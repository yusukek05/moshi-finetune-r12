"""Convert cascade synthesis output (stereo wav + per-turn timing JSON) into
LLM-jp-Moshi-v1 parquet training format.

This is the cascade-pipeline counterpart to mstts_tokens_to_v1_parquet.py.
Why we need a different script:
  - mstts saves pre-tokenized (17, T) tensors with text/audio already aligned
    at frame rate 12.5 Hz.
  - cascade saves raw wav (44.1 kHz stereo, L=A R=B) plus turns.json that
    holds per-turn (speaker, text, t_start, t_end).
  - We must Mimi-tokenize each channel and place text tokens at the right
    frame positions based on turn timings.

Output schema (matches v1 finetune.py expectations):

    {
        "dialogue_id": str,
        "A": int32 list shape (9, T),   # row 0 = text, rows 1-8 = audio CBs
        "B": int32 list shape (9, T),
    }

T = number of 12.5 Hz frames = ceil(audio_duration_sec * 12.5).
A and B have identical length T (full-dialogue padded streams).

Text placement within a turn (evenly distributed):
  Given turn (speaker=X, t_start=ts, t_end=te, text="..."):
    tokens = SP.encode(text)                  # K tokens
    frames = floor(ts*12.5) .. ceil(te*12.5)  # N frames
    Map tokens to frames evenly. If N > K, pad remainder with TEXT_PADDING_ID.
    If N < K (very rapid turn), pack tokens densely; surplus tokens are dropped
    with a warning.

Usage:
    uv run python -m mstts.data_prep.cascade_output_to_v1_parquet \\
        --cascade-dir output/cascade_synth_full/ \\
        --output-prefix output/cascade_synth_v1_parquet/synth \\
        --num-examples-per-parquet 5000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from tqdm import tqdm
from transformers import T5Tokenizer

TEXT_PADDING_ID = 3
FRAME_RATE = 12.5

# Mimi / text tokenizer defaults — match v1 base
DEFAULT_AUDIO_TOK_REPO = "kyutai/moshiko-pytorch-bf16"
DEFAULT_AUDIO_TOK_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
DEFAULT_TEXT_TOK_REPO = "rinna/japanese-gpt2-medium"
DEFAULT_TEXT_TOK_NAME = "spiece.model"


def sec_to_frame(t: float) -> int:
    return int(round(t * FRAME_RATE))


def encode_text_to_frames(
    text: str,
    tokenizer,
    t_start: float,
    t_end: float,
    num_total_frames: int,
) -> tuple[np.ndarray, int]:
    """Return (frame_token_array, num_dropped_tokens).

    frame_token_array: shape (num_total_frames,), int64. Has TEXT_PADDING_ID
    everywhere except in [floor(t_start*12.5), ceil(t_end*12.5)) where
    text tokens are evenly distributed.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return np.full(num_total_frames, TEXT_PADDING_ID, dtype=np.int64), 0

    f0 = max(0, sec_to_frame(t_start))
    f1 = min(num_total_frames, sec_to_frame(t_end))
    span = max(1, f1 - f0)

    track = np.full(num_total_frames, TEXT_PADDING_ID, dtype=np.int64)
    if span >= len(tokens):
        # Distribute tokens evenly across the span; pad rest with TEXT_PADDING_ID
        positions = np.linspace(0, span - 1, num=len(tokens), dtype=np.int64)
        for tok, p in zip(tokens, positions):
            track[f0 + p] = tok
        return track, 0
    # Span smaller than token count: pack one token per frame, drop overflow
    for i in range(span):
        track[f0 + i] = tokens[i]
    return track, len(tokens) - span


def tokenize_audio_channel(
    wav: torch.Tensor,  # (samples,) at mimi.sample_rate
    mimi,
    chunk_sec: int = 60,
) -> torch.LongTensor:
    """Return (8, T_audio) Mimi codes."""
    assert wav.dim() == 1, wav.shape
    chunk_samples = chunk_sec * mimi.sample_rate
    n_chunks = math.ceil(wav.shape[0] / chunk_samples)
    out = []
    for i in range(n_chunks):
        x = wav[i * chunk_samples : (i + 1) * chunk_samples]
        with torch.no_grad():
            codes = mimi.encode(x.reshape(1, 1, -1).to(mimi.device)).cpu()
        out.append(codes[0])  # (8, T_chunk)
    return torch.cat(out, dim=-1)


def process_one(
    dialogue_dir: Path,
    mimi,
    text_tok,
    resampler_cache: dict,
    speaker_main: str,
    audio_main_label: str,
) -> dict | None:
    """Convert one cascade output dir → {dialogue_id, A:[9,T], B:[9,T]}."""
    wav_path = dialogue_dir / "audio.wav"
    turns_path = dialogue_dir / "turns.json"
    if not wav_path.exists() or not turns_path.exists():
        return None

    turns = json.loads(turns_path.read_text(encoding="utf-8"))

    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] != 2:
        return None
    # Resample to mimi.sample_rate (24 kHz)
    if sr != mimi.sample_rate:
        key = (sr, mimi.sample_rate)
        if key not in resampler_cache:
            resampler_cache[key] = torchaudio.transforms.Resample(sr, mimi.sample_rate).to(mimi.device)
        wav = resampler_cache[key](wav.to(mimi.device))
    else:
        wav = wav.to(mimi.device)

    # Tokenize each channel → (8, T_audio)
    L_codes = tokenize_audio_channel(wav[0], mimi)
    R_codes = tokenize_audio_channel(wav[1], mimi)
    T_audio = min(L_codes.shape[-1], R_codes.shape[-1])
    L_codes = L_codes[:, :T_audio]
    R_codes = R_codes[:, :T_audio]

    # Build text streams (one per speaker)
    a_text = np.full(T_audio, TEXT_PADDING_ID, dtype=np.int64)
    b_text = np.full(T_audio, TEXT_PADDING_ID, dtype=np.int64)
    total_dropped = 0
    for turn in turns:
        spk = turn["speaker"]
        text = turn["text"]
        ts = float(turn["t_start"])
        te = float(turn["t_end"])
        track, dropped = encode_text_to_frames(text, text_tok, ts, te, T_audio)
        total_dropped += dropped
        # Only overlay frames where track has non-padding (preserve TEXT_PADDING_ID elsewhere)
        mask = track != TEXT_PADDING_ID
        if spk == "A":
            a_text[mask] = track[mask]
        elif spk == "B":
            b_text[mask] = track[mask]

    # L=A, R=B by cascade convention
    a_audio = L_codes.numpy().astype(np.int32)
    b_audio = R_codes.numpy().astype(np.int32)

    a_stream = np.concatenate([a_text[None, :], a_audio], axis=0)  # (9, T)
    b_stream = np.concatenate([b_text[None, :], b_audio], axis=0)

    return {
        "dialogue_id": dialogue_dir.name,
        "A": a_stream.astype(np.int32).tolist(),
        "B": b_stream.astype(np.int32).tolist(),
        "_dropped_text_tokens": total_dropped,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cascade-dir", required=True,
                    help="Root of cascade output: <root>/<dialogue_id>/{audio.wav, turns.json}")
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--num-examples-per-parquet", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, process only the first N dialogues (smoke test)")
    ap.add_argument("--audio-tokenizer-repo", default=DEFAULT_AUDIO_TOK_REPO)
    ap.add_argument("--audio-tokenizer-name", default=DEFAULT_AUDIO_TOK_NAME)
    ap.add_argument("--text-tokenizer-repo", default=DEFAULT_TEXT_TOK_REPO)
    ap.add_argument("--text-tokenizer-name", default=DEFAULT_TEXT_TOK_NAME)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cascade_dir = Path(args.cascade_dir)
    dirs = sorted([d for d in cascade_dir.iterdir() if d.is_dir()])
    if args.limit > 0:
        dirs = dirs[:args.limit]
    print(f"found {len(dirs)} dialogue dirs under {cascade_dir}")

    device = torch.device(args.device)
    print("loading Mimi codec ...")
    mimi = loaders.get_mimi(
        filename=hf_hub_download(args.audio_tokenizer_repo, args.audio_tokenizer_name),
        device=device,
    )
    mimi.device = device

    print("loading text tokenizer (rinna_gpt2 SP) ...")
    sp_path = hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name)
    text_tok = T5Tokenizer(vocab_file=sp_path)

    out_dir = Path(args.output_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    resampler_cache: dict = {}
    n_parquets = math.ceil(len(dirs) / args.num_examples_per_parquet)
    total_dropped = 0
    for i in range(n_parquets):
        chunk = dirs[i * args.num_examples_per_parquet: (i + 1) * args.num_examples_per_parquet]
        rows = []
        for d in tqdm(chunk, desc=f"shard {i+1}/{n_parquets}"):
            try:
                r = process_one(d, mimi, text_tok, resampler_cache,
                                speaker_main="B", audio_main_label="B")
            except Exception as exc:
                print(f"  SKIP {d.name}: {exc!r}")
                continue
            if r is None:
                continue
            total_dropped += r.pop("_dropped_text_tokens", 0)
            rows.append(r)
        df = pd.DataFrame(rows)
        out_path = f"{args.output_prefix}-{i + 1:03d}-of-{n_parquets:03d}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"wrote {len(rows)} rows → {out_path}")

    if total_dropped:
        print(f"WARNING: {total_dropped} text tokens were dropped (turns shorter than token count)")


if __name__ == "__main__":
    main()
