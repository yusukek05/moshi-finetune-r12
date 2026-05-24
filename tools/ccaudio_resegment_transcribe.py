"""Re-segment + re-transcribe ccaudio episodes into short TTS utterances.

The upstream `ccaudio_rss_transcribed_array` is unusable for TTS training:
each cut is a 6-68 min podcast episode, the word alignments are broken
(timestamps exceed episode length in ~98% of cuts), and long-form Whisper
transcripts are heavily hallucinated (French, `a a a a`, Kannada `ಠ`).

This script goes back to the raw episode audio (the recording.*.tar flac
files), runs faster-whisper with its Silero VAD filter to split each
episode into short speech segments, and transcribes each segment. Short
VAD-bounded segments are far more robust to hallucination, and the word
timestamps are in-range. Survivors are filtered (hallucination heuristics
+ Whisper's own quality signals) and Mimi-q16 + rinna-SP tokenized into
the standard mono parquet schema: {__key__, A_text, A_audio}.

A sidecar `shard-NNNNN.transcripts.jsonl` is written for quality inspection.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from sentencepiece import SentencePieceProcessor

from tools.ccaudio_to_parquet import (
    is_hallucinated,
    merge_text_audio,
    tokenize_audio_chunked,
    tokenize_text_aligned,
)

WHISPER_SR = 16000


def iter_episodes(tar_paths: list[Path]):
    """Yield (tar_stem, flac_stem, mono_wav_np, sr) for each episode flac."""
    for tar_path in tar_paths:
        tar_stem = tar_path.stem  # "recording.000000"
        with tarfile.open(tar_path, "r") as tar:
            names = sorted(n for n in tar.getnames() if n.endswith(".flac"))
            for name in names:
                fobj = tar.extractfile(name)
                if fobj is None:
                    continue
                wav, sr = sf.read(io.BytesIO(fobj.read()), dtype="float32", always_2d=True)
                # ccaudio recordings are stereo; channel 0 is the supervised one
                yield tar_stem, Path(name).stem, wav[:, 0], sr


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    print("[info] loading Mimi …")
    mimi = loaders.get_mimi(
        filename=hf_hub_download(args.audio_tokenizer_repo, args.audio_tokenizer_name),
        device=device,
    )
    mimi.set_num_codebooks(args.num_codebooks)
    print(f"[info] mimi sr={mimi.sample_rate} fr={mimi.frame_rate} K={mimi.num_codebooks}")

    print("[info] loading rinna SP tokenizer …")
    sp = SentencePieceProcessor(hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name))

    print("[info] loading faster-whisper large-v3 …")
    from faster_whisper import WhisperModel

    asr = WhisperModel("large-v3", device="cuda", compute_type="float16")

    # Two source modes:
    #   --tar_path : process a single recording.*.tar  (raw_all, one tar per shard)
    #   --slice_idx: process all recording.*.tar in cc_root/<slice_idx>/  (transcribed_array)
    if args.tar_path:
        tar_paths = [Path(args.tar_path)]
        shard_id = args.shard_id
    else:
        slice_dir = Path(args.cc_root) / str(args.slice_idx)
        tar_paths = sorted(slice_dir.glob("recording.*.tar"))
        shard_id = args.slice_idx
    print(f"[info] shard={shard_id} tars={[p.name for p in tar_paths]}")
    out_path = Path(args.output_dir) / f"shard-{shard_id:05d}.parquet"
    txt_path = Path(args.output_dir) / f"shard-{shard_id:05d}.transcripts.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path} exists. Use --overwrite to rebuild.")
        return

    rows_key: list[str] = []
    rows_text: list[list[int]] = []
    rows_audio: list[list[list[int]]] = []
    transcripts: list[dict] = []

    n_ep = n_seg = n_kept = 0
    n_drop_len = n_drop_qual = n_drop_text = n_drop_err = 0
    t0 = time.time()

    episodes = iter_episodes(tar_paths)
    for tar_stem, stem, mono, sr in episodes:
        n_ep += 1
        if args.max_episodes and n_ep > args.max_episodes:
            break
        mono_t = torch.from_numpy(mono)
        wav16 = torchaudio.functional.resample(mono_t, sr, WHISPER_SR).numpy()
        wav24 = torchaudio.functional.resample(mono_t, sr, mimi.sample_rate)

        segments, _info = asr.transcribe(
            wav16,
            language="ja",
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
            beam_size=5,
        )
        for seg in segments:
            n_seg += 1
            dur = seg.end - seg.start
            if dur < args.min_seg_s or dur > args.max_seg_s:
                n_drop_len += 1
                continue
            text = (seg.text or "").strip()
            if not text:
                n_drop_text += 1
                continue
            # Whisper's own quality signals
            if (
                seg.no_speech_prob > args.no_speech_thresh
                or seg.avg_logprob < args.logprob_thresh
                or seg.compression_ratio > args.compression_thresh
            ):
                n_drop_qual += 1
                continue
            bad, _reason = is_hallucinated(text, args.rep_ratio_thresh, args.non_ja_thresh)
            if bad:
                n_drop_text += 1
                continue
            words = [
                [w.word.strip(), w.start - seg.start, w.end - w.start, w.probability]
                for w in (seg.words or [])
                if w.word and w.word.strip()
            ]
            if not words:
                n_drop_text += 1
                continue
            try:
                a0 = int(seg.start * mimi.sample_rate)
                a1 = int(seg.end * mimi.sample_rate)
                seg_wav = wav24[a0:a1].to(device)
                audio_ids = tokenize_audio_chunked(seg_wav, mimi, args.audio_chunk_size_s).numpy()
                text_ids = tokenize_text_aligned(
                    words,
                    sp,
                    text_padding_id=args.text_padding_id,
                    end_of_text_padding_id=args.end_of_text_padding_id,
                    frame_rate=mimi.frame_rate,
                )
                if not text_ids:
                    n_drop_text += 1
                    continue
                text_list, audio_list = merge_text_audio(
                    np.asarray(text_ids, dtype=np.int64),
                    audio_ids,
                    text_padding_id=args.text_padding_id,
                )
            except Exception as e:  # noqa: BLE001
                n_drop_err += 1
                print(f"[err] {stem} seg@{seg.start:.1f}: {e}")
                continue
            key = f"ccaudio/shard{shard_id:05d}/{tar_stem}/{stem}/seg{n_seg:05d}"
            rows_key.append(key)
            rows_text.append(text_list)
            rows_audio.append(audio_list)
            transcripts.append(
                {
                    "key": key,
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "dur": round(dur, 2),
                    "text": text,
                }
            )
            n_kept += 1
        print(
            f"[ep {n_ep}] {stem}: kept={n_kept} seg={n_seg} "
            f"drop(len={n_drop_len} qual={n_drop_qual} text={n_drop_text} err={n_drop_err})"
        )

    elapsed = time.time() - t0
    print(
        f"[done] shard={shard_id} episodes={n_ep} segments={n_seg} kept={n_kept} "
        f"drop_len={n_drop_len} drop_qual={n_drop_qual} drop_text={n_drop_text} "
        f"drop_err={n_drop_err} elapsed={elapsed / 60:.1f}min"
    )

    with open(txt_path, "w") as f:
        for t in transcripts:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[done] wrote {txt_path} ({len(transcripts)} segments)")

    if not rows_key:
        print(f"[warn] no rows survived for shard {shard_id}; skipping shard write.")
        return

    table = pa.table(
        {
            "__key__": pa.array(rows_key, type=pa.string()),
            "A_text": pa.array(rows_text, type=pa.list_(pa.int64())),
            "A_audio": pa.array(rows_audio, type=pa.list_(pa.list_(pa.int64()))),
        }
    )
    pq.write_table(table, out_path, compression="zstd")
    total_frames = sum(len(a[0]) for a in rows_audio)
    print(
        f"[done] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
        f"{total_frames / mimi.frame_rate / 3600:.2f} audio hours)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cc_root",
        type=str,
        default="/groups/gcg51557/experiments/0167_cc_audio/asai/ccaudio_rss_transcribed_array",
        help="Root with per-slice dirs holding recording.*.tar (slice mode).",
    )
    ap.add_argument(
        "--slice_idx", type=int, default=-1, help="transcribed_array slice 0..99 (slice mode)."
    )
    ap.add_argument(
        "--tar_path",
        type=str,
        default=None,
        help="A single recording.*.tar to process (raw_all mode; one tar per shard).",
    )
    ap.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Output shard index for --tar_path mode -> shard-NNNNN.parquet.",
    )
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--max_episodes", type=int, default=0, help="0 = all; >0 for smoke.")
    # segmentation / quality thresholds
    ap.add_argument("--min_seg_s", type=float, default=2.0)
    ap.add_argument("--max_seg_s", type=float, default=30.0)
    ap.add_argument("--no_speech_thresh", type=float, default=0.6)
    ap.add_argument("--logprob_thresh", type=float, default=-1.0)
    ap.add_argument("--compression_thresh", type=float, default=2.4)
    ap.add_argument("--rep_ratio_thresh", type=float, default=0.05)
    ap.add_argument("--non_ja_thresh", type=float, default=0.10)
    # tokenizer config
    ap.add_argument("--num_codebooks", type=int, default=16)
    ap.add_argument("--audio_chunk_size_s", type=int, default=30)
    ap.add_argument("--text_padding_id", type=int, default=3)
    ap.add_argument("--end_of_text_padding_id", type=int, default=0)
    ap.add_argument("--audio_tokenizer_repo", default="kyutai/moshiko-pytorch-bf16")
    ap.add_argument(
        "--audio_tokenizer_name", default="tokenizer-e351c8d8-checkpoint125.safetensors"
    )
    ap.add_argument("--text_tokenizer_repo", default="rinna/japanese-gpt2-medium")
    ap.add_argument("--text_tokenizer_name", default="spiece.model")
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
