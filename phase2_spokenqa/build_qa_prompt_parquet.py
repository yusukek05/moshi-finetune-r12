"""Build a spoken-QA prompt parquet for LLM-jp-Moshi continuation.

Each spoken-magpie-ja question becomes one v1-schema row where the OTHER
speaker (B) asks the question and the MAIN speaker (A) is silent, so that
running generate.py with --moshi_speakers A makes the model generate A as the
answer. Per the model's main_speaker_streams(), the other speaker's TEXT row is
dropped — only B's 8 audio codebooks are seen — so this is the natural Moshi
full-duplex setup (user speaks, model listens and replies); no text alignment
for the question is needed.

Layout per row (frame rate 12.5 Hz, fixed PROMPT_FRAMES long):
  B = [ text=pad(3) ; Mimi(question audio, 16k->24k) padded with silence ]  [9, P]
  A = [ text=pad(3) ; Mimi(zeros) silence ]                                 [9, P]

generate.py then continues for --generation_length frames; decode_tokens emits
stereo L=A (answer) / R=B (question), and we ASR channel 0 (A).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders

TEXT_PAD = 3


def ceil_div(x, y):
    return int(-(-x // y))


def encode(wav: torch.Tensor, mimi) -> np.ndarray:
    """wav: 1D float tensor at mimi.sample_rate -> [8, T] int codebooks."""
    with torch.no_grad():
        ids = mimi.encode(wav.reshape(1, 1, -1).to(next(mimi.parameters()).device))
    return ids[0].cpu().numpy()  # [8, T]


def main(a: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mimi = loaders.get_mimi(
        filename=hf_hub_download(a.audio_tokenizer_repo, a.audio_tokenizer_name),
        device=device,
    )
    sr_target = mimi.sample_rate
    samples_per_frame = sr_target / mimi.frame_rate
    P = a.prompt_frames

    # silence reference: encode P frames of zeros once, reuse for A and B-pad.
    zeros = torch.zeros(int(P * samples_per_frame), dtype=torch.float32)
    sil = encode(zeros, mimi)[:, :P]                      # [8, P]
    if sil.shape[1] < P:                                  # pad if codec returned fewer
        sil = np.pad(sil, ((0, 0), (0, P - sil.shape[1])), mode="edge")

    meta = json.load(open(Path(a.questions_dir) / "meta.json"))
    rows = []
    for m in meta:
        qid = m["qid"]
        wav, sr = torchaudio.load(str(Path(a.questions_dir) / f"{qid}.wav"))
        wav = wav.mean(0)                                 # mono
        if sr != sr_target:
            wav = torchaudio.transforms.Resample(sr, sr_target)(wav)
        q = encode(wav.to(device), mimi)                  # [8, Tq]
        Tq = q.shape[1]
        if Tq >= P:
            q = q[:, :P]
        else:
            q = np.concatenate([q, sil[:, : P - Tq]], axis=1)  # pad tail with silence

        text_row = np.full((1, P), TEXT_PAD, dtype=np.int64)
        B = np.concatenate([text_row, q.astype(np.int64)], axis=0)        # [9, P]
        A = np.concatenate([text_row, sil.astype(np.int64)], axis=0)      # [9, P]
        rows.append({"dialogue_id": qid, "A": A.tolist(), "B": B.tolist()})

    out = Path(a.output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    print(f"wrote {out}: {len(rows)} prompts, P={P} frames "
          f"({P / mimi.frame_rate:.1f}s each), schema A/B = [9, {P}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-dir", required=True)
    ap.add_argument("--output-parquet", required=True)
    ap.add_argument("--prompt-frames", type=int, default=150)  # 12.0 s
    ap.add_argument("--audio-tokenizer-repo", default="kyutai/moshiko-pytorch-bf16")
    ap.add_argument("--audio-tokenizer-name",
                    default="tokenizer-e351c8d8-checkpoint125.safetensors")
    main(ap.parse_args())
