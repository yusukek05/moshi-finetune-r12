"""Decode the Ground-Truth (natural) continuation audio for the leaderboard.

For each of the first N rows of a v1-format test parquet {dialogue_id, A:[9,T], B:[9,T]},
take the *continuation* segment (frames prompt_len:example_len) of the real audio and
Mimi-decode it to a stereo wav (L = speaker A, R = speaker B), named <i>.wav to match
the models' generated_wavs (positional index == parquet row).

This is the human topline / reference row for llm-jp-moshi-eval. Run in the
moshi-finetune .venv (has moshi/Mimi).
"""
import argparse
import os

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--prompt_len", type=int, default=125)
    ap.add_argument("--example_len", type=int, default=375)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=args.device)
    mimi.set_num_codebooks(8)

    t = pq.read_table(args.parquet)
    A_all = t.column("A").to_pylist()
    B_all = t.column("B").to_pylist()
    n = min(args.n, len(A_all))
    made = 0
    for i in range(n):
        A = np.asarray(A_all[i]); B = np.asarray(B_all[i])
        end = min(args.example_len, A.shape[1], B.shape[1])
        if end <= args.prompt_len + 1:
            print(f"[skip] row {i}: too short (T={A.shape[1]})", flush=True)
            continue
        la = A[1:9, args.prompt_len:end]
        rb = B[1:9, args.prompt_len:end]
        with torch.no_grad():
            lw = mimi.decode(torch.from_numpy(la[None]).long().to(args.device)).cpu().numpy()[0, 0]
            rw = mimi.decode(torch.from_numpy(rb[None]).long().to(args.device)).cpu().numpy()[0, 0]
        m = min(len(lw), len(rw))
        stereo = np.stack([lw[:m], rw[:m]], axis=1)  # L=A, R=B
        sf.write(os.path.join(args.out_dir, f"{i}.wav"), stereo, 24000)
        made += 1
        if i % 10 == 0:
            print(f"  [{i}/{n}] {t.column('dialogue_id').to_pylist()[i]} dur={m/24000:.1f}s", flush=True)
    print(f"[done] wrote {made} GT wavs -> {args.out_dir}")


if __name__ == "__main__":
    main()
