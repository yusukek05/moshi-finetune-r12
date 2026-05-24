"""Decode sample rows from a mono (single-speaker) q16 parquet back to wav.

Used to A/B the *training-data quality* of the Stage-1 mono corpora
(LaboroTV / J-CHAT-mono / ccaudio) by ear. Each corpus stores Mimi q16
tokens with the same codec, so decoding all three with one Mimi gives a
fair apples-to-apples comparison of what the model actually ingests.

Mono parquet schema: {__key__: str, A_text: list<int64>, A_audio: list<list<int64>> (K=16, T)}.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from sentencepiece import SentencePieceProcessor

# rinna/japanese-gpt2-medium padding / end-of-text sentinels
_DROP_IDS = {0, 3}


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mimi = loaders.get_mimi(
        filename=hf_hub_download(args.audio_tokenizer_repo, args.audio_tokenizer_name),
        device=device,
    )
    mimi.set_num_codebooks(16)
    print(f"[info] mimi sr={mimi.sample_rate} fr={mimi.frame_rate} K={mimi.num_codebooks}")

    text_tok = SentencePieceProcessor(
        hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name)
    )
    os.makedirs(args.output_dir, exist_ok=True)

    tbl = pq.read_table(args.parquet)
    n = min(args.num_samples, tbl.num_rows)
    print(f"[info] {args.parquet}: {tbl.num_rows} rows, decoding first {n}")

    max_frames = int(args.max_seconds * mimi.frame_rate)
    for i in range(n):
        row = tbl.slice(i, 1).to_pylist()[0]
        audio = np.array(row["A_audio"], dtype=np.int64)  # (K, T)
        if audio.shape[1] > max_frames:
            audio = audio[:, :max_frames]
        with torch.no_grad():
            t = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, K, T)
            wav = mimi.decode(t).cpu().numpy().squeeze()  # (L,)
        out = os.path.join(args.output_dir, f"{args.prefix}_{i:02d}.wav")
        sf.write(out, wav.astype(np.float32), samplerate=mimi.sample_rate)

        text_ids = [
            x for x in row["A_text"]
            if x not in _DROP_IDS and 0 <= x < text_tok.vocab_size()
        ]
        txt = text_tok.decode(text_ids)
        print(f"  {args.prefix}_{i:02d}  key={row['__key__']}  "
              f"{wav.shape[0] / mimi.sample_rate:.1f}s  text={txt[:90]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prefix", required=True, help="wav filename prefix (corpus name)")
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--max_seconds", type=float, default=30.0)
    ap.add_argument("--text_tokenizer_repo", default="rinna/japanese-gpt2-medium")
    ap.add_argument("--text_tokenizer_name", default="spiece.model")
    ap.add_argument("--audio_tokenizer_repo", default="kyutai/moshiko-pytorch-bf16")
    ap.add_argument("--audio_tokenizer_name",
                    default="tokenizer-e351c8d8-checkpoint125.safetensors")
    main(ap.parse_args())
