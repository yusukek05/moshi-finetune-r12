#!/usr/bin/env python3
"""Decode the inner-monologue TEXT stream (stream 0) of generated Moshi token
arrays into clean, noise-free transcripts.

`generate.py` saves the full ``(1+K, T)`` token array per example
(stream 0 = text / inner-monologue, streams 1..K = Mimi audio codebooks) but
``tools/decode_tokens.py`` only decodes the audio and throws the text away.
This tool recovers the text: it drops the padding frames (``text_padding_id``)
and the word-boundary markers (``end_of_text_padding_id``) and decodes the
remaining SentencePiece ids -> the exact words Moshi "said", with ZERO ASR
noise. This is the inverse of ``tools/tokenize_text_from_dir.py``.

Used for the text-based meaningfulness reward (crowdsourcing MOS -> DPO): the
RL loop can judge meaning from this transcript instead of ASR-ing the (often
collapsed) generated audio, which is what limited the acoustic evaluator.

IMPORTANT: the v1 / v1.1 / v1.2 lineage uses the ``rinna/japanese-gpt2-medium``
``spiece.model`` text tokenizer with ``text_padding_id=3`` and
``end_of_text_padding_id=0`` (NOT the kyutai spm). Those are the defaults here;
override for other tokenizers.

Run (login node, no GPU needed — text only):
    uv run --no-project --python 3.12 --with sentencepiece --with numpy \
        --with huggingface-hub python tools/decode_text_from_tokens.py \
        --tokens_dir output/<run>/generated_tokens \
        --output_dir output/<run>/generated_text --jsonl output/<run>/transcripts.jsonl
"""
import argparse
import glob
import json
import os

import numpy as np
from huggingface_hub import hf_hub_download
from sentencepiece import SentencePieceProcessor


def decode_text_stream(
    text_tokens: np.ndarray,
    sp: SentencePieceProcessor,
    text_padding_id: int,
    end_of_text_padding_id: int,
) -> tuple[str, list[int]]:
    """stream-0 token ids -> (transcript, kept_content_ids).

    Drops the ``text_padding_id`` (frames with no token) and the
    ``end_of_text_padding_id`` (word-boundary marker) frames, leaving the real
    SentencePiece word-piece ids, then decodes them.
    """
    kept = [
        int(t)
        for t in text_tokens.tolist()
        if int(t) != text_padding_id and int(t) != end_of_text_padding_id
    ]
    return sp.decode(kept), kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens_dir", default=None,
                    help="Directory of generated *.npy token arrays (shape (1+K, T)).")
    ap.add_argument("--tokens", nargs="*", default=[],
                    help="Explicit *.npy token files (in addition to --tokens_dir).")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to write one <id>.txt transcript per example.")
    ap.add_argument("--jsonl", default=None,
                    help="Optional combined JSONL of {id, text, n_text_tokens, distinct_ratio}.")
    ap.add_argument("--text_tokenizer_repo", default="rinna/japanese-gpt2-medium")
    ap.add_argument("--text_tokenizer_name", default="spiece.model")
    ap.add_argument("--text_padding_id", type=int, default=3)
    ap.add_argument("--end_of_text_padding_id", type=int, default=0)
    ap.add_argument("--text_stream_index", type=int, default=0,
                    help="Row of the (1+K, T) array holding the text stream (0 = first).")
    args = ap.parse_args()

    paths = list(args.tokens)
    if args.tokens_dir:
        paths += sorted(glob.glob(os.path.join(args.tokens_dir, "*.npy")))
    if not paths:
        ap.error("no token files given (use --tokens_dir and/or --tokens)")

    sp = SentencePieceProcessor(
        hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name)
    )
    os.makedirs(args.output_dir, exist_ok=True)

    records = []
    for p in paths:
        arr = np.load(p)
        stream = arr[args.text_stream_index]
        text, kept = decode_text_stream(
            stream, sp, args.text_padding_id, args.end_of_text_padding_id
        )
        stem = os.path.splitext(os.path.basename(p))[0]
        with open(os.path.join(args.output_dir, f"{stem}.txt"), "w") as f:
            f.write(text + "\n")
        n = len(kept)
        # distinct_ratio: unique / total content tokens — a cheap repetition-loop
        # signal (a "反復ループ" collapse drives this toward 0).
        distinct_ratio = (len(set(kept)) / n) if n else 0.0
        records.append({
            "id": stem,
            "text": text,
            "n_text_tokens": n,
            "distinct_ratio": round(distinct_ratio, 3),
        })

    records.sort(key=lambda r: r["id"])
    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"decoded {len(records)} transcripts -> {args.output_dir}"
          + (f" (+ {args.jsonl})" if args.jsonl else ""))


if __name__ == "__main__":
    main()
