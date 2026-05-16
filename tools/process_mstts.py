"""Offline preprocessing for mstts (multi-stream TTS) training.

Builds the HF datasets cache for `preprocess_function_for_multistream_tts`
so that `mstts/finetune_ms_tts.py`'s `main_process_first()` map() returns
instantly on the real run (no NCCL watchdog timeout while rank 0
preprocesses ~5M J-CHAT multi-stream examples).

Cache fingerprint depends on fn_kwargs values, so we load the real base
model and pass its token IDs verbatim — must match what
`finetune_ms_tts.py` passes for the same `--moshi_speakers A B`.
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "mstts"))

from data_utils import preprocess_function_for_multistream_tts  # noqa: E402
from models import AutoMoshiForFinetuning  # noqa: E402


def main(args: argparse.Namespace) -> None:
    moshi_lm = AutoMoshiForFinetuning.from_pretrained(
        save_dir=args.model_dir,
        device=torch.device("cpu"),
        dtype=getattr(torch, args.model_dtype),
    )

    train_dataset = load_dataset(
        "parquet",
        split="train",
        data_files={"train": args.train_data_files},
        cache_dir=args.dataset_cache_dir,
    )

    preprocessing_kwargs = {
        "speakers": args.moshi_speakers,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "delays": moshi_lm.delays,
        "initial_token_ids": [moshi_lm.text_initial_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "padding_token_ids": [moshi_lm.text_padding_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "end_of_text_padding_token_id": moshi_lm.end_of_text_padding_id,
        "main_speaker_bos_id": args.main_speaker_bos_id,
        "other_speaker_bos_id": args.other_speaker_bos_id,
        "zero_token_id": moshi_lm.zero_token_id,
    }

    print(f"Mapping {len(train_dataset)} examples with num_proc={args.dataset_processing_workers}")
    train_dataset = train_dataset.map(
        preprocess_function_for_multistream_tts,
        remove_columns=train_dataset.column_names,
        batched=True,
        num_proc=args.dataset_processing_workers,
        fn_kwargs=preprocessing_kwargs,
        desc="Preprocessing train dataset (mstts)",
    )
    print(f"Done. Output examples: {len(train_dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_files", type=str, nargs="+", required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--min_length", type=int, default=128)
    parser.add_argument("--moshi_speakers", nargs="+", default=["A", "B"])
    parser.add_argument("--main_speaker_bos_id", type=int, default=1)
    parser.add_argument("--other_speaker_bos_id", type=int, default=2)
    parser.add_argument("--dataset_processing_workers", type=int, default=32)
    parser.add_argument("--dataset_cache_dir", type=str, default=".cache/huggingface/datasets")
    main(parser.parse_args())
