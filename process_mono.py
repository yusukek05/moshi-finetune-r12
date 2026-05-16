"""Offline preprocessing for mono training.

Builds the HF datasets cache for `preprocess_function_for_singlechannel` so that
`finetune_mono_text.py`'s `main_process_first()` map() returns instantly on the
real run (no NCCL watchdog timeout while rank 0 preprocesses for an hour).

Cache fingerprint depends on fn_kwargs values, so we load the real base model
and pass its token IDs verbatim (matches what finetune_mono_text.py passes).
"""

import argparse

import torch
from datasets import load_dataset

from data_utils import preprocess_function_for_singlechannel
from models import AutoMoshiForFinetuning


def main(args):
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
        "max_length": args.max_length,
        "num_audio_codebooks": moshi_lm.num_audio_codebooks,
        "max_audio_delay": args.max_audio_delay,
        "initial_token_ids": [moshi_lm.text_initial_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "padding_token_ids": [moshi_lm.text_padding_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "zero_token_id": moshi_lm.zero_token_id,
        "use_kana": args.use_kana,
    }

    print(f"Mapping {len(train_dataset)} examples with num_proc={args.dataset_processing_workers}")
    train_dataset = train_dataset.map(
        preprocess_function_for_singlechannel,
        remove_columns=train_dataset.column_names,
        batched=True,
        num_proc=args.dataset_processing_workers,
        fn_kwargs=preprocessing_kwargs,
        desc="Preprocessing train dataset",
    )
    print(f"Done. Output examples: {len(train_dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_files", type=str, nargs="+", required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_dtype", type=str, default="bfloat16")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_audio_delay", type=int, default=10)
    parser.add_argument("--dataset_processing_workers", type=int, default=32)
    parser.add_argument("--dataset_cache_dir", type=str, default=".cache/huggingface/datasets")
    parser.add_argument("--use_kana", action="store_true")
    main(parser.parse_args())
