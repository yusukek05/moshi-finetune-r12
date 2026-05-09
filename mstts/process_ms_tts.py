import os
import json
import argparse
from glob import glob
from dataclasses import dataclass
from datasets import load_dataset

from data_utils import preprocess_function_for_multistream_tts

@dataclass
class DummyMoshiLMForFinetuning:
    delays: list[int]
    end_of_text_padding_id: int
    text_padding_token_id: int
    zero_token_id: int
    text_initial_token_id: int
    initial_token_id: int
    num_audio_codebooks: int

    @classmethod
    def from_kwargs_file(cls, kwargs_file: str) -> "DummyMoshiLMForFinetuning":
        # get moshi_*_kwargs.json file
        moshi_lm_kwargs = json.load(open(kwargs_file))

        if kwargs_file.endswith("moshi_lm_kwargs.json"):
            end_of_text_padding_id = 0
        elif kwargs_file.endswith("moshi_llama_kwargs.json"):
            end_of_text_padding_id = moshi_lm_kwargs["end_of_text_padding_id"]

        return cls(
            delays=moshi_lm_kwargs["delays"],
            end_of_text_padding_id=end_of_text_padding_id,
            text_padding_token_id=moshi_lm_kwargs["existing_text_padding_id"],
            zero_token_id=-1,
            text_initial_token_id=moshi_lm_kwargs["text_card"],
            initial_token_id=moshi_lm_kwargs["card"],
            num_audio_codebooks=moshi_lm_kwargs["n_q"],
        )

def main(args):
    train_dataset = load_dataset(
        "parquet", split="train",
        data_files={"train": args.train_data_files},
        cache_dir=args.dataset_cache_dir,
    )

    moshi_lm = DummyMoshiLMForFinetuning.from_kwargs_file(args.ms_tts_kwargs_file)

    preprocessing_kwargs = {
        "speakers": args.moshi_speakers,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "delays": moshi_lm.delays,
        "initial_token_ids": [
            moshi_lm.text_initial_token_id
        ] + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "padding_token_ids": [
            moshi_lm.text_padding_token_id
        ] + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "end_of_text_padding_token_id": moshi_lm.end_of_text_padding_id,
        "main_speaker_bos_id": args.main_speaker_bos_id,
        "other_speaker_bos_id": args.other_speaker_bos_id,
        "zero_token_id": moshi_lm.zero_token_id,
    }

    dataset_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        preprocess_function_for_multistream_tts,
        remove_columns=dataset_columns,
        batched=True, num_proc=args.dataset_processing_workers,
        fn_kwargs=preprocessing_kwargs,
        desc="Preprocessing train dataset",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_files",
        type=str,
        required=True,
        help="Path to the training data files in parquet format",
    )
    parser.add_argument(
        "--ms_tts_kwargs_file",
        type=str,
        required=True,
        help="Path to the Moshi LM kwargs JSON file (moshi_lm_kwargs.json or similar)",
    )
    parser.add_argument(
        "--moshi_speakers",
        choices=["A", "B"],
        nargs="+",
        default=["A"],
        help="Speakers to use as the main stream",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum length of the input sequence",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=None,
        help="Minimum length of the input sequence",
    )
    parser.add_argument(
        "--dataset_processing_workers",
        type=int,
        default=16,
        help="Number of workers to use for processing the dataset.",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=".cache/huggingface/datasets",
        help="Directory to cache the datasets.",
    )

    parser.add_argument(
        "--main_speaker_bos_id", type=int, default=1,
        help="Text token id of the main speaker"
    )
    parser.add_argument(
        "--other_speaker_bos_id", type=int, default=2,
        help="Text token id of the other speaker"
    )

    args = parser.parse_args()
    main(args)