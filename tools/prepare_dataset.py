import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def merge_text_audio(
    text_ids: np.ndarray, audio_ids: np.ndarray, text_padding_id: int
) -> np.ndarray:
    """
    Merge the tokenized text and audio stream of a single speaker.
    Args:
        text_ids: Tokenized text stream. Shape: [T_text]
        audio_ids: Tokenized audio stream. Shape: [K=8, T_audio]
        text_padding_id: Padding id for text stream to fill the gap between audio and text streams.
    Returns:
        Merged tokenized text and audio stream. Shape: [K=8+1, T_audio]
    """
    assert text_ids.ndim == 1, f"Expected 1D tensor, got {text_ids.ndim}D tensor."
    assert audio_ids.ndim == 2, f"Expected 2D tensor, got {audio_ids.ndim}D tensor."
    # pad the text stream to match the audio stream
    audio_len = audio_ids.shape[-1]
    if text_ids.shape[0] > audio_len:
        text_ids = text_ids[:audio_len]
    elif text_ids.shape[0] < audio_len:
        text_ids = np.concat(
            [text_ids, np.full(audio_len - text_ids.shape[0], text_padding_id)], axis=0
        )
    return np.concat([text_ids[None], audio_ids], axis=0).astype(np.int32).tolist()


def main(args):
    text_dialogue_names = [
        os.path.splitext(f)[0] for f in os.listdir(args.tokenized_text_dir)
    ]
    audio_dialogue_names = [
        os.path.splitext(f)[0] for f in os.listdir(args.tokenized_audio_dir)
    ]
    missing_text_dialogue_names = set(audio_dialogue_names) - set(text_dialogue_names)
    missing_audio_dialogue_names = set(text_dialogue_names) - set(audio_dialogue_names)
    if missing_text_dialogue_names:
        print(
            f"Missing tokenized text for {len(missing_text_dialogue_names)} dialogues."
        )
        open("missing_text_dialogue_names.txt", "w").write(
            "\n".join(missing_text_dialogue_names)
        )
    if missing_audio_dialogue_names:
        print(
            f"Missing tokenized audio for {len(missing_audio_dialogue_names)} dialogues."
        )
        open("missing_audio_dialogue_names.txt", "w").write(
            "\n".join(missing_audio_dialogue_names)
        )
    # if missing_text_dialogue_names or missing_audio_dialogue_names:
    #     print("Both text and audio tokenized dialogues should match.")
    #     return

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    dialogue_names = sorted(set(text_dialogue_names) & set(audio_dialogue_names))
    num_dialogues = len(dialogue_names)
    num_parquets = -(-num_dialogues // args.num_examples_per_parquet)

    for i in range(num_parquets):
        dials_per_parquet = dialogue_names[
            i * args.num_examples_per_parquet : (i + 1) * args.num_examples_per_parquet
        ]

        # load the tokenized text and audio data
        data = []
        for dialogue_name in tqdm(
            dials_per_parquet, desc=f"Processing parquet {i + 1}/{num_parquets}"
        ):
            text_path = os.path.join(args.tokenized_text_dir, f"{dialogue_name}.npz")
            text_ids = np.load(text_path)
            audio_path = os.path.join(args.tokenized_audio_dir, f"{dialogue_name}.npz")
            audio_ids = np.load(audio_path)
            data.append(
                {
                    "dialogue_id": os.path.join(
                        args.output_prefix, dialogue_name
                    ),  # unique identifier
                    "A": merge_text_audio(
                        text_ids["A"], audio_ids["A"], args.text_padding_id
                    ),
                    "B": merge_text_audio(
                        text_ids["B"], audio_ids["B"], args.text_padding_id
                    ),
                }
            )

        # save the merged data
        df = pd.DataFrame(data)
        output_path = f"{args.output_prefix}-{i + 1:03d}-of-{num_parquets:03d}.parquet"
        df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge the tokenized text and audio data into a single dataset in parquet format."
    )
    parser.add_argument(
        "--tokenized_text_dir",
        type=str,
        required=True,
        help="Path to the directory containing the tokenized text data.",
    )
    parser.add_argument(
        "--tokenized_audio_dir",
        type=str,
        required=True,
        help="Path to the directory containing the tokenized audio data.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help=(
            "Prefix for the output dataset. Output files will be named as "
            "`{{output_prefix}}-001-of-002.parquet` etc."
        ),
    )
    parser.add_argument(
        "--text_padding_id",
        type=int,
        default=3,
        help="Padding id for text stream to fill the gap between audio and text streams.",
    )
    parser.add_argument(
        "--num_examples_per_parquet",
        type=int,
        default=100_000,
        help="Number of samples per parquet file.",
    )
    args = parser.parse_args()

    main(args)
