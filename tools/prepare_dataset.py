import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def merge_text_audio(
    text_ids: np.ndarray, audio_ids: np.ndarray, text_padding_id: int
) -> list[list[int]]:
    """Tokenized text [T_text] と audio [8, T_audio] を長さを合わせて結合 → [9, T_audio]"""
    assert text_ids.ndim == 1, f"Expected 1D tensor, got {text_ids.ndim}D."
    assert audio_ids.ndim == 2, f"Expected 2D tensor, got {audio_ids.ndim}D."

    T_audio = audio_ids.shape[-1]
    if text_ids.shape[0] > T_audio:
        text_ids = text_ids[:T_audio]
    elif text_ids.shape[0] < T_audio:
        pad = np.full(
            T_audio - text_ids.shape[0], text_padding_id, dtype=text_ids.dtype
        )
        text_ids = np.concatenate([text_ids, pad], axis=0)

    merged = np.concatenate([text_ids[None], audio_ids], axis=0)  # [9, T_audio]
    return merged.astype(np.int32).tolist()


def collect_npz(root_dir: str) -> dict[str, str]:
    """
    root_dir 以下を再帰的に探索し、拡張子 .npz を
    {相対パス（拡張子なし）: 絶対パス} で返す。
    例: /root/000/abc.npz → {'000/abc': '/root/000/abc.npz'}
    """
    root = Path(root_dir)
    mapping = {}
    for f in root.rglob("*.npz"):
        rel_without_ext = str(f.relative_to(root).with_suffix(""))
        mapping[rel_without_ext] = str(f)
    return mapping


def main(args: argparse.Namespace):
    text_map = collect_npz(args.tokenized_text_dir)
    audio_map = collect_npz(args.tokenized_audio_dir)

    missing_text = set(audio_map) - set(text_map)
    missing_audio = set(text_map) - set(audio_map)

    if missing_text:
        print(f"Missing tokenized text for {len(missing_text)} dialogues.")
        Path("missing_text_dialogue_names.txt").write_text(
            "\n".join(sorted(missing_text))
        )
    if missing_audio:
        print(f"Missing tokenized audio for {len(missing_audio)} dialogues.")
        Path("missing_audio_dialogue_names.txt").write_text(
            "\n".join(sorted(missing_audio))
        )

    dialogue_names = sorted(set(text_map) & set(audio_map))
    if not dialogue_names:
        print("No matching dialogue found. Check your directories.")
        return

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    num_dialogues = len(dialogue_names)
    num_parquets = -(
        -num_dialogues // args.num_examples_per_parquet
    )  # ceiling division

    for i in range(num_parquets):
        chunk = dialogue_names[
            i * args.num_examples_per_parquet : (i + 1) * args.num_examples_per_parquet
        ]

        rows = []
        for dname in tqdm(chunk, desc=f"Parquet {i + 1}/{num_parquets}"):
            text_ids = np.load(text_map[dname])
            audio_ids = np.load(audio_map[dname])

            rows.append(
                {
                    "dialogue_id": os.path.join(args.output_prefix, dname),
                    "A": merge_text_audio(
                        text_ids["A"], audio_ids["A"], args.text_padding_id
                    ),
                    "B": merge_text_audio(
                        text_ids["B"], audio_ids["B"], args.text_padding_id
                    ),
                }
            )

        df = pd.DataFrame(rows)
        out_path = f"{args.output_prefix}-{i + 1:03d}-of-{num_parquets:03d}.parquet"
        df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge tokenized text & audio into parquet (supports nested dirs)."
    )
    parser.add_argument(
        "--tokenized_text_dir",
        required=True,
        help="Root dir containing tokenized text .npz files.",
    )
    parser.add_argument(
        "--tokenized_audio_dir",
        required=True,
        help="Root dir containing tokenized audio .npz files.",
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        help="Parquet prefix: {prefix}-001-of-XXX.parquet などが生成される。",
    )
    parser.add_argument(
        "--text_padding_id",
        type=int,
        default=3,
        help="Padding ID used to fill gaps in text stream.",
    )
    parser.add_argument(
        "--num_examples_per_parquet",
        type=int,
        default=100_000,
        help="Number of dialogues per parquet file.",
    )
    main(parser.parse_args())
