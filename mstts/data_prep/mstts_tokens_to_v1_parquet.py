"""Convert mstts generated tokens to LLM-jp-Moshi-v1 parquet training format.

LLM-jp-Moshi-v1's finetune.py expects parquet files with per-row schema:

    {
        "dialogue_id": str,
        "A": [9, T_A],   # 1 text track + 8 audio codebooks for speaker A
        "B": [9, T_B],   # 1 text track + 8 audio codebooks for speaker B
    }

mstts saves generated tokens as `.npy` files of shape (17, T):

    row 0       : merged dialogue text (single channel, frame-aligned at 12.5 Hz)
                  - contains BOS markers (main_speaker_bos_id, other_speaker_bos_id)
                    inserted at speaker turn boundaries
                  - other frames carry the active speaker's text tokens or
                    text_padding_id / end_of_text_padding_id
    rows 1..8   : audio codebooks for "main speaker"  position
    rows 9..16  : audio codebooks for "other speaker" position

To convert mstts → v1 we demultiplex the row-0 text channel back into
per-speaker text tracks using the BOS markers as switch points. Audio
rows transfer directly, but we need to know which physical position
("main" rows 1-8 or "other" rows 9-16) corresponds to v1's "A" vs "B".

------------------------------------------------------------------
Speaker assignment convention (verified empirically on smoke output)
------------------------------------------------------------------
mstts/hf_inference/inference.py calls
`tokenize_text_chat_for_multistream_tts(..., main_speaker_first=False)`.
With v0c training using `--moshi_speakers A B` (both orderings) and a
text_chat starting at ["A", "..."]:
  • Turn 0 (A) receives `other_speaker_bos_id=2` ⇒ mstts treats A as "other"
  • Turn 1 (B) receives `main_speaker_bos_id=1`  ⇒ mstts treats B as "main"
  • Audio rows 1-8 (main position) carry **B's** audio
  • Audio rows 9-16 (other position) carry **A's** audio
  • After Mimi decode wavs[0]=rows 1-8 → LEFT channel ⇒ LEFT = B, RIGHT = A

(This contradicts a casual "L=A, R=B" claim that appears elsewhere in the
docs; the empirical signal — early audio-token diversity at the row where
the first speaker is active — confirmed L = B.)

Defaults below match this convention. Override with --audio-main-label A
and --main-speaker-first if your inference call differed.

Usage:
    python mstts_tokens_to_v1_parquet.py \
        --tokens-dir output/mstts_v0c_synth/0_50/generated_tokens \
        --output-prefix output/mstts_v0c_synth/0_50/parquet/synth \
        --num-examples-per-parquet 1000
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Token IDs (must match mstts training convention)
TEXT_PADDING_ID = 3
END_OF_TEXT_PADDING_ID = 0
MAIN_SPEAKER_BOS_ID = 1
OTHER_SPEAKER_BOS_ID = 2
PADDING_IDS = {TEXT_PADDING_ID, END_OF_TEXT_PADDING_ID}


def demux_text_channel(
    dialogue_text: np.ndarray,
    main_speaker_label: str,
    main_speaker_first: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the merged text channel back into (A_text, B_text) per-frame tracks.

    Args:
        dialogue_text: 1-D array of token ids at 12.5 Hz.
        main_speaker_label: which v1 label ("A" or "B") corresponds to mstts "main".
        main_speaker_first: whether the main speaker spoke first (mirrors inference flag).

    Returns:
        (a_text, b_text): each shape (T,) with the active speaker's token at
        each frame and TEXT_PADDING_ID elsewhere.
    """
    T = len(dialogue_text)
    a_text = np.full(T, TEXT_PADDING_ID, dtype=dialogue_text.dtype)
    b_text = np.full(T, TEXT_PADDING_ID, dtype=dialogue_text.dtype)
    other_speaker_label = "B" if main_speaker_label == "A" else "A"

    # Initial active speaker before any BOS is seen
    current = main_speaker_label if main_speaker_first else other_speaker_label
    for t in range(T):
        tok = int(dialogue_text[t])
        if tok == MAIN_SPEAKER_BOS_ID:
            current = main_speaker_label
            continue  # BOS is a merger artifact; not in per-speaker track
        if tok == OTHER_SPEAKER_BOS_ID:
            current = other_speaker_label
            continue
        if tok == TEXT_PADDING_ID:
            continue  # both stay padded
        # Real text token or end_of_text_padding (0) → write to active speaker only
        if current == "A":
            a_text[t] = tok
        else:
            b_text[t] = tok
    return a_text, b_text


def convert_one(
    tokens: np.ndarray,
    audio_main_label: str,
    main_speaker_first: bool,
) -> tuple[list[list[int]], list[list[int]]]:
    """Convert one mstts (17, T) token array to (A_9xT, B_9xT) lists."""
    assert tokens.shape[0] == 17, (
        f"Expected 17 channels (1 text + 16 audio), got {tokens.shape[0]}"
    )
    text_ch = tokens[0]
    audio_main = tokens[1:9]  # (8, T)
    audio_other = tokens[9:17]  # (8, T)
    a_text, b_text = demux_text_channel(
        text_ch,
        main_speaker_label=audio_main_label,
        main_speaker_first=main_speaker_first,
    )

    if audio_main_label == "A":
        a_audio = audio_main
        b_audio = audio_other
    else:
        a_audio = audio_other
        b_audio = audio_main

    a_stream = np.concatenate([a_text[None], a_audio], axis=0)  # (9, T)
    b_stream = np.concatenate([b_text[None], b_audio], axis=0)  # (9, T)
    return a_stream.astype(np.int32).tolist(), b_stream.astype(np.int32).tolist()


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tokens-dir", required=True,
                   help="Directory containing mstts generated_tokens *.npy files (shape (17, T))")
    p.add_argument("--output-prefix", required=True,
                   help="Parquet prefix: {prefix}-001-of-XXX.parquet etc.")
    p.add_argument("--audio-main-label", choices=["A", "B"], default="B",
                   help="Which v1 label corresponds to mstts 'main' position (rows 1-8). "
                        "Default B: matches mstts inference convention where text_chat "
                        "starts with A and main_speaker_first=False (verified on v0c output).")
    p.add_argument("--main-speaker-first", action="store_true", default=False,
                   help="Whether the main speaker spoke first (mirror mstts inference flag).")
    p.add_argument("--num-examples-per-parquet", type=int, default=10_000)
    args = p.parse_args()

    tokens_dir = Path(args.tokens_dir)
    files = sorted(tokens_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"No .npy files in {tokens_dir}")

    out_dir = Path(args.output_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    num_parquets = -(-len(files) // args.num_examples_per_parquet)
    for i in range(num_parquets):
        chunk = files[i * args.num_examples_per_parquet :
                      (i + 1) * args.num_examples_per_parquet]
        rows = []
        for fp in tqdm(chunk, desc=f"Parquet {i + 1}/{num_parquets}"):
            tokens = np.load(fp)
            try:
                a_stream, b_stream = convert_one(
                    tokens,
                    audio_main_label=args.audio_main_label,
                    main_speaker_first=args.main_speaker_first,
                )
            except AssertionError as e:
                print(f"Skip {fp.name}: {e}")
                continue
            rows.append({
                "dialogue_id": fp.stem,
                "A": a_stream,
                "B": b_stream,
            })
        df = pd.DataFrame(rows)
        out_path = f"{args.output_prefix}-{i + 1:03d}-of-{num_parquets:03d}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Wrote {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
