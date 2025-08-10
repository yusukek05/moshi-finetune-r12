import argparse
import json
import multiprocessing as mp
import os
import warnings

import numpy as np
from huggingface_hub import hf_hub_download
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
import logging
from pathlib import Path


def encode_as_pieces_wo_byte_fallback(
    sp: SentencePieceProcessor, text: str
) -> list[str]:
    """
    Tokenize the text without using byte fallback.
    """
    tokens = sp.encode_as_pieces(text)
    if not tokens:
        return []

    tokens_wo_byte = []
    last_byte_tokens = []
    for token in tokens:
        if not token.startswith("<0x"):
            tokens_wo_byte.append(token)
            text = text[len(token) :]
        else:
            last_byte_tokens.append(token)
            token = sp.decode_pieces(last_byte_tokens)
            if text.startswith(token):  # token is successfully decoded
                tokens_wo_byte.append(token)
                text = text[len(token) :]
                last_byte_tokens = []
    if last_byte_tokens:
        raise ValueError(
            f"Failed to decode the last byte tokens: {last_byte_tokens} in {text}"
        )
    return tokens_wo_byte


def tokenize_and_pad_text(
    word_transcript: list[dict[str, str | float]],
    no_whitespace_before_word: bool,
    text_tokenizer: SentencePieceProcessor,
    text_padding_id: int,
    end_of_text_padding_id: int,
    audio_tokenizer_frame_rate: float,
) -> list[int]:
    """
    Tokenize the word transcript of single speaker.
    Fill the appropriate frames with the tokens based on the word-level timestamps,
    and frames without tokens are filled with the padding token.
    """
    # ---------- empty transcript ----------
    if not word_transcript:
        # 無発話 → 空の token_id 配列（保存時に np.savez は長さ 0 を許容）
        return []

    # ---------- single-speaker check ----------
    speakers = {seg["speaker"] for seg in word_transcript}
    assert len(speakers) == 1, f"Expected single speaker but found {speakers}"

    # sort the word transcript by the start time
    word_transcript = sorted(word_transcript, key=lambda x: x["start"])

    # add whitespace to the beginning of each transcript word
    if not no_whitespace_before_word:
        # ensure that the first word has no whitespace before it
        word_transcript[0]["word"] = word_transcript[0]["word"].strip()
        for seg in word_transcript[1:]:
            seg["word"] = " " + seg["word"].strip()

    # tokenize the text
    text = "".join([seg["word"] for seg in word_transcript])
    tokens = encode_as_pieces_wo_byte_fallback(text_tokenizer, text)

    # word-level transcript to character-level transcript
    char_transcript = []
    for seg in word_transcript:
        num_chars = len(seg["word"])
        start = seg["start"]
        end = seg["end"]
        # split the duration into num_chars
        char_duration = (end - start) / num_chars
        for i, char in enumerate(seg["word"]):
            char_transcript.append(
                {
                    "speaker": seg["speaker"],
                    "start": start + i * char_duration,
                    "end": start + (i + 1) * char_duration,
                    "char": char,
                }
            )

    # make token-level transcript by aligning the timestamps
    token_transcript = []
    for i, token in enumerate(tokens):
        if i == 0 and token == "▁":
            # skip the first underscore of sentencepiece
            continue
        if i == 0 and token.startswith("▁"):
            # don't count the first underscore
            chars = char_transcript[: len(token) - 1]
        else:
            chars = char_transcript[: len(token)]
        token_transcript.append(
            {
                "speaker": chars[0]["speaker"],
                "start": chars[0]["start"],
                "end": chars[-1]["end"],
                "token": token,
            }
        )
        # remove the characters that are already processed
        char_transcript = char_transcript[len(chars) :]
    assert not char_transcript, f"Remaining characters: {char_transcript}"

    # make tokenized ids with padding
    if token_transcript:
        num_frames = int((token_transcript[-1]["end"] + 1) * audio_tokenizer_frame_rate)
    else:
        num_frames = 0
    seconds_per_frame = 1 / audio_tokenizer_frame_rate
    token_ids = [text_padding_id] * num_frames
    token_count = 0
    for seg in token_transcript:
        frame_index = int(seg["start"] // seconds_per_frame)
        try:
            # find the next padding index to insert the token
            while token_ids[frame_index] != text_padding_id:
                frame_index += 1
        except IndexError:
            warnings.warn(  # noqa: B028
                f"Last {len(token_transcript) - token_count} tokens out of {len(token_transcript)} tokens "
                f"are dropped due to the insufficient number of frames ({num_frames})."
            )
            break
        token_ids[frame_index] = text_tokenizer.piece_to_id(seg["token"])
        token_count += 1
        if frame_index > 0 and token_ids[frame_index - 1] == text_padding_id:
            # insert end_of_text_padding_id
            token_ids[frame_index - 1] = end_of_text_padding_id
    return token_ids


def worker(process_id: int, dialogue_paths: list[Path], args: argparse.Namespace):
    sp = SentencePieceProcessor(
        hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name)
    )
    pbar = tqdm(dialogue_paths, desc=f"Worker {process_id}", dynamic_ncols=True)
    in_root = Path(args.word_transcript_dir).resolve()
    out_root = Path(args.output_dir).resolve()

    for dialogue_path in pbar:
        rel_path = dialogue_path.relative_to(
            in_root
        )  # 例: 00000-of-01432/cuts.000000/foo.json
        pbar.set_postfix_str(str(rel_path))

        try:
            # ---------- load ----------
            with dialogue_path.open() as f:
                word_transcript = json.load(f)

            # ---------- tokenize ----------
            word_transcript_A = [
                seg for seg in word_transcript if seg["speaker"] == "A"
            ]
            token_ids_A = tokenize_and_pad_text(
                word_transcript=word_transcript_A,
                no_whitespace_before_word=args.no_whitespace_before_word,
                text_tokenizer=sp,
                text_padding_id=args.text_padding_id,
                end_of_text_padding_id=args.end_of_text_padding_id,
                audio_tokenizer_frame_rate=args.audio_tokenizer_frame_rate,
            )
            word_transcript_B = [
                seg for seg in word_transcript if seg["speaker"] == "B"
            ]
            token_ids_B = tokenize_and_pad_text(
                word_transcript=word_transcript_B,
                no_whitespace_before_word=args.no_whitespace_before_word,
                text_tokenizer=sp,
                text_padding_id=args.text_padding_id,
                end_of_text_padding_id=args.end_of_text_padding_id,
                audio_tokenizer_frame_rate=args.audio_tokenizer_frame_rate,
            )

            # ---------- save ----------
            out_path = (out_root / rel_path).with_suffix(".npz")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, A=token_ids_A, B=token_ids_B)

        except Exception:
            logging.exception(f"Failed to process '{dialogue_paths}' — skipped.")
            continue  # 次の JSON へ


def main(args):
    in_root = Path(args.word_transcript_dir).resolve()
    dialogue_paths = list(in_root.rglob("*.json"))  # ← 再帰的に収集

    # --resume オプション
    out_root = Path(args.output_dir).resolve()
    if args.resume:
        tokenized = {p.with_suffix(".npz") for p in out_root.rglob("*.npz")}
        dialogue_paths = [
            p
            for p in dialogue_paths
            if (out_root / p.relative_to(in_root)).with_suffix(".npz") not in tokenized
        ]

    # worker へは Path のリストを渡す
    if args.num_workers == 1:
        worker(0, dialogue_paths, args)
        return

    dialogue_chunks = np.array_split(dialogue_paths, args.num_workers)
    print(
        f"Spawning {args.num_workers} workers – "
        f"{len(dialogue_chunks[0])} dialogues / worker × {args.num_workers}"
    )

    procs: list[mp.Process] = []
    for rank, chunk in enumerate(dialogue_chunks):
        p = mp.Process(target=worker, args=(rank, list(chunk), args), daemon=False)
        p.start()
        procs.append(p)

    # 異常終了を拾う
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker {p.pid} exited with code {p.exitcode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize word-level transcripts using a text tokenizer."
    )
    parser.add_argument(
        "--word_transcript_dir",
        type=str,
        required=True,
        help=(
            "Path to the directory containing the transcripts with word level timestamps. "
            "Each file should contain a list of dictionaries "
            "(`{{'speaker': str, 'start': float, 'end': float, 'word': str}}`) "
            "and the filename should be the same as the corresponding audio file."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save the tokenized data.",
    )

    parser.add_argument(
        "--text_tokenizer_repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="Repository of the text tokenizer.",
    )
    parser.add_argument(
        "--text_tokenizer_name",
        type=str,
        default="tokenizer_spm_32k_3.model",
        help="Name of the text tokenizer.",
    )

    parser.add_argument(
        "--no_whitespace_before_word",
        action="store_true",
        help=(
            "No whitespace before each word. Set this flag if the language "
            "has no whitespace between words (e.g., Japanese and Chinese)."
        ),
    )

    parser.add_argument(
        "--text_padding_id", type=int, default=3, help="Padding id for text."
    )
    parser.add_argument(
        "--end_of_text_padding_id", type=int, default=0, help="End of text padding id."
    )
    parser.add_argument(
        "--audio_tokenizer_frame_rate",
        type=int,
        default=12.5,
        help="Frame rate for the audio tokenizer.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for multiprocessing.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume tokenization.")

    args = parser.parse_args()

    main(args)
