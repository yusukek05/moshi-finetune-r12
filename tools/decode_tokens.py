import argparse
import multiprocessing as mp
import os
from glob import glob

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm


def decode_text(
    text_tokens: np.ndarray,
    tokenizer: SentencePieceProcessor,
    text_padding_id: int = 3,
    end_of_text_padding_id: int = 0,
) -> str:
    """
    Decode the inner-monologue text stream into a clean transcript.

    Drops the padding frames (``text_padding_id``) and the word-boundary
    markers (``end_of_text_padding_id``) — inverse of
    ``tools/tokenize_text_from_dir.py`` — then decodes the remaining
    SentencePiece word-piece ids. Without this filtering, ids 3 and 0 would be
    decoded as literal vocab pieces and corrupt the text.

    Args:
        text_tokens (np.ndarray): Text stream. Shape: (seq_len,)
        tokenizer (SentencePieceProcessor): SentencePiece tokenizer (for the
            v1/v1.1/v1.2 lineage this is rinna/japanese-gpt2-medium spiece.model).
        text_padding_id (int): Padding id for no-token frames (rinna: 3).
        end_of_text_padding_id (int): Word-boundary marker id (rinna: 0).
    Returns:
        str: Decoded transcript.
    """
    kept = [
        int(t)
        for t in text_tokens.tolist()
        if int(t) != text_padding_id and int(t) != end_of_text_padding_id
    ]
    return tokenizer.decode(kept)


def decode_audio(audio_tokens: np.ndarray, mimi: loaders.MimiModel) -> np.ndarray:
    """
    Decode audio tokens.
    Args:
        audio_tokens (np.ndarray): Audio tokens. Shape: (K, T).
            K: Number of audio codebooks, T: Number of audio tokens.
        mimi (loaders.MimiModel): Mimi model.
    Returns:
        np.ndarray: Decoded audio. Shape: (C=2, wav_len).
            C: Number of audio channels, wav_len: Length of the audio waveform.

    Channel mapping convention: with default inference
    (`main_speaker_first=False`) and a text_chat starting at speaker A,
    mstts places A's audio at rows K/2..K (the "other" position) and B's
    at rows 0..K/2 (the "main" position). We reorder so output channel 0
    (LEFT) = first speaker (A), matching the project-wide L=A, R=B convention.
    """
    K, T = audio_tokens.shape
    assert K // 2 == mimi.num_codebooks, (
        f"Number of codebooks mismatch: {K}//2 != {mimi.num_codebooks}"
    )

    device = next(mimi.parameters()).device
    main_part, other_part = np.split(audio_tokens, 2, axis=0)
    # other (= A, first speaker) → channel 0 ; main (= B) → channel 1
    tokens = np.stack([other_part, main_part], axis=0)  # (2, K/2, T)
    with torch.no_grad():
        # use batch dimension of mimi as channel dimension
        wavs = mimi.decode(torch.from_numpy(tokens).to(device=device)).cpu().numpy()
        # wavs: (2, 1, wav_len)
    wav = wavs.squeeze(1)  # (2, wav_len) — channel 0 = A, channel 1 = B
    return wav


def decode_tokens(rank: int, list_of_tokens_path: list[str], args: argparse.Namespace):
    """
    Decode tokens.
    Args:
        rank (int): Rank of the process.
        list_of_tokens_path (list[str]): List of paths to the token files to decode.
            each file should be `*.npy` file.
        args (argparse.Namespace): Arguments.
    """
    # Load the text tokenizer (only needed when also saving transcripts)
    text_tokenizer = SentencePieceProcessor(
        hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name)
    )

    # Load the audio tokenizer
    mimi = loaders.get_mimi(
        filename=hf_hub_download(args.audio_tokenizer_repo, args.audio_tokenizer_name),
        device=torch.device("cuda", rank),
    )

    if args.text_output_dir is not None:
        os.makedirs(args.text_output_dir, exist_ok=True)

    for tokens_path in tqdm(list_of_tokens_path, desc=f"Rank {rank}"):
        tokens = np.load(tokens_path)  # (1+K, T)

        text_tokens = tokens[0]
        audio_tokens = tokens[1:]

        audio = decode_audio(audio_tokens, mimi)  # (2, wav_len)

        output_wav_path = os.path.join(
            args.output_dir, os.path.basename(tokens_path).replace(".npy", ".wav")
        )
        # save the audio
        sf.write(output_wav_path, audio.astype(np.float32).T, samplerate=mimi.sample_rate)

        # optionally save the noise-free inner-monologue transcript (for the
        # text-based meaningfulness reward). Gated: off unless --text_output_dir.
        if args.text_output_dir is not None:
            text = decode_text(
                text_tokens,
                text_tokenizer,
                text_padding_id=args.text_padding_id,
                end_of_text_padding_id=args.end_of_text_padding_id,
            )
            text_path = os.path.join(
                args.text_output_dir,
                os.path.basename(tokens_path).replace(".npy", ".txt"),
            )
            with open(text_path, "w") as f:
                f.write(text + "\n")


def main(args):
    # Get the list of token files
    list_of_tokens_path = glob(os.path.join(args.tokens_dir, "*.npy"))

    num_files = len(list_of_tokens_path)
    print(f"Number of token files: {num_files}")

    # Split the list of token files
    file_indices_per_rank = np.array_split(np.arange(num_files), args.num_workers)

    # Make the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Start the processes
    processes = []
    for rank, file_indices in enumerate(file_indices_per_rank):
        list_of_tokens_path_per_rank = [list_of_tokens_path[i] for i in file_indices]
        p = mp.Process(target=decode_tokens, args=(rank, list_of_tokens_path_per_rank, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens_dir",
        type=str,
        required=True,
        help="Directory containing the token files to decode. Each file should be `*.npy` file.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the decoded audio (wav)"
    )
    parser.add_argument(
        "--text_output_dir",
        type=str,
        default=None,
        help=(
            "If set, also decode the inner-monologue text stream and save one "
            "<id>.txt transcript per example here (noise-free; for the text-based "
            "meaningfulness reward). For the v1/v1.1/v1.2 lineage pass "
            "--text_tokenizer_repo rinna/japanese-gpt2-medium --text_tokenizer_name spiece.model."
        ),
    )
    parser.add_argument(
        "--text_tokenizer_repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="Hugging Face Hub repository for the text tokenizer.",
    )
    parser.add_argument(
        "--text_tokenizer_name",
        type=str,
        default="tokenizer_spm_32k_3.model",
        help="Model name of the text tokenizer.",
    )
    parser.add_argument(
        "--text_padding_id", type=int, default=3,
        help="Text-stream padding id to drop when decoding (rinna: 3).",
    )
    parser.add_argument(
        "--end_of_text_padding_id", type=int, default=0,
        help="Text-stream word-boundary marker id to drop when decoding (rinna: 0).",
    )
    parser.add_argument(
        "--audio_tokenizer_repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="Hugging Face Hub repository for the audio tokenizer.",
    )
    parser.add_argument(
        "--audio_tokenizer_name",
        type=str,
        default="tokenizer-e351c8d8-checkpoint125.safetensors",
        help="Model name of the audio tokenizer.",
    )

    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for multiprocessing."
    )

    args = parser.parse_args()

    main(args)
