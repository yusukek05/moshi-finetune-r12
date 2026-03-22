import argparse
import multiprocessing as mp
import os

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import MimiModel, loaders
from tqdm import tqdm
from pathlib import Path


def ceil(x, y):
    return int(-(-x // y))


def tokenize_audio(
    wav: torch.Tensor,
    mimi: MimiModel,
    audio_chunk_size: int,
) -> torch.LongTensor:
    """
    Tokenize the audio of a single channel.
    """
    assert wav.dim() == 1, f"Expected 1D tensor, got {wav.dim()}D tensor."

    wav_chunk_size = audio_chunk_size * mimi.sample_rate
    num_chunks = ceil(wav.shape[0], wav_chunk_size)
    device = next(mimi.parameters()).device

    list_of_audio_ids = []
    for i in range(num_chunks):
        wav_chunk = wav[i * wav_chunk_size : (i + 1) * wav_chunk_size]
        with torch.no_grad():
            list_of_audio_ids.append(
                mimi.encode(
                    wav_chunk.reshape(1, 1, -1).to(device)
                ).cpu()  # [B=1, K=8, T_chunk]
            )
    audio_ids = torch.cat(list_of_audio_ids, dim=-1)  # [B=1, K=8, T]
    audio_ids = audio_ids[0]  # [K=8, T]

    num_frames = ceil(wav.shape[-1], (mimi.sample_rate / mimi.frame_rate))
    assert audio_ids.shape == (
        mimi.num_codebooks,
        num_frames,
    ), f"{audio_ids.shape} != ({mimi.num_codebooks}, {num_frames})"
    return audio_ids


def worker(process_id: int, wav_paths: list[Path], args: argparse.Namespace):
    device = torch.device("cuda", process_id)
    mimi = loaders.get_mimi(
        filename=hf_hub_download(args.audio_tokenizer_repo, args.audio_tokenizer_name),
        device=device,
    )
    pbar = tqdm(wav_paths, desc=f"Worker {process_id}", dynamic_ncols=True)

    in_root = Path(args.audio_dir).resolve()
    out_root = Path(args.output_dir).resolve()

    for wav_path in pbar:
        rel_path = wav_path.relative_to(
            in_root
        )  # 例: 00000-of-01432/cuts.000000/foo.wav
        dialogue_name = rel_path.with_suffix("").as_posix()  # 進捗表示用
        pbar.set_postfix_str(dialogue_name)

        # load audio
        wavs, sr = torchaudio.load(wav_path)
        assert (
            wavs.shape[0] == 2
        ), f"Expected stereo audio, got {wavs.shape[0]} channels."
        resampler = torchaudio.transforms.Resample(sr, mimi.sample_rate).to(device)
        wavs = resampler(wavs.to(device))

        # tokenize audio
        audio_ids_A = tokenize_audio(wavs[0], mimi, args.audio_chunk_size)
        audio_ids_B = tokenize_audio(wavs[1], mimi, args.audio_chunk_size)

        # save tokenized audio
        out_path = (out_root / rel_path).with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(out_path, A=audio_ids_A.numpy(), B=audio_ids_B.numpy())
        except Exception as e:
            print(f"Failed to save {out_path}: {e}")
            if out_path.exists():
                out_path.unlink()


def main(args):
    in_root = Path(args.audio_dir).resolve()
    wav_paths = list(in_root.rglob("*.wav"))

    out_root = Path(args.output_dir).resolve()
    if args.resume:
        tokenized = {p.with_suffix(".npz") for p in out_root.rglob("*.npz")}
        wav_paths = [
            p
            for p in wav_paths
            if (out_root / p.relative_to(in_root)).with_suffix(".npz") not in tokenized
        ]
        print(f"Skipping {len(tokenized)} already tokenized dialogues.")

    if args.num_workers == 1:
        worker(0, wav_paths, args)

    else:
        num_devices = torch.cuda.device_count()
        if args.num_workers > num_devices:
            print(
                f"Number of workers ({args.num_workers}) exceeds number of available GPUs ({num_devices})."
            )
            args.num_workers = num_devices
            print(f"Using {args.num_workers} workers.")

        chunks = np.array_split(np.array(wav_paths), args.num_workers)
        print(
            f"Each of {args.num_workers} workers processes {len(chunks[0])} dialogues."
        )

        processes = []
        for i, chunk in enumerate(chunks):
            p = mp.Process(target=worker, args=(i, list(chunk), args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize audio files using a pretrained audio tokenizer."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help=(
            "Path to the directory containing the stereo wav files. "
            "Left and right channels should be the audio of speaker A and B respectively. "
            "and filenames should be the same as the dialogue names in the word transcript directory."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save the tokenized data.",
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
        help="Model name for the audio tokenizer.",
    )

    parser.add_argument(
        "--audio_chunk_size",
        type=int,
        default=1200,
        help="Split audio into chunks of this size (seconds) to fit into cuda memory.",
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
