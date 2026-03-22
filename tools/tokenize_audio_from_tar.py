import argparse
import io
import multiprocessing as mp
import tarfile

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import MimiModel, loaders
from tqdm import tqdm
from pathlib import Path


def ceil(x, y):
    return int(-(-x // y))


def load_wav_from_tar(tar_path: Path, internal_path: str) -> tuple[torch.Tensor, int]:
    """
    Load a WAV file from within a tar archive.
    Returns (wav_tensor, sample_rate).
    """
    with tarfile.open(tar_path, "r") as tar:
        member = tar.getmember(internal_path)
        file_obj = tar.extractfile(member)
        if file_obj is None:
            raise ValueError(f"Could not extract {internal_path} from {tar_path}")
        wav_bytes = file_obj.read()
        # Create a BytesIO object for torchaudio
        wav_io = io.BytesIO(wav_bytes)
        wavs, sr = torchaudio.load(wav_io, format="wav")
        return wavs, sr


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


def worker(process_id: int, wav_items: list[Path | tuple[Path, str]], args: argparse.Namespace):
    device = torch.device("cuda", process_id)
    mimi = loaders.get_mimi(
        filename=hf_hub_download(args.audio_tokenizer_repo, args.audio_tokenizer_name),
        device=device,
    )
    pbar = tqdm(wav_items, desc=f"Worker {process_id}", dynamic_ncols=True)

    in_root = Path(args.audio_dir).resolve()
    out_root = Path(args.output_dir).resolve()

    for wav_item in pbar:
        # Determine if this is a regular file path or a (tar_path, internal_path) tuple
        if isinstance(wav_item, tuple):
            # WebDataset tar file format: (tar_path, internal_wav_path)
            tar_path, internal_wav_path = wav_item
            dialogue_name = Path(internal_wav_path).with_suffix("").as_posix()
            pbar.set_postfix_str(dialogue_name)

            # Load audio from tar file
            wavs, sr = load_wav_from_tar(tar_path, internal_wav_path)
            
            # Generate output path: output_dir/internal_path.npz
            # (shard name is not included to match tokenize_text.py output structure)
            out_rel_path = Path(internal_wav_path).with_suffix(".npz")
            out_path = out_root / out_rel_path
        else:
            # Regular file path
            wav_path = wav_item
            rel_path = wav_path.relative_to(in_root)
            dialogue_name = rel_path.with_suffix("").as_posix()
            pbar.set_postfix_str(dialogue_name)

            # Load audio from regular file
            wavs, sr = torchaudio.load(wav_path)
            out_path = (out_root / rel_path).with_suffix(".npz")

        assert (
            wavs.shape[0] == 2
        ), f"Expected stereo audio, got {wavs.shape[0]} channels."
        resampler = torchaudio.transforms.Resample(sr, mimi.sample_rate).to(device)
        wavs = resampler(wavs.to(device))

        # tokenize audio
        audio_ids_A = tokenize_audio(wavs[0], mimi, args.audio_chunk_size)
        audio_ids_B = tokenize_audio(wavs[1], mimi, args.audio_chunk_size)

        # save tokenized audio
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(out_path, A=audio_ids_A.numpy(), B=audio_ids_B.numpy())
        except Exception as e:
            print(f"Failed to save {out_path}: {e}")
            if out_path.exists():
                out_path.unlink()


def collect_wav_items(in_root: Path) -> list[Path | tuple[Path, str]]:
    """
    Collect WAV files from either regular directories or tar files.
    Returns a list of either Path objects (for regular files) or
    (tar_path, internal_path) tuples (for tar files).
    """
    wav_items: list[Path | tuple[Path, str]] = []
    
    # First, check for tar files (WebDataset format)
    tar_files = sorted(set(in_root.rglob("*.tar")))
    
    if tar_files:
        print(f"Found {len(tar_files)} tar file(s). Processing as WebDataset format.")
        for tar_path in tar_files:
            try:
                with tarfile.open(tar_path, "r") as tar:
                    for member in tar.getmembers():
                        if member.name.endswith(".wav") and member.isfile():
                            wav_items.append((tar_path, member.name))
            except Exception as e:
                print(f"Warning: Failed to read {tar_path}: {e}")
                continue
        print(f"Found {len(wav_items)} WAV files in tar archives.")
    else:
        # No tar files found, look for regular WAV files
        wav_paths = list(in_root.rglob("*.wav"))
        wav_items = wav_paths
        print(f"Found {len(wav_paths)} regular WAV file(s).")
    
    return wav_items


def get_out_path_for_item(
    wav_item: Path | tuple[Path, str], in_root: Path, out_root: Path
) -> Path:
    """Get the output path for a given WAV item (regular file or tar entry)."""
    if isinstance(wav_item, tuple):
        tar_path, internal_wav_path = wav_item
        # Use internal path directly (without shard name) to match tokenize_text.py structure
        out_rel_path = Path(internal_wav_path).with_suffix(".npz")
        return out_root / out_rel_path
    else:
        rel_path = wav_item.relative_to(in_root)
        return (out_root / rel_path).with_suffix(".npz")


def main(args):
    in_root = Path(args.audio_dir).resolve()
    wav_items = collect_wav_items(in_root)

    out_root = Path(args.output_dir).resolve()
    if args.resume:
        tokenized = {p.with_suffix(".npz") for p in out_root.rglob("*.npz")}
        wav_items = [
            item
            for item in wav_items
            if get_out_path_for_item(item, in_root, out_root) not in tokenized
        ]
        print(f"Skipping {len(tokenized)} already tokenized dialogues.")

    if args.num_workers == 1:
        worker(0, wav_items, args)

    else:
        num_devices = torch.cuda.device_count()
        if args.num_workers > num_devices:
            print(
                f"Number of workers ({args.num_workers}) exceeds number of available GPUs ({num_devices})."
            )
            args.num_workers = num_devices
            print(f"Using {args.num_workers} workers.")

        # Split list into chunks manually to handle mixed types
        chunk_size = len(wav_items) // args.num_workers
        chunks = [
            wav_items[i * chunk_size : (i + 1) * chunk_size]
            for i in range(args.num_workers - 1)
        ]
        chunks.append(wav_items[(args.num_workers - 1) * chunk_size :])
        print(
            f"Each of {args.num_workers} workers processes approximately {len(chunks[0])} dialogues."
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
            "Can be either a directory with regular .wav files or a directory with .tar files (WebDataset format). "
            "Left and right channels should be the audio of speaker A and B respectively. "
            "Filenames should be the same as the dialogue names in the word transcript directory."
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
