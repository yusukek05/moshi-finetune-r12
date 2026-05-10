"""Self-contained inference for llm-jp-moshi-mstts-v0c-zoom1.

After downloading this repository (`huggingface-cli download
abePclWaseda/llm-jp-moshi-mstts-v0c-zoom1 --local-dir mstts-v0c`), `cd`
into the local directory and run:

    python inference.py \\
        --text-chat sample_dialogue.json \\
        --output-wav out.wav

A 24 kHz stereo wav (left = speaker A, right = speaker B) is written to
`out.wav`.

Required dependencies:
    pip install moshi==0.1.0 sentencepiece soundfile sphn huggingface_hub torch numpy
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import sentencepiece as sp
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders

from data_utils import (
    delay_and_pad_streams,
    find_last_non_padding,
    tokenize_text_chat_for_multistream_tts,
    undelay_tokens,
)
from models import AutoMoshiForFinetuning, MoshiForMultiStreamTTS


def decode_audio_with_mimi(audio_tokens: np.ndarray, mimi) -> np.ndarray:
    """Decode 16-codebook stereo audio tokens (8 per speaker × 2) to a 2-channel waveform."""
    K, _ = audio_tokens.shape
    assert K // 2 == mimi.num_codebooks, (
        f"Codebook mismatch: {K}//2 != {mimi.num_codebooks}"
    )
    device = next(mimi.parameters()).device
    # Split (16, T) into (2, 8, T): two speakers, each with their own 8 codebooks
    tokens = np.stack(np.split(audio_tokens, 2, axis=0), axis=0)
    with torch.no_grad():
        wavs = mimi.decode(torch.from_numpy(tokens).to(device=device)).cpu().numpy()
    return wavs.squeeze(1)  # (2, wav_len)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--text-chat", required=True,
        help='Path to a JSON file in the form `[["A", "..."], ["B", "..."], ...]`.',
    )
    p.add_argument(
        "--output-wav", required=True,
        help="Path to write the generated 24 kHz stereo wav.",
    )
    p.add_argument(
        "--model-dir", default=".",
        help="Directory holding model.safetensors + moshi_lm_kwargs.json (default: current directory).",
    )
    p.add_argument(
        "--prompt-npy", default=None,
        help="Path to the prompt streams .npy. Defaults to prompt_streams_default.npy in --model-dir.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-generation-length", type=int, default=750,
                   help="Maximum number of frames to generate (≈ 1 frame = 80 ms). Default 750 ≈ 60 s.")
    p.add_argument("--text-temperature", type=float, default=0.55)
    p.add_argument("--audio-temperature", type=float, default=0.6)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    os.environ.setdefault("NO_TORCH_COMPILE", "1")
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.model_dtype)

    # 1. Load Moshi LM weights from --model-dir.
    print(f"[1/5] Loading multi-stream TTS model from {args.model_dir} ...", file=sys.stderr)
    moshi_lm = AutoMoshiForFinetuning.from_pretrained(
        save_dir=args.model_dir, device=device, dtype=dtype,
    )
    moshi_lm.eval()
    mstts = MoshiForMultiStreamTTS(moshi_lm=moshi_lm)

    # 2. Load text tokenizer (rinna's japanese-gpt2-medium spiece, 32 k vocab).
    print("[2/5] Loading text tokenizer (rinna/japanese-gpt2-medium) ...", file=sys.stderr)
    tok_path = hf_hub_download("rinna/japanese-gpt2-medium", "spiece.model")
    text_tokenizer = sp.SentencePieceProcessor(tok_path)

    # 3. Load Mimi audio codec (Kyutai).
    print("[3/5] Loading Mimi audio codec ...", file=sys.stderr)
    mimi_path = hf_hub_download(
        "kyutai/moshika-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors",
    )
    mimi = loaders.get_mimi(filename=mimi_path, device=device)

    # 4. Build initial prompt tokens (audio context that conditions the generation).
    prompt_path = args.prompt_npy or os.path.join(args.model_dir, "prompt_streams_default.npy")
    prompt_streams = np.load(prompt_path)
    assert prompt_streams.shape[0] == moshi_lm.num_codebooks, (
        f"Prompt has {prompt_streams.shape[0]} channels but model expects "
        f"{moshi_lm.num_codebooks} (1 text + {moshi_lm.num_audio_codebooks} audio)"
    )
    prompt_padded = delay_and_pad_streams(
        list_of_streams=[prompt_streams],
        delays=moshi_lm.delays,
        initial_token_ids=(
            [moshi_lm.text_initial_token_id]
            + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks
        ),
        padding_token_ids=(
            [moshi_lm.text_padding_token_id]
            + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks
        ),
    )
    prompt_tokens = torch.tensor(prompt_padded[0], device=device, dtype=torch.long).unsqueeze(0)

    # 5. Tokenise the text dialogue and generate.
    text_chat = [text for _, text in json.load(open(args.text_chat))]
    text_tokens = tokenize_text_chat_for_multistream_tts(
        text_chat=text_chat,
        text_tokenizer=text_tokenizer,
        main_speaker_bos_id=1,
        other_speaker_bos_id=2,
        main_speaker_first=False,
    )

    print(f"[4/5] Generating up to {args.max_generation_length} frames ...", file=sys.stderr)
    sample_params = {"use_sampling": True, "top_k": args.top_k, "top_p": args.top_p}
    gen_tokens, gen_lens = mstts.generate(
        list_of_text_tokens=[text_tokens],
        generation_length=args.max_generation_length,
        text_generation_params={**sample_params, "temp": args.text_temperature},
        audio_generation_params={**sample_params, "temp": args.audio_temperature},
        prompt_tokens=prompt_tokens,
        num_speaker_embedding_frames=0,
    )
    undelayed = undelay_tokens(gen_tokens, moshi_lm.delays)
    gen_len = gen_lens[0].item() - max(moshi_lm.delays)
    tokens_i = undelayed[0, :, :gen_len]
    last_idx = find_last_non_padding(tokens_i[0], buffer_len=25)
    final_tokens = tokens_i[:, :last_idx].cpu().numpy()  # (1+16, T)

    # 6. Mimi-decode the audio rows to a stereo wav.
    print("[5/5] Decoding tokens to wav ...", file=sys.stderr)
    audio = decode_audio_with_mimi(final_tokens[1:], mimi)  # rows 1..16 are audio
    sf.write(args.output_wav, audio.astype(np.float32).T, samplerate=mimi.sample_rate)
    print(f"Saved: {args.output_wav}  ({audio.shape[1] / mimi.sample_rate:.1f} s, stereo)", file=sys.stderr)


if __name__ == "__main__":
    main()
