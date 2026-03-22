"""
Moshi server with --initial-audio support.

Extends moshi.server to accept an initial audio file as prompt.
When a client connects, the server first feeds the prompt audio through the model
to prime the context, then processes live microphone input.

Usage:
    module load python/3.12/3.12.9
    module load cuda/12.6/12.6.1

    uv run -m tools.server_with_initial_audio \
        --moshi-weight output/.../model.safetensors \
        --host 0.0.0.0 \
        --port 8998 \
        --tokenizer data/tokenizer/spiece.model \
        --static /path/to/client/dist \
        --initial-audio output/llmjp-zoom1_test_full/prompt_wavs/18.wav
"""

import argparse
import asyncio
import os
import random
import secrets
import sys
import tarfile
import time
from pathlib import Path

import aiohttp
import numpy as np
import sentencepiece
import sphn
import torch
import torchaudio
from aiohttp import web
from huggingface_hub import hf_hub_download

from moshi.client_utils import make_log
from moshi.models import MimiModel, LMModel, LMGen, loaders


def log(level: str, msg: str):
    print(make_log(level, msg))


def seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def load_initial_audio_pcm(
    wav_path: str,
    mimi: MimiModel,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Load WAV file and return PCM as mono tensor at mimi's sample rate.
    Returns None if loading fails.
    """
    if not os.path.exists(wav_path):
        log("error", f"Initial audio file not found: {wav_path}")
        return None

    try:
        wav, sr = torchaudio.load(wav_path)
    except Exception as e:
        log("error", f"Failed to load {wav_path}: {e}")
        return None

    # Use first channel if stereo
    if wav.dim() == 2:
        wav = wav[0]
    wav = wav.to(device)

    if sr != mimi.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, mimi.sample_rate).to(device)
        wav = resampler(wav)

    return wav


def create_chat_loops(
    state: "ServerStateWithPrompt",
    opus_reader: sphn.OpusStreamReader,
    opus_writer: sphn.OpusStreamWriter,
    ws: web.WebSocketResponse,
    close: list[bool],
):
    """Create recv, opus, and send loops for handle_chat."""

    async def recv_loop():
        try:
            async for message in ws:
                if message.type == aiohttp.WSMsgType.ERROR:
                    log("error", f"{ws.exception()}")
                    break
                elif message.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif message.type != aiohttp.WSMsgType.BINARY:
                    log("error", f"unexpected message type {message.type}")
                    continue
                message = message.data
                if not isinstance(message, bytes):
                    log("error", f"unsupported message type {type(message)}")
                    continue
                if len(message) == 0:
                    log("warning", "empty message")
                    continue
                kind = message[0]
                if kind == 1:  # audio
                    payload = message[1:]
                    opus_reader.append_bytes(payload)
                else:
                    log("warning", f"unknown message kind {kind}")
        finally:
            close[0] = True
            log("info", "connection closed")

    async def opus_loop():
        all_pcm_data = None
        prompt_consumed = state.initial_audio_pcm is None

        while True:
            if close[0]:
                return
            await asyncio.sleep(0.001)

            # Phase 1: Feed initial prompt audio before live input
            if not prompt_consumed and state.initial_audio_pcm is not None:
                prompt_pcm = state.initial_audio_pcm
                frame_size = state.frame_size
                num_frames = prompt_pcm.shape[-1] // frame_size

                if num_frames > 0:
                    log("info", f"Feeding initial prompt: {num_frames} frames ({num_frames * 0.08:.1f}s)")
                    for i in range(num_frames):
                        if close[0]:
                            return
                        if i % 10 == 0:
                            await asyncio.sleep(0)
                        chunk = prompt_pcm[i * frame_size : (i + 1) * frame_size]
                        chunk = chunk[None, None]
                        codes = state.mimi.encode(chunk)
                        for c in range(codes.shape[-1]):
                            tokens = state.lm_gen.step(codes[:, :, c : c + 1])
                            if tokens is None:
                                continue
                            main_pcm = state.mimi.decode(tokens[:, 1:])
                            main_pcm = main_pcm.cpu()
                            opus_writer.append_pcm(main_pcm[0, 0].numpy())
                            text_token = tokens[0, 0, 0].item()
                            if text_token not in (0, 3):
                                _text = state.text_tokenizer.id_to_piece(text_token)
                                _text = _text.replace("▁", " ")
                                msg = b"\x02" + bytes(_text, encoding="utf8")
                                await ws.send_bytes(msg)
                    log("info", "Initial prompt consumed, switching to live audio")

                prompt_consumed = True
                continue

            # Phase 2: Process live microphone input
            pcm = opus_reader.read_pcm()
            if pcm.shape[-1] == 0:
                continue
            if all_pcm_data is None:
                all_pcm_data = pcm
            else:
                all_pcm_data = np.concatenate((all_pcm_data, pcm))
            while all_pcm_data.shape[-1] >= state.frame_size:
                be = time.time()
                chunk = all_pcm_data[: state.frame_size]
                all_pcm_data = all_pcm_data[state.frame_size :]
                chunk = torch.from_numpy(chunk)
                chunk = chunk.to(device=state.device)[None, None]
                codes = state.mimi.encode(chunk)
                for c in range(codes.shape[-1]):
                    tokens = state.lm_gen.step(codes[:, :, c : c + 1])
                    if tokens is None:
                        continue
                    assert tokens.shape[1] == state.lm_gen.lm_model.dep_q + 1
                    main_pcm = state.mimi.decode(tokens[:, 1:])
                    main_pcm = main_pcm.cpu()
                    opus_writer.append_pcm(main_pcm[0, 0].numpy())
                    text_token = tokens[0, 0, 0].item()
                    if text_token not in (0, 3):
                        _text = state.text_tokenizer.id_to_piece(text_token)
                        _text = _text.replace("▁", " ")
                        msg = b"\x02" + bytes(_text, encoding="utf8")
                        log("info", f"text token '{_text}'")
                        await ws.send_bytes(msg)
                log("info", f"frame handled in {1000 * (time.time() - be):.1f}ms")

    async def send_loop():
        while True:
            if close[0]:
                return
            await asyncio.sleep(0.001)
            msg = opus_writer.read_bytes()
            if len(msg) > 0:
                await ws.send_bytes(b"\x01" + msg)

    return recv_loop, opus_loop, send_loop


class ServerStateWithPrompt:
    """Server state with optional initial audio prompt."""

    def __init__(
        self,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        device: str | torch.device,
        initial_audio_path: str | None = None,
    ):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm_gen = LMGen(lm)
        self.device = device
        self.frame_size = int(mimi.sample_rate / mimi.frame_rate)
        self.lock = asyncio.Lock()

        self.initial_audio_pcm: torch.Tensor | None = None
        if initial_audio_path:
            self.initial_audio_pcm = load_initial_audio_pcm(
                initial_audio_path, mimi, torch.device(device)
            )
            if self.initial_audio_pcm is not None:
                log("info", f"Loaded initial audio prompt: {initial_audio_path}")

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=torch.float32, device=self.device
            )
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])
        torch.cuda.synchronize()

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
        opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
        close = [False]

        recv_loop, opus_loop, send_loop = create_chat_loops(
            self, opus_reader, opus_writer, ws, close
        )

        log("info", "accepted connection")
        async with self.lock:
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            await ws.send_bytes(b"\x00")
            await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        log("info", "done with connection")
        return ws


def main():
    parser = argparse.ArgumentParser(
        description="Moshi server with --initial-audio prompt support"
    )
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action="store_true")
    parser.add_argument("--gradio-tunnel-token", type=str)

    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to Moshi checkpoint.")
    parser.add_argument("--mimi-weight", type=str, help="Path to Mimi checkpoint.")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HuggingFace repo for model artifacts.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--initial-audio",
        type=str,
        default=None,
        help="Path to WAV file to use as initial audio prompt (24kHz mono preferred).",
    )

    args = parser.parse_args()
    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ""
    if args.gradio_tunnel:
        try:
            from gradio import networking
        except ImportError:
            log("error", "pip install gradio required for --gradio-tunnel")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        tunnel_token = args.gradio_tunnel_token or secrets.token_urlsafe(32)

    log("info", "loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    log("info", "mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    log("info", "loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, args.device)
    log("info", "moshi loaded")

    state = ServerStateWithPrompt(
        mimi, text_tokenizer, lm, args.device, args.initial_audio
    )
    log("info", "warming up")
    state.warmup()

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    static_path = None
    if args.static is None:
        log("info", "retrieving static content")
        dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz")
        dist = Path(dist_tgz).parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=Path(dist_tgz).parent)
        static_path = str(dist)
    elif args.static != "none":
        static_path = args.static

    if static_path:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        log("info", f"serving static from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")

    log("info", f"Access Web UI at http://{args.host}:{args.port}")
    if args.initial_audio:
        log("info", f"Initial audio prompt: {args.initial_audio}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel("localhost", args.port, tunnel_token, None)
        log("info", f"Tunnel started: {tunnel}")

    with torch.no_grad():
        web.run_app(app, port=args.port)


if __name__ == "__main__":
    main()
