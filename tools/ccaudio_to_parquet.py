"""ccaudio (Lhotse cutset) → v1 mono parquet shard.

Process one slice of /groups/gcg51557/experiments/0167_cc_audio/asai/
ccaudio_rss_transcribed_array/{slice_idx}/{cuts.NNNNNN.jsonl.gz,
recording.NNNNNN.tar}, drop hallucinated cuts, Mimi+rinna_gpt2 tokenize the
survivors, and write a single shard-{slice_idx:05d}.parquet whose schema
matches J-CHAT-mono / reazonspeech (__key__, A_text, A_audio).
"""

import argparse
import gzip
import io
import json
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import MimiModel, loaders
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm


CC_JA_RANGES = (
    (0x3040, 0x309F),  # hiragana
    (0x30A0, 0x30FF),  # katakana
    (0x4E00, 0x9FFF),  # CJK
)
CC_JA_PUNCT = set("、。！？「」（）　 ・…ー〜")


def _is_ja_char(ch: str) -> bool:
    if ch in CC_JA_PUNCT:
        return True
    c = ord(ch)
    for lo, hi in CC_JA_RANGES:
        if lo <= c <= hi:
            return True
    return False


def non_japanese_ratio(text: str) -> float:
    """Fraction of 'content' chars that aren't Japanese.

    Content chars exclude whitespace and ASCII digits/punctuation (those are
    neutral — Japanese transcripts legitimately contain numbers and marks).
    ASCII *letters* DO count, and as non-Japanese: a transcript that is mostly
    Latin letters is a hallucination (French/English), not a Japanese cut with
    a few English words (those stay a small fraction and keep the ratio low).

    NOTE: the previous version skipped every `ch.isascii()` char, so a 100%
    French/English hallucination scored 0.0 and sailed through the filter.
    """
    n = 0
    bad = 0
    for ch in text:
        if ch.isspace():
            continue
        if ch.isascii() and not ch.isalpha():
            continue  # ASCII digits / punctuation: neutral
        n += 1
        if not _is_ja_char(ch):
            bad += 1
    return bad / n if n else 0.0


def max_ngram_repetition_ratio(text: str, n: int = 4) -> float:
    """Fraction of text covered by the most common 4-gram (length-normalized).

    Long radio podcasts naturally have moderate 4-gram counts from station IDs
    etc. An absolute threshold over-penalizes long good cuts; the ratio holds
    across cut lengths.
    """
    if len(text) < n:
        return 0.0
    c = Counter(text[i : i + n] for i in range(len(text) - n + 1))
    top = c.most_common(1)[0][1]
    return top / len(text)


def is_hallucinated(text: str, rep_ratio_thresh: float, non_ja_thresh: float) -> tuple[bool, str]:
    rep = max_ngram_repetition_ratio(text, 4)
    if rep > rep_ratio_thresh:
        return True, f"4gram_rep_ratio={rep:.3f}>{rep_ratio_thresh}"
    nja = non_japanese_ratio(text)
    if nja > non_ja_thresh:
        return True, f"non_ja_ratio={nja:.3f}>{non_ja_thresh}"
    return False, ""


def load_flac_from_tar(tar: tarfile.TarFile, name: str) -> tuple[torch.Tensor, int]:
    member = tar.getmember(name)
    fobj = tar.extractfile(member)
    if fobj is None:
        raise ValueError(f"Could not extract {name} from {tar.name}")
    buf = io.BytesIO(fobj.read())
    return torchaudio.load(buf)


def ceil_div(x: int, y: int) -> int:
    return -(-x // y)


def tokenize_audio_chunked(
    wav: torch.Tensor,
    mimi: MimiModel,
    audio_chunk_size_s: int,
) -> torch.LongTensor:
    """Encode a 1D mono waveform at mimi.sample_rate to [K, T_frames]."""
    assert wav.dim() == 1
    device = next(mimi.parameters()).device
    wav_chunk = audio_chunk_size_s * mimi.sample_rate
    n = ceil_div(wav.shape[0], wav_chunk)
    out = []
    for i in range(n):
        seg = wav[i * wav_chunk : (i + 1) * wav_chunk]
        with torch.no_grad():
            out.append(mimi.encode(seg.reshape(1, 1, -1).to(device)).cpu())
    ids = torch.cat(out, dim=-1)[0]  # [K, T]
    expected_frames = ceil_div(wav.shape[-1], int(mimi.sample_rate / mimi.frame_rate))
    assert ids.shape == (mimi.num_codebooks, expected_frames), (
        f"{ids.shape} vs ({mimi.num_codebooks}, {expected_frames})"
    )
    return ids


def tokenize_text_aligned(
    words: list[list],  # [[word, start, duration, conf], ...] from ccaudio alignment
    sp: SentencePieceProcessor,
    text_padding_id: int,
    end_of_text_padding_id: int,
    frame_rate: float,
) -> list[int]:
    """Same shape as tools/tokenize_text.tokenize_and_pad_text but tailored to
    ccaudio's alignment format. Single speaker (mono)."""
    if not words:
        return []
    words = sorted(words, key=lambda w: w[1])
    # add whitespace before all words but the first
    parts = []
    for i, (w, _s, _d, _c) in enumerate(words):
        parts.append(w.strip() if i == 0 else " " + w.strip())
    text = "".join(parts)

    # piece-level tokenization (no byte fallback) using rinna spiece
    pieces = _encode_pieces_no_byte(sp, text)

    # char-level transcript reconstruction from word transcript
    char_segs: list[tuple[float, float, str]] = []  # (start, end, char)
    for word, (_, s, d, _c) in zip(parts, words):
        if not word:
            continue
        per = d / max(1, len(word))
        for i, ch in enumerate(word):
            char_segs.append((s + i * per, s + (i + 1) * per, ch))

    # align pieces -> char ranges
    token_transcript: list[tuple[float, float, str]] = []  # (start, end, piece)
    cursor = 0
    for i, piece in enumerate(pieces):
        if i == 0 and piece == "▁":
            continue
        if i == 0 and piece.startswith("▁"):
            take = len(piece) - 1
        else:
            take = len(piece)
        if cursor + take > len(char_segs):
            # alignment exhausted — drop the rest silently
            break
        chunk = char_segs[cursor : cursor + take]
        cursor += take
        token_transcript.append((chunk[0][0], chunk[-1][1], piece))

    if not token_transcript:
        return []

    seconds_per_frame = 1.0 / frame_rate
    num_frames = int((token_transcript[-1][1] + 1) * frame_rate)
    token_ids = [text_padding_id] * num_frames
    for start, _end, piece in token_transcript:
        idx = int(start // seconds_per_frame)
        while idx < num_frames and token_ids[idx] != text_padding_id:
            idx += 1
        if idx >= num_frames:
            break
        token_ids[idx] = sp.piece_to_id(piece)
        if idx > 0 and token_ids[idx - 1] == text_padding_id:
            token_ids[idx - 1] = end_of_text_padding_id
    return token_ids


def _encode_pieces_no_byte(sp: SentencePieceProcessor, text: str) -> list[str]:
    pieces = sp.encode_as_pieces(text)
    if not pieces:
        return []
    out = []
    last_bytes = []
    for tok in pieces:
        if not tok.startswith("<0x"):
            out.append(tok)
            text = text[len(tok) :]
        else:
            last_bytes.append(tok)
            decoded = sp.decode_pieces(last_bytes)
            if text.startswith(decoded):
                out.append(decoded)
                text = text[len(decoded) :]
                last_bytes = []
    if last_bytes:
        raise ValueError(f"Unresolved byte fallback tokens: {last_bytes!r}")
    return out


def merge_text_audio(
    text_ids: np.ndarray, audio_ids: np.ndarray, text_padding_id: int
) -> tuple[list[int], list[list[int]]]:
    """Truncate/pad text to T_audio. Returns (A_text [T], A_audio [K][T])."""
    assert text_ids.ndim == 1 and audio_ids.ndim == 2
    T = audio_ids.shape[1]
    if text_ids.shape[0] > T:
        text_ids = text_ids[:T]
    elif text_ids.shape[0] < T:
        pad = np.full(T - text_ids.shape[0], text_padding_id, dtype=text_ids.dtype)
        text_ids = np.concatenate([text_ids, pad])
    return text_ids.astype(np.int64).tolist(), audio_ids.astype(np.int64).tolist()


def iter_cuts(cc_slice_dir: Path):
    for cuts_path in sorted(cc_slice_dir.glob("cuts.*.jsonl.gz")):
        idx = cuts_path.stem.split(".")[1]  # "000000" from cuts.000000.jsonl.gz
        tar_path = cc_slice_dir / f"recording.{idx}.tar"
        if not tar_path.exists():
            print(f"[warn] missing tar: {tar_path}")
            continue
        with gzip.open(cuts_path, "rt") as f, tarfile.open(tar_path, "r") as tar:
            for line in f:
                yield idx, json.loads(line), tar


def main(args: argparse.Namespace) -> None:
    slice_dir = Path(args.cc_root) / str(args.slice_idx)
    out_path = Path(args.output_dir) / f"shard-{args.slice_idx:05d}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path} exists. Use --overwrite to rebuild.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    print("[info] loading Mimi …")
    mimi = loaders.get_mimi(
        filename=hf_hub_download(args.audio_tokenizer_repo, args.audio_tokenizer_name),
        device=device,
    )
    # loaders.get_mimi() hard-codes set_num_codebooks(8); v1 mono parquets
    # (J-CHAT-mono, reazonspeech) are stored at q16 → match that.
    mimi.set_num_codebooks(args.num_codebooks)
    print(f"[info] mimi sr={mimi.sample_rate} fr={mimi.frame_rate} K={mimi.num_codebooks}")

    print("[info] loading rinna SP tokenizer …")
    sp = SentencePieceProcessor(
        hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name)
    )

    rows_key: list[str] = []
    rows_text: list[list[int]] = []
    rows_audio: list[list[list[int]]] = []

    n_total = n_kept = n_drop_text = n_drop_err = 0
    t0 = time.time()
    pbar = tqdm(iter_cuts(slice_dir), desc=f"slice {args.slice_idx}", unit="cut")
    for file_idx, cut, tar in pbar:
        n_total += 1
        cut_id = cut["id"]
        sup = cut["supervisions"][0]
        text = sup.get("text", "") or ""
        bad, reason = is_hallucinated(
            text, args.rep_ratio_thresh, args.non_ja_thresh
        )
        if bad:
            n_drop_text += 1
            pbar.set_postfix_str(f"kept={n_kept} drop_text={n_drop_text} drop_err={n_drop_err}")
            continue
        try:
            wav, sr = load_flac_from_tar(tar, f"{cut_id}.flac")
            # ccaudio recordings are stereo with channel 0 supervised → keep ch 0
            if wav.shape[0] >= 2:
                wav = wav[:1]
            wav = wav.to(device)
            if sr != mimi.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, mimi.sample_rate)
            wav = wav[0]  # 1-D
            audio_ids = tokenize_audio_chunked(wav, mimi, args.audio_chunk_size_s).numpy()

            words = sup.get("alignment", {}).get("word", []) or []
            text_ids = tokenize_text_aligned(
                words,
                sp,
                text_padding_id=args.text_padding_id,
                end_of_text_padding_id=args.end_of_text_padding_id,
                frame_rate=mimi.frame_rate,
            )
            text_ids_np = np.asarray(text_ids, dtype=np.int64)
            text_list, audio_list = merge_text_audio(
                text_ids_np, audio_ids, text_padding_id=args.text_padding_id
            )
            rows_key.append(f"ccaudio/slice{args.slice_idx:03d}/file{file_idx}/{cut_id}")
            rows_text.append(text_list)
            rows_audio.append(audio_list)
            n_kept += 1
        except Exception as e:
            n_drop_err += 1
            print(f"[err] {cut_id}: {e}")
        pbar.set_postfix_str(f"kept={n_kept} drop_text={n_drop_text} drop_err={n_drop_err}")

    elapsed = time.time() - t0
    print(
        f"[done] slice={args.slice_idx} cuts_total={n_total} kept={n_kept} "
        f"drop_text={n_drop_text} drop_err={n_drop_err} elapsed={elapsed/60:.1f}min"
    )

    if not rows_key:
        print(f"[warn] no rows survived for slice {args.slice_idx}; skipping shard write.")
        return

    table = pa.table(
        {
            "__key__": pa.array(rows_key, type=pa.string()),
            "A_text": pa.array(rows_text, type=pa.list_(pa.int64())),
            "A_audio": pa.array(
                rows_audio, type=pa.list_(pa.list_(pa.int64()))
            ),
        }
    )
    pq.write_table(table, out_path, compression="zstd")
    print(f"[done] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc_root", type=str,
                    default="/groups/gcg51557/experiments/0167_cc_audio/asai/ccaudio_rss_transcribed_array")
    ap.add_argument("--slice_idx", type=int, required=True, help="0..99")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Directory to write shard-{slice_idx:05d}.parquet")
    ap.add_argument("--audio_tokenizer_repo", default="kyutai/moshiko-pytorch-bf16")
    ap.add_argument("--audio_tokenizer_name", default="tokenizer-e351c8d8-checkpoint125.safetensors")
    ap.add_argument("--text_tokenizer_repo", default="rinna/japanese-gpt2-medium")
    ap.add_argument("--text_tokenizer_name", default="spiece.model")
    ap.add_argument("--audio_chunk_size_s", type=int, default=30)
    ap.add_argument("--num_codebooks", type=int, default=16,
                    help="Mimi quantizer codebook count (q16 to match J-CHAT/Reazon).")
    ap.add_argument("--text_padding_id", type=int, default=3)
    ap.add_argument("--end_of_text_padding_id", type=int, default=0)
    ap.add_argument("--rep_ratio_thresh", type=float, default=0.05,
                    help="Drop cut if (max 4-gram count) / text_length exceeds this.")
    ap.add_argument("--non_ja_thresh", type=float, default=0.10,
                    help="Drop cut if non-Japanese non-ascii char ratio exceeds this.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    main(args)
