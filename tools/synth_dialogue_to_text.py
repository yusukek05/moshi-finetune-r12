"""0386 の合成ターン分離ステレオ対話を Moshi 学習形式の word-level text JSON に変換する。

入力:
  - stereo wav: L=話者S1(=A) / R=話者S2(=B), 24kHz, 重なり無し
    (`0386/output/dialogue_arena/drop_full_pool_stereo/<id>.wav`)
  - 台本 JSON : {turns: [{speaker: "S1"|"S2", text}]}
    (`0386/data/dialogue_scripts/gen/<id>.json`)

処理:
  1. ステレオを話者別チャンネルに分割(ch0=A=S1, ch1=B=S2)。
  2. 各チャンネルの非ゼロ区間からターン境界を復元する。合成時に非アクティブ
     チャンネルは厳密に 0 埋め(gen_dialogue_stereo の `torch.zeros`)なので、
     非ゼロの連続塊 = その話者のターン(台本順)に 1:1 対応する。
  3. 復元した [start,end] と GT 台本テキストを segment として `whisperx.align(ja)`
     に渡し、文字単位のタイムスタンプを得る(認識ではなく整列。GTなので高精度)。
  4. A/B をマージし start 昇順で `text/<id>.json` を書く。
     形式は tools/tokenize_text_from_dir.py が要求する
     `[{"speaker":"A"|"B","word":str,"start":float,"end":float}, ...]`。
  5. stereo wav を audio dir に symlink(tokenize_audio_from_dir.py 用)。

whisperx を含む venv で実行すること(例: 本リポジトリ用に別途 uv sync した
`.venv_whisperx`)。GPU 1枚 / シャードで、infer_batch と同じ決定的シャーディング。
"""

from __future__ import annotations

import argparse
import json
import os
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
import whisperx

warnings.filterwarnings("ignore", message=".*TorchCodec.*", category=UserWarning)
warnings.filterwarnings(
    "ignore", message=".*StreamingMediaDecoder.*", category=UserWarning
)

ALIGN_SR = 16_000  # whisperx / wav2vec2 の入力 SR
# S1,S3 -> 左(A) / S2,S4 -> 右(B)。本データは S1/S2 のみ。
SPK_TO_CH = {"S1": 0, "S3": 0, "S2": 1, "S4": 1}
CH_TO_LABEL = {0: "A", 1: "B"}


def recover_turn_spans(
    ch_wav: np.ndarray, n_turns: int, sr: int, min_gap_ms: float
) -> list[tuple[float, float]]:
    """1チャンネル波形(原 SR)から非ゼロ連続区間=ターン境界を復元して秒で返す。

    非アクティブ部は厳密 0 なので |x|>0 でアクティブ判定。min_gap_ms より短い
    無音は同一ターン内とみなして結合する。復元数が台本ターン数と一致しない場合は
    全アクティブ範囲をテキスト長で按分する(whisperx が内部で精緻化するため、
    segment 境界は概略で十分)。
    """
    active = np.flatnonzero(np.abs(ch_wav) > 0.0)
    if active.size == 0:
        return []
    gap = int(sr * min_gap_ms / 1000.0)
    brk = np.flatnonzero(np.diff(active) > gap)
    starts = np.concatenate([[active[0]], active[brk + 1]])
    ends = np.concatenate([active[brk], [active[-1]]])
    spans = [(int(s) / sr, (int(e) + 1) / sr) for s, e in zip(starts, ends, strict=False)]
    if len(spans) == n_turns:
        return spans
    return []  # caller 側で按分フォールバック


def proportional_spans(
    ch_wav: np.ndarray, turns: list[dict], sr: int
) -> list[tuple[float, float]]:
    """全アクティブ範囲 [first,last] を各ターンのテキスト長で按分する。"""
    active = np.flatnonzero(np.abs(ch_wav) > 0.0)
    if active.size == 0:
        return []
    t0, t1 = int(active[0]) / sr, (int(active[-1]) + 1) / sr
    lens = [max(1, len(t["text"])) for t in turns]
    total = sum(lens)
    spans, cur = [], t0
    for n in lens:
        nxt = cur + (t1 - t0) * n / total
        spans.append((cur, nxt))
        cur = nxt
    return spans


def align_channel(
    ch_wav_align: np.ndarray,
    segments: list[dict],
    align_model,
    meta,
    device: str,
    label: str,
) -> list[dict]:
    """segments=[{start,end,text}] を whisperx で文字単位整列し word 項目を返す。"""
    if not segments:
        return []
    # AsReX(asrex_pkg.align_processor.AlignProcessor)と同じ呼び出し方に統一。
    # 日本語は whisperx が "words" を文字単位に分割するため文字レベル整列になる。
    aligned = whisperx.align(
        [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments],
        align_model,
        meta,
        ch_wav_align,
        device,
        return_char_alignments=False,
    )
    items: list[dict] = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            if w.get("start") is None or w.get("end") is None:
                continue
            word = w.get("word", "")
            if not isinstance(word, str) or not word.strip():
                continue
            # SentencePiece(rinna)は内部で NFKC 正規化する(例: "…"→"...")。
            # 先に揃えておかないと tokenize_text_from_dir の文字↔トークン対応が
            # ずれて IndexError になるため、書き出し時に NFKC を適用する。
            word = unicodedata.normalize("NFKC", word)
            if not word.strip():
                continue
            items.append(
                {
                    "speaker": label,
                    "word": word,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                }
            )
    return items


def process_one(
    stem: str,
    stereo_path: Path,
    script_path: Path,
    out_audio: Path,
    out_text: Path,
    align_model,
    meta,
    device: str,
    min_gap_ms: float,
) -> str:
    turns = json.load(open(script_path))["turns"]
    wav, sr = torchaudio.load(str(stereo_path))  # [2, T]
    assert wav.shape[0] == 2, f"{stem}: expected stereo, got {wav.shape[0]}ch"
    wav_np = wav.numpy()

    all_items: list[dict] = []
    for ch, label in CH_TO_LABEL.items():
        spk_turns = [t for t in turns if SPK_TO_CH.get(t["speaker"], 0) == ch]
        if not spk_turns:
            continue
        ch_wav = wav_np[ch]
        spans = recover_turn_spans(ch_wav, len(spk_turns), sr, min_gap_ms)
        if not spans:
            spans = proportional_spans(ch_wav, spk_turns, sr)
        segments = [
            {"start": s, "end": e, "text": t["text"]}
            for (s, e), t in zip(spans, spk_turns, strict=False)
        ]
        ch_align = torchaudio.functional.resample(
            torch.from_numpy(ch_wav).float().unsqueeze(0), sr, ALIGN_SR
        ).squeeze(0).numpy()
        all_items += align_channel(ch_align, segments, align_model, meta, device, label)

    all_items.sort(key=lambda x: x["start"])
    out_text.parent.mkdir(parents=True, exist_ok=True)
    out_text.write_text(json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8")

    out_audio.parent.mkdir(parents=True, exist_ok=True)
    if not out_audio.exists():
        os.symlink(os.path.abspath(stereo_path), out_audio)
    return f"{stem}: {len(all_items)} chars, {len(turns)} turns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stereo_dir",
        default="/groups/gcg51557/experiments/0386_dialogue_model/output/dialogue_arena/drop_full_pool_stereo",
    )
    ap.add_argument(
        "--scripts_dir",
        default="/groups/gcg51557/experiments/0386_dialogue_model/data/dialogue_scripts/gen",
    )
    ap.add_argument("--out_audio_dir", default="data/synth_dialogue/audio")
    ap.add_argument("--out_text_dir", default="data/synth_dialogue/text")
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min_gap_ms", type=float, default=120.0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    stereo_dir, scripts_dir = Path(args.stereo_dir), Path(args.scripts_dir)
    out_audio_dir, out_text_dir = Path(args.out_audio_dir), Path(args.out_text_dir)

    files = sorted(stereo_dir.glob("*.wav"))
    shard = [f for i, f in enumerate(files) if i % args.num_shards == args.shard_idx]
    if args.limit > 0:
        shard = shard[: args.limit]
    print(f"[shard {args.shard_idx}/{args.num_shards}] {len(shard)}/{len(files)} wav", flush=True)

    align_model, meta = whisperx.load_align_model(language_code=args.lang, device=args.device)

    done = skipped = failed = 0
    for wp in shard:
        stem = wp.stem
        out_text = out_text_dir / f"{stem}.json"
        out_audio = out_audio_dir / f"{stem}.wav"
        if args.skip_existing and out_text.exists():
            skipped += 1
            continue
        sp = scripts_dir / f"{stem}.json"
        if not sp.exists():
            print(f"  [SKIP {stem}] no script", flush=True)
            skipped += 1
            continue
        try:
            msg = process_one(
                stem, wp, sp, out_audio, out_text,
                align_model, meta, args.device, args.min_gap_ms,
            )
            done += 1
            print(f"  [OK] {msg}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [ERR {stem}] {e}", flush=True)

    print(f"[shard {args.shard_idx}] done={done} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
