#!/usr/bin/env python3
"""Build labelaudio annotation tasks from AsReX Zoom1 ASR transcripts.

For each sampled dialogue we cut a short stereo window (overlaps preserved,
boundaries snapped to silence so no utterance is split) and emit:
  <clip>.wav                 stereo 16 kHz, ch0 = speaker A, ch1 = speaker B
  <clip>.annotations.json    labelaudio AnnotationExport, prelabel segments
                             (source="vad") = the existing ReazonSpeech-ESPnet
                             transcript that crowd workers will correct.

Channel order matches how the prelabels were generated (AsReX speakers
["A","B"] -> stereo channel 0,1) and the project's L=A / R=B wav convention.
Use --swap-channels if a future corpus violates that.

Input layout (per session dir <S>):
  <base>_A.json, <base>_B.json   in  transcripts/<S>/   ({text, segments:[{start,end,text}]}, seconds)
  <base>.wav                     in  audio_merged/<S>/  (stereo 16 kHz)
where base = "<sess>_<Wa>_<Wb>_<T>".
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

NOW = datetime.now(timezone.utc).isoformat()
CHANNELS = [
    {"id": "ch_1", "index": 0, "name": "A (left)"},
    {"id": "ch_2", "index": 1, "name": "B (right)"},
]
CH_BY_SPK = {"A": "ch_1", "B": "ch_2"}


def load_segments(path: Path) -> list[dict]:
    segs = json.loads(path.read_text(encoding="utf-8")).get("segments", [])
    out = []
    for s in segs:
        try:
            st, en = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if en > st:
            out.append({"start": st, "end": en, "text": (s.get("text") or "").strip()})
    return out


def silence_cut_points(intervals: list[tuple[float, float]], total: float,
                       min_gap: float) -> list[float]:
    """Times where neither channel is active (gap >= min_gap). Cutting here
    never splits an utterance on either channel. Always includes 0 and total."""
    if not intervals:
        return [0.0, total]
    iv = sorted(intervals)
    merged = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    cuts = [0.0]
    for (_, e0), (s1, _) in zip(merged, merged[1:]):
        if s1 - e0 >= min_gap:
            cuts.append((e0 + s1) / 2.0)
    cuts.append(total)
    return sorted(set(cuts))


def pick_window(cuts: list[float], window_sec: float, rng: random.Random,
                total: float) -> tuple[float, float] | None:
    starts = [c for c in cuts if c < total - 1.0]
    if not starts:
        return None
    t0 = rng.choice(starts)
    later = [c for c in cuts if c >= t0 + window_sec]
    t1 = later[0] if later else total
    if t1 - t0 < 1.0:
        return None
    return t0, t1


def window_segments(segs: list[dict], t0: float, t1: float, channel_id: str) -> list[dict]:
    out = []
    for s in segs:
        st, en = max(s["start"], t0), min(s["end"], t1)
        if en - st <= 0.0:
            continue
        out.append({
            "start_ms": round((st - t0) * 1000),
            "end_ms": round((en - t0) * 1000),
            "text": s["text"],
            "channel_id": channel_id,
        })
    out.sort(key=lambda x: x["start_ms"])
    return out


def overlap_seconds(a: list[dict], b: list[dict]) -> float:
    """Total time both channels are simultaneously active (within a window)."""
    pts = []
    for seg in a:
        pts.append((seg["start_ms"], 1)); pts.append((seg["end_ms"], -1))
    for seg in b:
        pts.append((seg["start_ms"], 2)); pts.append((seg["end_ms"], -2))
    pts.sort()
    a_on = b_on = 0
    last = 0
    both_ms = 0
    for t, d in pts:
        if a_on > 0 and b_on > 0:
            both_ms += t - last
        last = t
        if d == 1: a_on += 1
        elif d == -1: a_on -= 1
        elif d == 2: b_on += 1
        elif d == -2: b_on -= 1
    return both_ms / 1000.0


def build_annotations(base: str, clip_name: str, sr: int, n_frames: int,
                      win_segs: list[dict]) -> dict:
    segments = []
    counters = {"ch_1": 0, "ch_2": 0}
    for seg in win_segs:
        cid = seg["channel_id"]
        counters[cid] += 1
        segments.append({
            "id": f"prelabel_{cid}_{counters[cid]:06d}",
            "channelId": cid,
            "startMs": seg["start_ms"],
            "endMs": seg["end_ms"],
            "labels": {},
            "source": "vad",
            "createdAt": NOW,
            "updatedAt": NOW,
            "transcript": seg["text"],
        })
    segments.sort(key=lambda s: (s["startMs"], s["channelId"]))
    return {
        "audio": {
            "filename": clip_name,
            "durationMs": round(n_frames / sr * 1000),
            "sampleRate": sr,
            "channelCount": 2,
        },
        "channels": CHANNELS,
        "segments": segments,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr-base", type=Path, required=True,
                    help="AsReX zoom1 dir containing transcripts/ and audio_merged/")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-windows", type=int, default=60)
    ap.add_argument("--window-sec", type=float, default=90.0)
    ap.add_argument("--min-gap", type=float, default=0.3,
                    help="min silence (s) on both channels to allow a cut")
    ap.add_argument("--max-per-session", type=int, default=1)
    ap.add_argument("--min-utts", type=int, default=4,
                    help="skip windows with fewer total prelabel segments")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--swap-channels", action="store_true",
                    help="emit ch0=B, ch1=A (use only if corpus violates L=A/R=B)")
    args = ap.parse_args()

    tdir = args.asr_base / "transcripts"
    adir = args.asr_base / "audio_merged"
    if not tdir.is_dir() or not adir.is_dir():
        raise SystemExit(f"expected transcripts/ and audio_merged/ under {args.asr_base}")

    rng = random.Random(args.seed)
    sessions = sorted(p.name for p in tdir.iterdir() if p.is_dir())
    rng.shuffle(sessions)

    clips_dir = args.out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    n_clips = 0

    for sess in sessions:
        if n_clips >= args.n_windows:
            break
        a_files = sorted((tdir / sess).glob("*_A.json"))
        if not a_files:
            continue
        a_json = a_files[0]
        base = a_json.name[:-len("_A.json")]
        b_json = tdir / sess / f"{base}_B.json"
        wav = adir / sess / f"{base}.wav"
        if not b_json.exists() or not wav.exists():
            continue
        try:
            info = sf.info(str(wav))
        except Exception as e:  # unreadable wav -> skip
            print(f"[skip] {wav}: {e}")
            continue
        if info.channels != 2:
            print(f"[skip] {wav}: not stereo ({info.channels}ch)")
            continue
        sr, total = info.samplerate, info.frames / info.samplerate
        segs_a, segs_b = load_segments(a_json), load_segments(b_json)
        cuts = silence_cut_points(
            [(s["start"], s["end"]) for s in segs_a + segs_b], total, args.min_gap)

        made_this_session = 0
        for _ in range(8):  # a few tries to land a non-trivial window
            if n_clips >= args.n_windows or made_this_session >= args.max_per_session:
                break
            win = pick_window(cuts, args.window_sec, rng, total)
            if win is None:
                break
            t0, t1 = win
            wa = window_segments(segs_a, t0, t1, "ch_1")
            wb = window_segments(segs_b, t0, t1, "ch_2")
            if args.swap_channels:
                for s in wa: s["channel_id"] = "ch_2"
                for s in wb: s["channel_id"] = "ch_1"
            if len(wa) + len(wb) < args.min_utts:
                continue

            start_f, stop_f = int(round(t0 * sr)), int(round(t1 * sr))
            data, _ = sf.read(str(wav), start=start_f, stop=stop_f,
                              dtype="float32", always_2d=True)
            if args.swap_channels:
                data = data[:, ::-1]
            clip_name = f"{base}__t{int(t0)}_{int(t1)}.wav"
            sf.write(str(clips_dir / clip_name), data, sr, subtype="PCM_16")

            ann = build_annotations(base, clip_name, sr, data.shape[0], wa + wb)
            (clips_dir / clip_name.replace(".wav", ".annotations.json")).write_text(
                json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")

            ov = overlap_seconds(wa, wb)
            manifest_rows.append({
                "clip": clip_name, "session": sess, "base": base,
                "t0_s": round(t0, 2), "t1_s": round(t1, 2),
                "dur_s": round(t1 - t0, 2), "n_seg_A": len(wa), "n_seg_B": len(wb),
                "overlap_s": round(ov, 2),
            })
            n_clips += 1
            made_this_session += 1

    with (args.out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows
                           else ["clip"])
        w.writeheader()
        w.writerows(manifest_rows)

    tot_min = sum(r["dur_s"] for r in manifest_rows) / 60.0
    tot_ov = sum(r["overlap_s"] for r in manifest_rows)
    print(f"clips: {n_clips} | audio: {tot_min:.1f} min "
          f"| overlap: {tot_ov:.1f} s ({100*tot_ov/max(tot_min*60,1):.1f}% of audio) "
          f"| out: {args.out_dir}")


if __name__ == "__main__":
    main()
