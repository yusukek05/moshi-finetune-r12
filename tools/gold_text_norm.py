#!/usr/bin/env python3
"""Canonical text normalization + scoring for Zoom1 gold annotation.

Normalization mirrors the training/inference path so that any CER comparison
(gold vs ASR prelabel, or worker vs worker) is apples-to-apples with what the
model actually consumes (data_utils.py: NFKC-style fullwidth->halfwidth ASCII,
optional lowercase, terminal period handled by the pipeline — not scored here).

Two scoring jobs (the QA backbone):
  - asr-vs-gold : per-channel CER(prelabel, corrected) = the ReazonSpeech-ESPnet
                  ASR error rate, recovered as a byproduct of correction.
  - iaa         : per-channel CER between two workers' corrections of the SAME
                  clip = inter-annotator agreement.

Both read labelaudio AnnotationExport JSON (the *.annotations.json this repo's
tools/build_labelaudio_tasks.py emits and that labelaudio exports).
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

# fullwidth ASCII (U+FF01..FF5E) -> halfwidth (U+21..7E), same map the pipeline uses
_FW2HW = str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})


def normalize_ja(text: str, lower: bool = False) -> str:
    t = unicodedata.normalize("NFKC", text or "").translate(_FW2HW)
    if lower:
        t = t.lower()
    return "".join(t.split())  # drop all whitespace; CER is char-level


def channel_text(export: dict, channel_id: str) -> str:
    segs = [s for s in export.get("segments", []) if s.get("channelId") == channel_id]
    segs.sort(key=lambda s: s.get("startMs", 0))
    return "".join((s.get("transcript") or "") for s in segs)


def cer(ref: str, hyp: str) -> float:
    import jiwer
    if not ref:
        return float("nan") if not hyp else 1.0
    return jiwer.cer(ref, hyp)


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_pair(ref_export: dict, hyp_export: dict, lower: bool) -> dict:
    out = {}
    for cid, name in (("ch_1", "A"), ("ch_2", "B")):
        r = normalize_ja(channel_text(ref_export, cid), lower)
        h = normalize_ja(channel_text(hyp_export, cid), lower)
        out[name] = {"ref_chars": len(r), "hyp_chars": len(h), "cer": cer(r, h)}
    return out


def _avg(vals: list[float]) -> float:
    vals = [v for v in vals if v == v]  # drop nan
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("asr-vs-gold",
                        help="CER(prelabel, corrected) per clip = ASR error rate")
    p1.add_argument("--prelabel-dir", type=Path, required=True,
                    help="dir of original *.annotations.json (from build_labelaudio_tasks.py)")
    p1.add_argument("--gold-dir", type=Path, required=True,
                    help="dir of worker-corrected *.annotations.json (same filenames)")
    p1.add_argument("--lower", action="store_true", help="lowercase (match training path)")

    p2 = sub.add_parser("iaa", help="CER between two workers' corrections")
    p2.add_argument("--dir-a", type=Path, required=True)
    p2.add_argument("--dir-b", type=Path, required=True)
    p2.add_argument("--lower", action="store_true")

    args = ap.parse_args()
    da = args.prelabel_dir if args.cmd == "asr-vs-gold" else args.dir_a
    db = args.gold_dir if args.cmd == "asr-vs-gold" else args.dir_b

    names = sorted(p.name for p in Path(da).glob("*.annotations.json"))
    rows, cers_a, cers_b = [], [], []
    for n in names:
        pb = Path(db) / n
        if not pb.exists():
            print(f"[skip] no counterpart for {n}")
            continue
        s = score_pair(load(Path(da) / n), load(pb), args.lower)
        rows.append((n, s))
        cers_a.append(s["A"]["cer"]); cers_b.append(s["B"]["cer"])

    label = "ASR->gold CER" if args.cmd == "asr-vs-gold" else "worker A/B CER (IAA)"
    print(f"== {label} ==  (clips compared: {len(rows)})")
    for n, s in rows:
        print(f"  {n:48} A {s['A']['cer']:.3f}  B {s['B']['cer']:.3f}")
    print(f"-- mean: A {_avg(cers_a):.3f}  B {_avg(cers_b):.3f}  "
          f"both {_avg(cers_a + cers_b):.3f}")


if __name__ == "__main__":
    main()
