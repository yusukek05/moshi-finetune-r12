#!/usr/bin/env python3
"""Validate worker-submitted gold annotation JSON (labelaudio AnnotationExport).

Intake QA before a submission is accepted: checks structure, the closed tag
set, and flags low-effort submissions (identical to the ASR prelabel). Run
against the prelabel batch the task was cut from.

  python tools/validate_gold_submission.py \
      --submission-dir <worker exports> --prelabel-dir output/zoom1_gold_pilot/clips

FAIL = structurally unusable (rejected). WARN = needs a human look.
Exit code is non-zero if any file FAILs.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

ALLOWED_TAGS = {"(笑)", "(不明)"}          # closed set, confirmed 2026-05-29
_FW2HW = str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})
_TAG_RE = re.compile(r"[（(][^（）()]*[）)]")  # any parenthesised span, FW or HW
VALID_CHANNELS = {"ch_1", "ch_2"}


def norm(s: str) -> str:
    return "".join(unicodedata.normalize("NFKC", s or "").translate(_FW2HW).split())


def channel_text(export: dict, cid: str) -> str:
    segs = [s for s in export.get("segments", []) if s.get("channelId") == cid]
    segs.sort(key=lambda s: s.get("startMs", 0))
    return "".join((s.get("transcript") or "") for s in segs)


def check(sub: dict, pre: dict | None) -> tuple[list[str], list[str]]:
    fails, warns = [], []

    if not isinstance(sub, dict) or set(sub) < {"audio", "channels", "segments"}:
        return ["not a valid AnnotationExport (missing audio/channels/segments)"], []
    segs = sub.get("segments", [])
    if not isinstance(segs, list) or not segs:
        fails.append("no segments")
        return fails, warns

    n_empty = 0
    for i, s in enumerate(segs):
        cid = s.get("channelId")
        if cid not in VALID_CHANNELS:
            fails.append(f"seg[{i}] bad channelId {cid!r}")
        st, en = s.get("startMs"), s.get("endMs")
        if not isinstance(st, (int, float)) or not isinstance(en, (int, float)) or en <= st:
            fails.append(f"seg[{i}] bad span {st}..{en}")
        txt = (s.get("transcript") or "").strip()
        if not txt:
            n_empty += 1
            continue
        for tag in _TAG_RE.findall(txt):
            if norm(tag) not in {norm(t) for t in ALLOWED_TAGS}:
                fails.append(f"seg[{i}] illegal tag {tag!r} (allowed: {sorted(ALLOWED_TAGS)})")

    if n_empty:
        warns.append(f"{n_empty}/{len(segs)} empty transcripts")

    if pre is not None:
        same = all(norm(channel_text(sub, c)) == norm(channel_text(pre, c))
                   for c in VALID_CHANNELS)
        if same:
            warns.append("identical to ASR prelabel (possible no-effort)")
        ns, npre = len(segs), len(pre.get("segments", []))
        if ns < 0.5 * npre:
            warns.append(f"segment count dropped {npre}->{ns} (deletions?)")
    return fails, warns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission-dir", type=Path, required=True)
    ap.add_argument("--prelabel-dir", type=Path,
                    help="originals; enables low-effort / deletion flags")
    args = ap.parse_args()

    subs = sorted(args.submission_dir.glob("*.annotations.json"))
    if not subs:
        raise SystemExit(f"no *.annotations.json in {args.submission_dir}")

    n_fail = n_warn = 0
    for sp in subs:
        try:
            sub = json.loads(sp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"FAIL {sp.name}: unreadable JSON ({e})")
            n_fail += 1
            continue
        pre = None
        if args.prelabel_dir:
            pp = args.prelabel_dir / sp.name
            if pp.exists():
                pre = json.loads(pp.read_text(encoding="utf-8"))
            else:
                print(f"WARN {sp.name}: no prelabel counterpart")
        fails, warns = check(sub, pre)
        status = "FAIL" if fails else ("WARN" if warns else "PASS")
        if fails:
            n_fail += 1
        elif warns:
            n_warn += 1
        print(f"{status} {sp.name}")
        for m in fails:
            print(f"    ! {m}")
        for m in warns:
            print(f"    ~ {m}")

    print(f"\n{len(subs)} files: {len(subs)-n_fail-n_warn} pass, {n_warn} warn, {n_fail} fail")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
