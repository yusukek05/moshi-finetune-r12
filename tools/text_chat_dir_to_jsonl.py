"""Convert a directory of mstts text_chat JSON files into a single
dialogues.jsonl compatible with cascade_synth.py.

Each input file dialogue_*.json contains: [["A", "..."], ["B", "..."], ...]
Each output line is: {"id": "<file_stem>", "turns": <input_content>}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="dir with dialogue_*.json")
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--skip-if-audio-exists-under", default=None,
                    help="if set, skip dialogues that already have "
                         "<this_dir>/<dialogue_id>/audio.wav (for supplemental runs)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no *.json in {in_dir}")

    skip_root = Path(args.skip_if_audio_exists_under) if args.skip_if_audio_exists_under else None
    n_written = n_skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for fp in files:
            if skip_root and (skip_root / fp.stem / "audio.wav").exists():
                n_skipped += 1
                continue
            turns = json.loads(fp.read_text(encoding="utf-8"))
            rec = {"id": fp.stem, "turns": turns}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"wrote {n_written} dialogue rows -> {out_path} (skipped {n_skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
