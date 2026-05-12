"""Convert JMultiWOZ dialogues to mstts-ready input JSON files.

JMultiWOZ (nu-dialogue/jmultiwoz, CC-BY-SA-4.0) stores task-oriented
dialogues in `dialogues.json` keyed by dialogue_name, with each entry's
`turns` list containing `{turn_id, speaker, utterance, ...}`. The
`speaker` field is either `"USER"` (customer) or `"SYSTEM"` (operator).

This script converts each dialogue to the mstts input format
`[["A", text], ["B", text], ...]` and writes one JSON per chunk. Long
dialogues are split into ≤ CHUNK_SECONDS-second segments so that the
mstts model stays within its stable generation window (~ 1 minute).

Speaker mapping: USER → A (left channel), SYSTEM → B (right channel).

Usage:
    # 1. Download once: https://github.com/nu-dialogue/jmultiwoz/blob/main/dataset/JMultiWOZ_1.0.zip
    python jmultiwoz_to_mstts.py \
        --zip-path JMultiWOZ_1.0.zip \
        --out-dir out/jmultiwoz_mstts \
        --splits train

License: JMultiWOZ dataset is CC-BY-SA-4.0. The upstream README states
"Models trained using the dataset are not considered copies or direct
derivatives of the dataset itself", so trained TTS / dialogue models
need not inherit ShareAlike. If you redistribute the *text or the
synthesised audio that closely reproduces the text*, attribution +
ShareAlike still apply.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

# Japanese speech: roughly 6 chars / sec at a calm conversational pace.
# Used only to bound chunk size, so a rough heuristic is sufficient.
CHARS_PER_SEC = 6.0
DEFAULT_CHUNK_SECONDS = 50.0

SPEAKER_MAP = {"USER": "A", "SYSTEM": "B"}


def chunk_turns(
    turns_ab: list[list[str]], char_budget: int
) -> list[list[list[str]]]:
    """Greedy chunking: accumulate turns until char_budget would be exceeded."""
    chunks: list[list[list[str]]] = []
    current: list[list[str]] = []
    used = 0
    for speaker, text in turns_ab:
        cost = len(text)
        if cost > char_budget and not current:
            chunks.append([[speaker, text]])
            continue
        if used + cost > char_budget and current:
            chunks.append(current)
            current = []
            used = 0
        current.append([speaker, text])
        used += cost
    if current:
        chunks.append(current)
    return chunks


def find_zip_entry(zf: zipfile.ZipFile, suffix: str) -> str:
    """Locate an entry whose name ends with `suffix`. JMultiWOZ_1.0/<file>."""
    for name in zf.namelist():
        if name.endswith(suffix):
            return name
    raise FileNotFoundError(f"No entry ending with {suffix!r} in {zf.filename}")


def convert(
    dialogues: dict,
    split_list: dict | None,
    splits_filter: set[str],
    out_dir: Path,
    char_budget: int,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()

    name_to_split: dict[str, str] = {}
    if split_list:
        for split, names in split_list.items():
            for n in names:
                name_to_split[n] = split

    for dialogue_name, d in dialogues.items():
        split = name_to_split.get(dialogue_name)
        if split_list and split not in splits_filter:
            stats["skipped_split"] += 1
            continue

        turns_ab: list[list[str]] = []
        for t in d.get("turns", []):
            spk_raw = (t.get("speaker") or "").upper()
            text = (t.get("utterance") or "").strip()
            if not text:
                continue
            spk = SPEAKER_MAP.get(spk_raw)
            if spk is None:
                stats["unknown_speaker"] += 1
                continue
            if turns_ab and turns_ab[-1][0] == spk:
                # Consecutive same-speaker — merge so alternation is preserved.
                turns_ab[-1][1] = turns_ab[-1][1].rstrip() + " " + text
                stats["merged_consecutive"] += 1
            else:
                turns_ab.append([spk, text])

        if not turns_ab:
            stats["empty_dialogue"] += 1
            continue

        chunks = chunk_turns(turns_ab, char_budget)
        stats["dialogues_converted"] += 1
        stats["chunks_written"] += len(chunks)
        for i, chunk in enumerate(chunks):
            out_path = out_dir / f"{dialogue_name}_part{i:02d}.json"
            out_path.write_text(
                json.dumps(chunk, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return dict(stats)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--zip-path", required=True, help="Path to JMultiWOZ_1.0.zip")
    p.add_argument("--out-dir", required=True, help="Output directory for per-chunk JSON files")
    p.add_argument(
        "--splits", default="train",
        help="Comma-separated splits to include (train|validation|test). Default: train",
    )
    p.add_argument(
        "--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS,
        help=f"Target chunk duration seconds (default {DEFAULT_CHUNK_SECONDS}).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Parse and report stats without writing files.",
    )
    args = p.parse_args()

    char_budget = int(CHARS_PER_SEC * args.chunk_seconds)
    wanted_splits = {s.strip() for s in args.splits.split(",") if s.strip()}

    zip_path = Path(args.zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        dialogues_entry = find_zip_entry(zf, "/dialogues.json")
        try:
            split_entry = find_zip_entry(zf, "/split_list.json")
        except FileNotFoundError:
            split_entry = None
        with zf.open(dialogues_entry) as f:
            dialogues = json.load(f)
        split_list = None
        if split_entry:
            with zf.open(split_entry) as f:
                split_list = json.load(f)

    out_dir = Path(args.out_dir)
    if args.dry_run:
        # Re-implement minimal accounting without writing files.
        out_dir = Path("/tmp/_jmultiwoz_dryrun_discard")
        out_dir.mkdir(parents=True, exist_ok=True)
    stats = convert(dialogues, split_list, wanted_splits, out_dir, char_budget)

    print("=== JMultiWOZ → mstts conversion stats ===", file=sys.stderr)
    print(f"  splits requested : {sorted(wanted_splits)}", file=sys.stderr)
    print(f"  char budget/chunk: {char_budget} (~{args.chunk_seconds:.0f}s)", file=sys.stderr)
    for k, v in stats.items():
        print(f"  {k:24s}: {v}", file=sys.stderr)
    if args.dry_run:
        print("  (dry-run — no files retained)", file=sys.stderr)
    else:
        print(f"  out_dir: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
