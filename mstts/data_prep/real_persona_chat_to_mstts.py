"""Convert RealPersonaChat dialogues to mstts-ready input JSON files.

RealPersonaChat (nu-dialogue/real-persona-chat, CC-BY-SA-4.0) stores
chitchat dialogues as one JSON per dialogue under
`real_persona_chat/dialogues/{00001..14000}.json`. Each file has:

    {
        "dialogue_id": int,
        "interlocutors": ["AA", "AB"],  # 2 speaker IDs
        "utterances": [
            {"utterance_id": 0, "interlocutor_id": "AA", "text": "...", "timestamp": "..."},
            ...
        ],
        "evaluations": [...]
    }

This script converts each dialogue to mstts input format
`[["A", text], ["B", text], ...]` and writes one JSON per chunk. Long
dialogues are split into ≤ CHUNK_SECONDS-second segments so mstts stays
in its stable generation window.

Speaker mapping: the first interlocutor (`interlocutors[0]`) → A,
                 the second                               (`[1]`)   → B.

Usage:
    # 1. Clone or download the repo:
    #    git clone https://github.com/nu-dialogue/real-persona-chat
    # 2. Convert:
    python real_persona_chat_to_mstts.py \
        --dialogues-dir real-persona-chat/real_persona_chat/dialogues \
        --out-dir out/rpc_mstts

License: RealPersonaChat is CC-BY-SA-4.0 (commercial allowed with
attribution + ShareAlike). Unlike JMultiWOZ, the upstream README does
NOT explicitly exempt trained models from ShareAlike, so be cautious
about how derivative model weights are licensed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

CHARS_PER_SEC = 6.0
DEFAULT_CHUNK_SECONDS = 50.0


def chunk_turns(
    turns_ab: list[list[str]], char_budget: int
) -> list[list[list[str]]]:
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


def convert_one(d: dict) -> list[list[str]]:
    interlocutors = d.get("interlocutors") or []
    if len(interlocutors) < 2:
        return []
    id_to_ab = {interlocutors[0]: "A", interlocutors[1]: "B"}
    turns_ab: list[list[str]] = []
    for u in d.get("utterances", []):
        text = (u.get("text") or "").strip()
        if not text:
            continue
        spk = id_to_ab.get(u.get("interlocutor_id"))
        if spk is None:
            continue
        if turns_ab and turns_ab[-1][0] == spk:
            turns_ab[-1][1] = turns_ab[-1][1].rstrip() + " " + text
        else:
            turns_ab.append([spk, text])
    return turns_ab


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dialogues-dir", required=True,
                   help="Path to real_persona_chat/dialogues/ containing 00001..14000.json")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for per-chunk JSON files")
    p.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS,
                   help=f"Target chunk duration seconds (default {DEFAULT_CHUNK_SECONDS}).")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse and report stats without writing files.")
    p.add_argument("--max-dialogues", type=int, default=None,
                   help="Cap the number of dialogues processed (for testing).")
    args = p.parse_args()

    char_budget = int(CHARS_PER_SEC * args.chunk_seconds)
    in_dir = Path(args.dialogues_dir)
    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    files = sorted(in_dir.glob("*.json"))
    if args.max_dialogues:
        files = files[: args.max_dialogues]

    for fp in files:
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            stats["read_error"] += 1
            continue
        turns_ab = convert_one(d)
        if not turns_ab:
            stats["empty_dialogue"] += 1
            continue
        chunks = chunk_turns(turns_ab, char_budget)
        stats["dialogues_converted"] += 1
        stats["chunks_total"] += len(chunks)
        if args.dry_run:
            continue
        stem = fp.stem  # "00001"
        for i, chunk in enumerate(chunks):
            out_path = out_dir / f"rpc_{stem}_part{i:02d}.json"
            out_path.write_text(
                json.dumps(chunk, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print("=== RealPersonaChat → mstts conversion stats ===", file=sys.stderr)
    print(f"  char budget/chunk: {char_budget} (~{args.chunk_seconds:.0f}s)", file=sys.stderr)
    for k, v in stats.items():
        print(f"  {k:24s}: {v}", file=sys.stderr)
    if args.dry_run:
        print("  (dry-run — no files written)", file=sys.stderr)
    else:
        print(f"  out_dir: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
