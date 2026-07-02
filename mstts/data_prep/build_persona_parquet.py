"""
Build a persona-conditioned v1 parquet for PersonaPlex-style prompt-control training.

Input : v1-format parquet(s) with columns {dialogue_id, A:[9,T], B:[9,T]}.
Output: same rows + per-example prompt fields consumed by
        utils.data.preprocess_function_with_system_prompt (via finetune.py
        --system_prompt_conditioning):
          - prompt_text_ids : list[int]      (tokenized Japanese persona prompt)
          - persona_json    : str            (the persona dict, for provenance)
          - prompt_audio    : [num_main_audio, Tv] | None  (voice-prompt mimi codes;
                              only emitted in --voice-prompt mode)

Two labeling modes:
  --mode bootstrap  : measure an attribute (formality) from the agent's own text
                      track and use it as the persona label. Cheap, no generation,
                      but only as diverse as the input corpus (see survey: existing
                      corpora are ~uniformly polite -> weak for style control).
  --mode passthrough: attach a persona supplied per-row via --persona-json-col
                      (for the causal path where LLM-jp-3 generated each dialogue
                      from a known persona).

For v0 we default to text-only prompts (no prompt_audio). Pass --voice-prompt to
also extract the first --voice-frames of the agent (speaker A) audio as a voice
prompt (note: same-dialogue voice prompt risks trivial copying; intended for the
held-out-speaker eval setup, not naive training).

Run (uv, per CLAUDE.md):
  uv run --no-project --python 3.12 --with pandas --with pyarrow --with numpy \
      --with sentencepiece python mstts/data_prep/build_persona_parquet.py \
      --input processed_data/synth_dialogue/synth_dialogue-001-of-001.parquet \
      --tokenizer output/v1.1_release/tokenizer_spm_32k_3.model \
      --output processed_data/persona_poc/persona_synth-001-of-001.parquet
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hashlib  # noqa: E402

from persona_schema import (  # noqa: E402
    FORMALITY_PARAPHRASES,
    Persona,
    classify_formality,
    decode_text_track,
    formality_prompt,
    tokenize_prompt,
)


def _paraphrase_for(dialogue_id: str, formality: str, holdout: int) -> str:
    """Deterministically pick a paraphrase for this row, excluding the held-out
    index (reserved for eval). Seeded by dialogue_id so runs are reproducible."""
    n = len(FORMALITY_PARAPHRASES[formality])
    choices = [i for i in range(n) if i != holdout]
    h = int(hashlib.md5(str(dialogue_id).encode()).hexdigest(), 16)
    return formality_prompt(formality, choices[h % len(choices)])


def _load_persona_map(map_dir):
    """Build {script_stem: persona_dict} from a dir of generator script JSONs.

    Each JSON (gen_persona_dialogues.py output) carries a "persona_json" string.
    The stem (e.g. "polite_food_0003") matches basename(dialogue_id) in the
    prep_synth_dialogue parquet, so we can attach the ground-truth persona that
    was used to *generate* the dialogue (the causal passthrough label)."""
    import glob
    m = {}
    for p in glob.glob(os.path.join(map_dir, "*.json")):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        pj = d.get("persona_json")
        if pj:
            m[stem] = json.loads(pj) if isinstance(pj, str) else pj
    return m


def build(args):
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    writer = None
    n_in = n_out = 0
    n_unmatched = 0
    formality_counts = {"polite": 0, "casual": 0, "mixed": 0}
    persona_map = _load_persona_map(args.persona_map_dir) if args.persona_map_dir else None
    if persona_map is not None:
        print(f"persona map : {len(persona_map)} entries from {args.persona_map_dir}")

    for path in args.input:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["dialogue_id", "A", "B"]):
            rows = batch.to_pylist()
            out = {"dialogue_id": [], "A": [], "B": [],
                   "prompt_text_ids": [], "persona_json": []}
            if args.voice_prompt:
                out["prompt_audio"] = []
            for row in rows:
                n_in += 1
                A = np.array(row["A"])
                # --- derive persona ---
                if args.mode == "bootstrap":
                    text = decode_text_track(A[0], sp)
                    formality = classify_formality(text)
                    formality_counts[formality] += 1
                    persona = Persona(formality=formality)
                elif args.mode == "passthrough":
                    if persona_map is not None:
                        key = os.path.basename(str(row["dialogue_id"]))
                        pd = persona_map.get(key)
                        if pd is None:
                            n_unmatched += 1
                            continue
                        persona = Persona(**pd)
                    else:
                        pj = row.get(args.persona_json_col)
                        persona = Persona(**json.loads(pj)) if pj else Persona()
                else:
                    raise ValueError(args.mode)

                if args.paraphrase_holdout >= 0 and persona.formality in FORMALITY_PARAPHRASES:
                    prompt = _paraphrase_for(row["dialogue_id"], persona.formality,
                                             args.paraphrase_holdout)
                else:
                    prompt = persona.to_prompt()
                prompt_ids = tokenize_prompt(prompt, sp)
                if not prompt_ids:
                    continue

                out["dialogue_id"].append(row["dialogue_id"])
                out["A"].append(row["A"])
                out["B"].append(row["B"])
                out["prompt_text_ids"].append(prompt_ids)
                out["persona_json"].append(json.dumps(persona.to_dict(), ensure_ascii=False))
                if args.voice_prompt:
                    # agent = speaker A; audio rows are A[1:9]; take first voice_frames
                    voice = A[1:9, : args.voice_frames].astype(np.int64)
                    out["prompt_audio"].append(voice.tolist())
                n_out += 1

            if not out["dialogue_id"]:
                continue
            table = pa.table(out)
            if writer is None:
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
                writer = pq.ParquetWriter(args.output, table.schema)
            writer.write_table(table)

    if writer is not None:
        writer.close()
    print(f"input rows : {n_in}")
    print(f"output rows: {n_out} -> {args.output}")
    if persona_map is not None:
        print(f"unmatched  : {n_unmatched} (dialogue_id with no persona-map entry, skipped)")
    if args.mode == "bootstrap":
        print(f"formality  : {formality_counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="v1-format parquet(s)")
    ap.add_argument("--tokenizer", required=True, help="SP text tokenizer .model")
    ap.add_argument("--output", required=True, help="output parquet path")
    ap.add_argument("--mode", choices=["bootstrap", "passthrough"], default="bootstrap")
    ap.add_argument("--persona-json-col", default="persona_json",
                    help="column holding a persona dict json (passthrough mode)")
    ap.add_argument("--persona-map-dir", default=None,
                    help="passthrough: dir of generator script JSONs; join persona by "
                         "basename(dialogue_id)==script stem (causal corpus path)")
    ap.add_argument("--paraphrase-holdout", type=int, default=-1,
                    help="if >=0, tokenize a per-row RANDOM formality paraphrase "
                         "(excluding this held-out index, reserved for eval) instead of "
                         "the single canonical prompt. Trains prompt-diversity.")
    ap.add_argument("--voice-prompt", action="store_true",
                    help="also emit prompt_audio from the agent's own audio")
    ap.add_argument("--voice-frames", type=int, default=50,
                    help="voice-prompt length in frames (12.5 Hz; 50 = 4s)")
    ap.add_argument("--limit", type=int, default=0, help="(debug) cap input rows; 0=all")
    args = ap.parse_args()
    if args.limit:
        # simple cap: read only the first --limit rows of the first file
        import pyarrow.parquet as _pq
        t = _pq.read_table(args.input[0]).slice(0, args.limit)
        tmp = args.output + ".cap.parquet"
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        _pq.write_table(t, tmp)
        args.input = [tmp]
    build(args)


if __name__ == "__main__":
    main()
