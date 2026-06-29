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
from persona_schema import (  # noqa: E402
    Persona,
    classify_formality,
    decode_text_track,
    tokenize_prompt,
)


def build(args):
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    writer = None
    n_in = n_out = 0
    formality_counts = {"polite": 0, "casual": 0, "mixed": 0}

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
                    pj = row.get(args.persona_json_col)
                    persona = Persona(**json.loads(pj)) if pj else Persona()
                else:
                    raise ValueError(args.mode)

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
