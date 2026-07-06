"""Build a TOPIC-conditioned v1 parquet for PersonaPlex-style content control.

The formality axis (build_persona_parquet.py) gave no usable control — formality is a
subtle, context-redundant signal. CONTENT/TOPIC is what a user actually wants to steer
("talk about X") and is a much stronger, less-redundant signal. The 100h corpus is already
labelled with 24 topics, so we attach a topic instruction as the system prompt.

Input : v1-format parquet {dialogue_id, A, B} + the generator script dir (for topic labels).
Output: same rows + prompt_text_ids (tokenized topic instruction) + topic + topic_label.

prompt template (declarative, matching the formality-prompt style):
    "{topic_label}について話します。"      e.g. "行ってみたい旅行先について話します。"

Run:
  uv run --no-project --python 3.12 --with pyarrow --with numpy --with sentencepiece python \
      mstts/data_prep/build_topic_parquet.py \
      --input processed_data/synth_dialogue_100h/synth_100h_cer20-001-of-001.parquet \
      --tokenizer output/v1.1_release/tokenizer_spm_32k_3.model \
      --scripts-dir <diverse_100h scripts> \
      --output processed_data/topic_100h/topic_synth_cer20-001-of-001.parquet
"""
import argparse
import glob
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm


def topic_prompt(topic_label: str) -> str:
    return f"{topic_label}について話します。"


def load_topic_map(scripts_dir):
    """{script_stem: (topic, topic_label)} from generator script JSONs."""
    m = {}
    for p in glob.glob(os.path.join(scripts_dir, "*.json")):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        m[stem] = (d.get("topic"), d.get("topic_label"))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--scripts-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    tmap = load_topic_map(args.scripts_dir)
    print(f"topic map: {len(tmap)} entries")
    # sanity: distinct labels
    labels = sorted({v[1] for v in tmap.values() if v[1]})
    print(f"distinct topic labels: {len(labels)}")

    writer = None
    n_in = n_out = n_unmatched = 0
    for path in args.input:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["dialogue_id", "A", "B"]):
            out = {"dialogue_id": [], "A": [], "B": [],
                   "prompt_text_ids": [], "topic": [], "topic_label": []}
            for row in batch.to_pylist():
                n_in += 1
                stem = os.path.basename(str(row["dialogue_id"]))
                tl = tmap.get(stem)
                if tl is None or tl[1] is None:
                    n_unmatched += 1
                    continue
                topic, topic_label = tl
                ids = sp.encode(topic_prompt(topic_label))
                if not ids:
                    continue
                out["dialogue_id"].append(row["dialogue_id"])
                out["A"].append(row["A"])
                out["B"].append(row["B"])
                out["prompt_text_ids"].append(ids)
                out["topic"].append(topic)
                out["topic_label"].append(topic_label)
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
    print(f"output rows: {n_out if False else '(see next)'}")
    # recount output
    orows = pq.ParquetFile(args.output).metadata.num_rows
    print(f"output rows: {orows} -> {args.output}")
    print(f"unmatched  : {n_unmatched}")


if __name__ == "__main__":
    main()
