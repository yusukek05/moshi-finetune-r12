#!/usr/bin/env python3
"""PersonaPlex TOPIC generation-flip (gold metric for content control).

Same idea as persona_gen_flip.py but for the TOPIC axis. Prime ONLY the topic system-prompt
prefix (no dialogue content), generate a fresh dialogue, decode the agent text track, and
classify which of the 24 topics it is about. If the prompt controls content, a "travel"
prompt yields travel talk and a "cooking" prompt cooking talk.

Topic classifier: char-2gram Jaccard similarity of the generated text to each topic's
reference text (concatenated training dialogues of that topic, from the scripts jsonl).
Fully offline, transparent, no extra model.

Reports:
  - matched top-1 accuracy (24-way; chance ~= 1/24 = 4%)
  - paired directional flip: matched-prompt -> matched topic AND swapped-prompt -> swapped topic

Run (GPU node):
  uv run python mstts/data_prep/persona_gen_flip_topic.py \
      --model_dir output/personaplex_topic_100h/step_675_fp32 \
      --carriers  processed_data/topic_100h/heldout_topic-001-of-001.parquet \
      --scripts   <diverse_100h_scripts.jsonl> --tokenizer <rinna spiece.model> --n 96
"""
import argparse
import json
import os
import sys

import torch
import sentencepiece as spm
from datasets import load_dataset

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "mstts", "data_prep"))

from models import MoshiForConditionalGeneration  # noqa: E402
from models.moshi_for_finetuning import MoshiForFinetuning  # noqa: E402
from utils.data import preprocess_function_with_system_prompt, DataCollator, undelay_tokens  # noqa: E402
from persona_schema import decode_text_track  # noqa: E402


def topic_prompt(topic_label: str) -> str:
    return f"{topic_label}について話します。"


def char2grams(t: str) -> set:
    t = "".join(t.split())
    return set(t[i:i + 2] for i in range(len(t) - 1)) if len(t) >= 2 else set(t)


def build_topic_refs(scripts_jsonl):
    """{topic: (topic_label, ref_char2gram_set)} from the generator scripts."""
    from collections import defaultdict
    texts = defaultdict(list)
    labels = {}
    for line in open(scripts_jsonl, encoding="utf-8"):
        d = json.loads(line)
        tp = d.get("topic"); labels[tp] = d.get("topic_label")
        texts[tp].append("".join(t["text"] for t in d["turns"]))
    refs = {tp: (labels[tp], char2grams("".join(txts))) for tp, txts in texts.items()}
    return refs


def classify_topic(text, refs):
    g = char2grams(text)
    if not g:
        return None
    best, bs = None, -1.0
    for tp, (_, rg) in refs.items():
        inter = len(g & rg)
        j = inter / (len(g) + len(rg) - inter + 1e-9)
        if j > bs:
            bs, best = j, tp
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--carriers", required=True)
    ap.add_argument("--scripts", required=True, help="diverse_100h_scripts.jsonl (topic refs)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--generation_length", type=int, default=250)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=250)
    ap.add_argument("--top_p", type=float, default=0.0)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--min_length", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    refs = build_topic_refs(args.scripts)
    all_topics = sorted(refs)
    print(f"topic refs: {len(all_topics)} topics", flush=True)

    print(f"loading model from {args.model_dir}", flush=True)
    m = MoshiForFinetuning.from_pretrained(save_dir=args.model_dir, device=args.device, dtype=torch.bfloat16)
    m.eval()
    cg = MoshiForConditionalGeneration(moshi_lm=m)
    max_delay = max(m.delays)
    kwargs = dict(
        speakers=["A"], max_length=args.max_length, min_length=args.min_length, delays=m.delays,
        initial_token_ids=[m.text_initial_token_id] + [m.initial_token_id] * m.num_audio_codebooks,
        padding_token_ids=[m.text_padding_token_id] + [m.initial_token_id] * m.num_audio_codebooks,
        zero_token_id=m.zero_token_id, num_main_audio=8, delimiter_text_id=m.end_of_text_padding_id,
    )
    collator = DataCollator(zero_token_id=m.zero_token_id)
    sampling = {"use_sampling": True, "temp": args.temperature, "top_k": args.top_k, "top_p": args.top_p}

    ds = load_dataset("parquet", split="train", data_files={"train": args.carriers})
    cols = ds.column_names
    # one carrier per topic, up to n
    seen = {}
    idxs = []
    for i, tp in enumerate(ds["topic"]):
        if tp not in seen:
            seen[tp] = i; idxs.append(i)
    idxs = idxs[: args.n] if args.n else idxs
    ds = ds.select(idxs)
    print(f"carriers: {len(ds)} (one per topic)", flush=True)

    @torch.no_grad()
    def gen_with_topic(row, topic_label):
        r = {k: [row[k]] for k in cols}
        r["prompt_text_ids"] = [sp.encode(topic_prompt(topic_label))]
        proc = preprocess_function_with_system_prompt(r, **kwargs)
        b = collator([{k: proc[k][0] for k in proc}])
        inp = b.input_ids.to(args.device)
        plen = proc["prompt_len"][0]
        prime = inp[:, :, : 1 + max_delay + plen]
        gen = cg.generate(prompt_tokens=prime, generation_length=args.generation_length,
                          text_sampling_params=sampling, audio_sampling_params=sampling)
        gen = undelay_tokens(gen, m.delays)
        if gen is None:
            return "", None
        txt = decode_text_track(gen[0, 0].cpu().numpy(), sp)
        return txt, classify_topic(txt, refs)

    rows = list(ds)
    dump = open(args.dump, "w") if args.dump else None
    matched_ok = 0
    paired_ok = 0
    n = len(rows)
    # deterministic swapped topic = next topic in the sorted list
    for idx, row in enumerate(rows):
        req = row["topic"]; req_label = row["topic_label"]
        swap = all_topics[(all_topics.index(req) + 7) % len(all_topics)]
        swap_label = refs[swap][0]
        mtxt, mcls = gen_with_topic(row, req_label)
        stxt, scls = gen_with_topic(row, swap_label)
        if mcls == req:
            matched_ok += 1
        if mcls == req and scls == swap:
            paired_ok += 1
        if dump:
            dump.write(json.dumps({"carrier": str(row["dialogue_id"]), "req": req, "matched_cls": mcls,
                                   "swap": swap, "swap_cls": scls, "matched_text": mtxt[:200]},
                                  ensure_ascii=False) + "\n")
        if (idx + 1) % 8 == 0:
            print(f"  {idx+1}/{n}  (matched_top1 so far {100*matched_ok/(idx+1):.0f}%)", flush=True)
    if dump:
        dump.close()

    print("\n=== PersonaPlex TOPIC generation flip ===")
    print(f"  matched top-1 (24-way): {matched_ok}/{n} ({100*matched_ok/max(n,1):.0f}%)  [chance ~4%]")
    print(f"  paired directional (matched->req AND swap->swap): {paired_ok}/{n} ({100*paired_ok/max(n,1):.0f}%)")
    print("\ninterpretation: matched top-1 >> 4% => the topic prompt steers generated content.")


if __name__ == "__main__":
    main()
