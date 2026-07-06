#!/usr/bin/env python3
"""PersonaPlex GENERATION-based flip (gold metric).

The NLL flip (persona_flip_eval.py) measures whether the prompt shifts teacher-forced
probability — an indirect, easily-diluted signal. This measures the real thing: prime the
model with ONLY the system-prompt prefix (no dialogue content), let it GENERATE a fresh
dialogue, decode the agent text track, and classify its formality. If the prompt controls
style, a "polite" prompt yields polite generations and a "casual" prompt casual ones.

For each carrier dialogue we generate under BOTH the polite and the casual prompt (same
seed carrier, only the prompt differs), so it is a paired test. No ASR needed — the model
emits the agent text track directly (row 0), which is exactly what classify_formality reads.

Run (GPU node):
  uv run python mstts/data_prep/persona_gen_flip.py \
      --model_dir output/personaplex_100h/step_675_fp32 \
      --carriers  processed_data/persona_100h/heldout_indist-001-of-001.parquet \
      --tokenizer <rinna spiece.model> --n 100 --paraphrase-idx 2
"""
import argparse
import os
import sys

import numpy as np
import torch
import sentencepiece as spm
from datasets import load_dataset

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "mstts", "data_prep"))

from models import MoshiForConditionalGeneration  # noqa: E402
from models.moshi_for_finetuning import MoshiForFinetuning  # noqa: E402
from utils.data import preprocess_function_with_system_prompt, DataCollator, undelay_tokens  # noqa: E402
from persona_schema import (  # noqa: E402
    Persona, formality_prompt, tokenize_prompt, decode_text_track, classify_formality,
)

STYLES = ("polite", "casual")


def style_of(dialogue_id: str) -> str:
    return os.path.basename(str(dialogue_id)).split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--carriers", required=True, help="persona parquet used as generation carriers")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=100, help="num carrier dialogues (balanced by style)")
    ap.add_argument("--generation_length", type=int, default=250)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=250)
    ap.add_argument("--top_p", type=float, default=0.0)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--min_length", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--paraphrase-idx", type=int, default=2,
                    help=">=0 uses a SEEN training paraphrase (recommended, since unseen "
                         "phrasings do not generalise); <0 uses the canonical prompt")
    ap.add_argument("--dump", default=None, help="optional: write per-sample jsonl here")
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    if args.paraphrase_idx >= 0:
        prompt_text = {s: formality_prompt(s, args.paraphrase_idx) for s in STYLES}
    else:
        prompt_text = {s: Persona(formality=s).to_prompt() for s in STYLES}
    prompts = {s: tokenize_prompt(prompt_text[s], sp) for s in STYLES}
    print("prompt_text:", prompt_text, flush=True)

    print(f"loading model from {args.model_dir}", flush=True)
    m = MoshiForFinetuning.from_pretrained(save_dir=args.model_dir, device=args.device, dtype=torch.bfloat16)
    m.eval()
    cg = MoshiForConditionalGeneration(moshi_lm=m)
    max_delay = max(m.delays)

    kwargs = dict(
        speakers=["A"], max_length=args.max_length, min_length=args.min_length,
        delays=m.delays,
        initial_token_ids=[m.text_initial_token_id] + [m.initial_token_id] * m.num_audio_codebooks,
        padding_token_ids=[m.text_padding_token_id] + [m.initial_token_id] * m.num_audio_codebooks,
        zero_token_id=m.zero_token_id,
        num_main_audio=8, delimiter_text_id=m.end_of_text_padding_id,
    )
    collator = DataCollator(zero_token_id=m.zero_token_id)
    sampling = {"use_sampling": True, "temp": args.temperature, "top_k": args.top_k, "top_p": args.top_p}

    ds = load_dataset("parquet", split="train", data_files={"train": args.carriers})
    # balanced carrier subset
    by = {s: [i for i, d in enumerate(ds["dialogue_id"]) if style_of(d) == s] for s in STYLES}
    per = max(1, args.n // 2)
    sel = (by["polite"][:per] + by["casual"][:per])
    ds = ds.select(sel)
    cols = ds.column_names
    print(f"carriers: {len(ds)} (polite {min(per,len(by['polite']))} + casual {min(per,len(by['casual']))})", flush=True)

    @torch.no_grad()
    def gen_style(row, req_style):
        # override prompt to the requested formality, preprocess, prime ONLY the prefix, generate
        r = {k: [row[k]] for k in cols}
        r["prompt_text_ids"] = [prompts[req_style]]
        proc = preprocess_function_with_system_prompt(r, **kwargs)
        b = collator([{k: proc[k][0] for k in proc}])
        inp = b.input_ids.to(args.device)  # (1, K, T)
        plen = proc["prompt_len"][0]
        prime = inp[:, :, : 1 + max_delay + plen]  # prefix region only
        gen = cg.generate(prompt_tokens=prime, generation_length=args.generation_length,
                          text_sampling_params=sampling, audio_sampling_params=sampling)
        gen = undelay_tokens(gen, m.delays)
        if gen is None:
            return "", "mixed"
        txt = decode_text_track(gen[0, 0].cpu().numpy(), sp)
        return txt, classify_formality(txt)

    rows = list(ds)
    results = []  # (requested, produced, text)
    dump = open(args.dump, "w") if args.dump else None
    import json as _json
    for idx, row in enumerate(rows):
        for req in STYLES:
            txt, prod = gen_style(row, req)
            results.append((req, prod, txt))
            if dump:
                dump.write(_json.dumps({"carrier": str(row["dialogue_id"]), "requested": req,
                                        "produced": prod, "text": txt}, ensure_ascii=False) + "\n")
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(rows)} carriers done", flush=True)
    if dump:
        dump.close()

    # ---- report ----
    def acc(req):
        sub = [(r, p) for (r, p, _) in results if r == req]
        n = len(sub)
        match = sum(1 for r, p in sub if p == req)
        mixed = sum(1 for r, p in sub if p == "mixed")
        opp = n - match - mixed
        return n, match, opp, mixed
    print("\n=== PersonaPlex GENERATION flip (prompt -> produced formality) ===")
    tot_match = tot = 0
    for req in STYLES:
        n, match, opp, mixed = acc(req)
        tot_match += match; tot += n
        print(f"  requested {req:6s}: n={n}  ->{req}={match} ({100*match/max(n,1):.0f}%)  "
              f"->{ 'casual' if req=='polite' else 'polite'}={opp}  mixed={mixed}")
    # paired flip: per carrier, polite-prompt->polite AND casual-prompt->casual
    per_carrier = {}
    for (r, p, _) in results:
        per_carrier.setdefault(None, None)
    # rebuild paired by carrier order (2 rows per carrier)
    paired_ok = 0; ncar = len(rows)
    for i in range(ncar):
        rp = results[2*i]     # (polite req, produced)
        rc = results[2*i + 1]  # (casual req, produced)
        if rp[1] == "polite" and rc[1] == "casual":
            paired_ok += 1
    print(f"\n  directional accuracy (both correct per carrier): {paired_ok}/{ncar} "
          f"({100*paired_ok/max(ncar,1):.0f}%)")
    print(f"  overall prompt->formality match: {tot_match}/{tot} ({100*tot_match/max(tot,1):.0f}%)")
    print("\ninterpretation: >>50% match (and high directional acc) => the text prompt "
          "genuinely controls generated formality. ~50% => weak/no control (NLL flip confirmed).")


if __name__ == "__main__":
    main()
