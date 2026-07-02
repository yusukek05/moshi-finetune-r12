#!/usr/bin/env python3
"""PersonaPlex flip evaluation (teacher-forced conditional NLL).

Question: does the text system-prompt actually control the agent's style?
Metric  : for each held-out dialogue, compute the model's mean NLL over the agent
          text track (row 0, dialogue region only; prompt region is loss-masked)
          under (a) the *matched* persona prompt and (b) the *swapped* (opposite
          formality) prompt. If conditioning was learned, matched NLL < swapped NLL.

This reuses the EXACT training preprocessing (preprocess_function_with_system_prompt)
and the exact tempformer text-loss, so the number is the same quantity training
optimizes -- just measured under two different prompts on held-out data.

Reports:
  - mean matched vs swapped NLL (overall and per requested style)
  - flip accuracy = fraction of dialogues with matched NLL < swapped NLL
  - control effect = mean(swapped - matched)   (>0 => prompt helps)

Run (uv, on a GPU node; needs the trained fp32 ckpt via tools.zero_to_fp32):
  uv run python mstts/data_prep/persona_flip_eval.py \
      --model_dir output/personaplex_poc_smoke/step_150_fp32 \
      --heldout   processed_data/persona_poc/persona_smoke_heldout-001-of-001.parquet \
      --tokenizer <rinna spiece.model>
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from datasets import load_dataset

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "mstts", "data_prep"))

from models.moshi_for_finetuning import MoshiForFinetuning  # noqa: E402
from utils.data import preprocess_function_with_system_prompt, DataCollator  # noqa: E402
from persona_schema import Persona, formality_prompt, tokenize_prompt  # noqa: E402

STYLES = ("polite", "casual")
OPP = {"polite": "casual", "casual": "polite"}


# Formality-bearing surface forms. The whole-track NLL is diluted by style-neutral
# content words; restricting to these markers isolates the style signal.
POLITE_MARKERS = ["です", "ます", "ました", "ません", "でした", "でしょう", "ですね",
                  "ますね", "ください", "ございます", "しています", "してます", "ですか", "ますか"]
CASUAL_MARKERS = ["だよ", "だね", "だな", "じゃん", "だろ", "かな", "よね", "なんだ",
                  "だわ", "だぜ", "じゃ", "してる", "たね", "よ", "ね", "さ"]


def build_marker_ids(sp) -> set:
    ids = set()
    for w in POLITE_MARKERS + CASUAL_MARKERS:
        ids.update(int(t) for t in sp.encode(w))
    return ids


@torch.no_grad()
def text_nll(moshi_lm, batch, marker_ids=None) -> dict:
    """Mean agent text-track NLL over the (non-pad, non-ignored) dialogue region.
    Replicates finetune.tempformer_forward's text loss exactly. Also returns the
    NLL restricted to formality-marker tokens (sharper, less-diluted signal)."""
    text_emb = moshi_lm.text_emb(batch.input_ids[:, 0])
    audio_emb = None
    for i in range(moshi_lm.num_audio_codebooks):
        e = moshi_lm.emb[i](batch.input_ids[:, moshi_lm.audio_offset + i])
        audio_emb = e if audio_emb is None else audio_emb + e
    out = moshi_lm.transformer(text_emb + audio_emb, attention_mask=batch.text_attention_mask)
    if moshi_lm.out_norm:
        out = moshi_lm.out_norm(out)
    logits = moshi_lm.text_linear(out).float()[..., :-1, :].contiguous()
    labels = batch.labels[:, 0, 1:].contiguous()
    loss = F.cross_entropy(
        logits.view(-1, moshi_lm.text_card), labels.view(-1),
        ignore_index=moshi_lm.zero_token_id, reduction="none",
    ).view(labels.size())
    keep = (labels != moshi_lm.text_padding_token_id) & (labels != moshi_lm.zero_token_id)
    all_nll = loss[keep].mean().item() if keep.sum() > 0 else float("nan")
    mark_nll = float("nan")
    if marker_ids is not None:
        mk = torch.zeros_like(labels, dtype=torch.bool)
        for tid in marker_ids:
            mk |= (labels == tid)
        mk &= keep
        if mk.sum() > 0:
            mark_nll = loss[mk].mean().item()
    return {"all": all_nll, "marker": mark_nll}


def style_of(dialogue_id: str) -> str:
    return os.path.basename(str(dialogue_id)).split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="trained fp32 ckpt (zero_to_fp32 output)")
    ap.add_argument("--heldout", required=True, help="held-out persona parquet")
    ap.add_argument("--tokenizer", required=True, help="rinna spiece.model (same vocab as text track)")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--min_length", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--paraphrase-idx", type=int, default=-1,
                    help="if >=0, use this formality paraphrase (e.g. the held-out one) "
                         "instead of the canonical prompt — tests prompt generalisation")
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    if args.paraphrase_idx >= 0:
        prompt_text = {s: formality_prompt(s, args.paraphrase_idx) for s in STYLES}
    else:
        prompt_text = {s: Persona(formality=s).to_prompt() for s in STYLES}
    prompts = {s: tokenize_prompt(prompt_text[s], sp) for s in STYLES}
    print("prompt_text:", prompt_text, flush=True)
    print("prompts:", {s: prompts[s] for s in STYLES}, flush=True)

    print(f"loading model from {args.model_dir}", flush=True)
    m = MoshiForFinetuning.from_pretrained(save_dir=args.model_dir, device=args.device, dtype=torch.bfloat16)
    m.eval()

    kwargs = dict(
        speakers=["A"], max_length=args.max_length, min_length=args.min_length,
        delays=m.delays,
        initial_token_ids=[m.text_initial_token_id] + [m.initial_token_id] * m.num_audio_codebooks,
        padding_token_ids=[m.text_padding_token_id] + [m.initial_token_id] * m.num_audio_codebooks,
        zero_token_id=m.zero_token_id,
        num_main_audio=8, delimiter_text_id=m.end_of_text_padding_id,
    )
    collator = DataCollator(zero_token_id=m.zero_token_id)

    ds = load_dataset("parquet", split="train", data_files={"train": args.heldout})
    cols = ds.column_names

    def make_mapped(swap: bool):
        def set_prompt(batch):
            out = {k: list(batch[k]) for k in batch}
            newp = []
            for did in batch["dialogue_id"]:
                s = style_of(did)
                newp.append(prompts[OPP[s] if swap else s])
            out["prompt_text_ids"] = newp
            return out
        d = ds.map(set_prompt, batched=True, desc=("swap" if swap else "match"))
        return d.map(preprocess_function_with_system_prompt, remove_columns=cols,
                     batched=True, num_proc=1, fn_kwargs=kwargs,
                     desc=("preprocess swap" if swap else "preprocess match"))

    marker_ids = build_marker_ids(sp)
    print(f"marker token ids: {len(marker_ids)}", flush=True)

    mapped = {"matched": make_mapped(False), "swapped": make_mapped(True)}
    ids = [style_of(x) for x in ds["dialogue_id"]]

    # per-dialogue NLL under each condition (order preserved by map)
    res = {"matched": [], "swapped": []}
    for cond in ("matched", "swapped"):
        for row in mapped[cond]:
            b = collator([row])
            b.input_ids = b.input_ids.to(args.device)
            b.labels = b.labels.to(args.device)
            b.text_attention_mask = b.text_attention_mask.to(args.device)
            res[cond].append(text_nll(m, b, marker_ids=marker_ids))

    n = len(ids)
    import statistics as st

    def rep(sel_idx, label, key):
        pairs = [(res["matched"][i][key], res["swapped"][i][key]) for i in sel_idx
                 if not (np.isnan(res["matched"][i][key]) or np.isnan(res["swapped"][i][key]))]
        if not pairs:
            print(f"  {label:16s} (no valid tokens)")
            return
        mt = [a for a, _ in pairs]
        sw = [b for _, b in pairs]
        flips = [a < b for a, b in pairs]
        acc = 100 * sum(flips) / len(flips)
        eff = st.mean([b - a for a, b in pairs])
        print(f"  {label:16s} n={len(pairs):3d}  NLL matched={st.mean(mt):.4f}  "
              f"swapped={st.mean(sw):.4f}  effect(swap-match)={eff:+.4f}  flip_acc={acc:.0f}%")

    for key, title in [("all", "whole agent text track"), ("marker", "formality-marker tokens only")]:
        print(f"\n=== PersonaPlex flip eval ({n} held-out) — {title} ===")
        rep(range(n), "ALL", key)
        for s in STYLES:
            rep([i for i, x in enumerate(ids) if x == s], f"requested {s}", key)
    print("\ninterpretation: flip_acc >> 50% and effect > 0 => the text prompt controls "
          "the agent's formality. The marker-only metric is the less-diluted signal.")


if __name__ == "__main__":
    main()
