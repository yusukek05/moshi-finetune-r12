#!/usr/bin/env python3
"""Text-based meaningfulness reward via a local Japanese instruct LLM judge.

The acoustic evaluator's meaningfulness head over-rates "fluent nonsense"
(docs/2026-07-01 §4.6); the fix it recommends is to judge meaning from the
noise-free inner-monologue TEXT (recovered by tools/decode_text_from_tokens.py
/ decode_tokens.py --text_output_dir) instead of ASR-ing the audio.

This scores each generated continuation's transcript with a strong local
instruct LLM (default llm-jp/llm-jp-3.1-13b-instruct4 — Japanese-tuned, cached
on ABCI), on a 1-5 meaningfulness/coherence scale (repetition / word-salad /
symbol spam -> low). Output: json { <npy_path>: score } consumed by
tools/build_dpo_pairs.py --meaningfulness_json.

Run (GPU): uv run python tools/score_meaningfulness_llm.py \
    --gen_root output/dpo_scaled --seeds 42 43 44 45 46 47 \
    --out_json output/dpo_scaled/meaningfulness.json
"""
import argparse
import glob
import json
import os
import re

import torch

PROMPT = (
    "あなたは対話音声の品質評価者です。次のテキストは、対話音声モデルが生成した発話の"
    "書き起こし(音声認識ノイズなし)です。内容が日本語として意味が通り、対話の発話として"
    "自然で一貫しているかを評価してください。同じ語や記号の反復、意味不明、支離滅裂、"
    "「!」等の羅列は低評価にします。\n"
    "1(全く意味をなさない)〜5(意味が通り自然)の整数を1つだけ出力してください。\n\n"
    "書き起こし:\n{transcript}\n\n評価(1〜5の整数のみ):"
)


def parse_score(text: str):
    m = re.search(r"[1-5]", text)
    return int(m.group(0)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--model", default="llm-jp/llm-jp-3.1-13b-instruct4")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=4)
    ap.add_argument("--default_score", type=float, default=1.0,
                    help="score when transcript empty or LLM output unparseable")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device).eval()
    print(f"[judge] loaded {args.model}", flush=True)

    # collect (npy_path, transcript)
    items = []
    for s in args.seeds:
        tdir = os.path.join(args.gen_root, f"seed{s}", "generated_text")
        for tp in sorted(glob.glob(os.path.join(tdir, "*.txt"))):
            stem = os.path.splitext(os.path.basename(tp))[0]
            npy = os.path.join(args.gen_root, f"seed{s}", "generated_tokens", f"{stem}.npy")
            transcript = open(tp).read().strip()
            items.append((npy, transcript))
    print(f"[judge] scoring {len(items)} transcripts", flush=True)

    scores = {}
    for i in range(0, len(items), args.batch_size):
        chunk = items[i:i + args.batch_size]
        prompts = []
        for _, tr in chunk:
            msg = [{"role": "user", "content": PROMPT.format(transcript=tr[:1000] or "(無音)")}]
            prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True)
        enc.pop("token_type_ids", None)  # llama-arch generate() rejects this key
        enc = enc.to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for (npy, _tr), t in zip(chunk, texts, strict=False):
            sc = parse_score(t)
            scores[npy] = float(sc) if sc is not None else args.default_score
        if (i // args.batch_size) % 5 == 0:
            print(f"[judge] {i + len(chunk)}/{len(items)}", flush=True)

    json.dump(scores, open(args.out_json, "w"), ensure_ascii=False, indent=2)
    import statistics as st
    vals = list(scores.values())
    print(f"[judge] wrote {len(scores)} -> {args.out_json} "
          f"(mean {st.mean(vals):.2f} min {min(vals)} max {max(vals)})", flush=True)


if __name__ == "__main__":
    main()
