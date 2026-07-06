#!/usr/bin/env python3
"""DPO stage 2: rank on-policy samples per context and mine preference pairs.

Input layout (from pbs/run_dpo_sample_gen.sh):
    <gen_root>/seed<S>/generated_tokens/<ctx>.npy   full 17-stream tokens
    <gen_root>/seed<S>/generated_wavs/<ctx>.wav      2ch audio (naturalness)
    <gen_root>/seed<S>/generated_text/<ctx>.txt      inner-monologue transcript

For each context we have N candidates (one per seed). We score each and, per
context, emit the (best, worst) preference pair when their reward gap clears a
confidence threshold (large-gap pairs only -> cleaner DPO signal).

Reward (anti-hacking aware — deliberately NOT the acoustic *meaningfulness*
head, which over-rates "fluent nonsense", per docs/2026-07-01 §4.6):

    reward = w_nat * z(naturalness_acoustic)   # reliable acoustic head (0.57 CV)
           + w_rep * distinct_ratio            # unique/total text tokens: anti repetition-loop
           + w_bal * channel_balance           # min/max ch RMS: anti monologue-collapse
    (hard veto: reward = -inf if channel_balance < --min_balance -> never "chosen")

`distinct_ratio` and token count come straight from stream 0 of the .npy (no
tokenizer needed). `naturalness` is supplied via --naturalness_json
(path -> predicted naturalness MOS), produced by the crowdsourcing reward model
(run score_audio.py in that venv; see pbs/run_dpo_score_naturalness.sh). When
the json is absent the naturalness term is dropped (0), so the pair logic is
testable without the reward model.

Output: <out_jsonl> with one line per pair:
    {context_id, chosen:{seed,reward,...}, rejected:{seed,reward,...}, gap}
plus a per-candidate scores dump for inspection.
"""
import argparse
import glob
import json
import os

import numpy as np


def text_health(npy_path: str, text_padding_id: int, end_of_text_padding_id: int):
    """(n_content_tokens, distinct_ratio) from the text stream (row 0)."""
    arr = np.load(npy_path)
    stream = arr[0].tolist()
    kept = [int(t) for t in stream
            if int(t) != text_padding_id and int(t) != end_of_text_padding_id]
    n = len(kept)
    distinct = (len(set(kept)) / n) if n else 0.0
    return n, distinct


def channel_balance(wav_path: str):
    """min/max per-channel RMS in [0,1]; ~0 => one speaker silent (monologue)."""
    if not wav_path or not os.path.exists(wav_path):
        return None
    import soundfile as sf
    x, _ = sf.read(wav_path)
    if x.ndim != 2 or x.shape[1] != 2:
        return None
    rms = np.sqrt((x.astype(np.float64) ** 2).mean(axis=0) + 1e-12)
    lo, hi = float(rms.min()), float(rms.max())
    return (lo / hi) if hi > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen_root", required=True, help="dir containing seed<S>/ subdirs")
    ap.add_argument("--seeds", nargs="+", required=True, help="seed ids, e.g. 42 43 44 45")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--scores_jsonl", default=None, help="optional per-candidate dump")
    ap.add_argument("--naturalness_json", default=None,
                    help="json map wav_path -> predicted naturalness MOS (crowdsourcing)")
    ap.add_argument("--meaningfulness_json", default=None,
                    help="json map npy_path -> LLM-judge meaningfulness 1-5 (score_meaningfulness_llm.py)")
    ap.add_argument("--w_nat", type=float, default=1.0)
    ap.add_argument("--w_mean", type=float, default=1.0)
    ap.add_argument("--w_rep", type=float, default=0.5)
    ap.add_argument("--w_bal", type=float, default=0.5)
    ap.add_argument("--min_balance", type=float, default=0.05,
                    help="veto a candidate as monologue-collapsed below this ch balance")
    ap.add_argument("--min_gap", type=float, default=0.5,
                    help="confidence filter: emit a pair only if reward gap >= this")
    ap.add_argument("--text_padding_id", type=int, default=3)
    ap.add_argument("--end_of_text_padding_id", type=int, default=0)
    args = ap.parse_args()

    nat_map = {}
    if args.naturalness_json and os.path.exists(args.naturalness_json):
        nat_map = json.load(open(args.naturalness_json))
        # allow either {path: nat} or score_audio rows [{path,naturalness,...}]
        if isinstance(nat_map, list):
            nat_map = {r["path"]: r["naturalness"] for r in nat_map}

    mean_map = {}
    if args.meaningfulness_json and os.path.exists(args.meaningfulness_json):
        mean_map = json.load(open(args.meaningfulness_json))  # {npy_path: score}

    # discover contexts from the first seed's tokens
    seed0 = os.path.join(args.gen_root, f"seed{args.seeds[0]}", "generated_tokens")
    contexts = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(seed0, "*.npy"))
    )
    if not contexts:
        ap.error(f"no .npy contexts under {seed0}")

    # gather raw per-candidate features
    cand = {}  # (ctx, seed) -> feature dict
    for ctx in contexts:
        for s in args.seeds:
            npy = os.path.join(args.gen_root, f"seed{s}", "generated_tokens", f"{ctx}.npy")
            wav = os.path.join(args.gen_root, f"seed{s}", "generated_wavs", f"{ctx}.wav")
            if not os.path.exists(npy):
                continue
            n, distinct = text_health(npy, args.text_padding_id, args.end_of_text_padding_id)
            bal = channel_balance(wav)
            cand[(ctx, s)] = {
                "context_id": ctx, "seed": s, "npy": npy, "wav": wav,
                "n_text_tokens": n, "distinct_ratio": round(distinct, 4),
                "channel_balance": (round(bal, 4) if bal is not None else None),
                "naturalness": nat_map.get(wav),
                "meaningfulness": mean_map.get(npy),
            }

    # z-normalise naturalness + meaningfulness across ALL candidates (if available)
    def _zstats(key):
        vals = [c[key] for c in cand.values() if c[key] is not None]
        if vals:
            return float(np.mean(vals)), float(np.std(vals) + 1e-6)
        return 0.0, 1.0
    mu, sd = _zstats("naturalness")
    mu_m, sd_m = _zstats("meaningfulness")

    def reward(c):
        bal = c["channel_balance"]
        if bal is not None and bal < args.min_balance:
            return float("-inf")  # monologue veto
        r = 0.0
        if c["naturalness"] is not None:
            r += args.w_nat * (c["naturalness"] - mu) / sd
        if c["meaningfulness"] is not None:
            r += args.w_mean * (c["meaningfulness"] - mu_m) / sd_m
        r += args.w_rep * c["distinct_ratio"]
        if bal is not None:
            r += args.w_bal * bal
        return r

    for c in cand.values():
        c["reward"] = reward(c)

    # per-context best/worst pair with confidence filter
    pairs, n_ctx_with_pair = [], 0
    for ctx in contexts:
        cs = [cand[(ctx, s)] for s in args.seeds if (ctx, s) in cand]
        cs = [c for c in cs if c["reward"] != float("-inf")]  # drop vetoed
        if len(cs) < 2:
            continue
        cs.sort(key=lambda c: c["reward"], reverse=True)
        best, worst = cs[0], cs[-1]
        gap = best["reward"] - worst["reward"]
        if gap < args.min_gap:
            continue
        n_ctx_with_pair += 1
        pairs.append({
            "context_id": ctx,
            "gap": round(gap, 4),
            "chosen": {k: best[k] for k in ("seed", "npy", "wav", "reward", "distinct_ratio",
                                            "channel_balance", "naturalness", "meaningfulness")},
            "rejected": {k: worst[k] for k in ("seed", "npy", "wav", "reward", "distinct_ratio",
                                               "channel_balance", "naturalness", "meaningfulness")},
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if args.scores_jsonl:
        with open(args.scores_jsonl, "w") as f:
            for c in sorted(cand.values(), key=lambda c: (c["context_id"], c["seed"])):
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"contexts={len(contexts)} candidates={len(cand)} "
          f"pairs={len(pairs)} (ctx_with_pair={n_ctx_with_pair}) "
          f"nat_used={'yes' if mean_map or mu != 0.0 else 'maybe'} "
          f"meaning_used={'yes' if mean_map else 'NO'} -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
