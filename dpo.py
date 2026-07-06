#!/usr/bin/env python3
"""DPO for Moshi multistream dialogue continuation (RL stage 3).

Improve the policy from on-policy preference pairs mined by the crowdsourcing
MOS reward (tools/build_dpo_pairs.py). DPO loss with a FROZEN reference (= the
initial policy), reference log-probs precomputed once and cached:

    L = -log sigmoid( beta * [ (lp_pol_w - lp_ref_w) - (lp_pol_l - lp_ref_l) ] )

lp_* = sequence_logprob over the CONTINUATION region only (tools/dpo_data.py),
summed over text + audio streams.

Modes:
  --selftest  : 1 GPU, no training. Validates the data path (alignment, prompt
                masking, finite log-probs; at init loss==log2, logits==0).
  --dump_ref  : 1 GPU, no training. Compute + cache ref log-probs for every pair
                (--ref_out json), from the initial policy. Run before training.
  (default)   : deepspeed ZeRO-3 training loop (mirrors finetune.py); loads the
                cached ref log-probs and updates the policy. Anti-hacking: keep
                beta/KL sane; SUCCESS is judged by human A/B, never this reward.
"""
import argparse
import collections
import json
import math
import os
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

from models import MoshiForFinetuning
from tools.dpo_data import (
    build_example,
    collate,
    prompt_undelayed_from_streams,
    sequence_logprob,
    token_id_config,
)
from utils import set_mpi_env_vars


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_jsonl", required=True)
    p.add_argument("--eval_data_files", required=True,
                   help="parquet of contexts (same as generation --eval_data_files)")
    p.add_argument("--model_dir", required=True, help="policy (= ref at init)")
    p.add_argument("--output_dir", default="output/dpo_pilot_run")
    p.add_argument("--model_dtype", choices=["float32", "float16", "bfloat16"],
                   default="bfloat16")
    p.add_argument("--prompt_length", type=int, default=125)
    p.add_argument("--example_length", type=int, default=375)
    p.add_argument("--model_user_stream", action="store_true", default=True)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--dataset_processing_workers", type=int, default=8)
    p.add_argument("--dataset_cache_dir", default=".cache/huggingface/datasets")
    # modes
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--num_selftest", type=int, default=8)
    p.add_argument("--dump_ref", action="store_true")
    p.add_argument("--ref_out", default=None, help="json to write/read ref logprobs")
    # training
    p.add_argument("--launcher", choices=["accelerate", "mpi"], default="mpi")
    p.add_argument("--use_deepspeed", action="store_true", default=False)
    p.add_argument("--deepspeed_config_file", default="ds_configs/zero3-bf16-warmlr-act_ckpt.json")
    p.add_argument("--parameters_to_finetune", default="all")
    p.add_argument("--per_device_train_batch_size", type=int, default=1,
                   help="pairs per device (each pair = 2 sequences: chosen+rejected)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--tempformer_learning_rate", type=float, default=5e-7)
    p.add_argument("--depformer_learning_rate", type=float, default=5e-7)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_warmup_steps", type=int, default=5)
    p.add_argument("--activation_checkpointing", action="store_true", default=False)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--process_group_timeout", type=int, default=1800)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.launcher == "mpi":
        set_mpi_env_vars()
    return args


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_context_streams(args, moshi_lm):
    """Reproduce generate.py's preprocessing so example_id -> delayed streams
    matches the saved continuations exactly."""
    from datasets import load_dataset

    from utils import preprocess_function

    ds = load_dataset("parquet", split="evaluation",
                      data_files={"evaluation": args.eval_data_files},
                      cache_dir=args.dataset_cache_dir)
    kw = {
        "speakers": ["A"],
        "max_length": args.example_length,
        "min_length": args.prompt_length,
        "delays": moshi_lm.delays,
        "initial_token_ids": [moshi_lm.text_initial_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "padding_token_ids": [moshi_lm.text_padding_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "zero_token_id": moshi_lm.zero_token_id,
    }
    ds = ds.map(preprocess_function, remove_columns=ds.column_names, batched=True,
                num_proc=args.dataset_processing_workers, fn_kwargs=kw,
                desc="Preprocessing contexts")
    return ds


def build_pair_examples(pairs, ds, cfg, prompt_length):
    """-> list of dicts {chosen, rejected, context_id} (numpy streams+labels)."""
    out = []
    for p in pairs:
        cid = int(p["context_id"])
        delayed_streams = np.asarray(ds[cid]["streams"])
        prompt_und = prompt_undelayed_from_streams(delayed_streams, cfg["delays"], prompt_length)
        if prompt_und is None:
            continue
        try:
            ch = build_example(prompt_und, np.load(p["chosen"]["npy"]), cfg)
            rj = build_example(prompt_und, np.load(p["rejected"]["npy"]), cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[skip ctx {cid}] {e}", flush=True)
            continue
        out.append({"chosen": ch, "rejected": rj, "context_id": cid})
    return out


class PairDataset(Dataset):
    def __init__(self, pair_examples, ref_map):
        self.data = pair_examples
        self.ref_map = ref_map  # context_id -> (lp_ref_w, lp_ref_l)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        pe = self.data[i]
        rw, rl = self.ref_map.get(pe["context_id"], (0.0, 0.0))
        return {"chosen": pe["chosen"], "rejected": pe["rejected"],
                "ref_w": rw, "ref_l": rl}


def make_collate(zero_token_id):
    def _collate(items):
        # concatenate: [chosen_0..chosen_{B-1}, rejected_0..rejected_{B-1}]
        exs = [it["chosen"] for it in items] + [it["rejected"] for it in items]
        batch = collate(exs, zero_token_id)
        ref_w = torch.tensor([it["ref_w"] for it in items], dtype=torch.float32)
        ref_l = torch.tensor([it["ref_l"] for it in items], dtype=torch.float32)
        return batch, ref_w, ref_l
    return _collate


def dpo_loss_from_logprob(lp, ref_w, ref_l, beta):
    """lp: (2B,) concatenated [chosen ; rejected]. Returns (loss, metrics)."""
    B = lp.shape[0] // 2
    lp_w, lp_l = lp[:B], lp[B:]
    logits = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    loss = -F.logsigmoid(logits).mean()
    with torch.no_grad():
        acc = (logits > 0).float().mean()
        margin = (lp_w - lp_l).mean()
    return loss, {"loss/dpo": loss.detach(), "dpo/acc": acc, "dpo/logits": logits.mean().detach(),
                  "dpo/margin": margin}


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def _load_pairs_and_examples(args, moshi_lm, limit=None):
    cfg = token_id_config(moshi_lm)
    ds = load_context_streams(args, moshi_lm)
    pairs = [json.loads(x) for x in open(args.pairs_jsonl)]
    if limit:
        pairs = pairs[:limit]
    pe = build_pair_examples(pairs, ds, cfg, args.prompt_length)
    return cfg, pe


def run_selftest(args):
    device = torch.device("cuda")
    moshi_lm = MoshiForFinetuning.from_pretrained(
        args.model_dir, device=device, dtype=getattr(torch, args.model_dtype))
    moshi_lm.eval()
    cfg, pe = _load_pairs_and_examples(args, moshi_lm, limit=args.num_selftest)
    if not pe:
        print("SELFTEST FAIL: no reconstructable pairs")
        return
    coll = make_collate(cfg["zero_token_id"])
    batch, _, _ = coll([{"chosen": x["chosen"], "rejected": x["rejected"],
                         "ref_w": 0.0, "ref_l": 0.0} for x in pe])
    batch = batch.to(device)
    scored = (batch.labels != cfg["zero_token_id"]).sum().item()
    print(f"built {len(pe)} pairs; batch input_ids {tuple(batch.input_ids.shape)}; "
          f"scored (continuation) tokens {scored}/{batch.labels.numel()} "
          f"({100*scored/batch.labels.numel():.1f}%)", flush=True)
    with torch.no_grad():
        lp = sequence_logprob(moshi_lm, batch, args.model_user_stream)
    B = lp.shape[0] // 2
    loss, m = dpo_loss_from_logprob(lp, lp[:B].clone(), lp[B:].clone(), args.beta)
    finite = bool(torch.isfinite(lp).all())
    print(f"logprobs finite: {finite}", flush=True)
    print(f"raw margin mean={m['dpo/margin'].item():.2f}", flush=True)
    print(f"init DPO loss={loss.item():.4f} (expect log2={math.log(2):.4f}); "
          f"logits mean={m['dpo/logits'].item():.2e} (expect 0)", flush=True)
    ok = finite and abs(loss.item() - math.log(2)) < 1e-3 and abs(m["dpo/logits"].item()) < 1e-3
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}", flush=True)


def run_dump_ref(args):
    assert args.ref_out, "--ref_out required for --dump_ref"
    device = torch.device("cuda")
    moshi_lm = MoshiForFinetuning.from_pretrained(
        args.model_dir, device=device, dtype=getattr(torch, args.model_dtype))
    moshi_lm.eval()
    cfg, pe = _load_pairs_and_examples(args, moshi_lm)
    coll = make_collate(cfg["zero_token_id"])
    ref = {}
    bs = 2
    for i in range(0, len(pe), bs):
        chunk = pe[i:i + bs]
        batch, _, _ = coll([{"chosen": x["chosen"], "rejected": x["rejected"],
                             "ref_w": 0.0, "ref_l": 0.0} for x in chunk])
        batch = batch.to(device)
        with torch.no_grad():
            lp = sequence_logprob(moshi_lm, batch, args.model_user_stream)
        B = lp.shape[0] // 2
        for j, x in enumerate(chunk):
            ref[str(x["context_id"])] = [float(lp[j].item()), float(lp[B + j].item())]
        print(f"ref {i + len(chunk)}/{len(pe)}", flush=True)
    json.dump(ref, open(args.ref_out, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {len(ref)} ref logprobs -> {args.ref_out}", flush=True)


def run_train(args):
    import deepspeed
    from accelerate import Accelerator, DeepSpeedPlugin
    from accelerate.utils import DummyOptim, DummyScheduler, InitProcessGroupKwargs, set_seed

    from finetune import get_parameters

    # Initialize the process group BEFORE Accelerator (deepspeed's comm backend
    # must be set for accelerator.prepare); matches finetune_mono_text.py.
    if args.use_deepspeed and args.launcher == "mpi":
        deepspeed.init_distributed(
            dist_backend="nccl",
            timeout=timedelta(seconds=args.process_group_timeout),
        )

    accel_kwargs = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "kwargs_handlers": [InitProcessGroupKwargs(timeout=timedelta(seconds=args.process_group_timeout))],
    }
    if args.use_deepspeed:
        accel_kwargs["deepspeed_plugin"] = DeepSpeedPlugin(
            hf_ds_config=args.deepspeed_config_file,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            zero3_save_16bit_model=True)  # let get_state_dict gather bf16 weights
    accelerator = Accelerator(**accel_kwargs)
    set_seed(args.seed)

    moshi_lm = MoshiForFinetuning.from_pretrained(
        save_dir=args.model_dir, device="cpu", dtype=torch.float32)
    if args.activation_checkpointing:
        import deepspeed as ds
        ac_kwargs = accelerator.deepspeed_plugin.hf_ds_config.config.get("activation_checkpointing", {})
        ds.checkpointing.configure(mpu_=None, deepspeed_config=None, **ac_kwargs)
        moshi_lm.enable_activation_checkpointing(ds.checkpointing.checkpoint)
    for prm in moshi_lm.parameters():
        prm.requires_grad = False
    for prm in get_parameters(moshi_lm, args.parameters_to_finetune):
        prm.requires_grad = True

    cfg = token_id_config(moshi_lm)
    if args.ref_out and os.path.exists(args.ref_out):
        ref_map_raw = json.load(open(args.ref_out))
        ref_map = {int(k): tuple(v) for k, v in ref_map_raw.items()}
    else:
        raise SystemExit("--ref_out (cached ref logprobs) required for training; run --dump_ref first")

    with accelerator.main_process_first():
        ds_ctx = load_context_streams(args, moshi_lm)
        pairs = [json.loads(x) for x in open(args.pairs_jsonl)]
        pair_examples = build_pair_examples(pairs, ds_ctx, cfg, args.prompt_length)
    dataset = PairDataset(pair_examples, ref_map)
    dataloader = DataLoader(dataset, batch_size=args.per_device_train_batch_size,
                            shuffle=True, collate_fn=make_collate(cfg["zero_token_id"]))

    param_groups = [
        {"params": get_parameters(moshi_lm, "tempformer"), "lr": args.tempformer_learning_rate,
         "weight_decay": args.weight_decay},
        {"params": get_parameters(moshi_lm, "depformer"), "lr": args.depformer_learning_rate,
         "weight_decay": args.weight_decay},
    ]
    optimizer = DummyOptim(param_groups, lr=args.tempformer_learning_rate, weight_decay=args.weight_decay)
    optimizer.defaults = {"lr": [args.tempformer_learning_rate, args.depformer_learning_rate]}
    lr_scheduler = None
    if "scheduler" in accelerator.deepspeed_plugin.hf_ds_config.config:
        lr_scheduler = DummyScheduler(optimizer=optimizer, warmup_num_steps=args.num_warmup_steps)

    moshi_lm, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        moshi_lm, optimizer, dataloader, lr_scheduler)

    os.makedirs(args.output_dir, exist_ok=True)
    if accelerator.is_main_process:
        json.dump(vars(args), open(os.path.join(args.output_dir, "config.json"), "w"), indent=2, default=str)

    steps = 0
    for epoch in range(args.num_train_epochs):
        buf = collections.defaultdict(list)
        for step, (batch, ref_w, ref_l) in enumerate(dataloader):
            moshi_lm.train()
            batch = batch.to(accelerator.device)
            ref_w = ref_w.to(accelerator.device)
            ref_l = ref_l.to(accelerator.device)
            lp = sequence_logprob(moshi_lm, batch, args.model_user_stream)
            loss, metrics = dpo_loss_from_logprob(lp, ref_w, ref_l, args.beta)
            accelerator.backward(loss)
            for k, v in metrics.items():
                buf[k].append(v.item())
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                steps += 1
                if steps % args.logging_steps == 0 and accelerator.is_main_process:
                    print(f"epoch {epoch} step {steps} "
                          f"loss={np.mean(buf['loss/dpo']):.4f} "
                          f"acc={np.mean(buf['dpo/acc']):.3f} "
                          f"logits={np.mean(buf['dpo/logits']):.3f} "
                          f"margin={np.mean(buf['dpo/margin']):.1f}", flush=True)
                    buf = collections.defaultdict(list)
                if args.save_steps and steps % args.save_steps == 0:
                    _save_policy(accelerator, moshi_lm, args, f"step_{steps}")
                if args.max_train_steps and steps >= args.max_train_steps:
                    break
    out = _save_policy(accelerator, moshi_lm, args, f"step_{steps}")
    accelerator.end_training()
    if accelerator.is_main_process:
        print(f"DONE. saved -> {out}", flush=True)


def _save_policy(accelerator, moshi_lm, args, tag):
    """Save ONLY the model weights in the MoshiForFinetuning inference format
    (model.safetensors + moshi_lm_kwargs.json), gathering ZeRO-3 shards as bf16
    (needs zero3_save_16bit_model=True). This is ~16GB vs ~94GB for
    accelerator.save_state (which also stores optimizer states we don't need) --
    critical when /groups is near-full -- and loads directly in generate.py
    (--model_dtype bfloat16), no zero_to_fp32 conversion step."""
    import shutil

    from safetensors.torch import save_file

    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(moshi_lm)  # gathers bf16 weights on main proc
    out = os.path.join(args.output_dir, tag)
    if accelerator.is_main_process:
        os.makedirs(out, exist_ok=True)
        save_file({k: v.contiguous() for k, v in state.items()},
                  os.path.join(out, "model.safetensors"))
        shutil.copy(os.path.join(args.model_dir, "moshi_lm_kwargs.json"),
                    os.path.join(out, "moshi_lm_kwargs.json"))
        print(f"saved policy -> {out}", flush=True)
    accelerator.wait_for_everyone()
    return out


def main():
    args = parse_args()
    if args.selftest:
        run_selftest(args)
    elif args.dump_ref:
        run_dump_ref(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
