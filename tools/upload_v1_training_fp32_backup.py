"""Upload llm-jp-moshi-v1 training-state fp32 ckpt to a private HF repo as backup.

Reads HF write token from ~/.cache/huggingface/stored_tokens (INI), entry
'for_abci_20260517'. Never prints the token.

Source ckpt: output/moshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_7epochs_1node_exp/step_9282_fp32
Target repo: abePclWaseda/llm-jp-moshi-v1-training-fp32 (private)
"""

from __future__ import annotations

import configparser
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

TOKEN_FILE = Path.home() / ".cache/huggingface/stored_tokens"
TOKEN_NAME = "for_abci_20260517"
REPO_ID = "abePclWaseda/llm-jp-moshi-v1-training-fp32"
SRC_DIR = Path(
    "/groups/gcg51557/experiments/0162_dialogue_model/moshi-finetune/"
    "output/moshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_7epochs_1node_exp/"
    "step_9282_fp32"
)

README = """---
license: cc-by-nc-4.0
language:
  - ja
base_model:
  - kyutai/moshiko-pytorch-bf16
library_name: moshi
tags:
  - private-backup
  - training-state
---

# llm-jp-moshi-v1-training-fp32 (Private backup)

Training-state fp32 checkpoint that is the **direct source** of the public
[abePclWaseda/llm-jp-moshi-v1](https://huggingface.co/abePclWaseda/llm-jp-moshi-v1).

Public v1 was produced from this ckpt via:

```bash
uv run -m tools.clean_moshi \\
    --moshi_ft_dir <this repo> \\
    --save_dir step_9282_cleaned \\
    --model_dtype bfloat16 \\
    --remove_modules_for_user_stream
```

## Contents
- `model.safetensors` (~33.5 GB, fp32) — `MoshiForFinetuning` wrapper format
  - `dep_q=16`, `depformer_context=16` (includes user-stream output modules)
  - Suitable for `finetune.py --model_dir <this repo>` continue-FT
- `moshi_lm_kwargs.json` — model config

## Purpose
- **Source-of-truth backup** against local-disk loss
- Base for any v1-derived continue-finetune (e.g. v1 + synth corpus, v1 + new
  corpus FT, v1.x lineage extensions)

## Curriculum (same as public v1)
1. **Stage 1 — J-CHAT** (Podcast + YouTube): 1 ep, step_8880, tlr=3e-5/dlr=3e-5
2. **Stage 2 — LLM-jp Zoom1**: 7 ep, step_9282, tlr=2e-6/dlr=4e-6

Loss weights: semantic=100.0, acoustic=1.0, text_padding=0.5

## Access
Private. Contact @abePclWaseda (abe@pcl.cs.waseda.ac.jp) for access.

## Why fp32 not bf16?
Continue-finetune needs fp32 master weights + the `MoshiForFinetuning` wrapper
(DeepSpeed Zero3-ready). The public bf16 cleaned model is inference-only and
cannot be resumed as a training base.
"""


def load_token() -> str:
    cfg = configparser.ConfigParser()
    cfg.read(TOKEN_FILE)
    if TOKEN_NAME not in cfg:
        raise SystemExit(f"token entry '{TOKEN_NAME}' not in {TOKEN_FILE}")
    return cfg[TOKEN_NAME]["hf_token"]


def main() -> int:
    token = load_token()
    print(f"loaded token (len={len(token)}) — will NOT print")
    api = HfApi(token=token)

    # Verify source files
    model_file = SRC_DIR / "model.safetensors"
    kwargs_file = SRC_DIR / "moshi_lm_kwargs.json"
    for f in (model_file, kwargs_file):
        if not f.exists():
            raise SystemExit(f"missing: {f}")
    size_gb = model_file.stat().st_size / 1e9
    print(f"src: {SRC_DIR}")
    print(f"  model.safetensors: {size_gb:.2f} GB")
    print(f"  moshi_lm_kwargs.json: {kwargs_file.stat().st_size} bytes")

    print(f"\ncreate_repo: {REPO_ID} (private)")
    create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True, token=token)

    # README first
    print("upload README.md")
    api.upload_file(
        path_or_fileobj=README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        token=token,
    )

    print("upload moshi_lm_kwargs.json")
    api.upload_file(
        path_or_fileobj=str(kwargs_file),
        path_in_repo="moshi_lm_kwargs.json",
        repo_id=REPO_ID,
        token=token,
    )

    print(f"upload model.safetensors ({size_gb:.1f} GB) — this will take a while")
    api.upload_file(
        path_or_fileobj=str(model_file),
        path_in_repo="model.safetensors",
        repo_id=REPO_ID,
        token=token,
    )

    print(f"\nDONE: https://huggingface.co/{REPO_ID} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
