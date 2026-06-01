"""Upload v1+v0csynth (bf16 cleaned) bundle to private HF model repo, matching
the structure of abePclWaseda/llm-jp-moshi-v1 for `uv run -m moshi.server` use.

Reads HF write token from ~/.cache/huggingface/stored_tokens 'for_abci_20260517'.
Never prints the token.
"""

from __future__ import annotations

import configparser
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, hf_hub_download

TOKEN_FILE = Path.home() / ".cache/huggingface/stored_tokens"
TOKEN_NAME = "for_abci_20260517"
REPO_ID = "abePclWaseda/llm-jp-moshi-v1-plus-v0csynth"
REPO_TYPE = "model"

ROOT = Path("/groups/gcg51557/experiments/0162_dialogue_model/moshi-finetune")
BF16_DIR = ROOT / "output/v1_plus_v0csynth/step_2757_cleaned_bf16"
STAGE = ROOT / ".tmp/v1_plus_v0csynth_upload"

README = """---
license: cc-by-nc-4.0
language:
  - ja
base_model:
  - abePclWaseda/llm-jp-moshi-v1
library_name: moshi
tags:
  - dialogue
  - spoken-dialogue
  - moshi
  - private
---

# llm-jp-moshi-v1-plus-v0csynth (private)

Continued-finetune of [`abePclWaseda/llm-jp-moshi-v1`](https://huggingface.co/abePclWaseda/llm-jp-moshi-v1)
on the **v0c mstts synth dialogue corpus** (46,266 two-speaker dialogues, ~530 h,
JMultiWOZ + RealPersonaChat seeds synthesized via the v0c mstts model).

Built for ICASSP 2027 Phase 2 "downstream utility of synthesized dialogue corpus" experiment.

## Training

| Stage | Data | Steps | LR | Notes |
|---|---|---:|---|---|
| 0 (base) | = HF llm-jp-moshi-v1 | step 9282 | — | J-CHAT → Zoom1 7ep |
| **+ this** | **v0c mstts synth (46,266 dialogues)** | **+2757 (1ep)** | tlr=2e-6 / dlr=4e-6 | 1 node × 8 GPU, eff bs 16, bf16 mixed-precision (DeepSpeed Zero3) |

## Format

- `model.safetensors` — bf16 cleaned (user-stream removed, `dep_q=8`)
- `moshi_lm_kwargs.json` — model config
- `tokenizer-e351c8d8-checkpoint125.safetensors` — Mimi codec
- `tokenizer_spm_32k_3.model` — text SentencePiece tokenizer

`moshi.server`-compatible — same interface as the public v1 release.

## Usage

```bash
uvx --from huggingface_hub hf download abePclWaseda/llm-jp-moshi-v1-plus-v0csynth \\
    --local-dir llm-jp-moshi-v1-plus-v0csynth
cd llm-jp-moshi-v1-plus-v0csynth
uv run -m moshi.server  # configure model_path / tokenizer_path appropriately
```

## Eval (Phase 2.2 Zoom1 continuation, 50 samples, T=0.8)

| Metric | v1 (baseline) | v1+v0csynth | Δ |
|---|---:|---:|---:|
| CER vs ref (faster-whisper) | 71.24 % | 105.30 % | -34 pp (worse) |
| UTMOS (L+R mean, normalized) | 1.857 | **1.963** | **+0.106** |

→ Trade-off: audio quality improves, lexical faithfulness to Zoom1 conversational
style decreases (likely because v0c synth distribution = task-oriented JMultiWOZ
+ casual RPC, not Zoom1 conversational).

## License

CC-BY-NC-4.0 — inherited from the v0c synth corpus (v0c model lineage contains
LaboroTVSpeech, non-commercial).

## Contact

Yuto Abe (@abePclWaseda, abe@pcl.cs.waseda.ac.jp)
"""


def load_token() -> str:
    cfg = configparser.ConfigParser()
    cfg.read(TOKEN_FILE)
    return cfg[TOKEN_NAME]["hf_token"]


def stage_files(token: str) -> None:
    """Build STAGE/ tree mirroring the public v1 release structure."""
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    # model + kwargs from bf16 cleaned dir
    for name in ("model.safetensors", "moshi_lm_kwargs.json"):
        src = BF16_DIR / name
        if not src.exists():
            raise SystemExit(f"missing source: {src}")
        # use symlinks for the large model file to save disk
        if name == "model.safetensors":
            (STAGE / name).symlink_to(src.resolve())
        else:
            shutil.copy(src, STAGE / name)

    # tokenizers from public v1 (identical Mimi + text spm)
    for name in (
        "tokenizer-e351c8d8-checkpoint125.safetensors",
        "tokenizer_spm_32k_3.model",
    ):
        p = hf_hub_download("abePclWaseda/llm-jp-moshi-v1", name, token=token)
        shutil.copy(p, STAGE / name)

    # README
    (STAGE / "README.md").write_text(README, encoding="utf-8")

    # Summary
    print("staged:")
    for f in sorted(STAGE.iterdir()):
        real = f.resolve() if f.is_symlink() else f
        sz = real.stat().st_size
        link = " (symlink)" if f.is_symlink() else ""
        print(f"  {f.name}: {sz / 1e6:.2f} MB{link}")


def main() -> int:
    if not BF16_DIR.exists():
        raise SystemExit(f"bf16 cleaned ckpt missing: {BF16_DIR}\n"
                          f"run pbs/run_v1_plus_v0csynth_clean_bf16.sh first")

    token = load_token()
    print(f"loaded token (len={len(token)}) — will NOT print")
    api = HfApi(token=token)

    print(f"\ncreate_repo: {REPO_ID} ({REPO_TYPE}, private)")
    create_repo(REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True, token=token)

    print("\nstaging bundle")
    stage_files(token)

    # Upload small files first (README + kwargs + tokenizers)
    print("\nupload small files (README + kwargs + tokenizers)")
    for name in (
        "README.md",
        "moshi_lm_kwargs.json",
        "tokenizer-e351c8d8-checkpoint125.safetensors",
        "tokenizer_spm_32k_3.model",
    ):
        print(f"  {name}")
        api.upload_file(
            path_or_fileobj=str(STAGE / name),
            path_in_repo=name,
            repo_id=REPO_ID, repo_type=REPO_TYPE, token=token,
        )

    # Upload model.safetensors (large, ~15 GB bf16)
    print(f"\nupload model.safetensors (~15 GB bf16, ~3-5 min)")
    api.upload_file(
        path_or_fileobj=str((BF16_DIR / "model.safetensors").resolve()),
        path_in_repo="model.safetensors",
        repo_id=REPO_ID, repo_type=REPO_TYPE, token=token,
    )

    print(f"\nDONE: https://huggingface.co/{REPO_ID} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
