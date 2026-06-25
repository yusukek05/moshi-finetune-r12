"""Upload v1+cascadesynth (bf16 cleaned) bundle to private HF model repo.
Mirrors upload_v1_plus_v0csynth.py exactly (same structure, different source).

License: CC-BY-4.0 — cascade lineage is LaboroTV-free (Style-Bert-VITS2 +
LLM-jp-3 + JMultiWOZ/RPC seeds, all commercially clean). This is the
**commercially-clean control** for v0c synth.

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
REPO_ID = "abePclWaseda/llm-jp-moshi-v1-plus-cascadesynth"
REPO_TYPE = "model"

ROOT = Path("/groups/gcg51557/experiments/0162_dialogue_model/moshi-finetune")
BF16_DIR = ROOT / "output/v1_plus_cascadesynth/step_2714_cleaned_bf16"
STAGE = ROOT / ".tmp/v1_plus_cascadesynth_upload"

README = """---
license: cc-by-4.0
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

# llm-jp-moshi-v1-plus-cascadesynth (private)

Continued-finetune of [`abePclWaseda/llm-jp-moshi-v1`](https://huggingface.co/abePclWaseda/llm-jp-moshi-v1)
on a **cascade-synthesized dialogue corpus** (46,266 two-speaker dialogues,
JMultiWOZ + RealPersonaChat seeds, voiced by Style-Bert-VITS2 with
`llmjp_zoom1_0001_{l,r}` speakers and a 200 ms heuristic turn merger).

Built as the **commercially-clean control** for `v1+v0csynth`. Both use the
exact same `text_chat` prompts (47,816 dialogues from `output/text_corpora`);
only the voicing pipeline differs:

- `v1+v0csynth` — multi-stream TTS (single LM joint generation, **CC-BY-NC-4.0**
  because v0c lineage includes LaboroTVSpeech)
- `v1+cascadesynth` — cascade TTS (LLM-jp-3 text + per-turn SBV2 + heuristic
  merge, **CC-BY-4.0** lineage)

## Training

| Stage | Data | Steps | LR | Notes |
|---|---|---:|---|---|
| 0 (base) | = HF llm-jp-moshi-v1 | step 9282 | — | J-CHAT → Zoom1 7ep |
| **+ this** | **cascade synth (46,266 dialogues)** | **+2714 (1ep)** | tlr=2e-6 / dlr=4e-6 | 1 node × 8 GPU, eff bs 16, bf16 mixed-precision (DeepSpeed Zero3) |

## Format

- `model.safetensors` — bf16 cleaned (user-stream removed, `dep_q=8`)
- `moshi_lm_kwargs.json` — model config
- `tokenizer-e351c8d8-checkpoint125.safetensors` — Mimi codec
- `tokenizer_spm_32k_3.model` — text SentencePiece tokenizer

`moshi.server`-compatible — same interface as the public v1 release.

## Usage

```bash
uvx --from huggingface_hub hf download abePclWaseda/llm-jp-moshi-v1-plus-cascadesynth \\
    --local-dir llm-jp-moshi-v1-plus-cascadesynth
cd llm-jp-moshi-v1-plus-cascadesynth
uv run -m moshi.server  # configure model_path / tokenizer_path appropriately
```

## Why this control exists

`v1+v0csynth` exhibited a slight audio-vs-text delay in moshi.server interactive
use (2026-06-01, user feedback). The hypothesis is that v0c synth dialogues
contain a learned "pre-roll" (breath / lip-smack / silence before each turn)
inherited from Zoom1 real audio. Cascade synthesis uses Style-Bert-VITS2
per-turn with explicit turn boundaries (200 ms heuristic gap), so no such
pre-roll should be learned. Comparing v1+v0csynth and v1+cascadesynth on the
exact same prompts isolates this effect.

## License

CC-BY-4.0 — all components (LLM-jp-Moshi v1 base, JMultiWOZ + RealPersonaChat
text seeds, LLM-jp-3 text generation, Style-Bert-VITS2 with zoom1-derived
voices) are commercially clean.

## Contact

Yuto Abe (@abePclWaseda, abe@pcl.cs.waseda.ac.jp)
"""


def load_token() -> str:
    cfg = configparser.ConfigParser()
    cfg.read(TOKEN_FILE)
    return cfg[TOKEN_NAME]["hf_token"]


def stage_files(token: str) -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    for name in ("model.safetensors", "moshi_lm_kwargs.json"):
        src = BF16_DIR / name
        if not src.exists():
            raise SystemExit(f"missing source: {src}")
        if name == "model.safetensors":
            (STAGE / name).symlink_to(src.resolve())
        else:
            shutil.copy(src, STAGE / name)

    for name in (
        "tokenizer-e351c8d8-checkpoint125.safetensors",
        "tokenizer_spm_32k_3.model",
    ):
        p = hf_hub_download("abePclWaseda/llm-jp-moshi-v1", name, token=token)
        shutil.copy(p, STAGE / name)

    (STAGE / "README.md").write_text(README, encoding="utf-8")

    print("staged:")
    for f in sorted(STAGE.iterdir()):
        real = f.resolve() if f.is_symlink() else f
        sz = real.stat().st_size
        link = " (symlink)" if f.is_symlink() else ""
        print(f"  {f.name}: {sz / 1e6:.2f} MB{link}")


def main() -> int:
    if not BF16_DIR.exists():
        raise SystemExit(f"bf16 cleaned ckpt missing: {BF16_DIR}\n"
                          f"run pbs/run_v1_plus_cascadesynth_post.sh first")

    token = load_token()
    print(f"loaded token (len={len(token)}) — will NOT print")
    api = HfApi(token=token)

    print(f"\ncreate_repo: {REPO_ID} ({REPO_TYPE}, private)")
    create_repo(REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True, token=token)

    print("\nstaging bundle")
    stage_files(token)

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
