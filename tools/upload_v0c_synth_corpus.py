"""Upload v0c mstts synth dialogue corpus to private HF dataset repo.

Contents:
  audio/<slot>/<id>.wav   — 46,266 wavs (24 kHz stereo PCM_16, L=A R=B), ~170 GB total
  tokens/<slot>/<id>.npy  — 46,266 17ch int tokens, ~3.4 GB total
  parquet/synth_<slot>.parquet — 10 v1 training-ready files, ~365 MB
  manifest.jsonl          — unified one-row-per-dialogue manifest
  README.md               — dataset card

Strategy: per-slot upload_folder loop (resumable via HF dedup; if interrupted,
re-run and only missing files will re-upload). Avoids the need for symlink staging
or upload_large_folder (which requires newer huggingface_hub).
"""

from __future__ import annotations

import configparser
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo

TOKEN_FILE = Path.home() / ".cache/huggingface/stored_tokens"
TOKEN_NAME = "for_abci_20260517"
REPO_ID = "abePclWaseda/mstts-v0c-synth-jmultiwoz-rpc"
REPO_TYPE = "dataset"

ROOT = Path("/groups/gcg51557/experiments/0162_dialogue_model/moshi-finetune")
SYNTH_DIR = ROOT / "output/mstts_v0c_synth"
PARQUET_DIR = ROOT / "output/mstts_v0c_synth_parquet"

PRODUCTION_SLOTS = [
    "0_5000", "5000_10000", "10000_15000", "15000_20000", "20000_25000",
    "25000_30000", "30000_35000", "35000_40000", "40000_45000", "45000_50000",
]

README = """---
license: cc-by-nc-4.0
language:
  - ja
tags:
  - synthetic-speech
  - dialogue
  - multi-stream-tts
  - japanese
size_categories:
  - 10K<n<100K
task_categories:
  - text-to-speech
  - automatic-speech-recognition
pretty_name: "mstts v0c synth dialogue corpus (JMultiWOZ + RealPersonaChat)"
---

# mstts v0c synth dialogue corpus (JMultiWOZ + RealPersonaChat)

Private backup of the v0c multi-stream TTS synthesis output:
**46,266 two-speaker Japanese dialogue audio clips, ~530 hours total.**

## Source

- **Synthesizer**: [`abePclWaseda/llm-jp-moshi-mstts-v0c-zoom1`](https://huggingface.co/abePclWaseda/llm-jp-moshi-mstts-v0c-zoom1) (private)
- **Text seeds**:
  - [JMultiWOZ](https://github.com/nu-dialogue/jmultiwoz) (CC-BY-SA-4.0, ~8,000 dialogues)
  - [RealPersonaChat](https://github.com/nu-dialogue/real-persona-chat) (CC-BY-SA-4.0, ~38,000 dialogues)

## License

CC-BY-NC-4.0 (inherited from v0c synthesizer, whose base model lineage includes
LaboroTVSpeech, non-commercial). Source text corpora are CC-BY-SA-4.0;
SA propagation does not apply to derived ML model outputs per dataset notes.

## Format

- `audio/<slot>/<dialogue_id>.wav` — 24 kHz stereo PCM_16, **L = speaker A, R = speaker B**
  (per c3455c1 channel swap fix)
- `tokens/<slot>/<dialogue_id>.npy` — 17-channel int tokens
  (row 0: dialogue text, rows 1-8: speaker B audio, rows 9-16: speaker A audio)
- `parquet/synth_<slot>-001-of-001.parquet` — v1 training-ready format:
  `{dialogue_id, A: [9, T_A], B: [9, T_B]}` (1 text + 8 audio rows per speaker)
- `manifest.jsonl` — one row per dialogue:
  `{dialogue_id, slot, corpus, wav, tokens, parquet_slot}`

## Slot structure

Synthesis was distributed across 10 PBS array-job slices of ~5,000 dialogues each:

| Slot | Dialogues |
|---|---:|
| 0_5000 | 5,000 |
| 5000_10000 | 5,000 |
| 10000_15000 | 5,000 |
| 15000_20000 | 5,000 |
| 20000_25000 | 5,000 |
| 25000_30000 | 5,000 |
| 30000_35000 | 5,000 |
| 35000_40000 | 5,000 |
| 40000_45000 | 5,000 |
| 45000_50000 | 1,266 |
| **Total** | **46,266** |

## Known caveats

- Quality not yet validated by random listening across all 46K samples
- v0c is a private NC model; flipping this dataset to public requires re-synthesis
  with a commercial-OK v0d-series model (work in progress)
- High-frequency loss / limited prosody variety (per high-priority feedback from
  Takamichi Sensei, see internal model card)

## Contact

Yuto Abe (@abePclWaseda, abe@pcl.cs.waseda.ac.jp)
"""


def load_token() -> str:
    cfg = configparser.ConfigParser()
    cfg.read(TOKEN_FILE)
    return cfg[TOKEN_NAME]["hf_token"]


def build_unified_manifest_jsonl() -> str:
    lines: list[str] = []
    total = 0
    for slot in PRODUCTION_SLOTS:
        mfp = SYNTH_DIR / slot / "manifest.json"
        if not mfp.exists():
            print(f"WARN: no manifest at {mfp}, skipping slot {slot}")
            continue
        entries = json.loads(mfp.read_text())
        for entry in entries:
            wav_path = Path(entry["wav"])
            dialogue_id = wav_path.stem
            lines.append(json.dumps({
                "dialogue_id": dialogue_id,
                "slot": slot,
                "corpus": entry.get("corpus", "?"),
                "wav": f"audio/{slot}/{wav_path.name}",
                "tokens": f"tokens/{slot}/{dialogue_id}.npy",
                "parquet_slot": f"parquet/synth_{slot}-001-of-001.parquet",
            }, ensure_ascii=False))
            total += 1
    print(f"unified manifest: {total} rows")
    return "\n".join(lines) + "\n"


def upload_dir_with_retry(api: HfApi, *, folder_path: Path, path_in_repo: str,
                          token: str, max_retries: int = 3) -> None:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_folder(
                folder_path=str(folder_path),
                path_in_repo=path_in_repo,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=token,
                ignore_patterns=[".DS_Store", "*.pyc", "__pycache__/"],
            )
            return
        except Exception as e:  # noqa: BLE001 — broad to retry on any transient HF error
            last_err = e
            wait = 30 * attempt
            print(f"  attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}")
            print(f"  sleeping {wait}s before retry")
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


def main() -> int:
    token = load_token()
    print(f"loaded token (len={len(token)}) — will NOT print")
    api = HfApi(token=token)

    print(f"\ncreate_repo: {REPO_ID} (dataset, private)")
    create_repo(REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True, token=token)

    # 1. README + manifest (real files, small)
    print("\n[1/3] upload README + manifest.jsonl")
    manifest_text = build_unified_manifest_jsonl()
    api.upload_file(
        path_or_fileobj=README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID, repo_type=REPO_TYPE, token=token,
    )
    api.upload_file(
        path_or_fileobj=manifest_text.encode("utf-8"),
        path_in_repo="manifest.jsonl",
        repo_id=REPO_ID, repo_type=REPO_TYPE, token=token,
    )

    # 2. parquet (~365 MB, fast)
    print("\n[2/3] upload parquet/")
    upload_dir_with_retry(
        api, folder_path=PARQUET_DIR, path_in_repo="parquet", token=token,
    )

    # 3. per-slot audio + tokens (resumable: HF dedup skips already-uploaded files)
    print("\n[3/3] upload audio + tokens per slot (10 slots, ~170 GB total)")
    t_total = time.time()
    for i, slot in enumerate(PRODUCTION_SLOTS, 1):
        print(f"\n--- slot {i}/{len(PRODUCTION_SLOTS)}: {slot} ---")
        audio_src = SYNTH_DIR / slot / "decoded_audio"
        tokens_src = SYNTH_DIR / slot / "generated_tokens"

        if audio_src.is_dir():
            t0 = time.time()
            print(f"  audio: {audio_src} -> audio/{slot}/")
            upload_dir_with_retry(
                api, folder_path=audio_src, path_in_repo=f"audio/{slot}", token=token,
            )
            print(f"  audio done ({time.time() - t0:.0f}s)")
        else:
            print(f"  skip audio (missing: {audio_src})")

        if tokens_src.is_dir():
            t0 = time.time()
            print(f"  tokens: {tokens_src} -> tokens/{slot}/")
            upload_dir_with_retry(
                api, folder_path=tokens_src, path_in_repo=f"tokens/{slot}", token=token,
            )
            print(f"  tokens done ({time.time() - t0:.0f}s)")
        else:
            print(f"  skip tokens (missing: {tokens_src})")

    print(f"\nALL DONE: https://huggingface.co/datasets/{REPO_ID} (private)")
    print(f"Total time: {(time.time() - t_total) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
