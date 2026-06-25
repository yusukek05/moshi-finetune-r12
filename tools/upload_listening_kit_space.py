"""Upload the 98-pair cascade vs mstts listening_kit to a private HF Space.

Source:
  /home/acg17145sv/projects/icassp-2027-mstts/dialogues/listening_kit/
    cascade/jmw_*.wav  rpc_*.wav  smj_*.wav   (98, symlinks → cascade_output/<id>/audio_norm.wav, 44.1 kHz stereo)
    mstts/  ...                                (98, symlinks → mstts_output/<id>/audio_norm.wav,   24   kHz stereo)
    index.html                                  (204 KB self-contained browser)

Symlinks are resolved with `cp -dereference` into a staging dir, then
upload_folder() pushes everything to the Space (static SDK, no build step).

Audio is uploaded AS-IS: cascade stays 44.1 kHz / mstts stays 24 kHz.
The recipient is an audio expert reviewing the *actual training data* —
re-sampling would mask the property we want them to hear.

Reads HF write token from ~/.cache/huggingface/stored_tokens 'for_abci_20260517'.
Never prints the token.
"""

from __future__ import annotations

import configparser
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo

TOKEN_FILE = Path.home() / ".cache/huggingface/stored_tokens"
TOKEN_NAME = "for_abci_20260517"
REPO_ID = "abePclWaseda/icassp-mstts-cascade-listening-kit"
REPO_TYPE = "space"
SPACE_SDK = "static"

SRC = Path("/home/acg17145sv/projects/icassp-2027-mstts/dialogues/listening_kit")
STAGE = Path("/home/acg17145sv/experiments/0162_dialogue_model/moshi-finetune/.tmp/listening_kit_stage")

README = """---
title: ICASSP mstts vs cascade listening kit
emoji: 🎧
colorFrom: indigo
colorTo: gray
sdk: static
pinned: false
short_description: Private A/B listening kit, 98 dialogues × 2 TTS pipelines.
---

# ICASSP 2027 — mstts (v0c) vs cascade (Style-Bert-VITS2) listening kit

98 dialogues, each voiced by **two TTS pipelines from the same script**:

| Pipeline | Voice | Sample rate | Pipeline details |
|---|---|---|---|
| **cascade** | Zoom1-derived SBV2 (`llmjp_zoom1_0001_l/r`) | **44.1 kHz** stereo (SBV2 native) | per-turn TTS + 200 ms heuristic gap, L=A R=B |
| **mstts**   | mstts v0c (LLM-jp-Moshi + Zoom1 mstts FT)    | **24 kHz**   stereo (Mimi codec native) | single LM joint generation, L=A R=B |

Open `index.html` and play the two columns side-by-side. Dialogues are grouped by style:

- **casual** — RealPersonaChat-style chat (`rpc_*`)
- **qa-expansion** — spoken-magpie-ja-style QA (`smj_*`)
- **task-oriented** — JMultiWOZ-style task dialogue (`jmw_*`)

## Important: sample rate disparity is *not* a TTS-quality difference

Cascade is 44.1 kHz because Style-Bert-VITS2 outputs at 44.1 kHz natively;
mstts is 24 kHz because Mimi codec operates at 24 kHz. The high-band roll-off
in mstts is a codec/architecture constraint, not a synthesis-quality issue.
This kit shows the **actual audio used as training data**, unmodified.

## License

Audio derived from LLM-jp-Zoom1 (LLM-JP internal use). Distribution restricted —
do not redistribute outside the listening-review context.

## Contact

Yuto Abe (@abePclWaseda, abe@pcl.cs.waseda.ac.jp)
"""


def load_token() -> str:
    cfg = configparser.ConfigParser()
    cfg.read(TOKEN_FILE)
    return cfg[TOKEN_NAME]["hf_token"]


def stage_files() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    # cp --dereference: follow symlinks and copy real files.
    # rsync -L would also work; cp is enough for this flat-ish dir.
    print(f"copying (dereferenced) {SRC} → {STAGE}")
    t0 = time.time()
    subprocess.run(
        ["cp", "-r", "-L", str(SRC) + "/.", str(STAGE)],
        check=True,
    )
    print(f"  copy took {time.time() - t0:.1f}s")

    # Overwrite README so HF Space picks up the YAML front-matter (no front-matter → no Space).
    (STAGE / "README.md").write_text(README, encoding="utf-8")

    n_cascade = len(list((STAGE / "cascade").glob("*.wav")))
    n_mstts = len(list((STAGE / "mstts").glob("*.wav")))
    size_cascade_mb = sum(f.stat().st_size for f in (STAGE / "cascade").glob("*.wav")) / 1e6
    size_mstts_mb = sum(f.stat().st_size for f in (STAGE / "mstts").glob("*.wav")) / 1e6
    print(f"staged: cascade {n_cascade} wavs ({size_cascade_mb:.1f} MB), "
          f"mstts {n_mstts} wavs ({size_mstts_mb:.1f} MB)")


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"source missing: {SRC}")

    token = load_token()
    print(f"loaded token (len={len(token)}) — will NOT print")
    api = HfApi(token=token)

    print(f"\ncreate_repo: {REPO_ID} ({REPO_TYPE}, private, sdk={SPACE_SDK})")
    create_repo(
        REPO_ID,
        repo_type=REPO_TYPE,
        space_sdk=SPACE_SDK,
        private=True,
        exist_ok=True,
        token=token,
    )

    print("\nstaging files (dereferencing symlinks)")
    stage_files()

    print("\nuploading folder (allow_patterns: *.html, *.md, *.wav)")
    api.upload_folder(
        folder_path=str(STAGE),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=token,
        allow_patterns=["*.html", "*.md", "cascade/*.wav", "mstts/*.wav"],
        commit_message="initial upload: 98-pair cascade vs mstts listening kit",
    )

    print(f"\nDONE: https://huggingface.co/spaces/{REPO_ID} (private)")
    print("Invite reviewers via the Space → Settings → Collaborators.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
