"""Upload ccaudio_v2_full (cleaned re-segmented + re-transcribed) tokenized
parquet + transcripts.jsonl shards to a private HF dataset repo.

Source:
  processed_data/ccaudio_v2_full/rinna_gpt2-kyutai_mimi-q16/
    shard-NNNNN.parquet           (Mimi q16 + rinna_gpt2 SP, 1,894,308 rows total)
    shard-NNNNN.transcripts.jsonl (raw text per segment)

Reads HF write token from ~/.cache/huggingface/stored_tokens 'for_abci_20260517'.
Never prints the token.
"""

from __future__ import annotations

import configparser
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo

TOKEN_FILE = Path.home() / ".cache/huggingface/stored_tokens"
TOKEN_NAME = "for_abci_20260517"
REPO_ID = "abePclWaseda/ccaudio-cc-clean-v2"
REPO_TYPE = "dataset"

ROOT = Path("/groups/gcg51557/experiments/0162_dialogue_model/moshi-finetune")
SRC_DIR = ROOT / "processed_data/ccaudio_v2_full/rinna_gpt2-kyutai_mimi-q16"
MANIFEST_FILE = ROOT / "processed_data/ccaudio_v2_full/tar_manifest.txt"

README = """---
license: cc-by-4.0
language:
  - ja
task_categories:
  - automatic-speech-recognition
  - text-to-speech
pretty_name: ccaudio CC-clean v2 (re-segmented + re-transcribed)
tags:
  - audio
  - japanese
  - podcast
  - tokenized
  - mimi
  - private
size_categories:
  - 1M<n<10M
---

# ccaudio-cc-clean-v2 (private)

**Cleaned** Japanese CC-licensed podcast corpus, re-segmented from long-form
podcast cuts and re-transcribed with Whisper. Tokenized via Mimi codec (q=16)
and rinna_gpt2 SentencePiece for direct ingestion into LLM-jp-Moshi / mstts
Stage-1 mono pretraining.

Built for the **LaboroTV-free mstts rebuild (v0d_*)** experiment as a
commercially-clean replacement for LaboroTVSpeech.

## Source

Raw RSS-podcast tarballs from
`/groups/gcg51557/experiments/0167_cc_audio/asai/ccaudio_rss_raw_all/`
(crawl by 0167 浅井さん, CC-license filtered upstream).

The v1 transcripts shipped with the upstream Lhotse cutset had Whisper
hallucination on long cuts (repetition collapse, language confusion). This v2
re-runs segmentation + ASR on the raw audio with stricter filtering.

## Scale

| | v2_full |
|---|---:|
| Shards | 587 |
| Rows (segments) | 1,894,308 |
| Approx audio hours | ~2,462 |
| Tokenized size | 2.55 GB |
| Includes raw transcripts | Yes (.transcripts.jsonl per shard) |

## Files

```
shard-00000.parquet            Mimi q16 audio + rinna_gpt2 text tokens
shard-00000.transcripts.jsonl  raw text per segment (key/start/end/dur/text)
...
shard-00593.parquet
shard-00593.transcripts.jsonl
```

## Schema (parquet)

| Column | Type | Note |
|---|---|---|
| `__key__` | string | e.g. `ccaudio/shard00000/recording.000000/audio_00000000/seg00003` |
| `A_text` | list&lt;int64&gt; | rinna_gpt2 SP tokens |
| `A_audio` | list&lt;list&lt;int64&gt;&gt; | Mimi codes, 16 codebooks × T frames @ 12.5 Hz |

Compatible with `data_utils.py` of LLM-jp-Moshi (same schema as J-CHAT-mono).

## transcripts.jsonl line format

```json
{"key": "ccaudio/shard00000/recording.000000/audio_00000000/seg00003",
 "start": 5.98, "end": 8.92, "dur": 2.94,
 "text": "さあここからは情熱ものづくりハイスクールのお時間です"}
```

## Filtering pipeline (v2)

1. Take raw stereo flac (28800 Hz, channel 0) from upstream Lhotse cutset.
2. **Re-segment** by VAD into ~3-20 s segments.
3. **Re-transcribe** each segment with Whisper (small enough chunks → no
   repetition collapse).
4. Filter: `rep_ratio &gt; 0.05` (4-gram repetition) or `non_ja_ratio &gt; 0.10`
   removed.
5. Tokenize: audio → Mimi q16 (8 dummy semantic + 8 acoustic codebooks), text
   → rinna_gpt2 SP.

## License

**CC-BY-4.0**. The upstream `ccaudio_rss_raw_all/` corpus is collected from
RSS feeds that publish under CC licenses. All non-CC items were filtered at
crawl time by the upstream collector. This re-segmentation + re-transcription
does not alter the underlying license.

→ Commercially usable (with attribution to the original podcast publishers,
   tracked in upstream `tar_manifest.txt`).

## Lineage (v0d_v3 mstts)

This corpus is the Stage-1 mono substrate for `v0d_v3` mstts rebuild
(replacing LaboroTVSpeech). See LLM-jp-Moshi-mstts repo for downstream use.

## Contact

Yuto Abe (@abePclWaseda, abe@pcl.cs.waseda.ac.jp)
"""


def load_token() -> str:
    cfg = configparser.ConfigParser()
    cfg.read(TOKEN_FILE)
    return cfg[TOKEN_NAME]["hf_token"]


def main() -> int:
    if not SRC_DIR.exists():
        raise SystemExit(f"source dir missing: {SRC_DIR}")

    parquets = sorted(SRC_DIR.glob("shard-*.parquet"))
    jsonls = sorted(SRC_DIR.glob("shard-*.transcripts.jsonl"))
    print(f"src: {SRC_DIR}")
    print(f"  parquet shards: {len(parquets)}")
    print(f"  jsonl shards:   {len(jsonls)}")
    total_bytes = sum(p.stat().st_size for p in parquets + jsonls)
    print(f"  total: {total_bytes / 1e9:.2f} GB")

    token = load_token()
    print(f"\nloaded token (len={len(token)}) — will NOT print")
    api = HfApi(token=token)

    print(f"\ncreate_repo: {REPO_ID} ({REPO_TYPE}, private)")
    create_repo(REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True, token=token)

    # 1) Upload README + tar_manifest first (small, fast feedback)
    print("\nupload README.md")
    api.upload_file(
        path_or_fileobj=README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID, repo_type=REPO_TYPE, token=token,
        commit_message="add README",
    )
    if MANIFEST_FILE.exists():
        print(f"upload tar_manifest.txt ({MANIFEST_FILE.stat().st_size} bytes)")
        api.upload_file(
            path_or_fileobj=str(MANIFEST_FILE),
            path_in_repo="tar_manifest.txt",
            repo_id=REPO_ID, repo_type=REPO_TYPE, token=token,
            commit_message="add upstream tar manifest",
        )

    # 2) Upload data/ folder (parquet + jsonl) in batches of ~50 shards per commit.
    #    upload_folder will batch internally; we just call it once with the source
    #    dir and let huggingface_hub handle chunking.
    print(f"\nupload_folder: {SRC_DIR} -> data/")
    t0 = time.time()
    try:
        api.upload_folder(
            folder_path=str(SRC_DIR),
            path_in_repo="data",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=token,
            commit_message=f"add {len(parquets)} parquet shards + transcripts",
            allow_patterns=["shard-*.parquet", "shard-*.transcripts.jsonl"],
        )
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"upload_folder failed after {elapsed/60:.1f} min: {exc!r}")
        print("retrying once after 30s")
        time.sleep(30)
        api.upload_folder(
            folder_path=str(SRC_DIR),
            path_in_repo="data",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=token,
            commit_message=f"add {len(parquets)} parquet shards + transcripts (retry)",
            allow_patterns=["shard-*.parquet", "shard-*.transcripts.jsonl"],
        )

    elapsed = time.time() - t0
    print(f"\nupload_folder done in {elapsed/60:.1f} min")
    print(f"\nDONE: https://huggingface.co/datasets/{REPO_ID} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
