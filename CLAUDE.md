# CLAUDE.md

このファイルは、Claude Code (または Claude Agent SDK) がこのリポジトリで作業する際に参照する開発ガイドです。**修正・追記は歓迎**。`@阿部雄斗 (abe@pcl.cs.waseda.ac.jp)` が管理。

## プロジェクト概要

LLM-jp-Moshi シリーズ + mstts (multi-stream dialogue TTS) サブプロジェクトの fine-tuning コード。Moshi/Moshika アーキテクチャを使った日本語対話音声モデル開発。

- **メインブランチ (main)**: LLM-jp-Moshi 本体 (v1〜v1.2)
- **feature/multistream-tts (本ブランチ)**: mstts サブプロジェクト。テキスト対話 → 2話者音声合成。0178_dialogue_tts/moshi-finetuning (名大・大橋さん) からインポート (2026-05-09、許諾あり)

## ディレクトリ構成

```
moshi-finetune/
├── finetune.py                  # LLM-jp-Moshi v1 系の学習エントリ
├── generate.py                  # v1 系の推論
├── data_utils.py / utils/       # 共通ユーティリティ (parquet → streams)
├── models/                      # LLM-jp-Moshi モデル定義
├── tools/                       # 補助ツール
│   ├── prepare_dataset.py       # tokenized text+audio → parquet
│   ├── tokenize_audio_from_dir.py
│   ├── tokenize_text_from_dir.py
│   ├── decode_tokens.py         # tokens → wav (Mimi codec)
│   └── zero_to_fp32.py          # DeepSpeed shard → 単一 fp32 safetensors
├── mstts/                       # mstts サブプロジェクト
│   ├── finetune_ms_tts.py       # mstts 学習エントリ
│   ├── run_ms_tts.py            # mstts バッチ推論
│   ├── data_utils.py            # multi-stream 用前処理 (merge_text_tokens 等)
│   ├── models/                  # Moshi LM / MoshiForMultiStreamTTS
│   ├── data_prep/               # コーパス → mstts JSON / mstts → v1 parquet
│   │   ├── jmultiwoz_to_mstts.py
│   │   ├── real_persona_chat_to_mstts.py
│   │   └── mstts_tokens_to_v1_parquet.py
│   ├── hf_inference/            # HF v0c repo に bundle される推論コード
│   ├── ds_configs/              # DeepSpeed config (zero3-bf16-act_ckpt 等)
│   ├── inference_inputs/        # text_chat サンプル + prompt_streams
│   └── MODEL_CARD_DRAFT.md      # 一般公開時のモデルカード雛形
├── pbs/                         # PBS ジョブスクリプト (ABCI 用)
├── docs/space/                  # HF Space (abePclWaseda/llm-jp-moshi-mstts) ソース
└── output/                      # 学習・推論成果物 (git無視)
```

## 開発環境 (重要)

ABCI 上で動作。**必ず module load + uv** を使うこと。

```bash
module purge
module load cuda/12.6/12.6.1
module load hpcx/2.20            # mpirun 用 (学習・推論時)
module load python/3.12/3.12.9
uv sync --python 3.12            # .venv セットアップ
```

### 必須環境変数

```bash
export NO_TORCH_COMPILE=1        # mstts/finetune_ms_tts.py に assert あり
export NCCL_NVLS_ENABLE=0        # マルチノード NCCL 安定化
unset NCCL_ASYNC_ERROR_HANDLING  # 古い変数を消す
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

### よくあるハマり

| 症状 | 原因 / 対処 |
|---|---|
| `libpython3.12.so.1.0: cannot open shared object file` | login ノードで `module load python/3.12/3.12.9` してから `uv run` |
| `assert use_deepspeed` で落ちる (mstts) | mstts/finetune.py が DeepSpeed 必須。1 GPU smoke でも `--use_deepspeed` + zero3 が要る |
| `AssertionError: NO_TORCH_COMPILE not set` | 上記環境変数を `mpirun -x` で渡す |

## ライセンス系譜 (2026-05-14 更新、要注意)

| 構成要素 | License | 商用 |
|---|---|---|
| Kyutai Moshika (base model) | **CC-BY-4.0** | ✅ |
| ReazonSpeech (Stage 1) | CC-BY-4.0 | ✅ |
| J-CHAT (Stage 1, 2) | 商用利用可 (2026-05-14 阿部確認) | ✅ |
| LaboroTVSpeech (Stage 1) | 非商用限定・申請制 | ❌ (**残る唯一の NC 継承主因**) |
| LLM-JP Zoom1 (Stage 3, 4) | LLM-JP 内利用 | ✅ (2026-05-14 阿部確認) |
| → **v0b/v0c 最終ライセンス** | **CC-BY-NC-4.0** | ❌ (LaboroTV 由来) |
| JMultiWOZ (新規) | CC-BY-SA-4.0 (学習モデルは SA 免除明記) | ✅ |
| RealPersonaChat (新規) | CC-BY-SA-4.0 | ✅ |

J-CHAT・Zoom1 は商用可と確認されたため、v1 系 (ReazonSpeech → J-CHAT → Zoom1) は商用クリーン。一方 **mstts v0b/v0c は系譜に LaboroTV (非商用) を含むため依然 NC 制約下**。MODEL_CARD_DRAFT / HF README / Space は v0c の NC 表記済 (理由は LaboroTV に集約された)。

**LaboroTV のライセンスはまだ完全には確認できておらず、Laboro.AI への問い合わせ必要**。合成対話音声を商用版 v1.x のデータに使うには、**LaboroTV を抜いた mstts を作り直す**必要がある可能性が高い。

## スピーカー / チャンネル規約 (重要)

- **入力 text_chat JSON**: `[["A", "..."], ["B", "..."], ...]` 形式。話者ラベルは "A" "B" 固定
- **出力 wav**: 24 kHz ステレオ、**channel 0 (LEFT) = A、channel 1 (RIGHT) = B**
- **mstts 内部 17ch token 表現** (`generated_tokens/*.npy` の形):
  - row 0: 統合 dialogue text 系列 (BOS=1 main, BOS=2 other)
  - rows 1-8: mstts "main" 位置の音声 (= 推論デフォルト `main_speaker_first=False` では **B**)
  - rows 9-16: mstts "other" 位置の音声 (= **A**)
  - `tools/decode_tokens.py:decode_audio` と `mstts/hf_inference/inference.py:decode_audio_with_mimi` が channel 順を swap して L=A R=B を出力 (2026-05-12 修正 commit c3455c1)

学習データ規約 (`README.md:47`): "left and right channel should contain speaker A's and B's audio, respectively" → 出力もこれに揃えてある。

## モデル系列

| ID | ベース | 追加学習 | Step | 状態 | Loss |
|---|---|---|---|---|---|
| (0178 mono) | moshika | J-CHAT-mono + ReazonSpeech + LaboroTV | 6,000 | 大橋さん作 | — |
| (0178 mstts) | 0178 mono | J-CHAT (multi-stream) | 17,892 | 大橋さん作、本プロジェクトの起点 | — |
| v0a | moshiko | J-CHAT 1shard | 500 | 試験のみ (ノイズ) | — |
| v0b | 0178 mstts | Zoom1 | +500 | HF private | 3.92 |
| **v0c** | v0b | Zoom1 継続 | +1500 | **HF private (推奨運用版)** | **3.04** (text 2.02, audio 1.02) |

### HF リポジトリ
- `abePclWaseda/llm-jp-moshi-mstts-v0b-zoom1` (private)
- `abePclWaseda/llm-jp-moshi-mstts-v0c-zoom1` (private、推論コード bundle 済)
- `abePclWaseda/llm-jp-moshi-mstts` (public Space、聴き比べデモ)

## v1 系 (LLM-jp-Moshi) 学習データフォーマット

`finetune.py` が読む parquet スキーマ:

```
{
  "dialogue_id": str,
  "A": [9, T_A],   # 1 text + 8 audio codebooks (per-speaker, frame aligned 12.5 Hz)
  "B": [9, T_B],
}
```

ビルド経路は 2 つ:
1. **実音声から**: wav (L=A, R=B) → `tokenize_audio_from_dir.py` + `tokenize_text_from_dir.py` (要 word-level transcript) → `prepare_dataset.py` でマージ
2. **mstts 合成から (新規)**: mstts `generated_tokens/*.npy` → `mstts/data_prep/mstts_tokens_to_v1_parquet.py` (text channel を BOS で demux、audio rows をそのまま分配)

## 主要コマンドリファレンス

### 学習 (mstts v0c の再現)

```bash
qsub pbs/run_mstts_v0c_zoom1.sh
# 8 GPU H100、effective batch 32、zero3-bf16-act_ckpt、tlr=1e-5 dlr=3e-5
```

### 推論 (HF v0c、ユーザ目線)

```bash
uvx --from huggingface_hub hf download abePclWaseda/llm-jp-moshi-mstts-v0c-zoom1 --local-dir mstts-v0c
cd mstts-v0c && uv sync
uv run python inference.py --text-chat sample_dialogue.json --output-wav out.wav
```

### バッチ合成 (テキストコーパス → wav)

```bash
qsub pbs/run_mstts_v0c_synth_text_corpora.sh
# デフォルト: 先頭 50 chunks (smoke 用)
# 本番: qsub -J 0-92 -v SLICE_SIZE=500 pbs/run_mstts_v0c_synth_text_corpora.sh
```

### mstts 合成 → v1 parquet

```bash
uv run --no-project --python 3.12 \
    --with pandas --with pyarrow --with tqdm python \
    mstts/data_prep/mstts_tokens_to_v1_parquet.py \
    --tokens-dir output/mstts_v0c_synth/<slice>/generated_tokens \
    --output-prefix output/mstts_v0c_synth/<slice>/parquet/synth
```

## 重要な既知バグ修正履歴

| Commit | 内容 |
|---|---|
| 6d9dd3a | mstts/finetune.py の `args.with_tracking` 属性未定義 (1行 fix) |
| 84c8617 | `run_ms_tts.py` ragged-last-batch crash (`prompt_tokens[:actual_bs]`) |
| c3455c1 | wav L/R 規約 (decode functions の channel swap、L=A R=B 化) |

## 関連プロジェクト

- `/groups/gcg51557/experiments/0178_dialogue_tts/moshi-finetuning` — 大橋さん (名大) のオリジナル mstts コード。本プロジェクトはここをベースに forking
- 連絡先: 阿部雄斗 <abe@pcl.cs.waseda.ac.jp> / 大橋厚元 (名大)

## 進捗 (2026-05-13 更新)

- [x] JMultiWOZ + RPC 全件本番合成 — Job 1758539[1-10]、**46,266 wavs / 約 530 時間** 合成完了。`output/mstts_v0c_synth/{slot}/` 配下
- [x] v1 parquet 化 — `output/mstts_v0c_synth_parquet/synth_<slot>-001-of-001.parquet` × 10、合計 357 MB

## 次のステップ

- [ ] **v1 追加学習 (本命)**: 上記 parquet を `finetune.py --train_data_files output/mstts_v0c_synth_parquet/synth_*.parquet` で投入。hyperparam・base model・mix ratio (実音声 vs 合成音声) 検討
- [ ] LaboroTV ライセンス確認 (Laboro.AI 問い合わせ)
- [ ] HF Space サンプル音声を L=A R=B 版に差し替え (任意、現状 legacy)
- [ ] RPC 雑談系の品質スポットチェック (production 後半 ~38K rpc wavs の random sample)
