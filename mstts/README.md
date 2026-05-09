# mstts — Multi-Stream Dialogue TTS

Multi-stream dialogue TTS の学習・推論用サブプロジェクト。Moshi/Moshika アーキテクチャをベースに、対話音声合成器を学習する。

## 出典

このディレクトリ配下のコードは、大橋厚元さん（名古屋大学 / `atsumoto`）の `0178_dialogue_tts/moshi-finetuning` から、本人の許諾を得て 2026-05-09 に取り込んだもの。元プロジェクトを破壊しないよう、ここでは独立コピーとして管理する。

- Original: `git@github.com:atsumoto/moshi-finetuning.git`
- Imported from: `/home/ach17826ug/0178_dialogue_tts/moshi-finetuning/`
- Imported on: 2026-05-09

## 構成

```
mstts/
├── finetune_ms_tts.py        # multi-stream TTS の学習スクリプト本体
├── finetune.py               # 共通 finetune ユーティリティ（mstts専用、ルートの finetune.py とは別物）
├── data_utils.py             # 前処理・collator（mstts専用）
├── process_ms_tts.py         # データセット前処理エントリポイント
├── run_ms_tts.py             # 推論エントリポイント
├── cond_gen.py               # 条件付き生成
├── models/
│   ├── moshi_for_finetuning.py
│   ├── moshi_for_generation.py
│   ├── modeling_moshi_llama.py
│   └── utils.py
└── ds_configs/               # mstts用 DeepSpeed 設定（ルート ds_configs/ とは別）
```

## ルートの J-Moshi 学習との関係

ルートの `finetune.py` / `data_utils.py` / `ds_configs/` は J-Moshi（対話モデル）の学習用で、本サブプロジェクトとは独立。共通化したい部分が出てきたら、別途リファクタする。

## moshi バージョンに関する注意

元プロジェクト `0178_dialogue_tts/moshi-finetuning` の `pyproject.toml` は `moshi==0.1.1a1` に依存していた。ルートの 0162 環境（`kyutai/moshiko-pytorch-bf16` ベース）と差分があり、`models/moshi_for_finetuning.py` 等がそのままでは動かない可能性がある。動作確認時にバージョン整合を取ること。
