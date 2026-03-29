# 研究室サーバ (g15) へのバックアップ内容

バックアップ先: `yuabe@g15:/mnt/work-qnap/yuabe/0162_dialogue_model/`

バックアップ日: 2026-03-29（ABCI 停止前日）

## ディレクトリ構成

```
/mnt/work-qnap/yuabe/0162_dialogue_model/
├── moshi-checkpoints/          # 学習済みチェックポイント（再学習コスト大）
│   ├── reazonspeech_step_50000_fp32/   # ReazonSpeech 中間 (32G)
│   ├── v1.2_jchat_step_8880_fp32/      # v1.2 J-CHAT 完了 (32G)
│   └── v1.1c_stage3_step_942_fp32/     # v1.1c 全部盛り完了 (32G)
├── processed_data/             # トークナイズ済み学習データ
│   ├── data_stage_3/                   # 全部盛りデータ (206M)
│   ├── llmjp-zoom1/                    # Zoom1 対話データ (994M)
│   └── VisualBank/                     # VisualBank データ (323M)
├── generated_wavs/             # 生成済み音声（評価用）
│   ├── v1/                             # v1 の continuation 結果
│   ├── v1.1c/                          # v1.1c の continuation 結果
│   └── v1.1d/                          # v1.1d の continuation 結果
├── Full-Duplex-Bench/          # FD-Bench-JA 評価コード・結果 (7.3G)
├── LLM-as-a-Judge/             # LLM-as-a-Judge 評価結果 (34M)
└── NISQA_data_sample_audio/    # NISQA 評価用音声サンプル (872M)
```

## チェックポイント詳細

| ファイル | 元パス (ABCI) | サイズ | 用途 |
|---|---|---|---|
| reazonspeech_step_50000_fp32 | `/groups/gcg51557/experiments/0215_audio_llm/.../20260107-1300+reazonspeech/step_50000_fp32` | 32G | v1.2 系全ての起点。ReazonSpeech で事前学習済み |
| v1.2_jchat_step_8880_fp32 | `.../output/v1.2_reazonspeech_jchat/step_8880_fp32` | 32G | v1.2 の J-CHAT 完了。Zoom1 学習や他の追加学習の起点 |
| v1.1c_stage3_step_942_fp32 | `.../output/moshi-finetuned_init_text_emb_train_ohashi_data_stage_3_3epochs/step_942_fp32` | 32G | 全部盛りステージ完了。v1.3 等のステージ学習の起点になり得る |

## HuggingFace にアップ済み（ダウンロード可能）

| モデル | HF リポジトリ | 備考 |
|---|---|---|
| v1 | abePclWaseda/llm-jp-moshi-v1 | private |
| v1.1b | abePclWaseda/llm-jp-moshi-v1.1-vb-pseudo | private |
| v1.1c | abePclWaseda/llm-jp-moshi-v1.1-all-staged | private |
| v1.1d | abePclWaseda/llm-jp-moshi-v1.1-all-mixed | private |
| v1.1e | abePclWaseda/llm-jp-moshi-v1.1-vb-pseudo-staged | private |
| v1.2 | abePclWaseda/llm-jp-moshi-v1.2 | private |

## バックアップしていないもの

| データ | 理由 |
|---|---|
| J-CHAT 生データ (58G) | sarulab-speech/J-CHAT から再取得可能 |
| J-CHAT トークナイズ済み (大部分) | 巨大。再トークナイズ可能 |
| init_models (108G) | HF/公開元から再ダウンロード可能 |
| DeepSpeed チェックポイント (step_XXXX/) | fp32 変換済みがあれば不要 |
| J-CHAT(大橋)モデル (32G) | v1系の追加学習にしか使わない。要判断 |
