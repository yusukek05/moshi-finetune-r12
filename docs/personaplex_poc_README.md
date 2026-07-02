# PersonaPlex-style prompt-control PoC — 共同研究者向けハンドオフ

LLM-jp-Moshi にテキスト system-prompt で「話し方（v0 は formality: 丁寧体/タメ口）」を
制御させる PoC の**再現手順・ファイル地図・結果・拡張方針**をまとめた入口です。
設計の詳細は [`personaplex_poc_design.md`](personaplex_poc_design.md)（特に §7）。

## TL;DR / 現況
- **機構は実装済み・学習も通る**（複数 checkpoint 生成済み）。
- **prompt が agent の文体予測に効くことは学習データ上で実証**（marker flip 82–96%）。
- **ただし held-out 汎化は 800 対話では未達**（marker flip ~50%）。主因＝データ量不足 + base の丁寧体 prior。
- → v0 は「機構実証済み・スケール要件を定量化」でクローズ。再開は (A2) データ桁増 or (C) 実データ混合。

## 機構（1 段落で）
system-prompt（voice/text prefix）を対話系列の**先頭に prepend** し、その prefix 区間の
**label を `moshi_lm.zero_token_id`(= loss の ignore_index) に設定して loss をマスク**する。
これで**学習ループ本体は無改修**のまま、prompt 条件付けが入る。
- 実装: `utils/data.py` の `build_system_prompt_prefix()` / `preprocess_function_with_system_prompt()`。
- 有効化: `finetune.py --system_prompt_conditioning --num_main_audio 8`
  （delimiter = `moshi_lm.end_of_text_padding_id`）。

## ファイル地図
| 役割 | パス |
|---|---|
| 機構(prefix+mask) | `utils/data.py`（`build_system_prompt_prefix`, `preprocess_function_with_system_prompt`） |
| 学習フラグ | `finetune.py`（`--system_prompt_conditioning`, `--num_main_audio`） |
| persona スキーマ/prompt | `mstts/data_prep/persona_schema.py`（`Persona`, `FORMALITY_PARAPHRASES`） |
| 台本生成(LLM-jp-4, style別few-shot) | `mstts/data_prep/gen_persona_dialogues.py` |
| persona 付与(parquet) | `mstts/data_prep/build_persona_parquet.py`（`--persona-map-dir`, `--paraphrase-holdout`） |
| flip 評価(条件付きNLL) | `mstts/data_prep/persona_flip_eval.py`（`--paraphrase-idx`） |
| PBS | `pbs/run_gen_persona_dialogues.sh` / `run_firered_synth_persona.sh` / `prep_synth_dialogue.pbs` / `run_train_personaplex_poc.sh` / `run_personaplex_flip_sweep.sh` |

## 再現パイプライン（ABCI）
前提: `module load cuda/12.6/12.6.1 python/3.12/3.12.9`、`uv sync`。学習は spot `qsub -q rt_HF -v RTYPE=rt_HF,...`
（予約 R9920261000 は preempt 多発時に spot 退避）。

```
# 1) 台本生成 (LLM-jp-4-8b, style別few-shot; polite=初対面/です・ます, casual=友達/タメ口)
qsub -q rt_HF -v RTYPE=rt_HF pbs/run_gen_persona_dialogues.sh          # 8GPU, 1000対話(500/500)

# 2) FireRedTTS-2 で合成 (Apache-2.0; L=A/R=B stereo)   ※0386 リポjの infer_batch を利用
qsub -q rt_HF -v RTYPE=rt_HF pbs/run_firered_synth_persona.sh          # 全1000 (LIMIT=25でsmoke)

# 3) wav -> v1 parquet (WhisperX整列→tokenize→merge)
qsub -q rt_HF -v RTYPE=rt_HF,STEREO=<synth_out>,SCRIPTS=<persona_gen>,DS=data/synth_dialogue_persona,\
OUTPREFIX=processed_data/persona_poc/persona_full_prep/synth pbs/prep_synth_dialogue.pbs

# 4) persona 付与 (prompt_text_ids + persona_json)。paraphrase 多様化は --paraphrase-holdout 5
uv run --no-project --with pyarrow --with numpy --with sentencepiece python \
  mstts/data_prep/build_persona_parquet.py --mode passthrough \
  --input <prep parquet> --tokenizer <rinna spiece.model> --persona-map-dir <persona_gen> \
  --output processed_data/persona_poc/persona_full-001-of-001.parquet
#   → 適宜 train/heldout に分割 (例: 各style 400 train / 100 heldout)

# 5) 学習 (base = v1.1 = output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32)
qsub -q rt_HF -v RTYPE=rt_HF,TRAIN_DATA=<persona_train parquet>,OUT=output/personaplex_poc_full,EPOCHS=5 \
  pbs/run_train_personaplex_poc.sh

# 6) flip 評価 (step を sweep, marker限定NLL, 未知言い回し=--paraphrase-idx 5)
qsub -q rt_HF -v RTYPE=rt_HF,OUT=output/personaplex_poc_full,\
HELDOUT=<persona_heldout parquet>,STEPS=step_100:step_150:step_200:step_250,PARAPHRASE_IDX=5 \
  pbs/run_personaplex_flip_sweep.sh
```

## 依存する外部リソース（ABCI 上）
- base v1.1: `output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32`（命名は歴史的、実体は v1.1）
- rinna tokenizer: `~/.cache/huggingface/hub/models--rinna--japanese-gpt2-medium/.../spiece.model`（text-track と同一 vocab）
- FireRedTTS-2: `/groups/gcg51557/experiments/0386_dialogue_model/`（`scripts/infer_batch.py`, weights, `data/voice_pool.json`）
- LLM-jp-4-8b-instruct（台本生成）: HF キャッシュ済

## 生成済み成果物（ABCI 上・git 管理外）
- 学習 ckpt: `output/personaplex_poc_{smoke,smoke_ep50,full,full_div}/step_*`（`*_fp32` は eval 用に consolidate 済）
- 学習データ: `processed_data/persona_poc/persona_*{train,heldout}*.parquet`
- 合成 wav: `0386/output/dialogue_arena/persona_drop_full_pool_stereo/`（1000本, 500/500）

## 結果（held-out, marker限定 flip_acc = 真の信号）
| 実験 | held-out | 判定 |
|---|---|---|
| smoke ep15 (train160) | 48% | chance |
| smoke ep50 (train160) | 42%（過学習） | chance |
| **同 ep50 を train データで** | **82–96%** | **機構◎** |
| full 800 (step best) | 54% | まだ弱い |
| full 800 + prompt多様化 (未知言い回し) | ~50% | まだ弱い |

## 再開時の拡張
- **(A2) データ桁増**: 5,000–10,000 対話（合成~1–2日）でデータ量仮説を本検証。
- **(C) prior を動かす**: 実 Zoom1（丁寧）+ casual 合成を混ぜ casual 方向を底上げ。
- 評価は `persona_flip_eval.py` をそのまま流用（marker flip / held-out paraphrase 対応）。
- 将来: 生成ベース flip（実生成→`classify_formality`）が gold standard。声(voice-prompt)軸・多軸 persona へ拡張。

連絡先: 阿部雄斗 <abe@pcl.cs.waseda.ac.jp>
