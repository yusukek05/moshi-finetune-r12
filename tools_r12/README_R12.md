# R12 再現キット（別ベースモデル対応）

R12 = 「合成4000h(プロンプト付き) + zoom1冒頭(プロンプト付き) を最初から混合し2段学習」。
北村さんの lb512_epoch2_half_lr 設定に、テキストプロンプト条件づけ(PersonaPlex方式)を足したもの。
このフォルダだけで、**任意のベースモデル(fp32, 学習フォーマット)から**同条件の学習を再現できます。

## 中身

**このリポジトリ本体が、R12の学習で実際に走ったコードそのもの**です(プロンプト条件づけ実装
= `utils/data.py` の `preprocess_function_with_system_prompt_v2` 系、適用済み。パッチ作業は不要)。

- `tools_r12/run_r12_repro.sh` … PBSジョブ(rt_HF 1ノード8GPU)。BASE をパラメータ化済み
- `tools_r12/run_finetune_with_subgroup_timeout.py` … NCCLサブグループtimeout延長ラッパー
- prefix_tokens.npz … ABCI上: `/groups/gcg51557/experiments/0378_spoken-dialogue-model/share_r12_repro/prefix_tokens.npz`
  (ジョブスクリプトはこのパスを既定参照)

## データ(グループ読取可・コピー不要)

```
合成4000h: /groups/gcg51557/experiments/0378_spoken-dialogue-model/personaplex_4000h/data/train_fix/train-00{0..7}-of-008.parquet
zoom1冒頭: /groups/gcg51557/experiments/0378_spoken-dialogue-model/personaplex_4000h/data/mix/z1nostyle.parquet
```
必要列: dialogue_id, A[9,T], B[9,T], prompt_text_ids, prompt_text ほか。zoom1は無音trimなし・
max_length2048で冒頭163.84秒だけが学習に入る(分割しない。分割+プロンプトは矛盾で品質が落ちる
=R22で実証済み)。

## 実行

```bash
git clone https://github.com/yusukek05/moshi-finetune-r12 && cd moshi-finetune-r12
uv sync                       # venv 構築 (ログインノードで。計算ノードのuv runはvenvを壊すので不可)
WRAP=$PWD/tools_r12/run_finetune_with_subgroup_timeout.py
qsub -v STAGE=1,BASE=<起点モデルのfp32dir>,OUTDIR=<出力先>,VENV=$PWD/.venv,WRAP=$WRAP tools_r12/run_r12_repro.sh
# 1段目完了後、fp32へ統合:
#   .venv/bin/python -m tools.zero_to_fp32 <OUTDIR>/stage1/step_5963 <OUTDIR>/stage1/step_5963_fp32 #       --moshi_lm_kwargs_path <BASEのmoshi_lm_kwargs.json>
qsub -v STAGE=2,BASEDIR=<stage1のfp32>,OUTDIR=<同じ出力先>,VENV=$PWD/.venv,WRAP=$WRAP tools_r12/run_r12_repro.sh
```

注意: スクリプト内の REPO 変数(コード本体の場所)を自分の clone 先に変えること。

- STAGE=1: lr 2e-6/4e-6, MOSHI_DATALOADER_SEED=1 / STAGE=2: lr半減, SEED=2 (各1エポック)
- batch16(PD1×ACC2×8GPU)・warmup0・スケジューラ無し・fp16・lb512・max_length2048
- 損失重み: text_pad 0.5 / semantic 100 / acoustic 1 (この系で実証済みの値。論文値0.3/0.02は
  こちらの検証では悪化した)
- NCCL対策(スクリプトに設定済み): IB無効化 + subgroup timeout 3600s → 当方の全学習で死亡ゼロ
- 目安: 95,402サンプル → 5,963 step/段, H200 1ノードで1段 約5時間

## 推論時の注意

- プロンプトは学習と同じ書式で与える(例:「あなたは気さくに雑談をする人です。{話題}の話から
  始めて、相手に質問しながら雑談します。」)。大文字英字・全角数字は語彙外(unk)なので使わない
- prefill実装の参考: /groups/gcg51557/experiments/0378_spoken-dialogue-model/models/infer/serve_role2.py
  (delimiter_id=5, pause_frames=6, 自分側=無音・相手側=サイン音トークン)

## 検証用の当方の結果(同条件・J-CHAT step_8880起点)

judge継続対話(プロンプト無し) 5.04 / cer_a 0.470 / 1段目終盤のtrain loss ~1.5前後。
人間評価は crowd 09-25 の R12/R12P 参照。
