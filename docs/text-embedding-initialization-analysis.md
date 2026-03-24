# テキスト埋め込み初期化の分析と改善案

## 現状の問題

### 現行の初期化処理 (`tools/init_moshi_for_ft.py`)

`--init_text_embeddings` フラグにより、以下の処理が行われている：

1. 元の英語テキスト埋め込みの**平均と共分散**を計算
2. その分布からサンプリングして**全トークンを再初期化**（ガウシアン）
3. 特殊トークン（0, 3, 32000）だけ元の重みを保持

対象レイヤー：
- `text_emb`（tempformer 入力側）
- `depformer_text_emb`（depformer 入力側）

### なぜ問題か

- tempformer の Transformer 層は「英語の埋め込み空間」で事前学習されている
- 埋め込みを再初期化すると、Transformer 層が期待する入力と実際の入力が乖離する
- 出力層（`text_linear`）も英語トークナイザー前提の重みのまま
- **結果として、tempformer の LLM としての能力（言語理解・生成）が破壊されている**
- J-CHAT 等の学習でテキスト埋め込みをゼロから学習し直していることになる
- v1.2 の Text loss (0.971) が v1.1c (0.798) より高いのも、この影響が一因と考えられる

## 改善案

### 案1: 日本語 LLM の埋め込みで初期化（低コスト）

- llm-jp-3 等の日本語 LLM のテキスト埋め込み層の重みを、Moshi の `text_emb` / `depformer_text_emb` にマッピング
- トークナイザーが異なるため、vocabulary の対応付けが必要
  - 同一サブワードは LLM の埋め込みをそのまま使用
  - 対応がないトークンはガウシアン初期化（現行と同様）
- tempformer の Transformer 層は英語 Moshi のまま → 埋め込み空間のミスマッチは残るが、ランダム初期化よりは良い初期値
- **実装コスト: 低、期待効果: 中**

### 案2: tempformer 全体を日本語 LLM で初期化（MoshiLlama アプローチ）

- tempformer の Transformer 層を日本語 LLM（llm-jp-3 等）の重みで置換
- テキスト埋め込み・出力層も日本語 LLM のものを使用
- depformer は英語 Moshi のものを維持（音声⇔テキストのアラインメントはここで学習）
- `modeling_moshi_llama.py` / `finetune_llm-jp-3.py` で既に検討中
- **課題**: tempformer のアーキテクチャ（dim, n_heads, n_layers 等）が Moshi 独自のため、LLM とのアーキテクチャ差分の吸収が必要
- **実装コスト: 高、期待効果: 高**

### 案3: テキストデータ混合で埋め込みを鍛える（現行延長）

- 現行のランダム初期化のまま、日本語テキストデータを大量に混ぜて学習
- `audio_loss_weight_when_text_pad` を低く設定して音声 loss への悪影響を抑制
- 詳細は `docs/text-only-training-plan.md` を参照
- **実装コスト: 低、期待効果: 低〜中**

## 推奨順序

1. **案3**（テキスト混合）を試す — 最もすぐ試せる。現行パイプラインの延長
2. **案1**（日本語 LLM 埋め込み初期化）を試す — 案3 と組み合わせ可能
3. **案2**（MoshiLlama）を長期的に開発 — 最もポテンシャルが高いが工数も大きい

## 関連ファイル

- `tools/init_moshi_for_ft.py` — 現行の初期化スクリプト
- `tools/init_moshi_llmjp_ft.py` — MoshiLlama 初期化スクリプト（案2）
- `models/modeling_moshi_llama.py` — MoshiLlama モデル定義（案2）
- `finetune_llm-jp-3.py` — MoshiLlama 学習スクリプト（案2）
- `docs/text-only-training-plan.md` — テキスト混合学習計画（案3）
