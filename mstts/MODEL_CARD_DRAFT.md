<!--
公開時の HuggingFace Model Card 用ドラフト。
本ファイルはレビュー用。一般公開時には次のいずれかを行う:
  (a) このファイルの内容で HF レポジトリの README.md を上書きする
  (b) このまま HF にも置く（README.md として）
公開前の最終チェック項目はファイル末尾の「Public release checklist」を参照。
-->

---
license: cc-by-nc-4.0
datasets:
- sarulab-speech/J-CHAT
language:
- ja
base_model:
- kyutai/moshika-pytorch-bf16
library_name: moshi
tags:
- multi-stream-tts
- dialogue-tts
- japanese
- spoken-dialogue
- tts
pipeline_tag: text-to-speech
---

# llm-jp-moshi-mstts-v0c-zoom1

**マルチストリーム日本語対話 TTS モデル**。テキストの2話者対話 (`[[A, "..."], [B, "..."], ...]`) を入力に、24 kHz ステレオ音声（左 ch = 話者 A、右 ch = 話者 B）を生成する。Moshi/Moshika アーキテクチャをベースに、4段階のカリキュラム学習で構築。

> **English summary:** A Japanese multi-stream text-to-speech model that generates 24 kHz stereo dialogue audio (left = speaker A, right = speaker B) from a turn-tagged text dialogue. Built on Kyutai's Moshika via 4-stage curriculum training. **License: CC-BY-NC 4.0 (non-commercial use only)**, inherited from the Moshika base model.

---

## Model Details

| 項目 | 値 |
|---|---|
| Architecture | Moshi LM (17 channels: 1 text + 8×2 audio codebooks) |
| Base model | [`kyutai/moshika-pytorch-bf16`](https://huggingface.co/kyutai/moshika-pytorch-bf16) |
| Audio codec | [Kyutai Mimi](https://huggingface.co/kyutai/moshika-pytorch-bf16) (24 kHz, 12.5 frame/s) |
| Text tokenizer | [`rinna/japanese-gpt2-medium`](https://huggingface.co/rinna/japanese-gpt2-medium) spiece.model (32 k vocab) |
| Parameters | ~7 B |
| Precision | bfloat16 (推奨), fp32 weights available |
| Input | JSON list of `[speaker_label, utterance]` |
| Output | 24 kHz stereo WAV (≈最大 60 s/segment) |
| Language | 日本語 (Japanese) only |
| Speakers | 2 名のみ（A / B、固定話者性） |

---

## Intended Use / 想定用途

### 想定する用途

- **学術研究**：日本語対話音声合成の研究、対話システム評価データの生成
- **合成データ作成**：dialogue model の学習用に**話者の同意済み**シナリオから対話音声を量産
- **デモ・プロトタイピング**：日本語対話 UI の音声プロトタイプ
- **教育**：full-duplex 音声対話モデルの教材

### 不適切な用途（Out-of-scope use）

以下は **明示的に禁止** します:

- ❌ **特定実在人物の声を再現する目的での使用**（voice cloning, impersonation）
- ❌ **本人同意のないなりすまし音声の生成**（詐欺、フェイクニュース、嫌がらせ）
- ❌ **商用利用全般**（CC-BY-NC 4.0 に準拠）
- ❌ **政治的なミスインフォメーション、選挙妨害**
- ❌ **未成年者を対象とした不適切な音声生成**
- ❌ **法令・公序良俗に反する音声生成**

> **English:** This model **must not** be used for impersonating real individuals, generating non-consensual deepfake audio, commercial purposes (forbidden by CC-BY-NC 4.0), political misinformation, or any illegal activity.

---

## Training Curriculum

4段階のカリキュラム。Stage 1, 2 は **大橋厚元（名古屋大学）** が学習した checkpoint をベースとして再利用しています（許諾済）。

### Stage 1 — mono pretraining (by 大橋厚元)
- **ベース**: `kyutai/moshika-pytorch-bf16`
- **データ**: J-CHAT-mono + ReazonSpeech + LaboroTVSpeech
- **規模**: 6,000 step、effective batch 512、tlr=3e-4 / dlr=3e-4
- **目的**: Moshika に日本語音声・テキストの分布を学習させる

### Stage 2 — multi-stream TTS pretraining (by 大橋厚元)
- **ベース**: Stage 1 産物
- **データ**: J-CHAT (multi-stream parquet)
- **規模**: 17,892 step、effective batch 512、tlr=3e-5 / dlr=1e-4

### Stage 3 — Zoom1 fine-tune (v0b, 阿部雄斗)
- **ベース**: Stage 2
- **データ**: LLM-JP Zoom1 対話データ（1,723 dialogues / 42,396 examples after split）
- **規模**: 500 step、effective batch 32、tlr=1e-5 / dlr=3e-5、warmup 50
- **最終 Loss**: 3.92（text 2.40、audio 1.52）

### Stage 4 — extended Zoom1 fine-tune (this model, v0c, 阿部雄斗)
- **ベース**: Stage 3
- **データ**: 同 Zoom1
- **規模**: 1,500 step、effective batch 32、tlr=1e-5 / dlr=3e-5、warmup 50
- **最終 Loss**: **3.04**（text 2.02、audio 1.02）

### Common Settings
- **Optimizer**: AdamW (β=[0.9, 0.95], eps=1e-5, weight_decay=0.1)
- **DeepSpeed**: ZeRO Stage 3, activation checkpointing
- **Loss weights**: semantic=100.0, acoustic=1.0, text_padding=0.5
- **Hardware**: NVIDIA H100 80GB × 8（GENIAC ABCI 環境、`gcg51557` プロジェクト）

---

## Training Data

| データセット | 用途 (stage) | 出典・ライセンス |
|---|---|---|
| J-CHAT-mono | Stage 1 | [sarulab-speech/J-CHAT](https://huggingface.co/datasets/sarulab-speech/J-CHAT)（独自ライセンス、要確認） |
| ReazonSpeech | Stage 1 | [reazon-research/reazonspeech](https://research.reazon.jp/projects/ReazonSpeech/)（CC-BY 4.0） |
| LaboroTVSpeech | Stage 1 | [Laboro.AI Inc.](https://laboro.ai/activity/column/engineer/laborotvspeech/)（**非商用限定**） |
| J-CHAT (multi-stream) | Stage 2 | 上記 J-CHAT を multi-stream 化 |
| LLM-JP Zoom1 | Stage 3, 4 | LLM-JP プロジェクト共同利用データ（要 LLM-JP との合意確認） |

> **公開前確認事項**:
> - LaboroTVSpeech・LLM-JP Zoom1 については **派生 TTS モデルの再配布**が許可されているか各データ提供元に明示的な確認が必要
> - 学習データに含まれる話者から「自分の声で TTS を作って公開する」ことへの同意があるか不明 → **特定話者の声を意図的に再現する用途を禁止する**ことで部分的に対応

---

## Evaluation

### Loss curves

| 段階 | Step | Total Loss | text | audio |
|---|---|---|---|---|
| v0a (moshiko 直、参考) | 500 | 6.55 | 3.29 | 3.25 |
| 0178 base (Stage 2 終了時) | — | — | — | — |
| v0b (Stage 3 終了) | 500 | 3.92 | 2.40 | 1.52 |
| **v0c (Stage 4 終了, this)** | 1,500 | **3.04** | 2.02 | 1.02 |

### 主観評価

> **TODO**: 公開前に MOS（Mean Opinion Score）を 5段階 × 50 サンプル × 10 評価者以上で実施し、結果を本セクションに記載する。

### 客観評価

> **TODO**: 公開前に以下を実施・記載:
> - **WER / CER**: 生成 wav を ASR (whisper-large-v3-ja 等) にかけて入力テキストと比較
> - **Speaker similarity**: 学習話者と意図せず似てしまっていないか（ECAPA-TDNN 等で計測）
> - **比較対象**: VOICEVOX、Style-Bert-VITS2、CosyVoice 等の OSS TTS との並列比較

---

## Limitations / 既知の制約

1. **言語**: 日本語のみ。英語等を入力すると無意味な音声が生成される。
2. **話者**: 2話者（A / B）固定。話者の声色は学習データ分布から自動決定され、明示的に制御不可。
3. **対話分布の偏り**: Stage 4 で Zoom1（オンライン会議形式）に強く適応しているため、独白・物語・ナレーションには弱い可能性。
4. **長尺生成**: 推論時の `max_generation_length` 上限内（≈1分）でのみ安定動作確認済。それ以上の長文では崩れる可能性。
5. **感情コントロール不可**: テキストに「（怒り）」「（笑い）」等を書いてもパラメトリックに反映されない。
6. **韻律・アクセント**: 学習データの分布に依存。固有名詞・専門用語のアクセントが不自然になることがある。
7. **prompt streams の依存**: 推論には prompt audio (.npy) が必須で、その音声特性が出力声質に影響する。
8. **計算コスト**: 推論に H100 1枚レベル＋ DeepSpeed なし環境でも 33 GB の重み + Mimi コーデック処理が必要。

---

## Risks and Ethical Considerations

### 音声 deepfake / 詐称リスク

本モデルは日本語の自然な対話音声を生成可能であり、悪意ある利用者が以下の用途に転用するリスクがあります:

- 振り込め詐欺等での声真似
- SNS でのなりすまし投稿
- 本人の同意のない再現音声の拡散

**対策（利用者側に要請）:**
- 生成音声には**生成物であることを明示**する（メタデータ、ラベル、可聴な告知）
- 公開前に**音声透かし**（[AudioSeal](https://github.com/facebookresearch/audioseal) 等）を埋め込むことを推奨
- 商用・準商用利用は禁止（ライセンス制約）

### バイアス

- **話者性**: 学習データの話者分布（性別・年代・方言）に偏る。具体的な分布の定量分析は未実施。
- **発話内容**: J-CHAT・Zoom1 由来のため、podcast / 会議調が支配的。日常会話・関西弁・敬語の使い分け等に弱い可能性。
- **誤発話**: 学習データに含まれるフィラー・言い間違い・差別的表現等を統計的に再現する可能性あり。

### プライバシー

- 学習データに含まれる話者の声特徴が、出力音声に部分的に再現される可能性がある（intentional でなくとも）。
- 学習データ作成時の話者同意の範囲が「TTS 派生モデルの公開」を含むか、各データセット提供元と確認すること。

---

## Environmental Impact

| 項目 | 値 |
|---|---|
| Stage 3 学習時間 | ≈30 分（H100 × 8） |
| Stage 4 学習時間 | ≈2 時間（H100 × 8） |
| Stage 1, 2 学習時間 (大橋さん側) | 推定 数日（H100 × 32） |
| 合計 GPU-hours (Stage 3-4 のみ) | ≈20 GPU-hours |
| 電力換算 (H100 ≈700W × 20h) | ≈14 kWh |
| CO2 換算 (日本平均 0.45 kg/kWh) | ≈6 kg CO2eq |

> Stage 1, 2 を含む total carbon は、大橋さん側の学習ログから別途算出が必要。

---

## How to Use / 推論クイックスタート

### Requirements

```bash
pip install moshi==0.1.0 torch torchaudio sphn safetensors sentencepiece soundfile
# + tools/decode_tokens.py のためにこちらも：
pip install huggingface_hub deepspeed accelerate
```

### Step 1: テキスト対話 → トークン

```bash
python run_ms_tts.py \
  --launcher accelerate \
  --model_dir <path-to-this-repo> \
  --model_dtype bfloat16 \
  --text_chat_data_dir <input-dir> \
  --text_tokenizer_repo rinna/japanese-gpt2-medium \
  --text_tokenizer_name spiece.model \
  --prompt_streams_path <prompt.npy> \
  --max_generation_length 750 \
  --use_sampling --text_temperature 0.55 --audio_temperature 0.6 \
  --output_dir <out-dir>
```

### Step 2: トークン → wav

```bash
python -m tools.decode_tokens \
  --tokens_dir <out-dir>/generated_tokens \
  --output_dir <out-dir>/decoded_audio
```

### Input Format

```json
[
  ["A", "おはようございます。今日は天気がいいですね。"],
  ["B", "本当にいい天気ですね。散歩でもしましょうか。"],
  ["A", "いいですね、行きましょう。"]
]
```

### Output

24 kHz 2-channel WAV：
- 左チャンネル = 話者 A
- 右チャンネル = 話者 B

完全な動作するスクリプト群は [abePclWaseda/moshi-finetune](https://github.com/abePclWaseda/moshi-finetune) の `pbs/run_mstts_v0c_*.sh` を参照。

---

## License / ライセンス

**CC-BY-NC 4.0** ([Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/))

このライセンスは、ベースモデル `kyutai/moshika-pytorch-bf16` から **継承** されています。

- ✅ 学術研究、教育、個人的非商用利用 OK
- ✅ 改変・再配布 OK（同一ライセンスでの公開）
- ✅ 帰属表示必須（**Citation** セクション参照）
- ❌ **商用利用 不可**
- ❌ 上記「Out-of-scope use」記載の用途 不可

---

## Attribution / 帰属

| 段階 | 担当 | 所属 |
|---|---|---|
| Base model | Kyutai team | Kyutai |
| Audio codec (Mimi) | Kyutai team | Kyutai |
| Text tokenizer | rinna | rinna |
| Stage 1, 2（学習コード＋ checkpoint） | 大橋 厚元 (Atsumoto Ohashi) | 名古屋大学 対話研究グループ |
| Stage 3, 4（追加学習・統合・公開） | 阿部 祐也 (Yuto Abe) | 早稲田大学 |
| Stage 1 学習データ作成 (J-CHAT-mono) | sarulab-speech | 東京大学 |
| Stage 1 学習データ作成 (ReazonSpeech) | Reazon Holdings | Reazon Holdings |
| Stage 1 学習データ作成 (LaboroTV) | Laboro.AI Inc. | Laboro.AI Inc. |
| Stage 3, 4 学習データ作成 (Zoom1) | LLM-JP プロジェクト | NII / 共同研究機関 |

---

## Citation

このモデルを利用する場合、以下を引用してください:

```bibtex
@misc{abe2026llm-jp-moshi-mstts-v0c-zoom1,
  author       = {Abe, Yuto and Ohashi, Atsumoto},
  title        = {llm-jp-moshi-mstts-v0c-zoom1: A Japanese multi-stream dialogue TTS model},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/abePclWaseda/llm-jp-moshi-mstts-v0c-zoom1}},
}

@article{ohashi2025towards,
    title   = {Towards a Japanese Full-duplex Spoken Dialogue System},
    author  = {Ohashi, Atsumoto and Iizuka, Shinya and Jiang, Jingjing and Higashinaka, Ryuichiro},
    journal = {arXiv preprint arXiv:2506.02979},
    year    = {2025}
}

@misc{defossez2024moshi,
    title   = {Moshi: a speech-text foundation model for real-time dialogue},
    author  = {Alexandre Défossez and Laurent Mazaré and Manu Orsini and Amélie Royer and Patrick Pérez and Hervé Jégou and Edouard Grave and Neil Zeghidour},
    year    = {2024},
    eprint  = {2410.00037},
    archivePrefix = {arXiv},
}
```

---

## Contact

- **Issues / questions**: [github.com/abePclWaseda/moshi-finetune/issues](https://github.com/abePclWaseda/moshi-finetune/issues)
- **Email**: abe@pcl.cs.waseda.ac.jp
- **Misuse / abuse の通報**: 同上

---

## Changelog

- **v0c-zoom1** (2026-05-10): Stage 4、Zoom1 で +1500 step 追加学習。final loss 3.04
- **v0b-zoom1** (2026-05-10): Stage 3、Zoom1 で 500 step fine-tune。final loss 3.92

---

<!--
==============================================================================
Public release checklist (公開前にこのセクションを削除すること)
==============================================================================

法務・倫理:
  [ ] Kyutai Moshika の CC-BY-NC 4.0 派生物条件を再確認、上流 attribution 完備
  [ ] LaboroTVSpeech 提供元に派生 TTS 公開可否を文書で確認
  [ ] LLM-JP に Zoom1 ベースの派生モデル公開可否を確認
  [ ] J-CHAT (sarulab) に派生 TTS 公開可否を確認
  [ ] 大橋さんから「checkpoint を public に再配布する」明示的同意を文書で取得
  [ ] J-Moshi 共同研究者（飯塚, Jiang, 東中先生）への通知・同意確認
  [ ] 学習データ話者プライバシー観点でのレビュー
  [ ] 「Out-of-scope use」節を社内法務 / 倫理委員会レビュー

評価:
  [ ] MOS 評価実施（最低 50 サンプル × 10 評価者）
  [ ] WER / CER 評価実施（whisper-large-v3-ja 等）
  [ ] Speaker similarity 評価（学習話者との意図せざる類似度）
  [ ] 既存 OSS TTS との比較表
  [ ] 失敗例（崩れる入力パターン）のサンプル

ドキュメント:
  [ ] サンプル wav を HF Space または gh-pages でホスト
  [ ] 推論 Quickstart の動作確認（CPU / 単一 GPU）
  [ ] Colab notebook 作成
  [ ] HuggingFace Space (Gradio デモ) 作成

リポジトリ:
  [ ] abePclWaseda/moshi-finetune を public 化（または公開用リポを別途作成）
  [ ] secrets / .env / 個人情報の確認・除去
  [ ] PBS 固有スクリプトの汎化版（汎用 GPU 用）
  [ ] LICENSE ファイル設置
  [ ] CONTRIBUTING.md / CODE_OF_CONDUCT.md
  [ ] CI 設定

その他:
  [ ] arXiv tech report 起草
  [ ] プロジェクトページ作成
  [ ] 音声透かしの埋め込み実装の検討
  [ ] semantic versioning 開始（v0c → v1.0.0）
==============================================================================
-->
