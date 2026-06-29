# PersonaPlex 型 prompt-control PoC 設計書

> 2026-06-29 / 阿部雄斗 <abe@pcl.cs.waseda.ac.jp>
> 目的: 「タスク毎 synth-finetune（トレッドミル）」から「**制御を一度学習→推論時に1モデルで多ペルソナ/多声を出し分け**」へ移行する最小実証。
> 戦略背景: `memory/project_personaplex_direction.md`、修論フレーム = 日本語 full-duplex ターンテイキングの評価・改善基盤（RQ3 = persona prompt で話し方/相槌/割り込み/関係性を制御）。

---

## 0. 一次情報（検証済み, 2026-06-29 web）

| 出典 | URL |
|---|---|
| PersonaPlex 論文 (ICASSP 2026) | https://arxiv.org/abs/2602.06053 / html: https://arxiv.org/html/2602.06053 |
| NVIDIA ADLR project page | https://research.nvidia.com/labs/adlr/personaplex/ |
| code (MIT) / weights (NVIDIA Open Model License) | https://github.com/NVIDIA/personaplex / https://huggingface.co/nvidia/personaplex-7b-v1 |
| F-Actor（兄弟研究, 行動制御） | https://arxiv.org/abs/2601.11329 |

**重要なライセンス注意**: PersonaPlex の **weights は NVIDIA Open Model License**（Apache ではない）。本 PoC は彼らの**手法（レシピ）のみ**を、我々の Moshi/moshiko（CC-BY-4.0）+ LLM-jp-Moshi-v1.1 に適用する。NVIDIA の checkpoint も彼らの合成データも**取り込まない**。データ源は FireRedTTS-2（Apache-2.0）+ LLM-jp-3（Apache-2.0）+ Zoom1 話者音声に限定 → **商用クリーンを維持**。

---

## 1. PersonaPlex レシピ要約（我々の用語にマップ）

### 1.1 ハイブリッド system prompt の構造（核心）

会話フレームの**前に** prompt 区間を prepend する。Moshi の各ストリーム（text / 自分=main audio / 相手=user audio）に次を置く:

```
[ 区間A: voice prompt ]  main_audio = 数秒の話者サンプル(audio tokens),  main_text = PAD
[ 区間B: text  prompt ]  main_text  = ペルソナの role テキスト tokens,   main_audio = silence
[ delimiter ]            text/audio の境界トークン
[ 会話本体 ]             既存の 2話者対話フレーム
```

- **順序**: voice(A) が text(B) に**先行**。理由 = 推論時に voice-clone 不要なら prefill でき latency 低減。
- **user/相手チャンネル**: prompt 区間中は **440 Hz サイン波**に置換（無音だと崩れるため "話していない明示信号"）。我々は silence/sine いずれでも可、要 ablation。
- **loss**: prompt 区間は**ロス backprop をマスク**（モデルに prompt を"生成"させない、条件付けにのみ使う）。
- **役割**: text prompt = "誰か・どう振る舞うか"、voice prompt = "どんな声か"。

### 1.2 学習データ構成

| 項目 | PersonaPlex 実績 | 含意 |
|---|---|---|
| 対話量 | service 1840h/105,410 dialog + QA 410h/39,322 dialog | **量より制御軸被覆**が効く（我々は桁違いに小規模で mechanism 実証） |
| transcript 生成 | Qwen-3-32B, GPT-OSS-120B | 我々 = **LLM-jp-3**（ICASSP Phase 0.3 で構築済） |
| TTS 合成 | Dia-TTS(service, 割り込み/timing 込み) + Chatterbox(QA, zero-shot clone) | 我々 = **FireRedTTS-2**(0386, Apache) |
| voice bank | 26,296 話者（VoxCeleb/Libriheavy/LibriTTS/CommonAccent/Fisher）, 2,630 held-out | 我々 = まず固定少数（Zoom1 話者 + FireRed clone 元の少数バンク） |
| 公開 ckpt 改善 | 実データ Fisher 1,217h + TortoiseTTS pitch/formant aug 追加 | 実データ併用が voice consistency を押し上げる（0.57→0.65） |

### 1.3 学習手順（我々の finetune.py とほぼ同一）

| ハイパラ | PersonaPlex | 我々の現状 | 差分 |
|---|---|---|---|
| base | Moshi weights → finetune | LLM-jp-Moshi-v1.1 (`step_9282_fp32`) | 同様 |
| tempformer lr | 2e-6 | 2e-6 | **一致** |
| depformer lr | 4e-6 | 4e-6 | **一致** |
| optimizer | Adam + cosine | (確認) | 概ね一致 |
| steps | 24,576 | PoC は小規模 (~1-3k) | スケールのみ |
| batch size | 32 | eff bs 16 (bs1×ga2×8) | 近い |
| max_len | 2048 (~163.84s) | 2048 | **一致** |
| loss downweight | 非意味音声 0.02 / pad text 0.3 | `--acoustic_loss_weight` / `--text_padding_loss_weight` で**設定可（引数既存）** | 値を合わせるだけ |
| 計算 | 6h / 8×A100 | 8×H100 で同等以下 | OK |

→ **学習ループは新規実装ほぼ不要**。新規は「prompt-prefix を作るデータ整形」と「prefix の loss マスク」だけ。

### 1.4 評価（PersonaPlex 公式）

- **persona 遵守**: GPT-4o 採点 on Service-Duplex-Bench（50 role × 7 Q; 固有名詞想起 / 文脈遵守 / 不可能要求 / 客の無礼対応）。PersonaPlex 4.48 vs Gemini 4.73。
- **声の一貫性**: WavLM-TDNN で voice-prompt と生成音声の埋め込み cosine 類似。Moshi 0.10 → PersonaPlex 0.57 → 公開 ckpt 0.65。
- **自然さ**: DMOS (MTurk, 1-5)。Full-Duplex-Bench 3.90、Service-Duplex-Bench 3.59。

### 1.5 F-Actor（行動制御, 兄弟手法）

- 制御対象: **backchannel / 割り込み確率 / turn-taking pace・overlap**。
- 制御信号: **離散 "F-Actor トークン"**（強度を段階値で prepend）。
- ラベル: 既存対話コーパスから **VAD + タイムスタンプで自動検出**（backchannel 語/overlap 区間/turn 境界）。
- 評価: backchannel rate / interruption rate / turn-taking latency + 人手。
- **我々への含意**: persona YAML の `backchannel_style` / `interruption_policy` は、テキスト指示だけでは弱い → **F-Actor 風の離散制御トークン**を併用すると強く効く（PoC v2 で追加）。

---

## 2. 制御軸スキーマ（persona YAML → 合成ラベル → prompt）

修論ログの YAML を、(a) 合成時の生成条件、(b) 学習時の prompt 表現、(c) 評価指標、にマップ:

```yaml
# persona.yaml （1 dialogue = 1 persona を causal に紐付け）
persona_id: jp_teacher_002
role: "高校の物理教師"                 # → text prompt 主文
relationship: "初対面の生徒"           # → text prompt 文脈
speaking_style:                        # → text prompt 形容
  formality: polite        # casual | polite | formal
  pace: medium             # slow | medium | fast
  warmth: high             # low | medium | high
backchannel_style: frequent            # none|sparse|frequent → text + F-Actor token(v2)
interruption_policy: avoid             # avoid|natural|assertive → text + F-Actor token(v2)
voice:
  voice_id: zoom1_spkA                 # 固定バンク → voice prompt(audio)
task_handoff: none                     # none|to_planner（arbitration v3）
```

### 2.1 制御軸の優先順位（PoC で扱う範囲）

| 軸 | 表現方法 | PoC | 根拠 |
|---|---|---|---|
| role / relationship / formality / warmth | **テキスト prompt（日本語自由文）** | **v0** | PersonaPlex の text-prompt 軸。最も実装容易・効果明瞭 |
| voice（話者性） | **voice prompt（audio tokens, 固定少数バンク）** | **v1** | Arena の声の多様性に直結。固定バンクなら zero-shot より遥かに容易 |
| backchannel / interruption / pace | テキスト + **F-Actor 離散トークン** | v2 | テキストだけでは弱い。VAD 自動ラベルが要る |
| task_handoff / arbitration | 別レイヤ（Arbitration Layer） | v3 | 修論 RQ4。PoC 範囲外 |

### 2.2 text prompt 直列化（日本語テンプレ案）

```
あなたは{role}です。{relationship}と話しています。
話し方: {formality}・{pace}なテンポ・親しみ{warmth}。あいづちは{backchannel_style}。
```
→ tokenize（`tokenizer_spm_32k_3.model`）して main_text チャンネルに forced tokens として配置、main_audio は silence。

---

## 3. 最小 PoC 計画（既存資産を最大流用）

### 3.1 パイプライン（新規は★のみ）

```
★persona.yaml サンプラ（8-16 role × 数 voice × style 組合せ）
   ↓ 条件付け
 LLM-jp-3 対話テキスト生成（ICASSP Phase 0.3 流用）        ← 既存
   ↓
 FireRedTTS-2 で2話者音声合成（0386 パイプライン流用）       ← 既存
   ↓ + persona ラベル添付
★data_prep: prompt-prefix 付き v1 parquet 構築（+ prompt_len フレーム数フィールド）
   ↓
★data_utils.py: preprocess_function / make_streams_labels に
   prefix 構築 & loss マスク（labels=-100 over prefix）を追加
   ↓
 finetune.py（既存レシピ: tlr2e-6/dlr4e-6/max_len2048、loss weight を PersonaPlex 値に）
   base = output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32 (= v1.1)
   ↓
 clean_moshi → dep_q=8 → moshi.server で「prompt を変えると振る舞いが変わる」を試聴
```

### 3.2 コード改修サーフェス（実名）

| ファイル | 改修 | 規模 |
|---|---|---|
| `mstts/data_prep/`（新規 `build_persona_corpus.py`） | persona サンプラ + LLM-jp-3 生成 + FireRed 合成の orchestrate | 中 |
| `mstts/data_prep/`（新規 `persona_streams_to_parquet.py`） | prefix(voice+text)+delimiter+dialogue を組み、`prompt_len` を付与 | 中 |
| `data_utils.py` | `preprocess_function`/`make_streams_labels` で prefix 区間の labels を `-100`（ignore）に。`delay_and_pad_streams` の initial-token 機構を流用 | **小** |
| `finetune.py` | loss が `batch.labels[:,0,1:]`/`[:,1:,1:]` を使う既存経路に ignore-index/mask を反映（`F.cross_entropy(ignore_index=-100)` か bool mask） | **小** |
| PBS | `pbs/run_train_personaplex_poc.sh`（既存 `run_train_v1.1_firered_poststage.sh` を雛形に） | 小 |

### 3.3 スケール（mechanism 実証なので小さく）

- 対話: **1,000–2,000**（PersonaPlex 144k に対し桁違いに小; 量でなく "prompt を変えると出力が変わる" の検証が目的）。
- ペルソナ: **8–16 role × 2–4 voice × 数 style** を held-out 込みで設計（**未学習 persona への汎化**を測れるよう train/test split）。
- 学習: 1 job、~1–3k step、数時間 / 8×H100。
- base: **v1.1 (step_9282_fp32)**。

### 3.4 成功基準（PoC ゲート）

1. **同一入力 × prompt 差し替え**で出力スタイル/声が**有意に変わる**（A/B 試聴 + 客観指標 §4）。
2. **held-out persona** でも遵守する（暗記でなく汎化）。
3. base（v1.1）対比で**素の対話品質を著しく壊さない**（自己整合 CER が崩壊しない: 参考 [[project_v1_lineage_cer_findings]]）。
→ 満たせば「制御を学習する」路線を本格投資。満たさなければ data 量/制御軸を絞って再試行。

### 3.5 期待値・正直な限界

- 我々の合成 1–2k 対話 + Zoom1 少数声 ≪ PersonaPlex（実 1,217h + 26k 声）→ **voice consistency は低め**想定。PoC は競争力でなく**機構実証**。
- **日本語知識の天井は不変**（英語 Helium バックボーン）。persona 制御は style/voice/behavior を動かすだけで知識は増えない → 真の上限突破は LLM-jp-3 バックボーン化（v2 級）と併走 [[project_v1_lineage_cer_findings]]。

---

## 4. Arena への persona-adherence 指標追加案

`abePclWaseda/speech-arena-next`（FD-DMOS Q1音質/Q2タイミング/Q3話し方/Q4内容 + pairwise → Bradley-Terry）に、PersonaPlex/F-Actor 由来の軸を足す:

### 4.1 客観指標（自動, 安い・先に回せる）

| 指標 | 方法 | 出典 |
|---|---|---|
| **persona 遵守** | JA LLM-judge（例: GPT-4o / LLM-jp）で「与えた persona text に応答が沿うか」採点（Service-Duplex-Bench 日本語版を簡略構築: 固有名詞想起/文脈遵守/役割外し検出） | PersonaPlex §eval |
| **声の一貫性** | 話者照合モデル（WavLM-TDNN 等, 日本語可なら ses model）で voice-prompt と生成 agent 音声の cosine | PersonaPlex |
| **行動制御の効き** | 生成ステレオを **VAD** で解析し backchannel rate / overlap(interruption) rate / turn-taking latency を測定 → **prompt を `frequent`↔`sparse` 等に振った時に指標が指示方向へ動くか** | F-Actor / 修論 RQ1 |

### 4.2 人手（Arena ライブ投票に追加質問）

- 既存 Q1–Q4 + pairwise はそのまま。
- 追加（任意）: **Q5 ペルソナ遵守**（与えた役割・話し方に沿っていたか）、**backchannel/割り込みの自然さ**、**handoff の自然さ**（v3 で）。修論ログの "persona adherence / backchannel style / handoff naturalness" に対応。

### 4.3 まず回す順
1. 客観 §4.1 を PoC モデルに適用（人手不要、即日）。
2. 効きが確認できたら Arena に Q5 + 行動軸を足して人手 pairwise。

---

## 5. ロードマップ

- **v0**（本 PoC 第一歩）: text-persona 軸のみ。data_utils マスク + persona サンプラ + 1k 対話 + 1 job。「prompt で話し方が変わる」を実証。
- **v1**: voice prompt（固定少数バンク）を追加。Arena の声多様性。
- **v2**: F-Actor 離散トークンで backchannel/interruption を明示制御（VAD 自動ラベル）。修論 RQ1+RQ3 本体。
- **v3**: Arbitration Layer / task_handoff（修論 RQ4）。
- 併走: LLM-jp-3 バックボーン化で知識天井を上げる（v2 級）。

## 実装状況 & 実測 (2026-06-29)

### 実装済み（検証済み, GPU 不要で確認）
- **コア機構**: `utils/data.py`（finetune.py が実際に使うのはこちら。トップレベル `data_utils.py` は別エントリ用の重複なので注意）に
  `build_system_prompt_prefix()` / `preprocess_function_with_system_prompt()`。prompt 区間の label を
  `zero_token_id`(= 既存 loss の `ignore_index`)にするだけでマスク成立 → **finetune.py の損失改修ゼロ**。
- **finetune.py 配線**: `--system_prompt_conditioning` / `--num_main_audio`(既定8)。ON で前処理を切替、
  delimiter = `moshi_lm.end_of_text_padding_id`。`prompt_len` 列は DataCollator が無視(collator 改修不要)。
- **データ整形**: `mstts/data_prep/persona_schema.py`(Persona schema / 日本語 prompt 直列化 / formality 計測)、
  `mstts/data_prep/build_persona_parquet.py`(bootstrap / passthrough モード, 任意 voice-prompt 抽出)。
- **PBS**: `pbs/run_train_personaplex_poc.sh`(v1.1 base, レシピ忠実ミラー + 新フラグ)。
- 検証: 単体4テスト + **end-to-end dry-run**(persona parquet → datasets.map → DataCollator → Batch、
  マスク保持を collation 後まで確認)。

### モデルの権威ある特殊トークン値 (lm.py / kwargs, 2026-06-29 確認)
v1.1 (`output/v1.2_reazonspeech_jchat_zoom1/step_9282_fp32`, text_card=32000, card=2048, n_q=dep_q=16):
`zero_token_id = -1`(ignore_index), `text_padding_token_id = 3`, `end_of_text_padding_id = 0`(delimiter に使用),
`initial_token_id = 2048`(audio), `text_initial_token_id = 32000`,
`delays = [0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1]`(text=0, semantic codebook=0, それ以外 audio=1)。
→ delimiter=0 は ignore(-1) ではなく実在の end_of_text_padding。対話中にも出る非ユニーク値なので、
強い境界が欲しければ将来 専用トークン化(embedding resize)を検討。v0 は 0 で許容。

### GPU smoke-train 結果 (Job 1962251, 2026-06-29) = plumbing 確定 ✅
bootstrap parquet(500, 丁寧偏り)で `pbs/run_train_personaplex_poc.sh` を実行:
- `system_prompt_conditioning ON … (num_main_audio=8, delimiter_id=0)` 発火。96 step/3ep/~4分、エラー無し。
- loss 5.65→~2.3 (text 3.57→1.0, audio 2.08→1.28)。prefix 追加で初期 loss 上昇→急降下=想定通り。
- ckpt `output/personaplex_poc/step_{50,96}` (各94GB, 使い捨て)。wandb run 1s7iy47w。
- → parquet→前処理(prefix+mask)→学習→保存→wandb が実機 GPU で一気通貫。**学習経路 production-ready**。
- 注: 制御の実証ではない(データが丁寧偏り)。本実証は causal コーパスが要る。

### 実測でのデータ戦略の決定 (重要)
agent(話者A)text track の formality 分布:
| コーパス | polite | casual | mixed |
|---|---|---|---|
| FireRed synth (500) | 469 | 13 | 18 |
| Zoom1 (400) | 400 | 0 | 0 |

→ **既存コーパスは丁寧体にほぼ偏り**。text-style(formality)制御を bootstrap で実証するのは casual 側
学習データ不足で**不可**。**正道は causal 生成**(persona をサンプリング→LLM-jp-3 で条件付き対話生成→
FireRedTTS-2 合成→`build_persona_parquet.py --mode passthrough`)。これは PersonaPlex が synth を使う理由と一致。

**次の分岐(user 判断)**:
- (A) **causal 生成**で style 制御 v0 を作る(LLM-jp-3=ICASSP Phase 0.3 / FireRed=0386 を流用)。styleの分散を persona が作る。
- (B) **voice-prompt 軸を先に**(既存音声から学習可。ただし同一対話 voice-prompt は trivial copy のリスク→
  cross-utterance / held-out 話者設計が要る)。Arena の声多様性に直結。

## 6. データ管理（合成）

HF Datasets(private, 正本) + provenance card（生成元: LLM-jp-3 版 + FireRedTTS-2 版 + persona サンプラ seed + フィルタ）+ W&B Artifact（run↔dataset 版 lineage）。再生成スクリプトを git に残す。詳細 [[project_personaplex_direction]]。
