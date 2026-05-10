---
license: cc-by-nc-4.0
datasets:
- sarulab-speech/J-CHAT
language:
- ja
base_model:
- abePclWaseda/llm-jp-moshi-mstts-v0b-zoom1
library_name: moshi
tags:
- multi-stream-tts
- dialogue-tts
- japanese
pipeline_tag: text-to-speech
---

# llm-jp-moshi-mstts-v0c-zoom1

日本語対話のマルチストリーム TTS モデル。`v0b` を Zoom1 で **追加 1,500 step** 学習させたもの。Final loss 3.04（text 2.02, audio 1.02）。

入力: `[[speaker, "..."], [speaker, "..."], ...]` 形式の対話 JSON
出力: 24 kHz ステレオ wav（左 ch = 話者 A、右 ch = 話者 B）

プロジェクトページ（聴き比べ可）: <https://huggingface.co/spaces/abePclWaseda/llm-jp-moshi-mstts>

## Quick start

```bash
# 1. Download this repository (model weights + inference scripts)
huggingface-cli download abePclWaseda/llm-jp-moshi-mstts-v0c-zoom1 \
    --local-dir mstts-v0c
cd mstts-v0c

# 2. Install runtime dependencies
pip install moshi==0.1.0 sentencepiece soundfile sphn huggingface_hub torch numpy

# 3. Run inference on the bundled sample dialogue
python inference.py --text-chat sample_dialogue.json --output-wav out.wav
# → Writes a ~30 s stereo wav (left = speaker A, right = speaker B).
```

CLI options:

```
--text-chat PATH               Input dialogue JSON (required)
--output-wav PATH              Output 24 kHz stereo wav (required)
--prompt-npy PATH              Audio prompt streams .npy (default: prompt_streams_default.npy)
--max-generation-length INT    Max frames to generate (default 750 ≈ 60 s)
--text-temperature FLOAT       Text-side sampling temperature (default 0.55)
--audio-temperature FLOAT      Audio-side sampling temperature (default 0.6)
--device cuda|cpu              Computation device (default: cuda if available)
--seed INT                     Sampling seed (default 1)
```

Custom dialogue example:

```json
[
  ["A", "最近、何か新しい趣味始めた？"],
  ["B", "あ、うん、双眼鏡を持って公園で鳥を見るやつ。何ていうんだっけ？"],
  ["A", "バードウォッチング？"],
  ["B", "そう、それ！"]
]
```

## Repository contents

| File | Description |
|---|---|
| `model.safetensors` (33 GB) | fp32 weights of the 7 B Moshi LM |
| `moshi_lm_kwargs.json` | architecture configuration |
| `inference.py` | self-contained CLI inference script |
| `prompt_streams_default.npy` | default audio prompt (17 ch × 60 frames) |
| `sample_dialogue.json` | sample input (greeting + lunch chat) |
| `sample_dialogue_hobby.json` | sample input (hobby recall scenario) |
| `models/` | model wrapper classes (loaded by `inference.py`) |
| `data_utils.py` | preprocessing utilities (loaded by `inference.py`) |

## Training Curriculum

4-stage curriculum. Stages 1–2 reuse Atsumoto Ohashi's checkpoint (with permission); stages 3–4 are this project's Zoom1 adaptation.

| Stage | Author | Base | Data | Steps | Eff. batch | Final Loss |
|---|---|---|---|---|---|---|
| 1 (mono) | A. Ohashi | `kyutai/moshika` | J-CHAT-mono + ReazonSpeech + LaboroTV | 6,000 | 512 | — |
| 2 (mstts) | A. Ohashi | Stage 1 | J-CHAT (multi-stream) | 17,892 | 512 | — |
| 3 (v0b) | Y. Abe | Stage 2 | LLM-JP Zoom1 | 500 | 32 | 3.92 |
| 4 (v0c, this) | Y. Abe | Stage 3 (v0b) | LLM-JP Zoom1 | 1,500 | 32 | **3.04** |

### Common Settings
- Optimizer: AdamW (β=[0.9, 0.95], eps=1e-5, weight_decay=0.1)
- DeepSpeed ZeRO Stage 3 with activation checkpointing
- Loss weights: semantic=100.0, acoustic=1.0, text_padding=0.5
- Audio codec: [Kyutai Mimi](https://huggingface.co/kyutai/moshika-pytorch-bf16) (1 text + 8×2 audio = 17 channels)
- Text tokenizer: [`rinna/japanese-gpt2-medium`](https://huggingface.co/rinna/japanese-gpt2-medium) spiece (32 k vocab)

## Limitations

- 日本語のみ。英語等は無意味な音声に。
- 2 話者（A / B）固定、声色は学習データから自動決定で明示的制御不可。
- Zoom 形式の対話に強く適応、独白・ナレーションには弱い。
- 安定生成は ≈ 1 分以内、それ以上の長文は崩れる可能性。
- 感情・スタイル制御不可。
- 推論には prompt audio (.npy) が必須で、その音声特性が出力声質に影響する。

## Out-of-scope use

- ❌ 特定実在人物の声を再現する目的（voice cloning, impersonation）
- ❌ 本人同意のないなりすまし音声
- ❌ 商用利用（CC-BY-NC 4.0 で禁止）
- ❌ 政治的ミスインフォメーション

## License

**CC-BY-NC 4.0**（[Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/)）

ベースモデル `kyutai/moshika-pytorch-bf16` から継承。**商用利用不可**。

## Attribution

- **Stage 1, 2**: 大橋厚元（名古屋大学 / `atsumoto`）
- **Stage 3, 4, integration, this checkpoint**: 阿部雄斗（早稲田大学）

Stage 1, 2 のコードと checkpoint は大橋さんから利用許諾を得て使用しています。

## Citation

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
```

## Contact

- **Issues / questions**: <https://github.com/abePclWaseda/moshi-finetune/issues>
- **Email**: abe@pcl.cs.waseda.ac.jp
