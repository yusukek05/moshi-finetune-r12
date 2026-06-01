# Audio sample attribution

このディレクトリの wav は LLM-jp-Moshi データフロー Sankey の "コーパス聴き比べ"
用に各原コーパスから短尺抜粋したものです。各ライセンスの帰属表記は以下:

| 配置先 | ファイル | サイズ | 出典 | License / Credit |
|---|---|---:|---|---|
| `jchat/` | `mono_sample.wav` (10s) | ~430 KB | `podcast_test/00000-of-00001/cuts.000000/01d54ff2...wav` | J-CHAT (商用 OK, 2026-05-14 確認) |
| `jchat/` | `mono_sample2.wav` (12s) | ~530 KB | `podcast_test/00000-of-00001/cuts.000000/` 別 wav | 同上 |
| `jchat/` | `mono_sample3.wav` (12s) | ~530 KB | 同上 別 wav | 同上 |
| `jchat/` | `podcast_sample.wav` (15s) | ~650 KB | `podcast_train/00000-of-01432/cuts.000000/00d51...wav` | 同上 |
| `jchat/` | `podcast_sample2.wav` (12s) | ~530 KB | `podcast_train/00000-of-01432/cuts.000000/005cbb...wav` | 同上 |
| `jchat/` | `podcast_sample3.wav` (12s) | ~530 KB | `podcast_train/00010-of-01432/cuts.000000/0067b1...wav` | 同上 |
| `cc_raw/` | `raw_sample.wav` (12s) | ~520 KB | `ccaudio_rss_raw_2/recording.000002.tar :: audio_00000200.flac` | CC, ccaudio aggregation |
| `cc_raw/` | `raw_sample_054.wav` (12s) | ~520 KB | `recording.000054.tar` の最初の flac | 同上 |
| `cc_raw/` | `raw_sample_069.wav` (12s) | ~520 KB | `recording.000069.tar` の最初の flac | 同上 |
| `synth_wav/` | `v0c_dialogue1.wav` (10s) | ~960 KB | `output/mstts_v0c_synth/0_5000/decoded_audio/dialogue_0002WLBV_part00.wav` の 0-10s | kobas-lab/llm-jp-moshi-mstts-v0c-zoom1 由来 (CC-BY-NC-4.0)、研究デモ目的 |
| `synth_wav/` | `v0c_dialogue2.wav` (12s) | ~1.1 MB | `dialogue_0003vFlb_part00.wav` の 5-17s | 同上 |
| `synth_wav/` | `v0c_dialogue3.wav` (12s) | ~860 KB | `dialogue_0004HAyh_part03.wav` の 3-15s | 同上 |
| `zoom1/` | `0001_dialogue.wav` (15s) | ~1.4 MB | `llmjp-zoom1/0001/0001_W02_W03_T01.m4a` の 5-5:15min | LLM-jp Zoom1 (内部利用, CC-BY) |
| `zoom1/` | `0002_dialogue.wav` (15s) | ~1.4 MB | `llmjp-zoom1/0002/0002_W02_W03_T02.m4a` の 5-5:15min | 同上 |
| `zoom1/` | `0005_dialogue.wav` (15s) | ~1.4 MB | `llmjp-zoom1/0005/0005_W02_W03_T05.m4a` の 5-5:15min | 同上 |
| `vb/` | `sample_1.wav` (12s) | ~1.1 MB | `VisualBank/音声データ/M001_222/M001_0424_aaax_stereo06_neutral.wav` 5min | VisualBank (要 license 確認) |
| `vb/` | `sample_2.wav` (12s) | ~1.1 MB | `M001_0425_aaCS_stereo06_neutral.wav` 8min | 同上 |
| `vb/` | `sample_3.wav` (12s) | ~1.1 MB | `M001_0426_aaCT_stereo06_neutral.wav` 11min | 同上 |
| `csj/` | `D01F0002_dialogue.wav` (15s) | ~960 KB | `0162/CSJ/audio/core/D01F0002.wav` の 2-2:15min | NINJAL CSJ (商用利用は要契約確認) |
| `csj/` | `D01M0009_dialogue.wav` (15s) | ~960 KB | `0162/CSJ/audio/core/D01M0009.wav` の 2-2:15min | 同上 |
| `csj/` | `D02M0028_dialogue.wav` (15s) | ~960 KB | `0162/CSJ/audio/core/D02M0028.wav` の 2-2:15min | 同上 |

## 抽出再現

```bash
JCHAT=/groups/gcg51557/experiments/0162_dialogue_model/J-CHAT/audio
SYNTH=/groups/gcg51557/experiments/0162_dialogue_model/moshi-finetune/output/mstts_v0c_synth/0_5000/decoded_audio
CC=/groups/gcg51557/experiments/0167_cc_audio/asai/ccaudio_rss_raw_all/ccaudio_rss_raw_2

# J-CHAT mono
ffmpeg -y -ss 30 -i "$JCHAT/podcast_test/00000-of-00001/cuts.000000/01d54ff2fc51137908fb398579397005.wav" \
    -t 10 -ar 22050 -ac 1 -af loudnorm jchat/mono_sample.wav

# J-CHAT podcast
ffmpeg -y -ss 5 -i "$JCHAT/podcast_train/00000-of-01432/cuts.000000/00d5175520567dd01c7e6066ea0a3703.wav" \
    -t 15 -ar 22050 -ac 1 -af loudnorm jchat/podcast_sample.wav

# v0c synth (stereo, L=A R=B)
ffmpeg -y -ss 0 -i "$SYNTH/dialogue_0002WLBV_part00.wav" -t 10 -ar 24000 -ac 2 synth_wav/v0c_dialogue1.wav
ffmpeg -y -ss 5 -i "$SYNTH/dialogue_0003vFlb_part00.wav" -t 12 -ar 24000 -ac 2 synth_wav/v0c_dialogue2.wav

# ccaudio (extract flac from tar then transcode)
tmpd=$(mktemp -d) && tar -xf "$CC/recording.000002.tar" -C "$tmpd" audio_00000200.flac
ffmpeg -y -ss 5 -i "$tmpd/audio_00000200.flac" -t 12 -ar 22050 -ac 1 -af loudnorm cc_raw/raw_sample.wav
rm -rf "$tmpd"
```

## Phase 2 (ライセンス確認後追加予定)

- `zoom1/` — LLM-jp Zoom1 (`0293/.../crowdsourcing-backchannel/audio/`)、要内部承認
- `vb/` — VisualBank (`0215/.../音声データ/`)、要 Sarulab/Toma 確認
- (新候補) CSJ 対話サブセット (`0162/CSJ/audio/`)、要 NINJAL ライセンス確認

## LaboroTV (NC ライセンス・研究参考用)

| 配置先 | ファイル | サイズ | 出典 | License / Credit |
|---|---|---:|---|---|
| `laboro/` | `sample_1.wav` (4s) | ~170 KB | `0178/data/archive/LaboroTVSpeech/shard-00000.tar :: dev/000/v001_dev_00ZhV6K9.wav` | LaboroTVSpeech (非商用限定、TV局著作権) |
| `laboro/` | `sample_2.wav` (5s) | ~200 KB | shard-00000.tar :: `v001_dev_01BfRDBT.wav` | 同上 |
| `laboro/` | `sample_3.wav` (5s) | ~230 KB | shard-00000.tar :: `v001_dev_01GKMQla.wav` | 同上 |

公開 Space に研究参考用として配置 (商用利用不可)。UI 上に NC 警告を表示。
