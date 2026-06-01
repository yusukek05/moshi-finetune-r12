// Data model for the LLM-jp-Moshi data-flow Sankey.
// Keep numbers in sync with the Python measurement script
// (tools/build_sankey_overview.py) — both render the same dataset.

export type License = 'ok' | 'nc' | 'unk' | 'plan' | 'base'
export type Group = 'src' | 'mid' | 'model'
export type Category = 'v1' | 'mstts' | 'clean' | 'shared'

export interface SampleDef {
  src: string         // path relative to /samples/<node-id>/
  caption?: string    // short label, e.g. "ニュース読み上げ・10s"
  topic?: string      // first ~40 chars of transcript / ASR / contents hint
  duration?: number   // seconds (display only)
  license?: string    // freeform credit, e.g. "CC-BY-4.0 / J-CHAT"
  source?: string     // origin trace, e.g. "podcast_test/<hash>.wav から 30-40s 抜粋"
}

export interface ExternalLinkDef {
  href: string
  label: string
}

export interface NodeDef {
  id: string
  label: string
  hover: string
  group: Group
  license: License
  cats: Category[]
  samples?: SampleDef[]            // bundled wavs under public/samples/<id>/
  externalLinks?: ExternalLinkDef[] // for corpora we can't bundle (Reazon HF, etc.)
}

export interface EdgeDef {
  source: string
  target: string
  hours: number
  label: string
  license: License
  cats: Category[]
}

export const PALETTE: Record<License, string> = {
  ok: '#3a8540',
  nc: '#c64a3b',
  unk: '#888888',
  plan: '#cba135',
  base: '#2f6fb7',
}

export const LICENSE_LABEL: Record<License, string> = {
  ok: '商用可',
  nc: '非商用 (NC) ⚠',
  unk: 'その他',
  plan: '予定',
  base: 'ベースモデル',
}

export const GROUP_LABEL: Record<Group, string> = {
  src: 'ソース',
  mid: '中間生成物',
  model: 'モデル',
}

export const NODES: NodeDef[] = [
  // ---- SOURCES ----
  { id: 'moshiko',  label: 'Moshiko (base)',     hover: 'Kyutai Moshiko — CC-BY-4.0（v1 系のベースモデル）', group: 'src', license: 'base', cats: ['v1', 'clean'] },
  { id: 'moshika',  label: 'Moshika (base)',     hover: 'Kyutai Moshika — CC-BY-4.0（mstts 系のベースモデル）', group: 'src', license: 'base', cats: ['mstts', 'clean'] },
  { id: 'reazon',   label: 'ReazonSpeech',       hover: 'ReazonSpeech: 4,851 h 利用（35,413 h 利用可能、large subset 相当）— CC-BY-4.0', group: 'src', license: 'ok', cats: ['v1', 'mstts', 'clean', 'shared'],
    samples: [
      { src: 'sample_1.wav', caption: 'Reazon サンプル 1・8s', topic: '今も相手にロンバルドのほうに肩口で握られても… (相撲実況)', duration: 8, license: 'CC-BY-4.0 (ReazonSpeech)', source: 'tiny subset / streaming' },
      { src: 'sample_2.wav', caption: 'Reazon サンプル 2・8s', topic: '積極的にお金を使うべきだと主張する政治家や省庁と… (財政ニュース)', duration: 8, license: 'CC-BY-4.0', source: 'tiny subset / streaming' },
      { src: 'sample_3.wav', caption: 'Reazon サンプル 3・5s', topic: '今大会のボキの泳ぎ杉内さんはどう感じてらっしゃいますか？ (スポーツ取材)', duration: 5, license: 'CC-BY-4.0', source: 'tiny subset / streaming' },
    ],
    externalLinks: [
      { href: 'https://huggingface.co/datasets/reazon-research/reazonspeech', label: 'HF dataset で全件試聴' },
    ],
  },
  { id: 'jchat',    label: 'J-CHAT',             hover: 'J-CHAT: 72,053 h mono + 57,466 h podcast multi-stream — 商用 OK（2026-05-14 確認）', group: 'src', license: 'ok', cats: ['v1', 'mstts', 'clean', 'shared'],
    samples: [
      { src: 'mono_sample.wav',    caption: 'mono サンプル 1・10s',  topic: 'これを聞いてほしい方は松山さんを探してください…(雑談ラジオ風)',  duration: 10, license: 'J-CHAT, 商用 OK', source: 'podcast_test/00000-of-00001/cuts.000000/' },
      { src: 'mono_sample2.wav',   caption: 'mono サンプル 2・12s',  topic: '350メートルは読んだんです。その下にちっちゃい字で… (商店街・建築の話題)',  duration: 12, license: 'J-CHAT, 商用 OK', source: 'podcast_test/00000-of-00001/cuts.000000/' },
      { src: 'mono_sample3.wav',   caption: 'mono サンプル 3・12s',  topic: '倉先輩はいろんなもの染めてやってますか… (染色・趣味)',  duration: 12, license: 'J-CHAT, 商用 OK', source: 'podcast_test/00000-of-00001/cuts.000000/' },
      { src: 'podcast_sample.wav',  caption: 'podcast サンプル 1・15s', topic: '子どもがそんなに頑張らなくても、1歩とか2歩とかで片づ… (育児・しつけ)', duration: 15, license: 'J-CHAT, 商用 OK', source: 'podcast_train/00000-of-01432/cuts.000000/' },
      { src: 'podcast_sample2.wav', caption: 'podcast サンプル 2・12s', topic: '負けちゃったのは残念だけど来年頑張ろうよの声… (スポーツの慰め会話)', duration: 12, license: 'J-CHAT, 商用 OK', source: 'podcast_train/00000-of-01432/cuts.000000/' },
      { src: 'podcast_sample3.wav', caption: 'podcast サンプル 3・12s', topic: '佐紀長が言ってたような共感できるような部分… (共感をめぐる雑談)', duration: 12, license: 'J-CHAT, 商用 OK', source: 'podcast_train/00010-of-01432/cuts.000000/' },
    ],
  },
  { id: 'laboro',   label: 'LaboroTV ⚠NC',       hover: 'LaboroTVSpeech: 6,614 h — 非商用限定（Laboro.AI 申請制）。v0c 系が NC な唯一の原因', group: 'src', license: 'nc', cats: ['mstts'],
    samples: [
      { src: 'sample_1.wav', caption: 'LaboroTV 1・4s (TV番組)', topic: 'わたしいつも公園なんかで一人で歌っているんです (バラエティ/トーク風)', duration: 4, license: 'LaboroTVSpeech (非商用限定)', source: 'shard-00000.tar :: dev/000/v001_dev_00ZhV6K9.wav' },
      { src: 'sample_2.wav', caption: 'LaboroTV 2・5s (TV番組)', topic: 'さらに生まれた赤ちゃんも何頭か感染してしまいました (ニュース・動物の感染症)', duration: 5, license: 'LaboroTVSpeech (非商用限定)', source: 'shard-00000.tar :: dev/000/v001_dev_01BfRDBT.wav' },
      { src: 'sample_3.wav', caption: 'LaboroTV 3・5s (TV番組)', topic: '森崎も見つかる危険を冒しアクアシティ屋上へ向かう (ドラマ/ナレーション)', duration: 5, license: 'LaboroTVSpeech (非商用限定)', source: 'shard-00000.tar :: dev/000/v001_dev_01GKMQla.wav' },
    ],
  },
  { id: 'zoom1',    label: 'Zoom1',              hover: 'Zoom1 (LLM-jp internal): 935 h train + 99 h test — CC-BY + LLM-jp、商用 OK', group: 'src', license: 'ok', cats: ['v1', 'mstts', 'clean', 'shared'],
    samples: [
      { src: '0001_dialogue.wav', caption: 'Zoom1 0001 (W02+W03 stereo)・15s', topic: 'バトルのためにイケメンを育成するみたいな… (ゲーム雑談)', duration: 15, license: 'LLM-jp Zoom1', source: 'llmjp-zoom1/0001/0001_W02_W03_T01.m4a 5min-5:15min' },
      { src: '0002_dialogue.wav', caption: 'Zoom1 0002 (W02+W03 stereo)・15s', topic: 'そこまでではなかった… その時は140円とかで買って… (相場の雑談)', duration: 15, license: 'LLM-jp Zoom1', source: 'llmjp-zoom1/0002/0002_W02_W03_T02.m4a 5min-5:15min' },
      { src: '0005_dialogue.wav', caption: 'Zoom1 0005 (W02+W03 stereo)・15s', topic: 'その人が多分その当時今の私ぐらいの年代だった気がするんですよね… (回想雑談)', duration: 15, license: 'LLM-jp Zoom1', source: 'llmjp-zoom1/0005/0005_W02_W03_T05.m4a 5min-5:15min' },
    ],
  },
  { id: 'vb',       label: 'VisualBank',         hover: 'VisualBank: 307 h — stereo録音、neutral発話多数', group: 'src', license: 'unk', cats: ['v1'],
    samples: [
      { src: 'sample_1.wav', caption: 'VB M001_0424 (neutral, stereo)・12s', topic: '確かにその通りですね マーケティング全般同じようなこと言えますので… (ビジネス雑談)', duration: 12, license: 'VisualBank (要ライセンス確認)', source: 'VisualBank/音声データ/M001_222/M001_0424 5min' },
      { src: 'sample_2.wav', caption: 'VB M001_0425 (neutral, stereo)・12s', topic: '経験したことないところで分からないところは分からない中で… (考察雑談)', duration: 12, license: 'VisualBank (要ライセンス確認)', source: 'VisualBank/音声データ/M001_222/M001_0425 8min' },
      { src: 'sample_3.wav', caption: 'VB M001_0426 (neutral, stereo)・12s', topic: 'アルファってことですね…日本の企業の中でも… (企業/オークション話題)', duration: 12, license: 'VisualBank (要ライセンス確認)', source: 'VisualBank/音声データ/M001_222/M001_0426 11min' },
    ],
  },
  { id: 'csj',      label: 'CSJ (候補)',         hover: 'CSJ 対話サブセット: 12.16 h (58 sessions, A/B speaker 分離済) + フル CSJ ~660h on gca50130/kusunoki — 商用利用は NINJAL ライセンス要確認', group: 'src', license: 'unk', cats: ['mstts', 'clean'],
    samples: [
      { src: 'D01F0002_dialogue.wav', caption: 'CSJ D01F0002 (女性×女性 対話)・15s', topic: 'そん時はお一人で…何かのプログラムで… (海外滞在の振り返り)', duration: 15, license: 'NINJAL CSJ (商用利用要契約確認)', source: '0162/CSJ/audio/core/D01F0002.wav 2min' },
      { src: 'D01M0009_dialogue.wav', caption: 'CSJ D01M0009 (男性×男性 対話)・15s', topic: '十何本くらいやってみてうまく行ったの一本だけだったんですけど… (実験の振り返り)', duration: 15, license: 'NINJAL CSJ (商用利用要契約確認)', source: '0162/CSJ/audio/core/D01M0009.wav 2min' },
      { src: 'D02M0028_dialogue.wav', caption: 'CSJ D02M0028 (男性×男性 対話)・15s', topic: '悪役商会の…次は中野浩一…元競…(芸能・スポーツトリビア)', duration: 15, license: 'NINJAL CSJ (商用利用要契約確認)', source: '0162/CSJ/audio/core/D02M0028.wav 2min' },
    ],
  },
  { id: 'cc_raw',   label: 'ccaudio raw_all',    hover: 'ccaudio raw_all: 23,685 h（11k h raw_1 + 13k h raw_2）podcast tars — CC、未書き起こし', group: 'src', license: 'ok', cats: ['mstts', 'clean'],
    samples: [
      { src: 'raw_sample.wav',    caption: 'CC ポッドキャスト 1・12s', topic: 'アイスエージとだんだんとかつかじゃんよろしく… (フリートーク冒頭)', duration: 12, license: 'CC, ccaudio aggregation', source: 'ccaudio_rss_raw_2/recording.000002.tar :: audio_00000200.flac' },
      { src: 'raw_sample_054.wav', caption: 'CC ポッドキャスト 2・12s', topic: 'なくてもいいけど、あると嬉しいよコンセプトに、九州にクラス… (商品紹介系)', duration: 12, license: 'CC, ccaudio aggregation', source: 'ccaudio_rss_raw_2/recording.000054.tar' },
      { src: 'raw_sample_069.wav', caption: 'CC ポッドキャスト 3・12s', topic: 'バスケ情報を拾っていきます 6月21日号ですね…b1の方は YouTube… (スポーツ情報番組)', duration: 12, license: 'CC, ccaudio aggregation', source: 'ccaudio_rss_raw_2/recording.000069.tar' },
    ],
  },
  { id: 'jmw',      label: 'JMultiWOZ',          hover: 'JMultiWOZ: 7,469 dialogues（テキスト）— CC-BY-SA、task-oriented 対話', group: 'src', license: 'ok', cats: ['mstts', 'clean'] },
  { id: 'rpc',      label: 'RealPersonaChat',    hover: 'RealPersonaChat: 38,797 dialogues（テキスト）— CC-BY-SA、casual chitchat', group: 'src', license: 'ok', cats: ['mstts', 'clean'] },
  // ---- INTERMEDIATES ----
  { id: 'mono0178',     label: '0178 mono ckpt',          hover: '0178 mono ckpt: moshika + Reazon + J-CHAT + LaboroTV を pretrain（LaboroTV 由来で NC）', group: 'mid', license: 'nc', cats: ['mstts'] },
  { id: 'mstts0178',    label: '0178 mstts ckpt',         hover: '0178 mstts ckpt: 0178 mono + J-CHAT podcast multi-stream で Stage 2', group: 'mid', license: 'nc', cats: ['mstts'] },
  { id: 'cc_v2',        label: 'ccaudio v2 filtered',     hover: 'ccaudio v2: 2,462 h / 587 shard（VAD 再分割 + 再 ASR で約10%採用、長尺アライメント問題を解消）', group: 'mid', license: 'ok', cats: ['mstts', 'clean'],
    samples: [
      { src: 'sample_1.wav', caption: 'cc_v2 セグメント 1・10s', topic: 'さあ早速ですが、お芋屋さん更新を経営する川子商店の5代目… (お店紹介・ラジオ番組)', duration: 10, license: 'CC, ccaudio v2 filtered (VAD 再分割後)', source: 'transcribed_array/0/recording.000000.tar :: audio_00000001.flac の 9.7s VAD セグメント' },
      { src: 'sample_2.wav', caption: 'cc_v2 セグメント 2・14s', topic: '14年目を迎えた当番組は、平成から令和に開元されたことで… (歴史・人文系の番組)', duration: 14, license: 'CC, ccaudio v2 filtered', source: 'transcribed_array/0/recording.000000.tar :: audio_00000002.flac の 14.2s VAD セグメント' },
      { src: 'sample_3.wav', caption: 'cc_v2 セグメント 3・9s',  topic: 'この神保宝記の紙というのは武漢や旗本名録で調べてもなかなか出てきませんが (郷土史・解説)', duration: 9, license: 'CC, ccaudio v2 filtered', source: 'transcribed_array/0/recording.000000.tar :: audio_00000003.flac の 9.4s VAD セグメント' },
    ],
  },
  { id: 'mstts_text',   label: 'mstts text inputs',       hover: 'mstts text inputs: JMultiWOZ + RPC を merge した 46,266 dialogues（合成エンジンへの入力）', group: 'mid', license: 'ok', cats: ['mstts', 'clean'] },
  { id: 'synth_wav',    label: 'v0c synth wavs',          hover: 'mstts v0c synth wavs: 46,266 wav / 527 h（合成済対話音声）', group: 'mid', license: 'nc', cats: ['mstts'],
    samples: [
      { src: 'v0c_dialogue1.wav', caption: 'v0c 合成対話 1・10s (L=A R=B)',  topic: '「東京に行きたい」←→「お問い合わせありがとうございます」(JMultiWOZ・観光案内)', duration: 10, license: 'kobas-lab/llm-jp-moshi-mstts-v0c-zoom1 由来 (CC-BY-NC-4.0)、研究デモ目的のみ', source: 'output/mstts_v0c_synth/0_5000/dialogue_0002WLBV_part00.wav' },
      { src: 'v0c_dialogue2.wav', caption: 'v0c 合成対話 2・12s (L=A R=B)',  topic: '京都旅行の飲食店問い合わせ (JMultiWOZ・観光案内)', duration: 12, license: '同上', source: 'output/mstts_v0c_synth/0_5000/dialogue_0003vFlb_part00.wav' },
      { src: 'v0c_dialogue3.wav', caption: 'v0c 合成対話 3・12s (L=A R=B)',  topic: '「是非楽しんでください」←→「これで大丈夫です」(JMultiWOZ・問い合わせ終結)', duration: 12, license: '同上', source: 'output/mstts_v0c_synth/0_5000/dialogue_0004HAyh_part03.wav' },
    ],
  },
  { id: 'synth_parquet',label: 'v0c synth → parquet',     hover: 'v0c synth → v1 parquet: 357 MB（v1.x 追加学習用フォーマット）', group: 'mid', license: 'nc', cats: ['mstts'] },
  { id: 'synth_commercial', label: 'commercial synth (plan)', hover: 'Commercial mstts synth corpus（v0d_v2 完成後、商用クリーンな合成コーパス）', group: 'mid', license: 'plan', cats: ['v1', 'clean'] },
  // ---- MODELS — v1 line ----
  { id: 'v1',    label: 'v1 (public)',           hover: 'v1: J-CHAT → Zoom1、LLM-jp org で publicly released', group: 'model', license: 'ok', cats: ['v1', 'clean'] },
  { id: 'v1_1',  label: 'v1.1 (ckpt)',           hover: 'v1.1: +ReazonSpeech 前段、ckpt 完成（HF private、公開準備中）', group: 'model', license: 'ok', cats: ['v1', 'clean'] },
  { id: 'v1_2',  label: 'v1.2 (MOS pending)',    hover: 'v1.2: +VisualBank、ckpt 完成・MOS 評価 pending', group: 'model', license: 'ok', cats: ['v1', 'clean'] },
  { id: 'v1_3',  label: 'v1.3 (planned)',        hover: 'v1.3: +商用合成コーパス（v0d_v2 完成後）— v1 を超える対話 LM の最終形', group: 'model', license: 'plan', cats: ['v1', 'clean'] },
  // ---- MODELS — mstts line ----
  { id: 'v0b',     label: 'v0b (NC mstts ckpt)', hover: 'v0b: 0178 mstts + Zoom1（+500 step、HF private）', group: 'model', license: 'nc', cats: ['mstts'] },
  { id: 'v0c',     label: 'v0c (NC synth engine)', hover: 'v0c: v0b + Zoom1（+1500 step）— 現運用版 mstts 合成エンジン（NC、HF private）', group: 'model', license: 'nc', cats: ['mstts'] },
  { id: 'v0d',     label: 'v0d ❌ clean 試行 失敗', hover: 'v0d: LaboroTV-free 再構築失敗（CER 63.6% vs v0c 36.1%）。ccaudio v1 フィルタバグが原因', group: 'model', license: 'nc', cats: ['mstts'] },
  { id: 'v0d_v2',  label: 'v0d_v2 (clean synth plan)', hover: 'v0d_v2: ccaudio v2 で Stage 1 から再構築、商用 OK な mstts 後継候補', group: 'model', license: 'plan', cats: ['mstts', 'clean'] },
]

export const EDGES: EdgeDef[] = [
  // === 0178 mono construction ===
  { source: 'moshika', target: 'mono0178', hours: 1,     label: 'base weights', license: 'base', cats: ['mstts'] },
  { source: 'reazon',  target: 'mono0178', hours: 4851,  label: 'ReazonSpeech 4,851h', license: 'ok', cats: ['mstts'] },
  { source: 'jchat',   target: 'mono0178', hours: 72053, label: 'J-CHAT-mono 72,053h', license: 'ok', cats: ['mstts'] },
  { source: 'laboro',  target: 'mono0178', hours: 6614,  label: 'LaboroTV 6,614h ⚠NC', license: 'nc', cats: ['mstts'] },
  // === 0178 mstts ===
  { source: 'mono0178', target: 'mstts0178', hours: 83519, label: 'mono ckpt', license: 'nc', cats: ['mstts'] },
  { source: 'jchat',    target: 'mstts0178', hours: 57466, label: 'J-CHAT-podcast 57,466h', license: 'ok', cats: ['mstts'] },
  // === v0b / v0c ===
  { source: 'mstts0178', target: 'v0b', hours: 140985, label: 'init ckpt', license: 'nc', cats: ['mstts'] },
  { source: 'zoom1',     target: 'v0b', hours: 935,    label: 'Zoom1 +500 step', license: 'ok', cats: ['mstts'] },
  { source: 'v0b',       target: 'v0c', hours: 141920, label: 'v0b ckpt', license: 'nc', cats: ['mstts'] },
  { source: 'zoom1',     target: 'v0c', hours: 935,    label: 'Zoom1 +1500 step', license: 'ok', cats: ['mstts'] },
  // === v0d (failed) ===
  { source: 'moshika', target: 'v0d', hours: 1,     label: 'base', license: 'base', cats: ['mstts'] },
  { source: 'reazon',  target: 'v0d', hours: 4851,  label: 'Stage 1 mono', license: 'ok', cats: ['mstts'] },
  { source: 'jchat',   target: 'v0d', hours: 72053, label: 'Stage 1 J-CHAT-mono', license: 'ok', cats: ['mstts'] },
  { source: 'jchat',   target: 'v0d', hours: 57466, label: 'Stage 2 J-CHAT-podcast', license: 'ok', cats: ['mstts'] },
  { source: 'cc_raw',  target: 'v0d', hours: 6613,  label: 'ccaudio v1 transcribed (broken filter)', license: 'ok', cats: ['mstts'] },
  { source: 'zoom1',   target: 'v0d', hours: 935,   label: 'Stage 3', license: 'ok', cats: ['mstts'] },
  // === ccaudio v2 ===
  { source: 'cc_raw', target: 'cc_v2', hours: 23685, label: 'raw → VAD re-seg + re-ASR (~9.4% kept)', license: 'ok', cats: ['mstts', 'clean'] },
  // === v0d_v2 ===
  { source: 'moshika', target: 'v0d_v2', hours: 1,     label: 'base', license: 'base', cats: ['mstts', 'clean'] },
  { source: 'reazon',  target: 'v0d_v2', hours: 4851,  label: 'Stage 1 mono', license: 'ok', cats: ['mstts', 'clean'] },
  { source: 'jchat',   target: 'v0d_v2', hours: 72053, label: 'Stage 1 J-CHAT-mono', license: 'ok', cats: ['mstts', 'clean'] },
  { source: 'jchat',   target: 'v0d_v2', hours: 57466, label: 'Stage 2 J-CHAT-podcast', license: 'ok', cats: ['mstts', 'clean'] },
  { source: 'cc_v2',   target: 'v0d_v2', hours: 2462,  label: 'ccaudio v2 (2,462h)', license: 'ok', cats: ['mstts', 'clean'] },
  { source: 'zoom1',   target: 'v0d_v2', hours: 935,   label: 'Stage 3', license: 'ok', cats: ['mstts', 'clean'] },
  { source: 'csj',     target: 'v0d_v2', hours: 12,    label: 'CSJ 対話 12h (候補・要 NINJAL 商用契約)', license: 'plan', cats: ['mstts', 'clean'] },
  // === mstts text → synth ===
  { source: 'jmw',        target: 'mstts_text', hours: 85,  label: '7,469 chunks → ~85h synth', license: 'ok', cats: ['mstts', 'clean'] },
  { source: 'rpc',        target: 'mstts_text', hours: 442, label: '38,797 chunks → ~442h synth', license: 'ok', cats: ['mstts', 'clean'] },
  { source: 'mstts_text', target: 'synth_wav',  hours: 527, label: 'text → v0c inference', license: 'ok', cats: ['mstts'] },
  { source: 'v0c',        target: 'synth_wav',  hours: 527, label: 'v0c synth engine ⚠NC', license: 'nc', cats: ['mstts'] },
  { source: 'synth_wav',  target: 'synth_parquet', hours: 527, label: 'wavs → v1 parquet', license: 'nc', cats: ['mstts'] },
  // === v1 line ===
  { source: 'moshiko', target: 'v1', hours: 1,     label: 'base', license: 'base', cats: ['v1', 'clean'] },
  { source: 'jchat',   target: 'v1', hours: 72053, label: 'Stage 1 J-CHAT-mono', license: 'ok', cats: ['v1', 'clean'] },
  { source: 'zoom1',   target: 'v1', hours: 935,   label: 'Stage 2 Zoom1', license: 'ok', cats: ['v1', 'clean'] },
  { source: 'moshiko', target: 'v1_1', hours: 1,    label: 'base', license: 'base', cats: ['v1', 'clean'] },
  { source: 'reazon',  target: 'v1_1', hours: 4851, label: 'Stage 1 ReazonSpeech', license: 'ok', cats: ['v1', 'clean'] },
  { source: 'jchat',   target: 'v1_1', hours: 72053,label: 'Stage 2 J-CHAT-mono', license: 'ok', cats: ['v1', 'clean'] },
  { source: 'zoom1',   target: 'v1_1', hours: 935,  label: 'Stage 3 Zoom1', license: 'ok', cats: ['v1', 'clean'] },
  { source: 'moshiko', target: 'v1_2', hours: 1,    label: 'base', license: 'base', cats: ['v1', 'clean'] },
  { source: 'reazon',  target: 'v1_2', hours: 4851, label: 'Stage 1', license: 'ok', cats: ['v1', 'clean'] },
  { source: 'jchat',   target: 'v1_2', hours: 72053,label: 'Stage 2', license: 'ok', cats: ['v1', 'clean'] },
  { source: 'vb',      target: 'v1_2', hours: 307,  label: 'Stage 3 VisualBank', license: 'unk', cats: ['v1', 'clean'] },
  { source: 'zoom1',   target: 'v1_2', hours: 935,  label: 'Stage 4 Zoom1', license: 'ok', cats: ['v1', 'clean'] },
  // === v1.3 (planned, needs commercial mstts) ===
  { source: 'v0d_v2',     target: 'synth_commercial', hours: 527, label: 'v0d_v2 as synth engine', license: 'plan', cats: ['v1', 'clean'] },
  { source: 'mstts_text', target: 'synth_commercial', hours: 527, label: 'same text inputs', license: 'ok', cats: ['v1', 'clean'] },
  { source: 'moshiko',          target: 'v1_3', hours: 1,    label: 'base', license: 'base', cats: ['v1', 'clean'] },
  { source: 'reazon',           target: 'v1_3', hours: 4851, label: 'Stage 1', license: 'ok', cats: ['v1', 'clean'] },
  { source: 'jchat',            target: 'v1_3', hours: 72053,label: 'Stage 2', license: 'ok', cats: ['v1', 'clean'] },
  { source: 'vb',               target: 'v1_3', hours: 307,  label: 'Stage 3', license: 'unk', cats: ['v1', 'clean'] },
  { source: 'synth_commercial', target: 'v1_3', hours: 527,  label: '+commercial synth', license: 'plan', cats: ['v1', 'clean'] },
  { source: 'zoom1',            target: 'v1_3', hours: 935,  label: 'Stage 5 Zoom1', license: 'ok', cats: ['v1', 'clean'] },
]

export const SNAPSHOT = '2026-05-30'
