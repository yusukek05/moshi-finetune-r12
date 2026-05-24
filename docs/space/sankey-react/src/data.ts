// Data model for the LLM-jp-Moshi data-flow Sankey.
// Keep numbers in sync with the Python measurement script
// (tools/build_sankey_overview.py) — both render the same dataset.

export type License = 'ok' | 'nc' | 'unk' | 'plan' | 'base'
export type Group = 'src' | 'mid' | 'model'
export type Category = 'v1' | 'mstts' | 'clean' | 'shared'

export interface NodeDef {
  id: string
  label: string
  hover: string
  group: Group
  license: License
  cats: Category[]
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
  { id: 'reazon',   label: 'ReazonSpeech',       hover: 'ReazonSpeech: 4,851 h 利用（35,413 h 利用可能、large subset 相当）— CC-BY-4.0', group: 'src', license: 'ok', cats: ['v1', 'mstts', 'clean', 'shared'] },
  { id: 'jchat',    label: 'J-CHAT',             hover: 'J-CHAT: 72,053 h mono + 57,466 h podcast multi-stream — 商用 OK（2026-05-14 確認）', group: 'src', license: 'ok', cats: ['v1', 'mstts', 'clean', 'shared'] },
  { id: 'laboro',   label: 'LaboroTV ⚠NC',       hover: 'LaboroTVSpeech: 6,614 h — 非商用限定（Laboro.AI 申請制）。v0c 系が NC な唯一の原因', group: 'src', license: 'nc', cats: ['mstts'] },
  { id: 'zoom1',    label: 'Zoom1',              hover: 'Zoom1 (LLM-jp internal): 935 h train + 99 h test — CC-BY + LLM-jp、商用 OK', group: 'src', license: 'ok', cats: ['v1', 'mstts', 'clean', 'shared'] },
  { id: 'vb',       label: 'VisualBank',         hover: 'VisualBank: 307 h', group: 'src', license: 'unk', cats: ['v1'] },
  { id: 'cc_raw',   label: 'ccaudio raw_all',    hover: 'ccaudio raw_all: 23,685 h（11k h raw_1 + 13k h raw_2）podcast tars — CC、未書き起こし', group: 'src', license: 'ok', cats: ['mstts', 'clean'] },
  { id: 'jmw',      label: 'JMultiWOZ',          hover: 'JMultiWOZ: 7,469 dialogues（テキスト）— CC-BY-SA、task-oriented 対話', group: 'src', license: 'ok', cats: ['mstts', 'clean'] },
  { id: 'rpc',      label: 'RealPersonaChat',    hover: 'RealPersonaChat: 38,797 dialogues（テキスト）— CC-BY-SA、casual chitchat', group: 'src', license: 'ok', cats: ['mstts', 'clean'] },
  // ---- INTERMEDIATES ----
  { id: 'mono0178',     label: '0178 mono ckpt',          hover: '0178 mono ckpt: moshika + Reazon + J-CHAT + LaboroTV を pretrain（LaboroTV 由来で NC）', group: 'mid', license: 'nc', cats: ['mstts'] },
  { id: 'mstts0178',    label: '0178 mstts ckpt',         hover: '0178 mstts ckpt: 0178 mono + J-CHAT podcast multi-stream で Stage 2', group: 'mid', license: 'nc', cats: ['mstts'] },
  { id: 'cc_v2',        label: 'ccaudio v2 filtered',     hover: 'ccaudio v2: 2,232 h（VAD 再分割 + 再 ASR で 9.4% 採用、長尺アライメント問題を解消）', group: 'mid', license: 'ok', cats: ['mstts', 'clean'] },
  { id: 'mstts_text',   label: 'mstts text inputs',       hover: 'mstts text inputs: JMultiWOZ + RPC を merge した 46,266 dialogues（合成エンジンへの入力）', group: 'mid', license: 'ok', cats: ['mstts', 'clean'] },
  { id: 'synth_wav',    label: 'v0c synth wavs',          hover: 'mstts v0c synth wavs: 46,266 wav / 527 h（合成済対話音声）', group: 'mid', license: 'nc', cats: ['mstts'] },
  { id: 'synth_parquet',label: 'v0c synth → parquet',     hover: 'v0c synth → v1 parquet: 357 MB（v1.x 追加学習用フォーマット）', group: 'mid', license: 'nc', cats: ['mstts'] },
  { id: 'synth_commercial', label: 'commercial synth (plan)', hover: 'Commercial mstts synth corpus（v0d_v2 完成後、商用クリーンな合成コーパス）', group: 'mid', license: 'plan', cats: ['v1', 'clean'] },
  // ---- MODELS — v1 line ----
  { id: 'v1',    label: 'v1 (public)',           hover: 'v1: J-CHAT → Zoom1、LLM-jp org で publicly released', group: 'model', license: 'ok', cats: ['v1', 'clean'] },
  { id: 'v1_1',  label: 'v1.1 (ckpt)',           hover: 'v1.1: +ReazonSpeech 前段、ckpt 完成（HF private、公開準備中）', group: 'model', license: 'ok', cats: ['v1', 'clean'] },
  { id: 'v1_2',  label: 'v1.2 (MOS pending)',    hover: 'v1.2: +VisualBank、ckpt 完成・MOS 評価 pending', group: 'model', license: 'ok', cats: ['v1', 'clean'] },
  { id: 'v1_3',  label: 'v1.3 (planned)',        hover: 'v1.3: +商用合成コーパス（v0d_v2 完成後）— v1 を超える対話 LM の最終形', group: 'model', license: 'plan', cats: ['v1', 'clean'] },
  // ---- MODELS — mstts line ----
  { id: 'v0b',     label: 'v0b',                 hover: 'v0b: 0178 mstts + Zoom1（+500 step、HF private）', group: 'model', license: 'nc', cats: ['mstts'] },
  { id: 'v0c',     label: 'v0c (prod)',          hover: 'v0c: v0b + Zoom1（+1500 step）— 現運用版 mstts 合成エンジン（NC、HF private）', group: 'model', license: 'nc', cats: ['mstts'] },
  { id: 'v0d',     label: 'v0d ❌ failed',        hover: 'v0d: LaboroTV-free 再構築失敗（CER 63.6% vs v0c 36.1%）。ccaudio v1 フィルタバグが原因', group: 'model', license: 'nc', cats: ['mstts'] },
  { id: 'v0d_v2',  label: 'v0d_v2 (planned)',    hover: 'v0d_v2: ccaudio v2 で Stage 1 から再構築、商用 OK な mstts 後継候補', group: 'model', license: 'plan', cats: ['mstts', 'clean'] },
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
  { source: 'cc_v2',   target: 'v0d_v2', hours: 2232,  label: 'ccaudio v2 (2,232h)', license: 'ok', cats: ['mstts', 'clean'] },
  { source: 'zoom1',   target: 'v0d_v2', hours: 935,   label: 'Stage 3', license: 'ok', cats: ['mstts', 'clean'] },
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

export const SNAPSHOT = '2026-05-24'
