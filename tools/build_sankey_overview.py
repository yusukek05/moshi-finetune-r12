"""Build a Sankey diagram of the 0162 data flow.

Sources (raw corpora & base models) → Intermediates (filtered / tokenized /
synth corpora & ckpt families) → Models (released or planned).

Edge widths are audio hours where known; estimates marked "~". Node colors
encode commercial usability: green=OK, red=NC-blocked, gray=undecided/info.
Output is self-contained HTML (Plotly bundled via CDN).
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# License color palette
# ---------------------------------------------------------------------------
C_OK = "#3a8540"  # commercial OK
C_NC = "#c64a3b"  # non-commercial (LaboroTV-derived)
C_UNK = "#888888"  # undecided / model family
C_PLAN = "#cba135"  # planned / not-yet-built
C_BASE = "#2f6fb7"  # base model

# ---------------------------------------------------------------------------
# Node table: (short_label, hover_desc, group, color)
# - short_label: rendered on diagram (kept compact)
# - hover_desc: revealed on hover (full hours + license)
# - group: column layout marker
# ---------------------------------------------------------------------------
NODES = [
    # SOURCES (raw corpora & base models)
    ("Moshiko (base)", "Kyutai Moshiko — CC-BY-4.0 (base weights)", "src", C_BASE),  # 0
    ("Moshika (base)", "Kyutai Moshika — CC-BY-4.0 (base weights)", "src", C_BASE),  # 1
    (
        "ReazonSpeech",
        "ReazonSpeech: 4,851h used (of 35,413h available) — CC-BY-4.0",
        "src",
        C_OK,
    ),  # 2
    ("J-CHAT", "J-CHAT: 72,053h mono + 57,466h podcast (mstts) — commercial OK", "src", C_OK),  # 3
    (
        "LaboroTV ⚠NC",
        "LaboroTVSpeech: 6,614h — NC ⚠ (non-commercial, the v0c NC source)",
        "src",
        C_NC,
    ),  # 4
    ("Zoom1", "Zoom1 (LLM-jp internal): 935h train + 99h test — CC-BY + LLM-jp", "src", C_OK),  # 5
    ("VisualBank", "VisualBank: 307h", "src", C_UNK),  # 6
    (
        "ccaudio raw_all",
        "ccaudio raw_all: 23,685h podcast tars (CC) — pre-VAD source",
        "src",
        C_OK,
    ),  # 7
    ("JMultiWOZ", "JMultiWOZ: 7,469 text dialogues (CC-BY-SA) — task-oriented", "src", C_OK),  # 8
    (
        "RealPersonaChat",
        "RealPersonaChat: 38,797 text dialogues (CC-BY-SA) — casual chitchat",
        "src",
        C_OK,
    ),  # 9
    ("pseudo_kanzaki", "pseudo_dialog_kanzaki: private synth corpus (神崎氏)", "src", C_UNK),  # 10
    # INTERMEDIATES
    (
        "0178 mono ckpt",
        "0178 mono ckpt: moshika + Reazon + J-CHAT + LaboroTV (NC-tainted)",
        "mid",
        C_NC,
    ),  # 11
    (
        "0178 mstts ckpt",
        "0178 mstts ckpt: 0178 mono + J-CHAT podcast multi-stream",
        "mid",
        C_NC,
    ),  # 12
    (
        "ccaudio v2 filtered",
        "ccaudio v2 filtered: 2,232h kept after VAD re-seg + re-ASR (~9.4%)",
        "mid",
        C_OK,
    ),  # 13
    (
        "mstts text inputs",
        "mstts text inputs: JMultiWOZ + RPC merged (46,266 dialogues)",
        "mid",
        C_OK,
    ),  # 14
    ("v0c synth wavs", "mstts v0c synth wavs: 46,266 wavs / ~527h synth audio", "mid", C_NC),  # 15
    (
        "v0c synth → parquet",
        "v0c synth → v1 parquet: 357MB (v1.x training input)",
        "mid",
        C_NC,
    ),  # 16
    (
        "commercial synth (plan)",
        "Commercial mstts synth corpus (planned, via v0d_v2)",
        "mid",
        C_PLAN,
    ),  # 17
    # MODELS — v1 line
    ("v1 (public)", "v1: J-CHAT → Zoom1 (publicly released, llm-jp org)", "model", C_OK),  # 18
    (
        "v1.1 (ckpt)",
        "v1.1: +ReazonSpeech, ckpt ready (HF private, pending public)",
        "model",
        C_OK,
    ),  # 19
    ("v1.2 (MOS pending)", "v1.2: +VisualBank, ckpt ready, MOS eval pending", "model", C_OK),  # 20
    ("v1.3 (planned)", "v1.3: +commercial synth corpus (needs v0d_v2)", "model", C_PLAN),  # 21
    (
        "v1.1b-e (ablation)",
        "v1.1b/c/d/e: synth ablation experiments (HF private)",
        "model",
        C_UNK,
    ),  # 22
    # MODELS — mstts line
    ("v0a (toy)", "v0a: moshiko + J-CHAT 1shard — toy test only (noisy)", "model", C_UNK),  # 23
    ("v0b", "v0b: 0178 mstts + Zoom1 (+500 step, HF private)", "model", C_NC),  # 24
    (
        "v0c (prod mstts)",
        "v0c: v0b + Zoom1 (+1500 step) — current production synth engine",
        "model",
        C_NC,
    ),  # 25
    (
        "v0d ❌ failed",
        "v0d: LaboroTV-free rebuild attempt — failed (CER 63.6% vs v0c 36.1%)",
        "model",
        C_NC,
    ),  # 26
    (
        "v0d_v2 (planned)",
        "v0d_v2: ccaudio v2 rebuild (in progress, planned production)",
        "model",
        C_PLAN,
    ),  # 27
]

LABELS = [n[0] for n in NODES]
HOVER = [n[1] for n in NODES]
NODE_COLORS = [n[3] for n in NODES]

# Symbolic name -> index lookup (for readability in EDGES below).
IDX = {
    "moshiko": 0,
    "moshika": 1,
    "reazon": 2,
    "jchat": 3,
    "laboro": 4,
    "zoom1": 5,
    "vb": 6,
    "cc_raw": 7,
    "jmw": 8,
    "rpc": 9,
    "pseudo": 10,
    "mono0178": 11,
    "mstts0178": 12,
    "cc_v2": 13,
    "mstts_text": 14,
    "synth_wav": 15,
    "synth_parquet": 16,
    "synth_commercial": 17,
    "v1": 18,
    "v1_1": 19,
    "v1_2": 20,
    "v1_3": 21,
    "v1_ablation": 22,
    "v0a": 23,
    "v0b": 24,
    "v0c": 25,
    "v0d": 26,
    "v0d_v2": 27,
}

# ---------------------------------------------------------------------------
# Edges: (src, tgt, value, label, color)
# value units: rough audio hours unless otherwise marked.
# ---------------------------------------------------------------------------
EDGES = [
    # === 0178 mono construction (foundation of v0b/c/d) ===
    ("moshika", "mono0178", 1, "base weights", C_BASE),
    ("reazon", "mono0178", 4851, "ReazonSpeech 4,851h", C_OK),
    ("jchat", "mono0178", 72053, "J-CHAT-mono 72,053h", C_OK),
    ("laboro", "mono0178", 6614, "LaboroTV 6,614h ⚠NC", C_NC),
    # === 0178 mstts (mono + J-CHAT multi-stream) ===
    ("mono0178", "mstts0178", 83519, "mono ckpt (all inputs)", C_NC),
    ("jchat", "mstts0178", 57466, "J-CHAT-podcast 57,466h (mstts)", C_OK),
    # === v0a-c training ===
    ("moshiko", "v0a", 1, "base", C_BASE),
    ("jchat", "v0a", 146, "J-CHAT 1shard (~146h)", C_OK),
    ("mstts0178", "v0b", 140985, "init ckpt (mono+mstts data)", C_NC),
    ("zoom1", "v0b", 935, "Zoom1 +500 step", C_OK),
    ("v0b", "v0c", 141920, "v0b ckpt", C_NC),
    ("zoom1", "v0c", 935, "Zoom1 +1500 step", C_OK),
    # === v0d (failed): moshika + Reazon + J-CHAT + ccaudio v1 → mstts → Zoom1 ===
    ("moshika", "v0d", 1, "base", C_BASE),
    ("reazon", "v0d", 4851, "Stage 1 mono", C_OK),
    ("jchat", "v0d", 72053, "Stage 1 J-CHAT-mono", C_OK),
    ("jchat", "v0d", 57466, "Stage 2 J-CHAT-podcast", C_OK),
    ("cc_raw", "v0d", 6613, "ccaudio v1 transcribed (broken filter)", C_OK),
    ("zoom1", "v0d", 935, "Stage 3", C_OK),
    # === ccaudio v2 reprocess (CURRENT WORK) ===
    ("cc_raw", "cc_v2", 23685, "raw → VAD re-seg + re-ASR (~9.4% kept)", C_OK),
    # === v0d_v2 (planned) ===
    ("moshika", "v0d_v2", 1, "base", C_BASE),
    ("reazon", "v0d_v2", 4851, "Stage 1 mono", C_OK),
    ("jchat", "v0d_v2", 72053, "Stage 1 J-CHAT-mono", C_OK),
    ("jchat", "v0d_v2", 57466, "Stage 2 J-CHAT-podcast", C_OK),
    ("cc_v2", "v0d_v2", 2232, "ccaudio v2 (2,232h)", C_OK),
    ("zoom1", "v0d_v2", 935, "Stage 3", C_OK),
    # === mstts text inputs → synth corpus generation ===
    # h here = synth output hours contributed (proportional to chunk count)
    ("jmw", "mstts_text", 85, "7,469 chunks → ~85h synth", C_OK),
    ("rpc", "mstts_text", 442, "38,797 chunks → ~442h synth", C_OK),
    ("mstts_text", "synth_wav", 527, "text → v0c inference", C_OK),
    ("v0c", "synth_wav", 527, "v0c synth engine ⚠NC", C_NC),
    ("synth_wav", "synth_parquet", 527, "wavs → v1 parquet", C_NC),
    # === v1 line ===
    ("moshiko", "v1", 1, "base", C_BASE),
    ("jchat", "v1", 72053, "Stage 1 J-CHAT-mono", C_OK),
    ("zoom1", "v1", 935, "Stage 2 Zoom1", C_OK),
    ("moshiko", "v1_1", 1, "base", C_BASE),
    ("reazon", "v1_1", 4851, "Stage 1 ReazonSpeech", C_OK),
    ("jchat", "v1_1", 72053, "Stage 2 J-CHAT-mono", C_OK),
    ("zoom1", "v1_1", 935, "Stage 3 Zoom1", C_OK),
    ("moshiko", "v1_2", 1, "base", C_BASE),
    ("reazon", "v1_2", 4851, "Stage 1", C_OK),
    ("jchat", "v1_2", 72053, "Stage 2", C_OK),
    ("vb", "v1_2", 307, "Stage 3 VisualBank", C_UNK),
    ("zoom1", "v1_2", 935, "Stage 4 Zoom1", C_OK),
    # === v1.3 (planned, needs commercial mstts) ===
    ("v0d_v2", "synth_commercial", 527, "v0d_v2 as synth engine", C_PLAN),
    ("mstts_text", "synth_commercial", 527, "same text inputs", C_OK),
    ("moshiko", "v1_3", 1, "base", C_BASE),
    ("reazon", "v1_3", 4851, "Stage 1", C_OK),
    ("jchat", "v1_3", 72053, "Stage 2", C_OK),
    ("vb", "v1_3", 307, "Stage 3", C_UNK),
    ("synth_commercial", "v1_3", 527, "+commercial synth", C_PLAN),
    ("zoom1", "v1_3", 935, "Stage 5 Zoom1", C_OK),
    # === v1.1 ablation cluster (experimental, private; mixed subsets) ===
    ("moshiko", "v1_ablation", 1, "base", C_BASE),
    ("jchat", "v1_ablation", 72053, "J-CHAT", C_OK),
    ("zoom1", "v1_ablation", 935, "Zoom1", C_OK),
    ("vb", "v1_ablation", 307, "VB subsets", C_UNK),
    ("pseudo", "v1_ablation", 100, "神崎 synth (~100h estimate)", C_UNK),
]

# Resolve symbolic edges to numeric indices.
src_idx = [IDX[e[0]] for e in EDGES]
tgt_idx = [IDX[e[1]] for e in EDGES]
real_val = [e[2] for e in EDGES]  # true audio hours
labels = [e[3] for e in EDGES]

# Visual edge widths: sqrt scaling so tiny flows (300h VB) stay visible next
# to giant ones (72k J-CHAT). Real hours go into customdata for hover.
vis_val = [max(1.0, math.sqrt(v)) for v in real_val]
custom = [[v] for v in real_val]  # plotly customdata expects 2D


def _to_rgba(hex6: str, alpha: float = 0.55) -> str:
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


ecolors = [_to_rgba(e[4]) for e in EDGES]


# ---------------------------------------------------------------------------
# Manual x positions (columns) so the diagram reads left→right cleanly.
# Plotly Sankey honors x/y when arrangement="snap" or "fixed".
# ---------------------------------------------------------------------------
def _col_x(group: str) -> float:
    return {"src": 0.005, "mid": 0.42, "model": 0.995}[group]


x_pos = [_col_x(n[2]) for n in NODES]

# y positions: spread within column
per_col: dict[str, list[int]] = defaultdict(list)
for i, (_lbl, _hover, grp, _c) in enumerate(NODES):
    per_col[grp].append(i)
y_pos: list[float] = [0.0] * len(NODES)
for idxs in per_col.values():
    n = len(idxs)
    for j, i in enumerate(idxs):
        # nudge into [0.02, 0.98] band to avoid clipping
        y_pos[i] = 0.02 + (j + 0.5) * (0.96 / n)

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig = go.Figure(
    data=[
        go.Sankey(
            arrangement="snap",
            node={
                "label": LABELS,
                "customdata": HOVER,
                "color": NODE_COLORS,
                "pad": 22,
                "thickness": 22,
                "line": {"color": "#222", "width": 0.8},
                "x": x_pos,
                "y": y_pos,
                "hovertemplate": "<b>%{label}</b><br>%{customdata}<extra></extra>",
            },
            link={
                "source": src_idx,
                "target": tgt_idx,
                "value": vis_val,
                "customdata": custom,
                "label": labels,
                "color": ecolors,
                "hovertemplate": (
                    "<b>%{source.label} → %{target.label}</b><br>"
                    "%{label}<br>"
                    "actual: %{customdata[0]:,} h"
                    "<extra></extra>"
                ),
            },
        )
    ]
)

# ---------------------------------------------------------------------------
# Legend (Plotly Sankey has no built-in legend; use HTML in title block).
# ---------------------------------------------------------------------------
LEGEND_HTML = (
    "<span style='font-size:12px;color:#666'>"
    "Legend (license):  "
    "<span style='color:#3a8540'>■ commercial OK</span>  "
    "<span style='color:#c64a3b'>■ NC ⚠</span>  "
    "<span style='color:#cba135'>■ planned</span>  "
    "<span style='color:#2f6fb7'>■ base ckpt</span>  "
    "<span style='color:#888888'>■ other / undecided</span>"
    "</span>"
)

fig.update_layout(
    title={
        "text": (
            "<b style='font-size:18px'>0162 LLM-jp-Moshi data flow</b>"
            "<br><span style='font-size:12px;color:#444'>"
            "Sources → Intermediates → Models. "
            "<b>Edge width = √(audio hours)</b> so small flows stay visible alongside "
            "J-CHAT (72k h). Hover any node or edge for exact numbers. "
            "Snapshot: 2026-05-24."
            "</span><br>" + LEGEND_HTML
        ),
        "x": 0.5,
        "xanchor": "center",
        "y": 0.97,
        "yanchor": "top",
    },
    font={"family": "Inter, -apple-system, system-ui, sans-serif", "size": 13},
    paper_bgcolor="#fafafa",
    margin={"l": 10, "r": 10, "t": 130, "b": 20},
    height=1200,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="output/sankey_overview/index.html",
        help="Output HTML path",
    )
    args = ap.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        out,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displaylogo": False},
    )
    print(f"[done] wrote {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
