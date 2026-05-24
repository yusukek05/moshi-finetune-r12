"""Build a Sankey diagram of the 0162 data flow.

Sources (raw corpora & base models) → Intermediates (filtered / tokenized /
synth corpora & ckpt families) → Models (released or planned).

Edge widths are audio hours where known; estimates marked "~". Node colors
encode commercial usability: green=OK, red=NC-blocked, gray=undecided/info.
Output is self-contained HTML (Plotly bundled via CDN).
"""

from __future__ import annotations

import argparse
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
# Node table: (label, group, color)
# Group is informational only (used for column layout).
# ---------------------------------------------------------------------------
NODES = [
    # SOURCES (raw corpora & base models)
    ("Kyutai Moshiko (base) — CC-BY-4.0", "src", C_BASE),  # 0
    ("Kyutai Moshika (base) — CC-BY-4.0", "src", C_BASE),  # 1
    ("ReazonSpeech — CC-BY-4.0", "src", C_OK),  # 2
    ("J-CHAT — commercial OK", "src", C_OK),  # 3
    ("LaboroTV — NC ⚠", "src", C_NC),  # 4
    ("Zoom1 (LLM-jp) — 935h train (CC-BY+LLM-jp)", "src", C_OK),  # 5
    ("VisualBank — 300h", "src", C_UNK),  # 6
    ("ccaudio raw_all — 23,685h (CC)", "src", C_OK),  # 7
    ("JMultiWOZ — 7,469 chunks (CC-BY-SA)", "src", C_OK),  # 8
    ("RealPersonaChat — 38,797 chunks (CC-BY-SA)", "src", C_OK),  # 9
    ("pseudo_dialog_kanzaki — private synth", "src", C_UNK),  # 10
    # INTERMEDIATES
    ("0178 mono ckpt (moshika+Reazon+J-CHAT+LaboroTV)", "mid", C_NC),  # 11
    ("0178 mstts ckpt (+J-CHAT multi-stream)", "mid", C_NC),  # 12
    ("ccaudio v2 filtered — 2,232h kept (re-VAD+re-ASR)", "mid", C_OK),  # 13
    ("mstts text inputs (JMultiWOZ + RPC merged)", "mid", C_OK),  # 14
    ("mstts v0c synth wavs — 46,266 / 530h", "mid", C_NC),  # 15
    ("mstts v0c synth → v1 parquet (357MB)", "mid", C_NC),  # 16
    ("commercial mstts synth corpus (planned)", "mid", C_PLAN),  # 17
    # MODELS — v1 line
    ("v1 (J-CHAT→Zoom1) — public", "model", C_OK),  # 18
    ("v1.1 (+ReazonSpeech) — ckpt ready", "model", C_OK),  # 19
    ("v1.2 (+VisualBank) — ckpt, MOS pending", "model", C_OK),  # 20
    ("v1.3 (+commercial synth) — planned", "model", C_PLAN),  # 21
    ("v1.1b/c/d/e — synth ablation (private)", "model", C_UNK),  # 22
    # MODELS — mstts line
    ("v0a (moshiko + J-CHAT 1shard) — toy", "model", C_UNK),  # 23
    ("v0b (0178 mstts + Zoom1, +500)", "model", C_NC),  # 24
    ("v0c (v0b + Zoom1, +1500) — production mstts", "model", C_NC),  # 25
    ("v0d (LaboroTV-free attempt) — failed", "model", C_NC),  # 26
    ("v0d_v2 (ccaudio v2 rebuild) — planned", "model", C_PLAN),  # 27
]

LABELS = [n[0] for n in NODES]
NODE_COLORS = [n[2] for n in NODES]

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
    ("reazon", "mono0178", 300, "~300h", C_OK),
    ("jchat", "mono0178", 300, "~300h", C_OK),
    ("laboro", "mono0178", 300, "~300h ⚠NC", C_NC),
    # === 0178 mstts (mono + J-CHAT multi-stream) ===
    ("mono0178", "mstts0178", 900, "mono ckpt", C_NC),
    ("jchat", "mstts0178", 200, "+J-CHAT mstts", C_OK),
    # === v0a-c training ===
    ("moshiko", "v0a", 1, "base", C_BASE),
    ("jchat", "v0a", 50, "J-CHAT 1shard", C_OK),
    ("mstts0178", "v0b", 1100, "init ckpt", C_NC),
    ("zoom1", "v0b", 935, "Zoom1 +500", C_OK),
    ("v0b", "v0c", 1200, "v0b ckpt", C_NC),
    ("zoom1", "v0c", 935, "Zoom1 +1500", C_OK),
    # === v0d (failed): moshika + Reazon + J-CHAT + ccaudio v1 → mstts → Zoom1 ===
    ("moshika", "v0d", 1, "base", C_BASE),
    ("reazon", "v0d", 300, "Stage 1", C_OK),
    ("jchat", "v0d", 300, "Stage 1/2", C_OK),
    ("cc_raw", "v0d", 6613, "ccaudio v1 transcribed (broken filter)", C_OK),
    ("zoom1", "v0d", 935, "Stage 3", C_OK),
    # === ccaudio v2 reprocess (CURRENT WORK) ===
    ("cc_raw", "cc_v2", 23685, "raw → VAD re-seg + re-ASR (filter ~9.4% kept)", C_OK),
    # === v0d_v2 (planned) ===
    ("moshika", "v0d_v2", 1, "base", C_BASE),
    ("reazon", "v0d_v2", 300, "Stage 1", C_OK),
    ("jchat", "v0d_v2", 300, "Stage 1/2", C_OK),
    ("cc_v2", "v0d_v2", 2232, "ccaudio v2 (2,232h)", C_OK),
    ("zoom1", "v0d_v2", 935, "Stage 3", C_OK),
    # === mstts text inputs → synth corpus generation ===
    ("jmw", "mstts_text", 60, "~60h text chunks", C_OK),
    ("rpc", "mstts_text", 320, "~320h text chunks", C_OK),
    ("mstts_text", "synth_wav", 380, "text → v0c inference", C_OK),
    ("v0c", "synth_wav", 380, "v0c synth engine ⚠NC", C_NC),
    ("synth_wav", "synth_parquet", 530, "wavs → v1 parquet", C_NC),
    # === v1 line ===
    ("moshiko", "v1", 1, "base", C_BASE),
    ("jchat", "v1", 300, "Stage 1", C_OK),
    ("zoom1", "v1", 935, "Stage 2", C_OK),
    ("moshiko", "v1_1", 1, "base", C_BASE),
    ("reazon", "v1_1", 300, "+ReazonSpeech", C_OK),
    ("jchat", "v1_1", 300, "Stage 2", C_OK),
    ("zoom1", "v1_1", 935, "Stage 3", C_OK),
    ("moshiko", "v1_2", 1, "base", C_BASE),
    ("reazon", "v1_2", 300, "Stage 1", C_OK),
    ("jchat", "v1_2", 300, "Stage 2", C_OK),
    ("vb", "v1_2", 300, "+VisualBank", C_UNK),
    ("zoom1", "v1_2", 935, "Stage 4", C_OK),
    # === v1.3 (planned, needs commercial mstts) ===
    ("v0d_v2", "synth_commercial", 380, "v0d_v2 as synth engine", C_PLAN),
    ("mstts_text", "synth_commercial", 380, "same text inputs", C_OK),
    ("moshiko", "v1_3", 1, "base", C_BASE),
    ("reazon", "v1_3", 300, "Stage 1", C_OK),
    ("jchat", "v1_3", 300, "Stage 2", C_OK),
    ("vb", "v1_3", 300, "Stage 3", C_UNK),
    ("synth_commercial", "v1_3", 530, "+commercial synth", C_PLAN),
    ("zoom1", "v1_3", 935, "Stage 5", C_OK),
    # === v1.1 ablation cluster (experimental, private) ===
    ("moshiko", "v1_ablation", 1, "base", C_BASE),
    ("jchat", "v1_ablation", 300, "J-CHAT", C_OK),
    ("zoom1", "v1_ablation", 935, "Zoom1", C_OK),
    ("vb", "v1_ablation", 100, "VB subsets", C_UNK),
    ("pseudo", "v1_ablation", 50, "神崎 synth", C_UNK),
]

# Resolve symbolic edges to numeric indices.
src_idx = [IDX[e[0]] for e in EDGES]
tgt_idx = [IDX[e[1]] for e in EDGES]
val = [e[2] for e in EDGES]
labels = [e[3] for e in EDGES]


def _to_rgba(hex6: str, alpha: float = 0.45) -> str:
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


ecolors = [_to_rgba(e[4]) for e in EDGES]


# ---------------------------------------------------------------------------
# Manual x positions (columns) so the diagram reads left→right cleanly.
# Plotly Sankey honors x/y when arrangement="snap" or "fixed".
# ---------------------------------------------------------------------------
def _col_x(group: str) -> float:
    return {"src": 0.01, "mid": 0.40, "model": 0.99}[group]


x_pos = [_col_x(n[1]) for n in NODES]

# y positions: spread within column
per_col: dict[str, list[int]] = defaultdict(list)
for i, (_lbl, grp, _c) in enumerate(NODES):
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
                "color": NODE_COLORS,
                "pad": 14,
                "thickness": 18,
                "line": {"color": "#333", "width": 0.6},
                "x": x_pos,
                "y": y_pos,
            },
            link={
                "source": src_idx,
                "target": tgt_idx,
                "value": val,
                "label": labels,
                "color": ecolors,
                "hovertemplate": "%{source.label} → %{target.label}<br>%{label}<br>weight: %{value}<extra></extra>",
            },
        )
    ]
)

fig.update_layout(
    title={
        "text": (
            "<b>0162 LLM-jp-Moshi data flow</b>"
            "<br><span style='font-size:13px'>Sources → Intermediates → Models. "
            "Color = license (green=commercial OK, red=NC-blocked, gold=planned, "
            "blue=base ckpt, gray=other). Hover edges for details. "
            "Snapshot: 2026-05-24.</span>"
        ),
        "x": 0.5,
        "xanchor": "center",
    },
    font={"family": "Inter, -apple-system, system-ui, sans-serif", "size": 11},
    paper_bgcolor="#fafafa",
    margin={"l": 10, "r": 10, "t": 80, "b": 20},
    height=900,
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
