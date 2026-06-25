"""ASR the model's answer (channel A = left) and tabulate vs the question.

For each pilot question we show: the question text, the model's A-text-track
(what it 'said' it would say, decoded from row 0 of the generated tokens), and
the faster-whisper transcript of channel A (what it actually produced). Lets a
human judge whether the casual dialogue model answers or just chats.
"""
from __future__ import annotations

import argparse
import html
import json
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()


def main(a: argparse.Namespace) -> None:
    from faster_whisper import WhisperModel
    from transformers import AutoTokenizer

    meta = {m["qid"]: m for m in json.load(open(Path(a.questions_dir) / "meta.json"))}
    tok = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")
    asr = WhisperModel("large-v3", device="cuda", compute_type="float16")

    gen = Path(a.gen_dir)
    tokens_dir = gen / "generated_tokens"
    wav_dir = gen / "generated_wavs"

    # example_id -> qid: generate.py numbers examples 0..N in parquet order,
    # which is meta.json order.
    order = [m["qid"] for m in json.load(open(Path(a.questions_dir) / "meta.json"))]

    rows = []
    for npy in sorted(tokens_dir.glob("*.npy"), key=lambda p: int(p.stem)):
        ex = int(npy.stem)
        qid = order[ex] if ex < len(order) else str(ex)
        toks = np.load(npy)                       # [17, P+G] undelayed
        # row 0 = A text track; keep only real tokens (drop pad 3 / 0, sentinels)
        text = toks[0]
        keep = [int(x) for x in text if int(x) not in (0, 3) and int(x) < tok.vocab_size]
        a_text = norm(tok.decode(keep)) if keep else ""

        wav = wav_dir / f"{ex}.wav"
        arr, sr = sf.read(str(wav), always_2d=True)
        chA = arr[:, 0]                           # L = A = answer
        # ASR only the generated (post-prompt) region of A: prompt is silence
        # on A anyway, so transcribing the whole channel is fine.
        segs, _ = asr.transcribe(chA.astype("float32"), language="ja", beam_size=5)
        a_asr = norm("".join(s.text for s in segs))

        rows.append({
            "qid": qid,
            "question": meta.get(qid, {}).get("instruction", ""),
            "ref_response_head": meta.get(qid, {}).get("response", "")[:120],
            "a_text_track": a_text,
            "a_asr": a_asr,
        })

    Path(a.output_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    cells = []
    for r in rows:
        cells.append(
            "<tr>"
            f"<td class=q>{html.escape(r['qid'])}<br><b>Q:</b> {html.escape(r['question'])}"
            f"<br><span class=ref><b>ref ans:</b> {html.escape(r['ref_response_head'])}…</span></td>"
            f"<td><b>A text-track:</b> {html.escape(r['a_text_track']) or '<i>(empty)</i>'}<br>"
            f"<b>A ASR:</b> {html.escape(r['a_asr']) or '<i>(empty)</i>'}</td>"
            "</tr>"
        )
    doc = (
        "<!DOCTYPE html><meta charset=utf-8><style>"
        "body{font-family:system-ui;margin:24px;font-size:13px}"
        "table{border-collapse:collapse;width:100%}"
        "td{border:1px solid #ddd;padding:8px;vertical-align:top}"
        ".q{width:42%;background:#fafafa}.ref{color:#888;font-size:11px}"
        "</style><h2>spoken-QA pilot — v1 answers (channel A)</h2>"
        "<p>Does the Zoom1 casual-dialogue model answer the spoken question, "
        "or just backchannel? Left = question + reference answer; right = the "
        "model's own text track and the ASR of its generated A-channel audio.</p>"
        "<table>" + "".join(cells) + "</table>"
    )
    Path(a.output_html).write_text(doc, encoding="utf-8")
    print(f"wrote {a.output_json} and {a.output_html} ({len(rows)} questions)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-dir", required=True)
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-html", required=True)
    main(ap.parse_args())
