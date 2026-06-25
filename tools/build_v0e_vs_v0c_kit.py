"""Build a v0c vs v0e A/B listening page (the commercial-clean tradeoff).

v0c (NC, LaboroTV lineage, CER 36.1%) vs v0e (commercial-clean, no LaboroTV,
CER 54.7%) on the SAME 5 smoke prompts / seed. CER is an intelligibility proxy;
this page is to judge the perceptual "綺麗さ" (quality/prosody) gap by ear.

Reads matched staged wavs from output/v0d_eval_compare/{v0c,v0e}/generated_wavs/
and the dialogue prompts from mstts/inference_inputs/text_chat/.
"""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

ROOT = Path("/groups/gcg51557/experiments/0162_dialogue_model/moshi-finetune")
WAVS = ROOT / "output/v0d_eval_compare"
PROMPTS = ROOT / "mstts/inference_inputs/text_chat"
OUT = ROOT / "output/v0e_vs_v0c_kit"
N = 5
CER = {"v0c": 36.1, "v0e": 54.7}


def load_prompt(i: int) -> list[tuple[str, str]]:
    # prompt file name is 1-000{i}.json; staged wav is {i}.wav
    p = PROMPTS / f"1-{i:04d}.json"
    return json.loads(p.read_text())


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "v0c").mkdir(parents=True)
    (OUT / "v0e").mkdir(parents=True)

    rows = []
    for i in range(1, N + 1):
        for tag in ("v0c", "v0e"):
            src = WAVS / tag / "generated_wavs" / f"{i}.wav"
            shutil.copy(src, OUT / tag / f"{i}.wav")
        turns = load_prompt(i)
        dlg = "<br>".join(
            f"<b>[{html.escape(spk)}]</b> {html.escape(txt)}" for spk, txt in turns
        )
        rows.append(
            "<tr>"
            f"<td class=p><div class=id>prompt {i}</div>{dlg}</td>"
            f"<td><audio controls preload=none src='v0c/{i}.wav'></audio></td>"
            f"<td><audio controls preload=none src='v0e/{i}.wav'></audio></td>"
            "</tr>"
        )

    doc = f"""<!DOCTYPE html><html lang=ja><head><meta charset=utf-8>
<title>mstts v0c vs v0e 聴き比べ</title>
<style>
 body{{font-family:system-ui;margin:24px;color:#222;font-size:14px}}
 h1{{font-size:19px;margin:0 0 6px}}
 table{{border-collapse:collapse;width:100%;margin-top:12px}}
 th,td{{border:1px solid #ddd;padding:10px;vertical-align:top;text-align:left}}
 th{{background:#f4f4f4;position:sticky;top:0}}
 .p{{width:46%;font-size:13px;line-height:1.6}}
 .id{{font-family:monospace;color:#888;font-size:11px;margin-bottom:4px}}
 audio{{width:300px}}
 .note{{font-size:12px;color:#555;background:#f8f8f8;padding:10px 14px;border-left:3px solid #c0392b;line-height:1.7}}
</style></head><body>
<h1>multi-stream TTS: v0c vs v0e 聴き比べ（商用クリーン化のトレードオフ）</h1>
<div class=note>
 同一の対話台本5件・同一 seed で合成。<b>v0c</b> = 現行（LaboroTV 由来で <b>CC-BY-NC</b>、自己整合 CER <b>{CER['v0c']}%</b>）／
 <b>v0e</b> = 新・<b>商用クリーン</b>（LaboroTV を除き Reazon+J-CHAT のみ、ccaudio 不使用、CER <b>{CER['v0e']}%</b>）。<br>
 CER は明瞭度の代理指標で <b>v0e の方が高い（=不利）</b>が、知りたいのは<b>音質・韻律の「綺麗さ」が耳でどれだけ違うか</b>。
 stereo: L=話者A / R=話者B（24kHz Mimi）。左=v0c, 右=v0e。
</div>
<table>
<tr><th>対話台本（prompt）</th><th>v0c（NC, CER {CER['v0c']}%）</th><th>v0e（商用クリーン, CER {CER['v0e']}%）</th></tr>
{chr(10).join(rows)}
</table>
</body></html>"""
    (OUT / "index.html").write_text(doc, encoding="utf-8")
    nwav = len(list((OUT / "v0c").glob("*.wav"))) + len(list((OUT / "v0e").glob("*.wav")))
    size = sum(f.stat().st_size for f in OUT.rglob("*")) / 1e6
    print(f"wrote {OUT}/index.html — {N} prompts, {nwav} wavs, {size:.1f} MB")


if __name__ == "__main__":
    main()
