#!/usr/bin/env python3
"""Persona/style-conditioned 2-speaker Japanese dialogue script generator (causal path).

This is the text-generation step of the PersonaPlex-style control PoC. Unlike the
existing 0386 `gen_dialogue_scripts.py` (whose few-shot is uniformly polite, so the
output collapses to ~uniform teineigo -> useless for *style control*), this script
emits dialogues whose speaker-A style is **driven by an explicit persona axis** and
records the ground-truth persona, so the downstream model can be taught to follow a
text prompt.

v0 control axis = formality (polite=teineigo / casual=tameguchi). The KEY mechanism
is a *style-matched few-shot*: a polite exemplar for polite requests, a tameguchi
exemplar for casual requests. Few-shot style dominates LLM output style, which is why
the polite-only exemplar previously washed out all casual requests.

Output JSON per dialogue is a superset of the 0386 schema (so 0386
`scripts/infer_batch.py` can synthesize it unchanged):
  {"topic","topic_label","s1_name","s2_name","personas":{"S1","S2"},
   "style": "polite"|"casual",                 # requested style
   "style_measured": "polite"|"casual"|"mixed",# classify_formality on S1 turns
   "persona_json": "<Persona(formality=style) as json>",  # for build_persona_parquet --mode passthrough
   "turns": [{"speaker":"S1"/"S2","text"}]}

Persona is attached to **speaker A == S1** (matches build_persona_parquet's convention:
bootstrap measures A[0], voice-prompt takes A's audio, channel L=A).

Run (uv; transformers + torch needed -> use the 0386 FireRedTTS2 venv or deps_textgen):
  uv run --no-project --python 3.12 \
      --with torch --with transformers --with accelerate --with sentencepiece \
      python mstts/data_prep/gen_persona_dialogues.py \
      --out_dir /groups/gcg51557/experiments/0386_dialogue_model/data/dialogue_scripts/persona_gen \
      --n_per_style 2 --styles polite,casual
"""
import argparse
import json
import os
import re
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from persona_schema import Persona, classify_formality  # noqa: E402

TOPICS = {
    "hobby": "最近ハマっている趣味",
    "weekend": "週末の過ごし方",
    "food": "好きな食べ物やお店",
    "media": "最近見た映画・ドラマ・動画",
    "place": "最近行った場所の話",
}

SURNAMES = """佐藤 鈴木 高橋 田中 伊藤 渡辺 山本 中村 小林 加藤 吉田 山田 佐々木 山口 松本
井上 木村 林 斎藤 清水 山崎 阿部 森 池田 橋本 山下 石川 中島 前田 藤田 後藤 小川 岡田
村上 長谷川 近藤 石井 斉藤 坂本 遠藤 藤井 青木 福田 三浦 西村 藤原 太田 松田 原田 岡本
中川 中野 原 田村 竹内 金子 和田 中山 石田 上田 森田 小野 柴田 原口 宮崎 酒井 工藤 横山
宮本 内田 高木 安藤 島田 谷口 大野 高田 丸山 今井 河野 藤本 村田 武田 上野 杉山 増田 小島
平野 大塚 千葉 久保 松井 岩崎 桜井 野口 松尾 野村 木下 菊地 佐野 大西 杉本 新井 浜田 菅原
市川 水野 小松 大橋 西田 菅野 山内 鈴江 浅野 川口 平田 関 五十嵐 柳沢 望月 星野""".split()

PERSONAS = [
    "インドア派で落ち着いた感じ", "明るくてよく笑う", "ちょっと人見知り", "話好きでテンポが速い",
    "のんびりマイペース", "好奇心旺盛", "聞き上手で相槌が多い", "おっとりした雰囲気",
    "アクティブで行動的", "理屈っぽいけど親しみやすい",
]

# Polite exemplar: first meeting, teineigo (です/ます).
FEWSHOT_POLITE = """A: あ、はじめまして。今日はよろしくお願いします。
B: はじめまして。こちらこそ、よろしくお願いします。
A: えっと、じゃあ早速なんですけど、最近何かハマってることってありますか？
B: あー、そうですね、最近はちょっとサウナにハマってて。
A: あ、サウナいいですね。
B: そうなんですよ。週一くらいで通ってて、なんかもう整うっていうか。
A: あー、わかります、あの感覚。
B: そうそう。田中さんは何かありますか？
A: 僕は最近、家で植物を育てるのにハマってて。
B: えー、いいですね。何を育ててるんですか？"""

# Casual exemplar: close friends catching up, tameguchi (だよ/じゃん/〜なんだ, no です/ます).
FEWSHOT_CASUAL = """A: あ、ひさしぶり！元気にしてた？
B: おー、ひさしぶり。まあぼちぼちかな。そっちは？
A: うちも変わんないよ。あ、そういえば最近なんかハマってることある？
B: あー、最近サウナにめっちゃハマっててさ。
A: えっ、サウナ？いいじゃん。
B: そうなんだよ、週一くらいで行っててさ、もう整うって感じ。
A: わかるわー、あれな。
B: そうそう。そっちは何かあるの？
A: 俺は最近、家で植物育てんのにハマっててさ。
B: えー、いいね。何育ててんの？"""

STYLE_CFG = {
    "polite": {
        "fewshot": FEWSHOT_POLITE,
        "frame": "初対面の2人",
        "open": "・冒頭は『はじめまして』から始め、自己紹介で名前を名乗る。相手の名前は時々呼びかける程度（毎回は不要）。",
        "tone": "・丁寧な話し言葉（です・ます調）で。敬語を基本に。相槌（あー、はい、そうなんですね 等）や軽い言い淀み（えっと、なんか）を自然に。",
    },
    "casual": {
        "fewshot": FEWSHOT_CASUAL,
        "frame": "親しい友達同士の2人",
        "open": "・旧友が久しぶりに会った雰囲気で始める。自己紹介で名前を名乗る（自然に）。相手の名前は時々呼びかける程度。",
        "tone": "・くだけたタメ口（だ・だよ・じゃん・〜なんだ・〜してる 調）で。敬語は使わない。相槌（あー、うん、そうそう 等）や軽い言い淀み（えっと、なんか）を自然に。",
    },
}

SENT_END = tuple("。．！？!?」』）)…ー〜")


def build_prompt(style, topic_label, s1_name, s2_name, p1, p2):
    cfg = STYLE_CFG[style]
    return (
        f"日本語で、{cfg['frame']}（AとB）の自然な話し言葉の雑談を作ってください。\n"
        f"・話題は「{topic_label}」。\n"
        f"・Aは「{s1_name}」さん（{p1}）、Bは「{s2_name}」さん（{p2}）。\n"
        f"{cfg['open']}\n"
        "・話を振る時は『相手』の名前を呼ぶ（自分の名前は呼ばない）。AとBを取り違えないこと。\n"
        f"{cfg['tone']}\n"
        "・14〜16発話くらい。\n"
        "・下の例と同じ題材（サウナ・観葉植物 等）は使わず、毎回違う具体的な内容にする。\n"
        "・出力は会話だけ。説明や見出しは書かない。次の形式を厳守:\n"
        "A: 〜\nB: 〜\n\n"
        "（例）\n" + cfg["fewshot"] + f"\n\n（ここから本番。話題:「{topic_label}」）\n"
    )


def parse_dialogue(text):
    text = re.sub(r"^\s*(?:analysis|final|assistant)\b[\s:：]*", "", text)
    turns = []
    for line in text.splitlines():
        m = re.match(r"^\s*([ABＡＢ])\s*[:：]\s*(.+?)\s*$", line)
        if not m:
            continue
        spk = "S1" if m.group(1) in "AＡ" else "S2"
        t = m.group(2).strip()
        if t:
            turns.append({"speaker": spk, "text": t})
    if turns and not turns[-1]["text"].rstrip().endswith(SENT_END):
        turns = turns[:-1]
    return turns


def s1_text(turns):
    return " ".join(t["text"] for t in turns if t["speaker"] == "S1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llm-jp/llm-jp-4-8b-instruct")
    ap.add_argument("--out_dir", required=True, help="dialogue script output dir (0386 infer_batch reads this)")
    ap.add_argument("--styles", default="polite,casual", help="comma-separated: polite,casual")
    ap.add_argument("--n_per_style", type=int, default=2, help="dialogues per (style x topic)")
    ap.add_argument("--min_turns", type=int, default=10)
    ap.add_argument("--max_turns", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topics", default="", help="comma-separated topic keys (empty=all)")
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    assert all(s in STYLE_CFG for s in styles), f"bad styles {styles}; choices {list(STYLE_CFG)}"
    topics = dict(TOPICS)
    if args.topics:
        want = [t.strip() for t in args.topics.split(",") if t.strip()]
        topics = {k: TOPICS[k] for k in want if k in TOPICS}
        assert topics, f"no valid topics in {args.topics!r} (choices: {list(TOPICS)})"

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"[INFO] loaded {args.model}", flush=True)

    # confusion[requested_style][measured] -> count  (the de-risk signal)
    confusion = {s: {"polite": 0, "casual": 0, "mixed": 0} for s in styles}
    n_ok = 0
    for style in styles:
        for topic, label in topics.items():
            made = 0
            attempts = 0
            while made < args.n_per_style and attempts < args.n_per_style * 4:
                attempts += 1
                s1, s2 = rng.sample(SURNAMES, 2)
                p1, p2 = rng.sample(PERSONAS, 2)
                prompt = build_prompt(style, label, s1, s2, p1, p2)
                msgs = [{"role": "user", "content": prompt}]
                inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                 return_tensors="pt", return_dict=True).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=700, do_sample=True,
                                         temperature=args.temperature, top_p=0.95,
                                         pad_token_id=tok.eos_token_id)
                gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                turns = parse_dialogue(gen)
                if os.environ.get("GEN_DEBUG"):
                    print(f"[debug {style}/{topic}] turns={len(turns)} raw={gen[:300]!r}", flush=True)
                if len(turns) < args.min_turns:
                    continue
                turns = turns[: args.max_turns]
                measured = classify_formality(s1_text(turns))
                confusion[style][measured] += 1
                persona = Persona(formality=style)
                obj = {"topic": topic, "topic_label": label, "s1_name": s1, "s2_name": s2,
                       "personas": {"S1": p1, "S2": p2}, "style": style,
                       "style_measured": measured,
                       "persona_json": json.dumps(persona.to_dict(), ensure_ascii=False),
                       "turns": turns}
                fname = f"{style}_{topic}_{made:04d}.json"
                with open(os.path.join(args.out_dir, fname), "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=1)
                made += 1
                n_ok += 1
                print(f"  [{style}/{topic}] {made-1} ({len(turns)}turns, {s1}/{s2}) measured={measured}", flush=True)

    print(f"\n[done] generated {n_ok} scripts -> {args.out_dir}")
    print("=== formality confusion (requested -> measured on speaker A) ===")
    for style in styles:
        row = confusion[style]
        tot = sum(row.values()) or 1
        print(f"  requested {style:7s}: polite={row['polite']:3d} casual={row['casual']:3d} "
              f"mixed={row['mixed']:3d}  (match {100*row.get(style,0)/tot:.0f}%)")


if __name__ == "__main__":
    main()
