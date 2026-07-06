#!/usr/bin/env python3
"""Diverse 2-speaker Japanese dialogue generator for the ~100h FireRed corpus.

Superset of gen_persona_dialogues.py, built for scale + diversity so the synthetic
corpus serves BOTH:
  - standard SFT (v1 line): general conversational ability, incl. how to OPEN a
    conversation (greeting) and how to CONTINUE one already in progress (midconv);
  - PersonaPlex prompt-control: persona_json label per dialogue (formality axis).

Diversity axes (a "cell" = topic x style x opening; ~24*2*2 = 96 cells):
  - topic     : 24 everyday chat topics
  - style     : polite (teineigo) | casual (tameguchi)   [style-matched few-shot]
  - opening   : greeting (はじめまして/久しぶり + 自己紹介) | midconv (会話の途中から、挨拶なし)
  - persona   : sampled trait per speaker (adds surface variety)

Output JSON (0386 infer_batch-compatible superset):
  {"topic","topic_label","style","opening","style_measured","s1_name","s2_name",
   "personas":{"S1","S2"}, "persona_json": "<Persona(formality=style) json>",
   "turns":[{"speaker":"S1"/"S2","text"}]}

Run (0386 FireRedTTS2 venv has torch+transformers; llm-jp-4-8b confirmed best generator):
  python gen_dialogues_diverse.py --model llm-jp/llm-jp-4-8b-instruct \
      --out_dir <dir> --n_per_cell 42 --topics <csv|all> --styles polite,casual \
      --openings greeting,midconv --seed 0
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
    "hobby": "最近ハマっている趣味", "weekend": "週末の過ごし方", "food": "好きな食べ物やお店",
    "media": "最近見た映画・ドラマ・動画", "place": "最近行った場所", "travel": "行ってみたい旅行先",
    "work": "仕事や勉強の近況", "health": "健康や運動のこと", "pet": "ペットや動物の話",
    "family": "家族や実家の話", "sports": "好きなスポーツや観戦", "music": "好きな音楽やライブ",
    "study": "最近学んでいること", "season": "季節や天気の話", "hometown": "出身地や地元の話",
    "future": "将来やってみたいこと", "shopping": "最近買ったもの", "cooking": "料理や自炊",
    "game": "好きなゲーム", "tech": "気になるガジェットやアプリ", "money": "節約やお金の使い方",
    "book": "最近読んだ本や漫画", "cafe": "お気に入りのカフェや喫茶店", "event": "最近行った/行きたいイベント",
}

SURNAMES = """佐藤 鈴木 高橋 田中 伊藤 渡辺 山本 中村 小林 加藤 吉田 山田 佐々木 山口 松本
井上 木村 林 斎藤 清水 山崎 阿部 森 池田 橋本 山下 石川 中島 前田 藤田 後藤 小川 岡田
村上 長谷川 近藤 石井 斉藤 坂本 遠藤 藤井 青木 福田 三浦 西村 藤原 太田 松田 原田 岡本
中川 中野 原 田村 竹内 金子 和田 中山 石田 上田 森田 小野 柴田 原口 宮崎 酒井 工藤 横山
宮本 内田 高木 安藤 島田 谷口 大野 高田 丸山 今井 河野 藤本 村田 武田 上野 杉山 増田 小島
平野 大塚 千葉 久保 松井 岩崎 桜井 野口 松尾 野村 木下 菊地 佐野 大西 杉本 新井 浜田 菅原""".split()

PERSONAS = [
    "インドア派で落ち着いた感じ", "明るくてよく笑う", "ちょっと人見知り", "話好きでテンポが速い",
    "のんびりマイペース", "好奇心旺盛", "聞き上手で相槌が多い", "おっとりした雰囲気",
    "アクティブで行動的", "理屈っぽいけど親しみやすい", "毒舌だけど優しい", "心配性で慎重",
    "楽観的でおおらか", "真面目でしっかり者", "天然で憎めない", "クールで大人っぽい",
]

# few-shots: [style][opening]. greeting = 会話の頭から / midconv = 途中から(挨拶なし).
FEWSHOTS = {
    ("polite", "greeting"): """A: あ、はじめまして。今日はよろしくお願いします。
B: はじめまして。こちらこそ、よろしくお願いします。
A: えっと、じゃあ早速なんですけど、最近何かハマってることってありますか？
B: あー、そうですね、最近はちょっとサウナにハマってて。
A: あ、サウナいいですね。
B: そうなんですよ。週一くらいで通ってて、なんかもう整うっていうか。
A: あー、わかります、あの感覚。
B: そうそう。田中さんは何かありますか？""",
    ("polite", "midconv"): """A: いや、それで結局どうなったんですか？
B: あー、結局その日は雨で中止になっちゃって。
A: えー、せっかくだったのに残念でしたね。
B: そうなんですよ。でもまた来月リベンジしようと思ってて。
A: いいですね。今度は晴れるといいですね。
B: ほんとそれです。山口さんは最近どこか行かれました？""",
    ("casual", "greeting"): """A: あ、ひさしぶり！元気にしてた？
B: おー、ひさしぶり。まあぼちぼちかな。そっちは？
A: うちも変わんないよ。あ、そういえば最近なんかハマってることある？
B: あー、最近サウナにめっちゃハマっててさ。
A: えっ、サウナ？いいじゃん。
B: そうなんだよ、週一くらいで行っててさ、もう整うって感じ。
A: わかるわー、あれな。
B: そうそう。そっちは何かあるの？""",
    ("casual", "midconv"): """A: で、その後どうなったんだよ？
B: いやー、結局その日雨降ってさ、中止になっちゃって。
A: えー、まじかよ。せっかくだったのにな。
B: な。でも来月リベンジしようと思っててさ。
A: いいじゃん。今度は晴れるといいな。
B: ほんとそれ。そっちは最近どっか行った？""",
}

STYLE_TONE = {
    "polite": "・丁寧な話し言葉（です・ます調）で。敬語を基本に。相槌（あー、はい、そうなんですね 等）や軽い言い淀み（えっと、なんか）を自然に。",
    "casual": "・くだけたタメ口（だ・だよ・じゃん・〜なんだ・〜してる 調）で。敬語は使わない。相槌（あー、うん、そうそう 等）や軽い言い淀み（えっと、なんか）を自然に。",
}
OPENING_INSTR = {
    "greeting": "・会話の冒頭から始める（{frame}）。自己紹介で名前を名乗り、相手の名前は時々呼びかける程度。",
    "midconv": "・すでに進行中の会話の途中から始める。挨拶・自己紹介はしない。いきなり話題の続きに入る。相手の名前は時々呼びかける程度。",
}
FRAME = {"polite": "初対面か顔見知り", "casual": "親しい友人同士"}

SENT_END = tuple("。．！？!?」』）)…ー〜")


def build_prompt(style, opening, topic_label, s1, s2, p1, p2):
    fs = FEWSHOTS[(style, opening)]
    open_line = OPENING_INSTR[opening].format(frame=FRAME[style])
    return (
        f"日本語で、2人（AとB）の自然な話し言葉の雑談を作ってください。\n"
        f"・話題は「{topic_label}」。\n"
        f"・Aは「{s1}」さん（{p1}）、Bは「{s2}」さん（{p2}）。\n"
        f"{open_line}\n"
        "・話を振る時は『相手』の名前を呼ぶ（自分の名前は呼ばない）。AとBを取り違えないこと。\n"
        f"{STYLE_TONE[style]}\n"
        "・14〜16発話くらい。下の例と同じ具体例（サウナ 等）は使わず、毎回違う内容にする。\n"
        "・出力は会話だけ。説明や見出しは書かない。次の形式を厳守:\nA: 〜\nB: 〜\n\n"
        "（例）\n" + fs + f"\n\n（ここから本番。話題:「{topic_label}」）\n"
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
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--styles", default="polite,casual")
    ap.add_argument("--openings", default="greeting,midconv")
    ap.add_argument("--topics", default="all")
    ap.add_argument("--n_per_cell", type=int, default=42, help="dialogues per (style x opening x topic)")
    ap.add_argument("--min_turns", type=int, default=10)
    ap.add_argument("--max_turns", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.85)
    args = ap.parse_args()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    openings = [o.strip() for o in args.openings.split(",") if o.strip()]
    topics = dict(TOPICS) if args.topics == "all" else {
        k: TOPICS[k] for k in (t.strip() for t in args.topics.split(",")) if k in TOPICS}
    assert topics and styles and openings
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"[INFO] loaded {args.model}", flush=True)

    from collections import Counter
    conf = Counter()
    n_ok = 0
    for style in styles:
        for opening in openings:
            for topic, label in topics.items():
                made = attempts = 0
                while made < args.n_per_cell and attempts < args.n_per_cell * 4:
                    attempts += 1
                    s1, s2 = rng.sample(SURNAMES, 2)
                    p1, p2 = rng.sample(PERSONAS, 2)
                    prompt = build_prompt(style, opening, label, s1, s2, p1, p2)
                    inp = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                  add_generation_prompt=True, return_tensors="pt",
                                                  return_dict=True).to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inp, max_new_tokens=700, do_sample=True,
                                             temperature=args.temperature, top_p=0.95,
                                             pad_token_id=tok.eos_token_id)
                    gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                    turns = parse_dialogue(gen)
                    if len(turns) < args.min_turns:
                        continue
                    turns = turns[: args.max_turns]
                    measured = classify_formality(s1_text(turns))
                    conf[(style, measured)] += 1
                    persona = Persona(formality=style)
                    obj = {"topic": topic, "topic_label": label, "style": style, "opening": opening,
                           "style_measured": measured, "s1_name": s1, "s2_name": s2,
                           "personas": {"S1": p1, "S2": p2},
                           "persona_json": json.dumps(persona.to_dict(), ensure_ascii=False),
                           "turns": turns}
                    fn = f"{style}_{opening}_{topic}_{made:04d}.json"
                    with open(os.path.join(args.out_dir, fn), "w", encoding="utf-8") as f:
                        json.dump(obj, f, ensure_ascii=False, indent=1)
                    made += 1; n_ok += 1
                print(f"  [{style}/{opening}/{topic}] {made} made", flush=True)

    print(f"\n[done] {n_ok} dialogues -> {args.out_dir}")
    for style in styles:
        tot = sum(v for (s, _), v in conf.items() if s == style) or 1
        print(f"  {style:7s} formality-match {100*conf.get((style, style), 0)/tot:.0f}% "
              f"({dict((m, v) for (s, m), v in conf.items() if s == style)})")


if __name__ == "__main__":
    main()
