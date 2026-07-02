"""
PersonaPlex-style persona schema + control-axis helpers for the prompt-control PoC.

This module is intentionally dependency-light (numpy + sentencepiece) so it can be
reused both for the v0 "bootstrap from existing synth" path (measure an attribute
from the data, then use it as the prompt label) and for the future causal path
(sample a persona -> LLM-jp-3 generates a matching dialogue -> FireRedTTS-2 synth).

Design doc: docs/personaplex_poc_design.md
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

# ---- text-track decoding ---------------------------------------------------
# In the v1 token streams, row 0 (text) mixes real SP token ids (0..vocab-1)
# with special ids (text_padding=3, zero=0, speaker-BOS/initial at the high end).
# To recover readable text we keep only "content" SP ids.
_SP_SPECIAL_LOW = {0, 1, 2, 3}  # unk/bos/eos/pad-ish low reserved ids


def decode_text_track(text_row: np.ndarray, sp, min_id: int = 4) -> str:
    """Decode a single text-track row (token ids) to a string via the SP model,
    dropping special / out-of-vocab ids."""
    vocab = sp.get_piece_size()
    ids = [int(t) for t in np.asarray(text_row).ravel()
           if min_id <= int(t) < vocab and int(t) not in _SP_SPECIAL_LOW]
    if not ids:
        return ""
    return sp.decode(ids)


# ---- formality (the v0 control axis) --------------------------------------
# Polite (teineigo) vs casual (futsuukei) markers. Heuristic but robust enough
# to *measure a label* and to *evaluate whether output formality flips* later.
_POLITE_PAT = re.compile(
    r"(です|ます|ました|ません|でした|でしょう|ください|ございます|しています|してます)"
)
_CASUAL_PAT = re.compile(
    r"(だよ|だね|だな|じゃん|だろ|かな|よね|してる(?!ます)|なんだ|わ。|さ。|っす|だぜ|だわ)"
)


def formality_scores(text: str) -> tuple[int, int]:
    return len(_POLITE_PAT.findall(text)), len(_CASUAL_PAT.findall(text))


def classify_formality(text: str, margin: int = 2) -> str:
    """Return 'polite' | 'casual' | 'mixed' from decoded dialogue text."""
    p, c = formality_scores(text)
    if p >= c + margin:
        return "polite"
    if c >= p + margin:
        return "casual"
    return "mixed"


# ---- prompt paraphrases (for prompt-diversity training) --------------------
# The v0 PoC used a single fixed prompt string per formality, so the model could
# memorise 2 token patterns instead of learning a "prompt -> style" function
# (held-out flip stayed ~chance). Training on many *paraphrases* of the same
# instruction, and evaluating on a HELD-OUT paraphrase, tests real generalisation.
FORMALITY_PARAPHRASES: dict[str, list[str]] = {
    "polite": [
        "丁寧な話し方で話します。",
        "敬語で丁寧に話します。",
        "です・ます調で話します。",
        "礼儀正しい口調で話します。",
        "フォーマルな言葉づかいで話します。",
        "ていねいな言葉づかいで応対します。",
    ],
    "casual": [
        "くだけた話し方で話します。",
        "タメ口で話します。",
        "友達みたいに砕けて話します。",
        "カジュアルな口調で話します。",
        "敬語を使わずに話します。",
        "フランクに話します。",
    ],
}


def formality_prompt(formality: str, idx: int) -> str:
    """Return the idx-th paraphrase for a formality (wraps modulo)."""
    bank = FORMALITY_PARAPHRASES[formality]
    return bank[idx % len(bank)]


# ---- persona schema --------------------------------------------------------
# Free-form text prompt (PersonaPlex uses free text). Fields below are the
# control axes we serialize into the Japanese role prompt. For v0 we only
# *populate* what we can measure (formality); the rest are placeholders for the
# causal path and are omitted from the serialized prompt when None.
@dataclass
class Persona:
    formality: str | None = None        # polite | casual
    role: str | None = None             # e.g. "親しい友人", "丁寧な受付"
    relationship: str | None = None     # e.g. "初対面", "旧友"
    pace: str | None = None             # slow | medium | fast
    warmth: str | None = None           # low | medium | high
    backchannel: str | None = None      # none | sparse | frequent
    interruption: str | None = None     # avoid | natural | assertive
    extra: dict[str, Any] = field(default_factory=dict)

    _FORMALITY_JA = {"polite": "丁寧な", "casual": "くだけた", "mixed": "自然な"}
    _PACE_JA = {"slow": "ゆっくりした", "medium": "ふつうの", "fast": "テンポの速い"}
    _WARMTH_JA = {"low": "あっさり", "medium": "ほどよく", "high": "とても親しみのある"}
    _BC_JA = {"none": "ほとんど打たない", "sparse": "控えめに打つ", "frequent": "よく打つ"}

    def to_prompt(self) -> str:
        """Serialize to a Japanese role/persona prompt string (text-prompt segment)."""
        parts: list[str] = []
        if self.role:
            rel = f"{self.relationship}の" if self.relationship else ""
            parts.append(f"あなたは{rel}{self.role}です。")
        if self.formality:
            f = self._FORMALITY_JA.get(self.formality, self.formality)
            parts.append(f"{f}話し方で話します。")
        style = []
        if self.pace:
            style.append(self._PACE_JA.get(self.pace, self.pace) + "テンポ")
        if self.warmth:
            style.append(self._WARMTH_JA.get(self.warmth, self.warmth) + "接し方")
        if style:
            parts.append("、".join(style) + "。")
        if self.backchannel:
            parts.append("あいづちは" + self._BC_JA.get(self.backchannel, self.backchannel) + "。")
        if not parts:
            parts.append("自然な話し方で話します。")
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, {}, [])}


def tokenize_prompt(prompt: str, sp) -> list[int]:
    """Tokenize a Japanese persona prompt into SP ids (same id space as the
    dialogue text track). No special/BOS tokens are added here; the preprocessor
    places the delimiter."""
    prompt = prompt.strip()
    pieces = sp.encode_as_pieces(prompt)
    if pieces and pieces[0] == "▁":
        pieces = pieces[1:]
    elif pieces and pieces[0].startswith("▁"):
        pieces[0] = pieces[0][1:]
    return sp.piece_to_id(pieces)


if __name__ == "__main__":
    # tiny self-check (no model needed for serialization)
    p = Persona(formality="polite", role="受付", relationship="初対面",
                pace="medium", warmth="high", backchannel="frequent")
    print("prompt:", p.to_prompt())
    print("dict  :", p.to_dict())
    p2 = Persona(formality="casual", role="友人", relationship="旧友")
    print("prompt:", p2.to_prompt())
    assert classify_formality("はい、そうですね。ありがとうございます。") == "polite"
    assert classify_formality("そうだよね、めっちゃいいじゃん、かな。") == "casual"
    print("OK")
