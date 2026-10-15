"""Addressed speech level of a Korean subtitle line.

The level is the one the speaker addresses the listener with, read off the
final 어절 of each sentence after trailing punctuation:

- hasipsio: ``(?:[ㅂ-final syllable]|습|읍)(?:니다|니까)$``, ``(?:십시오|시오)$``,
  ``[ㅂ-final syllable]시다$`` (갑니다, 했습니까, 가십시오, 갑시다)
- haeyo: ``(?:요|죠)$`` (갈게요, 했어요, 그렇죠, 기억하시네요)
- hae (반말, incl. 해라체 and 하게체): a final syllable in ``_HAE_FINALS``, or
  the promissive -ㄹ게 (가자, 간다, 갈래, 했어, 뭐야, 갈까, 하지 마, 알잖나, 올게)
- None: anything that is not addressed at a level:
  - exclamatives and self-talk: -네, -군, -구나 (기억하시네, 그걸 몰랐네)
  - connective and quotative fragments: -아니라, -(이)라, -라니/-다니, -거나,
    -더니, -든 (그게 아니라, 군대라니)
  - -는데/-ㄴ데, -(으)니, -는지/-ㄴ지/-ㄹ지, -(으)니까 unless the sentence ends
    with ``?`` (뭔데? and 알지? are addressed 반말 questions; 있으니 is not)
  - other -게 (adverbial 간단하게, or a 하게체 imperative it cannot tell apart)
  - vocatives, kin terms and common nouns whose last syllable looks like an
    ending (큰엄마!, 아버지, 여자, 군대)
  - single syllables (네, 응, 야), other nouns and names, and a sentence whose
    final word is not hangul

A line is split into speaker turns at dashes (``-가자\\n-네, 갈게요``); a
turn's level is that of its last sentence that has one. Sentences end at
``.?!…`` followed by whitespace and at line breaks (official subtitles often
drop the period).
"""

from __future__ import annotations

import re
from typing import Literal

Level = Literal["hae", "haeyo", "hasipsio"]
LEVELS: tuple[Level, ...] = ("hasipsio", "haeyo", "hae")

DASHES = ("-", "–", "—")
_TURN_RE = re.compile(r"(?:^|(?<=\s))[-–—]+\s*")
_SENT_RE = re.compile(r"(?<=[.?!…])\s+|\n")
_FINAL_RE = re.compile(r"([가-힣]+)([^가-힣]*)$")
_LETTER_RE = re.compile(r"[^\W\d_]")
# Bidi controls (LRM/RLM, embeddings, isolates); Netflix files prefix lines with U+200E.
_BIDI_RE = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069]")


def strip_bidi(text: str) -> str:
    """*text* without Unicode bidi control characters."""
    return _BIDI_RE.sub("", text)


_JONG_N = 4  # ㄴ as a final consonant
_JONG_L = 8  # ㄹ
_JONG_B = 17  # ㅂ

_HAE_FINALS = frozenset("아어여야지자래냐다라걸나까대봐줘와워돼해써마렴")
# Never an addressed level, whatever the punctuation.
_UNADDRESSED = (
    "네", "군", "구나", "구먼", "아니라", "이라", "라니", "다니", "거나", "더니", "든", "든지",
)
# Connective endings; a sentence-final ``?`` makes them an addressed 반말 question.
_CONNECTIVE = ("데", "니")
# Vocatives, kin terms and common nouns whose last syllable looks like an
# ending.
_NOUNS = (
    "엄마", "아줌마", "아버지", "어머니", "할머니", "언니", "누나",
    "여자", "남자", "의자", "모자", "과자", "기자", "부자", "환자", "편지", "바지", "거지",
    "미래", "바다", "나라", "아까", "그걸", "이걸", "저걸",
    "군대", "침대", "시대", "반대", "상대", "세대", "무대", "피해", "올해", "영어", "단어",
    "필요", "중요", "오래",
)


def jongseong(ch: str) -> int:
    """Final-consonant index of a hangul syllable (0 = none); -1 if not hangul."""
    code = ord(ch) - 0xAC00
    return code % 28 if 0 <= code < 11172 else -1


def _token_level(token: str, question: bool) -> Level | None:
    if len(token) < 2 or token.endswith(_NOUNS):
        return None
    if token.endswith(("니다", "니까")):
        prev = token[-3] if len(token) >= 3 else ""
        if prev and (jongseong(prev) == _JONG_B or prev in "습읍"):
            return "hasipsio"
        return None if token.endswith("니까") else "hae"
    if token.endswith(("십시오", "시오")):
        return "hasipsio"
    if token.endswith("시다") and len(token) >= 3 and jongseong(token[-3]) == _JONG_B:
        return "hasipsio"
    if token.endswith(("요", "죠")):
        return "haeyo"
    if token.endswith(_UNADDRESSED):
        return None
    if token.endswith(_CONNECTIVE) or (token[-1] == "지" and jongseong(token[-2]) in (_JONG_N, _JONG_L)):
        return "hae" if question else None
    if token[-1] == "게":  # promissive -ㄹ게; adverbial -게 and 하게체 imperatives are ambiguous
        return "hae" if jongseong(token[-2]) == _JONG_L else None
    return "hae" if token[-1] in _HAE_FINALS else None


def ending_level(sentence: str) -> Level | None:
    """Level of one sentence, read off its final 어절 (see module docstring)."""
    m = _FINAL_RE.search(sentence.strip())
    if not m or _LETTER_RE.search(m.group(2)):  # trailing non-hangul word → unclassified
        return None
    return _token_level(m.group(1), "?" in m.group(2))


def speaker_turns(ko: str) -> list[str]:
    """Dash-marked two-speaker lines split into turns; otherwise the whole text."""
    text = strip_bidi(ko).strip()
    if not text:
        return []
    if any(ln.strip().startswith(DASHES) for ln in text.split("\n")):
        turns = [t.strip() for t in _TURN_RE.split(text) if t.strip()]
        if len(turns) >= 2:
            return turns
        return [text.lstrip("".join(DASHES)).strip()]
    return [text]


def _turn_level(turn: str) -> Level | None:
    levels = [ending_level(s) for s in _SENT_RE.split(turn)]
    known = [lv for lv in levels if lv is not None]
    return known[-1] if known else None


def addressed_levels(ko: str) -> list[Level | None]:
    """One level (or None) per speaker turn of a subtitle line."""
    return [_turn_level(t) for t in speaker_turns(ko)]


def addressed_level(ko: str) -> Level | None:
    """Level of a single-speaker line; None for unleveled lines and for
    two-speaker dash lines (use ``addressed_levels`` for those)."""
    turns = addressed_levels(ko)
    return turns[0] if len(turns) == 1 else None
