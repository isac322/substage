"""Addressed speech level and 3-axis Korean honorific analysis via Sejong POS morphology.

Uses Kiwi (``kiwipiepy``) with the 21st Century Sejong Project POS tagset to
disambiguate sentence-final endings (``EF``), connective endings (``EC``),
polite auxiliary particles (``JX: 요/이요``), subject honorific prefinal
endings (``EP: 시/으시``), pronouns (``NP``), interjections (``IC``), and
nouns/adverbs (``NNG/NNP/MAG/MM``) without hand-rolled character or noun lists:

- ``hasipsio``: formal-polite ``EF`` (``-습니다/-습니까/-십시오/-십시다/-사옵니다`` and
  Unicode jongseong ``-\u11b8니다/-\u11b8니까/-\u11b8시오/-\u11b8시다``)
- ``haeyo``: informal-polite ``EF`` (``-어요/-아요/-여요/-죠/-네요/-거든요/-는데요/-세요``)
  or any clause ending upgraded by trailing ``JX`` ``요/이요``
- ``hae``: addressed plain/intimate ``EF`` (해체·해라체·하게체, including vowel
  contractions ``꺼져 -> 꺼지/VV+어/EF``, ``닥쳐 -> 닥치/VV+어/EF`` and
  sentence-final ``-거든/EF``), plus sentence-final quotative/colloquial
  endings (``-다며/-라며/-라고/-자고/-잖아/-잖나``)
- ``None``: non-final connective ``EC`` (e.g. conditional ``-거든/EC`` in
  ``시간이 나거든 연락하세요`` or ``네가 가게 되거든 나도 하렴``, background ``-는데/EC``
  without ``?``, adverbial ``-게/EC``), mirative/self-talk monologues
  (``-구나/-군/-구먼/-네/-다니/-라니/-랄까`` without ``요``), and nominal/adverbial
  fragments (``NNG``, ``NNP``, ``MAG``, ``IC``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Literal

from kiwipiepy import Kiwi

Level = Literal["hae", "haeyo", "hasipsio"]
LEVELS: tuple[Level, ...] = ("hasipsio", "haeyo", "hae")

DASHES = ("-", "–", "—")
_TURN_RE = re.compile(r"(?:^|(?<=\s))[-–—]+\s*")
_SENT_RE = re.compile(r"(?<=[.?!…])\s+|\n")
_LETTER_RE = re.compile(r"[^\W\d_]")
_HANGUL_RE = re.compile(r"[가-힣]")
# Bidi controls (LRM/RLM, embeddings, isolates); Netflix files prefix lines with U+200E.
_BIDI_RE = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069]")

_JONG_L = 8  # ㄹ

# Published Sejong EF suffixes for 하십시오체 (both compatibility jamo and Hangul jongseong U+11B8)
_HASIPSIO_EF_SUFFIXES = (
    "습니다", "습니까", "십시오", "십시다", "사옵니다", "옵소서", "올시다",
    "\u11b8니다", "\u11b8니까", "\u11b8시오", "\u11b8시다",
    "ㅂ니다", "ㅂ니까", "ㅂ시오", "ㅂ시다", "소서", "옵니다",
)

# Mirative / exclamative / deliberative EF lemmas that represent self-talk or unaddressed exclamations
_MIRATIVE_UNADDRESSED_EF = frozenset({
    "구나", "군", "구먼", "구만", "더구나", "더군", "로군", "로구나",
    "네", "다니", "라니", "랄까", "달까", "람", "담", "거나", "더니", "든", "든지",
})

# Connective endings that can masquerade as EF before a matrix clause when unpunctuated
_HOMOGRAPHIC_CONNECTIVES = frozenset({
    "거든", "는데", "은데", "\u11ab데", "던데", "지", "게", "나", "며", "고", "라", "자",
})

# Sentence-final colloquial/quotative EC endings that carry addressed 해체 force at utterance end
_FINAL_ADDRESSED_EC = frozenset({
    "다며", "라며", "냐며", "자며", "라고", "자고", "냐고", "다고",
    "다니까", "라니까", "잖아", "잖니", "잖나", "야", "이야", "어", "아", "래", "대",
})


@lru_cache(maxsize=1)
def _kiwi() -> Kiwi:
    return Kiwi()


def strip_bidi(text: str) -> str:
    """*text* without Unicode bidi control characters."""
    return _BIDI_RE.sub("", text)


def jongseong(ch: str) -> int:
    """Final-consonant index of a hangul syllable (0 = none); -1 if not hangul."""
    code = ord(ch) - 0xAC00
    return code % 28 if 0 <= code < 11172 else -1


@dataclass(frozen=True)
class TurnMorphemeAnalysis:
    """3-axis Korean honorific feature bundle for one speaker turn."""

    clause_levels: tuple[Level | Literal["mirative"], ...]
    addressed_levels: tuple[Level, ...]
    turn_level: Level | None
    has_subject_hon_si: bool
    has_humble_1p: bool
    has_plain_1p: bool
    has_plain_2p: bool
    has_banmal_ic: bool
    has_internal_clash: bool
    has_bare_final_ec: bool = False

def _classify_ef_token(
    tokens: list,
    idx: int,
    *,
    has_trailing_question: bool,
) -> Level | Literal["mirative"] | None:
    tk = tokens[idx]
    form: str = tk.form
    n = len(tokens)
    next_is_jx_yo = idx + 1 < n and tokens[idx + 1].tag == "JX" and tokens[idx + 1].form in ("요", "이요")

    if next_is_jx_yo or form.endswith(("요", "죠")):
        return "haeyo"
    if form.endswith(_HASIPSIO_EF_SUFFIXES):
        return "hasipsio"
    if form in _MIRATIVE_UNADDRESSED_EF:
        return "mirative"
    if form in ("지", "\u11ab지", "\u11af지", "는지", "은지"):
        # Ambiguous with indirect question -(으)ㄹ지 when stem ends with ㄹ and there is no '?'
        if not has_trailing_question and idx > 0:
            prev_form = tokens[idx - 1].form
            if prev_form and jongseong(prev_form[-1]) == _JONG_L:
                return None
            if form in ("\u11ab지", "\u11af지", "는지", "은지"):
                return None
        return "hae"
    if form in ("는데", "은데", "\u11ab데", "던데", "니", "으니", "\u11ab데"):
        return "hae" if has_trailing_question else None
    if form == "게":
        # Only promissive -(으)ㄹ게 is unambiguously addressed 해체 without context
        if idx > 0 and tokens[idx - 1].form and jongseong(tokens[idx - 1].form[-1]) == _JONG_L:
            return "hae"
        return None
    return "hae"


@lru_cache(maxsize=8192)
def analyze_turn_morphemes(turn_text: str) -> TurnMorphemeAnalysis:
    """Analyze one speaker turn with Kiwi top-2 Sejong POS lattice and 3-axis honorific features."""
    clean = strip_bidi(turn_text).strip()
    if not clean or not _HANGUL_RE.search(clean):
        return TurnMorphemeAnalysis((), (), None, False, False, False, False, False, False)

    # Trailing non-Hangul ASCII word (e.g. "그건 OK") makes final clause unclassified
    kiwi = _kiwi()
    analyses = kiwi.analyze(clean, top_n=2)
    if not analyses:
        return TurnMorphemeAnalysis((), (), None, False, False, False, False, False, False)

    tokens1, score1 = analyses[0]
    tokens2, score2 = analyses[1] if len(analyses) > 1 else (tokens1, score1)
    margin = abs(score1 - score2)

    alt_tags: dict[int, str] = {}
    if margin < 2.5 and len(tokens1) == len(tokens2):
        for i, (tk1, tk2) in enumerate(zip(tokens1, tokens2)):
            if tk1.tag != tk2.tag:
                alt_tags[i] = tk2.tag

    n = len(tokens1)
    # Check if the very last content token is non-Hangul foreign/alphabetic (SL)
    last_content_idx = n - 1
    while last_content_idx >= 0 and tokens1[last_content_idx].tag in ("SF", "SP", "SS", "SE", "SO", "SW"):
        last_content_idx -= 1
    if last_content_idx >= 0 and tokens1[last_content_idx].tag == "SL":
        return TurnMorphemeAnalysis((), (), None, False, False, False, False, False, False)

    # Also split into sentences via Kiwi so line-break two-clause turns ('그런 셈이죠\n그렇다고 치자')
    # preserve each sentence's EF while conditional EC ('시간이 나거든\n연락하세요') stays one sentence
    sents = kiwi.split_into_sents(clean, return_tokens=True)
    ef_positions = [i for i, tk in enumerate(tokens1) if tk.tag == "EF"]

    clause_levels: list[Level | Literal["mirative"]] = []
    if len(sents) >= 2:
        for s_idx, st in enumerate(sents):
            st_tokens = st.tokens
            if not st_tokens:
                continue
            # If this non-final sentence has no SF punctuation (.?!) and ends with a homographic connective
            # like '-거든' whose full-turn lattice has a later EF and an EC alternative (e.g. '네가 가게 되거든 나도 하렴'),
            # treat it as a conditional/connective clause (EC), not an independent EF!
            st_has_sf = any(t.tag == "SF" for t in st_tokens)
            st_has_q = "?" in st.text
            st_ef_indices = [i for i, t in enumerate(st_tokens) if t.tag == "EF"]
            if st_ef_indices:
                last_ef_i = st_ef_indices[-1]
                ef_tk = st_tokens[last_ef_i]
                next_yo = (
                    last_ef_i + 1 < len(st_tokens)
                    and st_tokens[last_ef_i + 1].tag == "JX"
                    and st_tokens[last_ef_i + 1].form in ("요", "이요")
                )
                if (
                    s_idx < len(sents) - 1
                    and not st_has_sf
                    and not next_yo
                    and ef_tk.form in _HOMOGRAPHIC_CONNECTIVES
                    and (any(t in alt_tags.values() for t in ("EC",)) or ef_tk.form in ("거든", "는데", "은데", "\u11ab데", "던데"))
                ):
                    continue
                lv = _classify_ef_token(st_tokens, last_ef_i, has_trailing_question=st_has_q)
                if lv is not None:
                    clause_levels.append(lv)
            else:
                # Sentence without EF: check if it ends withJX '요' or sentence-final addressed EC
                li = len(st_tokens) - 1
                while li >= 0 and st_tokens[li].tag in ("SF", "SP", "SS", "SE", "SO", "SW"):
                    li -= 1
                if li >= 0:
                    ltk = st_tokens[li]
                    if ltk.tag == "JX" and ltk.form in ("요", "이요"):
                        clause_levels.append("haeyo")
                    elif s_idx == len(sents) - 1 and ltk.tag == "JX" and ltk.form in ("이야", "야"):
                        clause_levels.append("hae")
                    elif ltk.tag == "IC":
                        if ltk.form.endswith("요"):
                            clause_levels.append("haeyo")
                        elif ltk.form in ("아니야", "아냐"):
                            clause_levels.append("hae")
                    elif s_idx == len(sents) - 1 and ltk.tag == "EC":
                        if ltk.form in _FINAL_ADDRESSED_EC:
                            clause_levels.append("hae")
                        elif ltk.form in ("는데", "은데", "\u11ab데", "던데", "니", "으니", "나", "는가", "던가") and st_has_q:
                            clause_levels.append("hae")
                        elif ltk.form == "나" and li > 0 and st_tokens[li - 1].form == "않":
                            clause_levels.append("hae")
    else:
        for i, tk in enumerate(tokens1):
            if tk.tag == "EF":
                next_is_sf = i + 1 < n and tokens1[i + 1].tag == "SF"
                next_is_jx_yo = i + 1 < n and tokens1[i + 1].tag == "JX" and tokens1[i + 1].form in ("요", "이요")
                later_ef_exists = any(j > i for j in ef_positions)
                if (
                    later_ef_exists
                    and not next_is_sf
                    and not next_is_jx_yo
                    and (alt_tags.get(i) == "EC" or tk.form in _HOMOGRAPHIC_CONNECTIVES)
                ):
                    continue
                has_q = (next_is_sf and "?" in tokens1[i + 1].form) or clean.endswith("?")
                lv = _classify_ef_token(tokens1, i, has_trailing_question=has_q)
                if lv is not None:
                    clause_levels.append(lv)

        if not clause_levels and last_content_idx >= 0:
            ltk = tokens1[last_content_idx]
            has_q = "?" in clean
            if ltk.tag == "JX" and ltk.form in ("요", "이요"):
                clause_levels.append("haeyo")
            elif ltk.tag == "JX" and ltk.form in ("이야", "야"):
                clause_levels.append("hae")
            elif ltk.tag == "IC":
                if ltk.form.endswith("요"):
                    clause_levels.append("haeyo")
                elif ltk.form in ("아니야", "아냐"):
                    clause_levels.append("hae")
            elif ltk.tag == "EC":
                if ltk.form in _FINAL_ADDRESSED_EC:
                    clause_levels.append("hae")
                elif ltk.form in ("는데", "은데", "\u11ab데", "던데", "니", "으니", "나", "는가", "던가") and has_q:
                    clause_levels.append("hae")
                elif ltk.form == "나" and last_content_idx > 0 and tokens1[last_content_idx - 1].form == "않":
                    clause_levels.append("hae")

    addressed = tuple(lv for lv in clause_levels if lv in ("hae", "haeyo", "hasipsio"))
    bins = {"J" if lv in ("haeyo", "hasipsio") else "B" for lv in addressed}

    has_si = any(tk.tag == "EP" and tk.form in ("시", "으시", "샤") for tk in tokens1)
    has_plain_1p = any(tk.tag == "NP" and tk.form in ("나", "내") for tk in tokens1)
    has_humble_1p = False
    for i, tk in enumerate(tokens1):
        if tk.tag == "NP" and tk.form in ("저", "제", "저희"):
            if i > 0 and tokens1[i - 1].form == "어" and tk.form == "저":
                continue
            if i + 1 < n and tokens1[i + 1].tag in ("XSN", "NR", "NNG", "NNB") and tokens1[i + 1].form in ("들", "둘", "셋", "사람", "것", "거", "기", "쪽"):
                continue
            if i + 1 < n and tokens1[i + 1].tag in ("JKS", "JX", "JKG", "JKB", "JKO"):
                has_humble_1p = True

    has_plain_2p = any(
        tk.tag == "NP" and tk.form in ("너", "네", "니", "너희")
        and (i + 1 < n and tokens1[i + 1].tag in ("JKS", "JX", "JKG", "JKO", "JKB"))
        for i, tk in enumerate(tokens1)
    )
    has_banmal_ic = any(
        (tk.tag == "IC" and tk.form in ("응", "야", "아니야", "아냐"))
        or (
            tk.form == "미안"
            and (i + 1 == n or tokens1[i + 1].tag in ("SF", "SP", "SS", "SE"))
        )
        for i, tk in enumerate(tokens1)
    )
    has_bare_ec = (
        last_content_idx >= 0
        and tokens1[last_content_idx].tag == "EC"
        and tokens1[last_content_idx].form in ("어서", "아서", "여서", "해서", "니까", "으니까", "길래", "는데", "은데", "\u11ab데", "던데")
    )

    has_clash = (
        len(bins) > 1
        or ("B" in bins and has_humble_1p)
        or ("J" in bins and (has_plain_2p or has_banmal_ic))
    )
    turn_level: Level | None = addressed[-1] if addressed else None
    return TurnMorphemeAnalysis(
        clause_levels=tuple(clause_levels),
        addressed_levels=addressed,
        turn_level=turn_level,
        has_subject_hon_si=has_si,
        has_humble_1p=has_humble_1p,
        has_plain_1p=has_plain_1p,
        has_plain_2p=has_plain_2p,
        has_banmal_ic=has_banmal_ic,
        has_internal_clash=has_clash,
        has_bare_final_ec=has_bare_ec,
    )


def ending_level(sentence: str) -> Level | None:
    """Addressed speech level of one sentence or clause via Sejong POS morphology."""
    return analyze_turn_morphemes(sentence).turn_level


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


def has_internal_register_clash(ko: str) -> bool:
    """True when any single speaker turn internally mixes 반말 (B) and 존댓말 (J)
    clauses or clashes humble 1st-person / plain 2nd-person / banmal interjection with the ending."""
    return any(analyze_turn_morphemes(t).has_internal_clash for t in speaker_turns(ko))


def addressed_levels(ko: str) -> list[Level | None]:
    """One level (or None) per speaker turn of a subtitle line."""
    return [analyze_turn_morphemes(t).turn_level for t in speaker_turns(ko)]


def addressed_level(ko: str) -> Level | None:
    """Level of a single-speaker line; None for unleveled lines and for
    two-speaker dash lines (use ``addressed_levels`` for those)."""
    turns = addressed_levels(ko)
    return turns[0] if len(turns) == 1 else None
