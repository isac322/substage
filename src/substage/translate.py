"""S3 scene translation: text-first with AV cascade, plus on-screen-text cues.

Every LLM call goes through a ``generate`` call with a stage label: Gemini for
AV calls, ``llm.make_client(settings.model_translate)`` for text-mode calls when
that override is set (else Gemini).
Checkpoints are per (scene, mode) keyed by input_hash(raw scene inputs, the
exact rendered prompt — prev KO tail, callback KO, retry flags, translator
memory examples — system template, model, and av clip bounds) so unchanged
scenes are skipped on resume; the orchestrator invalidates a scene's
checkpoints when it rebuilds that scene's snapshot.

Translator memory (``memory.py``): when the corpus exists and
Settings.memory_k > 0, each scene prompt carries a TRANSLATOR EXAMPLES block —
the top-k official (EN → KO) pairs of *other* videos retrieved for the scene's
English cues; the video being translated (its base id, arm suffix stripped) is
always excluded.

Scheduling per contract: sequential inside an act so each scene sees the
previous scene's Korean tail (last 6 translated lines); acts run in parallel
threads (max 4). The first scene of an act starts cold — act seams are covered
by S4 windowed review.
"""

import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from . import layout, media, memory, perception
from .capability import Architecture, CapabilityError, RoutePlan, resolve_route
from .config import Settings, Workspace, narrative_context, narrative_text
from .gemini import Clip, GeminiBlocked, GeminiClient, UploadedFile
from .llm import LLMClient, make_client, text_client
from .models import (
    Cue,
    CueObservation,
    VideoInfo,
    MediaNeed,
    OnScreenText,
    OmittedAct,
    OstBeat,
    PerceptionLedger,
    Register,
    RetainedFragment,
    Scene,
    SceneTranslation,
    SemanticAttributionProvenance,
    SemanticAttributionResult,
    SemanticDraft,
    SemanticLineage,
    SemanticProvenance,
    SemanticSceneTranslation,
    SemanticSource,
    Snapshot,
    TranslatedCue,
)
from .register import render_scene_register
from .store import Checkpoints, input_hash
from .style import with_guide
from .validate import (
    OST_KIND,
    _may_stay_empty,
    classify_lyric_cue,
    en_speaker_turns,
    is_gag_credit,
    is_translated_lyric,
    ost_kind,
)

KO_TAIL = 6          # previous-scene KO lines passed for continuity
ACT_WORKERS = 4      # max parallel act threads
CLIP_PAD_S = 1.0     # av clip padding around the scene
OST_OVERLAP_MS = 500  # time slack when matching on-screen text to existing cues
OST_MIN_MS = 1000    # on-screen-text cue duration floor
OST_MAX_MS = 6000    # on-screen-text cue duration cap
OST_MERGE_GAP_MS = 2000  # repeated reports this close (same text) are one appearance
OST_COVER_RATIO = 0.6  # share of the text's words overlapping cues must carry
ESCALATION_NEIGHBORS = 1
MAX_VIRTUAL_CONTINUATIONS = 2
VIRTUAL_CONTINUATION_GAP_MS = 1200

_STAGE = "s3_translate"
_SEMANTIC_EXTRACT_STAGE = "s3_semantic_extract"
_SEMANTIC_WRITE_STAGE = "s3_semantic_write"
_SEMANTIC_ROUTE_VERSION = "semantic-route-v1"
_SEMANTIC_SOURCE_VERSION = "semantic-source-v1"
_SEMANTIC_TARGET_VERSION = "semantic-target-v1"
_SEMANTIC_PROMPT_VERSION = "semantic-prompt-v1"
_MAX_SEMANTIC_ATTEMPTS = 3

logger = logging.getLogger(__name__)

#: media_reason on cues left empty because the provider refused them
_BLOCKED_REASON = "blocked by provider"

_NON_TRANSLATED = re.compile(r"^\s*$")



class _LunaCheck(BaseModel):
    """Small no-media consistency response used only when a caller supplies Luna."""

    flagged_cue_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

@lru_cache(maxsize=8)
def _prompt(name: str) -> str:
    return resources.files("substage.prompts").joinpath(name).read_text(encoding="utf-8")


def _system(narrative: str) -> str:
    """translate.md for one narrative level (lite | full) plus the style guide."""
    return with_guide(narrative_text(_prompt("translate.md"), narrative))


def _semantic_safety_system(narrative: str) -> str:
    """Translation task system without the style guide safety examples."""
    return _semantic_system(narrative_text(_prompt("translate.md"), narrative))


# ------------------------------------------------------------ translator memory


def memory_k(settings: Any) -> int:
    """The client's memory_k; duck-typed settings without it get the Settings default."""
    return getattr(settings, "memory_k", Settings.memory_k)


def scene_examples(settings: Any, scene_cues: list[Cue], exclude_video: str) -> list[memory.Pair]:
    """Translator-memory examples for one scene (never from *exclude_video*);
    [] when memory_k is 0 or no corpus exists."""
    k = memory_k(settings)
    corpus = memory.load_corpus() if k > 0 else None
    if corpus is None:
        return []
    return memory.retrieve(corpus, [c.text for c in scene_cues if c.text.strip()], memory.base_video_id(exclude_video), k)


def memory_tag(settings: Any, exclude_video: str) -> list[str]:
    """Hash parts identifying the memory a video's translation sees: [] when
    disabled or absent, else memory_k and the visible corpus fingerprint."""
    k = memory_k(settings)
    corpus = memory.load_corpus() if k > 0 else None
    if corpus is None:
        return []
    return [f"memory_k={k}", corpus.fingerprint(memory.base_video_id(exclude_video))]


_EXAMPLES_HEAD = (
    "## TRANSLATOR EXAMPLES (official Korean subtitles of other videos by the reference translator; "
    "imitate their register choices, condensation, profanity strength and phrasing habits — never copy content)"
)


def _render_examples(examples: Sequence[memory.Pair]) -> str:
    if not examples:
        return ""
    rows = []
    for p in examples:
        prev = f" (prev: {p.prev_en})" if p.prev_en else ""
        rows.append(f"  EN: {p.en}{prev} → KO: {p.ko.replace(chr(10), ' / ')}")
    return _EXAMPLES_HEAD + "\n" + "\n".join(rows)


# --------------------------------------------------------------------- prompt


def _fmt_ms(ms: int) -> str:
    s, rem = divmod(max(0, ms), 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{rem:03d}"


def _budget(cue: Cue, next_start: int | None) -> int:
    """Max Korean characters (validate.cps_chars) for the cue: the display time
    S5c layout can give it (until the next cue's start minus min_gap, capped at
    max_duration) times cps_max, at most two full lines."""
    s = Settings  # class defaults: the prompt builders carry no settings object
    avail = layout.available_ms(cue.start_ms, cue.end_ms, next_start, s)
    return max(1, min(2 * s.cpl_limit, int(avail / 1000.0 * s.cps_max)))


def _aim(budget: int, ratio: float) -> int:
    """Soft Korean length target: professional subtitles land well below the limit."""
    return max(1, math.floor(budget * ratio))


def aim_ratio(settings: Any) -> float:
    """The client's aim ratio; duck-typed settings without it get the Settings default."""
    return getattr(settings, "aim_ratio", Settings.aim_ratio)


def _render_snapshot(snapshot: Snapshot, callback_ko: dict[str, str]) -> str:
    """Snapshot block: video tone/rating, character names and personas,
    registers, character states, cue deltas, glossary, live callbacks (with
    established KO phrasing), ambiguity flags."""
    lines: list[str] = ["## CONTEXT (causal state at scene entry; no future facts)"]
    video = [f"tone={snapshot.tone_baseline}"] if snapshot.tone_baseline else []
    if snapshot.rating:
        video.append(f"rating={snapshot.rating}")
    if video:
        lines.append(f"VIDEO: {'; '.join(video)}")
    if snapshot.character_names:
        lines.append("CHARACTER NAMES (use exactly):")
        for cid, ko_name in sorted(snapshot.character_names.items()):
            lines.append(f"  {cid} -> {ko_name}")
    if snapshot.personas:
        lines.append("PERSONAS (attitude drives register and profanity):")
        for cid, persona in sorted(snapshot.personas.items()):
            lines.append(f"  {cid}: {persona}")
    if snapshot.registers:
        lines.append("REGISTERS (speaker>addressee: level · vocative · pronoun; shifts at their cue ids):")
        for pair, reg in sorted(snapshot.registers.items()):
            lines.extend(render_scene_register(pair, reg))
    if snapshot.characters:
        lines.append("CHARACTERS:")
        for cid, st in sorted(snapshot.characters.items()):
            parts = []
            if st.emotion:
                parts.append(f"emotion={st.emotion}")
            if st.knows:
                parts.append(f"knows=[{'; '.join(st.knows)}]")
            if st.recent_events:
                parts.append(f"recent=[{'; '.join(st.recent_events)}]")
            lines.append(f"  {cid}: {'; '.join(parts) if parts else '-'}")
    if snapshot.voice_cards:
        lines.append("VOICE CARDS (speaker: idiolect | vocab | profanity ceiling 0-3 | signature KO):")
        for cid, vc in sorted(snapshot.voice_cards.items()):
            vocab = ", ".join(vc.vocabulary) if vc.vocabulary else "-"
            sig = vc.signature_ko or "-"
            lines.append(f"  {cid}: {vc.idiolect} | vocab: {vocab} | ceiling {vc.profanity_ceiling} | {sig}")
    if snapshot.cue_deltas:
        lines.append("CUE DELTAS (state changes at specific cues):")
        for d in snapshot.cue_deltas:
            for ec in d.emotion_changes:
                lines.append(f"  {d.cue_id} emotion: {ec.character} {ec.from_state}->{ec.to_state}")
            for rv in d.revelations:
                lines.append(f"  {d.cue_id} reveals to {', '.join(rv.to_whom)}: {rv.fact}")
    if snapshot.glossary:
        lines.append("GLOSSARY (en -> ko, use exactly):")
        for g in snapshot.glossary:
            note = f" ({g.note})" if g.note else ""
            lines.append(f"  {g.en} -> {g.ko}{note}")
    if snapshot.callbacks_live:
        lines.append("CALLBACKS (payoff cues MUST reuse the fixed Korean wording):")
        for cb in snapshot.callbacks_live:
            ko = callback_ko.get(cb.motif, "")
            ko_part = f" | fixed KO: {ko}" if ko else ""
            note = f" ({cb.note})" if cb.note else ""
            lines.append(
                f"  {cb.motif}{note}: setups [{', '.join(cb.setup_cues)}] "
                f"-> payoffs [{', '.join(cb.payoff_cues)}]{ko_part}"
            )
    if snapshot.preserve_ambiguity:
        lines.append("PRESERVE AMBIGUITY (keep the line double-edged; do not resolve):")
        for f in snapshot.preserve_ambiguity:
            lines.append(f"  {f.cue_id}: {f.note}")
    return "\n".join(lines)


def _render_observations(scene_cue_ids: list[str], obs_by_id: dict[str, CueObservation]) -> str:
    rows = []
    for cid in scene_cue_ids:
        o = obs_by_id.get(cid)
        if o is None:
            continue
        parts = []
        if o.speaker:
            parts.append(f"speaker={o.speaker}")
        if o.addressee:
            parts.append(f"to={o.addressee}")
        if o.delivery:
            parts.append(f"delivery={o.delivery}")
        if o.nonverbal:
            parts.append(f"nonverbal={o.nonverbal}")
        if parts:
            rows.append(f"  {cid}: {'; '.join(parts)}")
    if not rows:
        return ""
    return "## OBSERVATIONS (from the viewing pass; ground register, tone, deixis)\n" + "\n".join(rows)

def _render_perception(
    scene_cue_ids: Sequence[str], ledger: PerceptionLedger | None
) -> str:
    """Render advisory perception through the canonical perception module."""
    return perception.render_perception_ledger(ledger, cue_ids=scene_cue_ids)


def _render_virtual_continuations(cues: Sequence[Cue]) -> str:
    """Render bounded source continuations without introducing output cues."""
    if not cues:
        return ""
    rows = []
    for cue in cues[:MAX_VIRTUAL_CONTINUATIONS]:
        text = cue.text.replace("\n", " / ").strip()
        if text:
            rows.append(
                f"  {cue.id} ({_fmt_ms(cue.start_ms)}-{_fmt_ms(cue.end_ms)}): {text}"
            )
    if not rows:
        return ""
    return (
        "## VIRTUAL CONTINUATION HINTS (context only; do not emit or merge these "
        "cue IDs)\n" + "\n".join(rows)
    )


def _render_prev_tail(prev_ko_tail: list[tuple[str, str]]) -> str:
    if not prev_ko_tail:
        return ""
    rows = "\n".join(f"  {cid}: {ko}" for cid, ko in prev_ko_tail)
    return "## PREVIOUS SCENE KOREAN TAIL (continuity anchor; do not re-translate)\n" + rows


_SPAN_TRAIL_RE = re.compile(r"(?:\.\.\.|…|--|—|,)\s*[\"'’”)\]]*\s*$")
_SPAN_TERM_RE = re.compile(r"[.?!][\"'’”)\]]*\s*$")
_SPAN_PP_MOD_RE = re.compile(
    r"^(?:(?:but\s+|and\s+|not\s+|even\s+|just\s+|only\s+)?(?:with|without|for|from|in|into|between|of|by|through|on|at|to|as|"
    r"about|against|over|under|out\s+of|instead\s+of|after|since|unless|where|when|like|despite|toward|towards|within|"
    r"across|around|behind|beyond|during|near|upon)|"
    r"who|whom|whose|which|because|before|until|than|while)\b",
    re.IGNORECASE,
)
_SPAN_COORD_RE = re.compile(
    r"^(?:(?:and|but)(?!\s+(?:with|without|for|from|in|into|between|of|by|through|on|at|to|as|about|against|over|under|out\s+of|instead\s+of|after|since|unless|where|when|like|despite|toward|towards|within)\b)|or|so|then|yet)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CueSpan:
    span_id: str
    cue_ids: tuple[str, str]
    policy: str  # "REVERSE_SOV_PERMITTED" | "DRAMATIC_REVEAL_KEEP_ORDER"


def detect_syntactic_spans(
    cues: Sequence[Cue],
    obs_by_id: dict[str, CueObservation],
    *,
    max_gap_ms: int = 1500,
) -> dict[str, CueSpan]:
    """Map cue_id -> CueSpan for consecutive 2-cue split English sentences."""
    by_cid: dict[str, CueSpan] = {}
    i = 0
    while i < len(cues) - 1:
        c1, c2 = cues[i], cues[i + 1]
        gap_ms = c2.start_ms - c1.end_ms
        if gap_ms < 0 or gap_ms > max_gap_ms or c1.lyric or c2.lyric:
            i += 1
            continue
        o1, o2 = obs_by_id.get(c1.id), obs_by_id.get(c2.id)
        spk1 = (o1.speaker if o1 and o1.speaker else c1.speaker_hint) or ""
        spk2 = (o2.speaker if o2 and o2.speaker else c2.speaker_hint) or ""
        if spk1 and spk2 and spk1.strip().lower() != spk2.strip().lower():
            i += 1
            continue
        if en_speaker_turns(c2, observation=o2) >= 2:
            i += 1
            continue
        t1 = c1.text.replace("\n", " ").strip()
        t2 = c2.text.replace("\n", " ").strip()
        t2_clean = t2.lstrip(" -\"'‘“(.[")
        has_trail = bool(_SPAN_TRAIL_RE.search(t1))
        no_term = not bool(_SPAN_TERM_RE.search(t1))
        c2_pp_mod = bool(_SPAN_PP_MOD_RE.match(t2_clean))
        c2_lower = bool(t2_clean and t2_clean[0].islower()) or t2.lstrip().startswith(("...", "…"))
        if not (has_trail or (no_term and (c2_pp_mod or c2_lower)) or c2_pp_mod):
            i += 1
            continue
        words2 = [w for w in re.split(r"\s+", t2_clean) if w]
        is_short_reveal = has_trail and len(words2) <= 2 and not c2_pp_mod
        is_coord = bool(_SPAN_COORD_RE.match(t2_clean))
        is_that_if_after_complete = bool(re.match(r"^(?:that|if)\b", t2_clean, re.I)) and len(t1.split()) >= 5 and not c2_pp_mod
        policy = (
            "DRAMATIC_REVEAL_KEEP_ORDER"
            if (is_short_reveal or is_coord or is_that_if_after_complete or gap_ms >= 1000)
            else "REVERSE_SOV_PERMITTED"
        )
        span = CueSpan(span_id=f"{c1.id}..{c2.id}", cue_ids=(c1.id, c2.id), policy=policy)
        by_cid[c1.id] = span
        by_cid[c2.id] = span
        i += 2
    return by_cid

_DISCOURSE_SUSPEND_END_RE = re.compile(
    r"(?:(?:\.\.\.|…|--|—|[,:])\s*[\"'’”)\]]*|"
    r"\b(?:a|an|the|of|to|for|with|from|in|on|at|by|as|and|but|or|so|because|if|unless|until|when|where|which|that|who)\s*[\"'’”)\]]*)\s*$",
    re.IGNORECASE,
)
_DISCOURSE_META_START_RE = re.compile(
    r"^(?:let's\s+(?:just\s+)?(?:call\s+it|say|pretend|assume|put\s+it|think\s+of\s+it)|"
    r"think\s+of\s+it\s+as|call\s+it\b|or\s+as\s+i\s+like\s+to\s+call|i\s+mean\b|if\s+you\s+will\b)",
    re.IGNORECASE,
)
_DISCOURSE_CONT_START_RE = re.compile(
    r"^(?:which|that|who|whom|whose|where|when|while|because|since|unless|until|although|though|"
    r"even\s+if|even\s+though|so\s+that|in\s+order\s+to|at\s+how|with\s+this|like\s+i'm|after\s+all)\b",
    re.IGNORECASE,
)
_HOLD_ELLIPSIS_END_RE = re.compile(r"(?:\.\.\.|…|--|—)\s*[\"'’”)\]]*\s*$")


@dataclass(frozen=True)
class DiscourseUnit:
    unit_id: str
    cue_ids: tuple[str, ...]
    start_ms: int
    end_ms: int
    speech_act_type: str  # "hesitation_completion" | "extended_clause" | "ellipsis_referent" | "setup_punchline"
    hold_ellipsis_cue_ids: tuple[str, ...] = ()


def build_discourse_units(
    cues: Sequence[Cue],
    obs_by_id: dict[str, CueObservation],
    *,
    max_gap_ms: int = 2500,
    max_unit_len: int = 5,
) -> list[DiscourseUnit]:
    """Group 2..5 contiguous spoken cues into unified DiscourseUnits across hesitation, metalinguistic completion, and clause chains."""
    spoken = [
        c for c in cues
        if not c.lyric and not c.id.startswith("ost-") and c.source != "ost" and c.text.strip()
    ]
    units: list[DiscourseUnit] = []

    def _can_link(c1: Cue, c2: Cue) -> bool:
        gap_ms = c2.start_ms - c1.end_ms
        if gap_ms < 0 or gap_ms > max_gap_ms:
            return False
        o1, o2 = obs_by_id.get(c1.id), obs_by_id.get(c2.id)
        if en_speaker_turns(c1, observation=o1) >= 2 or en_speaker_turns(c2, observation=o2) >= 2:
            return False
        spk1 = (o1.speaker if o1 and o1.speaker else c1.speaker_hint) or ""
        spk2 = (o2.speaker if o2 and o2.speaker else c2.speaker_hint) or ""
        if spk1 and spk2 and spk1.strip().lower() != spk2.strip().lower():
            return False
        t1 = c1.text.replace("\n", " ").strip()
        t2 = c2.text.replace("\n", " ").strip()
        t2_clean = t2.lstrip(" -\"'‘“(.[")
        if _DISCOURSE_SUSPEND_END_RE.search(t1):
            return True
        if _DISCOURSE_META_START_RE.match(t2_clean) or _DISCOURSE_CONT_START_RE.match(t2_clean):
            return True
        if not _SPAN_TERM_RE.search(t1) and (t2_clean[:1].islower() or t2.lstrip().startswith(("...", "…"))):
            return True
        return False

    i = 0
    while i < len(spoken) - 1:
        if not _can_link(spoken[i], spoken[i + 1]):
            i += 1
            continue
        chain = [spoken[i], spoken[i + 1]]
        j = i + 2
        while j < len(spoken) and len(chain) < max_unit_len and _can_link(chain[-1], spoken[j]):
            chain.append(spoken[j])
            j += 1
        # If the cue immediately after a hesitation/metalinguistic completion is a short same-speaker follow-up (< 400ms), include it up to max_unit_len
        if j < len(spoken) and len(chain) < max_unit_len:
            last_t = chain[-1].text.replace("\n", " ").strip().lstrip(" -\"'‘“(.[")
            if _DISCOURSE_META_START_RE.match(last_t) and 0 <= spoken[j].start_ms - chain[-1].end_ms <= 400:
                o_last, o_nxt = obs_by_id.get(chain[-1].id), obs_by_id.get(spoken[j].id)
                s_last = (o_last.speaker if o_last and o_last.speaker else chain[-1].speaker_hint) or ""
                s_nxt = (o_nxt.speaker if o_nxt and o_nxt.speaker else spoken[j].speaker_hint) or ""
                if (not s_last or not s_nxt or s_last.strip().lower() == s_nxt.strip().lower()) and en_speaker_turns(spoken[j], observation=o_nxt) == 1:
                    chain.append(spoken[j])
                    j += 1
        hold_ids = tuple(
            c.id for c in chain[:-1] if _HOLD_ELLIPSIS_END_RE.search(c.text.replace("\n", " ").strip())
        )
        has_meta = any(
            _DISCOURSE_META_START_RE.match(c.text.replace("\n", " ").strip().lstrip(" -\"'‘“(.["))
            for c in chain[1:]
        )
        if hold_ids and has_meta:
            act_type = "hesitation_completion"
        elif hold_ids and any(len(c.text.split()) <= 3 for c in chain[1:]):
            act_type = "setup_punchline"
        elif any(len(c.text.split()) <= 4 or c.text.strip().endswith("?") for c in chain):
            act_type = "ellipsis_referent"
        else:
            act_type = "extended_clause"
        units.append(
            DiscourseUnit(
                unit_id=f"{chain[0].id}..{chain[-1].id}",
                cue_ids=tuple(c.id for c in chain),
                start_ms=chain[0].start_ms,
                end_ms=chain[-1].end_ms,
                speech_act_type=act_type,
                hold_ellipsis_cue_ids=hold_ids,
            )
        )
        i = j
    return units


_PRAGMA_PARTITIVE_RE = re.compile(
    r"\b(?:(?:not\s+)?(?:all|part|most|half|none|rest|every\s+inch|every\s+part)\s+of\s+(?:you|me|him|her|us|it)\b"
    r"|\b(?:all|part)\s+of\s+(?:you|me)\s+(?:was|is|wasn't|isn't)\b"
    r"|\blove\s+\w+'s\s+brain\b"
    r"|\byou\s+are\s+haunting\b"
    r"|\balways\s+the\s+wrong\s+guy\b)",
    re.IGNORECASE,
)
_PRAGMA_DEIXIS_RE = re.compile(
    r"\b(?:you\s+look\s+(?:ridiculous|insane|crazy|like\s+shit|great)|"
    r"put\s+(?:him|her|it|them)\s+(?:down|up)\s+(?:over\s+)?(?:here|there)|"
    r"stay\s+(?:down|up|back)\s+(?:here|there)|"
    r"(?:up,\s*up|down,\s*down)|"
    r"fall\s+backwards|"
    r"keep\s+it\s+steady|"
    r"just\s+take\s+it\s+slow|"
    r"right\s+there\b|"
    r"look\s+at\s+(?:this|that)\b|"
    r"shooting\s+myself\s+with\s+it\b|"
    r"get\s+this\s+thing\s+off|"
    r"as\s+soon\s+as\s+you\s+do\s+that|"
    r"screaming\s+at\s+a\s+machine)",
    re.IGNORECASE,
)
_PRAGMA_IDIOM_RE = re.compile(
    r"\b(?:smell\s+what\s+you(?:'re|\s+are)\s+stepp?in'?|"
    r"sit\s+on\s+a\s+stick|"
    r"have\s+nuts\s+in\s+it|"
    r"money\s+shot|"
    r"shake\s+weight|"
    r"calling\s+wang|"
    r"flicking\s+my\s+bean|"
    r"full\s+of\s+shit|"
    r"doesn't\s+run\s+shit|"
    r"there's\s+a\s+stupid|"
    r"what\s+does\s+.*\s+taste\s+like|"
    r"cinnamon\s+ring|"
    r"i'd\s+love\s+a\s+blow\s+job|"
    r"i'd\s+hit\s+that|"
    r"she\s+don't\s+play|"
    r"gargling\s+juggernaut|"
    r"pixie\s+slab|"
    r"metaphysical\s+junkyard|"
    r"afrangry|"
    r"are\s+you\s+there\s+god\??\s+it's\s+me,\s+margaret)\b",
    re.IGNORECASE,
)
_PRAGMA_IMPLICATURE_RE = re.compile(
    r"\b(?:where\s+else\s+(?:do|would|could)\s+you\s+put|"
    r"maybe\s+not\s+start\s+with\s+that|"
    r"that\s+suit\s+says\s+different|"
    r"and\s+there\s+he\s+goes|"
    r"great,\s+it's\s+just\s+some\s+random\s+guy|"
    r"your\s+piece\s+of\s+it\s+is\s+at\s+its\s+very\s+center|"
    r"nothing\s+easy\s+about\s+you|"
    r"forces,\s+forces|"
    r"well,\s+maybe\s+not\s+the\s+most|"
    r"well,\s+he\s+better|"
    r"tell\s+the\s+difference|"
    r"army\s+of\s+the\s+insane|"
    r"you\s+come\s+for\s+me.*you\s+come\s+for|"
    r"it\s+says,\s*['\"]?ibuprofen|"
    r"endeared\s+herself\s+to\s+him|"
    r"spanked\s+it\s+to|"
    r"how\s+you\s+screwed\s+up\s+everything|"
    r"not\s+what\s+it\s+looks\s+like|"
    r"not\s+for\s+me\b|"
    r"how\s+tough\s+can\s+he\s+be|"
    r"with\s+a\s+name\s+like|"
    r"is\s+that\s+a\s+band|"
    r"showed\s+up\s+in\s+the\s+elevator)\b",
    re.IGNORECASE,
)
def detect_pragmatic_risk_cues(
    cues: Sequence[Cue],
    obs_by_id: dict[str, CueObservation] | None = None,
) -> dict[str, str]:
    """Map cue_id -> pragmatic risk category for cues requiring situational/implicature grounding."""
    out: dict[str, str] = {}
    for c in cues:
        if c.lyric or ost_kind(c) == "credit":
            continue
        t = c.text.replace("\n", " ").strip()
        if _PRAGMA_PARTITIVE_RE.search(t):
            out[c.id] = "body_or_partitive_metonymy"
        elif _PRAGMA_IDIOM_RE.search(t):
            out[c.id] = "idiom_or_slang"
        elif _PRAGMA_IMPLICATURE_RE.search(t):
            out[c.id] = "conversational_implicature"
        elif _PRAGMA_DEIXIS_RE.search(t):
            out[c.id] = "visual_action_deixis"
    return out


def _render_cues(
    cues: list[Cue], obs_by_id: dict[str, CueObservation], aim_ratio: float
) -> str:
    lines = ["## CUES (id | start-end | speaker | sound | dur s | aim/max chars | turns | text)"]
    spans = detect_syntactic_spans(cues, obs_by_id)
    d_units = {cid: du for du in build_discourse_units(cues, obs_by_id) for cid in du.cue_ids}
    for n, c in enumerate(cues):
        dur = (c.end_ms - c.start_ms) / 1000.0
        speaker = c.speaker_hint or "-"
        sound = ",".join(c.sound_cues) if c.sound_cues else "-"
        tags: list[str] = []
        o = obs_by_id.get(c.id)
        if c.lyric:
            lmode = classify_lyric_cue(c, obs_by_id, cues)
            if lmode == "parody_or_rhyme_game":
                tags.append("LYRIC PLOT-RELEVANT | PARODY/RHYME CHAIN")
            elif lmode == "spoken_with_stray_note":
                tags.append("LYRIC PLOT-RELEVANT | SPEECH WITH STRAY ♪")
            elif lmode == "performed_song":
                tags.append("LYRIC PLOT-RELEVANT")
            else:
                tags.append("LYRIC INCIDENTAL")
        if ost_kind(c) == "credit" and not is_gag_credit(c.text):
            tags.append("ON-SCREEN CREDIT")
        elif c.source == "ost" or c.forced:
            tags.append("ON-SCREEN TEXT")
        if _NON_TRANSLATED.match(c.text):
            tags.append("SOUND-ONLY")
        if o and o.delivery:
            tags.append(o.delivery)
        du = d_units.get(c.id)
        sp = spans.get(c.id)
        if du is not None and len(du.cue_ids) >= 3:
            hold_tag = " | HOLD-ELLIPSIS" if c.id in du.hold_ellipsis_cue_ids else ""
            tags.append(f"DISCOURSE UNIT {du.unit_id}: {du.speech_act_type}{hold_tag}")
        elif sp is not None:
            tags.append(f"SPAN {sp.span_id}: {sp.policy}")
        tag = f" [{' | '.join(tags)}]" if tags else ""
        text = c.text.replace("\n", " / ")
        budget = _budget(c, cues[n + 1].start_ms if n + 1 < len(cues) else None)
        lines.append(
            f"  {c.id} | {_fmt_ms(c.start_ms)}-{_fmt_ms(c.end_ms)} | {speaker} | "
            f"{sound} | {dur:.1f} | {_aim(budget, aim_ratio)}/{budget} | turns={en_speaker_turns(c, observation=o)} | {text}{tag}"
        )
    return "\n".join(lines)

def _scene_prompt(
    scene: Scene,
    scene_cues: list[Cue],
    snapshot: Snapshot,
    observations: list[CueObservation],
    prev_ko_tail: list[tuple[str, str]],
    *,
    mode: str,
    callback_ko: dict[str, str],
    flagged: set[str] | None = None,
    aim_ratio: float,
    examples: Sequence[memory.Pair] = (),
    register_note: str = "",
    perception_ledger: PerceptionLedger | None = None,
    continuation_hints: Sequence[Cue] = (),
) -> str:
    obs_by_id = {o.cue_id: o for o in observations}
    blocks = [
        _render_snapshot(snapshot, callback_ko),
        _render_observations([c.id for c in scene_cues], obs_by_id),
        _render_perception([c.id for c in scene_cues], perception_ledger),
        _render_examples(examples),
        _render_prev_tail(prev_ko_tail),
        _render_virtual_continuations(continuation_hints),
        _render_cues(scene_cues, obs_by_id, aim_ratio),
    ]
    if flagged:
        rows = "\n".join(f"  {cid}" for cid in sorted(flagged))
        blocks.append("## RETRY (these cues could not be translated confidently without media)\n" + rows)
    if register_note:
        blocks.append(register_note)
    media_line = (
        "A low-res clip of this scene (plus 1s padding) is attached: watch and listen before translating."
        if mode == "av"
        else "No video/audio is attached. Translate from text and context only; "
        "set needs_media on any cue you cannot render faithfully without hearing or seeing it."
    )
    body = "\n\n".join(b for b in blocks if b)
    return f"Scene {scene.id} ({_fmt_ms(scene.start_ms)}-{_fmt_ms(scene.end_ms)}). {media_line}\n\n{body}"


# --------------------------------------------------------------- translation


def _untranslated_ids(
    scene_cues: list[Cue], observations: list[CueObservation]
) -> set[str]:
    """Cues that must keep ko "": sound-only text plus incidental lyrics."""
    obs_by_id = {o.cue_id: o for o in observations}
    skip = {c.id for c in scene_cues if _NON_TRANSLATED.match(c.text)}
    for c in scene_cues:
        if c.lyric and not is_translated_lyric(c, obs_by_id, scene_cues):
            skip.add(c.id)
        elif not c.lyric and classify_lyric_cue(c, obs_by_id, scene_cues) in (
            "omitted_vocable",
            "omitted_background_or_credit",
        ):
            skip.add(c.id)
    return skip


def _align(
    scene: Scene,
    scene_cues: list[Cue],
    result: SceneTranslation,
    mode: str,
    observations: list[CueObservation],
) -> SceneTranslation:
    """Reconcile model output with the input cue list.

    Sound-only cues (text '') and incidental lyrics are never translated -> ko ''.
    Missing cues get a synthesized empty entry (S5 will flag it); extra ids are
    dropped.
    """
    untranslated = _untranslated_ids(scene_cues, observations)
    by_id = {t.cue_id: t for t in result.cues}
    cues: list[TranslatedCue] = []
    for c in scene_cues:
        if c.id in untranslated:
            cues.append(
                TranslatedCue(
                    cue_id=c.id,
                    speaker=c.speaker_hint,
                    addressee=None,
                    speech_level=Register.UNKNOWN,
                    ko="",
                    needs_media=MediaNeed.NONE,
                    media_reason=None,
                )
            )
        else:
            t = by_id.get(c.id)
            if t is None:
                t = TranslatedCue(cue_id=c.id, ko="")
            cues.append(t)
    cues = [
        cue.model_copy(update={"model": result.model}) if cue.model is None and result.model else cue
        for cue in cues
    ]
    return result.model_copy(update={
        "scene_id": scene.id, "mode": mode, "cues": cues,
    })


def _bisect_scene_cues(
    gc: LLMClient,
    scene: Scene,
    subset: list[Cue],
    snapshot: Snapshot,
    observations: list[CueObservation],
    prev_ko_tail: list[tuple[str, str]],
    flagged: set[str] | None,
    *,
    callback_ko: dict[str, str],
    stage: str,
    system: str,
    aim_ratio: float,
    examples: Sequence[memory.Pair] = (),
) -> SceneTranslation:
    """Text-mode bisection of a blocked scene call; single blocked cues are
    left empty with media_reason 'blocked by provider' so S5 flags them."""
    if len(subset) == 1:
        logger.warning(
            "%s: cue %s blocked; leaving ko empty", stage, subset[0].id
        )
        return SceneTranslation(
            scene_id=scene.id,
            mode="text",
            cues=[
                TranslatedCue(cue_id=subset[0].id, ko="", media_reason=_BLOCKED_REASON)
            ],
        )
    mid = len(subset) // 2
    halves: list[SceneTranslation] = []
    for i, part in enumerate((subset[:mid], subset[mid:])):
        if not part:
            continue
        sub_flagged = {c.id for c in part} & flagged if flagged else None
        prompt = _scene_prompt(
            scene, part, snapshot, observations, prev_ko_tail,
            mode="text", callback_ko=callback_ko, flagged=sub_flagged or None,
            aim_ratio=aim_ratio, examples=examples,
        )
        try:
            raw = gc.generate(
                stage=f"{stage}{'ab'[i]}", prompt=prompt, schema=SceneTranslation,
                clips=[], system=system, thinking_level=_think(gc),
            )
            halves.append(_align(scene, part, raw, "text", observations))
        except GeminiBlocked:
            halves.append(
                _bisect_scene_cues(
                    gc, scene, part, snapshot, observations, prev_ko_tail,
                    flagged, callback_ko=callback_ko, stage=f"{stage}{'ab'[i]}",
                    system=system, aim_ratio=aim_ratio, examples=examples,
                )
            )
    return SceneTranslation(
        scene_id=scene.id,
        mode="text",
        cues=[t for half in halves for t in half.cues],
        errata=[e for half in halves for e in half.errata],
    )


def _degraded_translate(
    gc: LLMClient,
    scene: Scene,
    scene_cues: list[Cue],
    snapshot: Snapshot,
    observations: list[CueObservation],
    prev_ko_tail: list[tuple[str, str]],
    flagged: set[str] | None,
    *,
    retried_with_text: bool,
    callback_ko: dict[str, str],
    stage: str,
    system: str,
    aim_ratio: float,
    examples: Sequence[memory.Pair] = (),
) -> SceneTranslation:
    """Blocked scene call: one text-mode retry when media was attached, then
    cue bisection. Never aborts; never checkpoints the degraded result."""
    if retried_with_text:
        logger.warning(
            "%s: scene %s av call blocked; retrying text-only",
            stage, scene.id,
        )
        prompt = _scene_prompt(
            scene, scene_cues, snapshot, observations, prev_ko_tail,
            mode="text", callback_ko=callback_ko, flagged=flagged,
            aim_ratio=aim_ratio, examples=examples,
        )
        try:
            raw = gc.generate(
                stage=f"{stage}:text", prompt=prompt, schema=SceneTranslation,
                clips=[], system=system, thinking_level=_think(gc),
            )
            return _align(scene, scene_cues, raw, "text", observations)
        except GeminiBlocked:
            pass  # still blocked — bisect below
    return _bisect_scene_cues(
        gc, scene, scene_cues, snapshot, observations, prev_ko_tail, flagged,
        callback_ko=callback_ko, stage=stage, system=system, aim_ratio=aim_ratio,
        examples=examples,
    )


def _client_model(client: LLMClient) -> str:
    return getattr(client, "model", client.settings.model_main)


def _written(result: SceneTranslation, model: str, *, degraded: bool = False) -> SceneTranslation:
    """Writer identity and route degradation are assigned by orchestration."""
    reason = _BLOCKED_REASON if degraded else None
    return result.model_copy(update={
        "model": model,
        "cues": [cue.model_copy(update={
            "model": model, "degraded": degraded, "degradation_reason": reason,
        }) for cue in result.cues],
        "degraded": degraded,
        "degradation_reason": reason,
    })


def _is_degraded(result: SceneTranslation) -> bool:
    return bool(result.degraded or any(cue.degraded for cue in result.cues))


def _generation_settings(client: LLMClient) -> list[str]:
    return [
        f"media_resolution={client.settings.media_resolution}",
        f"service_tier={client.settings.service_tier!r}",
    ]


def image_digests(images: Sequence[Path]) -> list[str]:
    """Content identity, not mtime/path identity, for attached image parts."""
    return [input_hash(Path(path).read_bytes()) for path in images]


def _check_route(plan: RoutePlan | None, mode: str, primary: LLMClient, av: LLMClient | None) -> None:
    if plan is None:
        return
    if plan.primary_model.startswith("anthropic/"):
        raise CapabilityError("explicit routes cannot generate new frozen Anthropic baseline translations")
    resolved = resolve_route(
        plan.primary_model, plan.architecture, translation_mode=plan.translation_mode,
        media_modality=plan.media_modality, mixed_team=plan.mixed_team,
    )
    if resolved != plan or mode != plan.translation_mode.value:
        raise CapabilityError("execution mode does not match the resolved route")
    if _client_model(primary) != plan.primary_model:
        raise CapabilityError("text writer must retain the selected primary model")
    if primary.settings.model_main != plan.primary_model or primary.settings.model_light != plan.primary_model:
        raise CapabilityError("primary client settings must bind both main and light to the selected model")
    if plan.av_passes and (av is None or _client_model(av) != plan.primary_model):
        raise CapabilityError("native AV passes must retain the selected primary model")
    if av is not None:
        resolve_route(_client_model(av), Architecture.DIRECT_AV)


def _semantic_target(
    source: SemanticSource,
    draft: SemanticDraft,
    scene: Scene,
    scene_cues: list[Cue],
    model: str,
    observations: Sequence[CueObservation] = (),
) -> tuple[list[TranslatedCue], SemanticLineage]:
    """Project actual target fragments without copying source counts into voices."""
    if draft.scene_id != scene.id or source.scene_id != scene.id:
        raise ValueError("semantic writer changed the scene identity")
    if [cue.cue_id for cue in draft.cues] != [cue.id for cue in scene_cues]:
        raise ValueError("semantic writer changed cue identities or order")
    untranslated = _untranslated_ids(scene_cues, list(observations))
    source_refs = {act.ref for act in source.acts}
    acts_by_cue: dict[str, list[str]] = {}
    for act in source.acts:
        acts_by_cue.setdefault(act.cue_id, []).append(act.ref)

    def _expand_refs(values: Sequence[str]) -> list[str]:
        expanded: list[str] = []
        for ref in values:
            expanded.extend(acts_by_cue.get(ref, [ref]))
        return list(dict.fromkeys(expanded))

    expanded_omissions: list[OmittedAct] = []
    for item in draft.omissions:
        for ref in _expand_refs([item.ref]):
            expanded_omissions.append(item.model_copy(update={"ref": ref}))
    omitted = {item.ref: item for item in expanded_omissions}
    if len(omitted) != len(expanded_omissions):
        raise ValueError("duplicate semantic writer omission refs")
    omitted = {
        ref: item for ref, item in omitted.items()
        if not (ref not in source_refs and ref.partition(":")[0] in untranslated)
    }
    normalized_cues = [
        target.model_copy(update={
            "fragments": [
                fragment.model_copy(update={
                    "refs": _expand_refs(fragment.refs),
                    "prior_refs": _expand_refs(fragment.prior_refs),
                })
                for fragment in target.fragments
            ],
        })
        for target in draft.cues
    ]
    policy_omitted_refs = {act.ref for act in source.acts if act.cue_id in untranslated}
    by_id = {cue.id: cue for cue in scene_cues}
    for act in source.acts:
        if act.ref in policy_omitted_refs:
            reason = (
                "pipeline policy: normalized sound-only/empty source cue"
                if not by_id[act.cue_id].text.strip()
                else "pipeline policy: incidental lyric; no plot-relevant observation"
            )
            omitted[act.ref] = OmittedAct(ref=act.ref, reason=reason)
    source_cue_by_ref = {act.ref: act.cue_id for act in source.acts}
    retained_by_ref: dict[str, set[str]] = {}
    for target in normalized_cues:
        if target.cue_id in untranslated:
            continue
        for fragment in target.fragments:
            for ref in fragment.refs:
                retained_by_ref.setdefault(ref, set()).add(target.cue_id)
    fragments_by_cue = {}
    translations: list[TranslatedCue] = []
    for cue in normalized_cues:
        if cue.cue_id not in untranslated and any(
            policy_omitted_refs.intersection(fragment.refs) for fragment in cue.fragments
        ):
            raise ValueError("semantic writer retained a policy-omitted source act in another display")
        fragments: list[RetainedFragment] = []
        for fragment in (cue.fragments if cue.cue_id not in untranslated else ()):
            if any(line.lstrip().startswith("- ") for line in fragment.text.splitlines()):
                raise ValueError(
                    f"semantic writer inserted a display dash in {cue.cue_id}: "
                    "remove leading '- ' from fragment.text"
                )
            lines = [
                line.strip() for line in perception.semantic_lexical_text(fragment.text).splitlines()
                if line.strip()
            ]
            if not lines:
                if not fragment.refs and not fragment.prior_refs:
                    continue
                raise ValueError(
                    f"empty fragment.text in {cue.cue_id} carries refs {fragment.refs}: "
                    "move refs to the cue displaying the merged Korean or to omissions[], "
                    "and use fragments=[] for empty cues"
                )
            speaker = (
                None if fragment.speaker is None or fragment.speaker.strip().casefold()
                in {"", "unknown", "?", "unidentified"} else fragment.speaker.strip()
            )
            fragments.extend(
                fragment.model_copy(update={"text": line, "speaker": speaker})
                for line in lines
            )
        for fragment in fragments:
            if fragment.attribution != "declared":
                raise ValueError(
                    f"writer fragment in {cue.cue_id} must use "
                    "attribution='declared' and prior_refs=[]"
                )
            if fragment.prior_refs:
                if not set(fragment.prior_refs).issubset(source_refs):
                    raise ValueError(f"invalid prior refs: {cue.cue_id}")
                empty_refs = not fragment.refs
                candidates = [
                    ref for ref in fragment.prior_refs
                    if empty_refs or source_cue_by_ref[ref] == cue.cue_id
                    or (
                        ref not in omitted
                        and not any(owner != cue.cue_id for owner in retained_by_ref.get(ref, ()))
                    )
                ]
                fragment.refs = list(dict.fromkeys([*fragment.refs, *candidates]))
                fragment.prior_refs = []
                for ref in candidates:
                    retained_by_ref.setdefault(ref, set()).add(cue.cue_id)
            if not fragment.refs:
                raise ValueError(
                    f"non-empty fragment in {cue.cue_id} has empty refs=[]: "
                    "cite at least one source act.ref from source.acts"
                )
        fragments_by_cue[cue.cue_id] = fragments
        voices = {fragment.speaker for fragment in fragments if fragment.speaker}
        if len(voices) > 2:
            raise ValueError(f"semantic display contains more than two known voices: {cue.cue_id}")
        two_voices = len(voices) == 2 and all(fragment.speaker for fragment in fragments)
        ko = "\n".join(("- " if two_voices else "") + fragment.text for fragment in fragments)
        translations.append(TranslatedCue(
            cue_id=cue.cue_id, ko=ko,
            speaker=(
                by_id[cue.cue_id].speaker_hint if cue.cue_id in untranslated
                else next(iter(voices)) if len(voices) == 1 else None
            ),
            addressee=None if cue.cue_id in untranslated else cue.addressee,
            speech_level=Register.UNKNOWN if cue.cue_id in untranslated else cue.speech_level,
            needs_media=MediaNeed.NONE if cue.cue_id in untranslated else cue.needs_media,
            media_reason=None if cue.cue_id in untranslated else cue.media_reason,
            model=model, degraded=False,
        ))
    retained_refs = {
        ref for fragments in fragments_by_cue.values()
        for fragment in fragments for ref in fragment.refs
    }
    for ref in retained_refs:
        omitted.pop(ref, None)
    for act in source.acts:
        if act.ref in retained_refs or act.ref in omitted:
            continue
        cue = by_id[act.cue_id]
        if (
            (not cue.lyric and _may_stay_empty(cue))
            or act.essential is False
        ):
            omitted[act.ref] = OmittedAct(
                ref=act.ref,
                reason="pipeline policy: non-essential or sound/interjection cue left empty",
            )
    lineage = perception.validate_semantic_lineage(SemanticLineage(
        source=source, fragments_by_cue=fragments_by_cue, omissions=list(omitted.values()),
    ), scene_cues, translations)
    return translations, lineage


def _semantic_span_anchors(cues: Sequence[Cue]) -> dict[str, list[dict[str, int | str]]]:
    """Canonical lexical boundaries in original text, including Unicode/newlines."""
    return {
        cue.id: [
            {"start_char": match.start(), "end_char": match.end(), "text": match.group()}
            for match in re.finditer(r"\S+", cue.text)
        ]
        for cue in cues
    }


_SEMANTIC_SPEAKER_POLICY = (
    "- Retain a question's essential answer or agreement that changes the viewer's "
    "understanding as the responding speaker's own turn; its wording may compress. "
    "A word such as 'Yeah' or 'Mm-hmm' alone never decides omission. An incidental "
    "backchannel adding no needed meaning may be omitted. Dashes depend on actual "
    "retained voices, not source `turns`: only two distinct retained speakers in "
    "the same displayed cue get '- ' on both lines. One retained speaker across "
    "multiple acts or long speech, an omitted incidental reply, or speakers in "
    "separate display events do not force dashes; an unknown label alone does not "
    "establish a speaker change. Never erase a short retained turn to balance line "
    "lengths. This SemanticDraft rule overrides legacy source-turn formatting."
)


_SEMANTIC_EXTRACT_SYSTEM = (
    "Extract source speech acts and the minimum meaning relationships needed "
    "for faithful subtitle writing from the original English and attached "
    "bounded media. Do not write Korean or an alternate translation. Preserve "
    "uncertainty, conflicting readings and unknown speakers; no truth locks "
    "or future facts. Output SemanticSource JSON only."
)
_SEMANTIC_EXTRACT_PROMPT = (
    "Extract every non-whitespace source character into non-overlapping exact "
    "half-open character spans in source order. Preserve existing cue IDs and "
    "times. Within each cue number turns and acts from 1 contiguously; ref is "
    "'{cue_id}:t{turn_ordinal}:a{act_ordinal}'. One speaker may have several acts. "
    "Offsets count Unicode code points in the EXACT source Cue.text, never "
    "UTF-8 bytes or rendered cue rows. Choose span start/end from the supplied "
    "canonical token boundaries; span text includes the exact original "
    "intervening whitespace/newlines. Do not recount normalized text. A line "
    "break alone is not evidence of a speaker change. "
    "Do not infer speaker changes from act counts. Unknown speaker is null. "
    "Summarize each act's meaning with uncertainties alongside its exact text. "
    "Mark essential answers/consequential agreement as advisory essential=true; "
    "a token such as Yeah alone does not establish essentiality. Unknown "
    "essentiality is null; false is an explicit incidental claim. Record only "
    "needed elapsed/relative-time, question-answer, order and list relations, "
    "referencing existing acts; uncertain temporal readings remain uncertain. "
    "Keep source ambiguity/conflicts, not a general meaning graph. Return the "
    "exact scene_id {scene_id}.\n\n"
)
_SEMANTIC_WRITE_PROMPT = (
    "Write Korean subtitles as SemanticDraft after inspecting the SAME media "
    "again. The complete original English and context below remain available; "
    "the source extraction is advisory, never the sole source or a truth lock. "
    "Resolve conflicts against original evidence and report source_conflicts "
    "without silently overwriting or trusting interpretation claims. Keep "
    "elapsed/relative-time meanings, essential replies, list membership/order "
    "and ambiguity; official-style compression/paraphrase and justified "
    "repetition/split/merge are allowed. Each existing cue has one entry in "
    "source order; each fragment is actual Korean retained in that display "
    "with its source act refs and speaker known or null. Multiple refs per "
    "fragment and repeated refs across fragments are legal, not proof of "
    "semantic preservation. Use declared attribution with empty prior_refs "
    "only; every source act needs retained refs or an explicit omitted ref/"
    "reason. Omission of an incidental meaning-free backchannel is allowed; "
    "retain an essential answer/consequential agreement as its responder's "
    "own turn even when short. Never manufacture a reply or dash to pass "
    "coverage. Speaker labels/counts in source are NOT retained voice counts. "
    "No dash inside fragment text: rendering uses two distinct known retained "
    "speakers in the same display only; null/one voice/separate events never "
    "establish a two-speaker display. A fragment contains one voice's actual "
    "Korean text, no embedded newlines. Empty cues have fragments=[].\n\n"
)
_SEMANTIC_SOURCE_HEADING = "\n\n## ADVISORY SOURCE ACTS AND RELATIONS\n"


def _semantic_system(system: str) -> str:
    legacy_rule = next(
        line for line in _prompt("translate.md").splitlines()
        if "Dashes follow the source cue's `turns`:" in line
    )
    if legacy_rule not in system:
        raise ValueError("semantic writer system is missing its explicit legacy policy replacement")
    return system.replace(legacy_rule, _SEMANTIC_SPEAKER_POLICY).replace(
        "All Korean output in ko fields; reasoning fields in English.",
        "All Korean output in fragments[].text fields; reasoning fields in English.",
    )


def _semantic_policy_hash() -> str:
    policy = json.dumps({
        "policy_version": "semantic-policy-v1",
        "effective_translation_policy": _semantic_system(_prompt("translate.md")),
        "style_guide": _prompt("style_guide.md"),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(policy.encode("utf-8")).hexdigest()


def semantic_contract_hash() -> str:
    """Actual new-route prompts and schemas for caller-owned outer cache keys."""
    return input_hash(
        _SEMANTIC_EXTRACT_SYSTEM, _SEMANTIC_EXTRACT_PROMPT,
        _SEMANTIC_WRITE_PROMPT, _SEMANTIC_SOURCE_HEADING,
        _semantic_system(_prompt("translate.md")), _semantic_policy_hash(),
        _SEMANTIC_ROUTE_VERSION, _SEMANTIC_SOURCE_VERSION,
        _SEMANTIC_TARGET_VERSION, _SEMANTIC_PROMPT_VERSION,
        json.dumps(SemanticSource.model_json_schema(), sort_keys=True),
        json.dumps(SemanticDraft.model_json_schema(), sort_keys=True),
        json.dumps(SemanticSceneTranslation.model_json_schema(), sort_keys=True),
    )


class _SemanticAttributionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fragments: list[RetainedFragment]
    omissions: list[OmittedAct] = Field(default_factory=list)


def _validate_reattribution(
    raw: _SemanticAttributionDraft,
    source: SemanticSource,
    cues: Sequence[Cue],
    ko: str,
    previous: Sequence[RetainedFragment],
) -> tuple[list[RetainedFragment], list[OmittedAct]]:
    """Require unchanged actual output; speaker conflicts remain unverified."""
    cue_ids = {cue.id for cue in cues}
    acts = {act.ref: act for act in source.acts}
    prior = {ref for fragment in previous for ref in (*fragment.refs, *fragment.prior_refs)}
    allowed = {act.ref for act in source.acts if act.cue_id in cue_ids} | prior
    if not allowed.issubset(acts):
        raise ValueError("reattribution has unknown previous source refs")
    actual = perception.semantic_text_identity(ko)
    expected = "".join(perception.semantic_text_identity(fragment.text) for fragment in raw.fragments)
    if expected != actual:
        raise ValueError("semantic reattribution changed actual Korean text or order")
    fragments: list[RetainedFragment] = []
    retained: set[str] = set()
    pending: set[str] = set()
    for fragment in raw.fragments:
        lexical_text = perception.semantic_lexical_text(fragment.text).strip()
        if not lexical_text or fragment.text.lstrip().startswith("- "):
            raise ValueError("semantic reattribution invented empty text or a display dash")
        refs = set(fragment.refs)
        previous_refs = set(fragment.prior_refs)
        if len(refs) != len(fragment.refs) or not (refs | previous_refs).issubset(allowed):
            raise ValueError("semantic reattribution invented or duplicated source refs")
        speaker = (
            None if fragment.speaker is None or fragment.speaker.strip().casefold()
            in {"", "unknown", "?", "unidentified"} else fragment.speaker.strip()
        )
        source_speakers = {acts[ref].speaker for ref in refs if acts[ref].speaker}
        conflicted = (
            len(source_speakers) > 1
            or (speaker is not None and source_speakers and speaker not in source_speakers)
        )
        if fragment.attribution == "unverified" or conflicted or not refs:
            if not (refs | previous_refs):
                raise ValueError("unverified reattribution requires explicit prior refs")
            normalized = fragment.model_copy(update={
                "text": lexical_text, "speaker": None, "refs": [],
                "prior_refs": sorted(refs | previous_refs), "attribution": "unverified",
            })
            pending.update(normalized.prior_refs)
        else:
            normalized = fragment.model_copy(update={
                "text": lexical_text, "speaker": speaker,
                "prior_refs": [], "attribution": "declared",
            })
            retained.update(refs)
        fragments.append(normalized)
    omitted: set[str] = set()
    for omission in raw.omissions:
        if omission.ref not in allowed or omission.ref in omitted or not omission.reason.strip():
            raise ValueError("semantic reattribution has invalid omission records")
        omitted.add(omission.ref)
    if omitted & retained or prior - retained - pending - omitted:
        raise ValueError("semantic reattribution lost previous coverage without an omission record")
    return fragments, raw.omissions


def reattribute_semantic_fragments(
    client: LLMClient,
    source: SemanticSource,
    cues: Sequence[Cue],
    ko: str,
    previous: Sequence[RetainedFragment],
    *,
    stage: str,
    context_cues: Sequence[Cue] = (),
    output_context: Mapping[str, str] | None = None,
) -> SemanticAttributionResult:
    """Attribute immutable postwriter output using the selected primary only."""
    plan = resolve_route(_client_model(client))
    if plan.primary_model.startswith("anthropic/") or (
        client.settings.model_main != plan.primary_model
        or client.settings.model_light != plan.primary_model
    ):
        raise CapabilityError("semantic reattribution must retain the selected primary")
    policy_hash = _semantic_policy_hash()
    system = (
        "Audit source-to-target lineage, not translation. Attribute the EXACT "
        "already-written Korean lines in order. Never rewrite text, create an "
        "answer, or change media. Source meaning/speaker/essentiality claims are "
        "advisory: preserve uncertainty and disagreements as unverified with "
        "null speaker and prior_refs. Output the requested typed JSON only."
    )
    prompt = (
        "Return ordered retained-voice fragment partitions of the ACTUAL Korean, "
        "independent of its present line wrapping or display dashes. Concatenating "
        "fragment text in order, ignoring only whitespace and display '- '/♪, MUST "
        "equal the existing Korean exactly: do not change any lexical character, "
        "punctuation, Unicode character or order. A currently flattened line can "
        "contain two retained voices; partition its unchanged text when evidence "
        "supports this. Never infer two voices from source act counts. Preserve "
        "current source refs only when the actual "
        "fragment supports them; refs assert attribution, not semantic truth. "
        "If attribution is unsupported/conflicting use attribution=unverified, "
        "refs=[], speaker=null and prior_refs recording the unresolved prior "
        "source identity. Unknown speaker stays null. Provide explicit omitted "
        "ref/reason records for previously covered acts you now declare omitted. "
        "Do not add dashes or infer voice count from source turn count. Existing "
        "EN, source relations, ambiguity/conflicts and actual prior fragments are "
        "all available below; no extra media or alternate writer is used.\n\n"
        "## SOURCE CUES\n" + "\n".join(cue.model_dump_json() for cue in cues)
        + "\n\n## ADVISORY SOURCE\n" + source.model_dump_json()
        + "\n\n## PRIOR FRAGMENTS\n"
        + "\n".join(fragment.model_copy(update={
            "text": perception.semantic_lexical_text(fragment.text),
        }).model_dump_json() for fragment in previous)
        + "\n\n## ACTUAL IMMUTABLE KOREAN (display marks excluded)\n"
        + json.dumps(perception.semantic_lexical_text(ko), ensure_ascii=False)
    )
    if context_cues:
        prompt += "\n\n## FULL SCENE SOURCE CONTEXT\n" + "\n".join(
            cue.model_dump_json() for cue in context_cues
        )
    if output_context is not None:
        prompt += "\n\n## FULL ACTUAL OUTPUT CONTEXT\n" + json.dumps(
            {cue_id: perception.semantic_lexical_text(text) for cue_id, text in output_context.items()},
            ensure_ascii=False, sort_keys=True,
        )
    h = input_hash(
        prompt, system, plan.primary_model, _think(client) or "",
        *_generation_settings(client), policy_hash, "semantic-attribution-v1",
        json.dumps(_SemanticAttributionDraft.model_json_schema(), sort_keys=True),
    )
    attempt_prompt = prompt
    for attempt in range(_MAX_SEMANTIC_ATTEMPTS):
        raw = client.generate(
            stage=stage, prompt=attempt_prompt, schema=_SemanticAttributionDraft, clips=[],
            system=system, thinking_level=_think(client),
        )
        try:
            fragments, omissions = _validate_reattribution(raw, source, cues, ko, previous)
        except ValueError as exc:
            if attempt + 1 == _MAX_SEMANTIC_ATTEMPTS:
                raise
            attempt_prompt = (
                prompt + f"\n\n## PREVIOUS ATTEMPT STRUCTURAL VALIDATION ERROR\n{exc}\n"
                "Fix this structural validation error and return valid JSON."
            )
        else:
            break
    return SemanticAttributionResult(
        fragments=fragments, omissions=omissions,
        provenance=SemanticAttributionProvenance(
            stage=stage, model=plan.primary_model, input_hash=h,
            schema_version="semantic-attribution-v1", policy_hash=policy_hash,
        ),
    )


def reattribute_semantic_lineage(
    client: LLMClient,
    lineage: SemanticLineage,
    cues: Sequence[Cue],
    ko: Mapping[str, str],
    *,
    stage: str,
) -> SemanticLineage:
    """Reattribute changed scene outputs; exact-remapped fragments need no call."""
    by_id = {cue.id: cue for cue in cues}
    fragments_by_cue = dict(lineage.fragments_by_cue)
    omissions = {omission.ref: omission for omission in lineage.omissions}
    passes = list(lineage.attribution_passes)
    source_ids = {act.cue_id for act in lineage.source.acts} | set(fragments_by_cue)
    scene_cues = [cue for cue in cues if cue.id in source_ids]
    for cue_id, previous in fragments_by_cue.items():
        empty_rewrite = not ko.get(cue_id, "").strip() and any(
            fragment.refs or fragment.prior_refs for fragment in previous
        )
        if not empty_rewrite and not any(fragment.attribution == "unverified" for fragment in previous):
            continue
        if cue_id not in by_id or cue_id not in ko:
            raise ValueError("semantic reattribution requires actual source/output cue")
        result = reattribute_semantic_fragments(
            client, lineage.source, [by_id[cue_id]], ko[cue_id], previous,
            stage=f"{stage}:{cue_id}",
            context_cues=scene_cues, output_context={cid: ko[cid] for cid in source_ids if cid in ko},
        )
        fragments_by_cue[cue_id] = result.fragments
        omissions.update({omission.ref: omission for omission in result.omissions})
        passes.append(result.provenance)
    retained = {ref for fragments in fragments_by_cue.values() for fragment in fragments for ref in fragment.refs}
    updated = lineage.model_copy(update={
        "fragments_by_cue": fragments_by_cue,
        "omissions": [omission for ref, omission in omissions.items() if ref not in retained],
        "attribution_passes": passes,
        "semantic_assessment": "unassessed",
    })
    translated = [TranslatedCue(cue_id=cue_id, ko=ko[cue_id]) for cue_id in fragments_by_cue]
    return perception.validate_semantic_lineage(
        updated, scene_cues, translated,
    )


def _semantic_generation(
    client: LLMClient,
    plan: RoutePlan,
    scene: Scene,
    scene_cues: list[Cue],
    prompt: str,
    system: str,
    clips: list[Clip],
    images: Sequence[Path],
    *,
    checkpoint: Checkpoints | None,
    key: str,
    inputs_hash: str,
    observations: Sequence[CueObservation] = (),
    snapshot: Snapshot | None = None,
) -> SemanticSceneTranslation:
    """Same-primary extraction and writing over identical bounded media/context."""
    if _client_model(client) != plan.primary_model:
        raise CapabilityError("semantic passes must use the selected primary model")
    if plan.architecture is Architecture.SEMANTIC_CLIP:
        if not clips or images:
            raise CapabilityError("semantic-clip requires clips and cannot switch to images")
        resolve_route(_client_model(client), Architecture.SEMANTIC_CLIP)
    elif plan.architecture is Architecture.SEMANTIC_IMAGE:
        if clips or not images:
            raise CapabilityError("semantic-image requires actual keyframes, without clips")
        resolve_route(_client_model(client), Architecture.SEMANTIC_IMAGE)
        prompt = prompt.replace(
            "No video/audio is attached. Translate from text and context only; ",
            "Scene keyframes are attached, without audio/video. Use the images, original English and context; ",
        )
    else:
        raise CapabilityError("semantic generation requires an explicit semantic route")
    # The exact source strings/times coexist with the full original context in BOTH
    # passes. Rendered cue rows collapse whitespace and cannot anchor exact spans.
    context = prompt + "\n\n## EXACT SOURCE CUES\n" + "\n".join(
        cue.model_dump_json() for cue in scene_cues
    )
    context += "\n\n## CANONICAL SOURCE SPAN ANCHORS\n" + json.dumps(
        _semantic_span_anchors(scene_cues), ensure_ascii=False,
    )
    policy_hash = _semantic_policy_hash()
    image_hashes = image_digests(images)
    media_hash = input_hash(
        *[f"{clip.file.uri}|{clip.file.mime_type}|{clip.start_s:g}|{clip.end_s:g}|{clip.fps:g}"
          for clip in clips],
        *image_hashes,
    )
    common = [
        inputs_hash, context, plan.primary_model, media_hash,
        _think(client) or "", *_generation_settings(client),
        _SEMANTIC_ROUTE_VERSION, _SEMANTIC_PROMPT_VERSION, policy_hash,
    ]
    extract_system = _SEMANTIC_EXTRACT_SYSTEM
    extract_prompt = _SEMANTIC_EXTRACT_PROMPT.replace("{scene_id}", repr(scene.id)) + context
    extract_hash = input_hash(
        *common, extract_prompt, extract_system, _SEMANTIC_SOURCE_VERSION,
        json.dumps(SemanticSource.model_json_schema(), sort_keys=True),
    )
    extract_key = f"{key}:{plan.architecture.value}:extract"
    source = checkpoint.get(extract_key, extract_hash, SemanticSource) if checkpoint else None
    media_clips = list(clips)
    media_images = list(images)
    def _sanitized_context() -> str:
        clean_cues = [cue.model_copy(update={"raw": "", "sound_cues": []}) for cue in scene_cues]
        cue_rows = [cue.model_copy(update={"raw": cue.text, "sound_cues": []}) for cue in scene_cues]
        register_block = ""
        if snapshot is not None and snapshot.registers:
            register_lines = ["## REGISTERS (speaker>addressee)"]
            for pair, reg in sorted(snapshot.registers.items()):
                register_lines.extend(render_scene_register(pair, reg))
            register_block = "\n".join(register_lines) + "\n\n"
        clean_context = (
            f"Scene {scene.id} ({_fmt_ms(scene.start_ms)}-{_fmt_ms(scene.end_ms)}).\n\n"
            + register_block
            + _render_cues(cue_rows, {}, aim_ratio(client.settings))
            + "\n\n## EXACT SOURCE CUES\n"
            + "\n".join(cue.model_dump_json() for cue in clean_cues)
            + "\n\n## CANONICAL SOURCE SPAN ANCHORS\n"
            + json.dumps(_semantic_span_anchors(scene_cues), ensure_ascii=False)
        )
        return clean_context

    def _sanitized_source() -> SemanticSource | None:
        if source is None:
            return None
        return source.model_copy(update={
            "acts": [
                act.model_copy(update={"meaning": "source speech act", "uncertainty": []})
                for act in source.acts
            ],
            "relations": [
                relation.model_copy(update={"claim": relation.kind})
                for relation in source.relations
            ],
            "ambiguities": [],
            "conflicts": [],
        })

    def _sanitize_safety_prompt(
        attempt_prompt: str,
        schema: type[BaseModel],
    ) -> str:
        clean_context = _sanitized_context()
        marker = "\n\n## PREVIOUS ATTEMPT STRUCTURAL VALIDATION ERROR"
        feedback = attempt_prompt[attempt_prompt.index(marker):] if marker in attempt_prompt else ""
        if schema is SemanticDraft:
            safe_source = _sanitized_source()
            rebuilt = (
                _SEMANTIC_WRITE_PROMPT + clean_context + _SEMANTIC_SOURCE_HEADING
                + (safe_source.model_dump_json() if safe_source is not None else "")
            )
        else:
            rebuilt = _SEMANTIC_EXTRACT_PROMPT.replace("{scene_id}", repr(scene.id)) + clean_context
        return rebuilt + feedback

    def _generate_semantic(
        stage: str,
        attempt_prompt: str,
        schema: type[BaseModel],
        stage_system: str,
    ) -> BaseModel:
        try:
            return client.generate(
                stage=stage, prompt=attempt_prompt, schema=schema,
                clips=media_clips, system=stage_system,
                thinking_level=_think(client),
                **({"images": media_images} if media_images else {}),
            )
        except GeminiBlocked:
            retry_system = (
                stage_system if schema is SemanticSource
                else _semantic_safety_system(narrative_context(client.settings))
            )
            return client.generate(
                stage=stage, prompt=_sanitize_safety_prompt(attempt_prompt, schema),
                schema=schema, clips=media_clips, system=retry_system,
                thinking_level=_think(client),
                **({"images": media_images} if media_images else {}),
            )

    if source is None:
        attempt_prompt = extract_prompt
        for attempt in range(_MAX_SEMANTIC_ATTEMPTS):
            source = _generate_semantic(
                _SEMANTIC_EXTRACT_STAGE, attempt_prompt, SemanticSource, extract_system,
            )
            try:
                if source.scene_id != scene.id:
                    raise ValueError("semantic extraction changed the scene identity")
                source = perception.validate_semantic_source(source, scene_cues)
            except ValueError as exc:
                if attempt + 1 == _MAX_SEMANTIC_ATTEMPTS:
                    raise
                attempt_prompt = (
                    extract_prompt + f"\n\n## PREVIOUS ATTEMPT STRUCTURAL VALIDATION ERROR\n{exc}\n"
                    "Fix this structural validation error and return valid JSON."
                )
            else:
                break
        if checkpoint:
            checkpoint.put(extract_key, extract_hash, source)
    else:
        if source.scene_id != scene.id:
            raise ValueError("cached semantic extraction changed the scene identity")
        perception.validate_semantic_source(source, scene_cues)
    write_system = _semantic_system(system)
    write_prompt = _SEMANTIC_WRITE_PROMPT + context + _SEMANTIC_SOURCE_HEADING + source.model_dump_json()
    write_hash = input_hash(
        *common, extract_hash, source.model_dump_json(), write_prompt, write_system,
        _SEMANTIC_TARGET_VERSION, json.dumps(SemanticDraft.model_json_schema(), sort_keys=True),
    )
    write_key = f"{key}:{plan.architecture.value}:write"
    cached = checkpoint.get(write_key, write_hash, SemanticSceneTranslation) if checkpoint else None
    if cached is not None:
        perception.validate_semantic_lineage(cached.lineage, scene_cues, cached.cues)
        return cached
    if image_digests(images) != image_hashes:
        raise ValueError("semantic image inputs changed between passes")
    attempt_prompt = write_prompt
    for attempt in range(_MAX_SEMANTIC_ATTEMPTS):
        draft = _generate_semantic(
            _SEMANTIC_WRITE_STAGE, attempt_prompt, SemanticDraft, write_system,
        )
        try:
            translations, lineage = _semantic_target(
                source, draft, scene, scene_cues, plan.primary_model, observations,
            )
        except ValueError as exc:
            if attempt + 1 == _MAX_SEMANTIC_ATTEMPTS:
                raise
            attempt_prompt = (
                write_prompt + f"\n\n## PREVIOUS ATTEMPT STRUCTURAL VALIDATION ERROR\n{exc}\n"
                "Fix this structural validation error and return valid JSON."
            )
        else:
            break
    result = SemanticSceneTranslation(
        scene_id=scene.id, mode="av" if clips else "text",
        cues=[cue.model_copy(update={"degraded": False, "degradation_reason": None})
              for cue in translations],
        errata=draft.errata, model=plan.primary_model,
        degraded=False, degradation_reason=None, lineage=lineage,
        provenance=SemanticProvenance(
            route_variant=plan.architecture.value, model=plan.primary_model,
            extract_stage=_SEMANTIC_EXTRACT_STAGE, write_stage=_SEMANTIC_WRITE_STAGE,
            extract_hash=extract_hash, write_hash=write_hash, media_hash=media_hash,
            route_version=_SEMANTIC_ROUTE_VERSION,
            source_schema_version=_SEMANTIC_SOURCE_VERSION,
            target_schema_version=_SEMANTIC_TARGET_VERSION,
            prompt_version=_SEMANTIC_PROMPT_VERSION, policy_hash=policy_hash,
        ),
    )
    if checkpoint:
        checkpoint.put(write_key, write_hash, result)
    return result


def _route_generation(
    client: LLMClient, primary: LLMClient, plan: RoutePlan | None,
    scene: Scene, scene_cues: list[Cue], prompt: str, system: str,
    clips: list[Clip], images: Sequence[Path],
    *, checkpoint: Checkpoints | None = None, key: str = "", inputs_hash: str = "",
    observations: Sequence[CueObservation] = (),
    snapshot: Snapshot | None = None,
) -> SceneTranslation:
    """Ground AV2 and pure-team escalation in typed evidence, never another draft."""
    if plan is not None and plan.semantic_passes:
        return _semantic_generation(
            client, plan, scene, scene_cues, prompt, system, clips, images,
            checkpoint=checkpoint, key=key, inputs_hash=inputs_hash,
            observations=observations,
            snapshot=snapshot,
        )
    if clips and plan is not None:
        resolve_route(_client_model(client), Architecture.DIRECT_AV)
    evidence_only = bool(clips and plan is not None and (
        plan.av_passes == 2 or (not plan.mixed_team and not plan.av_passes)
    ))
    if evidence_only:
        evidence_prompt = (
            "Inspect only the attached bounded clip and listed source cues. Return "
            "cue-keyed advisory perception: speaker/addressee candidates, delivery, "
            "prosody, visual deixis, ambiguity, confidence and direct evidence. "
            "Do not translate to Korean, infer future events, or reveal facts before "
            "their source cue. Omit unsupported observations.\n\n" + prompt
        )
        evidence_system = (
            "You are a bounded audiovisual perception assistant. Return grounded "
            "cue-level observations, not Korean translations. Preserve uncertainty, "
            "source cue identities, and reveal timing; never infer future events."
        )
        evidence_hash = input_hash(inputs_hash, evidence_prompt, evidence_system, "perception-ledger-v1")
        evidence = checkpoint.get(key + ":perception", evidence_hash, PerceptionLedger) if checkpoint else None
        if evidence is None:
            evidence = client.generate(
                stage=f"{_STAGE}:perception", prompt=evidence_prompt,
                schema=PerceptionLedger, clips=clips, system=evidence_system,
                thinking_level=_think(client),
            )
            evidence = perception.normalize_perception_ledger(
                evidence, scene_cues,
                window_start_ms=int(clips[0].start_s * 1000),
                window_end_ms=int(clips[0].end_s * 1000),
            )
            if checkpoint:
                checkpoint.put(key + ":perception", evidence_hash, evidence)
        prompt += (
            "\n\n## BOUNDED AV EVIDENCE (advisory; preserve uncertainty and "
            "cue-local reveal timing)\n" + evidence.model_dump_json()
        )
        if plan.av_passes == 2:
            prompt += "\nRealize the final Korean subtitles after watching the same clip again."
        else:
            client = primary
            clips = []
            prompt += "\nNo media is attached to this realization; use only the supplied evidence."
    final_hash = (
        input_hash(inputs_hash, prompt, _client_model(client), _think(client) or "",
                   *_generation_settings(client))
        if evidence_only else inputs_hash
    )
    cached = checkpoint.get(key, final_hash, SceneTranslation) if checkpoint else None
    if cached is not None and not (plan is not None and _is_degraded(cached)):
        return cached.model_copy(update={"mode": "av" if clips else "text"}) if plan is None else cached
    kwargs = {"images": list(images)} if images else {}
    raw = client.generate(
        stage=_STAGE, prompt=prompt, schema=SceneTranslation,
        clips=clips, system=system, thinking_level=_think(client), **kwargs,
    )
    actual_mode = "av" if clips else "text"
    result = _align(
        scene, scene_cues, _written(raw, _client_model(client)) if plan else raw,
        actual_mode, list(observations),
    )
    if checkpoint:
        checkpoint.put(key, final_hash, result)
    return result


def translate_scene(
    ws: Workspace,
    gc: GeminiClient,
    scene: Scene,
    cues_by_id: dict[str, Cue],
    snapshot: Snapshot,
    observations_for_scene: list[CueObservation],
    prev_ko_tail: list[tuple[str, str]],
    *,
    mode: str,
    proxy: UploadedFile | None,
    examples: Sequence[memory.Pair] = (),
    perception_ledger: PerceptionLedger | None = None,
    continuation_hints: Sequence[Cue] = (),
    av_client: LLMClient | None = None,
    route_plan: RoutePlan | None = None,
    images: Sequence[Path] | None = None,
    text_client_override: LLMClient | None = None,
    checkpoint_key: str | None = None,
) -> SceneTranslation:
    """Translate one scene with an optional dedicated native-AV client."""
    ratio = aim_ratio(gc.settings)
    narrative = narrative_context(gc.settings)
    snapshot = snapshot.for_prompt(narrative)
    system = _system(narrative)
    scene_cues = [cues_by_id[cid] for cid in scene.cue_ids if cid in cues_by_id]
    prompt = _scene_prompt(
        scene, scene_cues, snapshot, observations_for_scene, prev_ko_tail,
        mode=mode, callback_ko={}, aim_ratio=ratio, examples=examples,
        perception_ledger=perception_ledger, continuation_hints=continuation_hints,
    )
    clips: list[Clip] = []
    if mode == "av" and proxy is not None:
        clips = [
            Clip(
                file=proxy,
                start_s=max(0.0, scene.start_ms / 1000.0 - CLIP_PAD_S),
                end_s=scene.end_ms / 1000.0 + CLIP_PAD_S,
            )
        ]
    if text_client_override is not None:
        tc = text_client_override
    elif route_plan is not None:
        settings = replace(
            gc.settings, model_main=route_plan.primary_model,
            model_light=route_plan.primary_model, model_translate=route_plan.primary_model,
            model_text=route_plan.primary_model,
        )
        tc = make_client(ws, settings, route_plan.primary_model)
    else:
        tc, _ = _text_client(ws, gc)
    av = av_client or (tc if route_plan is not None and route_plan.av_passes else gc)
    _check_route(route_plan, mode, tc, av)
    if route_plan is not None and mode == "av" and not clips:
        raise CapabilityError("native AV route requires an uploaded scene clip")
    image_paths = list(images or ())
    if route_plan is not None and route_plan.architecture in {
        Architecture.TEXT_SNAPSHOT, Architecture.SEMANTIC_IMAGE,
    } and not image_paths:
        raise CapabilityError("text-snapshot requires actual scene image inputs")
    if image_paths:
        prompt += "\nAttached images are bounded scene keyframes; do not infer future events."
    client = av if clips else tc
    cp = Checkpoints(ws, "translate") if checkpoint_key else None
    h = input_hash(
        prompt, system, _client_model(client), _think(client) or "",
        *[f"{clip.file.uri} {clip.start_s:g}-{clip.end_s:g} @{clip.fps:g}fps" for clip in clips],
        *image_digests(image_paths),
        *([str(route_plan.to_dict())] if route_plan is not None else []),
        *(_generation_settings(client) if route_plan is not None else []),
    )
    try:
        result = _route_generation(
            client, tc, route_plan, scene, scene_cues, prompt, system, clips, image_paths,
            checkpoint=cp, key=checkpoint_key or "", inputs_hash=h,
            observations=observations_for_scene,
            snapshot=snapshot,
        )
    except GeminiBlocked:
        if route_plan is not None and route_plan.semantic_passes:
            raise  # semantic failures remain visible; no modality/writer fallback
        degraded = _degraded_translate(
            tc, scene, scene_cues, snapshot, observations_for_scene,
            prev_ko_tail, None, retried_with_text=bool(clips),
            callback_ko={}, stage=_STAGE, system=system, aim_ratio=ratio, examples=examples,
        )
        return _written(degraded, _client_model(tc), degraded=True) if route_plan else degraded
    if isinstance(result, SemanticSceneTranslation):
        return result
    return _align(scene, scene_cues, result, result.mode, observations_for_scene)


def _text_client(ws: Workspace, gc: GeminiClient) -> tuple[LLMClient, str]:
    """(client, model id) for text-mode calls: settings.model_translate when set, else gc."""
    model = getattr(gc.settings, "model_translate", None)
    if model:
        return make_client(ws, gc.settings, model), model
    return gc, gc.settings.model_main


def _think(client: object) -> str | None:
    """S3 thinking level (Settings.translate_thinking); None = the client's default."""
    return getattr(getattr(client, "settings", None), "translate_thinking", None)


def _tail(st: SceneTranslation | None) -> list[tuple[str, str]]:
    if st is None:
        return []
    return [(t.cue_id, t.ko) for t in st.cues if t.ko][-KO_TAIL:]


def _scene_inputs(
    scene: Scene,
    cues_by_id: dict[str, Cue],
    snapshots: dict[str, Snapshot],
    obs_by_scene: dict[str, list[CueObservation]],
) -> tuple[list[Cue], Snapshot | None, list[CueObservation], str]:
    sc = [cues_by_id[cid] for cid in scene.cue_ids if cid in cues_by_id]
    snap = snapshots.get(scene.id)
    obs = obs_by_scene.get(scene.id, [])
    # raw-input portion of the checkpoint hash; _generate folds in the rendered
    # prompt, system template, model, and clip bounds on top of this
    h = input_hash(
        *[c.model_dump_json() for c in sc],
        snap.model_dump_json() if snap else "",
        *[o.model_dump_json() for o in obs],
    )
    return sc, snap, obs, h

def _continuation_hints(scene: Scene, cues: Sequence[Cue]) -> list[Cue]:
    """Return at most one nearby source cue on either side of a scene."""
    ordered = sorted(cues, key=lambda cue: (cue.start_ms, cue.end_ms, cue.id))
    ids = set(scene.cue_ids)
    positions = [i for i, cue in enumerate(ordered) if cue.id in ids]
    if not positions:
        return []
    first, last = min(positions), max(positions)
    neighbors: list[Cue] = []
    if first:
        previous = ordered[first - 1]
        if scene.start_ms - previous.end_ms <= VIRTUAL_CONTINUATION_GAP_MS:
            neighbors.append(previous)
    if last + 1 < len(ordered):
        following = ordered[last + 1]
        if following.start_ms - scene.end_ms <= VIRTUAL_CONTINUATION_GAP_MS:
            neighbors.append(following)
    return neighbors[:MAX_VIRTUAL_CONTINUATIONS]


def _escalation_cue_ids(
    scene_cues: Sequence[Cue], flagged: set[str], neighbors: int = ESCALATION_NEIGHBORS
) -> list[str]:
    """Expand flagged cue IDs by a small bounded window, preserving order."""
    if not flagged:
        return []
    selected: set[str] = set()
    for index, cue in enumerate(scene_cues):
        if cue.id not in flagged:
            continue
        lo = max(0, index - neighbors)
        hi = min(len(scene_cues), index + neighbors + 1)
        selected.update(c.id for c in scene_cues[lo:hi])
    return [cue.id for cue in scene_cues if cue.id in selected]


def _merge_targeted(
    base: SceneTranslation, targeted: SceneTranslation, selected: set[str]
) -> SceneTranslation:
    """Overlay only selected cue results onto the text translation."""
    replacement = {cue.cue_id: cue for cue in targeted.cues}
    cues = [
        replacement.get(cue.cue_id, cue) if cue.cue_id in selected else cue
        for cue in base.cues
    ]
    models = {cue.model for cue in cues if cue.model is not None}
    known_degradation = [cue.degraded for cue in cues if cue.degraded is not None]
    reason = next(
        (cue.degradation_reason for cue in cues if cue.degraded and cue.degradation_reason),
        None,
    )
    return base.model_copy(update={
        "mode": targeted.mode,
        "model": next(iter(models)) if len(models) == 1 else None,
        "cues": cues,
        "errata": [*base.errata, *targeted.errata],
        "degraded": any(known_degradation) if known_degradation else None,
        "degradation_reason": reason,
    })

def _retry_cue_ids(note: object, scene_cues: Sequence[Cue]) -> set[str]:
    """Extract the bounded register-retry target set from a retry note."""
    valid = {cue.id for cue in scene_cues}
    explicit = {str(cid) for cid in (getattr(note, "cue_ids", ()) or ()) if str(cid) in valid}
    if explicit:
        return explicit
    text = str(note)
    return {cid for cid in valid if re.search(rf"\b{re.escape(cid)}\b", text)}


def _luna_flags(
    luna: LLMClient | None,
    scene: Scene,
    scene_cues: Sequence[Cue],
    translation: SceneTranslation,
) -> set[str]:
    """Ask an optional Luna client for cue-level consistency concerns.

    This route is advisory in what it returns, but provider, budget, and
    schema failures must remain visible to the caller instead of becoming a
    false-success empty flag set.
    """
    if luna is None:
        return set()
    ko_by_id = {cue.cue_id: cue.ko for cue in translation.cues}
    rows = "\n".join(
        f"{cue.id}: EN={cue.text.replace(chr(10), ' / ')} | KO={ko_by_id.get(cue.id, '')}"
        for cue in scene_cues
    )
    prompt = (
        f"Scene {scene.id}. Flag only cue IDs with a likely meaning, referent, "
        "speaker, or continuity problem. Do not rewrite translations.\n"
        "Return an empty list when uncertain.\n\n" + rows
    )
    result = luna.generate(
        stage=f"{_STAGE}:luna",
        prompt=prompt,
        schema=_LunaCheck,
        clips=[],
        system="You are a conservative subtitle consistency checker.",
        thinking_level=_think(luna),
    )
    valid = {cue.id for cue in scene_cues}
    return set(result.flagged_cue_ids) & valid


def translate_video(
    ws: Workspace,
    gc: GeminiClient,
    video: VideoInfo,
    scenes: list[Scene],
    cues: list[Cue],
    snapshots: dict[str, Snapshot],
    observations: Any,
    proxy: UploadedFile | None,
    *,
    mode: str,
    register_check: Callable[[SceneTranslation], str] | None = None,
    perception_ledger: PerceptionLedger | None = None,
    luna: LLMClient | None = None,
    text_client_override: LLMClient | None = None,
    av_client: LLMClient | None = None,
    route_tag: str = "",
    route_plan: RoutePlan | None = None,
    scene_images: Mapping[str, Sequence[Path]] | None = None,
) -> dict[str, SceneTranslation]:
    """Translate a video with optional DPAF perception and dedicated route clients."""
    if mode not in {"text", "av", "cascade", "dpaf", "mixed"}:
        raise ValueError(f"mode must be text|av|cascade|dpaf|mixed, got {mode!r}")
    cascade_mode = mode in {"cascade", "dpaf", "mixed"}

    if isinstance(observations, PerceptionLedger):
        perception_ledger = perception_ledger or observations
        observation_items: Sequence[Any] = ()
    elif isinstance(observations, dict) and "cues" in observations:
        perception_ledger = perception_ledger or PerceptionLedger.model_validate(observations)
        observation_items = ()
    else:
        observation_items = observations or ()

    cues_by_id = {c.id: c for c in cues}
    # observations may arrive flat or as ChunkObservations; normalize to per-scene
    flat_obs: list[CueObservation] = []
    for o in observation_items:
        if isinstance(o, CueObservation):
            flat_obs.append(o)
        elif hasattr(o, "cues"):  # ChunkObservations
            flat_obs.extend(o.cues)
    scene_of: dict[str, str] = {cid: s.id for s in scenes for cid in s.cue_ids}
    obs_by_scene: dict[str, list[CueObservation]] = {}
    for o in flat_obs:
        sid = scene_of.get(o.cue_id)
        if sid:
            obs_by_scene.setdefault(sid, []).append(o)

    cp = Checkpoints(ws, "translate")
    if text_client_override is not None:
        tc = text_client_override
        text_model = getattr(
            tc,
            "model",
            getattr(getattr(tc, "settings", None), "model_main", gc.settings.model_main),
        )
    elif route_plan is None:
        tc, text_model = _text_client(ws, gc)
    else:
        tc, text_model = gc, route_plan.primary_model
    if route_plan is not None and text_client_override is None:
        primary_settings = replace(
            gc.settings, model_main=route_plan.primary_model,
            model_light=route_plan.primary_model, model_translate=route_plan.primary_model,
            model_text=route_plan.primary_model,
        )
        tc = make_client(ws, primary_settings, route_plan.primary_model)
        text_model = route_plan.primary_model
    if route_plan is not None and route_plan.av_passes and av_client is None:
        av_client = tc
    _check_route(route_plan, mode, tc, av_client)
    if route_plan is not None and route_plan.av_passes and proxy is None:
        raise CapabilityError("native AV route requires an uploaded scene clip")
    image_inputs = dict(scene_images or {})
    if route_plan is not None and route_plan.architecture in {
        Architecture.TEXT_SNAPSHOT, Architecture.CASCADE, Architecture.MIXED,
        Architecture.SEMANTIC_IMAGE,
    }:
        for scene in scenes:
            if scene.id not in image_inputs:
                image_inputs[scene.id] = media.scene_keyframes(ws, video, scene)
            if not image_inputs[scene.id]:
                raise CapabilityError(f"scene {scene.id} has no actual snapshot images")
    narrative = narrative_context(gc.settings)
    system = _system(narrative)
    ratio = aim_ratio(gc.settings)
    snapshots = snapshots or {}
    results: dict[str, SceneTranslation] = {}
    exclude_video = memory.base_video_id(ws.video_id)

    def _callback_ko_for(scene: Scene) -> dict[str, str]:
        """motif -> established Korean line, from already-translated setup cues."""
        ko_by_id = {t.cue_id: t.ko for st in results.values() for t in st.cues}
        snap = snapshots.get(scene.id)
        if snap is None:
            return {}
        out: dict[str, str] = {}
        for cb in snap.callbacks_live:
            for cid in cb.setup_cues:
                if ko_by_id.get(cid):
                    out[cb.motif] = ko_by_id[cid]
                    break
        return out

    def _generate(
        scene: Scene,
        gen_mode: str,
        flagged: set[str] | None,
        prev_ko_tail: list[tuple[str, str]],
        register_note: str = "",
        selected_cue_ids: Sequence[str] | None = None,
    ) -> SceneTranslation:
        full_scene_cues, snap, obs, inputs_h = _scene_inputs(
            scene, cues_by_id, snapshots, obs_by_scene)
        selected = set(selected_cue_ids or ())
        scene_cues = (
            [cue for cue in full_scene_cues if cue.id in selected]
            if selected
            else full_scene_cues
        )
        callback_ko = _callback_ko_for(scene)
        view = (snap or Snapshot(scene_id=scene.id)).for_prompt(narrative)
        examples = scene_examples(gc.settings, scene_cues, exclude_video)
        prompt = _scene_prompt(
            scene, scene_cues, view, obs, prev_ko_tail,
            mode=gen_mode, callback_ko=callback_ko, flagged=flagged, aim_ratio=ratio,
            examples=examples, register_note=register_note,
            perception_ledger=perception_ledger,
            continuation_hints=(
                _continuation_hints(scene, cues) if perception_ledger is not None else ()
            ),
        )
        images = list(image_inputs.get(scene.id, ())) if gen_mode == "text" else []
        if images:
            prompt += "\nAttached images are bounded scene keyframes; do not infer future events."
        clips: list[Clip] = []
        if gen_mode == "av" and proxy is not None and scene_cues:
            if selected_cue_ids is None:
                clip_start, clip_end = scene.start_ms, scene.end_ms
            else:
                clip_start = min(c.start_ms for c in scene_cues)
                clip_end = max(c.end_ms for c in scene_cues)
            clips = [
                Clip(
                    file=proxy,
                    start_s=max(0.0, clip_start / 1000 - CLIP_PAD_S),
                    end_s=clip_end / 1000 + CLIP_PAD_S,
                )
            ]
        if gen_mode == "av" and route_plan is not None and not clips:
            raise CapabilityError("AV escalation requires an uploaded scene clip")
        client = (av_client or gc) if clips else tc
        if clips:
            model = getattr(
                client,
                "model",
                getattr(getattr(client, "settings", None), "model_main", text_model),
            )
        else:
            model = getattr(client, "model", text_model)
        hash_parts: list[str] = [
            inputs_h, narrative, prompt, system, model, _think(client) or "",
        ]
        if route_tag:
            hash_parts.append(route_tag)
        if route_plan is not None:
            hash_parts.append(str(route_plan.to_dict()))
            hash_parts.extend(_generation_settings(client))
        hash_parts.extend(image_digests(images))
        hash_parts.extend(
            [
                *([c.id for c in scene_cues] if selected else []),
                *[f"{c.file.uri} {c.start_s:g}-{c.end_s:g} @{c.fps:g}fps" for c in clips],
                *[f"memory:{p.id}" for p in examples],
            ]
        )
        h = input_hash(*hash_parts)
        key = f"{scene.id}:{gen_mode}" + (":register" if register_note else "")
        if selected:
            key += ":window=" + ",".join(c.id for c in scene_cues)
        try:
            st = _route_generation(
                client, tc, route_plan, scene, scene_cues, prompt, system, clips, images,
                checkpoint=cp, key=key, inputs_hash=h,
                observations=obs,
                snapshot=view,
            )
        except GeminiBlocked:
            if route_plan is not None and route_plan.semantic_passes:
                raise  # semantic failures remain visible; never use legacy fallback
            degraded = _degraded_translate(
                tc, scene, scene_cues, view, obs,
                prev_ko_tail, flagged, retried_with_text=bool(clips),
                callback_ko=callback_ko, stage=f"{_STAGE}:{scene.id}",
                system=system, aim_ratio=ratio, examples=examples,
            )
            return _written(degraded, text_model, degraded=True) if route_plan else degraded
        if isinstance(st, SemanticSceneTranslation):
            return st
        return _align(scene, scene_cues, st, st.mode, obs)

    def run_act(act_scenes: list[Scene]) -> None:
        """Sequential inside an act: each scene sees the previous scene's KO tail.
        The first scene of an act starts cold; review_windows covers act seams."""
        prev_st: SceneTranslation | None = None
        for s in act_scenes:
            first_mode = "av" if mode == "av" else "text"
            st = _generate(s, first_mode, None, _tail(prev_st))
            scene_cues = [cues_by_id[cid] for cid in s.cue_ids if cid in cues_by_id]
            luna_flagged = _luna_flags(luna, s, scene_cues, st) if cascade_mode else set()
            media_flagged = {
                t.cue_id for t in st.cues if t.needs_media != MediaNeed.NONE
            }
            flagged = media_flagged | luna_flagged
            if cascade_mode and proxy is not None and flagged:
                selected_ids = _escalation_cue_ids(scene_cues, flagged)
                targeted = _generate(
                    s, "av", flagged, _tail(prev_st), selected_cue_ids=selected_ids
                )
                st = _merge_targeted(st, targeted, flagged & set(selected_ids))
            if register_check is not None and (note := register_check(st)):
                if route_plan is not None and route_plan.semantic_passes:
                    # Re-run the whole scene in the SAME modality, preserving its
                    # source/retained sidecar; a targeted legacy overlay loses it.
                    st = _generate(s, first_mode, None, _tail(prev_st), str(note))
                    results[s.id] = st
                    prev_st = st
                    continue
                retry_ids = _retry_cue_ids(note, scene_cues)
                if retry_ids:
                    retry = _generate(
                        s, "text", None, _tail(prev_st), str(note),
                        selected_cue_ids=[cue.id for cue in scene_cues if cue.id in retry_ids],
                    )
                    st = _merge_targeted(st, retry, retry_ids)
                elif st.mode != "av":
                    st = _generate(s, "text", None, _tail(prev_st), str(note))
            results[s.id] = st
            prev_st = st

    # acts run in parallel threads (max 4); scenes inside an act run sequentially
    acts: list[list[Scene]] = []
    for s in scenes:
        if not acts or acts[-1][0].act != s.act:
            acts.append([s])
        else:
            acts[-1].append(s)

    if len(acts) <= 1:
        for act_scenes in acts:
            run_act(act_scenes)
    else:
        with ThreadPoolExecutor(max_workers=min(ACT_WORKERS, len(acts))) as ex:
            futs = [ex.submit(run_act, sc) for sc in acts]
            for f in futs:
                f.result()

    return {s.id: results[s.id] for s in scenes if s.id in results}


# -------------------------------------------------------------- on-screen text


def _norm_text(t: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", t.lower())


def _words(t: str) -> list[str]:
    return re.findall(r"[0-9a-z가-힣]+", t.lower())


#: kinds never subtitled: logos/watermarks and scrolling end-credit name lists
_OST_SKIP_KINDS = frozenset({"logo", "scroll"})
#: kinds a Korean release subtitles even when the viewing pass left plot_critical unset
_OST_ALWAYS_KINDS = frozenset({"title", "credit", "gag_credit", "caption", "dedication", "spoken", "message", "ui", "note", "headline"})
_OST_SIGNAL_KINDS = frozenset({"sign", "message", "ui", "note", "headline", "gag_credit", "caption", "title", "dedication"})
_AMBIENT_SINGLE_WORDS = frozenset({
    "exit", "stop", "open", "closed", "push", "pull", "men", "women", "restroom", "wc", "in", "out",
})

def _split_kind(text: str) -> tuple[str | None, str]:
    m = OST_KIND.match(text)
    if not m:
        return None, text.strip()
    return m.group(1).strip().lower(), text[m.end():].strip()


def _same_text(a: str, b: str) -> bool:
    na, nb = _norm_text(a), _norm_text(b)
    if not na or not nb:
        return False
    return na == nb or SequenceMatcher(None, na, nb).ratio() >= 0.85


def _carried_by(text: str, cue_texts: list[str]) -> bool:
    """True when the cue texts together carry *text*: most of its words, or
    near-identical spelling (OCR/ASR noise) in one cue."""
    words = set(_words(text))
    if not words or not cue_texts:
        return False
    have = {w for t in cue_texts for w in _words(t)}
    req_ratio = 0.85 if len(words) >= 4 else OST_COVER_RATIO
    if len(words & have) / len(words) >= req_ratio:
        return True
    nt = _norm_text(text)
    return any(SequenceMatcher(None, nt, _norm_text(t)).ratio() >= 0.8 for t in cue_texts)

def _is_credit_scroll(kind: str | None, text: str) -> bool:
    """A 'credit' that is really a list of names (crawl/roll), not a card."""
    if kind != "credit":
        return False
    lines = [s for s in re.split(r"\s*(?:\n|/|\|)\s*", text) if s.strip()]
    return len(lines) > 3 or len(_words(text)) > 12


def _decode_mmss_ms(ms: int) -> int:
    """Convert a base-100 MMSS*1000 timestamp (e.g. 101000 for 01:01) to ms (61000)."""
    s, rem = divmod(ms, 1000)
    if s >= 100:
        mm, ss = divmod(s, 100)
        if 0 <= ss < 60:
            return (mm * 60 + ss) * 1000 + rem
    return ms


def _normalize_chunk_ost_mmss(items: list[OnScreenText], start_ms: int, end_ms: int) -> list[OnScreenText]:
    """Fix v01/chunk on-screen-text times where the model wrote MM:SS as MMSS*1000."""
    def _is_mmss_cand(it: OnScreenText) -> bool:
        return (
            it.covered_by_cue is None
            and 100_000 <= it.start_ms < 600_000
            and it.start_ms != end_ms
            and it.start_ms % 100 == 0
            and it.end_ms % 100 == 0
            and (it.start_ms // 1000) % 100 < 60
            and (it.end_ms // 1000) % 100 < 60
        )

    high = [
        it for it in items
        if it.covered_by_cue is None and 100_000 <= it.start_ms < 600_000 and it.start_ms != end_ms
    ]
    has_clamped_or_over = end_ms > 0 and any(
        (it.start_ms == end_ms and it.end_ms == end_ms) or it.start_ms > end_ms for it in items
    )
    mmss_cands = [it for it in high if _is_mmss_cand(it)]
    batch_mmss = start_ms == 0 and (
        (has_clamped_or_over and len(mmss_cands) >= 1)
        or (len(high) >= 2 and len(mmss_cands) == len(high))
    )
    out: list[OnScreenText] = []
    for it in items:
        s, e = it.start_ms, it.end_ms
        if end_ms > 0 and s == end_ms and e == end_ms:
            continue
        if end_ms > 0 and s > end_ms:
            ds, de = _decode_mmss_ms(s), _decode_mmss_ms(e)
            if start_ms <= ds <= end_ms:
                s, e = ds, max(ds, min(de, end_ms))
        elif batch_mmss and _is_mmss_cand(it):
            s, e = _decode_mmss_ms(s), _decode_mmss_ms(e)
        if (s, e) != (it.start_ms, it.end_ms):
            it = it.model_copy(update={"start_ms": s, "end_ms": max(e, s)})
        out.append(it)
    return out


def _dedupe_ost(items: list[OnScreenText]) -> list[tuple[str | None, OnScreenText]]:
    """Merge repeated reports of one appearance (chunk boundaries, bisected
    clips): same text, overlapping or within OST_MERGE_GAP_MS. Returns
    (kind tag, item with the tag stripped)."""
    merged: list[tuple[str | None, OnScreenText]] = []
    for it in sorted(items, key=lambda x: (x.start_ms, x.end_ms)):
        kind, text = _split_kind(it.text)
        cur = OnScreenText(
            start_ms=it.start_ms,
            end_ms=max(it.end_ms, it.start_ms),
            text=text,
            plot_critical=it.plot_critical,
            covered_by_cue=it.covered_by_cue,
            beats=list(it.beats),
        )
        for i, (k, m) in enumerate(merged):
            if cur.start_ms <= m.end_ms + OST_MERGE_GAP_MS and _same_text(m.text, cur.text):
                merged[i] = (
                    k or kind,
                    OnScreenText(
                        start_ms=min(m.start_ms, cur.start_ms),
                        end_ms=max(m.end_ms, cur.end_ms),
                        text=m.text,
                        plot_critical=m.plot_critical or cur.plot_critical,
                        covered_by_cue=m.covered_by_cue or cur.covered_by_cue,
                        beats=m.beats or cur.beats,
                    ),
                )
                break
        else:
            merged.append((kind, cur))
    return merged


_BUBBLE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])|(?<=\b(?:AM|PM))\s+(?=[A-Z])|(?<=\b(?:a\.m\.|p\.m\.))\s+(?=[A-Z])"
)
_GREETING_OPEN_RE = re.compile(
    r"^(?:hey|hi|hello)\b.*\b(?:checking in|it's me|just texting)\.?$",
    re.IGNORECASE,
)


def split_ost_beats(
    kind: str | None, ost: OnScreenText
) -> list[tuple[int, int, str, str | None]]:
    """Decompose a multi-bubble / multi-beat OnScreenText into timed visual beats.

    Returns list of ``(start_ms, end_ms, text, ost_kind)``. Single static signs,
    short notes (<= 4000 ms), and single-sentence captions stay a single beat
    clamped to ``[OST_MIN_MS, OST_MAX_MS]``. Multi-bubble threads spanning > 4000 ms
    are decomposed into sequential non-overlapping beats across ``[start_ms, end_ms]``.
    """
    if ost.beats:
        out_b: list[tuple[int, int, str, str | None]] = []
        for b in ost.beats:
            if not b.text.strip():
                continue
            bs = b.start_ms
            be = min(max(b.end_ms, bs + OST_MIN_MS), bs + OST_MAX_MS)
            out_b.append((bs, be, b.text.strip(), b.ost_kind or kind))
        if out_b:
            return out_b

    start = ost.start_ms
    span = ost.end_ms - start
    text = ost.text.strip()
    if span > 4000 and kind in {"sign", "message", "note", None}:
        raw_parts = [p.strip() for line in text.splitlines() for p in _BUBBLE_SPLIT_RE.split(line) if p.strip()]
        parts: list[str] = []
        idx = 0
        while idx < len(raw_parts):
            cur = raw_parts[idx]
            if idx + 1 < len(raw_parts) and _GREETING_OPEN_RE.match(cur) and len(cur) + len(raw_parts[idx + 1]) <= 68:
                parts.append(f"{cur} {raw_parts[idx + 1]}")
                idx += 2
            else:
                parts.append(cur)
                idx += 1
        n = len(parts)
        if n >= 2 and span >= n * OST_MIN_MS + (n - 1) * 80:
            weights = [max(len(p), 6) for p in parts]
            total_w = sum(weights)
            usable = span - (n - 1) * 80
            durs = [max(OST_MIN_MS, min(OST_MAX_MS, int(round(usable * w / total_w)))) for w in weights]
            scale = usable / max(sum(durs), 1)
            if scale < 1.0:
                durs = [max(OST_MIN_MS, int(d * scale)) for d in durs]
            beats_out: list[tuple[int, int, str, str | None]] = []
            cur_t = start
            for i, (p, d) in enumerate(zip(parts, durs)):
                b_end = ost.end_ms if i == n - 1 else min(ost.end_ms, cur_t + d)
                b_end = min(max(b_end, cur_t + OST_MIN_MS), cur_t + OST_MAX_MS)
                beats_out.append((cur_t, b_end, p, "message" if kind == "sign" else kind))
                cur_t = b_end + 80
            return beats_out

    end = min(max(ost.end_ms, start + OST_MIN_MS), start + OST_MAX_MS)
    return [(start, end, text, kind)]


def ost_cues(
    observations: Any, cues: list[Cue], scenes: list[Scene]
) -> list[Cue]:
    """New cues for on-screen text (and unsubtitled audible words) that no
    existing English cue carries."""
    items: list[OnScreenText] = []
    for o in observations or []:
        if isinstance(o, OnScreenText):
            items.append(o)
        elif hasattr(o, "on_screen_text"):
            c_start = getattr(o, "start_ms", 0) or 0
            c_end = getattr(o, "end_ms", 0) or 0
            items.extend(_normalize_chunk_ost_mmss(list(o.on_screen_text), c_start, c_end))
    by_id = {c.id: c for c in cues}
    new: list[Cue] = []
    for kind, ost in _dedupe_ost(items):
        norm_t = _norm_text(ost.text)
        if not norm_t:
            continue
        if kind in _OST_SKIP_KINDS or _is_credit_scroll(kind, ost.text):
            continue
        is_signal_sign = kind in _OST_SIGNAL_KINDS and norm_t not in _AMBIENT_SINGLE_WORDS
        if not (ost.plot_critical or kind in _OST_ALWAYS_KINDS or is_signal_sign):
            continue
        linked = by_id.get(ost.covered_by_cue or "")
        if linked is not None and _carried_by(ost.text, [linked.text]):
            continue
        near = [
            c.text
            for c in cues
            if c.start_ms - OST_OVERLAP_MS < ost.end_ms and ost.start_ms < c.end_ms + OST_OVERLAP_MS
        ]
        if _carried_by(ost.text, near):
            continue
        for b_start, b_end, b_text, b_kind in split_ost_beats(kind, ost):
            raw_tag = kind or b_kind
            new.append(
                Cue(
                    id=f"ost-{len(new) + 1:04d}",
                    start_ms=b_start,
                    end_ms=b_end,
                    text=b_text,
                    raw=f"[{raw_tag}] {b_text}" if raw_tag else b_text,
                    forced=(raw_tag != "spoken"),
                    source="ost",
                    ost_kind=b_kind or kind,
                )
            )
    return new


def merge_ost_cues_into_spine(
    existing_cues: list[Cue], observations: Any, scenes: list[Scene]
) -> list[Cue]:
    """Ensure all promotable observations (captions, gag credits, signs, messages)
    are present in the cue spine without duplicating existing ``ost-`` cues."""
    spoken = [c for c in existing_cues if not c.id.startswith("ost-") and c.source != "ost"]
    existing_ost = [c for c in existing_cues if c.id.startswith("ost-") or c.source == "ost"]
    promoted = ost_cues(observations, spoken, scenes)
    merged_ost = list(existing_ost)
    max_num = max(
        [int(c.id.split("-")[1]) for c in merged_ost if c.id.startswith("ost-") and c.id.split("-")[1].isdigit()]
        or [0]
    )
    for cand in promoted:
        already = any(
            min(e.end_ms, cand.end_ms) - max(e.start_ms, cand.start_ms) >= 150
            and (_same_text(e.text, cand.text) or _carried_by(cand.text, [e.text]))
            for e in merged_ost
        )
        if not already:
            max_num += 1
            merged_ost.append(cand.model_copy(update={"id": f"ost-{max_num:04d}"}))
    out = sorted([*spoken, *merged_ost], key=lambda c: (c.start_ms, c.end_ms, c.id))
    return out
