"""Causal Series-Level State Ledger (SeriesBible B_t) and left-fold monoid Φ.

Implements a bi-temporal, strictly causal 6-ledger state machine across
multi-episode and multi-season series:

    B_t = Φ(B_{t-1}, E_t),   B_0 = ∅

Core invariants:
1. Causal Filtration (F_t / G_{t,k} — Zero Spoiler Guarantee):
   Episode t may only observe entities, anonymous slots, dyadic registers,
   glossary terms, translation-memory lines, and plot facts with
   ``first_seen < coord(t)`` (and within episode t, ``cue < k``).
2. Streaming ≡ Batch Equivalence:
   Whether episodes are translated one-by-one as released (streaming) or
   submitted together in a multi-episode batch, Stage 2..6 executes as a
   strict left fold over episode order `(season, episode)`, producing
   bit-for-bit identical `B_t` and identical Korean subtitles.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from . import config
from .levels import addressed_levels, speaker_turns
from .models import (
    Callback,
    Character,
    ContextMap,
    Cue,
    EdgeLevel,
    GlossaryEntry,
    RegisterEdge,
    RegisterMap,
    Timeline,
    VoiceCard,
)
from .register import _GENERIC_NAME_WORDS, _name_tokens, binary, same_level
from .store import _atomic_write

_EP_PATTERNS = (
    re.compile(r"^(?P<series>.+?)[-_.\s]+[sS](?P<season>\d{1,2})[eE](?P<ep>\d{1,3})(?:[-_.\s].*)?$"),
    re.compile(r"^(?P<series>.+?)[-_.\s]+(?P<season>\d{1,2})x(?P<ep>\d{1,3})(?:[-_.\s].*)?$"),
    re.compile(r"^(?P<series>[a-z0-9-]+?)-e(?P<ep>\d{1,3})$"),
)

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with", "by",
    "is", "am", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "do", "does", "did", "have", "has", "had", "will", "would", "can",
    "could", "should", "not", "no", "yes", "yeah", "oh", "well", "just", "so", "what", "who",
    "where", "when", "why", "how", "all", "out", "up", "down", "about", "into", "over", "after",
})

_RECAP_HEADER_RE = re.compile(
    r"^(?:[\"'“”‘’\s-]*(?:previously\s+on[^.?!:\n]*[.?!:\n]+|지난\s*이야기[^\"\n]*[\"'”’\s/:\n-]+))+",
    re.IGNORECASE,
)

_KO_PARTICLES = (
    "에게서는", "한테서는", "에게서", "한테서", "으로서는", "로서는", "이라고는", "라고는",
    "이라는", "라는", "이라고", "라고", "에게는", "한테는", "께서는", "에서부터", "으로부터",
    "로부터", "에게도", "한테도", "이랑은", "하고는", "보다는", "처럼은", "마저는", "조차도",
    "에게", "한테", "께서", "에서", "으로", "이랑", "하고", "보다", "처럼", "마저", "조차",
    "까지", "부터", "밖에", "만의", "만은", "만도", "이나", "이란", "라든가", "이든가",
    "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도", "만", "야", "아", "께", "랑",
)

_ANON_MONIKER_RE = re.compile(
    r"\b(tie[- ]dye\s+guy|glitter\s+guy|mystery\s+(?:man|woman|caller)|hooded\s+(?:man|figure)|"
    r"killer|stalker|anonymous\s+[a-z]+)\b",
    re.IGNORECASE,
)

_SPOILER_PERSONA_RE = re.compile(
    r"(?:[^.;,|]*\b(?:secretly\s+the\s+killer|is\s+the\s+(?:killer|murderer|culprit|real\s+killer)|"
    r"killed\s+[a-z]+|murdered\s+[a-z]+|real\s+identity\s+is|actually\s+becky\s+butler|"
    r"poisoned\s+[a-z]+|mastermind\s+behind\s+the\s+murder)\b[^.;,|]*[.;,|]?)",
    re.IGNORECASE,
)


@dataclass(frozen=True, order=True)
class EpisodeCoord:
    season: int
    episode: int
    series_id: str = ""
    video_id: str = ""

    @property
    def pair(self) -> tuple[int, int]:
        return (self.season, self.episode)


def parse_episode_coord(
    video_id: str,
    *,
    series_id: str | None = None,
    season: int | None = None,
    episode: int | None = None,
) -> EpisodeCoord | None:
    """Extract `(series_id, season, episode)` from `video_id` or explicit overrides."""
    base = video_id.split("__", 1)[0].strip()
    if season is not None and episode is not None:
        sid = series_id or re.sub(r"[-_.\s]+[sS]\d+[eE]\d+.*$", "", base) or base
        return EpisodeCoord(season=int(season), episode=int(episode), series_id=sid.lower(), video_id=base)
    for pat in _EP_PATTERNS:
        m = pat.match(base)
        if m:
            gd = m.groupdict()
            sid = (series_id or gd.get("series") or base).strip().lower()
            s_num = int(gd.get("season") or 1)
            e_num = int(gd["ep"])
            return EpisodeCoord(season=s_num, episode=e_num, series_id=sid, video_id=base)
    if series_id:
        return EpisodeCoord(season=1, episode=1, series_id=series_id.strip().lower(), video_id=base)
    return None


class EntityRecord(BaseModel):
    """Ledger 1: Canonical recurring character identity and Korean spelling."""

    char_id: str
    en_name: str
    aliases: list[str] = Field(default_factory=list)
    ko_name: str
    ko_short: str = ""
    ko_last: str = ""
    lock: Literal["provisional", "hard"] = "provisional"
    persona_immutable: str = ""
    voice: VoiceCard | None = None
    first_seen: tuple[int, int] = (1, 1)
    episodes_seen: list[tuple[int, int]] = Field(default_factory=list)
    cue_count: int = 0


class AnonSlot(BaseModel):
    """Ledger 2: Unrevealed / masked character slot (e.g. Tie-Dye Guy, Glitter Guy)."""

    anon_id: str
    en_label: str
    ko_label: str
    first_seen: tuple[int, int] = (1, 1)
    revealed_as: str | None = None
    reveal_coord: tuple[int, int] | None = None
    reveal_cue: str | None = None


class DyadRecord(BaseModel):
    """Ledger 3: Directed interpersonal register edge across episodes."""

    speaker: str
    addressee: str
    relation: str = "peer"
    apparent_generation: str = "same"
    history: str = "none"
    stance: str = "civil"
    step: int = 8
    level: EdgeLevel = "haeyo"
    pronoun: Literal["너", "자네", "당신", "none"] = "none"
    vocative: str = "none"
    lock: Literal["provisional", "hard"] = "provisional"
    first_seen: tuple[int, int] = (1, 1)
    last_updated: tuple[int, int] = (1, 1)
    episodes_seen: list[tuple[int, int]] = Field(default_factory=list)


class GlossaryLock(BaseModel):
    """Ledger 4: In-universe proper nouns, show/podcast titles, products, and recurring motifs."""

    en: str
    ko: str
    note: str = ""
    category: str = "term"
    lock: Literal["provisional", "hard"] = "provisional"
    first_seen: tuple[int, int] = (1, 1)
    episodes_seen: list[tuple[int, int]] = Field(default_factory=list)


class TMRecord(BaseModel):
    """Ledger 5: Causal Episodic Translation Memory for recaps and callbacks."""

    ep_coord: tuple[int, int]
    video_id: str
    cue_id: str
    start_ms: int
    en_norm: str
    en_raw: str
    ko: str
    speaker: str | None = None
    addressee: str | None = None
    speech_level: str | None = None
    content_tokens: list[str] = Field(default_factory=list)


class CausalFact(BaseModel):
    """Ledger 6: Causal plot / twist fact with explicit reveal coordinate."""

    fact_id: str
    fact: str
    revealed_at_ep: tuple[int, int]
    revealed_at_cue: str
    known_by: list[str] = Field(default_factory=list)


class CuePin(BaseModel):
    """Per-cue deterministic series constraints applied during and after S3/S4."""

    cue_id: str
    is_recap: bool = False
    tm_ko: str | None = None
    tm_level: str | None = None
    tm_source_ep: tuple[int, int] | None = None
    required_names: dict[str, str] = Field(default_factory=dict)  # en_token -> locked ko form
    required_terms: dict[str, str] = Field(default_factory=dict)  # en_phrase -> locked ko form


class SeriesBible(BaseModel):
    """Complete causal state B_t after episode t."""

    series_id: str
    last_committed_ep: tuple[int, int] = (0, 0)
    entities: dict[str, EntityRecord] = Field(default_factory=dict)
    anon_slots: dict[str, AnonSlot] = Field(default_factory=dict)
    dyads: dict[str, DyadRecord] = Field(default_factory=dict)
    glossary: dict[str, GlossaryLock] = Field(default_factory=dict)
    tm: list[TMRecord] = Field(default_factory=list)
    facts: list[CausalFact] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            self.last_committed_ep == (0, 0)
            and not self.entities
            and not self.anon_slots
            and not self.dyads
            and not self.glossary
            and not self.tm
        )

    def causal_slice(self, before_ep: tuple[int, int]) -> SeriesBible:
        """Strictly filter B to information from episodes `< before_ep` (F_{t-1})."""
        ep = tuple(before_ep)
        entities: dict[str, EntityRecord] = {}
        for k, ent in sorted(self.entities.items()):
            if tuple(ent.first_seen) < ep:
                eps_seen = [tuple(x) for x in ent.episodes_seen if tuple(x) < ep]
                lock: Literal["provisional", "hard"] = (
                    "hard" if len(eps_seen) >= 2 or ent.lock == "hard" else "provisional"
                )
                entities[k] = ent.model_copy(update={"episodes_seen": eps_seen, "lock": lock})

        anon_slots: dict[str, AnonSlot] = {}
        for k, slot in sorted(self.anon_slots.items()):
            if tuple(slot.first_seen) < ep:
                if slot.reveal_coord is not None and tuple(slot.reveal_coord) >= ep:
                    slot = slot.model_copy(
                        update={"revealed_as": None, "reveal_coord": None, "reveal_cue": None}
                    )
                anon_slots[k] = slot

        dyads: dict[str, DyadRecord] = {}
        for k, d in sorted(self.dyads.items()):
            if tuple(d.first_seen) < ep:
                eps_seen = [tuple(x) for x in d.episodes_seen if tuple(x) < ep]
                dyads[k] = d.model_copy(
                    update={
                        "episodes_seen": eps_seen,
                        "last_updated": max(eps_seen) if eps_seen else d.first_seen,
                    }
                )

        glossary: dict[str, GlossaryLock] = {}
        for k, g in sorted(self.glossary.items()):
            if tuple(g.first_seen) < ep:
                eps_seen = [tuple(x) for x in g.episodes_seen if tuple(x) < ep]
                lock_g: Literal["provisional", "hard"] = (
                    "hard" if len(eps_seen) >= 2 or g.lock == "hard" else "provisional"
                )
                glossary[k] = g.model_copy(update={"episodes_seen": eps_seen, "lock": lock_g})

        tm = [r for r in self.tm if tuple(r.ep_coord) < ep]
        facts = [f for f in self.facts if tuple(f.revealed_at_ep) < ep]
        committed = [
            *(tuple(e.first_seen) for e in entities.values()),
            *(tuple(r.ep_coord) for r in tm),
        ]
        last_ep = max(committed) if committed else (0, 0)
        sliced = SeriesBible(
            series_id=self.series_id,
            last_committed_ep=last_ep,
            entities=entities,
            anon_slots=anon_slots,
            dyads=dyads,
            glossary=glossary,
            tm=tm,
            facts=facts,
        )
        assert_causal_integrity(sliced, ep)
        return sliced

    def fingerprint(self) -> str:
        """Deterministic SHA-256 digest of the bible state."""
        if self.is_empty():
            return ""
        payload = json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def assert_causal_integrity(bible: SeriesBible, current_ep: tuple[int, int]) -> None:
    """Raise AssertionError if any record in `bible` comes from `>= current_ep`."""
    ep = tuple(current_ep)
    for ent in bible.entities.values():
        if tuple(ent.first_seen) >= ep:
            raise AssertionError(f"Causal leak: entity {ent.char_id} from {ent.first_seen} >= {ep}")
    for slot in bible.anon_slots.values():
        if tuple(slot.first_seen) >= ep:
            raise AssertionError(f"Causal leak: anon_slot {slot.anon_id} from {slot.first_seen} >= {ep}")
        if slot.reveal_coord is not None and tuple(slot.reveal_coord) >= ep:
            raise AssertionError(f"Causal leak: anon_slot {slot.anon_id} reveal {slot.reveal_coord} >= {ep}")
    for dyad in bible.dyads.values():
        if tuple(dyad.first_seen) >= ep:
            raise AssertionError(f"Causal leak: dyad {dyad.speaker}>{dyad.addressee} from {dyad.first_seen} >= {ep}")
    for g in bible.glossary.values():
        if tuple(g.first_seen) >= ep:
            raise AssertionError(f"Causal leak: glossary {g.en!r} from {g.first_seen} >= {ep}")
    for r in bible.tm:
        if tuple(r.ep_coord) >= ep:
            raise AssertionError(f"Causal leak: TM cue {r.video_id}:{r.cue_id} from {r.ep_coord} >= {ep}")
    for f in bible.facts:
        if tuple(f.revealed_at_ep) >= ep:
            raise AssertionError(f"Causal leak: fact {f.fact_id} from {f.revealed_at_ep} >= {ep}")


def default_bible_path(series_id: str) -> Path:
    slug = re.sub(r"[^a-z0-9_-]+", "-", series_id.strip().lower()).strip("-") or "series"
    return config.data_dir() / f"_series_{slug}" / "series_state.json"


def load_bible(
    series_id: str,
    before_ep: tuple[int, int] | None = None,
    *,
    path: Path | None = None,
) -> SeriesBible:
    """Load `series_state.json` and return its causal projection `< before_ep`."""
    p = path or default_bible_path(series_id)
    if not p.is_file():
        return SeriesBible(series_id=series_id)
    raw = SeriesBible.model_validate_json(p.read_text(encoding="utf-8"))
    return raw.causal_slice(before_ep) if before_ep is not None else raw


def save_bible(bible: SeriesBible, *, path: Path | None = None) -> Path:
    """Atomically persist `bible` to disk."""
    p = path or default_bible_path(bible.series_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(bible.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(p, data.encode("utf-8"))
    return p


# --------------------------------------------------------- text & token helpers


def _norm_en_utterance(text: str) -> str:
    s = _RECAP_HEADER_RE.sub("", text or "")
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)|♪", " ", s)
    s = re.sub(r"^[\s\-–—―:\"']+|[\s\-–—―:\"']+$", "", s)
    s = re.sub(r"[^a-z0-9'\s]+", " ", s.lower())
    return " ".join(s.split())


def _content_tokens(text: str) -> list[str]:
    norm = _norm_en_utterance(text)
    return [t for t in norm.split() if len(t) >= 2 and t not in _STOPWORDS]


def _strip_recap_ko_header(ko: str) -> str:
    s = _RECAP_HEADER_RE.sub("", (ko or "").strip())
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) >= 2 and re.match(r"^[\"'“”‘’]*지난\s*이야기", lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _split_ko_name_parts(ko_name: str) -> tuple[str, str]:
    """Return `(ko_short, ko_last)` from a Korean name string like `'찰스 헤이든 새비지'`."""
    parts = [p.strip() for p in re.split(r"[\s·-]+", (ko_name or "").strip()) if p.strip()]
    if not parts:
        return ("", "")
    if len(parts) == 1:
        return (parts[0], parts[0])
    return (parts[0], parts[-1])


def mask_spoiler_persona(persona: str, unrevealed_facts: Sequence[str] = ()) -> str:
    """Strip plot-reveal clauses from a character persona string."""
    if not persona:
        return ""
    cleaned = _SPOILER_PERSONA_RE.sub("", persona)
    if unrevealed_facts:
        clauses = re.split(r"([.;|])", cleaned)
        kept: list[str] = []
        fact_toks = [
            {w for w in re.findall(r"[a-z0-9]+", f.lower()) if len(w) >= 4 and w not in _STOPWORDS}
            for f in unrevealed_facts
        ]
        i = 0
        while i < len(clauses):
            seg = clauses[i]
            sep = clauses[i + 1] if i + 1 < len(clauses) else ""
            seg_toks = {w for w in re.findall(r"[a-z0-9]+", seg.lower()) if len(w) >= 4 and w not in _STOPWORDS}
            if any(len(ft) >= 2 and len(seg_toks & ft) >= min(2, len(ft)) for ft in fact_toks):
                i += 2
                continue
            kept.append(seg + sep)
            i += 2
        cleaned = "".join(kept)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;|-")
    return cleaned or persona.split(",")[0].strip()


def word_boundary_in(term_en: str, text_lower: str) -> bool:
    """True when `term_en` appears in `text_lower` on word boundaries (with optional plural `s`/`'s`)."""
    t = term_en.strip().lower()
    if not t:
        return False
    escaped = re.escape(t)
    pat = rf"(?<![a-z0-9]){escaped}(?:'s|s)?(?![a-z0-9])"
    return bool(re.search(pat, text_lower))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


# --------------------------------------------------------- prompt hint builders


def render_viewing_roster(bible: SeriesBible | None) -> str:
    """English-only character & anonymous-slot roster for S2v (`viewing_pass`)."""
    if bible is None or bible.is_empty():
        return ""
    lines: list[str] = [
        "\n\nSERIES CHARACTER ROSTER (from prior episodes — reuse these exact character names when seen/heard):"
    ]
    for ent in sorted(bible.entities.values(), key=lambda x: x.char_id):
        aliases = f" (aliases: {', '.join(ent.aliases)})" if ent.aliases else ""
        lines.append(f"- {ent.en_name} [id={ent.char_id}]{aliases}")
    unrevealed = [s for s in bible.anon_slots.values() if not s.revealed_as]
    if unrevealed:
        lines.append(
            "UNREVEALED ANONYMOUS FIGURES (keep these exact labels until on-screen dialogue explicitly reveals their name):"
        )
        for s in sorted(unrevealed, key=lambda x: x.anon_id):
            lines.append(f"- {s.en_label} [id={s.anon_id}]")
    return "\n".join(lines)


def render_context_map_locks(bible: SeriesBible | None) -> str:
    """Locked character IDs, Korean names, and glossary entries for S2a (`build_context_map`)."""
    if bible is None or bible.is_empty():
        return ""
    lines: list[str] = [
        "SERIES CONTINUITY LOCKS (established in prior episodes — MUST keep exact id and ko_name/ko spelling):"
    ]
    if bible.entities:
        lines.append("Locked characters:")
        for ent in sorted(bible.entities.values(), key=lambda x: x.char_id):
            lines.append(f"  - id={ent.char_id} | en_name={ent.en_name} | ko_name={ent.ko_name}")
    unrevealed = [s for s in bible.anon_slots.values() if not s.revealed_as]
    if unrevealed:
        lines.append("Unrevealed anonymous slots (keep separate from named characters unless revealed in this episode):")
        for s in sorted(unrevealed, key=lambda x: x.anon_id):
            lines.append(f"  - id={s.anon_id} | en_name={s.en_label} | ko_name={s.ko_label}")
    if bible.glossary:
        lines.append("Locked series glossary (use exact ko rendering):")
        for g in sorted(bible.glossary.values(), key=lambda x: x.en.lower()):
            lines.append(f"  - {g.en} -> {g.ko}")
    return "\n".join(lines)


def render_register_locks(bible: SeriesBible | None) -> str:
    """Locked interpersonal speech levels from prior episodes for S2r (`build_register`)."""
    if bible is None or not bible.dyads:
        return ""
    lines: list[str] = [
        "\n\nESTABLISHED SERIES REGISTERS (keep these exact baseline levels, pronouns, and vocatives unless an explicit relationship turning point occurs in this episode's transcript):"
    ]
    for key, d in sorted(bible.dyads.items()):
        lines.append(
            f"- {d.speaker}>{d.addressee}: level={d.level} | pronoun={d.pronoun} | vocative={d.vocative} | relation={d.relation} (step {d.step})"
        )
    return "\n".join(lines)


# --------------------------------------------------------- stage reconcilers


def _match_bible_entity(ch: Character, bible: SeriesBible) -> EntityRecord | None:
    """Match a newly drafted `Character` to an existing `EntityRecord` in `bible`."""
    if ch.id in bible.entities:
        return bible.entities[ch.id]
    ch_forms = [ch.id, ch.en_name, *ch.aliases]
    ch_norm_forms = {re.sub(r"[^a-z0-9가-힣]+", "", f.lower()) for f in ch_forms if f}
    for ent in bible.entities.values():
        ent_forms = [ent.char_id, ent.en_name, ent.ko_name, *ent.aliases]
        ent_norm = {re.sub(r"[^a-z0-9가-힣]+", "", f.lower()) for f in ent_forms if f}
        if ch_norm_forms & ent_norm:
            return ent

    # Unambiguous token match (first or last name, excluding generic words)
    ch_toks = set()
    for f in (ch.en_name, ch.id, *ch.aliases):
        for t in _name_tokens(f):
            if len(t) >= 3 and t not in _GENERIC_NAME_WORDS:
                ch_toks.add(t)
    if not ch_toks:
        return None
    candidates: list[EntityRecord] = []
    for ent in bible.entities.values():
        ent_toks = set()
        for f in (ent.en_name, ent.char_id, *ent.aliases):
            for t in _name_tokens(f):
                if len(t) >= 3 and t not in _GENERIC_NAME_WORDS:
                    ent_toks.add(t)
        if ch_toks & ent_toks:
            candidates.append(ent)
    return candidates[0] if len(candidates) == 1 else None


def reconcile_characters(
    context_map: ContextMap,
    bible: SeriesBible | None,
    cues: Sequence[Cue] = (),
) -> ContextMap:
    """Deterministically pin character IDs, Korean names, voice cards, and glossary to `bible`."""
    if bible is None or bible.is_empty():
        return context_map

    id_remap: dict[str, str] = {}
    new_chars: list[Character] = []
    seen_ids: set[str] = set()
    unrevealed_facts = [tf.fact for tf in context_map.twist_facts]

    for ch in context_map.characters:
        # Check unrevealed anonymous slot first
        norm_en = re.sub(r"[^a-z0-9]+", "_", (ch.en_name or ch.id).lower()).strip("_")
        if norm_en in bible.anon_slots and not bible.anon_slots[norm_en].revealed_as:
            slot = bible.anon_slots[norm_en]
            id_remap[ch.id] = slot.anon_id
            ch = ch.model_copy(
                update={
                    "id": slot.anon_id,
                    "en_name": slot.en_label,
                    "ko_name": slot.ko_label,
                    "persona": mask_spoiler_persona(ch.persona, unrevealed_facts),
                }
            )
            if ch.id not in seen_ids:
                seen_ids.add(ch.id)
                new_chars.append(ch)
            continue

        ent = _match_bible_entity(ch, bible)
        if ent is not None:
            id_remap[ch.id] = ent.char_id
            merged_aliases = list(
                dict.fromkeys(
                    a
                    for a in [*ent.aliases, *ch.aliases, ch.en_name, ent.en_name, ent.ko_short, ent.ko_last]
                    if a and a not in (ent.char_id, ent.ko_name)
                )
            )
            voice = ch.voice
            if ent.voice is not None:
                vocab = list(dict.fromkeys([*(ent.voice.vocabulary or []), *(ch.voice.vocabulary or [])]))[:8]
                voice = ch.voice.model_copy(
                    update={
                        "idiolect": ent.voice.idiolect or ch.voice.idiolect,
                        "vocabulary": vocab,
                        "profanity_ceiling": ent.voice.profanity_ceiling,
                        "signature_ko": ent.voice.signature_ko or ch.voice.signature_ko,
                    }
                )
            persona_clean = mask_spoiler_persona(ent.persona_immutable or ch.persona, unrevealed_facts)
            ch = ch.model_copy(
                update={
                    "id": ent.char_id,
                    "en_name": ent.en_name if len(ent.en_name) >= len(ch.en_name) else ch.en_name,
                    "ko_name": ent.ko_name,
                    "aliases": merged_aliases,
                    "persona": persona_clean,
                    "voice": voice,
                }
            )
        else:
            ch = ch.model_copy(update={"persona": mask_spoiler_persona(ch.persona, unrevealed_facts)})

        if ch.id not in seen_ids:
            seen_ids.add(ch.id)
            new_chars.append(ch)

    # Also inject any bible entities whose English name/alias appears in cues or speaker hints
    full_en_lower = " ".join(f"{c.speaker_hint or ''} {c.text}" for c in cues).lower()
    for ent in bible.entities.values():
        if ent.char_id in seen_ids:
            continue
        names_to_check = [ent.en_name, *[a for a in ent.aliases if re.search(r"[a-zA-Z]", a)]]
        if any(word_boundary_in(nm, full_en_lower) for nm in names_to_check if len(nm) >= 3):
            seen_ids.add(ent.char_id)
            new_chars.append(
                Character(
                    id=ent.char_id,
                    en_name=ent.en_name,
                    ko_name=ent.ko_name,
                    aliases=list(ent.aliases),
                    persona=ent.persona_immutable or ent.en_name,
                    voice=ent.voice or VoiceCard(idiolect=""),
                )
            )

    def _remap(cid: str) -> str:
        return id_remap.get(cid, cid)

    arcs = [a.model_copy(update={"character": _remap(a.character)}) for a in context_map.emotional_arcs]
    twists = [
        t.model_copy(
            update={
                "known_by": [kb.model_copy(update={"character": _remap(kb.character)}) for kb in t.known_by]
            }
        )
        for t in context_map.twist_facts
    ]

    # Reconcile glossary with locked bible glossary
    glossary_by_key: dict[str, GlossaryEntry] = {}
    for g in context_map.glossary:
        k = g.en.strip().lower()
        if k in bible.glossary:
            lock = bible.glossary[k]
            glossary_by_key[k] = GlossaryEntry(en=lock.en, ko=lock.ko, note=lock.note or g.note)
        else:
            glossary_by_key[k] = g
    for k, lock in bible.glossary.items():
        if k not in glossary_by_key and word_boundary_in(lock.en, full_en_lower):
            glossary_by_key[k] = GlossaryEntry(en=lock.en, ko=lock.ko, note=lock.note)

    # Inject cross-episode callbacks for cues matching prior episode TM
    callbacks = list(context_map.callbacks)
    existing_motifs = {cb.motif.lower() for cb in callbacks}
    for cid, pin in pin_cues(cues, bible).items():
        if pin.tm_ko and pin.is_recap:
            motif_key = f"series_recap_{cid}"
            if motif_key not in existing_motifs:
                callbacks.append(
                    Callback(
                        motif=motif_key,
                        setup_cues=[],
                        payoff_cues=[cid],
                        note=f"LOCKED_SERIES_TM::{pin.tm_ko}",
                    )
                )

    return context_map.model_copy(
        update={
            "characters": new_chars,
            "emotional_arcs": arcs,
            "twist_facts": twists,
            "glossary": list(glossary_by_key.values()),
            "callbacks": callbacks,
        }
    )


_VALID_CROSS_EP_SHIFT_REASONS = frozenset({
    "identity_reveal",
    "first_private_talk_equals",
    "relationship_change",
})


def lock_edges(
    register_map: RegisterMap,
    bible: SeriesBible | None,
    *,
    context_map: ContextMap | None = None,
    cues: Sequence[Cue] = (),
    clef_client: Any = None,
) -> RegisterMap:
    """Lock interpersonal `(speaker, addressee)` registers to `bible.dyads` unless
    an explicit relational turning point is evidenced in this episode."""
    if bible is None or not bible.dyads:
        return register_map

    cues_by_id = {c.id: c for c in cues}
    edges_by_pair: dict[tuple[str, str], RegisterEdge] = {
        (e.speaker, e.addressee): e for e in register_map.edges
    }

    # Verify any proposed binary flip against locked dyads
    flip_candidates: list[tuple[tuple[str, str], RegisterEdge, DyadRecord]] = []
    for pair, edge in list(edges_by_pair.items()):
        pair_key = f"{pair[0]}>{pair[1]}"
        locked = bible.dyads.get(pair_key)
        if locked is None:
            continue
        if same_level(edge.level, locked.level):
            # Keep locked pronoun/vocative when new edge left them 'none' or drifted
            updates: dict[str, Any] = {}
            if locked.pronoun != "none" and edge.pronoun != locked.pronoun:
                updates["pronoun"] = locked.pronoun
            if locked.vocative != "none" and (edge.vocative == "none" or not edge.vocative):
                updates["vocative"] = locked.vocative
            if updates:
                edges_by_pair[pair] = edge.model_copy(update=updates)
            continue

        # Binary flip proposed! Check if edge cites an explicit valid transition reason
        has_explicit_shift = any(
            sh.reason in _VALID_CROSS_EP_SHIFT_REASONS and sh.until_cue is None
            for sh in edge.shifts
        )
        valid_ev = [cid for cid in edge.evidence_cues if cid in cues_by_id]
        if has_explicit_shift and len(valid_ev) >= 2 and clef_client is not None:
            flip_candidates.append((pair, edge, locked))
        else:
            # Revert unlicensed cross-episode binary flip to locked baseline
            edges_by_pair[pair] = edge.model_copy(
                update={
                    "level": locked.level,
                    "pronoun": locked.pronoun if locked.pronoun != "none" else edge.pronoun,
                    "vocative": locked.vocative if locked.vocative != "none" else edge.vocative,
                    "step": locked.step,
                }
            )

    if flip_candidates and clef_client is not None:
        questions: dict[str, dict[str, Any]] = {}
        state_items: list[dict[str, Any]] = []
        for (spk, adr), edge, locked in flip_candidates:
            qid = f"flip_{spk}_{adr}"
            ev_lines = [
                f"{cid}: {cues_by_id[cid].text}"
                for cid in edge.evidence_cues
                if cid in cues_by_id
            ][:6]
            state_items.append(
                {
                    "pair": f"{spk}>{adr}",
                    "prior_relation": locked.relation,
                    "prior_level": locked.level,
                    "proposed_relation": edge.relation,
                    "proposed_level": edge.level,
                    "basis": edge.basis,
                    "evidence_lines": ev_lines,
                }
            )
            questions[qid] = {
                "type": "noul",
                "instructions": (
                    f"Does the dialogue evidence for {spk}>{adr} prove an explicit, permanent "
                    f"relationship turning point (such as unmasking identity, firing/betrayal, "
                    f"or strangers becoming close partners) that justifies flipping Korean "
                    f"speech level from {locked.level} to {edge.level}?"
                ),
            }
        try:
            ans = clef_client.decide(
                stage="s2r:series_lock_gate",
                state={"candidates": state_items},
                questions=questions,
            )
        except Exception:
            ans = {}
        for (spk, adr), edge, locked in flip_candidates:
            qid = f"flip_{spk}_{adr}"
            p_yes = float((ans.get(qid) or {}).get("noul", 0.0)) if isinstance(ans.get(qid), dict) else 0.0
            if p_yes < 0.70:
                edges_by_pair[(spk, adr)] = edge.model_copy(
                    update={
                        "level": locked.level,
                        "pronoun": locked.pronoun if locked.pronoun != "none" else edge.pronoun,
                        "vocative": locked.vocative if locked.vocative != "none" else edge.vocative,
                        "step": locked.step,
                    }
                )

    # Also ensure any required_pairs present in bible.dyads have an edge
    for req_str in register_map.required_pairs:
        if ">" not in req_str:
            continue
        s, a = req_str.split(">", 1)
        if (s, a) not in edges_by_pair and req_str in bible.dyads:
            d = bible.dyads[req_str]
            edges_by_pair[(s, a)] = RegisterEdge(
                speaker=d.speaker,
                addressee=d.addressee,
                relation=d.relation,  # type: ignore[arg-type]
                apparent_generation=d.apparent_generation,  # type: ignore[arg-type]
                history=d.history,  # type: ignore[arg-type]
                stance=d.stance,  # type: ignore[arg-type]
                step=d.step,
                level=d.level,
                pronoun=d.pronoun,
                vocative=d.vocative,
                basis="inherited from series bible",
            )

    ordered_edges = list(edges_by_pair.values())
    return register_map.model_copy(update={"edges": ordered_edges})


# --------------------------------------------------------- cue pinning & enforcement


def pin_cues(
    cues: Sequence[Cue],
    bible: SeriesBible | None,
    context_map: ContextMap | None = None,
) -> dict[str, CuePin]:
    """Build deterministic per-cue pins from `bible` (TM recaps/callbacks, names, glossary)."""
    if bible is None or bible.is_empty():
        return {}

    tm_by_norm: dict[str, TMRecord] = {}
    tm_by_toks: list[tuple[set[str], TMRecord]] = []
    for r in bible.tm:
        if r.en_norm and r.ko.strip():
            tm_by_norm.setdefault(r.en_norm, r)
            if len(r.content_tokens) >= 3:
                tm_by_toks.append((set(r.content_tokens), r))

    in_recap_window = False
    recap_cutoff_ms = 95_000
    for c in cues:
        if c.start_ms <= 30_000 and "previously on" in c.text.lower():
            in_recap_window = True
            recap_cutoff_ms = max(recap_cutoff_ms, min(120_000, c.end_ms + 75_000))
            break

    pins: dict[str, CuePin] = {}
    for c in cues:
        if not c.text.strip():
            continue
        is_recap = c.start_ms <= recap_cutoff_ms and (
            in_recap_window or c.start_ms <= 65_000
        )
        en_norm = _norm_en_utterance(c.text)
        c_toks = _content_tokens(c.text)
        c_tok_set = set(c_toks)

        matched_tm: TMRecord | None = None
        if en_norm and len(en_norm) >= 12 and en_norm in tm_by_norm:
            cand = tm_by_norm[en_norm]
            if is_recap or len(c_toks) >= 4:
                matched_tm = cand
        if matched_tm is None and is_recap and len(c_tok_set) >= 3:
            best_j = 0.0
            for r_toks, r in tm_by_toks:
                inter = len(c_tok_set & r_toks)
                if inter < 3:
                    continue
                union = len(c_tok_set | r_toks)
                j = inter / union if union else 0.0
                if j >= 0.80 and j > best_j:
                    best_j = j
                    matched_tm = r

        en_lower = c.text.lower()
        req_names: dict[str, str] = {}
        for ent in bible.entities.values():
            if word_boundary_in(ent.en_name, en_lower):
                req_names[ent.en_name] = ent.ko_name
            else:
                en_parts = [p for p in re.split(r"[\s-]+", ent.en_name) if len(p) >= 3]
                if len(en_parts) >= 2:
                    if ent.ko_short and word_boundary_in(en_parts[0], en_lower):
                        req_names[en_parts[0]] = ent.ko_short
                    if ent.ko_last and word_boundary_in(en_parts[-1], en_lower):
                        req_names[en_parts[-1]] = ent.ko_last

        req_terms: dict[str, str] = {}
        for g in bible.glossary.values():
            if word_boundary_in(g.en, en_lower):
                req_terms[g.en] = g.ko

        if matched_tm is not None or req_names or req_terms:
            pins[c.id] = CuePin(
                cue_id=c.id,
                is_recap=is_recap,
                tm_ko=matched_tm.ko if matched_tm else None,
                tm_level=matched_tm.speech_level if matched_tm else None,
                tm_source_ep=matched_tm.ep_coord if matched_tm else None,
                required_names=req_names,
                required_terms=req_terms,
            )
    return pins


def _replace_fuzzy_ko_name(ko_text: str, target_ko_name: str) -> str:
    """Replace near-homophone Hangul tokens (syllable edit distance <= 2) with `target_ko_name`."""
    if not ko_text or not target_ko_name or target_ko_name in ko_text:
        return ko_text
    target_parts = [p for p in target_ko_name.split() if len(p) >= 2]
    out = ko_text
    for tp in target_parts:
        if tp in out:
            continue
        # Scan Hangul tokens in `out`, stripping trailing Korean particles
        def _sub_tok(m: re.Match[str]) -> str:
            word = m.group(0)
            stem, josa = word, ""
            for p in _KO_PARTICLES:
                if len(word) > len(p) + 1 and word.endswith(p):
                    stem = word[: -len(p)]
                    josa = p
                    break
            max_dist = 1 if len(tp) <= 3 else 2
            if (
                len(stem) >= 2
                and abs(len(stem) - len(tp)) <= 1
                and stem[0] == tp[0]
                and _levenshtein(stem, tp) <= max_dist
            ):
                return tp + josa
            return word

        out = re.sub(r"[가-힣]{2,10}", _sub_tok, out)
    return out


def enforce_series_pins(
    ko: Mapping[str, str],
    cues: Sequence[Cue],
    bible: SeriesBible | None,
    pins: Mapping[str, CuePin] | None = None,
    *,
    context_map: ContextMap | None = None,
) -> dict[str, str]:
    """Deterministically enforce recap TM matches, locked character names, and glossary terms."""
    out = dict(ko)
    if bible is None or bible.is_empty():
        return out
    pins = pins if pins is not None else pin_cues(cues, bible, context_map)
    cues_by_id = {c.id: c for c in cues}

    for cid, pin in pins.items():
        curr = out.get(cid, "")
        cue = cues_by_id.get(cid)
        if cue is None or not curr.strip():
            continue

        # 1. Opening recap exact TM restoration (preserving '지난 이야기' header if present)
        if pin.is_recap and pin.tm_ko:
            has_recap_hdr = bool(re.search(r"지난\s*이야기|전편\s*줄거리", curr)) or (
                "previously on" in cue.text.lower()
            )
            clean_tm = _strip_recap_ko_header(pin.tm_ko)
            if clean_tm:
                if has_recap_hdr and "지난 이야기" not in clean_tm:
                    curr = f"\"지난 이야기...\"\n{clean_tm}"
                else:
                    curr = clean_tm

        # 2. Fuzzy Hangul name stabilization (e.g. 레반도스키/레반다우스키 -> 레반도프스키, 신디 -> 신다)
        for _en_name, locked_ko in pin.required_names.items():
            curr = _replace_fuzzy_ko_name(curr, locked_ko)

        out[cid] = curr
    return out


# --------------------------------------------------------- pure left-fold commit Φ


def commit_bible(
    bible_prev: SeriesBible | None,
    ep_coord: EpisodeCoord,
    context_map: ContextMap,
    register_map: RegisterMap,
    cues: Sequence[Cue],
    ko: Mapping[str, str],
    *,
    timeline: Timeline | None = None,
    pairs_by_cue: Mapping[str, tuple[str, str]] | None = None,
) -> SeriesBible:
    """Pure left-fold transition `B_t = Φ(B_{t-1}, E_t)`."""
    ep = ep_coord.pair
    base = (
        bible_prev.causal_slice(ep)
        if bible_prev is not None
        else SeriesBible(series_id=ep_coord.series_id)
    )
    b = copy.deepcopy(base)
    b.series_id = ep_coord.series_id or b.series_id
    b.last_committed_ep = ep

    cues_by_id = {c.id: c for c in cues}
    pairs_by_cue = pairs_by_cue or {}
    cue_counts: Counter[str] = Counter()
    for cid, (spk, adr) in pairs_by_cue.items():
        cue_counts[spk] += 1
        cue_counts[adr] += 1

    # 1. Fold Entities & AnonSlots
    unrevealed_facts = [tf.fact for tf in context_map.twist_facts]
    for ch in context_map.characters:
        if _ANON_MONIKER_RE.search(ch.en_name or ch.id):
            anon_key = re.sub(r"[^a-z0-9]+", "_", (ch.en_name or ch.id).lower()).strip("_")
            if anon_key not in b.anon_slots:
                b.anon_slots[anon_key] = AnonSlot(
                    anon_id=ch.id,
                    en_label=ch.en_name,
                    ko_label=ch.ko_name,
                    first_seen=ep,
                )
            continue

        existing = _match_bible_entity(ch, b)
        cid = existing.char_id if existing else ch.id
        ko_short, ko_last = _split_ko_name_parts(existing.ko_name if existing else ch.ko_name)
        n_cues = cue_counts.get(ch.id, 0) + cue_counts.get(cid, 0)
        if existing is not None:
            eps_seen = list(dict.fromkeys([*existing.episodes_seen, ep]))
            aliases = list(
                dict.fromkeys(
                    a
                    for a in [*existing.aliases, *ch.aliases, ch.en_name]
                    if a and a not in (cid, existing.ko_name)
                )
            )
            b.entities[cid] = existing.model_copy(
                update={
                    "aliases": aliases,
                    "lock": "hard",
                    "episodes_seen": eps_seen,
                    "cue_count": existing.cue_count + n_cues,
                }
            )
        else:
            lock: Literal["provisional", "hard"] = "hard" if n_cues >= 3 else "provisional"
            b.entities[cid] = EntityRecord(
                char_id=cid,
                en_name=ch.en_name,
                aliases=list(dict.fromkeys(a for a in ch.aliases if a)),
                ko_name=ch.ko_name,
                ko_short=ko_short,
                ko_last=ko_last,
                lock=lock,
                persona_immutable=mask_spoiler_persona(ch.persona, unrevealed_facts),
                voice=ch.voice,
                first_seen=ep,
                episodes_seen=[ep],
                cue_count=n_cues,
            )

    # 2. Fold Dyad Registers (including open-ended shifts that persist at episode end)
    for edge in register_map.edges:
        if edge.speaker.startswith("ep:") or edge.addressee.startswith("ep:"):
            continue
        pair_key = f"{edge.speaker}>{edge.addressee}"
        open_shifts = [sh for sh in edge.shifts if sh.until_cue is None and sh.at_cue in cues_by_id]
        end_level: EdgeLevel = open_shifts[-1].level if open_shifts else edge.level
        if pair_key in b.dyads:
            prev_d = b.dyads[pair_key]
            eps_seen = list(dict.fromkeys([*prev_d.episodes_seen, ep]))
            b.dyads[pair_key] = prev_d.model_copy(
                update={
                    "level": end_level,
                    "pronoun": edge.pronoun if edge.pronoun != "none" else prev_d.pronoun,
                    "vocative": edge.vocative if edge.vocative != "none" else prev_d.vocative,
                    "relation": edge.relation,
                    "step": edge.step,
                    "lock": "hard",
                    "last_updated": ep,
                    "episodes_seen": eps_seen,
                }
            )
        else:
            b.dyads[pair_key] = DyadRecord(
                speaker=edge.speaker,
                addressee=edge.addressee,
                relation=edge.relation,
                apparent_generation=edge.apparent_generation,
                history=edge.history,
                stance=edge.stance,
                step=edge.step,
                level=end_level,
                pronoun=edge.pronoun,
                vocative=edge.vocative,
                lock="hard" if len(edge.evidence_cues) >= 2 else "provisional",
                first_seen=ep,
                last_updated=ep,
                episodes_seen=[ep],
            )

    # 3. Fold Glossary
    for g in context_map.glossary:
        k = g.en.strip().lower()
        if not k:
            continue
        if k in b.glossary:
            prev_g = b.glossary[k]
            eps_seen = list(dict.fromkeys([*prev_g.episodes_seen, ep]))
            b.glossary[k] = prev_g.model_copy(update={"lock": "hard", "episodes_seen": eps_seen})
        else:
            b.glossary[k] = GlossaryLock(
                en=g.en.strip(),
                ko=g.ko.strip(),
                note=g.note,
                lock="provisional",
                first_seen=ep,
                episodes_seen=[ep],
            )

    # 4. Fold Episodic Translation Memory (TM)
    existing_tm_keys = {(tuple(r.ep_coord), r.cue_id) for r in b.tm}
    for c in cues:
        if c.id.startswith("ost-") or c.lyric:
            continue
        ko_txt = _strip_recap_ko_header(ko.get(c.id, ""))
        if not ko_txt or not re.search(r"[가-힣]", ko_txt):
            continue
        en_norm = _norm_en_utterance(c.text)
        c_toks = _content_tokens(c.text)
        if len(c_toks) < 3 or len(en_norm) < 12:
            continue
        if (ep, c.id) in existing_tm_keys:
            continue
        pair = pairs_by_cue.get(c.id)
        lvls = [lv for lv in addressed_levels(ko_txt) if lv is not None]
        b.tm.append(
            TMRecord(
                ep_coord=ep,
                video_id=ep_coord.video_id,
                cue_id=c.id,
                start_ms=c.start_ms,
                en_norm=en_norm,
                en_raw=c.text,
                ko=ko_txt,
                speaker=pair[0] if pair else c.speaker_hint,
                addressee=pair[1] if pair else None,
                speech_level=lvls[-1] if lvls else None,
                content_tokens=c_toks,
            )
        )

    # 5. Fold Causal Twist Facts
    existing_fact_ids = {f.fact_id for f in b.facts}
    for idx, tf in enumerate(context_map.twist_facts, 1):
        fid = f"s{ep[0]:02d}e{ep[1]:02d}_tf{idx:02d}"
        if fid not in existing_fact_ids:
            b.facts.append(
                CausalFact(
                    fact_id=fid,
                    fact=tf.fact,
                    revealed_at_ep=ep,
                    revealed_at_cue=tf.audience_reveal_cue,
                    known_by=[kb.character for kb in tf.known_by],
                )
            )

    return b
