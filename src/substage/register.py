"""S2r register table: one speech level per speaker->addressee pair.

The stage runs after the timeline (S2b). Required pairs come from two
sources: the deterministic cue pairs (viewing-observation speaker and
addressee, else the SDH speaker hint, resolved to character ids with titles
stripped and unambiguous first/last-name matches) and one LLM enumeration
pass over the transcript (``prompts/register_pairs.md``, constrained to the
map's ids). A pair is required with at least ``_MIN_LINES`` lines in either
source (speaker>group edges included), plus the reverse of each required
character pair that is spoken at all. Pairs go to the text model in
batches of at most ``_BATCH_PAIRS``, each call carrying the full transcript;
the answer is ``register.yaml`` (human-editable, with the required pairs
recorded for audit). Pairs the model skipped are re-asked once; coverage
below ``_MIN_COVERAGE`` then fails the run.

Downstream the table is authoritative:

* snapshots deliver each scene's pairs (by cue speaker) to S3 and S4;
* :class:`RegisterGate` lets an edit through only if its new level is the
  table level, unknown, or the line's old level, and never swaps the table
  pronoun (S4 review/finale/regen, C7 picture check, S5c condense);
* :func:`firm_violations` checks S3 lines against *firm* edges; the logged
  result drives one S3 retry only when ``Settings.register_retry`` is on.

Levels compare at the 반말/존댓말 grain (:func:`same_level`).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .levels import addressed_levels, speaker_turns
from .models import (
    ChunkObservations,
    Cue,
    CueObservation,
    CuePerception,
    PerceptionCandidate,
    VideoInfo,
    ContextMap,
    RegisterEdge,
    RegisterMap,
    SceneRegister,
    SceneTranslation,
    shifted_level,
)
from .store import Checkpoints, input_hash, load_model, load_models

logger = logging.getLogger(__name__)

_MIN_LINES = 3  # addressed lines that make a pair required
_BATCH_PAIRS = 25  # pairs per S2r call
_MIN_COVERAGE = 0.95  # required pairs the table must cover after the re-ask
GROUP = "group"
#: addressee words that mean "everyone present" (normalized; a leading
#: "the"/"a" is dropped first)
GROUP_WORDS = frozenset(
    {"group", "audience", "crowd", "press", "viewers", "viewer", "everyone", "everybody", "all"}
)
#: honorifics and titles dropped from the front of a name before lookup
_TITLES = frozenset({
    "dr", "doctor", "father", "fr", "president", "mr", "mister", "mrs", "ms", "miss", "prof",
    "professor", "agent", "captain", "capt", "detective", "det", "officer", "sergeant", "sgt",
    "lieutenant", "lt", "colonel", "col", "general", "gen", "admiral", "commander", "chief",
    "judge", "senator", "governor", "mayor", "sir", "dame", "lady", "lord", "madam", "madame",
    "reverend", "rev", "pastor", "sister", "brother", "uncle", "aunt", "auntie", "coach",
})
_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})

#: 반말 (B) vs 존댓말 (J)
_BIN = {"hae": "B", "haera": "B", "haeyo": "J", "hasipsio": "J"}
#: Korean name of each level, as prompts render it
LEVEL_KO = {"hae": "반말", "haera": "해라", "haeyo": "해요", "hasipsio": "하십시오"}

# firm rule (decision.md §2.2): ladder steps with a fixed level (step 3 —
# family trait — fixes none), and feature contradictions that point to an
# earlier step than the one claimed
_STEP_LEVEL: dict[int, str | None] = {1: "J", 2: "B", 3: None, 4: "J", 5: "B", 6: "J", 7: "B", 10: "B"}
SYSTEM_AI = "system_ai"
_ACRONYM_AI_RE = re.compile(r"^(?:[A-Z]\.){2,}[A-Z]?\.?$")
_SYSTEM_VOICE_WORDS = frozenset({"ai", "autopilot", "computer", "hud", "pa"})
# a pronoun (너희 included) followed by up to three stacked particles:
# 당신한테만, 너에게는, 너희한테서도
_PRONOUN_PARTICLE = (
    r"(?:에게서|한테서|께서|한테|에게|이랑|하고|보다|까지|부터|처럼|조차|마저|밖에|이야|으로"
    r"|은|는|이|가|을|를|도|만|께|의|랑|과|와|야|로)"
)
_PRONOUN_RE = re.compile(
    r"(?<![가-힣])(당신|자네|너|네가|니가|널|넌)"
    rf"(?:희)?{_PRONOUN_PARTICLE}{{0,3}}(?![가-힣])"
)
_PRONOUN_CANON = {"네가": "너", "니가": "너", "널": "너", "넌": "너"}


def binary(level: str | None) -> str | None:
    """'B' (반말: hae, haera), 'J' (존댓말: haeyo, hasipsio), or None."""
    return _BIN.get(level) if level else None


def same_level(a: str | None, b: str | None) -> bool:
    """True when both levels are known and fall on the same side of 반말/존댓말."""
    return a is not None and b is not None and binary(a) == binary(b) and binary(a) is not None


def _is_companion_ai(edge: RegisterEdge, reverse: RegisterEdge | None = None) -> bool:
    """True when an AI speaker is characterized as an intimate peer/buddy companion."""
    if edge.history == "longtime_crew_or_buddies" or edge.relation in ("peer", "crew_buddy", "partner"):
        return True
    if reverse is not None and (reverse.relation == "partner" or "companion" in reverse.basis.lower()):
        return True
    basis_low = edge.basis.lower()
    return "companion" in basis_low or "suit ai" in basis_low


def _earliest_step(edge: RegisterEdge) -> int | None:
    """The first ladder step the edge's features point to, or None when no
    feature names a step."""
    if edge.ai_user or edge.relation == "ai_to_user":
        return 5 if _is_companion_ai(edge) else 1
    if edge.stance != "civil" or edge.relation == "enemy":
        return 2
    if edge.relation == "own_parent":
        return 3 if edge.history == "raised_by_addressee" else 4
    if edge.relation in ("elder_kin", "surrogate_parent") and edge.apparent_generation == "addressee_older":
        return 4
    if edge.history == "longtime_crew_or_buddies" and edge.relation not in ("elder_kin", "surrogate_parent"):
        return 5
    if edge.relation in ("mentor", "boss", "kid_to_adult") or edge.apparent_generation == "addressee_older":
        return 6
    if edge.relation in ("protege", "subordinate", "adult_to_kid", "own_child") or edge.apparent_generation == "speaker_older":
        return 7
    if edge.relation in ("partner", "sibling", "crew_buddy") or (
        edge.relation == "peer" and edge.apparent_generation == "same" and edge.history != "met_as_adults"
    ):
        return 10
    return None


def canonicalize_edge(edge: RegisterEdge, *, reverse: RegisterEdge | None = None) -> RegisterEdge:
    """Reconcile an edge's step and level with relational invariants."""
    updates: dict[str, object] = {}
    if edge.step == 1:
        if _is_companion_ai(edge, reverse):
            updates["step"] = 5
            updates["relation"] = "crew_buddy"
            updates["history"] = "longtime_crew_or_buddies"
            updates["ai_user"] = False
            if binary(edge.level) != "B":
                updates["level"] = "hae"
        elif binary(edge.level) != "J":
            updates["level"] = "haeyo"
    elif (
        edge.step in (7, 10)
        and (edge.ai_user or edge.relation == "ai_to_user")
        and reverse is not None
        and (reverse.step == 1 or reverse.ai_user or reverse.relation == "ai_to_user")
    ):
        updates["ai_user"] = False
        if edge.relation == "ai_to_user":
            updates["relation"] = "subordinate" if edge.step == 7 else "peer"
        if binary(edge.level) != "B":
            updates["level"] = "hae"
    elif (
        edge.relation == "own_parent"
        and edge.history == "raised_by_addressee"
        and reverse is not None
        and reverse.relation in ("adult_to_kid", "own_child")
    ):
        basis_low = edge.basis.lower()
        if not any(w in basis_low for w in ("adult", "traditional", "strict", "formal", "patriarch")):
            updates["step"] = 3
            updates["level"] = "hae"
    elif (
        edge.relation == "partner"
        and edge.apparent_generation == "same"
        and not edge.shifts
        and reverse is not None
        and reverse.relation == "partner"
    ):
        updates["step"] = 10
        if binary(edge.level) != "B":
            updates["level"] = "hae"
    if edge.relation in ("elder_kin", "surrogate_parent") and edge.apparent_generation == "addressee_older" and edge.shifts:
        kept = [sh for sh in edge.shifts if binary(sh.level) == "J" or sh.reason in ("role_on", "role_off")]
        if len(kept) != len(edge.shifts):
            updates["shifts"] = kept
    return edge.model_copy(update=updates) if updates else edge


def canonicalize_register_map(rmap: RegisterMap) -> RegisterMap:
    """Canonicalize all directed edges in *rmap* using paired reverse context."""
    by_pair = {(e.speaker, e.addressee): e for e in rmap.edges}
    edges = [canonicalize_edge(e, reverse=by_pair.get((e.addressee, e.speaker))) for e in rmap.edges]
    return rmap.model_copy(update={"edges": edges})

def edge_firm(edge: RegisterEdge, cue_ids: Iterable[str]) -> bool:
    """The edge's level is beyond dispute: its ladder step fixes a level and
    the edge carries that level, no feature points to an earlier step, and it
    cites at least two existing cues."""
    step = edge.step
    if step not in _STEP_LEVEL:
        return False
    earliest = _earliest_step(edge)
    if earliest is not None and earliest < step:
        return False
    want = _STEP_LEVEL[step]
    if want is not None and binary(edge.level) != want:
        return False
    known = set(cue_ids)
    return len({c for c in edge.evidence_cues if c in known}) >= 2


class RegisterTable:
    """Lookup over register.yaml with shifts applied in video cue order."""

    def __init__(self, rmap: RegisterMap, cue_order: Sequence[str]):
        self.map = canonicalize_register_map(rmap)
        self._pos = {cid: i for i, cid in enumerate(cue_order)}
        self._edges = {(e.speaker, e.addressee): e for e in self.map.edges}
        self._firm = {k: edge_firm(e, self._pos) for k, e in self._edges.items()}
    @classmethod
    def load(cls, data_dir: Path) -> RegisterTable | None:
        """Table from a workspace data dir (register.yaml + cues.json for cue
        order); None when the dir has no register.yaml."""
        path = Path(data_dir) / "register.yaml"
        if not path.is_file():
            return None
        cues_path = Path(data_dir) / "cues.json"
        cues = load_models(cues_path, Cue) if cues_path.is_file() else []
        return cls(load_model(path, RegisterMap), [c.id for c in sorted(cues, key=lambda c: c.start_ms)])

    def pairs(self) -> list[str]:
        return [f"{s}>{a}" for s, a in self._edges]

    def edge(self, spk: str, adr: str) -> RegisterEdge | None:
        return self._edges.get((spk, adr))

    def level_at(self, spk: str, adr: str, cue_id: str) -> str | None:
        e = self.edge(spk, adr)
        if e is None:
            if spk == SYSTEM_AI:
                return "haeyo"
            if adr == SYSTEM_AI:
                return "hae"
            return None
        return shifted_level(e.level, e.shifts, cue_id, self._pos)
    def in_shift(self, spk: str, adr: str, cue_id: str) -> bool:
        """The cue lies inside one of the pair's shift spans."""
        e = self.edge(spk, adr)
        if e is None or cue_id not in self._pos:
            return False
        at, end = self._pos[cue_id], len(self._pos)
        for sh in e.shifts:
            start = self._pos.get(sh.at_cue)
            stop = self._pos.get(sh.until_cue, end) if sh.until_cue else end
            if start is not None and start <= at < stop:
                return True
        return False

    def is_firm(self, spk: str, adr: str, cue_id: str) -> bool:
        if spk == SYSTEM_AI or adr == SYSTEM_AI:
            return True
        return self._firm.get((spk, adr), False) and not self.in_shift(spk, adr, cue_id)
    def address(self, spk: str, adr: str) -> tuple[str, str]:
        e = self.edge(spk, adr)
        return ("none", "none") if e is None else (e.pronoun, e.vocative)


# ---------------------------------------------------------------- pairs


def _flat_observations(observations: Iterable[ChunkObservations | CueObservation]) -> dict[str, CueObservation]:
    out: dict[str, CueObservation] = {}
    for o in observations or []:
        for c in o.cues if isinstance(o, ChunkObservations) else [o]:
            out.setdefault(c.cue_id, c)
    return out


#: name words too generic to resolve a character by first/last name alone
_GENERIC_NAME_WORDS = frozenset({
    "man", "woman", "boy", "girl", "guy", "kid", "lady", "old", "young", "big", "little", "the",
    "mom", "mother", "dad", "father", "son", "daughter", "baby", "guard", "officer", "doctor",
})

Pair = tuple[str, str]


def _words(name: str) -> list[str]:
    from .narrative import _norm

    return [t for t in (_norm(w) for w in name.split()) if t]


def _name_tokens(name: str) -> list[str]:
    """Normalized name words with leading honorifics/titles and trailing
    suffixes (Jr., III) dropped; a lone title word is kept."""
    toks = _words(name)
    while len(toks) > 1 and toks[0] in _TITLES:
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in _NAME_SUFFIXES:
        toks = toks[:-1]
    return toks


def is_group(name: str | None) -> bool:
    """The addressee is everyone present: group, (the) audience, crowd,
    press, viewers, everyone…"""
    toks = _words(name or "")
    if toks and toks[0] in ("the", "a"):
        toks = toks[1:]
    return len(toks) == 1 and toks[0] in GROUP_WORDS


def name_resolver(context_map: ContextMap) -> Callable[[str | None], str | None]:
    """name -> character id or None. Each ``/``-separated part is tried in
    turn: exact normalized match on id / en_name / ko_name / aliases; then
    the same with honorifics and titles stripped on both sides (``Dr.
    Randall Mindy`` = ``Randall Mindy``); then a first-name or last-name
    match, accepted only when exactly one character carries it (a character
    matching both first and last name beats single-name matches). Never
    returns ``group`` (see :func:`is_group`)."""
    from .narrative import _alias_map, _norm  # narrative stays free of this module

    exact = _alias_map(context_map)
    full: dict[str, set[str]] = {}
    first: dict[str, set[str]] = {}
    last: dict[str, set[str]] = {}
    for ch in context_map.characters:
        for form in (ch.id, ch.en_name, ch.ko_name, *ch.aliases):
            toks = _name_tokens(form)
            if not toks:
                continue
            full.setdefault("".join(toks), set()).add(ch.id)
            first.setdefault(toks[0], set()).add(ch.id)
            last.setdefault(toks[-1], set()).add(ch.id)

    def partial(index: dict[str, set[str]], tok: str) -> set[str]:
        return set() if len(tok) < 2 or tok in _GENERIC_NAME_WORDS else index.get(tok, set())

    def one(part: str) -> str | None:
        if (r := exact.get(_norm(part))) is not None:
            return r
        toks = _name_tokens(part)
        if not toks:
            return None
        ids = full.get("".join(toks), set())
        if ids:
            return next(iter(ids)) if len(ids) == 1 else None
        f, l = partial(first, toks[0]), partial(last, toks[-1])
        cands = (f & l) or (f | l)
        return next(iter(cands)) if len(cands) == 1 else None

    def resolve(name: str | None) -> str | None:
        for part in (name or "").split("/"):
            if part.strip() and (r := one(part)) is not None:
                return r
        return None

    return resolve

def _is_system_ai_speaker(name: str | None) -> bool:
    """True for unmapped acronym or system/HUD interface speaker labels."""
    if not name or not name.strip():
        return False
    s = name.strip()
    if _ACRONYM_AI_RE.match(s):
        return True
    toks = _words(s)
    return bool(toks) and any(t in _SYSTEM_VOICE_WORDS for t in toks)


def cue_pairs(
    context_map: ContextMap,
    cues: list[Cue],
    observations: Iterable[ChunkObservations | CueObservation],
) -> dict[str, Pair]:
    """cue id -> (speaker id, addressee id) for addressed lines: the viewing
    observation names the speaker (else the SDH speaker hint) and addressee,
    both resolved to video-map character ids (:func:`name_resolver`); the
    addressee may be ``group`` (audience, crowd, press, viewers, everyone...).
    Unmapped acronym/system AI voices resolve to ``system_ai``.
    Lines to nobody, oneself or unresolved names are skipped."""
    rid = name_resolver(context_map)
    sys_ai_chars = {
        ch.id for ch in context_map.characters
        if _is_system_ai_speaker(ch.en_name) or any(_is_system_ai_speaker(al) for al in ch.aliases)
    }
    obs = _flat_observations(observations)
    out: dict[str, Pair] = {}
    for c in cues:
        o = obs.get(c.id)
        raw_spk = o.speaker if o and o.speaker else c.speaker_hint
        spk = rid(o.speaker if o and o.speaker else None) or rid(c.speaker_hint)
        if spk is None and _is_system_ai_speaker(raw_spk):
            spk = SYSTEM_AI
        if spk is None or o is None or not o.addressee:
            continue
        adr = GROUP if is_group(o.addressee) else rid(o.addressee)
        if adr is None and _is_system_ai_speaker(o.addressee):
            adr = SYSTEM_AI
        if adr is not None and adr != spk:
            out[c.id] = (spk, adr)
    if sys_ai_chars:
        counts = line_counts(out)
        sparse_ai = {
            ch_id for ch_id in sys_ai_chars
            if all(n < _MIN_LINES for (s, a), n in counts.items() if s == ch_id or a == ch_id)
        }
        if sparse_ai:
            out = {
                cid: (SYSTEM_AI if s in sparse_ai else s, SYSTEM_AI if a in sparse_ai else a)
                for cid, (s, a) in out.items()
            }
    return out

def cue_turn_pairs(
    context_map: ContextMap,
    observations: Iterable[ChunkObservations | CueObservation | CuePerception],
) -> dict[str, tuple[Pair | None, ...]]:
    """Resolve ordered sub-turn candidates without guessing between identities.

    Missing or conflicting candidates leave that turn's pair unknown. Perception
    identifies a possible table row, never supplies or changes its register.
    """
    resolve = name_resolver(context_map)

    def identity(
        candidates: Sequence[PerceptionCandidate], *, addressee: bool = False
    ) -> str | None:
        ids = {
            GROUP if addressee and is_group(c.value) else resolve(c.value)
            for c in candidates
        }
        return next(iter(ids)) if len(ids) == 1 and None not in ids else None

    out: dict[str, tuple[Pair | None, ...]] = {}
    for observation in observations:
        for cue in observation.cues if isinstance(observation, ChunkObservations) else [observation]:
            if cue.turns is None:
                continue
            pairs: list[Pair | None] = []
            for turn in cue.turns:
                speaker = identity(turn.speaker_candidates)
                addressee = identity(turn.addressee_candidates, addressee=True)
                pairs.append(
                    (speaker, addressee)
                    if speaker is not None and addressee is not None and speaker != addressee
                    else None
                )
            out.setdefault(cue.cue_id, tuple(pairs))
    return out


def _lines_by_pair(pairs_by_cue: dict[str, Pair]) -> dict[Pair, list[str]]:
    out: dict[Pair, list[str]] = {}
    for cid, p in pairs_by_cue.items():
        out.setdefault(p, []).append(cid)
    return out


def line_counts(pairs_by_cue: dict[str, Pair], llm_lines: dict[Pair, list[str]] | None = None) -> dict[Pair, int]:
    """Lines per pair: the larger of the deterministic and the LLM count."""
    counts = {p: len(v) for p, v in _lines_by_pair(pairs_by_cue).items()}
    for p, v in (llm_lines or {}).items():
        counts[p] = max(counts.get(p, 0), len(set(v)))
    return counts


def required_pairs(
    pairs_by_cue: dict[str, Pair],
    llm_lines: dict[Pair, list[str]] | None = None,
    cue_order: Sequence[str] | None = None,
) -> list[Pair]:
    """Pairs with >= _MIN_LINES lines in either source (deterministic cue
    pairs, LLM enumeration; ``group`` edges included), in order of first
    line, then the reverse of each such character pair that has at least one
    line in either source."""
    det = _lines_by_pair(pairs_by_cue)
    llm = llm_lines or {}
    counts = line_counts(pairs_by_cue, llm)
    order = list(cue_order) if cue_order is not None else [
        *pairs_by_cue, *(c for v in llm.values() for c in v)
    ]
    pos: dict[str, int] = {}
    for c in order:
        pos.setdefault(c, len(pos))
    end = len(pos)

    def first(p: Pair) -> int:
        return min((pos.get(c, end) for c in (*det.get(p, ()), *llm.get(p, ()))), default=end)

    req = sorted(
        (p for p, n in counts.items() if n >= _MIN_LINES and SYSTEM_AI not in p),
        key=first,
    )
    seen = set(req)
    for spk, adr in list(req):
        rev = (adr, spk)
        if adr != GROUP and SYSTEM_AI not in rev and rev not in seen and counts.get(rev, 0) >= 1:
            req.append(rev)
            seen.add(rev)
    return req


class EnumeratedPair(BaseModel):
    """One directed pair the LLM enumeration found, with the cues it speaks."""

    speaker: str  # character id
    addressee: str  # character id or "group"
    cue_ids: list[str] = Field(default_factory=list)


class RegisterPairs(BaseModel):
    """Response of the LLM enumeration pass (prompts/register_pairs.md)."""

    pairs: list[EnumeratedPair] = Field(default_factory=list)


def pairs_template() -> str:
    """register_pairs.md (checkpoint identity of the enumeration pass)."""
    from .narrative import _prompt

    return _prompt("register_pairs.md")


def _enum_transcript(cues: list[Cue], observations: Iterable[ChunkObservations | CueObservation]) -> str:
    """``c0001|mm:ss|speaker -> addressee|text`` with the raw viewing names
    (speaker falls back to the SDH hint; ``?`` when unknown)."""
    from .narrative import _mmss

    obs = _flat_observations(observations)
    rows = []
    for c in cues:
        o = obs.get(c.id)
        spk = (o.speaker if o and o.speaker else None) or c.speaker_hint or "?"
        adr = (o.addressee if o and o.addressee else None) or "?"
        text = re.sub(r"\s+", " ", c.text).strip()
        rows.append(f"{c.id}|{_mmss(c.start_ms)}|{spk} -> {adr}|{text}")
    return "\n".join(rows)


def llm_pair_lines(rp: RegisterPairs, context_map: ContextMap, cues: list[Cue]) -> dict[Pair, list[str]]:
    """Enumerated pairs constrained to map ids (addressee group words ->
    ``group``) -> their existing cue ids; pairs without one are dropped."""
    ids = {ch.id for ch in context_map.characters}
    known = {c.id for c in cues}
    out: dict[Pair, list[str]] = {}
    for p in rp.pairs:
        spk = p.speaker.strip()
        adr = GROUP if is_group(p.addressee) else p.addressee.strip()
        if spk not in ids or (adr != GROUP and adr not in ids) or spk == adr:
            continue
        lines = out.setdefault((spk, adr), [])
        lines += [c for c in dict.fromkeys(p.cue_ids) if c in known and c not in lines]
    return {p: v for p, v in out.items() if v}


def enumerate_pairs(
    ws,
    gc,
    video: VideoInfo,
    context_map: ContextMap,
    cues: list[Cue],
    observations: Iterable[ChunkObservations | CueObservation],
) -> dict[Pair, list[str]]:
    """One LLM call over the whole transcript (stage ``s2r:pairs``, inner
    checkpoint ``s2r/pairs`` keyed by the rendered prompt + model): every
    directed pair exchanging at least one line, constrained to map ids."""
    if not context_map.characters or not cues:
        return {}
    observations = list(observations or [])
    characters = "\n".join(
        f"{ch.id} | {ch.en_name} | {ch.ko_name} | {', '.join(ch.aliases) or '-'}" for ch in context_map.characters
    )
    prompt = pairs_template().format(
        title=video.title,
        year=video.year or "?",
        characters=characters,
        transcript=_enum_transcript(cues, observations),
    )
    h = input_hash(prompt, gc.settings.model_main)
    cp = Checkpoints(ws, "s2r")
    hit = cp.get("pairs", h, RegisterPairs)
    if hit is None:
        hit = gc.generate(stage="s2r:pairs", prompt=prompt, schema=RegisterPairs)
        cp.put("pairs", h, hit)
    return llm_pair_lines(hit, context_map, cues)


# ---------------------------------------------------------------- S2r


class RegisterCoverageError(RuntimeError):
    """The register table covers too few required pairs after the re-ask."""


class _EdgeBatch(BaseModel):
    """S2r batch response: RegisterMap's edges only (required_pairs is ours)."""

    model_config = ConfigDict(title="RegisterMap")

    edges: list[RegisterEdge] = Field(default_factory=list)


def _transcript(cues: list[Cue], pairs_by_cue: dict[str, tuple[str, str]]) -> str:
    from .narrative import _mmss

    rows = []
    for c in cues:
        pair = pairs_by_cue.get(c.id)
        who = f"{pair[0]}>{pair[1]}" if pair else (c.speaker_hint or "-")
        text = re.sub(r"\s+", " ", c.text).strip()
        rows.append(f"{c.id}|{_mmss(c.start_ms)}|{who}|{text}")
    return "\n".join(rows)


def template() -> str:
    """register.md with the style guide's §3 embedded (checkpoint identity)."""
    from .narrative import _prompt

    return _prompt("register.md")


def build_register(
    ws,
    gc,
    video: VideoInfo,
    context_map: ContextMap,
    cues: list[Cue],
    observations: Iterable[ChunkObservations | CueObservation],
) -> RegisterMap:
    """Enumerate the required pairs (deterministic cue pairs ∪ the LLM
    enumeration pass), ask the text model for every one (batches of <= 25,
    full transcript each), re-ask the missing pairs once, and fail when
    coverage stays below 95 %. The table records its required pairs."""
    observations = list(observations or [])
    pairs_by_cue = cue_pairs(context_map, cues, observations)
    llm_lines = enumerate_pairs(ws, gc, video, context_map, cues, observations)
    required = required_pairs(pairs_by_cue, llm_lines, [c.id for c in sorted(cues, key=lambda c: c.start_ms)])
    counts = line_counts(pairs_by_cue, llm_lines)
    tmpl = template()
    characters = "\n".join(
        f"{ch.id} | {ch.en_name} | {ch.ko_name} | {' '.join(ch.persona.split())}" for ch in context_map.characters
    ) or "(none)"
    transcript = _transcript(cues, pairs_by_cue)
    cp = Checkpoints(ws, "s2r")
    got: dict[tuple[str, str], RegisterEdge] = {}

    def ask(pairs: list[tuple[str, str]], label: str) -> None:
        for k in range(0, len(pairs), _BATCH_PAIRS):
            batch = pairs[k : k + _BATCH_PAIRS]
            prompt = tmpl.format(
                title=video.title,
                year=video.year or "?",
                synopsis=context_map.synopsis,
                tone=context_map.tone_baseline,
                characters=characters,
                transcript=transcript,
                pairs="\n".join(f"{s}>{a} ({counts.get((s, a), 0)} lines)" for s, a in batch),
            )
            h = input_hash(prompt, gc.settings.model_main)
            key = f"{label}-{k // _BATCH_PAIRS + 1:02d}"
            hit = cp.get(key, h, _EdgeBatch)
            if hit is None:
                hit = gc.generate(stage=f"s2r:{key}", prompt=prompt, schema=_EdgeBatch)
                cp.put(key, h, hit)
            wanted = set(batch)
            for e in hit.edges:
                k2 = (e.speaker, e.addressee)
                if k2 in wanted and k2 not in got:
                    got[k2] = e

    ask(required, "batch")
    missing = [p for p in required if p not in got]
    if missing:
        logger.warning("s2r: %d/%d pairs missing; re-asking once", len(missing), len(required))
        ask(missing, "reask")
        missing = [p for p in required if p not in got]
    if required and (len(required) - len(missing)) / len(required) < _MIN_COVERAGE:
        raise RegisterCoverageError(
            f"register table covers {len(required) - len(missing)}/{len(required)} required pairs "
            f"(< {_MIN_COVERAGE:.0%}); missing: {', '.join(f'{s}>{a}' for s, a in missing)}"
        )
    if missing:
        logger.warning("s2r: pairs left without an edge: %s", ", ".join(f"{s}>{a}" for s, a in missing))
    return canonicalize_register_map(
        RegisterMap(
            edges=[got[p] for p in required if p in got],
            required_pairs=[f"{s}>{a}" for s, a in required],
        )
    )


# ---------------------------------------------------------------- delivery


def cue_levels(
    table: RegisterTable | None, pairs_by_cue: dict[str, tuple[str, str]]
) -> dict[str, tuple[str, str]]:
    """cue id -> ("spk>adr", table level at the cue) for review/finale rows."""
    if table is None:
        return {}
    out: dict[str, tuple[str, str]] = {}
    for cid, (spk, adr) in pairs_by_cue.items():
        level = table.level_at(spk, adr, cid)
        if level is not None:
            out[cid] = (f"{spk}>{adr}", level)
    return out


def render_scene_register(pair: str, reg: SceneRegister) -> list[str]:
    """``spk>adr: 해요 · 아저씨 · 당신`` plus one row per shift."""
    parts = [LEVEL_KO.get(reg.level, reg.level)]
    parts += [x for x in (reg.vocative, reg.pronoun) if x and x != "none"]
    rows = [f"  {pair}: {' · '.join(parts)}"]
    for sh in reg.shifts:
        until = f" until {sh.until_cue}" if sh.until_cue else ""
        rows.append(f"    shift from {sh.at_cue}{until}: {LEVEL_KO.get(sh.level, sh.level)} ({sh.reason})")
    return rows


# ---------------------------------------------------------------- gate


def _turn_levels(ko: str) -> list[str | None]:
    """One addressed level per speaker turn (dash lines split)."""
    return list(addressed_levels(ko)) or [None]


def turn_levels_match(level: str | None, table_level: str | None, old_level: str | None) -> bool:
    """Unknown endings may pass; a known ending must retain or match its row."""
    return level is None or same_level(level, table_level) or same_level(level, old_level)


def _pronouns(ko: str) -> set[str]:
    """Second-person pronouns (너/자네/당신) the line uses."""
    return {_PRONOUN_CANON.get(p, p) for p in _PRONOUN_RE.findall(ko)}


class GateRejection(BaseModel):
    stage: str
    cue_ids: list[str]
    reason: str  # level | pronoun | turn_count
    old_ko: str
    new_ko: str
    pair: str | None = None
    table_level: str | None = None
    old_levels: list[str | None] = Field(default_factory=list)
    new_levels: list[str | None] = Field(default_factory=list)
    turn_index: int | None = None  # one-based source/display turn
    turn_ko: str | None = None
    expected_turns: int | None = None
    actual_turns: int | None = None


class FirmViolation(BaseModel):
    scene_id: str
    cue_id: str
    pair: str
    table_level: str
    line_level: str | None
    ko: str
    reason: str = "level"  # level | turn_count
    turn_index: int | None = None
    turn_ko: str | None = None
    expected_turns: int | None = None
    actual_turns: int | None = None


class RegisterReport(BaseModel):
    """register_report.json: what the table vetoed or sent back."""

    gate_rejections: list[GateRejection] = Field(default_factory=list)
    firm_retried_scenes: list[str] = Field(default_factory=list)  # S3 scenes retranslated this run
    firm_violations: list[FirmViolation] = Field(default_factory=list)  # left after the retry


class RegisterGate:
    """Edit gate: a changed line passes only if each new level is the table
    level, unknown, or the line's old level, and it brings in no address
    pronoun other than the table's (none at all when the pair's edge has
    pronoun "none"). Rejections accumulate in ``rejections``."""

    def __init__(
        self,
        table: RegisterTable | None,
        pairs_by_cue: dict[str, tuple[str, str]],
        *,
        turn_pairs_by_cue: Mapping[str, Sequence[Pair | None]] | None = None,
    ):
        self.table = table
        self.pairs_by_cue = pairs_by_cue
        self.turn_pairs_by_cue = turn_pairs_by_cue or {}
        self.rejections: list[GateRejection] = []

    def _pair(self, cue_ids: Sequence[str]) -> tuple[str, str] | None:
        """The single pair every cue shares, else None."""
        pairs = {self.pairs_by_cue.get(c) for c in cue_ids}
        return next(iter(pairs)) if len(pairs) == 1 and None not in pairs else None

    def _check_turns(
        self,
        stage: str,
        cue_id: str,
        old_ko: str,
        new_ko: str,
        pairs: Sequence[Pair | None],
        *,
        use_table: bool,
    ) -> bool:
        old, new = _turn_levels(old_ko), _turn_levels(new_ko)
        old_text, new_text = speaker_turns(old_ko), speaker_turns(new_ko)
        actual = len(new_text)
        if actual != len(pairs):
            self.rejections.append(
                GateRejection(
                    stage=stage, cue_ids=[cue_id], reason="turn_count",
                    old_ko=old_ko, new_ko=new_ko, old_levels=old, new_levels=new,
                    expected_turns=len(pairs), actual_turns=actual,
                )
            )
            return False
        rejected = False
        aligned_old = len(old_text) == len(pairs)
        for i, pair in enumerate(pairs):
            if pair is None and len(pairs) == 1:
                pair = self.pairs_by_cue.get(cue_id)
            table_level = None
            allowed_pronouns: set[str] | None = None
            if pair is not None and self.table is not None:
                if use_table:
                    table_level = self.table.level_at(*pair, cue_id)
                edge = self.table.edge(*pair)
                if edge is not None:
                    allowed_pronouns = set() if edge.pronoun == "none" else {edge.pronoun}
            old_level = old[i] if aligned_old else None
            reason = None
            if not turn_levels_match(new[i], table_level, old_level):
                reason = "level"
            elif allowed_pronouns is not None:
                prior_pronouns = _pronouns(old_text[i]) if aligned_old else set()
                if _pronouns(new_text[i]) - prior_pronouns - allowed_pronouns:
                    reason = "pronoun"
            if reason is not None:
                self.rejections.append(
                    GateRejection(
                        stage=stage, cue_ids=[cue_id], reason=reason,
                        old_ko=old_ko, new_ko=new_ko,
                        pair=f"{pair[0]}>{pair[1]}" if pair else None,
                        table_level=table_level, old_levels=old, new_levels=new,
                        turn_index=i + 1, turn_ko=new_text[i],
                        expected_turns=len(pairs), actual_turns=actual,
                    )
                )
                rejected = True
        return not rejected

    def check(
        self, stage: str, cue_ids: Sequence[str], old_ko: str, new_ko: str, *, use_table: bool = True
    ) -> bool:
        """True when the edit may land; ``use_table=False`` (C7 picture check,
        which has no register authority) allows only the old level."""
        if new_ko == old_ko:
            return True
        if len(cue_ids) == 1 and (turn_pairs := self.turn_pairs_by_cue.get(cue_ids[0])):
            return self._check_turns(
                stage, cue_ids[0], old_ko, new_ko, turn_pairs, use_table=use_table
            )
        pair = self._pair(cue_ids)
        table_level = None
        allowed_pronouns: set[str] | None = None  # None: no table edge, pronouns unchecked
        if pair and self.table is not None and cue_ids:
            if use_table:
                table_level = self.table.level_at(*pair, cue_ids[0])
            edge = self.table.edge(*pair)
            if edge is not None:
                allowed_pronouns = set() if edge.pronoun == "none" else {edge.pronoun}
        old, new = _turn_levels(old_ko), _turn_levels(new_ko)
        reason = None
        if len(old) == 1 and len(new) == 1:
            if not turn_levels_match(new[0], table_level, old[0]):
                reason = "level"
        else:
            rev_level = (
                self.table.level_at(pair[1], pair[0], cue_ids[0])
                if use_table and pair and self.table is not None and cue_ids
                else None
            )
            for i, lvl in enumerate(new):
                allowed = [old[i]] if len(old) == len(new) else list(old)
                if table_level is not None and rev_level is not None:
                    if len(new) == 2:
                        allowed.append(table_level if i == 0 else rev_level)
                    elif same_level(table_level, rev_level):
                        allowed.append(table_level)
                if lvl is not None and not any(same_level(lvl, a) for a in allowed):
                    reason = "level"
                    break
        if reason is None and allowed_pronouns is not None:
            if _pronouns(new_ko) - _pronouns(old_ko) - allowed_pronouns:
                reason = "pronoun"
        if reason is None:
            return True
        self.rejections.append(
            GateRejection(
                stage=stage, cue_ids=list(cue_ids), reason=reason, old_ko=old_ko, new_ko=new_ko,
                pair=f"{pair[0]}>{pair[1]}" if pair else None, table_level=table_level,
                old_levels=old, new_levels=new,
            )
        )
        return False


# ---------------------------------------------------------------- S3 post-pass


def firm_violations(
    table: RegisterTable | None,
    pairs_by_cue: dict[str, tuple[str, str]],
    st: SceneTranslation,
    *,
    turn_pairs_by_cue: Mapping[str, Sequence[Pair | None]] | None = None,
) -> list[FirmViolation]:
    """Known turns on firm table edges whose count or level violates their row.

    Without sub-turn metadata, retain the single-speaker-only legacy check.
    """
    if table is None:
        return []
    out: list[FirmViolation] = []
    turn_pairs_by_cue = turn_pairs_by_cue or {}
    for tc in st.cues:
        turn_pairs = turn_pairs_by_cue.get(tc.cue_id)
        if turn_pairs:
            text_turns = speaker_turns(tc.ko)
            levels = _turn_levels(tc.ko)
            aligned = len(text_turns) == len(turn_pairs)
            for i, pair in enumerate(turn_pairs):
                if pair is None and len(turn_pairs) == 1:
                    pair = pairs_by_cue.get(tc.cue_id)
                if pair is None or not table.is_firm(*pair, tc.cue_id):
                    continue
                want = table.level_at(*pair, tc.cue_id)
                level = levels[i] if aligned else None
                if aligned and (level is None or same_level(level, want)):
                    continue
                out.append(
                    FirmViolation(
                        scene_id=st.scene_id, cue_id=tc.cue_id,
                        pair=f"{pair[0]}>{pair[1]}", table_level=want or "",
                        line_level=level, ko=tc.ko,
                        reason="level" if aligned else "turn_count",
                        turn_index=i + 1, turn_ko=text_turns[i] if aligned else None,
                        expected_turns=len(turn_pairs), actual_turns=len(text_turns),
                    )
                )
            continue
        pair = pairs_by_cue.get(tc.cue_id)
        if pair is None or not tc.ko.strip() or not table.is_firm(*pair, tc.cue_id):
            continue
        levels = _turn_levels(tc.ko)
        want = table.level_at(*pair, tc.cue_id)
        if len(levels) != 1 or levels[0] is None or same_level(levels[0], want):
            continue
        out.append(
            FirmViolation(
                scene_id=st.scene_id, cue_id=tc.cue_id, pair=f"{pair[0]}>{pair[1]}",
                table_level=want or "", line_level=levels[0], ko=tc.ko,
            )
        )
    return out


def retry_note(violations: list[FirmViolation]) -> str:
    """S3 retry block naming each line that broke a firm register row."""
    rows = []
    for violation in violations:
        turn = f" turn {violation.turn_index}" if violation.turn_index is not None else ""
        if violation.reason == "turn_count":
            detail = (
                f"needs {violation.expected_turns} ordered speaker turns"
                f" (was {violation.actual_turns}); preserve one '- ' line per turn"
            )
        else:
            detail = (
                f"must be {LEVEL_KO.get(violation.table_level, violation.table_level)}"
                f" (was {LEVEL_KO.get(violation.line_level, violation.line_level)})"
            )
        text = violation.turn_ko if violation.turn_ko is not None else violation.ko
        rows.append(f"  {violation.cue_id}{turn} {violation.pair}: {detail}: {text.replace(chr(10), ' / ')}")
    return "## REGISTER RETRY (these lines left their pair's firm REGISTERS level; translate the scene again with each pair at its level)\n" + "\n".join(rows)
