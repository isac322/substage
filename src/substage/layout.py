"""S5c layout: re-segment final Korean cues into readable display events.

English cue boundaries are a poor fit for Korean: short cues become
unreadably fast, long ones overflow two lines. Professional Korean subtitlers
re-segment; this stage applies their measured habits (official subtitles of
six videos, see Settings comments) generically:

1. **events** — one per cue with Korean text. Empty-Korean cues yield none;
   a filler-only cue (English "uh"/"oh"/"whoa"…, Korean "어"/"오"/"와"…) is
   dropped.
2. **split** — a dash (two-speaker) event lasting ≥ ``split_min_ms`` splits
   into one event per speaker; a one-speaker event lasting ≥ ``split_min_ms``
   with ≥2 Korean sentences (or ≥ ``split_en_chars`` English characters)
   splits at the sentence (else clause) boundary nearest its middle; an event
   whose Korean overflows 2 lines of ≤ cpl splits the same way at any length.
   Time is divided in proportion to the parts' characters with
   ``min_gap_ms`` between them; every piece needs ≥ ``split_piece_min_ms``.
3. **merge** — adjacent events in the same scene merge when the English gap is
   < ``merge_max_gap_ms``, the combined span is ≤ ``merge_max_span_ms``, the
   merged Korean has ≤ ``merge_max_chars`` characters and fits 2 lines of
   ≤ cpl at ≤ ``cps_max``, and either the first event's English does not finish
   its sentence (same speaker) or one event is too short/fast to read alone.
   Different speakers render as "- A" / "- B" (one line per turn, never more
   than two turns); one speaker's text joins and rewraps. Lyric, sound-only
   and on-screen-text cues never merge.
4. **timing** — start = first cue's English start. The end extends toward
   ``max(min_duration_ms, chars / cps_target)`` but at most ``extend_ms`` past
   the English end (more only to reach ``min_duration_ms``), never beyond
   ``max_duration_ms``. Events never overlap: the end stays ≥ ``min_gap_ms``
   before the next start, and a remaining gap below ``pause_min_ms`` snaps to
   a chained ``min_gap_ms`` gap.
5. **condense** — events still over ``cps_max`` go to the text model
   (stage ``layout-condense``) to be shortened to the characters their
   duration allows; a provider block keeps the text.

Characters are counted with ``validate.cps_chars`` (whitespace excluded), the
same count ``validate.check`` and the translator budget use.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

from pydantic import BaseModel

from .models import (
    Cue, OmittedAct, OutEvent, RetainedFragment, SemanticAttributionProvenance, SemanticLineage,
)
from .style import style_guide
from .validate import (
    LYRIC_MARK, _DASHES, _dual_line, _speakers, cps_chars, normalize_punct, reconcile_lineage,
    render_retained, rewrap,
)

# Settings fields that change the layout (part of the stage's checkpoint key).
KNOBS: tuple[str, ...] = (
    "cps_target",
    "cps_max",
    "cpl_limit",
    "cpl_one_line",
    "min_duration_ms",
    "short_min_ms",
    "short_text_chars",
    "max_duration_ms",
    "extend_ms",
    "min_gap_ms",
    "pause_min_ms",
    "merge_max_gap_ms",
    "merge_max_span_ms",
    "merge_max_chars",
    "split_min_ms",
    "split_piece_min_ms",
    "split_en_chars",
)


def knobs_tag(settings) -> str:
    """Stable 'name=value' string of every layout knob (checkpoint hash part)."""
    return ",".join(f"{k}={getattr(settings, k)}" for k in KNOBS)


@dataclass(frozen=True)
class _Turn:
    speaker: str  # normalized speaker key; unknown speakers get a unique key
    text: str  # one line, no dash
    fragments: tuple[RetainedFragment, ...] = ()


@dataclass
class _Ev:
    start: int
    en_end: int  # English end the event is timed from
    cue_ids: list[str]
    turns: list[_Turn]
    ko: str
    fixed: bool  # lyric / sound-only / on-screen text: never merged
    scene_first: str | None
    scene_last: str | None
    en_text: str  # English of the covered cues (sentence-continuation test)
    end: int = 0
    tags: list[str] = field(default_factory=list)
    semantic: bool = False
    omissions: list[OmittedAct] = field(default_factory=list)
    attribution_passes: list[SemanticAttributionProvenance] = field(default_factory=list)
    lyric: bool = False


# ---------------------------------------------------------------- timing


def fit_end(start: int, en_end: int, chars: int, next_start: int | None, settings) -> int:
    """End time for an event of `chars` characters (see module doc, step 4)."""
    s = settings
    need = start + max(s.min_duration_ms, math.ceil(chars * 1000 / s.cps_target))
    grow_cap = max(en_end + s.extend_ms, start + s.min_duration_ms)
    end = max(en_end, min(need, grow_cap, start + s.max_duration_ms))
    if next_start is None:
        return end
    limit = next_start - s.min_gap_ms
    if limit < end:  # never overlap the next event
        return limit if limit - start >= s.short_min_ms else end
    if limit - end < s.pause_min_ms and limit - start <= s.max_duration_ms:
        return limit  # a near-miss gap becomes a chained one
    return end


def available_ms(start: int, en_end: int, next_start: int | None, settings) -> int:
    """Longest display time a cue can get under the timing rules: the end an
    arbitrarily long text would be given, minus the start."""
    return max(0, fit_end(start, en_end, 10**9, next_start, settings) - start)


def _cps(ko: str, start: int, end: int) -> float:
    dur = (end - start) / 1000
    return cps_chars(ko) / dur if dur > 0 else math.inf


def _fit(ev: _Ev, next_start: int | None, settings) -> int:
    return fit_end(ev.start, ev.en_end, cps_chars(ev.ko), next_start, settings)


def _retime(events: list[_Ev], settings) -> None:
    for i, ev in enumerate(events):
        ev.end = _fit(ev, events[i + 1].start if i + 1 < len(events) else None, settings)


# ---------------------------------------------------------------- text


def _flat(text: str) -> str:
    return " ".join(line.strip().lstrip(_DASHES).strip() for line in text.splitlines() if line.strip())


def _fits(ko: str, cpl: int) -> bool:
    lines = ko.split("\n")
    return len(lines) <= 2 and all(len(line) <= cpl for line in lines)


def _render(turns: list[_Turn], cpl: int, one_line: int) -> str | None:
    """Render known retained voices; legacy events keep their old turn policy."""
    fragments = [fragment for turn in turns for fragment in turn.fragments]
    if fragments:
        text = render_retained(fragments, cpl, one_line)
    elif len(turns) == 2:
        text = "\n".join(_dual_line(t.text) for t in turns)
    elif len(turns) == 1:
        text = rewrap(turns[0].text, cpl, one_line)
    else:
        return None
    return text if _fits(text, cpl) else None


def _speaker_key(cue: Cue, speakers: dict[str, str | None]) -> str:
    name = (speakers.get(cue.id) or cue.speaker_hint or "").strip().lower()
    return name if name else f"?{cue.id}"


def _is_fixed(cue: Cue) -> bool:
    return cue.lyric or cue.forced or cue.source == "ost" or not cue.text.strip()


_EN_FILLER = re.compile(r"(?:u+h+|u+m+|o+h+|a+h+|whoa+|h*m{2,}|h+m+|huh|hey|u+g+h+|e+r+|e+h+)")
_KO_FILLER = re.compile(r"(?:[어음오와흠응헉아으에엇허휴하헐흐후윽억]|우와)+")


def _tokens(text: str) -> list[str]:
    return re.sub(r"[^\w\s]|_", " ", text).lower().split()


def is_filler(cue: Cue, ko: str) -> bool:
    """English and Korean are both bare interjections (no meaning to keep)."""
    en, kr = _tokens(cue.text), _tokens(ko)
    return (
        bool(en) and bool(kr)
        and all(_EN_FILLER.fullmatch(t) for t in en)
        and all(_KO_FILLER.fullmatch(t) for t in kr)
    )


_EN_SENTENCE_END = re.compile(r"[.?!…]['\"’”)\]]*$")
_EN_TRAIL = re.compile(r"(\.\.\.|…)['\"’”)\]]*$")


def _continues(en: str) -> bool:
    """The English stops mid-sentence (no final . ? ! …, or trails off '...')."""
    en = en.strip()
    return bool(en) and (not _EN_SENTENCE_END.search(en) or bool(_EN_TRAIL.search(en)))


# ---------------------------------------------------------------- build


def _absorbed_timing_cues(cues, ko, scene_of, lineage, source_cue) -> dict[str, list[str]]:
    """Assign only contiguous blank-source absorption, never timing from ancestry."""
    if lineage is None:
        return {}
    claims: dict[str, set[str]] = {}
    references = {
        cid: {source_cue[ref] for fragment in fragments for ref in fragment.refs if ref in source_cue}
        for cid, fragments in lineage.fragments_by_cue.items()
        if fragments and ko.get(cid, "").strip()
    }
    for index, owner in enumerate(cues):
        if owner.id not in references:
            continue
        for direction in (-1, 1):
            pos = index + direction
            while 0 <= pos < len(cues):
                candidate = cues[pos]
                if (
                    candidate.id not in references[owner.id] or ko.get(candidate.id, "").strip()
                    or _is_fixed(candidate) or scene_of.get(candidate.id) != scene_of.get(owner.id)
                ):
                    break
                claims.setdefault(candidate.id, set()).add(owner.id)
                pos += direction
    absorbed: dict[str, list[str]] = {}
    for index, owner in enumerate(cues):
        if owner.id not in references:
            continue
        for direction in (-1, 1):
            pos = index + direction
            while 0 <= pos < len(cues) and claims.get(cues[pos].id) == {owner.id}:
                absorbed.setdefault(owner.id, []).append(cues[pos].id)
                pos += direction
    return absorbed


def _initial(
    cues: list[Cue], ko: dict[str, str], scene_of: dict[str, str], speakers: dict[str, str | None],
    lineage: SemanticLineage | None = None,
) -> list[_Ev]:
    out: list[_Ev] = []
    by_id = {cue.id: cue for cue in cues}
    source_cue = {act.ref: act.cue_id for act in lineage.source.acts} if lineage is not None else {}
    ordered = sorted(cues, key=lambda cue: (cue.start_ms, cue.id))
    absorbed = _absorbed_timing_cues(ordered, ko, scene_of, lineage, source_cue)
    for cue in ordered:
        text = ko.get(cue.id, "").strip()
        fragments = lineage.fragments_by_cue.get(cue.id) if lineage is not None else None
        if not text or (fragments is None and not _is_fixed(cue) and is_filler(cue, text)):
            continue
        if fragments is not None:
            turns = []
            for n, fragment in enumerate(fragments):
                key = fragment.speaker.strip().lower() if fragment.speaker else f"?{cue.id}#{n}"
                turn = _Turn(key, _flat(fragment.text), (fragment,))
                if turns and not key.startswith("?") and turns[-1].speaker == key:
                    prev = turns[-1]
                    turns[-1] = _Turn(key, f"{prev.text} {turn.text}", (*prev.fragments, fragment))
                else:
                    turns.append(turn)
            if not turns:
                continue
            text = render_retained(fragments, 10**9, 10**9)
        else:
            dual = _speakers([line for line in text.split("\n") if line.strip()])
            if dual is not None:
                turns = [_Turn(f"?{cue.id}#{n}", t) for n, t in enumerate(dual)]
            else:
                turns = [_Turn(_speaker_key(cue, speakers), _flat(text))]
        # Refs retain source ancestry. Only an adjacent cue without its own
        # display, uniquely absorbed by this owner, contributes display timing.
        covered_ids = {cue.id, *absorbed.get(cue.id, ())}
        covered = sorted((by_id[cid] for cid in covered_ids), key=lambda item: (item.start_ms, item.id))
        scene = scene_of.get(covered[0].id)
        out.append(
            _Ev(
                start=min(item.start_ms for item in covered),
                en_end=max(item.end_ms for item in covered),
                cue_ids=[item.id for item in covered],
                turns=turns,
                ko=text,
                fixed=any(_is_fixed(item) for item in covered),
                scene_first=scene,
                scene_last=scene_of.get(covered[-1].id),
                en_text="\n".join(item.text for item in covered),
                semantic=fragments is not None,
                lyric=cue.lyric,
                omissions=[
                    omission for omission in lineage.omissions
                    if omission.ref.partition(":")[0] in covered_ids
                ] if lineage is not None else [],
            )
        )
    return out


# ---------------------------------------------------------------- split

_SENTENCE_GAP = re.compile(r"(?<=[.?!…~])\s+")
_CLAUSE_GAP = re.compile(r"(?<=,)\s+")


def _mid_split(text: str, clauses: bool) -> tuple[str, str] | None:
    """Split at the sentence boundary nearest the middle; with `clauses`, fall
    back to the comma boundary nearest the middle. None when there is none."""
    mid = len(text) / 2
    for rx in (_SENTENCE_GAP, _CLAUSE_GAP) if clauses else (_SENTENCE_GAP,):
        cuts = list(rx.finditer(text))
        if not cuts:
            continue
        m = min(cuts, key=lambda m: (abs(m.start() - mid), m.start()))
        a, b = text[: m.start()].strip(), text[m.end() :].strip()
        if a and b:
            return a, b
    return None


def _split_fragments(turn: _Turn, texts: list[str]) -> list[list[RetainedFragment]]:
    """Partition actual text, retaining act ancestry without claiming full meaning."""
    out: list[list[RetainedFragment]] = []
    offset = 0
    spans = []
    for fragment in turn.fragments:
        flat = _flat(fragment.text)
        spans.append((offset, offset + len(flat), fragment))
        offset += len(flat) + 1
    offset = 0
    for text in texts:
        flat = _flat(text)
        end = offset + len(flat)
        retained = []
        for start, stop, fragment in spans:
            left, right = max(start, offset), min(stop, end)
            if left >= right:
                continue
            piece = normalize_punct(_flat(fragment.text)[left - start:right - start]).strip()
            retained.append(fragment.model_copy(update={"text": piece}))
        out.append(retained)
        offset = end + 1
    return out


def _split_once(ev: _Ev, next_start: int | None, settings) -> list[_Ev] | None:
    if ev.fixed:
        return None
    cpl, one_line = settings.cpl_limit, settings.cpl_one_line
    overflow = not _fits(ev.ko, cpl)
    long = ev.en_end - ev.start >= settings.split_min_ms
    if len(ev.turns) >= 2:
        if not (overflow or (ev.semantic and long)):
            return None
        turn_sets = [[ev.turns[0]], ev.turns[1:]]
        texts = [
            rewrap(ev.turns[0].text, cpl, one_line),
            render_retained([f for t in ev.turns[1:] for f in t.fragments], cpl, one_line)
            if ev.semantic else rewrap(ev.turns[1].text, cpl, one_line),
        ]
    else:
        if not (long or overflow):
            return None
        long_en = len(" ".join(ev.en_text.split())) >= settings.split_en_chars
        parts = _mid_split(ev.turns[0].text, clauses=overflow or long_en)
        if parts is None and ev.semantic and len(ev.turns[0].fragments) > 1:
            fragment_texts = [_flat(fragment.text) for fragment in ev.turns[0].fragments]
            candidates = [
                (" ".join(fragment_texts[:cut]), " ".join(fragment_texts[cut:]))
                for cut in range(1, len(fragment_texts))
            ]
            parts = min(candidates, key=lambda pair: abs(len(pair[0]) - len(pair[1])))
        if parts is None:
            return None
        texts = [rewrap(normalize_punct(p), cpl, one_line) for p in parts]
        fragment_sets = _split_fragments(ev.turns[0], list(parts))
        turn_sets = [
            [_Turn(ev.turns[0].speaker, _flat(text), tuple(fragments))]
            for text, fragments in zip(texts, fragment_sets, strict=True)
        ]
    end = _fit(ev, next_start, settings)
    c1, c2 = (max(1, cps_chars(t)) for t in texts)
    t1 = ev.start + round((end - ev.start - settings.min_gap_ms) * c1 / (c1 + c2))
    s2 = t1 + settings.min_gap_ms
    if min(t1 - ev.start, end - s2) < settings.split_piece_min_ms:
        return None
    tags = [*ev.tags, "split"]
    first = replace(ev, en_end=t1, turns=turn_sets[0], ko=texts[0], scene_last=ev.scene_first, tags=tags)
    second = replace(ev, start=s2, en_end=end, turns=turn_sets[1], ko=texts[1], scene_first=ev.scene_last, tags=tags)
    first.cue_ids, second.cue_ids = list(ev.cue_ids), list(ev.cue_ids)
    return [first, second]


def _split_all(events: list[_Ev], settings) -> list[_Ev]:
    out: list[_Ev] = []
    queue = list(events)
    while queue:
        ev = queue.pop(0)
        parts = _split_once(ev, queue[0].start if queue else None, settings)
        if parts is None:
            out.append(ev)
        else:
            queue[:0] = parts  # pieces may split further
    return out


# ---------------------------------------------------------------- merge


def _merge_turns(a: list[_Turn], b: list[_Turn]) -> list[_Turn] | None:
    turns = list(a)
    for t in b:
        if turns and turns[-1].speaker == t.speaker:
            turns[-1] = _Turn(t.speaker, f"{turns[-1].text} {t.text}",
                              (*turns[-1].fragments, *t.fragments))
        else:
            turns.append(t)
    return turns if len(turns) <= 2 else None


def _weak(ev: _Ev, end: int, settings) -> bool:
    """Too short or too fast to read on its own."""
    dur = end - ev.start
    short_ok = cps_chars(ev.ko) <= settings.short_text_chars and dur >= settings.short_min_ms
    return (dur < settings.min_duration_ms and not short_ok) or _cps(ev.ko, ev.start, end) > settings.cps_max


def _try_merge(a: _Ev, b: _Ev, next_start: int | None, settings) -> _Ev | None:
    s = settings
    if a.fixed or b.fixed or a.scene_last != b.scene_first:
        return None
    if a.semantic != b.semantic:
        return None
    if b.start - a.en_end >= s.merge_max_gap_ms or max(a.en_end, b.en_end) - a.start > s.merge_max_span_ms:
        return None
    if cps_chars(a.ko) + cps_chars(b.ko) > s.merge_max_chars:
        return None
    turns = _merge_turns(a.turns, b.turns)
    if turns is None:
        return None
    same_speaker = len(turns) < len(a.turns) + len(b.turns)
    a_last_en = a.en_text.splitlines()[-1].strip() if a.en_text else ""
    b_last_en = b.en_text.splitlines()[-1].strip() if b.en_text else ""
    if b_last_en.endswith(("...", "…")) and cps_chars(a.ko) >= 6 and not _weak(a, _fit(a, b.start, s), s):
        return None
    weak = _weak(a, _fit(a, b.start, s), s) or _weak(b, _fit(b, next_start, s), s)
    if not (weak or (same_speaker and _continues(a_last_en))):
        return None
    ko = _render(turns, s.cpl_limit, s.cpl_one_line)
    if ko is None:
        return None
    m = _Ev(
        start=a.start,
        en_end=max(a.en_end, b.en_end),
        cue_ids=list(dict.fromkeys([*a.cue_ids, *b.cue_ids])),
        turns=turns,
        ko=ko,
        fixed=False,
        scene_first=a.scene_first,
        scene_last=b.scene_last,
        en_text=f"{a.en_text}\n{b.en_text}",
        tags=list(dict.fromkeys([*a.tags, *b.tags, "merge"])),
        semantic=a.semantic,
        omissions=[*a.omissions, *b.omissions],
        attribution_passes=[*a.attribution_passes, *b.attribution_passes],
    )
    if _cps(ko, m.start, _fit(m, next_start, s)) > s.cps_max:
        return None
    return m


def _merge_all(events: list[_Ev], settings) -> list[_Ev]:
    out = list(events)
    i = 0
    while i + 1 < len(out):
        nxt = out[i + 2].start if i + 2 < len(out) else None
        m = _try_merge(out[i], out[i + 1], nxt, settings)
        if m is None:
            i += 1
        else:
            out[i : i + 2] = [m]  # the merged event may absorb the next one too
    return out


# ---------------------------------------------------------------- condense


class _Condensed(BaseModel):
    event_id: str
    ko: str


class _CondenseBatch(BaseModel):
    items: list[_Condensed] = []


CONDENSE_PROMPT = """\
Shorten Korean video subtitle events that are too fast to read. Each event shows
its on-screen duration and the maximum number of characters (spaces not
counted) a viewer can read in that time. Rewrite each Korean event to at most
that many characters. Keep the meaning the English line needs for the plot,
each speaker's speech level (존댓말 vs 반말, never changed while shortening),
address terms, tone, names, and any punchline word the ear must catch. Cut
filler, honorific padding, obvious pronouns and
repetition. An event with two '- ' lines has two speakers: keep exactly two
'- ' lines, one per speaker. Use a single '\\n' if the text needs two lines.
The style guide (system text) decides what to cut and how to phrase it.

Events (id | duration | max chars | speakers | Korean):
{events}

English source:
{sources}

Return items only for events you shortened, with the full new Korean text.\
"""

SEMANTIC_CONDENSE_PROMPT = """\
Shorten the displayed Korean events to their reading budgets. Preserve source
meaning, essential answers, elapsed/relative time, list structure, register,
protected names and ambiguity. Source acts and retained voices are different:
lawful incidental backchannels may be omitted, essential answers must remain.
Do not invent a reply, infer unknown speakers, or require two voices just
because the English has two acts. Keep the actual Korean fragments in order.
Use '- ' only for two different known voices retained in the same display.

Events (id | duration | max chars | retained speakers | current Korean):
{events}

Original English and cue identities:
{sources}

Return items only for shortened events, with event_id and full new ko.
"""


CONDENSE_BATCH = 60


def _condense(gc, events: list[_Ev], settings, gate=None, *, lineage=None, cues=None) -> None:
    """Shorten events still over cps_max with the client's light model (the
    text model when one is configured), in place. With *gate*
    (:class:`substage.register.RegisterGate`) a candidate that moves a line off
    its register level is rejected."""
    from .gemini import GeminiBlocked  # local: layout stays importable without the SDK

    over = [
        (f"e{i + 1:04d}", ev)
        for i, ev in enumerate(events)
        if not ev.fixed and _cps(ev.ko, ev.start, ev.end) > settings.cps_max
    ]
    for k in range(0, len(over), CONDENSE_BATCH):
        batch = over[k : k + CONDENSE_BATCH]
        rows, sources = [], []
        for eid, ev in batch:
            dur = (ev.end - ev.start) / 1000
            budget = max(1, math.floor(dur * settings.cps_max))
            who = " / ".join(t.speaker for t in ev.turns if not t.speaker.startswith("?")) or "?"
            rows.append(f"{eid} | {dur:.1f}s | {budget} | {who} | {ev.ko.replace(chr(10), ' / ')}")
            sources.append(
                f"{eid}{' [' + ','.join(ev.cue_ids) + ']' if ev.semantic else ''}: "
                f"{ev.en_text.replace(chr(10), ' / ')}"
            )
        template = SEMANTIC_CONDENSE_PROMPT if any(ev.semantic for _, ev in batch) else CONDENSE_PROMPT
        prompt = template.format(events="\n".join(rows), sources="\n".join(sources))
        try:
            result: _CondenseBatch = gc.generate(
                stage="layout-condense",
                prompt=prompt,
                schema=_CondenseBatch,
                model=gc.settings.model_light,
                system=style_guide(),
            )
        except GeminiBlocked:
            if any(ev.semantic for _, ev in batch):
                raise
            continue  # legacy provider refusal keeps the texts
        by_id = dict(batch)
        for item in result.items:
            ev = by_id.pop(item.event_id, None)
            if ev is None or not item.ko.strip():
                continue
            if ev.semantic:
                from .translate import reattribute_semantic_fragments

                candidate = normalize_punct(item.ko.strip())
                if cps_chars(candidate) >= cps_chars(ev.ko):
                    continue
                if gate is not None and not gate.check("layout", ev.cue_ids, ev.ko, candidate):
                    continue
                attributed = reattribute_semantic_fragments(
                    gc, lineage.source, [cue for cue in cues if cue.id in ev.cue_ids],
                    candidate, [fragment for turn in ev.turns for fragment in turn.fragments],
                    stage="layout-condense:semantic-reattribute",
                )
                ev.ko = render_retained(attributed.fragments, settings.cpl_limit, settings.cpl_one_line)
                ev.turns = [
                    _Turn(fragment.speaker.strip().lower() if fragment.speaker else f"?{ev.cue_ids[0]}#{n}",
                          _flat(fragment.text), (fragment,))
                    for n, fragment in enumerate(attributed.fragments)
                ]
                replacement_refs = {omission.ref for omission in attributed.omissions}
                active_refs = {ref for fragment in attributed.fragments for ref in fragment.refs}
                ev.omissions = [
                    omission for omission in ev.omissions
                    if omission.ref not in replacement_refs and omission.ref not in active_refs
                ] + attributed.omissions
                ev.attribution_passes = [*ev.attribution_passes, attributed.provenance]
                ev.tags = [*ev.tags, "condense", "semantic-reattribute"]
                continue
            new = rewrap(normalize_punct(item.ko.strip()), settings.cpl_limit, settings.cpl_one_line)
            new_turns = _speakers(new.split("\n"))
            was_dual = len(ev.turns) == 2
            if (new_turns is not None) != was_dual or cps_chars(new) >= cps_chars(ev.ko):
                continue  # broke the speaker structure or did not shorten
            if gate is not None and not gate.check("layout", ev.cue_ids, ev.ko, new):
                continue  # left the register table's level or pronoun
            ev.ko = new
            if was_dual:
                ev.turns = [_Turn(t.speaker, s) for t, s in zip(ev.turns, new_turns, strict=True)]
            else:
                ev.turns = [_Turn(ev.turns[0].speaker, _flat(new))]
            ev.tags = [*ev.tags, "condense"]


# ---------------------------------------------------------------- driver


def layout(
    cues: list[Cue],
    ko: dict[str, str],
    scene_of: dict[str, str],
    speakers: dict[str, str | None],
    settings,
    gc=None,
    gate=None,
    *,
    lineage: SemanticLineage | None = None,
) -> list[OutEvent]:
    """Display events for the video: split -> merge -> timing, then (with `gc`)
    condense events still over cps_max and retime. Deterministic given the
    model's answers; `speakers` maps cue id -> speaker id (S3 metadata);
    `gate` (a :class:`substage.register.RegisterGate`) vetoes condensed
    candidates that change the register level."""
    if lineage is not None:
        reconcile_lineage(lineage, ko)
    initial = _initial(cues, ko, scene_of, speakers, lineage)
    for event in initial:
        if event.semantic:
            reserve = len(LYRIC_MARK) if event.lyric else 0
            event.ko = render_retained(
                [fragment for turn in event.turns for fragment in turn.fragments],
                settings.cpl_limit - reserve, settings.cpl_one_line - reserve,
            )
            if event.lyric and event.ko:
                event.ko = LYRIC_MARK + event.ko
    events = _merge_all(_split_all(initial, settings), settings)
    _retime(events, settings)
    if gc is not None:
        _condense(gc, events, settings, gate, lineage=lineage, cues=cues)
        _retime(events, settings)
    return [
        OutEvent(
            start_ms=e.start, end_ms=e.end, ko=e.ko, cue_ids=e.cue_ids,
            fragments=[fragment for turn in e.turns for fragment in turn.fragments],
            omissions=e.omissions,
            attribution_passes=e.attribution_passes,
        )
        for e in events
    ]


def stats(events: list[OutEvent], settings) -> dict:
    """Reading-speed summary of display events (dry runs, reports)."""
    if not events:
        return {"events": 0, "cps_over_rate": None, "median_cps": None, "median_duration_ms": None}
    cps = sorted(_cps(e.ko, e.start_ms, e.end_ms) for e in events)
    durs = sorted(e.end_ms - e.start_ms for e in events)
    return {
        "events": len(events),
        "cps_over_rate": round(sum(c > settings.cps_max for c in cps) / len(events), 4),
        "median_cps": round(cps[len(cps) // 2], 2),
        "median_duration_ms": durs[len(durs) // 2],
    }
