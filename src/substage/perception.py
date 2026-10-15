"""Selective, cue-keyed multimodal perception.

The viewing pass in :mod:`substage.narrative` remains the coarse narrative pass.
This module is the opt-in DPAF path: it asks Gemini about a bounded set of
existing cues and returns advisory records without changing the subtitle timing
spine or inventing cue identifiers.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from typing import Any

from .config import Workspace
from .gemini import Clip, GeminiBlocked, GeminiClient, UploadedFile
from .models import (
    ChunkObservations,
    Cue,
    CueObservation,
    CuePerception,
    VideoInfo,
    MediaNeed,
    PerceptionCandidate,
    PerceptionLedger,
    Scene,
    SubTurnObservation,
    SemanticLineage,
    SemanticSource,
    TranslatedCue,
    VisualDeixis,
)
from .store import Checkpoints, input_hash

logger = logging.getLogger(__name__)

_DEFAULT_PAD_MS = 1_000
_DEFAULT_WINDOW_MS = 45_000
_DEFAULT_MAX_CUES = 12
_MAX_EVIDENCE = 4
_MAX_RENDER_EVIDENCE = 2
_MAX_PROMPT_OBSERVATION_CHARS = 320


def validate_semantic_source(source: SemanticSource, cues: Sequence[Cue]) -> SemanticSource:
    """Validate source identities/spans/order, never the model's interpretation."""
    by_id = {cue.id: cue for cue in cues}
    positions = {cue.id: index for index, cue in enumerate(cues)}
    refs: set[str] = set()
    covered: dict[str, set[int]] = {cue.id: set() for cue in cues}
    previous: tuple[int, int, int] | None = None
    ordinals: dict[tuple[str, int], int] = {}
    turns: dict[str, int] = {}
    cursors: dict[str, int] = {}
    normalized_acts: list[SourceAct] = []
    ref_aliases: dict[str, str] = {}
    for original_act in source.acts:
        act = original_act
        cue = by_id.get(act.cue_id)
        if cue is None:
            raise ValueError(f"invalid semantic source identity: {act.ref}")
        if not cue.text.strip():
            ref_aliases[act.ref] = ""
            continue
        cue_count = sum(count for (cid, _), count in ordinals.items() if cid == cue.id)
        turn = turns.get(cue.id, 0)
        pair = (cue.id, act.turn_ordinal)
        canonical_ref = f"{act.cue_id}:t{act.turn_ordinal}:a{act.act_ordinal}"
        if (
            act.turn_ordinal > 1
            and act.turn_ordinal == turn + 1
            and ordinals.get(pair, 0) == 0
            and act.act_ordinal == cue_count + 1
        ):
            old_ref = act.ref
            act = act.model_copy(update={
                "act_ordinal": 1,
                "ref": f"{act.cue_id}:t{act.turn_ordinal}:a1",
            })
            if old_ref != act.ref:
                ref_aliases[old_ref] = act.ref
        elif act.ref != canonical_ref:
            raise ValueError(f"invalid semantic source identity: {act.ref}")
        if act.ref in refs:
            raise ValueError(f"duplicate semantic source ref: {act.ref}")
        cursor = cursors.get(cue.id, 0)
        if (
            not 0 <= act.start_char < act.end_char <= len(cue.text)
            or cue.text[act.start_char:act.end_char] != act.text
        ):
            candidate = act.text.strip()
            idx = cue.text.find(candidate, cursor) if candidate else -1
            if (
                candidate
                and idx >= cursor
                and cue.text[cursor:idx].strip() == ""
                and cue.text[idx:idx + len(candidate)] == candidate
            ):
                act = act.model_copy(update={
                    "start_char": idx,
                    "end_char": idx + len(candidate),
                    "text": cue.text[idx:idx + len(candidate)],
                })
            elif not 0 <= act.start_char < act.end_char <= len(cue.text):
                raise ValueError(f"invalid semantic source span: {act.ref}")
            else:
                raise ValueError(f"semantic source span text mismatch: {act.ref}")
        order = (positions[cue.id], act.start_char, act.end_char)
        if previous is not None and order <= previous:
            raise ValueError(f"semantic source acts out of order: {act.ref}")
        previous = order
        span = set(range(act.start_char, act.end_char))
        if covered[cue.id] & span:
            raise ValueError(f"overlapping semantic source spans: {act.ref}")
        covered[cue.id].update(span)
        if act.turn_ordinal not in {turn, turn + 1} or (turn == 0 and act.turn_ordinal != 1):
            raise ValueError(f"non-contiguous semantic source turns: {act.ref}")
        turns[cue.id] = act.turn_ordinal
        pair = (cue.id, act.turn_ordinal)
        if act.act_ordinal != ordinals.get(pair, 0) + 1:
            raise ValueError(f"non-contiguous semantic source acts: {act.ref}")
        ordinals[pair] = act.act_ordinal
        cursors[cue.id] = act.end_char
        refs.add(act.ref)
        normalized_acts.append(act)
    for cue in cues:
        required = {index for index, char in enumerate(cue.text) if not char.isspace()}
        if not required.issubset(covered[cue.id]):
            raise ValueError(f"semantic extraction left source text uncovered: {cue.id}")
    normalized_relations: list[SourceRelation] = []
    for relation in source.relations:
        relation_refs: list[str] = []
        removed_empty = False
        for ref in relation.refs:
            mapped = ref_aliases.get(ref, ref)
            if ref in ref_aliases and not mapped:
                removed_empty = True
                continue
            if mapped in relation_refs:
                raise ValueError(
                    f"invalid semantic relation refs: {relation.kind} "
                    f"({relation.refs}; must be unique exact act.ref values from acts)"
                )
            relation_refs.append(mapped)
        if not set(relation_refs).issubset(refs):
            raise ValueError(
                f"invalid semantic relation refs: {relation.kind} "
                f"({relation.refs}; must be unique exact act.ref values from acts)"
            )
        if not relation_refs or (removed_empty and len(relation_refs) <= 1):
            continue
        normalized_relations.append(relation.model_copy(update={"refs": relation_refs}))
    return source.model_copy(update={
        "acts": [
            act.model_copy(update={"speaker": (
                None if act.speaker is None or act.speaker.strip().casefold()
                in {"", "unknown", "?", "unidentified"} else act.speaker.strip()
            )})
            for act in normalized_acts
        ],
        "relations": normalized_relations,
    })


def semantic_lexical_text(text: str) -> str:
    """Remove target display prefixes only, keeping lexical text and line order."""
    return "\n".join(
        re.sub(r"^\s*(?:(?:-\s+)|(?:♪\s*))*", "", line)
        for line in text.splitlines()
    )


def semantic_text_identity(text: str) -> str:
    """Exact lexical order, independent of target dashes, lyric marks or wrapping."""
    return re.sub(r"\s+", "", semantic_lexical_text(text))


def validate_semantic_lineage(
    lineage: SemanticLineage,
    cues: Sequence[Cue],
    translated: Sequence[TranslatedCue] | None = None,
) -> SemanticLineage:
    """Validate actual target references and coverage independently of fidelity."""
    source = validate_semantic_source(lineage.source, cues)
    ids = {cue.id for cue in cues}
    refs = {act.ref for act in lineage.source.acts}
    retained: set[str] = set()
    prior: set[str] = set()
    for cue_id, fragments in lineage.fragments_by_cue.items():
        if cue_id not in ids:
            raise ValueError(f"unknown semantic target cue: {cue_id}")
        for fragment in fragments:
            if not fragment.text.strip():
                raise ValueError(f"empty semantic target fragment: {cue_id}")
            if len(set(fragment.refs)) != len(fragment.refs) or not set(fragment.refs).issubset(refs):
                raise ValueError(
                    f"invalid retained refs: {cue_id} "
                    f"({fragment.refs}; must be unique exact act.ref values from source.acts)"
                )
            if not set(fragment.prior_refs).issubset(refs):
                raise ValueError(f"invalid prior refs: {cue_id}")
            if fragment.attribution == "unverified" and (fragment.refs or fragment.speaker):
                raise ValueError(f"unverified fragment asserts current attribution: {cue_id}")
            retained.update(fragment.refs)
            prior.update(fragment.prior_refs)
    omitted: set[str] = set()
    for omission in lineage.omissions:
        if omission.ref not in refs or omission.ref in omitted or not omission.reason.strip():
            raise ValueError(
                f"invalid semantic omission: {omission.ref} "
                "(must be a unique existing act.ref in source.acts with a non-empty reason; "
                "sound-only cues without source.acts need no omission record)"
            )
        omitted.add(omission.ref)
    if retained & omitted:
        overlap = sorted(retained & omitted)
        raise ValueError(f"semantic acts declared both retained and omitted: {overlap}")
    if refs - retained - omitted - prior:
        missing = sorted(refs - retained - omitted - prior)
        raise ValueError(
            f"semantic source acts have neither retained nor omitted records: {missing} "
            "(cite each ref in the fragment.refs of the cue displaying its meaning, "
            "or list it in omissions[] with a non-empty reason)"
        )
    if translated is not None:
        by_id = {cue.cue_id: cue for cue in translated}
        if len(by_id) != len(translated) or set(by_id) != ids or set(lineage.fragments_by_cue) != ids:
            raise ValueError("semantic writer must retain exactly the existing cue identities")
        for cue_id, fragments in lineage.fragments_by_cue.items():
            expected = "".join(semantic_text_identity(fragment.text) for fragment in fragments)
            actual = semantic_text_identity(by_id[cue_id].ko)
            if actual != expected:
                raise ValueError(f"semantic lineage does not match actual Korean: {cue_id}")
    return lineage.model_copy(update={"source": source, "structural_valid": True})


def _text(value: Any) -> str | None:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value or None


def _list_text(values: Iterable[Any], *, limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _text(value)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def _confidence(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _candidate_key(item: PerceptionCandidate) -> tuple[str, float]:
    return (item.value.casefold(), -(item.confidence if item.confidence is not None else -1.0))


def _normalize_candidates(values: Iterable[Any]) -> list[PerceptionCandidate]:
    candidates: list[PerceptionCandidate] = []
    seen: set[tuple[str, float | None]] = set()
    for value in values:
        try:
            item = value if isinstance(value, PerceptionCandidate) else PerceptionCandidate.model_validate(value)
        except Exception:
            continue
        item = item.model_copy(
            update={
                "value": _text(item.value) or "",
                "confidence": _confidence(item.confidence),
                "evidence": _list_text(item.evidence, limit=_MAX_EVIDENCE),
            }
        )
        if not item.value:
            continue
        key = (item.value.casefold(), item.confidence)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(item)
    return sorted(candidates, key=_candidate_key)


def _normalize_keyframe(value: int | None, lower_ms: int | None, upper_ms: int | None) -> int | None:
    if value is None:
        return None
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return None
    if timestamp < 0:
        return None
    if lower_ms is not None and timestamp < lower_ms:
        return None
    if upper_ms is not None and timestamp > upper_ms:
        return None
    return timestamp


def _normalize_deixis(
    value: VisualDeixis | dict[str, Any] | None,
    *,
    lower_ms: int | None,
    upper_ms: int | None,
) -> VisualDeixis | None:
    if value is None:
        return None
    try:
        item = value if isinstance(value, VisualDeixis) else VisualDeixis.model_validate(value)
    except Exception:
        return None
    item = item.model_copy(
        update={
            "target": _text(item.target),
            "relation": _text(item.relation),
            "description": _text(item.description),
            "keyframe_ms": _normalize_keyframe(item.keyframe_ms, lower_ms, upper_ms),
            "confidence": _confidence(item.confidence),
            "evidence": _list_text(item.evidence, limit=_MAX_EVIDENCE),
        }
    )
    if not any((item.target, item.relation, item.description, item.keyframe_ms is not None)):
        return None
    return item


def _normalize_turn(
    value: SubTurnObservation | dict[str, Any],
    *,
    lower_ms: int | None,
    upper_ms: int | None,
) -> SubTurnObservation | None:
    try:
        item = value if isinstance(value, SubTurnObservation) else SubTurnObservation.model_validate(value)
    except Exception:
        return None
    return item.model_copy(
        update={
            "text": _text(item.text),
            "speaker_candidates": _normalize_candidates(item.speaker_candidates),
            "addressee_candidates": _normalize_candidates(item.addressee_candidates),
            "delivery": _text(item.delivery),
            "prosody": _text(item.prosody),
            "visual_deixis": _normalize_deixis(item.visual_deixis, lower_ms=lower_ms, upper_ms=upper_ms),
            "ambiguity": _text(item.ambiguity),
            "keyframe_ms": _normalize_keyframe(item.keyframe_ms, lower_ms, upper_ms),
            "confidence": _confidence(item.confidence),
            "evidence": _list_text(item.evidence, limit=_MAX_EVIDENCE),
        }
    )


def normalize_cue_perception(
    value: CuePerception | dict[str, Any] | None,
    cue_id: str,
    *,
    window_start_ms: int | None = None,
    window_end_ms: int | None = None,
) -> CuePerception:
    """Return a canonical advisory record for exactly ``cue_id``.

    A mismatching model id is discarded rather than re-keyed.  This is the
    important no-new-ID invariant: a caller can only get a record for a cue it
    already supplied.  Malformed fields degrade independently where possible;
    a wholly malformed record becomes an empty, schema-valid record.
    """
    if not cue_id:
        raise ValueError("cue_id must be non-empty")
    if value is None:
        return CuePerception(cue_id=cue_id)
    try:
        if isinstance(value, dict):
            supplied = value.get("cue_id")
            if supplied not in (None, cue_id):
                return CuePerception(cue_id=cue_id)
            raw = dict(value)
            raw["cue_id"] = cue_id
            item = CuePerception.model_validate(raw)
        else:
            if value.cue_id != cue_id:
                return CuePerception(cue_id=cue_id)
            item = value
    except Exception:
        return CuePerception(cue_id=cue_id)

    turns: list[SubTurnObservation] = []
    for turn in item.turns or ():
        normalized = _normalize_turn(turn, lower_ms=window_start_ms, upper_ms=window_end_ms)
        if normalized is not None:
            turns.append(normalized)
    return item.model_copy(
        update={
            "cue_id": cue_id,
            "speaker_candidates": _normalize_candidates(item.speaker_candidates),
            "addressee_candidates": _normalize_candidates(item.addressee_candidates),
            "delivery": _text(item.delivery),
            "prosody": _text(item.prosody),
            "visual_deixis": _normalize_deixis(item.visual_deixis, lower_ms=window_start_ms, upper_ms=window_end_ms),
            "ambiguity": _text(item.ambiguity),
            "keyframe_ms": _normalize_keyframe(item.keyframe_ms, window_start_ms, window_end_ms),
            "confidence": _confidence(item.confidence),
            "evidence": _list_text(item.evidence, limit=_MAX_EVIDENCE),
            "turns": turns or None,
        }
    )


def normalize_perception_ledger(
    value: PerceptionLedger | dict[str, Any] | None,
    cues: Sequence[Cue] | None = None,
    *,
    cue_ids: Iterable[str] | None = None,
    window_start_ms: int | None = None,
    window_end_ms: int | None = None,
) -> PerceptionLedger:
    """Canonicalize a ledger and drop every key not in the existing cue set."""
    allowed = {c.id for c in cues} if cues is not None else None
    if cue_ids is not None:
        requested = {str(cid) for cid in cue_ids if str(cid)}
        allowed = requested if allowed is None else allowed & requested
    try:
        if value is None:
            raw_cues: dict[str, Any] = {}
        elif isinstance(value, PerceptionLedger):
            raw_cues = dict(value.cues)
        elif isinstance(value, dict):
            payload = value.get("cues", value)
            raw_cues = payload if isinstance(payload, dict) else {}
        else:
            raw_cues = {}
    except Exception:
        raw_cues = {}

    keys = sorted(set(raw_cues) if allowed is None else set(raw_cues) & allowed)
    # Include requested cues as empty records so API failures remain advisory
    # and callers can distinguish "asked but unavailable" from "not selected".
    if allowed is not None:
        keys = sorted(set(keys) | allowed)
    result: dict[str, CuePerception] = {}
    for cid in keys:
        record = normalize_cue_perception(
            raw_cues.get(cid),
            cid,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
        )
        result[cid] = record
    return PerceptionLedger(cues=result)


def _render_candidates(items: Sequence[PerceptionCandidate]) -> str:
    rows: list[str] = []
    for item in items:
        value = item.value.strip()
        if not value:
            continue
        confidence = f" ({item.confidence:.2f})" if item.confidence is not None else ""
        rows.append(value + confidence)
    return ", ".join(rows)


def _render_turn(turn: SubTurnObservation, index: int) -> str:
    parts: list[str] = []
    speakers = _render_candidates(turn.speaker_candidates)
    addressees = _render_candidates(turn.addressee_candidates)
    if speakers:
        parts.append(f"speaker?={speakers}")
    if addressees:
        parts.append(f"to?={addressees}")
    for key in ("delivery", "prosody", "ambiguity"):
        if value := getattr(turn, key):
            parts.append(f"{key}={value}")
    if turn.visual_deixis is not None:
        vd = turn.visual_deixis
        visual = [x for x in (vd.relation, vd.target, vd.description) if x]
        if visual:
            parts.append("deixis=" + " / ".join(visual))
    if turn.media_need is not None and turn.media_need != MediaNeed.NONE:
        parts.append(f"media_need={turn.media_need.value}")
    if turn.keyframe_ms is not None:
        parts.append(f"keyframe={turn.keyframe_ms}ms")
    return f"turn{index}: " + "; ".join(parts) if parts else ""


def render_cue_perception(perception: CuePerception) -> str:
    """Render one record deterministically for an advisory text prompt."""
    item = normalize_cue_perception(perception, perception.cue_id)
    parts: list[str] = []
    speakers = _render_candidates(item.speaker_candidates)
    addressees = _render_candidates(item.addressee_candidates)
    if speakers:
        parts.append(f"speaker?={speakers}")
    if addressees:
        parts.append(f"to?={addressees}")
    for key in ("delivery", "prosody", "ambiguity"):
        if value := getattr(item, key):
            parts.append(f"{key}={value}")
    if item.visual_deixis is not None:
        vd = item.visual_deixis
        visual = [x for x in (vd.relation, vd.target, vd.description) if x]
        if visual:
            parts.append("deixis=" + " / ".join(visual))
    if item.media_need is not None and item.media_need != MediaNeed.NONE:
        parts.append(f"media_need={item.media_need.value}")
    if item.keyframe_ms is not None:
        parts.append(f"keyframe={item.keyframe_ms}ms")
    if item.turns:
        parts.extend(row for i, turn in enumerate(item.turns, 1) if (row := _render_turn(turn, i)))
    if item.evidence:
        parts.append("evidence=" + " / ".join(item.evidence[:_MAX_RENDER_EVIDENCE]))
    return "; ".join(parts)


def render_perception_ledger(
    ledger: PerceptionLedger | dict[str, Any] | None,
    cue_ids: Sequence[str] | None = None,
) -> str:
    """Render only requested existing cue ids, in stable caller/order order."""
    normalized = normalize_perception_ledger(ledger, cue_ids=cue_ids)
    ids = list(cue_ids) if cue_ids is not None else sorted(normalized.cues)
    rows = []
    for cid in ids:
        item = normalized.cues.get(cid)
        if item is None:
            continue
        rendered = render_cue_perception(item)
        if rendered:
            rows.append(f"  {cid}: {rendered}")
    if not rows:
        return ""
    return (
        "## ADVISORY PERCEPTION (uncertain AV evidence; do not hard-lock; "
        "preserve the source cue IDs)\n" + "\n".join(rows)
    )


def _cue_line(cue: Cue) -> str:
    text = re.sub(r"\s+", " ", cue.text).strip()
    suffix = "".join((" ♪" if cue.lyric else "", " " + " ".join(cue.sound_cues) if cue.sound_cues else ""))
    return f"{cue.id}|{cue.start_ms}-{cue.end_ms}|{cue.speaker_hint or '-'}|{text}{suffix}"


def _observation_context(
    observations: Iterable[ChunkObservations | CueObservation | CuePerception],
    cue_ids: set[str],
    start_ms: int,
    end_ms: int,
) -> str:
    rows: list[str] = []
    for observation in observations:
        if isinstance(observation, ChunkObservations):
            if observation.end_ms <= start_ms or observation.start_ms >= end_ms:
                continue
            if observation.summary:
                summary = (_text(observation.summary) or "")[:_MAX_PROMPT_OBSERVATION_CHARS]
                rows.append(f"window-summary: {summary}")
            for cue in observation.cues:
                if cue.cue_id not in cue_ids:
                    continue
                fields = [
                    (_text(value) or "")[:_MAX_PROMPT_OBSERVATION_CHARS]
                    for value in (cue.speaker, cue.addressee, cue.delivery, cue.nonverbal)
                    if value
                ]
                if fields:
                    rows.append(f"{cue.cue_id}: " + "; ".join(fields))
        elif getattr(observation, "cue_id", None) in cue_ids:
            cid = str(observation.cue_id)
            if isinstance(observation, CuePerception):
                rendered = render_cue_perception(observation)
            else:
                fields = [
                    (_text(value) or "")[:_MAX_PROMPT_OBSERVATION_CHARS]
                    for value in (
                        getattr(observation, "speaker", None),
                        getattr(observation, "addressee", None),
                        getattr(observation, "delivery", None),
                        getattr(observation, "nonverbal", None),
                    )
                    if value
                ]
                for i, turn in enumerate(getattr(observation, "turns", None) or (), 1):
                    if rendered_turn := _render_turn(turn, i):
                        fields.append(rendered_turn[:_MAX_PROMPT_OBSERVATION_CHARS])
                rendered = "; ".join(fields)
            if rendered:
                rows.append(f"{cid}: {rendered}")
    return "\n".join(rows[:32])


def format_perception_prompt(
    video: VideoInfo,
    start_ms: int,
    end_ms: int,
    cues: Sequence[Cue],
    observations: Iterable[ChunkObservations | CueObservation | CuePerception] = (),
) -> str:
    """Format one bounded prompt; no cue outside ``cues`` is mentioned."""
    cue_ids = {cue.id for cue in cues}
    cue_lines = "\n".join(_cue_line(cue) for cue in cues) or "(no cues)"
    context = _observation_context(observations, cue_ids, start_ms, end_ms) or "(none)"
    return f"""You are a selective subtitle perception assistant. Inspect only the video window {start_ms}-{end_ms}ms and the existing cue lines below.
Return JSON matching PerceptionLedger. Every returned record MUST use one of these exact existing cue IDs: {', '.join(cue.id for cue in cues) or '(none)'}. Never invent, rename, merge, split, or omit the cue identity. Return advisory evidence only; uncertainty is preferable to guessing.
For each cue, report likely speaker/addressee candidates, delivery/prosody, picture-dependent deixis, ambiguity, media_need, and an absolute keyframe_ms only when useful. Use turns only when one displayed cue contains two observable turns, with at most two turns. Do not translate or rewrite subtitle text.

VIDEO: {video.title} ({video.year or '?'})
CUE WINDOW:
{cue_lines}
EXISTING VIEWING OBSERVATIONS (bounded to this window):
{context}
"""


def _windows(cues: Sequence[Cue], *, max_window_ms: int, pad_ms: int, max_cues: int) -> list[tuple[list[Cue], int, int]]:
    ordered = sorted(cues, key=lambda cue: (cue.start_ms, cue.end_ms, cue.id))
    groups: list[list[Cue]] = []
    for cue in ordered:
        if not groups:
            groups.append([cue])
            continue
        group = groups[-1]
        span = max([cue.end_ms, *(item.end_ms for item in group)]) - group[0].start_ms
        if len(group) >= max_cues or span > max_window_ms:
            groups.append([cue])
        else:
            group.append(cue)
    windows: list[tuple[list[Cue], int, int]] = []
    for group in groups:
        start = max(0, group[0].start_ms - pad_ms)
        end = max(start + 1, max(item.end_ms for item in group) + pad_ms)
        windows.append((group, start, end))
    return windows


def selective_perception_pass(
    ws: Workspace,
    gc: GeminiClient,
    video: VideoInfo,
    proxy: UploadedFile,
    cues: Sequence[Cue],
    scenes: Sequence[Scene] | None = None,
    observations: Iterable[ChunkObservations | CueObservation | CuePerception] = (),
    *,
    cue_ids: Iterable[str] | None = None,
    model: str | None = None,
    window_pad_ms: int = _DEFAULT_PAD_MS,
    max_window_ms: int = _DEFAULT_WINDOW_MS,
    max_cues_per_call: int = _DEFAULT_MAX_CUES,
) -> PerceptionLedger:
    """Opt-in selective Gemini AV pass for a bounded set of existing cues.

    Calls are independently checkpointed by prompt, model, media URI, cue
    payload, and bounded observations. Provider, budget, or validation failures
    propagate to the caller; blocked media responses remain advisory but are
    not checkpointed as successful empty perception.
    """
    if window_pad_ms < 0 or max_window_ms <= 0 or max_cues_per_call <= 0:
        raise ValueError("window_pad_ms must be >= 0; max_window_ms and max_cues_per_call must be > 0")
    by_id = {cue.id: cue for cue in cues}
    selected_ids = {str(cid) for cid in cue_ids or () if str(cid) in by_id}
    if scenes is not None:
        scene_ids = {cid for scene in scenes for cid in scene.cue_ids if cid in by_id}
        selected_ids = scene_ids if cue_ids is None else scene_ids & selected_ids
    selected = [cue for cue in cues if not selected_ids or cue.id in selected_ids]
    if cue_ids is not None or scenes is not None:
        selected = [cue for cue in selected if cue.id in selected_ids]
    if not selected:
        return PerceptionLedger()

    obs_list = list(observations)
    use_model = model or getattr(gc.settings, "dpaf_perception_model", None) or gc.settings.model_main
    cp = Checkpoints(ws, "s2v_perception")
    result: dict[str, CuePerception] = {}
    for group, start_ms, end_ms in _windows(
        selected, max_window_ms=max_window_ms, pad_ms=window_pad_ms, max_cues=max_cues_per_call
    ):
        group_ids = [cue.id for cue in group]
        prompt = format_perception_prompt(video, start_ms, end_ms, group, obs_list)
        overlap_obs = [
            observation.model_dump_json() if hasattr(observation, "model_dump_json") else str(observation)
            for observation in obs_list
            if not isinstance(observation, ChunkObservations)
            or (observation.start_ms < end_ms and observation.end_ms > start_ms)
        ]
        h = input_hash(
            proxy.uri,
            use_model,
            prompt,
            video.model_dump_json(),
            *[cue.model_dump_json() for cue in group],
            *overlap_obs,
        )
        key = f"{start_ms}-{end_ms}-{'_'.join(group_ids)}"
        hit: PerceptionLedger | None = cp.get(key, h, PerceptionLedger)
        if hit is None:
            try:
                generated = gc.generate(
                    stage=f"s2v_perception:{key}",
                    prompt=prompt,
                    schema=PerceptionLedger,
                    clips=[Clip(file=proxy, start_s=start_ms / 1000.0, end_s=end_ms / 1000.0)],
                    model=use_model,
                )
                hit = normalize_perception_ledger(
                    generated,
                    cue_ids=group_ids,
                    window_start_ms=start_ms,
                    window_end_ms=end_ms,
                )
            except GeminiBlocked as exc:
                logger.warning("s2v_perception window %s blocked (%s); leaving it uncached", key, exc.reason)
                hit = normalize_perception_ledger(None, cue_ids=group_ids)
            else:
                cp.put(key, h, hit)
        for cid in group_ids:
            result[cid] = normalize_cue_perception(
                hit.cues.get(cid), cid, window_start_ms=start_ms, window_end_ms=end_ms
            )
    return PerceptionLedger(cues={cid: result[cid] for cid in sorted(result)})


# Explicit alias for callers that prefer the layer's artifact name.
build_perception_ledger = selective_perception_pass


__all__ = [
    "build_perception_ledger",
    "format_perception_prompt",
    "normalize_cue_perception",
    "normalize_perception_ledger",
    "render_cue_perception",
    "render_perception_ledger",
    "selective_perception_pass",
    "validate_semantic_source",
    "validate_semantic_lineage",
    "semantic_text_identity",
    "semantic_lexical_text",
]
