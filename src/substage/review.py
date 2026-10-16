"""S4 windowed review and S4b finale: edits only, plus the blind Korean read.

All LLM calls produce edits/flags — never rewrites of the whole text. Stages:
  s4_review   bilingual windows of ~3 scenes, checkpoints per window
  s4b_finale  one whole-video bilingual audit call (register arcs, callbacks,
              ambiguity breaches, drift)
  s4b_blind   Korean-only read on the light model — 번역투, unclear referents
  s4b_regen   per-scene regeneration of flagged lines
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from importlib import resources
from typing import Any

from pydantic import BaseModel, Field
from pathlib import Path
from .config import Workspace, narrative_context, narrative_text
from .gemini import GeminiBlocked, GeminiClient
from .llm import ClefDecisionsClient
from .models import (
    Cue,
    CueEdit,
    CueObservation,
    ContextMap,
    ReadFlag,
    Scene,
    SceneTranslation,
    Snapshot,
)
from .levels import addressed_levels, speaker_turns
from .register import RegisterTable, binary, render_scene_register, same_level
from .store import Checkpoints, input_hash
from .style import PRO_SUBTITLE_LAWS, with_guide
from .translate import CueSpan, _NON_TRANSLATED, _fmt_ms, detect_syntactic_spans
from .validate import ost_kind, rewrap

WINDOW_SCENES = 3

_STAGE_REVIEW = "s4_review"
_STAGE_FINALE = "s4b_finale"
_STAGE_BLIND = "s4b_blind"
_STAGE_REGEN = "s4b_regen"

logger = logging.getLogger(__name__)


class _EditBatch(BaseModel):
    edits: list[CueEdit] = Field(default_factory=list)


class _FlagBatch(BaseModel):
    flags: list[ReadFlag] = Field(default_factory=list)


@lru_cache(maxsize=8)
def _prompt(name: str) -> str:
    return resources.files("substage.prompts").joinpath(name).read_text(encoding="utf-8")


# fixed instruction text sent ahead of each stage's rendered inputs
_REVIEW_TASK = "Review window {n} ({total} windows): scenes {first}..{last}\n\n"
_FINALE_TASK = {
    "full": (
        "Final whole-video audit. Bilingual transcript follows; check against the video facts. "
        "Edits only for lines off their row's register LEVEL (move them to it, never away), callback "
        "mismatches, ambiguity breaches, and whole-video voice/terminology drift — nothing else.\n\n"
    ),
    "lite": (
        "Final whole-video audit. Bilingual transcript follows; check against the video facts. "
        "Edits only for lines off their row's register LEVEL (move them to it, never away), callback "
        "mismatches, ambiguity breaches, and whole-video name/terminology drift — nothing else.\n\n"
    ),
}
_BLIND_TASK = "Korean subtitle text, in video order:\n\n"
_REGEN_TASK = (
    "Scene {sid}: rewrite the flagged Korean lines so they read as natural "
    "spoken Korean while keeping the pair's speech level and address term, meaning, and the scene's facts. "
    "Emit one edit per flagged cue; leave unflagged lines alone.\n\n"
)


def _system(narrative: str) -> str:
    """review.md for one narrative level (lite | full) plus the style guide it checks against."""
    return with_guide(narrative_text(_prompt("review.md"), narrative))


def prompt_texts(stage: str, narrative: str) -> tuple[str, ...]:
    """Fixed prompt text ``stage`` (review|finale|blind|regen) sends besides its
    inputs at narrative level ``narrative`` — system template plus inline
    instructions — so pipeline-level checkpoint keys change whenever any of it does."""
    return {
        "review": (narrative, _system(narrative), _REVIEW_TASK),
        "finale": (narrative, _system(narrative), _FINALE_TASK[narrative]),
        "blind": (_prompt("blind_read.md"), _BLIND_TASK),
        "regen": (narrative, _system(narrative), _REGEN_TASK),
    }[stage]


# ------------------------------------------------------------ prompt helpers


def _video_header(context_map: ContextMap, narrative: str, cue_ids: set[str] | None = None) -> str:
    """Compact stable video facts relevant to the given cue ids (None = all).
    ``lite`` drops voice cards; both keep tone, rating and personas. Register
    levels come from the register table (row LEVEL column, scene REGISTERS)."""
    full = narrative == "full"
    lines = ["## VIDEO"]
    if context_map.tone_baseline:
        lines.append(f"tone: {context_map.tone_baseline}")
    if context_map.rating:
        lines.append(f"rating: {context_map.rating}")
    if context_map.characters and full:
        lines.append("characters (id / KO name: persona | idiolect | profanity ceiling):")
        for c in context_map.characters:
            lines.append(
                f"  {c.id} / {c.ko_name}: {' '.join(c.persona.split())} | {c.voice.idiolect} "
                f"| ceiling {c.voice.profanity_ceiling}"
            )
    elif context_map.characters:
        lines.append("characters (id / KO name: persona):")
        for c in context_map.characters:
            lines.append(f"  {c.id} / {c.ko_name}: {' '.join(c.persona.split())}")
    if context_map.callbacks:
        lines.append("callbacks:")
        for cb in context_map.callbacks:
            lines.append(
                f"  {cb.motif} ({cb.note}): setups [{', '.join(cb.setup_cues)}] "
                f"-> payoffs [{', '.join(cb.payoff_cues)}]"
            )
    if context_map.ambiguous:
        lines.append("ambiguity flags (must stay double-edged):")
        for f in context_map.ambiguous:
            if cue_ids is not None and f.cue_id not in cue_ids:
                continue
            lines.append(f"  {f.cue_id}: {f.note}")
    if context_map.glossary:
        lines.append("glossary:")
        for g in context_map.glossary:
            lines.append(f"  {g.en} -> {g.ko}")
    return "\n".join(lines)


def _scene_context(snapshot: Snapshot | None, narrative: str) -> str:
    """Per-scene context for review: names, registers, deltas, ambiguity,
    callbacks — at the given narrative level (Snapshot.for_prompt)."""
    if snapshot is None:
        return ""
    snapshot = snapshot.for_prompt(narrative)
    lines: list[str] = []
    if snapshot.character_names:
        lines.append(
            "CHARACTER NAMES (use exactly): "
            + ", ".join(
                f"{cid} -> {ko_name}" for cid, ko_name in sorted(snapshot.character_names.items())
            )
        )
    if snapshot.registers:
        lines.append("REGISTERS (speaker>addressee: level · vocative · pronoun; shifts at their cue ids):")
        for pair, reg in sorted(snapshot.registers.items()):
            lines.extend(render_scene_register(pair, reg))
    if snapshot.cue_deltas:
        for d in snapshot.cue_deltas:
            for ec in d.emotion_changes:
                lines.append(f"DELTA {d.cue_id} emotion {ec.character} {ec.from_state}->{ec.to_state}")
            for rv in d.revelations:
                lines.append(f"DELTA {d.cue_id} reveals to {', '.join(rv.to_whom)}: {rv.fact}")
    if snapshot.preserve_ambiguity:
        for f in snapshot.preserve_ambiguity:
            lines.append(f"AMBIGUITY {f.cue_id}: {f.note}")
    if snapshot.callbacks_live:
        for cb in snapshot.callbacks_live:
            lines.append(
                f"CALLBACK {cb.motif}: setups [{', '.join(cb.setup_cues)}] "
                f"-> payoffs [{', '.join(cb.payoff_cues)}]"
            )
    if snapshot.glossary:
        lines.append("GLOSSARY: " + ", ".join(f"{g.en}->{g.ko}" for g in snapshot.glossary))
    return "\n".join(lines)


def _obs_by_id(observations) -> dict[str, CueObservation]:
    """Flatten CueObservation lists or ChunkObservations into cue_id -> observation."""
    out: dict[str, CueObservation] = {}
    for o in observations or []:
        if isinstance(o, CueObservation):
            out[o.cue_id] = o
        elif hasattr(o, "cues"):  # ChunkObservations
            for c in o.cues:
                out[c.cue_id] = c
    return out


def _reverse_level_map(
    levels: dict[str, tuple[str, str]] | None,
    ws: Workspace | None = None,
) -> dict[tuple[str, str], str]:
    """(spk, adr) -> latest level string from *levels* and optional workspace table."""
    rev: dict[tuple[str, str], str] = {}
    if ws is not None:
        table = RegisterTable.load(ws.data_dir)
        if table is not None:
            for e in table.map.edges:
                rev[(e.speaker, e.addressee)] = e.level
    for pair, lvl in (levels or {}).values():
        if ">" in pair and "/" not in pair and binary(lvl) is not None:
            s, a = pair.split(">", 1)
            rev.setdefault((s, a), lvl)
            if s == "system_ai":
                rev.setdefault((a, s), "hae")
    return rev


def _effective_cue_level(
    c: Cue | None,
    ko_txt: str,
    pair: str,
    lvl: str,
    rev_map: dict[tuple[str, str], str],
) -> tuple[str, str]:
    """Expand a 2-turn dash cue into ``(spk>adr/adr>spk, T1:lvl1/T2:lvl2)`` when asymmetric."""
    if ">" in pair and "/" not in pair and binary(lvl) is not None:
        en_turns = len(speaker_turns(c.text)) if c is not None else 1
        ko_turns = len(speaker_turns(ko_txt)) if ko_txt else 1
        if en_turns == 2 or ko_turns == 2:
            s, a = pair.split(">", 1)
            rev_lvl = rev_map.get((a, s))
            if rev_lvl is not None and not same_level(lvl, rev_lvl):
                return (f"{s}>{a}/{a}>{s}", f"{s}:{lvl}/{a}:{rev_lvl}")
    return (pair, lvl)


_BANMAL_MARKER_RE = (
    r"(?:(?:텐데|건데|는데|던데)[.?!…\"'’”)]*$)"
    r"|(?<![가-힣])안녕(?![가-힣])"
    r"|(?:^|[\s,\-—])(?:난|내가)(?=[\s,.?!]|$)"
)


def _violates_target_level(ko_txt: str, target_spec: str) -> bool:
    """True when *ko_txt* has a clear morphological ending that violates *target_spec*."""
    import re

    if not ko_txt or not target_spec or target_spec == "-":
        return False
    turns = list(addressed_levels(ko_txt))
    t_strs = speaker_turns(ko_txt)
    if ":" in target_spec and "/" in target_spec:
        p1, p2 = target_spec.split("/", 1)
        w1, w2 = p1.split(":", 1)[1], p2.split(":", 1)[1]
        if len(t_strs) == 2 and len(turns) == 2:
            bins = []
            for idx in (0, 1):
                b = binary(turns[idx])
                if b is None and re.search(_BANMAL_MARKER_RE, t_strs[idx].strip()):
                    b = "B"
                bins.append(b)
            allowed_bins = {b for b in (binary(w1), binary(w2)) if b is not None}
            if bins[0] is not None and bins[1] is not None:
                return set(bins) != allowed_bins
            return any(b is not None and b not in allowed_bins for b in bins)
        if len(turns) == 1 and turns[0] is not None:
            return not (same_level(turns[0], w1) or same_level(turns[0], w2))
        return False
    want_bin = binary(target_spec)
    if want_bin is None:
        return False
    known = [t for t in turns if t is not None]
    if known:
        return any(not same_level(t, target_spec) for t in known)
    if len(speaker_turns(ko_txt)) == 1 and want_bin == "J" and re.search(_BANMAL_MARKER_RE, ko_txt.strip()):
        return True
    return False

def _cue_lines(
    cues: list[Cue],
    ko_by_id: dict[str, str],
    observations,
    mark: set[str] | None = None,
    levels: dict[str, tuple[str, str]] | None = None,
    ws: Workspace | None = None,
) -> str:
    """Rows ``id | spk>adr | LEVEL | kind | EN | KO``; *levels* maps cue id ->
    (pair, table level at the cue) (:func:`substage.register.cue_levels`)."""
    obs_by_id = _obs_by_id(observations)
    levels = levels or {}
    rev_map = _reverse_level_map(levels, ws)
    rows = []
    for c in cues:
        ko = ko_by_id.get(c.id, "")
        en = c.text.replace("\n", " / ") or "(sound only)"
        if _NON_TRANSLATED.match(c.text):
            kind = "sound"
        elif c.lyric:
            o = obs_by_id.get(c.id)
            kind = "lyric-relevant" if (o and o.lyric_plot_relevant) else "lyric-incidental"
        elif ost_kind(c) == "credit":
            kind = "credit"
        elif c.source == "ost" or c.forced:
            kind = "ost"
        else:
            kind = "dialogue"
        flag = "*" if mark and c.id in mark else " "
        shown = ko if ko else "∅"
        raw_pair, raw_level = levels.get(c.id, ("-", "-"))
        pair, level = _effective_cue_level(c, ko, raw_pair, raw_level, rev_map)
        rows.append(f"{flag}{c.id} | {pair} | {level} | kind={kind} | EN: {en} | KO: {shown}")
    return "\n".join(rows)


# ------------------------------------------------------------------ S4 review


def review_windows(
    ws: Workspace,
    gc: GeminiClient,
    scenes: list[Scene],
    cues: list[Cue],
    translations: dict[str, SceneTranslation],
    snapshots: dict[str, Snapshot],
    context_map: ContextMap,
    observations,
    levels: dict[str, tuple[str, str]] | None = None,
) -> list[CueEdit]:
    """Bilingual continuity review over non-overlapping ~3-scene windows.

    Returns edits only for real problems (lines off the register table,
    omissions, callbacks, glossary, 말맛/condensation/joke function, act
    seams). *levels*: cue id -> (pair, table level) for the row LEVEL column.
    """
    cues_by_id = {c.id: c for c in cues}
    ko_by_id: dict[str, str] = {
        t.cue_id: t.ko for st in translations.values() for t in st.cues
    }
    cp = Checkpoints(ws, "review")
    edits: list[CueEdit] = []
    narrative = narrative_context(gc.settings)
    system = _system(narrative)

    for w in range(0, len(scenes), WINDOW_SCENES):
        win = scenes[w : w + WINDOW_SCENES]
        win_cue_ids = {cid for s in win for cid in s.cue_ids}
        win_cues = [cues_by_id[cid] for s in win for cid in s.cue_ids if cid in cues_by_id]
        if not win_cues:
            continue
        blocks = [
            _video_header(context_map, narrative, win_cue_ids),
        ]
        for s in win:
            ctx = _scene_context(snapshots.get(s.id), narrative)
            scene_cues = [cues_by_id[cid] for cid in s.cue_ids if cid in cues_by_id]
            part = f"### SCENE {s.id} ({_fmt_ms(s.start_ms)}-{_fmt_ms(s.end_ms)})"
            if ctx:
                part += "\n" + ctx
            part += "\n" + _cue_lines(scene_cues, ko_by_id, observations, levels=levels)
            blocks.append(part)
        prompt = _REVIEW_TASK.format(
            n=w // WINDOW_SCENES + 1,
            total=len(scenes) // WINDOW_SCENES + 1,
            first=win[0].id,
            last=win[-1].id,
        ) + "\n\n".join(blocks)
        # raw inputs + the exact rendered prompt, system template and model:
        # anything that can change the response forces a re-call
        h = input_hash(
            context_map,
            narrative,
            *[snapshots[s.id].model_dump_json() for s in win if s.id in snapshots],
            json.dumps([[c.id, ko_by_id.get(c.id, "")] for c in win_cues], ensure_ascii=False),
            # lyric relevance feeds the kind= column in the rendered rows
            json.dumps(
                [[o.cue_id, o.lyric_plot_relevant] for o in _obs_by_id(observations).values()],
                ensure_ascii=False,
            ),
            prompt,
            system,
            gc.settings.model_main,
        )
        key = f"win:{win[0].id}-{win[-1].id}"
        batch = cp.get(key, h, _EditBatch)
        if batch is None:
            try:
                batch = gc.generate(
                    stage=_STAGE_REVIEW, prompt=prompt, schema=_EditBatch,
                    system=system,
                )
            except GeminiBlocked as exc:
                logger.warning(
                    "review window %s..%s blocked (%s); skipping edits",
                    win[0].id, win[-1].id, exc.reason,
                )
                continue
            cp.put(key, h, batch)
        edits.extend(batch.edits)
    return edits


# ------------------------------------------------------------------ S4b parts


def finale_audit(
    ws: Workspace,
    gc: GeminiClient,
    cues: list[Cue],
    ko: dict[str, str],
    context_map: ContextMap,
    observations,
    levels: dict[str, tuple[str, str]] | None = None,
) -> list[CueEdit]:
    """One whole-video bilingual audit: lines off the register table, callback
    mismatches, ambiguity breaches, whole-video drift. Edits only."""
    transcript = _cue_lines(cues, ko, observations, levels=levels)
    narrative = narrative_context(gc.settings)
    prompt = (
        _FINALE_TASK[narrative]
        + _video_header(context_map, narrative)
        + "\n\n## TRANSCRIPT (id | speaker>addressee | LEVEL | kind | EN | KO; "
        "∅ is intentional only for sound and lyric-incidental rows)\n"
        + transcript
    )
    system = _system(narrative)
    h = input_hash(
        context_map,
        narrative,
        json.dumps([[c.id, ko.get(c.id, "")] for c in cues], ensure_ascii=False),
        json.dumps(
            [[o.cue_id, o.lyric_plot_relevant] for o in _obs_by_id(observations).values()],
            ensure_ascii=False,
        ),
        prompt,
        system,
        gc.settings.model_main,
    )
    cp = Checkpoints(ws, "review")
    batch = cp.get("finale", h, _EditBatch)
    if batch is None:
        try:
            batch = gc.generate(
                stage=_STAGE_FINALE, prompt=prompt, schema=_EditBatch,
                system=system,
            )
        except GeminiBlocked as exc:
            logger.warning("finale audit blocked (%s); no edits", exc.reason)
            return []
        cp.put("finale", h, batch)
    return batch.edits


def blind_read(
    ws: Workspace,
    gc: GeminiClient,
    ko_in_order: list[tuple[str, str]],
) -> list[ReadFlag]:
    """Korean-only read on the light model: 번역투, unclear referents, unnatural
    lines. Intentionally untranslated cues are skipped — a viewer sees nothing."""
    ko_pairs = [[cid, k] for cid, k in ko_in_order if k]
    body = "\n".join(f"{cid}: {k}" for cid, k in ko_pairs)
    prompt = f"{_BLIND_TASK}{body}"
    system = _prompt("blind_read.md")
    model = getattr(getattr(gc, "settings", None), "model_light", None)
    h = input_hash(json.dumps(ko_pairs, ensure_ascii=False), prompt, system, model or "")
    cp = Checkpoints(ws, "review")
    batch = cp.get("blind", h, _FlagBatch)
    if batch is None:
        try:
            batch = gc.generate(
                stage=_STAGE_BLIND,
                prompt=prompt,
                schema=_FlagBatch,
                system=system,
                model=model,
            )
        except GeminiBlocked as exc:
            logger.warning("blind read blocked (%s); no flags", exc.reason)
            return []
        cp.put("blind", h, batch)
    return batch.flags


_DECISIONS_CHUNK = 10  # 10 cues * 6 atomic questions = 60 questions <= 64 limit


def diagnose_with_decisions(
    ws: Workspace,
    client: ClefDecisionsClient,
    scenes: list[Scene],
    cues_by_id: dict[str, Cue],
    ko: dict[str, str],
    *,
    levels: dict[str, tuple[str, str]] | None = None,
    scene_images: dict[str, list[Path]] | None = None,
    threshold: float = 0.55,
) -> list[ReadFlag]:
    """Decomposed 6-axis System One diagnosis (60 atomic questions per 10-cue batch).

    Evaluates each translated cue across 6 orthogonal atomic axes (`omission`,
    `mistrans`, `translat`, `register`, `verbose`, `dash_err`) in a single
    forward pass (`output_tokens = 0`) over the bilingual scene context,
    ``PRO_SUBTITLE_LAWS``, and optional scene keyframes. Emits ``ReadFlag``
    items carrying the structured diagnosis vector for surgical regeneration.
    """
    levels = levels or {}
    rev_map = _reverse_level_map(levels, ws)
    scene_images = scene_images or {}
    cp = Checkpoints(ws, "review")
    flags: list[ReadFlag] = []
    tasks: list[tuple[str, str, list[Cue], dict[str, Any], dict[str, dict[str, Any]], list[Path]]] = []
    for scene in scenes:
        active_cues = [
            cues_by_id[cid]
            for cid in scene.cue_ids
            if cid in cues_by_id and ko.get(cid, "").strip()
        ]
        if not active_cues:
            continue
        imgs = (scene_images.get(scene.id) or [])[:4]
        for c_idx in range(0, len(active_cues), _DECISIONS_CHUNK):
            sub = active_cues[c_idx : c_idx + _DECISIONS_CHUNK]
            cues_state = []
            questions: dict[str, dict[str, Any]] = {}
            for c in sub:
                cid = c.id
                en_txt = c.text.replace("\n", " / ")
                ko_txt = ko.get(cid, "")
                raw_pair, raw_lvl = levels.get(cid, ("-", "-"))
                pair, lvl = _effective_cue_level(c, ko_txt, raw_pair, raw_lvl, rev_map)
                cues_state.append({
                    "id": cid,
                    "pair": pair,
                    "target_level": lvl,
                    "en": en_txt,
                    "ko": ko_txt,
                })
                questions[f"{cid}_omission"] = {
                    "type": "noul",
                    "instructions": (
                        f"Cue {cid}: Is KO ({ko_txt!r}) cut off mid-sentence or missing the core verb/point "
                        f"of EN ({en_txt!r})? (Dropping filler/noise per law_1_drop_noise is GOOD condensation, NOT omission.)"
                    ),
                }
                questions[f"{cid}_mistrans"] = {
                    "type": "noul",
                    "instructions": (
                        f"Cue {cid}: Does KO ({ko_txt!r}) mistranslate the pragmatic meaning, wordplay, or joke "
                        f"of EN ({en_txt!r}) in this scene?"
                    ),
                }
                questions[f"{cid}_translat"] = {
                    "type": "noul",
                    "instructions": (
                        f"Cue {cid}: Does KO ({ko_txt!r}) violate PRO_SUBTITLE_LAWS with stiff English translationese "
                        "(번역투: literal noun calques, phonetic US slang/names, unnecessary pronouns, or uncondensed vocal noise)?"
                    ),
                }
                questions[f"{cid}_register"] = {
                    "type": "noul",
                    "instructions": (
                        f"Cue {cid}: Does KO ({ko_txt!r}) violate target speech level {lvl!r} for pair {pair!r} "
                        "or use an unnatural vocative?"
                    ),
                }
                questions[f"{cid}_verbose"] = {
                    "type": "noul",
                    "instructions": (
                        f"Cue {cid}: Is KO ({ko_txt!r}) overly wordy/cluttered instead of briskly condensed for screen reading?"
                    ),
                }
                questions[f"{cid}_dash_err"] = {
                    "type": "noul",
                    "instructions": (
                        f"Cue {cid}: Does KO ({ko_txt!r}) have a multi-speaker '- ' dash error relative to EN ({en_txt!r})?"
                    ),
                }
            state_obj = {
                "scene_id": scene.id,
                "PRO_SUBTITLE_LAWS": PRO_SUBTITLE_LAWS,
                "cues": cues_state,
            }
            h = input_hash(
                "clef-6axis-v1",
                client.model,
                f"{threshold:.3f}",
                json.dumps(state_obj, ensure_ascii=False, sort_keys=True),
                json.dumps(questions, ensure_ascii=False, sort_keys=True),
                *[str(p) for p in imgs],
            )
            key = f"clef_diag:{scene.id}:{c_idx}"
            tasks.append((key, h, sub, state_obj, questions, imgs))

    def _ensure_reg_flags(base_flags: list[ReadFlag], sub: list[Cue], state_obj: dict[str, Any]) -> list[ReadFlag]:
        by_cid = {f.cue_id: f for f in base_flags}
        cues_meta = {item_c["id"]: item_c for item_c in state_obj.get("cues", [])}
        out: list[ReadFlag] = []
        reg_note = "register_mismatch(P=0.95: keep target speech level and natural vocative)"
        for c in sub:
            cid = c.id
            meta = cues_meta.get(cid) or {}
            viol = _violates_target_level(str(meta.get("ko", "")), str(meta.get("target_level", "-")))
            existing = by_cid.get(cid)
            if existing is not None:
                if viol and "register_mismatch" not in existing.problem:
                    out.append(ReadFlag(cue_id=cid, problem=f"{existing.problem}; {reg_note}"))
                else:
                    out.append(existing)
            elif viol:
                out.append(ReadFlag(cue_id=cid, problem=f"[6-Axis Diagnosis] {reg_note}"))
        return out

    def _run_diag(item: tuple[str, str, list[Cue], dict[str, Any], dict[str, dict[str, Any]], list[Path]]) -> list[ReadFlag]:
        key, h, sub, state_obj, questions, imgs = item
        cached = cp.get(key, h, _FlagBatch)
        if cached is not None:
            return _ensure_reg_flags(cached.flags, sub, state_obj)
        answers = client.decide(
            stage=_STAGE_BLIND,
            state=state_obj,
            questions=questions,
            images=imgs,
        )
        chunk_flags: list[ReadFlag] = []
        cues_meta = {item_c["id"]: item_c for item_c in state_obj.get("cues", [])}
        for c in sub:
            cid = c.id
            p_om = float((answers.get(f"{cid}_omission") or {}).get("noul", 0.0))
            p_mi = float((answers.get(f"{cid}_mistrans") or {}).get("noul", 0.0))
            p_tr = float((answers.get(f"{cid}_translat") or {}).get("noul", 0.0))
            p_rg = float((answers.get(f"{cid}_register") or {}).get("noul", 0.0))
            p_vb = float((answers.get(f"{cid}_verbose") or {}).get("noul", 0.0))
            p_da = float((answers.get(f"{cid}_dash_err") or {}).get("noul", 0.0))
            meta = cues_meta.get(cid) or {}
            if _violates_target_level(str(meta.get("ko", "")), str(meta.get("target_level", "-"))):
                p_rg = max(p_rg, 0.95)
            reasons: list[str] = []
            if p_om >= threshold:
                reasons.append(f"omission/truncation(P={p_om:.2f}: complete the missing predicate/point)")
            if p_mi >= threshold:
                reasons.append(f"pragmatic_mistranslation(P={p_mi:.2f}: land the intended joke/sarcasm/wordplay)")
            if p_tr >= max(threshold, 0.68):
                reasons.append(f"translationese_calque(P={p_tr:.2f}: replace literal calque with spoken Korean verb phrase)")
            if p_rg >= threshold:
                reasons.append(f"register_mismatch(P={p_rg:.2f}: keep target speech level and natural vocative)")
            if p_vb >= threshold:
                reasons.append(f"verbose_uncondensed(P={p_vb:.2f}: drop English filler/noise per subtitle laws)")
            if p_da >= threshold:
                reasons.append(f"speaker_dash_error(P={p_da:.2f}: fix 2-speaker '- ' dash formatting)")
            if reasons:
                chunk_flags.append(
                    ReadFlag(
                        cue_id=cid,
                        problem="[6-Axis Diagnosis] " + "; ".join(reasons),
                    )
                )
        cp.put(key, h, _FlagBatch(flags=chunk_flags))
        return chunk_flags

    if len(tasks) <= 1:
        for item in tasks:
            flags.extend(_run_diag(item))
    else:
        with ThreadPoolExecutor(max_workers=min(16, len(tasks))) as ex:
            for chunk_flags in ex.map(_run_diag, tasks):
                flags.extend(chunk_flags)
    return flags


def resolve_decisions_model(model: str, *, has_images: bool) -> str:
    """Select the optimal Clef variant based on whether keyframe images are attached.

    When ``model == "auto"``, routes multimodal calls (with keyframe images) to
    ``clef-omni`` and pure-text calls to ``clef`` (27B dense, sharper text
    calibration) or ``clef-flash`` when explicitly requested.
    """
    if model == "auto":
        return "clef-omni" if has_images else "clef"
    return model


class _AcceptanceCache(BaseModel):
    accepted_ids: list[str] = Field(default_factory=list)


def filter_edits_with_decisions(
    ws: Workspace,
    client: ClefDecisionsClient,
    edits: list[CueEdit],
    cues_by_id: dict[str, Cue],
    ko: dict[str, str],
    *,
    levels: dict[str, tuple[str, str]] | None = None,
) -> list[CueEdit]:
    """Clef Role-3 4-Pillar Acceptance Gate for surgical regeneration / polish edits.

    Evaluates ``current_ko`` vs ``new_ko`` side-by-side (`output_tokens = 0`) and
    commits an edit if and only if:
      1. Line count <= 2 and non-empty.
      2. Meaning/omission guard: ``delta_omiss <= +0.05`` (does not introduce truncation).
      3. Verbosity/collateral guard: ``delta_verbose <= +0.08``.
      4. Positive progress: target error drops (`delta_mistrans <= -0.15` or
         `delta_reg <= -0.15` or `delta_verbose <= -0.15`) OR spoken fluency
         improves (`delta_spoken >= +0.08`).
    """
    candidates = [
        e for e in edits
        if e.cue_id in cues_by_id
        and e.new_ko.strip()
        and e.new_ko.strip() != ko.get(e.cue_id, "").strip()
        and len(e.new_ko.strip().splitlines()) <= 2
    ]
    if not candidates:
        return []
    levels = levels or {}
    rev_map = _reverse_level_map(levels, ws)
    cp = Checkpoints(ws, "review")
    accepted: list[CueEdit] = []
    for idx in range(0, len(candidates), 5):
        chunk = candidates[idx : idx + 5]
        cases = []
        questions: dict[str, dict[str, Any]] = {}
        for e in chunk:
            cid = e.cue_id
            c = cues_by_id[cid]
            en_txt = c.text.replace("\n", " / ")
            curr_ko = ko.get(cid, "").strip()
            cand_ko = e.new_ko.strip()
            raw_pair, raw_lvl = levels.get(cid, ("-", "-"))
            pair, lvl = _effective_cue_level(c, curr_ko, raw_pair, raw_lvl, rev_map)
            cases.append({
                "id": cid,
                "pair": pair,
                "target_level": lvl,
                "en": en_txt,
                "current_ko": curr_ko,
                "candidate_ko": cand_ko,
                "reason": e.reason,
            })
            for side, val in (("curr", curr_ko), ("cand", cand_ko)):
                questions[f"{cid}_{side}_omiss"] = {
                    "type": "noul",
                    "instructions": f"Cue {cid}: Does {val!r} omit or cut off the core meaning of {en_txt!r}?",
                }
                questions[f"{cid}_{side}_mistrans"] = {
                    "type": "noul",
                    "instructions": f"Cue {cid}: Does {val!r} mistranslate the joke/wordplay/intent of {en_txt!r}?",
                }
                questions[f"{cid}_{side}_reg"] = {
                    "type": "noul",
                    "instructions": f"Cue {cid}: Does {val!r} violate target speech level {lvl!r} for pair {pair!r}?",
                }
                questions[f"{cid}_{side}_verbose"] = {
                    "type": "noul",
                    "instructions": f"Cue {cid}: Is {val!r} overly wordy/cluttered for a screen subtitle?",
                }
                questions[f"{cid}_{side}_spoken"] = {
                    "type": "noul",
                    "instructions": f"Cue {cid}: Does {val!r} read as natural spoken Korean dialogue (말맛)?",
                }
        state_obj = {"PRO_SUBTITLE_LAWS": PRO_SUBTITLE_LAWS, "cases": cases}
        h = input_hash(
            "clef-role3-gate-v1",
            client.model,
            json.dumps(state_obj, ensure_ascii=False, sort_keys=True),
            json.dumps(questions, ensure_ascii=False, sort_keys=True),
        )
        key = f"clef_gate:{idx}"
        cases_by_id = {item_c["id"]: item_c for item_c in cases}
        cached = cp.get(key, h, _AcceptanceCache)
        if cached is not None:
            keep_set = set(cached.accepted_ids)
            for e in chunk:
                cid = e.cue_id
                cm_meta = cases_by_id.get(cid) or {}
                tl = str(cm_meta.get("target_level", "-"))
                ck = str(cm_meta.get("current_ko", ""))
                nk = str(cm_meta.get("candidate_ko", ""))
                if _violates_target_level(nk, tl):
                    continue
                if cid in keep_set or (_violates_target_level(ck, tl) and not _violates_target_level(nk, tl)):
                    accepted.append(e)
            continue
        answers = client.decide(stage="s4b_gate", state=state_obj, questions=questions)
        chunk_keep: list[str] = []
        for e in chunk:
            cid = e.cue_id
            case_meta = cases_by_id.get(cid) or {}
            target_lvl = str(case_meta.get("target_level", "-"))
            curr_ko_s = str(case_meta.get("current_ko", ""))
            cand_ko_s = str(case_meta.get("candidate_ko", ""))
            if _violates_target_level(cand_ko_s, target_lvl):
                continue
            fixed_reg_violation = _violates_target_level(curr_ko_s, target_lvl) and not _violates_target_level(cand_ko_s, target_lvl)
            d_omiss = float((answers.get(f"{cid}_cand_omiss") or {}).get("noul", 0.0)) - float((answers.get(f"{cid}_curr_omiss") or {}).get("noul", 0.0))
            d_mist = float((answers.get(f"{cid}_cand_mistrans") or {}).get("noul", 0.0)) - float((answers.get(f"{cid}_curr_mistrans") or {}).get("noul", 0.0))
            d_reg = float((answers.get(f"{cid}_cand_reg") or {}).get("noul", 0.0)) - float((answers.get(f"{cid}_curr_reg") or {}).get("noul", 0.0))
            d_verb = float((answers.get(f"{cid}_cand_verbose") or {}).get("noul", 0.0)) - float((answers.get(f"{cid}_curr_verbose") or {}).get("noul", 0.0))
            d_spk = float((answers.get(f"{cid}_cand_spoken") or {}).get("noul", 0.0)) - float((answers.get(f"{cid}_curr_spoken") or {}).get("noul", 0.0))
            if (
                d_omiss <= (0.25 if fixed_reg_violation else 0.05)
                and d_verb <= (0.35 if fixed_reg_violation else 0.08)
                and (fixed_reg_violation or d_mist <= -0.15 or d_reg <= -0.15 or d_verb <= -0.15 or d_spk >= 0.08)
            ):
                chunk_keep.append(cid)
                accepted.append(e)
        cp.put(key, h, _AcceptanceCache(accepted_ids=chunk_keep))
    return accepted

def regenerate_flagged(
    ws: Workspace,
    gc: GeminiClient,
    flags: list[ReadFlag],
    cues_by_id: dict[str, Cue],
    snapshots: dict[str, Snapshot],
    scene_of: dict[str, str],
    ko: dict[str, str],
    observations,
    levels: dict[str, tuple[str, str]] | None = None,
) -> list[CueEdit]:
    """Rewrite blind-read-flagged lines per scene with full context.

    One call per affected scene; the model sees the scene's bilingual lines,
    its snapshot context, and which cues are flagged. Emits CueEdits.
    """
    by_scene: dict[str, list[ReadFlag]] = {}
    for f in flags:
        sid = scene_of.get(f.cue_id)
        if sid:
            by_scene.setdefault(sid, []).append(f)
    cp = Checkpoints(ws, "review")
    edits: list[CueEdit] = []
    narrative = narrative_context(gc.settings)
    system = _system(narrative)
    obs_json = json.dumps(
        [[o.cue_id, o.lyric_plot_relevant] for o in _obs_by_id(observations).values()],
        ensure_ascii=False,
    )
    sorted_scenes = sorted(by_scene.items())

    def _run_regen(item: tuple[str, list[ReadFlag]]) -> list[CueEdit]:
        sid, scene_flags = item
        scene_cues = sorted(
            (c for cid, c in cues_by_id.items() if scene_of.get(cid) == sid),
            key=lambda c: c.start_ms,
        )
        flagged = {f.cue_id for f in scene_flags}
        problems = "\n".join(f"  {f.cue_id}: {f.problem}" for f in scene_flags)
        ctx = _scene_context(snapshots.get(sid), narrative)
        prompt = (
            _REGEN_TASK.format(sid=sid)
            + (ctx + "\n\n" if ctx else "")
            + "## LINES (* = flagged; id | speaker>addressee | LEVEL | kind | EN | current KO)\n"
            + _cue_lines(scene_cues, ko, observations, mark=flagged, levels=levels, ws=ws)
            + "\n\n## FLAGGED PROBLEMS\n" + problems
        )
        h = input_hash(
            sid,
            narrative,
            json.dumps([[f.cue_id, f.problem] for f in scene_flags], ensure_ascii=False),
            json.dumps([[c.id, ko.get(c.id, "")] for c in scene_cues], ensure_ascii=False),
            snapshots[sid].model_dump_json() if sid in snapshots else "",
            obs_json,
            prompt,
            system,
            gc.settings.model_main,
        )
        key = f"regen:{sid}"
        batch = cp.get(key, h, _EditBatch)
        if batch is None:
            try:
                batch = gc.generate(
                    stage=_STAGE_REGEN, prompt=prompt, schema=_EditBatch,
                    system=system,
                )
            except GeminiBlocked as exc:
                logger.warning(
                    "regeneration for scene %s blocked (%s); keeping flagged lines",
                    sid, exc.reason,
                )
                return []
            cp.put(key, h, batch)
        return batch.edits

    if len(sorted_scenes) <= 2:
        for item in sorted_scenes:
            edits.extend(_run_regen(item))
    else:
        with ThreadPoolExecutor(max_workers=min(12, len(sorted_scenes))) as ex:
            for sc_edits in ex.map(_run_regen, sorted_scenes):
                edits.extend(sc_edits)
    return edits


_FINITE_CLOSE_RE = re.compile(
    r"(?:잖아|잖아요|다|란다|단다|네|네요|군|군요|구나|구먼|래|래요|대|대요|답니다|"
    r"어요|아요|해요|지요|죠|습니다|ㅂ니다|소|오|까|까요|니|냐|거든|거든요|거야|거예요|"
    r"건가|건가요|테야|텐데|텐데요|말이야|말이죠|말이다|일걸|일걸요|할걸|할걸요|했지|했죠|됐지|됐죠|"
    r"있지|있죠|없지|없죠|맞지|맞죠|알지|알죠|보지|보죠|주지|주죠|야지|야죠|겠지|겠죠|"
    r"했어|봤어|왔어|갔어|줬어|났어|졌어|렸어|겼어|켰어|쳤어|었어|았어|였어|겠어|싶어|없어|있어|"
    r"몰라|알아|맞아|같아|싶네요|던데요|불공평해|무리야|위험해|불가능해)[.?!…\"'’”)]*$"
)
_DANGLING_PP_RE = re.compile(
    r"(?:랑|이랑|랑도|이랑도|랑요|이랑요|와|과|와도|과도|에서|에서요|에선|에선요|"
    r"으로|로|으로도|로도|으로부터|으로부터요|로부터|로부터요|까지|까지요|부터|"
    r"한테|한테요|한테서|에게|에게요|에게서|사이에서|통해서|통해서요|위해서|위해서요|"
    r"소리를요|혐의로|죄목으로|유골로|시체로|파내서|말이야|말이지|말이다|일에|일에요|"
    r"을요|를요|은요|는요|이요|가요|도요|함께|함께요)[.?!…\"'’”)]*$"
)


class _AVSpanPairEdit(BaseModel):
    strategy: str = Field(description="One of: 'full_sov_inversion', 'verb_deferral_sov', 'preserve_reveal_order'")
    c1_ko: str
    c2_ko: str


class _HybridCoTSpanPairEdit(BaseModel):
    strategy: str = Field(description="One of: 'full_sov_inversion', 'verb_deferral_sov', 'preserve_reveal_order'")
    unified_sov_ko: str
    c1_ko: str
    c2_ko: str


def _last_turn_text(ko_txt: str) -> str:
    turns = speaker_turns(ko_txt)
    return turns[-1].strip() if turns else ko_txt.strip()


_STOP_STEMS = {"그리고", "하지만", "그래서", "그런데", "어쩌면", "그렇게", "이렇게", "우리도", "우리가"}


def _has_cross_cue_duplicate(ko1: str, ko2: str) -> bool:
    t1 = [w.strip(".,?!…\"'’”)-") for w in ko1.split() if len(w.strip(".,?!…\"'’”)-")) >= 3]
    t2 = [w.strip(".,?!…\"'’”)-") for w in ko2.split() if len(w.strip(".,?!…\"'’”)-")) >= 3]
    if set(t1) & set(t2):
        return True
    s1 = {w[:3] for w in t1 if len(w) >= 4 and w[:3] not in _STOP_STEMS}
    s2 = {w[:3] for w in t2 if len(w) >= 4 and w[:3] not in _STOP_STEMS}
    return bool(s1 & s2)

def _is_svo_premature_verb_close(ko1: str, ko2: str) -> bool:
    """True when Cue 1 closes with a finite verb or Cue 2 ends with a dangling PP/duplicate."""
    last1 = _last_turn_text(ko1)
    last2 = _last_turn_text(ko2)
    if _has_cross_cue_duplicate(ko1, ko2):
        return True
    if _FINITE_CLOSE_RE.search(last1) or _DANGLING_PP_RE.search(last2):
        return True
    return False


def _is_valid_sov_rewrite(
    curr1: str,
    curr2: str,
    new1: str,
    new2: str,
    target_lvl1: str,
    target_lvl2: str,
    *,
    require_open_c1: bool = True,
) -> bool:
    """Gate B: deterministic physical, dash, register, and SOV open-head check."""
    lines1 = [l for l in new1.splitlines() if l.strip()]
    lines2 = [l for l in new2.splitlines() if l.strip()]
    if not lines1 or not lines2 or len(lines1) > 2 or len(lines2) > 2:
        return False
    if any(len(l) > 16 for l in lines1 + lines2):
        return False
    if curr1.lstrip().startswith("-") != new1.lstrip().startswith("-"):
        return False
    if curr2.lstrip().startswith("-") != new2.lstrip().startswith("-"):
        return False
    if _violates_target_level(new1, target_lvl1) or _violates_target_level(new2, target_lvl2):
        return False
    if _has_cross_cue_duplicate(new1, new2):
        return False
    if require_open_c1:
        last1 = _last_turn_text(new1)
        last2 = _last_turn_text(new2)
        if _FINITE_CLOSE_RE.search(last1) or _DANGLING_PP_RE.search(last2):
            return False
    return True


def repair_span_calques(
    ws: Workspace,
    txt_client: Any,
    gate_client: ClefDecisionsClient,
    scenes: list[Scene],
    cues_by_id: dict[str, Cue],
    ko: dict[str, str],
    observations,
    *,
    levels: dict[str, tuple[str, str]] | None = None,
    calque_threshold: float = 0.25,
) -> list[CueEdit]:
    """Detect right-branching English multi-cue spans with SVO calque and jointly invert into Korean SOV."""
    levels = levels or {}
    rev_map = _reverse_level_map(levels, ws)
    obs_map = _obs_by_id(observations)
    candidate_spans: list[CueSpan] = []
    seen_cues: set[str] = set()
    for sc in scenes:
        sc_cues = [cues_by_id[cid] for cid in sc.cue_ids if cid in cues_by_id and ko.get(cid, "").strip()]
        idx_c = 0
        while idx_c < len(sc_cues) - 1:
            c1, c2 = sc_cues[idx_c], sc_cues[idx_c + 1]
            k1, k2 = ko.get(c1.id, "").strip(), ko.get(c2.id, "").strip()
            if c1.id in seen_cues or c2.id in seen_cues or not k1 or not k2 or k2.lstrip().startswith("-"):
                idx_c += 1
                continue
            sp_pair = detect_syntactic_spans([c1, c2], obs_map)
            sp = sp_pair.get(c1.id)
            if sp is not None and sp.policy == "REVERSE_SOV_PERMITTED" and _is_svo_premature_verb_close(k1, k2):
                seen_cues.add(c1.id)
                seen_cues.add(c2.id)
                candidate_spans.append(sp)
                idx_c += 2
                continue
            idx_c += 1
    if not candidate_spans:
        return []

    # Role-2 Clef Span-Pair Calque Diagnosis (batched <= 20 spans per decide call)
    flagged_spans: list[CueSpan] = []
    for b_start in range(0, len(candidate_spans), 20):
        sub_spans = candidate_spans[b_start : b_start + 20]
        state_obj: dict[str, Any] = {}
        questions: dict[str, dict[str, Any]] = {}
        for idx, sp in enumerate(sub_spans):
            c1_id, c2_id = sp.cue_ids
            c1, c2 = cues_by_id[c1_id], cues_by_id[c2_id]
            k1, k2 = ko.get(c1_id, "").strip(), ko.get(c2_id, "").strip()
            pid = f"sp{idx}"
            state_obj[pid] = {
                "en_cue1": c1.text.replace("\n", " "),
                "en_cue2": c2.text.replace("\n", " "),
                "ko_cue1": k1.replace("\n", " "),
                "ko_cue2": k2.replace("\n", " "),
            }
            questions[f"{pid}_calque"] = {
                "type": "noul",
                "instructions": (
                    f"Span {sp.span_id}: EN splits one sentence across Cue 1 ({state_obj[pid]['en_cue1']!r}) "
                    f"and Cue 2 ({state_obj[pid]['en_cue2']!r}). Current KO is Cue 1 ({state_obj[pid]['ko_cue1']!r}) "
                    f"-> Cue 2 ({state_obj[pid]['ko_cue2']!r}). Did Current KO prematurely close the English main verb "
                    "in Cue 1 (or repeat a phrase across both cues) so Cue 2 is a dangling post-verbal modifier/PP "
                    "that would read more naturally in Korean SOV order with the main verb at the end of Cue 2?"
                ),
            }
        ans = gate_client.decide(stage=_STAGE_BLIND, state=state_obj, questions=questions)
        for idx, sp in enumerate(sub_spans):
            c1_id, c2_id = sp.cue_ids
            k1, k2 = ko.get(c1_id, "").strip(), ko.get(c2_id, "").strip()
            p_calq = float((ans.get(f"sp{idx}_calque") or {}).get("noul", 0.0))
            if p_calq >= calque_threshold or _has_cross_cue_duplicate(k1, k2) or _DANGLING_PP_RE.search(_last_turn_text(k2)):
                flagged_spans.append(sp)

    if not flagged_spans:
        return []

    model_name = getattr(txt_client.settings, "model_main", "")
    is_av_model = "gemini" in model_name.lower()
    sys_prompt = "You are a professional subtitle localization specialist translating fictional motion picture dialogue into Korean."

    def _rewrite_one(sp: CueSpan) -> tuple[CueSpan, str, str] | None:
        c1_id, c2_id = sp.cue_ids
        c1, c2 = cues_by_id[c1_id], cues_by_id[c2_id]
        curr1, curr2 = ko.get(c1_id, "").strip(), ko.get(c2_id, "").strip()
        _, tl1 = _effective_cue_level(c1, curr1, *levels.get(c1_id, ("-", "-")), rev_map)
        _, tl2 = _effective_cue_level(c2, curr2, *levels.get(c2_id, ("-", "-")), rev_map)
        dur1 = (c1.end_ms - c1.start_ms) / 1000.0
        dur2 = (c2.end_ms - c2.start_ms) / 1000.0
        max1 = min(26, max(10, int(dur1 * 10)))
        max2 = min(26, max(10, int(dur2 * 10)))
        base_prompt = (
            f"You are fixing English SVO word-order calque across a 2-cue subtitle span into natural Korean head-final (SOV) order.\n"
            f"- Cue 1 ({c1_id}, {dur1:.1f}s, max={max1} chars, target_level={tl1}): EN={c1.text.replace(chr(10),' ')!r} | Current KO={curr1!r}\n"
            f"- Cue 2 ({c2_id}, {dur2:.1f}s, max={max2} chars, target_level={tl2}): EN={c2.text.replace(chr(10),' ')!r} | Current KO={curr2!r}\n\n"
            f"Current KO prematurely closed the English main verb in Cue 1 (or duplicated a phrase across Cue 1 and Cue 2), leaving Cue 2 as an unnatural post-verbal fragment.\n"
            f"## Mandatory Korean SOV Rules:\n"
            f"1. Move the main verb predicate from Cue 1 to the VERY END of Cue 2!\n"
            f"2. Cue 1 (`c1_ko`) MUST end with an OPEN modifier/PP/object particle (~의, ~와의, ~에서, ~로, ~부터, ~까지, ~을/를, ~은/는, ~해서) or noun, NEVER with a closed sentence-final verb ending (-다, -잖아, -지, -요, -죠, -거든, -네)!\n"
            f"   - Example A ('full_sov_inversion'): EN1 'You couldn't even save a relationship...' + EN2 'with a goddamn stripper!' -> c1_ko: '스트리퍼 따위와의 관계 하나' + c2_ko: '못 지켜내는 새끼가!'\n"
            f"   - Example B ('full_sov_inversion'): EN1 'To distribute gene therapy discreetly through everything...' + EN2 'from sweet drinks to breakfast cereals.' -> c1_ko: '단 음료부터 시리얼까지 모든 식품에' + c2_ko: '유전자 치료제를 몰래 퍼뜨렸고'\n"
            f"   - Example C ('verb_deferral_sov' with 2-speaker dash): EN1 '-It looks great. -And you decimated dozens of my men...' + EN2 'with the exhumed corpse of a hero.' -> c1_ko: '- 보기 좋잖아\\n- 내 부하 열댓 명을' + c2_ko: '파묘한 영웅의 유골로 몰살했지'\n"
            f"3. Preserve exact speech level ({tl1} / {tl2}) and 2-speaker '- ' dashes if present in Current KO. Never repeat the same phrase in both cues."
        )
        schema_cls = _AVSpanPairEdit if is_av_model else _HybridCoTSpanPairEdit
        for attempt in range(2):
            p_txt = base_prompt if attempt == 0 else (
                base_prompt
                + "\n\n[RETRY NOTICE: Your previous attempt left a closed finite verb at the end of c1_ko or a dangling particle in c2_ko! "
                "Ensure the last line of c1_ko ends with an open modifier/object particle (~의/~에서/~로/~에/~을/~를/~은/~는/~해서) and the main verb is at the end of c2_ko!]"
            )
            try:
                res = txt_client.generate(stage=_STAGE_REGEN, prompt=p_txt, schema=schema_cls, system=sys_prompt)
            except GeminiBlocked:
                return None
            if res.strategy == "preserve_reveal_order":
                return None
            raw1 = res.c1_ko.strip() if curr1.lstrip().startswith("-") else " ".join(res.c1_ko.splitlines())
            raw2 = res.c2_ko.strip() if curr2.lstrip().startswith("-") else " ".join(res.c2_ko.splitlines())
            n1 = rewrap(raw1, 14, 12)
            n2 = rewrap(raw2, 14, 12)
            if _is_valid_sov_rewrite(curr1, curr2, n1, n2, tl1, tl2, require_open_c1=True):
                return (sp, n1, n2)
        return None

    candidates_out: list[tuple[CueSpan, str, str]] = []
    with ThreadPoolExecutor(max_workers=min(8, len(flagged_spans))) as ex:
        for item_res in ex.map(_rewrite_one, flagged_spans):
            if item_res is not None:
                candidates_out.append(item_res)
    if not candidates_out:
        return []

    # Gate A: Span-Aware Clef Role-3 Semantic & Fluency Gate
    accepted_edits: list[CueEdit] = []
    for b_start in range(0, len(candidates_out), 10):
        sub_cands = candidates_out[b_start : b_start + 10]
        state_obj = {}
        questions = {}
        for idx, (sp, n1, n2) in enumerate(sub_cands):
            c1_id, c2_id = sp.cue_ids
            en_span = f"{cues_by_id[c1_id].text.replace(chr(10),' ').strip()} {cues_by_id[c2_id].text.replace(chr(10),' ').strip()}"
            curr_span = f"{ko.get(c1_id,'').replace(chr(10),' ').strip()} {ko.get(c2_id,'').replace(chr(10),' ').strip()}"
            cand_span = f"{n1.replace(chr(10),' ').strip()} {n2.replace(chr(10),' ').strip()}"
            pid = f"g{idx}"
            state_obj[pid] = {"en_span": en_span, "curr_span": curr_span, "cand_span": cand_span}
            for side, val in (("curr", curr_span), ("cand", cand_span)):
                questions[f"{pid}_{side}_omiss"] = {
                    "type": "noul",
                    "instructions": f"Span {sp.span_id}: Does combined Korean {val!r} omit the core point of combined English {en_span!r} (allowing normal subtitle condensation)?",
                }
                questions[f"{pid}_{side}_mistrans"] = {
                    "type": "noul",
                    "instructions": f"Span {sp.span_id}: Does combined Korean {val!r} mistranslate the intent or joke of combined English {en_span!r}?",
                }
                questions[f"{pid}_{side}_spoken"] = {
                    "type": "noul",
                    "instructions": f"Span {sp.span_id}: Does combined Korean {val!r} read in natural Korean head-final SOV order without English afterthought calque?",
                }
        ans = gate_client.decide(stage="s4b_gate", state=state_obj, questions=questions)
        for idx, (sp, n1, n2) in enumerate(sub_cands):
            pid = f"g{idx}"
            c1_id, c2_id = sp.cue_ids
            d_om = float((ans.get(f"{pid}_cand_omiss") or {}).get("noul", 0.0)) - float((ans.get(f"{pid}_curr_omiss") or {}).get("noul", 0.0))
            d_mi = float((ans.get(f"{pid}_cand_mistrans") or {}).get("noul", 0.0)) - float((ans.get(f"{pid}_curr_mistrans") or {}).get("noul", 0.0))
            cand_om = float((ans.get(f"{pid}_cand_omiss") or {}).get("noul", 0.0))
            cand_mi = float((ans.get(f"{pid}_cand_mistrans") or {}).get("noul", 0.0))
            if (d_om <= 0.25 or cand_om <= 0.35) and (d_mi <= 0.15 or cand_mi <= 0.25):
                accepted_edits.append(CueEdit(cue_id=c1_id, new_ko=n1, reason="span_sov_inversion", kind="polish"))
                accepted_edits.append(CueEdit(cue_id=c2_id, new_ko=n2, reason="span_sov_inversion", kind="polish"))
    return accepted_edits
