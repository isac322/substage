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
from .register import render_scene_register
from .store import Checkpoints, input_hash
from .style import PRO_SUBTITLE_LAWS, with_guide
from .translate import _NON_TRANSLATED, _fmt_ms
from .validate import ost_kind

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


def _cue_lines(
    cues: list[Cue],
    ko_by_id: dict[str, str],
    observations,
    mark: set[str] | None = None,
    levels: dict[str, tuple[str, str]] | None = None,
) -> str:
    """Rows ``id | spk>adr | LEVEL | kind | EN | KO``; *levels* maps cue id ->
    (pair, table level at the cue) (:func:`substage.register.cue_levels`)."""
    obs_by_id = _obs_by_id(observations)
    levels = levels or {}
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
        pair, level = levels.get(c.id, ("-", "-"))
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
    scene_images = scene_images or {}
    cp = Checkpoints(ws, "review")
    flags: list[ReadFlag] = []
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
                pair, lvl = levels.get(cid, ("-", "-"))
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
            cached = cp.get(key, h, _FlagBatch)
            if cached is not None:
                flags.extend(cached.flags)
                continue
            answers = client.decide(
                stage=_STAGE_BLIND,
                state=state_obj,
                questions=questions,
                images=imgs,
            )
            chunk_flags: list[ReadFlag] = []
            for c in sub:
                cid = c.id
                p_om = float((answers.get(f"{cid}_omission") or {}).get("noul", 0.0))
                p_mi = float((answers.get(f"{cid}_mistrans") or {}).get("noul", 0.0))
                p_tr = float((answers.get(f"{cid}_translat") or {}).get("noul", 0.0))
                p_rg = float((answers.get(f"{cid}_register") or {}).get("noul", 0.0))
                p_vb = float((answers.get(f"{cid}_verbose") or {}).get("noul", 0.0))
                p_da = float((answers.get(f"{cid}_dash_err") or {}).get("noul", 0.0))
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
            batch = _FlagBatch(flags=chunk_flags)
            cp.put(key, h, batch)
            flags.extend(chunk_flags)
    return flags

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
    for sid, scene_flags in sorted(by_scene.items()):
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
            + _cue_lines(scene_cues, ko, observations, mark=flagged, levels=levels)
            + "\n\n## FLAGGED PROBLEMS\n" + problems
        )
        h = input_hash(
            sid,
            narrative,
            json.dumps([[f.cue_id, f.problem] for f in scene_flags], ensure_ascii=False),
            json.dumps([[c.id, ko.get(c.id, "")] for c in scene_cues], ensure_ascii=False),
            snapshots[sid].model_dump_json() if sid in snapshots else "",
            json.dumps(
                [[o.cue_id, o.lyric_plot_relevant] for o in _obs_by_id(observations).values()],
                ensure_ascii=False,
            ),
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
                continue
            cp.put(key, h, batch)
        edits.extend(batch.edits)
    return edits
