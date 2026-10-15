"""C7 picture check: re-verify finished Korean lines against picture + audio.

After S4 edits and S5 validation, the cues that changed (plus comedy/turning/
twist-sensitive cues chosen by the pipeline) are grouped into <=60 s windows of
the uploaded proxy and shown to the model one clip at a time. The model returns
edits only; run_s5 then revalidates. The check has no register authority: it
sees no register rows, and an edit that changes a line's speech level (or
swaps the table pronoun) is dropped through the register gate.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from .config import narrative_context, narrative_text
from .gemini import Clip, GeminiBlocked, UploadedFile
from .models import Cue, CueEdit, Snapshot
from .register import RegisterGate
from .store import Checkpoints, input_hash
from .validate import _keeps_protected, _protected_row

logger = logging.getLogger(__name__)

_MAX_CLIP_MS = 60_000  # group cues into clips no longer than this
_GROUP_GAP_MS = 8_000  # cues closer than this share a clip
_PAD_MS = 2_000  # clip padding around the group's first/last cue


class _PictureBatch(BaseModel):
    edits: list[CueEdit] = []

class _PictureConcern(BaseModel):
    cue_id: str
    problem: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class _PictureEvidence(BaseModel):
    """Native AV verifier has no Korean-writing authority in a pure text team."""
    concerns: list[_PictureConcern] = Field(default_factory=list)



_PICTURE_PROMPT = """\
You are checking finished Korean subtitles against the actual picture and audio
of the video. For each listed cue decide whether the Korean line works as a
subtitle *in this moment*:

- meaning matches what is said AND what is on screen (deixis, off-screen speech,
  visible action the line leans on)
- the punchline, reveal, or emotional beat lands in sync with the actor's
  delivery, not early or late
- keep each line's speech level (존댓말 vs 반말) and address terms as they are;
  register is not judged here
- no leftover translationese a viewer would hear clash with the picture
- preserve these protected forms verbatim and do not resolve protected
  ambiguity (see each cue's PROTECT line)

Cues (id | time | EN source | KO shown; PROTECT lines per cue):
{cues}

Return only cues that must change, kind "picture", with the full corrected
Korean line (single '\\n' if two lines). If a line is fine, omit it.\
"""


def _groups(cues: list[Cue]) -> list[list[Cue]]:
    """Group chronologically adjacent cues into <=_MAX_CLIP_MS windows."""
    groups: list[list[Cue]] = []
    for cue in sorted(cues, key=lambda c: c.start_ms):
        if groups and cue.start_ms - groups[-1][-1].end_ms <= _GROUP_GAP_MS and cue.end_ms - groups[-1][0].start_ms <= _MAX_CLIP_MS:
            groups[-1].append(cue)
        else:
            groups.append([cue])
    return groups


def _fmt(ms: int) -> str:
    return f"{ms // 60000}:{(ms % 60000) // 1000:02d}.{(ms % 1000) // 100:01d}"


def _clip_lines(
    group: list[Cue],
    ko: dict[str, str],
    snapshots: dict[str, Snapshot],
    scene_of: dict[str, str],
) -> str:
    lines: list[str] = []
    for c in group:
        lines.append(
            f"{c.id} | {_fmt(c.start_ms)}–{_fmt(c.end_ms)} | "
            f"{c.text.replace(chr(10), ' / ')} | {ko.get(c.id, '').replace(chr(10), ' / ')}"
        )
        protected = _protected_row(c, snapshots.get(scene_of.get(c.id, "")), ko)
        if protected:
            lines.append(protected)
    return "\n".join(lines)


def picture_check(
    ws,
    gc,
    proxy: UploadedFile,
    cues_by_id: dict[str, Cue],
    ko: dict[str, str],
    cue_ids: set[str],
    snapshots: dict[str, Snapshot],
    scene_of: dict[str, str],
    gate: RegisterGate | None = None,
    *,
    writer=None,
    strict: bool = False,
) -> list[CueEdit]:
    """Show each group of selected cues to the model inside its clip.

    Returns edits only; ids outside `cue_ids`, empty texts, edits that drop a
    protected form and (with *gate*) edits that change the line's speech
    level are dropped.
    """
    if proxy is None:
        return []
    selected = sorted(
        (cues_by_id[cid] for cid in cue_ids if cid in cues_by_id),
        key=lambda c: c.start_ms,
    )
    ck = Checkpoints(ws, "picture-check")
    edits: list[CueEdit] = []
    narrative = narrative_context(gc.settings)
    template = narrative_text(_PICTURE_PROMPT, narrative)
    for group in _groups(selected):
        clip = Clip(
            file=proxy,
            start_s=max(0.0, (group[0].start_ms - _PAD_MS) / 1000),
            end_s=(group[-1].end_ms + _PAD_MS) / 1000,
        )
        prompt = template.format(
            cues=_clip_lines(group, ko, snapshots, scene_of),
        )
        if writer is not None:
            prompt = (
                "Inspect the bounded clip against the listed subtitles. Return only "
                "cue-keyed concerns supported by visible/audible evidence, with confidence. "
                "Do not write corrected Korean or infer future plot facts. Preserve "
                "uncertain referents and ambiguity.\n\n"
                + _clip_lines(group, ko, snapshots, scene_of)
            )
        # raw inputs + the exact rendered prompt, model and clip bounds
        h = input_hash(
            "|".join(c.id for c in group),
            "|".join(ko.get(c.id, "") for c in group),
            narrative,
            prompt,
            gc.settings.model_main,
            f"{proxy.uri} {clip.start_s:g}-{clip.end_s:g}",
        )
        if writer is not None:
            h = input_hash(
                h, writer.settings.model_main, writer.settings.model_light,
                template, "evidence-writer-v1",
            )
        batch = ck.get(group[0].id, h, _PictureBatch)
        if batch is None:
            try:
                if writer is None:
                    batch = gc.generate(
                        stage="picture-check", prompt=prompt,
                        schema=_PictureBatch, clips=[clip],
                    )
                else:
                    evidence = gc.generate(
                        stage="picture-check:perception", prompt=prompt,
                        schema=_PictureEvidence, clips=[clip],
                    )
                    evidence = evidence.model_copy(update={
                        "concerns": [item for item in evidence.concerns
                                     if item.cue_id in {cue.id for cue in group}],
                    })
                    if not evidence.concerns:
                        batch = _PictureBatch()
                    else:
                        batch = writer.generate(
                            stage="picture-check:realize",
                            prompt=(
                                template.format(cues=_clip_lines(group, ko, snapshots, scene_of))
                                + "\nNo clip is attached. Native AV evidence is advisory; "
                                "correct only grounded concerns, preserving ambiguity and "
                                "cue-local reveal timing.\n" + evidence.model_dump_json()
                            ),
                            schema=_PictureBatch, clips=[],
                        )
            except GeminiBlocked as exc:
                if strict:
                    raise
                # Provider refused this clip: keep the reviewed text, check the rest.
                logger.warning("picture-check %s-%s blocked (%s); skipped", group[0].id, group[-1].id, exc)
                continue
            ck.put(group[0].id, h, batch)
        seen: set[str] = set()
        for e in batch.edits:
            if e.cue_id not in cue_ids or not e.new_ko.strip() or e.cue_id in seen:
                continue
            snap = snapshots.get(scene_of.get(e.cue_id, ""))
            if not _keeps_protected(cues_by_id[e.cue_id], ko.get(e.cue_id, ""), e.new_ko, snap, ko):
                continue  # dropped a protected form: keep the original text
            if gate is not None and not gate.check(
                "picture", [e.cue_id], ko.get(e.cue_id, ""), e.new_ko, use_table=False
            ):
                continue  # changed the speech level: C7 has no register authority
            seen.add(e.cue_id)
            edits.append(e.model_copy(update={"kind": "picture"}))
    return edits
