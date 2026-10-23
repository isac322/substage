"""S6 emission: the final .ko.srt plus a machine-readable sidecar.

<data_dir>/<video_id>.ko.srt   — plain UTF-8 SRT of the S5c layout events
<data_dir>/<video_id>.ko.json  — {"events": display events with the cue ids
                                each covers, "cues": per-cue speaker/addressee/
                                register + edits history}
"""

from __future__ import annotations
import re
from pathlib import Path

from pydantic import BaseModel, Field

from .models import Cue, CueEdit, VideoInfo, OutEvent, TranslatedCue
from .store import save_model
from .subs import write_srt
from .validate import strip_clause_slashes


_AN_PREFIX = re.compile(r"^(?:\{\\an\d\})+")


def _collapse_an_prefix(text: str) -> str:
    """Collapse any leading ASS alignment-tag run to one top-position tag."""
    return _AN_PREFIX.sub(lambda _: r"{\an8}", text, count=1)


def _strip_an_prefix(text: str) -> str:
    """Drop any leading ASS alignment-tag run (bottom-position text)."""
    return _AN_PREFIX.sub("", text).lstrip()


def _normalize_body(event: OutEvent) -> str:
    """Normalize displayed punctuation and dialogue dashes without changing ASS tag presence."""
    text = _strip_an_prefix(strip_clause_slashes(event.ko)).replace("…", "...")
    if event.channel == "dialogue":
        lines = text.splitlines()
        dashed = [bool(re.match(r"^\s*-(?!\d)\s*", line)) for line in lines]
        if len(lines) == 1 and dashed[0]:
            text = re.sub(r"^\s*-(?!\d)\s*", "", lines[0])
        elif len(lines) == 2 and any(dashed):
            speakers = [
                fragment.speaker.strip().casefold()
                for fragment in event.fragments
                if fragment.text.strip() and fragment.speaker
            ]
            two_voices = (
                len(set(speakers)) == 2
                and all(fragment.speaker for fragment in event.fragments if fragment.text.strip())
            )
            bodies = [re.sub(r"^\s*-(?!\d)\s*", "", line) for line in lines]
            text = "\n".join(
                f"- {body}" if all(dashed) or two_voices else body
                for body in bodies
            )
    return text


def _event_text(event: OutEvent) -> str:
    """Normalize displayed punctuation/dashes and apply positioning."""
    text = _normalize_body(event)
    if event.position == "an8":
        text = r"{\an8}" + text
    return text

def _sort_key(event: OutEvent) -> tuple[int, int]:
    return event.start_ms, event.end_ms


def _sanitize_lane(lane: list[OutEvent]) -> list[OutEvent]:
    """Resolve overlaps among events sharing one screen position."""
    pending = sorted(lane, key=_sort_key)
    out: list[OutEvent] = []
    while pending:
        b = pending.pop(0)
        if not out or out[-1].end_ms <= b.start_ms:
            out.append(b)
            continue
        a = out[-1]
        if _strip_an_prefix(a.ko).strip() == _strip_an_prefix(b.ko).strip():
            out[-1] = a.model_copy(update={
                "start_ms": min(a.start_ms, b.start_ms),
                "end_ms": max(a.end_ms, b.end_ms),
                "cue_ids": list(dict.fromkeys([*a.cue_ids, *b.cue_ids])),
            })
            continue
        if b.start_ms - 1 - a.start_ms >= 200:
            out[-1] = a.model_copy(update={"end_ms": b.start_ms - 1})
            out.append(b)
            continue
        # Identical or near-simultaneous starts (<200ms gap): share the interval.
        mid = a.start_ms + max(200, (max(a.end_ms, b.end_ms) - a.start_ms) // 2)
        out[-1] = a.model_copy(update={"end_ms": mid - 1})
        pending.append(b.model_copy(update={"start_ms": mid, "end_ms": max(b.end_ms, mid + 200)}))
        pending.sort(key=_sort_key)
    return out


def sanitize_display_events(events: list[OutEvent]) -> list[OutEvent]:
    """Normalize displayed text and resolve overlaps at each screen position."""
    events = [
        event.model_copy(
            update={
                "ko": (r"{\an8}" if event.position == "an8" and _AN_PREFIX.match(event.ko) else "")
                + _normalize_body(event)
            }
        )
        for event in events
    ]
    an2 = [e for e in events if e.position != "an8"]
    an8 = [e for e in events if e.position == "an8"]
    return sorted([*_sanitize_lane(an2), *_sanitize_lane(an8)], key=_sort_key)


class SidecarCue(TranslatedCue):
    """TranslatedCue plus the post-translation edits it accumulated."""

    edits: list[CueEdit] = Field(default_factory=list)


class Sidecar(BaseModel):
    events: list[OutEvent] = Field(default_factory=list)
    cues: list[SidecarCue] = Field(default_factory=list)


def emit(
    ws,
    video: VideoInfo,
    cues: list[Cue],
    ko: dict[str, str],
    meta: dict[str, TranslatedCue],
    events: list[OutEvent],
) -> tuple[Path, Path]:
    """Write the final subtitle file (one SRT entry per non-empty layout event)
    and its sidecar; returns their paths.

    Every cue with English or a translation keeps a sidecar row for
    auditability, including cues no event displays.
    """
    srt_path = ws.path(f"{video.video_id}.ko.srt")
    sidecar_path = ws.path(f"{video.video_id}.ko.json")

    events = sanitize_display_events([e for e in events if e.ko.strip()])
    shown = [e.model_copy(update={"ko": _event_text(e)}) for e in events]
    rows: list[SidecarCue] = []
    for cue in cues:
        m = meta.get(cue.id)
        if m is not None:
            row = m if isinstance(m, SidecarCue) else SidecarCue(**m.model_dump())
            rows.append(row.model_copy(update={"ko": _collapse_an_prefix(row.ko)}))
        elif cue.text.strip():
            rows.append(
                SidecarCue(cue_id=cue.id, ko=_collapse_an_prefix(ko.get(cue.id, "")))
            )

    write_srt(
        [
            (
                e.start_ms,
                e.end_ms,
                e.ko,
            )
            for e in shown
        ],
        srt_path,
    )
    save_model(sidecar_path, Sidecar(events=shown, cues=rows))
    return srt_path, sidecar_path
