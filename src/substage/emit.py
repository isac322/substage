"""S6 emission: the final .ko.srt plus a machine-readable sidecar.

<data_dir>/<video_id>.ko.srt   — plain UTF-8 SRT of the S5c layout events
<data_dir>/<video_id>.ko.json  — {"events": display events with the cue ids
                                each covers, "cues": per-cue speaker/addressee/
                                register + edits history}
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .models import Cue, CueEdit, VideoInfo, OutEvent, TranslatedCue
from .store import save_model
from .subs import write_srt


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

    shown = [e for e in events if e.ko.strip()]
    rows: list[SidecarCue] = []
    for cue in cues:
        m = meta.get(cue.id)
        if m is not None:
            rows.append(m if isinstance(m, SidecarCue) else SidecarCue(**m.model_dump()))
        elif cue.text.strip():
            rows.append(SidecarCue(cue_id=cue.id, ko=ko.get(cue.id, "")))

    write_srt(
        [
            (
                e.start_ms,
                e.end_ms,
                f"{{\\an8}}{e.ko}" if e.position == "an8" and not e.ko.startswith("{\\an8}") else e.ko,
            )
            for e in shown
        ],
        srt_path,
    )
    save_model(sidecar_path, Sidecar(events=shown, cues=rows))
    return srt_path, sidecar_path
