"""Shared evaluation primitives (used by refeval).

Legacy arm specs and blind labels, final SRT lookup and cue formatting, plus
opt-in paired-evaluation metadata and arithmetic candidate/reference deltas.
The six-label restriction intentionally belongs only to legacy blind ranking.
"""

from __future__ import annotations

import random
from pathlib import Path

from .models import Cue, VideoInfo
from .store import load_model

LABELS: tuple[str, ...] = ("X", "Y", "Z", "W", "V", "U")
REF = "ref"

EVALUATION_MODES: tuple[str, ...] = ("listwise", "paired")
EVALUATION_SPLITS: tuple[str, ...] = ("dev", "heldout")


def evaluation_metadata(
    mode: str, *, split: str | None = None, cycle_id: str | None = None
) -> dict[str, str | None]:
    """Validate opt-in evaluation metadata without changing legacy defaults."""
    if mode not in EVALUATION_MODES:
        raise ValueError(f"evaluation mode must be one of {EVALUATION_MODES}, got {mode!r}")
    if split is not None and split not in EVALUATION_SPLITS:
        raise ValueError(f"split must be one of {EVALUATION_SPLITS}, got {split!r}")
    if cycle_id is not None and not str(cycle_id).strip():
        raise ValueError("cycle_id must not be blank")
    return {"evaluation_mode": mode, "split": split, "cycle_id": cycle_id}


def numeric_deltas(candidate: dict, reference: dict) -> dict[str, float | None]:
    """Candidate minus frozen reference; missing values stay missing, not zero.

    Signs are arithmetic, not a claim that higher is better for every metric
    (for example, a positive error-rate delta is a regression).
    """
    return {
        key: round(value - reference[key], 4)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and isinstance(reference.get(key), (int, float)) and not isinstance(reference[key], bool)
        else None
        for key, value in candidate.items()
        if not isinstance(value, dict)
    }


# ---------------------------------------------------------------- arms


def parse_arms(spec: str) -> dict[str, str]:
    """``"A=video__a,B=video"`` → ``{"A": "video__a", "B": "video"}`` (order kept)."""
    arms: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, wid = part.partition("=")
        name, wid = name.strip(), wid.strip()
        if not sep or not name or not wid:
            raise ValueError(f"arm spec must be NAME=WORKSPACE_ID, got {part!r}")
        if name == REF or name in arms:
            raise ValueError(f"duplicate or reserved arm name {name!r}")
        arms[name] = wid
    if not arms:
        raise ValueError("no arms given")
    return arms


def shuffle_labels(names: list[str], scene_id: str, seed: int) -> dict[str, str]:
    """Per-scene deterministic label→candidate assignment."""
    if len(names) > len(LABELS):
        raise ValueError(f"at most {len(LABELS)} candidates supported, got {len(names)}")
    order = list(names)
    random.Random(f"{seed}:{scene_id}").shuffle(order)
    return dict(zip(LABELS, order))


def arm_srt(data_dir: Path) -> Path:
    """The final ``<video_id>.ko.srt`` of a workspace data dir."""
    video_json = data_dir / "video.json"
    if video_json.is_file():
        p = data_dir / f"{load_model(video_json, VideoInfo).video_id}.ko.srt"
        if p.is_file():
            return p
    found = sorted(data_dir.glob("*.ko.srt"))
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(f"no unique final .ko.srt in {data_dir} (found {len(found)})")


# ---------------------------------------------------------------- formatting


def mmss(ms: int) -> str:
    s = max(ms, 0) // 1000
    h, m = divmod(s // 60, 60)
    return f"{h}:{m:02d}:{s % 60:02d}" if h else f"{m:02d}:{s % 60:02d}"


def _en_text(c: Cue) -> str:
    tail = " ".join(([f"♪"] if c.lyric else []) + c.sound_cues)
    body = c.text.strip()
    return " ".join(p for p in (f"{c.speaker_hint}:" if c.speaker_hint and body else "", body, tail) if p)
