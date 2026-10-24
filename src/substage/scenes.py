"""S1: group cues into scenes, audit dialogue gaps, plan viewing chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .models import Cue, Scene


_ACT_COUNT = 4


def _assign_acts(scenes: list[Scene]) -> None:
    """Assign stable, contiguous work partitions without inferring story acts.

    Translation runs scenes sequentially within an act and runs acts in
    parallel.  S1 has no semantic input from which to infer narrative acts,
    so divide the ordered scene list into at most four balanced contiguous
    partitions.  Keeping each partition contiguous preserves cue order and
    keeps scene-to-scene context local while making the existing scheduler
    usable for every multi-scene video.
    """
    if not scenes:
        return
    act_count = min(_ACT_COUNT, len(scenes))
    base, remainder = divmod(len(scenes), act_count)
    if len(scenes) <= 12 or act_count <= 1:
        offset = 0
        for act in range(1, act_count + 1):
            width = base + (1 if act <= remainder else 0)
            for scene in scenes[offset : offset + width]:
                scene.act = act
            offset += width
        return
    cuts: list[int] = [0]
    radius = max(1, base // 5)
    for k in range(1, act_count):
        target = k * base + min(k, remainder)
        lo = max(cuts[-1] + 1, target - radius)
        hi = min(len(scenes) - (act_count - k), target + radius)
        best_cut = target
        best_gap = -1
        for idx in range(lo, hi + 1):
            gap = scenes[idx].start_ms - scenes[idx - 1].end_ms
            if gap > best_gap:
                best_gap = gap
                best_cut = idx
        cuts.append(best_cut)
    cuts.append(len(scenes))
    for act_idx in range(act_count):
        for scene in scenes[cuts[act_idx] : cuts[act_idx + 1]]:
            scene.act = act_idx + 1

def group_scenes(cues: list[Cue], *, gap_ms: int, max_cues: int) -> list[Scene]:
    """Split sorted cues into scenes.

    A scene break happens at a silence >= ``gap_ms``; when a run reaches
    ``max_cues`` cues without a silence, it is cut at the largest gap inside
    the last third of the run (mid-dialogue cuts land where they hurt least).
    """
    cues = sorted(cues, key=lambda c: (c.start_ms, c.end_ms))
    if not cues:
        return []

    runs: list[list[Cue]] = []
    run = [cues[0]]
    for prev, cur in zip(cues, cues[1:]):
        if cur.start_ms - prev.end_ms >= gap_ms:
            runs.append(run)
            run = []
        run.append(cur)
    runs.append(run)

    pieces: list[list[Cue]] = []
    for run in runs:
        while len(run) > max_cues:
            head = run[:max_cues]
            third = max_cues // 3
            # Largest gap between consecutive cues inside the last third of
            # the head; falls back to the max_cues boundary when the tail is
            # one continuous block of talk (all gaps equal -> first wins).
            window_start = len(head) - third
            best_i, best_gap = window_start, -1
            for i in range(window_start, len(head)):
                gap = head[i].start_ms - head[i - 1].end_ms
                if gap > best_gap:
                    best_i, best_gap = i, gap
            pieces.append(head[:best_i])
            run = head[best_i:] + run[max_cues:]
        pieces.append(run)

    scenes: list[Scene] = []
    for i, piece in enumerate(pieces, 1):
        scenes.append(
            Scene(
                id=f"s{i:03d}",
                start_ms=piece[0].start_ms,
                end_ms=max(c.end_ms for c in piece),
                cue_ids=[c.id for c in piece],
            )
        )
    _assign_acts(scenes)
    return scenes


def gap_audit(cues: list[Cue], duration_ms: int, min_gap_ms: int = 3_000) -> list[tuple[int, int]]:
    """Cue-free intervals >= ``min_gap_ms`` — candidates for un-subtitled on-screen text."""
    gaps: list[tuple[int, int]] = []
    ordered = sorted(cues, key=lambda c: (c.start_ms, c.end_ms))
    cursor = 0
    for cue in ordered:
        if cue.start_ms - cursor >= min_gap_ms:
            gaps.append((cursor, cue.start_ms))
        cursor = max(cursor, cue.end_ms)
    if duration_ms - cursor >= min_gap_ms:
        gaps.append((cursor, duration_ms))
    return gaps


def _scene_boundary(a: Scene, b: Scene) -> int:
    """Cut point between consecutive scenes: midpoint of the silence."""
    return (a.end_ms + b.start_ms) // 2


def viewing_chunks(
    scenes: list[Scene], duration_ms: int, chunk_ms: int
) -> list[tuple[str, int, int, list[str]]]:
    """Partition 0..``duration_ms`` into chunks cut only at scene
    boundaries (silence midpoints). Scenes are packed greedily: a scene is
    added while the span from the chunk's start to *its own* end boundary
    stays within ``chunk_ms``. A single scene whose span exceeds
    ``chunk_ms`` forms its own chunk so coverage stays complete.
    Returns (chunk_id "v01", start_ms, end_ms, scene_ids)."""
    if duration_ms <= 0:
        return []
    if not scenes:
        return [("v01", 0, duration_ms, [])]

    # End boundary of each scene: midpoint of the silence after it, or the
    # video's end for the last one.
    end_bounds = [
        _scene_boundary(scene, nxt) for scene, nxt in zip(scenes, scenes[1:])
    ] + [duration_ms]

    groups: list[tuple[list[Scene], int]] = []  # (scenes, end boundary)
    group: list[Scene] = []
    group_start = 0
    for scene, end_b in zip(scenes, end_bounds):
        if group and end_b - group_start > chunk_ms:
            # Close the group at the boundary before this scene.
            prev_b = _scene_boundary(group[-1], scene)
            groups.append((group, prev_b))
            group_start = prev_b
            group = []
        group.append(scene)
    if group:
        groups.append((group, end_bounds[-1]))

    chunks: list[tuple[str, int, int, list[str]]] = []
    start = 0
    for i, (grp, end) in enumerate(groups):
        chunks.append((f"v{i + 1:02d}", start, end, [s.id for s in grp]))
        start = end
    return chunks


@dataclass(frozen=True)
class ConversationalEpisode:
    """Homogeneous participation-framework episode inside a silence-bounded Scene."""

    id: str
    scene_id: str
    cue_ids: tuple[str, ...]
    participants: frozenset[str]
    has_episodic_participant: bool


_GENERIC_PARTICIPANTS = frozenset({"group", "audience", "narration", "self", "system_ai", "unknown", "none", ""})


def partition_episodes(
    scene_cue_ids: Sequence[str],
    cues_by_id: Mapping[str, Cue],
    pairs_by_cue: Mapping[str, tuple[str, str]],
    *,
    scene_id: str = "s000",
) -> list[ConversationalEpisode]:
    """Partition a scene's cue ids into homogeneous conversational episodes.

    Splits a scene when the active interpersonal participant set shifts
    (Jaccard similarity < 0.5 across blocks of >= 2 interpersonal cues, or
    across a gap >= 2500 ms / an intervening OST location/title card).
    Unresolved ``group`` cues attach to their nearest contiguous episode.
    """
    cids = [cid for cid in scene_cue_ids if cid in cues_by_id]
    if not cids:
        return []

    def _Specific(cid: str) -> frozenset[str]:
        p = pairs_by_cue.get(cid)
        if p is None:
            return frozenset()
        return frozenset(x for x in p if x and x.lower() not in _GENERIC_PARTICIPANTS)

    # Identify contiguous runs of cues sharing compatible specific participant sets
    blocks: list[tuple[list[str], frozenset[str]]] = []
    cur_cids: list[str] = []
    cur_parts: frozenset[str] = frozenset()

    for idx, cid in enumerate(cids):
        c = cues_by_id[cid]
        if c.id.startswith("ost-") or c.source == "ost":
            if cur_cids:
                blocks.append((cur_cids, cur_parts))
                cur_cids, cur_parts = [], frozenset()
            continue
        sp = _Specific(cid)
        if not cur_cids:
            cur_cids = [cid]
            cur_parts = sp
            continue
        prev_c = cues_by_id[cur_cids[-1]]
        gap = c.start_ms - prev_c.end_ms
        if len(sp) >= 2 and len(cur_parts) >= 2:
            inter = len(sp & cur_parts)
            union = len(sp | cur_parts)
            jacc = inter / union if union else 1.0
            if jacc < 0.5:
                blocks.append((cur_cids, cur_parts))
                cur_cids = [cid]
                cur_parts = sp
                continue
        if gap >= 2500 and len(cur_parts) >= 2 and len(sp) < 2:
            # Look ahead to see if the next specific participant set after this gap shifts
            ahead_parts: frozenset[str] = frozenset()
            for nxt_cid in cids[idx:]:
                np = _Specific(nxt_cid)
                if len(np) >= 2:
                    ahead_parts = np
                    break
            if len(ahead_parts) >= 2:
                inter = len(ahead_parts & cur_parts)
                union = len(ahead_parts | cur_parts)
                if (inter / union if union else 1.0) < 0.5:
                    blocks.append((cur_cids, cur_parts))
                    cur_cids = [cid]
                    cur_parts = ahead_parts
                    continue
        cur_cids.append(cid)
        if len(sp) >= len(cur_parts):
            cur_parts = cur_parts | sp if len(cur_parts) < 2 else cur_parts

    if cur_cids:
        blocks.append((cur_cids, cur_parts))

    if not blocks:
        return [
            ConversationalEpisode(
                id=f"{scene_id}e01",
                scene_id=scene_id,
                cue_ids=tuple(cids),
                participants=frozenset(),
                has_episodic_participant=False,
            )
        ]

    # Merge 1-cue blocks or blocks without specific participants into the closest neighbor
    merged: list[tuple[list[str], frozenset[str]]] = []
    for b_cids, b_parts in blocks:
        if merged and (not b_parts or not merged[-1][1] or b_parts == merged[-1][1]):
            prev_cids, prev_parts = merged[-1]
            merged[-1] = (prev_cids + b_cids, prev_parts or b_parts)
        elif merged and len(b_cids) < 2 and len(merged[-1][0]) >= 2:
            prev_cids, prev_parts = merged[-1]
            merged[-1] = (prev_cids + b_cids, prev_parts)
        else:
            merged.append((list(b_cids), b_parts))

    episodes: list[ConversationalEpisode] = []
    for i, (e_cids, e_parts) in enumerate(merged, 1):
        has_ep = any(p.startswith("ep:") for p in e_parts)
        episodes.append(
            ConversationalEpisode(
                id=f"{scene_id}e{i:02d}",
                scene_id=scene_id,
                cue_ids=tuple(e_cids),
                participants=e_parts,
                has_episodic_participant=has_ep,
            )
        )
    return episodes
