"""S2 narrative layer: viewing pass (S2v), video map (S2a), timeline (S2b),
snapshot materialization (S2c), and the errata path.

The causal rules live in :func:`materialize_snapshots`:

* register rows come from the S2r register table (``register.yaml``), not
  from scene presence: a scene carries every table edge whose pair speaks a
  line in the scene, plus edges between the scene's cue speakers (and their
  ``group`` edges). Each row keeps the table level, address terms and the
  shifts whose span overlaps the scene.
* a character's knowledge contains only revelations anchored before the scene;
  twist facts additionally stay hidden until their ``audience_reveal_cue`` —
  before that they surface only as ``preserve_ambiguity`` flags.
* ``recent_events`` keeps at most the 5 most recent events per present
  character; voice cards and present-character state are emitted only for
  characters present in the scene's beat.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from importlib import resources
from typing import TYPE_CHECKING

from .gemini import Clip, GeminiBlocked
from .models import (
    AmbiguityFlag,
    CharacterState,
    ChunkObservations,
    Cue,
    CueDelta,
    VideoInfo,
    ContextMap,
    PerceptionLedger,
    RegisterMap,
    Revelation,
    Scene,
    SceneBeat,
    SceneEvent,
    SceneRegister,
    Snapshot,
    Timeline,
    TimelineErratum,
)
from .perception import selective_perception_pass as _selective_perception_pass
from .store import Checkpoints, input_hash
from .style import REGISTER_TOKEN, register_rules

_PROMPTS = resources.files("substage") / "prompts"
_TIMELINE_SCENES_PER_CALL = 20
_RECENT_EVENTS_MAX = 5
_ERRATUM_PREFIX = "ERRATUM: "

logger = logging.getLogger(__name__)

#: a blocked viewing clip splits in half until it is about this short
_VIEW_MIN_CLIP_MS = 60_000
#: on-screen text this far before its clip is read as clip-relative time
_OST_REL_SLACK_MS = 2_000


def _prompt(name: str) -> str:
    """Prompt template text with the style guide's register section embedded,
    so template-keyed checkpoints also change when that section does."""
    text = (_PROMPTS / name).read_text(encoding="utf-8")
    return text.replace(REGISTER_TOKEN, register_rules()) if REGISTER_TOKEN in text else text


def _mmss(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def _transcript_line(cue: Cue) -> str:
    """Compact transcript line: ``id|mm:ss|speaker_hint|text``."""
    text = re.sub(r"\s+", " ", cue.text).strip()
    suffixes = [f"♪" if cue.lyric else "", *cue.sound_cues]
    tail = " ".join(s for s in suffixes if s)
    return f"{cue.id}|{_mmss(cue.start_ms)}|{cue.speaker_hint or '-'}|{text}{(' ' + tail) if tail else ''}"


def _compact_observations(observations: list[ChunkObservations]) -> str:
    """Compacts viewing observations to text lines for the video-map prompt."""
    lines: list[str] = []
    for o in sorted(observations, key=lambda x: x.start_ms):
        lines.append(f"== {o.chunk_id} {o.start_ms}-{o.end_ms}ms: {o.summary}")
        if o.present_characters:
            lines.append("present: " + ", ".join(o.present_characters))
        for c in o.cues:
            parts = [c.cue_id]
            if c.speaker:
                parts.append(f"speaker={c.speaker}")
            if c.addressee:
                parts.append(f"to={c.addressee}")
            if c.delivery:
                parts.append(f"delivery={c.delivery}")
            if c.nonverbal:
                parts.append(f"note={c.nonverbal}")
            if c.lyric_plot_relevant:
                parts.append("lyric=plot-relevant")
            if len(parts) > 1:
                lines.append(" ".join(parts))
        for e in o.events:
            chars = f" [{','.join(e.characters)}]" if e.characters else ""
            kc = f" | knowledge: {e.knowledge_change}" if e.knowledge_change else ""
            lines.append(f"event@{e.t_ms}{chars}: {e.description}{kc}")
        for t in o.on_screen_text:
            pc = " CRITICAL" if t.plot_critical else ""
            cov = f" covered_by={t.covered_by_cue}" if t.covered_by_cue else ""
            lines.append(f"text@{t.start_ms}-{t.end_ms}: {t.text}{pc}{cov}")
    return "\n".join(lines)



def _merge_observations(
    chunk_id: str, start_ms: int, end_ms: int, halves: list[ChunkObservations]
) -> ChunkObservations:
    """Merge bisected-clip observations into the parent chunk's record."""
    merged = ChunkObservations(chunk_id=chunk_id, start_ms=start_ms, end_ms=end_ms)
    for h in halves:
        merged.present_characters += [
            c for c in h.present_characters if c not in merged.present_characters
        ]
        merged.cues += h.cues
        merged.events += h.events
        merged.on_screen_text += h.on_screen_text
    merged.summary = " / ".join(s for h in halves if (s := h.summary.strip()))
    return merged


def _decode_mmss_ms(ms: int) -> int:
    """Convert a base-100 MMSS*1000 timestamp (e.g. 101000 for 01:01) to ms (61000)."""
    s, rem = divmod(ms, 1000)
    if s >= 100:
        mm, ss = divmod(s, 100)
        if 0 <= ss < 60:
            return (mm * 60 + ss) * 1000 + rem
    return ms


def _absolute_ost(obs: ChunkObservations, start_ms: int, end_ms: int) -> None:
    """Put on-screen-text times on the video clock inside the clip window.

    Times well before the clip that fit its length were reported relative to
    the clip start and are shifted by it; base-100 MMSS*1000 times are decoded
    to milliseconds before clamping; all spans are then clamped to the
    window. Empty texts are dropped."""
    span = end_ms - start_ms
    non_empty = [t for t in obs.on_screen_text if t.text.strip()]
    def _is_mmss_cand(it: OnScreenText) -> bool:
        return (
            it.covered_by_cue is None
            and 100_000 <= it.start_ms < 600_000
            and it.start_ms != end_ms
            and it.start_ms % 100 == 0
            and it.end_ms % 100 == 0
            and (it.start_ms // 1000) % 100 < 60
            and (it.end_ms // 1000) % 100 < 60
        )

    high = [
        t for t in non_empty
        if t.covered_by_cue is None and 100_000 <= t.start_ms < 600_000 and t.start_ms != end_ms
    ]
    has_clamped_or_over = end_ms > 0 and any(
        (t.start_ms == end_ms and t.end_ms == end_ms) or t.start_ms > end_ms for t in non_empty
    )
    mmss_cands = [t for t in high if _is_mmss_cand(t)]
    batch_mmss = start_ms == 0 and (
        (has_clamped_or_over and len(mmss_cands) >= 1)
        or (len(high) >= 2 and len(mmss_cands) == len(high))
    )
    kept = []
    for t in non_empty:
        if end_ms > 0 and t.start_ms == end_ms and t.end_ms == end_ms:
            continue
        if end_ms > 0 and t.start_ms > end_ms:
            ds, de = _decode_mmss_ms(t.start_ms), _decode_mmss_ms(t.end_ms)
            if 0 <= ds <= span and start_ms > 0 and ds < start_ms - _OST_REL_SLACK_MS:
                t.start_ms, t.end_ms = ds + start_ms, de + start_ms
            elif start_ms <= ds <= end_ms:
                t.start_ms, t.end_ms = ds, max(ds, de)
            else:
                continue
        elif batch_mmss and _is_mmss_cand(t):
            t.start_ms = _decode_mmss_ms(t.start_ms)
            t.end_ms = _decode_mmss_ms(t.end_ms)
        if t.start_ms < start_ms - _OST_REL_SLACK_MS and 0 <= t.start_ms <= span:
            t.start_ms += start_ms
            t.end_ms += start_ms
        t.start_ms = min(max(t.start_ms, start_ms), end_ms)
        t.end_ms = min(max(t.end_ms, t.start_ms), end_ms)
        if end_ms > 0 and t.start_ms == end_ms and t.end_ms == end_ms:
            continue
        kept.append(t)
    obs.on_screen_text = kept

def _observe_clip(
    gc: GeminiClient,
    proxy: UploadedFile,
    chunk_id: str,
    start_ms: int,
    end_ms: int,
    prompt: str,
    stage: str,
) -> ChunkObservations:
    """Observe one clip; on a provider block split the clip in half and merge,
    down to ~_VIEW_MIN_CLIP_MS, then degrade to an empty 'blocked' record."""
    try:
        obs = gc.generate(
            stage=stage,
            prompt=prompt,
            schema=ChunkObservations,
            clips=[Clip(file=proxy, start_s=start_ms / 1000.0, end_s=end_ms / 1000.0)],
        )
        # the clip window is authoritative, not the model's echo
        obs.chunk_id = chunk_id
        obs.start_ms = start_ms
        obs.end_ms = end_ms
        _absolute_ost(obs, start_ms, end_ms)
        return obs
    except GeminiBlocked as exc:
        if end_ms - start_ms <= _VIEW_MIN_CLIP_MS * 2:
            logger.warning(
                "%s: ~%ss clip blocked (%s); recording empty observations",
                stage, (end_ms - start_ms) // 1000, exc.reason,
            )
            return ChunkObservations(
                chunk_id=chunk_id, start_ms=start_ms, end_ms=end_ms, summary="blocked"
            )
        mid_ms = start_ms + (end_ms - start_ms) // 2
        logger.warning(
            "%s: chunk blocked (%s); splitting clip %d-%dms in half",
            stage, exc.reason, start_ms, end_ms,
        )
        halves = [
            _observe_clip(gc, proxy, chunk_id, s, e, prompt, f"{stage}{suffix}")
            for (s, e), suffix in zip(
                ((start_ms, mid_ms), (mid_ms, end_ms)), ("a", "b")
            )
        ]
        return _merge_observations(chunk_id, start_ms, end_ms, halves)


# --------------------------------------------------------------------- S2v


def viewing_pass(
    ws: Workspace,
    gc: GeminiClient,
    video: VideoInfo,
    proxy: UploadedFile,
    cues: list[Cue],
    chunks: list[tuple[str, int, int, list[str]]],
    *,
    series_bible: object | None = None,
) -> list[ChunkObservations]:
    """Watch the video chunk by chunk and report observations only.

    ``chunks`` are ``(chunk_id, start_ms, end_ms, scene_ids)`` covering the
    video, as produced by ``scenes.viewing_chunks``. Clip offsets are absolute
    video seconds. Each chunk is checkpointed independently.
    """
    from .series_state import render_viewing_roster

    cp = Checkpoints(ws, "s2v")
    template = _prompt("viewing.md")
    roster_block = render_viewing_roster(series_bible)  # type: ignore[arg-type]
    bible_tag = (
        (series_bible.fingerprint(),)  # type: ignore[attr-defined]
        if series_bible is not None and hasattr(series_bible, "fingerprint") and series_bible.fingerprint()  # type: ignore[attr-defined]
        else ()
    )
    out: list[ChunkObservations] = []
    for chunk_id, start_ms, end_ms, _scene_ids in chunks:
        chunk_cues = [c for c in cues if start_ms <= c.start_ms < end_ms]
        h = input_hash(
            proxy.uri,
            f"{start_ms}-{end_ms}",
            template,
            gc.settings.model_main,
            *[c.model_dump_json() for c in chunk_cues],
            *bible_tag,
        )
        hit = cp.get(chunk_id, h, ChunkObservations)
        if hit is not None:
            out.append(hit)
            continue
        cue_lines = "\n".join(
            f"{c.id}|{c.start_ms}-{c.end_ms}|{c.speaker_hint or '-'}|{re.sub(r'\s+', ' ', c.text).strip()}"
            + (" ♪" if c.lyric else "")
            + ((" " + " ".join(c.sound_cues)) if c.sound_cues else "")
            for c in chunk_cues
        )
        prompt = template.format(
            title=video.title,
            year=video.year or "?",
            start_ms=start_ms,
            cues=cue_lines or "(no subtitle cues in this window)",
        ) + roster_block
        obs = _observe_clip(
            gc, proxy, chunk_id, start_ms, end_ms, prompt, f"s2v:{chunk_id}"
        )
        spoken = [c.start_ms for c in chunk_cues if c.text.strip()]
        if start_ms == 0 and end_ms > 240_000 and spoken and min(spoken) >= 90_000:
            open_end = min(spoken)
            open_cues = [c for c in chunk_cues if c.start_ms < open_end]
            open_lines = "\n".join(
                f"{c.id}|{c.start_ms}-{c.end_ms}|{c.speaker_hint or '-'}|{re.sub(r'\s+', ' ', c.text).strip()}"
                + (" ♪" if c.lyric else "")
                + ((" " + " ".join(c.sound_cues)) if c.sound_cues else "")
                for c in open_cues
            )
            open_prompt = (
                template.format(
                    title=video.title,
                    year=video.year or "?",
                    start_ms=0,
                    cues=open_lines or "(no subtitle cues in this window)",
                )
                + roster_block
                + "\n\nThis dialogue-free opening window often contains title/credit cards and readable English text on props (cups, covers, cards, flyers). Report every readable prop text as [sign] and every credit card as [credit] with exact video milliseconds (1 minute = 60,000 ms, 2 minutes = 120,000 ms)."
            )
            open_obs = _observe_clip(
                gc, proxy, f"{chunk_id}_open", 0, open_end, open_prompt, f"s2v:{chunk_id}_open"
            )
            obs.on_screen_text = sorted(
                [*open_obs.on_screen_text, *[t for t in obs.on_screen_text if t.start_ms >= open_end]],
                key=lambda x: (x.start_ms, x.end_ms),
            )
        cp.put(chunk_id, h, obs)
        out.append(obs)
    return out

def selective_perception_pass(
    ws: Workspace,
    gc: GeminiClient,
    video: VideoInfo,
    proxy: UploadedFile,
    cues: Sequence[Cue],
    scenes: Sequence[Scene] | None = None,
    observations: Iterable[object] = (),
    *,
    cue_ids: Iterable[str] | None = None,
    model: str | None = None,
    window_pad_ms: int = 1_000,
    max_window_ms: int = 45_000,
    max_cues_per_call: int = 12,
) -> PerceptionLedger:
    """Opt in to bounded cue-level AV perception without changing viewing_pass.

    ``viewing_pass`` remains the coarse, whole-chunk narrative observation API.
    This helper delegates to the cue-keyed ledger path and is intentionally
    selective: callers may bound it by existing scenes and/or cue ids.
    """
    return _selective_perception_pass(
        ws,
        gc,
        video,
        proxy,
        cues,
        scenes=scenes,
        cue_ids=cue_ids,
        observations=observations,
        model=model,
        window_pad_ms=window_pad_ms,
        max_window_ms=max_window_ms,
        max_cues_per_call=max_cues_per_call,
    )


build_perception_ledger = selective_perception_pass


def render_perception_ledger(
    ledger: PerceptionLedger | dict[str, object] | None,
    cue_ids: Sequence[str] | None = None,
) -> str:
    """Render advisory perception through the canonical ledger renderer."""
    from .perception import render_perception_ledger as _render

    return _render(ledger, cue_ids)


# --------------------------------------------------------------------- S2a


def build_context_map(
    ws: Workspace,
    gc: GeminiClient,
    video: VideoInfo,
    cues: list[Cue],
    observations: list[ChunkObservations],
    tmdb_hint: str | None = None,
    *,
    series_bible: object | None = None,
) -> ContextMap:
    """One text call over the whole transcript + compacted observations."""
    from .series_state import reconcile_characters, render_context_map_locks

    cp = Checkpoints(ws, "s2a")
    template = _prompt("context_map.md")
    locks_block = render_context_map_locks(series_bible)  # type: ignore[arg-type]
    combined_hint = "\n".join(x for x in (tmdb_hint, locks_block) if x)
    bible_tag = (
        (series_bible.fingerprint(),)  # type: ignore[attr-defined]
        if series_bible is not None and hasattr(series_bible, "fingerprint") and series_bible.fingerprint()  # type: ignore[attr-defined]
        else ()
    )
    # generator identity (exact template text + model) is part of the key so a
    # prompt or model change regenerates instead of reusing a stale map
    h = input_hash(
        template,
        gc.settings.model_main,
        video.model_dump_json(),
        tmdb_hint or "",
        *[c.model_dump_json() for c in cues],
        *[o.model_dump_json() for o in observations],
        *bible_tag,
    )
    hit = cp.get("context_map", h, ContextMap)
    if hit is None:
        prompt = template.format(
            title=video.title,
            year=video.year or "?",
            tmdb_hint=(f"Metadata hint: {combined_hint}" if combined_hint else ""),
            transcript="\n".join(_transcript_line(c) for c in cues),
            observations=_compact_observations(observations),
        )
        try:
            hit = gc.generate(stage="s2a", prompt=prompt, schema=ContextMap)
        except GeminiBlocked as exc:
            # likely the compacted observations tripped the filter — retry once
            # on transcript alone; a video map is required, so re-raise if still
            # blocked
            logger.warning("s2a blocked (%s); retrying transcript-only", exc.reason)
            prompt = template.format(
                title=video.title,
                year=video.year or "?",
                tmdb_hint=(f"Metadata hint: {combined_hint}" if combined_hint else ""),
                transcript="\n".join(_transcript_line(c) for c in cues),
                observations="(unavailable: viewing-pass observations blocked)",
            )
            hit = gc.generate(stage="s2a:transcript", prompt=prompt, schema=ContextMap)
        cp.put("context_map", h, hit)
    if series_bible is not None:
        hit = reconcile_characters(hit, series_bible, cues)  # type: ignore[arg-type]
    return hit

# --------------------------------------------------------------------- S2b


def _scene_batches(scenes: list[Scene], per_call: int = _TIMELINE_SCENES_PER_CALL) -> list[list[Scene]]:
    """~Act-sized batches of ≤ ``per_call`` scenes. When every scene carries an
    act number, batches never straddle acts."""
    if scenes and all(s.act is not None for s in scenes):
        by_act: dict[int, list[Scene]] = {}
        for s in scenes:
            by_act.setdefault(s.act, []).append(s)  # type: ignore[union-attr]
        batches: list[list[Scene]] = []
        for act in sorted(by_act):
            g = by_act[act]
            batches += [g[i : i + per_call] for i in range(0, len(g), per_call)]
        return batches
    return [scenes[i : i + per_call] for i in range(0, len(scenes), per_call)]


def build_timeline(
    ws: Workspace,
    gc: GeminiClient,
    context_map: ContextMap,
    scenes: list[Scene],
    cues: list[Cue],
    observations: list[ChunkObservations],
) -> Timeline:
    """Narrative timeline in ~act batches; every scene gets a beat."""
    cp = Checkpoints(ws, "s2b")
    ordered = sorted(scenes, key=lambda s: s.start_ms)
    cues_by_id = {c.id: c for c in cues}
    template = _prompt("timeline.md")
    characters = ", ".join(
        f"{ch.id} ({ch.en_name} / {ch.ko_name})" for ch in context_map.characters
    )
    context_map_block = (
        f"synopsis: {context_map.synopsis}\n"
        f"tone: {context_map.tone_baseline}\n"
        f"characters: {characters}"
    )
    beats: dict[str, SceneBeat] = {}

    def _timeline_call(batch: list[Scene], label: str, stage: str) -> Timeline:
        """One timeline call for `batch`; on a provider block bisect the batch
        recursively — a single blocked scene degrades to an empty beat."""
        try:
            return gc.generate(stage=stage, prompt=_batch_prompt(batch, label),
                               schema=Timeline)
        except GeminiBlocked as exc:
            if len(batch) == 1:
                logger.warning(
                    "%s: scene %s blocked (%s); leaving an empty beat",
                    stage, batch[0].id, exc.reason,
                )
                return Timeline(scenes=[SceneBeat(scene_id=batch[0].id, beat="")])
            mid = len(batch) // 2
            logger.warning(
                "%s: batch %s..%s blocked (%s); bisecting",
                stage, batch[0].id, batch[-1].id, exc.reason,
            )
            halves = [
                _timeline_call(batch[:mid], f"{label}a", f"{stage}a"),
                _timeline_call(batch[mid:], f"{label}b", f"{stage}b"),
            ]
            return Timeline(scenes=[b for half in halves for b in half.scenes])

    def _batch_prompt(batch: list[Scene], label: str) -> str:
        start_ms, end_ms = batch[0].start_ms, batch[-1].end_ms
        obs_slice = [
            o for o in observations if o.start_ms < end_ms and o.end_ms > start_ms
        ]
        scene_blocks: list[str] = []
        for s in batch:
            scene_blocks.append(f"## {s.id} ({s.start_ms}-{s.end_ms}ms)")
            scene_blocks += [
                _transcript_line(cues_by_id[cid]) for cid in s.cue_ids if cid in cues_by_id
            ]
        return template.format(
            batch_label=f"{label} ({batch[0].id}–{batch[-1].id})",
            characters=characters or "(none)",
            context_map=context_map_block,
            scenes="\n".join(scene_blocks),
            observations=_compact_observations(obs_slice) or "(none)",
        )

    for n, batch in enumerate(_scene_batches(ordered), start=1):
        start_ms = batch[0].start_ms
        end_ms = batch[-1].end_ms
        obs_slice = [
            o for o in observations if o.start_ms < end_ms and o.end_ms > start_ms
        ]
        h = input_hash(
            template,
            gc.settings.model_main,
            context_map.model_dump_json(),
            *[s.model_dump_json() for s in batch],
            *[
                cues_by_id[cid].model_dump_json()
                for s in batch
                for cid in s.cue_ids
                if cid in cues_by_id
            ],
            *[o.model_dump_json() for o in obs_slice],
        )
        key = f"{batch[0].id}-{batch[-1].id}"
        hit = cp.get(key, h, Timeline)
        if hit is None:
            hit = _timeline_call(batch, str(n), f"s2b:{key}")
            cp.put(key, h, hit)
        for b in hit.scenes:
            beats.setdefault(b.scene_id, b)
    # a beat for every scene, in scene order; empty beats for gaps
    result = Timeline(
        scenes=[beats.get(s.id, SceneBeat(scene_id=s.id, beat="")) for s in ordered]
    )
    return result




# --------------------------------------------------------------------- S2c


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", "", name.lower())


def _alias_map(context_map: ContextMap) -> dict[str, str]:
    """normalized surface form -> character id."""
    m: dict[str, str] = {}
    for ch in context_map.characters:
        for form in (ch.id, ch.en_name, ch.ko_name, *ch.aliases):
            m.setdefault(_norm(form), ch.id)
    return m


def _resolve(name: str | None, aliases: dict[str, str]) -> str | None:
    if name is None:
        return None
    return aliases.get(_norm(name), name)


def _pair(speaker: str, addressee: str) -> str:
    return f"{speaker}>{addressee}"


def _norm_fact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def apply_video_facts(
    snap: Snapshot,
    context_map: ContextMap,
    *,
    unrevealed_facts: Sequence[str] = (),
) -> None:
    """Copy the video's tone baseline, rating and the one-line persona of each
    character named in *snap* (``character_names``) onto the snapshot. The
    persona carries the voice's idiolect too: attitude (irreverent, profane)
    often lives there, and register/profanity choices depend on it."""
    from .series_state import mask_spoiler_persona

    snap.tone_baseline = context_map.tone_baseline
    snap.rating = context_map.rating
    snap.personas = {}
    for ch in context_map.characters:
        if ch.id not in snap.character_names or not ch.persona.strip():
            continue
        raw_persona = (
            mask_spoiler_persona(ch.persona, unrevealed_facts)
            if unrevealed_facts
            else ch.persona
        )
        line = " ".join(raw_persona.split())
        if ch.voice.idiolect.strip():
            line += " | talks: " + " ".join(ch.voice.idiolect.split())
        snap.personas[ch.id] = line

def materialize_snapshots(
    context_map: ContextMap,
    timeline: Timeline,
    scenes: list[Scene],
    cues: list[Cue],
    register: RegisterMap | None = None,
    pairs_by_cue: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Snapshot]:
    """Deterministically fold context_map + timeline (+ the register table and
    the cue -> (speaker, addressee) map of :func:`substage.register.cue_pairs`)
    into per-scene causal state.

    Pure Python, no model calls. See module docstring for the causal rules.
    """
    pos = {c.id: i for i, c in enumerate(cues)}
    text_by_id = {c.id: c.text for c in cues}
    hint_by_id = {c.id: c.speaker_hint for c in cues}
    INF = len(cues) + 1

    def p(cid: str) -> int:
        return pos.get(cid, INF)

    aliases = _alias_map(context_map)
    ko_names = {ch.id: ch.ko_name for ch in context_map.characters}

    def rid(name: str | None) -> str | None:
        return _resolve(name, aliases)

    beats = {b.scene_id: b for b in timeline.scenes}
    ordered = sorted(scenes, key=lambda s: s.start_ms)
    edges = {(e.speaker, e.addressee): e for e in (register.edges if register else [])}
    pairs_by_cue = pairs_by_cue or {}
    # waypoints seed entry emotions; consume them in order as scenes pass them
    waypoints: dict[str, list[tuple[int, str]]] = {}
    for arc in context_map.emotional_arcs:
        ch = rid(arc.character) or arc.character
        waypoints[ch] = sorted((p(w.cue_id), w.state) for w in arc.waypoints)
    wp_idx = {ch: 0 for ch in waypoints}
    # normalized twist facts -> reveal position, to gate lookalike revelations
    twist_by_text = {_norm_fact(t.fact): p(t.audience_reveal_cue) for t in context_map.twist_facts}

    knows: dict[str, list[str]] = {}
    emo: dict[str, str] = {}
    recent: dict[str, list[str]] = {}

    snapshots: dict[str, Snapshot] = {}
    for sc in ordered:
        beat = beats.get(sc.id)
        cue_set = set(sc.cue_ids)
        member = [p(cid) for cid in sc.cue_ids]
        first = min(member) if member else INF
        last = max(member) if member else -1
        present = [r for r in (rid(x) for x in (beat.chars_present if beat else [])) if r]

        # ---- running state: everything strictly before this scene
        for ch, wps in waypoints.items():
            while wp_idx[ch] < len(wps) and wps[wp_idx[ch]][0] < first:
                emo[ch] = wps[wp_idx[ch]][1]
                wp_idx[ch] += 1

        snap = Snapshot(scene_id=sc.id)
        # scene registers follow the cue speakers, never chars_present
        scene_pairs = {pairs_by_cue[cid] for cid in sc.cue_ids if cid in pairs_by_cue}
        speakers = {spk for spk, _ in scene_pairs}
        for cid in sc.cue_ids:
            for part in (hint_by_id.get(cid) or "").split("/"):
                if (r := aliases.get(_norm(part))) is not None:
                    speakers.add(r)
        for (spk, adr), e in edges.items():
            if (spk, adr) not in scene_pairs and not (spk in speakers and (adr in speakers or adr == "group")):
                continue
            snap.registers[_pair(spk, adr)] = SceneRegister(
                level=e.level,
                pronoun=e.pronoun,
                vocative=e.vocative,
                shifts=[
                    sh for sh in e.shifts
                    if p(sh.at_cue) <= last and (p(sh.until_cue) if sh.until_cue else INF) > first
                ],
            )
        snap.character_names = {ch: ko_names[ch] for ch in present if ch in ko_names}
        for cid in sc.cue_ids:
            for part in (hint_by_id.get(cid) or "").split("/"):
                if (r := aliases.get(_norm(part))) is not None:
                    snap.character_names[r] = ko_names[r]
        for ch in present:
            snap.characters[ch] = CharacterState(
                emotion=emo.get(ch),
                knows=list(knows.get(ch, [])),
                recent_events=list(recent.get(ch, [])),
            )
        snap.voice_cards = {
            ch.id: ch.voice for ch in context_map.characters if ch.id in present
        }
        unrevealed_facts = [
            tf.fact for tf in context_map.twist_facts if p(tf.audience_reveal_cue) >= first
        ]
        apply_video_facts(snap, context_map, unrevealed_facts=unrevealed_facts)
        from .series_state import word_boundary_in

        scene_text = " ".join(text_by_id.get(cid, "") for cid in sc.cue_ids).lower()
        snap.glossary = [g for g in context_map.glossary if word_boundary_in(g.en, scene_text)]
        snap.callbacks_live = [
            cb
            for cb in context_map.callbacks
            if any(c in cue_set for c in cb.payoff_cues)
            or (
                any(p(c) < first for c in cb.setup_cues)
                and (not cb.payoff_cues or any(p(c) >= first for c in cb.payoff_cues))
            )
        ]
        snap.preserve_ambiguity = [
            a for a in context_map.ambiguous if a.cue_id in cue_set
        ]
        for tf in context_map.twist_facts:
            # fact still concealed from the audience at scene entry
            if p(tf.audience_reveal_cue) >= first and sc.cue_ids:
                snap.preserve_ambiguity.append(
                    AmbiguityFlag(
                        cue_id=sc.cue_ids[0],
                        note=(
                            f"Concealed until {tf.audience_reveal_cue}: {tf.fact} "
                            "(keep related lines double-edged)"
                        ),
                    )
                )

        # ---- in-scene deltas; they mutate running state for later scenes
        deltas: dict[str, CueDelta] = {}

        def delta(cid: str) -> CueDelta:
            return deltas.setdefault(cid, CueDelta(cue_id=cid))

        if beat:
            for r in beat.revelations:
                # suppress lookalike restatements of a still-hidden twist fact
                reveal_p = twist_by_text.get(_norm_fact(r.fact))
                if reveal_p is not None and p(r.cue_id) < reveal_p:
                    continue
                delta(r.cue_id).revelations.append(r)
                for w in r.to_whom:
                    ch = rid(w)
                    if ch and ch != "audience":
                        k = knows.setdefault(ch, [])
                        if r.fact not in k:
                            k.append(r.fact)
            for ec in beat.emotion_changes:
                delta(ec.cue_id).emotion_changes.append(ec)
                ch = rid(ec.character) or ec.character
                emo[ch] = ec.to_state
            for e in beat.events:
                targets = [rid(c) or c for c in e.characters] or present
                entry = f"{e.cue_id}: {e.description}"
                for ch in targets:
                    lst = recent.setdefault(ch, [])
                    lst.append(entry)
                    del lst[:-_RECENT_EVENTS_MAX]
        for tf in context_map.twist_facts:
            tp = p(tf.audience_reveal_cue)
            if first <= tp <= last:
                delta(tf.audience_reveal_cue).revelations.append(
                    Revelation(
                        cue_id=tf.audience_reveal_cue,
                        fact=tf.fact,
                        to_whom=["audience"],
                    )
                )
                # pre-reveal knowledge gates lift once the fact is public
                for kb in tf.known_by:
                    if p(kb.from_cue) < first:
                        ch = rid(kb.character) or kb.character
                        k = knows.setdefault(ch, [])
                        if tf.fact not in k:
                            k.append(tf.fact)

        snap.cue_deltas = [d for _, d in sorted(deltas.items(), key=lambda kv: p(kv[0]))]
        snapshots[sc.id] = snap
    return snapshots


# ------------------------------------------------------------------- errata


def _cue_refs(beat: SceneBeat) -> set[str]:
    return (
        {e.cue_id for e in beat.events}
        | {r.cue_id for r in beat.revelations}
        | {e.cue_id for e in beat.emotion_changes}
    )


def _beat_index_for_cue(timeline: Timeline, cue_id: str) -> int | None:
    """Beat holding the cue, else first beat referencing a later cue, else last."""
    if not timeline.scenes:
        return None
    future: int | None = None
    for i, b in enumerate(timeline.scenes):
        refs = _cue_refs(b)
        if cue_id in refs:
            return i
        if future is None and any(r > cue_id for r in refs):
            future = i
    return future if future is not None else len(timeline.scenes) - 1


def apply_errata(
    context_map: ContextMap,
    timeline: Timeline,
    errata: list[TimelineErratum],
    accepted_ids: set[int],
) -> tuple[ContextMap, Timeline]:
    """Fold validator/human-accepted errata into the narrative artifacts.

    ``accepted_ids`` are indices into ``errata``. For each accepted erratum the
    cue-anchored entries it disputes are removed (timeline events, revelations,
    emotion changes; video-map ambiguity flags/arc waypoints/reveal anchors
    keyed by those cues) and the claim is recorded as an ``ERRATUM:`` scene
    event so snapshot materialization propagates it. The register table is
    never touched (humans edit register.yaml).
    Rejected errata change nothing. Returns new models; inputs are untouched.
    """
    fm = context_map.model_copy(deep=True)
    tl = timeline.model_copy(deep=True)
    for i, er in enumerate(errata):
        if i not in accepted_ids:
            continue
        disputed = set(er.cue_ids)
        # locate the anchor beat before removals erase the cue refs it uses
        anchor = min(er.cue_ids) if er.cue_ids else ""
        idx = _beat_index_for_cue(tl, anchor)
        for beat in tl.scenes:
            beat.events = [e for e in beat.events if e.cue_id not in disputed]
            beat.revelations = [r for r in beat.revelations if r.cue_id not in disputed]
            beat.emotion_changes = [
                e for e in beat.emotion_changes if e.cue_id not in disputed
            ]
        fm.ambiguous = [a for a in fm.ambiguous if a.cue_id not in disputed]
        for arc in fm.emotional_arcs:
            arc.waypoints = [w for w in arc.waypoints if w.cue_id not in disputed]
        for tf in fm.twist_facts:
            if tf.audience_reveal_cue in disputed:
                tf.audience_reveal_cue = ""
            tf.known_by = [k for k in tf.known_by if k.from_cue not in disputed]
        for cb in fm.callbacks:
            cb.setup_cues = [c for c in cb.setup_cues if c not in disputed]
            cb.payoff_cues = [c for c in cb.payoff_cues if c not in disputed]
        if idx is not None:
            tl.scenes[idx].events.append(
                SceneEvent(
                    cue_id=anchor,
                    description=f"{_ERRATUM_PREFIX}{er.claim} (evidence: {er.evidence})",
                )
            )
    return fm, tl
