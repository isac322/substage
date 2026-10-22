"""End-to-end pipeline driver.

run() order (contract):
  ingest -> scenes -> upload proxy -> viewing_pass -> context_map -> timeline
  -> register (S2r) -> snapshots -> ost cues -> translate_video (+ firm-register
  retry when Settings.register_retry) -> review_windows -> finale_audit (applied) -> blind_read
  -> regenerate -> run_s5 -> picture_check(changed ∪ edited ∪ special cues)
  -> normalize/rewrap edited -> validate.check -> layout (S5c: split/merge/
  retime/condense) -> emit

Every stage writes its artifact into the workspace (video.json, cues.json,
scenes.json, observations.json, context_map.yaml, timeline.yaml, register.yaml,
snapshots.json, translations.json, edits.json, issues.json, layout.json,
register_report.json) so stages can resume via Checkpoints and humans can edit
the narrative artifacts (context_map/timeline/register.yaml) between runs.

The register table gates every edit after S3 (S4 review/finale/regen, C7
picture check, S5c condense): see substage.register.RegisterGate.

Experiment arms (`arm=`) run in workspace `<video_id>__<arm>`, seeded once from
the base workspace's pre-translation checkpoints (see seed_arm). `no_viewing`
skips S2v; the narrative is built from text only under distinct hashes.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
import shutil
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, Field, field_validator

from . import acquire, layout, media, narrative, recheck, review, style, subs, translate, validate
from . import emit as emit_mod
from . import register as register_mod
from . import scenes as scenes_mod
from . import llm
from . import series_state
from .capability import Architecture, RoutePlan, resolve_route
from .config import Workspace, load_secrets, load_settings, workspace
from .emit import SidecarCue
from .gemini import GeminiClient, UploadedFile
from .llm import LLMClient, make_client, text_client
from .models import (
    ChunkObservations,
    Cue,
    CueEdit,
    CueObservation,
    CuePerception,
    VideoInfo,
    ContextMap,
    OutEvent,
    PerceptionLedger,
    ReadFlag,
    RegisterMap,
    Scene,
    SceneTranslation,
    SemanticLineage,
    SemanticSceneTranslation,
    Snapshot,
    Timeline,
)
from .store import Checkpoints, _atomic_write, input_hash, load_model, save_model

# --stop-after names, in pipeline order
STAGES: tuple[str, ...] = (
    "ingest",
    "scenes",
    "proxy",
    "observations",
    "context_map",
    "timeline",
    "register",
    "snapshots",
    "ost",
    "translate",
    "review",
    "s5",
    "recheck",
    "layout",
    "emit",
)

# ---------------------------------------------------------------- checkpoint wrappers


class _Cues(BaseModel):
    items: list[Cue] = Field(default_factory=list)


class _Scenes(BaseModel):
    items: list[Scene] = Field(default_factory=list)


class _Observations(BaseModel):
    items: list[ChunkObservations] = Field(default_factory=list)


class _Snapshots(BaseModel):
    items: list[Snapshot] = Field(default_factory=list)


class _Translations(BaseModel):
    items: list[SemanticSceneTranslation | SceneTranslation] = Field(default_factory=list)

    @field_validator("items", mode="before")
    @classmethod
    def _strict_semantic_results(cls, values):
        return [validate.parse_translation_result(value) for value in values]


class _Edits(BaseModel):
    items: list[CueEdit] = Field(default_factory=list)


class _Flags(BaseModel):
    items: list[ReadFlag] = Field(default_factory=list)


class _Layout(BaseModel):
    items: list[OutEvent] = Field(default_factory=list)
    condensed_cue_ids: list[str] = Field(default_factory=list)


def _layout_result(cues, ko, scene_of, speakers, settings, client, gate, *, routed: bool,
                   lineage: SemanticLineage | None = None) -> _Layout:
    """Compare the deterministic layout against its only LLM-writing step."""
    before = layout.layout(cues, ko, scene_of, speakers, settings, lineage=lineage) if routed else []
    after = layout.layout(cues, ko, scene_of, speakers, settings, client, gate=gate, lineage=lineage)
    condensed = sorted({
        cid for old, new in zip(before, after, strict=True)
        if old.ko != new.ko for cid in new.cue_ids
    }) if routed else []
    return _Layout(items=after, condensed_cue_ids=condensed)


_SEMANTIC_ADVISORY_SUFFIX = (
    "\n\nAdvisory semantic source and actual retained output (not a truth lock). "
    "Check source against EN; keep uncertainty, essential answers and list relationships. "
    "Do not invent replies or force two dashes from source turn counts. "
    "Cross-cue refs record ancestry, not display timing ownership. "
    "Declared refs alone do not prove preserved meaning.\n"
)


class _SemanticContextClient:
    """Carry advisory source/retained evidence into downstream primary writes."""

    def __init__(self, client, lineage: SemanticLineage, trace: list[dict], ws, metadata):
        self._client, self._lineage, self._trace = client, lineage, trace
        self._ws, self._metadata = ws, metadata
        self.settings = client.settings

    def __getattr__(self, name):
        return getattr(self._client, name)

    def _save_trace(self) -> None:
        self._metadata["semantic_stages"] = self._trace
        _write_route_artifact(self._ws, self._metadata)

    def generate(self, **kwargs):
        stage = kwargs["stage"]
        prompt = kwargs["prompt"]
        if stage != "s4b_blind" and ":semantic-reattribute" not in stage:
            selected = {cid for cid in self._lineage.fragments_by_cue if cid in prompt}
            ancestry = {
                ref for cid in selected for fragment in self._lineage.fragments_by_cue[cid]
                for ref in [*fragment.refs, *fragment.prior_refs]
            }
            selected.update(act.cue_id for act in self._lineage.source.acts if act.ref in ancestry)
            acts = [act for act in self._lineage.source.acts if act.cue_id in selected]
            refs = {act.ref for act in acts}
            context = {
                "source_acts": [act.model_dump(mode="json") for act in acts],
                "source_relations": [relation.model_dump(mode="json") for relation in self._lineage.source.relations
                                     if any(ref in refs for ref in relation.refs)],
                "retained_fragments": {cid: [f.model_dump(mode="json") for f in fragments]
                                       for cid, fragments in self._lineage.fragments_by_cue.items() if cid in selected},
                "omissions": [omission.model_dump(mode="json") for omission in self._lineage.omissions
                              if omission.ref in refs],
                "ambiguities": self._lineage.source.ambiguities,
                "conflicts": self._lineage.source.conflicts,
            }
            kwargs["prompt"] = prompt + _SEMANTIC_ADVISORY_SUFFIX + json.dumps(context, ensure_ascii=False)
        record = {
            "stage": stage, "model": self.settings.model_main, "status": "attempted",
            "schema": kwargs["schema"].__name__, "input_hash": input_hash(kwargs["prompt"]),
            "clips": len(kwargs.get("clips") or ()), "images": len(kwargs.get("images") or ()),
        }
        self._trace.append(record)
        self._save_trace()
        try:
            result = self._client.generate(**kwargs)
        except Exception as exc:
            record.update(status="failed", error=type(exc).__name__)
            self._save_trace()
            raise
        record["status"] = "completed"
        self._save_trace()
        return result


def _semantic_picture_check(ws, client, scenes, scene_images, cues_by_id, ko, targets,
                            snapshots, scene_of, gate, lineage: SemanticLineage) -> list[CueEdit]:
    """Image-route C7 stays on the selected writer and existing scene frames."""
    edits = []
    for scene in scenes:
        selected = [cues_by_id[cid] for cid in scene.cue_ids if cid in targets and cid in cues_by_id]
        if not selected:
            continue
        images = scene_images.get(scene.id, [])
        if not images:
            raise RuntimeError(f"semantic-image picture check {scene.id}: missing scene keyframes")
        prompt = (
            "Check finished Korean subtitles against the attached scene keyframes and original EN. "
            "No audio/video is attached; do not claim unseen delivery. Preserve register and protected "
            "forms; return items only for necessary changes, kind picture.\n"
            + recheck._clip_lines(selected, ko, snapshots, scene_of)
        )
        batch = _ck(
            Checkpoints(ws, "semantic-image-picture"), scene.id,
            input_hash(prompt, lineage.model_dump_json(), _SEMANTIC_ADVISORY_SUFFIX,
                       *translate.image_digests(images), *translate._generation_settings(client), style.style_guide()),
            _Edits,
            lambda: client.generate(stage="picture-check:semantic-image", prompt=prompt, schema=_Edits,
                                    images=images, system=style.style_guide()),
        )
        seen = set()
        for edit in batch.items:
            if edit.cue_id not in {cue.id for cue in selected} or edit.cue_id in seen or not edit.new_ko.strip():
                continue
            cue = cues_by_id[edit.cue_id]
            if not validate._keeps_protected(cue, ko.get(cue.id, ""), edit.new_ko,
                                             snapshots.get(scene_of.get(cue.id, "")), ko):
                continue
            if gate is not None and not gate.check("picture", [cue.id], ko.get(cue.id, ""),
                                                   edit.new_ko, use_table=False):
                continue
            seen.add(cue.id)
            edits.append(edit.model_copy(update={"kind": "picture"}))
    return edits


def _semantic_workspace(ws, lineage: SemanticLineage | None, stage: str):
    """Keep semantic post-writer inputs out of legacy inner checkpoint keys."""
    if lineage is None:
        return ws
    return replace(ws, data_dir=ws.data_dir / "semantic-output" / stage / input_hash(
        "semantic-output-v1", lineage.model_dump_json(), style.style_guide(), _SEMANTIC_ADVISORY_SUFFIX,
    ))


# ---------------------------------------------------------------- helpers

_JUNK = re.compile(
    r"(?i)\b(2160p|1080p|720p|480p|4k|uhd|hdr|dv|bluray|blu-ray|bdrip|webrip|web-dl|web|hdrip|"
    r"remux|proper|repack|extended|unrated|imax|x264|x265|h264|h265|hevc|avc|aac|ac3|atmos|dts|"
    r"truehd|eac3|flac|5\.1|7\.1|10bit|8bit|amzn|nf|hulu|dsnp)\b"
)

_DPAF_SECOND_PERSON = re.compile(
    r"\b(?:you|your|yours|yourself|yourselves|this|that|these|those|here|there)\b",
    re.IGNORECASE,
)
_DPAF_MAX_CUES_PER_SCENE = 6
_DPAF_MAX_CUES = 96


def _flat_cue_observations(observations: Any) -> list[CueObservation]:
    if isinstance(observations, PerceptionLedger):
        return []
    out: list[CueObservation] = []
    for item in observations or ():
        if isinstance(item, CueObservation):
            out.append(item)
        elif hasattr(item, "cues"):
            out.extend(item.cues)
    return out


def _observation_by_cue(
    observations: Any,
    perception_ledger: PerceptionLedger | None = None,
) -> dict[str, CueObservation | CuePerception]:
    out: dict[str, CueObservation | CuePerception] = {
        item.cue_id: item for item in _flat_cue_observations(observations)
    }
    if perception_ledger is not None:
        for cue_id, perception in perception_ledger.cues.items():
            existing = out.get(cue_id)
            if existing is not None and getattr(existing, "turns", None) and not perception.turns:
                continue
            out[cue_id] = perception
    return out


def _dpaf_cue_ids(
    cues: list[Cue],
    scenes: list[Scene],
    observations: Any,
) -> list[str]:
    """Select a bounded, deterministic subset for cue-level AV perception."""
    obs_by_id = {item.cue_id: item for item in _flat_cue_observations(observations)}
    selected: list[str] = []
    by_id = {cue.id: cue for cue in cues}
    for scene in scenes:
        ranked: list[tuple[int, int, str]] = []
        for order, cue_id in enumerate(scene.cue_ids):
            cue = by_id.get(cue_id)
            if cue is None:
                continue
            observation = obs_by_id.get(cue_id)
            text = cue.text or ""
            score = 0
            if cue.speaker_hint:
                score += 3
            if cue.sound_cues:
                score += 3
            if observation is not None and (
                observation.addressee or observation.nonverbal or observation.delivery
            ):
                score += 2
            if observation is not None and observation.turns:
                score += 5
            if _DPAF_SECOND_PERSON.search(text):
                score += 2
            if "?" in text or "!" in text:
                score += 1
            if score:
                ranked.append((-score, order, cue_id))
        ranked.sort()
        selected.extend(cue_id for _, _, cue_id in ranked[:_DPAF_MAX_CUES_PER_SCENE])
        if len(selected) >= _DPAF_MAX_CUES:
            break
    return selected[:_DPAF_MAX_CUES]


def _slug(text: str) -> str:
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def _title_tokens(stem: str) -> list[str]:
    stem = re.sub(r"[._+\[\](){}]", " ", stem)
    return [t for t in stem.split() if t and not _JUNK.fullmatch(t)]


def _guess_year(stem: str) -> int | None:
    for tok in _title_tokens(stem):
        if re.fullmatch(r"(19|20)\d{2}", tok):
            return int(tok)
    return None


def _derive_video_id(stem: str, title: str | None, year: int | None) -> str:
    tokens = _title_tokens(title or stem)
    if year is not None and str(year) not in tokens:
        tokens.append(str(year))
    return _slug(" ".join(tokens)) or "video"


def _pretty_title(stem: str) -> str:
    tokens = [t for t in _title_tokens(stem) if not re.fullmatch(r"(19|20)\d{2}", t)]
    return " ".join(tokens) or stem


def _stopped(stop_after: str | None, stage: str, ws: Workspace) -> bool:
    if stage == stop_after:
        print(f"substage: stopped after '{stage}' — artifacts in {ws.data_dir}")
        return True
    return False


def _ck[T: BaseModel](
    ck: Checkpoints, key: str, h: str, cls: type[T], make: Callable[[], T],
    *, cache_if: Callable[[T], bool] | None = None,
) -> T:
    """Apply the same cacheability policy to existing hits and new results."""
    hit = ck.get(key, h, cls)
    if hit is not None and (cache_if is None or cache_if(hit)):
        return hit
    obj = make()
    if cache_if is None or cache_if(obj):
        ck.put(key, h, obj)
    return obj


def _editable(ck: Checkpoints, key: str, h: str, cls, artifact: Path, make):
    """_ck for user-editable artifacts: on a checkpoint hit the on-disk
    artifact (which a human may have edited) wins over the stored copy; the
    checkpoint is updated to the loaded model so later runs stay consistent."""
    hit = ck.get(key, h, cls)
    if hit is not None:
        if artifact.exists():
            hit = load_model(artifact, cls)
            ck.put(key, h, hit)
        return hit
    obj = make()
    save_model(artifact, obj)
    ck.put(key, h, obj)
    return obj


def _apply_edits(
    ko: dict[str, str],
    edits: list[CueEdit],
    gate: register_mod.RegisterGate | None = None,
    stage: str = "review",
    *,
    allow_omission: bool = False,
) -> tuple[dict[str, str], list[CueEdit]]:
    """Apply non-empty edits in order; with *gate*, an edit that moves a line
    off the register table (level or pronoun) is rejected and logged on the
    gate. Returns (new ko, applied edits)."""
    out = dict(ko)
    applied: list[CueEdit] = []
    for e in edits:
        if e.cue_id not in out or (
            not e.new_ko.strip() and not (allow_omission and e.kind == "omission")
        ):
            continue
        if gate is not None and not gate.check(stage, [e.cue_id], out[e.cue_id], e.new_ko):
            continue
        out[e.cue_id] = e.new_ko
        applied.append(e)
    return out, applied


def _apply_picture_edits(
    ko: dict[str, str], edits: list[CueEdit], settings, cues_by_id: dict[str, Cue],
    lineage: SemanticLineage | None = None,
) -> dict[str, str]:
    """Apply picture-check edits, then the deterministic S5 pass
    (validate.finish) on each edited cue. No model call and no second
    picture check."""
    out, _ = _apply_edits(ko, edits)
    if lineage is not None:
        validate.reconcile_lineage(lineage, out)
    for cid in {e.cue_id for e in edits}:
        if out.get(cid, "").strip():
            out[cid] = validate.finish(
                cues_by_id.get(cid), out[cid], settings,
                retained_fragments=lineage.fragments_by_cue.get(cid) if lineage is not None else None,
            )
    return out


def _hash_ko(ko: dict[str, str]) -> str:
    return input_hash(json.dumps(ko, ensure_ascii=False, sort_keys=True))


def _route_tag(
    plan: RoutePlan | None,
    *,
    mode: str,
    legacy: bool,
    roles: dict[str, str] | None = None,
    experiment_id: str | None = None,
) -> str:
    """Canonical routing identity folded into provider-dependent checkpoints."""
    payload: dict[str, object] = {
        "legacy": legacy,
        "mode": mode,
        "route": plan.to_dict() if plan is not None else None,
        "roles": roles or {},
    }
    if experiment_id is not None:
        payload["experiment_id"] = experiment_id
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_route_artifact(ws: Workspace, metadata: dict[str, object]) -> None:
    _atomic_write(
        ws.path("route.json"),
        (json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _route_default_model(settings, architecture: str | Architecture | None) -> str:
    """Choose a role default without changing Settings' legacy routing."""
    value = architecture.value if isinstance(architecture, Architecture) else architecture
    if value in {Architecture.DIRECT_AV.value, Architecture.AV_TWO_PASS.value, "semantic-clip"}:
        return settings.model_main
    return getattr(settings, "model_translate", None) or settings.model_main


def _route_clients(
    ws: Workspace,
    settings,
    plan: RoutePlan | None,
) -> tuple[LLMClient | None, LLMClient | None]:
    """Return primary text writer and native AV writer/evidence observer."""
    if plan is None:
        return None, None
    if plan.primary_model.startswith("anthropic/"):
        raise ValueError("explicit routes cannot generate new frozen Anthropic baseline translations")
    primary_settings = replace(
        settings, model_main=plan.primary_model, model_light=plan.primary_model,
        model_translate=plan.primary_model, model_text=plan.primary_model,
    )
    primary = make_client(ws, primary_settings, plan.primary_model)
    if plan.av_passes:
        return primary, primary
    if plan.architecture in {Architecture.DPAF, Architecture.CASCADE, Architecture.MIXED}:
        av_model = getattr(settings, "dpaf_perception_model", settings.model_main)
        resolve_route(av_model, Architecture.DIRECT_AV)
        av_settings = replace(settings, model_main=av_model, model_light=av_model)
        return primary, make_client(ws, av_settings, av_model)
    return primary, None


def _route_roles(settings, plan: RoutePlan | None, effective_mode: str) -> dict[str, str]:
    if plan is None:
        return {
            "narrative": settings.model_main,
            "translation": getattr(settings, "model_translate", None) or settings.model_main,
            "av": settings.model_main,
        }
    if plan.semantic_passes:
        roles = {stage: plan.primary_model for stage in (
            "narrative", "translation", "review", "finale", "regenerate", "compression",
            "layout", "blind_read", "fallback", "picture_check", "picture_writer",
            "semantic_extract", "semantic_write",
        )}
        roles.update(viewing=settings.model_main, av=plan.primary_model)
        return roles
    observer = (
        plan.primary_model if plan.av_passes else
        getattr(settings, "dpaf_perception_model", settings.model_main)
    )
    resolve_route(observer, Architecture.DIRECT_AV)
    roles = {
        stage: plan.primary_model for stage in (
            "narrative", "translation", "review", "finale", "regenerate",
            "compression", "layout", "blind_read", "fallback",
        )
    }
    roles.update({
        "viewing": settings.model_main,
        "av": observer,
        "picture_check": observer,
        "picture_writer": observer if plan.mixed_team else plan.primary_model,
    })
    if plan.av_passes == 2:
        roles["av_pass2"] = plan.primary_model
    if effective_mode in {"dpaf", "cascade", "mixed"}:
        roles["perception"] = observer
        roles["router"] = getattr(settings, "dpaf_router_model", None) or settings.model_main
    return roles



# LLM-produced checkpoints key on their generator identity too — the exact
# prompt template text (read through the stage's own loader) and the model —
# so a prompt or model change regenerates instead of reusing a stale artifact.


def _context_map_hash(settings, cues: list[Cue], observations: list[ChunkObservations], view_tag: tuple[str, ...]) -> str:
    return input_hash(
        _Cues(items=cues),
        _Observations(items=observations),
        narrative._prompt("context_map.md"),
        settings.model_main,
        *view_tag,
    )


def _timeline_hash(
    settings,
    context_map: ContextMap,
    scenes: list[Scene],
    cues: list[Cue],
    observations: list[ChunkObservations],
    view_tag: tuple[str, ...],
) -> str:
    return input_hash(
        context_map,
        _Scenes(items=scenes),
        _Cues(items=cues),
        _Observations(items=observations),
        narrative._prompt("timeline.md"),
        settings.model_main,
        *view_tag,
    )


def _register_hash(
    settings, context_map: ContextMap, cues: list[Cue], observations: list[ChunkObservations], view_tag: tuple[str, ...]
) -> str:
    return input_hash(
        context_map,
        _Cues(items=cues),
        _Observations(items=observations),
        register_mod.template(),
        register_mod.pairs_template(),
        settings.model_main,
        *view_tag,
    )


def _firm_retry(
    settings,
    table: register_mod.RegisterTable,
    pairs_by_cue: dict[str, tuple[str, str]],
    report: register_mod.RegisterReport,
    turn_pairs_by_cue: dict[str, tuple[register_mod.Pair | None, ...]] | None = None,
):
    """S3 post-pass for firm-register violations, including known sub-turns."""
    if not settings.register_retry:
        return None

    class _RetryNote(str):
        def __new__(cls, text: str, cue_ids: list[str]):
            item = super().__new__(cls, text)
            item.cue_ids = tuple(cue_ids)
            return item

    def check(st: SceneTranslation) -> str:
        violations = register_mod.firm_violations(
            table,
            pairs_by_cue,
            st,
            turn_pairs_by_cue=turn_pairs_by_cue,
        )
        if not violations:
            return ""
        report.firm_retried_scenes.append(st.scene_id)
        return _RetryNote(
            register_mod.retry_note(violations),
            [violation.cue_id for violation in violations],
        )

    return check


def _print_rows(rows: list[tuple[str, int, int, int, float]]) -> None:
    print(f"{'stage':<16}{'input':>12}{'output':>12}{'cached':>12}{'cost_usd':>12}")
    for label, ti, to, tc, cost in rows:
        print(f"{label:<16}{ti:>12,}{to:>12,}{tc:>12,}{cost:>12.4f}")


def _print_cost(gc: GeminiClient) -> None:
    summary = gc.usage_summary()
    rows = [
        (
            stage,
            int(r.get("input_tokens", 0)),
            int(r.get("output_tokens", 0)),
            int(r.get("cached_tokens", 0)),
            float(r.get("cost_usd", 0.0)),
        )
        for stage, r in summary.items()
        if stage != "total"
    ]
    total = summary.get("total", {})
    rows.append(
        (
            "TOTAL",
            int(total.get("input_tokens", 0)),
            int(total.get("output_tokens", 0)),
            int(total.get("cached_tokens", 0)),
            float(total.get("cost_usd", 0.0)),
        )
    )
    _print_rows(rows)


# ---------------------------------------------------------------- stages


def _do_ingest(
    ws: Workspace,
    gc: GeminiClient,
    video: Path,
    en_sub: Path | None,
    title: str | None,
    year: int | None,
    media_id: str | None = None,
) -> VideoInfo:
    streams, duration_ms = media.probe(video)
    audio = media.pick_dialogue_audio(streams)
    if en_sub is not None:
        en_sub_path, forced_path, origin, off, scale = Path(en_sub).resolve(), None, "user", None, None
    else:
        en_sub_path, forced_path, origin, off, scale = acquire.acquire_english(ws, gc, video, streams, media_id=media_id)
    # The low-res proxy is built lazily at the proxy stage (_proxy_upload):
    # ingest-only runs (--stop-after ingest/scenes) never encode video.
    return VideoInfo(
        video_id=ws.video_id,
        title=title or _pretty_title(video.stem),
        year=year,
        video_path=str(video),
        duration_ms=duration_ms,
        audio_stream=audio.index,
        en_sub_path=str(en_sub_path),
        en_sub_origin=origin,
        forced_sub_path=str(forced_path) if forced_path else None,
        proxy_path=None,
        sync_offset_s=off,
        sync_scale=scale,
    )


def _ingest(
    ws: Workspace,
    gc: GeminiClient,
    ck: Checkpoints,
    video: Path,
    en_sub: Path | None,
    title: str | None,
    year: int | None,
    media_id: str | None = None,
) -> VideoInfo:
    st = video.stat()
    h = input_hash(str(video), str(st.st_size), str(int(st.st_mtime)), str(en_sub) if en_sub else "", media_id or "")
    video = _ck(ck, "video", h, VideoInfo, lambda: _do_ingest(ws, gc, video, en_sub, title, year, media_id))
    save_model(ws.path("video.json"), video)
    return video


def _load_cues(ck: Checkpoints, video: VideoInfo) -> list[Cue]:
    en_bytes = Path(video.en_sub_path).read_bytes()
    forced_bytes = Path(video.forced_sub_path).read_bytes() if video.forced_sub_path else b""
    # The parser is part of the input: a parsing fix must re-derive cues (downstream
    # stages hash the cue list, so they only rerun where the cue text changed).
    h = input_hash(en_bytes, forced_bytes, Path(subs.__file__).read_bytes())

    def make() -> _Cues:
        main = subs.load_cues(Path(video.en_sub_path))
        cues = (
            subs.merge_forced(main, subs.load_cues(Path(video.forced_sub_path), forced=True, source="forced"))
            if video.forced_sub_path
            else main
        )
        return _Cues(items=cues)

    return _ck(ck, "cues", h, _Cues, make).items


def _proxy_upload(ws: Workspace, gc: GeminiClient, video: VideoInfo) -> UploadedFile:
    """Encode the proxy once per video (cached in the workspace cache dir, which
    arms share by symlink; make_proxy skips an up-to-date file) and upload it."""
    if not video.proxy_path:
        streams, _ = media.probe(Path(video.video_path))
        audio = next(s for s in streams if s.index == video.audio_stream)
        video.proxy_path = str(media.make_proxy(Path(video.video_path), audio, ws.cache("proxy.mp4")))
        save_model(ws.path("video.json"), video)
    return gc.upload(Path(video.proxy_path), "video/mp4")


def _merge_ost(cues: list[Cue], scenes: list[Scene], ost: list[Cue]) -> tuple[list[Cue], list[Scene]]:
    """Insert ost cues into the cue list; the containing/nearest scene grows to
    cover each ost cue (one scene is created when none exist)."""
    cues = sorted([*cues, *ost], key=lambda c: (c.start_ms, c.id))
    pos = {c.id: c.start_ms for c in cues}
    if not scenes:
        span = Scene(
            id="s001",
            start_ms=min(o.start_ms for o in ost),
            end_ms=max(o.end_ms for o in ost),
            cue_ids=sorted((o.id for o in ost), key=lambda cid: pos[cid]),
        )
        return cues, [span]
    ordered = sorted(scenes, key=lambda s: s.start_ms)
    for o in ost:
        mid = (o.start_ms + o.end_ms) // 2
        scene = next((s for s in ordered if s.start_ms <= mid < s.end_ms), None)
        if scene is None:
            scene = min(ordered, key=lambda s: min(abs(mid - s.start_ms), abs(mid - s.end_ms)))
        scene.cue_ids = sorted([*scene.cue_ids, o.id], key=lambda cid: pos.get(cid, 0))
        scene.start_ms = min(scene.start_ms, o.start_ms)
        scene.end_ms = max(scene.end_ms, o.end_ms)
    scenes.sort(key=lambda s: s.start_ms)
    return cues, scenes


def _special_cues(context_map: ContextMap, timeline: Timeline) -> set[str]:
    """Comedy/turning/twist-sensitive cues that always get a picture check
    (register shifts are not: C7 has no register authority)."""
    ids: set[str] = set()
    for cb in context_map.callbacks:
        ids.update(cb.setup_cues)
        ids.update(cb.payoff_cues)
    for arc in context_map.emotional_arcs:
        ids.update(w.cue_id for w in arc.waypoints)
    for t in context_map.twist_facts:
        ids.add(t.audience_reveal_cue)
        ids.update(k.from_cue for k in t.known_by)
    for flag in context_map.ambiguous:
        ids.add(flag.cue_id)
    for beat in timeline.scenes:
        for rv in beat.revelations:
            ids.add(rv.cue_id)
        for ec in beat.emotion_changes:
            ids.add(ec.cue_id)
    return ids


# ---------------------------------------------------------------- arms

# pipeline checkpoint keys (and their user-facing artifacts) an arm inherits
# from its base workspace; everything from translation on is arm-specific.
_SEED_SHARED: tuple[tuple[str, str], ...] = (
    ("video", "video.json"),
    ("cues", "cues.json"),
    ("scenes", "scenes.json"),
)
_SEED_VIEWING: tuple[tuple[str, str | None], ...] = (
    ("observations", "observations.json"),
    ("context_map", "context_map.yaml"),
    ("timeline", "timeline.yaml"),
    ("register", "register.yaml"),
    ("snapshots", "snapshots.json"),
    ("ost", None),
)
# inner narrative checkpoint stores (viewing pass, video map, timeline, register)
_SEED_INNER: tuple[str, ...] = ("s2v", "s2a", "s2b", "s2r")


def arm_video_id(video_id: str, arm: str) -> str:
    """Workspace id of an experiment arm: ``<video_id>__<arm>``."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", arm):
        raise ValueError(f"arm name must be [A-Za-z0-9_-]+, got {arm!r}")
    return f"{video_id}__{arm}"


def seed_arm(base: Workspace, arm: Workspace, *, no_viewing: bool = False) -> list[str]:
    """First-use seeding of an arm workspace from its base workspace.

    Copies the pipeline checkpoints (and artifacts) for video/cues/scenes and —
    unless ``no_viewing`` — the viewing-derived narrative stages through
    snapshots + ost plus the inner s2v/s2a/s2b stores. Translation and later
    stages are never copied. The arm's cache_dir becomes a symlink to the base
    cache_dir (when absent) so the proxy, subtitles and Gemini upload cache are
    shared. No-op once the arm has checkpoints of its own. Returns the copied
    checkpoint keys (pipeline keys and inner store names)."""
    if (arm.data_dir / "checkpoints").exists():
        return []
    if not arm.cache_dir.exists() and not arm.cache_dir.is_symlink() and base.cache_dir.is_dir():
        arm.cache_dir.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(base.cache_dir.resolve(), arm.cache_dir, target_is_directory=True)

    src_ck = base.data_dir / "checkpoints"
    if not (src_ck / "pipeline").is_dir():
        return []
    dst_ck = arm.data_dir / "checkpoints"
    entries = list(_SEED_SHARED) + ([] if no_viewing else list(_SEED_VIEWING))
    copied: list[str] = []
    for key, artifact in entries:
        src = src_ck / "pipeline" / f"{key}.json"
        if not src.is_file():
            continue
        (dst_ck / "pipeline").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_ck / "pipeline" / f"{key}.json")
        copied.append(key)
        if artifact and (base.data_dir / artifact).is_file():
            shutil.copy2(base.data_dir / artifact, arm.path(artifact))
    if not no_viewing:
        for store in _SEED_INNER:
            if (src_ck / store).is_dir():
                shutil.copytree(src_ck / store, dst_ck / store, dirs_exist_ok=True)
                copied.append(store)
    return copied


# ---------------------------------------------------------------- run


def run(
    video: Path,
    *,
    video_id: str | None = None,
    title: str | None = None,
    year: int | None = None,
    en_sub: Path | None = None,
    media_id: str | None = None,
    mode: str = "cascade",
    stop_after: str | None = None,
    arm: str | None = None,
    no_viewing: bool = False,
    perception_ledger: PerceptionLedger | None = None,
    dpaf: bool = False,
    model: str | None = None,
    architecture: str | Architecture | None = None,
    route: str | Architecture | None = None,
    experiment_id: str | None = None,
    series_id: str | None = None,
    season: int | None = None,
    episode: int | None = None,
    series_bible: series_state.SeriesBible | None = None,
    series_bible_path: Path | None = None,
) -> Path:
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    if architecture is not None and route is not None and architecture != route:
        raise ValueError("architecture and route cannot both be set to different values")
    architecture = architecture if architecture is not None else route
    if mode not in {"text", "av", "cascade", "dpaf", "mixed"}:
        raise ValueError(f"mode must be text|av|cascade|dpaf|mixed, got {mode!r}")
    if architecture is not None and mode in {"text", "av"}:
        actual_architecture = architecture.value if isinstance(architecture, Architecture) else architecture
        allowed_architectures = (
            {"text-snapshot", "semantic-image"} if mode == "text"
            else {"direct-av", "av-two-pass", "semantic-clip"}
        )
        if actual_architecture not in allowed_architectures:
            raise ValueError(
                f"mode {mode!r} conflicts with architecture {actual_architecture!r}; "
                "choose a matching route or omit --mode"
            )
    if stop_after is not None and stop_after not in STAGES:
        raise ValueError(f"stop_after must be one of {', '.join(STAGES)}")
    if perception_ledger is not None and not isinstance(perception_ledger, PerceptionLedger):
        perception_ledger = PerceptionLedger.model_validate(perception_ledger)

    load_secrets()
    settings = load_settings()
    if experiment_id is not None:
        if settings.experiment_id is not None and settings.experiment_id != experiment_id:
            raise ValueError(
                "explicit experiment_id conflicts with configured experiment_id "
                f"{settings.experiment_id!r}"
            )
        settings = replace(settings, experiment_id=experiment_id)
    if architecture is not None:
        requested_architecture = architecture
    elif model is not None and mode == "text":
        requested_architecture = Architecture.TEXT_SNAPSHOT
    elif model is not None and mode == "av":
        requested_architecture = Architecture.DIRECT_AV
    else:
        requested_architecture = mode if mode in {"dpaf", "mixed"} else Architecture.AUTO
    explicit_route = model is not None or architecture is not None or mode in {"dpaf", "mixed"}
    requested_model = model or _route_default_model(settings, requested_architecture)
    plan = resolve_route(requested_model, requested_architecture) if explicit_route else None
    effective_mode = plan.translation_mode.value if plan is not None else mode
    route_roles = _route_roles(settings, plan, effective_mode)
    route_tag = (
        _route_tag(
            plan,
            mode=effective_mode,
            legacy=plan is None,
            roles=route_roles,
            experiment_id=experiment_id,
        )
        if plan is not None or experiment_id is not None
        else ""
    )
    year = year if year is not None else _guess_year(video.stem)
    video_id = video_id or _derive_video_id(video.stem, title, year)
    if arm:
        base_ws = workspace(video_id)
        ws = workspace(arm_video_id(video_id, arm))
        seeded = seed_arm(base_ws, ws, no_viewing=no_viewing)
        if seeded:
            print(f"substage: arm '{ws.video_id}' seeded from '{video_id}': {', '.join(seeded)}")
    else:
        ws = workspace(video_id)
    route_metadata: dict[str, object] = {
        "requested_model": model,
        "requested_architecture": (
            architecture.value if isinstance(architecture, Architecture) else architecture
        ),
        "model": plan.primary_model if plan is not None else settings.model_main,
        "translation_model": route_roles["translation"],
        "av_model": route_roles["av"],
        "architecture": plan.architecture.value if plan is not None else "legacy",
        "translation_mode": effective_mode,
        "legacy": plan is None,
        "mixed": bool(plan is not None and plan.mixed_team),
        "route_label": (
            "mixed multi-model"
            if plan is not None and plan.mixed_team
            else (plan.architecture.value if plan is not None else "legacy")
        ),
        "roles": route_roles,
    }
    if experiment_id is not None:
        route_metadata["experiment_id"] = experiment_id
    dedicated_text, dedicated_av = _route_clients(ws, settings, plan)
    gc = GeminiClient(ws, settings)
    # Explicit routes own every Korean-writing text stage, including stages
    # that address model_light. Legacy routing keeps its historical clients.
    txt = dedicated_text if plan is not None else text_client(ws, gc)
    picture_client = dedicated_text if plan is not None and plan.semantic_passes else dedicated_av or (
        make_client(ws, replace(settings, model_main=route_roles["picture_check"],
                                model_light=route_roles["picture_check"]),
                    route_roles["picture_check"])
        if plan is not None else gc
    )
    _write_route_artifact(ws, route_metadata)
    ck = Checkpoints(ws, "pipeline")
    ep_coord = (
        series_state.parse_episode_coord(video_id, series_id=series_id, season=season, episode=episode)
        if (series_id is not None or series_bible is not None or season is not None or episode is not None)
        else None
    )
    effective_series_id = (
        f"{ep_coord.series_id}__{arm}" if (ep_coord is not None and arm) else (ep_coord.series_id if ep_coord else None)
    )
    bible_file = series_bible_path or (
        series_state.default_bible_path(effective_series_id) if effective_series_id else None
    )
    active_bible: series_state.SeriesBible | None = None
    if series_bible is not None:
        active_bible = series_bible.causal_slice(ep_coord.pair) if ep_coord is not None else series_bible
    elif ep_coord is not None and bible_file is not None:
        active_bible = series_state.load_bible(effective_series_id or ep_coord.series_id, before_ep=ep_coord.pair, path=bible_file)
    bible_fp = active_bible.fingerprint() if active_bible is not None else ""
    # no-viewing runs must never share narrative checkpoints with viewing runs;
    # viewing runs keep their historical hashes (no marker) so caches stay valid
    view_tag: tuple[str, ...] = ("no-viewing",) if no_viewing else ()
    if bible_fp:
        view_tag = (*view_tag, f"bible:{bible_fp}")
    # ---- S0 ingest (+ cue extraction) ------------------------------------
    video = _ingest(ws, gc, ck, video, en_sub, title, year, media_id)
    cues = _load_cues(ck, video)
    save_model(ws.path("cues.json"), cues)
    if _stopped(stop_after, "ingest", ws):
        return ws.data_dir

    # ---- S1 scenes --------------------------------------------------------
    scenes = _ck(
        ck,
        "scenes",
        input_hash(_Cues(items=cues), str(video.duration_ms), str(settings.scene_gap_ms), str(settings.max_cues_per_scene)),
        _Scenes,
        lambda: _Scenes(
            items=scenes_mod.group_scenes(cues, gap_ms=settings.scene_gap_ms, max_cues=settings.max_cues_per_scene)
        ),
    ).items
    save_model(ws.path("scenes.json"), scenes)
    if _stopped(stop_after, "scenes", ws):
        return ws.data_dir

    # ---- upload proxy ------------------------------------------------------
    proxy = _proxy_upload(ws, gc, video)
    if _stopped(stop_after, "proxy", ws):
        return ws.data_dir

    # ---- S2v viewing pass (skipped with --no-viewing: text-only narrative) ---
    if no_viewing:
        observations = []
    else:
        observations = _ck(
            ck,
            "observations",
            input_hash(
                video,
                _Cues(items=cues),
                _Scenes(items=scenes),
                narrative._prompt("viewing.md"),
                settings.model_main,
            ),
            _Observations,
            lambda: _Observations(
                items=narrative.viewing_pass(
                    ws,
                    gc,
                    video,
                    proxy,
                    cues,
                    scenes_mod.viewing_chunks(scenes, video.duration_ms, settings.viewing_chunk_ms),
                )
            ),
        ).items
    save_model(ws.path("observations.json"), observations)
    if _stopped(stop_after, "observations", ws):
        return ws.data_dir

    # ---- selective DPAF perception (explicit ledger or ``dpaf=True``) ----
    dpaf_enabled = dpaf or perception_ledger is not None or (
        plan is not None and plan.architecture in {Architecture.DPAF, Architecture.MIXED}
    )
    dpaf_requested = dpaf_enabled
    if dpaf_requested and perception_ledger is None and proxy is not None:
        perception_model = getattr(settings, "dpaf_perception_model", settings.model_main)
        perception_cue_ids = _dpaf_cue_ids(cues, scenes, observations)
        # The inner selective pass owns its prompt/window/model checkpoint.
        # Avoid a redundant outer cache that could persist blocked/degraded
        # empty records and could drift from the actual prompt template.
        perception_ledger = narrative.build_perception_ledger(
            ws,
            gc,
            video,
            proxy,
            cues,
            scenes,
            observations,
            cue_ids=perception_cue_ids,
            model=perception_model,
        )
    if perception_ledger is not None and perception_ledger.cues:
        save_model(ws.path("perception.json"), perception_ledger)

    # ---- S2a video map ------------------------------------------------------
    context_map = _editable(
        ck,
        "context_map",
        _context_map_hash(txt.settings, cues, observations, view_tag),
        ContextMap,
        ws.path("context_map.yaml"),
        lambda: narrative.build_context_map(
            ws, txt, video, cues, observations,
            **({"series_bible": active_bible} if bible_fp else {}),
        ),
    )
    if active_bible is not None and not active_bible.is_empty():
        context_map = series_state.reconcile_characters(context_map, active_bible, cues)
        save_model(ws.path("context_map.yaml"), context_map)
        ck.put("context_map", _context_map_hash(txt.settings, cues, observations, view_tag), context_map)
    if _stopped(stop_after, "context_map", ws):
        return ws.data_dir
    # ---- S2b timeline ------------------------------------------------------
    decisions_cfg = getattr(settings, "decisions_model", None)
    lean_decisions_pipeline = bool(
        decisions_cfg
        and (
            plan is None
            or (
                plan.architecture in {Architecture.DIRECT_AV, Architecture.MIXED}
                and not plan.semantic_passes
            )
        )
    )
    if lean_decisions_pipeline and stop_after != "timeline":
        timeline = Timeline(scenes=[])
    else:
        timeline = _editable(
            ck,
            "timeline",
            _timeline_hash(txt.settings, context_map, scenes, cues, observations, view_tag),
            Timeline,
            ws.path("timeline.yaml"),
            lambda: narrative.build_timeline(ws, txt, context_map, scenes, cues, observations),
        )
    if _stopped(stop_after, "timeline", ws):
        return ws.data_dir

    # ---- S2r register table ------------------------------------------------
    series_clef = (
        llm.ClefDecisionsClient(ws, settings, review.resolve_decisions_model(decisions_cfg, has_images=False))
        if decisions_cfg and active_bible is not None and not active_bible.is_empty()
        else None
    )
    register_map = _editable(
        ck,
        "register",
        _register_hash(txt.settings, context_map, cues, observations, view_tag),
        RegisterMap,
        ws.path("register.yaml"),
        lambda: register_mod.build_register(
            ws, txt, video, context_map, cues, observations,
            **({"series_bible": active_bible, "clef_client": series_clef} if bible_fp else {}),
        ),
    )
    if active_bible is not None and not active_bible.is_empty():
        register_map = series_state.lock_edges(
            register_map, active_bible, context_map=context_map, cues=cues, clef_client=series_clef
        )
        save_model(ws.path("register.yaml"), register_map)
        ck.put("register", _register_hash(txt.settings, context_map, cues, observations, view_tag), register_map)
    register_observations = list(observations)
    if perception_ledger is not None:
        register_observations.extend(perception_ledger.cues.values())
    pairs_by_cue = register_mod.cue_pairs(context_map, cues, observations)
    cues_by_id_init = {c.id: c for c in cues}
    for sc in scenes:
        pairs_by_cue.update(register_mod.lock_scene_pairs(sc.cue_ids, cues_by_id_init, pairs_by_cue, observations, context_map=context_map))
    turn_pairs_by_cue = register_mod.cue_turn_pairs(context_map, register_observations)
    for sc in scenes:
        for cid, dp in register_mod.scene_dash_turn_pairs(sc.cue_ids, cues_by_id_init, pairs_by_cue).items():
            turn_pairs_by_cue.setdefault(cid, dp)
    if _stopped(stop_after, "register", ws):
        return ws.data_dir

    # ---- S2c snapshots -----------------------------------------------------
    snapshots_list = _ck(
        ck,
        "snapshots",
        # the Snapshot field set keys the stage so a schema change re-derives old snapshots
        input_hash(context_map, timeline, register_map, _Scenes(items=scenes), _Cues(items=cues),
                   _Observations(items=observations), *view_tag,
                   "snapshot-fields:" + ",".join(sorted(Snapshot.model_fields))),
        _Snapshots,
        lambda: _Snapshots(items=list(narrative.materialize_snapshots(
            context_map, timeline, scenes, cues, register_map, pairs_by_cue).values())),
    ).items
    snapshots = {s.scene_id: s for s in snapshots_list}
    save_model(ws.path("snapshots.json"), snapshots_list)
    if _stopped(stop_after, "snapshots", ws):
        return ws.data_dir

    # ---- on-screen-text cues ------------------------------------------------
    ost = _ck(
        ck,
        "ost",
        input_hash(_Observations(items=observations), _Cues(items=cues), _Scenes(items=scenes), "ost-mmss-norm-v2", *view_tag),
        _Cues,
        lambda: _Cues(items=translate.ost_cues(observations, cues, scenes)),
    ).items
    if ost:
        if video.video_path and Path(video.video_path).is_file():
            vpath = Path(video.video_path)
            snap_ck = Checkpoints(ws, "ost_snap")
            snap_h = input_hash(_Cues(items=ost), str(vpath))
            cached_snap = snap_ck.get("snapped", snap_h, _Cues)
            if cached_snap is not None:
                ost = cached_snap.items
            else:
                snapped_ost: list[Cue] = []
                for o in ost:
                    ns, ne = media.snap_ost_shot(
                        vpath, o.start_ms, o.end_ms, frame_exact=translate._is_frame_exact_cue(o)
                    )
                    if (ns, ne) != (o.start_ms, o.end_ms):
                        o = o.model_copy(update={"start_ms": ns, "end_ms": max(ne, ns + 800)})
                    snapped_ost.append(o)
                ost = snapped_ost
                snap_ck.put("snapped", snap_h, _Cues(items=ost))
        cues, scenes = _merge_ost(cues, scenes, ost)
        save_model(ws.path("cues.json"), cues)
        save_model(ws.path("scenes.json"), scenes)
    scene_of = {cid: s.id for s in scenes for cid in s.cue_ids}
    cues_by_id = {c.id: c for c in cues}
    table = register_mod.RegisterTable(register_map, [c.id for c in sorted(cues, key=lambda c: c.start_ms)])
    gate = register_mod.RegisterGate(
        table, pairs_by_cue, turn_pairs_by_cue=turn_pairs_by_cue
    )
    report = register_mod.RegisterReport()

    def _save_report() -> None:
        report.gate_rejections = list(gate.rejections)
        save_model(ws.path("register_report.json"), report)

    if _stopped(stop_after, "ost", ws):
        return ws.data_dir

    register_check = _firm_retry(
        settings, table, pairs_by_cue, report, turn_pairs_by_cue
    )

    # DPAF roles are instantiated only when explicitly requested; legacy
    # routing remains unchanged. An explicit route always owns its translator.
    dpaf_text_client = dedicated_text
    dpaf_translation_model_id = getattr(dedicated_text, "model", "") if dedicated_text is not None else ""
    dpaf_router_model_id = ""
    dpaf_luna = None
    if dpaf_enabled:
        translation_model = (
            plan.primary_model
            if plan is not None
            else getattr(settings, "model_translate", None)
        )
        if dpaf_text_client is None and translation_model:
            dpaf_translation_model_id = translation_model
            dpaf_text_client = make_client(ws, settings, translation_model)
    if effective_mode in {"cascade", "dpaf", "mixed"} and (dpaf_enabled or plan is not None):
        router_model = getattr(settings, "dpaf_router_model", None) or settings.model_main
        dpaf_router_model_id = router_model
        dpaf_luna = make_client(ws, settings, router_model)

    scene_images: dict[str, list[Path]] = {}
    if plan is not None and plan.architecture in {
        Architecture.TEXT_SNAPSHOT, Architecture.CASCADE, Architecture.MIXED, Architecture.SEMANTIC_IMAGE,
    }:
        scene_images = {
            scene.id: media.scene_keyframes(ws, video, scene) for scene in scenes
        }
    translation_hash_parts: list[Any] = [
        video,
        _Scenes(items=scenes),
        _Cues(items=cues),
        _Snapshots(items=list(snapshots.values())),
        _Observations(items=observations),
        effective_mode,
        settings.model_main,
        # text-mode translator; absent when unset so Gemini-only keys stay valid
        *([settings.model_translate] if settings.model_translate else []),
        settings.translate_thinking,
        settings.narrative_context,
        translate._system(settings.narrative_context),
        *view_tag,
        # firm edges (S3 retry) depend on the table's features, not only its levels
        register_map,
        f"register_retry={int(settings.register_retry)}",
        f"aim_ratio={settings.aim_ratio:g}",
        json.dumps(turn_pairs_by_cue, ensure_ascii=False, sort_keys=True, default=str),
        # [] when memory is off/absent, so memory-less checkpoints stay valid
        *translate.memory_tag(settings, ws.video_id),
    ]
    if route_tag:
        translation_hash_parts.append(route_tag)
    if plan is not None:
        translation_hash_parts.extend(translate._generation_settings(dedicated_text))
        if dedicated_av is not None:
            translation_hash_parts.extend(translate._generation_settings(dedicated_av))
        if plan.semantic_passes:
            translation_hash_parts.append(translate.semantic_contract_hash())
    if scene_images:
        translation_hash_parts.append(input_hash(*[
            input_hash(scene.id, *translate.image_digests(scene_images[scene.id]))
            for scene in scenes
        ]))
    if dpaf_enabled:
        translation_hash_parts.extend(
            (
                perception_ledger.model_dump_json() if perception_ledger else "",
                dpaf_translation_model_id,
                (
                    dpaf_router_model_id or getattr(dpaf_luna, "model", "")
                    if effective_mode in {"cascade", "dpaf", "mixed"}
                    else ""
                ),
            )
        )

    translations_list = _ck(
        ck,
        "translations",
        input_hash(*translation_hash_parts),
        _Translations,
        lambda: _Translations(
            items=list(
                translate.translate_video(
                    ws, gc, video, scenes, cues, snapshots, observations, proxy, mode=effective_mode,
                    register_check=register_check,
                    perception_ledger=perception_ledger,
                    luna=dpaf_luna,
                    text_client_override=dpaf_text_client or dedicated_text,
                    av_client=dedicated_av,
                    route_tag=route_tag,
                    route_plan=plan,
                    scene_images=scene_images,
                ).values()
            )
        ),
        cache_if=(
            lambda batch: not any(translate._is_degraded(scene) for scene in batch.items)
        ) if plan is not None else None,
    ).items
    translations = {t.scene_id: t for t in translations_list}
    save_model(ws.path("translations.json"), translations_list)
    report.firm_violations = [
        v for st in translations_list
        for v in register_mod.firm_violations(
            table, pairs_by_cue, st, turn_pairs_by_cue=turn_pairs_by_cue
        )
    ]
    _save_report()
    if _stopped(stop_after, "translate", ws):
        return ws.data_dir

    # ---- S4 review -----------------------------------------------------------
    series_pins = (
        series_state.pin_cues(cues, active_bible, context_map)
        if (ep_coord is not None or (active_bible is not None and not active_bible.is_empty()))
        else {}
    )
    ko: dict[str, str] = {c.id: "" for c in cues}
    for st in translations.values():
        for tc in st.cues:
            ko[tc.cue_id] = tc.ko
    if series_pins:
        ko = series_state.enforce_series_pins(
            ko, cues, active_bible, series_pins, context_map=context_map,
            register_map=register_map, observations=observations, scenes=scenes,
        )
    ko = register_mod.enforce_pragmatic_address(
        ko, cues, register_map, context_map=context_map, observations=observations, scenes=scenes,
    )
    lineage = validate.combine_lineage(list(translations.values()))
    semantic_checked_ko = dict(ko)
    semantic_checked_fragments = {
        cid: [fragment.model_copy(deep=True) for fragment in fragments]
        for cid, fragments in lineage.fragments_by_cue.items()
    } if lineage is not None else {}
    if lineage is not None:
        txt = _SemanticContextClient(txt, lineage, semantic_trace, ws, route_metadata)
        picture_client = txt

    def semantic_stage(stage: str) -> None:
        if lineage is None:
            return
        validate.reconcile_lineage(lineage, ko)
        changed_ids = {cid for cid, text in ko.items() if semantic_checked_ko.get(cid) != text}
        for st in translations.values():
            if not isinstance(st, SemanticSceneTranslation):
                continue
            ids = set(st.lineage.fragments_by_cue)
            if not any(
                cid in changed_ids and (
                    any(f.attribution == "unverified" for f in lineage.fragments_by_cue[cid])
                    or (not ko.get(cid, "").strip() and bool(semantic_checked_fragments.get(cid)))
                )
                for cid in ids
            ):
                continue
            scene_lineage = st.lineage.model_copy(deep=True, update={
                "fragments_by_cue": {
                    cid: semantic_checked_fragments.get(cid, []) if not ko.get(cid, "").strip()
                    else lineage.fragments_by_cue[cid] for cid in ids
                },
                "omissions": [omission for omission in lineage.omissions
                              if omission.ref.partition(":")[0] in ids],
            })
            attributed = translate.reattribute_semantic_lineage(
                txt, scene_lineage, [cue for cue in cues if cue.id in ids],
                {cid: ko.get(cid, "") for cid in ids}, stage=f"{stage}:semantic-reattribute",
            )
            lineage.fragments_by_cue.update(attributed.fragments_by_cue)
            lineage.omissions = [
                omission for omission in lineage.omissions if omission.ref.partition(":")[0] not in ids
            ] + attributed.omissions
            lineage.attribution_passes.extend(attributed.attribution_passes)
            for cid in ids:
                if ko.get(cid, "").strip():
                    ko[cid] = validate.finish(
                        cues_by_id.get(cid), ko[cid], settings,
                        retained_fragments=lineage.fragments_by_cue[cid],
                    )
        semantic_checked_ko.update(ko)
        semantic_checked_fragments.update({
            cid: [fragment.model_copy(deep=True) for fragment in fragments]
            for cid, fragments in lineage.fragments_by_cue.items()
        })
        semantic_trace.append({
            "stage": stage, "kind": "output-lineage",
            "lineage": lineage.model_dump(mode="json"),
        })
        save_model(ws.path("output_lineage.json"), lineage)
        route_metadata["semantic_stages"] = semantic_trace
        _write_route_artifact(ws, route_metadata)

    semantic_stage("s3-consumer-input")

    levels = register_mod.cue_levels(
        table, pairs_by_cue, scenes=scenes, cues_by_id=cues_by_id, ko=ko, context_map=context_map
    )
    if lean_decisions_pipeline:
        review_edits = []
        edits = []
        finale_edits = []
    else:
        review_edits = list(
            _ck(
                ck,
                "review",
                input_hash(
                    _Translations(items=translations_list),
                    _Snapshots(items=list(snapshots.values())),
                    context_map,
                    register_map,
                    txt.settings.model_main,
                    *review.prompt_texts("review", settings.narrative_context),
                    *([lineage.model_dump_json(), "semantic-output-v1", _SEMANTIC_ADVISORY_SUFFIX]
                      if lineage is not None else []),
                ),
                _Edits,
                lambda: _Edits(items=review.review_windows(
                    _semantic_workspace(ws, lineage, "review"), txt, scenes, cues, translations,
                    snapshots, context_map, observations, levels=levels)),
            ).items
        )
        ko, edits = _apply_edits(ko, review_edits, gate, "review", allow_omission=lineage is not None)
        semantic_stage("s4-review")
        ko_hash = _hash_ko(ko)

        finale_raw = _ck(
            ck,
            "finale_audit",
            input_hash(ko_hash, context_map, register_map, txt.settings.model_main,
                       *review.prompt_texts("finale", settings.narrative_context),
                       *([lineage.model_dump_json(), _SEMANTIC_ADVISORY_SUFFIX] if lineage is not None else [])),
            _Edits,
            lambda: _Edits(items=review.finale_audit(
                _semantic_workspace(ws, lineage, "finale"), txt, cues, ko, context_map, observations, levels=levels)),
        ).items
        ko, finale_edits = _apply_edits(ko, list(finale_raw), gate, "finale", allow_omission=lineage is not None)
        semantic_stage("s4-finale")
    ko_hash = _hash_ko(ko)
    if decisions_cfg:
        diag_model = review.resolve_decisions_model(decisions_cfg, has_images=bool(scene_images))
        clef_client = llm.ClefDecisionsClient(ws, settings, diag_model)
        flags = _ck(
            ck,
            "blind_read",
            input_hash(
                ko_hash,
                clef_client.model,
                "clef-6axis-v1",
                register_map,
            ),
            _Flags,
            lambda: _Flags(items=review.diagnose_with_decisions(
                ws, clef_client, scenes, cues_by_id, ko, levels=levels, scene_images=scene_images,
            )),
        ).items
    else:
        flags = _ck(
            ck,
            "blind_read",
            input_hash(ko_hash, (txt if plan is not None else gc).settings.model_light,
                       *review.prompt_texts("blind", settings.narrative_context)),
            _Flags,
            lambda: _Flags(items=review.blind_read(
                ws, txt if plan is not None else gc, [(c.id, ko.get(c.id, "")) for c in cues])),
        ).items
    regen_raw = _ck(
        ck,
        "regenerate",
        input_hash(ko_hash, _Flags(items=list(flags)), register_map, txt.settings.model_main,
                   *review.prompt_texts("regen", settings.narrative_context),
                   *([lineage.model_dump_json(), _SEMANTIC_ADVISORY_SUFFIX] if lineage is not None else [])),
        _Edits,
        lambda: _Edits(items=review.regenerate_flagged(
            _semantic_workspace(ws, lineage, "regenerate"), txt, list(flags), cues_by_id,
            snapshots, scene_of, ko, observations, levels=levels)),
    ).items
    if decisions_cfg and regen_raw:
        gate_model = review.resolve_decisions_model(decisions_cfg, has_images=False)
        gate_client = llm.ClefDecisionsClient(ws, settings, gate_model)
        prot_forms = {
            cid: [*pin.required_names.values(), *pin.required_terms.values()]
            for cid, pin in series_pins.items()
        } if series_pins else None
        regen_raw = review.filter_edits_with_decisions(
            ws, gate_client, list(regen_raw), cues_by_id, ko, levels=levels,
            protected_forms=prot_forms,
        )
    ko, regen_edits = _apply_edits(ko, list(regen_raw), gate, "regen", allow_omission=lineage is not None)
    if series_pins:
        ko = series_state.enforce_series_pins(
            ko, cues, active_bible, series_pins, context_map=context_map,
            register_map=register_map, observations=observations, scenes=scenes,
        )
    semantic_stage("s4-regeneration")
    edits += finale_edits + regen_edits
    save_model(ws.path("edits.json"), edits)
    _save_report()
    if _stopped(stop_after, "review", ws):
        return ws.data_dir

    # ---- S5 validate ---------------------------------------------------------
    s5_observations = _observation_by_cue(observations, perception_ledger)
    normalized_before_s5 = {
        cue.id: (
            validate.finish(cue, ko.get(cue.id, ""), settings,
                            observation=s5_observations.get(cue.id),
                            retained_fragments=lineage.fragments_by_cue.get(cue.id) if lineage is not None else None)
            if ko.get(cue.id, "").strip() else ko.get(cue.id, "")
        )
        for cue in cues
    } if plan is not None else {}
    ko, _issues, changed = validate.run_s5(
        ws,
        txt,
        cues,
        ko,
        snapshots,
        scene_of,
        settings,
        observations_by_cue=s5_observations,
        lineage=lineage,
    )
    if series_pins:
        ko = series_state.enforce_series_pins(
            ko, cues, active_bible, series_pins, context_map=context_map,
            register_map=register_map, observations=observations, scenes=scenes,
        )
    s5_written = {
        cid for cid, normalized in normalized_before_s5.items() if ko.get(cid) != normalized
    }
    semantic_stage("s5-finish-compression")
    if _stopped(stop_after, "s5", ws):
        return ws.data_dir

    # ---- C7 picture check + revalidation -------------------------------------
    targets = (
        changed
        | {e.cue_id for e in finale_edits}
        | {e.cue_id for e in regen_edits}
        | _special_cues(context_map, timeline)
    )
    picture_kwargs = {"strict": True} if plan is not None else {}
    if plan is not None and not plan.mixed_team and not plan.av_passes:
        picture_kwargs["writer"] = txt
    if lean_decisions_pipeline:
        pic_edits = []
    elif plan is not None and plan.architecture is Architecture.SEMANTIC_IMAGE:
        pic_edits = _semantic_picture_check(
            ws, txt, scenes, scene_images, cues_by_id, ko, targets, snapshots, scene_of, gate, lineage,
        )
    else:
        pic_edits = recheck.picture_check(
            replace(ws, data_dir=ws.data_dir / "semantic-picture" / input_hash(
                lineage.model_dump_json(), _SEMANTIC_ADVISORY_SUFFIX,
            ))
            if lineage is not None else ws,
            picture_client, proxy, cues_by_id, ko, targets, snapshots, scene_of,
            gate=gate, **picture_kwargs,
        )
    edits += pic_edits
    if pic_edits:
        ko = _apply_picture_edits(ko, pic_edits, settings, {c.id: c for c in cues}, lineage=lineage)
        save_model(ws.path("edits.json"), edits)
    if series_pins:
        ko = series_state.enforce_series_pins(
            ko, cues, active_bible, series_pins, context_map=context_map,
            register_map=register_map, observations=observations, scenes=scenes,
        )
    semantic_stage("c7-picture-output")
    issues = [
        issue for cue in cues for issue in validate.check(
            cue, ko.get(cue.id, ""), settings,
            observation=s5_observations.get(cue.id) if lineage is not None else None,
            retained_fragments=lineage.fragments_by_cue.get(cue.id) if lineage is not None else None,
        )
    ]
    if lineage is not None:
        issues.extend(validate.lineage_issues(lineage))
    save_model(ws.path("issues.json"), issues)
    if _stopped(stop_after, "recheck", ws):
        return ws.data_dir

    # ---- S5c layout: re-segment into readable display events ------------------
    tc_by_id = {tc.cue_id: tc for st in translations.values() for tc in st.cues}
    speakers = {cid: tc.speaker for cid, tc in tc_by_id.items()}
    layout_result = _ck(
        ck,
        "layout",
        input_hash(
            _hash_ko(ko),
            _Cues(items=cues),
            json.dumps(scene_of, sort_keys=True),
            json.dumps(speakers, sort_keys=True),
            layout.knobs_tag(settings),
            layout.CONDENSE_PROMPT,
            style.style_guide(),
            txt.settings.model_light,
            # the register gate vetoes condensed candidates
            register_map,
            json.dumps(pairs_by_cue, sort_keys=True),
            *([lineage.model_dump_json(), layout.SEMANTIC_CONDENSE_PROMPT, _SEMANTIC_ADVISORY_SUFFIX]
              if lineage is not None else []),
        ),
        _Layout,
        lambda: _layout_result(
            cues, ko, scene_of, speakers, settings, txt, gate, routed=plan is not None, lineage=lineage,
        ),
    )
    events = layout_result.items
    if series_pins or register_map is not None:
        events = series_state.enforce_event_pins(
            events, cues, ko, active_bible, series_pins, context_map=context_map,
            register_map=register_map, observations=observations, scenes=scenes,
        )
        for ev in events:
            clean_ev = ev.ko.removeprefix("{\\an8}")
            if len(ev.cue_ids) == 1 and ev.cue_ids[0] in ko:
                ko[ev.cue_ids[0]] = clean_ev
            elif len(ev.cue_ids) == 2 and all(cid in ko for cid in ev.cue_ids):
                c0_id, c1_id = ev.cue_ids
                c0_alnum = re.sub(r"[^0-9a-zA-Z가-힣]+", "", ko[c0_id])
                ev_alnum = re.sub(r"[^0-9a-zA-Z가-힣]+", "", clean_ev)
                if c0_alnum and c0_alnum not in ev_alnum:
                    ko[c0_id] = clean_ev
                    ko[c1_id] = clean_ev
    if lineage is not None:
        semantic_trace.append({
            "stage": "s5c-layout", "kind": "actual-display-events",
            "events": [event.model_dump(mode="json") for event in events],
        })
        issues = [
            issue for issue in issues if issue.rule not in {"essential_coverage", "lineage_unverified"}
        ]
        issues.extend(validate.check_output_events(events, settings))
        issues.extend(validate.lineage_issues(lineage, events=events))
        save_model(ws.path("issues.json"), issues)
        route_metadata["semantic_stages"] = semantic_trace
        _write_route_artifact(ws, route_metadata)
    save_model(ws.path("layout.json"), events)
    _save_report()
    if _stopped(stop_after, "layout", ws):
        return ws.data_dir

    # ---- S6 emit --------------------------------------------------------------
    edit_hist: dict[str, list[CueEdit]] = {}
    for e in edits:
        edit_hist.setdefault(e.cue_id, []).append(e)
    final_writer = {cid: tc.model for cid, tc in tc_by_id.items()}
    if plan is not None:
        # S4/S5 and deterministic normalization do not expose new cue IDs.
        for edit in edits:
            final_writer[edit.cue_id] = plan.primary_model
        for cid in s5_written:
            final_writer[cid] = plan.primary_model
        for edit in pic_edits:
            final_writer[edit.cue_id] = route_roles["picture_writer"]
        for cid in layout_result.condensed_cue_ids:
            final_writer[cid] = plan.primary_model
        route_metadata["final_events"] = [
            {
                "index": index + 1,
                "cue_ids": event.cue_ids,
                "writer_models": sorted({
                    final_writer[cid] for cid in event.cue_ids if final_writer.get(cid)
                }),
                "degraded": any(
                    tc_by_id[cid].degraded for cid in event.cue_ids if cid in tc_by_id
                ),
                "degraded_cue_ids": [
                    cid for cid in event.cue_ids if cid in tc_by_id and tc_by_id[cid].degraded
                ],
                "degradation_reasons": {
                    cid: tc_by_id[cid].degradation_reason
                    for cid in event.cue_ids
                    if cid in tc_by_id and tc_by_id[cid].degraded
                },
                **({"fragments": [fragment.model_dump(mode="json") for fragment in event.fragments],
                    "omissions": [omission.model_dump(mode="json") for omission in event.omissions]}
                   if lineage is not None else {}),
            }
            for index, event in enumerate(events)
        ]
        _write_route_artifact(ws, route_metadata)
    meta = {
        cid: SidecarCue(**{
            **tc.model_dump(), "ko": ko.get(cid, tc.ko),
            "model": final_writer.get(cid), "edits": edit_hist.get(cid, []),
        })
        for cid, tc in tc_by_id.items()
    }
    srt, _sidecar = emit_mod.emit(ws, video, cues, ko, meta, events)
    if lineage is not None:
        payload = json.loads(_sidecar.read_text(encoding="utf-8"))
        payload["semantic"] = {
            "lineage": lineage.model_dump(mode="json"),
            "scenes": [st.model_dump(mode="json") for st in translations.values()
                       if isinstance(st, SemanticSceneTranslation)],
            "stages": semantic_trace,
            "issues": [issue.model_dump(mode="json") for issue in issues],
        }
        _atomic_write(_sidecar, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    if ep_coord is not None and (series_id is not None or series_bible is not None) and stop_after is None:
        committed_bible = series_state.commit_bible(
            active_bible,
            ep_coord,
            context_map,
            register_map,
            cues,
            ko,
            timeline=timeline,
            pairs_by_cue=pairs_by_cue,
            events=events,
        )
        if bible_file is not None:
            series_state.save_bible(committed_bible, path=bible_file)
        save_model(ws.path("series_state.json"), committed_bible)
    _print_cost(gc)
    return srt


def run_series(
    episodes: list[dict[str, Any]],
    *,
    series_id: str,
    series_bible_path: Path | None = None,
    **common_kwargs: Any,
) -> tuple[list[Path], series_state.SeriesBible]:
    """Execute a multi-episode series with strict causal left-fold equivalence:

        B_t = Φ(B_{t-1}, E_t),   B_0 = ∅

    Episodes are sorted by `(season, episode)` coordinate and folded in causal
    order so batch execution produces bit-for-bit identical `B_t` and Korean
    subtitles to one-by-one streaming execution.
    """
    arm = common_kwargs.get("arm")
    eff_sid = f"{series_id}__{arm}" if arm else series_id
    bible_file = series_bible_path or series_state.default_bible_path(eff_sid)

    def _coord_of(item: dict[str, Any], idx: int) -> series_state.EpisodeCoord:
        vid = item.get("video_id") or Path(item["video"]).stem
        c = series_state.parse_episode_coord(
            vid,
            series_id=series_id,
            season=item.get("season"),
            episode=item.get("episode"),
        )
        return c or series_state.EpisodeCoord(season=1, episode=idx + 1, series_id=series_id, video_id=vid)

    ordered = sorted(
        ((_coord_of(ep, i), ep) for i, ep in enumerate(episodes)),
        key=lambda pair: pair[0].pair,
    )
    bible = series_state.load_bible(eff_sid, before_ep=ordered[0][0].pair if ordered else None, path=bible_file)
    srts: list[Path] = []
    for coord, ep_spec in ordered:
        kw = {**common_kwargs, **ep_spec}
        video_path = Path(kw.pop("video"))
        kw["series_id"] = series_id
        kw["season"] = coord.season
        kw["episode"] = coord.episode
        kw["series_bible"] = bible.causal_slice(coord.pair)
        kw["series_bible_path"] = bible_file
        srt = run(video_path, **kw)
        srts.append(srt)
        bible = series_state.load_bible(eff_sid, path=bible_file)
    return srts, bible
