"""Architecture benchmarks on exact, frozen scene samples.

Explicit routes execute the production ``translate_video`` dispatcher, including
native perception, real frame inputs, AV comprehension/realization and mixed
writers. Legacy fake text models retain the lightweight scene ablation path.
Gold subtitles are scoring-only inputs. Emitted SRTs and sidecars cover the
sample, not a full-video review or a claim of quality improvement.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


from pydantic import BaseModel, field_validator

from . import capability
from . import emit, experiments, layout, media, memory, narrative, refeval, translate, validate
from .config import Settings, Workspace, load_settings, narrative_context
from .evaluate import LABELS, evaluation_metadata
from .models import (
    ChunkObservations,
    Cue,
    CueObservation,
    VideoInfo,
    ContextMap,
    RegisterMap,
    Scene,
    SceneTranslation,
    SemanticLineage,
    SemanticSceneTranslation,
    Snapshot,
    UsageRecord,
)
from .store import input_hash, load_model, load_models, save_model

TARGETED_SHARE = 0.5
WORKERS = 8  # parallel (config, run-of-consecutive-scenes) jobs
THINKING_LEVELS = ("low", "medium", "high", "xhigh")
MODES = ("auto", "text", "av", "dpaf", "cascade", "mixed")
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_FLAGS = {"on": True, "off": False}
NARRATIVE_LEVELS = ("on", "lite", "off")


# ---------------------------------------------------------------- configs


@dataclass(frozen=True)
class AblateConfig:
    name: str
    model: str
    thinking: str = "low"
    mode: str = "text"
    observations: bool = True
    narrative: str = "on"  # on (narrative_context=full) | lite (narrative_context=lite) | off (names only)
    aim: float | None = None  # Settings.aim_ratio override (cue aim = floor(max * aim)); None = settings
    memory: int | None = None  # Settings.memory_k override (0 = no translator examples); None = settings
    architecture: str = "auto"
    route: capability.RoutePlan | None = field(default=None, compare=False, repr=False)
    roles: dict[str, str] = field(default_factory=dict)

    @property
    def narrative_context(self) -> str:
        """Settings.narrative_context the config's prompts render at."""
        return "full" if self.narrative == "on" else "lite"

    @property
    def stage(self) -> str:
        return f"ablate:{self.name}"


def parse_config(spec: str) -> AblateConfig:
    """``NAME:model=...,thinking=low|medium|high,mode=text|av,architecture=auto|direct-av|text-snapshot|...,
    observations=on|off,narrative=on|lite|off,aim=<float in (0, 1]>,memory=<int >= 0>``."""
    name, sep, body = spec.partition(":")
    name = name.strip()
    if not sep or not _NAME_RE.match(name):
        raise ValueError(f"config must look like NAME:model=...,... with NAME in [A-Za-z0-9_-], got {spec!r}")
    kv: dict[str, str] = {}
    for part in body.split(","):
        if not part.strip():
            continue
        key, eq, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not eq or not value:
            raise ValueError(f"config {name}: expected key=value, got {part!r}")
        kv[key] = value
    unknown = set(kv) - {"model", "thinking", "mode", "observations", "narrative", "aim", "memory", "architecture",
                         "perception", "router", "av"}
    if unknown:
        raise ValueError(f"config {name}: unknown keys {sorted(unknown)}")
    if "model" not in kv:
        raise ValueError(f"config {name}: model= is required")
    thinking = kv.get("thinking", "low").lower()
    mode = kv.get("mode", "auto").lower()
    if thinking not in THINKING_LEVELS:
        raise ValueError(f"config {name}: thinking must be one of {THINKING_LEVELS}, got {thinking!r}")
    if mode not in MODES:
        raise ValueError(f"config {name}: mode must be one of {MODES}, got {mode!r}")
    raw = kv.get("observations", "on").lower()
    if raw not in _FLAGS:
        raise ValueError(f"config {name}: observations must be on|off, got {raw!r}")
    narrative = kv.get("narrative", "on").lower()
    if narrative not in NARRATIVE_LEVELS:
        raise ValueError(f"config {name}: narrative must be one of {NARRATIVE_LEVELS}, got {narrative!r}")
    aim: float | None = None
    if "aim" in kv:
        try:
            aim = float(kv["aim"])
        except ValueError:
            raise ValueError(f"config {name}: aim must be a number, got {kv['aim']!r}") from None
        if not 0 < aim <= 1:
            raise ValueError(f"config {name}: aim must be in (0, 1], got {aim}")
    mem: int | None = None
    if "memory" in kv:
        try:
            mem = int(kv["memory"])
        except ValueError:
            raise ValueError(f"config {name}: memory must be an integer, got {kv['memory']!r}") from None
        if mem < 0:
            raise ValueError(f"config {name}: memory must be >= 0, got {mem}")
    architecture = kv.get("architecture", "auto").lower()
    return _resolve_config_route(AblateConfig(
        name=name, model=kv["model"], thinking=thinking, mode=mode, observations=_FLAGS[raw], narrative=narrative,
        aim=aim, memory=mem, architecture=architecture,
        roles={key: kv[key] for key in ("perception", "router", "av") if key in kv},
    ))


def _resolve_config_route(cfg: AblateConfig) -> AblateConfig:
    """Resolve real model routes while leaving offline fake IDs untouched."""
    real_model = (
        cfg.mode not in {"text", "auto"}
        or cfg.model in experiments.EXPERIMENT_MODEL_POOL
        or cfg.model.startswith(("codex/", "anthropic/"))
        or cfg.architecture != "auto"
    )
    if not real_model:
        if cfg.roles:
            raise ValueError(f"config {cfg.name}: explicit roles require a supported capability route")
        return replace(cfg, mode="text", route=None)
    architecture = cfg.architecture
    mode = None if cfg.mode == "auto" else cfg.mode
    if architecture == "auto" and mode is not None:
        architecture = {"text": "text-snapshot", "av": "direct-av"}.get(mode, mode)
    try:
        route = capability.resolve_route(cfg.model, architecture, translation_mode=mode)
    except capability.CapabilityError as exc:
        raise capability.CapabilityError(
            f"config {cfg.name}: {exc}", model=exc.model, architecture=exc.architecture
        ) from exc
    if cfg.model.startswith("anthropic/"):
        raise ValueError(f"config {cfg.name}: new Anthropic translations are forbidden; use the frozen baseline")
    return replace(cfg, mode=route.translation_mode.value, route=route)


class _ConfigClient:
    """Pins model, thinking level and the ``ablate:<config>`` stage label on
    every call translate_scene (and its blocked-call fallbacks) makes; a
    caller's own thinking_level is kept.
    ``settings`` names the config's model as model_main with no
    model_translate (so translate_scene sends text calls through this client)
    and carries the config's narrative level, aim ratio and memory_k."""

    def __init__(self, inner: Any, cfg: AblateConfig, settings: Settings, *, model: str | None = None,
                 role: str = "translation", call_trace: list[dict[str, Any]] | None = None):
        self._inner = inner
        self._cfg = cfg
        self.model = model or cfg.model
        self._role = role
        self.call_trace = call_trace if call_trace is not None else []
        self.settings = replace(
            settings,
            model_main=cfg.model,
            model_light=cfg.model,
            model_translate=None,
            model_text=cfg.model,
            thinking_level=cfg.thinking,
            translate_thinking=cfg.thinking,
            narrative_context=cfg.narrative_context,
            aim_ratio=settings.aim_ratio if cfg.aim is None else cfg.aim,
            memory_k=settings.memory_k if cfg.memory is None else cfg.memory,
        )

    def generate(self, **kw: Any) -> Any:
        source_stage = kw.get("stage", "")
        stage = self._cfg.stage
        if self._cfg.route is not None:
            stage += f":{self._role}:{source_stage}"
        kw.update(stage=stage, model=self.model, thinking_level=self._cfg.thinking)
        record = {
            "stage": source_stage, "role": self._role, "model": self.model,
            "schema": kw["schema"].__name__, "clips": len(kw.get("clips") or ()),
            "images": len(kw.get("images") or ()), "effort": self._cfg.thinking,
            "status": "attempted",
        }
        self.call_trace.append(record)
        try:
            result = self._inner.generate(**kw)
        except Exception as exc:
            record.update(status="failed", error=type(exc).__name__)
            raise
        record["status"] = "completed"
        return result

    def upload(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.upload(*args, **kwargs)


# ---------------------------------------------------------------- inputs


@dataclass
class BaseInputs:
    cues: list[Cue]
    scenes: list[Scene]
    snapshots: dict[str, Snapshot]
    obs_by_scene: dict[str, list[CueObservation]]
    context_map: ContextMap | None
    register: RegisterMap | None
    video: VideoInfo | None


def load_base(ws: Workspace) -> BaseInputs:
    d = ws.data_dir
    for name in ("cues.json", "scenes.json"):
        if not (d / name).is_file():
            raise FileNotFoundError(f"base workspace has no {name}: {d / name}")
    cues = load_models(d / "cues.json", Cue)
    scenes = load_models(d / "scenes.json", Scene)
    snaps = load_models(d / "snapshots.json", Snapshot) if (d / "snapshots.json").is_file() else []
    context_map = load_model(d / "context_map.yaml", ContextMap) if (d / "context_map.yaml").is_file() else None
    if context_map is not None:
        # snapshots written before tone/rating/personas existed get them from the map
        for s in snaps:
            narrative.apply_video_facts(s, context_map)
    chunks = load_models(d / "observations.json", ChunkObservations) if (d / "observations.json").is_file() else []
    scene_of = {cid: s.id for s in scenes for cid in s.cue_ids}
    obs_by_scene: dict[str, list[CueObservation]] = {}
    for o in (o for ch in chunks for o in ch.cues):
        if o.cue_id in scene_of:
            obs_by_scene.setdefault(scene_of[o.cue_id], []).append(o)
    return BaseInputs(
        cues=cues,
        scenes=scenes,
        snapshots={s.scene_id: s for s in snaps},
        obs_by_scene=obs_by_scene,
        context_map=context_map,
        register=load_model(d / "register.yaml", RegisterMap) if (d / "register.yaml").is_file() else None,
        video=load_model(d / "video.json", VideoInfo) if (d / "video.json").is_file() else None,
    )


# ---------------------------------------------------------------- sampling


def sample_scenes(
    scenes: list[Scene],
    cues: list[Cue],
    context_map: ContextMap | None,
    register: RegisterMap | None,
    n: int,
    seed: int,
) -> list[tuple[Scene, str]]:
    """Seeded stratified sample of scenes with English dialogue: up to half
    containing a register-shift boundary or callback setup/payoff cue
    ("targeted"), the rest uniformly from the remainder ("random"). Returned in
    video order with the reason."""
    has_text = {c.id for c in cues if c.text.strip()}
    eligible = [s for s in scenes if any(cid in has_text for cid in s.cue_ids)]
    targeted = refeval.targeted_cue_ids(context_map, register)
    rng = random.Random(seed)
    hot = [s for s in eligible if targeted.intersection(s.cue_ids)]
    picked_t = rng.sample(hot, min(round(n * TARGETED_SHARE), len(hot)))
    taken = {s.id for s in picked_t}
    rest = [s for s in eligible if s.id not in taken]
    picked_r = rng.sample(rest, min(n - len(picked_t), len(rest)))
    reason = {s.id: "targeted" for s in picked_t} | {s.id: "random" for s in picked_r}
    return [(s, reason[s.id]) for s in sorted(picked_t + picked_r, key=lambda s: (s.start_ms, s.id))]


def scene_runs(scenes: list[Scene], sampled_ids: set[str]) -> list[list[Scene]]:
    """Maximal runs of sampled scenes that are consecutive in the video; a run
    is translated in order so each scene sees its predecessor's KO tail."""
    runs: list[list[Scene]] = []
    prev_sampled = False
    for s in scenes:
        if s.id in sampled_ids:
            if prev_sampled:
                runs[-1].append(s)
            else:
                runs.append([s])
        prev_sampled = s.id in sampled_ids
    return runs


# ---------------------------------------------------------------- translation


class _Entry(BaseModel):
    input_hash: str
    translation: SemanticSceneTranslation | SceneTranslation
    media_identity: dict[str, Any] | None = None
    request_spec: dict[str, Any] | None = None

    @field_validator("translation", mode="before")
    @classmethod
    def _strict_semantic_result(cls, value):
        return validate.parse_translation_result(value)


def _checkpoint(ws: Workspace, cfg: AblateConfig, scene_id: str) -> Path:
    return ws.data_dir / "ablate" / cfg.name / f"{scene_id}.json"


def _scene_snapshot(cfg: AblateConfig, base: BaseInputs, scene: Scene) -> Snapshot:
    """The base snapshot (on/lite: translate renders it at the config's
    narrative_context via Snapshot.for_prompt) or, for off, names only."""
    snap = base.snapshots.get(scene.id) or Snapshot(scene_id=scene.id)
    if cfg.narrative == "off":
        return Snapshot(scene_id=scene.id, character_names=dict(snap.character_names))
    return snap


def _media_identity(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = experiments.file_digest(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"media changed while freezing benchmark inputs: {path}")
    return {"path": str(path.resolve()), "size": after.st_size,
            "mtime_ns": after.st_mtime_ns, "sha256": digest}


def _native_request_spec(proxy: Any, scene: Scene, settings: Settings, *, semantic: bool = False) -> dict[str, Any]:
    from .gemini import Clip

    clip = Clip(
        file=proxy, start_s=max(0.0, scene.start_ms / 1000.0 - translate.CLIP_PAD_S),
        end_s=scene.end_ms / 1000.0 + translate.CLIP_PAD_S,
    )
    return {
        **({"passes": [
            {"stage": "s3_semantic_extract", "schema": "SemanticSource"},
            {"stage": "s3_semantic_write", "schema": "SemanticDraft"},
        ]} if semantic else {"schema": SceneTranslation.__name__}),
        "clip": {"uri": clip.file.uri, "mime_type": clip.file.mime_type,
                 "start_s": clip.start_s, "end_s": clip.end_s, "fps": clip.fps},
        "media_resolution": settings.media_resolution, "service_tier": settings.service_tier,
    }


def _translate_run(
    ws: Workspace,
    gc: Any,
    cfg: AblateConfig,
    base: BaseInputs,
    run: list[Scene],
    proxy: Any,
    cues_by_id: dict[str, Cue],
    media_identity: dict[str, Any] | None = None,
) -> dict[str, SceneTranslation]:
    narrative = narrative_context(gc.settings)
    system = translate._system(narrative)
    out: dict[str, SceneTranslation] = {}
    exclude_video = memory.base_video_id(ws.video_id)
    prev: SceneTranslation | None = None
    for scene in run:
        snap = _scene_snapshot(cfg, base, scene)
        obs = base.obs_by_scene.get(scene.id, []) if cfg.observations else []
        tail = translate._tail(prev)
        scene_cues = [cues_by_id[cid] for cid in scene.cue_ids if cid in cues_by_id]
        view = snap.for_prompt(narrative)
        examples = translate.scene_examples(gc.settings, scene_cues, exclude_video)
        prompt = translate._scene_prompt(
            scene, scene_cues, view, obs, tail, mode=cfg.mode, callback_ko={}, aim_ratio=gc.settings.aim_ratio,
            examples=examples,
        )
        h = input_hash(
            cfg.model, cfg.thinking, cfg.mode, narrative, system, prompt, *[f"memory:{p.id}" for p in examples]
        )
        path = _checkpoint(ws, cfg, scene.id)
        request_spec = _native_request_spec(proxy, scene, gc.settings) if cfg.route is not None else None
        if cfg.route is not None:
            # Retain undigested historical outputs as diagnostics, never overwrite
            # or use them as proof that the current media is identical.
            path = path.with_suffix(".strict.json")
            h = input_hash(h, json.dumps(media_identity, sort_keys=True), json.dumps(request_spec, sort_keys=True))
        st: SceneTranslation | None = None
        if path.is_file():
            try:
                entry = load_model(path, _Entry)
            except Exception:  # noqa: BLE001 — corrupt checkpoint is a miss
                entry = None
            if (
                entry is not None and entry.input_hash == h and entry.media_identity == media_identity
                and entry.request_spec == request_spec
                and not (cfg.route is not None and entry.translation.degraded)
            ):
                st = entry.translation
        if st is None:
            route_kwargs = {}
            if cfg.route is not None:
                route_kwargs = {"route_plan": cfg.route, "text_client_override": gc}
            st = translate.translate_scene(
                ws, gc, scene, cues_by_id, snap, obs, tail, mode=cfg.mode, proxy=proxy, examples=examples,
                **route_kwargs,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path = path.with_suffix(".degraded.json") if cfg.route is not None and st.degraded else path
            save_model(checkpoint_path, _Entry(
                input_hash=h, translation=st, media_identity=media_identity, request_spec=request_spec,
            ))
        if cfg.route is not None:
            # Add primary-writer identity without changing the runtime's media
            # degradation status or its reason.
            st = st.model_copy(update={
                "model": cfg.model,
                "cues": [cue.model_copy(update={"model": cfg.model}) for cue in st.cues],
            })
        out[scene.id] = st
        prev = st
    return out


def _execution_roles(cfg: AblateConfig, settings: Settings) -> dict[str, str]:
    """Validate roles before constructing clients; never accept metadata-only roles."""
    requested = cfg.roles
    unknown = set(requested) - {"translator", "translation", "perception", "router", "av"}
    if unknown:
        raise ValueError(f"config {cfg.name}: unsupported execution roles {sorted(unknown)}")
    for key in ("translator", "translation"):
        if key in requested and requested[key] != cfg.model:
            raise ValueError(f"config {cfg.name}: {key} must equal primary model {cfg.model}")
    roles = {"translation": cfg.model, "fallback": cfg.model}
    if cfg.route is None:
        if requested:
            raise ValueError(f"config {cfg.name}: roles require an explicit supported architecture")
        return roles
    arch = cfg.route.architecture
    if arch.value in {"semantic-clip", "semantic-image"}:
        if set(requested) - {"translator", "translation", "av"}:
            raise ValueError(f"config {cfg.name}: semantic passes do not execute auxiliary roles")
        if requested.get("av", cfg.model) != cfg.model:
            raise ValueError(f"config {cfg.name}: both semantic passes must use the primary model")
        roles.update(semantic_extract=cfg.model, semantic_write=cfg.model)
        if arch.value == "semantic-clip":
            roles["av"] = cfg.model
        elif "av" in requested:
            raise ValueError(f"config {cfg.name}: semantic-image does not execute an AV role")
        return roles
    native = {capability.Architecture.DIRECT_AV, capability.Architecture.AV_TWO_PASS}
    hybrid = {capability.Architecture.DPAF, capability.Architecture.CASCADE, capability.Architecture.MIXED}
    if arch in native:
        if set(requested) - {"translator", "translation", "av"}:
            raise ValueError(f"config {cfg.name}: {arch.value} does not execute perception/router roles")
        if requested.get("av", cfg.model) != cfg.model:
            raise ValueError(f"config {cfg.name}: native AV passes must use the primary model")
        roles["av"] = cfg.model
        if arch is capability.Architecture.AV_TWO_PASS:
            roles["av_pass2"] = cfg.model
    elif arch in hybrid:
        perception = requested.get("perception", requested.get("av", settings.dpaf_perception_model))
        if requested.get("av", perception) != perception:
            raise ValueError(f"config {cfg.name}: production shares perception and AV client; identities must match")
        if perception not in experiments.EXPERIMENT_MODEL_POOL:
            raise ValueError(f"config {cfg.name}: perception must be in the fixed model pool")
        capability.resolve_route(perception, "direct-av")
        router = requested.get("router", settings.dpaf_router_model)
        if router not in experiments.EXPERIMENT_MODEL_POOL:
            raise ValueError(f"config {cfg.name}: router must be in the fixed model pool")
        roles.update(av=perception, router=router)
        if arch in {capability.Architecture.DPAF, capability.Architecture.MIXED}:
            roles["perception"] = perception
    elif set(requested) - {"translator", "translation"}:
        raise ValueError(f"config {cfg.name}: snapshot route does not execute auxiliary roles")
    return roles


def _translate_architecture(
    ws: Workspace, settings: Settings, cfg: AblateConfig, base: BaseInputs,
    sampled: list[Scene], client_factory: Callable[[Workspace, Settings, str], Any],
    call_trace: list[dict[str, Any]],
) -> dict[str, SceneTranslation]:
    from .pipeline import _dpaf_cue_ids, _route_tag

    roles = _execution_roles(cfg, settings)
    route_settings = replace(
        settings, model_main=cfg.model, model_translate=cfg.model, model_text=cfg.model,
        model_light=cfg.model, thinking_level=cfg.thinking, translate_thinking=cfg.thinking,
        narrative_context=cfg.narrative_context,
        aim_ratio=settings.aim_ratio if cfg.aim is None else cfg.aim,
        memory_k=settings.memory_k if cfg.memory is None else cfg.memory,
        dpaf_perception_model=roles.get("av", settings.dpaf_perception_model),
        dpaf_router_model=roles.get("router", settings.dpaf_router_model),
    )
    arm = Workspace(
        video_id=ws.video_id, data_dir=ws.data_dir / "ablate" / cfg.name,
        cache_dir=ws.cache_dir / "ablate" / cfg.name, log_dir=ws.log_dir,
    )

    def client(model: str, role: str) -> _ConfigClient:
        return _ConfigClient(client_factory(arm, route_settings, model), cfg, route_settings,
                             model=model, role=role, call_trace=call_trace)

    selected_media = base.video.proxy_path if base.video.proxy_path is not None else base.video.video_path
    identity = _media_identity(Path(selected_media))
    arm.data_dir.mkdir(parents=True, exist_ok=True)
    (arm.data_dir / "benchmark_inputs.json").write_text(json.dumps(
        {"media_identity": identity, "route": cfg.route.to_dict(), "roles": roles,
         "effort": cfg.thinking, "scenes": [scene.model_dump() for scene in sampled]},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")
    primary = client(cfg.model, "translation")
    av = client(roles["av"], "av") if "av" in roles else None
    router = client(roles["router"], "router") if "router" in roles else None
    proxy = av.upload(Path(base.video.proxy_path), "video/mp4") if av is not None else None
    if cfg.route.av_passes:
        inputs_path = arm.data_dir / "benchmark_inputs.json"
        inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        inputs["request_specs"] = {
            scene.id: _native_request_spec(proxy, scene, route_settings, semantic=bool(cfg.route.semantic_passes))
            for scene in sampled
        }
        inputs_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding="utf-8")
    scene_images = {}
    if cfg.route.architecture.value == "semantic-image":
        scene_images = {scene.id: media.scene_keyframes(arm, base.video, scene) for scene in sampled}
        inputs_path = arm.data_dir / "benchmark_inputs.json"
        inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        inputs["request_specs"] = {
            scene.id: {
                "passes": [
                    {"stage": "s3_semantic_extract", "schema": "SemanticSource"},
                    {"stage": "s3_semantic_write", "schema": "SemanticDraft"},
                ],
                "images": translate.image_digests(scene_images[scene.id]),
            }
            for scene in sampled
        }
        inputs_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding="utf-8")
    runs = scene_runs(base.scenes, {scene.id for scene in sampled})
    if cfg.route.architecture is capability.Architecture.DIRECT_AV:
        direct_results: dict[str, SceneTranslation] = {}
        if len(runs) > 1:
            with ThreadPoolExecutor(max_workers=min(translate.ACT_WORKERS, len(runs))) as ex:
                futures = [
                    ex.submit(
                        _translate_run, ws, primary, cfg, base, run, proxy,
                        {cue.id: cue for cue in base.cues}, identity,
                    )
                    for run in runs
                ]
                for future in futures:
                    direct_results.update(future.result())
        else:
            for run in runs:
                direct_results.update(_translate_run(
                    ws, primary, cfg, base, run, proxy,
                    {cue.id: cue for cue in base.cues}, identity,
                ))
        return direct_results
    cue_ids = {cid for scene in sampled for cid in scene.cue_ids}
    cues = [cue for cue in base.cues if cue.id in cue_ids]
    observations = [o for scene in sampled for o in base.obs_by_scene.get(scene.id, [])] if cfg.observations else []
    perception = None
    if "perception" in roles:
        perception_identity = input_hash(json.dumps(identity, sort_keys=True), cfg.thinking, roles["perception"])
        perception_ws = replace(arm, data_dir=arm.data_dir / "perception-inputs" / perception_identity)
        perception_client = _ConfigClient(av._inner, cfg, route_settings,
                                          model=roles["perception"], role="perception", call_trace=call_trace)
        perception = narrative.build_perception_ledger(
            perception_ws, perception_client, base.video, proxy, cues, sampled, observations,
            cue_ids=_dpaf_cue_ids(cues, sampled, observations), model=roles["perception"],
        )
    snapshots = {scene.id: _scene_snapshot(cfg, base, scene) for scene in sampled}
    tag = _route_tag(plan=cfg.route, mode=cfg.mode, legacy=False,
                     roles=roles, experiment_id=route_settings.experiment_id)
    tag += json.dumps({"media_identity": identity}, sort_keys=True)
    results: dict[str, SceneTranslation] = {}
    if len(runs) > 1:
        with ThreadPoolExecutor(max_workers=min(translate.ACT_WORKERS, len(runs))) as ex:
            futures = [
                ex.submit(
                    translate.translate_video,
                    arm, primary, base.video, run, cues, snapshots, observations, proxy,
                    mode=cfg.mode, perception_ledger=perception, luna=router,
                    text_client_override=primary, av_client=av, route_tag=tag, route_plan=cfg.route,
                    scene_images=scene_images,
                )
                for run in runs
            ]
            for future in futures:
                results.update(future.result())
    else:
        for run in runs:
            results.update(translate.translate_video(
                arm, primary, base.video, run, cues, snapshots, observations, proxy,
                mode=cfg.mode, perception_ledger=perception, luna=router,
                text_client_override=primary, av_client=av, route_tag=tag, route_plan=cfg.route,
                scene_images=scene_images,
            ))
    return results


def translate_configs(
    ws: Workspace,
    settings: Settings,
    base: BaseInputs,
    sampled: list[Scene],
    configs: list[AblateConfig],
    client_factory: Callable[[Workspace, Settings, str], Any],
) -> dict[str, dict[str, SceneTranslation]]:
    """config name -> scene id -> translation for every sampled scene."""
    cues_by_id = {c.id: c for c in base.cues}
    runs = scene_runs(base.scenes, {s.id for s in sampled})
    # Validate every config and prerequisite before even creating a real client.
    for cfg in configs:
        _execution_roles(cfg, settings)
        if cfg.route is not None:
            if base.video is None:
                raise RuntimeError(f"config {cfg.name}: architecture benchmarks require video.json")
            if cfg.mode != "text" and not base.video.proxy_path:
                raise RuntimeError(f"config {cfg.name}: {cfg.mode} needs video.json proxy_path")
    results: dict[str, dict[str, SceneTranslation]] = {cfg.name: {} for cfg in configs}
    call_traces: dict[str, list[dict[str, Any]]] = {cfg.name: [] for cfg in configs}
    with ThreadPoolExecutor(max_workers=max(1, min(WORKERS, len(configs)))) as ex:
        futs = []
        for cfg in configs:
            if cfg.route is not None:
                futs.append((cfg, ex.submit(_translate_architecture, ws, settings, cfg, base, sampled,
                                           client_factory, call_traces[cfg.name])))
            else:
                gc = _ConfigClient(client_factory(ws, settings, cfg.model), cfg, settings,
                                   call_trace=call_traces[cfg.name])
                for run in runs:
                    futs.append((cfg, ex.submit(_translate_run, ws, gc, cfg, base, run, None, cues_by_id)))
        for cfg, fut in futs:
            results[cfg.name].update(fut.result())
    for cfg in configs:
        inputs_path = ws.data_dir / "ablate" / cfg.name / "benchmark_inputs.json"
        inputs_path.parent.mkdir(parents=True, exist_ok=True)
        inputs = json.loads(inputs_path.read_text(encoding="utf-8")) if inputs_path.is_file() else {}
        inputs["stage_scope"] = _benchmark_stage_scope(cfg, call_traces[cfg.name])
        inputs_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def normalize(st: SceneTranslation, cues_by_id: dict[str, Cue], settings) -> dict[str, str]:
    """Deterministic S5 pass (validate.finish, as run_s5 without compression)."""
    out: dict[str, str] = {}
    for t in st.cues:
        cue = cues_by_id.get(t.cue_id)
        if cue is None or not t.ko.strip():
            continue
        fragments = st.lineage.fragments_by_cue.get(t.cue_id, []) if isinstance(st, SemanticSceneTranslation) else None
        out[t.cue_id] = validate.finish(cue, t.ko, settings, retained_fragments=fragments)
    return out


# ---------------------------------------------------------------- scoring


def restrict_official(official: list[refeval.Line], scenes: list[Scene]) -> list[refeval.Line]:
    """Official lines whose midpoint lies inside a sampled scene's span."""
    spans = [(s.start_ms, s.end_ms) for s in scenes]
    kept = [ln for ln in official if any(a <= (ln.start_ms + ln.end_ms) / 2 <= b for a, b in spans)]
    return [refeval.Line(i, ln.start_ms, ln.end_ms, ln.text) for i, ln in enumerate(kept)]


def candidate_lines(ko: dict[str, str], cues_by_id: dict[str, Cue]) -> list[refeval.Line]:
    rows = sorted(
        ((cues_by_id[cid].start_ms, cues_by_id[cid].end_ms, text) for cid, text in ko.items() if text.strip()),
    )
    return [refeval.Line(i, s, e, t) for i, (s, e, t) in enumerate(rows)]


def usage_by_stage(ws: Workspace) -> dict[str, dict[str, float]]:
    per: dict[str, dict[str, float]] = {}
    log = ws.log_dir / "usage.jsonl"
    if not log.is_file():
        return per
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = UsageRecord.model_validate_json(line)
        d = per.setdefault(r.stage, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        d["calls"] += 1
        d["input_tokens"] += r.input_tokens
        d["output_tokens"] += r.output_tokens
        d["cost_usd"] += r.cost_usd
    return per


def _config_usage(usage: experiments.UsageDelta, cfg: AblateConfig) -> experiments.UsageDelta:
    return experiments.select_usage(usage, cfg.stage)


def _benchmark_stage_scope(cfg: AblateConfig, call_trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Record actual provider substeps separately from available route structure."""
    return {
        "pipeline_scope": "scene-translation-and-deterministic-output",
        "translation_execution": "generated-or-checkpoint-reused",
        "configured_av_passes": cfg.route.av_passes if cfg.route else 0,
        "executed_stages": (
            ["s3-semantic-extraction", "s3-semantic-writing"]
            if cfg.route is not None and cfg.route.semantic_passes else ["s3-scene-translation"]
        ),
        "omitted_stages": [
            "s4-review", "s4-finale", "s4-blind-read", "s4-regeneration",
            "s5-model-compression", "c7-picture-check-and-rewrite", "s5c-llm-condensation",
        ],
        "current_run_provider_calls": call_trace,
        "current_run_completed_substeps": [
            {
                **call,
                "substep": (
                    "semantic-extraction" if call["stage"] == "s3_semantic_extract"
                    else "semantic-writing" if call["stage"] == "s3_semantic_write"
                    else "native-perception" if call["schema"] == "PerceptionLedger"
                    else "consistency-router" if call["role"] == "router"
                    else "av-realization" if call["clips"]
                    else "snapshot-realization" if call["images"]
                    else "text-realization"
                ),
            }
            for call in call_trace if call["status"] == "completed"
        ],
        "checkpoint_reuse_note": "An empty current-run trace does not mean provider substeps ran again.",
    }


class _SampleSidecar(emit.Sidecar):
    """Sample output also preserves scene-level media degradation provenance."""

    scenes: list[SemanticSceneTranslation | SceneTranslation]
    lineage: SemanticLineage | None = None

    @field_validator("scenes", mode="before")
    @classmethod
    def _strict_semantic_results(cls, values):
        return [validate.parse_translation_result(value) for value in values]


def _write_sample_artifacts(
    out_dir: Path, cfg: AblateConfig, sampled: list[Scene], cues_by_id: dict[str, Cue],
    translations: dict[str, SceneTranslation], ko: dict[str, str], settings: Settings,
    *, full_source: bool,
) -> dict[str, Any]:
    from .subs import write_srt

    cues = [cues_by_id[cid] for scene in sampled for cid in scene.cue_ids if cid in cues_by_id]
    metadata = {cue.cue_id: cue for st in translations.values() for cue in st.cues}
    lineage = validate.combine_lineage(list(translations.values()))
    events = layout.layout(
        cues, ko, {cid: scene.id for scene in sampled for cid in scene.cue_ids},
        {cid: cue.speaker for cid, cue in metadata.items()}, settings, lineage=lineage,
    )
    directory = out_dir / cfg.name
    directory.mkdir(parents=True, exist_ok=True)
    srt_path = directory / "sample.ko.srt"
    sidecar_path = directory / "sample.ko.json"
    write_srt([(event.start_ms, event.end_ms, event.ko) for event in events], srt_path)
    save_model(sidecar_path, _SampleSidecar(
        events=events, scenes=list(translations.values()), lineage=lineage,
        cues=[emit.SidecarCue(**cue.model_copy(update={"ko": ko.get(cid, cue.ko)}).model_dump())
              for cid, cue in metadata.items()],
    ))
    rows = [refeval.Line(i, event.start_ms, event.end_ms, event.ko) for i, event in enumerate(events)]
    return {
        "srt": str(srt_path), "sidecar": str(sidecar_path),
        "scope": "all-source-scenes-output" if full_source else "sampled-scenes-output",
        "full_video_review": False,
        "layout": layout.stats(events, settings),
        "display_rates": refeval.display_rates(rows, settings),
        "scenes": {sid: st.model_dump(mode="json") for sid, st in translations.items()},
        "output_events": [event.model_dump(mode="json") for event in events],
        "lineage": lineage.model_dump(mode="json") if lineage is not None else None,
        "lineage_issues": [
            issue.model_dump(mode="json") for issue in validate.lineage_issues(lineage, events=events)
        ]
        if lineage is not None else [],
    }


def run_ablation(
    ws: Workspace,
    configs: list[AblateConfig],
    ref_path: Path | str,
    n_scenes: int,
    seed: int,
    *,
    judge: bool = False,
    out_dir: Path | str | None = None,
    settings: Settings | None = None,
    client_factory: Callable[[Workspace, Settings, str], Any] | None = None,
    judge_models: list[str] | None = None,
    judge_factory: refeval.JudgeFactory | None = None,
    evaluation_mode: str = "listwise",
    opus_ref: Path | str | None = None,
    split: str | None = None,
    cycle_id: str | None = None,
    sample_from: Path | str | None = None,
    experiment_id: str | None = None,
    declared_architecture: str = "legacy-ablation",
    effort: str | None = None,
    roles: dict[str, str] | None = None,
    cycle: experiments.CycleState | dict[str, Any] | None = None,
    budget_status: dict[str, Any] | None = None,
) -> dict:
    """Sample, translate per config, normalize, score; with ``judge``, every
    model in ``judge_models`` (default ``settings.model_main``) judges and is
    calibrated on the same windows; write ablate.json/.md."""
    metadata = evaluation_metadata(evaluation_mode, split=split, cycle_id=cycle_id)
    if evaluation_mode == "paired" and opus_ref is None:
        raise ValueError("paired evaluation requires a frozen opus_ref subtitle artifact")
    if evaluation_mode != "paired" and opus_ref is not None:
        raise ValueError("opus_ref requires evaluation_mode='paired'")
    frozen_opus = refeval.load_ko(Path(opus_ref)) if opus_ref is not None else None
    if not configs:
        raise ValueError("at least one --config is required")
    # Global options are execution settings, not manifest labels.
    if effort is not None and effort not in THINKING_LEVELS:
        raise ValueError(f"effort must be one of {THINKING_LEVELS}")
    if declared_architecture != "legacy-ablation":
        for cfg in configs:
            if cfg.architecture not in {"auto", declared_architecture}:
                raise ValueError(f"config {cfg.name}: global and per-config architectures conflict")
        configs = [replace(cfg, architecture=declared_architecture, mode="auto", route=None) for cfg in configs]
    configs = [_resolve_config_route(replace(
        cfg, thinking=effort or cfg.thinking, roles={**cfg.roles, **(roles or {})},
    )) for cfg in configs]
    names = [c.name for c in configs]
    if len(set(names)) != len(names):
        raise ValueError(f"config names must be unique, got {names}")
    if judge and evaluation_mode == "listwise" and len(configs) > len(LABELS):
        raise ValueError(f"--judge supports at most {len(LABELS)} configs per call, got {len(configs)}")
    settings = settings or load_settings()
    if experiment_id is not None:
        configured_id = getattr(settings, "experiment_id", None)
        if configured_id is not None and configured_id != experiment_id:
            raise ValueError("experiment_id conflicts with settings.experiment_id")
        settings = replace(settings, experiment_id=experiment_id)
        if budget_status is not None:
            budget_id = budget_status.get("experiment_id")
            if budget_id is not None and budget_id != experiment_id:
                raise ValueError("budget_status experiment_id conflicts with experiment_id")
            budget_status = {**budget_status, "experiment_id": experiment_id}
    if client_factory is None:
        from .llm import make_client

        client_factory = make_client
    default_out = ws.data_dir / "ablate"
    out_was_default = out_dir is None
    out_dir = Path(out_dir) if out_dir else default_out
    base = load_base(ws)
    usage_log = ws.log_dir / "usage.jsonl"
    usage_start = experiments.usage_cursor(usage_log)
    if sample_from is not None:
        source = Path(sample_from)
        if source.is_dir():
            source = source / "ablate.json"
        if source.resolve() == (out_dir / "ablate.json").resolve():
            if out_was_default:
                out_dir = source.parent / "rerun"
            else:
                raise ValueError("sample-from report must not be overwritten by this run")
        frozen_report, _ = experiments.frozen_sample(source)
        picked = list(experiments.validate_frozen_sample(
            frozen_report, ws.video_id, base.scenes,
            input_paths={"cues": ws.data_dir / "cues.json", "scenes": ws.data_dir / "scenes.json"},
        ))
    else:
        picked = sample_scenes(base.scenes, base.cues, base.context_map, base.register, n_scenes, seed)
    if experiment_id is not None:
        if cycle_id is None:
            raise ValueError("experiment_id requires cycle_id for an experiment manifest")
        invalid_models = {cfg.model for cfg in configs if cfg.model not in experiments.EXPERIMENT_MODEL_POOL}
        if invalid_models:
            raise ValueError(f"experiment model(s) outside the fixed pool: {sorted(invalid_models)}")
        if roles:
            invalid_roles = {model for model in roles.values() if model not in experiments.EXPERIMENT_MODEL_POOL}
            if invalid_roles:
                raise ValueError(f"experiment role model(s) outside the fixed pool: {sorted(invalid_roles)}")
    sampled = [s for s, _ in picked]
    cues_by_id = {c.id: c for c in base.cues}
    translations = translate_configs(ws, settings, base, sampled, configs, client_factory)

    ko_by_cfg: dict[str, dict[str, str]] = {}
    for cfg in configs:
        ko: dict[str, str] = {}
        for s in sampled:
            ko.update(normalize(translations[cfg.name][s.id], cues_by_id, settings))
        ko_by_cfg[cfg.name] = ko
    artifacts = {
        cfg.name: _write_sample_artifacts(
            out_dir, cfg, sampled, cues_by_id, translations[cfg.name], ko_by_cfg[cfg.name], settings,
            full_source={scene.id for scene in sampled} == {scene.id for scene in base.scenes},
        )
        for cfg in configs
    }

    en = [cues_by_id[cid] for s in sampled for cid in s.cue_ids if cid in cues_by_id and cues_by_id[cid].text.strip()]
    official = restrict_official(refeval.load_ko(Path(ref_path)), sampled)
    cands = {
        name: [
            refeval.Line(i, event["start_ms"], event["end_ms"], event["ko"])
            for i, event in enumerate(artifacts[name]["output_events"])
        ] if any(isinstance(st, SemanticSceneTranslation) for st in translations[name].values())
        else candidate_lines(ko_by_cfg[name], cues_by_id)
        for name in names
    }
    paired_result = None
    if evaluation_mode == "paired":
        jws = Workspace(video_id=ws.video_id, data_dir=out_dir, cache_dir=ws.cache_dir, log_dir=ws.log_dir)
        paired_result = refeval.evaluate_paired(
            jws, settings, en, official, cands, restrict_official(frozen_opus, sampled),
            scenes=sampled, seed=seed, judge=judge, judge_models=judge_models,
            judge_factory=judge_factory, split=split or "dev", cycle_id=cycle_id,
        )
        groups = []
        metrics = paired_result["metrics"]
    else:
        groups = refeval.align(en, official, cands)
        refeval.assign_scenes(groups, sampled)
        metrics = {name: refeval.arm_metrics(groups, name, settings) for name in names}

    usage_delta = experiments.usage_delta(usage_log, usage_start)
    input_digest = input_hash(
        ws.video_id,
        (ws.data_dir / "cues.json").read_bytes(),
        (ws.data_dir / "scenes.json").read_bytes(),
    )
    opus_digest = input_hash(Path(opus_ref).read_bytes()) if opus_ref is not None else None
    report: dict = {
        "video_id": ws.video_id,
        "ref": str(ref_path),
        "seed": seed,
        **({k: v for k, v in metadata.items() if v is not None} if split is not None or cycle_id is not None else {}),
        "n_scenes": len(sampled),
        "input_digest": input_digest,
        "provenance": {
            name: {"path": str(ws.data_dir / f"{name}.json"),
                   "sha256": experiments.file_digest(ws.data_dir / f"{name}.json")}
            for name in ("cues", "scenes")
        },
        "sample_from": str(sample_from) if sample_from is not None else None,
        # aim / memory: the effective cue aim ratio and memory_k (the config's
        # override or the Settings value)
        "configs": {
            cfg.name: asdict(cfg) | {
                "architecture": cfg.route.architecture.value if cfg.route is not None else cfg.architecture,
                "aim": settings.aim_ratio if cfg.aim is None else cfg.aim,
                "memory": settings.memory_k if cfg.memory is None else cfg.memory,
                "route": cfg.route.to_dict() if cfg.route is not None else None,
                "roles": _execution_roles(cfg, settings),
                "frozen_inputs": (
                    json.loads((ws.data_dir / "ablate" / cfg.name / "benchmark_inputs.json").read_text(encoding="utf-8"))
                    if cfg.route is not None else None
                ),
                "previous_undigested_results": {
                    scene.id: {"path": str(_checkpoint(ws, cfg, scene.id)), "eligible_for_strict_reuse": False,
                               "reason": "historical checkpoint lacks producer-frozen media identity"}
                    for scene in sampled
                    if cfg.route is not None and _checkpoint(ws, cfg, scene.id).is_file()
                },
            }
            for cfg in configs
        },
        "scenes": [
            {
                "id": s.id,
                "reason": reason,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "cue_ids": list(s.cue_ids),
                "cues": len(s.cue_ids),
            }
            for s, reason in picked
        ],
        "alignment": {
            "groups": len(groups),
            "official_lines": len(official),
            "en_cues": len(en),
            "official_rule": "official lines whose midpoint lies inside a sampled scene",
        },
        "metrics": metrics,
        "artifacts": artifacts,
        "review_scope": "sampled-scenes",
        "full_video_review": False,
        "official": refeval.display_rates(official, settings),
        "usage": usage_delta.model_dump(mode="json"),
        "cost": {
            cfg.name: experiments.usage_totals(_config_usage(usage_delta, cfg)).model_dump(mode="json")
            for cfg in configs
        },
        "judge": None,
    }
    if experiment_id is not None:
        manifest_cycle = experiments.CycleState.model_validate(cycle or {"phase": "decision"})
        sampled_records = tuple(
            experiments.SampledScene(scene_id=s.id, start_ms=s.start_ms, end_ms=s.end_ms,
                                     cue_ids=tuple(s.cue_ids), reason=reason)
            for s, reason in picked
        )
        manifests: dict[str, dict[str, Any]] = {}
        for cfg in configs:
            manifest = experiments.ExperimentManifest(
                experiment_id=experiment_id,
                cycle_id=cycle_id,
                declared_architecture=(
                    cfg.route.architecture.value if cfg.route is not None else declared_architecture
                ),
                translator=cfg.model,
                model=cfg.model,
                effort=cfg.thinking,
                mode=cfg.mode,
                roles=_execution_roles(cfg, settings),
                input_digest=input_digest,
                frozen_opus_digest=opus_digest,
                sampled_scenes=sampled_records,
                usage=_config_usage(usage_delta, cfg),
                budget_status=budget_status,
                cycle=manifest_cycle,
                heldout=experiments.HeldoutMetadata(split=split, label=split, unread_new_data=False),
            ).model_dump(mode="json")
            if cfg.route is not None:
                manifest["route"] = cfg.route.to_dict()
            manifests[cfg.name] = manifest
        report["manifests"] = manifests
        if len(manifests) == 1:
            report["manifest"] = next(iter(manifests.values()))
    if judge and evaluation_mode == "listwise":
        models = judge_models or [settings.model_main]
        jws = Workspace(video_id=ws.video_id, data_dir=out_dir, cache_dir=ws.cache_dir, log_dir=ws.log_dir)
        windows = refeval.build_windows(groups)
        results = refeval.run_judges(
            jws, settings, models, windows, names, official, refeval.name_swaps(base.context_map),
            seed=seed, factory=judge_factory,
        )
        for name, s in refeval.judge_summaries(results, names).items():
            metrics[name].update(s)
        report["judge"] = {
            "models": models,
            "calibration": {m: r["calibration"] for m, r in results.items()},
            "windows": [
                {
                    "id": w.id,
                    "scene_ids": list(dict.fromkeys(g.scene_id for g in w.groups)),
                    "start_ms": w.start_ms,
                    "end_ms": w.end_ms,
                    "verdicts": refeval.merge_verdicts({m: r["judged"][i]["verdicts"] for m, r in results.items()}),
                }
                for i, w in enumerate(windows)
            ],
        }
    report["groups"] = [refeval.group_record(g, names) for g in groups]
    if paired_result is not None:
        report.update(paired_result)
        report["opus_ref"] = str(opus_ref)
        report["opus_artifact_hash"] = input_hash(Path(opus_ref).read_bytes())
        report["alignment"].update({
            "scope": "independent_candidate_pairs",
            "groups": {name: pair["alignment"]["groups"] for name, pair in paired_result["paired"].items()},
        })
        report["groups"] = [
            {**row, "comparison_arm": name}
            for name, pair in paired_result["paired"].items() for row in pair["groups"]
        ]
        if judge:
            report["judge"] = {
                "models": list(dict.fromkeys(judge_models or [settings.model_main])),
                "calibration": paired_result["calibration"],
                "windows": [
                    {**row, "comparison_arm": name}
                    for name, pair in paired_result["paired"].items() for row in pair["windows"]
                ],
            }
    usage_delta = experiments.usage_delta(usage_log, usage_start)
    report["usage"] = usage_delta.model_dump(mode="json")
    report["configuration_usage"] = {
        cfg.name: _config_usage(usage_delta, cfg).model_dump(mode="json") for cfg in configs
    }
    report["evaluator_usage"] = {
        stage: experiments.select_usage(usage_delta, stage).model_dump(mode="json")
        for stage in (refeval.STAGE, refeval.PAIRED_STAGE)
    }
    report["cost"] = {
        cfg.name: experiments.usage_totals(_config_usage(usage_delta, cfg)).model_dump(mode="json")
        for cfg in configs
    }
    if "manifests" in report:
        if budget_status is None:
            from . import budget

            budget_status = budget.snapshot(settings)
        for cfg in configs:
            manifest = report["manifests"][cfg.name]
            manifest["usage"] = _config_usage(usage_delta, cfg).model_dump(mode="json")
            manifest["budget_status"] = budget_status
        if "manifest" in report:
            report["manifest"]["usage"] = report["manifests"][configs[0].name]["usage"]
    report["stage_scope"] = {}
    for cfg in configs:
        inputs_path = ws.data_dir / "ablate" / cfg.name / "benchmark_inputs.json"
        inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        scope = inputs["stage_scope"]
        scope["executed_stages"].extend([
            "s5-deterministic-finish", "s5c-deterministic-layout",
            "s6-sample-srt-emission", "official-reference-scoring",
        ])
        if judge:
            scope["executed_stages"].append("independent-reference-evaluation")
        scope["source_scene_coverage"] = (
            "all-source-scenes" if {scene.id for scene in sampled} == {scene.id for scene in base.scenes}
            else "sampled-scenes"
        )
        scope["full_pipeline_review"] = False
        report["stage_scope"][cfg.name] = scope
        report["configs"][cfg.name]["stage_scope"] = scope
        report["artifacts"][cfg.name]["stage_scope"] = scope
        if cfg.route is not None:
            report["configs"][cfg.name]["frozen_inputs"]["stage_scope"] = scope
        if "manifests" in report:
            report["manifests"][cfg.name]["stage_scope"] = scope
        inputs_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "ablate.md").write_text(render_md(report), encoding="utf-8")
    return report


# ---------------------------------------------------------------- report


def render_md(report: dict) -> str:
    fmt = refeval._fmt
    errs_cols = refeval.ERROR_CATEGORIES
    judged = report.get("judge") is not None
    head = ["config", "model", "think", "mode", "obs", "narr", "aim", "mem", "chrF++", "coverage", "len ratio",
            "level agr", "dash agr", "dash recall", "CPS>12", "CPL>16"]
    if judged:
        head += ["judge", *errs_cols, "sev sum"]
    head += ["calls", "cost $"]
    out = [
        f"# Architecture benchmark — {report['video_id']}",
        "",
        f"Gold: `{report['ref']}` restricted to {report['n_scenes']} sampled scenes (seed {report['seed']}; "
        f"{report['alignment']['official_lines']} official lines, {report['alignment']['en_cues']} EN cues, "
        f"{report['alignment']['groups']} aligned groups)."
        + (f" Judge: {', '.join(report['judge']['models'])}, {len(report['judge']['windows'])} windows"
           + (" (judge columns = mean over judges)." if len(report["judge"]["models"]) > 1 else ".")
           if judged else ""),
        "Generated SRT and layout metrics cover the sampled output only; no full-video review or quality gain is claimed.",
        "",
        "| " + " | ".join(head) + " |",
        "|---" * len(head) + "|",
    ]
    for name, cfg in report["configs"].items():
        m = report["metrics"][name]
        c = report["cost"][name]
        row = [
            name, cfg["model"], cfg["thinking"], cfg["mode"],
            "on" if cfg["observations"] else "off", cfg["narrative"], fmt(cfg.get("aim")), fmt(cfg.get("memory")),
            fmt(m["chrf"]), fmt(m["coverage"], True), fmt(m["length_ratio"]),
            fmt(m["speech_level_agreement"], True), fmt(m["dash_agreement"], True), fmt(m["dash_recall"], True),
            fmt(m["cps_over_rate"], True), fmt(m["cpl_over_rate"], True),
        ]
        if judged:
            j = m.get("judge") or {}
            row += [fmt(j.get("score")), *(fmt(j.get("errors", {}).get(k)) for k in errs_cols),
                    fmt(j.get("severity_sum"))]
        row += [str(int(c["calls"])), f"{c['cost_usd']:.4f}"]
        out.append("| " + " | ".join(row) + " |")
    o = report["official"]
    gold = ["official (gold)", "", "", "", "", "", "", "", "–", "–", "1.00", "–", "–", "–",
            fmt(o["cps_over_rate"], True), fmt(o["cpl_over_rate"], True)]
    if judged:
        gold += ["–"] * (len(errs_cols) + 2)
    gold += ["", ""]
    out.append("| " + " | ".join(gold) + " |")
    n_t = sum(1 for s in report["scenes"] if s["reason"] == "targeted")
    out += [
        "",
        f"Scenes: {n_t} targeted (register shift / callback), {report['n_scenes'] - n_t} random. "
        "Metrics as in refeval (after the deterministic S5 normalization). "
        "calls/cost = translation, perception and router stages prefixed `ablate:<config>`; "
        "independent evaluator usage is reported separately (checkpoint hits are free).",
    ]
    if judged:
        out += refeval.render_judges_md(
            {name: report["metrics"][name]["judges"] for name in report["configs"]}, report["judge"]["models"]
        )
        out += refeval.render_calibration_md(report["judge"]["calibration"])
    if report.get("paired"):
        out += refeval.render_paired_md(report)
    return "\n".join(out) + "\n"
