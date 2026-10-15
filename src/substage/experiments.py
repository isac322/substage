"""Reusable experiment manifests, frozen samples, usage deltas, and cycle state.

This module is deliberately provider-agnostic.  It records what an experiment
actually used; it does not select models or dispatch network calls.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from .models import Scene, UsageRecord

EXPERIMENT_MODEL_POOL: tuple[str, ...] = (
    "gemini-3.8-flash",
    "gemini-4-argon",
    "codex/gpt-6-luna",
    "codex/gpt-6.1-sol",
)

CyclePhase = Literal["hypothesis", "change", "translation", "score", "debate", "decision"]


class SampledScene(BaseModel):
    """Immutable scene identity captured in an experiment report/manifest."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    scene_id: str = Field(validation_alias=AliasChoices("scene_id", "id"))
    start_ms: int
    end_ms: int
    cue_ids: tuple[str, ...] = Field(min_length=1)
    reason: str | None = None

    @property
    def id(self) -> str:
        return self.scene_id


# Descriptive alias for callers that prefer the report terminology.
SceneSampleRecord = SampledScene


class UsageTotals(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0


class UsageDelta(BaseModel):
    """Per-run accounting, split by stage, model, and their pair."""

    records: int = 0
    per_stage: dict[str, UsageTotals] = Field(default_factory=dict)
    per_model: dict[str, UsageTotals] = Field(default_factory=dict)
    per_stage_model: dict[str, UsageTotals] = Field(default_factory=dict)


class BudgetReservationLike(Protocol):
    """Small typed contract for the prospective budget controller."""

    reservation_id: str
    group: str
    upper_usd: float

    def settle(self, cost_usd: float) -> None: ...


class BudgetControllerLike(Protocol):
    def reserve(
        self, settings: Any, model: str, stage: str, upper_usd: float, *, provider: str | None = None
    ) -> BudgetReservationLike | None: ...

    def snapshot(self, settings: Any) -> dict[str, Any]: ...


class CycleState(BaseModel):
    """Evidence-bearing state for one adaptive experiment cycle."""

    phase: CyclePhase = "hypothesis"
    hypothesis: str | None = None
    change: str | None = None
    translation: str | None = None
    score: str | None = None
    debate: str | None = None
    decision: str | None = None
    completed: bool = False
    # Legacy completion marker, never evidence that subtitle quality increased.
    actual_improvement: bool = False
    quality_change: Literal["unknown", "improved", "unchanged", "regressed"] = "unknown"
    quality_evidence: str | None = None
    verification_only: bool = False
    architecture_changed: bool = False


class HeldoutMetadata(BaseModel):
    """Truthful split labels; a label is not evidence of unread data."""

    split: str | None = None
    label: str | None = None
    unread_new_data: bool = False
    note: str | None = None


class ExperimentManifest(BaseModel):
    """Portable record of one actual model/architecture experiment run."""

    experiment_id: str
    cycle_id: str
    declared_architecture: str
    translator: str | None = None
    model: str | None = None
    effort: str | None = None
    mode: str | None = None
    roles: dict[str, str] = Field(default_factory=dict)
    stage_scope: dict[str, Any] = Field(default_factory=dict)
    input_digest: str
    frozen_opus_digest: str | None = None
    sampled_scenes: tuple[SampledScene, ...] = ()
    usage: UsageDelta = Field(default_factory=UsageDelta)
    budget_status: dict[str, Any] | None = None
    cycle: CycleState = Field(default_factory=CycleState)
    heldout: HeldoutMetadata = Field(default_factory=HeldoutMetadata)
    @field_validator("translator", "model")
    @classmethod
    def _validate_pool_model(cls, value: str | None) -> str | None:
        if value is not None and value not in EXPERIMENT_MODEL_POOL:
            raise ValueError(f"model is outside the fixed experiment pool: {value!r}")
        return value
    @field_validator("roles")
    @classmethod
    def _validate_pool_roles(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = {model for model in value.values() if model not in EXPERIMENT_MODEL_POOL}
        if invalid:
            raise ValueError(f"role model(s) outside the fixed experiment pool: {sorted(invalid)}")
        return value

    @property
    def actual_translator(self) -> str | None:
        return self.translator

    @property
    def actual_model(self) -> str | None:
        return self.model

    @property
    def actual_effort(self) -> str | None:
        return self.effort

    @property
    def actual_mode(self) -> str | None:
        return self.mode
    @property
    def opus_digest(self) -> str | None:
        return self.frozen_opus_digest

    @property
    def sampled_scene_records(self) -> tuple[SampledScene, ...]:
        return self.sampled_scenes
SceneRecord = SampledScene
ExperimentCycle = CycleState
ModelUsage = UsageTotals
RunUsage = UsageDelta


def _totals(rows: Iterable[UsageRecord]) -> UsageTotals:
    total = UsageTotals()
    for row in rows:
        total.calls += 1
        total.input_tokens += row.input_tokens
        total.output_tokens += row.output_tokens
        total.cached_tokens += row.cached_tokens
        total.cache_write_tokens += row.cache_write_tokens
        total.cost_usd += row.cost_usd
    return total


def summarize_usage(rows: Iterable[UsageRecord]) -> UsageDelta:
    """Summarize only the supplied rows (normally a new-run suffix)."""
    rows = list(rows)
    by_stage: dict[str, list[UsageRecord]] = {}
    by_model: dict[str, list[UsageRecord]] = {}
    by_pair: dict[str, list[UsageRecord]] = {}
    for row in rows:
        by_stage.setdefault(row.stage, []).append(row)
        by_model.setdefault(row.model, []).append(row)
        by_pair.setdefault(f"{row.stage}|{row.model}", []).append(row)
    return UsageDelta(
        records=len(rows),
        per_stage={key: _totals(value) for key, value in by_stage.items()},
        per_model={key: _totals(value) for key, value in by_model.items()},
        per_stage_model={key: _totals(value) for key, value in by_pair.items()},
    )


def select_usage(usage: UsageDelta, stage_prefix: str) -> UsageDelta:
    """Select one executed configuration, excluding independent evaluator calls."""
    stages = {key: value for key, value in usage.per_stage.items()
              if key == stage_prefix or key.startswith(stage_prefix + ":")}
    pairs = {key: value for key, value in usage.per_stage_model.items()
             if key.split("|", 1)[0] in stages}
    models: dict[str, UsageTotals] = {}
    for key, value in pairs.items():
        model = key.split("|", 1)[1]
        total = models.setdefault(model, UsageTotals())
        for field in UsageTotals.model_fields:
            setattr(total, field, getattr(total, field) + getattr(value, field))
    return UsageDelta(records=sum(value.calls for value in stages.values()),
                      per_stage=stages, per_model=models, per_stage_model=pairs)


def usage_totals(usage: UsageDelta) -> UsageTotals:
    total = UsageTotals()
    for value in usage.per_stage.values():
        for field in UsageTotals.model_fields:
            setattr(total, field, getattr(total, field) + getattr(value, field))
    return total


@dataclass(frozen=True)
class UsageCursor:
    """Byte/row position captured immediately before a run starts."""

    offset: int
    records: int


def usage_cursor(log_path: Path) -> UsageCursor:
    if not log_path.is_file():
        return UsageCursor(0, 0)
    raw = log_path.read_bytes()
    return UsageCursor(len(raw), raw.count(b"\n"))


def usage_delta(log_path: Path, cursor: UsageCursor) -> UsageDelta:
    """Read the usage-log suffix after *cursor*, excluding all prior history."""
    if not log_path.is_file():
        return UsageDelta()
    with log_path.open("rb") as fh:
        fh.seek(cursor.offset)
        suffix = fh.read().decode("utf-8")
    rows: list[UsageRecord] = []
    for line in suffix.splitlines():
        if line.strip():
            rows.append(UsageRecord.model_validate_json(line))
    return summarize_usage(rows)


def _scene_from_report(raw: Mapping[str, Any]) -> SampledScene:
    scene_id = raw.get("scene_id", raw.get("id"))
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("frozen sample scene is missing a non-empty id")
    cue_ids = raw.get("cue_ids")
    if cue_ids is None:
        cues = raw.get("cues")
        cue_ids = cues if isinstance(cues, list) else None
    if not isinstance(cue_ids, list) or not cue_ids or not all(isinstance(c, str) and c for c in cue_ids):
        raise ValueError(f"frozen sample scene {scene_id!r} must include cue_ids")
    try:
        start_ms = int(raw["start_ms"])
        end_ms = int(raw["end_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"frozen sample scene {scene_id!r} is missing start_ms/end_ms") from exc
    return SampledScene(
        scene_id=scene_id,
        start_ms=start_ms,
        end_ms=end_ms,
        cue_ids=tuple(cue_ids),
        reason=raw.get("reason"),
    )


def frozen_sample(path: Path | str) -> tuple[dict[str, Any], tuple[SampledScene, ...]]:
    """Load a prior ablation report without modifying it."""
    source = Path(path)
    if source.is_dir():
        source = source / "ablate.json"
    if not source.is_file():
        raise FileNotFoundError(f"frozen ablation report not found: {source}")
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid frozen ablation report: {source}") from exc
    if not isinstance(report, dict) or not isinstance(report.get("scenes"), list):
        raise ValueError(f"frozen ablation report has no scenes list: {source}")
    scenes = tuple(_scene_from_report(row) for row in report["scenes"] if isinstance(row, Mapping))
    if len(scenes) != len(report["scenes"]):
        raise ValueError("frozen ablation report contains a non-object scene record")
    return report, scenes


def validate_frozen_sample(
    report: Mapping[str, Any], expected_video_id: str, scenes: Sequence[Scene],
    *, input_paths: Mapping[str, Path] | None = None,
) -> tuple[tuple[Scene, str], ...]:
    """Validate every frozen scene identity before selecting it for a rerun."""
    report_video = report.get("video_id")
    if report_video != expected_video_id:
        raise ValueError(f"frozen sample video mismatch: report={report_video!r}, expected={expected_video_id!r}")
    if "provenance" in report:
        provenance = report["provenance"]
        if not isinstance(provenance, Mapping):
            raise ValueError("frozen sample provenance must contain cues/scenes SHA256 records")
        for name in ("cues", "scenes"):
            record = provenance.get(name)
            if not isinstance(record, Mapping) or not isinstance(record.get("sha256"), str):
                raise ValueError(f"frozen sample provenance is missing {name}.sha256")
            expected = record["sha256"]
            if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected.lower()):
                raise ValueError(f"frozen sample provenance has invalid {name}.sha256")
            if input_paths is None or name not in input_paths:
                raise ValueError(f"frozen sample {name} provenance requires current input bytes")
            if file_digest(input_paths[name]) != expected.lower():
                raise ValueError(f"frozen sample {name} SHA256 mismatch: input bytes changed")
    records = tuple(_scene_from_report(row) for row in report.get("scenes", []))
    if "n_scenes" in report and report["n_scenes"] != len(records):
        raise ValueError("frozen sample n_scenes does not match its scene records")
    by_id = {scene.id: scene for scene in scenes}
    if len(by_id) != len(scenes):
        raise ValueError("base workspace contains duplicate scene ids")
    selected: list[tuple[Scene, str]] = []
    seen: set[str] = set()
    for record in records:
        if record.scene_id in seen:
            raise ValueError(f"frozen sample contains duplicate scene id {record.scene_id!r}")
        seen.add(record.scene_id)
        current = by_id.get(record.scene_id)
        if current is None:
            raise ValueError(f"frozen sample scene is absent from base workspace: {record.scene_id!r}")
        if (current.start_ms, current.end_ms, tuple(current.cue_ids)) != (
            record.start_ms,
            record.end_ms,
            record.cue_ids,
        ):
            raise ValueError(f"frozen sample scene identity mismatch: {record.scene_id!r}")
        selected.append((current, record.reason or "frozen"))
    return tuple(selected)


def actual_improvement_completed(cycle: CycleState | Mapping[str, Any]) -> bool:
    """Count a completed experiment, not a claimed quality gain.

    The historical function/field names are retained for existing experiment
    reports. All six evidence-bearing phases are required regardless of the
    legacy boolean or measured quality outcome.
    """
    state = cycle if isinstance(cycle, CycleState) else CycleState.model_validate(cycle)
    required = (state.hypothesis, state.change, state.translation, state.score, state.debate, state.decision)
    return state.completed and not state.verification_only and all(
        value is not None and value.strip() for value in required
    )


def count_actual_improvement_cycles(cycles: Iterable[CycleState | Mapping[str, Any]]) -> int:
    return sum(actual_improvement_completed(cycle) for cycle in cycles)


def can_pause_for_no_improvement(
    cycles: Iterable[CycleState | Mapping[str, Any]], *, minimum_cycles: int = 10,
) -> bool:
    """The caller supplies the experiment's required research-cycle minimum."""
    if minimum_cycles < 1:
        raise ValueError("minimum_cycles must be positive")
    return count_actual_improvement_cycles(cycles) >= minimum_cycles


# More explicit spelling for callers and reports.
no_improvement_pause_allowed = can_pause_for_no_improvement


def file_digest(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
