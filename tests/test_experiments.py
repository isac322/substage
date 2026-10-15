"""Offline tests for reusable experiment manifest primitives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from substage import experiments
from substage.models import Scene, UsageRecord


def _usage(stage: str, model: str, cost: float) -> UsageRecord:
    return UsageRecord(
        stage=stage,
        model=model,
        input_tokens=10,
        output_tokens=4,
        cached_tokens=2,
        cache_write_tokens=1,
        cost_usd=cost,
    )


def test_usage_delta_excludes_historic_rows(tmp_path: Path):
    log = tmp_path / "usage.jsonl"
    log.write_text(_usage("old", "gemini-3.8-flash", 9).model_dump_json() + "\n", encoding="utf-8")
    cursor = experiments.usage_cursor(log)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(_usage("new", "codex/gpt-6.1-sol", 0.25).model_dump_json() + "\n")
    delta = experiments.usage_delta(log, cursor)
    assert delta.records == 1
    assert set(delta.per_stage) == {"new"}
    assert set(delta.per_model) == {"codex/gpt-6.1-sol"}
    assert delta.per_stage["new"].cost_usd == 0.25


def test_frozen_sample_validates_every_scene_identity(tmp_path: Path):
    report = tmp_path / "prior.json"
    report.write_text(json.dumps({
        "video_id": "video",
        "n_scenes": 1,
        "scenes": [{"id": "s001", "start_ms": 10, "end_ms": 20, "cue_ids": ["c1"], "reason": "targeted"}],
    }), encoding="utf-8")
    loaded, records = experiments.frozen_sample(report)
    selected = experiments.validate_frozen_sample(loaded, "video", [Scene(id="s001", start_ms=10, end_ms=20, cue_ids=["c1"])])
    assert [scene.id for scene, _ in selected] == ["s001"]
    assert records[0].cue_ids == ("c1",)

    with pytest.raises(ValueError, match="identity mismatch"):
        experiments.validate_frozen_sample(loaded, "video", [Scene(id="s001", start_ms=10, end_ms=21, cue_ids=["c1"])])


def test_cycle_count_ignores_incomplete_and_verification_only_cycles():
    complete = experiments.CycleState(
        phase="decision", hypothesis="h", change="c", translation="t", score="s", debate="d", decision="keep",
        completed=True, actual_improvement=True,
    )
    incomplete = complete.model_copy(update={"completed": False})
    verification = complete.model_copy(update={"verification_only": True})
    assert experiments.count_actual_improvement_cycles([complete, incomplete, verification]) == 1
    assert not experiments.can_pause_for_no_improvement([complete] * 9)
    assert experiments.can_pause_for_no_improvement([complete] * 10)


def test_manifest_preserves_actual_model_origin_and_scene_records():
    manifest = experiments.ExperimentManifest(
        experiment_id="exp-1",
        cycle_id="cycle-1",
        declared_architecture="dpaf",
        translator="codex/gpt-6.1-sol",
        model="codex/gpt-6.1-sol",
        effort="high",
        mode="text",
        roles={"translator": "codex/gpt-6.1-sol"},
        input_digest="input",
        sampled_scenes=(experiments.SampledScene(scene_id="s1", start_ms=0, end_ms=100, cue_ids=("c1",)),),
    )
    assert manifest.actual_model == "codex/gpt-6.1-sol"
    assert manifest.sampled_scenes[0].cue_ids == ("c1",)
