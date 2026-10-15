"""Offline tests for experiment arms (pipeline.seed_arm) and the blind
evaluation helpers (evaluate.py). No Gemini, no ffmpeg."""

from __future__ import annotations

from pathlib import Path

import pytest

from substage import evaluate
from substage.config import Workspace
from substage.models import VideoInfo, ContextMap
from substage.pipeline import arm_video_id, seed_arm
from substage.store import Checkpoints, save_model


def _ws(root: Path, video_id: str) -> Workspace:
    return Workspace(
        video_id=video_id,
        data_dir=root / "data" / video_id,
        cache_dir=root / "cache" / video_id,
        log_dir=root / "log" / video_id,
    )


# ---------------------------------------------------------------- arms

_ALL_PIPELINE = (
    "video", "cues", "scenes", "observations", "context_map", "timeline", "snapshots", "ost",
    "translations", "review", "finale_audit", "blind_read", "regenerate",
)
_ARTIFACTS = (
    "video.json", "cues.json", "scenes.json", "observations.json", "context_map.yaml",
    "timeline.yaml", "snapshots.json", "translations.json", "edits.json", "base.ko.srt",
)


def _populated_base(root: Path) -> Workspace:
    base = _ws(root, "base")
    for key in _ALL_PIPELINE:
        Checkpoints(base, "pipeline").put(key, f"h-{key}", ContextMap(synopsis=key, tone_baseline="t"))
    for store in ("s2v", "s2a", "s2b", "translate", "review", "picture-check"):
        Checkpoints(base, store).put("k", "h", ContextMap(synopsis=store, tone_baseline="t"))
    for name in _ARTIFACTS:
        base.path(name).write_text(name, encoding="utf-8")
    base.cache("proxy.mp4").write_bytes(b"proxy")
    return base


def _listing(ws: Workspace, store: str) -> set[str]:
    d = ws.data_dir / "checkpoints" / store
    return {p.stem for p in d.glob("*.json")} if d.is_dir() else set()


def test_arm_video_id():
    assert arm_video_id("sample-2016", "a") == "sample-2016__a"
    with pytest.raises(ValueError):
        arm_video_id("x", "../evil")


def test_seed_arm_copies_only_pre_translation_checkpoints(tmp_path):
    base = _populated_base(tmp_path)
    arm = _ws(tmp_path, "base__c")

    copied = seed_arm(base, arm)

    expected = {"video", "cues", "scenes", "observations", "context_map", "timeline", "snapshots", "ost"}
    assert _listing(arm, "pipeline") == expected
    assert set(copied) == expected | {"s2v", "s2a", "s2b"}
    for store in ("s2v", "s2a", "s2b"):
        assert _listing(arm, store) == {"k"}
    for store in ("translate", "review", "picture-check"):
        assert not (arm.data_dir / "checkpoints" / store).exists()
    present = {p.name for p in arm.data_dir.iterdir() if p.is_file()}
    assert present == {
        "video.json", "cues.json", "scenes.json", "observations.json",
        "context_map.yaml", "timeline.yaml", "snapshots.json",
    }
    # cache shared through a symlink to the base cache
    assert arm.cache_dir.is_symlink()
    assert (arm.cache_dir / "proxy.mp4").read_bytes() == b"proxy"
    # copied checkpoints still hit with the base's hashes
    assert Checkpoints(arm, "pipeline").get("context_map", "h-context_map", ContextMap).synopsis == "context_map"


def test_seed_arm_no_viewing_copies_only_video_cues_scenes(tmp_path):
    base = _populated_base(tmp_path)
    arm = _ws(tmp_path, "base__a")

    copied = seed_arm(base, arm, no_viewing=True)

    assert set(copied) == {"video", "cues", "scenes"}
    assert _listing(arm, "pipeline") == {"video", "cues", "scenes"}
    for store in ("s2v", "s2a", "s2b", "translate", "review"):
        assert not (arm.data_dir / "checkpoints" / store).exists()
    present = {p.name for p in arm.data_dir.iterdir() if p.is_file()}
    assert present == {"video.json", "cues.json", "scenes.json"}
    assert arm.cache_dir.is_symlink()


def test_seed_arm_only_on_first_use(tmp_path):
    base = _populated_base(tmp_path)
    arm = _ws(tmp_path, "base__c")
    Checkpoints(arm, "pipeline").put("video", "own", ContextMap(synopsis="own", tone_baseline="t"))

    assert seed_arm(base, arm) == []
    assert _listing(arm, "pipeline") == {"video"}
    assert Checkpoints(arm, "pipeline").get("video", "own", ContextMap).synopsis == "own"


def test_seed_arm_without_base_checkpoints_runs_normally(tmp_path):
    base = _ws(tmp_path, "base")
    arm = _ws(tmp_path, "base__c")

    assert seed_arm(base, arm) == []
    assert not (arm.data_dir / "checkpoints").exists()
    assert not arm.cache_dir.exists() and not arm.cache_dir.is_symlink()


def test_seed_arm_keeps_existing_arm_cache(tmp_path):
    base = _populated_base(tmp_path)
    arm = _ws(tmp_path, "base__c")
    arm.cache("own.bin").write_bytes(b"x")

    seed_arm(base, arm)

    assert not arm.cache_dir.is_symlink()
    assert (arm.cache_dir / "own.bin").exists()


# ---------------------------------------------------------------- evaluate helpers


def test_shuffle_labels_deterministic_per_scene():
    names = ["A", "B", "C", "ref"]
    a = evaluate.shuffle_labels(names, "s001", 7)
    assert a == evaluate.shuffle_labels(names, "s001", 7)
    assert list(a) == ["X", "Y", "Z", "W"]
    assert sorted(a.values()) == sorted(names)
    orders = {tuple(evaluate.shuffle_labels(names, f"s{i:03d}", 7).values()) for i in range(30)}
    assert len(orders) > 1  # labels really are reshuffled per scene


def test_parse_arms():
    assert evaluate.parse_arms("A=f__a, B=f ,C=f__c") == {"A": "f__a", "B": "f", "C": "f__c"}
    for bad in ("A", "A=x,A=y", "ref=x", ""):
        with pytest.raises(ValueError):
            evaluate.parse_arms(bad)


# ---------------------------------------------------------------- arm srt


def test_arm_srt_prefers_video_id_then_unique_glob(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    (d / "video.ko.srt").write_text("", encoding="utf-8")
    (d / "other.ko.srt").write_text("", encoding="utf-8")
    save_model(d / "video.json", VideoInfo(video_id="video", title="Video", video_path="v", duration_ms=1,
                                         audio_stream=1, en_sub_path="e", en_sub_origin="user"))
    assert evaluate.arm_srt(d) == d / "video.ko.srt"

    lone = tmp_path / "lone"
    lone.mkdir()
    (lone / "x.ko.srt").write_text("", encoding="utf-8")
    assert evaluate.arm_srt(lone) == lone / "x.ko.srt"
    with pytest.raises(FileNotFoundError):
        evaluate.arm_srt(d.parent)


def test_mmss():
    assert evaluate.mmss(65_400) == "01:05"
    assert evaluate.mmss(3_725_000) == "1:02:05"
