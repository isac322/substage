"""Offline tests for store.py: serialization round-trips, hashing, checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from substage.config import Workspace
from substage.models import Cue, Scene
from substage.store import Checkpoints, input_hash, load_model, load_models, save_model


class Widget(BaseModel):
    name: str
    count: int = 0
    tags: list[str] = Field(default_factory=list)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(
        video_id="test-video",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
    )


# ------------------------------------------------------------------ save/load


def test_save_load_json_roundtrip(tmp_path: Path):
    w = Widget(name="w", count=3, tags=["a", "b"])
    p = tmp_path / "w.json"
    save_model(p, w)
    assert load_model(p, Widget) == w


def test_save_load_yaml_roundtrip(tmp_path: Path):
    w = Widget(name="한국어", count=1)
    p = tmp_path / "w.yaml"
    save_model(p, w)
    assert "한국어" in p.read_text(encoding="utf-8")  # unicode preserved
    assert load_model(p, Widget) == w


def test_load_models_list(tmp_path: Path):
    items = [Widget(name="a"), Widget(name="b", count=2)]
    for name in ("l.json", "l.yaml"):
        p = tmp_path / name
        save_model(p, items)
        assert load_models(p, Widget) == items


def test_save_model_writes_atomically_and_parent_dirs(tmp_path: Path):
    p = tmp_path / "nested" / "deep" / "w.json"
    save_model(p, Widget(name="x"))
    assert p.exists()
    assert not list(p.parent.glob("*.tmp"))  # no temp files left behind


def test_save_model_rejects_unknown_suffix(tmp_path: Path):
    with pytest.raises(ValueError):
        save_model(tmp_path / "w.txt", Widget(name="x"))


def test_load_real_model_roundtrip(tmp_path: Path):
    cue = Cue(id="c0001", start_ms=100, end_ms=2500, text="Hello.", raw="Hello.")
    p = tmp_path / "cue.json"
    save_model(p, cue)
    assert load_model(p, Cue) == cue


# ------------------------------------------------------------------ input_hash


def test_input_hash_deterministic_and_sensitive():
    h1 = input_hash("scene-1", b"bytes", Widget(name="w"))
    h2 = input_hash("scene-1", b"bytes", Widget(name="w"))
    assert h1 == h2 and len(h1) == 16
    assert input_hash("scene-1", b"bytes", Widget(name="w2")) != h1
    assert input_hash("scene-2", b"bytes", Widget(name="w")) != h1


def test_input_hash_model_canonicalization():
    # same semantic content must hash identically regardless of dump order
    a = Scene(id="s001", start_ms=0, end_ms=1000, cue_ids=["c0001"], act=1)
    b = Scene.model_validate({"id": "s001", "start_ms": 0, "end_ms": 1000,
                              "cue_ids": ["c0001"], "act": 1})
    assert input_hash(a) == input_hash(b)


def test_input_hash_no_cross_type_collision():
    # str "abc" vs b"abc" vs a model dumping to the same text must differ
    assert input_hash("abc") != input_hash(b"abc")


def test_input_hash_part_boundary_unambiguous():
    assert input_hash("ab", "c") != input_hash("a", "bc")


# ----------------------------------------------------------------- Checkpoints


def test_checkpoint_miss_then_hit(ws: Workspace):
    cp = Checkpoints(ws, "s3")
    h = input_hash("inputs-v1")
    assert cp.get("s001", h, Widget) is None
    cp.put("s001", h, Widget(name="done", count=7))
    hit = cp.get("s001", h, Widget)
    assert hit is not None and hit.name == "done" and hit.count == 7


def test_checkpoint_hit_only_when_input_hash_matches(ws: Workspace):
    cp = Checkpoints(ws, "s3")
    h1, h2 = input_hash("v1"), input_hash("v2")
    cp.put("s001", h1, Widget(name="old"))
    assert cp.get("s001", h1, Widget) is not None
    assert cp.get("s001", h2, Widget) is None  # stale input -> miss


def test_checkpoint_overwrite_on_rerun(ws: Workspace):
    cp = Checkpoints(ws, "s3")
    cp.put("k", input_hash("v1"), Widget(name="old"))
    cp.put("k", input_hash("v2"), Widget(name="new"))
    assert cp.get("k", input_hash("v1"), Widget) is None
    assert cp.get("k", input_hash("v2"), Widget).name == "new"


def test_checkpoint_invalidate(ws: Workspace):
    cp = Checkpoints(ws, "s3")
    h = input_hash("v1")
    cp.put("k", h, Widget(name="x"))
    cp.invalidate("k")
    assert cp.get("k", h, Widget) is None
    cp.invalidate("k")  # idempotent


def test_checkpoint_key_sanitized_for_filename(ws: Workspace):
    cp = Checkpoints(ws, "s3")
    h = input_hash("v1")
    cp.put("scene s001/act 2", h, Widget(name="x"))
    assert cp.get("scene s001/act 2", h, Widget) is not None
    assert all("/" not in p.name for p in cp.dir.iterdir())


def test_checkpoint_stages_isolated(ws: Workspace):
    h = input_hash("v1")
    Checkpoints(ws, "s3").put("k", h, Widget(name="three"))
    assert Checkpoints(ws, "s4").get("k", h, Widget) is None


def test_checkpoint_corrupt_file_is_miss(ws: Workspace):
    cp = Checkpoints(ws, "s3")
    (cp.dir / "k.json").write_text("{not json", encoding="utf-8")
    assert cp.get("k", input_hash("v1"), Widget) is None
