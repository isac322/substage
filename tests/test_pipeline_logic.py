"""Offline unit tests for the pure helpers in pipeline.py: the user-editable
artifact checkpoint wrapper, generator-aware checkpoint keys and arm seeding,
the OST cue/scene merge, and the deterministic pass on picture-check edits.
No Gemini, no ffmpeg."""

from __future__ import annotations

from types import SimpleNamespace

from substage import narrative
from substage.config import Settings, Workspace
from substage.models import Cue, CueEdit, VideoInfo, ContextMap, Scene
from substage.pipeline import (
    _apply_picture_edits,
    _editable,
    _context_map_hash,
    _merge_ost,
    _timeline_hash,
    seed_arm,
)
from substage.store import Checkpoints, input_hash, load_model, save_model


def _ws(tmp_path, video_id: str = "t") -> Workspace:
    return Workspace(
        video_id=video_id,
        data_dir=tmp_path / video_id / "data",
        cache_dir=tmp_path / video_id / "cache",
        log_dir=tmp_path / video_id / "log",
    )


def _cue(cid: str, start: int, end: int, source: str = "main") -> Cue:
    return Cue(id=cid, start_ms=start, end_ms=end, text="x", raw="x", source=source)


def _scene(sid: str, start: int, end: int, cue_ids: list[str]) -> Scene:
    return Scene(id=sid, start_ms=start, end_ms=end, cue_ids=cue_ids)


def _map(synopsis: str = "syn") -> ContextMap:
    return ContextMap(synopsis=synopsis, tone_baseline="tone")


# ---------------------------------------------------------------- _editable


def test_editable_miss_generates_and_saves(tmp_path):
    ws = _ws(tmp_path)
    ck = Checkpoints(ws, "pipeline")
    artifact = ws.path("context_map.yaml")
    made = _map("generated")

    out = _editable(ck, "context_map", "h0", ContextMap, artifact, lambda: made)

    assert out == made
    assert load_model(artifact, ContextMap) == made  # artifact written
    assert ck.get("context_map", "h0", ContextMap) == made


def test_editable_hit_loads_existing_yaml(tmp_path):
    ws = _ws(tmp_path)
    ck = Checkpoints(ws, "pipeline")
    artifact = ws.path("context_map.yaml")
    generated = _map("generated")
    h = input_hash("inputs")

    # First run: generate + checkpoint.
    assert _editable(ck, "context_map", h, ContextMap, artifact, lambda: generated) == generated
    # Human edits the YAML between runs.
    edited = _map("human edited")
    save_model(artifact, edited)

    def boom():  # must not be called: checkpoint hits
        raise AssertionError("regenerated on a checkpoint hit")

    out = _editable(ck, "context_map", h, ContextMap, artifact, boom)

    assert out == edited  # on-disk artifact wins over the checkpoint copy
    assert ck.get("context_map", h, ContextMap) == edited  # checkpoint updated to it


def test_editable_hit_without_yaml_uses_checkpoint(tmp_path):
    ws = _ws(tmp_path)
    ck = Checkpoints(ws, "pipeline")
    artifact = ws.path("context_map.yaml")
    generated = _map("generated")
    h = input_hash("inputs")
    assert _editable(ck, "context_map", h, ContextMap, artifact, lambda: generated) == generated

    artifact.unlink()  # YAML deleted; checkpoint still valid

    out = _editable(ck, "context_map", h, ContextMap, artifact, lambda: (_ for _ in ()).throw(AssertionError))
    assert out == generated


# ---------------------------------------------------------------- generator-aware keys


def _patch_template(monkeypatch, name: str, suffix: str) -> None:
    real = narrative._prompt
    monkeypatch.setattr(narrative, "_prompt", lambda n: real(n) + (suffix if n == name else ""))


def test_context_map_and_timeline_keys_change_with_template_and_model(monkeypatch):
    settings = SimpleNamespace(model_main="m1")
    cues, scenes, fm = [_cue("c1", 0, 1_000)], [_scene("s001", 0, 1_000, ["c1"])], _map()
    fm_h = _context_map_hash(settings, cues, [], ())
    tl_h = _timeline_hash(settings, fm, scenes, cues, [], ())
    # stable for unchanged generator inputs; --no-viewing stays distinct
    assert _context_map_hash(settings, cues, [], ()) == fm_h
    assert _context_map_hash(settings, cues, [], ("no-viewing",)) != fm_h
    assert _timeline_hash(settings, fm, scenes, cues, [], ("no-viewing",)) != tl_h
    # a different model misses
    other = SimpleNamespace(model_main="m2")
    assert _context_map_hash(other, cues, [], ()) != fm_h
    assert _timeline_hash(other, fm, scenes, cues, [], ()) != tl_h

    _patch_template(monkeypatch, "context_map.md", "\nnew rule")
    assert _context_map_hash(settings, cues, [], ()) != fm_h
    assert _timeline_hash(settings, fm, scenes, cues, [], ()) == tl_h  # timeline.md untouched
    _patch_template(monkeypatch, "timeline.md", "\nnew rule")
    assert _timeline_hash(settings, fm, scenes, cues, [], ()) != tl_h


class _MapGC:
    def __init__(self, synopsis: str):
        self.settings = SimpleNamespace(model_main="m1")
        self.synopsis = synopsis
        self.prompts: list[str] = []

    def generate(self, *, stage, prompt, schema, **_kw):
        self.prompts.append(prompt)
        return ContextMap(synopsis=self.synopsis, tone_baseline="tone")


def test_arm_seeded_from_older_template_regenerates_context_map(tmp_path, monkeypatch):
    base, arm = _ws(tmp_path, "t"), _ws(tmp_path, "t__v2")
    settings = SimpleNamespace(model_main="m1")
    cues = [_cue("c1", 0, 1_000)]
    video = VideoInfo(video_id="t", title="M", year=2001, video_path="v", duration_ms=1_000,
                    audio_stream=0, en_sub_path="s", en_sub_origin="user")

    def context_map(ws: Workspace, gc: _MapGC) -> ContextMap:
        return _editable(
            Checkpoints(ws, "pipeline"), "context_map", _context_map_hash(settings, cues, [], ()),
            ContextMap, ws.path("context_map.yaml"),
            lambda: narrative.build_context_map(ws, gc, video, cues, []),
        )

    old_gc = _MapGC("old")
    assert context_map(base, old_gc).synopsis == "old"

    _patch_template(monkeypatch, "context_map.md", "\nnew rule")
    assert "context_map" in seed_arm(base, arm)
    assert load_model(arm.path("context_map.yaml"), ContextMap).synopsis == "old"  # seeded copy

    new_gc = _MapGC("new")
    out = context_map(arm, new_gc)

    assert out.synopsis == "new"  # neither pipeline nor inner s2a checkpoint reused
    assert len(new_gc.prompts) == 1 and "new rule" in new_gc.prompts[0]
    assert load_model(arm.path("context_map.yaml"), ContextMap).synopsis == "new"
    # an unchanged generator in the arm is a hit again, and human YAML edits win
    save_model(arm.path("context_map.yaml"), _map("human edited"))
    assert context_map(arm, new_gc).synopsis == "human edited"
    assert len(new_gc.prompts) == 1


# ---------------------------------------------------------------- _merge_ost


def test_merge_ost_inserts_into_containing_scene_and_expands_it():
    cues = [_cue("c1", 0, 1_000), _cue("c2", 5_000, 6_000)]
    scenes = [_scene("s001", 0, 8_000, ["c1", "c2"])]
    ost = [_cue("ost-1", 3_000, 9_500, source="ost")]  # inside midpoint, ends past scene end

    cues, scenes = _merge_ost(cues, scenes, ost)

    assert [c.id for c in cues] == ["c1", "ost-1", "c2"]
    assert scenes[0].cue_ids == ["c1", "ost-1", "c2"]  # ordered by cue position
    assert scenes[0].start_ms == 0
    assert scenes[0].end_ms == 9_500  # expanded to cover the ost cue


def test_merge_ost_falls_back_to_nearest_scene():
    cues = [_cue("c1", 0, 1_000)]
    near = _scene("s001", 0, 2_000, ["c1"])
    far = _scene("s002", 60_000, 70_000, [])
    ost = [_cue("ost-1", 4_000, 6_000, source="ost")]  # outside every scene

    _, scenes = _merge_ost(cues, [near, far], ost)

    assert near.cue_ids == ["c1", "ost-1"] and far.cue_ids == []
    assert near.end_ms == 6_000  # grew to include the ost cue


def test_merge_ost_keeps_scenes_sorted_after_expansion():
    cues = [_cue("c1", 4_500, 4_800), _cue("c2", 9_000, 10_000)]
    scenes = [_scene("s001", 4_000, 5_000, ["c1"]), _scene("s002", 8_000, 12_000, ["c2"])]
    # midpoint 8_500 is inside s002, but the ost cue starts before s001.
    ost = [_cue("ost-1", 3_000, 14_000, source="ost")]

    _, scenes = _merge_ost(cues, scenes, ost)

    assert [s.id for s in scenes] == ["s002", "s001"]  # re-sorted after growth
    s002 = scenes[0]
    assert s002.start_ms == 3_000 and s002.end_ms == 14_000
    assert s002.cue_ids == ["ost-1", "c2"]  # by cue position


def test_merge_ost_no_scenes_creates_one():
    cues = [_cue("c1", 0, 1_000)]
    ost = [
        _cue("ost-1", 20_000, 22_000, source="ost"),
        _cue("ost-2", 50_000, 55_000, source="ost"),
    ]

    cues, scenes = _merge_ost(cues, [], ost)

    assert [c.id for c in cues] == ["c1", "ost-1", "ost-2"]
    assert len(scenes) == 1
    s = scenes[0]
    assert s.id == "s001"
    assert s.start_ms == 20_000 and s.end_ms == 55_000
    assert s.cue_ids == ["ost-1", "ost-2"]


# ---------------------------------------------------------------- _apply_picture_edits


def _pic(cid: str, new_ko: str) -> CueEdit:
    return CueEdit(cue_id=cid, new_ko=new_ko, reason="picture", kind="picture")


def test_picture_edits_get_deterministic_normalization():
    ko = {"c1": "원래", "c2": "원래", "c3": "그대로.", "c4": "원래", "c5": "원래"}
    cues = {
        "c2": Cue(id="c2", start_ms=0, end_ms=2000, text="- What?\n- Nothing.", raw=""),
        "c5": Cue(id="c5", start_ms=0, end_ms=3000, text="Let's go, hurry.", raw=""),
    }
    edits = [
        _pic("c1", "건배, 네 건강을 위해.\n엿이나 먹어"),
        _pic("c2", "뭐라고...\n아무것도."),  # two English speakers, dashes lost by the model
        _pic("c4", "하나.\n둘,\n셋"),
        _pic("c5", "- 가자\n- 빨리"),  # one English speaker, dashes added by the model
    ]
    out = _apply_picture_edits(ko, edits, Settings(), cues)
    assert out["c1"] == "건배, 네 건강을 위해\n엿이나 먹어"
    assert out["c2"] == "- 뭐라고...\n- 아무것도"
    assert len(out["c4"].split("\n")) <= 2 and not out["c4"].endswith((".", ","))
    assert "-" not in out["c5"]
    assert out["c3"] == "그대로."  # untouched cues are not re-normalized here


def test_picture_edits_mark_lyric_cues():
    cues = {"c1": Cue(id="c1", start_ms=0, end_ms=2000, text="We are the champions", raw="", lyric=True)}
    out = _apply_picture_edits({"c1": "♪ 원래"}, [_pic("c1", "우리는 챔피언")], Settings(), cues)
    assert out["c1"] == "♪ 우리는 챔피언"


def test_picture_edits_empty_new_ko_is_ignored():
    assert _apply_picture_edits({"c1": "원래"}, [_pic("c1", " ")], Settings(), {}) == {"c1": "원래"}


def test_explicit_semantic_omission_changes_output_without_relaxing_legacy_edits():
    from substage.pipeline import _apply_edits

    incidental = CueEdit(cue_id="c1", new_ko="", reason="incidental acknowledgment", kind="omission")
    assert _apply_edits({"c1": "응"}, [incidental]) == ({"c1": "응"}, [])
    output, applied = _apply_edits({"c1": "응"}, [incidental], allow_omission=True)
    assert output == {"c1": ""} and applied == [incidental]
    not_an_omission = CueEdit(cue_id="c1", new_ko="", reason="shorter", kind="condensation")
    assert _apply_edits({"c1": "응"}, [not_an_omission], allow_omission=True) == ({"c1": "응"}, [])


def test_semantic_checkpoint_metadata_cannot_silently_fall_through_to_legacy_result():
    import pytest
    from pydantic import ValidationError

    from substage.pipeline import _Translations

    legacy = {
        "scene_id": "s1", "mode": "text", "model": "codex/gpt-6.1-sol",
        "cues": [{"cue_id": "c1", "ko": "가자", "model": "codex/gpt-6.1-sol"}],
    }
    assert _Translations(items=[legacy]).items[0].cues[0].ko == "가자"
    for semantic_key in ("lineage", "provenance"):
        with pytest.raises(ValidationError):
            _Translations(items=[{**legacy, semantic_key: {}}])
