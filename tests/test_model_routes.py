"""Offline end-to-end route QA: identity, real images, AV2, and fail-closed media.

No provider, upload or ffmpeg execution is needed. The final-SRT tests retain
real translation/review/S5/C7/layout/emission and fake only ingestion, narrative
preparation and provider responses.
"""
from __future__ import annotations

import json
from dataclasses import replace
import re
from types import SimpleNamespace

import pytest

from substage import pipeline, recheck, review, translate
from substage.capability import Architecture, CapabilityError, resolve_route
from substage.config import Settings, Workspace
from substage.gemini import GeminiBlocked
from substage.models import (
    Cue, CueEdit, CuePerception, VideoInfo, ContextMap, MediaNeed,
    PerceptionLedger, ReadFlag, RegisterMap, Scene, SceneTranslation, Snapshot,
    Timeline, TranslatedCue,
)
from substage.store import Checkpoints

ARGON = "gemini-4-argon"
SOL = "codex/gpt-6.1-sol"
FLASH = "gemini-3.8-flash"
PROXY = SimpleNamespace(name="proxy", uri="gs://scene", mime_type="video/mp4")
LONG_KO = "지금 이 상황에서는 아주 길고 자세하게 충분히 설명하면서 말해야 한다"


def _ws(tmp_path):
    return Workspace(video_id="route-video", data_dir=tmp_path / "data",
                     cache_dir=tmp_path / "cache", log_dir=tmp_path / "log")


def _inputs():
    cue = Cue(id="c0001", start_ms=1000, end_ms=2000, text="Tell me now.", raw="Tell me now.")
    scene = Scene(id="s001", start_ms=1000, end_ms=2000, cue_ids=[cue.id], act=1)
    video = VideoInfo(video_id="route-video", title="T", video_path="video.mkv",
                    duration_ms=3000, audio_stream=0, en_sub_path="en.srt", en_sub_origin="user")
    return video, scene, cue, Snapshot(scene_id=scene.id)


class _Client:
    def __init__(self, settings, model, calls, *, blocked=False, flagged=False, dialogue=False):
        self.settings = settings
        self.model = model
        self.calls = calls
        self.blocked = blocked
        self.flagged = flagged
        self.dialogue = dialogue

    def generate(self, *, stage, prompt, schema, clips=(), images=(), model=None, **kwargs):
        effective = model or self.model
        call = dict(stage=stage, prompt=prompt, schema=schema, clips=list(clips),
                    images=list(images), model=effective, **kwargs)
        self.calls.append(call)
        assert not (effective.startswith("codex/") and clips), "Codex received AV"
        assert "GOLD-SECRET" not in prompt
        if clips and self.blocked:
            self.blocked = False
            raise GeminiBlocked(stage, "safety")
        if schema is SceneTranslation:
            return SceneTranslation(scene_id="s001", mode="untrusted", model="wrong-model",
                cues=[TranslatedCue(cue_id="c0001", ko=self._spoken(LONG_KO), model="wrong-model",
                      needs_media=MediaNeed.VIDEO if self.flagged and not clips and "BOUNDED AV EVIDENCE" not in prompt else MediaNeed.NONE)])
        if schema is PerceptionLedger:
            return PerceptionLedger(cues={
                "c0001": CuePerception(cue_id="c0001", delivery="urgent", confidence=.8,
                                       evidence=["The speaker raises their voice in the clip"]),
                "future": CuePerception(cue_id="future", evidence=["unlisted future fact"]),
            })
        if schema is recheck._PictureEvidence:
            return recheck._PictureEvidence(concerns=[recheck._PictureConcern(
                cue_id="c0001", problem="Urgent delivery", evidence=["Raised voice"], confidence=.8)])
        if schema is review._FlagBatch:
            return schema(flags=[ReadFlag(cue_id="c0001", problem="Unnatural wording")])
        if schema is ContextMap:
            return ContextMap(synopsis="A conversation", tone_baseline="ordinary")
        if schema is Timeline:
            return Timeline()
        if schema is RegisterMap:
            return RegisterMap()
        if "edits" in schema.model_fields:
            text = "이 상황에서는 아주 자세하게 말해야 한다" if stage == "s5-compress" else LONG_KO
            if stage.startswith("picture-check"):
                text = "이 순간에는 아주 자세하고 길게 말해야 한다"
            return schema(edits=[CueEdit(cue_id="c0001", new_ko=self._spoken(text), reason="Grounded correction", kind="picture" if stage.startswith("picture") else "naturalness")])
        if stage == "layout-condense":
            return schema(items=[dict(event_id="e0001", ko="- 말해\n- 응" if self.dialogue else "지금 말해")])
        raise AssertionError(f"Unexpected schema/stage: {schema.__name__}/{stage}")

    def _spoken(self, text):
        return f"- {text}\n- 알았어" if self.dialogue else text


def _primary(settings, model, calls, **kwargs):
    settings = replace(settings, model_main=model, model_light=model,
                       model_translate=model, model_text=model)
    return _Client(settings, model, calls, **kwargs)


def _video_translate(tmp_path, plan, primary, *, av=None, images=None, register_check=None):
    video, scene, cue, snapshot = _inputs()
    return translate.translate_video(
        _ws(tmp_path), primary, video, [scene], [cue], {scene.id: snapshot}, [], PROXY,
        mode=plan.translation_mode.value, text_client_override=primary, av_client=av,
        route_plan=plan, scene_images=images, register_check=register_check,
    )[scene.id]




def test_blocked_argon_retries_only_argon_and_records_degradation(tmp_path):
    calls = []
    client = _primary(Settings(memory_k=0), ARGON, calls, blocked=True)
    st = _video_translate(tmp_path, resolve_route(ARGON), client, av=client)
    assert [call["model"] for call in calls] == [ARGON, ARGON]
    assert calls[0]["clips"] and not calls[1]["clips"]
    assert st.model == st.cues[0].model == ARGON
    assert st.mode == "text" and st.degraded is True and st.degradation_reason
    assert st.cues[0].degraded is True and st.cues[0].degradation_reason
    assert not list((_ws(tmp_path).data_dir / "checkpoints" / "translate").glob("*.json"))


def test_av_two_pass_retains_primary_clip_and_grounded_perception(tmp_path):
    calls = []
    primary = _primary(Settings(memory_k=0), ARGON, calls)
    plan = resolve_route(ARGON, Architecture.AV_TWO_PASS)
    st = _video_translate(tmp_path, plan, primary, av=primary)
    assert [call["schema"] for call in calls] == [PerceptionLedger, SceneTranslation]
    assert all(call["model"] == ARGON and call["clips"] for call in calls)
    assert calls[0]["clips"] == calls[1]["clips"]
    assert "BOUNDED AV EVIDENCE" in calls[1]["prompt"]
    assert "unlisted future fact" not in calls[1]["prompt"]
    assert st.mode == "av" and st.model == st.cues[0].model == ARGON
    _video_translate(tmp_path, plan, primary, av=primary)
    assert len(calls) == 2  # both passes checkpoint independently
    assert len(list((_ws(tmp_path).data_dir / "checkpoints" / "translate").glob("*.json"))) == 2


def test_snapshot_attaches_actual_frame_paths_and_hashes_bytes(tmp_path, monkeypatch):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"actual PNG frame one")
    supplied = []
    def frames(ws, video, scene):
        supplied.append((video.video_id, scene.id))
        return [frame]
    monkeypatch.setattr(translate.media, "scene_keyframes", frames)
    calls = []
    primary = _primary(Settings(memory_k=0), SOL, calls)
    plan = resolve_route(SOL)
    st = _video_translate(tmp_path, plan, primary)
    assert supplied == [("route-video", "s001")]
    assert calls[0]["images"] == [frame] and not calls[0]["clips"]
    assert st.model == SOL
    _video_translate(tmp_path, plan, primary)
    assert len(calls) == 1
    frame.write_bytes(b"actual PNG frame two")
    _video_translate(tmp_path, plan, primary)
    assert len(calls) == 2  # same path with different bytes invalidates the checkpoint


@pytest.mark.parametrize("architecture", [Architecture.DPAF, Architecture.CASCADE])
def test_pure_text_primary_rewrites_native_av_evidence(tmp_path, architecture):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"bounded scene image")
    calls = []
    settings = Settings(memory_k=0)
    primary = _primary(settings, SOL, calls, flagged=True)
    native = _primary(settings, FLASH, calls)
    st = _video_translate(tmp_path, resolve_route(SOL, architecture), primary, av=native,
                         images={"s001": [frame]})
    assert [(call["model"], call["schema"]) for call in calls] == [
        (SOL, SceneTranslation), (FLASH, PerceptionLedger), (SOL, SceneTranslation)]
    assert calls[1]["clips"] and not calls[2]["clips"]
    assert st.mode == "text" and st.model == st.cues[0].model == SOL


def test_explicit_mixed_av_writer_and_primary_register_rewrite_are_truthful(tmp_path):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"scene frame")
    calls = []
    primary = _primary(Settings(memory_k=0), SOL, calls, flagged=True)
    native = _primary(Settings(memory_k=0), FLASH, calls)
    plan = resolve_route(SOL, Architecture.MIXED)
    st = _video_translate(tmp_path / "mixed", plan, primary, av=native, images={"s001": [frame]})
    assert st.model == st.cues[0].model == FLASH and st.mode == "av"
    st = _video_translate(tmp_path / "retry", plan, primary, av=native, images={"s001": [frame]},
                         register_check=lambda st: SimpleNamespace(cue_ids=["c0001"]))
    assert st.model == st.cues[0].model == SOL and st.mode == "text"
    assert calls[-1]["model"] == SOL and not calls[-1]["clips"]


def test_invalid_native_route_fails_before_any_provider_call(tmp_path):
    calls = []
    primary = _primary(Settings(memory_k=0), SOL, calls)
    bad = replace(resolve_route(ARGON), primary_model=SOL)
    with pytest.raises(CapabilityError):
        _video_translate(tmp_path, bad, primary, av=primary)
    assert calls == []


def test_new_anthropic_baseline_generation_is_rejected(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "make_client", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="frozen Anthropic"):
        pipeline._route_clients(_ws(tmp_path), Settings(), resolve_route("anthropic/claude-opus-5-5"))
    assert calls == []


@pytest.mark.parametrize("model", [ARGON, SOL])
@pytest.mark.parametrize("dialogue", [False, True])
def test_final_srt_pipeline_keeps_selected_writer_at_every_stage(tmp_path, monkeypatch, model, dialogue):
    settings = Settings(memory_k=0, max_duration_ms=1000, extend_ms=0)
    calls = []
    ws = _ws(tmp_path)
    video_info, scene, cue, snapshot = _inputs()
    if dialogue:
        cue = cue.model_copy(update={"text": "- Tell me now.\n- Okay.", "raw": "- Tell me now.\n- Okay."})
    video = tmp_path / "video.mkv"
    video.write_bytes(b"offline video")
    frame = tmp_path / "scene.png"
    frame.write_bytes(b"actual extracted frame bytes")
    monkeypatch.setattr(pipeline, "load_secrets", lambda: None)
    monkeypatch.setattr(pipeline, "load_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "workspace", lambda _: ws)
    monkeypatch.setattr(pipeline, "GeminiClient", lambda ws, settings: _Client(settings, settings.model_main, calls, dialogue=dialogue))
    monkeypatch.setattr(pipeline, "make_client", lambda ws, settings, model: _Client(settings, model, calls, dialogue=dialogue))
    monkeypatch.setattr(pipeline, "_ingest", lambda *args: video_info)
    monkeypatch.setattr(pipeline, "_load_cues", lambda *args: [cue])
    monkeypatch.setattr(pipeline, "_proxy_upload", lambda *args: PROXY)
    monkeypatch.setattr(pipeline.media, "scene_keyframes", lambda *args: [frame])
    monkeypatch.setattr(pipeline, "_print_cost", lambda _: None)
    monkeypatch.setattr(pipeline.narrative, "build_context_map", lambda ws, gc, *args: gc.generate(stage="map", prompt="source", schema=ContextMap))
    monkeypatch.setattr(pipeline.narrative, "build_timeline", lambda ws, gc, *args: gc.generate(stage="timeline", prompt="source", schema=Timeline))
    monkeypatch.setattr(pipeline.register_mod, "build_register", lambda ws, gc, *args: gc.generate(stage="register", prompt="source", schema=RegisterMap))
    monkeypatch.setattr(pipeline.narrative, "materialize_snapshots", lambda *args: {scene.id: snapshot})
    monkeypatch.setattr(pipeline.register_mod.RegisterGate, "check", lambda *args, **kwargs: True)
    out = pipeline.run(video, model=model, video_id=video_info.video_id, no_viewing=True)
    expected = "- 말해\n- 응" if dialogue else "지금 말해"
    assert out.is_file()
    srt = out.read_text()
    assert srt.split("\n\n")[0].splitlines()[2:] == expected.splitlines()
    assert LONG_KO not in srt  # the consumer receives the shortened final realization
    assert srt.count("\n- ") == (2 if dialogue else 0)
    stages = {call["stage"] for call in calls}
    for stage in ("s3_translate", "s4_review", "s4b_finale", "s4b_blind", "s4b_regen", "s5-compress", "layout-condense"):
        assert stage in stages
        assert {call["model"] for call in calls if call["stage"] == stage} == {model}
    assert all(not call["model"].startswith("anthropic/") for call in calls)
    picture = [call for call in calls if call["stage"].startswith("picture-check")]
    if model == SOL:
        assert [(call["model"], call["stage"]) for call in picture] == [
            (FLASH, "picture-check:perception"), (SOL, "picture-check:realize")]
        assert picture[0]["clips"] and not picture[1]["clips"]
        assert any(call["images"] == [frame] for call in calls)
    else:
        assert [(call["model"], call["stage"]) for call in picture] == [(ARGON, "picture-check")]
    sidecar = json.loads(ws.path(f"{video_info.video_id}.ko.json").read_text())
    assert sidecar["cues"][0]["model"] == model
    assert sidecar["cues"][0]["cue_id"] == cue.id
    assert sidecar["events"][0]["cue_ids"] == [cue.id]
    assert sidecar["events"][0]["ko"] == expected
    route = json.loads(ws.path("route.json").read_text())
    assert route["final_events"][0]["writer_models"] == [model]


def test_strict_c7_block_is_visible_without_substitute_writer(tmp_path):
    _, scene, cue, snapshot = _inputs()
    calls = []
    native = _primary(Settings(memory_k=0), FLASH, calls, blocked=True)
    primary = _primary(Settings(memory_k=0), SOL, calls)
    with pytest.raises(GeminiBlocked):
        recheck.picture_check(
            _ws(tmp_path), native, PROXY, {cue.id: cue}, {cue.id: "지금 말해"},
            {cue.id}, {scene.id: snapshot}, {cue.id: scene.id},
            writer=primary, strict=True,
        )
    assert len(calls) == 1 and calls[0]["model"] == FLASH


@pytest.mark.parametrize("model,error", [
    ("anthropic/claude-opus-5-5", ValueError),
    (ARGON, RuntimeError),
])
def test_rejected_or_unavailable_route_preserves_existing_published_artifact(tmp_path, monkeypatch, model, error):
    ws = _ws(tmp_path)
    route_path = ws.path("route.json")
    published = b'{"model":"gemini-4-argon","architecture":"direct-av","legacy":false}\n'
    route_path.write_bytes(published)
    video = tmp_path / "video.mkv"
    video.write_bytes(b"offline video")
    constructors = []
    monkeypatch.setattr(pipeline, "load_secrets", lambda: None)
    monkeypatch.setattr(pipeline, "load_settings", lambda: Settings(memory_k=0))
    monkeypatch.setattr(pipeline, "workspace", lambda _: ws)
    def native_client(ws, settings):
        constructors.append("default-native")
        return _Client(settings, settings.model_main, [])
    def unavailable_client(ws, settings, model):
        constructors.append(model)
        raise RuntimeError("offline client unavailable")
    monkeypatch.setattr(pipeline, "GeminiClient", native_client)
    monkeypatch.setattr(pipeline, "make_client", unavailable_client)
    with pytest.raises(error):
        pipeline.run(video, video_id=ws.video_id, model=model, no_viewing=True)
    assert route_path.read_bytes() == published
    assert "default-native" not in constructors


class _PartialRouteClient:
    """Faithful provider boundary for a one-time native refusal on one cue."""

    def __init__(self, settings, model, calls, state):
        self.settings, self.model = settings, model
        self.calls, self.state = calls, state

    def generate(self, *, stage, prompt, schema, clips=(), images=(), model=None, **kwargs):
        scene = re.search(r"Scene (s\d+)", prompt)
        ids = re.findall(r"^  (c\d+) \|", prompt, re.MULTILINE)
        self.calls.append({
            "stage": stage, "schema": schema, "scene": scene.group(1) if scene else None,
            "ids": ids, "clips": list(clips), "model": model or self.model,
        })
        if clips and self.state["block_native"]:
            self.state["block_native"] = False
            raise GeminiBlocked(stage, "bounded clip refused")
        if schema is SceneTranslation:
            ko = {"c0001": "안녕", "c0002": "문 닫아", "c0003": "고마워"}
            return SceneTranslation(
                scene_id=scene.group(1), mode="untrusted",
                cues=[TranslatedCue(
                    cue_id=cid, ko=ko[cid],
                    needs_media=MediaNeed.VIDEO if cid == "c0002" and "BOUNDED AV EVIDENCE" not in prompt else MediaNeed.NONE,
                ) for cid in ids],
            )
        if schema is PerceptionLedger:
            return PerceptionLedger(cues={
                cid: CuePerception(cue_id=cid, confidence=.9, evidence=["The indicated door is visible"])
                for cid in ids
            })
        if schema is ContextMap:
            return ContextMap(synopsis="Three lines of dialogue", tone_baseline="ordinary")
        if schema is Timeline:
            return Timeline()
        if schema is RegisterMap:
            return RegisterMap()
        return schema()


def test_full_pipeline_partial_fallback_is_visible_and_only_degraded_window_retries(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    settings = Settings(memory_k=0)
    calls, state = [], {"block_native": True}
    cues = [
        Cue(id="c0001", start_ms=1000, end_ms=2000, text="Hello.", raw="Hello."),
        Cue(id="c0002", start_ms=7000, end_ms=8000, text="Close that door.", raw="Close that door."),
        Cue(id="c0003", start_ms=11000, end_ms=12000, text="Thank you.", raw="Thank you."),
    ]
    scenes = [
        Scene(id="s001", start_ms=1000, end_ms=8000, cue_ids=["c0001", "c0002"], act=1),
        Scene(id="s002", start_ms=11000, end_ms=12000, cue_ids=["c0003"], act=2),
    ]
    video = tmp_path / "video.mkv"
    video.write_bytes(b"offline frozen video")
    frame = tmp_path / "scene.png"
    frame.write_bytes(b"frozen frame bytes")
    video_info = VideoInfo(video_id=ws.video_id, title="T", video_path=str(video),
                    duration_ms=16000, audio_stream=0, en_sub_path="en.srt", en_sub_origin="user")
    monkeypatch.setattr(pipeline, "load_secrets", lambda: None)
    monkeypatch.setattr(pipeline, "load_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "workspace", lambda _: ws)
    monkeypatch.setattr(pipeline, "GeminiClient", lambda ws, settings: _PartialRouteClient(settings, settings.model_main, calls, state))
    monkeypatch.setattr(pipeline, "make_client", lambda ws, settings, model: _PartialRouteClient(settings, model, calls, state))
    monkeypatch.setattr(pipeline, "_ingest", lambda *args: video_info)
    monkeypatch.setattr(pipeline, "_load_cues", lambda *args: cues)
    monkeypatch.setattr(pipeline.scenes_mod, "group_scenes", lambda *args, **kwargs: scenes)
    monkeypatch.setattr(pipeline, "_proxy_upload", lambda *args: PROXY)
    monkeypatch.setattr(pipeline.media, "scene_keyframes", lambda *args: [frame])
    monkeypatch.setattr(pipeline, "_print_cost", lambda _: None)
    monkeypatch.setattr(pipeline.narrative, "build_context_map", lambda ws, gc, *args: gc.generate(stage="map", prompt="source", schema=ContextMap))
    monkeypatch.setattr(pipeline.narrative, "build_timeline", lambda ws, gc, *args: gc.generate(stage="timeline", prompt="source", schema=Timeline))
    monkeypatch.setattr(pipeline.register_mod, "build_register", lambda ws, gc, *args: gc.generate(stage="register", prompt="source", schema=RegisterMap))
    monkeypatch.setattr(pipeline.narrative, "materialize_snapshots", lambda *args: {
        scene.id: Snapshot(scene_id=scene.id) for scene in scenes
    })
    checkpoint_hashes = {}
    real_ck = pipeline._ck
    def capture_checkpoint(ck, key, h, cls, make, **kwargs):
        checkpoint_hashes[key] = h
        return real_ck(ck, key, h, cls, make, **kwargs)
    monkeypatch.setattr(pipeline, "_ck", capture_checkpoint)
    kwargs = dict(video_id=ws.video_id, model=SOL, architecture=Architecture.CASCADE, no_viewing=True)
    srt = pipeline.run(video, **kwargs)
    assert all(text in srt.read_text() for text in ("안녕", "문 닫아", "고마워"))
    sidecar = json.loads(ws.path(f"{ws.video_id}.ko.json").read_text())
    by_id = {cue["cue_id"]: cue for cue in sidecar["cues"]}
    assert set(by_id) == {cue.id for cue in cues}
    assert by_id["c0002"]["degraded"] is True and by_id["c0002"]["degradation_reason"]
    assert by_id["c0001"]["degraded"] is False and by_id["c0003"]["degraded"] is False
    assert all(cue["model"] == SOL for cue in by_id.values())
    route = json.loads(ws.path("route.json").read_text())
    for event in route["final_events"]:
        affected = ["c0002"] if "c0002" in event["cue_ids"] else []
        assert event["degraded_cue_ids"] == affected
        assert event["degraded"] is bool(affected)
        assert set(event["degradation_reasons"]) == set(affected)
    ck = Checkpoints(ws, "pipeline")
    assert ck.get("translations", checkpoint_hashes["translations"], pipeline._Translations) is None

    # Simulate the old whole-video cache with only scene-level degradation.
    old_scenes = [
        SceneTranslation.model_validate(row).model_copy(update={
            "cues": [TranslatedCue.model_validate(cue).model_copy(update={
                "degraded": None, "degradation_reason": None,
            }) for cue in row["cues"]],
        })
        for row in json.loads(ws.path("translations.json").read_text())
    ]
    ck.put("translations", checkpoint_hashes["translations"], pipeline._Translations(items=old_scenes))
    first_end = len(calls)
    pipeline.run(video, **kwargs)
    new_writes = [call for call in calls[first_end:] if call["schema"] is SceneTranslation]
    assert len(new_writes) == 1 and new_writes[0]["scene"] == "s001"
    assert new_writes[0]["model"] == SOL and not new_writes[0]["clips"]
    assert any(call["schema"] is PerceptionLedger for call in calls[first_end:])
    repaired = json.loads(ws.path(f"{ws.video_id}.ko.json").read_text())
    assert all(cue["degraded"] is False and cue["degradation_reason"] is None for cue in repaired["cues"])
    assert all(not event["degraded_cue_ids"] for event in json.loads(ws.path("route.json").read_text())["final_events"])
    healthy = ck.get("translations", checkpoint_hashes["translations"], pipeline._Translations)
    assert healthy is not None and not any(scene.degraded for scene in healthy.items)
    second_end = len(calls)
    pipeline.run(video, **kwargs)
    assert not any(call["schema"] in (SceneTranslation, PerceptionLedger) for call in calls[second_end:])
