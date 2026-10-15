"""Offline assertions for executed routes, not just architecture labels."""
from __future__ import annotations

import base64
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from substage import ablate, capability, cli, experiments, media
from substage.config import Settings, Workspace
from substage.gemini import Clip, GeminiBlocked, UploadedFile
from substage.models import Cue, CuePerception, VideoInfo, PerceptionLedger, Scene, SceneTranslation, Snapshot, TranslatedCue, UsageRecord
from substage.store import save_model


class RecordingClient:
    def __init__(self, ws, settings, model, calls):
        self.ws, self.settings, self.model, self.calls = ws, settings, model, calls

    def upload(self, path, mime):
        return UploadedFile(name="frozen", uri="offline://same-proxy", mime_type=mime)

    def generate(self, *, stage, prompt, schema, model, thinking_level, clips=(), images=(), **kwargs):
        self.calls.append({"stage": stage, "model": model, "effort": thinking_level,
                           "schema": schema, "clips": clips, "images": list(images), "prompt": prompt})
        ids = list(dict.fromkeys(re.findall(r"\b(c\d+)\b", prompt)))
        if schema is PerceptionLedger:
            return PerceptionLedger(cues={cid: CuePerception(cue_id=cid, confidence=0.8,
                                                             evidence=["bounded scene evidence"]) for cid in ids})
        if schema is SceneTranslation:
            sid = re.search(r"Scene (s\d+)", prompt).group(1)
            source_ids = re.findall(r"^  (c\d+) \|", prompt, re.MULTILINE)
            return SceneTranslation(scene_id=sid, mode="av" if clips else "text", cues=[
                TranslatedCue(cue_id=cid, ko="저걸 봐요", needs_media="none" if clips or "BOUNDED AV EVIDENCE" in prompt else "video")
                for cid in source_ids
            ])
        return schema.model_validate({"flagged_cue_ids": ids})


def fixture(root: Path, n=1):
    ws = Workspace(video_id="frozen-video", data_dir=root / "data", cache_dir=root / "cache", log_dir=root / "logs")
    cues = [Cue(id=f"c{i:03}", start_ms=i * 10_000, end_ms=i * 10_000 + 2_000, text="Look at that.", raw="") for i in range(n)]
    scenes = [Scene(id=f"s{i:03}", start_ms=c.start_ms, end_ms=c.end_ms, cue_ids=[c.id]) for i, c in enumerate(cues)]
    proxy = root / "frozen.mp4"
    proxy.write_bytes(b"offline frozen media")
    original = root / "original-video.mp4"
    original.write_bytes(b"different original video bytes")
    video = VideoInfo(video_id=ws.video_id, title="Frozen", video_path=str(original), duration_ms=n * 10_000,
                    audio_stream=1, en_sub_path="source.srt", en_sub_origin="user", proxy_path=str(proxy))
    save_model(ws.path("cues.json"), cues)
    save_model(ws.path("scenes.json"), scenes)
    save_model(ws.path("video.json"), video)
    save_model(ws.path("snapshots.json"), [Snapshot(scene_id=s.id) for s in scenes])
    ref = root / "gold.srt"
    ref.write_text("1\n00:00:00,000 --> 00:00:02,000\n저걸 봐요\n", encoding="utf-8")
    return ws, ref, cues, scenes


@pytest.fixture
def frames(tmp_path, monkeypatch):
    path = tmp_path / "bounded.png"
    path.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jF1sAAAAASUVORK5CYII="
    ))
    monkeypatch.setattr(media, "scene_keyframes", lambda *args, **kwargs: [path])
    return path


def execute(tmp_path, architecture, *, model="gemini-4-argon", roles=None, frames=None, n=1):
    ws, ref, cues, scenes = fixture(tmp_path, n=n)
    calls = []
    cfg = ablate.parse_config(f"candidate:model={model},architecture={architecture},thinking=high,memory=0")
    report = ablate.run_ablation(ws, [cfg], ref, n, 7, settings=Settings(memory_k=0), roles=roles,
                                client_factory=lambda w, s, m: RecordingClient(w, s, m, calls))
    return report, calls, ws


def test_native_auto_and_fake_text_defaults():
    assert ablate.parse_config("native:model=gemini-4-argon").route.architecture is capability.Architecture.DIRECT_AV
    fake = ablate.parse_config("fake:model=offline-fixture")
    assert fake.mode == "text" and fake.route is None
    assert ablate.parse_config("av2:model=gemini-4-argon,architecture=av-two-pass").mode == "av"
    assert ablate.parse_config("mixed:model=codex/gpt-6.1-sol,architecture=mixed").mode == "mixed"


@pytest.mark.parametrize("architecture,schemas", [
    ("direct-av", [SceneTranslation]),
    ("av-two-pass", [PerceptionLedger, SceneTranslation]),
])
def test_native_calls_and_artifact_provenance(tmp_path, architecture, schemas):
    report, calls, _ = execute(tmp_path, architecture)
    assert [call["schema"] for call in calls] == schemas
    assert all(call["model"] == "gemini-4-argon" and call["effort"] == "high" and call["clips"] for call in calls)
    if architecture == "av-two-pass":
        assert calls[0]["clips"] == calls[1]["clips"]
        assert "BOUNDED AV EVIDENCE" in calls[1]["prompt"]
    artifact = report["artifacts"]["candidate"]
    assert Path(artifact["srt"]).is_file()
    assert "저걸 봐요" in Path(artifact["srt"]).read_text(encoding="utf-8")
    sidecar = json.loads(Path(artifact["sidecar"]).read_text(encoding="utf-8"))
    assert sidecar["cues"][0]["model"] == "gemini-4-argon"
    assert sidecar["events"][0]["cue_ids"] == ["c000"]
    assert artifact["layout"]["events"] == 1 and artifact["full_video_review"] is False
    completed = artifact["stage_scope"]["current_run_completed_substeps"]
    assert [step["schema"] for step in completed] == [schema.__name__ for schema in schemas]
    assert all(step["clips"] == len(call["clips"]) for step, call in zip(completed, calls, strict=True))


def test_snapshot_passes_real_paths_and_invalidates_changed_frame_bytes(tmp_path, frames):
    report, calls, ws = execute(tmp_path, "text-snapshot", model="codex/gpt-6.1-sol")
    assert len(calls) == 1
    assert calls[0]["images"] == [frames] and calls[0]["clips"] == []
    assert report["configs"]["candidate"]["roles"]["translation"] == "codex/gpt-6.1-sol"
    assert report["configs"]["candidate"]["frozen_inputs"]["roles"] == {
        "translation": "codex/gpt-6.1-sol", "fallback": "codex/gpt-6.1-sol",
    }
    assert {call["model"] for call in calls} == {"codex/gpt-6.1-sol"}
    cfg = ablate.parse_config("candidate:model=codex/gpt-6.1-sol,architecture=text-snapshot,thinking=high,memory=0")
    base = ablate.load_base(ws)
    calls.clear()
    factory = lambda w, s, m: RecordingClient(w, s, m, calls)
    ablate.translate_configs(ws, Settings(memory_k=0), base, base.scenes, [cfg], factory)
    assert calls == []
    frames.write_bytes(frames.read_bytes() + b"changed")
    ablate.translate_configs(ws, Settings(memory_k=0), base, base.scenes, [cfg], factory)
    assert len(calls) == 1 and calls[0]["images"] == [frames]


@pytest.mark.parametrize("architecture", ["dpaf", "cascade", "mixed"])
def test_hybrid_roles_execute_and_pure_routes_keep_selected_writer(tmp_path, frames, architecture):
    roles = {"perception": "gemini-4-argon", "router": "codex/gpt-6-luna"}
    report, calls, _ = execute(tmp_path, architecture, model="codex/gpt-6.1-sol", roles=roles)
    assert any(c["model"] == "codex/gpt-6-luna" and c["schema"] is not SceneTranslation for c in calls)
    assert any(c["model"] == "gemini-4-argon" and c["clips"] for c in calls)
    writers = [c for c in calls if c["schema"] is SceneTranslation]
    expected_writer = "gemini-4-argon" if architecture == "mixed" else "codex/gpt-6.1-sol"
    assert writers[-1]["model"] == expected_writer
    assert all(not c["model"].startswith("anthropic/") for c in calls)
    sidecar = json.loads(Path(report["artifacts"]["candidate"]["sidecar"]).read_text(encoding="utf-8"))
    assert sidecar["cues"][0]["model"] == expected_writer
    assert report["configs"]["candidate"]["model"] == "codex/gpt-6.1-sol"
    if architecture != "dpaf":
        assert writers[0]["images"] == [frames]


def test_same_frozen_thirty_scenes_execute_av2_and_mixed(tmp_path, frames):
    ws, ref, cues, scenes = fixture(tmp_path, n=30)
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"video_id": ws.video_id, "n_scenes": 30,
                                  "scenes": [s.model_dump() for s in scenes]}), encoding="utf-8")
    calls = []
    configs = [ablate.parse_config("native:model=gemini-4-argon,architecture=av-two-pass,memory=0"),
               ablate.parse_config("team:model=codex/gpt-6.1-sol,architecture=mixed,memory=0")]
    report = ablate.run_ablation(ws, configs, ref, 1, 999, sample_from=frozen, settings=Settings(memory_k=0),
                                client_factory=lambda w, s, m: RecordingClient(w, s, m, calls))
    assert [row["id"] for row in report["scenes"]] == [s.id for s in scenes]
    assert report["n_scenes"] == 30 and report["sample_from"] == str(frozen)
    assert len(report["artifacts"]["native"]["scenes"]) == len(report["artifacts"]["team"]["scenes"]) == 30


@pytest.mark.parametrize("spec,roles", [
    ("bad:model=codex/gpt-6.1-sol,architecture=direct-av", None),
    ("bad:model=codex/gpt-6.1-sol,architecture=text-snapshot", {"router": "codex/gpt-6-luna"}),
    ("bad:model=codex/gpt-6.1-sol,architecture=mixed", {"perception": "codex/gpt-6-luna"}),
    ("bad:model=codex/gpt-6.1-sol,architecture=mixed", {"perception": "gemini-4-argon", "av": "gemini-3.8-flash"}),
    ("bad:model=anthropic/claude-opus-5-5,architecture=text-snapshot", None),
])
def test_unsupported_routes_and_unused_roles_fail_before_client(tmp_path, spec, roles):
    ws, ref, _, _ = fixture(tmp_path)
    calls = []
    with pytest.raises(ValueError):
        cfg = ablate.parse_config(spec)
        ablate.run_ablation(ws, [cfg], ref, 1, 1, roles=roles, settings=Settings(),
                            client_factory=lambda *args: calls.append(args))
    assert calls == []


def test_global_architecture_and_effort_are_executed(tmp_path):
    ws, ref, _, _ = fixture(tmp_path)
    calls = []
    report = ablate.run_ablation(ws, [ablate.parse_config("actual:model=gemini-4-argon")], ref, 1, 2,
                                declared_architecture="av-two-pass", effort="xhigh", settings=Settings(memory_k=0),
                                client_factory=lambda w, s, m: RecordingClient(w, s, m, calls))
    assert [c["schema"] for c in calls] == [PerceptionLedger, SceneTranslation]
    assert all(c["effort"] == "xhigh" for c in calls)
    assert report["configs"]["actual"]["thinking"] == "xhigh"


def test_usage_roles_are_separate_from_evaluator():
    rows = [UsageRecord(stage=stage, model=model, input_tokens=1, output_tokens=2, cost_usd=0.1)
            for stage, model in [("ablate:a:translation:s3", "codex/gpt-6.1-sol"),
                                 ("ablate:a:av:s3:perception", "gemini-4-argon"),
                                 ("ablate:a:router:s3:luna", "codex/gpt-6-luna"),
                                 ("refeval:judge", "gemini-4-argon"),
                                 ("ablate:ab:translation:s3", "codex/gpt-6.1-sol")]]
    selected = experiments.select_usage(experiments.summarize_usage(rows), "ablate:a")
    assert selected.records == 3 and len(selected.per_model) == 3
    assert experiments.usage_totals(selected).cost_usd == pytest.approx(0.3)


def test_completed_cycle_does_not_claim_quality_gain():
    cycle = experiments.CycleState(completed=True, actual_improvement=False, hypothesis="h", change="c",
                                   translation="t", score="s", debate="d", decision="continue")
    assert experiments.actual_improvement_completed(cycle)
    assert cycle.quality_change == "unknown"
    assert experiments.can_pause_for_no_improvement([cycle] * 12, minimum_cycles=12)
    assert not experiments.can_pause_for_no_improvement([cycle] * 11, minimum_cycles=12)


def test_cli_run_exposes_av_two_pass_and_mixed(monkeypatch):
    from substage import pipeline
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: calls.append(kwargs) or Path("output"))
    for architecture in ("av-two-pass", "mixed"):
        assert cli.main(["run", "video.mp4", "--model", "gemini-4-argon", "--architecture", architecture]) == 0
    assert [call["architecture"] for call in calls] == ["av-two-pass", "mixed"]


def test_undigested_av_results_remain_diagnostic_and_new_strict_media_cache_is_reusable(tmp_path):
    ws, ref, _, _ = fixture(tmp_path)
    cfg = ablate.parse_config("candidate:model=gemini-4-argon,architecture=direct-av,memory=0")
    old = ablate._checkpoint(ws, cfg, "s000")
    old.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({"input_hash": "historical", "translation": {
        "scene_id": "s000", "mode": "av", "cues": [{"cue_id": "c000", "ko": "진단 자료"}],
    }}).encode()
    old.write_bytes(original)
    calls = []
    factory = lambda w, s, m: RecordingClient(w, s, m, calls)
    report = ablate.run_ablation(ws, [cfg], ref, 1, 1, settings=Settings(memory_k=0), client_factory=factory)
    assert len(calls) == 1 and old.read_bytes() == original
    assert report["configs"]["candidate"]["previous_undigested_results"]["s000"]["eligible_for_strict_reuse"] is False
    identity = report["configs"]["candidate"]["frozen_inputs"]["media_identity"]
    assert identity["sha256"] == experiments.file_digest(tmp_path / "frozen.mp4")
    calls.clear()
    ablate.run_ablation(ws, [cfg], ref, 1, 1, settings=Settings(memory_k=0), client_factory=factory)
    assert calls == [] and old.read_bytes() == original
    (tmp_path / "frozen.mp4").write_bytes(b"different media")
    ablate.run_ablation(ws, [cfg], ref, 1, 1, settings=Settings(memory_k=0), client_factory=factory)
    assert len(calls) == 1 and old.read_bytes() == original


@pytest.mark.parametrize("spec", [
    "missing:thinking=low",
    "invalid:model=codex/gpt-6.1-sol,thinking=max",
    "unsupported:model=codex/gpt-6.1-sol,mode=av",
])
def test_parser_rejects_missing_model_invalid_effort_and_unsupported_media(spec):
    with pytest.raises(ValueError):
        ablate.parse_config(spec)


@pytest.mark.parametrize("preexisting_degraded_checkpoint", [False, True])
def test_blocked_native_av_preserves_sidecar_degradation_and_retries_fresh(
    tmp_path, preexisting_degraded_checkpoint,
):
    ws, ref, _, _ = fixture(tmp_path)
    cfg = ablate.parse_config("candidate:model=gemini-4-argon,architecture=direct-av,memory=0")
    calls = []
    blocked = {"value": True}

    class BlockedAVClient(RecordingClient):
        def generate(self, **kwargs):
            if blocked["value"] and kwargs.get("clips"):
                calls.append({"model": kwargs["model"], "clips": kwargs["clips"],
                              "schema": kwargs["schema"], "stage": kwargs["stage"]})
                raise GeminiBlocked(kwargs["stage"], "blocked bounded scene")
            return super().generate(**kwargs)

    factory = lambda w, s, m: BlockedAVClient(w, s, m, calls)

    def run():
        return ablate.run_ablation(ws, [cfg], ref, 1, 1, settings=Settings(memory_k=0),
                                   client_factory=factory)

    first = run()
    strict = ablate._checkpoint(ws, cfg, "s000").with_suffix(".strict.json")
    diagnostic = strict.with_suffix(".degraded.json")
    assert not strict.exists() and diagnostic.is_file()
    assert len(calls) == 2 and all(call["model"] == "gemini-4-argon" for call in calls)
    assert calls[0]["clips"] and not calls[1]["clips"]
    scene = first["artifacts"]["candidate"]["scenes"]["s000"]
    reason = scene["degradation_reason"]
    assert scene["degraded"] is True and reason and scene["mode"] == "text"
    scope_calls = first["artifacts"]["candidate"]["stage_scope"]["current_run_provider_calls"]
    assert [call["status"] for call in scope_calls] == ["failed", "completed"]
    assert scope_calls[0]["clips"] == 1 and scope_calls[1]["clips"] == 0
    assert scope_calls[0]["error"] == "GeminiBlocked"
    sidecar = json.loads(Path(first["artifacts"]["candidate"]["sidecar"]).read_text(encoding="utf-8"))
    assert sidecar["scenes"][0]["degraded"] is True
    assert sidecar["scenes"][0]["degradation_reason"] == reason
    assert sidecar["cues"][0]["model"] == "gemini-4-argon"
    assert sidecar["cues"][0]["degraded"] is True
    assert sidecar["cues"][0]["degradation_reason"] == reason

    if preexisting_degraded_checkpoint:
        strict.write_bytes(diagnostic.read_bytes())
        previous_strict = strict.read_bytes()
    calls.clear()
    second = run()
    assert len(calls) == 2 and calls[0]["clips"] and not calls[1]["clips"]
    assert all(call["model"] == "gemini-4-argon" for call in calls)
    assert second["artifacts"]["candidate"]["scenes"]["s000"]["degraded"] is True
    sidecar = json.loads(Path(second["artifacts"]["candidate"]["sidecar"]).read_text(encoding="utf-8"))
    assert sidecar["scenes"][0]["degradation_reason"] == reason
    assert sidecar["cues"][0]["degraded"] is True
    assert sidecar["cues"][0]["degradation_reason"] == reason
    if preexisting_degraded_checkpoint:
        assert strict.read_bytes() == previous_strict
    else:
        assert not strict.exists()

    blocked["value"] = False
    calls.clear()
    recovered = run()
    assert len(calls) == 1 and calls[0]["clips"] and strict.is_file()
    assert recovered["artifacts"]["candidate"]["scenes"]["s000"]["degraded"] is False
    sidecar = json.loads(Path(recovered["artifacts"]["candidate"]["sidecar"]).read_text(encoding="utf-8"))
    assert sidecar["scenes"][0]["degraded"] is False
    assert sidecar["scenes"][0]["degradation_reason"] is None
    assert sidecar["cues"][0]["degraded"] is False
    assert sidecar["cues"][0]["degradation_reason"] is None
    calls.clear()
    run()
    assert calls == []


def test_snapshot_freezes_actual_proxy_source_when_original_is_unavailable(tmp_path, frames):
    ws, ref, _, _ = fixture(tmp_path)
    (tmp_path / "original-video.mp4").unlink()
    cfg = ablate.parse_config("candidate:model=codex/gpt-6.1-sol,architecture=text-snapshot,memory=0")
    calls = []
    report = ablate.run_ablation(ws, [cfg], ref, 1, 1, settings=Settings(memory_k=0),
                                client_factory=lambda w, s, m: RecordingClient(w, s, m, calls))
    identity = report["configs"]["candidate"]["frozen_inputs"]["media_identity"]
    assert identity["path"] == str((tmp_path / "frozen.mp4").resolve())
    assert identity["sha256"] == experiments.file_digest(tmp_path / "frozen.mp4")
    assert calls[0]["images"] == [frames]
    assert calls[0]["model"] == "codex/gpt-6.1-sol"


@pytest.mark.parametrize("changed_input", ["cues", "scenes"])
def test_frozen_provenance_rejects_changed_source_bytes_before_clients(tmp_path, changed_input):
    ws, ref, cues, scenes = fixture(tmp_path)
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({
        "video_id": ws.video_id, "n_scenes": 1, "scenes": [scene.model_dump() for scene in scenes],
        "provenance": {name: {"sha256": experiments.file_digest(ws.data_dir / f"{name}.json")}
                       for name in ("cues", "scenes")},
    }), encoding="utf-8")
    if changed_input == "cues":
        save_model(ws.path("cues.json"), [cues[0].model_copy(update={"text": "Look at this."})])
    else:
        save_model(ws.path("scenes.json"), [scenes[0].model_copy(update={"act": 2})])
    calls = []
    cfg = ablate.parse_config("candidate:model=gemini-4-argon,architecture=direct-av,memory=0")
    with pytest.raises(ValueError, match=f"{changed_input} SHA256 mismatch"):
        ablate.run_ablation(ws, [cfg], ref, 1, 1, sample_from=sample, settings=Settings(memory_k=0),
                            client_factory=lambda *args: calls.append(args))
    assert calls == []


@pytest.mark.parametrize("changed_spec", ["padding", "fps", "media_resolution"])
def test_strict_native_checkpoint_binds_actual_request_spec(tmp_path, monkeypatch, changed_spec):
    from substage import translate

    ws, ref, _, _ = fixture(tmp_path)
    calls = []
    cfg = ablate.parse_config("candidate:model=gemini-4-argon,architecture=direct-av,memory=0")
    settings = Settings(memory_k=0)
    factory = lambda w, s, m: RecordingClient(w, s, m, calls)
    first = ablate.run_ablation(ws, [cfg], ref, 1, 1, settings=settings, client_factory=factory)
    before = first["configs"]["candidate"]["frozen_inputs"]["request_specs"]["s000"]
    assert before["clip"]["start_s"] == calls[0]["clips"][0].start_s
    assert before["clip"]["end_s"] == calls[0]["clips"][0].end_s
    assert before["clip"]["fps"] == calls[0]["clips"][0].fps
    calls.clear()
    ablate.run_ablation(ws, [cfg], ref, 1, 1, settings=settings, client_factory=factory)
    assert calls == []
    if changed_spec == "padding":
        monkeypatch.setattr(translate, "CLIP_PAD_S", translate.CLIP_PAD_S + 0.5)
    elif changed_spec == "fps":
        monkeypatch.setattr(Clip.__init__, "__defaults__", (2.0,))
    else:
        settings = replace(settings, media_resolution="high")
    changed = ablate.run_ablation(ws, [cfg], ref, 1, 1, settings=settings, client_factory=factory)
    assert len(calls) == 1 and calls[0]["clips"]
    after = changed["configs"]["candidate"]["frozen_inputs"]["request_specs"]["s000"]
    assert before != after
    assert after["clip"]["start_s"] == calls[0]["clips"][0].start_s
    assert after["clip"]["end_s"] == calls[0]["clips"][0].end_s
    assert after["clip"]["fps"] == calls[0]["clips"][0].fps
    assert after["media_resolution"] == settings.media_resolution
