"""Offline tests for S3 ablations (ablate.py). No model calls, no network."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from substage import ablate, translate
from substage.config import Settings, Workspace
from substage.models import (
    Callback,
    Character,
    CharacterState,
    ChunkObservations,
    Cue,
    CueObservation,
    ContextMap,
    Scene,
    SceneRegister,
    SceneTranslation,
    Snapshot,
    TranslatedCue,
    VoiceCard,
)
from substage.store import save_model

N_SCENES = 10


def _ws(root: Path) -> Workspace:
    return Workspace(video_id="video", data_dir=root / "data", cache_dir=root / "cache", log_dir=root / "log")


def _video(root: Path) -> Workspace:
    """10 scenes x 2 cues, 10 s apart; scenes s003 and s007 hold callback cues."""
    ws = _ws(root)
    cues, scenes = [], []
    for i in range(N_SCENES):
        base = i * 10_000
        ids = [f"c{i:02d}a", f"c{i:02d}b"]
        cues += [
            Cue(id=ids[0], start_ms=base, end_ms=base + 2000, text=f"Hello {i}.", raw=""),
            Cue(id=ids[1], start_ms=base + 3000, end_ms=base + 5000, text=f"Bye {i}.", raw=""),
        ]
        scenes.append(Scene(id=f"s{i:03d}", start_ms=base, end_ms=base + 5000, cue_ids=ids))
    save_model(ws.path("cues.json"), cues)
    save_model(ws.path("scenes.json"), scenes)
    save_model(ws.path("snapshots.json"), [
        Snapshot(scene_id=s.id, registers={"a>b": SceneRegister(level="haeyo")}, character_names={"a": "에이"})
        for s in scenes
    ])
    save_model(ws.path("observations.json"), [
        ChunkObservations(chunk_id="v01", start_ms=0, end_ms=100_000,
                          cues=[CueObservation(cue_id=c.id, delivery="SARCASTIC_TONE") for c in cues]),
    ])
    save_model(ws.path("context_map.yaml"), ContextMap(
        synopsis="x", tone_baseline="y",
        callbacks=[Callback(motif="m", setup_cues=["c03a"], payoff_cues=["c07b"])],
    ))
    return ws


def _official(path: Path) -> Path:
    """One gold line per English cue across the whole video."""
    blocks = []
    n = 0
    for i in range(N_SCENES):
        for off, word in ((0, "안녕"), (3000, "잘가")):
            n += 1
            s, e = i * 10_000 + off, i * 10_000 + off + 2000
            blocks.append(f"{n}\n{_ts(s)} --> {_ts(e)}\n{word} {i}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def _ts(ms: int) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class FakeClient:
    """Translates every cue of the prompted scene as '<tag> <n>요.'"""

    def __init__(self, tag: str, log: list):
        self.tag = tag
        self.log = log

    def generate(self, *, stage, prompt, schema, model=None, thinking_level=None, **kw):
        assert schema is SceneTranslation
        self.log.append({"stage": stage, "model": model, "thinking": thinking_level, "prompt": prompt,
                         "system": kw.get("system")})
        sid = re.match(r"Scene (\S+) ", prompt).group(1)
        n = int(sid[1:])
        cues = [TranslatedCue(cue_id=f"c{n:02d}{x}", ko=f"{self.tag} {n}요.") for x in "ab"]
        return SceneTranslation(scene_id=sid, mode="text", cues=cues)


def _factory(log: list):
    return lambda ws, settings, model: FakeClient(model, log)


# ---------------------------------------------------------------- configs




def test_load_base_backfills_video_facts_on_old_snapshots(tmp_path):
    """Snapshots saved before tone/rating/personas existed get them from the video map."""
    ws = _video(tmp_path)
    save_model(ws.path("context_map.yaml"), ContextMap(
        synopsis="x", tone_baseline="y", rating="R",
        characters=[Character(id="a", en_name="A", ko_name="에이", persona="insolent merc",
                              voice=VoiceCard(idiolect="i"))],
    ))
    snap = ablate.load_base(ws).snapshots["s000"]
    assert (snap.tone_baseline, snap.rating, snap.personas) == ("y", "R", {"a": "insolent merc | talks: i"})


# ---------------------------------------------------------------- sampling


def test_sampling_deterministic_and_stratified(tmp_path):
    ws = _video(tmp_path)
    base = ablate.load_base(ws)
    a = ablate.sample_scenes(base.scenes, base.cues, base.context_map, base.register, 4, seed=3)
    b = ablate.sample_scenes(base.scenes, base.cues, base.context_map, base.register, 4, seed=3)
    assert [(s.id, r) for s, r in a] == [(s.id, r) for s, r in b]
    assert {s.id for s, r in a if r == "targeted"} == {"s003", "s007"}
    assert [s.start_ms for s, _ in a] == sorted(s.start_ms for s, _ in a)
    seeds = {tuple(s.id for s, _ in ablate.sample_scenes(base.scenes, base.cues, base.context_map, base.register, 4, seed=k))
             for k in range(10)}
    assert len(seeds) > 1
    assert len(ablate.sample_scenes(base.scenes, base.cues, base.context_map, base.register, 99, seed=1)) == N_SCENES


def test_scene_runs_follow_video_order():
    scenes = [Scene(id=f"s{i}", start_ms=i, end_ms=i + 1, cue_ids=[]) for i in range(6)]
    runs = ablate.scene_runs(scenes, {"s1", "s2", "s4"})
    assert [[s.id for s in r] for r in runs] == [["s1", "s2"], ["s4"]]


# ---------------------------------------------------------------- translation


def test_configs_translate_checkpoint_and_ablate_inputs(tmp_path):
    ws = _video(tmp_path)
    ref = _official(tmp_path / "official.srt")
    cfgs = [
        ablate.parse_config("base:model=m1,thinking=low"),
        ablate.parse_config("bare:model=m2,thinking=high,observations=off,narrative=off"),
    ]
    log: list = []
    rep = ablate.run_ablation(ws, cfgs, ref, 4, 3, settings=Settings(), client_factory=_factory(log))
    sampled = [s["id"] for s in rep["scenes"]]
    assert len(log) == 2 * len(sampled)
    by_stage = {s: [c for c in log if c["stage"] == s] for s in ("ablate:base", "ablate:bare")}
    assert {(c["model"], c["thinking"]) for c in by_stage["ablate:base"]} == {("m1", "low")}
    assert {(c["model"], c["thinking"]) for c in by_stage["ablate:bare"]} == {("m2", "high")}
    assert all("SARCASTIC_TONE" in c["prompt"] for c in by_stage["ablate:base"])
    assert not any("SARCASTIC_TONE" in c["prompt"] for c in by_stage["ablate:bare"])
    assert all("에이" in c["prompt"] for c in log)  # character names survive narrative=off
    for sid in sampled:
        assert (ws.data_dir / "ablate" / "base" / f"{sid}.json").is_file()
        assert (ws.data_dir / "ablate" / "bare" / f"{sid}.json").is_file()
    # previous-scene KO tail only when that scene was sampled, from the same config
    nums = sorted(int(s[1:]) for s in sampled)
    for c in by_stage["ablate:base"]:
        n = int(re.match(r"Scene s(\d+) ", c["prompt"]).group(1))
        has_tail = "PREVIOUS SCENE KOREAN TAIL" in c["prompt"]
        assert has_tail == (n - 1 in nums)
        if has_tail:
            assert f"m1 {n - 1}요." in c["prompt"] and "m2" not in c["prompt"]

    # rerun: every (config, scene) is a checkpoint hit
    log2: list = []
    rep2 = ablate.run_ablation(ws, cfgs, ref, 4, 3, settings=Settings(), client_factory=_factory(log2))
    assert log2 == []
    assert rep2["metrics"] == rep["metrics"]

    # a changed config invalidates only its own checkpoints
    log3: list = []
    cfgs3 = [cfgs[0], ablate.parse_config("bare:model=m2,thinking=medium,observations=off,narrative=off")]
    ablate.run_ablation(ws, cfgs3, ref, 4, 3, settings=Settings(), client_factory=_factory(log3))
    assert {c["stage"] for c in log3} == {"ablate:bare"} and len(log3) == len(sampled)


def test_narrative_levels_map_onto_production_rendering(tmp_path):
    """narrative=lite renders exactly what production translate_scene sends at
    Settings.narrative_context="lite" (one implementation); narrative=on is the
    full context with the speech-level table and voice cards."""
    ws = _video(tmp_path)
    base = ablate.load_base(ws)
    snaps = [
        Snapshot(
            scene_id=sc.id,
            registers={"a>b": SceneRegister(level="haeyo", vocative="선생님")},
            characters={"a": CharacterState(emotion="FURIOUS", knows=["KNOWS_SECRET"])},
            voice_cards={"a": VoiceCard(idiolect="IDIOLECT_X")},
            character_names={"a": "에이"},
        )
        for sc in base.scenes
    ]
    save_model(ws.path("snapshots.json"), snaps)
    ref = _official(tmp_path / "official.srt")
    cfgs = [ablate.parse_config("full:model=m1"), ablate.parse_config("lite:model=m1,narrative=lite")]
    log: list = []
    rep = ablate.run_ablation(ws, cfgs, ref, 4, 3, settings=Settings(narrative_context="full"),
                              client_factory=_factory(log))
    full = [c for c in log if c["stage"] == "ablate:full"]
    lite = [c for c in log if c["stage"] == "ablate:lite"]
    assert full and len(full) == len(lite)
    for c in full:
        assert "REGISTERS" in c["prompt"] and "IDIOLECT_X" in c["prompt"] and "FURIOUS" in c["prompt"]
    for c in lite:
        # lite keeps the register table rows but no voice cards or emotions
        assert "KNOWS_SECRET" in c["prompt"] and "에이" in c["prompt"]
        assert "a>b: 해요 · 선생님" in c["prompt"]
        for banned in ("VOICE CARDS", "IDIOLECT_X", "FURIOUS"):
            assert banned not in c["prompt"], banned

    # production lite (default Settings) sends the identical prompt + system
    sampled = {s["id"] for s in rep["scenes"]}
    sid = next(s for s in sorted(sampled) if f"s{int(s[1:]) - 1:03d}" not in sampled)
    scene = next(sc for sc in base.scenes if sc.id == sid)
    prod_log: list = []
    prod = FakeClient("p", prod_log)
    prod.settings = Settings(model_translate=None)  # text calls stay on this fake client
    assert prod.settings.narrative_context == "lite"
    # production snapshots carry the video facts materialize_snapshots adds;
    # load_base backfills them from the video map
    prod_snap = ablate.load_base(ws).snapshots[sid]
    assert prod_snap.tone_baseline == "y"
    translate.translate_scene(
        ws, prod, scene, {c.id: c for c in base.cues}, prod_snap, base.obs_by_scene.get(sid, []), [], mode="text", proxy=None,
    )
    ablated = next(c for c in lite if c["prompt"].startswith(f"Scene {sid} "))
    assert prod_log[0]["prompt"] == ablated["prompt"]
    assert prod_log[0]["system"] == ablated["system"] == translate._system("lite")


def test_parse_config_aim():
    c = ablate.parse_config("ga:model=m,narrative=lite,aim=0.7")
    assert c.aim == 0.7
    d = ablate.parse_config("d:model=m")
    assert d.aim is None
    for bad in ("x:model=m,bogus=1", "x:model=m,aim=0", "x:model=m,aim=1.5", "x:model=m,aim=short"):
        with pytest.raises(ValueError):
            ablate.parse_config(bad)


def test_aim_reaches_the_prompt_and_changes_checkpoint_keys(tmp_path):
    ws = _video(tmp_path)
    ref = _official(tmp_path / "official.srt")

    def run(spec: str) -> list:
        log: list = []
        ablate.run_ablation(ws, [ablate.parse_config(spec)], ref, 4, 3, settings=Settings(),
                            client_factory=_factory(log))
        return log

    plain = run("c:model=m1,narrative=lite")
    sampled = len(plain)
    assert sampled == 4
    assert all("15/18" in c["prompt"] for c in plain)  # 2 s cue + 250 ms extension: max 2.25 s x 8, aim floor(18 * 0.85)

    assert run("c:model=m1,narrative=lite") == []  # checkpoints hit

    aimed = run("c:model=m1,narrative=lite,aim=0.5")
    assert len(aimed) == sampled  # different prompt → checkpoint misses
    assert all("9/18" in c["prompt"] for c in aimed)
    rep = ablate.run_ablation(ws, [ablate.parse_config("c:model=m1,narrative=lite,aim=0.5")], ref, 4, 3,
                              settings=Settings(), client_factory=_factory([]))
    assert rep["configs"]["c"]["aim"] == 0.5
    md = (ws.data_dir / "ablate" / "ablate.md").read_text(encoding="utf-8")
    assert "| aim |" in md and "| 0.50 |" in md


# ---------------------------------------------------------------- scoring


def test_scoring_restricted_to_sampled_scenes(tmp_path):
    ws = _video(tmp_path)
    ref = _official(tmp_path / "official.srt")
    cfgs = [ablate.parse_config("base:model=m1")]
    rep = ablate.run_ablation(ws, cfgs, ref, 4, 3, settings=Settings(), client_factory=_factory([]))
    sampled = {s["id"] for s in rep["scenes"]}
    assert len(sampled) == 4
    assert rep["alignment"]["official_lines"] == 2 * len(sampled)  # 20 lines video-wide
    assert rep["alignment"]["en_cues"] == 2 * len(sampled)
    assert {g["scene_id"] for g in rep["groups"]} == sampled
    for g in rep["groups"]:
        n = int(g["scene_id"][1:])
        assert all(t.endswith(f" {n}") for t in g["official"])
    m = rep["metrics"]["base"]
    assert m["coverage"] == 1.0  # nothing outside the sample counts as an omission
    assert m["lines"] == 2 * len(sampled)
    assert m["speech_level_agreement"] is None or m["speech_level_n"] >= 0
    assert (ws.data_dir / "ablate" / "ablate.md").read_text(encoding="utf-8").count("| base |") == 1


def test_restrict_official_by_midpoint():
    from substage.refeval import Line

    scenes = [Scene(id="s1", start_ms=1000, end_ms=2000, cue_ids=[])]
    lines = [Line(0, 0, 1200, "a"), Line(1, 900, 1500, "b"), Line(2, 1800, 3000, "c")]
    kept = ablate.restrict_official(lines, scenes)
    assert [(ln.idx, ln.text) for ln in kept] == [(0, "b")]


def test_normalize_applies_s5_rules():
    cue = Cue(id="c1", start_ms=0, end_ms=2000, text="- Hi.\n- Hey.", raw="")
    st = SceneTranslation(scene_id="s1", mode="text", cues=[
        TranslatedCue(cue_id="c1", ko="안녕. 반가워."), TranslatedCue(cue_id="c2", ko=""),
    ])
    out = ablate.normalize(st, {"c1": cue}, Settings())
    assert out == {"c1": "- 안녕\n- 반가워"}


class _Judge:
    """Judge fake: a constant score for every label; records the models asked."""

    def __init__(self, score: int = 70):
        self.score = score
        self.models: list[str] = []

    def generate(self, *, stage, prompt, schema, model=None, **kw):
        self.models.append(model)
        labels_line = next(ln for ln in prompt.splitlines() if ln.startswith("Return exactly one entry"))
        labels = labels_line.split("(")[1].split(")")[0].split(", ")
        n = int(labels_line.rsplit("1..", 1)[1].split(".")[0])
        return schema.model_validate({"candidates": [
            {"label": f"Candidate {lb}", "score": self.score,
             "items": [{"item": i, "category": "none", "severity": 0} for i in range(1, n + 1)]}
            for lb in labels
        ]})


def test_judge_all_configs_in_one_call_per_window(tmp_path):
    ws = _video(tmp_path)
    ref = _official(tmp_path / "official.srt")
    cfgs = [ablate.parse_config("a:model=m1"), ablate.parse_config("b:model=m2")]
    judge = _Judge()
    rep = ablate.run_ablation(ws, cfgs, ref, 10, 3, judge=True, settings=Settings(),
                              client_factory=_factory([]), judge_factory=lambda ws, s, m: judge)
    # one arms call + one calibration call per window, default judge model
    assert len(judge.models) == 2 * len(rep["judge"]["windows"]) >= 2
    assert set(judge.models) == {Settings().model_main} and rep["judge"]["models"] == [Settings().model_main]
    assert rep["metrics"]["a"]["judge"]["score"] == 70
    assert set(rep["judge"]["calibration"]) == {Settings().model_main}
    assert (ws.data_dir / "ablate" / "checkpoints" / "refeval-judge").is_dir()


def test_multiple_judges_report_per_judge_and_mean(tmp_path):
    ws = _video(tmp_path)
    ref = _official(tmp_path / "official.srt")
    cfgs = [ablate.parse_config("a:model=m1"), ablate.parse_config("b:model=m2")]
    judges = {"gemini-j": _Judge(90), "codex/gpt-j": _Judge(60), "anthropic/claude-j": _Judge(30)}
    factory = lambda ws, s, m: judges[m]  # noqa: E731
    rep = ablate.run_ablation(ws, cfgs, ref, 10, 3, judge=True, settings=Settings(), client_factory=_factory([]),
                              judge_models=list(judges), judge_factory=factory)
    n_win = len(rep["judge"]["windows"])
    for m, j in judges.items():
        assert j.models == [m] * (2 * n_win)
    a = rep["metrics"]["a"]
    assert {m: s["score"] for m, s in a["judges"].items()} == {"gemini-j": 90, "codex/gpt-j": 60, "anthropic/claude-j": 30}
    assert a["judge"]["score"] == 60
    assert rep["judge"]["windows"][0]["verdicts"]["a"]["scores"] == {"gemini-j": 90, "codex/gpt-j": 60,
                                                                      "anthropic/claude-j": 30}
    assert rep["judge"]["calibration"]["gemini-j"]["official"]["score"] == 90
    assert set(rep["judge"]["calibration"]) == set(judges)
    md = (ws.data_dir / "ablate" / "ablate.md").read_text(encoding="utf-8")
    assert "mean over judges" in md and "### Judge: mean" in md and "### Judge: codex/gpt-j" in md
    # separate checkpoint per judge: a rerun with a judge dropped makes no calls
    for j in judges.values():
        j.models.clear()
    rep2 = ablate.run_ablation(ws, cfgs, ref, 10, 3, judge=True, settings=Settings(), client_factory=_factory([]),
                               judge_models=["anthropic/claude-j", "gemini-j"], judge_factory=factory)
    assert all(not j.models for j in judges.values())
    assert rep2["metrics"]["a"]["judge"]["score"] == 60  # (30 + 90) / 2


def test_cli_ablate_judge_models(tmp_path, monkeypatch):
    from substage import cli

    ref = _official(tmp_path / "official.srt")
    seen: dict = {}

    def fake_run(ws, configs, ref_path, n, seed, *, judge, out_dir, judge_models):
        seen.update(judge=judge, judge_models=judge_models)
        (out_dir / "ablate.md").parent.mkdir(parents=True, exist_ok=True)
        (out_dir / "ablate.md").write_text("x", encoding="utf-8")

    monkeypatch.setattr(ablate, "run_ablation", fake_run)
    monkeypatch.setattr("substage.config.load_settings", lambda: Settings())
    base = ["ablate", "video", "--ref", str(ref), "--judge", "--out", str(tmp_path / "out"), "--config", "a:model=m"]
    cli.main([*base, "--judge-model", "gemini-x,codex/gpt-y", "--judge-model", "anthropic/claude-z"])
    assert seen == {"judge": True, "judge_models": ["gemini-x", "codex/gpt-y", "anthropic/claude-z"]}
    cli.main(base)
    assert seen["judge_models"] == [Settings().model_main]


def test_semantic_sample_normalize_layout_srt_retains_actual_two_voice_lineage(tmp_path):
    import json

    import pysubs2

    from substage.models import (
        RetainedFragment, SemanticLineage, SemanticProvenance, SemanticSceneTranslation,
        SemanticSource, SourceAct, SourceRelation,
    )

    writer = "codex/gpt-6.1-sol"
    source = "Where? Outside."
    c = Cue(id="c1", start_ms=0, end_ms=2000, text=source, raw=source)
    acts = [
        SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=6, text="Where?", speaker="a", meaning="question", essential=True),
        SourceAct(ref="c1:t2:a1", cue_id="c1", turn_ordinal=2, act_ordinal=1,
                  start_char=7, end_char=len(source), text="Outside.", speaker="b", meaning="answer", essential=True),
    ]
    fragments = [
        RetainedFragment(text="어디야?", refs=[acts[0].ref], speaker="a"),
        RetainedFragment(text="밖이야", refs=[acts[1].ref], speaker="b"),
    ]
    st = SemanticSceneTranslation(
        scene_id="s1", mode="text", model=writer,
        cues=[TranslatedCue(cue_id="c1", ko="- 어디야?\n- 밖이야", model=writer)],
        lineage=SemanticLineage(
            source=SemanticSource(
                scene_id="s1", acts=acts,
                relations=[SourceRelation(kind="question_answer", refs=[act.ref for act in acts],
                                          claim="location question followed by its answer",
                                          uncertainty=["speaker identity remains advisory"])],
            ),
            fragments_by_cue={"c1": fragments}, structural_valid=True,
        ),
        provenance=SemanticProvenance(
            route_variant="semantic-image", model=writer,
            extract_stage="s3_semantic_extract", write_stage="s3_semantic_write",
            extract_hash="extract", write_hash="write", media_hash="media",
            route_version="semantic-route-v1", source_schema_version="source-v1",
            target_schema_version="target-v1", prompt_version="prompt-v1", policy_hash="policy",
        ),
    )
    ko = ablate.normalize(st, {"c1": c}, Settings())
    assert ko == {"c1": "- 어디야?\n- 밖이야"}
    artifacts = ablate._write_sample_artifacts(
        tmp_path, ablate.AblateConfig(name="semantic", model=writer),
        [Scene(id="s1", start_ms=0, end_ms=2000, cue_ids=["c1"])], {"c1": c}, {"s1": st},
        ko, Settings(), full_source=False,
    )
    rendered = pysubs2.load(artifacts["srt"], encoding="utf-8")
    assert [event.plaintext for event in rendered] == ["- 어디야?\n- 밖이야"]
    sidecar = json.loads(Path(artifacts["sidecar"]).read_text(encoding="utf-8"))
    event = sidecar["events"][0]
    assert event["cue_ids"] == ["c1"]
    assert [fragment["refs"] for fragment in event["fragments"]] == [["c1:t1:a1"], ["c1:t2:a1"]]
    assert sidecar["cues"][0]["model"] == writer
    assert sidecar["scenes"][0]["lineage"]["source"]["acts"][1]["text"] == "Outside."
    assert sidecar["lineage"]["semantic_assessment"] == "unassessed"
    relation = sidecar["lineage"]["source"]["relations"][0]
    assert relation["refs"] == ["c1:t1:a1", "c1:t2:a1"]
    assert relation["uncertainty"] == ["speaker identity remains advisory"]
    assert artifacts["lineage_issues"] == []


def test_semantic_lyric_sample_srt_keeps_one_note_and_declared_markerless_lineage(tmp_path):
    import json

    import pysubs2

    from substage.models import (
        RetainedFragment, SemanticLineage, SemanticProvenance, SemanticSceneTranslation, SemanticSource, SourceAct,
    )

    writer = "codex/gpt-6.1-sol"
    c = Cue(id="c1", start_ms=0, end_ms=2000, text="A story lyric.", raw="A story lyric.", lyric=True)
    act = SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                    start_char=0, end_char=len(c.text), text=c.text,
                    meaning="plot-relevant sung line", essential=True)
    st = SemanticSceneTranslation(
        scene_id="s1", mode="text", model=writer,
        cues=[TranslatedCue(cue_id="c1", ko="가사", model=writer)],
        lineage=SemanticLineage(
            source=SemanticSource(scene_id="s1", acts=[act]),
            fragments_by_cue={"c1": [RetainedFragment(text="가사", refs=[act.ref], speaker=None)]},
            structural_valid=True,
        ),
        provenance=SemanticProvenance(
            route_variant="semantic-image", model=writer,
            extract_stage="s3_semantic_extract", write_stage="s3_semantic_write",
            extract_hash="extract", write_hash="write", media_hash="media",
            route_version="semantic-route-v1", source_schema_version="source-v1",
            target_schema_version="target-v1", prompt_version="prompt-v1", policy_hash="policy",
        ),
    )
    ko = ablate.normalize(st, {"c1": c}, Settings())
    artifacts = ablate._write_sample_artifacts(
        tmp_path, ablate.AblateConfig(name="lyric", model=writer),
        [Scene(id="s1", start_ms=0, end_ms=2000, cue_ids=["c1"])], {"c1": c}, {"s1": st},
        ko, Settings(), full_source=False,
    )
    rendered = pysubs2.load(artifacts["srt"], encoding="utf-8")
    assert [event.plaintext for event in rendered] == ["♪ 가사"]
    sidecar = json.loads(Path(artifacts["sidecar"]).read_text(encoding="utf-8"))
    fragment = sidecar["events"][0]["fragments"][0]
    assert fragment["text"] == "가사" and fragment["attribution"] == "declared"
    assert fragment["refs"] == ["c1:t1:a1"] and fragment["prior_refs"] == []
    assert sidecar["cues"][0]["model"] == writer
    assert artifacts["lineage_issues"] == []
