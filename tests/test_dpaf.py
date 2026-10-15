"""Offline DPAF regressions: stable cue identities, bounded AV and paired judging."""

from __future__ import annotations

from dataclasses import replace
import json
import re
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from substage import perception, pipeline, refeval, register, scenes, translate, validate
from substage.config import Settings, Workspace
from substage.models import (
    Character,
    ChunkObservations,
    Cue,
    CueObservation,
    CuePerception,
    VideoInfo,
    ContextMap,
    MediaNeed,
    PerceptionCandidate,
    PerceptionLedger,
    RegisterEdge,
    RegisterMap,
    Scene,
    SceneTranslation,
    Snapshot,
    SubTurnObservation,
    Timeline,
    TranslatedCue,
    VisualDeixis,
    VoiceCard,
)
from substage.refeval import Line
from substage.store import load_model, save_model


@pytest.fixture
def dpaf_ws(tmp_path):
    return Workspace(
        video_id="dpaf-video", data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache", log_dir=tmp_path / "log",
    )


def _cue(cid="c0001", start=1000, end=4000, text="Who is she?", **kwargs):
    return Cue(id=cid, start_ms=start, end_ms=end, text=text, raw=text, **kwargs)


def _turn(speaker, addressee, **kwargs):
    return SubTurnObservation(
        speaker_candidates=[PerceptionCandidate(value=speaker, confidence=0.9)],
        addressee_candidates=[PerceptionCandidate(value=addressee, confidence=0.9)],
        **kwargs,
    )


@pytest.mark.parametrize("suffix", ["json", "yaml"])
def test_perception_artifacts_roundtrip_without_changing_cue_identity(tmp_path, suffix):
    turn = _turn("Peter", "Tony", text="Who is she?", prosody="rising")
    ledger = PerceptionLedger(cues={"c0001": CuePerception(
        cue_id="c0001", turns=[turn], media_need=MediaNeed.BOTH,
        visual_deixis=VisualDeixis(target="woman", keyframe_ms=2500),
        evidence=["Peter points toward the woman"],
    )})
    path = tmp_path / f"perception.{suffix}"
    save_model(path, ledger)
    loaded = load_model(path, PerceptionLedger)
    assert loaded == ledger
    assert loaded.cues["c0001"].turns[0].text == "Who is she?"
    assert loaded.cues["c0001"].visual_deixis.keyframe_ms == 2500
    assert loaded.model_dump(mode="json")["cues"]["c0001"]["media_need"] == "both"

    chunk = ChunkObservations(
        chunk_id="v01", start_ms=0, end_ms=5000,
        cues=[CueObservation(cue_id="c0001", speaker="Peter", addressee="Tony", turns=[turn])],
    )
    chunk_path = tmp_path / f"observations.{suffix}"
    save_model(chunk_path, chunk)
    assert load_model(chunk_path, ChunkObservations) == chunk


def test_legacy_observations_remain_loadable_and_defaults_are_independent():
    observation = CueObservation.model_validate({
        "cue_id": "c0001", "speaker": "Peter", "addressee": "Tony",
    })
    assert observation.turns is None
    assert observation.speaker == "Peter"
    first, second = CuePerception(cue_id="c0001"), CuePerception(cue_id="c0002")
    first.speaker_candidates.append(PerceptionCandidate(value="Peter"))
    first.evidence.append("visible mouth movement")
    assert second.speaker_candidates == []
    assert second.evidence == []
    assert PerceptionLedger().cues == {}


@pytest.mark.parametrize("count", [0, 3])
def test_sub_turn_models_reject_counts_outside_one_or_two(count):
    for model in (CuePerception, CueObservation):
        with pytest.raises(ValidationError):
            model(cue_id="c0001", turns=[SubTurnObservation() for _ in range(count)])


def test_ledger_filters_unknown_ids_and_does_not_rekey_mismatched_records():
    source = [_cue("c0001"), _cue("c0002"), _cue("c0003")]
    raw = {"cues": {
        "c0001": {"cue_id": "invented", "delivery": "shouting"},
        "c0002": {"cue_id": "c0002", "delivery": "whispered"},
        "invented": {"cue_id": "invented", "delivery": "shouting"},
    }}
    ledger = perception.normalize_perception_ledger(
        raw, source, cue_ids=["c0001", "c0002", "c0003", "invented"],
    )
    assert list(ledger.cues) == ["c0001", "c0002", "c0003"]
    assert ledger.cues["c0001"] == CuePerception(cue_id="c0001")
    assert ledger.cues["c0002"].delivery == "whispered"
    assert ledger.cues["c0003"] == CuePerception(cue_id="c0003")
    subset = perception.normalize_perception_ledger(ledger, source, cue_ids=["c0002", "invented"])
    assert list(subset.cues) == ["c0002"]
    assert subset.cues["c0002"].cue_id == "c0002"


def test_perception_normalization_is_idempotent_and_bounds_nested_keyframes():
    item = CuePerception(
        cue_id="c0001", delivery="  whispered\n softly  ", confidence=0.91234567,
        speaker_candidates=[
            PerceptionCandidate(value=" Tony ", confidence=0.6),
            PerceptionCandidate(value="Peter", confidence=0.91234567, evidence=[" lips move ", "LIPS MOVE"]),
            PerceptionCandidate(value=" peter ", confidence=0.91234567),
        ],
        evidence=[" points\n left ", "POINTS LEFT", "  "], keyframe_ms=999,
        visual_deixis=VisualDeixis(target="  doorway  ", keyframe_ms=4001),
        turns=[
            _turn("Peter", "Tony", text="  Who\n is she? ", keyframe_ms=1000,
                  visual_deixis=VisualDeixis(target=" woman ", keyframe_ms=4000)),
            _turn("Tony", "Peter", keyframe_ms=4001),
        ],
    )
    normalized = perception.normalize_perception_ledger(
        PerceptionLedger(cues={"c0001": item}), [_cue()],
        window_start_ms=1000, window_end_ms=4000,
    )
    record = normalized.cues["c0001"]
    assert record.delivery == "whispered softly"
    assert record.confidence == 0.912346
    assert [c.value for c in record.speaker_candidates] == ["Peter", "Tony"]
    assert record.speaker_candidates[0].evidence == ["lips move"]
    assert record.evidence == ["points left"]
    assert record.keyframe_ms is None
    assert record.visual_deixis.target == "doorway"
    assert record.visual_deixis.keyframe_ms is None
    assert record.turns[0].text == "Who is she?"
    assert record.turns[0].keyframe_ms == 1000
    assert record.turns[0].visual_deixis.keyframe_ms == 4000
    assert record.turns[1].keyframe_ms is None
    assert perception.normalize_perception_ledger(
        normalized, [_cue()], window_start_ms=1000, window_end_ms=4000,
    ) == normalized
    assert item.delivery == "  whispered\n softly  "  # input artifact is not mutated
    assert len(item.speaker_candidates) == 3


@pytest.mark.parametrize("timestamp, expected", [(999, None), (1000, 1000), (2500, 2500), (4000, 4000), (4001, None)])
def test_keyframe_bounds_are_inclusive(timestamp, expected):
    result = perception.normalize_cue_perception(
        {"cue_id": "c0001", "keyframe_ms": timestamp}, "c0001",
        window_start_ms=1000, window_end_ms=4000,
    )
    assert result.keyframe_ms == expected


@pytest.mark.parametrize("count, expected", [
    (0, []), (1, [1]), (3, [1, 2, 3]), (5, [1, 1, 2, 3, 4]),
    (10, [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]),
])
def test_scene_acts_are_deterministic_balanced_contiguous_partitions(count, expected):
    cues = [_cue(f"c{i:04d}", i * 5000, i * 5000 + 1000) for i in range(1, count + 1)]
    ordered = scenes.group_scenes(cues, gap_ms=3000, max_cues=20)
    shuffled = scenes.group_scenes(list(reversed(cues)), gap_ms=3000, max_cues=20)
    assert [s.act for s in ordered] == expected
    assert shuffled == ordered
    assert [cid for s in ordered for cid in s.cue_ids] == [c.id for c in cues]


class _TranslationClient:
    def __init__(self, *, av=False, media_flag=True):
        self.av, self.media_flag = av, media_flag
        self.calls = []
        self.settings = SimpleNamespace(
            model_main="gemini-3.8-flash" if av else "codex/gpt-6.1-sol",
            model_light="fake-light", narrative_context="lite", memory_k=0,
        )

    def generate(self, *, stage, prompt, schema, clips=(), **kwargs):
        self.calls.append({"stage": stage, "prompt": prompt, "clips": list(clips)})
        ids = re.findall(r"^  (c\d+) \|", prompt, re.M)
        assert schema is SceneTranslation
        assert bool(clips) is self.av
        # The AV fake deliberately attempts to rewrite every context neighbor,
        # including IDs it was not asked to translate. Only flagged IDs may land.
        output_ids = [f"c{i:04d}" for i in range(1, 6)] if self.av else ids
        return SceneTranslation(scene_id="s001", mode="av" if self.av else "text", cues=[
            TranslatedCue(
                cue_id=cid, ko=f"{'영상' if self.av else '원문'}-{cid}",
                needs_media=MediaNeed.VIDEO if not self.av and self.media_flag and cid == "c0003" else MediaNeed.NONE,
            ) for cid in output_ids
        ])


@pytest.mark.parametrize("flag_source", ["translation", "luna"])
def test_targeted_cascade_preserves_every_unflagged_translation(dpaf_ws, flag_source):
    cues = [_cue(f"c{i:04d}", i * 3000, i * 3000 + 2000) for i in range(1, 6)]
    scene = Scene(id="s001", start_ms=3000, end_ms=17000, cue_ids=[c.id for c in cues], act=1)
    video = VideoInfo(
        video_id=dpaf_ws.video_id, title="DPAF", video_path="/unused/video.mkv",
        duration_ms=20000, audio_stream=0, en_sub_path="/unused/en.srt", en_sub_origin="user",
    )
    text = _TranslationClient(media_flag=flag_source == "translation")
    av = _TranslationClient(av=True)

    class Luna:
        def generate(self, *, schema, clips=(), **kwargs):
            assert not clips
            return schema(flagged_cue_ids=["c0003", "invented"])

    proxy = SimpleNamespace(name="proxy", uri="gs://fake-proxy", mime_type="video/mp4")
    result = translate.translate_video(
        dpaf_ws, av, video, [scene], cues, {scene.id: Snapshot(scene_id=scene.id)}, [], proxy,
        mode="cascade", text_client_override=text, luna=Luna() if flag_source == "luna" else None,
    )[scene.id]
    assert len(text.calls) == len(av.calls) == 1
    assert result.mode == "av"
    assert [(tc.cue_id, tc.ko) for tc in result.cues] == [
        (c.id, f"{'영상' if c.id == 'c0003' else '원문'}-{c.id}") for c in cues
    ]
    assert re.findall(r"^  (c\d+) \|", av.calls[0]["prompt"], re.M) == ["c0002", "c0003", "c0004"]
    clip = av.calls[0]["clips"][0]
    assert clip.start_s == pytest.approx(5.0)
    assert clip.end_s == pytest.approx(15.0)


def _register_context():
    fm = ContextMap(synopsis="test", tone_baseline="test", characters=[
        Character(id=cid, en_name=name, ko_name=name, persona="test", voice=VoiceCard(idiolect="test"))
        for cid, name in (("peter", "Peter"), ("tony", "Tony"), ("may", "May"))
    ])
    edge = RegisterEdge(
        speaker="peter", addressee="tony", relation="mentor", apparent_generation="addressee_older",
        history="met_as_child", stance="civil", evidence_cues=["c0001", "c0002"], step=6, level="haeyo",
    )
    table = register.RegisterTable(RegisterMap(edges=[edge]), ["c0001", "c0002", "c0003"])
    return fm, table


def test_turn_pair_resolution_preserves_ambiguity_instead_of_guessing():
    fm, _ = _register_context()
    ambiguous = _turn("Tony", "Peter")
    ambiguous.speaker_candidates.append(PerceptionCandidate(value="May", confidence=0.1))
    known = _turn("Peter", "Tony")
    known.speaker_candidates.append(PerceptionCandidate(value="peter", confidence=0.2))
    observations = [CuePerception(cue_id="c0001", turns=[known, ambiguous])]
    pairs = register.cue_turn_pairs(fm, observations)
    assert pairs == {"c0001": (("peter", "tony"), None)}
    group = CueObservation(cue_id="c0002", turns=[_turn("Tony", "audience")])
    assert register.cue_turn_pairs(fm, [group])["c0002"] == (("tony", "group"),)


def test_known_turns_can_move_to_their_row_but_ambiguous_turns_keep_their_old_level():
    _, table = _register_context()
    turns = {"c0003": (("peter", "tony"), None)}
    gate = register.RegisterGate(table, {"c0003": ("peter", "tony")}, turn_pairs_by_cue=turns)
    old = "- 누가 그랬어?\n- 내가 그랬어"
    correct = "- 누가 그랬어요?\n- 내가 그랬어"
    assert gate.check("review", ["c0003"], old, correct)
    assert not gate.check("review", ["c0003"], correct, "- 누가 그랬어요?\n- 제가 그랬어요")
    rejection = gate.rejections[-1]
    assert rejection.reason == "level"
    assert rejection.turn_index == 2
    assert rejection.pair is None
    assert rejection.table_level is None
    assert rejection.turn_ko == "제가 그랬어요"
    assert not gate.check("layout", ["c0003"], correct, "누가 그랬어요?")
    assert gate.rejections[-1].reason == "turn_count"
    assert (gate.rejections[-1].expected_turns, gate.rejections[-1].actual_turns) == (2, 1)
    assert not gate.check("picture", ["c0003"], old, correct, use_table=False)
    assert gate.rejections[-1].turn_index == 1


def test_firm_validation_checks_known_turns_and_skips_ambiguous_turns():
    _, table = _register_context()
    st = SceneTranslation(scene_id="s001", mode="text", cues=[
        TranslatedCue(cue_id="c0003", ko="- 누가 그랬어?\n- 내가 그랬어"),
    ])
    pairs = {"c0003": ("peter", "tony")}
    turns = {"c0003": (("peter", "tony"), None)}
    violations = register.firm_violations(table, pairs, st, turn_pairs_by_cue=turns)
    assert len(violations) == 1
    violation = violations[0]
    assert (violation.reason, violation.turn_index, violation.pair) == ("level", 1, "peter>tony")
    assert violation.turn_ko == "누가 그랬어?"
    assert "c0003 turn 1" in register.retry_note(violations)
    st.cues[0].ko = "- 누가 그랬어요?\n- 내가 그랬어"
    assert register.firm_violations(table, pairs, st, turn_pairs_by_cue=turns) == []
    st.cues[0].ko = "누가 그랬어요?"
    violations = register.firm_violations(table, pairs, st, turn_pairs_by_cue=turns)
    assert [(v.reason, v.expected_turns, v.actual_turns) for v in violations] == [("turn_count", 2, 1)]


def test_register_validation_without_sub_turn_metadata_keeps_legacy_behavior():
    _, table = _register_context()
    pairs = {"c0003": ("peter", "tony")}
    gate = register.RegisterGate(table, pairs)
    old = "- 누가 그랬어?\n- 내가 그랬어"
    new = "- 누가 그랬어요?\n- 내가 그랬어"
    assert not gate.check("review", ["c0003"], old, new)
    st = SceneTranslation(scene_id="s001", mode="text", cues=[TranslatedCue(cue_id="c0003", ko=old)])
    assert register.firm_violations(table, pairs, st) == []
    st.cues[0].ko = "누가 그랬어?"
    assert len(register.firm_violations(table, pairs, st)) == 1
    # A single ambiguous sub-turn still retains the existing cue-level row.
    gate = register.RegisterGate(table, pairs, turn_pairs_by_cue={"c0003": (None,)})
    assert gate.check("review", ["c0003"], "누가 그랬어?", "누가 그랬어요?")


def test_display_validation_uses_sub_turn_count_and_ignores_other_cue_metadata():
    settings = Settings()
    cue = _cue(text="Hello. Goodbye.")
    two = CueObservation(cue_id=cue.id, turns=[_turn("Peter", "Tony"), _turn("Tony", "Peter")])
    issues = validate.check(cue, "안녕", settings, observation=two)
    assert [(i.rule, i.cue_id) for i in issues] == [("speakers", cue.id)]
    assert validate.check(cue, "- 안녕\n- 잘 가", settings, observation=two) == []
    assert validate.check(cue, "안녕", settings) == []
    wrong_id = two.model_copy(update={"cue_id": "c9999"})
    assert validate.check(cue, "안녕", settings, observation=wrong_id) == []
    dashed = _cue(text="- Hello.\n- Goodbye.")
    one = CuePerception(cue_id=dashed.id, turns=[_turn("Peter", "Tony")])
    assert validate.check(dashed, "안녕", settings, observation=one) == []
    assert [i.rule for i in validate.check(dashed, "- 안녕\n- 잘 가", settings, observation=one)] == ["speakers"]
    assert [i.rule for i in validate.check(dashed, "안녕", settings)] == ["speakers"]


class _PointJudge:
    """Scripted transport/cache fixtures; actual model quality is validated separately."""

    def __init__(self):
        self.calls = []

    def generate(self, *, stage, prompt, schema, model):
        assert schema is refeval.PointJudgment
        ids = [int(n) for n in re.findall(r"^\[(\d+)\]", prompt, re.M)]
        context_line = re.search(r"^UNIT_CONTEXT: (.+)$", prompt, re.M)
        assert context_line is not None
        contexts = json.loads(context_line.group(1))
        texts = [event["text"] for context in contexts for event in context["events"] if "text" in event]
        self.calls.append({"stage": stage, "prompt": prompt, "model": model, "texts": texts})
        degraded = texts == ["여기 영원히 있어야 해."]
        alignment_anchor = any(
            source["text"] == "I want you"
            for context in contexts for group in context["source_gold"] for source in group["en"]
        )
        omitted = alignment_anchor and texts == ["네가"]
        duplicated = alignment_anchor and texts == ["네가 떠났으면 좋겠어"] * 2
        timing_error = any(
            context["unmatched_outside_window"]["events"]
            or any(not event["span_group_ids"] for event in context["events"])
            for context in contexts
        )
        score = 20 if degraded or omitted or duplicated or timing_error else (
            70 if any("기준" in text for text in texts) else 100
        )
        return schema(
            score=score,
            items=[
                refeval.ItemVerdict(
                    item=i,
                    category="omission" if omitted else "verbosity" if duplicated else "meaning" if timing_error else "none",
                    severity=3 if omitted or duplicated or timing_error else 0, note="fake",
                )
                for i in ids
            ],
            source_meaning_score=20 if degraded or omitted or timing_error else 95,
            source_items=[
                refeval.SourceItemVerdict(
                    item=i, severity=3 if degraded or omitted or timing_error else 0,
                    safe_harbor=not (degraded or omitted or timing_error), note="fake",
                )
                for i in ids
            ],
            presentation_items=[
                refeval.PresentationItemVerdict(
                    item=i, timing_severity=3 if timing_error else 0, display_severity=0,
                    duplicate_severity=3 if duplicated else 0, note="fake",
                )
                for i in ids
            ],
        )


def _paired_inputs():
    en = [_cue("c0001", 1000, 3000, "Leave now."), _cue("c0002", 5000, 7000, "All right.")]
    gold = [Line(0, 1000, 3000, "지금 가자"), Line(1, 5000, 7000, "알았어")]
    opus = [Line(0, 1000, 3000, "기준 번역"), Line(1, 5000, 7000, "알았어")]
    return en, gold, opus


def test_paired_evaluation_accepts_more_than_six_candidates_and_isolates_each_pair(dpaf_ws):
    en, gold, opus = _paired_inputs()
    judge = _PointJudge()
    factory_models = []

    def factory(ws, settings, model):
        assert ws is dpaf_ws
        factory_models.append(model)
        return judge

    names = ["gemini-3.8-flash", "gemini-4-argon", "codex/gpt-6-luna", "codex/gpt-6.1-sol", "arm4", "arm5", "arm6", "arm7"]
    candidates = {name: [replace(gold[0], text=f"후보 {i}"), gold[1]] for i, name in enumerate(names)}
    kwargs = dict(judge=True, judge_models=["fake-judge"], judge_factory=factory, n_windows=1)
    report = refeval.evaluate_paired(dpaf_ws, Settings(), en, gold, candidates, opus, cycle_id="cycle-1", **kwargs)
    assert set(report["paired"]) == set(names)
    assert factory_models == ["fake-judge"]
    assert report["calibration"]["fake-judge"]["passed"] is True
    opus_calls = [call for call in judge.calls if call["texts"] == [line.text for line in opus]]
    assert len(opus_calls) == len(report["paired"][names[0]]["windows"])
    calls_after_first = len(judge.calls)
    for name, pair in report["paired"].items():
        assert pair["candidate_hash"] == refeval.subtitle_hash(candidates[name])
        assert pair["opus_hash"] == report["opus_hash"]
        assert pair["judge_delta"]["score"] == 30
        assert len(pair["windows"]) == 1
        assert report["metrics"][name]["source_safety"]["source_meaning_score"] == 95
    assert len({p["candidate_hash"] for p in report["paired"].values()}) == 8
    assert all(name not in call["prompt"] for call in judge.calls for name in names)

    # Other arms can radically change their own alignment without contaminating
    # an existing pair or its checkpoint; arm names/order/cycle are not judge inputs.
    expanded = dict(reversed(list(candidates.items())))
    expanded["unrelated"] = [Line(0, 1000, 7000, "무관한 병합 후보")]
    again = refeval.evaluate_paired(dpaf_ws, Settings(), en, gold, expanded, opus, cycle_id="cycle-2", **kwargs)
    assert again["paired"][names[0]] == report["paired"][names[0]]
    added_calls = judge.calls[calls_after_first:]
    assert len(added_calls) == len(again["paired"]["unrelated"]["windows"])
    assert all(call["texts"] == ["무관한 병합 후보"] for call in added_calls)
    assert [call for call in judge.calls if call["texts"] == [line.text for line in opus]] == opus_calls
    before = len(judge.calls)
    renamed = {"renamed": candidates[names[0]]}
    rename_report = refeval.evaluate_paired(dpaf_ws, Settings(), en, gold, renamed, opus, cycle_id="cycle-3", **kwargs)
    assert rename_report["paired"]["renamed"] == report["paired"][names[0]]
    assert len(judge.calls) == before

    changed = dict(candidates)
    changed[names[0]] = [replace(candidates[names[0]][0], text="변경된 후보"), gold[1]]
    changed_report = refeval.evaluate_paired(dpaf_ws, Settings(), en, gold, changed, opus, **kwargs)
    assert len(judge.calls) == before + 1
    assert changed_report["paired"][names[0]]["candidate_hash"] != report["paired"][names[0]]["candidate_hash"]
    assert changed_report["paired"][names[1]] == report["paired"][names[1]]
    assert changed_report["opus_hash"] == report["opus_hash"]


@pytest.mark.parametrize("changed_input", ["candidate", "source", "gold", "timing", "judge"])
def test_pointwise_checkpoint_invalidates_only_when_scoring_inputs_change(dpaf_ws, changed_input):
    en, gold, _ = _paired_inputs()
    group = refeval.Group(
        id="g0001", start_ms=1000, end_ms=3000, en=[en[0]], official=[gold[0]],
        candidates={refeval.PAIRED_CANDIDATE: [Line(0, 1000, 3000, "가자")]},
    )
    window = refeval.Window(id="w001", groups=[group])
    judge = _PointJudge()
    first = refeval.judge_point(dpaf_ws, judge, "fake-judge", window, [[group]], candidate_hash="frozen-candidate")
    repeat = refeval.judge_point(dpaf_ws, judge, "fake-judge", window, [[group]], candidate_hash="frozen-candidate")
    assert repeat == first
    assert len(judge.calls) == 1
    model = "fake-judge"
    if changed_input == "candidate":
        group = replace(group, candidates={refeval.PAIRED_CANDIDATE: [Line(0, 1000, 3000, "어서 가자")]})
    elif changed_input == "source":
        group = replace(group, en=[en[0].model_copy(update={"text": "Come here."})])
    elif changed_input == "gold":
        group = replace(group, official=[replace(gold[0], text="이리 와")])
    elif changed_input == "timing":
        group = replace(group, end_ms=3500)
    else:
        model = "another-judge"
    updated = refeval.judge_point(
        dpaf_ws, judge, model, replace(window, groups=[group]), [[group]], candidate_hash="frozen-candidate",
    )
    assert len(judge.calls) == 2
    assert updated["checkpoint_hash"] != first["checkpoint_hash"]


def test_subtitle_hash_freezes_timing_and_text_not_track_index():
    lines = [Line(0, 1000, 3000, "- 가자\n- 알았어")]
    assert refeval.subtitle_hash(lines) == refeval.subtitle_hash([replace(lines[0], idx=99)])
    assert refeval.subtitle_hash(lines) != refeval.subtitle_hash([replace(lines[0], end_ms=3001)])
    assert refeval.subtitle_hash(lines) != refeval.subtitle_hash([replace(lines[0], text="가자 알았어")])

def test_pipeline_dpaf_is_explicit_and_keeps_selected_text_translator(tmp_path, monkeypatch):
    video = tmp_path / "video.mkv"
    video.write_bytes(b"fake")
    ws = Workspace(
        video_id="pipeline-dpaf", data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache", log_dir=tmp_path / "log",
    )
    settings = Settings(model_main="gemini-3.8-flash", model_translate="codex/gpt-6.1-sol")
    video_info = VideoInfo(
        video_id=ws.video_id, title="Pipeline DPAF", video_path=str(video),
        duration_ms=5000, audio_stream=0, en_sub_path=str(tmp_path / "en.srt"),
        en_sub_origin="user",
    )
    cues = [_cue("c0001", 1000, 3000, "Leave now.")]
    scene = Scene(id="s001", start_ms=1000, end_ms=3000, cue_ids=["c0001"], act=1)
    proxy = SimpleNamespace(name="proxy", uri="gs://fake-proxy", mime_type="video/mp4")
    calls: dict[str, object] = {"perception": 0, "models": [], "translate": []}

    class FakeCheckpoints:
        def __init__(self, *_args, **_kwargs):
            pass

        def get(self, *_args, **_kwargs):
            return None

        def put(self, *_args, **_kwargs):
            pass

    class FakeGemini:
        def __init__(self):
            self.settings = settings

    text = SimpleNamespace(settings=settings)

    def fake_make_client(_ws, _settings, model):
        calls["models"].append(model)
        return text

    def fake_perception(*_args, **_kwargs):
        calls["perception"] += 1
        return PerceptionLedger(cues={"c0001": CuePerception(cue_id="c0001")})

    def fake_translate(*_args, **kwargs):
        calls["translate"].append(kwargs)
        return {}

    monkeypatch.setattr(pipeline, "load_secrets", lambda: None)
    monkeypatch.setattr(pipeline, "load_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "workspace", lambda _video_id: ws)
    monkeypatch.setattr(pipeline, "GeminiClient", lambda *_args: FakeGemini())
    monkeypatch.setattr(pipeline, "text_client", lambda *_args: text)
    monkeypatch.setattr(pipeline, "make_client", fake_make_client)
    monkeypatch.setattr(pipeline, "Checkpoints", FakeCheckpoints)
    monkeypatch.setattr(pipeline, "save_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "_ingest", lambda *_args, **_kwargs: video_info)
    monkeypatch.setattr(pipeline, "_load_cues", lambda *_args, **_kwargs: cues)
    monkeypatch.setattr(pipeline, "_proxy_upload", lambda *_args, **_kwargs: proxy)
    monkeypatch.setattr(pipeline.scenes_mod, "group_scenes", lambda *_args, **_kwargs: [scene])
    monkeypatch.setattr(pipeline.narrative, "viewing_pass", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(pipeline.narrative, "build_perception_ledger", fake_perception)
    monkeypatch.setattr(pipeline.narrative, "build_context_map", lambda *_args: ContextMap(synopsis="", tone_baseline=""))
    monkeypatch.setattr(pipeline.narrative, "build_timeline", lambda *_args: Timeline())
    monkeypatch.setattr(pipeline.register_mod, "build_register", lambda *_args: RegisterMap())
    monkeypatch.setattr(pipeline.narrative, "materialize_snapshots", lambda *_args: {scene.id: Snapshot(scene_id=scene.id)})
    monkeypatch.setattr(pipeline.translate, "ost_cues", lambda *_args: [])
    monkeypatch.setattr(pipeline.translate, "translate_video", fake_translate)
    monkeypatch.setattr(pipeline.register_mod, "cue_pairs", lambda *_args: {})
    monkeypatch.setattr(pipeline.register_mod, "cue_turn_pairs", lambda *_args: {})

    pipeline.run(video, video_id=ws.video_id, mode="text", stop_after="translate")
    assert calls["perception"] == 0
    assert calls["models"] == []
    assert calls["translate"][0]["text_client_override"] is None
    calls["translate"].clear()

    pipeline.run(video, video_id=ws.video_id, mode="text", dpaf=True, stop_after="translate")

    assert calls["perception"] == 1
    assert calls["models"] == ["codex/gpt-6.1-sol"]
    assert calls["translate"][0]["text_client_override"] is text
    assert calls["translate"][0]["luna"] is None

def test_failed_perception_is_not_checkpointed_as_empty_success(dpaf_ws):
    video = VideoInfo(
        video_id=dpaf_ws.video_id, title="Failure", video_path="/unused/video.mkv",
        duration_ms=5000, audio_stream=0, en_sub_path="/unused/en.srt",
        en_sub_origin="user",
    )
    cue = _cue("c0001")
    proxy = SimpleNamespace(uri="gs://fake-proxy")

    class FailingGemini:
        settings = Settings(model_main="gemini-3.8-flash")

        def __init__(self):
            self.calls = 0

        def generate(self, **_kwargs):
            self.calls += 1
            raise RuntimeError("provider unavailable")

    gc = FailingGemini()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="provider unavailable"):
            perception.selective_perception_pass(
                dpaf_ws, gc, video, proxy, [cue], cue_ids=[cue.id],
            )
    assert gc.calls == 2


def test_register_retry_after_cascade_only_rewrites_violating_cues(dpaf_ws):
    cues = [_cue(f"c{i:04d}", i * 3000, i * 3000 + 2000) for i in range(1, 6)]
    scene = Scene(id="s001", start_ms=3000, end_ms=17000, cue_ids=[c.id for c in cues], act=1)
    video = VideoInfo(
        video_id=dpaf_ws.video_id, title="DPAF", video_path="/unused/video.mkv",
        duration_ms=20000, audio_stream=0, en_sub_path="/unused/en.srt",
        en_sub_origin="user",
    )
    text = _TranslationClient(media_flag=True)
    av = _TranslationClient(av=True)

    class RetryNote(str):
        cue_ids = ("c0003",)

    result = translate.translate_video(
        dpaf_ws, av, video, [scene], cues, {scene.id: Snapshot(scene_id=scene.id)}, [],
        SimpleNamespace(name="proxy", uri="gs://fake-proxy", mime_type="video/mp4"),
        mode="cascade",
        register_check=lambda _st: RetryNote("retry c0003"),
        text_client_override=text,
    )[scene.id]
    assert len(text.calls) == 2
    assert len(av.calls) == 1
    assert [(tc.cue_id, tc.ko) for tc in result.cues] == [
        (c.id, f"원문-{c.id}") for c in cues
    ]
