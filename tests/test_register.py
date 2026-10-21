"""Offline tests for substage.register: S2r pair selection and coverage, the
firm rule, the edit gate, the S3 firm retry and the snapshot row rendering."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from substage import register, translate
from substage.config import Workspace
from substage.models import (
    Character,
    ChunkObservations,
    Cue,
    CueObservation,
    VideoInfo,
    ContextMap,
    RegisterEdge,
    RegisterMap,
    RegisterShift,
    Scene,
    SceneRegister,
    SceneTranslation,
    Snapshot,
    TranslatedCue,
    VoiceCard,
)


def _ws(tmp_path) -> Workspace:
    return Workspace(video_id="t", data_dir=tmp_path / "d", cache_dir=tmp_path / "c", log_dir=tmp_path / "l")


def _cue(n: int, text: str = "line", hint: str | None = None) -> Cue:
    return Cue(id=f"c{n:04d}", start_ms=n * 1000, end_ms=n * 1000 + 800, text=text, raw=text, speaker_hint=hint)


def _char(cid: str, name: str) -> Character:
    return Character(id=cid, en_name=name, ko_name=name, persona="p", voice=VoiceCard(idiolect="i"))


def _edge(spk, adr, level="haeyo", step=6, **kw) -> RegisterEdge:
    base = dict(
        speaker=spk, addressee=adr, relation="mentor", apparent_generation="addressee_older",
        history="met_as_child", stance="civil", step=step, level=level,
        evidence_cues=["c0001", "c0002"], basis="b",
    )
    return RegisterEdge(**{**base, **kw})


def _context_map() -> ContextMap:
    return ContextMap(synopsis="syn", tone_baseline="tone",
                   characters=[_char("peter", "Peter"), _char("tony", "Tony"), _char("may", "May")])


def _video() -> VideoInfo:
    return VideoInfo(video_id="t", title="T", video_path="/v", duration_ms=1, audio_stream=0,
                    en_sub_path="/s", en_sub_origin="x")


# ---------------------------------------------------------------- pairs


def _observed(pairs: list[tuple[str | None, str | None]]) -> tuple[list[Cue], list[ChunkObservations]]:
    cues = [_cue(i + 1) for i in range(len(pairs))]
    obs = ChunkObservations(chunk_id="v01", start_ms=0, end_ms=10**6, cues=[
        CueObservation(cue_id=c.id, speaker=s, addressee=a) for c, (s, a) in zip(cues, pairs, strict=True)
    ])
    return cues, [obs]


def test_required_pairs_are_frequent_pairs_spoken_reverses_and_group_edges():
    cues, obs = _observed(
        [("Peter", "Tony")] * 3 + [("TONY", "group")] * 3 + [("May", "Peter")] * 2
        + [("Peter", "audience"), ("Peter", None), ("Ghost", "Peter"), ("Tony", "Peter")]
    )
    pairs = register.cue_pairs(_context_map(), cues, obs)
    # names resolve to ids; audience is the group; missing/unknown are not addressed lines
    assert pairs["c0001"] == ("peter", "tony") and pairs["c0004"] == ("tony", "group")
    assert pairs["c0009"] == ("peter", "group")
    assert set(pairs) == {f"c{n:04d}" for n in (*range(1, 10), 12)}
    # >= 3 lines: peter>tony and tony>group; tony>peter is a spoken reverse (1 line);
    # may>peter (2 lines) is not required, nor its silent reverse peter>may
    assert register.required_pairs(pairs) == [("peter", "tony"), ("tony", "group"), ("tony", "peter")]
    # a reverse nobody speaks is not required
    del pairs["c0012"]
    assert register.required_pairs(pairs) == [("peter", "tony"), ("tony", "group")]


def _names_map(*names: tuple[str, str]) -> ContextMap:
    return ContextMap(synopsis="s", tone_baseline="t", characters=[_char(cid, n) for cid, n in names])


def test_name_resolver_strips_titles_and_takes_unambiguous_partial_names():
    rid = register.name_resolver(_names_map(
        ("randall", "Dr. Randall Mindy"), ("janie", "Janie Orlean"), ("jud", "Jud Duplenticy"),
        ("kate", "Kate Dibiasky"),
    ))
    assert rid("Dr. Randall Mindy") == "randall"  # exact
    assert rid("Randall Mindy") == "randall"  # the map's title stripped
    assert rid("President Janie Orlean") == "janie"  # the observation's title stripped
    assert rid("Father Jud Duprat") == "jud"  # misspelt surname: the first name is unique
    assert rid("Mindy") == rid("Mr. Mindy") == "randall"  # unique last name
    assert rid("Kate") == "kate" and rid("Ghost/Kate") == "kate"
    assert rid("Duprat") is None and rid("Man") is None and rid(None) is None


def test_name_resolver_refuses_ambiguous_partial_names():
    rid = register.name_resolver(_names_map(
        ("randall", "Dr. Randall Mindy"), ("june", "June Mindy"), ("jud", "Jud Duplenticy"),
        ("judd", "Jud Wicks"),
    ))
    assert rid("Mindy") is None and rid("Dr. Mindy") is None  # two Mindys
    assert rid("Randall Mindy") == "randall"  # the full name still resolves
    assert rid("Father Jud Duprat") is None  # two Juds, surname matches neither
    assert rid("Jud Wicks") == "judd"  # first and last name pick one


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Emil Michael", "ep:emilmichael"),
        ("David Bonderman", "ep:davidbonderman"),
        ("Bonnie Kalanick", "ep:bonniekalanick"),
        ("Michael Ovitz", "ovitz"),
        ("Michael", "ovitz"),
        ("Kalanick", "travis"),
    ],
)
def test_episode_ids_do_not_collapse_distinct_full_names(name, expected):
    known = {
        "michaelovitz": "ovitz", "michael": "ovitz",
        "daviddrummond": "drummond", "david": "drummond",
        "traviskalanick": "travis", "kalanick": "travis",
    }
    assert register._mint_ep_id(name, known) == expected


def test_group_words_map_to_group():
    for word in ("group", "audience", "the audience", "Crowd", "the press", "viewers", "Everyone"):
        assert register.is_group(word), word
    assert not register.is_group("Peter") and not register.is_group("self") and not register.is_group(None)
    cues, obs = _observed([("Peter", w) for w in ("audience", "the crowd", "Press", "viewers", "everyone")])
    assert set(register.cue_pairs(_context_map(), cues, obs).values()) == {("peter", "group")}


# ---------------------------------------------------------------- S2r


class _FakeTextClient:
    """Answers the enumeration pass with `enumerated` and each S2r batch with
    edges for the listed pairs minus `skip`."""

    def __init__(self, skip_first: set[str], skip_reask: set[str], enumerated: list[dict] | None = None):
        self.settings = SimpleNamespace(model_main="m")
        self.calls: list[dict] = []
        self.skip_first, self.skip_reask = skip_first, skip_reask
        self.enumerated = enumerated or []

    def generate(self, *, stage, prompt, schema, **kw):
        self.calls.append({"stage": stage, "prompt": prompt})
        if stage == "s2r:pairs":
            return schema(pairs=self.enumerated)
        listed = re.findall(r"^(\w+)>(\w+) \(\d+ lines\)$", prompt, re.M)
        skip = self.skip_first if ":batch" in stage else self.skip_reask
        return schema(edges=[_edge(s, a) for s, a in listed if f"{s}>{a}" not in skip])


def _s2r_world(n_pairs: int):
    """n_pairs speaker>group pairs, 3 lines each (no reverses)."""
    chars = [_char(f"x{i:02d}", f"X{i:02d}") for i in range(n_pairs)]
    fm = ContextMap(synopsis="syn", tone_baseline="tone", characters=chars)
    cues, obs = _observed([(f"X{i:02d}", "group") for i in range(n_pairs) for _ in range(3)])
    return fm, cues, obs


def test_s2r_batches_pairs_with_the_full_transcript(tmp_path):
    fm, cues, obs = _s2r_world(30)
    gc = _FakeTextClient(set(), set())
    rmap = register.build_register(_ws(tmp_path), gc, _video(), fm, cues, obs)
    assert len(rmap.edges) == 30
    # one enumeration pass, then <= 25 pairs per call
    assert [c["stage"] for c in gc.calls] == ["s2r:pairs", "s2r:batch-01", "s2r:batch-02"]
    for call in gc.calls:
        assert all(f"{c.id}|" in call["prompt"] for c in cues)  # whole video each time
    assert "c0001|00:01|x00>group|line" in gc.calls[1]["prompt"]
    assert rmap.required_pairs == [f"x{i:02d}>group" for i in range(30)]
    # checkpointed: a rerun makes no calls
    register.build_register(_ws(tmp_path), gc, _video(), fm, cues, obs)
    assert len(gc.calls) == 3


def test_s2r_reasks_missing_pairs_once(tmp_path):
    fm, cues, obs = _s2r_world(4)
    gc = _FakeTextClient({"x01>group", "x02>group"}, set())
    rmap = register.build_register(_ws(tmp_path), gc, _video(), fm, cues, obs)
    assert [(e.speaker, e.addressee) for e in rmap.edges] == [(f"x{i:02d}", "group") for i in range(4)]
    assert [c["stage"] for c in gc.calls] == ["s2r:pairs", "s2r:batch-01", "s2r:reask-01"]
    reask = re.findall(r"^\w+>group \(", gc.calls[2]["prompt"], re.M)
    assert reask == ["x01>group (", "x02>group ("]


def test_s2r_coverage_below_95_percent_fails_loudly(tmp_path):
    fm, cues, obs = _s2r_world(20)
    gc = _FakeTextClient({"x03>group", "x07>group"}, {"x03>group", "x07>group"})
    with pytest.raises(register.RegisterCoverageError, match=r"18/20 .*x03>group, x07>group"):
        register.build_register(_ws(tmp_path), gc, _video(), fm, cues, obs)
    assert len(gc.calls) == 3  # enumeration + one batch + exactly one re-ask
    # 1 of 20 missing is 95 %: accepted, the pair is left out
    gc = _FakeTextClient({"x03>group"}, {"x03>group"})
    rmap = register.build_register(_ws(tmp_path / "ok"), gc, _video(), fm, cues, obs)
    assert len(rmap.edges) == 19
    assert len(rmap.required_pairs) == 20  # the audit list keeps the uncovered pair


def test_s2r_required_pairs_are_the_union_of_cue_pairs_and_the_llm_enumeration(tmp_path):
    # viewing: peter>tony 3 lines; may>peter 2 lines; the rest unaddressed or unresolved
    cues, obs = _observed(
        [("Peter", "Tony")] * 3 + [("May", "Peter")] * 2
        + [("May", None), ("Peter", "Mr. Stark?"), ("Tony", None), ("Tony", None), ("Tony", None)]
    )
    enumerated = [
        {"speaker": "peter", "addressee": "tony", "cue_ids": ["c0001", "c0002"]},  # fewer than viewing
        {"speaker": "may", "addressee": "peter", "cue_ids": ["c0004", "c0005", "c0006"]},  # 3 here, 2 in viewing
        {"speaker": "tony", "addressee": "audience", "cue_ids": ["c0008", "c0009", "c0010"]},  # -> group
        {"speaker": "tony", "addressee": "peter", "cue_ids": ["c0007", "c0007", "c9999"]},  # 1 real line
        {"speaker": "ghost", "addressee": "peter", "cue_ids": ["c0001", "c0002", "c0003"]},  # not a map id
        {"speaker": "may", "addressee": "may", "cue_ids": ["c0001", "c0002", "c0003"]},  # self
        {"speaker": "tony", "addressee": "may", "cue_ids": ["c9997", "c9998", "c9999"]},  # no real cue
    ]
    gc = _FakeTextClient(set(), set(), enumerated)
    rmap = register.build_register(_ws(tmp_path), gc, _video(), _context_map(), cues, obs)
    want = ["peter>tony", "may>peter", "tony>group", "tony>peter"]  # tony>peter: spoken reverse
    assert rmap.required_pairs == want  # persisted in register.yaml for audit
    assert [f"{e.speaker}>{e.addressee}" for e in rmap.edges] == want
    enum, batch = gc.calls[0]["prompt"], gc.calls[1]["prompt"]
    # the enumeration sees the roster and the raw viewing names, and is in the checkpoint identity
    assert "peter | Peter | Peter | -" in enum and "c0007|00:07|Peter -> Mr. Stark?|line" in enum
    assert "c0008|00:08|Tony -> ?|line" in enum and "{" not in enum
    # line counts are the larger of the two sources
    assert re.findall(r"^\S+ \(\d+ lines\)$", batch, re.M) == [
        "peter>tony (3 lines)", "may>peter (3 lines)", "tony>group (3 lines)", "tony>peter (1 lines)",
    ]
    register.build_register(_ws(tmp_path), gc, _video(), _context_map(), cues, obs)
    assert len(gc.calls) == 2  # enumeration and batch checkpointed


def test_register_stage_hash_includes_the_enumeration_prompt(monkeypatch):
    from substage import pipeline

    args = (SimpleNamespace(model_main="m"), _context_map(), [_cue(1)], [], ())
    before = pipeline._register_hash(*args)
    monkeypatch.setattr(register, "pairs_template", lambda: "another enumeration prompt")
    assert pipeline._register_hash(*args) != before


# ---------------------------------------------------------------- table + firm


def _table(*edges: RegisterEdge, n_cues: int = 20) -> register.RegisterTable:
    return register.RegisterTable(RegisterMap(edges=list(edges)), [f"c{n:04d}" for n in range(1, n_cues + 1)])


def test_firm_rule():
    assert register.edge_firm(_edge("a", "b", "haeyo", step=6), ["c0001", "c0002"])
    assert not register.edge_firm(_edge("a", "b", "hae", step=6), ["c0001", "c0002"])  # off the step's level
    assert not register.edge_firm(_edge("a", "b", "haeyo", step=8), ["c0001", "c0002"])  # step without a fixed level
    assert not register.edge_firm(_edge("a", "b", "haeyo", step=6), ["c0001"])  # one existing cue
    assert not register.edge_firm(_edge("a", "b", "haeyo", step=6, stance="open_enmity"), ["c0001", "c0002"])
    assert not register.edge_firm(_edge("a", "b", "hae", step=7, ai_user=True), ["c0001", "c0002"])
    assert register.edge_firm(_edge("a", "b", "hae", step=3), ["c0001", "c0002"])  # step 3 fixes no level


@pytest.mark.parametrize("stance", ["civil", "open_enmity", "contempt_from_power"])
def test_polite_stranger_step_nine_is_firm_with_two_distinct_existing_cues(stance):
    edge = _edge(
        "driver", "passenger", "haeyo", step=9, relation="stranger_civil",
        apparent_generation="same", history="met_as_adults", stance=stance,
    )
    assert register._earliest_step(edge) is None
    assert register.edge_firm(edge, ["c0001", "c0002"])
    assert _table(edge).is_firm("driver", "passenger", "c0003")
    assert not register.edge_firm(edge, ["c0001"])
    duplicate = edge.model_copy(update={"evidence_cues": ["c0001", "c0001", "c9999"]})
    assert not register.edge_firm(duplicate, ["c0001", "c0002"])


@pytest.mark.parametrize("relation", ["peer", "mentor", "professional"])
def test_step_nine_other_relations_remain_non_firm(relation):
    edge = _edge(
        "a", "b", "haeyo", step=9, relation=relation,
        apparent_generation="same", history="met_as_adults",
    )
    assert not register.edge_firm(edge, ["c0001", "c0002"])


@pytest.mark.parametrize("relation", ["stranger_civil", "professional"])
def test_heated_polite_encounters_do_not_become_enemy_edges(relation):
    edge = _edge(
        "a", "b", "haeyo", step=9, relation=relation,
        apparent_generation="same", history="met_as_adults", stance="open_enmity",
    )
    assert register._earliest_step(edge) is None
    assert register._earliest_step(edge.model_copy(update={"relation": "enemy"})) == 2
    assert register._earliest_step(edge.model_copy(update={"step": 8})) == 2
    assert register._earliest_step(edge.model_copy(update={"level": "hae"})) == 2
    assert not register.edge_firm(edge.model_copy(update={"level": "hae"}), ["c0001", "c0002"])
    assert not register.edge_firm(
        edge.model_copy(update={"apparent_generation": "addressee_older"}), ["c0001", "c0002"]
    )


def test_table_shifts_and_firmness_by_cue():
    t = _table(_edge("a", "b", "haeyo", shifts=[
        RegisterShift(at_cue="c0005", level="hae", reason="role_on", until_cue="c0008")]))
    assert [t.level_at("a", "b", c) for c in ("c0004", "c0005", "c0007", "c0008")] == ["haeyo", "hae", "hae", "haeyo"]
    assert t.is_firm("a", "b", "c0004") and not t.is_firm("a", "b", "c0006") and t.is_firm("a", "b", "c0008")
    assert t.level_at("b", "a", "c0001") is None and t.address("b", "a") == ("none", "none")


# ---------------------------------------------------------------- gate


@pytest.mark.parametrize(
    ("table_level", "old", "new", "ok"),
    [
        ("haeyo", "누가 그랬어?", "누가 그랬어요?", True),  # moves to the table level
        ("haeyo", "누가 그랬어요?", "누가 그랬어?", False),  # moves away from it
        ("hae", "누가 그랬어?", "누가 그랬어요?", False),  # off both the table and the old level
        ("hae", "누가 그랬어요?", "누가 그랬어요, 진짜?", True),  # keeps the old level
        ("hae", "누가 그랬어요?", "토니 씨", True),  # unleveled fragment
        ("hasipsio", "누가 그랬어?", "누가 그랬어요?", True),  # 존댓말 grain: 해요 ≈ 하십시오
        (None, "누가 그랬어?", "누가 그랬어요?", False),  # no edge: only the old level
    ],
)
def test_gate_level_matrix(table_level, old, new, ok):
    table = _table(_edge("a", "b", table_level)) if table_level else _table()
    gate = register.RegisterGate(table, {"c0003": ("a", "b")})
    assert gate.check("review", ["c0003"], old, new) is ok
    assert [r.reason for r in gate.rejections] == ([] if ok else ["level"])


def test_gate_rejects_table_pronoun_swap_and_picture_level_edits():
    table = _table(_edge("a", "b", "hae", pronoun="너"))
    gate = register.RegisterGate(table, {"c0003": ("a", "b")})
    assert not gate.check("finale", ["c0003"], "네가 그랬어?", "당신이 그랬어?")
    assert gate.check("finale", ["c0003"], "누가 그랬어?", "네가 그랬어?")  # the table's own pronoun
    # C7: no register authority — the table level does not license a change
    table2 = _table(_edge("a", "b", "haeyo"))
    gate2 = register.RegisterGate(table2, {"c0003": ("a", "b")})
    assert not gate2.check("picture", ["c0003"], "누가 그랬어?", "누가 그랬어요?", use_table=False)
    # dash lines judge each turn against its old level
    assert gate2.check("layout", ["c0003"], "- 누가 그랬어?\n- 제가요", "- 누가?\n- 제가 그랬어요")
    assert not gate2.check("layout", ["c0003"], "- 누가 그랬어?\n- 제가 그랬어요", "- 누가 그랬어요?\n- 제가 그랬어요")
    assert [(r.stage, r.reason) for r in gate.rejections + gate2.rejections] == [
        ("finale", "pronoun"), ("picture", "level"), ("layout", "level")]


@pytest.mark.parametrize(
    "kw",
    [
        dict(relation="mentor", apparent_generation="addressee_older", history="met_as_child", step=7, level="hae"),
        dict(relation="boss", apparent_generation="same", history="met_as_adults", step=7, level="hae"),
        dict(relation="kid_to_adult", apparent_generation="same", history="none", step=10, level="hae"),
        dict(relation="peer", apparent_generation="addressee_older", history="met_as_adults", step=10, level="hae"),
        dict(relation="own_parent", apparent_generation="addressee_older", history="raised_by_addressee", step=6,
             level="haeyo"),
        dict(relation="ai_to_user", apparent_generation="same", history="none", step=10, level="hae"),
        dict(relation="protege", apparent_generation="speaker_older", history="none", step=10, level="hae"),
    ],
)
def test_firm_rule_rejects_earlier_step_features(kw):
    """Features that point to an earlier ladder step leave the edge non-firm
    (the table level stays as given), so no retry pushes toward it."""
    edge = _edge("a", "b", **kw)
    assert not register.edge_firm(edge, ["c0001", "c0002"])
    assert _table(edge).level_at("a", "b", "c0003") == kw["level"]


def test_mentor_edge_claiming_senior_step_triggers_no_retry():
    table = _table(_edge("peter", "tony", "hae", step=7))  # younger -> mentor, wrongly at step 7
    st = SceneTranslation(scene_id="s001", mode="text", cues=[TranslatedCue(cue_id="c0001", ko="아저씨도 좀 쉬어요")])
    assert register.firm_violations(table, {"c0001": ("peter", "tony")}, st) == []


def test_enemy_relation_points_to_step_two():
    edge = _edge("peter", "tony", "haeyo", step=6, relation="enemy", stance="civil", apparent_generation="same",
                 history="met_as_adults")
    assert not register.edge_firm(edge, ["c0001", "c0002"])
    st = SceneTranslation(scene_id="s001", mode="text", cues=[TranslatedCue(cue_id="c0001", ko="당신이 한 짓이야")])
    assert register.firm_violations(_table(edge), {"c0001": ("peter", "tony")}, st) == []


def test_gate_rejects_new_pronoun_when_table_edge_has_none():
    table = _table(_edge("a", "b", "haeyo", step=4, relation="surrogate_parent", vocative="아저씨"))
    gate = register.RegisterGate(table, {"c0003": ("a", "b"), "c0004": ("x", "y")})
    assert not gate.check("finale", ["c0003"], "아저씨도 좀 쉬어요", "너도 좀 쉬어요")
    assert gate.check("finale", ["c0003"], "아저씨도 좀 쉬어요", "아저씨는 좀 쉬어요")
    # an absent edge (no table entry) keeps the pronoun unchecked
    assert gate.check("finale", ["c0004"], "그만 쉬어", "너도 그만 쉬어")
    assert [r.reason for r in gate.rejections] == ["pronoun"]


@pytest.mark.parametrize("ko,want", [
    ("당신한테만 한 말이야", {"당신"}), ("너에게는 안 줘", {"너"}), ("자네한테도 있지", {"자네"}),
    ("너희한테서도 왔어", {"너"}), ("너무 좋아", set()), ("널리 알려", set()), ("당신들 모두", set()),
])
def test_pronouns_with_compound_particles(ko, want):
    assert register._pronouns(ko) == want


def test_gate_rejects_pronoun_swap_behind_compound_particle():
    gate = register.RegisterGate(_table(_edge("a", "b", "hae", pronoun="너")), {"c0003": ("a", "b")})
    assert not gate.check("finale", ["c0003"], "너한테 한 말이야", "당신한테만 한 말이야")
    assert [r.reason for r in gate.rejections] == ["pronoun"]


# ---------------------------------------------------------------- S3 firm retry


def test_pipeline_apply_edits_runs_every_edit_through_the_gate():
    from substage.models import CueEdit
    from substage.pipeline import STAGES, _apply_edits

    assert STAGES.index("timeline") < STAGES.index("register") < STAGES.index("snapshots")
    gate = register.RegisterGate(_table(_edge("a", "b", "hae")), {"c0001": ("a", "b"), "c0002": ("a", "b")})
    ko = {"c0001": "누가 그랬어?", "c0002": "누가 그랬어요?"}
    edits = [
        CueEdit(cue_id="c0001", new_ko="누가 그랬어요?", reason="r", kind="register"),  # away from 반말
        CueEdit(cue_id="c0002", new_ko="누가 그랬어?", reason="r", kind="register"),  # toward the table
    ]
    out, applied = _apply_edits(ko, edits, gate, "finale")
    assert out == {"c0001": "누가 그랬어?", "c0002": "누가 그랬어?"}
    assert [e.cue_id for e in applied] == ["c0002"]
    assert [(r.stage, r.cue_ids) for r in gate.rejections] == [("finale", ["c0001"])]


def test_firm_violation_retranslates_the_scene_once(tmp_path):
    table = _table(_edge("peter", "tony", "haeyo", step=6))
    pairs = {"c0001": ("peter", "tony"), "c0002": ("peter", "tony")}
    scenes = [Scene(id="s001", start_ms=0, end_ms=3000, cue_ids=["c0001", "c0002"], act=1),
              Scene(id="s002", start_ms=4000, end_ms=6000, cue_ids=["c0004"], act=1)]
    cues = [_cue(1, "Who did it?"), _cue(2, "Fine."), _cue(4, "Hi.")]
    snaps = {s.id: Snapshot(scene_id=s.id) for s in scenes}
    calls: list[str] = []

    class GC:
        settings = SimpleNamespace(model_main="m", model_light="l", narrative_context="lite")

        def generate(self, *, stage, prompt, schema, **kw):
            calls.append(prompt)
            sid = re.search(r"Scene (s\d+)", prompt).group(1)
            ids = re.findall(r"^  (c\d+) \|", prompt, re.M)
            retry = "REGISTER RETRY" in prompt
            ko = {"c0001": "누가 그랬어요?" if retry else "누가 그랬어?", "c0002": "좋아요", "c0004": "안녕"}
            return SceneTranslation(scene_id=sid, mode="text",
                                    cues=[TranslatedCue(cue_id=i, ko=ko[i]) for i in ids])

    retried: list[str] = []

    def check(st):
        v = register.firm_violations(table, pairs, st)
        retried.extend(x.cue_id for x in v)
        return register.retry_note(v) if v else ""

    out = translate.translate_video(_ws(tmp_path), GC(), _video(), scenes, cues, snaps, [], None,
                                   mode="text", register_check=check)
    assert retried == ["c0001"]
    assert len(calls) == 3  # s001, s001 retry, s002
    assert "c0001 peter>tony: must be 해요 (was 반말): 누가 그랬어?" in calls[1]
    assert out["s001"].cues[0].ko == "누가 그랬어요?"
    assert register.firm_violations(table, pairs, out["s001"]) == []


def test_non_firm_edges_never_trigger_a_retry():
    table = _table(_edge("peter", "tony", "haeyo", step=9))  # step 9 is not firm
    st = SceneTranslation(scene_id="s001", mode="text", cues=[TranslatedCue(cue_id="c0001", ko="누가 그랬어?")])
    assert register.firm_violations(table, {"c0001": ("peter", "tony")}, st) == []


def test_firm_retry_is_off_by_default_and_firm_violations_stay_logged():
    from dataclasses import replace

    from substage.config import Settings
    from substage.pipeline import _firm_retry

    table = _table(_edge("peter", "tony", "haeyo", step=6))
    pairs = {"c0001": ("peter", "tony")}
    st = SceneTranslation(scene_id="s001", mode="text", cues=[TranslatedCue(cue_id="c0001", ko="누가 그랬어?")])
    report = register.RegisterReport()
    assert Settings().register_retry is False
    assert _firm_retry(Settings(), table, pairs, report) is None  # translate_video gets no retry hook
    # the firm computation still runs and is what register_report.json logs after S3
    assert [v.cue_id for v in register.firm_violations(table, pairs, st)] == ["c0001"]
    check = _firm_retry(replace(Settings(), register_retry=True), table, pairs, report)
    assert "c0001 peter>tony: must be 해요" in check(st) and report.firm_retried_scenes == ["s001"]


# ---------------------------------------------------------------- rendering


def test_lite_snapshot_renders_level_address_and_shifts():
    snap = Snapshot(scene_id="s001", registers={
        "may>peter": SceneRegister(level="hae", vocative="피터", pronoun="너"),
        "peter>tony": SceneRegister(level="haeyo", vocative="아저씨", shifts=[
            RegisterShift(at_cue="c0040", level="hae", reason="first_private_talk_equals", until_cue="c0090")]),
    })
    lite = snap.for_prompt("lite")
    assert lite.registers == snap.registers
    block = translate._render_snapshot(lite, {})
    assert "  may>peter: 반말 · 피터 · 너" in block
    assert "  peter>tony: 해요 · 아저씨" in block
    assert "    shift from c0040 until c0090: 반말 (first_private_talk_equals)" in block


def test_tactical_ai_vs_companion_ai_and_paired_canonicalization():
    tactical_ai = _edge(
        "sys_ai", "user_a", "hae", step=1, relation="ai_to_user",
        apparent_generation="same", history="none", ai_user=True,
        basis="Tactical defense AI addressing its designated operator.",
    )
    user_to_tactical = _edge(
        "user_a", "sys_ai", "hae", step=10, relation="ai_to_user",
        apparent_generation="same", history="none", ai_user=True,
        basis="Operator commands tactical AI utility.",
    )
    companion_ai = _edge(
        "buddy_ai", "user_a", "hae", step=1, relation="ai_to_user",
        apparent_generation="same", history="none", ai_user=True,
        basis="Suit AI companion speaking to teen wearer.",
    )
    user_to_companion = _edge(
        "user_a", "buddy_ai", "hae", step=10, relation="partner",
        apparent_generation="same", history="none", ai_user=False,
        basis="Teen wearer talks to suit AI companion.",
    )
    spouse_a = _edge(
        "sp_a", "sp_b", "haeyo", step=9, relation="partner",
        apparent_generation="same", history="met_as_adults",
        basis="Married spouses.",
    )
    spouse_b = _edge(
        "sp_b", "sp_a", "haeyo", step=9, relation="partner",
        apparent_generation="same", history="met_as_adults",
        basis="Married spouses.",
    )
    table = _table(tactical_ai, user_to_tactical, companion_ai, user_to_companion, spouse_a, spouse_b)
    assert table.level_at("sys_ai", "user_a", "c0001") == "haeyo"
    assert table.is_firm("sys_ai", "user_a", "c0001")
    assert table.level_at("user_a", "sys_ai", "c0001") == "hae"
    assert table.is_firm("user_a", "sys_ai", "c0001")
    assert table.level_at("buddy_ai", "user_a", "c0001") == "hae"
    assert table.is_firm("buddy_ai", "user_a", "c0001")
    assert table.level_at("sp_a", "sp_b", "c0001") == "hae"
    assert table.is_firm("sp_a", "sp_b", "c0001")


def test_unmapped_acronym_ai_resolves_to_system_ai_haeyo_and_two_turn_dash_gate():
    cues, obs = _observed([("A.I.S.", "Peter"), ("Peter", "A.I.S.")])
    pairs = register.cue_pairs(_context_map(), cues, obs)
    assert pairs["c0001"] == (register.SYSTEM_AI, "peter")
    assert pairs["c0002"] == ("peter", register.SYSTEM_AI)
    assert register.required_pairs(pairs) == []
    table = _table(
        _edge("peter", "tony", "hae", step=10, relation="peer", apparent_generation="same", history="raised_together"),
        _edge("tony", "peter", "haeyo", step=1, relation="ai_to_user", apparent_generation="same", history="none", ai_user=True),
    )
    assert table.level_at(register.SYSTEM_AI, "peter", "c0001") == "haeyo"
    assert table.level_at("peter", register.SYSTEM_AI, "c0002") == "hae"
    gate = register.RegisterGate(table, {"c0001": (register.SYSTEM_AI, "peter"), "c0003": ("peter", "tony")})
    assert gate.check("regen", ["c0001"], "드론 공격을 시작할 거야?", "드론 공격을 시작할까요?")
    assert gate.check("regen", ["c0003"], "- 멋지다\n- 내 기능 한번 볼래?", "- 멋지다\n- 제 기능 보실래요?")


def test_canonicalize_edge_clears_upward_and_civil_mentor_pronouns_and_renders_none_ban():
    parent_edge = _edge(
        "travis", "bonnie", "hae", step=3, relation="own_parent",
        apparent_generation="addressee_older", history="raised_by_addressee",
        pronoun="너", vocative="엄마",
    )
    mentor_edge = _edge(
        "travis", "bill", "haeyo", step=6, relation="mentor",
        apparent_generation="addressee_older", history="met_as_adults",
        pronoun="당신", vocative="빌",
    )
    spouse_edge = _edge(
        "donald", "bonnie", "hae", step=10, relation="partner",
        apparent_generation="same", history="met_as_adults",
        pronoun="당신", vocative="당신",
    )
    assert register.canonicalize_edge(parent_edge).pronoun == "none"
    assert register.canonicalize_edge(mentor_edge).pronoun == "none"
    assert register.canonicalize_edge(spouse_edge).pronoun == "당신"

    sr = register.SceneRegister(level="hae", pronoun="none", vocative="엄마")
    rendered = "\n".join(register.render_scene_register("travis>bonnie", sr))
    assert "2인칭대명사생략(너/너희/당신 금지)" in rendered


def test_enforce_pragmatic_address_repairs_parent_neohui_and_mentor_dangsin():
    from substage.models import Character, ChunkObservations, ContextMap, Cue, CueObservation, RegisterMap, Scene, VoiceCard

    cm = ContextMap(
        synopsis="test",
        tone_baseline="drama",
        characters=[
            Character(id="travis", en_name="Travis Kalanick", ko_name="트래비스 캘러닉", aliases=["TK"], persona="CEO", voice=VoiceCard(idiolect="test")),
            Character(id="bonnie", en_name="Bonnie Kalanick", ko_name="보니 캘러닉", aliases=["Mom"], persona="Mom", voice=VoiceCard(idiolect="test")),
            Character(id="bill", en_name="Bill Gurley", ko_name="빌 걸리", aliases=["Bill"], persona="VC", voice=VoiceCard(idiolect="test")),
        ],
    )
    cues = [
        Cue(id="c0485", start_ms=1000, end_ms=2500, text="I invited all you guys.", raw="I invited all you guys."),
        Cue(id="c0201", start_ms=3000, end_ms=4500, text="But if I had a mentor, like you...", raw="But if I had a mentor, like you..."),
        Cue(id="c0799", start_ms=5000, end_ms=6500, text="You are Uber.", raw="You are Uber."),
    ]
    obs = [
        ChunkObservations(
            chunk_id="chunk_000", start_ms=0, end_ms=10000,
            cues=[
                CueObservation(cue_id="c0485", speaker="Travis Kalanick", addressee="Bonnie Kalanick"),
                CueObservation(cue_id="c0201", speaker="Travis Kalanick", addressee="Bill Gurley"),
                CueObservation(cue_id="c0799", speaker="Travis Kalanick", addressee="audience"),
            ],
        )
    ]
    rm = RegisterMap(
        edges=[
            _edge(
                "travis", "bonnie", "hae", step=3, relation="own_parent",
                apparent_generation="addressee_older", history="raised_by_addressee",
                pronoun="none", vocative="엄마",
            ),
            _edge(
                "travis", "bill", "haeyo", step=6, relation="mentor",
                apparent_generation="addressee_older", history="met_as_adults",
                pronoun="none", vocative="빌",
            ),
        ]
    )
    scenes = [Scene(id="s001", start_ms=0, end_ms=10000, cue_ids=["c0485", "c0201", "c0799"])]
    ko = {
        "c0485": "너희 다 초대했잖아",
        "c0201": "하지만 당신 같은 멘토가 있다면...",
        "c0799": "너희가 바로 우버야",
    }
    fixed = register.enforce_pragmatic_address(
        ko, cues, rm, context_map=cm, observations=obs, scenes=scenes
    )
    assert "너희" not in fixed["c0485"]
    assert fixed["c0485"] == "다 초대했잖아"
    assert "당신" not in fixed["c0201"]
    assert "빌" in fixed["c0201"]
    assert "너희" not in fixed["c0799"]
