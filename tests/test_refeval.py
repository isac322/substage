"""Offline tests for reference-based evaluation (refeval.py). No Gemini, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from substage import refeval
from substage.config import Settings, Workspace
from substage.models import Callback, Character, Cue, ContextMap, RegisterEdge, RegisterMap, RegisterShift, Scene, VoiceCard
from substage.evaluate import LABELS
from substage.refeval import Line, align
from substage.store import save_model


def _cue(cid: str, start: int, end: int, text: str = "hello") -> Cue:
    return Cue(id=cid, start_ms=start, end_ms=end, text=text, raw=text)


def _lines(*spans: tuple[int, int, str]) -> list[Line]:
    return [Line(i, s, e, t) for i, (s, e, t) in enumerate(spans)]


# ---------------------------------------------------------------- alignment


def test_align_one_to_many_many_to_one_and_omissions():
    en = [
        _cue("c1", 1000, 3000, "a"),      # split into two official lines (1:N)
        _cue("c2", 5000, 6000, "b"),      # merged with c3 into one official line (N:1)
        _cue("c3", 6000, 7000, "c"),
        _cue("c4", 9000, 10000, "d"),     # candidate only (official dropped it)
    ]
    official = _lines(
        (1000, 2000, "하나"), (2000, 3100, "둘"),
        (5000, 7000, "셋 넷"),
        (12000, 13000, "다섯"),            # nothing overlaps: candidate omission
    )
    cand = _lines(
        (1000, 3000, "하나 둘"),
        (5000, 6000, "셋"), (6000, 7000, "넷"),
        (9000, 10000, "여섯"),
        (20000, 21000, "덤"),              # nothing overlaps: extra candidate line
    )
    groups = align(en, official, {"A": cand})
    shape = [([c.id for c in g.en], [l.text for l in g.official], [l.text for l in g.candidates["A"]]) for g in groups]
    assert shape == [
        (["c1"], ["하나", "둘"], ["하나 둘"]),
        (["c2", "c3"], ["셋 넷"], ["셋", "넷"]),
        (["c4"], [], ["여섯"]),
        ([], ["다섯"], []),
        ([], [], ["덤"]),
    ]
    assert [g.id for g in groups] == ["g0001", "g0002", "g0003", "g0004", "g0005"]


def test_overlap_threshold_ms_or_fraction():
    assert refeval.overlaps(0, 1000, 800, 2000)          # 200 ms
    assert not refeval.overlaps(0, 1000, 850, 2000)      # 150 ms of a 1000 ms cue
    assert refeval.overlaps(0, 400, 280, 3000)           # 120 ms = 30% of the 400 ms cue
    assert not refeval.overlaps(0, 1000, 1000, 2000)     # touching is not overlapping


def test_same_track_never_links():
    en = [_cue("c1", 0, 2000), _cue("c2", 1000, 3000)]
    groups = align(en, [], {})
    assert len(groups) == 2


def test_assign_scenes_by_cue_then_overlap():
    en = [_cue("c1", 0, 1000)]
    groups = align(en, _lines((0, 1000, "가"), (50_000, 51_000, "나")), {})
    scenes = [
        Scene(id="s1", start_ms=0, end_ms=1000, cue_ids=["c1"]),
        Scene(id="s2", start_ms=40_000, end_ms=60_000, cue_ids=[]),
    ]
    refeval.assign_scenes(groups, scenes)
    assert [g.scene_id for g in groups] == ["s1", "s2"]


# ---------------------------------------------------------------- speech level


def test_line_levels_split_dash_turns_and_sentences():
    assert refeval.line_levels("-가자\n-네, 갈게요") == ["hae", "haeyo"]
    assert refeval.line_levels("알았어요. 가자") == ["hae"]
    assert refeval.line_levels("공기 청정제 말고\n저 여자") == []
    assert refeval.line_levels("이제 누군지 기억하시네") == []


def test_group_level_majority_tie_latest():
    assert refeval.group_level(_lines((0, 1, "가자"), (1, 2, "갈게요"))) == "haeyo"
    assert refeval.group_level(_lines((0, 1, "가자"), (1, 2, "갈게요"), (2, 3, "했어"))) == "hae"
    assert refeval.group_level(_lines((0, 1, "여섯"))) == "other"


# ---------------------------------------------------------------- loading


_SAMI = """<SAMI>
<HEAD><TITLE>x</TITLE>
<STYLE TYPE="text/css"><!--
.KRCC { Name:한국어; lang:ko-KR; SAMIType:CC; }
--></STYLE></HEAD>
<BODY>
<-- comment -->
<SYNC Start=51299><P Class=KRCC>“롭 L.”
<SYNC Start=52429><P Class=KRCC>&nbsp;
<SYNC Start=57848><P Class=KRCC>제작<br>개허접 필름
<SYNC Start=60348><P Class=KRCC>&nbsp;
<SYNC Start=61810><P Class=KRCC><font color="#ffff00">출연</font>&amp; 또라이
<SYNC Start=64310><P Class=KRCC>네, 바로
</BODY>
</SAMI>
"""


def test_sami_cp949_with_nbsp_clears(tmp_path):
    p = tmp_path / "official.smi"
    p.write_bytes(_SAMI.encode("cp949"))
    lines = refeval.load_ko(p)
    assert [(ln.idx, ln.start_ms, ln.end_ms, ln.text) for ln in lines] == [
        (0, 51299, 52429, "“롭 L.”"),
        (1, 57848, 60348, "제작\n개허접 필름"),
        (2, 61810, 64310, "출연& 또라이"),
        (3, 64310, 64310 + 3000, "네, 바로"),
    ]


def test_sami_prefers_korean_class():
    text = (
        "<SYNC Start=0><P Class=ENCC>Hello\n<SYNC Start=0><P Class=KRCC>안녕\n"
        "<SYNC Start=1000><P Class=KRCC>&nbsp;\n<SYNC Start=1000><P Class=ENCC>&nbsp;\n"
    )
    assert [ln.text for ln in refeval.parse_sami(text)] == ["안녕"]


def test_srt_utf8_and_cp949(tmp_path):
    body = "1\n00:00:01,000 --> 00:00:02,500\n<i>-가자</i>\n-네\n\n2\n00:00:03,000 --> 00:00:04,000\n좋아\n"
    for enc in ("utf-8", "cp949"):
        p = tmp_path / f"o-{enc}.srt"
        p.write_bytes(body.encode(enc))
        assert [(ln.start_ms, ln.end_ms, ln.text) for ln in refeval.load_ko(p)] == [
            (1000, 2500, "-가자\n-네"),
            (3000, 4000, "좋아"),
        ]


def test_srt_bidi_controls_stripped(tmp_path):
    p = tmp_path / "o.srt"
    p.write_text("1\n00:00:01,000 --> 00:00:02,500\n\u200e- 건배!\n\u202a- 가자\u202c\n", encoding="utf-8")
    lines = refeval.load_ko(p)
    assert [ln.text for ln in lines] == ["- 건배!\n- 가자"]
    assert refeval.has_dash(lines)


# ---------------------------------------------------------------- chrF++


def test_chrf_matches_sacrebleu():
    hyps = ["나랑 결혼했으면 좋았을 텐데", "네, 기타예요. 진짜 예쁘죠?", ""]
    refs = ["나랑 결혼했어야 했는데", "네, 기타예요 정말 예쁘죠", "프란시스"]
    # sacrebleu 2.x CHRF(word_order=2): corpus 36.928..., sentence 26.240...
    assert refeval.corpus_chrf(list(zip(hyps, refs))) == pytest.approx(36.92826982819208, abs=1e-9)
    assert refeval.chrf_score(refeval.chrf_stats(hyps[0], refs[0])) == pytest.approx(26.240293065951718, abs=1e-9)
    assert refeval.corpus_chrf([("같다", "같다")]) == pytest.approx(100.0)


def test_listwise_metrics_keep_groupwise_chrf_and_coverage_without_lexical_units():
    groups = [
        refeval.Group(
            id="g0001", start_ms=1000, end_ms=3000,
            official=[Line(0, 1000, 3000, "네가")],
            candidates={"arm": [Line(0, 1000, 5000, "네가 떠났으면 좋겠어")]},
        ),
        refeval.Group(
            id="g0002", start_ms=3000, end_ms=5000,
            official=[Line(1, 3000, 5000, "떠났으면 좋겠어")],
        ),
    ]
    metrics = refeval.arm_metrics(groups, "arm", Settings())
    expected = round(refeval.corpus_chrf([
        ("네가 떠났으면 좋겠어", "네가"), ("", "떠났으면 좋겠어"),
    ]), 2)

    assert metrics["chrf"] == metrics["chrf_fixed_group"] == expected
    assert metrics["coverage"] == 0.5
    assert metrics["length_ratio"] == 1
    assert metrics["lines"] == 1


# ---------------------------------------------------------------- degradation


def test_swap_register_both_directions():
    assert refeval.swap_register("갈게요") == "갈게"
    assert refeval.swap_register("뭐예요?") == "뭐야?"
    assert refeval.swap_register("그렇죠") == "그렇지"
    assert refeval.swap_register("가자") == "가요"
    assert refeval.swap_register("했다") == "했어요"
    assert refeval.swap_register("갈까?") == "갈까요?"
    assert refeval.swap_register("-가자\n-네, 갈게요") == "-가요\n-네, 갈게"
    assert refeval.swap_register("갑니다") == "갑니다"


def _context_map() -> ContextMap:
    voice = VoiceCard(idiolect="x")
    return ContextMap(
        synopsis="s",
        tone_baseline="t",
        characters=[
            Character(id="wade", en_name="Wade", ko_name="웨이드 윌슨", persona="p", voice=voice),
            Character(id="vanessa", en_name="Vanessa", ko_name="바네사", persona="p", voice=voice),
            Character(id="weasel", en_name="Weasel", ko_name="위즐", persona="p", voice=voice),
        ],
        callbacks=[Callback(motif="m", setup_cues=["c0010"], payoff_cues=["c0090"])],
    )


def test_targeted_cue_ids_shift_bounds_and_callbacks():
    shift = RegisterShift(at_cue="c0050", level="haeyo", reason="role_on", until_cue="c0060")
    edge = RegisterEdge(
        speaker="wade", addressee="vanessa", relation="partner", apparent_generation="same",
        history="met_as_adults", stance="civil", step=10, level="hae", shifts=[shift],
    )
    ids = refeval.targeted_cue_ids(_context_map(), RegisterMap(edges=[edge]))
    assert ids == {"c0010", "c0050", "c0060", "c0090"}
    assert refeval.targeted_cue_ids(None, None) == set()


def test_degrade_is_deterministic():
    names = refeval.name_swaps(_context_map())
    # characters by id: vanessa → wade → weasel → vanessa
    assert names == {"바네사": "웨이드 윌슨", "웨이드 윌슨": "위즐", "웨이드": "위즐", "위즐": "바네사"}
    lines = _lines(*[(i * 1000, i * 1000 + 900, t) for i, t in enumerate(
        ["웨이드, 갈게요", "바네사", "가자", "위즐이야", "좋네요", "삭제될 줄", "했어", "뭐요?"]
    )])
    first = refeval.degrade(lines, names)
    assert first == refeval.degrade(lines, names)
    assert first == {
        0: "위즐, 갈게",            # idx 0: register swap + name
        1: "웨이드 윌슨",
        2: "가자",
        3: "바네사이에요",           # idx 3: 야→이에요, then name rotation
        4: "좋네요",
        5: "",                     # idx 5: deleted
        6: "했어요",
        7: "뭐요?",
    }


# ---------------------------------------------------------------- windows


def _groups(n: int, scene_size: int) -> list[refeval.Group]:
    en = [_cue(f"c{i:04d}", i * 2000, i * 2000 + 1500) for i in range(n)]
    off = _lines(*[(i * 2000, i * 2000 + 1500, f"줄{i}") for i in range(n)])
    groups = align(en, off, {})
    for i, g in enumerate(groups):
        g.scene_id = f"s{i // scene_size:03d}"
    return groups


def test_windows_sizes_and_seeded_stratified_selection():
    groups = _groups(100, 13)
    windows = refeval.build_windows(groups)
    sizes = [len(w.groups) for w in windows]
    assert sum(sizes) == 100
    assert all(refeval.WINDOW_MIN <= s <= refeval.WINDOW_MAX for s in sizes), sizes
    targeted = {"c0010", "c0050", "c0090"}
    a = refeval.select_windows(refeval.build_windows(groups), 4, 7, targeted)
    b = refeval.select_windows(refeval.build_windows(groups), 4, 7, targeted)
    assert [w.id for w in a] == [w.id for w in b]
    assert sum(w.reason == "targeted" for w in a) == 2
    assert [w.start_ms for w in a] == sorted(w.start_ms for w in a)


# ---------------------------------------------------------------- judge + run


class FakeGC:
    """Returns a verdict for every label; score by the candidate's text."""

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, *, stage, prompt, schema, model=None, **kw):
        assert stage == refeval.STAGE
        self.prompts.append(prompt)
        labels_line = next(ln for ln in prompt.splitlines() if ln.startswith("Return exactly one entry"))
        labels = labels_line.split("(")[1].split(")")[0].split(", ")
        n = int(labels_line.rsplit("1..", 1)[1].split(".")[0])
        items = [b for b in prompt.split("\n\n") if b.startswith("[")]
        cands = []
        for label in labels:
            same = all(
                next(r for r in it.splitlines() if r.startswith(f"{label}: ")).split(": ", 1)[1]
                == next(r for r in it.splitlines() if r.startswith("GOLD: ")).split(": ", 1)[1]
                for it in items
            )
            cands.append({
                "label": f"Candidate {label}",
                "score": 100 if same else 40,
                "items": [{"item": i, "category": "none" if same else "register", "severity": 0 if same else 2}
                          for i in range(1, n + 1)],
            })
        return schema.model_validate({"candidates": cands})


def _ws(root: Path, video_id: str) -> Workspace:
    return Workspace(video_id=video_id, data_dir=root / "data" / video_id,
                     cache_dir=root / "cache" / video_id, log_dir=root / "log" / video_id)


def _write_srt(path: Path, spans: list[tuple[int, int, str]]) -> None:
    def ts(ms: int) -> str:
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{i}\n{ts(s)} --> {ts(e)}\n{t}\n" for i, (s, e, t) in enumerate(spans, 1)), encoding="utf-8"
    )


def _refeval_video(tmp_path, monkeypatch, n: int = 40):
    """Base video with 4 scenes; arm B == gold, arm A = 반말 on the first 30 lines."""
    video = "video"
    base = _ws(tmp_path, video)
    cues = [_cue(f"c{i:04d}", i * 3000, i * 3000 + 2000, f"line {i}") for i in range(n)]
    save_model(base.path("cues.json"), cues)
    save_model(base.path("scenes.json"), [
        Scene(id=f"s{k}", start_ms=k * 30000, end_ms=k * 30000 + 29000, cue_ids=[c.id for c in cues[k * 10:(k + 1) * 10]])
        for k in range(4)
    ])
    save_model(base.path("context_map.yaml"), _context_map())
    gold = [(i * 3000, i * 3000 + 2000, f"웨이드, {i}번 갈게요") for i in range(n)]
    ref = tmp_path / "official.srt"
    _write_srt(ref, gold)
    _write_srt(base.data_dir / f"{video}.ko.srt", gold)  # arm B == gold
    arm_a = _ws(tmp_path, f"{video}__a")
    _write_srt(arm_a.data_dir / f"{video}.ko.srt", [(s, e, t.replace("갈게요", "갈게")) for s, e, t in gold[:30]])
    monkeypatch.setattr(refeval, "workspace", lambda fid: _ws(tmp_path, fid))
    return video, ref, {video: base.data_dir, f"{video}__a": arm_a.data_dir}.__getitem__


def test_run_refeval_end_to_end(tmp_path, monkeypatch):
    n = 40
    video, ref, resolve = _refeval_video(tmp_path, monkeypatch, n)
    gc = FakeGC()
    report = refeval.run_refeval(
        video, {"A": f"{video}__a", "B": video}, ref, n_scenes=3, seed=1, out_dir=tmp_path / "out",
        settings=Settings(), judge_factory=lambda ws, s, m: gc, resolve=resolve,
    )
    mA, mB = report["metrics"]["A"], report["metrics"]["B"]
    assert mB["chrf"] == pytest.approx(100.0) and mB["coverage"] == 1.0 and mB["speech_level_agreement"] == 1.0
    assert mA["coverage"] == 0.75 and mA["speech_level_agreement"] == 0.0
    assert mB["judge"]["score"] == 100 and mA["judge"]["score"] == 40
    assert mA["judge"]["errors"]["register"] > 0
    judge = Settings().model_main
    assert report["judge_models"] == [judge]
    assert report["calibration"][judge]["official"]["score"] == 100
    assert report["calibration"][judge]["degraded"]["score"] == 40
    assert report["calibration"][judge]["passed"] is True
    assert len(report["scenes"]) == 3
    # blind: arm names and workspace ids never reach the prompt
    for p in gc.prompts:
        rows = p.splitlines()
        assert "__a" not in p
        assert not any(r.startswith(("A: ", "B: ", "official: ", "degraded: ")) for r in rows)
        assert any(r.startswith(tuple(f"{lb}: " for lb in LABELS)) for r in rows)
    on_disk = json.loads((tmp_path / "out" / "refeval.json").read_text(encoding="utf-8"))
    assert on_disk["groups"][0]["candidates"]["B"] == ["웨이드, 0번 갈게요"]
    assert "| A (video__a) |" in (tmp_path / "out" / "refeval.md").read_text(encoding="utf-8")
    # checkpointed: rerun makes no calls
    before = len(gc.prompts)
    refeval.run_refeval(video, {"A": f"{video}__a", "B": video}, ref, 3, 1, tmp_path / "out2",
                        settings=Settings(), judge_factory=lambda ws, s, m: gc, resolve=resolve)
    assert len(gc.prompts) == before

    exported = refeval.export_groups(video, {"A": f"{video}__a"}, ref, resolve=resolve)
    assert len(exported) == n
    assert set(exported[0]) >= {"id", "scene_id", "start_ms", "end_ms", "en", "official", "candidates"}
    assert exported[35]["candidates"]["A"] == [] and exported[35]["official"] == ["웨이드, 35번 갈게요"]


def test_parse_judge_models():
    assert refeval.parse_judge_models(None, "g") == ["g"]
    assert refeval.parse_judge_models([], "g") == ["g"]
    assert refeval.parse_judge_models([" , "], "g") == ["g"]
    assert refeval.parse_judge_models(
        ["gemini-x,codex/gpt-y", "anthropic/claude-z", "gemini-x"], "g"
    ) == ["gemini-x", "codex/gpt-y", "anthropic/claude-z"]


class ConstJudge(FakeGC):
    """FakeGC with every score replaced by a constant."""

    def __init__(self, score: int):
        super().__init__()
        self.score = score
        self.models: list[str] = []

    def generate(self, *, model=None, **kw):
        self.models.append(model)
        v = super().generate(model=model, **kw)
        for c in v.candidates:
            c.score = self.score
        return v


def test_run_refeval_multiple_judges(tmp_path, monkeypatch):
    video, ref, resolve = _refeval_video(tmp_path, monkeypatch)
    judges = {"gemini-j": FakeGC(), "anthropic/claude-j": ConstJudge(60)}
    made: list[str] = []

    def factory(ws, settings, model):
        made.append(model)
        return judges[model]

    arms = {"A": f"{video}__a", "B": video}
    models = list(judges)
    report = refeval.run_refeval(video, arms, ref, 3, 1, tmp_path / "out", settings=Settings(),
                                 judge_models=models, judge_factory=factory, resolve=resolve)
    assert sorted(made) == sorted(models)
    assert judges["anthropic/claude-j"].models and set(judges["anthropic/claude-j"].models) == {"anthropic/claude-j"}
    mA, mB = report["metrics"]["A"], report["metrics"]["B"]
    assert mB["judges"]["gemini-j"]["score"] == 100 and mB["judges"]["anthropic/claude-j"]["score"] == 60
    assert mB["judge"]["score"] == 80 and mA["judge"]["score"] == 50  # (40 + 60) / 2
    assert mA["judge"]["errors"]["register"] == mA["judges"]["gemini-j"]["errors"]["register"]
    # calibration per judge: the constant judge cannot separate official from degraded
    assert report["calibration"]["gemini-j"]["passed"] is True
    assert report["calibration"]["anthropic/claude-j"]["passed"] is False
    sc = report["scenes"][0]
    assert sc["verdicts"]["B"]["scores"] == {"gemini-j": 100, "anthropic/claude-j": 60}
    assert sc["verdicts"]["B"]["score"] == 80
    assert {ln["judge"] for ln in sc["verdicts"]["A"]["lines"]} == set(models)
    md = (tmp_path / "out" / "refeval.md").read_text(encoding="utf-8")
    for head in ("### Judge: gemini-j", "### Judge: anthropic/claude-j", "### Judge: mean",
                 "### anthropic/claude-j"):
        assert head in md
    # checkpoints are keyed per judge model: one file per (kind, window, judge)
    ck = sorted(p.name for p in (tmp_path / "data" / video / "checkpoints" / refeval.STAGE).iterdir())
    assert len(ck) == 2 * 2 * 3  # arms + calib, 2 judges, 3 windows
    assert sum("anthropic_claude-j" in n for n in ck) == 6 and sum("gemini-j" in n for n in ck) == 6
    # adding a judge reuses the others' checkpoints; only the new judge is called
    before = {m: len(j.prompts) for m, j in judges.items()}
    judges["codex/gpt-j"] = ConstJudge(70)
    report2 = refeval.run_refeval(video, arms, ref, 3, 1, tmp_path / "out2", settings=Settings(),
                                  judge_models=[*models, "codex/gpt-j"], judge_factory=factory, resolve=resolve)
    assert {m: len(judges[m].prompts) for m in models} == before
    assert len(judges["codex/gpt-j"].prompts) == 6
    assert report2["metrics"]["B"]["judge"]["score"] == round((100 + 60 + 70) / 3, 2)


def test_display_rates_reading_speed_metrics():
    lines = [Line(0, 0, 1000, "가나 다라"), Line(1, 2000, 4000, "가" * 20), Line(2, 5000, 8000, "가나다")]
    r = refeval.display_rates(lines, Settings())
    # 4 / 1 s, 20 / 2 s (> cps_max 8), 3 / 3 s; whitespace is not counted
    assert r["cps_over_rate"] == round(1 / 3, 4)
    assert r["median_cps"] == 4.0 and r["median_duration_s"] == 2.0
    assert r["cpl_over_rate"] == round(1 / 3, 4)
