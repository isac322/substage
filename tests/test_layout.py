"""S5c layout: timing extension, merge, split, filler drop, condense, emit."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pysubs2
import pytest

from substage import emit as emit_mod
from substage import layout as lay
from substage.config import Settings, Workspace
from substage.gemini import GeminiBlocked
from substage.models import (
    Cue, VideoInfo, OutEvent, RetainedFragment, SemanticLineage, SemanticSource, SourceAct, TranslatedCue,
)
from substage.pipeline import STAGES

S = Settings()


def cue(cid: str, start: int, end: int, text: str = "Line.", **kw) -> Cue:
    return Cue(id=cid, start_ms=start, end_ms=end, text=text, raw=text, **kw)


def run(cues: list[Cue], ko: dict[str, str], scene_of=None, speakers=None, gc=None) -> list[OutEvent]:
    scene_of = scene_of if scene_of is not None else {c.id: "s001" for c in cues}
    return lay.layout(cues, ko, scene_of, speakers or {}, S, gc)


# ---------------------------------------------------------------- timing


def test_short_cue_stretches_to_min_duration_when_free():
    assert lay.fit_end(0, 800, 3, None, S) == 1000


def test_extension_stops_extend_ms_past_english_end():
    assert lay.fit_end(0, 2000, 20, None, S) == 2000 + S.extend_ms


def test_extension_respects_next_start_and_chains_near_misses():
    # free gap after the 250 ms extension would be 67 ms: snap to a chained gap
    assert lay.fit_end(0, 2000, 20, 2400, S) == 2400 - S.min_gap_ms
    # a real pause stays a pause
    assert lay.fit_end(0, 2000, 20, 3000, S) == 2250
    # never overlap the next event, even when the English does
    assert lay.fit_end(0, 3000, 5, 2500, S) == 2500 - S.min_gap_ms


def test_extension_capped_by_max_duration_but_never_below_english_end():
    assert lay.fit_end(0, 5600, 100, None, S) == S.max_duration_ms
    assert lay.fit_end(0, 6500, 100, None, S) == 6500


def test_events_keep_english_start_and_never_overlap():
    cues = [cue("c1", 0, 2000, "Hello there."), cue("c2", 2050, 4000, "Hi."), cue("c3", 9000, 9300, "Go!")]
    ev = run(cues, {"c1": "안녕하세요 반가워요", "c2": "안녕", "c3": "가"}, speakers={"c1": "a", "c2": "b", "c3": "a"})
    assert [e.start_ms for e in ev] == [0, 2050, 9000]
    assert all(a.end_ms + S.min_gap_ms <= b.start_ms for a, b in zip(ev, ev[1:]))
    assert ev[2].end_ms == 10000  # 0.3 s cue stretched to 1 s


# ---------------------------------------------------------------- merge


def _pair(**kw):
    a = cue("c1", 0, 600, "Are you really ready?", **kw.pop("a", {}))
    b = cue("c2", kw.pop("b_start", 700), 1500, "Yes, let's go.", **kw.pop("b", {}))
    return [a, b], {"c1": "너 진짜 준비됐어?", "c2": "응 가자"}


def test_two_short_cues_by_two_speakers_merge_into_dash_lines():
    cues, ko = _pair()
    ev = run(cues, ko, speakers={"c1": "peter", "c2": "mj"})
    assert len(ev) == 1
    assert ev[0].ko == "- 너 진짜 준비됐어?\n- 응 가자"
    assert ev[0].cue_ids == ["c1", "c2"]
    assert ev[0].start_ms == 0 and ev[0].end_ms == 1750


def test_unfinished_sentence_merges_with_same_speaker_continuation():
    cues = [cue("c1", 0, 1500, "I think that we"), cue("c2", 1600, 3000, "should go now.")]
    ev = run(cues, {"c1": "내 생각엔 우리가", "c2": "지금 가야 해"}, speakers={"c1": "peter", "c2": "peter"})
    assert [e.cue_ids for e in ev] == [["c1", "c2"]]
    assert "-" not in ev[0].ko and ev[0].ko.replace("\n", " ") == "내 생각엔 우리가 지금 가야 해"


def test_finished_sentences_of_readable_cues_stay_apart():
    cues = [cue("c1", 0, 1500, "I'm home."), cue("c2", 1600, 3000, "Where is everyone?")]
    ev = run(cues, {"c1": "나 왔어", "c2": "다들 어디 갔어?"}, speakers={"c1": "peter", "c2": "peter"})
    assert len(ev) == 2


def test_no_merge_across_scenes_large_gaps_or_lyrics():
    speakers = {"c1": "peter", "c2": "mj"}
    cues, ko = _pair()
    assert len(run(cues, ko, scene_of={"c1": "s001", "c2": "s002"}, speakers=speakers)) == 2
    cues, ko = _pair(b_start=1000)  # 400 ms gap
    assert len(run(cues, ko, speakers=speakers)) == 2
    cues, ko = _pair(a={"lyric": True})
    assert len(run(cues, ko, speakers=speakers)) == 2
    cues, ko = _pair(b={"source": "ost"})
    assert len(run(cues, ko, speakers=speakers)) == 2


def test_merge_never_creates_three_speaker_lines():
    cues = [cue("c1", 0, 600, "- Now?\n- Now."), cue("c2", 650, 1200, "Go!")]
    ev = run(cues, {"c1": "- 지금?\n- 지금", "c2": "가!"}, speakers={"c2": "tony"})
    assert len(ev) == 2


# ---------------------------------------------------------------- split


def test_long_two_sentence_cue_splits_with_proportional_times():
    cues = [cue("c1", 0, 4000, "I can't. I have to go home now.")]
    ev = run(cues, {"c1": "난 못 해. 지금 집에 가야 해"}, speakers={"c1": "peter"})
    assert [e.ko for e in ev] == ["난 못 해", "지금 집에 가야 해"]
    # 4000 ms - min_gap split 3:7 by characters of the pieces (난못해 = 3, 지금집에가야해 = 7)
    t1 = round((4000 - S.min_gap_ms) * 3 / 10)
    assert (ev[0].start_ms, ev[0].end_ms) == (0, t1)
    assert (ev[1].start_ms, ev[1].end_ms) == (t1 + S.min_gap_ms, 4000)
    assert ev[0].cue_ids == ev[1].cue_ids == ["c1"]


def test_short_cue_with_two_sentences_is_not_split():
    ev = run([cue("c1", 0, 2500, "No. Go.")], {"c1": "안 돼. 가"})
    assert len(ev) == 1


def test_long_dash_cue_that_fits_cpl_stays_whole():
    ev = run([cue("c1", 0, 4000, "- Where were you?\n- Out.")], {"c1": "- 어디 있었어?\n- 밖에"})
    assert [e.ko for e in ev] == ["- 어디 있었어?\n- 밖에"]


def test_overflowing_dash_cue_splits_into_its_two_turns():
    ev = run(
        [cue("c1", 0, 4000, "- Where were you all night?\n- Out with friends.")],
        {"c1": "- 어젯밤 내내 대체 어디 가 있었던 거야?\n- 친구들이랑 밖에 있었어"},
    )
    assert len(ev) == 2
    assert all(e.end_ms - e.start_ms >= S.split_piece_min_ms for e in ev)

# ---------------------------------------------------------------- drops


def test_empty_and_filler_only_cues_produce_no_event():
    cues = [cue("c1", 0, 900, "Uh..."), cue("c2", 3000, 4000, "[explosion]"), cue("c3", 6000, 7000, "Hey.")]
    ev = run(cues, {"c1": "어…", "c2": "", "c3": "이봐"})
    assert [e.cue_ids for e in ev] == [["c3"]]


# ---------------------------------------------------------------- determinism


def test_layout_is_deterministic_and_retiming_is_a_fixpoint():
    cues = [cue(f"c{i}", i * 1100, i * 1100 + 700, "Wait" if i % 2 else "Go.") for i in range(12)]
    ko = {c.id: ("잠깐만" if i % 2 else "가자") for i, c in enumerate(cues)}
    speakers = {c.id: ("a" if i % 3 else "b") for i, c in enumerate(cues)}
    first = run(cues, ko, speakers=speakers)
    assert first == run(cues, ko, speakers=speakers)
    evs = lay._merge_all(lay._split_all(lay._initial(cues, ko, {c.id: "s" for c in cues}, speakers), S), S)
    lay._retime(evs, S)
    ends = [e.end for e in evs]
    lay._retime(evs, S)
    assert ends == [e.end for e in evs]


# ---------------------------------------------------------------- condense


class FakeGC:
    def __init__(self, items=None, blocked=False):
        self.items, self.blocked, self.calls = items or [], blocked, []
        self.settings = SimpleNamespace(model_light="fake-light")

    def generate(self, *, stage, prompt, schema, model=None, **kw):
        self.calls.append({"stage": stage, "prompt": prompt, "model": model})
        if self.blocked:
            raise GeminiBlocked(stage, "blocked")
        return schema(items=self.items)


def _fast():
    cues = [cue("c1", 0, 1000, "I told you a hundred times already!"), cue("c2", 1100, 3000, "Fine.")]
    return cues, {"c1": "내가 이미 백 번은 말했잖아 정말로", "c2": "알았어"}


def test_condense_shortens_only_events_over_cps_max():
    cues, ko = _fast()
    gc = FakeGC([{"event_id": "e0001", "ko": "백 번 말했잖아"}])
    ev = run(cues, ko, speakers={"c1": "a", "c2": "b"}, gc=gc)
    assert [c["stage"] for c in gc.calls] == ["layout-condense"] and gc.calls[0]["model"] == "fake-light"
    assert "e0001" in gc.calls[0]["prompt"] and "e0002" not in gc.calls[0]["prompt"]
    assert ev[0].ko == "백 번 말했잖아"


def test_condense_blocked_keeps_text():
    cues, ko = _fast()
    ev = run(cues, ko, speakers={"c1": "a", "c2": "b"}, gc=FakeGC(blocked=True))
    assert ev[0].ko == ko["c1"]


# ---------------------------------------------------------------- emit / pipeline


def test_emit_writes_events_and_sidecar_cue_ids(tmp_path):
    ws = Workspace("video", tmp_path / "d", tmp_path / "c", tmp_path / "l")
    video = VideoInfo(video_id="video", title="F", video_path="v", duration_ms=1, audio_stream=1,
                    en_sub_path="e", en_sub_origin="user")
    cues, ko = _pair()
    events = run(cues, ko, speakers={"c1": "peter", "c2": "mj"})
    meta = {c.id: TranslatedCue(cue_id=c.id, ko=ko[c.id]) for c in cues}
    srt, side = emit_mod.emit(ws, video, cues, ko, meta, events)
    subs = pysubs2.load(str(srt), encoding="utf-8")
    assert [(e.start, e.end, e.plaintext) for e in subs] == [(0, 1750, "- 너 진짜 준비됐어?\n- 응 가자")]
    data = json.loads(side.read_text(encoding="utf-8"))
    assert data["events"][0]["cue_ids"] == ["c1", "c2"]
    assert [c["cue_id"] for c in data["cues"]] == ["c1", "c2"]


def test_layout_is_a_stage_before_emit():
    assert STAGES.index("recheck") < STAGES.index("layout") < STAGES.index("emit")


def _lineage(cues, fragments):
    return SemanticLineage(
        source=SemanticSource(scene_id="s1", acts=[
            SourceAct(ref=f"{c.id}:t1:a1", cue_id=c.id, turn_ordinal=1, act_ordinal=1,
                      start_char=0, end_char=len(c.text), text=c.text, meaning="source act")
            for c in cues
        ]),
        fragments_by_cue=fragments, structural_valid=True,
    )


def test_retained_two_voice_split_preserves_exact_piece_refs_without_per_event_dashes():
    c = cue("c1", 0, 4000, "Where? Outside.")
    fragments = [
        RetainedFragment(text="어디야?", refs=["c1:t1:a1"], speaker="a"),
        RetainedFragment(text="밖이야", refs=["c1:t2:a1"], speaker="b"),
    ]
    lineage = _lineage([c], {"c1": fragments})
    lineage.source.acts = [
        SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=6, text="Where?", speaker="a", meaning="question"),
        SourceAct(ref="c1:t2:a1", cue_id="c1", turn_ordinal=2, act_ordinal=1,
                  start_char=7, end_char=len(c.text), text="Outside.", speaker="b", meaning="answer"),
    ]
    events = lay.layout([c], {"c1": "- 어디야?\n- 밖이야"}, {"c1": "s1"}, {}, S, lineage=lineage)
    assert [event.ko for event in events] == ["어디야?", "밖이야"]
    assert [event.fragments[0].refs for event in events] == [["c1:t1:a1"], ["c1:t2:a1"]]
    assert all(event.cue_ids == ["c1"] for event in events)
    assert [event.fragments[0].speaker for event in events] == ["a", "b"]


def test_retained_merge_uses_actual_known_voice_not_source_turn_count():
    cues = [cue("c1", 0, 600, "Ready?"), cue("c2", 700, 1500, "Yes.")]
    fragments = {
        "c1": [RetainedFragment(text="준비됐어?", refs=["c1:t1:a1"], speaker="a")],
        "c2": [RetainedFragment(text="응", refs=["c2:t1:a1"], speaker="b")],
    }
    events = lay.layout(cues, {"c1": "준비됐어?", "c2": "응"}, {"c1": "s1", "c2": "s1"}, {}, S,
                        lineage=_lineage(cues, fragments))
    assert len(events) == 1
    assert events[0].ko == "- 준비됐어?\n- 응"
    assert events[0].cue_ids == ["c1", "c2"]
    assert [f.refs for f in events[0].fragments] == [["c1:t1:a1"], ["c2:t1:a1"]]


def test_unknown_retained_voices_keep_words_and_metadata_without_artificial_dashes():
    cues = [cue("c1", 0, 600, "Wait"), cue("c2", 700, 1500, "there.")]
    fragments = {
        "c1": [RetainedFragment(text="잠깐", refs=["c1:t1:a1"], speaker=None)],
        "c2": [RetainedFragment(text="거기서", refs=["c2:t1:a1"], speaker=None)],
    }
    events = lay.layout(cues, {"c1": "잠깐", "c2": "거기서"}, {"c1": "s1", "c2": "s1"}, {}, S,
                        lineage=_lineage(cues, fragments))
    assert [word for event in events for word in event.ko.split()] == ["잠깐", "거기서"]
    assert all(event.start_ms < event.end_ms for event in events)
    assert all(a.end_ms + S.min_gap_ms <= b.start_ms for a, b in zip(events, events[1:]))
    assert all("-" not in event.ko for event in events)
    retained = [fragment for event in events for fragment in event.fragments]
    assert [word for fragment in retained for word in fragment.text.split()] == ["잠깐", "거기서"]
    assert all(fragment.speaker is None for fragment in retained)


def test_single_act_text_split_keeps_lawful_one_to_many_ancestry_and_word_order():
    c = cue("c1", 0, 4000, "I cannot. I have to go home.")
    fragments = [RetainedFragment(text="난 못 해. 지금 집에 가야 해", refs=["c1:t1:a1"], speaker="a")]
    events = lay.layout([c], {"c1": fragments[0].text}, {"c1": "s1"}, {}, S,
                        lineage=_lineage([c], {"c1": fragments}))
    assert [event.ko for event in events] == ["난 못 해", "지금 집에 가야 해"]
    assert all(event.fragments[0].refs == ["c1:t1:a1"] for event in events)
    assert all(event.fragments[0].speaker == "a" for event in events)
    assert all("-" not in event.ko for event in events)


def test_actual_event_coverage_detects_lost_declared_essential_ref_after_layout():
    from substage.validate import lineage_issues

    c = cue("c1", 0, 2000, "Answer.")
    lineage = _lineage([c], {"c1": [
        RetainedFragment(text="답이야", refs=["c1:t1:a1"], speaker="a")
    ]})
    lineage.source.acts[0].essential = True
    assert lineage_issues(lineage) == []
    events = [OutEvent(start_ms=0, end_ms=2000, ko="그러네", cue_ids=["c1"], fragments=[
        RetainedFragment(text="그러네", refs=[], speaker=None, attribution="unverified",
                         prior_refs=["c1:t1:a1"])
    ])]
    assert {issue.rule for issue in lineage_issues(lineage, events=events)} == {
        "lineage_unverified", "essential_coverage"
    }


def test_alternating_two_known_voices_split_without_folding_third_turn_into_second():
    source = "First. Second. Third."
    c = cue("c1", 0, 6000, source)
    fragments = [
        RetainedFragment(text=text, refs=[f"c1:t{n}:a1"], speaker=speaker)
        for n, (text, speaker) in enumerate((("하나", "a"), ("둘", "b"), ("셋", "a")), 1)
    ]
    lineage = _lineage([c], {"c1": fragments})
    lineage.source.acts = [
        SourceAct(ref=f"c1:t{n}:a1", cue_id="c1", turn_ordinal=n, act_ordinal=1,
                  start_char=start, end_char=end, text=source[start:end], speaker=speaker,
                  meaning="ordered list item")
        for n, (start, end, speaker) in enumerate(((0, 6, "a"), (7, 14, "b"), (15, 21, "a")), 1)
    ]
    events = lay.layout([c], {"c1": "하나 둘 셋"}, {"c1": "s1"}, {}, S, lineage=lineage)
    assert [event.ko for event in events] == ["하나", "둘", "셋"]
    assert [event.fragments[0].speaker for event in events] == ["a", "b", "a"]
    assert [event.fragments[0].refs for event in events] == [["c1:t1:a1"], ["c1:t2:a1"], ["c1:t3:a1"]]


def test_writer_many_to_one_fragment_keeps_all_source_ids_and_time_evidence():
    from substage.validate import check, lineage_issues

    cues = [cue("c1", 0, 1500, "We should"), cue("c2", 1600, 3000, "leave now.")]
    fragment = RetainedFragment(text="이제 가야 해", refs=["c1:t1:a1", "c2:t1:a1"], speaker="a")
    lineage = _lineage(cues, {"c1": [fragment], "c2": []})
    for act in lineage.source.acts:
        act.essential = True
    events = lay.layout(cues, {"c1": fragment.text, "c2": ""}, {"c1": "s1", "c2": "s1"}, {}, S,
                        lineage=lineage)
    assert len(events) == 1
    assert events[0].cue_ids == ["c1", "c2"]
    assert events[0].fragments[0].refs == ["c1:t1:a1", "c2:t1:a1"]
    assert events[0].start_ms == cues[0].start_ms and events[0].end_ms >= cues[-1].end_ms
    assert events[0].ko == "이제 가야 해" and "-" not in events[0].ko
    assert check(cues[1], "", S, retained_fragments=[]) == []
    assert lineage_issues(lineage, events=events) == []


def test_cross_cue_ancestry_keeps_each_display_owners_start_without_overlap():
    cues = [cue("c1", 0, 2000, "Go now."), cue("c2", 2200, 4000, "Go.")]
    fragments = {
        "c1": [RetainedFragment(text="지금", refs=["c1:t1:a1"], speaker=None)],
        "c2": [RetainedFragment(text="가", refs=["c1:t1:a1", "c2:t1:a1"], speaker=None)],
    }
    events = lay.layout(cues, {"c1": "지금", "c2": "가"}, {"c1": "s1", "c2": "s1"}, {}, S,
                        lineage=_lineage(cues, fragments))
    assert [event.start_ms for event in events] == [0, 2200]
    assert [event.cue_ids for event in events] == [["c1"], ["c2"]]
    assert events[0].end_ms + S.min_gap_ms <= events[1].start_ms
    assert events[1].fragments[0].refs == ["c1:t1:a1", "c2:t1:a1"]


def test_cross_cue_word_order_swap_preserves_display_timing_and_separate_ancestry():
    cues = [cue("c1", 0, 2000, "We go."), cue("c2", 2200, 4000, "Now.")]
    fragments = {
        "c1": [RetainedFragment(text="이제", refs=["c2:t1:a1"], speaker=None)],
        "c2": [RetainedFragment(text="가", refs=["c1:t1:a1"], speaker=None)],
    }
    events = lay.layout(cues, {"c1": "이제", "c2": "가"}, {"c1": "s1", "c2": "s1"}, {}, S,
                        lineage=_lineage(cues, fragments))
    assert [event.start_ms for event in events] == [0, 2200]
    assert [event.cue_ids for event in events] == [["c1"], ["c2"]]
    assert events[0].end_ms + S.min_gap_ms <= events[1].start_ms
    assert [event.fragments[0].refs for event in events] == [["c2:t1:a1"], ["c1:t1:a1"]]


def test_blank_source_referenced_by_two_displays_is_not_absorbed_into_both_timelines():
    cues = [cue("c1", 0, 1000, "No."), cue("c2", 1500, 2500, "Go."), cue("c3", 3000, 4000, "Now.")]
    fragments = {
        "c1": [RetainedFragment(text="안 돼", refs=["c1:t1:a1", "c2:t1:a1"], speaker=None)],
        "c2": [],
        "c3": [RetainedFragment(text="지금 가", refs=["c2:t1:a1", "c3:t1:a1"], speaker=None)],
    }
    events = lay.layout(cues, {"c1": "안 돼", "c2": "", "c3": "지금 가"}, {c.id: "s1" for c in cues}, {}, S,
                        lineage=_lineage(cues, fragments))
    assert [event.cue_ids for event in events] == [["c1"], ["c3"]]
    assert [event.start_ms for event in events] == [0, 3000]
    assert events[0].end_ms + S.min_gap_ms <= events[1].start_ms
    assert all("c2:t1:a1" in event.fragments[0].refs for event in events)


def test_unique_contiguous_blank_sources_can_be_absorbed_by_one_display():
    cues = [cue("c1", 0, 1000, "We should"), cue("c2", 1100, 2100, "leave"), cue("c3", 2200, 3200, "now.")]
    refs = [f"{c.id}:t1:a1" for c in cues]
    fragment = RetainedFragment(text="이제 가야 해", refs=refs, speaker="a")
    lineage = _lineage(cues, {"c1": [fragment], "c2": [], "c3": []})
    events = lay.layout(cues, {"c1": fragment.text, "c2": "", "c3": ""}, {c.id: "s1" for c in cues}, {}, S,
                        lineage=lineage)
    assert len(events) == 1 and events[0].cue_ids == ["c1", "c2", "c3"]
    assert events[0].start_ms == 0 and events[0].end_ms >= cues[-1].end_ms
    assert events[0].fragments[0].refs == refs


def test_semantic_plot_lyric_layout_preserves_one_display_note_and_markerless_fragments():
    from substage.validate import check, lineage_issues

    c = cue("c1", 0, 2000, "A story lyric.", lyric=True)
    fragment = RetainedFragment(text="가사", refs=["c1:t1:a1"], speaker=None)
    lineage = _lineage([c], {"c1": [fragment]})
    lineage.source.acts[0].essential = True
    events = lay.layout([c], {"c1": "♪ 가사"}, {"c1": "s1"}, {}, S, lineage=lineage)
    assert len(events) == 1 and events[0].ko == "♪ 가사"
    assert events[0].ko.count("♪") == 1
    assert events[0].fragments[0].text == "가사"
    assert events[0].fragments[0].refs == ["c1:t1:a1"]
    assert events[0].fragments[0].attribution == "declared"
    assert check(c, events[0].ko, S, retained_fragments=events[0].fragments) == []
    assert lineage_issues(lineage, events=events) == []


def test_guarded_sound_and_omitted_incidental_lyric_do_not_create_semantic_display_events():
    from substage.models import OmittedAct
    from substage.validate import lineage_issues

    sound = cue("c1", 0, 1000, "", sound_cues=["[gunshot]"])
    song = cue("c2", 2000, 3000, "La la.", lyric=True)
    act = SourceAct(ref="c2:t1:a1", cue_id="c2", turn_ordinal=1, act_ordinal=1,
                    start_char=0, end_char=len(song.text), text=song.text,
                    meaning="incidental lyric", essential=False)
    lineage = SemanticLineage(
        source=SemanticSource(scene_id="s1", acts=[act]),
        fragments_by_cue={"c1": [], "c2": []},
        omissions=[OmittedAct(ref=act.ref, reason="incidental lyric remains untranslated")],
        structural_valid=True,
    )
    events = lay.layout([sound, song], {"c1": "", "c2": ""}, {"c1": "s1", "c2": "s1"}, {}, S, lineage=lineage)
    assert events == [] and lineage_issues(lineage, events=events) == []


def test_co_existence_ost_and_dialogue_preserves_overlap_and_an8_positioning(tmp_path):
    # Even a 129 ms tail overlap must stay visible on the top channel.
    c_ost = cue("ost-0008", 687000, 688400, "EARTH-10005", source="ost")
    c_dlg = cue("c0177", 688271, 689417, "your locker at work opens.")
    evs = run([c_ost, c_dlg], {"ost-0008": "'지구-10005'", "c0177": "직장 사물함이 열린다고"})
    assert len(evs) == 2
    assert evs[0].channel == "ost" and evs[0].position == "an8"
    assert evs[0].end_ms >= 688400
    assert evs[1].channel == "dialogue" and evs[1].position == "an2"
    assert evs[1].start_ms == 688271

    # Case 2: Truly simultaneous OST ('공존' [732690..735200]) overlapping dialogue ([730813..735568])
    # keeps both events, sets OST position='an8' (top) and dialogue position='an2' (bottom), and emits {\an8}.
    c_d1 = cue("c0188", 730813, 735568, "And look, you can always go back to superhero-ing.")
    c_o1 = cue("ost-0011", 732690, 735200, "COEXIST", source="ost")
    evs2 = run([c_d1, c_o1], {"c0188": "언제든 히어로로 돌아가도 되잖아", "ost-0011": "'공존'"})
    assert len(evs2) == 2
    dlg_ev = next(e for e in evs2 if e.channel == "dialogue")
    ost_ev = next(e for e in evs2 if e.channel == "ost")
    assert dlg_ev.position == "an2"
    assert ost_ev.position == "an8"

    ws = Workspace("demo", tmp_path, tmp_path / "cache", tmp_path / "logs")
    srt_p, _ = emit_mod.emit(
        ws,
        SimpleNamespace(video_id="demo"),
        [c_d1, c_o1],
        {"c0188": "언제든 히어로로 돌아가도 되잖아", "ost-0011": "'공존'"},
        {},
        evs2,
    )
    srt_txt = srt_p.read_text(encoding="utf-8")
    assert "{\\an8}'공존'" in srt_txt
    from substage.review_server import load_cue_list
    loaded = load_cue_list(srt_p)
    top_cues = [x for x in loaded if x.get("pos") == "top"]
    assert len(top_cues) == 1 and top_cues[0]["t"] == "'공존'"

    # Case 3: Logan chauffeur permit (ost-0051 [295101..296838]) overlapping radio dialogue
    # c0055 [292450..294290] and c0056 [294370..296750] is NEVER truncated to c0056.start - 83,
    # gets position='an8', and _normalize_to_srt preserves {\an8}.
    c0055 = cue("c0055", 292450, 294290, "Sleepwalking.\nBetween the ice caps,")
    ost51 = cue("ost-0051", 295101, 296838, "CHRYSLER EL PASO, TEXAS CHAUFFEUR PERMIT HOWLETT, JAMES", source="ost")
    c0056 = cue("c0056", 294370, 296750, "the pornographers,\npoisoned water, mutants...")
    evs3 = run(
        [c0055, ost51, c0056],
        {
            "c0055": "녹는 만년설\n포르노 업자",
            "ost-0051": "'운전면허증\n제임스 하울렛'",
            "c0056": "오염수, 뮤턴트 사이를\n몽유병자처럼 걷지",
        },
    )
    ost_ev3 = next(e for e in evs3 if e.channel == "ost")
    assert ost_ev3.start_ms == 295101 and ost_ev3.end_ms >= 296838
    assert ost_ev3.position == "an8"

    from substage.review_server import _normalize_to_srt
    norm_dest = tmp_path / "norm.srt"
    _normalize_to_srt(srt_p, norm_dest)
    assert "{\\an8}'공존'" in norm_dest.read_text(encoding="utf-8")
    loaded_norm = load_cue_list(norm_dest)
    assert [x["t"] for x in loaded_norm if x.get("pos") == "top"] == ["'공존'"]


def test_ost_chain_position_invariance_no_jump():
    cues = [
        cue("ost-1", 0, 1800, "FIRST", source="ost"),
        cue("ost-2", 2000, 4000, "SECOND", source="ost"),
        cue("c1", 2500, 4500, "Spoken dialogue."),
    ]
    events = run(cues, {"ost-1": "첫 화면", "ost-2": "다음 화면", "c1": "말하는 중"})
    ost = [e for e in events if e.channel == "ost"]
    assert len(ost) == 2
    assert [e.position for e in ost] == ["an8", "an8"]
    assert all(e.ko.startswith("'") and e.ko.endswith("'") for e in ost)
    assert events[-1].position == "an2"


def test_isolated_ost_chain_stays_bottom_and_does_not_join_after_400ms():
    cues = [
        cue("ost-1", 0, 1800, "FIRST", source="ost"),
        cue("ost-2", 2601, 4000, "SECOND", source="ost"),
        cue("c1", 2500, 4500, "Spoken dialogue."),
    ]
    ost = [e for e in run(cues, {"ost-1": "첫 화면", "ost-2": "다음 화면", "c1": "말하는 중"})
           if e.channel == "ost"]
    assert [e.position for e in ost] == ["an2", "an8"]


def test_touching_ost_and_dialogue_boundaries_do_not_move_ost_top():
    cues = [
        cue("ost-1", 0, 1800, "FIRST", source="ost"),
        cue("c1", 1800, 4000, "Spoken dialogue."),
    ]
    ost = next(e for e in run(cues, {"ost-1": "첫 화면", "c1": "말하는 중"}) if e.channel == "ost")
    assert ost.position == "an2"
    assert ost.end_ms == 1800


def test_overlapping_top_ost_single_lines_merge_and_keep_source_cues():
    cues = [
        cue("ost-1", 0, 2000, "FIRST", source="ost"),
        cue("ost-2", 100, 2500, "SECOND", source="ost"),
        cue("c1", 0, 4000, "Spoken dialogue."),
    ]
    events = run(cues, {"ost-1": "첫 화면", "ost-2": "다음 화면", "c1": "말하는 중"})
    ost = [e for e in events if e.channel == "ost"]
    assert len(ost) == 1
    assert ost[0].position == "an8"
    assert ost[0].ko == "'첫 화면'\n'다음 화면'"
    assert ost[0].cue_ids == ["ost-1", "ost-2"]
    assert ost[0].start_ms == 0 and ost[0].end_ms >= 2500


def test_overlapping_multiline_top_ost_sequences_without_collisions():
    cues = [
        cue("ost-1", 0, 2000, "FIRST", source="ost"),
        cue("ost-2", 100, 2500, "SECOND", source="ost"),
        cue("ost-3", 200, 3000, "THIRD", source="ost"),
        cue("c1", 0, 4000, "Spoken dialogue."),
    ]
    events = run(
        cues,
        {
            "ost-1": "'첫 줄\n둘째 줄'",
            "ost-2": "'다음 줄\n이어지는 줄'",
            "ost-3": "'마지막 줄\n끝 줄'",
            "c1": "말하는 중",
        },
    )
    ost = [e for e in events if e.channel == "ost"]
    assert len(ost) == 3
    assert all(e.position == "an8" for e in ost)
    assert all(a.end_ms <= b.start_ms for a, b in zip(ost, ost[1:]))
    assert all(e.end_ms - e.start_ms >= S.short_min_ms for e in ost)
    assert events == sorted(events, key=lambda e: e.start_ms)


def test_rapid_montage_top_ost_clamps_predecessor_instead_of_cascading_drift():
    # The first simultaneous pair merges into a two-line card whose extended
    # tail overlaps the next cut. Later cards must keep their original starts.
    cues = [
        cue("ost-1", 0, 900, "FIRST HEADLINE", source="ost"),
        cue("ost-2", 100, 900, "FIRST CALLOUT", source="ost"),
        cue("ost-3", 700, 1100, "SECOND HEADLINE", source="ost"),
        cue("ost-4", 1400, 1800, "THIRD HEADLINE", source="ost"),
        cue("c1", 0, 3000, "Spoken dialogue."),
    ]
    events = run(
        cues,
        {
            "ost-1": "'첫 기사'",
            "ost-2": "'둘째 기사'",
            "ost-3": "'세 번째 기사\n추가 내용'",
            "ost-4": "'네 번째 기사\n추가 내용'",
            "c1": "말하는 중",
        },
    )
    ost = [e for e in events if e.channel == "ost"]
    assert [e.cue_ids for e in ost] == [["ost-1", "ost-2"], ["ost-3"], ["ost-4"]]
    assert [e.start_ms for e in ost] == [0, 700, 1400]
    assert len(ost[0].ko.splitlines()) == 2
    assert ost[0].end_ms == ost[1].start_ms - S.min_gap_ms
    assert all(e.position == "an8" for e in ost)
    assert all(a.end_ms + S.min_gap_ms <= b.start_ms for a, b in zip(ost, ost[1:]))
    assert all(e.end_ms - e.start_ms >= S.short_min_ms for e in ost)


def test_one_millisecond_ost_dialogue_overlap_keeps_full_ost_on_top():
    cues = [
        cue("ost-1", 0, 1800, "FIRST", source="ost"),
        cue("c1", 1799, 4000, "Spoken dialogue."),
    ]
    ost = next(e for e in run(cues, {"ost-1": "첫 화면", "c1": "말하는 중"}) if e.channel == "ost")
    assert ost.position == "an8"
    assert ost.end_ms == 1800


# ---------------------------------------------------------------- duplicate displays


def test_dedupe_normalization_ignores_ass_tags_punctuation_and_case():
    assert lay._norm_ko_dedupe(r"{\an8}'JFK 택시 파업!'") == "jfk택시파업"
    assert lay._ko_tokens_dedupe(r"{\an8}'JFK 택시 파업!'") == {"jfk", "택시", "파업"}


@pytest.mark.parametrize("start", [1_040_000, 1_061_000])  # 17:20 and 17:41
def test_full_tweet_duplicate_yields_to_overlapping_ost_beats(start):
    cues = [
        cue("c1", start, start + 6000, "This is unrelated dialogue."),
        cue("ost-full", start, start + 4000, "Taxi drivers strike. Uber prices surge.", source="ost"),
        cue("ost-beat1", start, start + 1800, "Taxi drivers strike.", source="ost"),
        cue("ost-beat2", start + 2000, start + 4000, "Uber prices surge.", source="ost"),
    ]
    events = run(cues, {
        "c1": "지금 어디 가는 거야",
        "ost-full": "택시 기사 파업\n우버 요금 폭등",
        "ost-beat1": "택시 기사 파업",
        "ost-beat2": "우버 요금 폭등",
    })
    ost = [e for e in events if e.channel == "ost"]
    assert [e.cue_ids for e in ost] == [["ost-beat1"], ["ost-beat2"]]
    assert all(e.position == "an8" for e in ost)
    assert all(a.end_ms + S.min_gap_ms <= b.start_ms for a, b in zip(ost, ost[1:]))


@pytest.mark.parametrize("gap", [500, 5000, 12_000])
def test_nearby_ost_translations_share_one_display_without_bridging_long_gaps(gap):
    cues = [
        cue("ost-1", 0, 1500, "Taxi drivers on strike", source="ost"),
        cue("ost-2", 1500 + gap, 3000 + gap, "Taxi drivers strike today", source="ost"),
    ]
    events = run(cues, {"ost-1": "택시 기사 파업", "ost-2": "택시 기사 오늘 파업"})
    assert len(events) == 1
    assert events[0].cue_ids == ["ost-1", "ost-2"]
    assert "오늘" in events[0].ko
    if gap > 2000:
        assert events[0].end_ms < cues[1].start_ms
    else:
        assert events[0].end_ms >= cues[1].end_ms


def test_ost_repeated_after_dedupe_window_stays_separate():
    cues = [
        cue("ost-1", 0, 1500, "Taxi drivers on strike", source="ost"),
        cue("ost-2", 13_501, 15_000, "Taxi drivers on strike", source="ost"),
    ]
    assert len(run(cues, {"ost-1": "택시 기사 파업", "ost-2": "택시 기사 파업"})) == 2


def test_coextensive_identical_ost_is_not_displayed_over_single_dialogue():
    cues = [
        cue("ost-1", 0, 2000, "ERIC HIMPTON HOLDER JR.", source="ost"),
        cue("c1", 200, 2200, "Eric Himpton Holder, Jr."),
    ]
    events = run(cues, {"ost-1": "'에릭 힘턴 홀더 주니어'", "c1": "에릭 힘턴 홀더 주니어"})
    assert len(events) == 1
    assert events[0].channel == "dialogue"
    assert events[0].cue_ids == ["c1"]


@pytest.mark.parametrize("spoken", [
    "지금 택시 기사 파업 중이래",
    "택시 기사 파업 소식이야",
])
def test_non_identical_or_multi_cue_ost_and_dialogue_stay_split_top_and_bottom(spoken):
    cues = [
        cue("ost-1", 0, 2000, "TAXI DRIVERS STRIKE", source="ost"),
        cue("c1", 100, 2200, "Taxi drivers are on strike."),
    ]
    events = run(cues, {"ost-1": "택시 기사 파업", "c1": spoken})
    assert len(events) == 2
    assert [e.position for e in events] == ["an8", "an2"]

@pytest.mark.parametrize("later_ko", ["우리가 지금 가야 한다고", "우리가 지금 가야 한다고 했어"])
def test_adjacent_dialogue_echo_keeps_one_translation_and_both_cue_ids(later_ko):
    cues = [
        cue("c1", 0, 1800, "I told you we should leave."),
        cue("c2", 2300, 4100, "That is what I said."),
    ]
    events = run(cues, {"c1": "우리가 지금 가야 한다고", "c2": later_ko})
    assert len(events) == 1
    assert events[0].cue_ids == ["c1", "c2"]
    assert events[0].ko.replace("\n", " ") == later_ko
    assert events[0].start_ms == 0 and events[0].end_ms >= 4100


def test_intentional_english_repetition_is_not_a_translation_echo():
    cues = [
        cue("c1", 0, 1800, "We have to leave now."),
        cue("c2", 2300, 4100, "We have to leave now!"),
    ]
    events = run(cues, {"c1": "우리가 지금 떠나야 한다고", "c2": "우리가 지금 떠나야 한다고"})
    assert len(events) == 2


def test_dialogue_echo_guard_preserves_scene_changes_and_real_pauses():
    cues = [
        cue("c1", 0, 1800, "We have to leave."),
        cue("c2", 2300, 4100, "That is what I said."),
    ]
    ko = {"c1": "우리가 지금 떠나야 한다고", "c2": "우리가 지금 떠나야 한다고"}
    assert len(run(cues, ko, scene_of={"c1": "s1", "c2": "s2"})) == 2
    cues[1] = cues[1].model_copy(update={"start_ms": 4000, "end_ms": 5800})
    assert len(run(cues, ko)) == 2


@pytest.mark.parametrize("with_ost", [False, True])
def test_rapid_dialogue_events_never_overlap_on_bottom_channel(with_ost):
    cues = [
        cue("c1", 0, 800, "First."),
        cue("c2", 50, 900, "Second."),
        cue("c3", 100, 1000, "Third."),
    ]
    ko = {"c1": "처음이야", "c2": "아니잖아", "c3": "알았어"}
    if with_ost:
        cues.append(cue("ost-1", 0, 1200, "NOTICE", source="ost"))
        ko["ost-1"] = "긴급 공지"
    events = run(cues, ko, scene_of={c.id: c.id for c in cues})
    assert len([e for e in events if e.channel == "dialogue"]) == 3
    for pos in ("an2", "an8"):
        lane = sorted((e for e in events if e.position == pos), key=lambda e: e.start_ms)
        assert all(a.end_ms + S.min_gap_ms <= b.start_ms for a, b in zip(lane, lane[1:]))
    assert all(e.end_ms > e.start_ms for e in events)


@pytest.mark.parametrize("with_dialogue", [False, True])
def test_simultaneous_multiline_ost_events_never_overlap_in_either_position(with_dialogue):
    cues = [
        cue("ost-1", 0, 1800, "FIRST CARD", source="ost"),
        cue("ost-2", 50, 2000, "SECOND CARD", source="ost"),
        cue("ost-3", 100, 2200, "THIRD CARD", source="ost"),
    ]
    ko = {
        "ost-1": "택시 기사\n파업 시작",
        "ost-2": "우버 요금\n폭등 사태",
        "ost-3": "대중 교통\n긴급 대책",
    }
    if with_dialogue:
        cues.append(cue("c1", 0, 3000, "Unrelated dialogue."))
        ko["c1"] = "다른 얘기를 해 보자"
    events = run(cues, ko)
    ost = [e for e in events if e.channel == "ost"]
    assert len(ost) == 3
    assert all(e.position == ("an8" if with_dialogue else "an2") for e in ost)
    assert all(a.end_ms + S.min_gap_ms <= b.start_ms for a, b in zip(ost, ost[1:]))
    assert all(e.end_ms - e.start_ms >= S.short_min_ms for e in ost)


def test_source_ost_matching_keeps_numbers_and_prefers_complete_text():
    from substage.models import OnScreenText
    from substage.translate import _dedupe_ost, _same_text

    assert _same_text("JFK Taxi Strike", "JFK Taxi Strike / Uber Surge")
    assert not _same_text("Uber Surge 4x", "Uber Surge 5x")
    assert not _same_text("JFK Taxi Strike", "JFK Taxi Strike / Uber Surge 4x")
    items = [
        OnScreenText(start_ms=0, end_ms=1500, text="[headline] JFK Taxi Strike"),
        OnScreenText(start_ms=1000, end_ms=2500, text="[headline] JFK Taxi Strike / Uber Surge"),
    ]
    [(kind, merged)] = _dedupe_ost(items)
    assert kind == "headline" and merged.text == "JFK Taxi Strike / Uber Surge"
    assert (merged.start_ms, merged.end_ms) == (0, 2500)


def test_source_full_tweet_yields_to_frame_exact_overlapping_beats():
    from substage.translate import dedupe_ost_shadows

    cues = [
        cue("ost-full", 1_040_000, 1_044_000, "Taxi drivers strike. Uber prices surge.", source="ost"),
        cue("ost-beat1", 1_040_001, 1_041_801, "Taxi drivers strike.", source="ost"),
        cue("ost-beat2", 1_042_001, 1_044_001, "Uber prices surge.", source="ost"),
    ]
    assert [c.id for c in dedupe_ost_shadows(cues)] == ["ost-beat1", "ost-beat2"]


def test_source_nearby_ost_duplicates_merge_spans_and_preserve_frame_exact_identity():
    from substage.translate import dedupe_ost_shadows

    coarse = cue("ost-coarse", 0, 2000, "Taxi drivers strike", source="ost")
    exact = cue("ost-exact", 1001, 3001, "Taxi drivers strike today", source="ost")
    [merged] = dedupe_ost_shadows([coarse, exact])
    assert merged.id == "ost-exact" and merged.text == exact.text
    assert (merged.start_ms, merged.end_ms) == (0, 3001)
    assert (exact.start_ms, exact.end_ms) == (1001, 3001)


@pytest.mark.parametrize("kind", ["headline", "callout"])
@pytest.mark.parametrize("linked", [None, "c1"])
def test_source_ost_read_aloud_across_dialogue_cues_is_removed(kind, linked):
    from substage.models import OnScreenText
    from substage.translate import ost_cues

    spoken = [
        cue("c1", 0, 1500, "Taxi drivers strike."),
        cue("c2", 1500, 3000, "Uber prices surge."),
    ]
    observation = OnScreenText(
        start_ms=0, end_ms=3000, text=f"[{kind}] Taxi drivers strike. Uber prices surge.",
        covered_by_cue=linked,
    )
    assert ost_cues([observation], spoken, []) == []


def test_dialogue_echo_dedupe_preserves_both_semantic_source_references():
    from substage.validate import lineage_issues

    cues = [
        cue("c1", 0, 1800, "I said we should leave."),
        cue("c2", 2300, 4100, "That is what I told you."),
    ]
    text = "우리가 지금 떠나야 한다고"
    fragments = {
        c.id: [RetainedFragment(text=text, refs=[f"{c.id}:t1:a1"], speaker="a")]
        for c in cues
    }
    lineage = _lineage(cues, fragments)
    for act in lineage.source.acts:
        act.essential = True
    events = lay.layout(cues, {c.id: text for c in cues}, {c.id: "s1" for c in cues}, {}, S,
                        lineage=lineage)
    assert len(events) == 1
    assert events[0].fragments[0].refs == ["c1:t1:a1", "c2:t1:a1"]
    assert lineage_issues(lineage, events=events) == []


def _raw_ev(start, en_end, ko, channel="dialogue"):
    return lay._Ev(start=start, en_end=en_end, cue_ids=[f"c{start}"], turns=[], ko=ko, fixed=channel == "ost",
                   scene_first="s", scene_last="s", en_text="x", channel=channel)


def test_retime_non_overlapping_ost_drops_stale_top_tag():
    dlg = _raw_ev(0, 1500, "안녕")
    ost = _raw_ev(10_000, 12_000, r"{\an8}'소독용 알코올'", channel="ost")
    lay._retime([dlg, ost], S)
    assert ost.position == "an2"
    assert ost.ko == "'소독용 알코올'"


def test_retime_non_overlapping_ost_gains_quotes_without_top_tag():
    ost = _raw_ev(10_000, 12_000, r"{\an8}공존", channel="ost")
    lay._retime([ost], S)
    assert ost.position == "an2"
    assert ost.ko == "'공존'"


def test_event_text_an2_never_emits_top_tag():
    ev = OutEvent(start_ms=0, end_ms=1000, ko=r"{\an8}'공존'", position="an2", channel="ost")
    assert emit_mod._event_text(ev) == "'공존'"
    top = OutEvent(start_ms=0, end_ms=1000, ko=r"{\an2}{\an8}'공존'", position="an8", channel="ost")
    assert emit_mod._event_text(top) == r"{\an8}'공존'"


def _lane_overlaps(events):
    n = 0
    for pos in ("an2", "an8"):
        lane = sorted((e for e in events if e.position == pos), key=lambda e: (e.start_ms, e.end_ms))
        n += sum(1 for a, b in zip(lane, lane[1:]) if a.end_ms > b.start_ms)
    return n


def test_sanitize_display_events_resolves_same_channel_overlaps():
    events = [
        OutEvent(start_ms=0, end_ms=3000, ko="첫째"),
        OutEvent(start_ms=2000, end_ms=4000, ko="둘째"),  # staggered
        OutEvent(start_ms=5000, end_ms=7000, ko=r"{\an8}'스팸'", position="an8", channel="ost"),
        OutEvent(start_ms=5000, end_ms=6000, ko="'햄'", position="an8", channel="ost"),  # same start
        OutEvent(start_ms=8000, end_ms=9000, ko="'같은'", position="an8", channel="ost"),
        OutEvent(start_ms=8500, end_ms=9500, ko=r"{\an8}'같은'", position="an8", channel="ost"),  # duplicate
        OutEvent(start_ms=5500, end_ms=6500, ko="대사"),  # other lane: may overlap the top
    ]
    out = emit_mod.sanitize_display_events(events)
    assert _lane_overlaps(out) == 0
    assert all(e.end_ms > e.start_ms for e in out)
    assert [(e.start_ms, e.end_ms) for e in out] == sorted((e.start_ms, e.end_ms) for e in out)
    first = next(e for e in out if e.ko == "첫째")
    assert (first.start_ms, first.end_ms) == (0, 1999)
    same = [e for e in out if "같은" in e.ko]
    assert len(same) == 1 and (same[0].start_ms, same[0].end_ms) == (8000, 9500)
    assert {e.ko for e in out if e.position == "an8" and e.start_ms < 8000} == {r"{\an8}'스팸'", "'햄'"}
    assert any(e.ko == "대사" and e.start_ms == 5500 for e in out)


def test_sanitize_display_events_preserves_positive_duration_on_1ms_start_gap():
    events = [
        OutEvent(start_ms=1000, end_ms=2000, ko="앞대사", position="an2"),
        OutEvent(start_ms=1001, end_ms=3000, ko="뒤대사", position="an2"),
        OutEvent(start_ms=4000, end_ms=5000, ko=r"{\an8}'앞자막'", position="an8", channel="ost"),
        OutEvent(start_ms=4001, end_ms=6000, ko=r"{\an8}'뒤자막'", position="an8", channel="ost"),
    ]
    out = emit_mod.sanitize_display_events(events)
    assert _lane_overlaps(out) == 0
    assert all(e.end_ms - e.start_ms >= 199 for e in out)
