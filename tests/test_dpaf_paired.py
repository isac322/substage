"""Consumer-visible invariants for fixed-window paired evaluation."""

from __future__ import annotations

import json
import re

import pytest

from substage import refeval
from substage.config import Settings, Workspace
from substage.models import Cue
from substage.refeval import Line


class _PointJudge:
    """Scripted protocol fixtures, not evidence of a real judge's semantic quality."""

    def __init__(self, *, fail_calibration: bool = False):
        self.calls: list[dict] = []
        self.fail_calibration = fail_calibration

    def generate(self, *, stage, prompt, schema, model):
        ids = [int(value) for value in re.findall(r"^\[(\d+)\]", prompt, re.M)]
        references = re.findall(r"^CANDIDATE: (.+)$", prompt, re.M)
        context_line = re.search(r"^UNIT_CONTEXT: (.+)$", prompt, re.M)
        assert context_line is not None
        contexts = json.loads(context_line.group(1))
        texts = [event["text"] for context in contexts for event in context["events"] if "text" in event]
        self.calls.append({
            "stage": stage, "prompt": prompt, "texts": texts, "model": model,
            "references": references, "contexts": contexts,
        })
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
        incorrect = degraded or omitted or duplicated or timing_error
        score = 80 if self.fail_calibration else (20 if incorrect else 100)
        return schema(
            score=score,
            items=[
                refeval.ItemVerdict(
                    item=item, category="omission" if omitted else "verbosity" if duplicated else "meaning" if timing_error else "none",
                    severity=3 if omitted or duplicated or timing_error else 0, note="fake",
                )
                for item in ids
            ],
            source_meaning_score=80 if self.fail_calibration else (20 if degraded or omitted or timing_error else 100),
            source_items=[
                refeval.SourceItemVerdict(
                    item=item, severity=3 if degraded or omitted or timing_error else 0,
                    safe_harbor=not (degraded or omitted or timing_error), note="fake"
                )
                for item in ids
            ],
            presentation_items=[
                refeval.PresentationItemVerdict(
                    item=item, timing_severity=3 if timing_error else 0,
                    display_severity=0, duplicate_severity=3 if duplicated else 0, note="fake",
                )
                for item in ids
            ],
        )


def _inputs():
    en = [
        Cue(id="c0001", start_ms=1000, end_ms=3000, text="Leave now.", raw="Leave now."),
        Cue(id="c0002", start_ms=5000, end_ms=7000, text="All right.", raw="All right."),
    ]
    official = [Line(0, 1000, 3000, "지금 가자"), Line(1, 5000, 7000, "알았어")]
    opus = [Line(0, 1000, 3000, "기준 번역"), Line(1, 5000, 7000, "알았어")]
    return en, official, opus


def _run(tmp_path, judge, candidates, *, n_windows=1):
    en, official, opus = _inputs()
    ws = Workspace(video_id="paired-test", data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", log_dir=tmp_path / "log")
    return refeval.evaluate_paired(
        ws,
        Settings(),
        en,
        official,
        candidates,
        opus,
        judge=True,
        judge_models=["fake-judge"],
        judge_factory=lambda *_: judge,
        n_windows=n_windows,
        seed=7,
    )


def test_paired_windows_are_pinned_before_candidate_timing(tmp_path):
    judge = _PointJudge()
    first = {
        "arm-a": [Line(0, 1000, 3000, "후보 A"), Line(1, 5000, 7000, "알았어")],
        "arm-b": [Line(0, 2500, 4500, "후보 B"), Line(1, 5000, 7000, "알았어")],
    }
    report = _run(tmp_path, judge, first, n_windows=2)
    windows = [report["paired"][name]["windows"] for name in first]
    def pinned(rows):
        return [(row["id"], row["group_ids"], row["scene_ids"], row["start_ms"], row["end_ms"]) for row in rows]

    assert pinned(windows[0]) == pinned(windows[1])
    assert report["pinned"]["window_ids"] == [row["id"] for row in windows[0]]
    assert report["pinned"]["group_ids"] == [group["id"] for group in report["paired"]["arm-a"]["groups"]]
    assert report["paired"]["arm-a"]["alignment"]["scope"] == "fixed_source+official_with_attached_tracks"


def test_shifted_candidate_does_not_invalidate_cached_opus_judgment(tmp_path):
    judge = _PointJudge()
    original = {"arm": [Line(0, 1000, 3000, "후보"), Line(1, 5000, 7000, "알았어")]}
    shifted = {"arm": [Line(0, 3200, 4800, "후보"), Line(1, 5000, 7000, "알았어")]}
    first = _run(tmp_path, judge, original)
    calls_after_first = len(judge.calls)
    second = _run(tmp_path, judge, shifted)

    assert [
        (row["id"], row["group_ids"], row["scene_ids"], row["start_ms"], row["end_ms"])
        for row in second["paired"]["arm"]["windows"]
    ] == [
        (row["id"], row["group_ids"], row["scene_ids"], row["start_ms"], row["end_ms"])
        for row in first["paired"]["arm"]["windows"]
    ]
    assert len(judge.calls) == calls_after_first + 1
    assert second["paired"]["arm"]["candidate_hash"] != first["paired"]["arm"]["candidate_hash"]
    assert second["paired"]["arm"]["opus_hash"] == first["paired"]["arm"]["opus_hash"]


def test_adding_candidate_keeps_existing_pair_and_opus_cache(tmp_path):
    judge = _PointJudge()
    arm_a = [Line(0, 1000, 3000, "후보 A"), Line(1, 5000, 7000, "알았어")]
    first = _run(tmp_path, judge, {"arm-a": arm_a})
    calls_after_first = len(judge.calls)
    second = _run(
        tmp_path,
        judge,
        {
            "arm-a": arm_a,
            "arm-b": [Line(0, 1000, 3000, "후보 B"), Line(1, 5000, 7000, "알았어")],
        },
    )

    assert [
        (row["id"], row["group_ids"], row["scene_ids"], row["start_ms"], row["end_ms"])
        for row in second["paired"]["arm-a"]["windows"]
    ] == [
        (row["id"], row["group_ids"], row["scene_ids"], row["start_ms"], row["end_ms"])
        for row in first["paired"]["arm-a"]["windows"]
    ]
    assert len(judge.calls) == calls_after_first + 1


def test_failed_pointwise_calibration_invalidates_published_aggregate(tmp_path):
    judge = _PointJudge(fail_calibration=True)
    report = _run(
        tmp_path,
        judge,
        {"arm": [Line(0, 1000, 3000, "후보"), Line(1, 5000, 7000, "알았어")]},
    )

    assert report["calibration"]["fake-judge"]["passed"] is False
    assert report["judge_status"] == "invalid_calibration"
    assert report["paired"]["arm"]["judge_status"] == "invalid_calibration"
    assert report["paired"]["arm"]["judge_delta"] is None
    assert report["metrics"]["arm"]["judge"] is None
    assert report["metrics"]["arm"]["presentation"] is None
    assert report["paired"]["arm"]["opus_metrics"]["presentation"] is None
    assert report["paired"]["arm"]["judges"]["fake-judge"]["candidate"]["timing_error_rate"] == 0
    assert "INVALID" in "\n".join(refeval.render_paired_md(report))


def test_entirely_shifted_candidate_is_unmatched_without_moving_pinned_items(tmp_path):
    judge = _PointJudge()
    _, official, _ = _inputs()
    aligned = _run(tmp_path, judge, {"arm": official})
    shifted_lines = [
        Line(line.idx, line.start_ms + 600_000, line.end_ms + 600_000, line.text)
        for line in official
    ]
    shifted = _run(tmp_path, judge, {"arm": list(reversed(shifted_lines))})
    metrics = shifted["metrics"]["arm"]
    pair = shifted["paired"]["arm"]

    assert aligned["metrics"]["arm"]["chrf"] == 100
    assert metrics["chrf"] == 0
    assert metrics["coverage"] == 0
    assert metrics["lines"] == len(shifted_lines)
    assert metrics["length_ratio"] == metrics["event_ratio"] == 1
    assert metrics["median_cps"] == aligned["metrics"]["arm"]["median_cps"]
    assert shifted["pinned"] == aligned["pinned"]
    assert [
        (row["id"], row["group_ids"], row["scene_ids"], row["start_ms"], row["end_ms"])
        for row in pair["windows"]
    ] == [
        (row["id"], row["group_ids"], row["scene_ids"], row["start_ms"], row["end_ms"])
        for row in aligned["paired"]["arm"]["windows"]
    ]
    assert [group["id"] for group in pair["groups"]] == shifted["pinned"]["group_ids"]
    assert all(group["candidates"]["candidate"] == [] for group in pair["groups"])
    extras = pair["unmatched"]["candidate"]
    assert [group["id"] for group in extras] == ["candidate-x0001", "candidate-x0002"]
    assert [
        (group["start_ms"], group["end_ms"], group["candidates"]["candidate"])
        for group in extras
    ] == [(line.start_ms, line.end_ms, [line.text]) for line in shifted_lines]
    assert all(not group["en"] and not group["official"] for group in extras)
    assert pair["unmatched"]["opus"] == []
    assert judge.calls[-1]["references"] == ["(none)", "(none)"]
    assert judge.calls[-1]["texts"] == []
    assert judge.calls[-1]["contexts"][0]["unmatched_outside_window"] == {
        "events": len(shifted_lines), "start_ms": shifted_lines[0].start_ms, "end_ms": shifted_lines[-1].end_ms,
    }


def test_unmatched_extra_penalizes_precision_and_counts_display_metrics(tmp_path):
    judge = _PointJudge()
    _, official, _ = _inputs()
    aligned = _run(tmp_path, judge, {"arm": official})
    extra = Line(2, 600_000, 600_100, "가" * 20)
    report = _run(tmp_path, judge, {"arm": [extra, *reversed(official)]})
    metrics = report["metrics"]["arm"]
    pair = report["paired"]["arm"]

    assert 0 < metrics["chrf"] < aligned["metrics"]["arm"]["chrf"]
    assert metrics["coverage"] == 1
    assert metrics["lines"] == 3
    assert metrics["length_ratio"] == round(27 / 7, 4)
    assert metrics["event_ratio"] == 1.5
    assert metrics["cps_over_rate"] == metrics["cpl_over_rate"] == round(1 / 3, 4)
    assert metrics["median_cps"] == metrics["median_duration_s"] == 2
    assert pair["deterministic_delta"]["chrf"] < aligned["paired"]["arm"]["deterministic_delta"]["chrf"]
    assert pair["unmatched"]["candidate"][0]["candidates"]["candidate"] == [extra.text]
    assert report["pinned"] == aligned["pinned"]
    assert judge.calls[-1]["texts"] == [line.text for line in official]
    assert judge.calls[-1]["contexts"][0]["unmatched_outside_window"]["events"] == 1
    assert extra.text not in judge.calls[-1]["prompt"]


@pytest.mark.parametrize(
    ("start_ms", "end_ms", "matched"),
    [(2950, 4950, False), (2800, 4800, True), (2950, 3050, True)],
)
def test_paired_attachment_requires_existing_overlap_thresholds(tmp_path, start_ms, end_ms, matched):
    report = _run(
        tmp_path,
        _PointJudge(),
        {"arm": [Line(0, start_ms, end_ms, "지금 가자")]},
        n_windows=0,
    )

    assert report["metrics"]["arm"]["coverage"] == (0.5 if matched else 0)
    assert report["metrics"]["arm"]["lines"] == 1
    assert bool(report["paired"]["arm"]["unmatched"]["candidate"]) is not matched


def test_split_and_merged_candidate_events_keep_fixed_items_and_full_metrics(tmp_path):
    _, official, _ = _inputs()
    report = _run(
        tmp_path,
        _PointJudge(),
        {
            "split": [
                Line(0, 1000, 2000, "지금"), Line(1, 2000, 3000, "가자"),
                official[1],
            ],
            "merged": [Line(0, 1000, 7000, "지금 가자 알았어")],
        },
        n_windows=0,
    )

    split = report["metrics"]["split"]
    merged = report["metrics"]["merged"]
    assert split["chrf"] == 100
    assert split["coverage"] == 1
    assert split["length_ratio"] == 1
    assert split["lines"] == 3
    assert split["event_ratio"] == 1.5
    assert merged["chrf"] == 100
    assert merged["coverage"] == 1
    assert merged["chrf_fixed_group"] < 100
    assert merged["length_ratio"] == 1
    assert merged["lines"] == 1
    assert merged["event_ratio"] == 0.5
    assert [
        group["candidates"]["candidate"]
        for group in report["paired"]["merged"]["groups"]
    ] == [["지금 가자 알았어"], []]
    for pair in report["paired"].values():
        assert pair["pinned"] == report["pinned"]
        assert [group["id"] for group in pair["groups"]] == report["pinned"]["group_ids"]
        assert pair["unmatched"]["candidate"] == []


def test_opus_unmatched_extras_are_track_local_and_candidate_independent(tmp_path):
    en, official, _ = _inputs()
    extra = Line(2, 600_000, 600_100, "가" * 20)
    ws = Workspace(video_id="paired-test", data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", log_dir=tmp_path / "log")
    report = refeval.evaluate_paired(
        ws, Settings(), en, official,
        {"aligned": official, "empty": []}, [extra, *official], n_windows=0,
    )

    assert report["metrics"]["aligned"]["chrf"] == 100
    assert report["metrics"]["aligned"]["lines"] == 2
    assert report["metrics"]["empty"]["chrf"] == 0
    assert report["metrics"]["empty"]["lines"] == 0
    aligned, empty = (report["paired"][name] for name in ("aligned", "empty"))
    assert aligned["opus_metrics"] == empty["opus_metrics"]
    assert aligned["opus_metrics"]["chrf"] < 100
    assert aligned["opus_metrics"]["coverage"] == 1
    assert aligned["opus_metrics"]["lines"] == 3
    assert aligned["opus_metrics"]["event_ratio"] == 1.5
    assert aligned["opus_metrics"]["cps_over_rate"] == round(1 / 3, 4)
    assert aligned["unmatched"]["opus"] == empty["unmatched"]["opus"]
    assert aligned["unmatched"]["opus"][0]["id"] == "opus-x0001"
    assert aligned["unmatched"]["opus"][0]["start_ms"] == extra.start_ms
    assert aligned["unmatched"]["opus"][0]["end_ms"] == extra.end_ms
    assert aligned["unmatched"]["opus"][0]["candidates"]["opus"] == [extra.text]
    assert aligned["unmatched"]["candidate"] == empty["unmatched"]["candidate"] == []


def _evaluate_track(tmp_path, official, candidates, *, judge=None, n_windows=None):
    en = [
        Cue(id=f"c{index:04d}", start_ms=line.start_ms, end_ms=line.end_ms,
            text=f"Source {index}", raw=f"Source {index}")
        for index, line in enumerate(official)
    ]
    ws = Workspace(video_id="paired-test", data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", log_dir=tmp_path / "log")
    return refeval.evaluate_paired(
        ws, Settings(), en, official, candidates, official, n_windows=n_windows,
        judge=judge is not None, judge_models=["fake-judge"], judge_factory=lambda *_: judge,
    )


def test_merged_content_covers_all_original_gold_without_copying_the_event(tmp_path):
    gold = [Line(0, 1000, 3000, "네가"), Line(1, 3000, 5000, "떠났으면 좋겠어")]
    report = _evaluate_track(tmp_path, gold, {"merged": [Line(0, 1000, 5000, "네가 떠났으면 좋겠어")]})
    metrics = report["metrics"]["merged"]
    pair = report["paired"]["merged"]

    assert metrics["chrf"] == 100
    assert metrics["coverage"] == metrics["length_ratio"] == 1
    assert metrics["chrf_fixed_group"] < 100
    assert metrics["lines"] == 1
    assert metrics["event_ratio"] == 0.5
    assert pair["lexical_units"]["candidate"] == [["g0001", "g0002"]]
    assert [group["candidates"]["candidate"] for group in pair["groups"]] == [
        ["네가 떠났으면 좋겠어"], [],
    ]


def test_boundary_resplit_is_lexically_complete_and_idx_collisions_do_not_drop_events(tmp_path):
    gold = [
        Line(0, 1000, 3000, "네가"), Line(1, 3000, 5000, "떠났으면"),
        Line(2, 5000, 7000, "좋겠어"),
    ]
    candidates = [Line(0, 1000, 4000, "네가 떠났으면"), Line(0, 4000, 7000, "좋겠어")]
    report = _evaluate_track(tmp_path, gold, {"resplit": candidates})
    metrics = report["metrics"]["resplit"]

    assert metrics["chrf"] == 100
    assert metrics["coverage"] == metrics["length_ratio"] == 1
    assert metrics["lines"] == 2
    assert metrics["event_ratio"] == round(2 / 3, 4)
    assert metrics["chrf_fixed_group"] < 100
    assert report["paired"]["resplit"]["lexical_units"]["candidate"] == [["g0001", "g0002", "g0003"]]


def test_true_duplicate_events_are_never_deduplicated_by_idx_or_text(tmp_path):
    gold = [Line(0, 1000, 3000, "지금 떠나야 해")]
    report = _evaluate_track(tmp_path, gold, {"single": gold, "duplicate": [gold[0], gold[0]]})
    single, duplicate = (report["metrics"][name] for name in ("single", "duplicate"))

    assert single["chrf"] == 100
    assert duplicate["chrf"] < single["chrf"]
    assert duplicate["coverage"] == 1
    assert duplicate["lines"] == duplicate["length_ratio"] == duplicate["event_ratio"] == 2
    assert report["paired"]["duplicate"]["groups"][0]["candidates"]["candidate"] == [
        gold[0].text, gold[0].text,
    ]




def test_point_judge_sees_merged_text_once_for_every_covered_fixed_item(tmp_path):
    gold = [Line(0, 1000, 3000, "네가"), Line(1, 3000, 5000, "떠났으면 좋겠어")]
    judge = _PointJudge()
    report = _evaluate_track(
        tmp_path, gold, {"merged": [Line(0, 1000, 5000, "네가 떠났으면 좋겠어")]}, judge=judge,
    )
    call = next(
        call for call in judge.calls
        if call["texts"] == ["네가 떠났으면 좋겠어"]
        and any(source["text"] == "Source 0"
                for context in call["contexts"] for group in context["source_gold"] for source in group["en"])
    )

    assert call["references"] == ["e0001", "e0001"]
    assert call["contexts"][0]["group_ids"] == ["g0001", "g0002"]
    assert len(call["contexts"][0]["events"]) == 1
    event = call["contexts"][0]["events"][0]
    assert (event["event"], event["start_ms"], event["end_ms"], event["text"]) == (
        "e0001", 1000, 5000, "네가 떠났으면 좋겠어",
    )
    assert event["span_group_ids"] == ["g0001", "g0002"]
    assert event["text_omitted"] is False
    assert [
        gold_line["text"] for group in call["contexts"][0]["source_gold"] for gold_line in group["gold"]
    ] == [line.text for line in gold]
    assert "Return verdicts only for the numbered fixed items" in call["prompt"]
    verdict = report["paired"]["merged"]["windows"][0]["judges"]["fake-judge"]["candidate"]
    assert [line["group_id"] for line in verdict["lines"]] == report["pinned"]["group_ids"]


def test_point_judge_bounds_wide_shared_events_without_changing_opus_prompt(tmp_path):
    gold = [Line(i, 1000 + 2000 * i, 3000 + 2000 * i, f"대사 {i}") for i in range(16)]
    merged = [Line(0, gold[0].start_ms, gold[-1].end_ms, " ".join(line.text for line in gold))]
    judge = _PointJudge()
    first = _evaluate_track(tmp_path, gold, {"aligned": gold}, judge=judge)
    before = len(judge.calls)
    second = _evaluate_track(tmp_path, gold, {"aligned": gold, "merged": merged}, judge=judge)

    assert len(judge.calls) == before + len(second["paired"]["merged"]["windows"])
    assert first["paired"]["aligned"] == second["paired"]["aligned"]
    assert second["pinned"] == first["pinned"]
    for call in judge.calls[before:]:
        assert call["texts"] == []
        assert merged[0].text not in call["prompt"]
        assert len(call["references"]) == 8
        assert set(call["references"]) == {"e0001"}
        context = call["contexts"][0]
        assert len(context["source_gold"]) == len(call["references"]) + 1
        event = context["events"][0]
        assert event["span_group_count"] == len(gold)
        assert event["span_group_ids"] == context["group_ids"]
        assert event["text_omitted"] is True
        assert event["text_chars"] == refeval.cps_chars(merged[0].text)
        assert context["scope"]["omitted_scored_event_bodies"] == 1
        assert context["scope"]["scored_event_text_complete"] is False
        assert context["scope"]["remote_unit_group_count"] == len(gold) - len(context["group_ids"])
    assert second["metrics"]["merged"]["source_safety"]["context_partial_windows"] == len(judge.calls) - before


def test_boundary_merge_body_is_visible_in_both_windows_with_only_direct_neighbors(tmp_path):
    gold = [Line(i, 1000 + 2000 * i, 3000 + 2000 * i, f"대사 {i}") for i in range(16)]
    merged_text = gold[7].text + " " + gold[8].text
    merged = [*gold[:7], Line(7, gold[7].start_ms, gold[8].end_ms, merged_text), *gold[9:]]
    judge = _PointJudge()
    _evaluate_track(tmp_path, gold, {"aligned": gold}, judge=judge)
    before = len(judge.calls)
    report = _evaluate_track(tmp_path, gold, {"merged": merged}, judge=judge)
    calls = judge.calls[before:]

    assert len(calls) == len(report["paired"]["merged"]["windows"])
    for call in calls:
        assert call["texts"].count(merged_text) == 1
        context = call["contexts"][0]
        event = next(event for event in context["events"] if event.get("text") == merged_text)
        assert event["span_group_ids"] == ["g0008", "g0009"]
        assert event["text_omitted"] is False
        assert len(context["source_gold"]) == len(call["references"]) + 1
        assert len(context["scope"]["readonly_group_ids"]) == 1
        assert context["scope"]["scored_event_text_complete"] is True
    assert report["metrics"]["merged"]["chrf"] == 100


def test_chained_lexical_unit_does_not_expand_judge_context_to_entire_track(tmp_path):
    gold = [Line(i, 1000 + 2000 * i, 3000 + 2000 * i, f"대사 {i}") for i in range(24)]
    chained = [
        Line(i, line.start_ms, line.end_ms + (200 if i + 1 < len(gold) else 0), f"CHAIN_BODY_{i}")
        for i, line in enumerate(gold)
    ]
    judge = _PointJudge()
    report = _evaluate_track(tmp_path, gold, {"chain": chained}, judge=judge)
    calls = [
        call for call in judge.calls if any(text.startswith("CHAIN_BODY_") for text in call["texts"])
    ]

    assert report["paired"]["chain"]["lexical_units"]["candidate"] == [report["pinned"]["group_ids"]]
    assert len(calls) == len(report["paired"]["chain"]["windows"])
    for call in calls:
        context = call["contexts"][0]
        assert len(context["source_gold"]) == len(call["references"]) + len(context["scope"]["readonly_group_ids"])
        assert len(context["scope"]["readonly_group_ids"]) <= 2
        assert len(context["source_gold"]) < len(gold)
        assert context["scope"]["remote_unit_group_count"] == len(gold) - len(context["group_ids"])
        assert context["scope"]["omitted_remote_unit_event_bodies"] > 0
        assert context["scope"]["scored_event_text_complete"] is True
    assert "CHAIN_BODY_23" not in calls[0]["prompt"]
    assert "Source 23" not in calls[0]["prompt"]
    assert chained[0].text not in calls[-1]["texts"]


def test_only_temporally_visible_unmatched_body_is_sent_to_a_window(tmp_path):
    gold = [Line(i, 1000 + 4000 * i, 3000 + 4000 * i, f"대사 {i}") for i in range(16)]
    local_extra = Line(16, 3500, 4000, "LOCAL_UNMATCHED_BODY")
    far_extra = Line(17, 600_000, 601_000, "REMOTE_UNMATCHED_BODY")
    judge = _PointJudge()
    _evaluate_track(tmp_path, gold, {"aligned": gold}, judge=judge)
    before = len(judge.calls)
    report = _evaluate_track(tmp_path, gold, {"extras": [*gold, local_extra, far_extra]}, judge=judge)
    calls = judge.calls[before:]

    assert len(calls) == len(report["paired"]["extras"]["windows"])
    assert sum(local_extra.text in call["texts"] for call in calls) == 1
    assert all(far_extra.text not in call["prompt"] for call in calls)
    assert calls[0]["contexts"][0]["unmatched_outside_window"] == {
        "events": 1, "start_ms": far_extra.start_ms, "end_ms": far_extra.end_ms,
    }
    assert calls[-1]["contexts"][0]["unmatched_outside_window"] == {
        "events": 2, "start_ms": local_extra.start_ms, "end_ms": far_extra.end_ms,
    }
    assert report["metrics"]["extras"]["lines"] == len(gold) + 2
    assert len(report["paired"]["extras"]["unmatched"]["candidate"]) == 2


def test_point_context_keeps_identical_event_ordinals_separate_despite_idx_collision(tmp_path):
    gold = [Line(0, 1000, 3000, "가자")]
    judge = _PointJudge()
    _evaluate_track(tmp_path, gold, {"duplicate": [gold[0], gold[0]]}, judge=judge)
    call = next(call for call in judge.calls if call["texts"] == ["가자", "가자"])

    assert call["references"] == ["e0001, e0002"]
    assert [event["event"] for event in call["contexts"][0]["events"]] == ["e0001", "e0002"]


def test_merge_across_fixed_windows_uses_full_lexical_unit_without_clipping(tmp_path):
    gold = [Line(i, 1000 + 2000 * i, 3000 + 2000 * i, f"대사 {i}") for i in range(16)]
    merged = [
        *gold[:7], Line(7, gold[7].start_ms, gold[8].end_ms, gold[7].text + " " + gold[8].text),
        *gold[9:],
    ]
    report = _evaluate_track(tmp_path, gold, {"aligned": gold, "merged": merged})
    pair = report["paired"]["merged"]
    metrics = report["metrics"]["merged"]

    assert len(pair["windows"]) == 2
    assert "g0008" in pair["windows"][0]["group_ids"]
    assert "g0009" in pair["windows"][1]["group_ids"]
    assert metrics["chrf"] == 100
    assert metrics["coverage"] == metrics["length_ratio"] == 1
    assert metrics["lines"] == 15
    assert pair["lexical_units"]["candidate"] == [["g0008", "g0009"]]
    assert pair["pinned"] == report["paired"]["aligned"]["pinned"]


def test_rotated_content_retains_bad_fixed_group_diagnostic(tmp_path):
    gold = [
        Line(0, 1000, 3000, "오늘은 떠나자"), Line(1, 3000, 5000, "내일 돌아올게"),
        Line(2, 5000, 7000, "잘 지내고 있어"),
    ]
    rotated = [
        Line(0, 1000, 3200, gold[2].text), Line(1, 3000, 5200, gold[0].text),
        Line(2, 5000, 7000, gold[1].text),
    ]
    report = _evaluate_track(tmp_path, gold, {"rotated": rotated})
    metrics = report["metrics"]["rotated"]

    expected = refeval.corpus_chrf([(line.text, reference.text) for line, reference in zip(rotated, gold)])
    assert metrics["chrf_fixed_group"] == round(expected, 2)
    assert metrics["chrf_fixed_group"] < 100
    assert metrics["coverage"] == metrics["length_ratio"] == 1
    assert report["paired"]["rotated"]["lexical_units"]["candidate"] == [["g0001", "g0002", "g0003"]]


def test_giant_overlay_is_not_clean_display_quality_despite_complete_lexical_content(tmp_path):
    gold = [Line(i, 1000 + 2000 * i, 3000 + 2000 * i, f"아주 중요한 대사 {i}") for i in range(16)]
    overlay = Line(0, gold[0].start_ms, gold[-1].end_ms, " ".join(line.text for line in gold))
    report = _evaluate_track(tmp_path, gold, {"overlay": [overlay], "doubled": [*gold, overlay]})
    metrics = report["metrics"]["overlay"]
    doubled = report["metrics"]["doubled"]

    assert metrics["chrf"] == 100
    assert metrics["coverage"] == metrics["length_ratio"] == 1
    assert metrics["chrf_fixed_group"] < 100
    assert metrics["cpl_over_rate"] == 1
    assert metrics["median_duration_s"] == 32
    assert metrics["event_ratio"] == round(1 / 16, 4)
    assert metrics["lines"] == 1
    assert doubled["length_ratio"] == 2
    assert doubled["lines"] == 17
    assert doubled["chrf"] < 100
    assert report["paired"]["overlay"]["pinned"] == report["paired"]["doubled"]["pinned"]


@pytest.mark.parametrize("invalid_presentation", ["missing", "duplicate"])
def test_incomplete_presentation_verdicts_cannot_be_published_or_cached(tmp_path, invalid_presentation):
    class IncompleteJudge(_PointJudge):
        def generate(self, **kwargs):
            verdict = super().generate(**kwargs)
            first = verdict.presentation_items[0]
            verdict.presentation_items = [first] if invalid_presentation == "missing" else [first, first]
            return verdict

    en, gold, _ = _inputs()
    groups = refeval.align(en, gold, {})
    window = refeval.Window(id="w001", groups=groups)
    ws = Workspace(video_id="paired-test", data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", log_dir=tmp_path / "log")
    judge = IncompleteJudge()
    _, units = refeval._attach_track(groups, refeval.PAIRED_CANDIDATE, gold)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="incomplete or duplicate item verdicts"):
            refeval.judge_point(ws, judge, "fake-judge", window, units, candidate_hash="candidate")


def test_presentation_annotations_round_trip_separately_from_source_meaning(tmp_path):
    class AnnotatedJudge:
        def generate(self, *, schema, **_kwargs):
            return schema(
                score=40,
                items=[refeval.ItemVerdict(item=1, category="verbosity", severity=3, note="display annotation")],
                source_meaning_score=100,
                source_items=[refeval.SourceItemVerdict(item=1, severity=0, safe_harbor=True, note="meaning intact")],
                presentation_items=[refeval.PresentationItemVerdict(
                    item=1, timing_severity=1, display_severity=3, duplicate_severity=2, note="raw evidence",
                )],
            )

    _, gold, _ = _inputs()
    group = refeval.Group(id="g0001", start_ms=1000, end_ms=3000, official=[gold[0]])
    window = refeval.Window(id="w001", groups=[group])
    _, units = refeval._attach_track([group], refeval.PAIRED_CANDIDATE, [gold[0]])
    ws = Workspace(video_id="paired-test", data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", log_dir=tmp_path / "log")
    verdict = refeval.judge_point(
        ws, AnnotatedJudge(), "fake-judge", window, units, candidate_hash="candidate",
    )
    summary = refeval.point_summary([verdict])

    assert verdict["presentation_lines"] == [{
        "group_id": "g0001", "timing_severity": 1, "display_severity": 3,
        "duplicate_severity": 2, "note": "raw evidence",
    }]
    assert summary["source_meaning_score"] == 100
    assert summary["source_error_rate"] == 0
    assert summary["score"] == 40
    assert summary["error_rate"] == 1
    assert summary["timing_error_rate"] == summary["display_error_rate"] == summary["duplicate_error_rate"] == 1
    assert summary["presentation_error_rate"] == 1
    assert summary["presentation_severity_sum"] == 6


def test_empty_opus_is_rejected_for_nonempty_evaluation(tmp_path):
    en, official, _ = _inputs()
    ws = Workspace(video_id="paired-test", data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", log_dir=tmp_path / "log")
    with pytest.raises(ValueError, match="frozen Opus must not be empty"):
        refeval.evaluate_paired(
            ws,
            Settings(),
            en,
            official,
            {"arm": official},
            [],
            n_windows=1,
        )
