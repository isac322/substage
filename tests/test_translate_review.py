"""Offline unit tests for translate.py and review.py.

A fake GeminiClient records every generate() call and returns scripted
responses keyed on the prompt's scene marker, so we test real behavior —
scheduling, prompt content, checkpointing, ost cue coverage — not wiring.
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import pytest

from substage.config import Settings, Workspace, narrative_text
from substage.gemini import GeminiBlocked
from substage.models import (
    AmbiguityFlag,
    Callback,
    Character,
    CharacterState,
    ChunkObservations,
    Cue,
    CueDelta,
    CueObservation,
    VideoInfo,
    ContextMap,
    GlossaryEntry,
    MediaNeed,
    OnScreenText,
    ReadFlag,
    RegisterShift,
    SceneRegister,
    Scene,
    SceneTranslation,
    SemanticDraft,
    SemanticDraftCue,
    SemanticLineage,
    SemanticSource,
    SourceAct,
    SourceRelation,
    RetainedFragment,
    OmittedAct,
    Snapshot,
    TranslatedCue,
    VoiceCard,
    CueEdit,
)
from substage import narrative, perception, review, style, translate
from substage.capability import resolve_route
from substage.store import Checkpoints


# ------------------------------------------------------------------ fixtures


def ws(tmp_path) -> Workspace:
    return Workspace(
        video_id="test-video",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "log",
    )


class FakeGemini:
    """Duck-typed GeminiClient: records calls, answers via a responder fn."""

    def __init__(self, responder=None, narrative="lite"):
        self.calls: list[dict] = []
        self.responder = responder or self._default
        self.settings = SimpleNamespace(model_main="fake-main-model",
                                        model_light="fake-light-model",
                                        narrative_context=narrative)

    def generate(self, *, stage, prompt, schema, clips=(), images=(),
                 model=None, thinking_level=None, system=None):
        call = dict(stage=stage, prompt=prompt, schema=schema,
                    clips=list(clips), model=model, system=system)
        self.calls.append(call)
        return self.responder(call)

    _CUE_LINE = re.compile(r"^  ([A-Za-z0-9_-]+) \| \d{2}:", re.M)

    def _default(self, call):
        if call["schema"] is SceneTranslation:
            sid = re.search(r"Scene (s\d+)", call["prompt"]).group(1)
            ids = self._CUE_LINE.findall(call["prompt"])
            return SceneTranslation(
                scene_id=sid, mode="text",
                cues=[TranslatedCue(cue_id=i, ko=f"KO-{i}") for i in ids],
            )
        if call["schema"] is review._FlagBatch:
            return review._FlagBatch()
        return review._EditBatch()


def cue(cid, start, end, text="line", **kw):
    return Cue(id=cid, start_ms=start, end_ms=end, text=text, raw=text, **kw)


def scene(sid, start, end, cue_ids, act=1):
    return Scene(id=sid, start_ms=start, end_ms=end, cue_ids=cue_ids, act=act)


def video() -> VideoInfo:
    return VideoInfo(video_id="test-video", title="T", video_path="/v.mkv",
                    duration_ms=60_000, audio_stream=1,
                    en_sub_path="/en.srt", en_sub_origin="embedded:2")


PROXY = SimpleNamespace(name="proxy", uri="gs://proxy", mime_type="video/mp4")

# ------------------------------------------------------------------ cascade


def test_cascade_reruns_only_flagged_scenes(tmp_path):
    """Cascade re-sends with media only scenes whose cues flagged needs_media."""
    scenes = [scene("s001", 0, 5000, ["c0001"]), scene("s002", 6000, 11000, ["c0002"])]
    cues = [cue("c0001", 0, 4000, "who is she?"), cue("c0002", 6000, 10000, "hi")]
    snaps = {"s001": Snapshot(scene_id="s001"), "s002": Snapshot(scene_id="s002")}

    def respond(call):
        sid = re.search(r"Scene (s\d+)", call["prompt"]).group(1)
        ids = FakeGemini._CUE_LINE.findall(call["prompt"])
        if not call["clips"]:  # text pass
            if sid == "s001":
                return SceneTranslation(scene_id=sid, mode="text", cues=[
                    TranslatedCue(cue_id=ids[0], ko="저 여자 누구야",
                                  needs_media=MediaNeed.VIDEO,
                                  media_reason="deixis: who is 'she' on screen?")])
            return SceneTranslation(scene_id=sid, mode="text",
                                    cues=[TranslatedCue(cue_id=ids[0], ko="안녕")])
        return SceneTranslation(scene_id=sid, mode="av",
                                cues=[TranslatedCue(cue_id=ids[0], ko="저 여자 누구야")])

    gc = FakeGemini(respond)
    out = translate.translate_video(ws(tmp_path), gc, video(), scenes, cues,
                                   snaps, [], PROXY, mode="cascade")

    text_calls = [c for c in gc.calls if not c["clips"]]
    av_calls = [c for c in gc.calls if c["clips"]]
    assert len(text_calls) == 2 and len(av_calls) == 1
    assert "Scene s001" in av_calls[0]["prompt"]
    assert "c0001" in av_calls[0]["prompt"]  # retry names the flagged cue
    assert "Scene s002" not in av_calls[0]["prompt"]
    assert out["s001"].mode == "av" and out["s002"].mode == "text"


def test_av_mode_attaches_clip(tmp_path):
    scenes = [scene("s001", 1000, 5000, ["c0001"])]
    cues = [cue("c0001", 1000, 4000, "hi")]
    gc = FakeGemini()
    translate.translate_video(ws(tmp_path), gc, video(), scenes, cues,
                             {"s001": Snapshot(scene_id="s001")}, [], PROXY,
                             mode="av")
    clip = gc.calls[0]["clips"][0]
    assert clip.start_s == pytest.approx(0.0)  # padded -1s, clamped at 0
    assert clip.end_s == pytest.approx(6.0)


def _junk_mode(call):
    sid = re.search(r"Scene (s\d+)", call["prompt"]).group(1)
    ids = FakeGemini._CUE_LINE.findall(call["prompt"])
    return SceneTranslation(scene_id=sid, mode="translation",
                            cues=[TranslatedCue(cue_id=i, ko=f"KO-{i}") for i in ids])


def test_mode_is_set_by_code_not_model(tmp_path):
    sc = scene("s001", 1000, 5000, ["c0001"])
    cues = {"c0001": cue("c0001", 1000, 4000, "hi")}
    st = translate.translate_scene(ws(tmp_path), FakeGemini(_junk_mode), sc, cues,
                                   Snapshot(scene_id="s001"), [], [], mode="text", proxy=None)
    assert st.mode == "text"
    out = translate.translate_video(ws(tmp_path / "av"), FakeGemini(_junk_mode), video(), [sc],
                                   list(cues.values()), {"s001": Snapshot(scene_id="s001")},
                                   [], PROXY, mode="av")
    assert out["s001"].mode == "av"


def test_mode_overridden_on_legacy_checkpoint_hit(tmp_path, monkeypatch):
    w = ws(tmp_path)
    scenes = [scene("s001", 1000, 5000, ["c0001"])]
    cues = [cue("c0001", 1000, 4000, "hi")]
    snaps = {"s001": Snapshot(scene_id="s001")}
    gc = FakeGemini(_junk_mode)
    # simulate a checkpoint written before mode was code-owned
    with monkeypatch.context() as mp:
        mp.setattr(translate, "_align", lambda scene, sc, result, mode, obs: result)
        translate.translate_video(w, gc, video(), scenes, cues, snaps, [], None, mode="text")
    out = translate.translate_video(w, gc, video(), scenes, cues, snaps, [], None, mode="text")
    assert len(gc.calls) == 1  # checkpoint hit
    assert out["s001"].mode == "text"


# ------------------------------------------------------------ prompt content


def _rich_snapshot() -> Snapshot:
    return Snapshot(
        scene_id="s001",
        registers={"peter>tony": SceneRegister(
            level="haeyo", vocative="스타크 씨",
            shifts=[RegisterShift(at_cue="c0002", level="hae", reason="relationship_change")])},
        characters={"peter": CharacterState(
            emotion="anxious", knows=["his suit is Stark-made"],
            recent_events=["stopped a bike thief"])},
        voice_cards={"peter": VoiceCard(idiolect="rambling teen",
                                        vocabulary=["저기"], profanity_ceiling=0)},
        glossary=[GlossaryEntry(en="Stark", ko="스타크")],
        callbacks_live=[Callback(motif="power line", setup_cues=["c0009"],
                                 payoff_cues=["c0002"])],
        preserve_ambiguity=[AmbiguityFlag(cue_id="c0001", note="'them' hides the culprit's identity")],
        character_names={"peter": "피터", "tony": "토니"},
        tone_baseline="earnest teen comedy",
        rating="PG-13",
        personas={"peter": "eager teen hero, cheeky with his mentor"},
    )


def _rich_scene():
    sc = scene("s001", 0, 9000, ["c0001", "c0002"])
    cues = {"c0001": cue("c0001", 0, 4000, "They said it was them"),
            "c0002": cue("c0002", 5000, 8000, "you can call me Tony")}
    obs = [CueObservation(cue_id="c0001", speaker="peter", addressee="tony",
                          delivery="sarcastic")]
    return sc, cues, obs


def test_prompt_has_deltas_and_no_future_facts(tmp_path):
    """Full narrative context: the scene prompt carries snapshot state, deltas,
    glossary, callbacks, ambiguity flags, observations — and nothing from
    beyond the scene's causal horizon."""
    sc, cues, obs = _rich_scene()
    gc = FakeGemini(narrative="full")
    translate.translate_scene(ws(tmp_path), gc, sc, cues, _rich_snapshot(), obs, [],
                              mode="text", proxy=None)
    p = gc.calls[0]["prompt"]
    assert "REGISTERS" in p and "VOICE CARDS" in p
    system = gc.calls[0]["system"]
    assert "REGISTERS table" in system and "voice card" in system
    assert "[full]" not in system and "[lite]" not in system
    assert "48 | They said" in p or "max chars" in p  # budget column rendered
    for expected in ("peter>tony: 해요 · 스타크 씨", "shift from c0002: 반말 (relationship_change)", "스타크",
                     "power line", "hides the culprit", "anxious",
                     "stopped a bike thief", "speaker=peter", "sarcastic",
                     "저기", "PERSONAS", "rating=PG-13"):
        assert expected in p, expected
    # future fact lives only in a twist fact upstream, never in the snapshot
    assert "Vance" not in p and "Falcon" not in p


def test_lite_prompt_has_story_facts_and_registers_but_no_voice(tmp_path):
    """Default (lite) narrative context: names, knowledge, recent events,
    glossary, callbacks, ambiguity, observations and the register table rows
    (level · address term, shifts) — no voice cards or emotion state, and a
    system prompt whose register rule rests on the conventions + table."""
    sc, cues, obs = _rich_scene()
    gc = FakeGemini()
    translate.translate_scene(ws(tmp_path), gc, sc, cues, _rich_snapshot(), obs, [],
                              mode="text", proxy=None)
    p = gc.calls[0]["prompt"]
    for expected in ("CHARACTER NAMES", "peter -> 피터", "his suit is Stark-made",
                     "stopped a bike thief", "GLOSSARY", "스타크", "power line",
                     "hides the culprit", "speaker=peter", "sarcastic",
                     "VIDEO: tone=earnest teen comedy; rating=PG-13", "PERSONAS",
                     "peter: eager teen hero, cheeky with his mentor",
                     "REGISTERS", "peter>tony: 해요 · 스타크 씨", "shift from c0002: 반말"):
        assert expected in p, expected
    for banned in ("VOICE CARDS", "voice", "anxious", "emotion",
                   "rambling teen", "저기", "CUE DELTAS"):
        assert banned not in p, banned
    system = gc.calls[0]["system"]
    assert "voice card" not in system and "CUE DELTAS" not in system
    assert "[full]" not in system and "[lite]" not in system
    assert "never oscillate" in system and "Own parents who raised the speaker" in system
    for expected in ("Never mirror", "Fourth-wall", "PERSONAS", "빌어먹을", "-잖아", "천만에",
                     "telegraphic", "never inside [ ]", "ON-SCREEN CREDIT", "pure vocalizations"):
        assert expected in system, expected
    assert "wrap short labels" not in system


def test_review_lite_context_has_registers_but_no_voice(tmp_path):
    scenes = [scene("s001", 0, 9000, ["c0001", "c0002"])]
    cues = [cue("c0001", 0, 4000, "They said it was them"), cue("c0002", 5000, 8000, "call me Tony")]
    tr = {"s001": SceneTranslation(scene_id="s001", mode="text",
                                   cues=[TranslatedCue(cue_id="c0001", ko="걔들이래")])}
    fm = ContextMap(
        synopsis="syn", tone_baseline="warm",
        characters=[Character(id="peter", en_name="Peter", ko_name="피터", persona="p",
                              voice=VoiceCard(idiolect="rambling teen", profanity_ceiling=0))],
        glossary=[GlossaryEntry(en="Stark", ko="스타크")],
    )
    snaps = {"s001": _rich_snapshot()}
    gc = FakeGemini()
    levels = {"c0001": ("peter>tony", "haeyo")}
    review.review_windows(ws(tmp_path), gc, scenes, cues, tr, snaps, fm, [], levels=levels)
    review.finale_audit(ws(tmp_path), gc, cues, {"c0001": "걔들이래"}, fm, [], levels=levels)
    review.regenerate_flagged(ws(tmp_path), gc, [ReadFlag(cue_id="c0001", problem="번역투")],
                              {c.id: c for c in cues}, snaps, {"c0001": "s001", "c0002": "s001"},
                              {"c0001": "걔들이래"}, [], levels=levels)
    assert len(gc.calls) == 3
    for call in gc.calls:
        p, system = call["prompt"], call["system"]
        # every stage's rows read id | spk>adr | LEVEL | kind | EN | KO
        assert "c0001 | peter>tony | haeyo | kind=dialogue | EN: They said it was them | KO: 걔들이래" in p
        assert "c0002 | - | - | kind=dialogue" in p
        for banned in ("register shifts", "idiolect", "ceiling", "rambling teen",
                       "anxious", "DELTA"):
            assert banned not in p, banned
        assert "[lite]" not in system and "[full]" not in system
    # review + regen carry the scene's REGISTERS rows; the finale reads the LEVEL column
    assert "peter>tony: 해요 · 스타크 씨" in gc.calls[0]["prompt"]
    assert "peter>tony: 해요 · 스타크 씨" in gc.calls[2]["prompt"]
    assert "peter -> 피터" in gc.calls[0]["prompt"] and "스타크" in gc.calls[0]["prompt"]
    assert "peter / 피터: p" in gc.calls[1]["prompt"]
    # full mode still renders the register table and voice cards
    full = FakeGemini(narrative="full")
    review.review_windows(ws(tmp_path), full, scenes, cues, tr, snaps, fm, [])
    assert "REGISTERS" in full.calls[0]["prompt"] and "rambling teen" in full.calls[0]["prompt"]
    assert "LEVEL" in full.calls[0]["system"]


def test_cue_rows_carry_speaker_turns_and_length_aim(tmp_path):
    """Each S3 cue row shows its English speaker structure (turns=) and a soft
    length aim below the hard budget; the system prompt carries the dash rule,
    the register rubric, and fidelity-first guidance."""
    sc = scene("s001", 0, 9000, ["c0001", "c0002"])
    cues = {"c0001": cue("c0001", 0, 1500, "- Where were you?\n- Out."),
            "c0002": cue("c0002", 1600, 9000, "I waited all night for you.")}
    gc = FakeGemini()
    translate.translate_scene(ws(tmp_path), gc, sc, cues, Snapshot(scene_id="s001"),
                              [], [], mode="text", proxy=None)
    p = gc.calls[0]["prompt"]
    assert "aim/max chars | turns" in p
    # display time until the next cue start minus min_gap (1.517 s) x cps_max 8.0 -> 12; aim floor(12 * 0.85)
    assert "| 10/12 | turns=2 | - Where were you?" in p
    # 7.4 s would allow 59 chars; capped at two full lines (2 x cpl 14)
    assert "| 23/28 | turns=1 | I waited all night" in p
    system = gc.calls[0]["system"]
    for expected in ("turns=2", "turns=1", "OCR", "aim", "REGISTERS table",
                     "Korean subtitle style guide", "Decision procedure", "AI → user",
                     "해요체", "하십시오체", "nickname"):
        assert expected in system, expected
    assert system.rstrip().endswith(style.style_guide().rstrip())


def test_register_rubric_shared_by_register_stage_and_review():
    """The S2r register prompt embeds the style guide's register ladder (one
    source of truth), the video map no longer decides register, and review puts
    the cue-by-cue meaning check first."""
    rp = narrative._prompt("register.md")
    assert style.REGISTER_TOKEN not in rp and style.register_rules() in rp
    for expected in ("Own parents who raised the speaker", "classmates", "AI → user",
                     "the first step that applies wins", "first_private_talk_equals", "evidence_cues"):
        assert expected in rp, expected
    rp.format(title="T", year="2000", synopsis="", tone="", characters="", transcript="", pairs="")
    fm = narrative._prompt("context_map.md")
    assert "register_matrix" not in fm and "register_shifts" not in fm
    fm.format(title="T", year="2000", tmdb_hint="", transcript="", observations="")
    tl = narrative._prompt("timeline.md")
    assert "register_observed" not in tl
    for narrative_level in ("lite", "full"):
        rv = review._system(narrative_level)
        assert rv.index("- meaning (top priority)") < rv.index("- register:")
        assert "meaning | register" in rv
        assert style.style_guide() in rv


def test_narrative_context_setting_and_tagged_prompt_lines():
    assert Settings().narrative_context == "lite"
    with pytest.raises(ValueError):
        Settings(narrative_context="on")
    text = "shared\n- [full] F rule\n- [lite] L rule\n[lite]   indented\n"
    assert narrative_text(text, "full") == "shared\n- F rule\n"
    assert narrative_text(text, "lite") == "shared\n- L rule\n  indented\n"


def test_style_guide_feeds_s3_and_review_checkpoint_keys(monkeypatch):
    """S3 and S4 keys hash their system text, which carries the guide: editing
    the guide changes them (and the video map template's register section)."""
    before = (translate._system("lite"), review.prompt_texts("review", "lite"),
              review.prompt_texts("regen", "full"), narrative._prompt("context_map.md"))
    real = style.style_guide()
    monkeypatch.setattr(style, "style_guide", lambda: real.replace("## 3. Register and address",
                                                                  "## 3. Register and address\nNEW RULE"))
    after = (translate._system("lite"), review.prompt_texts("review", "lite"),
             review.prompt_texts("regen", "full"), narrative._prompt("context_map.md"))
    assert all(a != b and "NEW RULE" in str(b) for a, b in zip(before, after, strict=True))



def test_prev_ko_tail_feeds_next_scene_in_act(tmp_path):
    """Sequential within an act: scene 2's prompt shows scene 1's KO tail."""
    scenes = [scene("s001", 0, 5000, ["c0001"]),
              scene("s002", 6000, 11000, ["c0002"])]
    cues = [cue("c0001", 0, 4000, "first"), cue("c0002", 6000, 10000, "second")]
    snaps = {s.id: Snapshot(scene_id=s.id) for s in scenes}
    gc = FakeGemini()
    translate.translate_video(ws(tmp_path), gc, video(), scenes, cues,
                             snaps, [], None, mode="text")
    p2 = gc.calls[1]["prompt"]
    assert "KO-c0001" in p2          # previous scene's tail
    assert "PREVIOUS SCENE KOREAN TAIL" in p2
    assert "KO-c0001" not in gc.calls[0]["prompt"]


# -------------------------------------------------------------- checkpoints


def test_checkpoint_skips_identical_rerun(tmp_path):
    """Same inputs on the same workspace: zero new API calls on run two."""
    scenes = [scene("s001", 0, 5000, ["c0001"]), scene("s002", 6000, 11000, ["c0002"])]
    cues = [cue("c0001", 0, 4000, "a"), cue("c0002", 6000, 10000, "b")]
    snaps = {s.id: Snapshot(scene_id=s.id) for s in scenes}
    w = ws(tmp_path)
    gc = FakeGemini()
    translate.translate_video(w, gc, video(), scenes, cues, snaps, [], None, mode="text")
    assert len(gc.calls) == 2
    translate.translate_video(w, gc, video(), scenes, cues, snaps, [], None, mode="text")
    assert len(gc.calls) == 2  # all hits


def test_checkpoint_missed_when_inputs_change(tmp_path):
    """Changed cue text must miss the checkpoint."""
    scenes = [scene("s001", 0, 5000, ["c0001"])]
    snaps = {"s001": Snapshot(scene_id="s001")}
    w = ws(tmp_path)
    gc = FakeGemini()
    translate.translate_video(w, gc, video(), scenes, [cue("c0001", 0, 4000, "a")],
                             snaps, [], None, mode="text")
    translate.translate_video(w, gc, video(), scenes, [cue("c0001", 0, 4000, "changed")],
                             snaps, [], None, mode="text")
    assert len(gc.calls) == 2


# ------------------------------------------------------------------- ost cues


def test_ost_cues_only_uncovered_plot_critical(tmp_path):
    cues = [cue("c0001", 1000, 3000, "DANGER KEEP OUT")]  # covers the sign
    obs = [ChunkObservations(chunk_id="v01", start_ms=0, end_ms=60_000, on_screen_text=[
        OnScreenText(start_ms=1100, end_ms=2900, text="Danger keep out",
                     plot_critical=True, covered_by_cue="c0001"),
        OnScreenText(start_ms=9000, end_ms=11000, text="EXIT",
                     plot_critical=False),                       # ambient
        OnScreenText(start_ms=20000, end_ms=23000, text="Meet me at dawn",
                     plot_critical=True),                        # uncovered
        OnScreenText(start_ms=30000, end_ms=32000, text="DANGER KEEP OUT",
                     plot_critical=True),                        # same text near a cue? no cue there
    ])]
    new = translate.ost_cues(obs, cues, [])
    assert [c.id for c in new] == ["ost-0001", "ost-0002"]
    assert new[0].text == "Meet me at dawn" and new[0].forced and new[0].source == "ost"
    assert new[1].text == "DANGER KEEP OUT"  # plot-critical recurrence = own cue


def test_ost_cues_text_overlap_coverage(tmp_path):
    """A cue overlapping in time with the same words counts as coverage."""
    cues = [cue("c0001", 1000, 4000, "meet me at dawn")]
    obs = [OnScreenText(start_ms=1500, end_ms=3500, text="Meet me at dawn!",
                        plot_critical=True)]
    assert translate.ost_cues(obs, cues, []) == []


def test_realign_ost_caption_over_dialogue_stays_out_of_later_silent_gap():
    caption = OnScreenText(
        start_ms=969_000, end_ms=992_000, text="[caption] UBER HEADQUARTERS",
    )
    observations = [ChunkObservations(
        chunk_id="v03", start_ms=950_000, end_ms=1_080_000, on_screen_text=[caption],
    )]
    cues = [
        cue("c1", 960_000, 1_024_000, "We need to discuss the next meeting."),
        cue("c2", 1_040_000, 1_050_000, "Everyone is already here."),
    ]
    (chunk,) = translate.realign_ost_to_anchors_and_gaps(observations, cues)
    (retained,) = chunk.on_screen_text
    assert (retained.start_ms, retained.end_ms) == (969_000, 992_000)


def test_realign_ost_keeps_supported_dialogue_anchor_correction():
    observations = [ChunkObservations(
        chunk_id="v01", start_ms=0, end_ms=100_000, on_screen_text=[
            OnScreenText(start_ms=1000, end_ms=3000, text="[sign] Find the hidden door"),
        ],
    )]
    cues = [
        cue("c1", 1000, 3000, "Everyone is already here."),
        cue("c2", 50_000, 53_000, "Find the hidden door."),
    ]
    (chunk,) = translate.realign_ost_to_anchors_and_gaps(observations, cues)
    (corrected,) = chunk.on_screen_text
    assert (corrected.start_ms, corrected.end_ms) == (51_650, 53_650)


@pytest.mark.parametrize("kind", ["callout", "headline"])
@pytest.mark.parametrize("linked", [None, "c1"])
def test_ost_callout_and_headline_survive_partial_dialogue_coverage(kind, linked):
    cues = [cue("c1", 1000, 5000, "Travis talks to me. He does not listen to you.")]
    observations = [OnScreenText(
        start_ms=1500, end_ms=3500, text=f"[{kind}] Travis talks to me, not you.",
        covered_by_cue=linked,
    )]
    (overlay,) = translate.ost_cues(observations, cues, [])
    assert overlay.text == "Travis talks to me, not you."
    assert overlay.ost_kind == kind
    assert overlay.source == "ost" and overlay.forced


@pytest.mark.parametrize("kind", ["callout", "headline"])
@pytest.mark.parametrize("linked", [None, "c1"])
def test_ost_callout_and_headline_skip_verbatim_dialogue(kind, linked):
    cues = [cue("c1", 1000, 5000, "Travis talks to me, not you.")]
    observations = [OnScreenText(
        start_ms=1500, end_ms=3500, text=f"[{kind}] TRAVIS TALKS TO ME, NOT YOU!",
        covered_by_cue=linked,
    )]
    assert translate.ost_cues(observations, cues, []) == []


@pytest.mark.parametrize("kind", ["callout", "headline"])
def test_ost_callout_and_headline_are_kept_without_plot_critical_flag(kind):
    observations = [OnScreenText(start_ms=1000, end_ms=3000, text=f"[{kind}] THE NEXT BIG THING")]
    assert [c.ost_kind for c in translate.ost_cues(observations, [], [])] == [kind]


@pytest.mark.parametrize("offset,expected", [
    (250, [(350, 550), (1250, 1350)]),
    (0, [(100, 300), (1000, 1100)]),
    (-150, [(0, 150), (850, 950)]),
    (-248, [(0, 100), (752, 852)]),
    (-500, [(0, 100), (500, 600)]),
])
def test_apply_sync_offset_clamps_and_preserves_cue_metadata(offset, expected):
    from substage.acquire import apply_sync_offset

    cues = [cue("c1", 100, 300, "First", speaker_hint="Travis"), cue("c2", 1000, 1020, "Second")]
    shifted = apply_sync_offset(cues, offset)
    assert [(c.start_ms, c.end_ms) for c in shifted] == expected
    assert [(c.id, c.text, c.raw, c.speaker_hint) for c in shifted] == [
        (c.id, c.text, c.raw, c.speaker_hint) for c in cues
    ]
    assert [(c.start_ms, c.end_ms) for c in cues] == [(100, 300), (1000, 1020)]
    assert all(original is not changed for original, changed in zip(cues, shifted, strict=True))
    assert apply_sync_offset([], offset) == []


def test_evaluate_ost_and_sync_counts_coverage_emphasis_and_chain_position():
    from substage.refeval import evaluate_ost_and_sync

    ost = [
        {"start_ms": 1000, "end_ms": 2000, "text": "'첫 문구'", "pos": "top"},
        SimpleNamespace(start=2400, end=3000, ko="'둘째 문구'\n'셋째 문구'", position="an8"),
    ]
    dialogue = [cue("c1", 2950, 3500)]
    gold = [{"start_ms": 1200, "end_ms": 1800}, {"start_ms": 5000, "end_ms": 6000}]
    assert evaluate_ost_and_sync(ost, dialogue, gold_ost=gold, sync_delta_ms=-200.0) == {
        "ost_cov": 50.0, "ost_emp": 100.0, "ost_pos": 100.0, "sync_err_ms": 48.0,
    }
    ost[0].update(text="첫 문구", pos="bottom")
    scores = evaluate_ost_and_sync(ost, dialogue)
    assert scores == {"ost_cov": None, "ost_emp": 50.0, "ost_pos": 50.0, "sync_err_ms": None}


def test_evaluate_ost_and_sync_uses_strict_overlap_and_ass_tags():
    from substage.refeval import evaluate_ost_and_sync

    ost = [
        {"start": 1000, "end": 2000, "text": r"{\an8}'문구'"},
        {"start": 5000, "end": 6000, "text": r"{\an2}'독립 문구'"},
    ]
    scores = evaluate_ost_and_sync(ost, [cue("c1", 1999, 2500), cue("c2", 6000, 7000)])
    assert scores["ost_pos"] == scores["ost_emp"] == 100.0
    assert evaluate_ost_and_sync([], [], gold_ost=[], sync_delta_ms=-248.0) == {
        "ost_cov": 100.0, "ost_emp": 100.0, "ost_pos": 100.0, "sync_err_ms": 0.0,
    }
    assert evaluate_ost_and_sync([], [], gold_ost=[{"start_ms": 1, "end_ms": 2}])["ost_cov"] == 0.0


def test_ost_cues_sign_without_english_cue_becomes_cue():
    """A tagged sign no cue carries becomes an ost cue; the tag is stripped and
    a dialogue cue overlapping it with other words does not cover it."""
    cues = [cue("c0001", 4000, 7000, "Come on, we're late.")]
    obs = [OnScreenText(start_ms=5000, end_ms=6500, text="[sign] 42nd Street Station",
                        plot_critical=True)]
    new = translate.ost_cues(obs, cues, [])
    assert [(c.id, c.text, c.source, c.forced) for c in new] == [
        ("ost-0001", "42nd Street Station", "ost", True)
    ]


def test_credit_card_is_tagged_and_may_stay_empty():
    """An ost cue keeps its viewing-pass kind in raw: a credit card is tagged
    ON-SCREEN CREDIT for S3, rows as kind=credit in review and may stay empty,
    while a sign is plain ON-SCREEN TEXT and must be translated."""
    from substage import validate
    obs = [OnScreenText(start_ms=1000, end_ms=3000, text="[credit] Directed by Jane Roe",
                        plot_critical=True),
           OnScreenText(start_ms=9000, end_ms=11000, text="[sign] NO EXIT", plot_critical=True)]
    credit, sign = translate.ost_cues(obs, [], [])
    assert credit.text == "Directed by Jane Roe" and credit.raw == "[credit] Directed by Jane Roe"
    assert validate.ost_kind(credit) == "credit" and validate.ost_kind(sign) == "sign"
    rows = translate._render_cues([credit, sign], {}, 0.85)
    assert "Directed by Jane Roe [ON-SCREEN CREDIT]" in rows and "NO EXIT [ON-SCREEN TEXT]" in rows
    assert "kind=credit" in review._cue_lines([credit], {}, [])
    assert validate.check(credit, "", Settings()) == []
    assert [i.rule for i in validate.check(sign, "", Settings())] == ["empty"]


def test_ost_cues_covered_text_not_duplicated():
    """Covered: words carried by overlapping cues (even split across two), or by
    the linked cue. A wrong covered_by_cue link does not suppress the text."""
    cues = [
        cue("c0001", 1000, 2500, "Three months"),
        cue("c0002", 2600, 4000, "later"),
        cue("c0003", 9000, 11000, "Hey, Flash!"),
        cue("c0004", 20000, 22000, "Where are we going?"),
    ]
    obs = [
        OnScreenText(start_ms=1200, end_ms=3800, text="[caption] THREE MONTHS LATER"),
        OnScreenText(start_ms=9500, end_ms=10000, text="[spoken] Flash!"),
        OnScreenText(start_ms=20500, end_ms=21500, text="[sign] MOTEL", plot_critical=True,
                     covered_by_cue="c0004"),
    ]
    new = translate.ost_cues(obs, cues, [])
    assert [c.text for c in new] == ["MOTEL"]


def test_ost_cues_duplicate_reports_merged():
    """The same card reported by two chunks across their boundary is one cue
    spanning both reports (capped); a later reappearance stays its own cue."""
    obs = [
        ChunkObservations(chunk_id="v01", start_ms=0, end_ms=60_000, on_screen_text=[
            OnScreenText(start_ms=58_000, end_ms=60_000, text="[title] THE LONG ROAD"),
        ]),
        ChunkObservations(chunk_id="v02", start_ms=60_000, end_ms=120_000, on_screen_text=[
            OnScreenText(start_ms=60_000, end_ms=61_500, text="[title] The Long Road."),
            OnScreenText(start_ms=90_000, end_ms=90_200, text="[title] THE LONG ROAD"),
        ]),
    ]
    new = translate.ost_cues(obs, [], [])
    assert [(c.start_ms, c.end_ms, c.text) for c in new] == [
        (58_000, 61_500, "THE LONG ROAD"),
        (90_000, 91_000, "THE LONG ROAD"),  # 1 s floor
    ]


def test_ost_cues_duration_capped():
    obs = [OnScreenText(start_ms=0, end_ms=20_000, text="[dedication] For my father")]
    (c,) = translate.ost_cues(obs, [], [])
    assert (c.start_ms, c.end_ms) == (0, 6000)


def test_ost_cues_skip_logos_and_credit_scrolls():
    """Key credit cards are kept even without plot_critical; logos, scroll
    lists and name-list 'credits' are skipped."""
    obs = [
        OnScreenText(start_ms=0, end_ms=3000, text="[logo] ACME PICTURES", plot_critical=True),
        OnScreenText(start_ms=10_000, end_ms=13_000, text="[credit] Directed by Jane Roe"),
        OnScreenText(start_ms=100_000, end_ms=106_000, plot_critical=True,
                     text="[scroll] Grip / Best Boy / Gaffer / Caterer"),
        OnScreenText(start_ms=110_000, end_ms=116_000, plot_critical=True,
                     text="[credit] Ann Lee\nBo Park\nCy Diaz\nDee Moss\nEd Wu"),
    ]
    new = translate.ost_cues(obs, [], [])
    assert [c.text for c in new] == ["Directed by Jane Roe"]



def test_ost_cues_normalizes_v01_mmss_timestamps():
    obs = [
        ChunkObservations(chunk_id="v01", start_ms=0, end_ms=548_340, on_screen_text=[
            OnScreenText(start_ms=58_000, end_ms=61_000, text="[credit] SOME DOUCHEBAG'S VIDEO", plot_critical=True),
            OnScreenText(start_ms=101_000, end_ms=105_000, text="[credit] STARRING GOD'S PERFECT IDIOT", plot_critical=True),
            OnScreenText(start_ms=207_000, end_ms=210_000, text="[credit] PRODUCED BY ASSHATS", plot_critical=True),
            OnScreenText(start_ms=745_000, end_ms=748_000, text="[sign] HI, I'M FRANCIS!", plot_critical=True),
        ])
    ]
    new = translate.ost_cues(obs, [], [])
    assert [(c.start_ms, c.end_ms, c.text) for c in new] == [
        (58_000, 61_000, "SOME DOUCHEBAG'S VIDEO"),
        (61_000, 65_000, "STARRING GOD'S PERFECT IDIOT"),
        (127_000, 130_000, "PRODUCED BY ASSHATS"),
        (465_000, 468_000, "HI, I'M FRANCIS!"),
    ]


def test_ost_cues_normalizes_100ms_and_mixed_mmss_and_drops_clamped_boundary():
    obs = [
        ChunkObservations(chunk_id="v01", start_ms=0, end_ms=565_106, on_screen_text=[
            OnScreenText(start_ms=100_400, end_ms=103_000, text="[caption] A MAXIMUM EFFORT PRODUCTION", plot_critical=True),
            OnScreenText(start_ms=199_000, end_ms=204_000, text="[title] LOGAN", plot_critical=True),
            OnScreenText(start_ms=404_000, end_ms=406_000, text="[credit] A FILM BY SHAWN LEVY", plot_critical=True),
            OnScreenText(start_ms=565_106, end_ms=565_106, text="[sign] CLAMPED BOUNDARY ARTIFACT", plot_critical=True),
        ])
    ]
    new = translate.ost_cues(obs, [], [])
    assert [(c.start_ms, c.end_ms, c.text) for c in new] == [
        (60_400, 63_000, "A MAXIMUM EFFORT PRODUCTION"),
        (199_000, 204_000, "LOGAN"),
        (244_000, 246_000, "A FILM BY SHAWN LEVY"),
    ]


def test_ost_cues_splits_multi_bubble_sms_thread_and_keeps_unflagged_signs():
    obs = [
        ChunkObservations(
            chunk_id="v02",
            start_ms=600_000,
            end_ms=1_200_000,
            on_screen_text=[
                OnScreenText(
                    start_ms=606_000,
                    end_ms=615_200,
                    text="[sign] Hey Happy just checking in. I'm out of school at 2:45 PM Ready for my next mission! It's Peter BTW. Parker",
                    plot_critical=True,
                ),
                OnScreenText(
                    start_ms=1_017_000,
                    end_ms=1_019_000,
                    text="[sign] IS THIS YOUR BIKE? IF NOT, DON'T STEAL IT! SPIDER-MAN",
                    plot_critical=True,
                ),
                OnScreenText(
                    start_ms=1_100_000,
                    end_ms=1_103_000,
                    text="[sign] United States Academic Decathlon",
                    plot_critical=False,
                ),
            ],
        )
    ]
    new = translate.ost_cues(obs, [], [])
    assert [c.text for c in new] == [
        "Hey Happy just checking in. I'm out of school at 2:45 PM",
        "Ready for my next mission!",
        "It's Peter BTW.",
        "Parker",
        "IS THIS YOUR BIKE? IF NOT, DON'T STEAL IT! SPIDER-MAN",
        "United States Academic Decathlon",
    ]
    assert new[0].start_ms == 606_000
    assert new[3].end_ms >= 614_500
    assert all(c.end_ms - c.start_ms >= translate.OST_MIN_MS for c in new[:4])


def test_clef_ost_atomic_divide_and_conquer_questions():
    items = [{"id": "ost-0001", "text": "United States Academic Decathlon", "nearby_en": ""}]
    qs = review.build_clef_ost_intent_questions(items)
    assert set(qs.keys()) == {
        "ost-0001_viewer_intent",
        "ost-0001_not_logo_noise",
        "ost-0001_distinct_from_speech",
        "ost-0001_needs_quotes",
    }
    q_qual = review.build_clef_ost_quality_questions(
        [{"id": "ost-0001", "en": "Ready for my next mission!", "ko": "'다음 임무 준비됐어요!'", "target_level": "haeyo"}]
    )
    assert set(q_qual.keys()) == {
        "ost-0001_complete",
        "ost-0001_accurate",
        "ost-0001_register_ok",
        "ost-0001_layout_ok",
        "ost-0001_prop_gag_caption_ok",
    }


def test_gag_credit_cannot_stay_empty_and_merge_ost_cues_promotes_missing_captions():
    from substage import validate
    real_credit = Cue(
        id="ost-0001", start_ms=1000, end_ms=3000,
        text="Directed by Jane Roe", raw="[credit] Directed by Jane Roe",
        forced=True, source="ost", ost_kind="credit",
    )
    gag_credit = Cue(
        id="ost-0002", start_ms=61_000, end_ms=65_000,
        text="STARRING GOD'S PERFECT IDIOT", raw="[credit] STARRING GOD'S PERFECT IDIOT",
        forced=True, source="ost", ost_kind="credit",
    )
    assert validate.check(real_credit, "", Settings()) == []
    assert [i.rule for i in validate.check(gag_credit, "", Settings())] == ["empty"]
    rows = translate._render_cues([real_credit, gag_credit], {}, 0.85)
    assert "Directed by Jane Roe [ON-SCREEN CREDIT]" in rows
    assert "STARRING GOD'S PERFECT IDIOT [ON-SCREEN TEXT]" in rows

    obs = [
        ChunkObservations(
            chunk_id="v01", start_ms=0, end_ms=600_000,
            on_screen_text=[
                OnScreenText(start_ms=48_000, end_ms=51_500, text="[caption] IXTENCO, MEXICO.", plot_critical=True),
                OnScreenText(start_ms=61_000, end_ms=65_000, text="[credit] STARRING GOD'S PERFECT IDIOT", plot_critical=True),
            ],
        )
    ]
    merged = translate.merge_ost_cues_into_spine([real_credit], obs, [])
    assert [c.text for c in merged] == [
        "Directed by Jane Roe",
        "IXTENCO, MEXICO.",
        "STARRING GOD'S PERFECT IDIOT",
    ]
# --------------------------------------------------------------- sound/lyric


def test_sound_only_cue_never_translated(tmp_path):
    sc = scene("s001", 0, 9000, ["c0001", "c0002"])
    cues = {"c0001": cue("c0001", 0, 4000, "", sound_cues=["[gunshot]"]),
            "c0002": cue("c0002", 5000, 8000, "real line")}
    gc = FakeGemini()
    st = translate.translate_scene(ws(tmp_path), gc, sc, cues,
                                   Snapshot(scene_id="s001"), [], [],
                                   mode="text", proxy=None)
    by_id = {t.cue_id: t for t in st.cues}
    assert by_id["c0001"].ko == "" and by_id["c0001"].needs_media == MediaNeed.NONE
    assert by_id["c0002"].ko


# -------------------------------------------------------------------- review


def test_review_windows_emit_edits_and_group_scenes(tmp_path):
    scenes = [scene(f"s00{i}", i * 10000, i * 10000 + 9000, [f"c{i:03d}"]) for i in range(1, 5)]
    cues = [cue(f"c{i:03d}", i * 10000, i * 10000 + 8000, f"en {i}") for i in range(1, 5)]
    tr = {
        s.id: SceneTranslation(scene_id=s.id, mode="text",
                               cues=[TranslatedCue(cue_id=s.cue_ids[0], ko=f"KO {s.cue_ids[0]}")])
        for s in scenes
    }
    snaps = {s.id: Snapshot(scene_id=s.id,
                            registers={"a>b": SceneRegister(level="hae")}) for s in scenes}
    fm = ContextMap(synopsis="syn", tone_baseline="warm")

    def respond(call):
        if "s001" in call["prompt"]:
            return review._EditBatch(edits=[CueEdit(
                cue_id="c001", new_ko="다시 써", reason="register flip", kind="register")])
        return review._EditBatch()

    gc = FakeGemini(respond, narrative="full")
    edits = review.review_windows(ws(tmp_path), gc, scenes, cues, tr, snaps, fm, [])
    assert len(gc.calls) == 2                      # 4 scenes / window of 3 -> 2 calls
    assert [e.cue_id for e in edits] == ["c001"]   # edits only
    assert "REGISTERS" in gc.calls[0]["prompt"]
    assert "a>b: 반말" in gc.calls[0]["prompt"]
    assert "EN: en 1 | KO: KO c001" in gc.calls[0]["prompt"]


def test_finale_audit_single_whole_video_call(tmp_path):
    cues = [cue("c1", 0, 2000, "x"), cue("c2", 3000, 5000, "y")]
    ko = {"c1": "가", "c2": "나"}
    fm = ContextMap(synopsis="s", tone_baseline="t")
    gc = FakeGemini()
    review.finale_audit(ws(tmp_path), gc, cues, ko, fm, [])
    assert len(gc.calls) == 1
    assert gc.calls[0]["stage"] == "s4b_finale"
    assert "가" in gc.calls[0]["prompt"] and "x" in gc.calls[0]["prompt"]


def test_blind_read_uses_light_model_and_korean_only(tmp_path):
    gc = FakeGemini()
    flags = review.blind_read(ws(tmp_path), gc,
                              [("c1", "그녀가 그것을 그에게 주었다"), ("c2", "")])
    assert flags == []
    call = gc.calls[0]
    assert call["model"] == "fake-light-model"
    assert "그녀가" in call["prompt"] and "c2" not in call["prompt"]


def test_regenerate_flagged_one_call_per_scene(tmp_path):
    cues = {c.id: c for c in [cue("c1", 0, 2000, "x"), cue("c2", 3000, 5000, "y")]}
    scene_of = {"c1": "s001", "c2": "s002"}
    snaps = {"s001": Snapshot(scene_id="s001"), "s002": Snapshot(scene_id="s002")}
    flags = [ReadFlag(cue_id="c1", problem="번역투"),
             ReadFlag(cue_id="c2", problem="unclear referent")]
    ko = {"c1": "그것을 했다", "c2": "그녀가 했다"}

    def respond(call):
        if "s001" in call["prompt"]:
            return review._EditBatch(edits=[CueEdit(
                cue_id="c1", new_ko="했어", reason="natural", kind="naturalness")])
        return review._EditBatch()

    gc = FakeGemini(respond)
    edits = review.regenerate_flagged(ws(tmp_path), gc, flags, cues, snaps, scene_of, ko, [])
    assert len(gc.calls) == 2
    assert all(c["stage"] == "s4b_regen" for c in gc.calls)
    assert [e.cue_id for e in edits] == ["c1"]
    p = gc.calls[0]["prompt"]
    assert "*c1" in p and "번역투" in p


# ------------------------------------------------- checkpoint key coverage


def _suffix_responder(state):
    """Default-shaped responder whose KO for c0001 carries state['suffix']."""
    def respond(call):
        sid = re.search(r"Scene (s\d+)", call["prompt"]).group(1)
        ids = FakeGemini._CUE_LINE.findall(call["prompt"])
        return SceneTranslation(scene_id=sid, mode="text", cues=[
            TranslatedCue(cue_id=i, ko=f"KO-{i}{state['suffix'] if i == 'c0001' else ''}")
            for i in ids])
    return respond


def test_checkpoint_missed_when_prev_ko_tail_changes(tmp_path):
    """Scene 2's own inputs are unchanged, but scene 1's new KO changes the
    tail in scene 2's prompt -> scene 2 must be re-called."""
    scenes = [scene("s001", 0, 5000, ["c0001"]), scene("s002", 6000, 11000, ["c0002"])]
    snaps = {s.id: Snapshot(scene_id=s.id) for s in scenes}
    state = {"suffix": ""}
    gc = FakeGemini(_suffix_responder(state))
    w = ws(tmp_path)
    translate.translate_video(w, gc, video(), scenes,
                             [cue("c0001", 0, 4000, "first"), cue("c0002", 6000, 10000, "second")],
                             snaps, [], None, mode="text")
    assert len(gc.calls) == 2
    state["suffix"] = "-v2"
    translate.translate_video(w, gc, video(), scenes,
                             [cue("c0001", 0, 4000, "first, reworded"),
                              cue("c0002", 6000, 10000, "second")],
                             snaps, [], None, mode="text")
    assert len(gc.calls) == 4
    assert "Scene s002" in gc.calls[3]["prompt"] and "KO-c0001-v2" in gc.calls[3]["prompt"]


def test_checkpoint_missed_when_callback_ko_changes(tmp_path):
    """Setup cue c0001 falls outside the 6-line tail, so only the callback's
    fixed KO changes in scene 2's prompt -> scene 2 must be re-called."""
    s1_ids = [f"c000{i}" for i in range(1, 8)]  # 7 cues: tail keeps c0002..c0007
    scenes = [scene("s001", 0, 20000, s1_ids), scene("s002", 30000, 35000, ["c0009"])]
    snaps = {
        "s001": Snapshot(scene_id="s001"),
        "s002": Snapshot(scene_id="s002", callbacks_live=[
            Callback(motif="the line", setup_cues=["c0001"], payoff_cues=["c0009"])]),
    }

    def cues_with(first_text):
        return [cue("c0001", 0, 2000, first_text)] + [
            cue(cid, i * 2500, i * 2500 + 2000, f"l{i}") for i, cid in enumerate(s1_ids[1:], 1)
        ] + [cue("c0009", 30000, 34000, "payoff")]

    state = {"suffix": ""}
    gc = FakeGemini(_suffix_responder(state))
    w = ws(tmp_path)
    translate.translate_video(w, gc, video(), scenes, cues_with("setup"), snaps, [], None, mode="text")
    assert len(gc.calls) == 2
    assert "fixed KO: KO-c0001" in gc.calls[1]["prompt"]
    assert "PREVIOUS SCENE KOREAN TAIL" in gc.calls[1]["prompt"]
    assert "  c0001: KO-c0001" not in gc.calls[1]["prompt"]  # not in the tail
    state["suffix"] = "-v2"
    translate.translate_video(w, gc, video(), scenes, cues_with("setup, reworded"), snaps, [], None,
                             mode="text")
    assert len(gc.calls) == 4
    assert "Scene s002" in gc.calls[3]["prompt"]
    assert "fixed KO: KO-c0001-v2" in gc.calls[3]["prompt"]


# ----------------------------------------------------------------- lyrics


def test_lyric_relevance_rendered_and_incidental_returns_empty(tmp_path):
    sc = scene("s001", 0, 12000, ["c0001", "c0002", "c0003"])
    cues = {"c0001": cue("c0001", 0, 3000, "I'll never let you go", lyric=True),
            "c0002": cue("c0002", 4000, 7000, "la la la tonight", lyric=True),
            "c0003": cue("c0003", 8000, 11000, "ooh baby", lyric=True)}  # no observation
    obs = [CueObservation(cue_id="c0001", lyric_plot_relevant=True),
           CueObservation(cue_id="c0002", lyric_plot_relevant=False)]
    gc = FakeGemini()  # answers every cue with non-empty KO
    st = translate.translate_scene(ws(tmp_path), gc, sc, cues, Snapshot(scene_id="s001"),
                                   obs, [], mode="text", proxy=None)
    p = gc.calls[0]["prompt"]
    assert "I'll never let you go [LYRIC PLOT-RELEVANT]" in p
    assert "la la la tonight [LYRIC INCIDENTAL]" in p
    assert "ooh baby [LYRIC INCIDENTAL]" in p  # None -> incidental
    by_id = {t.cue_id: t.ko for t in st.cues}
    assert by_id == {"c0001": "KO-c0001", "c0002": "", "c0003": ""}


def test_review_rows_carry_cue_kind(tmp_path):
    sc = scene("s001", 0, 20000, ["c1", "c2", "c3", "c4", "ost-0001"])
    cues = [cue("c1", 0, 2000, "hello"),
            cue("c2", 3000, 5000, "", sound_cues=["[gunshot]"]),
            cue("c3", 6000, 8000, "sung story line", lyric=True),
            cue("c4", 9000, 11000, "la la", lyric=True),
            cue("ost-0001", 12000, 14000, "EXIT", source="ost", forced=True)]
    tr = {"s001": SceneTranslation(scene_id="s001", mode="text", cues=[
        TranslatedCue(cue_id="c1", ko="안녕"), TranslatedCue(cue_id="c3", ko="노래")])}
    obs = [ChunkObservations(chunk_id="v01", start_ms=0, end_ms=20000, cues=[
        CueObservation(cue_id="c3", lyric_plot_relevant=True),
        CueObservation(cue_id="c4", lyric_plot_relevant=False)])]
    gc = FakeGemini()
    review.review_windows(ws(tmp_path), gc, [sc], cues, tr, {}, ContextMap(synopsis="s", tone_baseline="t"),
                          obs)
    p = gc.calls[0]["prompt"]
    for row in (" c1 | - | - | kind=dialogue | EN: hello",
                " c2 | - | - | kind=sound | EN: (sound only)",
                " c3 | - | - | kind=lyric-relevant | EN: sung story line",
                " c4 | - | - | kind=lyric-incidental | EN: la la",
                " ost-0001 | - | - | kind=ost | EN: EXIT"):
        assert row in p, row
    gc2 = FakeGemini()
    review.finale_audit(ws(tmp_path), gc2, cues, {"c1": "안녕"}, ContextMap(synopsis="s", tone_baseline="t"),
                        obs)
    assert " c4 | - | - | kind=lyric-incidental" in gc2.calls[0]["prompt"]



def test_hwang_lyric_classification_and_bgm_guard():
    from substage.validate import classify_lyric_cue, is_translated_lyric

    c_spoken1 = cue("c1", 0, 2000, "Because I'm Tutti, and you're Frutti.")
    c_hook = cue("c2", 2100, 4000, "A wop bop b-luma, b-lop bam bom.", lyric=True)
    c_spoken2 = cue("c3", 4100, 6000, "So who else are you gonna listen to?")
    c_parody = cue("c4", 7000, 10000, "Kleenex-ica, Windex-ica\nShe's sexy-ca, oh, boy", lyric=True)
    c_vocable = cue("c5", 10100, 12000, "Oy, oy, oy, oy", lyric=True)
    c_tail = cue("c6", 14000, 17000, "Hyen-ica, hyen-ica, hyen-ica, hiya", lyric=True)
    c_bgm_under_speech = cue("c7", 18000, 21000, "A pretty woman makes her husband look small", lyric=True)
    sc_cues = [c_spoken1, c_hook, c_spoken2, c_parody, c_vocable, c_tail, c_bgm_under_speech]
    obs_map = {
        "c1": CueObservation(cue_id="c1", speaker="Bennie Loewy"),
        "c2": CueObservation(cue_id="c2", speaker="Bennie Loewy", lyric_plot_relevant=False),
        "c3": CueObservation(cue_id="c3", speaker="Bennie Loewy"),
        "c4": CueObservation(cue_id="c4", speaker="Bennie Loewy", lyric_plot_relevant=False),
        "c5": CueObservation(cue_id="c5", speaker="Bennie Loewy", lyric_plot_relevant=False),
        "c6": CueObservation(cue_id="c6", speaker="Bennie Loewy", lyric_plot_relevant=False),
        "c7": CueObservation(cue_id="c7", speaker=None, lyric_plot_relevant=True),
    }
    assert classify_lyric_cue(c_hook, obs_map, sc_cues) == "parody_or_rhyme_game"
    assert is_translated_lyric(c_hook, obs_map, sc_cues) is True
    assert classify_lyric_cue(c_parody, obs_map, sc_cues) == "parody_or_rhyme_game"
    assert is_translated_lyric(c_parody, obs_map, sc_cues) is True
    assert classify_lyric_cue(c_vocable, obs_map, sc_cues) == "omitted_vocable"
    assert is_translated_lyric(c_vocable, obs_map, sc_cues) is False
    assert classify_lyric_cue(c_tail, obs_map, [c_tail]) == "omitted_background_or_credit"
    assert is_translated_lyric(c_tail, obs_map, [c_tail]) is False
    assert classify_lyric_cue(c_bgm_under_speech, obs_map, [c_spoken1, c_bgm_under_speech]) == "omitted_background_or_credit"
    assert is_translated_lyric(c_bgm_under_speech, obs_map, [c_spoken1, c_bgm_under_speech]) is False

    qs = review.build_clef_lyric_questions([{"id": "c4", "en": c_parody.text, "ko": "크리넥사이카, 윈덱사이카", "lyric_mode": "parody_or_rhyme_game"}])
    assert set(qs.keys()) == {"c4_no_bgm_leak", "c4_rhyme_or_sense_ok", "c4_no_note_and_clean_format"}



def test_build_discourse_units_and_hesitation_merge_guard():
    from substage.translate import build_discourse_units
    from substage import layout
    from substage.config import Settings

    c36 = cue("c0036", 201162, 203408, "And as a more concrete gesture of our\ngratitude,")
    c37 = cue("c0037", 203491, 205408, "we'd like to offer you a...")
    c38 = cue("c0038", 206836, 208636, "Let's call it a favor.")
    c39 = cue("c0039", 208667, 210239, "The nature of it is your choice.")
    obs_map = {
        cid: CueObservation(cue_id=cid, speaker="harry", addressee="michelle")
        for cid in ("c0036", "c0037", "c0038", "c0039")
    }
    units = build_discourse_units([c36, c37, c38, c39], obs_map)
    assert len(units) == 1
    du = units[0]
    assert du.cue_ids == ("c0036", "c0037", "c0038", "c0039")
    assert du.speech_act_type == "hesitation_completion"
    assert du.hold_ellipsis_cue_ids == ("c0037",)

    qs = review.build_clef_discourse_questions([
        {
            "unit_id": du.unit_id,
            "speech_act_type": du.speech_act_type,
            "en": [c36.text, c37.text, c38.text, c39.text],
            "ko": ["그리고 감사의 표시로", "저희가 부인께...", "도움을 하나 드린다고 생각하시죠", "무슨 일이든 괜찮으니"],
        }
    ])
    assert set(qs.keys()) == {
        "c0036..c0039__q1_disconnected_or_broken_flow",
        "c0036..c0039__q2_premature_object_or_idiom_calque",
        "c0036..c0039__q3_referent_or_subject_mismatch",
        "c0036..c0039__q4_unnatural_speech_act",
    }

    # Ensure layout._try_merge preserves intentional '...' hesitation beat on c0037 when c0036 is already a readable phrase
    evs = layout.layout(
        [c36, c37, c38, c39],
        {
            "c0036": "감사의 표시로\n저희가",
            "c0037": "부인께...",
            "c0038": "도움을 하나\n드린다고 생각하시죠",
            "c0039": "무슨 일이든\n괜찮으니",
        },
        {cid: "s002" for cid in ("c0036", "c0037", "c0038", "c0039")},
        {cid: "harry" for cid in ("c0036", "c0037", "c0038", "c0039")},
        Settings(),
    )
    assert [e.cue_ids for e in evs] == [["c0036"], ["c0037"], ["c0038"], ["c0039"]]


def test_detect_echo_and_moniker_chains_and_scene_register_lock():
    from substage.translate import detect_echo_and_moniker_chains
    from substage.register import lock_scene_dyad_levels, lock_scene_pairs, scene_dash_turn_pairs

    # 1. Cross-scene conditional 'if' echo (4.13s pause between c1134 and c1135)
    c1128 = cue("c1128", 3670584, 3671418, "If they can fix your world...")
    c1129 = cue("c1129", 3671585, 3673212, "what's the first thing you're gonna do?")
    c1134 = cue("c1134", 3682136, 3683805, "If... they can fix your world?")
    c1135 = cue("c1135", 3687934, 3689500, 'What do you mean, "if?"')
    links = detect_echo_and_moniker_chains([c1128, c1129, c1134, c1135])
    cond_links = [el for el in links if el.kind == "conditional_echo"]
    assert len(cond_links) == 1
    assert cond_links[0].setup_ids == ("c1128", "c1129", "c1134")
    assert cond_links[0].payoff_ids == ("c1135",)

    # 2. Descriptive moniker gloss ('Blind Al' -> 'You call her "Blind Al?"' -> "Well, she's blind.")
    c1032 = cue("c1032", 3380000, 3383000, "My blind, elderly roommate, Blind Al, always says")
    c1036 = cue("c1036", 3390000, 3392000, 'That\'s her name? You call her "Blind Al?"')
    c1037 = cue("c1037", 3392200, 3394000, "Well, she's blind.")
    m_links = detect_echo_and_moniker_chains([c1032, c1036, c1037])
    assert any(el.kind == "moniker_gloss" and "c1032" in el.setup_ids for el in m_links)

    # 3. Scene register lock: spurious addressee="group" in a 2-person scene is locked to the dyad partner,
    # while theatrical vocative ('Your Honor') is preserved as exempt.
    c_a1 = cue("c1044", 3450000, 3451000, "Who are you?")
    c_a2 = cue("c1045", 3451200, 3452500, "I'm Nicepool.")
    c_grp = cue("c1050", 3455000, 3456500, "Can't even tell.")
    c_vh = cue("c1067", 3470000, 3472000, "Guilty on all charges, Your Honor.")
    cues_map = {c.id: c for c in (c_a1, c_a2, c_grp, c_vh)}
    pairs_in = {
        "c1044": ("wade", "nicepool"),
        "c1045": ("nicepool", "wade"),
        "c1050": ("wade", "group"),
        "c1067": ("wade", "group"),
    }
    locked = lock_scene_pairs(["c1044", "c1045", "c1050", "c1067"], cues_map, pairs_in)
    assert locked["c1050"] == ("wade", "nicepool")
    assert locked["c1067"] == ("wade", "group")

    c_dash = cue("c1051", 3457000, 3459000, "- Sorry!\n- Fuck!")
    cues_map["c1051"] = c_dash
    locked["c1051"] = ("nicepool", "wade")
    dp = scene_dash_turn_pairs(["c1044", "c1045", "c1050", "c1051", "c1067"], cues_map, locked)
    assert dp["c1051"] == (("nicepool", "wade"), ("wade", "nicepool"))
    dyad_lvls = lock_scene_dyad_levels(
        None,
        ["c1044", "c1045", "c1050", "c1067"],
        cues_map,
        locked,
        ko={"c1044": "넌 누구야?", "c1045": "난 데드풀이야", "c1050": "티도 안 나", "c1067": "할 말이 없사옵니다 폐하"},
    )
    assert dyad_lvls[("wade", "nicepool")] == "hae"
    assert dyad_lvls[("nicepool", "wade")] == "hae"
    # 4. Consensus 4-axis Clef questions
    qs = review.build_clef_echo_register_questions([
        {"window_id": "w1", "en": [c1128.text, c1134.text, c1135.text], "ko": ["만약 네 세계를 고친다면", "만약 고칠 수 있다면?", "만약이라니?"]}
    ])
    assert set(qs.keys()) == {
        "w1__q1_echo_and_moniker_anchor",
        "w1__q2_scene_register_consistency",
        "w1__q3_contextual_meaning_and_pragmatics",
        "w1__q4_natural_korean_and_subtitle_limits",
    }


def test_episode_partition_salesman_vs_locker_and_voiceover_bridge():
    from substage.models import Character, CueObservation, VoiceCard
    from substage.register import NARRATION, lock_scene_dyad_levels, lock_scene_pairs
    from substage.review import _violates_target_level
    from substage.scenes import partition_episodes

    # Fixture A (s012): Kia Carnival salesman->customer (Tammy) followed by DRIVEMAX card and Locker Room (Peter)
    s12_cues = {
        "c0179": cue("c0179", 705162, 708374, "Technically, the, the Carnival's not a minivan."),
        "c0180": cue("c0180", 709376, 710460, "It's an MPV."),
        "c0181": cue("c0181", 710543, 713672, "So how does the Kia compare to the Honda Odyssey?"),
        "c0183": cue("c0183", 713755, 716549, "That's a great question. It doesn't fucking suck."),
        "c0184": cue("c0184", 716966, 719261, "You know, you can answer the question without swearing."),
        "c0185": cue("c0185", 719344, 721513, "I'm sorry, Tammy, I don't have kids."),
        "ost-0010": Cue(id="ost-0010", start_ms=727477, end_ms=729230, text="DRIVEMAX", raw="DRIVEMAX", source="ost"),
        "c0187": cue("c0187", 729271, 730730, "You'll get 'em next time, pal."),
        "c0188": cue("c0188", 730813, 735568, "And look, you can always go back to superhero-ing."),
        "c0193": cue("c0193", 736027, 738780, "I don't keep it in my locker so that I can wear it."),
        "c0194": cue("c0194", 738863, 741450, "I keep it in my locker in case we need to saddle up again."),
        "c0201": cue("c0201", 741533, 744411, "Hey, hey, hey. What is that doing in there?"),
    }
    s12_cids = ["c0179", "c0180", "c0181", "c0183", "c0184", "c0185", "c0187", "c0188", "c0193", "c0194", "c0201"]
    raw_pairs = {
        "c0179": ("wade", "group"),
        "c0180": ("wade", "group"),
        # c0181..c0185 were dropped by cue_pairs because Tammy is not in film_map.yaml
        "c0188": ("peter", "wade"),
        "c0193": ("wade", "peter"),
        "c0194": ("wade", "peter"),
        "c0201": ("peter", "wade"),
    }
    obs = [
        CueObservation(cue_id="c0179", speaker="Wade Wilson", addressee="group"),
        CueObservation(cue_id="c0180", speaker="Wade Wilson", addressee="group"),
        CueObservation(cue_id="c0181", speaker="Tammy", addressee="Wade Wilson"),
        CueObservation(cue_id="c0183", speaker="Wade Wilson", addressee="Tammy"),
        CueObservation(cue_id="c0184", speaker="Tammy", addressee="Wade Wilson"),
        CueObservation(cue_id="c0185", speaker="Wade Wilson", addressee="Tammy"),
        CueObservation(cue_id="c0187", speaker="Peter Pool", addressee="Wade Wilson"),
        CueObservation(cue_id="c0188", speaker="Peter Pool", addressee="Wade Wilson"),
        CueObservation(cue_id="c0193", speaker="Wade Wilson", addressee="Peter Pool"),
        CueObservation(cue_id="c0194", speaker="Wade Wilson", addressee="Peter Pool"),
        CueObservation(cue_id="c0201", speaker="Peter Pool", addressee="Wade Wilson"),
    ]
    locked = lock_scene_pairs(s12_cids, s12_cues, raw_pairs, obs)
    # Carnival group cues must resolve to ep:tammy, NEVER to peter!
    assert locked["c0179"] == ("wade", "ep:tammy")
    assert locked["c0180"] == ("wade", "ep:tammy")
    assert locked["c0181"] == ("ep:tammy", "wade")
    assert locked["c0183"] == ("wade", "ep:tammy")
    assert locked["c0187"] == ("peter", "wade")
    assert locked["c0193"] == ("wade", "peter")

    eps = partition_episodes(s12_cids + ["ost-0010"], s12_cues, locked, scene_id="s012")
    assert len(eps) == 2
    assert eps[0].has_episodic_participant is True
    assert eps[1].has_episodic_participant is False

    cmap = ContextMap(
        synopsis="", tone_baseline="action",
        characters=[
            Character(id="wade", en_name="Wade Wilson", ko_name="웨이드", persona="merc", voice=VoiceCard(idiolect="Informal 반말 to everyone")),
            Character(id="peter", en_name="Peter Pool", ko_name="피터", persona="buddy", voice=VoiceCard(idiolect="Warm 반말 to Wade")),
        ],
    )
    ko_s12 = {
        "c0179": "엄밀히 말하면\n카니발은 미니밴이 아니야",
        "c0180": "MPV지",
        "c0181": "기아차는 비교하면 어떤가요?",
        "c0183": "좋은 질문이야",
        "c0184": "욕 안 쓰고도 대답할 수 있잖아요",
        "c0185": "미안, 태미\n난 애가 없어서",
        "c0187": "다음엔 잘될 거야",
        "c0188": "언제든 히어로로 돌아가도 되잖아",
        "c0193": "내가 입으려고 사물함에 둔 게 아냐",
        "c0194": "다시 출동할 때를 대비해서지",
        "c0201": "야, 저게 왜 거기 있어?",
    }
    dyad_lvls = lock_scene_dyad_levels(None, s12_cids, s12_cues, locked, ko=ko_s12, context_map=cmap)
    assert dyad_lvls[("wade", "ep:tammy")] == "haeyo"
    assert dyad_lvls[("ep:tammy", "wade")] == "haeyo"
    assert dyad_lvls[("wade", "peter")] == "hae"
    assert dyad_lvls[("peter", "wade")] == "hae"
    # Elliptical bare -어서 + plain 1p '난' + bare '미안' violates target 'haeyo'
    assert _violates_target_level("미안, 태미\n난 애가 없어서", "haeyo") is True
    assert _violates_target_level("죄송해요, 태미 씨\n제가 애가 없어서요", "haeyo") is False

    # Fixture B (s011): Cross-cut voiceover sentence bridging EARTH-10005 card
    s11_cues = {
        "c0175": cue("c0175", 679053, 681889, "-Yeah, no, thank you.\n-Oh, okay."),
        "c0176": cue("c0176", 683850, 686936, "You know what they say.\nWhen one door closes…"),
        "ost-0008": Cue(id="ost-0008", start_ms=687000, end_ms=688400, text="EARTH-10005", raw="EARTH-10005", source="ost"),
        "c0177": cue("c0177", 688271, 689314, "your locker at work opens."),
    }
    s11_raw = {
        "c0175": ("happy", "wade"),
        "c0176": ("wade", "happy"),
        "c0177": ("wade", "happy"),
    }
    s11_locked = lock_scene_pairs(["c0175", "c0176", "c0177"], s11_cues, s11_raw)
    assert s11_locked["c0176"] == ("wade", NARRATION)
    assert s11_locked["c0177"] == ("wade", NARRATION)
    s11_lvls = lock_scene_dyad_levels(None, ["c0175", "c0176", "c0177"], s11_cues, s11_locked)
    assert s11_lvls[("wade", NARRATION)] == "hae"


def test_dedupe_ost_shadows_and_digit_guard():
    from substage.translate import _same_text, dedupe_ost_shadows

    # 1. Digit-exact guard: TIMELINE INTEGRITY 77% vs 2% must never match
    assert _same_text("TIMELINE INTEGRITY 77%", "TIMELINE INTEGRITY 2%") is False
    assert _same_text("TIMELINE INTEGRITY 2%", "Timeline Integrity 2%") is True

    # 2. Logan phantom shadows (ost-0049 at 79540, ost-0050 at 199000, ost-0063 at 7709000) dropped;
    #    frame-exact ost-0068 (119736..122417) and ost-0072 (7868170..7872503) kept!
    spoken = [
        cue("c0014", 79910, 80910, "Fuck."),
        cue("c0047", 199440, 201030, "(MAN SPEAKING SPANISH)"),
    ]
    logan_ost = [
        Cue(id="ost-0049", start_ms=79540, end_ms=82420, text="LOGAN", raw="[title] LOGAN", source="ost"),
        Cue(id="ost-0068", start_ms=119736, end_ms=122417, text="LOGAN", raw="[title] LOGAN", source="ost"),
        Cue(id="ost-0050", start_ms=199000, end_ms=204000, text="LOGAN", raw="[title] LOGAN", source="ost"),
        Cue(id="ost-0063", start_ms=7709000, end_ms=7711500, text="LOGAN", raw="[title] LOGAN", source="ost"),
        Cue(id="ost-0072", start_ms=7868170, end_ms=7872503, text="LOGAN", raw="[title] LOGAN", source="ost"),
    ]
    kept = dedupe_ost_shadows(logan_ost, spoken)
    assert [c.id for c in kept] == ["ost-0068", "ost-0072"]

    # 3. Genuine cross-scene repeats (STARK INDUSTRIES > 60s apart) are both preserved
    stark_ost = [
        Cue(id="ost-0033", start_ms=2159850, end_ms=2162820, text="STARK INDUSTRIES", raw="STARK INDUSTRIES", source="ost"),
        Cue(id="ost-0057", start_ms=2407893, end_ms=2409295, text="STARK INDUSTRIES", raw="STARK INDUSTRIES", source="ost"),
    ]
    assert [c.id for c in dedupe_ost_shadows(stark_ost, [])] == ["ost-0033", "ost-0057"]
# ------------------------------------------------------------ character names


def test_character_names_rendered_in_s3_and_regen(tmp_path):
    snap = Snapshot(scene_id="s001", character_names={"peter": "피터", "tony": "토니"})
    sc = scene("s001", 0, 5000, ["c1"])
    cues = {"c1": cue("c1", 0, 4000, "Mr. Stark!")}
    gc = FakeGemini()
    translate.translate_scene(ws(tmp_path), gc, sc, cues, snap, [], [], mode="text", proxy=None)
    p = gc.calls[0]["prompt"]
    assert "CHARACTER NAMES (use exactly):" in p
    assert "peter -> 피터" in p and "tony -> 토니" in p

    gc2 = FakeGemini()
    review.regenerate_flagged(ws(tmp_path), gc2, [ReadFlag(cue_id="c1", problem="번역투")],
                              cues, {"s001": snap}, {"c1": "s001"}, {"c1": "스타크 씨"}, [])
    p2 = gc2.calls[0]["prompt"]
    assert "CHARACTER NAMES (use exactly): peter -> 피터, tony -> 토니" in p2


# ------------------------------------------------------------------- blocked


def _bisect_responder(blocked_id):
    """Default response unless the CUES section renders `blocked_id`."""
    def respond(call):
        if blocked_id in FakeGemini._CUE_LINE.findall(call["prompt"]):
            raise GeminiBlocked(call["stage"], "prompt blocked")
        return FakeGemini()._default(call)
    return respond


def test_translate_blocked_bisects_to_single_cue(tmp_path, caplog):
    """A refused prompt narrows to the one offending cue: it alone gets an
    empty ko + media_reason; its sibling is translated normally."""
    scenes = [scene("s001", 0, 9000, ["c0001", "c0002"])]
    cues = [cue("c0001", 0, 4000, "hi"), cue("c0002", 5000, 9000, "the line")]
    gc = FakeGemini(_bisect_responder("c0002"))

    with caplog.at_level(logging.WARNING, logger="substage.translate"):
        out = translate.translate_video(
            ws(tmp_path), gc, video(), scenes, cues,
            {"s001": Snapshot(scene_id="s001")}, [], None, mode="text")

    by_id = {t.cue_id: t for t in out["s001"].cues}
    assert by_id["c0001"].ko  # sibling translated via the clean half
    assert by_id["c0002"].ko == ""
    assert by_id["c0002"].media_reason == "blocked by provider"
    assert "blocked" in caplog.text


def test_translate_av_blocked_retries_text_then_bisects(tmp_path):
    """av mode: one text-mode retry of the whole scene before cue bisection."""
    scenes = [scene("s001", 0, 9000, ["c0001", "c0002"])]
    cues = [cue("c0001", 0, 4000, "hi"), cue("c0002", 5000, 9000, "the line")]
    gc = FakeGemini(_bisect_responder("c0002"))

    out = translate.translate_video(
        ws(tmp_path), gc, video(), scenes, cues,
        {"s001": Snapshot(scene_id="s001")}, [], PROXY, mode="av")

    assert gc.calls[0]["clips"]              # original av attempt
    assert gc.calls[1]["clips"] == []        # text-mode retry of whole scene
    assert gc.calls[1]["stage"].endswith(":text")
    assert all(c["clips"] == [] for c in gc.calls[1:])  # bisection is text-only
    by_id = {t.cue_id: t for t in out["s001"].cues}
    assert by_id["c0002"].media_reason == "blocked by provider"
    assert out["s001"].mode == "text"        # degraded result records real mode


def test_review_window_blocked_skips_keeps_rest(tmp_path, caplog):
    """A refused window contributes no edits; later windows still review."""
    scenes = [scene(f"s00{i}", i * 10000, i * 10000 + 9000, [f"c{i:03d}"])
              for i in range(1, 5)]
    cues = [cue(f"c{i:03d}", i * 10000, i * 10000 + 2000, f"en {i}")
            for i in range(1, 5)]
    translations = {
        f"s00{i}": SceneTranslation(
            scene_id=f"s00{i}", mode="text",
            cues=[TranslatedCue(cue_id=f"c{i:03d}", ko=f"KO c{i:03d}")])
        for i in range(1, 5)
    }

    def respond(call):
        if "c002" in call["prompt"]:
            raise GeminiBlocked(call["stage"], "prompt blocked")
        return review._EditBatch(
            edits=[CueEdit(cue_id="c004", new_ko="x", reason="r", kind="consistency")])

    gc = FakeGemini(respond)
    with caplog.at_level(logging.WARNING, logger="substage.review"):
        edits = review.review_windows(
            ws(tmp_path), gc, scenes, cues, translations, {}, ContextMap(synopsis="s", tone_baseline="t"), [])

    assert [e.cue_id for e in edits] == ["c004"]
    assert len(gc.calls) == 2
    assert "blocked" in caplog.text


def _semantic_fixture(*, essential=None):
    source_cue = cue("c1", 0, 2500, "Ready? Yeah.")
    source = SemanticSource(scene_id="s1", acts=[
        SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=6, text="Ready?", speaker="a", meaning="readiness question"),
        SourceAct(ref="c1:t2:a1", cue_id="c1", turn_ordinal=2, act_ordinal=1,
                  start_char=7, end_char=12, text="Yeah.", speaker="b",
                  meaning="affirmative response", essential=essential,
                  uncertainty=["answer or incidental acknowledgment depends on scene"]),
    ], relations=[SourceRelation(kind="question_answer", refs=["c1:t1:a1", "c1:t2:a1"],
                                 claim="possible answer", uncertainty=["check scene"])])
    return source_cue, source


@pytest.mark.parametrize("second_speaker,expected", [
    ("b", "- 준비됐어?\n- 응"),
    ("a", "준비됐어?\n응"),
    (None, "준비됐어?\n응"),
    ("unknown", "준비됐어?\n응"),
])
def test_semantic_target_uses_actual_retained_voices_not_source_turn_count(second_speaker, expected):
    source_cue, source = _semantic_fixture()
    draft = SemanticDraft(scene_id="s1", cues=[SemanticDraftCue(cue_id="c1", fragments=[
        RetainedFragment(text="준비됐어?", refs=["c1:t1:a1"], speaker="a"),
        RetainedFragment(text="응", refs=["c1:t2:a1"], speaker=second_speaker),
    ])])
    output, lineage = translate._semantic_target(
        source, draft, Scene(id="s1", start_ms=0, end_ms=2500, cue_ids=["c1"]),
        [source_cue], "codex/gpt-6.1-sol",
    )
    assert output[0].ko == expected
    assert lineage.structural_valid is True
    assert lineage.semantic_assessment == "unassessed"
    assert [fragment.text for fragment in lineage.fragments_by_cue["c1"]] == ["준비됐어?", "응"]


@pytest.mark.parametrize("essential", [True, False, None])
def test_semantic_omission_is_a_declared_coverage_record_not_fake_reply_or_dash(essential):
    source_cue, source = _semantic_fixture(essential=essential)
    draft = SemanticDraft(scene_id="s1", cues=[SemanticDraftCue(cue_id="c1", fragments=[
        RetainedFragment(text="준비됐어?", refs=["c1:t1:a1"], speaker="a"),
    ])], omissions=[OmittedAct(ref="c1:t2:a1", reason="declared incidental backchannel")])
    output, lineage = translate._semantic_target(
        source, draft, Scene(id="s1", start_ms=0, end_ms=2500, cue_ids=["c1"]),
        [source_cue], "codex/gpt-6.1-sol",
    )
    assert output[0].ko == "준비됐어?"
    assert lineage.source.acts[1].essential is essential
    assert lineage.omissions[0].ref == "c1:t2:a1"
    assert lineage.structural_valid is True
    assert lineage.semantic_assessment == "unassessed"  # not confirmed semantic fidelity


def test_semantic_lineage_permits_one_voice_multi_act_merge_and_repetition():
    source_cue, source = _semantic_fixture()
    source.acts[1] = source.acts[1].model_copy(update={
        "ref": "c1:t1:a2", "turn_ordinal": 1, "act_ordinal": 2, "speaker": "a",
    })
    source.relations = []
    draft = SemanticDraft(scene_id="s1", cues=[SemanticDraftCue(cue_id="c1", fragments=[
        RetainedFragment(text="준비됐다고", refs=["c1:t1:a1", "c1:t1:a2"], speaker="a"),
        RetainedFragment(text="준비됐어", refs=["c1:t1:a1", "c1:t1:a2"], speaker="a"),
    ])])
    output, lineage = translate._semantic_target(
        source, draft, Scene(id="s1", start_ms=0, end_ms=2500, cue_ids=["c1"]),
        [source_cue], "codex/gpt-6.1-sol",
    )
    assert output[0].ko == "준비됐다고\n준비됐어"
    assert lineage.fragments_by_cue["c1"][0].refs == lineage.fragments_by_cue["c1"][1].refs
    assert lineage.structural_valid


@pytest.mark.parametrize("failure", ["identity", "span", "duplicate", "order", "uncovered", "relation"])
def test_semantic_source_rejects_corrupt_identity_spans_order_and_refs(failure):
    source_cue, source = _semantic_fixture()
    if failure == "identity":
        source.acts[0].ref = "invented"
    elif failure == "span":
        source.acts[0].text = "a different source"
    elif failure == "duplicate":
        source.acts.append(source.acts[0])
    elif failure == "order":
        source.acts.reverse()
    elif failure == "uncovered":
        source.acts.pop()
    else:
        source.relations[0].refs = ["not-a-source-ref"]
    with pytest.raises(ValueError):
        perception.validate_semantic_source(source, [source_cue])


def test_semantic_lineage_rejects_declared_refs_without_actual_output_alignment():
    source_cue, source = _semantic_fixture()
    lineage = SemanticLineage(source=source, fragments_by_cue={"c1": [
        RetainedFragment(text="준비됐어?", refs=["c1:t1:a1"], speaker="a"),
        RetainedFragment(text="응", refs=["c1:t2:a1"], speaker="b"),
    ]})
    with pytest.raises(ValueError, match="actual Korean"):
        perception.validate_semantic_lineage(
            lineage, [source_cue], [TranslatedCue(cue_id="c1", ko="준비됐어?")],
        )


def test_semantic_reattribution_preserves_actual_text_and_leaves_speaker_conflict_unknown():
    source_cue, source = _semantic_fixture()
    previous = [RetainedFragment(text="준비됐어?", refs=["c1:t1:a1"], speaker="a")]
    raw = translate._SemanticAttributionDraft(fragments=[
        RetainedFragment(text="준비됐니?", refs=["c1:t1:a1"], speaker="b"),
    ])
    fragments, omissions = translate._validate_reattribution(raw, source, [source_cue], "준비됐니?", previous)
    assert fragments[0].text == "준비됐니?"
    assert fragments[0].speaker is None
    assert fragments[0].refs == []
    assert fragments[0].prior_refs == ["c1:t1:a1"]
    assert fragments[0].attribution == "unverified"
    assert omissions == []
    with pytest.raises(ValueError, match="changed actual Korean"):
        translate._validate_reattribution(raw, source, [source_cue], "준비됐어?", previous)


def test_semantic_reattribution_rejects_lost_prior_refs_without_omission_reason():
    source_cue, source = _semantic_fixture()
    previous = [RetainedFragment(text="응", refs=["c1:t2:a1"], speaker="b")]
    raw = translate._SemanticAttributionDraft(fragments=[
        RetainedFragment(text="준비됐어?", refs=["c1:t1:a1"], speaker="a"),
    ])
    with pytest.raises(ValueError, match="lost previous coverage"):
        translate._validate_reattribution(raw, source, [source_cue], "준비됐어?", previous)


def test_semantic_target_allows_cross_cue_split_merge_without_renaming_sources():
    cues = [cue("c1", 0, 2000, "Go now."), cue("c2", 2200, 4000, "Go.")]
    source = SemanticSource(scene_id="s1", acts=[
        SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=7, text="Go now.", speaker="a", meaning="go immediately"),
        SourceAct(ref="c2:t1:a1", cue_id="c2", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=3, text="Go.", speaker="a", meaning="repeat imperative"),
    ])
    draft = SemanticDraft(scene_id="s1", cues=[
        SemanticDraftCue(cue_id="c1", fragments=[
            RetainedFragment(text="지금", refs=["c1:t1:a1"], speaker="a"),
        ]),
        SemanticDraftCue(cue_id="c2", fragments=[
            RetainedFragment(text="가", refs=["c1:t1:a1", "c2:t1:a1"], speaker="a"),
        ]),
    ])
    output, lineage = translate._semantic_target(
        source, draft, Scene(id="s1", start_ms=0, end_ms=4000, cue_ids=["c1", "c2"]),
        cues, "codex/gpt-6.1-sol",
    )
    assert [row.cue_id for row in output] == ["c1", "c2"]
    assert [row.ko for row in output] == ["지금", "가"]
    assert lineage.fragments_by_cue["c2"][0].refs == ["c1:t1:a1", "c2:t1:a1"]
    assert [act.cue_id for act in lineage.source.acts] == ["c1", "c2"]
    assert lineage.structural_valid
    assert lineage.semantic_assessment == "unassessed"


def test_semantic_reattribution_empty_output_needs_explicit_omissions_and_no_fake_fragment():
    source_cue, source = _semantic_fixture(essential=False)
    previous = [RetainedFragment(text="응", refs=["c1:t2:a1"], speaker="b")]
    raw = translate._SemanticAttributionDraft(
        fragments=[], omissions=[OmittedAct(ref="c1:t2:a1", reason="declared incidental acknowledgment")],
    )
    fragments, omissions = translate._validate_reattribution(raw, source, [source_cue], "", previous)
    assert fragments == []
    assert [record.ref for record in omissions] == ["c1:t2:a1"]
    with pytest.raises(ValueError, match="lost previous coverage"):
        translate._validate_reattribution(
            translate._SemanticAttributionDraft(fragments=[]), source, [source_cue], "", previous,
        )


def test_semantic_source_spans_preserve_newlines_and_unicode_codepoint_offsets():
    source_cue = cue("c1", 0, 3000, "Wait…\nI'm ready 🙂")
    anchors = translate._semantic_span_anchors([source_cue])["c1"]
    assert [item["text"] for item in anchors] == ["Wait…", "I'm", "ready", "🙂"]
    for item in anchors:
        assert source_cue.text[item["start_char"]:item["end_char"]] == item["text"]
    source = SemanticSource(scene_id="s1", acts=[
        SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=len(source_cue.text), text=source_cue.text,
                  speaker=None, meaning="one possible continuous statement", essential=None),
    ])
    normalized = perception.validate_semantic_source(source, [source_cue])
    assert normalized.acts[0].text == "Wait…\nI'm ready 🙂"
    assert normalized.acts[0].turn_ordinal == 1  # a newline is not a second voice
    source.acts[0].text = "Wait… / I'm ready 🙂"
    with pytest.raises(ValueError, match="span text mismatch"):
        perception.validate_semantic_source(source, [source_cue])


def test_semantic_reattribution_recovers_retained_voice_partitions_from_flat_output():
    source_cue, source = _semantic_fixture(essential=True)
    previous = [RetainedFragment(
        text="준비됐어? 응", attribution="unverified",
        prior_refs=["c1:t1:a1", "c1:t2:a1"],
    )]
    raw = translate._SemanticAttributionDraft(fragments=[
        RetainedFragment(text="준비됐어?", refs=["c1:t1:a1"], speaker="a"),
        RetainedFragment(text="응", refs=["c1:t2:a1"], speaker="b"),
    ])
    fragments, omissions = translate._validate_reattribution(
        raw, source, [source_cue], "준비됐어? 응", previous,
    )
    assert [fragment.speaker for fragment in fragments] == ["a", "b"]
    assert [fragment.attribution for fragment in fragments] == ["declared", "declared"]
    lineage = perception.validate_semantic_lineage(SemanticLineage(
        source=source, fragments_by_cue={"c1": fragments}, omissions=omissions,
    ), [source_cue], [TranslatedCue(cue_id="c1", ko="준비됐어? 응")])
    assert lineage.structural_valid
    assert lineage.semantic_assessment == "unassessed"
    for changed in ("준비됐어? 아니", "응 준비됐어?", "준비됐어 응"):
        with pytest.raises(ValueError, match="changed actual Korean"):
            translate._validate_reattribution(raw, source, [source_cue], changed, previous)


@pytest.mark.parametrize("plot_relevant", [False, True])
def test_semantic_target_preserves_existing_incidental_and_plot_relevant_lyric_guard(plot_relevant):
    source_cue = cue("c1", 0, 2500, "A song.")
    source_cue.lyric = True
    source = SemanticSource(scene_id="s1", acts=[
        SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=7, text="A song.", speaker=None,
                  meaning="song content", essential=None),
    ])
    draft = SemanticDraft(scene_id="s1", cues=[SemanticDraftCue(
        cue_id="c1", needs_media="video", media_reason="model request",
        fragments=[RetainedFragment(text="음악", refs=["c1:t1:a1"])],
    )])
    observations = [CueObservation(cue_id="c1", lyric_plot_relevant=True)] if plot_relevant else []
    output, lineage = translate._semantic_target(
        source, draft, Scene(id="s1", start_ms=0, end_ms=2500, cue_ids=["c1"]),
        [source_cue], "codex/gpt-6.1-sol", observations,
    )
    assert output[0].ko == ("음악" if plot_relevant else "")
    assert lineage.source.acts[0].essential is None
    assert lineage.semantic_assessment == "unassessed"
    if plot_relevant:
        assert lineage.fragments_by_cue["c1"][0].refs == ["c1:t1:a1"]
        assert lineage.omissions == []
    else:
        assert lineage.fragments_by_cue["c1"] == []
        assert output[0].needs_media == MediaNeed.NONE
        assert output[0].media_reason is None
        assert lineage.omissions[0].ref == "c1:t1:a1"
        assert lineage.omissions[0].reason.startswith("pipeline policy: incidental lyric")
    assert lineage.structural_valid


@pytest.mark.parametrize("phantom_omission", [False, True])
def test_semantic_target_preserves_normalized_empty_sound_only_guard_without_fake_source_act(phantom_omission):
    source_cue = cue("c1", 0, 2000, "")
    source = SemanticSource(scene_id="s1", acts=[])
    draft = SemanticDraft(scene_id="s1", cues=[SemanticDraftCue(
        cue_id="c1", needs_media="audio", media_reason="model request",
        fragments=[RetainedFragment(text="음악", refs=[])],
    )], omissions=[OmittedAct(ref="c1:t1:a1", reason="sound-only")] if phantom_omission else [])
    output, lineage = translate._semantic_target(
        source, draft, Scene(id="s1", start_ms=0, end_ms=2000, cue_ids=["c1"]),
        [source_cue], "codex/gpt-6.1-sol",
    )
    assert output[0].ko == ""
    assert output[0].needs_media == MediaNeed.NONE
    assert lineage.source.acts == []
    assert lineage.fragments_by_cue == {"c1": []}
    assert lineage.omissions == []
    assert lineage.structural_valid


def test_semantic_lyric_display_marker_does_not_enter_fragment_lexical_attribution():
    source_cue, source = _semantic_fixture()
    previous = [RetainedFragment(text="준비됐어?", refs=["c1:t1:a1"], speaker="a")]
    raw = translate._SemanticAttributionDraft(fragments=[
        RetainedFragment(text="♪ 준비됐어?", refs=["c1:t1:a1"], speaker="a"),
    ])
    fragments, _ = translate._validate_reattribution(
        raw, source, [source_cue], "♪ 준비됐어?", previous,
    )
    assert fragments[0].text == "준비됐어?"
    assert fragments[0].speaker == "a"
    assert fragments[0].attribution == "declared"
    assert perception.semantic_text_identity("♪ 준비됐어?") == perception.semantic_text_identity("준비됐어?")
    assert perception.semantic_lexical_text("♪ - 준비됐어?\n- ♪ 응") == "준비됐어?\n응"


def _semantic_retry_client(responder):
    client = FakeGemini(responder)
    client.settings = Settings(
        model_main="gemini-3.8-flash", model_light="gemini-3.8-flash",
        model_translate="gemini-3.8-flash", model_text="gemini-3.8-flash",
    )
    return client


def _semantic_retry_runner(tmp_path, *, failing_stage=None, failures=0, invalid_fragment=None):
    source_cue, source = _semantic_fixture(essential=True)
    draft = SemanticDraft(scene_id="s1", cues=[SemanticDraftCue(cue_id="c1", fragments=[
        RetainedFragment(text="준비됐어?", refs=["c1:t1:a1"], speaker="a"),
        RetainedFragment(text="응", refs=["c1:t2:a1"], speaker="b"),
    ])])
    counts = {"extract": 0, "write": 0}

    def respond(call):
        if call["schema"] is SemanticSource:
            counts["extract"] += 1
            result = source.model_copy(deep=True)
            if failing_stage == "extract" and counts["extract"] <= failures:
                result.relations[0].refs = ["c1:t1:a1", "c1:t1:a1"]
            return result
        assert call["schema"] is SemanticDraft
        counts["write"] += 1
        result = draft.model_copy(deep=True)
        if failing_stage == "write" and counts["write"] <= failures:
            if invalid_fragment is None:
                result.cues[0].fragments.pop()  # an actual retained/omitted coverage defect
            else:
                result.cues[0].fragments[0] = invalid_fragment.model_copy(deep=True)
        return result

    client = _semantic_retry_client(respond)
    checkpoint = Checkpoints(ws(tmp_path), "translate")
    plan = resolve_route("gemini-3.8-flash", "semantic-clip")
    target_scene = Scene(id="s1", start_ms=0, end_ms=2500, cue_ids=["c1"])
    clips = [translate.Clip(file=PROXY, start_s=0.0, end_s=2.5)]

    def run():
        return translate._semantic_generation(
            client, plan, target_scene, [source_cue], "Scene s1. Frozen source/context.",
            translate._system("lite"), clips, [],
            checkpoint=checkpoint, key="s1:av", inputs_hash="frozen-input-hash",
        )

    return client, run


@pytest.mark.parametrize("failing_stage,error", [
    ("extract", "invalid semantic relation refs: question_answer "
     "(['c1:t1:a1', 'c1:t1:a1']; must be unique exact act.ref values from acts)"),
    ("write", "semantic source acts have neither retained nor omitted records: ['c1:t2:a1'] "
     "(cite each ref in the fragment.refs of the cue displaying its meaning, "
     "or list it in omissions[] with a non-empty reason)"),
])
def test_semantic_structural_retry_recovers_without_changing_checkpoint_hashes(tmp_path, failing_stage, error):
    baseline_client, baseline_run = _semantic_retry_runner(tmp_path / "baseline")
    baseline = baseline_run()
    client, run = _semantic_retry_runner(tmp_path / "retry", failing_stage=failing_stage, failures=1)
    recovered = run()
    assert recovered == baseline
    assert recovered.provenance.extract_hash == baseline.provenance.extract_hash
    assert recovered.provenance.write_hash == baseline.provenance.write_hash
    schema = SemanticSource if failing_stage == "extract" else SemanticDraft
    attempts = [call for call in client.calls if call["schema"] is schema]
    original = next(call["prompt"] for call in baseline_client.calls if call["schema"] is schema)
    assert len(attempts) == 2
    assert attempts[0]["prompt"] == original
    assert attempts[1]["prompt"] == (
        original + f"\n\n## PREVIOUS ATTEMPT STRUCTURAL VALIDATION ERROR\n{error}\n"
        "Fix this structural validation error and return valid JSON."
    )
    calls_before_resume = len(client.calls)
    assert run() == recovered
    assert len(client.calls) == calls_before_resume  # successful result uses the original checkpoint key/hash


@pytest.mark.parametrize("failing_stage,error", [
    ("extract", "invalid semantic relation refs: question_answer"),
    ("write", "semantic source acts have neither retained nor omitted records"),
])
def test_semantic_structural_retry_fails_closed_after_three_attempts(tmp_path, monkeypatch, failing_stage, error):
    client, run = _semantic_retry_runner(tmp_path, failing_stage=failing_stage, failures=3)
    validator_owner = perception if failing_stage == "extract" else translate
    validator_name = "validate_semantic_source" if failing_stage == "extract" else "_semantic_target"
    validator = getattr(validator_owner, validator_name)
    failures = []

    def record_actual_failure(*args, **kwargs):
        try:
            return validator(*args, **kwargs)
        except ValueError as exc:
            failures.append(exc)
            raise

    monkeypatch.setattr(validator_owner, validator_name, record_actual_failure)
    with pytest.raises(ValueError, match=error) as caught:
        run()
    schema = SemanticSource if failing_stage == "extract" else SemanticDraft
    assert len([call for call in client.calls if call["schema"] is schema]) == 3
    assert len(failures) == 3
    assert caught.value is failures[-1]  # preserve the final validator exception, not a replacement/fallback


def _semantic_reattribution_retry_runner(*, failures=0):
    source_cue, source = _semantic_fixture(essential=True)
    previous = [RetainedFragment(
        text="준비됐어? 응", attribution="unverified", prior_refs=["c1:t1:a1", "c1:t2:a1"],
    )]
    count = 0

    def respond(call):
        nonlocal count
        count += 1
        return translate._SemanticAttributionDraft(fragments=[
            RetainedFragment(
                text="다른 질문" if count <= failures else "준비됐어?",
                refs=["c1:t1:a1"], speaker="a",
            ),
            RetainedFragment(text="응", refs=["c1:t2:a1"], speaker="b"),
        ])

    client = _semantic_retry_client(respond)

    def run():
        return translate.reattribute_semantic_fragments(
            client, source, [source_cue], "준비됐어? 응", previous, stage="s5:semantic-reattribute",
        )

    return client, run


def test_semantic_reattribution_structural_retry_preserves_input_hash():
    baseline_client, baseline_run = _semantic_reattribution_retry_runner()
    baseline = baseline_run()
    client, run = _semantic_reattribution_retry_runner(failures=1)
    recovered = run()
    assert recovered == baseline
    assert recovered.provenance.input_hash == baseline.provenance.input_hash
    assert len(client.calls) == 2
    assert client.calls[0]["prompt"] == baseline_client.calls[0]["prompt"]
    assert client.calls[1]["prompt"] == (
        client.calls[0]["prompt"]
        + "\n\n## PREVIOUS ATTEMPT STRUCTURAL VALIDATION ERROR\n"
        "semantic reattribution changed actual Korean text or order\n"
        "Fix this structural validation error and return valid JSON."
    )


def test_semantic_reattribution_structural_retry_raises_last_error_after_three_attempts(monkeypatch):
    client, run = _semantic_reattribution_retry_runner(failures=3)
    validator = translate._validate_reattribution
    failures = []

    def record_actual_failure(*args, **kwargs):
        try:
            return validator(*args, **kwargs)
        except ValueError as exc:
            failures.append(exc)
            raise

    monkeypatch.setattr(translate, "_validate_reattribution", record_actual_failure)
    with pytest.raises(ValueError, match="changed actual Korean") as caught:
        run()
    assert len(client.calls) == 3
    assert len(failures) == 3
    assert caught.value is failures[-1]


def _semantic_multiline_fixture():
    source_cue = cue("c1", 0, 5000, "There is a reason for this trip.")
    source = SemanticSource(scene_id="s1", acts=[
        SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=len(source_cue.text), text=source_cue.text,
                  speaker="speaker", meaning="explains a reason for the trip"),
    ])
    draft = SemanticDraft(scene_id="s1", cues=[SemanticDraftCue(cue_id="c1", fragments=[
        RetainedFragment(text="오늘 이 택시에 탄 건\n이유가 있나 봐",
                         refs=["c1:t1:a1"], speaker=" speaker "),
    ])])
    target_scene = Scene(id="s1", start_ms=0, end_ms=5000, cue_ids=["c1"])
    return source_cue, source, draft, target_scene


def test_semantic_single_voice_multiline_fragment_expands_without_dialogue_dashes():
    source_cue, source, draft, target_scene = _semantic_multiline_fixture()
    output, lineage = translate._semantic_target(
        source, draft, target_scene, [source_cue], "codex/gpt-6.1-sol",
    )
    assert output[0].ko == "오늘 이 택시에 탄 건\n이유가 있나 봐"
    fragments = lineage.fragments_by_cue["c1"]
    assert [fragment.text for fragment in fragments] == ["오늘 이 택시에 탄 건", "이유가 있나 봐"]
    assert [fragment.speaker for fragment in fragments] == ["speaker", "speaker"]
    assert [fragment.refs for fragment in fragments] == [["c1:t1:a1"], ["c1:t1:a1"]]
    assert all(fragment.attribution == "declared" and fragment.prior_refs == [] for fragment in fragments)
    assert lineage.structural_valid and lineage.semantic_assessment == "unassessed"


@pytest.mark.parametrize("text", ["- 첫 줄\n둘째 줄", "첫 줄\n  - 둘째 줄"])
def test_semantic_multiline_fragment_rejects_display_dash_on_any_line(text):
    source_cue, source, draft, target_scene = _semantic_multiline_fixture()
    draft.cues[0].fragments[0].text = text
    with pytest.raises(ValueError) as caught:
        translate._semantic_target(source, draft, target_scene, [source_cue], "codex/gpt-6.1-sol")
    assert str(caught.value) == (
        "semantic writer inserted a display dash in c1: remove leading '- ' from fragment.text"
    )


def test_semantic_merged_empty_cue_discards_ref_free_dummy_fragment():
    source_cue, source, draft, target_scene = _semantic_multiline_fixture()
    second = cue("c2", 5100, 6500, "Please.")
    source.acts.append(SourceAct(
        ref="c2:t1:a1", cue_id="c2", turn_ordinal=1, act_ordinal=1,
        start_char=0, end_char=len(second.text), text=second.text,
        speaker="speaker", meaning="request politeness",
    ))
    draft.cues[0].fragments[0].refs.append("c2:t1:a1")
    draft.cues.append(SemanticDraftCue(
        cue_id="c2", fragments=[RetainedFragment(text="", refs=[], speaker=None)],
    ))
    target_scene.cue_ids.append("c2")
    target_scene.end_ms = second.end_ms
    output, lineage = translate._semantic_target(
        source, draft, target_scene, [source_cue, second], "codex/gpt-6.1-sol",
    )
    assert output[1].ko == ""
    assert lineage.fragments_by_cue["c2"] == []
    assert lineage.fragments_by_cue["c1"][0].refs == ["c1:t1:a1", "c2:t1:a1"]
    assert lineage.omissions == []
    assert lineage.structural_valid


@pytest.mark.parametrize("fragment,error", [
    (
        RetainedFragment(text="준비됐어?\n  - 응", refs=["c1:t1:a1"], speaker="a"),
        "semantic writer inserted a display dash in c1: remove leading '- ' from fragment.text",
    ),
    (
        RetainedFragment(text="", refs=["c1:t1:a1"], speaker="a"),
        "empty fragment.text in c1 carries refs ['c1:t1:a1']: move refs to the cue displaying "
        "the merged Korean or to omissions[], and use fragments=[] for empty cues",
    ),
    (
        RetainedFragment(text="준비됐어?", refs=[], speaker="a"),
        "non-empty fragment in c1 has empty refs=[]: cite at least one source act.ref from source.acts",
    ),
    (
        RetainedFragment(text="준비됐어?", refs=["c1:t1:a1"], speaker="a",
                         attribution="unverified", prior_refs=["c1:t1:a1"]),
        "writer fragment in c1 must use attribution='declared' and prior_refs=[]",
    ),
])
def test_semantic_writer_retry_receives_specific_fragment_error_without_hash_change(tmp_path, fragment, error):
    baseline_client, baseline_run = _semantic_retry_runner(tmp_path / "baseline")
    baseline = baseline_run()
    contract_hash = translate.semantic_contract_hash()
    client, run = _semantic_retry_runner(
        tmp_path / "retry", failing_stage="write", failures=1, invalid_fragment=fragment,
    )
    recovered = run()
    assert recovered == baseline
    assert recovered.provenance.extract_hash == baseline.provenance.extract_hash
    assert recovered.provenance.write_hash == baseline.provenance.write_hash
    assert translate.semantic_contract_hash() == contract_hash
    baseline_prompt = next(call["prompt"] for call in baseline_client.calls if call["schema"] is SemanticDraft)
    attempts = [call for call in client.calls if call["schema"] is SemanticDraft]
    assert len(attempts) == 2
    assert attempts[0]["prompt"] == baseline_prompt
    assert attempts[1]["prompt"] == (
        baseline_prompt + f"\n\n## PREVIOUS ATTEMPT STRUCTURAL VALIDATION ERROR\n{error}\n"
        "Fix this structural validation error and return valid JSON."
    )


def test_semantic_missing_refs_diagnostic_identifies_exact_source_act():
    source_cue, source = _semantic_fixture()
    lineage = SemanticLineage(source=source, fragments_by_cue={"c1": [
        RetainedFragment(text="준비됐어?", refs=["c1:t1:a1"], speaker="a"),
    ]})
    with pytest.raises(ValueError, match="semantic source acts have neither retained nor omitted records") as caught:
        perception.validate_semantic_lineage(lineage, [source_cue])
    assert str(caught.value) == (
        "semantic source acts have neither retained nor omitted records: ['c1:t2:a1'] "
        "(cite each ref in the fragment.refs of the cue displaying its meaning, "
        "or list it in omissions[] with a non-empty reason)"
    )


def test_semantic_invalid_retained_refs_diagnostic_lists_attempted_refs():
    source_cue, source = _semantic_fixture()
    lineage = SemanticLineage(source=source, fragments_by_cue={"c1": [
        RetainedFragment(text="준비됐어?", refs=["c1:t1:a1", "c1:t1:a1"], speaker="a"),
    ]})
    with pytest.raises(ValueError, match="invalid retained refs: c1") as caught:
        perception.validate_semantic_lineage(lineage, [source_cue])
    assert str(caught.value) == (
        "invalid retained refs: c1 "
        "(['c1:t1:a1', 'c1:t1:a1']; must be unique exact act.ref values from source.acts)"
    )


def test_semantic_retained_omitted_overlap_diagnostic_lists_sorted_exact_refs():
    source_cue, source = _semantic_fixture()
    lineage = SemanticLineage(source=source, fragments_by_cue={"c1": [
        RetainedFragment(text="준비됐어? 응", refs=["c1:t2:a1", "c1:t1:a1"]),
    ]}, omissions=[
        OmittedAct(ref="c1:t2:a1", reason="declared omitted"),
        OmittedAct(ref="c1:t1:a1", reason="declared omitted"),
    ])
    with pytest.raises(ValueError, match="semantic acts declared both retained and omitted") as caught:
        perception.validate_semantic_lineage(lineage, [source_cue])
    assert str(caught.value) == (
        "semantic acts declared both retained and omitted: ['c1:t1:a1', 'c1:t2:a1']"
    )


@pytest.mark.parametrize("unknown_ref", ["c1:t3:a1", "unknown:t1:a1"])
def test_semantic_target_rejects_unknown_omission_refs_outside_untranslated_cues(unknown_ref):
    source_cue, source = _semantic_fixture()
    draft = SemanticDraft(scene_id="s1", cues=[SemanticDraftCue(cue_id="c1", fragments=[
        RetainedFragment(text="준비됐어?", refs=["c1:t1:a1"], speaker="a"),
        RetainedFragment(text="응", refs=["c1:t2:a1"], speaker="b"),
    ])], omissions=[OmittedAct(ref=unknown_ref, reason="model declared omitted")])
    with pytest.raises(ValueError, match="invalid semantic omission") as caught:
        translate._semantic_target(
            source, draft, Scene(id="s1", start_ms=0, end_ms=2500, cue_ids=["c1"]),
            [source_cue], "codex/gpt-6.1-sol",
        )
    assert str(caught.value) == (
        f"invalid semantic omission: {unknown_ref} "
        "(must be a unique existing act.ref in source.acts with a non-empty reason; "
        "sound-only cues without source.acts need no omission record)"
    )


def test_semantic_sound_only_omission_filter_preserves_duplicate_ref_rejection():
    source_cue = cue("c1", 0, 2000, "")
    source = SemanticSource(scene_id="s1", acts=[])
    draft = SemanticDraft(scene_id="s1", cues=[SemanticDraftCue(cue_id="c1", fragments=[])],
                          omissions=[OmittedAct(ref="c1:t1:a1", reason="sound-only"),
                                     OmittedAct(ref="c1:t1:a1", reason="duplicate sound-only")])
    with pytest.raises(ValueError, match="duplicate semantic writer omission refs"):
        translate._semantic_target(
            source, draft, Scene(id="s1", start_ms=0, end_ms=2000, cue_ids=["c1"]),
            [source_cue], "codex/gpt-6.1-sol",
        )


def _semantic_continuation_fixture():
    cues = [cue("c1", 0, 2000, "Let's go."), cue("c2", 2200, 4200, "Now.")]
    source = SemanticSource(scene_id="s1", acts=[
        SourceAct(ref=f"{item.id}:t1:a1", cue_id=item.id, turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=len(item.text), text=item.text,
                  speaker="speaker", meaning="departure instruction")
        for item in cues
    ])
    draft = SemanticDraft(scene_id="s1", cues=[
        SemanticDraftCue(cue_id="c1", fragments=[
            RetainedFragment(text="가자", refs=["c1:t1:a1"], speaker="speaker"),
        ]),
        SemanticDraftCue(cue_id="c2", fragments=[
            RetainedFragment(text="지금", refs=["c2:t1:a1"], prior_refs=["c1:t1:a1"],
                             speaker="speaker", attribution="declared"),
        ]),
    ])
    target_scene = Scene(id="s1", start_ms=0, end_ms=4200, cue_ids=["c1", "c2"])
    return cues, source, draft, target_scene


def test_semantic_declared_prior_context_normalizes_without_redundant_cross_cue_refs():
    cues, source, draft, target_scene = _semantic_continuation_fixture()
    output, lineage = translate._semantic_target(source, draft, target_scene, cues, "codex/gpt-6.1-sol")
    fragment = lineage.fragments_by_cue["c2"][0]
    assert fragment.refs == ["c2:t1:a1"]
    assert fragment.prior_refs == []
    assert fragment.attribution == "declared"
    assert [item.ko for item in output] == ["가자", "지금"]
    assert draft.cues[1].fragments[0].prior_refs == ["c1:t1:a1"]  # preserve the raw draft
    assert lineage.structural_valid and lineage.semantic_assessment == "unassessed"


def test_semantic_declared_prior_refs_recover_unclaimed_merged_source_coverage_in_order():
    cues, source, draft, target_scene = _semantic_continuation_fixture()
    draft.cues[0].fragments = []
    draft.cues[1].fragments[0].prior_refs = ["c1:t1:a1", "c1:t1:a1"]
    _, lineage = translate._semantic_target(source, draft, target_scene, cues, "codex/gpt-6.1-sol")
    fragment = lineage.fragments_by_cue["c2"][0]
    assert fragment.refs == ["c2:t1:a1", "c1:t1:a1"]
    assert fragment.prior_refs == []
    assert lineage.fragments_by_cue["c1"] == []
    assert lineage.omissions == []
    assert lineage.structural_valid


def test_semantic_declared_prior_refs_preserve_explicit_source_omissions():
    cues, source, draft, target_scene = _semantic_continuation_fixture()
    draft.cues[0].fragments = []
    draft.omissions = [OmittedAct(ref="c1:t1:a1", reason="declared incidental repetition")]
    _, lineage = translate._semantic_target(source, draft, target_scene, cues, "codex/gpt-6.1-sol")
    fragment = lineage.fragments_by_cue["c2"][0]
    assert fragment.refs == ["c2:t1:a1"]
    assert fragment.prior_refs == []
    assert lineage.omissions == draft.omissions
    assert lineage.structural_valid


def test_semantic_declared_empty_current_refs_can_be_filled_from_valid_prior_refs():
    cues, source, draft, target_scene = _semantic_continuation_fixture()
    draft.cues[1].fragments[0].refs = []
    draft.cues[1].fragments[0].prior_refs = ["c1:t1:a1", "c2:t1:a1", "c1:t1:a1"]
    _, lineage = translate._semantic_target(source, draft, target_scene, cues, "codex/gpt-6.1-sol")
    fragment = lineage.fragments_by_cue["c2"][0]
    assert fragment.refs == ["c1:t1:a1", "c2:t1:a1"]
    assert fragment.prior_refs == []
    assert lineage.structural_valid


def test_semantic_declared_prior_ref_belonging_to_current_cue_is_retained():
    cues, source, draft, target_scene = _semantic_continuation_fixture()
    draft.cues[1].fragments[0].refs = ["c1:t1:a1"]
    draft.cues[1].fragments[0].prior_refs = ["c2:t1:a1"]
    _, lineage = translate._semantic_target(source, draft, target_scene, cues, "codex/gpt-6.1-sol")
    fragment = lineage.fragments_by_cue["c2"][0]
    assert fragment.refs == ["c1:t1:a1", "c2:t1:a1"]
    assert fragment.prior_refs == []
    assert lineage.structural_valid


@pytest.mark.parametrize("attribution,prior_refs,error", [
    ("declared", ["unknown:t1:a1"], "invalid prior refs: c2"),
    ("unverified", ["c1:t1:a1"],
     "writer fragment in c2 must use attribution='declared' and prior_refs=[]"),
])
def test_semantic_prior_ref_normalization_still_rejects_unknown_and_unverified(attribution, prior_refs, error):
    cues, source, draft, target_scene = _semantic_continuation_fixture()
    draft.cues[1].fragments[0].attribution = attribution
    draft.cues[1].fragments[0].prior_refs = prior_refs
    with pytest.raises(ValueError) as caught:
        translate._semantic_target(source, draft, target_scene, cues, "codex/gpt-6.1-sol")
    assert str(caught.value) == error


def test_semantic_source_drops_empty_cue_acts_and_relation_refs():
    empty = Cue(id="c-empty", start_ms=0, end_ms=1000, text="", raw="(LAUGHS)")
    spoken = cue("c1", 1200, 3000, "Ready.")
    source = SemanticSource(scene_id="s1", acts=[
        SourceAct(ref="c-empty:t1:a1", cue_id="c-empty", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=1, text="x", meaning="laugh"),
        SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=6, text="Ready.", meaning="ready"),
    ], relations=[SourceRelation(
        kind="order", refs=["c-empty:t1:a1", "c1:t1:a1"], claim="then",
    )])
    normalized = perception.validate_semantic_source(source, [empty, spoken])
    assert [act.ref for act in normalized.acts] == ["c1:t1:a1"]
    assert normalized.relations == []


def test_semantic_source_repairs_whitespace_span_and_multiturn_ordinal_refs():
    source_cue = cue("c1", 0, 3000, "Ready?\nYes.")
    source = SemanticSource(scene_id="s1", acts=[
        SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=6, text="Ready?", meaning="question"),
        SourceAct(ref="c1:t2:a2", cue_id="c1", turn_ordinal=2, act_ordinal=2,
                  start_char=7, end_char=11, text="Yes.", meaning="answer"),
    ], relations=[SourceRelation(
        kind="question_answer", refs=["c1:t1:a1", "c1:t2:a2"], claim="answer",
    )])
    normalized = perception.validate_semantic_source(source, [source_cue])
    assert [(act.ref, act.start_char, act.end_char, act.text) for act in normalized.acts] == [
        ("c1:t1:a1", 0, 6, "Ready?"),
        ("c1:t2:a1", 7, 11, "Yes."),
    ]
    assert normalized.relations[0].refs == ["c1:t1:a1", "c1:t2:a1"]


def test_semantic_source_rejects_normalized_separator_and_invalid_original_ref():
    cue_with_newline = cue("c1", 0, 3000, "Ready?\nYes.")
    source = SemanticSource(scene_id="s1", acts=[
        SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=6, text="Ready?", meaning="question"),
        SourceAct(ref="c1:t2:a2", cue_id="c1", turn_ordinal=2, act_ordinal=2,
                  start_char=7, end_char=11, text="Yes. /", meaning="answer"),
    ])
    with pytest.raises(ValueError, match="semantic source span text mismatch"):
        perception.validate_semantic_source(source, [cue_with_newline])
    bad = _semantic_fixture()[1]
    bad.acts[0].ref = "invented"
    with pytest.raises(ValueError, match="invalid semantic source identity"):
        perception.validate_semantic_source(bad, [_semantic_fixture()[0]])


def test_semantic_target_expands_bare_cue_refs_and_auto_omits_empty_policy_act():
    empty = cue("c-empty", 0, 1000, "Uh...")
    source = SemanticSource(scene_id="s1", acts=[
        SourceAct(ref="c-empty:t1:a1", cue_id="c-empty", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=4, text="Uh...", meaning="interjection",
                  essential=None),
    ])
    draft = SemanticDraft(
        scene_id="s1",
        cues=[SemanticDraftCue(cue_id="c-empty", fragments=[])],
        omissions=[OmittedAct(ref="c-empty", reason="empty")],
    )
    output, lineage = translate._semantic_target(
        source, draft, Scene(id="s1", start_ms=0, end_ms=1000, cue_ids=["c-empty"]),
        [empty], "codex/gpt-6.1-sol",
    )
    assert output[0].ko == ""
    assert [item.ref for item in lineage.omissions] == ["c-empty:t1:a1"]
    assert lineage.omissions[0].reason == "empty"
    assert lineage.structural_valid


def test_semantic_target_auto_omits_declared_nonessential_empty_act():
    source_cue = cue("c1", 0, 1000, "Uh...")
    source = SemanticSource(scene_id="s1", acts=[
        SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=4, text="Uh...", meaning="incidental",
                  essential=False),
    ])
    draft = SemanticDraft(scene_id="s1", cues=[SemanticDraftCue(cue_id="c1", fragments=[])])
    _, lineage = translate._semantic_target(
        source, draft, Scene(id="s1", start_ms=0, end_ms=1000, cue_ids=["c1"]),
        [source_cue], "codex/gpt-6.1-sol",
    )
    assert lineage.omissions[0].reason == (
        "pipeline policy: non-essential or sound/interjection cue left empty"
    )


def test_semantic_target_keeps_unknown_essential_empty_act_fail_closed():
    source_cue = cue("c1", 0, 1000, "A complete answer.")
    source = SemanticSource(scene_id="s1", acts=[
        SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=len(source_cue.text), text=source_cue.text,
                  meaning="unknown importance", essential=None),
    ])
    draft = SemanticDraft(scene_id="s1", cues=[SemanticDraftCue(cue_id="c1", fragments=[])])
    with pytest.raises(ValueError, match="semantic source acts have neither retained nor omitted"):
        translate._semantic_target(
            source, draft, Scene(id="s1", start_ms=0, end_ms=1000, cue_ids=["c1"]),
            [source_cue], "codex/gpt-6.1-sol",
        )


def test_semantic_target_keeps_unknown_bare_ref_strict():
    source_cue = cue("c1", 0, 1000, "A line.")
    source = SemanticSource(scene_id="s1", acts=[
        SourceAct(ref="c1:t1:a1", cue_id="c1", turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=7, text="A line.", meaning="line"),
    ])
    draft = SemanticDraft(
        scene_id="s1",
        cues=[SemanticDraftCue(cue_id="c1", fragments=[])],
        omissions=[OmittedAct(ref="unknown", reason="bad")],
    )
    with pytest.raises(ValueError, match="invalid semantic omission"):
        translate._semantic_target(
            source, draft, Scene(id="s1", start_ms=0, end_ms=1000, cue_ids=["c1"]),
            [source_cue], "codex/gpt-6.1-sol",
        )


def test_semantic_media_safety_block_retries_same_media_with_sanitized_prompt_and_system(tmp_path):
    source_cue, source = _semantic_fixture(essential=True)
    draft = SemanticDraft(scene_id="s1", cues=[SemanticDraftCue(cue_id="c1", fragments=[
        RetainedFragment(text="준비됐어?", refs=["c1:t1:a1"], speaker="a"),
        RetainedFragment(text="응", refs=["c1:t2:a1"], speaker="b"),
    ])])
    calls: list[dict] = []
    state = {"writes": 0}

    def respond(call):
        calls.append(call)
        if call["schema"] is SemanticDraft:
            state["writes"] += 1
            if state["writes"] == 1:
                invalid = draft.model_copy(deep=True)
                invalid.cues[0].fragments.pop()
                return invalid
            if state["writes"] == 2:
                raise GeminiBlocked(call["stage"], "PROHIBITED_CONTENT")
        return source.model_copy(deep=True) if call["schema"] is SemanticSource else draft.model_copy(deep=True)

    client = _semantic_retry_client(respond)
    plan = resolve_route("gemini-3.8-flash", "semantic-clip")
    result = translate._semantic_generation(
        client, plan, Scene(id="s1", start_ms=0, end_ms=2500, cue_ids=["c1"]),
        [source_cue], "Scene s1. Frozen source/context.", translate._system("lite"),
        [translate.Clip(file=PROXY, start_s=0.0, end_s=2.5)], [],
        checkpoint=Checkpoints(ws(tmp_path), "translate"), key="s1:av",
        inputs_hash="frozen-input-hash",
    )
    assert result.degraded is False
    assert result.degradation_reason is None
    write_calls = [call for call in calls if call["schema"] is SemanticDraft]
    assert len(write_calls) == 3
    assert write_calls[0]["clips"] and write_calls[1]["clips"] and write_calls[2]["clips"]
    assert write_calls[2]["system"] == translate._semantic_safety_system("lite")
    assert '"raw":""' in write_calls[2]["prompt"]
    assert translate._SEMANTIC_SPEAKER_POLICY in write_calls[2]["system"]
    assert "fragments[].text fields" in write_calls[2]["system"]
    assert '"sound_cues":[]' in write_calls[2]["prompt"]
    assert '"meaning":"source speech act"' in write_calls[2]["prompt"]
    assert "## CUES (id | start-end" in write_calls[2]["prompt"]
    assert "## EXACT SOURCE CUES" in write_calls[2]["prompt"]
    assert "Scene s1 (00:00:00.000-00:00:02.500)." in write_calls[2]["prompt"]
    assert '"claim":"question_answer"' in write_calls[2]["prompt"]
    assert "## PREVIOUS ATTEMPT STRUCTURAL VALIDATION ERROR" in write_calls[2]["prompt"]
    assert "['c1:t2:a1']" in write_calls[2]["prompt"]
    assert "[Media attachment was safety-blocked" not in write_calls[2]["prompt"]


def test_detect_syntactic_spans_and_sov_rewrite_gates():
    c1 = Cue(id="c1172", index=1, start_ms=1000, end_ms=3000, text="You couldn't even save a relationship…", raw="", speaker_hint="Wade")
    c2 = Cue(id="c1173", index=2, start_ms=3080, end_ms=5000, text="with a goddamn stripper!", raw="", speaker_hint="Wade")
    c3 = Cue(id="c0920", index=3, start_ms=7000, end_ms=8500, text="Said I was...", raw="", speaker_hint="Beck")
    c4 = Cue(id="c0921", index=4, start_ms=8600, end_ms=10000, text="unstable.", raw="", speaker_hint="Beck")
    spans = translate.detect_syntactic_spans([c1, c2, c3, c4], {})
    assert spans["c1172"].policy == "REVERSE_SOV_PERMITTED"
    assert spans["c1173"].policy == "REVERSE_SOV_PERMITTED"
    assert spans["c0920"].policy == "DRAMATIC_REVEAL_KEEP_ORDER"
    assert spans["c0921"].policy == "DRAMATIC_REVEAL_KEEP_ORDER"
    assert review._is_svo_premature_verb_close("연애 하나 못 지킨 놈이!", "그것도 망할 스트리퍼랑!")
    assert not review._is_svo_premature_verb_close("스트리퍼 따위와의\n관계 하나", "못 지켜내는 새끼가!")
    assert review._is_valid_sov_rewrite(
        "연애 하나 못 지킨 놈이!",
        "그것도 망할 스트리퍼랑!",
        "스트리퍼 따위와의\n관계 하나",
        "못 지켜내는 새끼가!",
        "hae",
        "hae",
    )
    assert not review._is_valid_sov_rewrite(
        "연애 하나 못 지킨 놈이!",
        "그것도 망할 스트리퍼랑!",
        "그깟 관계 하나도\n제대로 못 지켰잖아",
        "빌어먹을 스트리퍼랑도!",
        "hae",
        "hae",
    )


def test_detect_pragmatic_risk_cues_and_scene_situation_card():
    c_sleep = Cue(id="c0767", index=1, start_ms=1000, end_ms=2000, text="How long was I asleep?", raw="", speaker_hint="Deadpool")
    c_part = Cue(id="c0768", index=2, start_ms=2100, end_ms=3500, text="Not all of you was asleep.", raw="", speaker_hint="Wolverine")
    c_deix = Cue(id="c1456", index=3, start_ms=4000, end_ms=5500, text="Put her down over here.", raw="", speaker_hint="Ajax")
    c_idiom = Cue(id="c0376", index=4, start_ms=6000, end_ms=7500, text="Oh, I smell what you're steppin' in.", raw="", speaker_hint="Deadpool")
    c_imp = Cue(id="c0403", index=5, start_ms=8000, end_ms=9500, text="Where else do you put one out?", raw="", speaker_hint="Wade")
    risks = translate.detect_pragmatic_risk_cues([c_sleep, c_part, c_deix, c_idiom, c_imp])
    assert "c0767" not in risks
    assert risks["c0768"] == "body_or_partitive_metonymy"
    assert risks["c1456"] == "visual_action_deixis"
    assert risks["c0376"] == "idiom_or_slang"
    assert risks["c0403"] == "conversational_implicature"


# ---------------------------------------------------------------- reference-free quality


def _write_quality_track(tmp_path, events, cues=None, video_id="sample"):
    """Write real SRT tags, so tests exercise the plain-text loader boundary."""
    from substage.store import save_model

    def timestamp(ms):
        seconds, millis = divmod(ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{video_id}.ko.srt"
    path.write_text("\n\n".join(
        f"{index}\n{timestamp(start)} --> {timestamp(end)}\n{text}"
        for index, (start, end, text) in enumerate(events, 1)
    ) + "\n", encoding="utf-8")
    if cues is not None:
        save_model(tmp_path / "cues.json", cues)
    return path


def test_quality_reference_free_perfect_and_optional_reference(tmp_path):
    from substage import refeval

    srt = _write_quality_track(
        tmp_path, [(1000, 4000, "안녕하세요.")],
        [cue("c1", 1000, 4000, "Hello.")],
    )
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert report["quality_score"] == 100.0
    assert report["passes_targets"] is True
    assert report["actionable_defects"] == []
    assert "reference_eval" not in report
    assert report["counts"]["spoken_cues"] == 1
    assert report["coverage"] == 1.0
    assert report["ost_pos"] == 100.0
    assert "context_map.yaml" in report["unassessed"]
    markdown = refeval.render_quality_md(report)
    assert "Reference-free" in markdown and "**PASS**" in markdown
    assert all(axis in markdown for axis in (
        "NETFLIX-FMT", "ZERO-DUPLICATE-COLLISION", "REGISTER-CONSISTENCY",
        "PRAGMATIC-ADDRESS", "OST-QUALITY", "SEMANTIC-COMPLETENESS",
    ))
    reference = tmp_path / "official.ko.srt"
    reference.write_text(srt.read_text(encoding="utf-8").replace("안녕하세요.", "잘 가요."), encoding="utf-8")
    with_ref = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, ref_path=reference, settings=Settings())
    assert set(with_ref["reference_eval"]) == {"path", "chrF++", "coverage", "level_agreement", "dash_agreement"}
    assert with_ref["reference_eval"]["chrF++"] < 100.0
    assert with_ref["quality_score"] == report["quality_score"]
    assert with_ref["passes_targets"] is True
    missing_ref = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, ref_path=tmp_path / "absent.srt", settings=Settings())
    assert "reference_eval" not in missing_ref


def test_quality_detects_1720_double_top_and_dialogue_redundancy(tmp_path):
    from substage import refeval

    srt = _write_quality_track(tmp_path, [
        (1_040_000, 1_044_000, r"{\an8}'우주의 새로운 영웅'"),
        (1_041_000, 1_045_000, r"{\an8}'새로운 영웅 등장'"),
        (1_041_000, 1_045_000, "그 마이크가 바로 트위터다"),
        (1_211_700, 1_214_800, r"{\an8}'에릭 힘턴 홀더 주니어'"),
        (1_211_944, 1_215_448, "에릭 힘턴 홀더 주니어"),
    ], [
        cue("c1", 1_041_000, 1_045_000, "That microphone is Twitter."),
        cue("c2", 1_211_944, 1_215_448, "Eric Himpton Holder, Jr."),
    ])
    assert all(r"{\an8}" not in line.text for line in refeval.load_ko(srt))
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert report["counts"]["same_channel_overlaps"] == 1
    assert report["counts"]["ost_duplicate_translations"] == 1
    assert report["counts"]["ost_dialogue_redundancies"] == 1
    assert report["collision_free_score"] == 92.5
    assert report["ost_pos"] == report["ost_emp"] == 100.0
    assert report["passes_targets"] is False
    collisions = [item for item in report["actionable_defects"] if item["stage"] == "layout/ost"]
    assert len(collisions) == 3
    assert all(item["start_ms"] >= 1_040_000 and item["other_text"] for item in collisions)
    assert "17:20" in refeval.render_quality_md(report)


@pytest.mark.parametrize("overlap,expected", [(49, 0), (50, 0), (51, 1)])
def test_quality_same_channel_requires_strict_50ms_overlap(tmp_path, overlap, expected):
    from substage import refeval

    _write_quality_track(tmp_path, [
        (1000, 4000, "첫 대사입니다."),
        (4000 - overlap, 7000, "다른 대사입니다."),
    ])
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert report["counts"]["same_channel_overlaps"] == expected


def test_quality_counts_all_nested_same_channel_pairs(tmp_path):
    from substage import refeval

    _write_quality_track(tmp_path, [
        (1000, 9000, "긴 대사입니다."), (2000, 7000, "중간 대사입니다."), (3000, 5000, "짧은 대사입니다."),
    ])
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert report["counts"]["same_channel_overlaps"] == 3


@pytest.mark.parametrize("first,second", [
    ("세 번째 기사", "네 번째 기사"),
    ("제3의 새로운 기사", "제4의 새로운 기사"),
    ("12번 출입구 폐쇄", "13번 출입구 폐쇄"),
])
def test_quality_ost_distinct_ordinals_and_digits_are_not_duplicates(tmp_path, first, second):
    from substage import refeval

    _write_quality_track(tmp_path, [
        (1000, 4000, f"'{first}'"), (5000, 8000, f"'{second}'"),
    ])
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert report["counts"]["ost_duplicate_translations"] == 0
    assert report["collision_free_score"] == 100.0


@pytest.mark.parametrize("second_start,expected", [(16_000, 1), (16_001, 0)])
def test_quality_ost_duplicate_window_includes_12000ms_gap(tmp_path, second_start, expected):
    from substage import refeval

    _write_quality_track(tmp_path, [
        (1000, 4000, "'긴급 출입 통제'"), (second_start, second_start + 3000, "'긴급 출입 통제 중'"),
    ])
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert report["counts"]["ost_duplicate_translations"] == expected


@pytest.mark.parametrize("english,expected", [
    (("I will leave now.", "I will leave now."), 0),
    (("I will leave now.", "Don't wait for me."), 1),
])
def test_quality_dialogue_echo_respects_intentional_source_repetition(tmp_path, english, expected):
    from substage import refeval

    _write_quality_track(tmp_path, [
        (1000, 4000, "이제 떠날 거예요."), (5000, 8000, "이제 떠날 거예요."),
    ], [cue("c1", 1000, 4000, english[0]), cue("c2", 5000, 8000, english[1])])
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert report["counts"]["adjacent_dialogue_echoes"] == expected


def test_quality_format_completeness_and_empty_tracks(tmp_path):
    from substage import refeval

    srt = _write_quality_track(tmp_path, [
        (1000, 2000, "아주 긴 자막입니다 " * 4 + " / 뒤\n둘째\n셋째♪"),
        (3000, 5000, "Untranslated."),
    ], [
        cue("c1", 1000, 2000, "A long subtitle."),
        cue("c2", 3000, 5000, "This is untranslated."),
        cue("c3", 6000, 8000, "Missing."),
    ])
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert report["format_score"] == 0.0
    assert report["counts"]["cpl_over"] == report["counts"]["slash_count"] == report["counts"]["over_two_lines_count"] == 1
    assert report["counts"]["missing_dialogue"] == report["counts"]["raw_music_count"] == report["counts"]["english_only_count"] == 1
    assert report["completeness_score"] == 0.0
    assert report["passes_targets"] is False
    srt.write_text("", encoding="utf-8")
    empty = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert empty["completeness_score"] == 0.0
    assert empty["passes_targets"] is False
    assert any(item["kind"] == "empty_subtitles" for item in empty["actionable_defects"])


def _write_quality_register_artifacts(tmp_path, *, shifted=False, collision=False):
    from substage.models import RegisterEdge, RegisterMap
    from substage.store import save_model

    cues = [
        cue("c1", 1000, 4000, "Please leave."),
        cue("c2", 5000, 8000, "Go now."),
        cue("c3", 9000, 12000, "Come back."),
    ]
    _write_quality_track(tmp_path, [
        (1000, 4000, "먼저 가세요."), (5000, 8000, "빨리 가."), (9000, 12000, "다시 오세요."),
    ], cues)
    characters = [
        Character(id="alex", en_name="Alex Stone", ko_name="알렉스", persona="adult",
                  aliases=["Stone"] if collision else [], voice=VoiceCard(idiolect="polite")),
        Character(id="bob", en_name="Bob Stone", ko_name="밥", persona="adult", voice=VoiceCard(idiolect="polite")),
    ]
    save_model(tmp_path / "context_map.yaml", ContextMap(synopsis="test", tone_baseline="neutral", characters=characters))
    shifts = [RegisterShift(at_cue="c2", until_cue="c3", level="hae", reason="relationship_change")] if shifted else []
    edge = RegisterEdge(
        speaker="alex", addressee="bob", relation="stranger_civil", apparent_generation="same",
        history="none", stance="civil", step=9, level="haeyo", shifts=shifts,
    )
    save_model(tmp_path / "register.yaml", RegisterMap(edges=[edge]))
    save_model(tmp_path / "scenes.json", [Scene(id="s1", start_ms=1000, end_ms=12000, cue_ids=[c.id for c in cues])])
    save_model(tmp_path / "observations.json", [
        CueObservation(cue_id=c.id, speaker="Alex Stone", addressee="Bob Stone") for c in cues
    ])
    save_model(tmp_path / "snapshots.json", [
        Snapshot(scene_id="s1", registers={"alex>bob": SceneRegister(level="haeyo" if shifted else "hae", shifts=shifts)}),
    ])
    return cues


def test_quality_register_alias_snapshot_stranger_and_scene_defects(tmp_path):
    from substage import refeval

    _write_quality_register_artifacts(tmp_path, collision=True)
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert report["counts"]["alias_collisions"] == 1
    assert report["counts"]["stranger_banmal_count"] == 1
    assert report["counts"]["scene_oscillations"] == 1
    assert report["counts"]["snap_disagree"] == 3
    assert report["register_score"] == 90.0
    assert report["passes_targets"] is False
    assert {item["kind"] for item in report["actionable_defects"]} >= {
        "alias_collision", "stranger_banmal", "scene_oscillation", "snapshot_register_disagreement",
    }


def test_quality_honors_register_shifts_and_translation_pair_metadata(tmp_path):
    from substage import refeval
    from substage.store import save_model

    cues = _write_quality_register_artifacts(tmp_path, shifted=True)
    save_model(tmp_path / "observations.json", [])
    save_model(tmp_path / "translations.json", [
        SceneTranslation(scene_id="s1", mode="text", cues=[
            TranslatedCue(cue_id=c.id, speaker="alex", addressee="bob", ko="old draft") for c in cues
        ]),
    ])
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert report["counts"]["stranger_banmal_count"] == 0
    assert report["counts"]["scene_oscillations"] == 0
    assert report["counts"]["snap_disagree"] == 0
    assert report["register_score"] == 100.0
    assert report["quality_score"] == 100.0


def test_quality_resolves_arm_artifacts_without_official_subtitles(tmp_path, monkeypatch):
    from substage import refeval

    base = tmp_path / "sample"
    arm = tmp_path / "sample__flash38"
    _write_quality_track(base, [(1000, 4000, "원본 초안입니다.")], [cue("c1", 1000, 4000, "Hello.")])
    _write_quality_track(arm, [(1000, 4000, "안녕하세요.")])
    monkeypatch.setattr(refeval, "workspace", lambda video_id: SimpleNamespace(data_dir=base))
    report = refeval.evaluate_media_quality("sample", settings=Settings())
    assert report["srt_path"] == str(arm / "sample.ko.srt")
    assert report["artifact_paths"]["cues.json"] == str(base / "cues.json")
    assert report["quality_score"] == 100.0


def test_quality_corpus_markdown_and_gating(tmp_path):
    from substage import refeval

    first, second = tmp_path / "first", tmp_path / "second"
    _write_quality_track(first, [(1000, 4000, "안녕하세요.")])
    _write_quality_track(second, [(1000, 4000, "같은 대사입니다."), (2000, 5000, "다른 대사입니다.")])
    report = refeval.evaluate_corpus_quality([
        {"video_id": "first", "ws_dir": first}, {"video_id": "second", "ws_dir": second},
    ], settings=Settings())
    assert report["summary"]["media_count"] == 2
    assert report["summary"]["passed_count"] == 1
    assert report["quality_score"] > 97.5
    assert report["passes_targets"] is False
    assert report["summary"]["quality_score"] == report["quality_score"]
    markdown = refeval.render_quality_md(report)
    assert "Subtitle corpus quality" in markdown and "first" in markdown and "second" in markdown
    assert "**FAIL**" in markdown
    with pytest.raises(ValueError, match="at least one"):
        refeval.evaluate_corpus_quality([], settings=Settings())


def test_quality_cli_json_markdown_output_and_fail_thresholds(tmp_path, capsys):
    import json

    from substage import cli

    path = _write_quality_track(tmp_path, [(1000, 4000, "안녕하세요.")])
    out = tmp_path / "report"
    args = ["quality", "sample", "--ws-dir", str(tmp_path), "--srt", str(path)]
    assert cli.main([*args, "--out", str(out), "--json", "--fail-under", "100"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["quality_score"] == 100.0
    assert json.loads((out / "quality.json").read_text(encoding="utf-8")) == report
    assert "Subtitle quality" in (out / "quality.md").read_text(encoding="utf-8")
    assert cli.main([*args, "--fail-under", "0"]) == 0
    assert "**PASS**" in capsys.readouterr().out
    _write_quality_track(tmp_path, [(1000, 4000, "첫 대사입니다."), (2000, 5000, "다른 대사입니다.")])
    assert cli.main([*args, "--fail-under", "0"]) == 1
    capsys.readouterr()
    assert cli.main([*args, "--fail-under", "100"]) == 1
    capsys.readouterr()
    assert cli.main(args) == 0
    capsys.readouterr()


def test_quality_cli_multiple_media_uses_corpus_report(tmp_path, monkeypatch, capsys):
    import json

    from substage import cli, refeval

    for video_id in ("first", "second"):
        _write_quality_track(tmp_path / video_id, [(1000, 4000, "안녕하세요.")], video_id=video_id)
    monkeypatch.setattr(refeval, "workspace", lambda video_id: SimpleNamespace(data_dir=tmp_path / video_id))
    assert cli.main(["quality", "first", "second", "--json", "--fail-under", "0"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["media_count"] == 2
    assert report["passes_targets"] is True


def test_quality_pragmatic_address_and_sync_defects_are_actionable(tmp_path):
    from substage import refeval
    from substage.store import load_model, save_model
    from substage.models import RegisterMap

    _write_quality_register_artifacts(tmp_path, shifted=True)
    rmap = load_model(tmp_path / "register.yaml", RegisterMap)
    edge = rmap.edges[0].model_copy(update={
        "relation": "own_parent", "apparent_generation": "addressee_older",
        "history": "raised_by_addressee", "step": 8, "shifts": [],
    })
    save_model(tmp_path / "register.yaml", RegisterMap(edges=[edge]))
    _write_quality_track(tmp_path, [(1000, 4000, "당신은 가세요.")], [cue("c1", 1000, 4000, "You should leave.")])
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, sync_delta_ms=0.0, settings=Settings())
    assert report["pragmatic_score"] == 0.0
    assert report["sync_err_ms"] == 248.0
    assert any(item["kind"] == "civil_dangsin" and item["stage"] == "pragmatic/address" for item in report["actionable_defects"])
    assert any(item["kind"] == "sync_error" for item in report["actionable_defects"])


def test_quality_overlapping_dialogue_echo_is_distinct_from_time_collision(tmp_path):
    from substage import refeval

    _write_quality_track(tmp_path, [
        (1000, 4000, "이제 떠날 거예요."), (3000, 6000, "이제 떠날 거예요."),
    ], [cue("c1", 1000, 3000, "I will leave now."), cue("c2", 4000, 6000, "Don't wait for me.")])
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert report["counts"]["same_channel_overlaps"] == 1
    assert report["counts"]["adjacent_dialogue_echoes"] == 1


def test_rewrap_and_quality_reject_newline_adjacent_slashes_while_keeping_fixed_tokens(tmp_path):
    from substage import refeval, validate

    assert validate.rewrap("전 남친 /\n어쩌고 하던데", 16, 14) == "전 남친\n어쩌고 하던데"
    assert validate.rewrap("내 고무 재질\n/\n딸딸이 신발?", 16, 14) == "내 고무 재질\n딸딸이 신발?"
    assert validate.rewrap("고마워할 줄도\n/ 알아야지", 16, 14) == "고마워할 줄도\n알아야지"
    assert validate.rewrap("케토롤락 주사액\n30mg/mL", 16, 14) == "케토롤락 주사액\n30mg/mL"
    _write_quality_track(tmp_path, [(1000, 3000, "전 남친 /\n어쩌고 하던데"), (4000, 6000, "케토롤락 30mg/mL")])
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert report["counts"]["slash_count"] == 1
    assert report["passes_targets"] is False


def test_compound_caption_splits_into_subsequent_silent_gap_and_quality_flags_missing_ost(tmp_path):
    from substage import refeval

    cues = [
        cue("c0076", 437_604, 440_148, "I've waited for this moment for so long."),
        cue("c0077", 441_483, 443_610, "Thank you, sir, for seeing me."),
        cue("c0078", 444_694, 446_571, "I firmly believe..."),
    ]
    obs = [
        ChunkObservations(
            chunk_id="v01", start_ms=0, end_ms=600_000,
            on_screen_text=[
                OnScreenText(
                    start_ms=440_232, end_ms=441_399,
                    text="[caption] EARTH - 616 THE SACRED TIMELINE", plot_critical=True,
                ),
            ],
        )
    ]
    ost = translate.ost_cues(obs, cues, [])
    assert [(c.start_ms, c.end_ms, c.text) for c in ost] == [
        (440_232, 441_399, "EARTH - 616"),
        (443_690, 444_654, "THE SACRED TIMELINE"),
    ]
    _write_quality_track(
        tmp_path,
        [
            (437_604, 440_148, "오랫동안 기다려 온\n순간이에요"),
            (440_232, 441_399, "'지구-616'"),
            (441_483, 443_610, "만나 주셔서\n감사합니다"),
            (444_694, 446_571, "제 능력이 도움이\n될 거라 믿어요"),
        ],
        [*cues, *ost],
    )
    report = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert report["counts"]["missing_ost"] == 1
    assert report["passes_targets"] is False
    assert any(d["kind"] == "missing_ost" and d["text"] == "THE SACRED TIMELINE" for d in report["actionable_defects"])


def test_quality_detects_carnival_sales_episodic_group_oscillation(tmp_path):
    import json
    from substage import refeval
    from substage.models import Character, CueObservation, Scene, VoiceCard
    from substage.store import save_model

    cues = [
        cue("c0179", 705_162, 708_124, "Technically, the,\nthe Carnival's not a minivan."),
        cue("c0180", 709_376, 710_377, "It's an MPV."),
        cue("c0181", 710_543, 712_128, "So how does the Kia compare to..."),
        cue("c0182", 712_295, 713_588, "-the Honda Odyssey?\n-The Odyssey?"),
        cue("c0183", 713_755, 716_299, "That's a great question.\nIt doesn't fucking suck."),
        cue("c0184", 716_966, 719_176, "You know, you can answer\nthe question without swearing."),
        cue("c0185", 719_344, 721_263, "I'm sorry, Tammy, I don't, I don't have kids."),
    ]
    obs = [
        CueObservation(cue_id="c0179", speaker="Wade Wilson", addressee="group"),
        CueObservation(cue_id="c0180", speaker="Wade Wilson", addressee="group"),
        CueObservation(cue_id="c0181", speaker="Tammy", addressee="Wade Wilson"),
        CueObservation(cue_id="c0182", speaker="Wade Wilson", addressee="Tammy"),
        CueObservation(cue_id="c0183", speaker="Wade Wilson", addressee="Tammy"),
        CueObservation(cue_id="c0184", speaker="Tammy", addressee="Wade Wilson"),
        CueObservation(cue_id="c0185", speaker="Wade Wilson", addressee="Tammy"),
    ]
    cm = ContextMap(
        synopsis="s", tone_baseline="t",
        characters=[Character(id="deadpool", en_name="Wade Wilson / Deadpool", ko_name="데드풀", aliases=["Wade", "Wade Wilson"], persona="p", voice=VoiceCard(idiolect="banter"))],
    )
    save_model(tmp_path / "context_map.yaml", cm)
    (tmp_path / "observations.json").write_text(json.dumps([x.model_dump() for x in obs]), encoding="utf-8")
    (tmp_path / "scenes.json").write_text(json.dumps([Scene(id="s012", start_ms=705_000, end_ms=722_000, cue_ids=[c.id for c in cues]).model_dump()]), encoding="utf-8")
    _write_quality_track(
        tmp_path,
        [
            (705_162, 708_124, "엄밀히 말하면\n카니발은 미니밴이 아니야"),
            (709_376, 710_377, "MPV지"),
            (710_543, 712_128, "기아차는\n비교하면 어떤가요?"),
            (712_295, 713_588, "- 혼다 오딧세이요?\n- 오딧세이?"),
            (713_755, 716_299, "좋은 질문이에요\n존나 안 구리거든요"),
            (716_966, 719_176, "욕 안 쓰고도\n대답할 수 있잖아요"),
            (719_344, 721_263, "죄송해요, 태미 씨\n제가 애가 없어서요"),
        ],
        cues,
    )
    bad_rep = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert bad_rep["counts"]["scene_oscillations"] == 3
    assert bad_rep["passes_targets"] is False

    _write_quality_track(
        tmp_path,
        [
            (705_162, 708_124, "엄밀히 말하면 카니발은\n미니밴이 아니에요"),
            (709_376, 710_377, "MPV예요"),
            (710_543, 712_128, "기아차는\n비교하면 어떤가요?"),
            (712_295, 713_588, "- 혼다 오딧세이요?\n- 오딧세이요?"),
            (713_755, 716_299, "좋은 질문이에요\n존나 안 구리거든요"),
            (716_966, 719_176, "욕 안 쓰고도\n대답할 수 있잖아요"),
            (719_344, 721_263, "죄송해요, 태미 씨\n제가 애가 없어서요"),
        ],
        cues,
    )
    good_rep = refeval.evaluate_media_quality("sample", ws_dir=tmp_path, settings=Settings())
    assert good_rep["counts"]["scene_oscillations"] == 0
    assert good_rep["register_score"] == 100.0
