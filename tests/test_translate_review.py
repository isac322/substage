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
