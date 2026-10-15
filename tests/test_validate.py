"""Deterministic tests for the S5 validators (rewrap, check, normalize_punct)
plus the per-cue register / protected-context handling of compress and
picture_check (fake Gemini client, no network)."""

from types import SimpleNamespace

from substage import recheck
from substage.config import Settings, Workspace
from substage.models import (
    AmbiguityFlag,
    Callback,
    Cue,
    CueEdit,
    GlossaryEntry,
    RegisterShift,
    SceneRegister,
    Snapshot,
    ValidationIssue,
    OmittedAct,
    RetainedFragment,
    SemanticLineage,
    SemanticSource,
    SourceAct,
)
from substage.register import RegisterGate
from substage.validate import check, compress, en_speaker_turns, enforce_speakers, finish, normalize_punct, rewrap

SETTINGS = Settings()  # cpl_limit=14, cpl_one_line=12, cps_max=8.0


def cue(start_ms: int = 0, end_ms: int = 2000, text: str = "line", cid: str = "c0001", **kw) -> Cue:
    return Cue(id=cid, start_ms=start_ms, end_ms=end_ms, text=text, raw=text, **kw)


def rules(c: Cue, ko: str) -> set[str]:
    return {i.rule for i in check(c, ko, SETTINGS)}


# ------------------------------------------------------------------ rewrap


def test_rewrap_never_splits_inside_eojeol():
    ko = "나는 정말로 괜찮아 보인다고 생각했어요"
    out = rewrap(ko, 12)
    for line in out.split("\n"):
        for word in line.split(" "):
            assert word in ko.split(" ")
    assert " ".join(out.split("\n")) == ko  # words intact, order preserved


def test_rewrap_keeps_single_word_whole():
    assert rewrap("가나다라마바사아자차카타파하가나다라마바사", 8) == "가나다라마바사아자차카타파하가나다라마바사"


def test_rewrap_max_two_lines():
    out = rewrap("가 나 다 라 마 바 사 아 자 차 카 타 파 하 가 나 다 라 마 바", 5)
    assert len(out.split("\n")) <= 2


def test_rewrap_prefers_bottom_heavy():
    # splits "aaa|b ccc" (3,5) and "aaa b|ccc" (5,3) both minimize max to 5;
    # the bottom-heavy one (line2 >= line1) wins
    out = rewrap("aaa b ccc", 6)
    assert out == "aaa\nb ccc"


def test_rewrap_bottom_heavy_unachievable_keeps_min_max():
    # only top-heavy splits minimize the max (5): smallest l1 chosen -> 5/4
    assert rewrap("aaaaa bb c", 6) == "aaaaa\nbb c"


def test_rewrap_single_line_when_it_fits():
    assert rewrap("짧은 대사", 16) == "짧은 대사"


def test_rewrap_flattens_excess_lines():
    out = rewrap("가나다\n라마바\n사아자", 16)
    assert out == "가나다 라마바 사아자"  # 11 chars: fits one line


def test_rewrap_dual_speaker_keeps_speakers():
    out = rewrap("-뭐라고 했어?\n-아무것도 아니야 진짜", 16)
    lines = out.split("\n")
    assert lines == ["- 뭐라고 했어?", "- 아무것도 아니야 진짜"]
    assert all(line.startswith("- ") for line in lines)


def test_rewrap_normalizes_dual_speaker_hyphens():
    out = rewrap("- 뭐라고 했어?\n- 아무것도 아니야", 16)
    assert out == "- 뭐라고 했어?\n- 아무것도 아니야"
    assert rewrap("-A\n-B", 16) == "- A\n- B"


def test_rewrap_dual_speaker_over_cpl_keeps_two_lines():
    long = "- 아무것도 아니야 정말 진짜로 아무것도"
    out = rewrap(f"{long}\n-그래?", 16)
    assert out == f"{long}\n- 그래?"
    assert "cpl" in rules(cue(end_ms=10000, text="- Nothing.\n- Yeah?"), out)


def test_rewrap_dual_speaker_never_exceeds_two_lines():
    out = rewrap("-가자\n-어디로\n가는데?\n-몰라", 16)
    assert len(out.split("\n")) == 2
    assert all(line.startswith("- ") for line in out.split("\n"))


# ------------------------------------------------------------------ check


def test_check_cps_boundary_ok():
    # 16 chars in 2.0 s = exactly 8.0 cps -> allowed
    assert "cps" not in rules(cue(end_ms=2000), "가" * 16)


def test_check_cps_boundary_over():
    # 17 chars in 2.0 s = 8.5 cps -> flagged
    assert "cps" in rules(cue(end_ms=2000), "가" * 17)


def test_check_cps_ignores_whitespace():
    # 16 non-whitespace chars in 2.0 s -> ok even though the raw string is longer
    assert "cps" not in rules(cue(end_ms=2000), "가" * 4 + " " + "가" * 4 + "\n" + "나" * 8)


def test_run_s5_leaves_reading_speed_to_layout():
    from substage.validate import run_s5

    gc = FakeGC([])
    out, unresolved, _ = run_s5(None, gc, [cue(0, 1000, "Hi.", "c0001")], {"c0001": "가" * 12}, {}, {}, SETTINGS)
    assert out == {"c0001": "가" * 12} and gc.calls == []  # no compress call for a cps-only issue
    assert [i.rule for i in unresolved] == ["cps"]  # still reported


def test_check_cpl_flags_long_line():
    assert "cpl" in rules(cue(), "가" * 15)
    assert "cpl" not in rules(cue(), "가" * 7 + "\n" + "가" * 14)


def test_check_cpl_flags_long_one_line_cue():
    # one-line cues stay <= 12 chars (official practice); 13-14 chars belong on two lines
    assert "cpl" not in rules(cue(), "가" * 12)
    assert "cpl" in rules(cue(), "가" * 13)


def test_rewrap_breaks_one_line_cue_over_one_line_limit():
    s = Settings()
    assert (s.cpl_limit, s.cpl_one_line) == (14, 12)
    assert rewrap("가나다라 마바사 아자차", s.cpl_limit, s.cpl_one_line) == "가나다라 마바사 아자차"  # 12: one line
    assert rewrap("가나다라 마바사 아자차카", s.cpl_limit, s.cpl_one_line) == "가나다라\n마바사 아자차카"


def test_check_lines_flags_three_lines():
    assert "lines" in rules(cue(), "가\n나\n다")
    assert "lines" not in rules(cue(), "가\n나")


def test_check_empty_when_source_has_text():
    assert "empty" in rules(cue(), "")
    assert "empty" in rules(cue(), "  ")
    assert "empty" in rules(cue(text="Oh, no."), "")
    assert "empty" in rules(cue(text="Hey, you!"), "")


def test_check_empty_ok_for_sound_description_only():
    assert rules(cue(text="(MEN SCREAMING)"), "") == set()
    assert rules(cue(text="(INDISTINCT SHOUTING AND SCREAMING)"), "") == set()
    assert rules(cue(text="[grunts]"), "") == set()


def test_check_empty_ok_for_lyric():
    assert rules(cue(text="♪ Take on me ♪", lyric=True), "") == set()


def test_check_empty_ok_for_interjection():
    for text in ("Hmm.", "Oh!", "Uh, um...", "Hey?", "MMM", "Whoa!", "(GROANS) Ugh."):
        assert rules(cue(text=text), "") == set(), text


def test_check_empty_ok_for_sound_cue():
    # pure SDH sound cue (no dialogue text) needs no Korean text
    assert rules(cue(text=""), "") == set()


def test_check_punctuation_flags_terminal_period_unicode_ellipsis_and_quotes():
    assert "punctuation" in rules(cue(), "그래.")
    assert "punctuation" in rules(cue(), "그래…")
    assert "punctuation" not in rules(cue(), "그래...")
    assert "punctuation" in rules(cue(), "‘타이타닉’ 봤어?")
    assert "punctuation" in rules(cue(), '"타이타닉" 봤어?')
    assert "punctuation" not in rules(cue(), "'타이타닉' 봤어?")
    assert "punctuation" not in rules(cue(), '<font color="red">위험</font>')


def test_check_lyric_needs_one_leading_note():
    lyric = cue(text="We are the champions", lyric=True)
    assert "punctuation" not in rules(lyric, "♪ 우리는 챔피언")
    assert "punctuation" in rules(lyric, "우리는 챔피언")
    assert "punctuation" in rules(lyric, "♪ 우리는 챔피언 ♪")


def test_check_clean_line_passes():
    assert rules(cue(), "괜찮아?") == set()


def test_check_dual_line_needs_hyphen_space():
    two = cue(text="- Hi.\n- Hey.")
    assert rules(two, "- 안녕\n- 어") == set()
    assert "linebreak" in rules(two, "-안녕\n-어")


def test_check_flags_speaker_mismatch():
    assert "speakers" in rules(cue(text="- Hi.\n- Hey."), "안녕\n어")
    assert "speakers" in rules(cue(text="Let's go. Hurry."), "- 가자\n- 빨리")
    assert "speakers" not in rules(cue(text="Let's go. Hurry."), "가자 빨리")


# ------------------------------------------------------------------ speaker turns


def test_en_speaker_turns():
    assert en_speaker_turns(cue(text="- Hi.\n- Hey.")) == 2
    assert en_speaker_turns(cue(text="– Hi.\n– Hey.")) == 2
    assert en_speaker_turns(cue(text="Hi.\n- Hey.")) == 2
    assert en_speaker_turns(cue(text="Hi there.\nHow are you?", speaker_hint="A/B")) == 2
    assert en_speaker_turns(cue(text="Hi there.\nHow are you?", speaker_hint="A")) == 1
    assert en_speaker_turns(cue(text="- Hi there,\nhow are you?")) == 1  # lone dash
    assert en_speaker_turns(cue(text="I said\n--never mind.")) == 1
    assert en_speaker_turns(cue(text="Got it?\nYeah.")) == 2
    assert en_speaker_turns(cue(text="Wade...\nHey.", speaker_hint="COLOSSUS").model_copy(update={"raw": "COLOSSUS: Wade...\nHey."})) == 2
    assert en_speaker_turns(cue(text="-See? All right. Man: All right.")) == 2

def test_enforce_speakers_splits_at_line_break():
    assert enforce_speakers(cue(text="- Hi.\n- Hey."), "안녕\n어", 16) == "- 안녕\n- 어"


def test_enforce_speakers_splits_one_line_at_middle_sentence_boundary():
    c = cue(text="- Where are you going?\n- Home.")
    assert enforce_speakers(c, "어디 가? 집에", 16) == "- 어디 가?\n- 집에"
    assert enforce_speakers(c, "그래. 어디 가? 집에 가", 16) == "- 그래. 어디 가?\n- 집에 가"


def test_enforce_speakers_no_safe_split_keeps_text_and_check_flags():
    c = cue(text="- Where?\n- Home.")
    assert enforce_speakers(c, "어디 집에", 16) == "어디 집에"
    assert "speakers" in rules(c, "어디 집에")


def test_enforce_speakers_single_english_speaker_drops_dashes():
    assert enforce_speakers(cue(text="Let's go. Hurry."), "- 가자\n- 빨리", 16) == "가자 빨리"


def test_enforce_speakers_hint_two_speakers():
    c = cue(text="Hi there.\nHow are you?", speaker_hint="A/B")
    assert enforce_speakers(c, "안녕\n잘 지내?", 16) == "- 안녕\n- 잘 지내?"


def test_enforce_speakers_matching_structure_unchanged():
    assert enforce_speakers(cue(text="- Hi.\n- Hey."), "- 안녕\n- 어", 16) == "- 안녕\n- 어"
    assert enforce_speakers(cue(text="Hi."), "안녕", 16) == "안녕"


def test_run_s5_enforces_speaker_structure():
    from substage.validate import run_s5

    cues = [
        cue(0, 4000, "- Hi.\n- Hey.", "c0001"),
        cue(5000, 9000, "Let's go. Hurry.", "c0002"),
    ]
    out, unresolved, changed = run_s5(
        None, FakeGC([]), cues, {"c0001": "안녕.\n어.", "c0002": "-가자.\n-빨리."}, {}, {}, SETTINGS
    )
    assert out == {"c0001": "- 안녕\n- 어", "c0002": "가자 빨리"}
    assert not [i for i in unresolved if i.rule == "speakers"]
    assert changed == {"c0001", "c0002"}



# ------------------------------------------------------------------ normalize_punct


def test_normalize_strips_terminal_period():
    assert normalize_punct("그래.") == "그래"
    assert normalize_punct("그래..") == "그래..."  # '..' is an ellipsis too
    assert normalize_punct("첫 줄.\n둘째 줄.") == "첫 줄\n둘째 줄"


def test_normalize_writes_three_period_ellipsis():
    # the official files write '...' (12 of 13 videos): '…' and odd runs become '...'
    assert normalize_punct("그래...") == "그래..."
    assert normalize_punct("그래…") == "그래..."
    assert normalize_punct("그래.....") == "그래..."
    assert normalize_punct("글쎄… 모르겠어") == "글쎄... 모르겠어"
    # a period or comma left dangling before a trailing ellipsis collapses
    assert normalize_punct("다.…") == "다..."
    assert normalize_punct("그러니까,...") == "그러니까..."


def test_normalize_keeps_question_and_exclamation():
    assert normalize_punct("뭐?") == "뭐?"
    assert normalize_punct("조심해!") == "조심해!"


def test_normalize_idempotent():
    once = normalize_punct("글쎄... 그런가 본데.")
    assert normalize_punct(once) == once


def test_normalize_strips_terminal_comma_and_period_every_line():
    assert normalize_punct("건배, 네 건강을 위해.\n엿이나 먹어") == "건배, 네 건강을 위해\n엿이나 먹어"
    assert normalize_punct("그러니까,\n가자.") == "그러니까\n가자"
    assert normalize_punct("정말?\n그래!\n글쎄…") == "정말?\n그래!\n글쎄..."


def test_normalize_dual_speaker_adds_missing_first_dash():
    assert normalize_punct("A\n- B") == "- A\n- B"
    assert normalize_punct("가자.\n-어디로?") == "- 가자\n- 어디로?"


def test_normalize_dual_speaker_hyphen_style():
    assert normalize_punct("-A\n-B") == "- A\n- B"
    assert normalize_punct("- 뭐라고?\n– 아무것도.") == "- 뭐라고?\n- 아무것도"


def test_normalize_dual_speaker_inline_boundary_split():
    assert normalize_punct("뭐라고? / - 아무것도 아냐") == "- 뭐라고?\n- 아무것도 아냐"
    assert normalize_punct("뭐라고? - 아무것도 아냐.") == "- 뭐라고?\n- 아무것도 아냐"


def test_normalize_dual_speaker_continuation_line_joins_turn():
    assert normalize_punct("-가자\n-어디로\n가는데?") == "- 가자\n- 어디로 가는데?"


def test_normalize_drops_stray_dash_on_single_speaker():
    assert normalize_punct("- 가자.") == "가자"


def test_normalize_dual_idempotent():
    once = normalize_punct("A...\n- B, C.")
    assert once == "- A...\n- B, C"
    assert normalize_punct(once) == once


def test_normalize_then_rewrap_three_lines_to_two():
    out = rewrap(normalize_punct("하나.\n둘…\n셋, 넷."), 14)
    assert len(out.split("\n")) <= 2
    assert "…" not in out and "둘..." in out and not out.endswith(".")


def test_normalize_angle_bracket_titles_become_quotes():
    """SRT players strip '<제목>' as a markup tag, so the title vanishes;
    titles become '제목' while real formatting tags survive."""
    assert normalize_punct("프로그램 <코쿤>은 완전 막장이야") == "프로그램 '코쿤'은 완전 막장이야"
    assert normalize_punct("<127시간> 봤어?\n<재즈 싱어>네요.") == "'127시간' 봤어?\n'재즈 싱어'네요"
    assert normalize_punct("<i>기억해</i>") == "<i>기억해</i>"
    assert normalize_punct('<font color="red">위험</font>') == '<font color="red">위험</font>'
    assert normalize_punct("- <스타워즈> 봤어?\n- 응") == "- '스타워즈' 봤어?\n- 응"


def test_normalize_quotes_become_straight_single():
    # one quote style per video: the corpus majority writes straight single quotes
    assert normalize_punct("‘타이타닉’ 봤어?") == "'타이타닉' 봤어?"
    assert normalize_punct("“나는 슬픔의 도시로 가는 길”") == "'나는 슬픔의 도시로 가는 길'"
    assert normalize_punct('"고마워"라더라') == "'고마워'라더라"


def test_finish_marks_lyric_cues():
    s = Settings()
    lyric = cue(text="We are the champions", lyric=True)
    assert finish(lyric, "우리는 챔피언 ♪", s) == "♪ 우리는 챔피언"
    assert finish(lyric, "♪ ♪ 우리는 챔피언이다.", s) == "♪ 우리는 챔피언이다"
    out = finish(lyric, "내가 내일까지 돌아오지 않더라도", s)
    assert out.startswith("♪ ") and out.count("♪") == 1
    assert all(len(line) <= s.cpl_limit for line in out.split("\n"))
    assert rules(lyric, out) & {"punctuation", "cpl"} == set()
    # dialogue keeps its text; an empty lyric stays empty
    assert finish(cue(), "그래.", s) == "그래"
    assert finish(lyric, "", s) == ""


def test_run_s5_marks_lyric_cues():
    from substage.validate import run_s5

    lyric = cue(text="We are the champions", lyric=True)
    out, unresolved, _ = run_s5(None, FakeGC([]), [lyric], {"c0001": "우리는 챔피언"}, {}, {}, SETTINGS)
    assert out == {"c0001": "♪ 우리는 챔피언"} and unresolved == []


# ------------------------------------------------------------------ compress / picture_check context


class FakeGC:
    """Duck-typed GeminiClient: records prompts, returns queued edits."""

    def __init__(self, edits: list[CueEdit], narrative: str = "lite"):
        self.edits = edits
        self.calls: list[dict] = []
        self.settings = SimpleNamespace(
            model_main="fake-main", model_light="fake-light", narrative_context=narrative
        )

    def generate(self, *, stage, prompt, schema, clips=(), model=None, **kw):
        self.calls.append({"stage": stage, "prompt": prompt, "clips": list(clips)})
        return schema(edits=self.edits)


def _shift_snapshot() -> Snapshot:
    """peter>tony sits at haeyo and shifts to hae at c0002 (in-scene)."""
    return Snapshot(
        scene_id="s001",
        registers={
            "peter>tony": SceneRegister(
                level="haeyo", shifts=[RegisterShift(at_cue="c0002", level="hae", reason="relationship_change")]
            ),
            "tony>peter": SceneRegister(level="hae"),
        },
        character_names={"peter": "피터", "tony": "토니"},
    )


def _scene_cues() -> list[Cue]:
    return [
        cue(0, 2000, "Mr. Stark, look at this", "c0001", speaker_hint="PETER"),
        cue(3000, 5000, "Tony, call me Tony", "c0002", speaker_hint="PETER"),
        cue(6000, 8000, "Tony, look", "c0003", speaker_hint="PETER"),
        cue(9000, 11000, "Who said that?", "c0004"),
    ]


def _issue(cid: str) -> list[ValidationIssue]:
    return [ValidationIssue(cue_id=cid, rule="cps", detail="too fast")]


def test_compress_renders_register_effective_at_each_cue():
    cues = _scene_cues()
    snaps = {"s001": _shift_snapshot()}
    scene_of = {c.id: "s001" for c in cues}
    order = {c.id: i for i, c in enumerate(cues)}
    ko = {"c0001": "토니 씨 이것 좀 봐요", "c0003": "토니 이것 좀 봐", "c0004": "누가 그랬어?"}
    items = [(c, ko[c.id], _issue(c.id)) for c in cues if c.id in ko]
    gc = FakeGC([], narrative="full")
    compress(None, gc, items, snaps, scene_of, ko=ko, order=order)
    lines = gc.calls[0]["prompt"].splitlines()
    assert "REG lines = the register table's 'speaker>addressee | level' at that cue;" in lines
    # before the in-scene shift: entry level; after it: shifted level
    assert "   REG c0001 | peter>tony | haeyo" in lines
    assert "   REG c0003 | peter>tony | hae" in lines
    # only the cue's own speaker for known speakers
    assert not any(l.startswith("   REG c0001 | tony>") for l in lines)
    # unknown speaker falls back to every present pair (at that cue's position)
    assert "   REG c0004 | peter>tony | hae" in lines
    assert "   REG c0004 | tony>peter | hae" in lines
    # no shared merged register table
    assert "Register targets" not in gc.calls[0]["prompt"]


def test_compress_lite_keeps_register_rows():
    cues = _scene_cues()
    snaps = {"s001": _shift_snapshot()}
    scene_of = {c.id: "s001" for c in cues}
    order = {c.id: i for i, c in enumerate(cues)}
    ko = {"c0001": "토니 씨 이것 좀 봐요", "c0003": "토니 이것 좀 봐"}
    items = [(c, ko[c.id], _issue(c.id)) for c in cues if c.id in ko]
    gc = FakeGC([])  # default narrative_context: lite
    compress(None, gc, items, snaps, scene_of, ko=ko, order=order)
    prompt = gc.calls[0]["prompt"]
    assert "   REG c0003 | peter>tony | hae" in prompt.splitlines()
    assert "[lite]" not in prompt and "[full]" not in prompt
    assert "PROTECT c0001:" in prompt and "tony=토니" in prompt


def test_compress_rejects_edit_dropping_protected_name():
    cues = _scene_cues()
    snaps = {"s001": _shift_snapshot()}
    scene_of = {c.id: "s001" for c in cues}
    ko = {"c0001": "토니 씨 이것 좀 봐요 진짜로", "c0003": "토니 이것 좀 봐 진짜"}
    items = [(c, ko[c.id], _issue(c.id)) for c in cues if c.id in ko]
    gc = FakeGC([
        CueEdit(cue_id="c0001", new_ko="이것 좀 봐요", reason="shorter", kind="condensation"),  # drops 토니
        CueEdit(cue_id="c0003", new_ko="토니 이거 봐", reason="shorter", kind="condensation"),  # keeps 토니
    ])
    edits = compress(None, gc, items, snaps, scene_of, ko=ko)
    assert [e.cue_id for e in edits] == ["c0003"]
    prompt = gc.calls[0]["prompt"]
    assert "PROTECT c0001:" in prompt and "tony=토니" in prompt
    assert "Preserve these protected forms verbatim" in prompt


def test_compress_protects_glossary_callback_and_ambiguity():
    c_payoff = cue(5000, 7000, "Remember: with great power, kid", "c0002", speaker_hint="TONY")
    snap = Snapshot(
        scene_id="s001",
        glossary=[GlossaryEntry(en="kid", ko="꼬맹이")],
        callbacks_live=[Callback(motif="power", setup_cues=["c0001"], payoff_cues=["c0002"])],
        preserve_ambiguity=[AmbiguityFlag(cue_id="c0002", note="is he proud or angry?")],
    )
    ko = {"c0001": "큰 힘에는", "c0002": "기억해 큰 힘에는 책임이 따른다 꼬맹이"}
    items = [(c_payoff, ko["c0002"], _issue("c0002"))]
    scene_of = {"c0001": "s001", "c0002": "s001"}
    for bad in ("기억해 책임이 따른다 꼬맹이", "기억해 큰 힘에는 책임이 따른다"):
        gc = FakeGC([CueEdit(cue_id="c0002", new_ko=bad, reason="shorter", kind="condensation")])
        assert compress(None, gc, items, {"s001": snap}, scene_of, ko=ko) == []
    prompt = gc.calls[0]["prompt"]
    assert "c0001 KO 큰 힘에는" in prompt
    assert "kid=꼬맹이" in prompt
    assert "ambiguity: is he proud or angry?" in prompt
    gc = FakeGC([CueEdit(cue_id="c0002", new_ko="큰 힘에는 책임이야 꼬맹이", reason="shorter", kind="condensation")])
    assert len(compress(None, gc, items, {"s001": snap}, scene_of, ko=ko)) == 1


def test_picture_check_shows_no_registers_and_rejects_level_or_protected_edits(tmp_path):
    ws = Workspace(
        video_id="t", data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", log_dir=tmp_path / "log"
    )
    cues = _scene_cues()
    cues_by_id = {c.id: c for c in cues}
    snaps = {"s001": _shift_snapshot()}
    scene_of = {c.id: "s001" for c in cues}
    ko = {"c0001": "토니 씨 이것 좀 봐요", "c0003": "토니 이것 좀 봐", "c0004": "누가 그랬어?"}
    gc = FakeGC([
        CueEdit(cue_id="c0001", new_ko="이것 좀 봐요", reason="picture", kind="picture"),  # drops 토니
        CueEdit(cue_id="c0003", new_ko="토니, 이거 봐", reason="picture", kind="picture"),
        CueEdit(cue_id="c0004", new_ko="누가 그랬어요?", reason="picture", kind="picture"),  # 반말 -> 해요
    ], narrative="full")
    proxy = SimpleNamespace(name="proxy", uri="gs://proxy", mime_type="video/mp4")
    gate = RegisterGate(None, {})
    edits = recheck.picture_check(
        ws, gc, proxy, cues_by_id, ko, {"c0001", "c0003", "c0004"}, snaps, scene_of, gate=gate
    )
    assert [e.cue_id for e in edits] == ["c0003"]
    assert [(r.stage, r.cue_ids, r.reason) for r in gate.rejections] == [("picture", ["c0004"], "level")]
    prompt = gc.calls[0]["prompt"]
    assert "REG" not in prompt and "[lite]" not in prompt and "[full]" not in prompt
    assert "register is not judged here" in prompt
    assert "PROTECT c0001:" in prompt


def _retained_case(*, second_essential=False, speakers=("a", "b")):
    first, second = ("Ready?", "Yes.") if second_essential else ("Ready.", "Mm-hm.")
    source = f"{first} {second}"
    c = cue(text=source)
    acts = [
        SourceAct(ref=f"{c.id}:t1:a1", cue_id=c.id, turn_ordinal=1, act_ordinal=1,
                  start_char=0, end_char=len(first), text=first, speaker=speakers[0],
                  meaning="question" if second_essential else "statement", essential=True),
        SourceAct(ref=f"{c.id}:t2:a1", cue_id=c.id, turn_ordinal=2, act_ordinal=1,
                  start_char=len(first) + 1, end_char=len(source), text=second, speaker=speakers[1],
                  meaning="answer" if second_essential else "incidental acknowledgment",
                  essential=second_essential),
    ]
    fragments = [
        RetainedFragment(text="준비됐어?" if second_essential else "준비됐어",
                         refs=[acts[0].ref], speaker=speakers[0]),
        RetainedFragment(text="응", refs=[acts[1].ref], speaker=speakers[1]),
    ]
    return c, SemanticLineage(
        source=SemanticSource(scene_id="s1", acts=acts),
        fragments_by_cue={c.id: fragments}, structural_valid=True,
    )


def test_actual_retained_two_voices_survive_single_source_line_normalization():
    c, lineage = _retained_case(second_essential=True)
    fragments = lineage.fragments_by_cue[c.id]
    output = finish(c, "준비됐어?\n응", SETTINGS, retained_fragments=fragments)
    assert output == "- 준비됐어?\n- 응"
    assert "speakers" not in {i.rule for i in check(c, output, SETTINGS, retained_fragments=fragments)}


def test_source_two_turns_do_not_invent_retained_second_voice_after_incidental_omission():
    from substage.validate import lineage_issues

    c, lineage = _retained_case()
    dropped = lineage.fragments_by_cue[c.id].pop()
    lineage.omissions = [OmittedAct(ref=dropped.refs[0], reason="incidental backchannel")]
    output = finish(c, "준비됐어", SETTINGS, retained_fragments=lineage.fragments_by_cue[c.id])
    assert output == "준비됐어"
    assert lineage_issues(lineage) == []


def test_declared_essential_missing_answer_is_coverage_audit_not_fake_dash_or_reply():
    from substage.validate import lineage_issues

    c, lineage = _retained_case(second_essential=True)
    lineage.fragments_by_cue[c.id].pop()
    output = finish(c, "준비됐어?", SETTINGS, retained_fragments=lineage.fragments_by_cue[c.id])
    assert output == "준비됐어?" and "-" not in output
    issues = lineage_issues(lineage)
    assert [issue.rule for issue in issues] == ["essential_coverage"]
    assert "declared essential ref" in issues[0].detail


def test_unknown_and_same_speaker_multiple_acts_never_create_dialogue_dashes():
    for speakers in ((None, None), ("a", "a"), ("a", None)):
        c, lineage = _retained_case(second_essential=True, speakers=speakers)
        output = finish(c, "준비됐어?\n응", SETTINGS, retained_fragments=lineage.fragments_by_cue[c.id])
        assert output.replace("\n", " ") == "준비됐어? 응"
        assert "-" not in output


def test_punctuation_and_rewrap_preserve_attribution_but_actual_rewrite_invalidates_it():
    from substage.validate import reconcile_lineage

    c, lineage = _retained_case(second_essential=True)
    before_refs = [fragment.refs for fragment in lineage.fragments_by_cue[c.id]]
    reconcile_lineage(lineage, {c.id: "- 준비됐어?\n- 응."})
    assert [fragment.refs for fragment in lineage.fragments_by_cue[c.id]] == before_refs
    reconcile_lineage(lineage, {c.id: "준비됐으면 가자"})
    actual = lineage.fragments_by_cue[c.id]
    assert len(actual) == 1 and actual[0].text == "준비됐으면 가자"
    assert actual[0].refs == [] and actual[0].speaker is None
    assert actual[0].attribution == "unverified"
    assert actual[0].prior_refs == [ref for refs in before_refs for ref in refs]


def test_legitimate_repeated_fragment_keeps_text_and_shared_source_ancestry():
    from substage.validate import reconcile_lineage

    c, lineage = _retained_case(speakers=("a", "a"))
    original = lineage.fragments_by_cue[c.id][1]
    lineage.fragments_by_cue[c.id] = [original, original.model_copy(deep=True)]
    reconcile_lineage(lineage, {c.id: "응 응"})
    assert finish(c, "응 응", SETTINGS, retained_fragments=lineage.fragments_by_cue[c.id]) == "응 응"
    assert [f.refs for f in lineage.fragments_by_cue[c.id]] == [original.refs, original.refs]


def _semantic_lyric_case(fragment_text="가사"):
    c = cue(text="A story lyric.", lyric=True)
    act = SourceAct(ref=f"{c.id}:t1:a1", cue_id=c.id, turn_ordinal=1, act_ordinal=1,
                    start_char=0, end_char=len(c.text), text=c.text,
                    meaning="plot-relevant sung line", essential=True)
    lineage = SemanticLineage(
        source=SemanticSource(scene_id="s1", acts=[act]),
        fragments_by_cue={c.id: [RetainedFragment(text=fragment_text, refs=[act.ref], speaker=None)]},
        structural_valid=True,
    )
    return c, lineage


def test_semantic_lyric_display_marker_does_not_invalidate_attribution_or_duplicate_on_finish():
    from substage.validate import lineage_issues, reconcile_lineage

    c, lineage = _semantic_lyric_case()
    first = finish(c, "가사", SETTINGS, retained_fragments=lineage.fragments_by_cue[c.id])
    assert first == "♪ 가사"
    reconcile_lineage(lineage, {c.id: first})
    fragment = lineage.fragments_by_cue[c.id][0]
    assert fragment.text == "가사" and fragment.attribution == "declared"
    assert fragment.refs == [f"{c.id}:t1:a1"] and fragment.prior_refs == []
    second = finish(c, first, SETTINGS, retained_fragments=lineage.fragments_by_cue[c.id])
    assert second == first and second.count("♪") == 1
    assert lineage_issues(lineage) == []


def test_semantic_lyric_strips_all_display_notes_from_fragments_before_single_marker_render():
    from substage.validate import reconcile_lineage

    c, lineage = _semantic_lyric_case("♪ ♪ 가사 ♪")
    output = finish(c, "♪ ♪ 가사 ♪", SETTINGS, retained_fragments=lineage.fragments_by_cue[c.id])
    assert output == "♪ 가사"
    reconcile_lineage(lineage, {c.id: output})
    assert lineage.fragments_by_cue[c.id][0].text == "가사"
    assert lineage.fragments_by_cue[c.id][0].attribution == "declared"


def test_semantic_lyric_s5_finishing_keeps_declared_refs_without_a_postwriter_call():
    from substage.validate import run_s5

    c, lineage = _semantic_lyric_case()
    output, unresolved, changed = run_s5(None, None, [c], {c.id: "가사"}, {}, {}, SETTINGS, lineage=lineage)
    assert output == {c.id: "♪ 가사"} and unresolved == [] and changed == {c.id}
    assert lineage.fragments_by_cue[c.id][0].attribution == "declared"
    assert lineage.fragments_by_cue[c.id][0].refs == [f"{c.id}:t1:a1"]
