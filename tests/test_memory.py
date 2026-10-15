"""Translator memory: pair building, corpus storage, BM25 retrieval, and the
TRANSLATOR EXAMPLES prompt block / checkpoint wiring in S3 and ablate."""

from __future__ import annotations

import random
import re
from types import SimpleNamespace

import pytest

from substage import ablate, cli, memory, translate
from substage.config import Workspace
from substage.memory import Pair
from substage.models import Cue, VideoInfo, Scene, SceneTranslation, Snapshot, TranslatedCue
from substage.refeval import Line
from substage.store import save_model


def cue(cid, start, end, text, **kw):
    return Cue(id=cid, start_ms=start, end_ms=end, text=text, raw=text, **kw)


def line(start, end, text):
    return Line(0, start, end, text)


def ws(tmp_path, video_id="test-video") -> Workspace:
    return Workspace(video_id=video_id, data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", log_dir=tmp_path / "log")


class FakeGemini:
    _CUE_LINE = re.compile(r"^  ([A-Za-z0-9_-]+) \| \d{2}:", re.M)

    def __init__(self, **settings):
        self.calls: list[dict] = []
        self.settings = SimpleNamespace(model_main="fake-main", narrative_context="lite", **settings)

    def generate(self, *, stage, prompt, schema, clips=(), images=(), model=None, thinking_level=None, system=None):
        self.calls.append(dict(stage=stage, prompt=prompt))
        sid = re.search(r"Scene (s\d+)", prompt).group(1)
        return SceneTranslation(
            scene_id=sid, mode="text", cues=[TranslatedCue(cue_id=i, ko=f"KO-{i}") for i in self._CUE_LINE.findall(prompt)]
        )


# ---------------------------------------------------------------- build_pairs


def test_build_pairs_keeps_small_groups_and_skips_noise():
    cues = [
        cue("c1", 0, 2000, "Hello there."),
        cue("c2", 2500, 4000, "Get   down!"),
        cue("c3", 5000, 7000, "La la la", lyric=True),
        cue("c4", 8000, 9000, ""),  # sound-only
        cue("c5", 10000, 12000, "I'm fine,\nthanks."),
        # 4 EN cues under one official line: too large a group
        cue("c6", 20000, 21500, "One."), cue("c7", 22000, 23500, "Two."),
        cue("c8", 24000, 25500, "Three."), cue("c9", 26000, 27500, "Four."),
        cue("c10", 30000, 31000, "Yeah."),  # official line without Hangul
    ]
    official = [
        line(0, 2000, "안녕"),
        line(2500, 4000, "엎드려!"),
        line(5000, 7000, "♪ 라라라"),
        line(10000, 12000, "괜찮아,\n  고마워"),
        line(20000, 28000, "하나 둘 셋 넷"),
        line(30000, 31000, "♪♪"),
    ]
    pairs = memory.build_pairs("video-a", cues, official)
    assert pairs == [
        Pair("video-a", 0, "Hello there.", "안녕", ""),
        Pair("video-a", 2500, "Get down!", "엎드려!", "Hello there."),
        Pair("video-a", 10000, "I'm fine, thanks.", "괜찮아,\n고마워", "Get down!"),
    ]


def test_build_pairs_groups_up_to_three_en_and_two_ko():
    cues = [cue("c1", 0, 1500, "Wait."), cue("c2", 1600, 3000, "Wait, wait."), cue("c3", 3100, 4500, "Stop!")]
    official = [line(0, 2200, "잠깐"), line(2300, 4500, "멈춰!")]
    (p,) = memory.build_pairs("f", cues, official)
    assert p.en == "Wait. Wait, wait. Stop!" and p.ko == "잠깐\n멈춰!"


# ---------------------------------------------------------------- storage


def test_add_video_replaces_rows_and_stats(tmp_path):
    path = tmp_path / "memory.jsonl"
    memory.add_video("a", [Pair("a", 0, "hi", "안녕"), Pair("a", 10, "bye", "잘가")], path)
    memory.add_video("b", [Pair("b", 0, "yes", "응")], path)
    memory.add_video("a", [Pair("a", 5, "hey", "야")], path)
    assert memory.load_pairs(path) == [Pair("a", 5, "hey", "야"), Pair("b", 0, "yes", "응")]
    assert memory.stats(memory.load_pairs(path)) == {"a": 1, "b": 1}
    with pytest.raises(ValueError):
        memory.add_video("a", [Pair("b", 0, "x", "엑스")], path)


# ---------------------------------------------------------------- retrieval


def test_tokens_keep_contractions():
    assert memory.tokens("You DON’T know, I'm sure it's 42!") == ["you", "don't", "know", "i'm", "sure", "it's", "42"]


def _corpus() -> list[Pair]:
    return [
        Pair("video-a", 0, "Where is the money?", "돈 어딨어?"),
        Pair("video-a", 20000, "Give me the money now.", "돈 내놔"),
        Pair("video-b", 0, "The money is gone.", "돈은 없어"),
        Pair("video-b", 30000, "I love tacos.", "타코 좋아"),
        Pair("video-c", 0, "Show me the money!", "돈 보여줘!"),
        Pair("video-c", 40000, "Nice weather.", "날씨 좋네"),
    ]


def test_retrieve_excludes_video_and_ranks_by_bm25():
    out = memory.retrieve(_corpus(), ["Where's the money?", "money money"], "video-a", 10)
    assert out and all(p.video_id != "video-a" for p in out)
    assert {p.ko for p in out} == {"돈은 없어", "돈 보여줘!"}
    # unrelated pairs never score
    assert memory.retrieve(_corpus(), ["zebra"], "video-a", 5) == []
    assert memory.retrieve(_corpus(), ["money"], "video-a", 0) == []


def test_retrieve_is_deterministic_regardless_of_corpus_order():
    pairs = _corpus()
    want = memory.retrieve(pairs, ["the money"], "video-x", 4)
    for seed in range(5):
        shuffled = pairs[:]
        random.Random(seed).shuffle(shuffled)
        assert memory.retrieve(shuffled, ["the money"], "video-x", 4) == want
    # equal scores tie-break by (video_id, start_ms, en)
    ties = [Pair("z", 0, "hello", "안녕"), Pair("a", 50000, "hello", "안녕하세요"), Pair("a", 0, "hello", "여보세요")]
    assert [p.id for p in memory.retrieve(ties, ["hello"], "x", 3)] == ["a@0", "a@50000", "z@0"]


def test_retrieve_caps_pairs_per_ten_second_window_and_dedupes():
    pairs = [Pair("b", 1000 * i, f"run now {i}", f"뛰어 {i}") for i in range(6)]  # all in one 10 s window
    pairs += [Pair("b", 15000, "run now later", "나중에 뛰어"), Pair("c", 0, "run now 0", "뛰어 0")]  # dup text of b@0
    out = memory.retrieve(pairs, ["run now"], "a", 10)
    window = [p for p in out if p.video_id == "b" and p.start_ms < 10000]
    assert len(window) == memory.WINDOW_CAP
    assert any(p.start_ms == 15000 for p in out)
    texts = [(p.en, p.ko) for p in out]
    assert len(texts) == len(set(texts))


def test_index_stats_exclude_the_video():
    """The excluded video's rows don't even shape IDF (leave-one-out)."""
    pairs = _corpus()
    c = memory.Corpus(pairs)
    assert c.fingerprint("video-a") == memory.Corpus([p for p in pairs if p.video_id != "video-a"]).fingerprint("none")
    assert c.fingerprint("video-a") != c.fingerprint("video-b")


def test_base_video_id_strips_arm():
    assert memory.base_video_id("sample-2016__mem12") == "sample-2016"
    assert memory.base_video_id("sample-2016") == "sample-2016"


# ---------------------------------------------------------------- translate wiring


def _scene_setup():
    sc = Scene(id="s001", start_ms=0, end_ms=5000, cue_ids=["c0001"], act=1)
    cues = [cue("c0001", 0, 4000, "Where's the money?")]
    return sc, cues


def _video() -> VideoInfo:
    return VideoInfo(video_id="test-video", title="T", video_path="/v.mkv", duration_ms=60_000, audio_stream=1,
                    en_sub_path="/en.srt", en_sub_origin="embedded:2")


def _seed_corpus():
    memory.add_video("test-video", [Pair("test-video", 0, "Where is the money?", "돈 어딨냐고")])
    memory.add_video("other", [Pair("other", 0, "Give me the money.", "돈 내놔", "Hey.")])


def test_prompt_has_examples_only_when_enabled(tmp_path):
    sc, cues = _scene_setup()
    by_id = {c.id: c for c in cues}
    # no corpus: no block
    gc = FakeGemini()
    translate.translate_scene(ws(tmp_path), gc, sc, by_id, Snapshot(scene_id="s001"), [], [], mode="text", proxy=None,
                              examples=translate.scene_examples(gc.settings, cues, "test-video"))
    assert "TRANSLATOR EXAMPLES" not in gc.calls[0]["prompt"]

    _seed_corpus()
    off = FakeGemini(memory_k=0)
    assert translate.scene_examples(off.settings, cues, "test-video") == []
    assert translate.memory_tag(off.settings, "test-video") == []

    on = FakeGemini(memory_k=4)
    ex = translate.scene_examples(on.settings, cues, "test-video__arm")
    assert [p.video_id for p in ex] == ["other"]
    translate.translate_scene(ws(tmp_path), on, sc, by_id, Snapshot(scene_id="s001"), [], [], mode="text", proxy=None,
                              examples=ex)
    p = on.calls[0]["prompt"]
    assert "## TRANSLATOR EXAMPLES (official Korean subtitles of other videos" in p
    assert "EN: Give me the money. (prev: Hey.) → KO: 돈 내놔" in p
    assert "돈 어딨냐고" not in p  # the video being translated is never an example
    assert p.index("TRANSLATOR EXAMPLES") < p.index("## CUES")


def test_translate_video_checkpoint_hash_tracks_examples(tmp_path):
    sc, cues = _scene_setup()
    w = ws(tmp_path, "test-video__arm")

    def run():
        gc = FakeGemini()
        translate.translate_video(w, gc, _video(), [sc], cues, {"s001": Snapshot(scene_id="s001")}, [], None, mode="text")
        return gc.calls

    first = run()
    assert len(first) == 1 and "TRANSLATOR EXAMPLES" not in first[0]["prompt"]
    assert run() == []  # checkpoint hit
    _seed_corpus()
    second = run()
    assert len(second) == 1 and "돈 내놔" in second[0]["prompt"] and "돈 어딨냐고" not in second[0]["prompt"]
    assert run() == []
    memory.add_video("other", [Pair("other", 0, "Show me the money!", "돈 보여줘")])
    third = run()
    assert len(third) == 1 and "돈 보여줘" in third[0]["prompt"]


def test_memory_tag_tracks_visible_corpus():
    _seed_corpus()
    s = SimpleNamespace(memory_k=12)
    tag = translate.memory_tag(s, "test-video")
    assert tag[0] == "memory_k=12"
    memory.add_video("test-video", [Pair("test-video", 0, "Changed.", "바뀜")])
    assert translate.memory_tag(s, "test-video") == tag  # own rows are invisible
    memory.add_video("other", [Pair("other", 0, "Changed.", "바뀜")])
    assert translate.memory_tag(s, "test-video") != tag


# ---------------------------------------------------------------- ablate / cli


def test_ablate_memory_key():
    cfg = ablate.parse_config("m12:model=gemini-x,memory=12")
    assert cfg.memory == 12
    assert ablate.parse_config("m0:model=gemini-x,memory=0").memory == 0
    assert ablate.parse_config("d:model=gemini-x").memory is None
    for bad in ("x:model=m,memory=-1", "x:model=m,memory=lots"):
        with pytest.raises(ValueError):
            ablate.parse_config(bad)
    from substage.config import Settings

    client = ablate._ConfigClient(object(), cfg, Settings())
    assert client.settings.memory_k == 12
    assert ablate._ConfigClient(object(), ablate.parse_config("d:model=m"), Settings()).settings.memory_k == Settings.memory_k


def test_ablate_run_excludes_base_video(tmp_path):
    _seed_corpus()
    sc, cues = _scene_setup()
    base = ablate.BaseInputs(
        cues=cues, scenes=[sc], snapshots={}, obs_by_scene={}, context_map=None, register=None, video=None,
    )
    from substage.config import Settings

    prompts = {}
    for k in (0, 12):
        cfg = ablate.parse_config(f"m{k}:model=gemini-x,memory={k}")
        inner = FakeGemini()
        client = ablate._ConfigClient(inner, cfg, Settings())
        ablate._translate_run(ws(tmp_path, "test-video__arm"), client, cfg, base, [sc], None, {c.id: c for c in cues})
        prompts[k] = inner.calls[0]["prompt"]
    assert "TRANSLATOR EXAMPLES" not in prompts[0]
    assert "돈 내놔" in prompts[12] and "돈 어딨냐고" not in prompts[12]


def test_cli_memory_add_and_stats(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    from substage.config import workspace

    w = workspace("video-z__arm")
    save_model(w.path("cues.json"), [cue("c1", 0, 2000, "Run!"), cue("c2", 3000, 5000, "Now.")])
    ref = tmp_path / "ko.srt"
    ref.write_text("1\n00:00:00,000 --> 00:00:02,000\n뛰어!\n\n2\n00:00:03,000 --> 00:00:05,000\n지금\n", encoding="utf-8")
    assert cli.main(["memory", "add", "video-z__arm", "--ref", str(ref)]) == 0
    assert [p.video_id for p in memory.load_pairs()] == ["video-z", "video-z"]
    assert cli.main(["memory", "stats"]) == 0
    out = capsys.readouterr().out
    assert "video-z" in out and "TOTAL" in out
