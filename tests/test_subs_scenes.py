"""Deterministic, offline unit tests for subs.py, scenes.py and the
non-network parts of media.py / acquire.py. No ffmpeg/ffprobe/ffsubsync,
no Gemini, no subliminal — all external calls are faked."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from substage import acquire, media, subs
from substage.acquire import AcquireError
from substage.models import Cue
from substage.scenes import gap_audit, group_scenes, viewing_chunks


# ---------------------------------------------------------------- helpers


def _srt(cues: list[tuple[str, str, str]]) -> str:
    """Build SRT text from (start, end, text) tuples."""
    blocks = []
    for i, (start, end, text) in enumerate(cues, 1):
        blocks.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def _write(tmp_path: Path, name: str, content: str, encoding: str = "utf-8") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding=encoding)
    return p


def _cue(start: int, end: int, text: str = "x", **kw) -> Cue:
    return Cue(id="", start_ms=start, end_ms=end, text=text, raw=text, **kw)


# ---------------------------------------------------------------- SDH parse


# SampleVideo SDH-flavoured fixture: sound cue, label, dash+inline tag, lyrics,
# two-line cue, dual speakers.
SRT_SDH = _srt(
    [
        ("00:00:01,000", "00:00:03,000", "(POP MUSIC PLAYING)"),
        ("00:00:05,000", "00:00:08,000", "Kinda lonesome\nback here."),
        ("00:00:10,000", "00:00:13,000", "WADE: I know, right?\nWhose balls did I have to fondle?"),
        ("00:00:14,000", "00:00:16,000", "- [grunts] Hey!"),
        ("00:00:18,000", "00:00:21,000", "♪ Angel of the morning ♪"),
        ("00:00:23,000", "00:00:26,000", "- Wait.\n- What?"),
        ("00:00:28,000", "00:00:31,000", "WADE (V.O.): From the studio\nthat inexplicably sewed his mouth shut."),
    ]
)


def test_ocr_garbled_sdh_lyrics_and_fragments(tmp_path):
    # Real downloaded Part One SDH: ♪ OCR'd as j'/J'/j”, "[♪]" OCR'd as "[M1".
    srt = _srt(
        [
            ("00:00:01,000", "00:00:02,000", "J' yeah, you got satin shoes j'"),
            ("00:00:03,000", "00:00:04,000", "J' yeah, you gotj'"),
            ("00:00:05,000", "00:00:06,000", "J' can't you hear me knocking\non your window j”"),
            ("00:00:07,000", "00:00:08,000", "[M1"),
            ("00:00:09,000", "00:00:10,000", "Agent J is here."),
        ]
    )
    cues = subs.load_cues(_write(tmp_path, "en.srt", srt))
    assert [c.lyric for c in cues] == [True, True, True, False, False]
    assert cues[0].text == "yeah, you got satin shoes"
    assert cues[1].text == "yeah, you got"
    assert cues[3].text == "" and cues[3].sound_cues == ["[m1"]
    assert cues[4].text == "Agent J is here."


def test_sdh_parsing(tmp_path):
    cues = subs.load_cues(_write(tmp_path, "en.srt", SRT_SDH))

    assert [c.id for c in cues] == [f"c{i:04d}" for i in range(1, 8)]
    assert cues[0].start_ms == 1000 and cues[0].end_ms == 3000

    # Pure sound cue: kept, empty dialogue text.
    assert cues[0].text == ""
    assert cues[0].sound_cues == ["(pop music playing)"]

    # Plain multi-line dialogue: untouched, no false label.
    assert cues[1].text == "Kinda lonesome\nback here."
    assert cues[1].speaker_hint is None
    assert cues[1].sound_cues == []

    # Speaker label stripped into hint, dialogue kept clean.
    assert cues[2].speaker_hint == "WADE"
    assert cues[2].text == "I know, right?\nWhose balls did I have to fondle?"

    # A lone dashed line is one speaker: the dash is dropped; inline [grunts] becomes a sound cue.
    assert cues[3].text == "Hey!"
    assert cues[3].sound_cues == ["[grunts]"]
    # A dashed turn that is only a sound description is not a speaker turn.
    assert subs._parse_event_text("- Come on, kid.\n- [grunts]")[0] == "Come on, kid."

    # Lyric marker detected and stripped.
    assert cues[4].lyric is True
    assert "♪" not in cues[4].text
    assert "Angel of the morning" in cues[4].text

    # Dual-speaker dashes survive verbatim.
    assert cues[5].text == "- Wait.\n- What?"

    # Label with parenthesized delivery tag: name + sound cue.
    assert cues[6].speaker_hint == "WADE"
    assert "(v.o.)" in cues[6].sound_cues
    assert cues[6].text == "From the studio\nthat inexplicably sewed his mouth shut."


def test_html_tags_and_bom(tmp_path):
    content = "\ufeff" + _srt(
        [("00:00:01,000", "00:00:02,000", "<i>So that's a no?</i>")]
    )
    cues = subs.load_cues(_write(tmp_path, "en.srt", content))
    assert cues[0].text == "So that's a no?"
    assert cues[0].raw == "So that's a no?" or "<i>" in cues[0].raw


def test_cp949_fallback(tmp_path):
    # A .smi-style body that only decodes as cp949 (Korean bytes).
    srt_like = _srt([("00:00:01,000", "00:00:02,000", "자막 테스트")])
    p = tmp_path / "ko.smi"
    p.write_bytes(srt_like.encode("cp949"))
    cues = subs.load_cues(p)
    assert len(cues) == 1
    assert cues[0].text == "자막 테스트"


def test_merge_forced_dedupes(tmp_path):
    main = subs.load_cues(
        _write(
            tmp_path,
            "en.srt",
            _srt(
                [
                    ("00:00:01,000", "00:00:03,000", "Hello there."),
                    ("00:00:05,000", "00:00:07,000", "DUPONT CIRCLE"),
                ]
            ),
        )
    )
    forced = subs.load_cues(
        _write(
            tmp_path,
            "en.forced.srt",
            _srt(
                [
                    ("00:00:05,100", "00:00:06,900", "DUPONT CIRCLE"),  # dup of c0002
                    ("00:00:10,000", "00:00:12,000", "AIRPORT DEPARTURES"),  # new
                    ("00:00:12,500", "00:00:14,000", "Hello there."),  # non-overlap -> keep
                ]
            ),
        ),
        forced=True,
        source="forced",
    )
    merged = subs.merge_forced(main, forced)

    assert len(merged) == 4
    assert [c.id for c in merged] == ["c0001", "c0002", "c0003", "c0004"]
    assert merged[1].text == "DUPONT CIRCLE"
    assert not merged[1].forced  # main cue wins over its forced duplicate
    assert merged[2].forced and merged[2].source == "forced"
    assert merged[2].text == "AIRPORT DEPARTURES"
    assert merged[3].text == "Hello there."
    assert merged[3].forced


def test_write_srt_roundtrip(tmp_path):
    cues = subs.load_cues(_write(tmp_path, "en.srt", SRT_SDH))
    out = tmp_path / "out.ko.srt"
    rows = [(c.start_ms, c.end_ms, ko) for c, ko in ((cues[1], "여긴 좀 외롭네."), (cues[4], "♪ 아침의 천사 ♪"))]
    subs.write_srt(rows, out)
    back = subs.load_cues(out)
    assert len(back) == 2
    assert back[0].start_ms == 5000 and back[0].end_ms == 8000
    assert back[0].text == "여긴 좀 외롭네."
    assert back[1].lyric is True  # ♪ survives the roundtrip


# ---------------------------------------------------------------- scenes


def _dialogue(n: int, start: int, step: int = 3_000, start_id: int = 1) -> list[Cue]:
    cues = [_cue(start + i * step, start + i * step + 1_500) for i in range(n)]
    for i, c in enumerate(cues):
        c.id = f"c{start_id + i:04d}"
    return cues


def test_group_scenes_splits_on_gap():
    cues = _dialogue(3, 0) + _dialogue(2, 20_000, start_id=4)
    scenes = group_scenes(cues, gap_ms=4_000, max_cues=40)
    assert [s.id for s in scenes] == ["s001", "s002"]
    assert scenes[0].cue_ids == [f"c{i:04d}" for i in range(1, 4)]
    assert scenes[1].cue_ids == [f"c{i:04d}" for i in range(4, 6)]
    assert scenes[0].start_ms == 0 and scenes[0].end_ms == 7_500
    assert scenes[1].start_ms == 20_000


def test_group_scenes_respects_max_cues_and_prefers_largest_gap():
    # 9 cues, 2 s apart; one breathing pause of 6 s between cue 4 and 5.
    cues = _dialogue(9, 0, step=2_000)
    cues[4] = _cue(cues[3].end_ms + 6_000, cues[3].end_ms + 7_500)
    for i in range(5, 9):
        cues[i] = _cue(cues[i - 1].end_ms + 500, cues[i - 1].end_ms + 2_000)

    scenes = group_scenes(cues, gap_ms=10_000, max_cues=6)
    assert len(scenes) == 2
    # Largest gap inside the last third of the first 6 cues is after cue 4.
    assert len(scenes[0].cue_ids) == 4
    assert len(scenes[1].cue_ids) == 5

def test_gap_audit():
    cues = _dialogue(2, 0) + _dialogue(2, 70_000) + _dialogue(1, 300_000)
    gaps = gap_audit(cues, duration_ms=400_000, min_gap_ms=60_000)
    assert (4_500, 70_000) in gaps
    assert (301_500, 400_000) in gaps
    assert all(b - a >= 60_000 for a, b in gaps)


def test_viewing_chunks_cover_duration_and_cut_at_scene_boundaries():
    # 6 scenes: each ~100 s of dialogue (5 s cue cadence) then ~20 s silence.
    cues: list[Cue] = []
    for k in range(6):
        cues += _dialogue(20, start=k * 120_000, step=5_000, start_id=k * 20 + 1)
    scenes = group_scenes(cues, gap_ms=10_000, max_cues=40)
    assert len(scenes) == 6
    duration = 6 * 120_000

    chunks = viewing_chunks(scenes, duration, chunk_ms=250_000)

    # Full coverage, contiguous, ordered.
    assert chunks[0][1] == 0
    assert chunks[-1][2] == duration
    for (_, _, prev_end, _), (_, start, _, _) in zip(chunks, chunks[1:]):
        assert prev_end == start
    # Cuts only at scene boundaries (mid-gap), never inside a scene.
    scene_ids = [s.id for s in scenes]
    seen: list[str] = []
    for _, start, end, ids in chunks:
        assert end - start <= 250_000 + 1  # boundary midpoints add ≤ half a gap
        assert ids
        assert ids[0] == scene_ids[len(seen)]
        seen += ids
        for s in scenes:
            if s.id in ids:
                assert start <= s.start_ms and s.end_ms <= end
    assert seen == scene_ids
    assert chunks[0][0] == "v01"


def test_viewing_chunks_no_scenes():
    assert viewing_chunks([], 500_000, 600_000) == [("v01", 0, 500_000, [])]
    assert viewing_chunks([], 0, 600_000) == []


# ---------------------------------------------------------------- media

# ffprobe JSON modelled on the source's rip: TrueHD 7.1 + AC3 5.1 + two AC3
# commentary tracks, embedded English/SDH/Forced SubRip + a PGS image track.
FFPROBE_JSON = json.dumps(
    {
        "format": {"duration": "6486.400000"},
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "hevc",
             "disposition": {"default": 1}, "tags": {"language": "und"}},
            {"index": 1, "codec_type": "audio", "codec_name": "truehd",
             "channels": 8, "disposition": {"default": 1},
             "tags": {"language": "eng"}},
            {"index": 2, "codec_type": "audio", "codec_name": "ac3",
             "channels": 6, "disposition": {"default": 0},
             "tags": {"language": "eng"}},
            {"index": 3, "codec_type": "audio", "codec_name": "ac3",
             "channels": 2, "disposition": {"default": 0, "comment": 1},
             "tags": {"language": "eng",
                      "title": "Commentary by director Tim Miller"}},
            {"index": 4, "codec_type": "audio", "codec_name": "ac3",
             "channels": 2, "disposition": {"default": 0, "comment": 1},
             "tags": {"language": "eng", "title": "Commentary"}},
            {"index": 5, "codec_type": "subtitle", "codec_name": "subrip",
             "disposition": {"default": 1}, "tags": {"language": "eng"}},
            {"index": 6, "codec_type": "subtitle", "codec_name": "subrip",
             "disposition": {"default": 0, "hearing_impaired": 1},
             "tags": {"language": "eng", "title": "English SDH"}},
            {"index": 7, "codec_type": "subtitle", "codec_name": "subrip",
             "disposition": {"default": 0, "forced": 1},
             "tags": {"language": "eng", "title": "English Forced"}},
            {"index": 8, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle",
             "disposition": {"default": 0}, "tags": {"language": "kor"}},
        ],
    }
)


def _fake_ffprobe(payload: str):
    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr="")
    return run


def test_probe_parses_dispositions(monkeypatch, tmp_path):
    video = tmp_path / "video.mkv"
    video.write_bytes(b"")
    monkeypatch.setattr(media.subprocess, "run", _fake_ffprobe(FFPROBE_JSON))
    streams, duration_ms = media.probe(video)

    assert duration_ms == 6_486_400
    sdh = next(s for s in streams if s.index == 6)
    assert sdh.hearing_impaired and not sdh.forced
    forced = next(s for s in streams if s.index == 7)
    assert forced.forced
    # Title heuristics fire too.
    assert streams[6].title == "English SDH"


def test_probe_title_heuristics(monkeypatch, tmp_path):
    payload = json.dumps(
        {
            "format": {"duration": "100.0"},
            "streams": [
                {"index": 0, "codec_type": "subtitle", "codec_name": "subrip",
                 "disposition": {}, "tags": {"language": "eng", "title": "Forced"}},
                {"index": 1, "codec_type": "subtitle", "codec_name": "subrip",
                 "disposition": {}, "tags": {"language": "eng", "title": "CC"}},
            ],
        }
    )
    video = tmp_path / "f.mkv"
    video.write_bytes(b"")
    monkeypatch.setattr(media.subprocess, "run", _fake_ffprobe(payload))
    streams, dur = media.probe(video)
    assert streams[0].forced
    assert streams[1].hearing_impaired
    assert dur == 100_000


def test_pick_dialogue_audio_skips_commentary():
    streams, _ = _streams_from_fixture()
    pick = media.pick_dialogue_audio(streams)
    assert pick.index == 1  # TrueHD 7.1, most channels, not commentary


def test_pick_dialogue_audio_commentary_with_more_channels_still_loses():
    streams, _ = _streams_from_fixture()
    bumped = [
        media.Stream(
            index=s.index, codec_type=s.codec_type, codec_name=s.codec_name,
            language=s.language, title=s.title,
            channels=8 if s.index == 3 else s.channels,
            forced=s.forced, hearing_impaired=s.hearing_impaired,
        )
        for s in streams
    ]
    pick = media.pick_dialogue_audio(bumped)
    assert pick.index == 1


def test_pick_dialogue_audio_untagged_fallback():
    def audio(index, language, channels=6):
        return media.Stream(
            index=index, codec_type="audio", codec_name="aac", language=language,
            title=None, channels=channels, forced=False, hearing_impaired=False,
        )

    assert media.pick_dialogue_audio([audio(1, None)]).index == 1
    assert media.pick_dialogue_audio([audio(1, "und"), audio(2, "eng", 2)]).index == 2
    with pytest.raises(media.MediaError):
        media.pick_dialogue_audio([audio(1, "fre")])


def test_pick_english_text_subs_roles():
    streams, _ = _streams_from_fixture()
    picks = media.pick_english_text_subs(streams)
    assert picks["main"].index == 5
    assert picks["sdh"].index == 6
    assert picks["forced"].index == 7
    # PGS (image) subs never qualify even with eng tags.
    assert all(s.codec_name == "subrip" for s in picks.values())


def _streams_from_fixture():
    info = json.loads(FFPROBE_JSON)
    streams = []
    for st in info["streams"]:
        tags = st.get("tags") or {}
        disp = st.get("disposition") or {}
        title = tags.get("title")
        streams.append(
            media.Stream(
                index=st["index"], codec_type=st["codec_type"],
                codec_name=st["codec_name"], language=tags.get("language"),
                title=title, channels=st.get("channels"),
                forced=bool(disp.get("forced")), hearing_impaired=bool(disp.get("hearing_impaired")),
            )
        )
    return streams, 6_486_400


# ---------------------------------------------------------------- acquire


class _FakeWS:
    def __init__(self, root: Path):
        self.data_dir = root / "data"
        self.cache_dir = root / "cache"

    def path(self, name: str) -> Path:
        p = self.data_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def cache(self, name: str) -> Path:
        p = self.cache_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


class _FakeGC:
    """Returns one transcript per call (one audio clip per call); records kwargs."""

    def __init__(self):
        self.settings = SimpleNamespace(model_light="lite-model")
        self.calls: list[dict] = []
        self._queue: list[str] = []

    def push(self, transcripts: list[str]) -> None:
        self._queue.extend(transcripts)

    def generate(self, *, stage, prompt, schema, audio=(), model=None, **kw):
        self.calls.append({"stage": stage, "model": model, "n_audio": len(audio)})
        return schema(transcript=self._queue.pop(0))


def _touch(path: Path, *a, **kw) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"wav")
    return path


def _pg_only_streams() -> list[media.Stream]:
    """One dialogue audio + only image (PGS) subs: forces the download path."""
    return [
        media.Stream(0, "video", "hevc", None, None, None, False, False),
        media.Stream(1, "audio", "truehd", "eng", None, 8, False, False),
        media.Stream(2, "subtitle", "hdmv_pgs_subtitle", "eng", None, None, False, False),
        media.Stream(3, "audio", "ac3", "eng", "Commentary", 2, False, False),
    ]


def test_acquire_download_wrong_video_rejected_next_accepted(tmp_path):
    wrong = tmp_path / "wrong.srt"  # Action Series (2002) lines
    right = tmp_path / "right.srt"  # Part One lines
    wrong.write_text(
        _srt(
            [
                ("00:10:00,000", "00:10:02,000", "With great power comes great responsibility."),
                ("00:10:30,000", "00:10:32,000", "Go web go!"),
                ("00:11:00,000", "00:11:02,000", "I missed the part where that's my problem."),
                ("00:11:30,000", "00:11:32,000", "It's you who's out, Gobby."),
                ("00:12:00,000", "00:12:02,000", "Pizza time."),
                ("00:12:30,000", "00:12:32,000", "I had to beat an old lady with a stick."),
            ]
        )
    )
    right.write_text(
        _srt(
            [
                ("00:10:00,000", "00:10:02,000", "Okay, sorta. A weird thing happened."),
                ("00:10:30,000", "00:10:32,000", "I was in the car with my Aunt May."),
                ("00:11:00,000", "00:11:02,000", "Forget the flying monster guy."),
                ("00:11:30,000", "00:11:32,000", "There are people who handle this sort of thing."),
                ("00:12:00,000", "00:12:02,000", "You might even say I was a science teacher."),
                ("00:12:30,000", "00:12:32,000", "Can you spit venom?"),
            ]
        )
    )

    gc = _FakeGC()
    # Candidate 1's windows: model hears the real Part One audio -> wrong video -> ~0 overlap.
    gc.push(["okay sorta a weird thing happened"] * 6)
    # Candidate 2's windows: same audio, matching subs -> high overlap.
    gc.push(
        [
            "okay sorta a weird thing happened i was in the car with my aunt may",
            "i was in the car with my aunt may forget the flying monster guy",
            "forget the flying monster guy there are people who handle this sort of thing",
            "there are people who handle this sort of thing you might even say",
            "you might even say i was a science teacher can you spit venom",
            "can you spit venom",
        ]
    )

    def fake_sync(video, sub_in, sub_out):
        sub_out.write_bytes(Path(sub_in).read_bytes())
        return True, -15.4, 1.043  # high score, still wrong video!

    en, forced, origin, offset, scale = acquire.acquire_english(
        _FakeWS(tmp_path),
        gc,
        tmp_path / "SamplePartOne.mkv",
        _pg_only_streams(),
        candidate_lister=lambda ws, v: [
            acquire._Candidate(wrong, "podnapisi", False, 300, "wrong-video"),
            acquire._Candidate(right, "opensubtitles", True, 250, "right-video"),
        ],
        sync_runner=fake_sync,
        audio_cutter=lambda video, audio, s, e, out: _touch(out),
    )

    # Wrong video skipped despite the good sync score; second candidate wins.
    assert en.read_text() == right.read_text()
    assert origin == "downloaded:opensubtitles"
    assert offset == -15.4 and scale == 1.043
    assert forced is None
    assert len(gc.calls) == 12  # two candidates × six single-clip windows
    assert all(c["stage"] == "acquire-verify" for c in gc.calls)
    assert all(c["model"] == "lite-model" for c in gc.calls)
    assert all(c["n_audio"] == 1 for c in gc.calls)


def test_acquire_all_candidates_fail_raises(tmp_path):
    bad = tmp_path / "bad.srt"
    bad.write_text(_srt([("00:00:01,000", "00:00:02,000", "Unrelated dialogue.")]))

    gc = _FakeGC()
    gc.push(["something else entirely"] * 6)

    def fail_sync(video, sub_in, sub_out):
        sub_out.write_bytes(Path(sub_in).read_bytes())
        return False, None, None  # low-quality alignment

    with pytest.raises(AcquireError) as exc:
        acquire.acquire_english(
            _FakeWS(tmp_path),
            gc,
            tmp_path / "f.mkv",
            _pg_only_streams(),
            candidate_lister=lambda ws, v: [bad],
            sync_runner=fail_sync,
            audio_cutter=lambda video, audio, s, e, out: _touch(out),
        )
    assert "--en-sub" in str(exc.value)
    assert gc.calls == []  # never reached verification


def test_acquire_embedded_shortcircuits(tmp_path, monkeypatch):
    streams = _pg_only_streams() + [
        media.Stream(4, "subtitle", "subrip", "eng", None, None, False, False),
        media.Stream(5, "subtitle", "subrip", "eng", "SDH", None, False, True),
        media.Stream(6, "subtitle", "subrip", "eng", "Forced", None, True, False),
    ]
    calls: list[list[str]] = []

    def fake_extract(video, stream, out):
        calls.append([stream.index])
        Path(out).write_bytes(b"srt")
        return Path(out)

    monkeypatch.setattr(media, "extract_subtitle", fake_extract)
    en, forced, origin, off, sc = acquire.acquire_english(
        _FakeWS(tmp_path), _FakeGC(), tmp_path / "f.mkv", streams
    )

    # SDH preferred over plain main; forced extracted too; no download.
    assert [c[0] for c in calls] == [6, 5]
    assert origin == "embedded:5"
    assert forced is not None and en is not None
    assert off is None and sc is None


def test_title_year_filters():
    video = SimpleNamespace(title="Action Series: Part One", year=2017)
    assert acquire._year_ok(video, 2017)
    assert not acquire._year_ok(video, 2002)
    assert acquire._year_ok(video, None)
    assert not acquire._title_ok(video, "Action Series")  # short substring -> different video
    assert acquire._title_ok(video, "Action Series Part One")
    assert acquire._title_ok(video, "Action Series: Part One (2017)")
    assert acquire._title_ok(video, None)


def test_media_id_match_overrides_guessed_title():
    # guessit reads "Action Series - Part Two (2018)" as title "Action Series".
    video = SimpleNamespace(title="Action Series", year=2018, media_id="tt4633694")
    same = SimpleNamespace(media_id="tt4633694")
    other = SimpleNamespace(media_id="tt0145487")
    assert acquire._media_id_match(video, same) is True
    assert acquire._media_id_match(video, other) is False
    assert acquire._media_id_match(video, SimpleNamespace()) is None
    assert acquire._media_id_match(SimpleNamespace(media_id=None), same) is None


def test_parse_sync_log_variants():
    off, scale, low = acquire._parse_sync_log(
        "INFO: offset seconds: -15.400\nINFO: framerate scale factor: 1.043\n"
    )
    assert off == -15.4 and scale == 1.043 and not low

    off, scale, low = acquire._parse_sync_log(
        "WARNING: low-quality alignment detected, skipping sync"
    )
    assert low

    off, scale, low = acquire._parse_sync_log("nothing relevant")
    assert off is None and scale is None and not low
