"""Deterministic offline tests for substage.narrative.

Sibling modules (config/store/gemini) are faked via monkeypatch; the tests
exercise real behavior: causal snapshot folding, twist gating, recent-event
caps, errata application, and viewing-pass plumbing.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import substage.narrative as narrative
from substage.gemini import GeminiBlocked
from substage.models import (
    Callback,
    Character,
    ChunkObservations,
    Cue,
    EmotionChange,
    VideoInfo,
    ContextMap,
    GlossaryEntry,
    KnownBy,
    OnScreenText,
    RegisterEdge,
    RegisterMap,
    RegisterShift,
    Revelation,
    Scene,
    SceneBeat,
    SceneEvent,
    Timeline,
    TimelineErratum,
    TwistFact,
    VoiceCard,
)


# ------------------------------------------------------------------- fakes


@dataclass
class FakeClip:
    file: Any
    start_s: float
    end_s: float
    fps: float = 1.0


class FakeCheckpoints:
    """In-memory checkpoint store honoring the (key, input_hash) contract.

    State is shared across instances keyed by workspace identity so repeated
    calls in one test see earlier puts — mirroring the real on-disk store.
    """

    _stores: dict[int, dict[tuple[str, str, str], Any]] = {}

    def __init__(self, ws, stage):
        self.stage = stage
        self.store = self._stores.setdefault(id(ws), {})

    def get(self, key, input_hash, cls):
        return self.store.get((self.stage, key, input_hash))

    def put(self, key, input_hash, obj):
        self.store[(self.stage, key, input_hash)] = obj

    def invalidate(self, key):
        for k in [k for k in self.store if k[0] == self.stage and k[1] == key]:
            del self.store[k]


class FakeGC:
    """Stands in for GeminiClient.generate; returns queued objects by stage."""

    def __init__(self, model: str = "fake-main"):
        self.calls: list[dict] = []
        self._by_stage: dict[str, Any] = {}
        self.settings = SimpleNamespace(model_main=model)

    def respond(self, stage: str, obj):
        self._by_stage[stage] = obj
        return self

    def generate(self, *, stage, prompt, schema, clips=(), images=(), model=None,
                 thinking_level=None, system=None):
        self.calls.append(
            {"stage": stage, "prompt": prompt, "clips": list(clips), "schema": schema}
        )
        outcome = self._by_stage[stage]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _fake_hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:16]


class FakeWS:
    def __init__(self, tmp: Path):
        self.tmp = tmp

    def path(self, name: str) -> Path:
        return self.tmp / name


@pytest.fixture(autouse=True)
def patch_infra(monkeypatch):
    FakeCheckpoints._stores.clear()
    monkeypatch.setattr(narrative, "Checkpoints", FakeCheckpoints)
    monkeypatch.setattr(narrative, "Clip", FakeClip)
    monkeypatch.setattr(narrative, "input_hash", _fake_hash)


# ------------------------------------------------------------------ helpers


def _cue(n: int, text: str = "line") -> Cue:
    return Cue(id=f"c{n:04d}", start_ms=n * 10_000, end_ms=n * 10_000 + 2000,
               text=text, raw=text)


def _scene(n: int, cue_ns: list[int]) -> Scene:
    ids = [f"c{c:04d}" for c in cue_ns]
    return Scene(id=f"s{n:03d}", start_ms=cue_ns[0] * 10_000,
                 end_ms=cue_ns[-1] * 10_000 + 2000, cue_ids=ids)


def _char(cid: str, en: str) -> Character:
    return Character(id=cid, en_name=en, ko_name=en, persona="p",
                     voice=VoiceCard(idiolect=f"{cid}-voice"))


# ------------------------------------------------------------ S2v viewing


def test_viewing_pass_clip_offsets_and_prompt(tmp_path):
    gc = FakeGC().respond(
        "s2v:v01",
        ChunkObservations(chunk_id="WRONG", start_ms=0, end_ms=0, summary="s"),
    )
    video = VideoInfo(video_id="f", title="SampleVideo", year=2000, video_path="v",
                    duration_ms=120_000, audio_stream=1, en_sub_path="s",
                    en_sub_origin="embedded:2")
    cues = [_cue(1, "hello there"), _cue(2, "general")]
    cues[1].lyric = True
    chunks = [("v01", 0, 60_000, ["s001"])]
    proxy = type("UF", (), {"uri": "gs://x", "name": "n", "mime_type": "video/mp4"})()

    out = narrative.viewing_pass(FakeWS(tmp_path), gc, video, proxy, cues, chunks)

    assert len(gc.calls) == 1
    clip = gc.calls[0]["clips"][0]
    assert (clip.start_s, clip.end_s) == (0.0, 60.0)  # absolute video seconds
    assert "c0001" in gc.calls[0]["prompt"] and "hello there" in gc.calls[0]["prompt"]
    assert "lyric_plot_relevant" in gc.calls[0]["prompt"]
    assert "c0002|20000-22000|-|general ♪" in gc.calls[0]["prompt"]
    # the clip window is authoritative, not the model's echo
    assert out[0].chunk_id == "v01" and out[0].end_ms == 60_000


def test_viewing_pass_checkpoints_per_chunk(tmp_path):
    gc = FakeGC().respond(
        "s2v:v01", ChunkObservations(chunk_id="v01", start_ms=0, end_ms=0, summary="one")
    )
    video = VideoInfo(video_id="f", title="M", year=None, video_path="v",
                    duration_ms=10_000, audio_stream=0, en_sub_path="s",
                    en_sub_origin="user")
    proxy = type("UF", (), {"uri": "u", "name": "n", "mime_type": "m"})()
    chunks = [("v01", 0, 10_000, [])]
    ws = FakeWS(tmp_path)

    narrative.viewing_pass(ws, gc, video, proxy, [], chunks)
    narrative.viewing_pass(ws, gc, video, proxy, [], chunks)
    assert len(gc.calls) == 1  # second run served from checkpoint


def _viewing_video() -> VideoInfo:
    return VideoInfo(video_id="f", title="M", year=None, video_path="v",
                    duration_ms=900_000, audio_stream=0, en_sub_path="s",
                    en_sub_origin="user")


def test_viewing_prompt_asks_for_kinded_on_screen_text(tmp_path):
    gc = FakeGC().respond(
        "s2v:v01", ChunkObservations(chunk_id="v01", start_ms=0, end_ms=0)
    )
    proxy = type("UF", (), {"uri": "u", "name": "n", "mime_type": "m"})()
    narrative.viewing_pass(FakeWS(tmp_path), gc, _viewing_video(), proxy, [],
                           [("v01", 0, 60_000, [])])
    prompt = gc.calls[0]["prompt"]
    for kind in ("[sign]", "[caption]", "[title]", "[credit]", "[dedication]",
                 "[spoken]", "[logo]", "[scroll]"):
        assert kind in prompt
    assert "absolute video ms" in prompt


def test_viewing_pass_on_screen_text_on_video_clock(tmp_path):
    """Clip-relative on-screen-text times are shifted to video ms; spans are
    clamped to the clip window and empty texts dropped."""
    obs = ChunkObservations(chunk_id="v02", start_ms=0, end_ms=0, on_screen_text=[
        OnScreenText(start_ms=5_000, end_ms=8_000, text="[sign] EXIT"),       # relative
        OnScreenText(start_ms=301_000, end_ms=303_000, text="[caption] DAY 1"),  # absolute
        OnScreenText(start_ms=599_500, end_ms=601_000, text="[sign] LATE"),   # spills past end
        OnScreenText(start_ms=310_000, end_ms=311_000, text="  "),
    ])
    gc = FakeGC().respond("s2v:v02", obs)
    proxy = type("UF", (), {"uri": "u", "name": "n", "mime_type": "m"})()
    out = narrative.viewing_pass(FakeWS(tmp_path), gc, _viewing_video(), proxy, [],
                                 [("v02", 300_000, 600_000, [])])
    spans = [(t.start_ms, t.end_ms, t.text) for t in out[0].on_screen_text]
    assert spans == [
        (305_000, 308_000, "[sign] EXIT"),
        (301_000, 303_000, "[caption] DAY 1"),
        (599_500, 600_000, "[sign] LATE"),
    ]


def test_viewing_pass_checkpoint_keyed_on_prompt(tmp_path, monkeypatch):
    """A changed viewing prompt re-observes instead of serving stale output."""
    gc = FakeGC().respond(
        "s2v:v01", ChunkObservations(chunk_id="v01", start_ms=0, end_ms=0)
    )
    proxy = type("UF", (), {"uri": "u", "name": "n", "mime_type": "m"})()
    ws = FakeWS(tmp_path)
    chunks = [("v01", 0, 10_000, [])]
    narrative.viewing_pass(ws, gc, _viewing_video(), proxy, [], chunks)
    real = narrative._prompt
    monkeypatch.setattr(narrative, "_prompt", lambda n: real(n) + "\nnew rule")
    narrative.viewing_pass(ws, gc, _viewing_video(), proxy, [], chunks)
    assert len(gc.calls) == 2


# ------------------------------------------------------- S2c snapshot fold


@pytest.fixture
def causal_world():
    """c0001..c0060 across s001..s006 (10 cues/scene).

    Twist reveal c0049 lives in s005; a knowledge revelation lands at c0015
    in s002.
    """
    cues = [_cue(n) for n in range(1, 61)]
    scenes = [_scene(n, list(range((n - 1) * 10 + 1, n * 10 + 1))) for n in range(1, 7)]
    fm = ContextMap(
        synopsis="s", tone_baseline="t",
        characters=[_char("peter", "Alex Mercer"), _char("tony", "Tony Stark"),
                    _char("vance", "Victor Vance")],
        twist_facts=[
            TwistFact(fact="Vance is Liz's father",
                      audience_reveal_cue="c0049",
                      known_by=[KnownBy(character="vance", from_cue="c0001")]),
        ],
        callbacks=[Callback(motif="with great power", setup_cues=["c0002"],
                            payoff_cues=["c0055"])],
        glossary=[GlossaryEntry(en="Stark", ko="스타크")],
        ambiguous=[],
    )
    beats = [
        SceneBeat(scene_id="s001", beat="intro", chars_present=["peter", "tony"],
                  events=[SceneEvent(cue_id=f"c{n:04d}", description=f"ev{n}",
                                     characters=["peter"]) for n in range(1, 5)]),
        SceneBeat(scene_id="s002", beat="b2", chars_present=["peter"],
                  events=[SceneEvent(cue_id=f"c{n:04d}", description=f"ev{n}",
                                     characters=["peter"]) for n in range(11, 14)],
                  revelations=[Revelation(cue_id="c0015", fact="Tony bankrolls the suit",
                                          to_whom=["peter"])]),
        SceneBeat(scene_id="s003", beat="b3", chars_present=["peter", "vance"]),
        SceneBeat(scene_id="s004", beat="b4", chars_present=["vance"]),
        SceneBeat(scene_id="s005", beat="reveal", chars_present=["peter", "tony", "vance"]),
        SceneBeat(scene_id="s006", beat="b6", chars_present=["peter", "tony", "vance"]),
    ]
    tl = Timeline(scenes=beats)
    return fm, tl, scenes, cues


def _edge(spk: str, adr: str, level: str, shifts=(), vocative="none", pronoun="none") -> RegisterEdge:
    return RegisterEdge(
        speaker=spk, addressee=adr, relation="mentor", apparent_generation="addressee_older",
        history="met_as_child", stance="civil", step=6, level=level, shifts=list(shifts),
        vocative=vocative, pronoun=pronoun, evidence_cues=["c0001", "c0002"], basis="b",
    )


@pytest.fixture
def register_world(causal_world):
    fm, tl, scenes, cues = causal_world
    rmap = RegisterMap(edges=[
        _edge("peter", "tony", "haeyo", vocative="스타크 씨",
              shifts=[RegisterShift(at_cue="c0050", level="hae", reason="relationship_change")]),
        _edge("vance", "peter", "hae", pronoun="너"),
        _edge("peter", "group", "haeyo"),
        _edge("tony", "peter", "hae"),
    ])
    pairs = {
        "c0001": ("peter", "tony"),
        "c0012": ("peter", "group"),
        "c0035": ("vance", "peter"),  # s004: chars_present lists only vance
        "c0045": ("peter", "tony"),
        "c0055": ("peter", "tony"),
    }
    return fm, tl, scenes, cues, rmap, pairs


def test_registers_follow_cue_speakers_not_presence(register_world):
    fm, tl, scenes, cues, rmap, pairs = register_world
    snaps = narrative.materialize_snapshots(fm, tl, scenes, cues, rmap, pairs)

    # s001: peter speaks to tony -> that edge; tony says nothing, so tony>peter
    # is absent even though tony is present in the beat
    assert set(snaps["s001"].registers) == {"peter>tony", "peter>group"}
    s1 = snaps["s001"].registers["peter>tony"]
    assert (s1.level, s1.vocative, s1.shifts) == ("haeyo", "스타크 씨", [])
    # s002: peter addresses the group only; peter>tony rides along (speaker in
    # scene, addressee not) is NOT delivered
    assert set(snaps["s002"].registers) == {"peter>group"}
    # s004: the beat lists only vance, but the line is vance -> peter
    assert set(snaps["s004"].registers) == {"vance>peter"}
    assert snaps["s004"].registers["vance>peter"].pronoun == "너"
    # s003: nobody speaks an addressed line -> no rows
    assert snaps["s003"].registers == {}


def test_register_shifts_ride_with_overlapping_scenes(register_world):
    fm, tl, scenes, cues, rmap, pairs = register_world
    snaps = narrative.materialize_snapshots(fm, tl, scenes, cues, rmap, pairs)
    pos = {c.id: i for i, c in enumerate(cues)}

    s5 = snaps["s005"].registers["peter>tony"]
    assert [sh.at_cue for sh in s5.shifts] == ["c0050"]
    assert s5.level_at("c0045", pos) == "haeyo" and s5.level_at("c0050", pos) == "hae"
    # the open-ended shift still holds in s006 (span overlaps the scene)
    s6 = snaps["s006"].registers["peter>tony"]
    assert s6.level == "haeyo" and s6.level_at("c0055", pos) == "hae"


def test_registers_between_hinted_speakers(register_world):
    fm, tl, scenes, cues, rmap, pairs = register_world
    # s003: peter and tony both speak (SDH hints) without an observed addressee
    cues[21].speaker_hint = "PETER"
    cues[22].speaker_hint = "Tony Stark"
    snaps = narrative.materialize_snapshots(fm, tl, scenes, cues, rmap, pairs)
    assert set(snaps["s003"].registers) == {"peter>tony", "tony>peter", "peter>group"}


def test_snapshot_character_names(causal_world):
    fm, tl, scenes, cues = causal_world
    fm.characters[0].ko_name = "피터 파커"      # peter
    fm.characters[1].ko_name = "토니 스타크"    # tony
    fm.characters[2].ko_name = "빅터 밴스"  # vance
    fm.characters[2].aliases = ["Falcon"]
    # c0015 sits in s002 (peter only); its hint names absent vance by alias —
    # resolvable names join character_names, unresolvable parts are dropped
    cues[14].speaker_hint = "FALCON/NOBODY"
    snaps = narrative.materialize_snapshots(fm, tl, scenes, cues)

    assert snaps["s001"].character_names == {
        "peter": "피터 파커",
        "tony": "토니 스타크",
    }
    assert snaps["s002"].character_names == {
        "peter": "피터 파커",
        "vance": "빅터 밴스",
    }
    # present-only when no cue names anyone else
    assert snaps["s004"].character_names == {"vance": "빅터 밴스"}


def test_snapshot_video_facts_and_personas(causal_world):
    """Every snapshot carries the video's tone and rating plus a one-line persona
    for each named character in the scene, never for absent ones."""
    fm, tl, scenes, cues = causal_world
    fm.tone_baseline, fm.rating = "light teen comedy", "PG-13"
    personas = {c.id: f"{c.id} persona\n  second line" for c in fm.characters}
    for c in fm.characters:
        c.persona = personas[c.id]
    snaps = narrative.materialize_snapshots(fm, tl, scenes, cues)

    s1 = snaps["s001"]
    assert (s1.tone_baseline, s1.rating) == ("light teen comedy", "PG-13")
    assert s1.personas == {
        "peter": "peter persona second line | talks: peter-voice",
        "tony": "tony persona second line | talks: tony-voice",
    }
    assert set(snaps["s004"].personas) == set(snaps["s004"].character_names)
    # lite keeps them: attitude facts, not register prescriptions
    assert s1.for_prompt("lite").personas == s1.personas
    assert s1.for_prompt("lite").tone_baseline == "light teen comedy"



def test_twist_gating(causal_world):
    fm, tl, scenes, cues = causal_world
    snaps = narrative.materialize_snapshots(fm, tl, scenes, cues)

    # before the reveal cue the fact never appears as knowledge — not even for
    # the character who secretly knows it — only as preserve_ambiguity flags
    for sid in ("s001", "s003", "s004", "s005"):
        s = snaps[sid]
        for state in s.characters.values():
            assert not any("Vance" in k for k in state.knows)
        assert any("c0049" in a.note for a in s.preserve_ambiguity)
    # the reveal lands as a delta revelation at c0049 inside s005
    deltas = {d.cue_id: d for d in snaps["s005"].cue_deltas}
    assert deltas["c0049"].revelations[0].fact == "Vance is Liz's father"
    # afterwards: public — no flag, and the pre-knowing character's gate lifts
    s6 = snaps["s006"]
    assert all("c0049" not in a.note for a in s6.preserve_ambiguity)
    assert s6.characters["vance"].knows == ["Vance is Liz's father"]


def test_knowledge_only_from_prior_revelations(causal_world):
    fm, tl, scenes, cues = causal_world
    snaps = narrative.materialize_snapshots(fm, tl, scenes, cues)

    # revelation at c0015 inside s002: absent from s002's entry state,
    # present from s003 on
    assert snaps["s002"].characters["peter"].knows == []
    assert snaps["s003"].characters["peter"].knows == ["Tony bankrolls the suit"]
    deltas = {d.cue_id: d for d in snaps["s002"].cue_deltas}
    assert deltas["c0015"].revelations[0].fact == "Tony bankrolls the suit"


def test_recent_events_capped_at_five(causal_world):
    fm, tl, scenes, cues = causal_world
    snaps = narrative.materialize_snapshots(fm, tl, scenes, cues)

    # peter had 4 events in s001 + 3 in s002 -> 5 most recent at s003 entry
    p3 = snaps["s003"].characters["peter"]
    assert len(p3.recent_events) == 5
    assert p3.recent_events[0].startswith("c0003")  # ev1, ev2 dropped
    assert p3.recent_events[-1].startswith("c0013")


def test_present_only_scoped_state(causal_world):
    fm, tl, scenes, cues = causal_world
    snaps = narrative.materialize_snapshots(fm, tl, scenes, cues)

    assert set(snaps["s001"].characters) == {"peter", "tony"}
    assert set(snaps["s001"].voice_cards) == {"peter", "tony"}
    assert "vance" not in snaps["s001"].characters


def test_glossary_and_callbacks_live(causal_world):
    fm, tl, scenes, cues = causal_world
    cues[5].text = "Mr Stark is here"  # "Stark" appears only in s001
    snaps = narrative.materialize_snapshots(fm, tl, scenes, cues)

    assert [g.en for g in snaps["s001"].glossary] == ["Stark"]
    assert snaps["s002"].glossary == []
    # setup c0002 seen before s003, payoff c0055 still pending -> live
    assert snaps["s003"].callbacks_live[0].motif == "with great power"
    # payoff inside s006 -> live there too
    assert snaps["s006"].callbacks_live[0].motif == "with great power"


def test_emotion_changes_are_deltas_and_carry(causal_world):
    fm, tl, scenes, cues = causal_world
    tl.scenes[2].emotion_changes.append(
        EmotionChange(cue_id="c0025", character="peter",
                      from_state="calm", to_state="shaken")
    )
    snaps = narrative.materialize_snapshots(fm, tl, scenes, cues)

    assert snaps["s003"].characters["peter"].emotion is None
    deltas = {d.cue_id: d for d in snaps["s003"].cue_deltas}
    assert deltas["c0025"].emotion_changes[0].to_state == "shaken"
    assert snaps["s005"].characters["peter"].emotion == "shaken"


# ---------------------------------------------------------------- errata


def test_apply_errata_only_accepted(causal_world):
    fm, tl, scenes, cues = causal_world
    errata = [
        TimelineErratum(cue_ids=["c0015"], claim="Tony did NOT bankroll the suit",
                        evidence="c0016: he denies it"),
        TimelineErratum(cue_ids=["c0050"], claim="shift is wrong", evidence="x"),
    ]
    fm2, tl2 = narrative.apply_errata(fm, tl, errata, accepted_ids={0})

    # the register table is not part of the narrative errata path
    assert not hasattr(fm2, "register_shifts")
    # accepted erratum #0 removed the disputed revelation…
    beat2 = next(b for b in tl2.scenes if b.scene_id == "s002")
    assert all(r.cue_id != "c0015" for r in beat2.revelations)
    # …and recorded the claim as an ERRATUM event on that same beat
    err_events = [e for e in beat2.events if e.description.startswith("ERRATUM:")]
    assert len(err_events) == 1 and err_events[0].cue_id == "c0015"
    # inputs not mutated
    assert any(r.cue_id == "c0015" for r in tl.scenes[1].revelations)
    # re-materialized snapshots reflect the correction
    snaps = narrative.materialize_snapshots(fm2, tl2, scenes, cues)
    p3 = snaps["s003"].characters["peter"]
    assert p3.knows == []
    assert any(e.startswith("ERRATUM:") or "did NOT bankroll" in e
               for e in p3.recent_events)


def test_apply_errata_none_accepted(causal_world):
    fm, tl, scenes, cues = causal_world
    fm2, tl2 = narrative.apply_errata(
        fm, tl, [TimelineErratum(cue_ids=["c0015"], claim="x", evidence="y")],
        accepted_ids=set(),
    )
    assert tl2 == tl and fm2 == fm


# -------------------------------------------------------------- video map


def test_build_context_map_checkpoint(tmp_path):
    gc = FakeGC().respond("s2a", ContextMap(synopsis="syn", tone_baseline="t"))
    video = VideoInfo(video_id="f", title="M", year=2001, video_path="v",
                    duration_ms=1000, audio_stream=0, en_sub_path="s",
                    en_sub_origin="user")
    ws = FakeWS(tmp_path)

    fm = narrative.build_context_map(ws, gc, video, [_cue(1)], [])
    assert fm.synopsis == "syn"
    narrative.build_context_map(ws, gc, video, [_cue(1)], [])
    assert len(gc.calls) == 1
    # transcript reached the prompt in compact form
    assert "c0001|00:10|-|line" in gc.calls[0]["prompt"]


def test_build_timeline_covers_every_scene(tmp_path):
    beats = [SceneBeat(scene_id="s001", beat="one"),
             SceneBeat(scene_id="s002", beat="two")]
    gc = FakeGC().respond("s2b:s001-s003", Timeline(scenes=beats))
    ws = FakeWS(tmp_path)
    fm = ContextMap(synopsis="s", tone_baseline="t", characters=[_char("peter", "Peter")])
    scenes = [_scene(1, [1]), _scene(2, [2]), _scene(3, [3])]

    tl = narrative.build_timeline(ws, gc, fm, scenes, [_cue(1), _cue(2), _cue(3)], [])

    assert [b.scene_id for b in tl.scenes] == ["s001", "s002", "s003"]
    assert tl.scenes[2].beat == ""  # gap filled with an empty beat


def _map_video() -> VideoInfo:
    return VideoInfo(video_id="f", title="M", year=2001, video_path="v",
                    duration_ms=1000, audio_stream=0, en_sub_path="s",
                    en_sub_origin="user")


def test_build_context_map_checkpoint_keyed_on_template_and_model(tmp_path, monkeypatch):
    """A changed context_map.md or model regenerates instead of reusing the
    stored map; an unchanged generator is still served from the checkpoint."""
    ws = FakeWS(tmp_path)
    gc = FakeGC().respond("s2a", ContextMap(synopsis="syn", tone_baseline="t"))
    narrative.build_context_map(ws, gc, _map_video(), [_cue(1)], [])
    narrative.build_context_map(ws, gc, _map_video(), [_cue(1)], [])
    assert len(gc.calls) == 1

    real = narrative._prompt
    monkeypatch.setattr(narrative, "_prompt",
                        lambda n: real(n) + ("\nnew rule" if n == "context_map.md" else ""))
    narrative.build_context_map(ws, gc, _map_video(), [_cue(1)], [])
    assert len(gc.calls) == 2
    assert "new rule" in gc.calls[1]["prompt"]

    gc2 = FakeGC(model="other-model").respond("s2a", ContextMap(synopsis="syn", tone_baseline="t"))
    narrative.build_context_map(ws, gc2, _map_video(), [_cue(1)], [])
    assert len(gc2.calls) == 1


def test_build_timeline_checkpoint_keyed_on_template_and_model(tmp_path, monkeypatch):
    ws = FakeWS(tmp_path)
    fm = ContextMap(synopsis="s", tone_baseline="t")
    scenes = [_scene(1, [1])]
    beat = Timeline(scenes=[SceneBeat(scene_id="s001", beat="one")])
    gc = FakeGC().respond("s2b:s001-s001", beat)
    narrative.build_timeline(ws, gc, fm, scenes, [_cue(1)], [])
    narrative.build_timeline(ws, gc, fm, scenes, [_cue(1)], [])
    assert len(gc.calls) == 1

    real = narrative._prompt
    monkeypatch.setattr(narrative, "_prompt",
                        lambda n: real(n) + ("\nnew rule" if n == "timeline.md" else ""))
    narrative.build_timeline(ws, gc, fm, scenes, [_cue(1)], [])
    assert len(gc.calls) == 2

    gc2 = FakeGC(model="other-model").respond("s2b:s001-s001", beat)
    narrative.build_timeline(ws, gc2, fm, scenes, [_cue(1)], [])
    assert len(gc2.calls) == 1


# ------------------------------------------------------------ blocked input


def test_viewing_pass_blocked_chunk_splits_clip(tmp_path, caplog):
    """A refused viewing clip is split in half recursively; ~60s leaves degrade
    to empty 'blocked' observations instead of aborting the pass."""
    gc = FakeGC()
    for st in ("s2v:v01", "s2v:v01a", "s2v:v01b"):
        gc.respond(st, GeminiBlocked("x", "prompt blocked"))
    for st in ("s2v:v01aa", "s2v:v01ab", "s2v:v01ba", "s2v:v01bb"):
        gc.respond(st, GeminiBlocked("x", "prompt blocked"))
    ws = FakeWS(tmp_path)
    video = VideoInfo(video_id="f", title="M", year=2001, video_path="v",
                    duration_ms=300_000, audio_stream=0, en_sub_path="s",
                    en_sub_origin="user")
    proxy = SimpleNamespace(name="p", uri="gs://p", mime_type="video/mp4")

    with caplog.at_level(logging.WARNING, logger="substage.narrative"):
        out = narrative.viewing_pass(
            ws, gc, video, proxy, [], [("v01", 0, 300_000, ["s001"])]
        )

    assert len(out) == 1
    obs = out[0]
    assert obs.chunk_id == "v01" and obs.start_ms == 0 and obs.end_ms == 300_000
    assert obs.cues == [] and obs.events == [] and obs.on_screen_text == []
    assert "blocked" in obs.summary
    stages = {c["stage"] for c in gc.calls}
    assert "s2v:v01" in stages
    # 300s -> 2x150s -> 4x75s, all refused: 1 + 2 + 4 calls
    assert len(gc.calls) == 7
    assert "blocked" in caplog.text


def test_build_context_map_blocked_retries_transcript_only(tmp_path):
    gc = (FakeGC()
          .respond("s2a", GeminiBlocked("s2a", "prompt blocked"))
          .respond("s2a:transcript", ContextMap(synopsis="syn", tone_baseline="t")))
    ws = FakeWS(tmp_path)
    video = VideoInfo(video_id="f", title="M", year=2001, video_path="v",
                    duration_ms=1000, audio_stream=0, en_sub_path="s",
                    en_sub_origin="user")

    fm = narrative.build_context_map(ws, gc, video, [_cue(1)], [])

    assert fm.synopsis == "syn"
    assert len(gc.calls) == 2
    assert "unavailable" in gc.calls[1]["prompt"]


def test_build_context_map_blocked_twice_raises(tmp_path):
    gc = (FakeGC()
          .respond("s2a", GeminiBlocked("s2a", "prompt blocked"))
          .respond("s2a:transcript", GeminiBlocked("s2a", "prompt blocked")))
    ws = FakeWS(tmp_path)
    video = VideoInfo(video_id="f", title="M", year=2001, video_path="v",
                    duration_ms=1000, audio_stream=0, en_sub_path="s",
                    en_sub_origin="user")

    with pytest.raises(GeminiBlocked):
        narrative.build_context_map(ws, gc, video, [_cue(1)], [])


def test_build_timeline_blocked_batch_bisects(tmp_path, caplog):
    """The full batch is refused; bisection isolates the offending scene, which
    alone gets an empty beat — every other scene keeps its real beat."""
    gc = (FakeGC()
          .respond("s2b:s001-s003", GeminiBlocked("s2b", "prompt blocked"))
          .respond("s2b:s001-s003a", Timeline(scenes=[SceneBeat(scene_id="s001", beat="one")]))
          .respond("s2b:s001-s003b", GeminiBlocked("s2b", "prompt blocked"))
          .respond("s2b:s001-s003ba", Timeline(scenes=[SceneBeat(scene_id="s002", beat="two")]))
          .respond("s2b:s001-s003bb", GeminiBlocked("s2b", "prompt blocked")))
    ws = FakeWS(tmp_path)
    fm = ContextMap(synopsis="s", tone_baseline="t", characters=[_char("peter", "Peter")])
    scenes = [_scene(1, [1]), _scene(2, [2]), _scene(3, [3])]
    cues = [_cue(1), _cue(2), _cue(3)]

    with caplog.at_level(logging.WARNING, logger="substage.narrative"):
        tl = narrative.build_timeline(ws, gc, fm, scenes, cues, [])

    beats = {b.scene_id: b.beat for b in tl.scenes}
    assert beats == {"s001": "one", "s002": "two", "s003": ""}
    assert len(gc.calls) == 5  # 1 blocked batch + 2 halves (one blocked) + 2 quarters
    assert "s003" in caplog.text
