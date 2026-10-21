from __future__ import annotations

from substage import memory, narrative, refeval, series_state
from substage.models import (
    Character,
    ContextMap,
    Cue,
    GlossaryEntry,
    RegisterEdge,
    RegisterMap,
    Scene,
    SceneBeat,
    Timeline,
    TwistFact,
    VoiceCard,
)


def _char(cid: str, en_name: str, ko_name: str, persona: str = "character", idiolect: str = "neutral") -> Character:
    return Character(
        id=cid,
        en_name=en_name,
        ko_name=ko_name,
        persona=persona,
        voice=VoiceCard(idiolect=idiolect),
    )


def _cue(cid: str, start_ms: int, end_ms: int, text: str) -> Cue:
    return Cue(id=cid, start_ms=start_ms, end_ms=end_ms, text=text, raw=text)


def test_parse_episode_coord():
    c1 = series_state.parse_episode_coord("only-murders-in-the-building-s02e04__flash38")
    assert c1 is not None
    assert (c1.series_id, c1.season, c1.episode) == ("only-murders-in-the-building", 2, 4)

    c2 = series_state.parse_episode_coord("super-pumped-1x07")
    assert c2 is not None
    assert (c2.series_id, c2.season, c2.episode) == ("super-pumped", 1, 7)

    assert series_state.parse_episode_coord("deadpool-2016") is None


def test_causal_slice_blocks_future_spoilers_and_anon_reveals():
    ep1 = series_state.EpisodeCoord(season=1, episode=1, series_id="omitb", video_id="omitb-s01e01")
    ep2 = series_state.EpisodeCoord(season=1, episode=2, series_id="omitb", video_id="omitb-s01e02")

    cm1 = ContextMap(
        synopsis="Ep1",
        tone_baseline="wry mystery",
        characters=[
            _char("charles", "Charles-Haden Savage", "찰스 헤이든 새비지", "retired actor"),
            _char("tie_dye", "Tie-Dye Guy", "타이다이 사람", "hooded figure"),
        ],
        glossary=[GlossaryEntry(en="Gut Milk", ko="장에 좋은 우유")],
    )
    rm1 = RegisterMap(
        edges=[
            RegisterEdge(
                speaker="charles",
                addressee="mabel",
                relation="peer",
                apparent_generation="speaker_older",
                history="met_as_adults",
                stance="civil",
                step=5,
                level="hae",
                evidence_cues=["c0001", "c0002"],
            )
        ]
    )
    cues1 = [
        _cue("c0001", 10_000, 13_000, "A great true crime mystery unpeels itself like an onion."),
    ]
    ko1 = {"c0001": "제대로 된 실화 미스터리는 양파 껍질 벗기듯 풀려야죠"}
    b1 = series_state.commit_bible(None, ep1, cm1, rm1, cues1, ko1)

    # Commit episode 2 with a new character, a twist fact, and reveal of Tie-Dye Guy
    cm2 = ContextMap(
        synopsis="Ep2",
        tone_baseline="wry mystery",
        characters=[
            _char("jan", "Jan Bellows", "잰 벨로스", "bassoonist; secretly the killer who poisoned Tim"),
        ],
        twist_facts=[TwistFact(fact="Jan poisoned Tim Kono", audience_reveal_cue="c0099")],
    )
    b2 = series_state.commit_bible(b1, ep2, cm2, RegisterMap(), [], {})
    b2.anon_slots["tie_dye_guy"].revealed_as = "oscar"
    b2.anon_slots["tie_dye_guy"].reveal_coord = (1, 2)

    # Slicing B2 for episode 2 (before_ep=(1, 2)) MUST strip all ep2 entities/facts/reveals
    sliced_for_ep2 = b2.causal_slice((1, 2))
    assert "jan" not in sliced_for_ep2.entities
    assert not sliced_for_ep2.facts
    assert sliced_for_ep2.anon_slots["tie_dye_guy"].revealed_as is None
    assert "charles" in sliced_for_ep2.entities


def test_spoiler_persona_masked_before_reveal_cue():
    cm = ContextMap(
        synopsis="Finale",
        tone_baseline="tense",
        characters=[
            _char(
                "jan",
                "Jan",
                "잰",
                "Warm bassoonist in 6A; secretly the killer who poisoned Tim Kono.",
                "soft-spoken",
            )
        ],
        twist_facts=[TwistFact(fact="Jan poisoned Tim Kono", audience_reveal_cue="c0003")],
    )
    cues = [
        _cue("c0001", 1000, 2000, "Hello Charles."),
        _cue("c0003", 5000, 6000, "I poisoned Tim."),
    ]
    scenes = [
        Scene(id="s001", start_ms=1000, end_ms=2000, cue_ids=["c0001"]),
        Scene(id="s002", start_ms=5000, end_ms=6000, cue_ids=["c0003"]),
    ]
    tl = Timeline(
        scenes=[
            SceneBeat(scene_id="s001", beat="greeting", chars_present=["jan"]),
            SceneBeat(scene_id="s002", beat="confession", chars_present=["jan"]),
        ]
    )
    snaps = narrative.materialize_snapshots(cm, tl, scenes, cues)
    assert "poisoned" not in snaps["s001"].personas["jan"].lower()
    assert "killer" not in snaps["s001"].personas["jan"].lower()
    assert "Warm bassoonist" in snaps["s001"].personas["jan"]


def test_reconcile_lock_and_enforce_series_pins():
    ep1 = series_state.EpisodeCoord(season=1, episode=1, series_id="sp", video_id="sp-s01e01")
    cm1 = ContextMap(
        synopsis="Ep1",
        tone_baseline="fast",
        characters=[
            _char("levandowski", "Anthony Levandowski", "앤서니 레반도프스키", "engineer"),
            _char("travis", "Travis Kalanick", "트래비스 칼라닉", "CEO"),
            _char("gurley", "Bill Gurley", "빌 걸리", "VC"),
        ],
        glossary=[GlossaryEntry(en="Angel in Flip-Flops", ko="슬리퍼를 신은 천사")],
    )
    rm1 = RegisterMap(
        edges=[
            RegisterEdge(
                speaker="gurley",
                addressee="travis",
                relation="mentor",
                apparent_generation="speaker_older",
                history="longtime_crew_or_buddies",
                stance="civil",
                step=7,
                level="hae",
                pronoun="자네",
                vocative="트래비스",
                evidence_cues=["c0001", "c0002"],
            )
        ]
    )
    cues1 = [
        _cue("c0001", 15_000, 18_000, "It's the fucking drivers. They're defecting."),
    ]
    ko1 = {"c0001": "망할 기사들이 다 이탈 중이야"}
    b1 = series_state.commit_bible(None, ep1, cm1, rm1, cues1, ko1)

    # Episode 2 drafts drifted spelling ('레반도스키') and flipped register ('haeyo')
    cm2_draft = ContextMap(
        synopsis="Ep2",
        tone_baseline="fast",
        characters=[
            _char("anthony", "Anthony Levandowski", "앤서니 레반도스키", "engineer"),
            _char("travis", "Travis Kalanick", "트래비스 캘러닉", "CEO"),
            _char("bill", "Bill Gurley", "빌 걸리", "VC"),
        ],
    )
    cues2 = [
        _cue("c0001", 5_000, 8_000, "Previously on... It's the fucking drivers. They're defecting."),
        _cue("c0002", 25_000, 28_000, "We need to fire Levandowski right now."),
    ]
    cm2 = series_state.reconcile_characters(cm2_draft, b1, cues2)
    chars_by_id = {c.id: c for c in cm2.characters}
    assert chars_by_id["levandowski"].ko_name == "앤서니 레반도프스키"
    assert chars_by_id["travis"].ko_name == "트래비스 칼라닉"

    rm2_draft = RegisterMap(
        edges=[
            RegisterEdge(
                speaker="gurley",
                addressee="travis",
                relation="peer",
                apparent_generation="speaker_older",
                history="met_as_adults",
                stance="civil",
                step=8,
                level="haeyo",
                pronoun="당신",
                vocative="none",
            )
        ]
    )
    rm2 = series_state.lock_edges(rm2_draft, b1, context_map=cm2, cues=cues2)
    assert rm2.edges[0].level == "hae"
    assert rm2.edges[0].pronoun == "자네"
    assert rm2.edges[0].vocative == "트래비스"

    ko2_draft = {
        "c0001": "기사들이 전부 넘어가고 있어요",
        "c0002": "당장 레반도스키를 해고해야 해",
    }
    ko2_enforced = series_state.enforce_series_pins(ko2_draft, cues2, b1, context_map=cm2)
    assert "망할 기사들이 다 이탈 중이야" in ko2_enforced["c0001"]
    assert ko2_enforced["c0002"] == "당장 레반도프스키를 해고해야 해"


def test_streaming_equals_batch_equivalence():
    episodes_data = []
    for i in range(1, 4):
        coord = series_state.EpisodeCoord(season=1, episode=i, series_id="omitb", video_id=f"omitb-s01e0{i}")
        cm = ContextMap(
            synopsis=f"Ep {i}",
            tone_baseline="wry",
            characters=[
                _char("cinda", "Cinda Canning", "신다 캐닝" if i == 1 else "신디 캐닝"),
            ],
            glossary=[GlossaryEntry(en="Gut Milk", ko="장에 좋은 우유")],
        )
        rm = RegisterMap(
            edges=[
                RegisterEdge(
                    speaker="charles",
                    addressee="mabel",
                    relation="peer",
                    apparent_generation="speaker_older",
                    history="longtime_crew_or_buddies",
                    stance="civil",
                    step=5,
                    level="hae" if i == 1 else "haeyo",
                )
            ]
        )
        cues = [_cue("c0001", 10_000, 13_000, "Cinda Canning loves Gut Milk.")]
        ko = {"c0001": "신다 캐닝은 장에 좋은 우유를 좋아해" if i == 1 else "신디 캐닝은 장에 좋은 우유를 좋아해"}
        episodes_data.append((coord, cm, rm, cues, ko))

    # Streaming: serialize/deserialize bible to JSON after each episode
    b_stream: series_state.SeriesBible | None = None
    out_stream = []
    for coord, cm, rm, cues, ko in episodes_data:
        if b_stream is not None:
            b_stream = series_state.SeriesBible.model_validate_json(b_stream.model_dump_json())
            cm = series_state.reconcile_characters(cm, b_stream, cues)
            rm = series_state.lock_edges(rm, b_stream, context_map=cm, cues=cues)
            ko = series_state.enforce_series_pins(ko, cues, b_stream, context_map=cm)
        b_stream = series_state.commit_bible(b_stream, coord, cm, rm, cues, ko)
        out_stream.append(ko)

    # Batch: fold in memory
    b_batch: series_state.SeriesBible | None = None
    out_batch = []
    for coord, cm, rm, cues, ko in episodes_data:
        if b_batch is not None:
            b_slice = b_batch.causal_slice(coord.pair)
            cm = series_state.reconcile_characters(cm, b_slice, cues)
            rm = series_state.lock_edges(rm, b_slice, context_map=cm, cues=cues)
            ko = series_state.enforce_series_pins(ko, cues, b_slice, context_map=cm)
        b_batch = series_state.commit_bible(b_batch, coord, cm, rm, cues, ko)
        out_batch.append(ko)

    assert b_stream is not None and b_batch is not None
    assert b_stream.fingerprint() == b_batch.fingerprint()
    assert out_stream == out_batch
    assert all(k["c0001"] == "신다 캐닝은 장에 좋은 우유를 좋아해" for k in out_batch)


def test_memory_corpus_excludes_future_series_episodes():
    pairs = [
        memory.Pair("omitb-s01e01", 1000, "A great true crime mystery", "실화 미스터리 1화"),
        memory.Pair("omitb-s01e02", 1000, "A great true crime mystery", "실화 미스터리 2화"),
        memory.Pair("omitb-s01e03", 1000, "A great true crime mystery", "실화 미스터리 3화"),
        memory.Pair("knives-out-2019", 1000, "A great true crime mystery", "나이브스 아웃"),
    ]
    corpus = memory.Corpus(pairs)
    hits_e02 = memory.retrieve(corpus, ["A great true crime mystery"], "omitb-s01e02", k=10)
    vids = {p.video_id for p in hits_e02}
    assert "omitb-s01e01" in vids
    assert "knives-out-2019" in vids
    assert "omitb-s01e02" not in vids
    assert "omitb-s01e03" not in vids


def test_evaluate_series_consistency_detects_drift_and_clusters_shortenings():
    cues_e1 = [
        _cue("c0001", 10_000, 13_000, "I'm Cinda Canning on Only Murders in the Building."),
        _cue("c0002", 20_000, 23_000, "Cinda Canning is back."),
    ]
    cues_e2 = [
        _cue("c0001", 5_000, 8_000, "I'm Cinda Canning on Only Murders in the Building."),
    ]
    drift_res = refeval.evaluate_series_consistency(
        [
            {
                "ep_id": "s01e01",
                "cues": cues_e1,
                "ko_lines": {
                    "c0001": "저는 '아파트 건물의 살인'의 신디 캐닝입니다",
                    "c0002": "신다 캐닝이 돌아왔어요",
                },
            },
            {
                "ep_id": "s01e02",
                "cues": cues_e2,
                "ko_lines": {
                    "c0001": "저는 '아파트 이웃들이 수상해'의 신다입니다",
                },
            },
        ],
        tracked_concepts={
            "Cinda Canning": ["신디 캐닝", "신다 캐닝", "신다"],
            "Only Murders in the Building": ["아파트 건물의 살인", "아파트 이웃들이 수상해"],
        },
    )
    assert drift_res["scs"] < 90.0
    assert drift_res["counts"]["total_defects"] >= 2
    assert drift_res["concepts"]["Cinda Canning"]["clustered_forms"] == {"신다 캐닝": 2, "신디 캐닝": 1}


def test_evaluate_pragmatic_address_para_and_gra():
    from substage.models import ChunkObservations, CueObservation, Scene, VoiceCard

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
        _cue("c0485", 1000, 2500, "Mom, I invited all you guys."),
        _cue("c0201", 3000, 4500, "But if I had a mentor, like you..."),
    ]
    obs = [
        ChunkObservations(
            chunk_id="chunk_000", start_ms=0, end_ms=10000,
            cues=[
                CueObservation(cue_id="c0485", speaker="Travis Kalanick", addressee="Bonnie Kalanick"),
                CueObservation(cue_id="c0201", speaker="Travis Kalanick", addressee="Bill Gurley"),
            ],
        )
    ]
    rm = RegisterMap(
        edges=[
            RegisterEdge(
                speaker="travis", addressee="bonnie", relation="own_parent",
                apparent_generation="addressee_older", history="raised_by_addressee",
                stance="civil", step=3, level="hae", pronoun="none", vocative="엄마",
                evidence_cues=["c0485"], basis="mother",
            ),
            RegisterEdge(
                speaker="travis", addressee="bill", relation="mentor",
                apparent_generation="addressee_older", history="met_as_adults",
                stance="civil", step=6, level="haeyo", pronoun="none", vocative="빌",
                evidence_cues=["c0201"], basis="mentor",
            ),
        ]
    )
    scenes = [Scene(id="s001", start_ms=0, end_ms=10000, cue_ids=["c0485", "c0201"])]

    bad_res = refeval.evaluate_pragmatic_address(
        cues,
        {"c0485": "너희 다 초대했잖아", "c0201": "하지만 당신 같은 멘토가 있다면..."},
        context_map=cm, register_map=rm, observations=obs, scenes=scenes, ep_id="s01e01",
    )
    assert bad_res["para"] < 1.0
    assert len(bad_res["violations"]) == 2
    assert bad_res["gra"] == 1.0

    good_res = refeval.evaluate_pragmatic_address(
        cues,
        {"c0485": "다 초대했잖아", "c0201": "하지만 빌 같은 멘토가 있다면..."},
        context_map=cm, register_map=rm, observations=obs, scenes=scenes, ep_id="s01e01",
    )
    assert good_res["para"] == 1.0
    assert len(good_res["violations"]) == 0
    assert good_res["gra"] == 1.0


def test_slash_ost_sync_and_recap_bead_alignment():
    from substage import translate, validate
    from substage.models import ChunkObservations, OnScreenText

    # 1. Slash sanitization preserves fixed tokens and strips clause separators
    assert validate.normalize_punct("거래처 결제 / 잠깐 미뤄") == "거래처 결제 잠깐 미뤄"
    assert validate.normalize_punct("저희 의뢰인들이 /\n후원한 액수는 잘 아시겠죠") == "저희 의뢰인들이\n후원한 액수는 잘 아시겠죠"
    assert validate.normalize_punct("24/7 연중무휴 남/여 구분 없음") == "24/7 연중무휴 남/여 구분 없음"
    assert validate.normalize_punct("뭐라고? / - 아무것도 아냐") == "- 뭐라고?\n- 아무것도 아냐"

    # 2. OST sync: v01 keeps 120000/136000 intact when no clamped/over-end item exists, and v04 realigns +60s Vegas & Kalanick anchor
    ch_v01 = ChunkObservations(
        chunk_id="v01", start_ms=0, end_ms=532886,
        on_screen_text=[
            OnScreenText(start_ms=68000, end_ms=71000, text="[caption] SHOWTIME PRESENTS"),
            OnScreenText(start_ms=120000, end_ms=125000, text="[title] SUPER PUMPED: THE BATTLE FOR UBER"),
            OnScreenText(start_ms=136000, end_ms=138500, text="[caption] 2014"),
        ],
    )
    norm_v01 = translate._normalize_chunk_ost_mmss(ch_v01.on_screen_text, ch_v01.start_ms, ch_v01.end_ms)
    assert [o.start_ms for o in norm_v01] == [68000, 120000, 136000]

    cues_v04 = [
        _cue("c0621", 1846000, 1850000, "Wait, hold on to the bed."),
        _cue("c0642", 1900000, 1902358, "It was you."),
        _cue("c0644", 1919375, 1921000, "Hey, Travis."),
        _cue("c0713", 2066397, 2070902, "Now, if I want them to stay quiet about something,"),
        _cue("c0744", 2149063, 2153359, "Instead it's time for a lesson from Prof. Kalanick."),
    ]
    ch_v04 = ChunkObservations(
        chunk_id="v04", start_ms=1702116, end_ms=2289787,
        on_screen_text=[
            OnScreenText(start_ms=1847000, end_ms=1849000, text="[sign] WELCOME TO Fabulous LAS VEGAS NEVADA"),
            OnScreenText(start_ms=2068000, end_ms=2073500, text="[caption] PROF KALANICK"),
        ],
    )
    realigned = translate.realign_ost_to_anchors_and_gaps([ch_v04], cues_v04)[0]
    assert realigned.on_screen_text[0].start_ms == 1847000
    assert 2149000 <= realigned.on_screen_text[1].start_ms <= 2155000

    # 3. Recap bead alignment: short cue ("They're defecting."), 2:1 merge, and clause trim
    ep1_cues = [
        _cue("c0143", 100000, 103000, "Look, Bill, something is going to pop in the space."),
        _cue("c0358", 200000, 201500, "I think..."),
        _cue("c0359", 201600, 203500, "you're my new VC."),
        _cue("c0690", 300000, 302000, "They're defecting."),
    ]
    ep1_ko = {
        "c0143": "빌, 이 분야에서\n뭐 하나 터지긴 할 겁니다",
        "c0358": "제 생각엔...",
        "c0359": "제 새 투자자시네요",
        "c0690": "다 이탈 중이야",
    }
    b1 = series_state.commit_bible(
        None,
        series_state.EpisodeCoord(series_id="sp", season=1, episode=1, video_id="sp-s01e01"),
        ContextMap(synopsis="test", tone_baseline="drama", characters=[]),
        RegisterMap(edges=[]),
        ep1_cues,
        ep1_ko,
    )
    ep2_cues = [
        _cue("c0006", 14000, 15500, "Something is going to pop in this space."),
        _cue("c0007", 15600, 17500, "I think you're my new VC."),
        _cue("c0012", 27000, 28500, "They're defecting."),
    ]
    pins2 = series_state.pin_cues(ep2_cues, b1)
    assert pins2["c0006"].tm_ko == "빌, 이 분야에서\n뭐 하나 터지긴 할 겁니다"
    assert pins2["c0007"].tm_ko == "제 생각엔...\n제 새 투자자시네요"
    assert pins2["c0012"].tm_ko == "다 이탈 중이야"
