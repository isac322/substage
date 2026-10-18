"""substage.levels: addressed speech level. Cases are official (전문 번역가) lines that
the old refeval ending regex misread (ErrorCensus, HwangPatterns, SampleCases)
plus clearly addressed lines from the same pairs."""

from __future__ import annotations

import pytest

from substage.levels import addressed_level, addressed_levels, analyze_turn_morphemes, ending_level, has_internal_register_clash, speaker_turns


@pytest.mark.parametrize(
    ("sentence", "level"),
    [
        ("갈게요", "haeyo"),
        ("내가 갈게요!", "haeyo"),
        ("그렇죠?", "haeyo"),
        ("갑니다", "hasipsio"),
        ("했습니까?", "hasipsio"),
        ("이쪽으로 오십시오", "hasipsio"),
        ("같이 갑시다", "hasipsio"),
        ("가자", "hae"),
        ("간다", "hae"),
        ("갈래?", "hae"),
        ("뭐야…", "hae"),
        ("했어", "hae"),
        ("프란시스", None),
        ("네", None),
        ("그러니까", None),
        ("OK", None),
    ],
)
def test_ending_level_basic_ladder(sentence, level):
    assert ending_level(sentence) == level


@pytest.mark.parametrize(
    "line",
    [
        # -네 / -군 / -구나 exclamatives and self-talk, including -시- (Morgan → Charles)
        "이제 누군지 기억하시네",  # c0201
        "편해 봤으면 소원이 없겠네",  # c0951
        "그걸 몰랐네",  # c0987
        "그 간호사가 거지 같은 걸 줬네",  # c0852
        "확 쏠리네",  # SampleVideo, Wade → Blind Al
        "머리가 졸라 핑핑 도네",
        "참 실망스럽군",
        # connective and quotative fragments
        "그게 아니라",  # c0224
        "숨는 게 아니라",  # c0797
        "군대라니",  # c0212
        "돌연변이가 25년째\n안 태어나고 있으니",  # c0221
        "오메가급 뮤턴트를\n데려오다니",
        "멕시코 여자들의\n뱃속에서 키우는데",
        "피터의 도움을\n받으러 왔는데",
        "결과는 뻔한데",
        "지금 하려는 일의 목적이 뭔지",
        "병을 낫게만\n해드리는 게 아니라",
        # vocatives and kin terms
        "큰엄마!",
        "아버지",
        "그것도 아주 오래",  # c1100
    ],
)
def test_unaddressed_official_lines_are_none(line):
    assert addressed_level(line) is None


@pytest.mark.parametrize(
    ("line", "level"),
    [
        # rude English, polite ending (official, elder addressee)
        ("알아요, 아버지\n더럽게 못난 놈이죠", "haeyo"),  # c0954
        ("하긴 뭘 해요?", "haeyo"),
        ("이러지 마요", "haeyo"),  # c0185
        ("약 먹어요", "haeyo"),  # c0204
        ("아저씬 죽어가요", "haeyo"),  # Laura → Morgan c1602
        ("그냥 약이나 드시고\n주무시면 안 돼요?", "haeyo"),  # claire → Harlan
        ("정신 나가셨어요?", "haeyo"),  # devon → Harlan
        ("이럴거면 월세 내요", "haeyo"),  # Wade → Blind Al c1269
        ("웹슈터 설정에\n익숙하지 않으시네요", "haeyo"),  # -네요 is addressed
        ("반대하는 목소리가 큰데요", "haeyo"),  # -는데요 is addressed
        ("예전엔 가지치기했죠\n간단하게", "haeyo"),  # trailing adverbial -게
        ("그게 뭡니까?", "hasipsio"),  # John → Winston
        # 반말 / 하게체 from the senior side, and the one plain 반말 Morgan line
        ("며칠 있다 올게", "hae"),  # c0382
        ("자네도 알잖나", "hae"),  # c0620
        ("그렇게 모르겠나?", "hae"),
        ("그만 좀 해! 혼자 할 수 있어", "hae"),
        ("무슨 일인데?", "hae"),  # -는데 with ? is an addressed question
        ("어디 서명해?", "hae"),
    ],
)
def test_addressed_lines(line, level):
    assert addressed_level(line) == level


def test_dash_lines_split_per_speaker_turn():
    assert addressed_levels("- 당신을 위해서예요\n- 아니야!") == ["haeyo", "hae"]  # c0242
    assert addressed_levels("- 쟤가 그래요?\n- 교감으로 알지") == ["haeyo", None]  # c0474
    assert addressed_levels("- 얼마나 됐어요?\n- 몰라") == ["haeyo", "hae"]  # c0938
    assert addressed_levels("- 모건\n- 왜요!") == [None, "haeyo"]  # g0470
    assert addressed_levels("-가자\n-네, 갈게요") == ["hae", "haeyo"]
    assert addressed_levels("- 당신을 위해서예요 - 아니야!") == ["haeyo", "hae"]  # one display line
    # two speakers have no single level
    assert addressed_level("- 당신을 위해서예요\n- 아니야!") is None


def test_turns_and_sentences():
    assert speaker_turns("") == []
    assert speaker_turns("-가자") == ["가자"]
    assert speaker_turns("오메가급\n데려오다니") == ["오메가급\n데려오다니"]
    assert addressed_levels("알았어요. 가자") == ["hae"]  # last leveled sentence wins
    assert addressed_levels("공기 청정제 말고\n저 여자") == [None]
    assert addressed_levels("그건 OK") == [None]


def test_bidi_controls_do_not_hide_dash_turns():
    # Netflix official files (streaming) prefix every line with U+200E / U+202A
    assert addressed_levels("\u200e- 건배!\n\u200e- 가자") == [None, "hae"]
    assert addressed_levels("\u202a- 갈게요\u202c\n\u202a- 가자\u202c") == ["haeyo", "hae"]


def test_sejong_pos_disambiguates_conditional_ec_vs_final_ef():
    # Conditional connective -거든 (EC) does not override the matrix clause EF
    assert addressed_level("시간이 나거든 연락하세요") == "haeyo"
    assert addressed_level("네가 가게 되거든 나도 하렴") == "hae"
    assert addressed_level("비가 오거든 창문을 닫아라") == "hae"
    assert not has_internal_register_clash("시간이 나거든 연락하세요")
    # Sentence-final explanatory -거든 (EF) is addressed 해체, and -거든요 is 해요체
    assert addressed_level("난 페미니스트거든") == "hae"
    assert addressed_level("벳시는 믿음직하거든") == "hae"
    assert addressed_level("어제 돈봉투가 왔거든요") == "haeyo"
    # Vowel-contracted 해체 imperatives/declaratives (꺼져, 닥쳐, 멈춰, 살려 줘)
    assert addressed_level("여기서 당장 꺼져") == "hae"
    assert addressed_level("아가리 닥쳐") == "hae"
    assert addressed_level("야, 왜 멈춰?") == "hae"
    assert addressed_level("제발 살려 줘") == "hae"
    # Nouns not in any hardcoded list are tagged NNG -> None
    for noun in ("국어", "노래", "새해", "감자", "사자", "할머니"):
        assert addressed_level(noun) is None


def test_three_axis_honorific_clash_detection():
    # Intra-turn EF clash (해요체 + 해체 in one single-speaker turn)
    assert has_internal_register_clash("그런 셈이죠\n그렇다고 치자")
    # Banmal interjection (IC: 응) + 해요체 EF
    assert has_internal_register_clash("응\n힘들게 컸어요?")
    # Humble 1st-person NP (저/제) + 해체 EF
    assert has_internal_register_clash("제가 지금 갈게")
    # Colloquial 나/내 + 해요체 is valid in Korean dialogue (not a clash)
    assert not has_internal_register_clash("모두가 구세주 타입은 아니에요\n나도 아니지만 행복해요")
    # Demonstrative '저 둘' / '저들' is not humble 1st-person '저'
    assert not has_internal_register_clash("저 둘이 우리의\n기회일지 몰라")
    # Mirative exclamative (-네!) followed by addressed 해요체 is not a clash
    assert not has_internal_register_clash("더럽게 껴드네!\n미안해요")
