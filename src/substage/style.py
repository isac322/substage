"""House Korean subtitle style guide (prompts/style_guide.md).

One source of truth for stylistic choices: S3 translate and S4 review append
the whole guide to their system prompts, S5 compress and S5c condense send it
as system text, and the S2a video map embeds its register section. Callers put
the text into their checkpoint keys (through their system/template text), so
editing the guide regenerates every stage that reads it.
"""

from __future__ import annotations

from functools import cache
from importlib import resources

#: placeholder a prompt template uses to embed the guide's register section
REGISTER_TOKEN = "<<STYLE_GUIDE_REGISTER>>"
_REGISTER_HEADING = "## 3. Register and address"


@cache
def style_guide() -> str:
    """Full guide text."""
    return resources.files("substage.prompts").joinpath("style_guide.md").read_text(encoding="utf-8").strip()


def register_rules() -> str:
    """The guide's register-and-address section, up to the next '## ' heading."""
    text = style_guide()
    start = text.index(_REGISTER_HEADING)
    end = text.find("\n## ", start + len(_REGISTER_HEADING))
    return text[start : end if end != -1 else len(text)].strip()


def with_guide(system: str) -> str:
    """``system`` followed by the full guide."""
    return f"{system.rstrip()}\n\n{style_guide()}\n"


PRO_SUBTITLE_LAWS: dict[str, str] = {
    "law_1_drop_noise": (
        "Drop vocal noises ('Whoo!', 'Uh', 'Ah', 'Huh') and filler openers ('Well', 'Look'); "
        "keep only the core speech act (e.g. 'Excuse me. Whoo!' -> '실례', NEVER '실례, 우와!')."
    ),
    "law_2_cultural_transcreation": (
        "Transcreate obscure Anglo-American celebrity/pop-culture gags into instant Korean visual equivalents "
        "(e.g. 'sweaty after hot yoga' -> '핫요가 하고 나온 하마 암내', NEVER phonetic transliteration)."
    ),
    "law_3_wordplay_landing": (
        "When English inverts a name or pun (e.g. 'Last. First.'), recreate the Korean name-intro punch "
        "('이름이야 / 성은 성씨'), NEVER translate a name pun literally."
    ),
    "law_4_spoken_compression": (
        "Compress clunky English noun phrases ('someone on my naughty list') into active Korean verb phrases "
        "('못된 아이 혼내주러 왔거든'), NEVER literal translationese ('나쁜 아이 명단의 누굴')."
    ),
    "law_5_no_truncation": (
        "Never cut off a sentence leaving a dangling object without its verb "
        "('그러니까 사랑을' is broken; complete as '그러니까 사랑을 붙잡으란 말야')."
    ),
}
