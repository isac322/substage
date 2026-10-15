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
