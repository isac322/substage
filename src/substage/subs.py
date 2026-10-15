"""English subtitle loading: SDH-aware parsing into clean ``Cue`` objects.

pysubs2 handles srt/ass/ssa/vtt/smi. Encoding falls back utf-8-sig -> cp949
(Korean .smi files on the filesystem are cp949; the pipeline never reads
Korean subtitles, but the loader stays tolerant).

Parsing rules (e.g. the source's SDH track):
  "WADE: ..." / "WADE (V.O.): ..."   -> speaker_hint "WADE" (+ "(v.o.)" sound cue)
  "(POP MUSIC PLAYING)" / "[explosion]" / "- [grunts] Hey!" -> sound_cues
  "♪ ... ♪"                          -> lyric
  "- one / - two"                    -> dual-speaker dashes kept verbatim in text
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pysubs2

from .models import Cue

logger = logging.getLogger(__name__)

_ENCODINGS = ("utf-8-sig", "cp949")

# Any bracketed/parenthesized group is a non-dialogue annotation (sound,
# music, delivery direction). Non-greedy; may appear mid-line.
_SOUND_RE = re.compile(r"[\(\[][^\(\)\[\]]*[\)\]]")

# Leading "NAME:" label, up to 4 tokens, allowing "(V.O.)"-style suffixes and
# all-caps or capitalized names ("WADE:", "Wade:", "WADE (V.O.):", "TV ANNOUNCER:").
# The capture may include bracket groups; brackets are extracted as sound cues.
_LABEL_RE = re.compile(
    r"^\s*(?P<dash>-\s+)?"
    r"(?P<label>(?:[\(\[][^\(\)\[\]]*[\)\]]\s*)*"
    r"[A-Z][\w.'&-]*(?:\s+[A-Z][\w.'&-]*){0,3}"
    r"(?:\s*[\(\[][^\(\)\[\]]*[\)\]])*)\s*(?:[:;]|-\.)\s*"
)

# Dual-speaker dash: "- text" (also en/em dashes some rips use).
_SPEAKER_DASH_RE = re.compile(r"^\s*[-–—]\s+")

# Pure-bracket label fallback: "[WADE]:" / "(WADE):" conventions.
_BRACKET_LABEL_RE = re.compile(r"^\s*[\(\[](?P<name>[^\(\)\[\]:]{1,40}?)[\)\]]\s*:\s*")


def _norm_space(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line).strip()


def _norm_text_for_dedup(text: str) -> str:
    """Normalize for identical-text comparison in merge_forced."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[\s\.,!\?;:'\"\-–—…]+", "", t)
    return t


def _parse_line(line: str, speakers: list[str], sounds: list[str]) -> str:
    """Clean one subtitle line; collects speaker labels and sound cues."""
    dash = ""
    if _SPEAKER_DASH_RE.match(line):
        dash = "- "
        line = _SPEAKER_DASH_RE.sub("", line, count=1)

    m = _LABEL_RE.match(line)
    if m:
        label_raw = m.group("label")
        # Pull bracketed suffixes out of the label: "WADE (V.O.)" -> "WADE".
        for grp in _SOUND_RE.findall(label_raw):
            sounds.append(grp.strip().lower())
        name = _SOUND_RE.sub("", label_raw).strip()
        if name:
            speakers.append(name.upper())
        line = line[m.end():]
    else:
        bm = _BRACKET_LABEL_RE.match(line)
        if bm:
            speakers.append(bm.group("name").strip().upper())
            line = line[bm.end():]

    def _stash(match: re.Match[str]) -> str:
        sounds.append(match.group(0).strip().lower())
        return " "

    line = _norm_space(_SOUND_RE.sub(_stash, line))
    # "- [grunts]" leaves no words: the dash must not survive as an empty speaker turn.
    return dash + line if line else ""


# OCR'd ♪ ("j'", "J'", "j”", "j'j'") as a whole token at a line edge, or glued to the
# last word ("gotj'"). A bare "J" (e.g. "Agent J") never matches: a quote is required.
_NOTE = r"(?:[jJ]{1,2}['’”]{1,2})+"
_OCR_NOTE = re.compile(rf"^{_NOTE}(?=\s|$)|(?<=\s){_NOTE}$|(?<=[a-z]){_NOTE}$")
_STRAY_TAG = re.compile(r"[\[(][^\])]{0,8}")


def _parse_event_text(plaintext: str) -> tuple[str, str | None, list[str], bool]:
    """Split SDH markup out of one event's plaintext.

    Returns (text, speaker_hint, sound_cues, lyric). ``text`` is the clean
    dialogue and may be "" for pure sound-effect cues.
    """
    speakers: list[str] = []
    sounds: list[str] = []
    lyric = False
    lines: list[str] = []

    for raw_line in plaintext.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if _OCR_NOTE.search(line):
            # OCR'd SDH subtitles render ♪ as "j'", "J'" or "j”" at line edges.
            line = _OCR_NOTE.sub(" ♪ ", line).strip()
        if "♪" in line:
            lyric = True
            line = line.replace("♪", "")
        if _STRAY_TAG.fullmatch(line):
            sounds.append(line.lower())  # unclosed OCR fragment such as "[M1"
            continue
        cleaned = _parse_line(line, speakers, sounds)
        if cleaned:
            lines.append(cleaned)

    speaker_hint: str | None = None
    if speakers:
        first = speakers[0]
        speaker_hint = first if all(s == first for s in speakers) else "/".join(
            dict.fromkeys(speakers)
        )
    if len(lines) == 1 and lines[0].startswith("- "):
        lines[0] = lines[0][2:]  # one speaker left after removing sound-only turns
    return "\n".join(lines), speaker_hint, sounds, lyric


def load_cues(path: Path, *, forced: bool = False, source: str = "main") -> list[Cue]:
    """Load a subtitle file into sorted, numbered ``Cue`` objects.

    ``forced=True`` marks cues as coming from a forced track (on-screen text /
    foreign dialogue). ``source`` labels provenance ("main" | "forced" | "ost").
    """
    path = Path(path)
    subs: pysubs2.SSAFile | None = None
    last_exc: Exception | None = None
    for enc in _ENCODINGS:
        try:
            subs = pysubs2.load(str(path), encoding=enc)
            break
        except UnicodeDecodeError as exc:
            last_exc = exc
    if subs is None:
        raise ValueError(f"could not decode {path} (tried {_ENCODINGS})") from last_exc

    cues: list[Cue] = []
    for ev in subs.events:
        if ev.is_comment or not ev.is_text:
            continue
        raw = ev.plaintext.strip()
        if not raw:
            continue
        text, speaker_hint, sounds, lyric = _parse_event_text(ev.plaintext)
        if not text and not sounds:
            continue  # e.g. a cue that was only ♪ characters
        cues.append(
            Cue(
                id="",
                start_ms=int(ev.start),
                end_ms=int(ev.end),
                text=text,
                raw=raw,
                speaker_hint=speaker_hint,
                sound_cues=sounds,
                lyric=lyric,
                forced=forced,
                source=source,
            )
        )

    cues.sort(key=lambda c: (c.start_ms, c.end_ms))
    _renumber(cues)
    return cues


def _renumber(cues: list[Cue]) -> None:
    for i, cue in enumerate(cues, 1):
        cue.id = f"c{i:04d}"


def merge_forced(main: list[Cue], forced: list[Cue]) -> list[Cue]:
    """Merge a forced track into the main cue list.

    Forced cues duplicating existing dialogue (normalized identical text on an
    overlapping interval, in either track) are dropped; the rest are added,
    flagged forced, and ids are reassigned in time order.
    """
    merged: list[Cue] = [c.model_copy() for c in main]
    pool = list(main)

    for fc in forced:
        norm = _norm_text_for_dedup(fc.text)
        duplicate = any(
            norm
            and c.start_ms < fc.end_ms
            and fc.start_ms < c.end_ms
            and _norm_text_for_dedup(c.text) == norm
            for c in pool
        )
        if duplicate:
            continue
        clone = fc.model_copy(update={"forced": True, "source": "forced"})
        merged.append(clone)
        pool.append(clone)

    merged.sort(key=lambda c: (c.start_ms, c.end_ms))
    _renumber(merged)
    return merged


def write_srt(rows: list[tuple[int, int, str]], out: Path) -> None:
    """Write (start_ms, end_ms, korean_text) rows as a UTF-8 SRT."""
    subs = pysubs2.SSAFile()
    for start_ms, end_ms, ko in rows:
        subs.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=ko.replace("\n", "\\N")))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(out), encoding="utf-8", format_="srt")
