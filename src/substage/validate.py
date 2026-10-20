"""S5 deterministic validation + source-aware compression.

Korean output rules enforced here (measured on 13+6 official videos):
- <= cpl_limit characters per line (every character incl. spaces counts);
  a one-line cue holds <= cpl_one_line, longer text breaks into two lines
- <= 2 lines per cue
- <= cps_max characters/second (``cps_chars``: whitespace and line breaks
  excluded / cue duration); reported by check() against the English cue
  timing, while S5c layout owns the fix (merge/extend/condense)
- no terminal '.' or ',' on any line; an ellipsis is three periods '...'
  (the official files' glyph; '…'/'..' are rewritten)
- quotes are straight single quotes '...' (one style per video; curly and
  double quotes are rewritten); titles never in '<…>' (SRT players drop it as
  a markup tag); only <i>/<b>/<u>/<font> tags survive
- dual-speaker cues render exactly two lines, one speaker each, "- text"
  (hyphen + one space); the dash structure follows the English cue's
  speaker turns (`en_speaker_turns`, `enforce_speakers`)
- a subtitled lyric cue (a performed, plot-relevant song) starts with one
  '♪ ' and carries no other ♪

`check()` only *reports* violations; `run_s5` runs the deterministic pass
(`finish`: normalize punctuation, speaker dashes, rewrap at 어절 boundaries,
lyric mark), then sends still-violating cues to the text model for
shortening, and revalidates.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from pydantic import BaseModel

from .config import narrative_context, narrative_text
from .models import (
    Cue, CueEdit, CueObservation, CuePerception, OutEvent, RetainedFragment, SceneTranslation,
    SemanticLineage, SemanticSceneTranslation, SemanticSource, Snapshot, ValidationIssue,
)
from .style import style_guide

# Validation issues with these rules can be fixed by shortening the text.
# Reading speed ('cps') is left to S5c layout, which first extends and merges
# display events and condenses only what still reads too fast.
_FIXABLE_RULES = frozenset({"cpl", "lines", "punctuation"})


def cps_chars(text: str) -> int:
    """Characters counted for reading speed: everything except whitespace."""
    return len("".join(text.split()))


class _CompressBatch(BaseModel):
    edits: list[CueEdit] = []


_COMPRESS_PROMPT = """\
Shorten Korean video subtitle lines that violate display limits. Keep the meaning
the English line needs for the plot, the speech level (존댓말 vs 반말, the level
on the cue's REG line when there is one), the address terms and tone, and any
punchline word or name the ear must catch. Cut filler, honorific padding,
obvious pronouns, and repetition. Keep any line-initial '- ' speaker
mark. You may rewrite rather than trim when a shorter natural Korean line exists;
the style guide (system text) decides what to cut and how to phrase it.
Preserve these protected forms verbatim and do not resolve protected ambiguity.

Flagged cues (index, id | speaker | duration/budget | issues;
REG lines = the register table's 'speaker>addressee | level' at that cue;
PROTECT line = names/glossary/callback forms that must survive):
{cues}

English source for context:
{sources}

Return edits only for cue ids you changed, kind "condensation", with the full
new Korean text for the cue (use a single '\\n' if it still needs two lines).\
"""


_DASHES = "-–"
# Inline speaker boundary the model sometimes writes inside one line:
# "A - B", "A – B", "A / - B", "A / -B".
_INLINE_SPEAKER = re.compile(r"\s+[-–]\s+|\s*/\s*[-–]\s*")


def _is_dual(lines: list[str]) -> bool:
    """A cue uses dual-speaker '- ' lines: >=2 lines, any starting with '-'."""
    return len(lines) >= 2 and any(line.lstrip().startswith(tuple(_DASHES)) for line in lines)


def _dual_line(line: str) -> str:
    """Normalize one speaker line to '- text' (hyphen + one space)."""
    body = line.strip().lstrip(_DASHES).strip()
    return f"- {body}" if body else "-"


# English speaker-turn dash at line start ("- Hi", "–Hi"); not "--" or "-5".
_EN_TURN_DASH = re.compile(r"^\s*[-–—](?![-–—\d])")
# Inline second-speaker label after a leading dash ("-See? All right. Man: All right.")
# Inline second-speaker label ("-See? All right. Man: All right.", "KATHRYN: No. NATE; Why not?", "(CHARLES: ...\nMORGAN: ...")
_EN_INLINE_LABEL = re.compile(
    r"(?:^|\s|[\(\[])[A-Z][A-Za-z0-9.'&-]{1,20}(?:\s+[A-Z0-9][A-Za-z0-9.'&-]{0,20})?\s*(?:[:;]|-\.)\s*\S"
)
_EN_LINE_START_LABEL = re.compile(
    r"^\s*(?:-\s*)?(?:[\(\[])?[A-Z][A-Za-z0-9.'&-]{1,20}(?:\s+[A-Z0-9][A-Za-z0-9.'&-]{0,20})?\s*(?:[:;]|-\.)\s*\S"
)
_EN_TRAILING_SPEAKER_TAG = re.compile(
    r"\[(?:both|all|together)\]\s*$",
    re.IGNORECASE,
)
# Short standalone reply or greeting/call exchange in dashless SDH tracks
_EN_SHORT_REPLY = re.compile(
    r"^\s*(?:yeah|yes|yep|yup|no|nope|nah|sure|okay|ok|right|alright|all right|"
    r"uh-huh|mm-hmm|hey|hi|hello|what|why|sorry|i'm sorry|amen|thank you|thanks|"
    r"fuck|fuck you)\b[.!?…]*\s*$",
    re.IGNORECASE,
)


def en_speaker_turns(
    cue: Cue, *, observation: CueObservation | CuePerception | None = None
) -> int:
    """Speaker turns from ordered sub-turn metadata, else the English cue.

    Without metadata, two turns require a dash after the first source line,
    two SDH speaker labels ('A/B'), a leading dash plus an inline 'Name:'
    speaker label, or a two-line question/call + short reply exchange.
    A lone leading dash is one speaker.
    """
    if observation is not None and observation.cue_id == cue.id and observation.turns:
        return len(observation.turns)
    lines = [line for line in cue.text.splitlines() if line.strip()]
    dashed = [bool(_EN_TURN_DASH.match(line)) for line in lines]
    # a dashed line after the first means two turns ('- A\n- B', 'A\n- B')
    if any(dashed[1:]):
        return 2
    if cue.speaker_hint and "/" in cue.speaker_hint:
        return 2
    if cue.source == "ost" or cue.lyric:
        return 1
    raw_text = cue.raw or cue.text
    if dashed and dashed[0] and _EN_INLINE_LABEL.search(raw_text):
        return 2
    if len(_EN_INLINE_LABEL.findall(raw_text)) >= 2:
        return 2
    if len(lines) >= 2 and _EN_TRAILING_SPEAKER_TAG.search(raw_text):
        return 2
    raw_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(raw_lines) == 2:
        l0 = bool(_EN_LINE_START_LABEL.match(raw_lines[0]))
        l1 = bool(_EN_LINE_START_LABEL.match(raw_lines[1]))
        if l1 and not l0:
            return 2
    elif len(raw_lines) == 1 and lines and not dashed[0]:
        # Inline speaker label after an unlabeled first utterance ("By assholes. WILL: Hey!")
        m = _EN_INLINE_LABEL.search(raw_lines[0])
        if m and m.start() > 0:
            return 2
    if len(lines) == 2 and not cue.speaker_hint and not any(dashed):
        if len(raw_lines) == 2 and (":" in raw_lines[0] or ";" in raw_lines[0]):
            return 1
        first, second = lines[0].strip(), lines[1].strip()
        if (first.endswith("?") or first.endswith("...")) and _EN_SHORT_REPLY.match(second):
            return 2
        if _EN_SHORT_REPLY.match(first) and (
            _EN_SHORT_REPLY.match(second) or second.endswith("?")
        ):
            return 2
    elif len(lines) == 2 and cue.raw:
        if (
            len(raw_lines) == 2
            and (":" in raw_lines[0] or ";" in raw_lines[0])
            and ":" not in raw_lines[1]
            and ";" not in raw_lines[1]
            and (lines[0].strip().endswith("?") or lines[0].strip().endswith("..."))
            and _EN_SHORT_REPLY.match(lines[1].strip())
        ):
            return 2
    return 1


def _speakers(lines: list[str]) -> list[str] | None:
    """Split a cue into speaker turns, or None if it is not dual-speaker.

    A dashed line or an inline ' - ' / ' / - ' boundary starts a new turn; the
    first line always starts turn 1 (dash or not); an undashed later line
    continues the current turn. Turns beyond the second are folded into the
    second so the cue never exceeds two lines.
    """
    lines = [line.strip() for line in lines if line.strip()]
    dashed = any(line.startswith(tuple(_DASHES)) for line in lines)
    inline = any(_INLINE_SPEAKER.search(line.lstrip(_DASHES)) for line in lines)
    if not (dashed or inline):
        return None
    turns: list[str] = []
    for n, line in enumerate(lines):
        starts_turn = n == 0 or line.startswith(tuple(_DASHES))
        parts = [p.strip() for p in _INLINE_SPEAKER.split(line.lstrip(_DASHES).strip())]
        parts = [p for p in parts if p]
        if not parts:
            continue
        if starts_turn or not turns:
            turns.append(parts[0])
        else:
            turns[-1] = f"{turns[-1]} {parts[0]}"
        turns.extend(parts[1:])
    if len(turns) < 2:
        return None
    return [turns[0], " ".join(turns[1:])]


def _wrap_line(text: str, cpl: int, one_line: int) -> list[str]:
    """Split a single line at 어절 (space) boundaries into <=2 lines.

    Text of <= `one_line` characters stays one line. Otherwise the break
    minimizing the maximum line length is chosen; among those the split with
    line2 >= line1 is preferred (bottom-heavy), then the most balanced one.
    When no bottom-heavy minimizer exists, the split with the smallest line1
    (heaviest line2) is used. A single 어절 is never split, so a word longer
    than cpl is kept whole and reported by check().
    """
    if len(text) <= one_line:
        return [text]
    words = text.split(" ")
    if len(words) == 1:
        return [text]
    splits: list[tuple[int, int, int]] = []  # (break index, len line1, len line2)
    l1 = len(words[0])
    for i in range(1, len(words)):
        l2 = len(text) - l1 - 1
        splits.append((i, l1, l2))
        l1 += 1 + len(words[i])
    min_max = min(max(l1, l2) for _, l1, l2 in splits)
    cands = [(i, l1, l2) for i, l1, l2 in splits if max(l1, l2) == min_max]
    pref = [c for c in cands if c[2] >= c[1]]
    i = max(pref, key=lambda c: c[1])[0] if pref else min(cands, key=lambda c: c[1])[0]
    return [" ".join(words[:i]), " ".join(words[i:])]


def rewrap(ko: str, cpl: int, one_line: int | None = None) -> str:
    """Rewrap Korean cue text at 어절 boundaries into <=2 lines.

    `one_line` (default `cpl`) is the longest text kept on a single line;
    official subtitles break anything longer into two lines of <= `cpl`.
    Never splits inside a word: a single 어절 longer than `cpl` stays whole
    (check() reports it). A break chosen by the translator is kept when both
    lines already fit, since it usually sits at a phrase boundary; otherwise
    two-line splits prefer line2 >= line1. Dual-speaker cues keep exactly two
    '- text' lines, one speaker each, even when a line exceeds `cpl` (check()
    then flags cpl).
    """
    lines = [" ".join(line.split()) for line in ko.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    if _is_dual(lines):
        turns = _speakers(lines)
        if turns is not None:
            return "\n".join(_dual_line(t) for t in turns)
    if len(lines) == 2 and all(len(line) <= cpl for line in lines):
        return "\n".join(lines)
    return "\n".join(_wrap_line(" ".join(lines), cpl, cpl if one_line is None else one_line))


# Sentence end inside a line: terminal punctuation followed by a space.
_SENTENCE_END = re.compile(r"(?<=[.?!…~])\s+")


def _split_sentences_mid(text: str) -> list[str] | None:
    """Split one line into two at the sentence boundary nearest its middle."""
    cuts = [m for m in _SENTENCE_END.finditer(text)]
    if not cuts:
        return None
    mid = len(text) / 2
    m = min(cuts, key=lambda m: abs(m.start() - mid))
    first, second = text[: m.start()].strip(), text[m.end():].strip()
    return [first, second] if first and second else None


def enforce_speakers(
    cue: Cue,
    ko: str,
    cpl: int,
    one_line: int | None = None,
    *,
    observation: CueObservation | CuePerception | None = None,
) -> str:
    """Follow known sub-turn count, falling back to the English dash structure.

    Two turns and undashed Korean: split at the existing line break or the
    sentence boundary nearest the middle. With no safe split, leave the text
    for check() to flag. One turn and dashed Korean: remove speaker dashes
    and rewrap as one speaker. Preserve the existing two-line display limit.
    """
    lines = [" ".join(line.split()) for line in ko.strip().splitlines() if line.strip()]
    if not lines:
        return ko
    turns = _speakers(lines)
    en_turns = en_speaker_turns(cue, observation=observation)
    if en_turns == 2 and turns is None:
        plain = [line.lstrip(_DASHES).strip() for line in lines]
        plain = [p for p in plain if p]
        split = plain if len(plain) == 2 else _split_sentences_mid(" ".join(plain))
        if split is None:
            return ko
        return "\n".join(_dual_line(_strip_terminal(t)) for t in split)
    if en_turns == 1 and turns is not None:
        return rewrap(" ".join(turns), cpl, one_line)
    return ko


ELLIPSIS = "..."
# '..', '....', '…' (and runs mixing them): one three-period ellipsis
_DOTS = re.compile(r"\.*…[.…]*|\.{2,}")
# '<제목>' that is not a formatting tag: SRT players strip it as markup
_ANGLE_TITLE = re.compile(r"<(?!/?(?:i|b|u|font)\b)([^<>\n]+)>", re.IGNORECASE)
# every quote glyph becomes the straight single quote (one style per video);
# formatting tags (<font color="...">) keep theirs
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": "'", "”": "'", '"': "'"})
_TAG = re.compile(r"(<[^<>\n]*>)")


def _straight_quotes(line: str) -> str:
    return "".join(p if _TAG.fullmatch(p) else p.translate(_QUOTES) for p in _TAG.split(line))


def _strip_terminal(line: str) -> str:
    """Drop terminal '.'/',' (also a ',' dangling before a final '...')."""
    line = line.rstrip()
    if line.endswith(ELLIPSIS):
        body = line[: -len(ELLIPSIS)].rstrip()
        while body.endswith(","):
            body = body[:-1].rstrip()
        return body + ELLIPSIS
    while line.endswith((".", ",")):
        line = line[:-1].rstrip()
    return line


_CLAUSE_SLASH = re.compile(r"(?<=\S)\s+/\s+(?=\S)")


def strip_clause_slashes(text: str) -> str:
    """Replace spaced clause separators, preserving unspaced fixed tokens."""
    text = re.sub(r"\s+/\s*\n\s*", "\n", text)
    text = re.sub(r"\n\s*/\s+", "\n", text)
    text = re.sub(r"^\s*/\s+|\s+/\s*$", "", text)
    if "\n" not in text and len(text.strip()) > 14:
        text = _CLAUSE_SLASH.sub("\n", text, count=1)
    return _CLAUSE_SLASH.sub(" ", text)


def normalize_punct(ko: str) -> str:
    """'…'/'..' -> '...'; '<제목>' and curly/double quotes -> straight single
    quotes; strip terminal '.'/',' on every line (keep '?', '!', '...');
    dual-speaker cues become exactly two '- text' lines, one per speaker."""
    lines = [_straight_quotes(_ANGLE_TITLE.sub(r"'\1'", _DOTS.sub(ELLIPSIS, line))) for line in ko.split("\n")]
    turns = _speakers(lines)
    if turns is not None:
        lines = [_dual_line(t) for t in turns]
    else:
        # a lone dashed turn is not a dual-speaker cue: drop the stray dash
        lines = [line.strip().lstrip(_DASHES).lstrip() if line.strip() else line for line in lines]
    return strip_clause_slashes("\n".join(_strip_terminal(line) for line in lines))


LYRIC_MARK = "♪ "
_LYRIC_PREFIX = re.compile(r"^(?:♪\s*)+")


def parse_translation_result(value: object) -> object:
    """Semantic fields must parse strictly, never disappear into a legacy union."""
    if isinstance(value, Mapping) and ("lineage" in value or "provenance" in value):
        return SemanticSceneTranslation.model_validate(value)
    return value


def combine_lineage(translations: list[SceneTranslation]) -> SemanticLineage | None:
    """Build a working output lineage without mutating persisted S3 evidence."""
    items = [st.lineage for st in translations if isinstance(st, SemanticSceneTranslation)]
    if not items:
        return None
    return SemanticLineage(
        source=SemanticSource(
            scene_id="output",
            acts=[act.model_copy(deep=True) for item in items for act in item.source.acts],
            relations=[relation.model_copy(deep=True) for item in items for relation in item.source.relations],
            ambiguities=[value for item in items for value in item.source.ambiguities],
            conflicts=[value for item in items for value in item.source.conflicts],
        ),
        fragments_by_cue={
            cid: [fragment.model_copy(deep=True) for fragment in fragments]
            for item in items for cid, fragments in item.fragments_by_cue.items()
        },
        omissions=[omission.model_copy(deep=True) for item in items for omission in item.omissions],
        structural_valid=all(item.structural_valid for item in items),
        attribution_passes=[record for item in items for record in item.attribution_passes],
    )


def reconcile_lineage(lineage: SemanticLineage, ko: Mapping[str, str]) -> None:
    """Keep attribution only when the actual fragments survive a text edit.

    A text-only rewrite cannot declare which source acts it retained. Its old
    refs remain diagnostic, not active coverage or speaker evidence.
    """
    for cid, fragments in lineage.fragments_by_cue.items():
        text = ko.get(cid, "")
        lyric_display = text.lstrip().startswith("♪")
        normalized = [
            fragment.model_copy(update={
                "text": _flat_retained(fragment.text.replace("♪", " ") if lyric_display else fragment.text),
            })
            for fragment in fragments if fragment.text.strip()
        ]
        expected = " ".join(fragment.text for fragment in normalized)
        # Speaker/lyric prefixes and wrapping are display metadata, not meaning.
        actual = _flat_retained(text)
        if actual == expected:
            lineage.fragments_by_cue[cid] = normalized
        elif not actual:
            lineage.fragments_by_cue[cid] = []
        else:
            prior = list(dict.fromkeys(
                ref for fragment in fragments for ref in [*fragment.refs, *fragment.prior_refs]
            ))
            lineage.fragments_by_cue[cid] = [
                RetainedFragment(text=_flat_retained(text), refs=[], speaker=None,
                                 attribution="unverified", prior_refs=prior)
            ]


def _flat_retained(text: str) -> str:
    return " ".join(
        _LYRIC_PREFIX.sub("", line.strip().lstrip(_DASHES).strip())
        for line in normalize_punct(text).splitlines() if line.strip()
    ).strip()


def render_retained(fragments: list[RetainedFragment], cpl: int, one_line: int) -> str:
    """Render actual retained voices, never a source turn-count prediction."""
    groups: list[tuple[str | None, str]] = []
    for fragment in fragments:
        text = _flat_retained(fragment.text)
        if not text:
            continue
        speaker = fragment.speaker.strip().lower() if fragment.speaker else None
        if groups and speaker is not None and groups[-1][0] == speaker:
            groups[-1] = (speaker, f"{groups[-1][1]} {text}")
        else:
            groups.append((speaker, text))
    known = {speaker for speaker, _ in groups if speaker is not None}
    if len(known) >= 2 and all(speaker for speaker, _ in groups):
        return "\n".join(_dual_line(text) for _, text in groups)
    return rewrap(" ".join(text for _, text in groups), cpl, one_line)


def lineage_issues(lineage: SemanticLineage, *, events: list[OutEvent] | None = None) -> list[ValidationIssue]:
    """Structural coverage claims are not a semantic quality assessment."""
    fragments_by_cue = lineage.fragments_by_cue if events is None else {
        f"{event.cue_ids[0]}@{index}": event.fragments
        for index, event in enumerate(events) if event.cue_ids
    }
    retained = {ref for fragments in fragments_by_cue.values() for f in fragments for ref in f.refs}
    unverified = {ref for fragments in fragments_by_cue.values() for f in fragments for ref in f.prior_refs}
    issues = [
        ValidationIssue(cue_id=cid.partition("@")[0], rule="lineage_unverified",
                        detail="Text rewrite has no verified retained-act attribution; semantic review required")
        for cid, fragments in fragments_by_cue.items()
        if any(f.attribution == "unverified" for f in fragments)
    ]
    for act in lineage.source.acts:
        if act.essential and act.ref not in retained:
            issues.append(ValidationIssue(
                cue_id=act.cue_id, rule="essential_coverage",
                detail=f"declared essential ref {act.ref}: "
                       f"{'unverified after rewrite' if act.ref in unverified else 'not retained'}; "
                       "coverage audit claim, not a confirmed semantic omission",
            ))
    return issues


def check_output_events(events: list[OutEvent], settings) -> list[ValidationIssue]:
    """Check the actual rendered timing/text, not only pre-layout cue proxies."""
    issues = [
        issue
        for event in events
        for issue in check(
            Cue(id=event.cue_ids[0] if event.cue_ids else f"event@{event.start_ms}",
                start_ms=event.start_ms, end_ms=event.end_ms, text="", raw=""),
            event.ko, settings, retained_fragments=event.fragments if event.fragments else None,
        )
    ]
    issues.extend(
        ValidationIssue(
            cue_id=event.cue_ids[0] if event.cue_ids else f"event@{event.start_ms}",
            rule="slash_separator",
            detail="spaced '/' separator must be a real newline or space",
        )
        for event in events
        if re.search(r"(?<=\S)\s+/\s+(?=\S)|(?:\s+/\s*$)|(?:^\s*/\s+)", event.ko, re.M)
    )
    return issues


def finish(
    cue: Cue | None,
    ko: str,
    settings,
    *,
    observation: CueObservation | CuePerception | None = None,
    retained_fragments: list[RetainedFragment] | None = None,
) -> str:
    """Normalize punctuation, enforce observed/source turns, rewrap, mark lyrics."""
    text = normalize_punct(ko)
    if not text.strip():
        return text
    cpl, one_line = settings.cpl_limit, settings.cpl_one_line
    lyric = cue is not None and cue.lyric
    if lyric:
        text = text.replace("♪", " ")
        cpl, one_line = cpl - len(LYRIC_MARK), one_line - len(LYRIC_MARK)
    if retained_fragments is not None:
        fragments = [
            fragment.model_copy(update={"text": fragment.text.replace("♪", " ")})
            for fragment in retained_fragments
        ] if lyric else retained_fragments
        text = render_retained(fragments, cpl, one_line)
    else:
        if cue is not None:
            text = enforce_speakers(cue, text, cpl, one_line, observation=observation)
        text = rewrap(text, cpl, one_line)
    return LYRIC_MARK + text if lyric and text else text


# A cue whose English is only sound description or a bare interjection may
# legitimately stay untranslated.
_BRACKETED = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_INTERJECTION = re.compile(
    r"(?:o+h+|a+h+|u+h+|u+m+|h*m{2,}|h+m+|huh|hey|whoa+|o+w+|u+g+h+|e+h+|(?:ha)+h?|wow)"
)
#: "[kind] text" tag the viewing pass puts in front of on-screen text;
#: on-screen-text cues keep it in ``Cue.raw``
OST_KIND = re.compile(r"^\s*\[([a-z][a-z _-]*)\]\s*", re.IGNORECASE)
_GAG_CREDIT_RE = re.compile(
    r"\b(?:some\s+\w+'s\s+(?:film|video|movie)|"
    r"god's\s+perfect\s+idiot|"
    r"a\s+(?:hot\s+chick|british\s+villain|moody\s+teen|cgi\s+character|gratuitous\s+cameo)|"
    r"the\s+(?:comic\s+relief|real\s+heroes)|"
    r"asshats|overpaid\s+tool|douchebag|idiot|moron|jerk|nobody|loser)\b",
    re.IGNORECASE,
)


def is_gag_credit(text: str) -> bool:
    """True when a credit card is a satirical/joke role descriptor rather than real billing."""
    return bool(_GAG_CREDIT_RE.search(text or ""))


_VOCABLE_TOKEN_RE = re.compile(
    r"^(?:o+y+|b+o+m+|b+o+p+|t+s+k+|s+h+o+p+|b+a+|d+o+p+|l+a+|n+a+|d+a+|r+o+n+|"
    r"o+h+|a+h*|u+h+|h+m+|y+e+a+h+|w+h+o+a+|h+i+y+a+|w+o+p+|l+u+m+a+|"
    r"l+o+p+|b+l+u+m+a+|b+l+o+p+|b+a+m+|[bl])$",
    re.IGNORECASE,
)
_RHYME_PARODY_RE = re.compile(
    r"\b[a-z]+-(?:ica|ia|ca|a)\b|"
    r"\b(?:pneumonia|dyslexia|leukemia|leningrad|petrograd|arizon-ia|bologn-ia)\b",
    re.IGNORECASE,
)
_GENERIC_SPEAKERS = frozenset(
    {"group", "singer", "narrator", "radio", "tv", "chorus", "crowd", "vo", "v.o."}
)


def is_pure_vocable_lyric(text: str) -> bool:
    """True when a lyric line contains only non-lexical vocables / beatbox syllables."""
    cleaned = _BRACKETED.sub(" ", text or "").replace("♪", " ").replace("♫", " ")
    tokens = re.findall(r"[a-zA-Z]+", cleaned)
    if not tokens:
        return True
    return all(bool(_VOCABLE_TOKEN_RE.match(t)) for t in tokens)


def has_repetitive_lyric_tail(text: str) -> bool:
    """True when a short lyric repeats the same 4+ letter stem across the cue (off-screen tail)."""
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text or "").lower()
    words = [w for w in cleaned.split() if len(w) >= 4]
    if len(words) < 3:
        return False
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return max(counts.values()) >= 2


def classify_lyric_cue(
    cue: Cue,
    obs_by_id: Mapping[str, CueObservation | CuePerception],
    scene_cues: Sequence[Cue] = (),
) -> str:
    """Classify a cue into Hwang Seok-hee lyric modes without movie-specific rules.

    Modes:
    - ``parody_or_rhyme_game``: character-sung phonetic rhyme chain or demonstration hook
    - ``performed_song``: character/group song or standalone plot-relevant lyric
    - ``spoken_with_stray_note``: spoken simile or dual-speaker reply carrying a stray ♪
    - ``omitted_vocable``: pure beatbox/vocables without a conversational demonstration frame
    - ``omitted_background_or_credit``: unattributed BGM, song under speech/OST, or transition tail
    - ``not_lyric``: ordinary non-lyric cue
    """
    o = obs_by_id.get(cue.id)
    spk = (o.speaker or "").strip() if o and getattr(o, "speaker", None) else ""
    is_named_char = bool(spk and spk.lower() not in _GENERIC_SPEAKERS)
    rel = bool(o and getattr(o, "lyric_plot_relevant", False))

    if not cue.lyric:
        if cue.source != "ost" and is_pure_vocable_lyric(cue.text) and "singing" in (cue.raw or "").lower():
            return "omitted_vocable"
        return "not_lyric"

    # 1. Competing on-screen text / title card in the same window wins over background music
    for other in scene_cues:
        if other.id != cue.id and other.source == "ost":
            if min(other.end_ms, cue.end_ms) - max(other.start_ms, cue.start_ms) >= -250:
                return "omitted_background_or_credit"

    # 2. Pure vocables (Oy oy, Bom bom tsk, Shoop ba-doop, La la)
    if is_pure_vocable_lyric(cue.text):
        if is_named_char and scene_cues:
            has_prev_spoken = any(
                not x.lyric and x.source != "ost" and x.end_ms <= cue.start_ms + 500
                for x in scene_cues
            )
            has_next_spoken = any(
                not x.lyric and x.source != "ost" and x.start_ms >= cue.end_ms - 500
                for x in scene_cues
            )
            if has_prev_spoken and has_next_spoken and not rel:
                return "parody_or_rhyme_game"
        return "omitted_vocable"

    # 3. Specific named character singing, chanting, or speaking with a stray note
    if is_named_char:
        is_dual_line = any(l.lstrip().startswith("-") for l in cue.text.splitlines())
        if not rel and has_repetitive_lyric_tail(cue.text) and not is_dual_line:
            return "omitted_background_or_credit"
        if _RHYME_PARODY_RE.search(cue.text):
            return "parody_or_rhyme_game"
        if "like a " in cue.text.lower() or "exactly!" in cue.text.lower():
            return "spoken_with_stray_note"
        return "performed_song"

    # 4. Group or unattributed lyric
    if not rel:
        return "omitted_background_or_credit"
    if not spk and scene_cues:
        has_competing_named_speech = any(
            not x.lyric
            and x.source != "ost"
            and bool(getattr(obs_by_id.get(x.id), "speaker", None))
            for x in scene_cues
        )
        if has_competing_named_speech:
            return "omitted_background_or_credit"
    return "performed_song"


def is_translated_lyric(
    cue: Cue,
    obs_by_id: Mapping[str, CueObservation | CuePerception],
    scene_cues: Sequence[Cue] = (),
) -> bool:
    """True when a lyric cue should be translated under the Hwang Seok-hee standard."""
    return classify_lyric_cue(cue, obs_by_id, scene_cues) in (
        "parody_or_rhyme_game",
        "performed_song",
        "spoken_with_stray_note",
    )

def ost_kind(cue: Cue) -> str | None:
    """Viewing-pass kind ("sign", "credit", …) of an on-screen-text cue."""
    if cue.source != "ost":
        return None
    m = OST_KIND.match(cue.raw)
    return m.group(1).strip().lower() if m else None


def _may_stay_empty(cue: Cue) -> bool:
    """Lyric, real billing credit card, sound-description-only, or 1-2 token interjection-only cue."""
    kind = ost_kind(cue)
    if kind == "gag_credit":
        return False
    if cue.lyric or (kind == "credit" and not is_gag_credit(cue.text)):
        return True
    rest = _BRACKETED.sub(" ", cue.text)
    tokens = re.sub(r"[^\w\s]|_", " ", rest).lower().split()
    if not tokens:
        return True  # only bracketed descriptions / punctuation
    return len(tokens) <= 2 and all(_INTERJECTION.fullmatch(t) for t in tokens)

def check(
    cue: Cue,
    ko: str,
    settings,
    *,
    observation: CueObservation | CuePerception | None = None,
    retained_fragments: list[RetainedFragment] | None = None,
) -> list[ValidationIssue]:
    """Report display and ordered speaker-turn violations for one cue."""
    issues: list[ValidationIssue] = []

    def flag(rule: str, detail: str) -> None:
        issues.append(ValidationIssue(cue_id=cue.id, rule=rule, detail=detail))

    if not ko.strip():
        # Semantic coverage is scene/event-wide: this cue's act may be retained
        # elsewhere in a lawful N:1 output, or explicitly omitted with a reason.
        if retained_fragments is None and cue.text.strip() and not _may_stay_empty(cue):
            flag("empty", "missing Korean text")
        return issues

    lines = ko.split("\n")
    if len(lines) > 2:
        flag("lines", f"{len(lines)} lines > 2")
    if len(lines) == 1 and settings.cpl_one_line < len(lines[0]) <= settings.cpl_limit:
        flag("cpl", f"one-line cue: {len(lines[0])} chars > {settings.cpl_one_line}")
    for i, line in enumerate(lines, 1):
        if len(line) > settings.cpl_limit:
            flag("cpl", f"line {i}: {len(line)} chars > {settings.cpl_limit}")
        if line != line.strip():
            flag("linebreak", f"line {i}: stray whitespace at edge")
        if line.endswith((".", ",")) and not line.endswith(ELLIPSIS):
            flag("punctuation", f"line {i}: terminal '{line[-1]}'")
        if "…" in line:
            flag("punctuation", f"line {i}: '…' must be '...'")
        if _straight_quotes(line) != line:
            flag("punctuation", f"line {i}: quotes must be straight single quotes")
    if cue.lyric and (not ko.startswith(LYRIC_MARK) or ko.count("♪") != 1):
        flag("punctuation", "lyric cue must start with one '♪ ' and carry no other ♪")
    if _is_dual(lines):
        for i, line in enumerate(lines, 1):
            if not line.startswith("- ") or line.startswith("-  "):
                flag("linebreak", f"line {i}: dual-speaker line must start with '- '")
    ko_dual = _speakers(lines) is not None
    if retained_fragments is not None:
        rendered = render_retained(retained_fragments, settings.cpl_limit, settings.cpl_one_line)
        expected_dual = _speakers(rendered.splitlines()) is not None
        if expected_dual != ko_dual:
            flag("speakers", "Dash structure disagrees with actual known retained voices")
    else:
        en_turns = en_speaker_turns(cue, observation=observation)
        known_turns = bool(
            observation is not None and observation.cue_id == cue.id and observation.turns
        )
        if en_turns == 2 and not ko_dual:
            detail = (
                "known turns 1 and 2 need two ordered '- ' lines; Korean has one turn"
                if known_turns else "English has two speakers; Korean needs two '- ' lines"
            )
            flag("speakers", detail)
        elif en_turns == 1 and ko_dual:
            detail = (
                "known turn 1 must not use dual-speaker '- ' lines; Korean has two turns"
                if known_turns else "English has one speaker; Korean must not use '- ' lines"
            )
            flag("speakers", detail)

    duration_s = (cue.end_ms - cue.start_ms) / 1000
    n_chars = cps_chars(ko)
    if duration_s > 0 and n_chars / duration_s > settings.cps_max:
        flag("cps", f"{n_chars} chars / {duration_s:.1f}s = {n_chars / duration_s:.1f} cps > {settings.cps_max}")
    return issues


def _registers_at(snap: Snapshot, cue_id: str, order: dict[str, int]) -> dict[str, str]:
    """Effective register map at `cue_id`: each scene pair's table level with
    the shift whose span holds the cue applied (`order`: global cue order)."""
    return {pair: reg.level_at(cue_id, order) for pair, reg in snap.registers.items()}


def _speaker_id(hint: str | None, snap: Snapshot) -> str | None:
    """Resolve an SDH speaker hint to a character id present in the snapshot."""
    if not hint:
        return None
    h = hint.strip().lower()
    if not h:
        return None
    if h in snap.character_names:
        return h
    for cid in snap.character_names or snap.characters:
        if cid.lower() == h:
            return cid
    return None


def _register_rows(
    cue: Cue,
    snap: Snapshot | None,
    order: dict[str, int],
) -> list[str]:
    """'cue_id | speaker>addressee | level' rows effective at this cue:
    only the cue's own speaker, or every scene pair if the speaker is unknown."""
    if snap is None:
        return []
    regs = _registers_at(snap, cue.id, order)
    speaker = _speaker_id(cue.speaker_hint, snap)
    if speaker is not None:
        regs = {p: l for p, l in regs.items() if p.partition(">")[0] == speaker}
    return [f"   REG {cue.id} | {pair} | {level}" for pair, level in sorted(regs.items())]


def _required_forms(
    cue: Cue,
    orig_ko: str,
    snap: Snapshot | None,
    ko: dict[str, str],
) -> list[str]:
    """Protected forms present in `orig_ko` that an edit must not drop:
    character names, glossary ko for terms in the English, callback setup KO."""
    if snap is None or not orig_ko:
        return []
    forms: list[str] = []
    for ko_name in snap.character_names.values():
        if ko_name and ko_name in orig_ko:
            forms.append(ko_name)
    en_lower = cue.text.lower()
    for g in snap.glossary:
        if g.en.lower() in en_lower and g.ko and g.ko in orig_ko:
            forms.append(g.ko)
    for cb in snap.callbacks_live:
        if cue.id in cb.payoff_cues:
            if cb.note.startswith("LOCKED_SERIES_TM::"):
                for line in cb.note.removeprefix("LOCKED_SERIES_TM::").split("\n"):
                    phrase = line.strip().lstrip("-").strip()
                    if phrase and phrase in orig_ko:
                        forms.append(phrase)
            for sid in cb.setup_cues:
                for line in ko.get(sid, "").split("\n"):
                    phrase = line.strip().lstrip("-").strip()
                    if phrase and phrase in orig_ko:
                        forms.append(phrase)
    return list(dict.fromkeys(forms))


def _keeps_protected(
    cue: Cue,
    orig_ko: str,
    new_ko: str,
    snap: Snapshot | None,
    ko: dict[str, str],
) -> bool:
    """True if `new_ko` keeps every protected form that `orig_ko` carried."""
    return all(f in new_ko for f in _required_forms(cue, orig_ko, snap, ko))


def _protected_row(cue: Cue, snap: Snapshot | None, ko: dict[str, str]) -> str | None:
    """'   PROTECT <cue_id>: ...' row: preserve-ambiguity note, present
    character KO names, glossary hits in the English, callback setup KO."""
    if snap is None:
        return None
    parts: list[str] = []
    for flag in snap.preserve_ambiguity:
        if flag.cue_id == cue.id:
            parts.append(f"ambiguity: {flag.note}")
    if snap.character_names:
        names = ", ".join(f"{cid}={ko_name}" for cid, ko_name in sorted(snap.character_names.items()))
        parts.append(f"names: {names}")
    en_lower = cue.text.lower()
    hits = [g for g in snap.glossary if g.en.lower() in en_lower]
    if hits:
        parts.append("glossary: " + ", ".join(f"{g.en}={g.ko}" for g in hits))
    for cb in snap.callbacks_live:
        if cue.id in cb.payoff_cues:
            setups = "; ".join(
                f"{sid} KO {ko.get(sid, '').replace(chr(10), ' / ')}"
                for sid in cb.setup_cues
                if ko.get(sid, "").strip()
            )
            if setups:
                parts.append(f"callback {cb.motif} setups: {setups}")
    if not parts:
        return None
    return f"   PROTECT {cue.id}: " + " | ".join(parts)


def _format_cues(
    items: list[tuple[Cue, str, list[ValidationIssue]]],
    snapshots: dict[str, Snapshot],
    scene_of: dict[str, str],
    ko: dict[str, str],
    order: dict[str, int],
    narrative: str,
) -> str:
    rows = []
    for n, (cue, ko_text, issues) in enumerate(items, 1):
        dur = (cue.end_ms - cue.start_ms) / 1000
        budget = int(dur * 12) if dur > 0 else 0
        rules = ",".join(sorted({i.rule for i in issues}))
        speaker = cue.speaker_hint or "?"
        snap = snapshots.get(scene_of.get(cue.id, ""))
        rows.append(
            f"{n}. {cue.id} | {speaker} | {dur:.1f}s (budget ~{budget} chars) | {rules}\n"
            f"   KO: {ko_text.replace(chr(10), ' / ')}"
        )
        rows.extend(_register_rows(cue, snap.for_prompt(narrative) if snap else None, order))
        protected = _protected_row(cue, snap, ko)
        if protected:
            rows.append(protected)
    return "\n".join(rows)


def compress(
    ws,
    gc,
    items: list[tuple[Cue, str, list[ValidationIssue]]],
    snapshots: dict[str, Snapshot],
    scene_of: dict[str, str],
    ko: dict[str, str] | None = None,
    order: dict[str, int] | None = None,
) -> list[CueEdit]:
    """Ask the text model to shorten flagged cues, keeping meaning/register.

    `items` are (cue, current_ko, issues); `ko` is the full current Korean map
    (callback setup lookups) and `order` the global cue index for register
    deltas. Returns edits only for cues the model changed and that kept every
    protected form; callers re-validate the result.
    """
    if not items:
        return []
    ko = ko if ko is not None else {}
    # Fallback order covers flagged cues only: deltas on unknown cue ids apply
    # only to cues placed at/after them in this map.
    order = order if order is not None else {c.id: i for i, (c, _, _) in enumerate(items)}
    narrative = narrative_context(gc.settings)
    prompt = narrative_text(_COMPRESS_PROMPT, narrative).format(
        cues=_format_cues(items, snapshots, scene_of, ko, order, narrative),
        sources="\n".join(f"{cue.id}: {cue.text.replace(chr(10), ' / ')}" for cue, _, _ in items),
    )
    from .gemini import GeminiBlocked  # local: validate stays importable without the SDK

    try:
        result: _CompressBatch = gc.generate(
            stage="s5-compress",
            prompt=prompt,
            schema=_CompressBatch,
            model=gc.settings.model_light,
            system=style_guide(),
        )
    except GeminiBlocked:
        return []  # provider refused: issues stay flagged, texts unchanged
    valid_ids = {cue.id for cue, _, _ in items}
    edits: list[CueEdit] = []
    seen: set[str] = set()
    for e in result.edits:
        if e.cue_id not in valid_ids or not e.new_ko.strip() or e.cue_id in seen:
            continue
        orig = next(k for c, k, _ in items if c.id == e.cue_id)
        cue = next(c for c, _, _ in items if c.id == e.cue_id)
        snap = snapshots.get(scene_of.get(e.cue_id, ""))
        if not _keeps_protected(cue, orig, e.new_ko, snap, ko):
            continue  # dropped a protected form: keep the original text
        seen.add(e.cue_id)
        edits.append(e)
    return edits


def run_s5(
    ws,
    gc,
    cues: list[Cue],
    ko: dict[str, str],
    snapshots: dict[str, Snapshot],
    scene_of: dict[str, str],
    settings,
    *,
    observations_by_cue: Mapping[str, CueObservation | CuePerception] | None = None,
    lineage: SemanticLineage | None = None,
) -> tuple[dict[str, str], list[ValidationIssue], set[str]]:
    """finish (normalize -> speakers -> rewrap -> lyric mark) -> check ->
    compress stragglers -> finish -> check.

    Returns (final ko map, still-unresolved issues, ids whose text changed).
    """
    out: dict[str, str] = {}
    observations = observations_by_cue or {}
    if lineage is not None:
        reconcile_lineage(lineage, ko)
    retained = lineage.fragments_by_cue if lineage is not None else {}
    for cue in cues:
        text = ko.get(cue.id, "")
        out[cue.id] = (
            finish(cue, text, settings, observation=observations.get(cue.id),
                   retained_fragments=retained.get(cue.id))
            if text.strip() else text
        )

    by_id = {cue.id: cue for cue in cues}
    flagged: list[tuple[Cue, str, list[ValidationIssue]]] = []
    for cue in cues:
        issues = [
            i for i in check(cue, out[cue.id], settings, observation=observations.get(cue.id),
                             retained_fragments=retained.get(cue.id))
            if i.rule in _FIXABLE_RULES
        ]
        if issues:
            flagged.append((cue, out[cue.id], issues))

    cue_order = {c.id: i for i, c in enumerate(cues)}
    for e in compress(ws, gc, flagged, snapshots, scene_of, ko=out, order=cue_order):
        if e.cue_id in by_id:
            if lineage is not None:
                reconcile_lineage(lineage, {**out, e.cue_id: e.new_ko})
            out[e.cue_id] = finish(
                by_id[e.cue_id], e.new_ko, settings, observation=observations.get(e.cue_id),
                retained_fragments=retained.get(e.cue_id),
            )

    unresolved = [
        issue for cue in cues
        for issue in check(cue, out[cue.id], settings, observation=observations.get(cue.id),
                           retained_fragments=retained.get(cue.id))
    ]
    if lineage is not None:
        reconcile_lineage(lineage, out)
        unresolved.extend(lineage_issues(lineage))
    changed = {cue.id for cue in cues if out[cue.id] != ko.get(cue.id, "")}
    return out, unresolved, changed
