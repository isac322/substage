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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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


_KOREAN_BREAK = re.compile(r"(?:면|서|고|는데|지만|니까|거든|잖아)(?:요)?$|[,!?…~.]$")


def _wrap_line(text: str, cpl: int, one_line: int) -> list[str]:
    """Split a single line at 어절 (space) boundaries into <=2 lines.

    Text of <= `one_line` characters stays one line. Among splits fitting
    `cpl`, prefer Korean connective, particle, or punctuation boundaries so
    balancing does not split a noun phrase. Then minimize the maximum line
    length, preferring bottom-heavy splits and finally the most balanced one.
    A single 어절 is never split, so a word longer than cpl is kept whole and
    reported by check().
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
    fitting = [split for split in splits if max(split[1], split[2]) <= cpl]
    boundaries = [split for split in fitting if _KOREAN_BREAK.search(words[split[0] - 1])]
    pool = boundaries or fitting or splits
    min_max = min(max(l1, l2) for _, l1, l2 in pool)
    cands = [(i, l1, l2) for i, l1, l2 in pool if max(l1, l2) == min_max]
    pref = [c for c in cands if c[2] >= c[1]]
    i = max(pref, key=lambda c: c[1])[0] if pref else min(cands, key=lambda c: c[1])[0]
    return [" ".join(words[:i]), " ".join(words[i:])]


def rewrap(ko: str, cpl: int, one_line: int | None = None) -> str:
    """Rewrap Korean cue text at 어절 boundaries into <=2 lines.

    `one_line` (default `cpl`) is the longest text kept on a single line;
    official subtitles break anything longer into two lines of <= `cpl`.
    Never splits inside a word: a single 어절 longer than `cpl` stays whole
    (check() reports it). Keep a fitting translator break unless a Korean
    connective, particle, or punctuation boundary avoids splitting a phrase;
    otherwise two-line splits prefer line2 >= line1. Dual-speaker cues keep
    exactly two '- text' lines, one speaker each, even above `cpl` (check()
    then flags cpl).
    """
    lines = [" ".join(line.split()) for line in strip_clause_slashes(ko).strip().splitlines() if line.strip()]
    if not lines:
        return ""
    if _is_dual(lines):
        turns = _speakers(lines)
        if turns is not None:
            return "\n".join(_dual_line(t) for t in turns)
    if len(lines) == 2 and all(len(line) <= cpl for line in lines):
        w0 = lines[0].split(" ")
        stray_tail = len(w0) >= 2 and bool(_KOREAN_BREAK.search(w0[-2])) and not bool(_KOREAN_BREAK.search(w0[-1]))
        if not stray_tail:
            return "\n".join(lines)
    wrapped = _wrap_line(" ".join(lines), cpl, cpl if one_line is None else one_line)
    return "\n".join(wrapped)


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


_CLAUSE_SLASH = re.compile(r"(?<=\S)\s+/\s*(?=\S)|(?<=\S)\s*/\s+(?=\S)")


def strip_clause_slashes(text: str) -> str:
    """Turn spaced clause separators into line breaks, preserving fixed tokens."""
    text = re.sub(r"[ \t]*/[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*/[ \t]*", "\n", text)
    text = re.sub(r"^\s*/\s*|\s*/\s*$", "", text)
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


BGM_LYRIC_SUFFIX = " ♪"
_NAMED_CHARACTER_TOKENS = frozenset({
    "travis", "kalanick", "bill", "gurley", "emil", "michael", "austin", "geidt",
    "gabi", "holzwarth", "angi", "angie", "you", "sergey", "brin", "larry", "page",
    "tim", "cook", "david", "bonderman", "arianna", "huffington", "garrett", "camp",
    "ryan", "graves", "quentin", "pierre", "fawzi", "kamal", "susan", "fowler",
    "eric", "alexander", "holder", "charles", "oliver", "mabel", "deadpool", "wade", "logan",
})


def is_bgm_lyric(cue: Cue | None, observation: CueObservation | CuePerception | None = None) -> bool:
    """Return True when a cue is background music (wrapped with `♪ ... ♪`) rather than a character-sung song (`♪ ...`)."""
    if cue is None or cue.id.startswith("ost-") or cue.source == "ost":
        return False
    if observation is not None and getattr(observation, "lyric_plot_relevant", None) is False:
        return True
    if not cue.lyric and (observation is None or not getattr(observation, "lyric", False)):
        return False
    if observation is not None and getattr(observation, "lyric_plot_relevant", None) is True:
        spk = (getattr(observation, "speaker", None) or cue.speaker_hint or "").strip().lower()
        if any(tok in spk for tok in _NAMED_CHARACTER_TOKENS):
            return False
    return False


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
    bgm = is_bgm_lyric(cue, observation)
    lyric = (cue is not None and cue.lyric) or bgm
    if lyric:
        text = text.replace("♪", "").strip()
        reserve = len(LYRIC_MARK) + (len(BGM_LYRIC_SUFFIX) if bgm else 0)
        cpl, one_line = cpl - reserve, one_line - reserve
    if retained_fragments is not None:
        fragments = [
            fragment.model_copy(update={"text": fragment.text.replace("♪", "").strip()})
            for fragment in retained_fragments
        ] if lyric else retained_fragments
        text = render_retained(fragments, cpl, one_line)
    else:
        if cue is not None:
            text = enforce_speakers(cue, text, cpl, one_line, observation=observation)
        text = rewrap(text, cpl, one_line)
    if not text:
        return text
    if bgm:
        return f"{LYRIC_MARK}{text}{BGM_LYRIC_SUFFIX}"
    return LYRIC_MARK + text if lyric else text

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
    bgm_display = is_bgm_lyric(cue, observation)
    lyric_display = (cue is not None and cue.lyric) or bgm_display
    display_lines = list(lines)
    if bgm_display and display_lines and display_lines[-1].endswith(BGM_LYRIC_SUFFIX):
        display_lines[-1] = display_lines[-1][: -len(BGM_LYRIC_SUFFIX)]
    if len(display_lines) > 2:
        flag("lines", f"{len(lines)} lines > 2")
    if len(display_lines) == 1 and settings.cpl_one_line < len(display_lines[0]) <= settings.cpl_limit:
        flag("cpl", f"one-line cue: {len(display_lines[0])} chars > {settings.cpl_one_line}")
    for i, line in enumerate(display_lines, 1):
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
    if bgm_display:
        if not (ko.startswith(LYRIC_MARK) and ko.endswith(BGM_LYRIC_SUFFIX) and ko.count("♪") == 2):
            flag("punctuation", "BGM lyric cue must be wrapped as '♪ ... ♪'")
    elif lyric_display and (not ko.startswith(LYRIC_MARK) or ko.count("♪") != 1):
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


@dataclass(frozen=True)
class _ContinuityEvent:
    cue_id: str
    start_ms: int
    end_ms: int
    text: str
    cue_ids: tuple[str, ...] = ()
    speaker: str | None = None
    addressee: str | None = None


_REACTION_QUESTION = re.compile(r"^(?:(?:그래|정말|진짜|뭐|왜|네|응|어|설마)\s*[?!]\s*)+")
_DESCRIPTIVE_CALQUE = re.compile(
    r"(?:앤커|앵커)\s*(?:비잉|존재)|거트\s*밀크|블라인드\s*알|"
    r"(?:보조\s*바퀴|심문\s*강화|최후의\s*날)\s*프로토콜"
)


def _event_field(event: object, key: str, default: object = None) -> object:
    return event.get(key, default) if isinstance(event, Mapping) else getattr(event, key, default)


def check_utterance_span_continuity(
    en_cues: Sequence[Cue],
    ko_events: Sequence[object] | Mapping[str, str],
) -> list[ValidationIssue]:
    """Audit displayed dialogue, including tracks without workspace metadata.

    Cue identities take precedence over temporal alignment; a merged display
    event is never mistaken for two independently translated questions.
    Unknown-speaker register checks require an established two-event register
    run, rather than treating every neighboring speaker change as oscillation.
    """
    from .levels import addressed_levels

    cues = sorted(
        (cue for cue in en_cues if cue.source != "ost" and not cue.lyric),
        key=lambda cue: (cue.start_ms, cue.end_ms),
    )
    cues_by_id = {cue.id: cue for cue in cues}
    causal_cue_ids: set[str] = set()
    causal_continuation = False
    previous_cue: Cue | None = None
    for cue in cues:
        if previous_cue is not None and (
            cue.start_ms - previous_cue.end_ms > 3000
            or (cue.speaker_hint and previous_cue.speaker_hint and cue.speaker_hint != previous_cue.speaker_hint)
        ):
            causal_continuation = False
        source = cue.text.strip().rstrip("\"'”’")
        causal = "?" not in source and (
            causal_continuation or bool(re.search(r"\bbecause\b", source, re.I))
        )
        if causal:
            causal_cue_ids.add(cue.id)
        causal_continuation = causal and (
            source.endswith(("...", "…", ",")) or bool(re.search(r"\w$", source))
        )
        previous_cue = cue
    events: list[_ContinuityEvent] = []
    if isinstance(ko_events, Mapping):
        for cid, text in ko_events.items():
            cue = cues_by_id.get(cid)
            if cue is not None and text.strip():
                events.append(_ContinuityEvent(
                    cid, cue.start_ms, cue.end_ms, text, (cid,), cue.speaker_hint,
                ))
    else:
        for index, event in enumerate(ko_events):
            if _event_field(event, "ost", False) or _event_field(event, "position") == "an8":
                continue
            text = str(_event_field(event, "ko", _event_field(event, "text", "")))
            text = re.sub(r"\{\\an\d+\}|<[^>]*>", "", text).strip()
            if not text or "♪" in text:
                continue
            ids = tuple(str(cid) for cid in (_event_field(event, "cue_ids", ()) or ()))
            cid = str(_event_field(event, "cue_id", "") or (ids[0] if ids else f"event@{index}"))
            speaker = _event_field(event, "speaker")
            addressee = _event_field(event, "addressee")
            events.append(_ContinuityEvent(
                cid, int(str(_event_field(event, "start_ms", 0))),
                int(str(_event_field(event, "end_ms", 0))), text, ids,
                str(speaker) if speaker else None, str(addressee) if addressee else None,
            ))
    events.sort(key=lambda event: (event.start_ms, event.end_ms))
    issues: list[ValidationIssue] = []
    seen: set[tuple[str, str]] = set()

    def flag(event: _ContinuityEvent, rule: str, detail: str) -> None:
        key = (event.cue_id, rule)
        if key not in seen:
            seen.add(key)
            issues.append(ValidationIssue(cue_id=event.cue_id, rule=rule, detail=detail))

    by_cue: dict[str, list[int]] = {cue.id: [] for cue in cues}
    sources: list[list[Cue]] = []
    left = 0
    for index, event in enumerate(events):
        while left < len(cues) and cues[left].end_ms <= event.start_ms:
            left += 1
        if event.cue_ids:
            matched = [cues_by_id[cid] for cid in event.cue_ids if cid in cues_by_id]
        else:
            matched = []
            right = left
            while right < len(cues) and cues[right].start_ms < event.end_ms:
                if cues[right].end_ms > event.start_ms:
                    matched.append(cues[right])
                right += 1
        sources.append(matched)
        for cue in matched:
            by_cue[cue.id].append(index)
        lines = [line.strip() for line in event.text.splitlines() if line.strip()]
        dashed = [bool(_EN_TURN_DASH.match(line)) for line in lines]
        if (len(lines) == 1 and any(dashed)) or (len(lines) == 2 and sum(dashed) == 1):
            flag(event, "orphan_speaker_dash", "One speaker must not carry a lone/asymmetric leading dash")
        if "…" in event.text:
            flag(event, "unicode_ellipsis", "Dialogue ellipsis must be three ASCII periods")
        if _straight_quotes(event.text) != event.text:
            flag(event, "unnormalized_quotes", "Dialogue quotes must be straight single quotes")
        if _DESCRIPTIVE_CALQUE.search(event.text):
            flag(event, "descriptive_concept_transliteration", "Translate descriptive meaning, not phonetic labels")
        if re.search(r"인물(?:를|가|는|면|라고|고)(?=\s|[.?!,…'\"”’)\]]|$)", event.text):
            flag(event, "ungrammatical_josa", "Batchim noun stem carries vowel-only particle")
        source = " ".join(cue.text for cue in matched)
        causal_clauses = [
            clause for clause in re.split(r"(?<=[.!?])\s+", source)
            if re.search(r"\bbecause\b", clause, re.I) and "?" not in clause
        ]
        body = _REACTION_QUESTION.sub("", event.text.strip())
        if (causal_clauses or any(cue.id in causal_cue_ids for cue in matched)) and re.search(r"(?:다고|냐고)\s*\?[\"'’”]*$", body):
            flag(event, "causal_clause_question", "Declarative because-clause was changed into a question")

    for first, second in zip(cues, cues[1:]):
        if second.start_ms - first.end_ms > 3000:
            continue
        if first.speaker_hint and second.speaker_hint and first.speaker_hint != second.speaker_hint:
            continue
        first_matches, second_matches = by_cue[first.id], by_cue[second.id]
        if not first_matches or not second_matches:
            continue
        a, b = first_matches[-1], second_matches[0]
        if a >= b:
            continue
        source = first.text.strip().rstrip("\"'”’")
        continuing = source.endswith(("...", "…", ",")) or bool(re.search(r"[\w]$", source))
        if not continuing or "?" not in second.text:
            continue
        # A complete question before a fragment is a separate speech act.
        tail = source.rsplit("?", 1)[-1].strip()
        if not tail:
            continue
        ko_a = _REACTION_QUESTION.sub("", events[a].text.strip()).rstrip("\"'”’")
        ko_b = _REACTION_QUESTION.sub("", events[b].text.strip()).rstrip("\"'”’")
        if ko_a.endswith("?") and ko_b.endswith("?"):
            flag(events[a], "premature_question_split", f"Continuing source question closes twice across {first.id}/{second.id}")
        clean_ka = re.sub(r"[^가-힣]", "", ko_a)
        clean_kb = re.sub(r"[^가-힣]", "", ko_b)
        _stop_en = {"that", "this", "what", "with", "from", "have", "your", "they", "will", "just", "the", "and", "for", "are", "but", "not", "you"}
        en_wa = {w for w in re.findall(r"[a-z]{3,}", first.text.casefold())} - _stop_en
        en_wb = {w for w in re.findall(r"[a-z]{3,}", second.text.casefold())} - _stop_en
        if a != b and len(clean_ka) >= 4 and len(clean_kb) >= 6 and en_wa.isdisjoint(en_wb):
            if clean_ka in clean_kb or (len(clean_ka) >= 5 and clean_ka[:-1] in clean_kb):
                flag(events[b], "adjacent_dialogue_echoes", f"Continuation repeats clause from {first.id} in {second.id}")
    slogan_forms: dict[str, list[_ContinuityEvent]] = {}
    for event in events:
        for form in ("성장 아니면 죽음", "성장 아니면 사장"):
            if re.sub(r"\s+", "", form) in re.sub(r"\s+", "", event.text):
                slogan_forms.setdefault(form, []).append(event)
    if len(slogan_forms) > 1:
        for matches in slogan_forms.values():
            for event in matches:
                flag(event, "slogan_inconsistency", "The same track uses conflicting translations of one slogan")

    def flat(text: str) -> str:
        return re.sub(r"[^\w가-힣]", "", text).casefold()

    def pair_at(index: int) -> tuple[str | None, str | None]:
        event = events[index]
        hints = {cue.speaker_hint for cue in sources[index] if cue.speaker_hint}
        speaker = event.speaker or (next(iter(hints)) if len(hints) == 1 else None)
        return speaker, event.addressee

    level_cache: dict[int, str | None] = {}

    def level_at(idx: int) -> str | None:
        if idx not in level_cache:
            addressed = set(addressed_levels(events[idx].text)) - {None}
            polite = addressed & {"haeyo", "hasipsio", "hao", "haseo"}
            informal = addressed & {"hae", "haera"}
            level_cache[idx] = "J" if polite and not informal else "B" if informal and not polite else None
        return level_cache[idx]

    for index in range(1, len(events)):
        previous, event = events[index - 1], events[index]
        gap = event.start_ms - previous.end_ms
        if gap > 3000 or gap < -500:
            continue
        if gap <= 2000 and len(flat(previous.text)) >= 6:
            source_a = " ".join(cue.text for cue in sources[index - 1])
            source_b = " ".join(cue.text for cue in sources[index])
            _stop_en = {"that", "this", "what", "with", "from", "have", "your", "they", "will", "just", "the", "and", "for", "are", "but", "not", "you"}
            en_wa = {w for w in re.findall(r"[a-z]{3,}", source_a.casefold())} - _stop_en
            en_wb = {w for w in re.findall(r"[a-z]{3,}", source_b.casefold())} - _stop_en
            exact_dup = flat(event.text) == flat(previous.text) and ((not source_a and not source_b) or (flat(source_a) != flat(source_b) and en_wa.isdisjoint(en_wb)))
            sub_dup = flat(previous.text) in flat(event.text) and en_wa.isdisjoint(en_wb)
            if exact_dup or sub_dup:
                flag(event, "adjacent_dialogue_echoes", f"Different source dialogue overwrites {previous.cue_id} with identical Korean")
        if any(_EN_TURN_DASH.match(line) for ev in (previous, event) for line in ev.text.splitlines()):
            continue
        pair_a, pair_b = pair_at(index - 1), pair_at(index)
        if any(a and b and a != b for a, b in zip(pair_a, pair_b)):
            continue
        same_pair = bool(pair_a[0] and pair_a[0] == pair_b[0] and pair_a[1] == pair_b[1])
        self_intro_flip = bool(
            index >= 2
            and 0 <= previous.start_ms - events[index - 2].end_ms <= 2500
            and not any(_EN_TURN_DASH.match(line) for line in events[index - 2].text.splitlines())
            and (
                (
                    re.search(r"(?:^|\s)(?:전|제가|저도|저는)(?:\s|$)", events[index - 2].text + " " + previous.text)
                    and re.search(r"(?:^|\s)(?:난|내가|날|내|나는)(?:\s|$)", event.text)
                )
                or (
                    re.search(r"(?:^|\s)(?:난|내가|날|내|나는)(?:\s|$)", events[index - 2].text + " " + previous.text)
                    and re.search(r"(?:^|\s)(?:전|제가|저도|저는)(?:\s|$)", event.text)
                )
            )
        )
        if not (same_pair or self_intro_flip):
            continue
        lvl_prev, lvl_curr = level_at(index - 1), level_at(index)
        if not lvl_prev or not lvl_curr or lvl_prev == lvl_curr:
            continue
        if same_pair or (index >= 2 and level_at(index - 2) == lvl_prev):
            flag(event, "srt_register_oscillation", f"Same-speaker run switches {lvl_prev}->{lvl_curr} within 3s")
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
