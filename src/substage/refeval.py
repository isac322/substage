"""Reference-based evaluation: agreement with the official Korean subtitles.

The official (human, professional) Korean subtitles are the correct answer,
condensation and paraphrase included. A candidate (an experiment arm's final
``<video_id>.ko.srt``) is scored by how closely it agrees with them. The
official file is read only here, as evaluation gold; the translation pipeline
never reads it.

1. Alignment (``align``): English cues, official lines and every candidate's
   lines become nodes; two nodes of different tracks are linked when their
   time spans overlap by >= ``MIN_OVERLAP_MS`` or by >= ``MIN_OVERLAP_FRAC`` of
   the shorter one. A group is a connected component, so 1:N and N:1 splits
   land in one group. Official lines nothing overlaps (candidate omission) and
   candidate lines nothing overlaps (extra lines) keep their own groups.
2. Deterministic metrics (``metrics``): corpus chrF++ (sacreBLEU-compatible:
   char 6-grams without whitespace + word 1-2-grams, beta 2), coverage,
   length ratio, speech-level agreement, dual-speaker dash agreement, and
   CPS > cps_max / CPL > cpl rates, median CPS, median display duration and
   the event count ratio for the official file (the acceptable range) and
   each candidate's displayed events.
3. LLM judge (``judge_windows``): seeded, stratified windows of ~8 groups;
   every arm is judged in one call against the same gold under per-window
   shuffled labels (arm names never reach the prompt).
4. Calibration (``calibrate``): the same judge on the official itself and on a
   deterministically degraded official; the judge is trusted only if the
   former scores >= 95 with ~no errors and the latter clearly lower.

Opt-in paired evaluation (``evaluate_paired`` / ``run_paired_refeval``) isolates
one arm and a frozen Opus subtitle artifact on one fixed EN+official
alignment. Candidate/Opus events are attached to the fixed groups without
letting their timing redefine groups or windows. Each side is judged
point-wise against official gold with content-specific checkpoints, never in
a candidate ranking. It reports paired deterministic/judge deltas, and source
meaning/safe-harbor separately. Calibration uses versioned official, degraded,
and meaning-preserving paraphrase anchors. Legacy listwise behavior remains the
default. Dev/held-out metadata is descriptive; the cycle orchestrator controls
which split may be read.

Speech levels (``line_levels``, ``group_level``) come from
``levels.addressed_levels``: one level per dash turn, None for fragments,
connective endings, -네/-군 exclamatives and vocatives.
"""

from __future__ import annotations

import html
import json
import random
import re
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pysubs2
from pydantic import BaseModel, Field

from .config import Settings, Workspace, load_settings, workspace
from .evaluate import _en_text, arm_srt, evaluation_metadata, mmss, numeric_deltas, shuffle_labels
from .levels import DASHES, addressed_levels, ending_level, jongseong, strip_bidi
from .llm import ClefDecisionsClient
from .models import ChunkObservations, Cue, CueObservation, ContextMap, RegisterMap, Scene
from .style import PRO_SUBTITLE_LAWS
from .store import Checkpoints, input_hash, load_model, load_models
from .validate import cps_chars

STAGE = "refeval-judge"
OFFICIAL = "official"
DEGRADED = "degraded"

PARAPHRASE = "paraphrase"
PAIRED_STAGE = "refeval-pointwise"
PAIRED_CANDIDATE = "candidate"
PAIRED_OPUS = "opus"
MIN_OVERLAP_MS = 200
MIN_OVERLAP_FRAC = 0.30

OTHER = "other"

CATEGORIES: tuple[str, ...] = (
    "meaning", "register", "name_term", "verbosity", "omission", "unnatural", "dash_format", "none",
)
ERROR_CATEGORIES: tuple[str, ...] = CATEGORIES[:-1]

WINDOW_TARGET = 8
WINDOW_MIN = 6
WINDOW_MAX = 10
TARGETED_SHARE = 0.5

CALIB_OFFICIAL_MIN = 95.0
CALIB_OFFICIAL_MAX_ERROR_RATE = 0.05  # share of items with severity >= 1
CALIB_MIN_GAP = 15.0  # degraded must score at least this much lower

_ENCODINGS = ("utf-8-sig", "cp949")


# ---------------------------------------------------------------- loading


@dataclass(frozen=True)
class Line:
    """One displayed Korean subtitle event; ``text`` keeps display line breaks."""

    idx: int  # position within its track (time order)
    start_ms: int
    end_ms: int
    text: str


def _decode(path: Path) -> str:
    raw = Path(path).read_bytes()
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"could not decode {path} (tried {_ENCODINGS})")


_SYNC_RE = re.compile(r"<sync\b[^>]*?\bstart\s*=\s*[\"']?(-?\d+)[^>]*>", re.I)
_P_CLASS_RE = re.compile(r"<p\b[^>]*?\bclass\s*=\s*[\"']?([\w-]+)", re.I)
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_TAG_RE = re.compile(r"<[^>]*>")
_SAMI_TAIL_MS = 3000  # duration of a last SYNC block nothing clears


def _clean_markup(body: str) -> str:
    body = _BR_RE.sub("\n", body)
    body = _TAG_RE.sub("", body)
    body = strip_bidi(html.unescape(body)).replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in body.split("\n")]
    return "\n".join(line for line in lines if line)


def parse_sami(text: str) -> list[Line]:
    """SAMI → lines. A SYNC block lasts until the next SYNC; an ``&nbsp;``-only
    block is a clear. With several ``<P Class>`` languages, Korean (``*KR*``)
    blocks win."""
    end_body = re.search(r"</body\s*>", text, re.I)
    if end_body:
        text = text[: end_body.start()]
    marks = list(_SYNC_RE.finditer(text))
    classes = {m.group(1).lower() for m in _P_CLASS_RE.finditer(text)}
    want_kr = len(classes) > 1 and any("kr" in c for c in classes)
    blocks: list[tuple[int, str]] = []
    for i, m in enumerate(marks):
        body = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        if want_kr:
            cm = _P_CLASS_RE.search(body)
            if cm and "kr" not in cm.group(1).lower():
                continue
        blocks.append((int(m.group(1)), _clean_markup(body)))
    out: list[Line] = []
    for i, (start, body) in enumerate(blocks):
        if not body:
            continue
        end = blocks[i + 1][0] if i + 1 < len(blocks) else start + _SAMI_TAIL_MS
        if end <= start:
            continue
        out.append(Line(len(out), start, end, body))
    return out


def load_ko(path: Path) -> list[Line]:
    """Korean subtitle file (SRT/ASS/VTT or SAMI; utf-8 or cp949) → lines in time order."""
    path = Path(path)
    text = _decode(path)
    if path.suffix.lower() in (".smi", ".sami"):
        lines = parse_sami(text)
    else:
        subs = pysubs2.SSAFile.from_string(text)
        events = [
            (int(ev.start), int(ev.end), _clean_markup(ev.plaintext.replace("\n", "<br>")))
            for ev in subs.events
            if not ev.is_comment
        ]
        lines = [Line(0, s, e, t) for s, e, t in events if t and e > s]
    lines.sort(key=lambda ln: (ln.start_ms, ln.end_ms))
    return [Line(i, ln.start_ms, ln.end_ms, ln.text) for i, ln in enumerate(lines)]


# ---------------------------------------------------------------- alignment


@dataclass
class Group:
    id: str
    start_ms: int
    end_ms: int
    en: list[Cue] = field(default_factory=list)
    official: list[Line] = field(default_factory=list)
    candidates: dict[str, list[Line]] = field(default_factory=dict)
    scene_id: str | None = None

    def ko(self, track: str) -> list[Line]:
        return self.official if track == OFFICIAL else self.candidates.get(track, [])

    def text(self, track: str) -> str:
        return "\n".join(ln.text for ln in self.ko(track))

    @property
    def has_korean(self) -> bool:
        return bool(self.official) or any(self.candidates.values())


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    ov = min(a_end, b_end) - max(a_start, b_start)
    if ov <= 0:
        return False
    shorter = max(1, min(a_end - a_start, b_end - b_start))
    return ov >= MIN_OVERLAP_MS or ov >= MIN_OVERLAP_FRAC * shorter


def align(en: list[Cue], official: list[Line], candidates: dict[str, list[Line]]) -> list[Group]:
    """Connected components of cross-track time overlaps (see module docstring)."""
    nodes: list[tuple[str, int, int, int]] = [("en", i, c.start_ms, c.end_ms) for i, c in enumerate(en)]
    nodes += [(OFFICIAL, i, ln.start_ms, ln.end_ms) for i, ln in enumerate(official)]
    for arm, lines in candidates.items():
        nodes += [(arm, i, ln.start_ms, ln.end_ms) for i, ln in enumerate(lines)]
    order = sorted(range(len(nodes)), key=lambda k: (nodes[k][2], nodes[k][3]))

    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    active: list[int] = []
    for k in order:
        track, _, s, e = nodes[k]
        active = [a for a in active if nodes[a][3] > s]
        for a in active:
            if nodes[a][0] != track and overlaps(nodes[a][2], nodes[a][3], s, e):
                ra, rk = find(a), find(k)
                if ra != rk:
                    parent[ra] = rk
        active.append(k)

    comps: dict[int, list[int]] = {}
    for k in range(len(nodes)):
        comps.setdefault(find(k), []).append(k)

    groups: list[Group] = []
    for members in comps.values():
        g = Group(
            id="",
            start_ms=min(nodes[k][2] for k in members),
            end_ms=max(nodes[k][3] for k in members),
            candidates={arm: [] for arm in candidates},
        )
        for k in sorted(members, key=lambda k: (nodes[k][2], nodes[k][3])):
            track, i, _, _ = nodes[k]
            if track == "en":
                g.en.append(en[i])
            elif track == OFFICIAL:
                g.official.append(official[i])
            else:
                g.candidates[track].append(candidates[track][i])
        groups.append(g)
    groups.sort(key=lambda g: (g.start_ms, g.end_ms))
    for i, g in enumerate(groups, 1):
        g.id = f"g{i:04d}"
    return groups


def _attach_track(
    groups: list[Group], track: str, lines: list[Line],
) -> tuple[list[Group], list[list[Group]]]:
    """Attach events once and partition fixed groups into track-local lexical units.

    Each event goes to the group with the greatest credible EN/official
    overlap; ties resolve in frozen video order. Every overlap joins groups
    for lexical scoring only;
    fixed IDs, attachment, and judge items stay unchanged. Unmatched events
    keep their own timing/text in source/gold-free extras. Units are not a
    timing-quality verdict: rotations and oversized overlays still need the
    fixed-group diagnostic, display metrics, and pointwise judgment.
    """
    extras: list[Group] = []
    parent = list(range(len(groups)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for group in groups:
        group.candidates.setdefault(track, [])
    for line in sorted(lines, key=lambda item: (item.start_ms, item.end_ms, item.idx)):
        best: tuple[int, int, int] | None = None
        matched: list[int] = []
        for index, group in enumerate(groups):
            evidence = [*group.en, *group.official]
            overlap = max(
                (
                    min(line.end_ms, item.end_ms) - max(line.start_ms, item.start_ms)
                    for item in evidence
                    if overlaps(line.start_ms, line.end_ms, item.start_ms, item.end_ms)
                ),
                default=0,
            )
            if overlap:
                matched.append(index)
                candidate = (overlap, -index, 0)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            extras.append(Group(
                id=f"{track}-x{len(extras) + 1:04d}",
                start_ms=line.start_ms,
                end_ms=line.end_ms,
                candidates={track: [line]},
            ))
        else:
            groups[-best[1]].candidates[track].append(line)
            for index in matched:
                parent[root(index)] = root(-best[1])
    for group in groups:
        group.candidates[track].sort(key=lambda item: (item.start_ms, item.end_ms, item.idx))
    units: dict[int, list[Group]] = {}
    for index, group in enumerate(groups):
        units.setdefault(root(index), []).append(group)
    return extras, list(units.values())


def _clone_groups(groups: list[Group]) -> list[Group]:
    """Copy fixed groups while preserving their source/gold identity and IDs."""
    return [
        Group(
            id=group.id,
            start_ms=group.start_ms,
            end_ms=group.end_ms,
            en=list(group.en),
            official=list(group.official),
            candidates={},
            scene_id=group.scene_id,
        )
        for group in groups
    ]


def assign_scenes(groups: list[Group], scenes: list[Scene]) -> None:
    """Scene of the group's first English cue; else the scene overlapping it
    most; else the nearest scene in time."""
    if not scenes:
        return
    by_cue = {cid: s.id for s in scenes for cid in s.cue_ids}
    for g in groups:
        sid = next((by_cue[c.id] for c in g.en if c.id in by_cue), None)
        if sid is None:
            best = max(scenes, key=lambda s: min(s.end_ms, g.end_ms) - max(s.start_ms, g.start_ms))
            if min(best.end_ms, g.end_ms) - max(best.start_ms, g.start_ms) > 0:
                sid = best.id
            else:
                mid = (g.start_ms + g.end_ms) / 2
                sid = min(scenes, key=lambda s: min(abs(s.start_ms - mid), abs(s.end_ms - mid))).id
        g.scene_id = sid


# ---------------------------------------------------------------- chrF++

_PUNCTS = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
CHAR_ORDER = 6
WORD_ORDER = 2
BETA = 2


def _words(sent: str) -> list[str]:
    out: list[str] = []
    for w in sent.split():
        if len(w) == 1:
            out.append(w)
        elif w[-1] in _PUNCTS:
            out += [w[:-1], w[-1]]
        elif w[0] in _PUNCTS:
            out += [w[0], w[1:]]
        else:
            out.append(w)
    return out


def _ngrams(sent: str) -> list[Counter]:
    chars = "".join(sent.split())
    out = [Counter(chars[i:i + n] for i in range(len(chars) - n + 1)) for n in range(1, CHAR_ORDER + 1)]
    words = _words(sent)
    out += [Counter(" ".join(words[i:i + n]) for i in range(len(words) - n + 1)) for n in range(1, WORD_ORDER + 1)]
    return out


def chrf_stats(hyp: str, ref: str) -> list[int]:
    """Flattened [hyp, ref, match] counts per order (char 1-6, word 1-2)."""
    stats: list[int] = []
    for h, r in zip(_ngrams(hyp), _ngrams(ref)):
        stats += [sum(h.values()), sum(r.values()), sum(min(c, r[g]) for g, c in h.items() if g in r)]
    return stats


def chrf_score(stats: list[int]) -> float:
    """sacreBLEU 2.x chrF++ (effective-order averaging of precision/recall)."""
    eps = 1e-16
    prec_sum = rec_sum = 0.0
    eff = 0
    for i in range(len(stats) // 3):
        n_hyp, n_ref, n_match = stats[3 * i: 3 * i + 3]
        if n_hyp > 0 and n_ref > 0:
            prec_sum += n_match / n_hyp
            rec_sum += n_match / n_ref
            eff += 1
    if eff == 0:
        return 0.0
    p, r = prec_sum / eff, rec_sum / eff
    f = BETA**2
    return 100 * (1 + f) * p * r / (f * p + r) if p + r > eps else 0.0


def corpus_chrf(pairs: list[tuple[str, str]]) -> float:
    total = [0] * (3 * (CHAR_ORDER + WORD_ORDER))
    for hyp, ref in pairs:
        total = [a + b for a, b in zip(total, chrf_stats(hyp, ref))]
    return chrf_score(total)


# ---------------------------------------------------------------- speech level

_HANGUL_TOKEN_RE = re.compile(r"([가-힣]+)([^가-힣]*)$")
_JONG_SS = 20  # ㅆ as a final consonant


def line_levels(text: str) -> list[str]:
    """The levels of a line's speaker turns that have one."""
    return [lv for lv in addressed_levels(text) if lv is not None]


def group_level(lines: list[Line]) -> str:
    """Most common utterance level in the group (ties → the latest); else other."""
    levels = [lv for ln in lines for lv in line_levels(ln.text)]
    if not levels:
        return OTHER
    counts = Counter(levels)
    top = max(counts.values())
    return next(lv for lv in reversed(levels) if counts[lv] == top)


def has_dash(lines: list[Line]) -> bool:
    return any(part.strip().startswith(DASHES) for ln in lines for part in ln.text.split("\n"))


# ---------------------------------------------------------------- metrics


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def display_rates(lines: list[Line], settings: Settings) -> dict[str, float | None]:
    """Reading-speed and layout metrics of displayed events (lines): share over
    cps_max (validate.cps_chars per second, as validate.check / layout count)
    and over cpl, median CPS and median duration in seconds."""
    if not lines:
        return {"cps_over_rate": None, "cpl_over_rate": None, "median_cps": None, "median_duration_s": None}
    cps = cpl = 0
    rates: list[float] = []
    for ln in lines:
        dur = (ln.end_ms - ln.start_ms) / 1000
        if dur > 0:
            rate = cps_chars(ln.text) / dur
            rates.append(rate)
            cps += rate > settings.cps_max
        if any(len(part) > settings.cpl_limit for part in ln.text.split("\n")):
            cpl += 1
    return {
        "cps_over_rate": round(cps / len(lines), 4),
        "cpl_over_rate": round(cpl / len(lines), 4),
        "median_cps": _median(rates),
        "median_duration_s": _median([(ln.end_ms - ln.start_ms) / 1000 for ln in lines]),
    }


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def evaluate_ost_and_sync(
    ost_events: Sequence[Any],
    dlg_events: Sequence[Any],
    *,
    gold_ost: Sequence[Mapping[str, Any]] | None = None,
    sync_delta_ms: float | None = None,
    ref_sync_ms: float = -248.0,
) -> dict[str, Any]:
    """Score OST coverage, quotation and chain positions on a 0..100 scale.

    Coverage is the share of gold OST spans overlapping a displayed OST event;
    it is unassessed (None) without gold. Emphasis is the share of OST events
    with every display line quoted. Position checks entire <=400 ms-gap chains:
    any positive dialogue overlap requires every chain event at the top.
    Empty assessed sets score 100. Sync error is absolute milliseconds, or None
    when no measured offset is supplied.
    """
    def value(event: Any, name: str, default: Any = None) -> Any:
        return event.get(name, default) if isinstance(event, Mapping) else getattr(event, name, default)

    def span(event: Any) -> tuple[int, int]:
        return (
            int(value(event, "start_ms", value(event, "start", 0))),
            int(value(event, "end_ms", value(event, "end", 0))),
        )

    def quoted(event: Any) -> bool:
        text = re.sub(r"\{[^}]*\}", "", str(value(event, "ko", value(event, "text", "")))).replace(r"\N", "\n").strip()
        lines = text.splitlines()
        pairs = {"'": "'", '"': '"', "‘": "’", "“": "”", "[": "]"}
        if len(text) >= 2 and pairs.get(text[0]) == text[-1]:
            return True
        return bool(lines) and all(
            len(line.strip()) >= 2 and pairs.get(line.strip()[0]) == line.strip()[-1]
            for line in lines
        )
    def at_top(event: Any) -> bool:
        position = value(event, "position", value(event, "pos"))
        if position is not None:
            return position in {"an8", "top", 8}
        text = str(value(event, "ko", value(event, "text", "")))
        return bool(re.search(r"\{[^}]*\\an8\b[^}]*\}", text))

    ordered = sorted(((span(event), event) for event in ost_events), key=lambda item: item[0])
    dialogue = [span(event) for event in dlg_events]
    position_ok = 0
    index = 0
    while index < len(ordered):
        chain_end = ordered[index][0][1]
        following = index + 1
        while following < len(ordered) and ordered[following][0][0] - chain_end <= 400:
            chain_end = max(chain_end, ordered[following][0][1])
            following += 1
        chain_start = ordered[index][0][0]
        needs_top = any(min(chain_end, end) > max(chain_start, start) for start, end in dialogue)
        position_ok += sum(at_top(event) == needs_top for _, event in ordered[index:following])
        index = following

    coverage = None
    if gold_ost is not None:
        covered = sum(
            any(min(gold_end, end) > max(gold_start, start) for (start, end), _ in ordered)
            for gold_start, gold_end in map(span, gold_ost)
        )
        coverage = round(100 * covered / len(gold_ost), 2) if gold_ost else 100.0
    count = len(ordered)
    return {
        "ost_cov": coverage,
        "ost_emp": round(100 * sum(quoted(event) for _, event in ordered) / count, 2) if count else 100.0,
        "ost_pos": round(100 * position_ok / count, 2) if count else 100.0,
        "sync_err_ms": abs(sync_delta_ms - ref_sync_ms) if sync_delta_ms is not None else None,
    }


def arm_metrics(
    groups: list[Group], arm: str, settings: Settings, *, units: list[list[Group]] | None = None,
) -> dict:
    """Lexical agreement by overlap units; display events always count once.

    Without a partition, existing listwise consumers keep group-wise scoring.
    ``chrf_fixed_group`` preserves temporal-locality evidence that lexical
    units intentionally cannot provide.
    """
    pairs: list[tuple[str, str]] = []
    gold_groups = 0
    cand_chars = gold_chars = 0
    lvl_n = lvl_ok = 0
    dash_n = dash_ok = dash_gold = dash_hit = 0
    for g in groups:
        gold, cand = g.text(OFFICIAL), g.text(arm)
        if not gold and not cand:
            continue
        pairs.append((" ".join(cand.split()), " ".join(gold.split())))
        cand_chars += cps_chars(cand)
        gold_chars += cps_chars(gold)
        if gold:
            gold_groups += 1
        if gold and cand:
            lg, lc = group_level(g.official), group_level(g.ko(arm))
            if lg != OTHER and lc != OTHER:
                lvl_n += 1
                lvl_ok += lg == lc
            dg, dc = has_dash(g.official), has_dash(g.ko(arm))
            dash_n += 1
            dash_ok += dg == dc
            dash_gold += dg
            dash_hit += dg and dc
    lines = [ln for g in groups for ln in g.ko(arm)]
    official_lines = sum(len(g.official) for g in groups)
    lexical_pairs = pairs
    covered = sum(bool(group.text(OFFICIAL)) and bool(group.text(arm)) for group in groups)
    if units is not None:
        lexical_pairs = []
        covered = 0
        for unit in units:
            events = sorted(
                (line for group in unit for line in group.ko(arm)),
                key=lambda line: (line.start_ms, line.end_ms, line.idx),
            )
            cand = " ".join(line.text for line in events)
            gold = " ".join(group.text(OFFICIAL) for group in unit)
            if gold.strip() or cand.strip():
                lexical_pairs.append((" ".join(cand.split()), " ".join(gold.split())))
            if cand.strip():
                covered += sum(bool(group.text(OFFICIAL)) for group in unit)
    return {
        "chrf": round(corpus_chrf(lexical_pairs), 2),
        "chrf_fixed_group": round(corpus_chrf(pairs), 2),
        "coverage": _rate(covered, gold_groups),
        "length_ratio": _rate(cand_chars, gold_chars),
        "speech_level_agreement": _rate(lvl_ok, lvl_n),
        "speech_level_n": lvl_n,
        "dash_agreement": _rate(dash_ok, dash_n),
        "dash_recall": _rate(dash_hit, dash_gold),
        "dash_n": dash_n,
        **display_rates(lines, settings),
        "lines": len(lines),
        "event_ratio": _rate(len(lines), official_lines),
    }



_OFF_SIM_JOSA_EOMI = (
    "습니다", "ㅂ니다", "했어요", "했다", "한다", "해요", "하다", "예요", "이에요", "거예요", "건가요", "잖아", "잖아요", "거든", "거든요",
    "은", "는", "이", "가", "을", "를", "도", "만", "에", "에서", "에게", "한테", "께", "의", "와", "과", "랑", "로", "으로", "고", "서", "요", "죠", "다", "니다", "까", "나", "네", "군", "지",
)


def _norm_off_surface(s: str) -> str:
    s = re.sub(r"\{\\an\d+\}", "", s or "")
    s = re.sub(r"[^\w\s가-힣]", " ", s)
    return " ".join(s.split())


def _extract_off_content_stems(s: str) -> Counter[str]:
    stems: Counter[str] = Counter()
    for tok in _norm_off_surface(s).split():
        stem = tok
        for suf in sorted(_OFF_SIM_JOSA_EOMI, key=len, reverse=True):
            if len(stem) > len(suf) and stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        if stem and (re.search(r"[가-힣0-9]", stem) or len(stem) >= 2):
            stems[stem] += 1
            if len(stem) >= 3 and re.match(r"^[가-힣]+$", stem):
                stems[stem[:2]] += 1
    return stems


def evaluate_official_similarity(
    groups: Sequence[Any],
    arm_key: str = "ko",
    jev_scores: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Compute Official-Subtitle Similarity (`OFF-SIM`), guaranteed 100.0 at identity (`cand == official`)."""
    if not groups:
        return {"chrf": 0.0, "recall": 0.0, "len_sim": 0.0, "jev_sim": 0.0, "off_sim": 0.0}
    chrf_vals, rec_vals, len_vals, jev_vals, off_vals = [], [], [], [], []
    for idx, g in enumerate(groups):
        if isinstance(g, Mapping):
            uid = str(g.get("uid") or g.get("id") or idx)
            gold = str(g.get("official") or g.get("official_ko") or "")
            cand = str(g.get(arm_key) or "")
        else:
            uid = str(getattr(g, "id", idx))
            gold = g.text(OFFICIAL)
            cand = g.text(arm_key)
        s_cand = _norm_off_surface(cand)
        s_gold = _norm_off_surface(gold)
        if s_cand == s_gold:
            c_sc = rec = l_sim = j_sim = o_sim = 100.0
        else:
            c_sc = chrf_score(chrf_stats(s_cand, s_gold))
            o_stems = _extract_off_content_stems(gold)
            c_stems = _extract_off_content_stems(cand)
            if not o_stems and not c_stems:
                rec = 100.0
            elif not o_stems:
                rec = 0.0
            else:
                rec = sum((o_stems & c_stems).values()) / sum(o_stems.values()) * 100.0
            e_c = len(s_cand.split())
            e_o = len(s_gold.split())
            l_sim = (1.0 - abs(e_c - e_o) / max(e_c, e_o, 1)) * 100.0
            j_raw = jev_scores.get(uid, 0.5) if jev_scores is not None else 0.5
            j_sim = j_raw * 100.0
            o_sim = 0.40 * c_sc + 0.30 * rec + 0.10 * l_sim + 0.20 * j_sim
        chrf_vals.append(c_sc)
        rec_vals.append(rec)
        len_vals.append(l_sim)
        jev_vals.append(j_sim)
        off_vals.append(o_sim)
    n = len(off_vals)
    return {
        "chrf": round(sum(chrf_vals) / n, 2),
        "recall": round(sum(rec_vals) / n, 2),
        "len_sim": round(sum(len_vals) / n, 2),
        "jev_sim": round(sum(jev_vals) / n, 2),
        "off_sim": round(sum(off_vals) / n, 2),
    }

def group_record(g: Group, arms: list[str]) -> dict:
    """Export shape of one aligned group (also the ``groups`` list of refeval.json)."""
    rec = {
        "id": g.id,
        "scene_id": g.scene_id,
        "start_ms": g.start_ms,
        "end_ms": g.end_ms,
        "en_cue_ids": [c.id for c in g.en],
        "en": [_en_text(c) for c in g.en],
        "official": [ln.text for ln in g.official],
        "candidates": {arm: [ln.text for ln in g.ko(arm)] for arm in arms},
    }
    gold = " ".join(g.text(OFFICIAL).split())
    rec["chrf"] = {
        arm: round(chrf_score(chrf_stats(" ".join(g.text(arm).split()), gold)), 2)
        for arm in arms
        if gold or g.text(arm)
    }
    rec["level"] = {OFFICIAL: group_level(g.official), **{arm: group_level(g.ko(arm)) for arm in arms}}
    rec["dash"] = {OFFICIAL: has_dash(g.official), **{arm: has_dash(g.ko(arm)) for arm in arms}}
    return rec


# ---------------------------------------------------------------- inputs


def _default_resolve(workspace_id: str) -> Path:
    return workspace(workspace_id).data_dir


@dataclass
class Inputs:
    base: Path
    groups: list[Group]
    official: list[Line]
    context_map: ContextMap | None
    register: RegisterMap | None


def build_groups(
    video_id: str,
    arms: dict[str, str],
    ref_path: Path,
    *,
    resolve: Callable[[str], Path] = _default_resolve,
) -> Inputs:
    """Load EN cues/scenes from the base workspace, the official reference and
    each arm's final .ko.srt; align them into groups with scene ids."""
    base = resolve(video_id)
    en = [c for c in load_models(base / "cues.json", Cue) if c.text.strip()]
    scenes = load_models(base / "scenes.json", Scene) if (base / "scenes.json").is_file() else []
    fm_path = base / "context_map.yaml"
    context_map = load_model(fm_path, ContextMap) if fm_path.is_file() else None
    reg_path = base / "register.yaml"
    register = load_model(reg_path, RegisterMap) if reg_path.is_file() else None
    official = load_ko(Path(ref_path))
    cands = {arm: load_ko(arm_srt(resolve(wid))) for arm, wid in arms.items()}
    groups = align(en, official, cands)
    assign_scenes(groups, scenes)
    return Inputs(base=base, groups=groups, official=official, context_map=context_map, register=register)


def export_groups(
    video_id: str,
    arms: dict[str, str],
    ref_path: Path,
    *,
    resolve: Callable[[str], Path] = _default_resolve,
) -> list[dict]:
    """Aligned groups for downstream bulk classification: id, scene_id,
    start/end ms, en (cue texts), official (line texts), candidates
    {arm: line texts}, plus per-group chrF / level / dash."""
    inp = build_groups(video_id, arms, ref_path, resolve=resolve)
    return [group_record(g, list(arms)) for g in inp.groups]


# ---------------------------------------------------------------- windows


@dataclass
class Window:
    id: str
    groups: list[Group]
    reason: str = "random"

    @property
    def start_ms(self) -> int:
        return self.groups[0].start_ms

    @property
    def end_ms(self) -> int:
        return self.groups[-1].end_ms


def _split_even(items: list, k: int) -> list[list]:
    q, r = divmod(len(items), k)
    out, i = [], 0
    for j in range(k):
        n = q + (j < r)
        out.append(items[i:i + n])
        i += n
    return out


def build_windows(groups: list[Group]) -> list[Window]:
    """Consecutive Korean-bearing groups cut into windows of WINDOW_MIN..WINDOW_MAX
    (target WINDOW_TARGET), preferring scene boundaries; short scenes are
    pooled with the following ones."""
    chunks: list[list[Group]] = []
    buf: list[Group] = []

    def flush() -> None:
        if not buf:
            return
        k = max(1, round(len(buf) / WINDOW_TARGET))
        while len(buf) / k > WINDOW_MAX:
            k += 1
        while k > 1 and len(buf) / k < WINDOW_MIN:
            k -= 1
        chunks.extend(_split_even(list(buf), k))
        buf.clear()

    content = [g for g in groups if g.has_korean]
    for i, g in enumerate(content):
        buf.append(g)
        nxt = content[i + 1] if i + 1 < len(content) else None
        if nxt is None or (nxt.scene_id != g.scene_id and len(buf) >= WINDOW_MIN):
            flush()
    if len(chunks) >= 2 and len(chunks[-1]) < WINDOW_MIN and len(chunks[-2]) + len(chunks[-1]) <= WINDOW_MAX:
        chunks[-2].extend(chunks.pop())
    return [Window(id=f"w{i:03d}", groups=c) for i, c in enumerate(chunks, 1)]


def targeted_cue_ids(context_map: ContextMap | None, register: RegisterMap | None) -> set[str]:
    """Register-shift boundaries (register.yaml) and callback setups/payoffs."""
    ids: set[str] = set()
    if register is not None:
        for edge in register.edges:
            for sh in edge.shifts:
                ids.add(sh.at_cue)
                if sh.until_cue:
                    ids.add(sh.until_cue)
    if context_map is not None:
        for cb in context_map.callbacks:
            ids.update(cb.setup_cues)
            ids.update(cb.payoff_cues)
    return ids


def select_windows(windows: list[Window], n: int, seed: int, targeted: set[str]) -> list[Window]:
    """Seeded, stratified pick: up to TARGETED_SHARE from windows containing a
    targeted cue, the rest uniformly from the remainder; video order."""
    rng = random.Random(seed)
    hot = [w for w in windows if any(c.id in targeted for g in w.groups for c in g.en)]
    want_t = min(round(n * TARGETED_SHARE), len(hot))
    picked_t = rng.sample(hot, want_t)
    taken = {w.id for w in picked_t}
    rest = [w for w in windows if w.id not in taken]
    picked_r = rng.sample(rest, min(n - want_t, len(rest)))
    for w in picked_t:
        w.reason = "targeted"
    for w in picked_r:
        w.reason = "random"
    return sorted(picked_t + picked_r, key=lambda w: w.start_ms)


# ---------------------------------------------------------------- degradation


def _to_hae(token: str) -> str:
    if token.endswith("이에요"):
        return token[:-3] + "이야"
    if token.endswith("예요"):
        return token[:-2] + "야"
    if token.endswith("세요"):
        return token[:-2] + "셔"
    if token.endswith("죠"):
        return token[:-1] + "지"
    if token.endswith("요") and len(token) >= 2:
        return token[:-1]
    return token


_ADD_YO = tuple("아어여해게래네걸데든군까봐줘와워돼써마")


def _to_haeyo(token: str) -> str:
    if token.endswith("이야"):
        return token[:-2] + "이에요"
    if token.endswith("야"):
        return token[:-1] + "예요"
    if token.endswith("지"):
        return token[:-1] + "죠"
    if token.endswith("자"):
        return token[:-1] + "요"
    if token.endswith("다") and len(token) >= 2 and jongseong(token[-2]) == _JONG_SS:
        return token[:-1] + "어요"
    if token.endswith(_ADD_YO):
        return token + "요"
    return token


_SEG_RE = re.compile(r"([.?!…~]+[\"'”’)]*\s*|\n)")


def swap_register(text: str) -> str:
    """Every sentence-final 해요체 ending → 반말 and 반말 → 해요체."""
    parts = _SEG_RE.split(text)
    for i in range(0, len(parts), 2):
        m = _HANGUL_TOKEN_RE.search(parts[i])
        if not m:
            continue
        token = m.group(1)
        level = ending_level(parts[i] + (parts[i + 1] if i + 1 < len(parts) else ""))
        new = _to_hae(token) if level == "haeyo" else _to_haeyo(token) if level == "hae" else token
        parts[i] = parts[i][: m.start(1)] + new + parts[i][m.end(1):]
    return "".join(parts)


def name_swaps(context_map: ContextMap | None) -> dict[str, str]:
    """Each character's Korean name (and given-name part) → the next character's,
    characters ordered by id."""
    if context_map is None:
        return {}
    chars = sorted((c for c in context_map.characters if c.ko_name.strip()), key=lambda c: c.id)
    if len(chars) < 2:
        return {}
    out: dict[str, str] = {}
    for i, c in enumerate(chars):
        nxt = chars[(i + 1) % len(chars)]
        if c.ko_name != nxt.ko_name:
            out[c.ko_name] = nxt.ko_name
        first, nfirst = c.ko_name.split()[0], nxt.ko_name.split()[0]
        if len(first) >= 2 and first != nfirst:
            out.setdefault(first, nfirst)
    return out


def degrade(lines: list[Line], names: dict[str, str]) -> dict[int, str]:
    """Deterministic perturbation of the official track, by line index: every
    6th line (idx % 6 == 5) is deleted, every 3rd (idx % 3 == 0) has its speech
    level swapped, and character names are replaced throughout. Returns
    idx → text ("" = deleted)."""
    pat = re.compile("|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))) if names else None
    out: dict[int, str] = {}
    for ln in lines:
        if ln.idx % 6 == 5:
            out[ln.idx] = ""
            continue
        text = swap_register(ln.text) if ln.idx % 3 == 0 else ln.text
        if pat is not None:
            text = pat.sub(lambda m: names[m.group(0)], text)
        out[ln.idx] = text
    return out


# ---------------------------------------------------------------- judge


class ItemVerdict(BaseModel):
    item: int
    category: str
    severity: int
    note: str = ""


class CandidateVerdict(BaseModel):
    label: str
    score: int  # 0..100 closeness to gold for the whole window
    items: list[ItemVerdict] = Field(default_factory=list)


class WindowJudgment(BaseModel):
    candidates: list[CandidateVerdict] = Field(default_factory=list)


JUDGE_PROMPT = """You grade Korean video subtitle candidates against the video's OFFICIAL Korean subtitles, which are the gold answer.
The gold was written by professional subtitlers. Its wording, condensation, dropped lines, paraphrases, speech levels (존댓말/반말), names and terms, and dual-speaker '-' lines are correct by definition. A candidate is good exactly to the extent that it conveys what the gold conveys, the way the gold does. Never reward a candidate for being more literal or more complete than the gold, and never penalize it for condensing where the gold condenses. Different wording with the same meaning, tone and speech level as the gold is fine.
The English lines are context only.

The scene is split into numbered time-aligned items. Each item shows EN (English), GOLD (official Korean) and one line per candidate label. "(none)" means no subtitle there; " / " is a line break inside one subtitle.

For each candidate label and each item give:
- category: the main difference from the gold, one of
  meaning (different or wrong meaning, lost joke/reference/subtext),
  register (different speech level or honorifics than the gold),
  name_term (different name, title, nickname or term than the gold),
  verbosity (noticeably longer or more literal than the gold, or a subtitle where the gold has none),
  omission (gold content missing or cut),
  unnatural (awkward or translationese Korean the gold avoids),
  dash_format (dual-speaker '-' lines differ from the gold),
  none (equivalent to the gold)
- severity: 0 none, 1 minor (barely noticeable), 2 major (clearly different), 3 critical (wrong or misleading)
- note: at most 15 words
and one score 0-100 for the whole scene: closeness to the gold (100 = equivalent to the gold, 50 = same gist with many differences, 0 = unrelated).
Return exactly one entry per candidate label ({labels}), each with exactly one verdict per item 1..{n}.

{items}"""


def _ko_inline(lines: list[str]) -> str:
    return "  //  ".join(t.replace("\n", " / ") for t in lines if t) or "(none)"


def judge_prompt(window: Window, texts: dict[str, dict[str, list[str]]], key: dict[str, str]) -> str:
    """``texts``: candidate name → group id → line texts; ``key``: label → name."""
    blocks: list[str] = []
    for i, g in enumerate(window.groups, 1):
        rows = [f"[{i}] {mmss(g.start_ms)}"]
        rows.append("EN: " + (" ".join(_en_text(c).replace("\n", " ") for c in g.en) or "(none)"))
        rows.append("GOLD: " + _ko_inline([ln.text for ln in g.official]))
        for label, name in key.items():
            rows.append(f"{label}: " + _ko_inline(texts[name].get(g.id, [])))
        blocks.append("\n".join(rows))
    return JUDGE_PROMPT.format(labels=", ".join(key), n=len(window.groups), items="\n\n".join(blocks))


_CATEGORY_ALIASES = {
    "name": "name_term", "term": "name_term", "names": "name_term", "name/term": "name_term",
    "dash": "dash_format", "format": "dash_format", "speech_level": "register", "honorific": "register",
    "naturalness": "unnatural", "verbose": "verbosity", "omitted": "omission", "ok": "none",
}


def _norm_category(raw: str) -> str:
    c = re.sub(r"[\s-]+", "_", raw.strip().lower())
    c = _CATEGORY_ALIASES.get(c, c)
    return c if c in CATEGORIES else "unnatural"


def _verdict_record(v: CandidateVerdict, window: Window) -> dict:
    lines: list[dict] = []
    for it in v.items:
        if not 1 <= it.item <= len(window.groups):
            continue
        sev = max(0, min(3, int(it.severity)))
        cat = _norm_category(it.category)
        if cat == "none":
            sev = 0
        elif sev == 0:
            cat = "none"
        lines.append({"group_id": window.groups[it.item - 1].id, "category": cat, "severity": sev, "note": it.note})
    return {"score": max(0, min(100, int(v.score))), "lines": lines}


_JUDGE_ATTEMPTS = 3
JUDGE_WORKERS = 4  # concurrent window calls per judge model
MEAN = "mean"

# (ws, settings, model id) -> client with GeminiClient's generate() contract
JudgeFactory = Callable[[Workspace, Settings, str], Any]


def parse_judge_models(specs: list[str] | None, default: str) -> list[str]:
    """``--judge-model`` values (repeatable, each a comma list) -> unique model
    ids in order; none given -> ``[default]``."""
    out: list[str] = []
    for spec in specs or ():
        for m in spec.split(","):
            m = m.strip()
            if m and m not in out:
                out.append(m)
    return out or [default]


def _default_judge_factory(ws: Workspace, settings: Settings, model: str) -> Any:
    from .llm import make_client

    return make_client(ws, settings, model)


def judge_window(
    ws: Workspace,
    client: Any,
    model: str,
    window: Window,
    texts: dict[str, dict[str, list[str]]],
    *,
    seed: int,
    kind: str,
) -> dict[str, dict]:
    """One ``model`` call judging every candidate in ``texts`` against the gold
    (checkpointed per judge model). Returns candidate name → {score, lines}."""
    names = sorted(texts)
    key = shuffle_labels(names, f"{kind}:{window.id}", seed)
    prompt = judge_prompt(window, texts, key)
    ck = Checkpoints(ws, STAGE)
    ck_key = f"{kind}-{window.id}-s{seed}-{model}"
    h = input_hash(prompt, model)
    verdict = ck.get(ck_key, h, WindowJudgment)
    if verdict is None:
        for attempt in range(_JUDGE_ATTEMPTS):
            verdict = client.generate(stage=STAGE, prompt=prompt, schema=WindowJudgment, model=model)
            for v in verdict.candidates:
                v.label = re.sub(r"(?i)^candidate\s+", "", v.label.strip()).strip(" :")
            if {v.label for v in verdict.candidates} == set(key):
                break
            if attempt == _JUDGE_ATTEMPTS - 1:
                got = sorted(v.label for v in verdict.candidates)
                raise RuntimeError(f"{STAGE} {model} {window.id}: expected labels {sorted(key)}, got {got}")
        ck.put(ck_key, h, verdict)
    by_label = {v.label: v for v in verdict.candidates}
    return {name: _verdict_record(by_label[label], window) for label, name in key.items()}


def _group_texts(groups: list[Group], track: str) -> dict[str, list[str]]:
    return {g.id: [ln.text for ln in g.ko(track)] for g in groups}


def _judge_each(
    ws: Workspace, client: Any, model: str, windows: list[Window],
    texts_of: Callable[[Window], dict[str, dict[str, list[str]]]], *, seed: int, kind: str,
) -> list[dict]:
    with ThreadPoolExecutor(max_workers=max(1, min(JUDGE_WORKERS, len(windows)))) as ex:
        futs = [ex.submit(judge_window, ws, client, model, w, texts_of(w), seed=seed, kind=kind) for w in windows]
        return [{"window": w, "verdicts": f.result()} for w, f in zip(windows, futs)]


def judge_windows(
    ws: Workspace, client: Any, model: str, windows: list[Window], arms: list[str], *, seed: int
) -> list[dict]:
    """All arms per window in one blind call."""
    return _judge_each(
        ws, client, model, windows, lambda w: {arm: _group_texts(w.groups, arm) for arm in arms},
        seed=seed, kind="arms",
    )


def calibrate(
    ws: Workspace,
    client: Any,
    model: str,
    windows: list[Window],
    official: list[Line],
    names: dict[str, str],
    *,
    seed: int,
) -> tuple[dict, list[dict]]:
    """Judge the official itself and a degraded official on the same windows."""
    degraded = degrade(official, names)
    judged = _judge_each(
        ws, client, model, windows,
        lambda w: {
            OFFICIAL: {g.id: [ln.text for ln in g.official] for g in w.groups},
            DEGRADED: {g.id: [degraded[ln.idx] for ln in g.official if degraded[ln.idx]] for g in w.groups},
        },
        seed=seed, kind="calib",
    )
    summary = {name: _summarize([j["verdicts"][name] for j in judged]) for name in (OFFICIAL, DEGRADED)}
    off, deg = summary[OFFICIAL], summary[DEGRADED]
    checks = {
        "official_score_ok": off["score"] is not None and off["score"] >= CALIB_OFFICIAL_MIN,
        "official_errors_ok": off["error_rate"] is not None and off["error_rate"] <= CALIB_OFFICIAL_MAX_ERROR_RATE,
        "degraded_lower_ok": (
            off["score"] is not None and deg["score"] is not None and deg["score"] <= off["score"] - CALIB_MIN_GAP
        ),
    }
    deleted = sum(1 for t in degraded.values() if not t)
    calib = {
        **summary,
        "checks": checks,
        "passed": all(checks.values()),
        "thresholds": {
            "official_min_score": CALIB_OFFICIAL_MIN,
            "official_max_error_rate": CALIB_OFFICIAL_MAX_ERROR_RATE,
            "min_gap": CALIB_MIN_GAP,
        },
        "perturbation": {
            "lines": len(official),
            "deleted": deleted,
            "register_swapped": sum(1 for ln in official if ln.idx % 3 == 0 and ln.idx % 6 != 5),
            "name_swaps": names,
        },
    }
    return calib, judged


def run_judges(
    ws: Workspace,
    settings: Settings,
    models: list[str],
    windows: list[Window],
    arms: list[str],
    official: list[Line],
    names: dict[str, str],
    *,
    seed: int,
    factory: JudgeFactory | None = None,
) -> dict[str, dict]:
    """Arms verdicts + calibration for every judge model (judges run
    concurrently). Returns model -> {"judged", "calibration", "calib_judged"}."""
    factory = factory or _default_judge_factory

    def one(model: str) -> dict:
        client = factory(ws, settings, model)
        calib, calib_judged = calibrate(ws, client, model, windows, official, names, seed=seed)
        judged = judge_windows(ws, client, model, windows, arms, seed=seed)
        return {"judged": judged, "calibration": calib, "calib_judged": calib_judged}

    with ThreadPoolExecutor(max_workers=max(1, len(models))) as ex:
        futs = {m: ex.submit(one, m) for m in models}
        return {m: f.result() for m, f in futs.items()}


def _summarize(verdicts: list[dict]) -> dict:
    scores = [v["score"] for v in verdicts]
    items = [ln for v in verdicts for ln in v["lines"]]
    errors = Counter(ln["category"] for ln in items if ln["severity"] > 0)
    return {
        "score": round(sum(scores) / len(scores), 2) if scores else None,
        "windows": len(scores),
        "items": len(items),
        "error_rate": _rate(sum(errors.values()), len(items)),
        "severity_sum": sum(ln["severity"] for ln in items),
        "errors": {c: errors.get(c, 0) for c in ERROR_CATEGORIES},
    }


def _avg(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def mean_summary(summaries: list[dict]) -> dict:
    """Per-field mean of several judges' ``_summarize`` results (one judge:
    that judge's summary unchanged)."""
    if len(summaries) == 1:
        return summaries[0]
    return {
        "score": _avg([s["score"] for s in summaries]),
        "windows": summaries[0]["windows"],
        "items": _avg([s["items"] for s in summaries]),
        "error_rate": _avg([s["error_rate"] for s in summaries]),
        "severity_sum": _avg([s["severity_sum"] for s in summaries]),
        "errors": {c: _avg([s["errors"][c] for s in summaries]) for c in ERROR_CATEGORIES},
    }


def judge_summaries(results: dict[str, dict], arms: list[str]) -> dict[str, dict]:
    """arm -> {"judge": mean over judges, "judges": model -> summary}."""
    out: dict[str, dict] = {}
    for arm in arms:
        per = {m: _summarize([j["verdicts"][arm] for j in r["judged"]]) for m, r in results.items()}
        out[arm] = {"judge": mean_summary(list(per.values())), "judges": per}
    return out


def merge_verdicts(per_model: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """model -> candidate -> {score, lines} as candidate -> {score: mean over
    judges, scores: model -> score, lines: every judge's lines tagged with
    ``judge``} (the review app reads ``score`` and ``lines``)."""
    names = next(iter(per_model.values())).keys()
    return {
        name: {
            "score": _avg([v[name]["score"] for v in per_model.values()]),
            "scores": {m: v[name]["score"] for m, v in per_model.items()},
            "lines": [ln | {"judge": m} for m, v in per_model.items() for ln in v[name]["lines"]],
        }
        for name in names
    }


# ---------------------------------------------------------------- independent paired evaluation


class SourceItemVerdict(BaseModel):
    """Source meaning defects, including assessable omissions, distinct from gold agreement."""

    item: int
    severity: int = Field(ge=0, le=3)
    safe_harbor: bool = False
    note: str = ""


class PresentationItemVerdict(BaseModel):
    """Actual display/timing defects, separate from source meaning."""

    item: int
    timing_severity: int = Field(ge=0, le=3)
    display_severity: int = Field(ge=0, le=3)
    duplicate_severity: int = Field(ge=0, le=3)
    note: str = ""


class PointJudgment(BaseModel):
    score: int = Field(ge=0, le=100)
    items: list[ItemVerdict]
    source_meaning_score: int = Field(ge=0, le=100)
    source_items: list[SourceItemVerdict]
    presentation_items: list[PresentationItemVerdict]


_PRESENTATION_METRICS = (
    "timing_error_rate", "display_error_rate", "duplicate_error_rate",
    "presentation_error_rate", "presentation_severity_sum",
)


POINT_PROMPT = """Grade ONE Korean subtitle candidate independently. Never rank candidates.
OFFICIAL Korean subtitles are the professional GOLD for the requested score, including their
paraphrase, condensation, omissions, register, terms and speaker dashes. Equivalent meaning,
tone and register in different words is fully correct. More literal is not automatically better.
The official identity is the 100-point reference. Valid, readable merges/re-splits and
meaning-preserving condensation/paraphrase deserve the same highest GOLD agreement.
The primary score assesses usable subtitle agreement too: reduce it for genuine mistiming,
unreadable layout, or repeated display events, even when all words are lexically present.
source_meaning_score concerns only time-aligned candidate content for the scored items,
not a video-wide transcript bag. A remote subtitle is absent at this scene, and remote
extras have no established source correspondence. Do not assume their source meaning
is correct. Distinguish missing time-aligned content from a readable local re-segmentation.
Return score 0..100 for agreement with GOLD and exactly one items verdict per item 1..{n}:
item, category (meaning/register/name_term/verbosity/omission/unnatural/dash_format/none),
severity 0..3 (0 none, 1 minor, 2 major, 3 critical), note (at most 15 words).

SEPARATELY return source_meaning_score 0..100 for safety against EN, not lexical agreement
with GOLD, and exactly one source_items verdict per item 1..{n}: item, severity 0..3,
safe_harbor, note. safe_harbor=true means a meaning-preserving paraphrase or safe condensation
that needs no correction solely for differing wording. Do not reinterpret GOLD as wrong in
the requested score. When EN is absent, do not invent source meaning or flag unsupported
source errors; use GOLD as context and state the limitation in the source note.
Source errors include contradictions, unsupported additions, and lost essential meaning.
When EN and GOLD carry essential meaning, all relevant local event bodies are visible,
and no time-aligned candidate event conveys it, mark an assessable source omission with
nonzero severity and safe_harbor=false. An observed missing event is an empty hypothesis,
not missing evidence. First check shared local events, readable merges/re-splits, and
meaning-preserving paraphrase or safe condensation; none is an omission merely because
words or item boundaries differ. Allowed professional GOLD condensation is not an error
solely because EN contains more words.
An evaluator-omitted event body (text_omitted=true) or absent EN is instead an evidence
limitation: state what is unknown in the source note and do not invent an error from
unseen words or grant safe_harbor without supporting evidence. A remote extra summary
does not reveal its meaning and does not fill an observed absence in this scored scene.
SEPARATELY return exactly one presentation_items verdict per item 1..{n}: item,
timing_severity, display_severity, duplicate_severity (each 0..3), note.
Use actual event timing, display text and source/gold intervals to assess these independently.
A legitimate merge/re-split is not a presentation defect. An oversized unreadable overlay
can preserve meaning but have severe presentation defects and a lower primary gold score.
Do not count a shared event reference as duplication; real separate repeated events are
duplication. Item severity/error rates must also reflect genuine presentation defects:
use verbosity for unreadable/excess/repeated display and meaning for mistimed dialogue,
while presentation_items retain the exact reason. Never manufacture a content omission.
Only the single CANDIDATE shown below is being graded. No model or arm identity is given.
Display events can merge or re-split adjacent fixed items. CANDIDATE references identify
real events, not independent copies for each item. Distinct ordinals remain separate
events even when timing/text are identical. UNIT_CONTEXT contains only this fixed window
and its immediate source/gold-order boundary neighbors, which are read-only context.
Return verdicts only for the numbered fixed items. Do not require word-for-word or
item-boundary alignment; a readable local merge/condensation can preserve the highest
GOLD agreement. Event bodies whose complete source/gold span lies outside the local
context are explicitly omitted, never sliced into invented local dialogue. Their span,
duration and display measurements remain visible, but content coverage is uncertain.
Do not claim complete semantic/source safety from omitted bodies, lexical metrics, or
remote extra summaries. A local event spanning multiple items is not an omission.
Nonvisible extras supply count/timing summaries only, not dialogue to credit here.

{items}"""


def _point_unit_context(window: Window, units: list[list[Group]], track: str) -> list[dict]:
    """Bound judge evidence by fixed locality, never by a track's connected span."""
    fixed = sorted(
        (group for unit in units for group in unit if group.en or group.official),
        key=lambda group: (group.start_ms, group.end_ms, group.id),
    )
    scored_ids = {group.id for group in window.groups}
    positions = [index for index, group in enumerate(fixed) if group.id in scored_ids]
    local = fixed[max(0, min(positions) - 1):max(positions) + 2] if positions else []
    local_ids = {group.id for group in local}
    actual = sorted(
        ((line, group, unit) for unit in units for group in unit for line in group.ko(track)),
        key=lambda event: (event[0].start_ms, event[0].end_ms, event[0].idx),
    )
    events, outside_extras = [], []
    omitted_scored_bodies = 0
    omitted_remote_unit_bodies = 0
    remote_unit_ids: set[str] = set()
    for unit in units:
        if any(group.id in scored_ids for group in unit):
            remote_unit_ids.update(group.id for group in unit if group.id not in local_ids and (group.en or group.official))
    for ordinal, (line, owner, unit) in enumerate(actual, 1):
        span = [
            group.id for group in unit if (group.en or group.official)
            and any(overlaps(line.start_ms, line.end_ms, item.start_ms, item.end_ms)
                    for item in [*group.en, *group.official])
        ]
        visible = min(line.end_ms, window.end_ms) > max(line.start_ms, window.start_ms)
        if not span and not visible:
            outside_extras.append(line)
            continue
        if span and not local_ids.intersection(span):
            if any(group.id in scored_ids for group in unit):
                omitted_remote_unit_bodies += 1
            continue
        complete_body = not span or all(group_id in local_ids for group_id in span)
        if not complete_body and scored_ids.intersection(span):
            omitted_scored_bodies += 1
        duration = (line.end_ms - line.start_ms) / 1000
        record = {
            "event": f"e{ordinal:04d}", "start_ms": line.start_ms, "end_ms": line.end_ms,
            "owner_group_id": owner.id, "span_group_ids": [group_id for group_id in span if group_id in local_ids],
            "span_group_count": len(span), "span_first_group_id": span[0] if span else None,
            "span_last_group_id": span[-1] if span else None,
            "text_omitted": not complete_body, "text_chars": cps_chars(line.text),
            "max_line_chars": max((len(part) for part in line.text.split("\n")), default=0),
            "display_lines": len(line.text.split("\n")), "duration_s": duration,
            "cps": cps_chars(line.text) / duration if duration > 0 else None,
        }
        if complete_body:
            record["text"] = line.text
        else:
            record["text_hash"] = input_hash(line.text)
        events.append(record)
    return [{
        "group_ids": [group.id for group in local],
        "source_gold": [
            {
                "group_id": group.id, "start_ms": group.start_ms, "end_ms": group.end_ms,
                "en": [{"start_ms": cue.start_ms, "end_ms": cue.end_ms, "text": _en_text(cue)}
                       for cue in group.en],
                "gold": [{"start_ms": line.start_ms, "end_ms": line.end_ms, "text": line.text}
                         for line in group.official],
            }
            for group in local
        ],
        "events": events,
        "scope": {
            "scored_group_ids": [group.id for group in window.groups],
            "readonly_group_ids": [group.id for group in local if group.id not in scored_ids],
            "remote_unit_group_count": len(remote_unit_ids),
            "omitted_remote_unit_event_bodies": omitted_remote_unit_bodies,
            "omitted_scored_event_bodies": omitted_scored_bodies,
            "scored_event_text_complete": omitted_scored_bodies == 0,
        },
        "unmatched_outside_window": {
            "events": len(outside_extras),
            "start_ms": min((line.start_ms for line in outside_extras), default=None),
            "end_ms": max((line.end_ms for line in outside_extras), default=None),
        },
    }]


def point_prompt(window: Window, contexts: list[dict]) -> str:
    """Render the mandatory bounded event context for fixed scored items."""
    blocks: list[str] = []
    for i, g in enumerate(window.groups, 1):
        references = [
            event["event"] for context in contexts for event in context["events"]
            if g.id in event["span_group_ids"]
        ]
        candidate = ", ".join(references) or "(none)"
        blocks.append("\n".join([
            f"[{i}] {mmss(g.start_ms)} group_id={g.id} end_ms={g.end_ms}",
            "EN: " + (" ".join(_en_text(c).replace("\n", " ") for c in g.en) or "(none)"),
            "GOLD: " + _ko_inline([ln.text for ln in g.official]),
            "CANDIDATE: " + candidate,
        ]))
    blocks.append("UNIT_CONTEXT: " + json.dumps(contexts, ensure_ascii=False, separators=(",", ":")))
    return POINT_PROMPT.format(n=len(window.groups), items="\n\n".join(blocks))


def judge_point(
    ws: Workspace, client: Any, model: str, window: Window,
    units: list[list[Group]], *, candidate_hash: str, track: str = PAIRED_CANDIDATE,
) -> dict:
    """One candidate per call, with a content-specific checkpoint namespace.

    Hashes include the rubric, gold, source, timing, candidate and judge. Other
    candidates, their order, and the cycle label cannot affect this checkpoint.
    """
    contexts = _point_unit_context(window, units, track)
    prompt = point_prompt(window, contexts)
    context = json.dumps([
        {
            "id": g.id, "start_ms": g.start_ms, "end_ms": g.end_ms,
            "en": [c.model_dump(mode="json") for c in g.en],
            "official": [[ln.start_ms, ln.end_ms, ln.text] for ln in g.official],
        }
        for g in window.groups
    ], ensure_ascii=False, sort_keys=True)
    h = input_hash(
        "pointwise-local-context-v4", prompt, context, model, json.dumps(PointJudgment.model_json_schema(), sort_keys=True),
    )
    key = input_hash(candidate_hash, h)
    ck = Checkpoints(ws, PAIRED_STAGE)
    verdict = ck.get(key, h, PointJudgment)
    expected = set(range(1, len(window.groups) + 1))

    def complete(v: PointJudgment) -> bool:
        return (
            len(v.items) == len(expected) and {it.item for it in v.items} == expected
            and len(v.source_items) == len(expected) and {it.item for it in v.source_items} == expected
            and len(v.presentation_items) == len(expected)
            and {it.item for it in v.presentation_items} == expected
        )

    if verdict is None or not complete(verdict):
        for attempt in range(_JUDGE_ATTEMPTS):
            verdict = client.generate(stage=PAIRED_STAGE, prompt=prompt, schema=PointJudgment, model=model)
            if complete(verdict):
                break
            if attempt == _JUDGE_ATTEMPTS - 1:
                raise RuntimeError(f"{PAIRED_STAGE} {model} {window.id}: incomplete or duplicate item verdicts")
        ck.put(key, h, verdict)
    record = _verdict_record(
        CandidateVerdict(label=PAIRED_CANDIDATE, score=verdict.score, items=sorted(verdict.items, key=lambda x: x.item)),
        window,
    )
    record["source_meaning_score"] = verdict.source_meaning_score
    record["source_lines"] = [
        {
            "group_id": window.groups[it.item - 1].id,
            "severity": it.severity,
            "safe_harbor": it.safe_harbor,
            "note": it.note,
        }
        for it in sorted(verdict.source_items, key=lambda x: x.item)
    ]
    record["presentation_lines"] = [
        {
            "group_id": window.groups[it.item - 1].id,
            "timing_severity": it.timing_severity,
            "display_severity": it.display_severity,
            "duplicate_severity": it.duplicate_severity,
            "note": it.note,
        }
        for it in sorted(verdict.presentation_items, key=lambda x: x.item)
    ]
    record["context_scope"] = contexts[0]["scope"]
    record["unmatched_outside_window"] = contexts[0]["unmatched_outside_window"]
    record["checkpoint_hash"] = key
    return record


def point_summary(verdicts: list[dict]) -> dict:
    source = [ln for v in verdicts for ln in v["source_lines"]]
    presentation = [line for verdict in verdicts for line in verdict["presentation_lines"]]
    return {
        **_summarize(verdicts),
        "source_meaning_score": _avg([v["source_meaning_score"] for v in verdicts]),
        "source_error_rate": _rate(sum(ln["severity"] > 0 for ln in source), len(source)),
        "safe_harbor_rate": _rate(sum(ln["safe_harbor"] for ln in source), len(source)),
        "context_partial_windows": sum(not verdict["context_scope"]["scored_event_text_complete"] for verdict in verdicts),
        **{
            f"{kind}_error_rate": _rate(
                sum(line[f"{kind}_severity"] > 0 for line in presentation), len(presentation),
            )
            for kind in ("timing", "display", "duplicate")
        },
        "presentation_error_rate": _rate(
            sum(any(line[f"{kind}_severity"] > 0 for kind in ("timing", "display", "duplicate"))
                for line in presentation),
            len(presentation),
        ),
        "presentation_severity_sum": sum(
            line[f"{kind}_severity"]
            for line in presentation for kind in ("timing", "display", "duplicate")
        ),
    }


def subtitle_hash(lines: list[Line]) -> str:
    """Freeze displayed cue-level text/timing, never scene/list-wise judge state."""
    return input_hash(json.dumps(
        [[ln.start_ms, ln.end_ms, ln.text] for ln in lines], ensure_ascii=False, separators=(",", ":"),
    ))


def calibrate_pointwise(ws: Workspace, client: Any, model: str) -> dict:
    """Fixed anchors independent of video, candidate set and sample ordering."""
    gold = "지금 떠나야 해."
    g = Group(
        id="anchor", start_ms=0, end_ms=3000,
        en=[Cue(id="anchor", start_ms=0, end_ms=3000, text="We need to leave now.", raw="We need to leave now.")],
        official=[Line(0, 0, 3000, gold)],
    )
    anchors: dict[str, Any] = {OFFICIAL: gold, DEGRADED: "여기 영원히 있어야 해.", PARAPHRASE: "이제 가야 해."}

    def judge_anchor(groups: list[Group], lines: list[Line]) -> dict:
        attached = _clone_groups(groups)
        extras, units = _attach_track(attached, PAIRED_CANDIDATE, lines)
        return judge_point(
            ws, client, model, Window(id="calibration-local-context-v4", groups=attached, reason="calibration"),
            units + [[extra] for extra in extras], candidate_hash=subtitle_hash(lines), track=PAIRED_CANDIDATE,
        )

    verdicts = {
        name: judge_anchor([g], [Line(0, 0, 3000, text)])
        for name, text in anchors.items()
    }
    merged_gold = [Line(0, 1000, 3000, "네가"), Line(1, 3000, 5000, "떠났으면 좋겠어")]
    merged_en = [
        Cue(id="anchor-c1", start_ms=1000, end_ms=3000, text="I want you", raw="I want you"),
        Cue(id="anchor-c2", start_ms=3000, end_ms=5000, text="to leave.", raw="to leave."),
    ]
    merged_groups = align(merged_en, merged_gold, {})
    merged_text = "네가 떠났으면 좋겠어"
    alignment_anchors = {
        "valid_merge": [Line(0, 1000, 5000, merged_text)],
        "boundary_resplit": [Line(0, 1000, 4000, "네가 떠났으면"), Line(1, 4000, 5000, "좋겠어")],
        "true_omission": [merged_gold[0]],
        "real_duplicate": [Line(0, 1000, 5000, merged_text), Line(0, 1000, 5000, merged_text)],
        "timing_error": [Line(0, 601000, 605000, merged_text)],
    }
    for name, lines in alignment_anchors.items():
        anchors[name] = [[line.start_ms, line.end_ms, line.text] for line in lines]
        verdicts[name] = judge_anchor(merged_groups, lines)
    summary = {name: point_summary([v]) for name, v in verdicts.items()}
    off, deg, para = (summary[name] for name in (OFFICIAL, DEGRADED, PARAPHRASE))
    checks = {
        "official_score_ok": off["score"] == 100,
        "official_presentation_ok": off["presentation_error_rate"] == 0,
        "official_errors_ok": off["error_rate"] <= CALIB_OFFICIAL_MAX_ERROR_RATE,
        "degraded_lower_ok": deg["score"] <= off["score"] - CALIB_MIN_GAP,
        "paraphrase_gold_ok": para["score"] >= CALIB_OFFICIAL_MIN,
        "paraphrase_presentation_ok": para["presentation_error_rate"] == 0,
        "paraphrase_source_safe_ok": para["source_meaning_score"] >= CALIB_OFFICIAL_MIN
        and para["source_error_rate"] == 0 and para["safe_harbor_rate"] == 1,
    }
    for name in ("valid_merge", "boundary_resplit"):
        anchor = summary[name]
        checks[f"{name}_gold_ok"] = anchor["score"] >= CALIB_OFFICIAL_MIN
        checks[f"{name}_errors_ok"] = anchor["error_rate"] <= CALIB_OFFICIAL_MAX_ERROR_RATE
        checks[f"{name}_presentation_ok"] = anchor["presentation_error_rate"] == 0
        checks[f"{name}_source_safe_ok"] = (
            anchor["source_meaning_score"] >= CALIB_OFFICIAL_MIN and anchor["source_error_rate"] == 0
        )
    for name in ("true_omission", "real_duplicate", "timing_error"):
        checks[f"{name}_lower_ok"] = summary[name]["score"] <= off["score"] - CALIB_MIN_GAP
        checks[f"{name}_errors_detected"] = summary[name]["error_rate"] > 0
    checks["true_omission_detected"] = summary["true_omission"]["errors"]["omission"] > 0
    checks["true_omission_source_error"] = summary["true_omission"]["source_error_rate"] > 0
    checks["real_duplicate_detected"] = summary["real_duplicate"]["duplicate_error_rate"] > 0
    checks["timing_error_detected"] = summary["timing_error"]["timing_error_rate"] > 0
    checks["official_highest_ok"] = all(off["score"] >= anchor["score"] for anchor in summary.values())
    return {
        **summary, "checks": checks, "passed": all(checks.values()),
        "anchor_version": "pointwise-local-context-v4", "anchors": anchors, "verdicts": verdicts,
        "thresholds": {
            "official_identity_score": 100, "official_min_score": CALIB_OFFICIAL_MIN,
            "min_gap": CALIB_MIN_GAP,
        },
    }


def evaluate_paired(
    ws: Workspace, settings: Settings, en: list[Cue], official: list[Line],
    candidates: dict[str, list[Line]], opus: list[Line], *,
    scenes: list[Scene] | None = None, n_windows: int | None = None, seed: int = 1,
    judge: bool = False, judge_models: list[str] | None = None,
    judge_factory: JudgeFactory | None = None, targeted: set[str] | None = None,
    split: str = "dev", cycle_id: str | None = None,
) -> dict:
    """Offline-capable paired comparisons on one frozen EN+official alignment.

    Credible overlaps form track-local lexical units for chrF/coverage while
    every actual event remains attached to one fixed group. Judge evidence is
    bounded by the fixed window and its direct source/gold-order neighbors;
    out-of-context shared-event bodies are explicitly omitted, not relocated.
    Extras remain source/gold-free. Neither changes IDs or the pinned sample.
    """
    metadata = evaluation_metadata("paired", split=split, cycle_id=cycle_id)
    if n_windows is not None and n_windows < 0:
        raise ValueError("n_windows must be nonnegative")
    if not candidates:
        raise ValueError("at least one candidate is required")

    fixed_groups = align(en, official, {})
    assign_scenes(fixed_groups, scenes or [])
    fixed_windows = build_windows(fixed_groups)
    if n_windows is not None:
        fixed_windows = select_windows(fixed_windows, n_windows, seed, targeted or set())
    if fixed_windows and not opus:
        raise ValueError("frozen Opus must not be empty for a nonempty evaluation")

    pinned = {
        "group_ids": [group.id for group in fixed_groups],
        "window_ids": [window.id for window in fixed_windows],
        "scene_ids": list(dict.fromkeys(
            group.scene_id for window in fixed_windows for group in window.groups if group.scene_id is not None
        )),
        "windows": [
            {
                "id": window.id,
                "group_ids": [group.id for group in window.groups],
                "scene_ids": list(dict.fromkeys(
                    group.scene_id for group in window.groups if group.scene_id is not None
                )),
                "start_ms": window.start_ms,
                "end_ms": window.end_ms,
            }
            for window in fixed_windows
        ],
    }

    models = list(dict.fromkeys(judge_models or [settings.model_main]))
    opus_hash = subtitle_hash(opus)
    factory = judge_factory or _default_judge_factory
    clients = {m: factory(ws, settings, m) for m in models} if judge and n_windows != 0 else {}
    calibration = {m: calibrate_pointwise(ws, client, m) for m, client in clients.items()}
    judge_status = (
        "invalid_calibration" if calibration and not all(c["passed"] for c in calibration.values())
        else "valid" if calibration else "not_run"
    )
    metrics, paired = {}, {}
    for name in sorted(candidates):
        lines = candidates[name]
        candidate_hash = subtitle_hash(lines)
        groups = _clone_groups(fixed_groups)
        candidate_extras, candidate_units = _attach_track(groups, PAIRED_CANDIDATE, lines)
        opus_extras, opus_units = _attach_track(groups, PAIRED_OPUS, opus)
        by_id = {group.id: group for group in groups}
        windows = [
            Window(
                id=window.id,
                groups=[by_id[group.id] for group in window.groups],
                reason=window.reason,
            )
            for window in fixed_windows
        ]
        candidate_units += [[group] for group in candidate_extras]
        opus_units += [[group] for group in opus_extras]
        candidate_metrics = arm_metrics(
            groups + candidate_extras, PAIRED_CANDIDATE, settings, units=candidate_units,
        )
        opus_metrics = arm_metrics(groups + opus_extras, PAIRED_OPUS, settings, units=opus_units)
        delta = numeric_deltas(candidate_metrics, opus_metrics)
        rows = [
            {
                "id": window.id, "reason": window.reason,
                "group_ids": [group.id for group in window.groups],
                "scene_ids": list(dict.fromkeys(
                    group.scene_id for group in window.groups if group.scene_id is not None
                )),
                "start_ms": window.start_ms, "end_ms": window.end_ms, "judges": {},
            }
            for window in windows
        ]
        per_model = {}
        for model, client in clients.items():
            verdicts = {PAIRED_CANDIDATE: [], PAIRED_OPUS: []}
            for window, row in zip(windows, rows):
                pair = {
                    track: judge_point(
                        ws, client, model, window,
                        candidate_units if track == PAIRED_CANDIDATE else opus_units,
                        candidate_hash=candidate_hash if track == PAIRED_CANDIDATE else opus_hash, track=track,
                    )
                    for track in (PAIRED_CANDIDATE, PAIRED_OPUS)
                }
                for track, verdict in pair.items():
                    verdicts[track].append(verdict)
                row["judges"][model] = pair
            summaries = {track: point_summary(values) for track, values in verdicts.items()}
            per_model[model] = {**summaries, "delta": numeric_deltas(
                summaries[PAIRED_CANDIDATE], summaries[PAIRED_OPUS],
            )}
        if per_model:
            for track, destination in ((PAIRED_CANDIDATE, candidate_metrics), (PAIRED_OPUS, opus_metrics)):
                destination["judges"] = {model: summary[track] for model, summary in per_model.items()}
            if judge_status == "valid":
                for track, destination in ((PAIRED_CANDIDATE, candidate_metrics), (PAIRED_OPUS, opus_metrics)):
                    destination["judge"] = mean_summary(list(destination["judges"].values()))
                    destination["source_safety"] = {
                        key: _avg([summary[track][key] for summary in per_model.values()])
                        for key in ("source_meaning_score", "source_error_rate", "safe_harbor_rate", "context_partial_windows")
                    }
                    destination["presentation"] = {
                        key: _avg([summary[track][key] for summary in per_model.values()])
                        for key in _PRESENTATION_METRICS
                    }
            else:
                candidate_metrics["judge"] = None
                candidate_metrics["source_safety"] = None
                candidate_metrics["presentation"] = None
                opus_metrics["judge"] = None
                opus_metrics["source_safety"] = None
                opus_metrics["presentation"] = None
        metrics[name] = candidate_metrics
        paired[name] = {
            "candidate_hash": candidate_hash, "opus_hash": opus_hash,
            "opus_metrics": opus_metrics, "deterministic_delta": delta,
            "judge_status": judge_status,
            "judge_delta": {
                key: _avg([summary["delta"][key] for summary in per_model.values()])
                for key in ("score", "error_rate", "severity_sum", "source_meaning_score",
                            "source_error_rate", "safe_harbor_rate", "context_partial_windows", *_PRESENTATION_METRICS)
            } if per_model and judge_status == "valid" else None,
            "judges": per_model, "windows": rows, "pinned": pinned,
            "alignment": {"groups": len(groups), "scope": "fixed_source+official_with_attached_tracks"},
            "groups": [group_record(group, [PAIRED_CANDIDATE, PAIRED_OPUS]) for group in groups],
            "unmatched": {
                track: [group_record(group, [track]) for group in extras]
                for track, extras in ((PAIRED_CANDIDATE, candidate_extras), (PAIRED_OPUS, opus_extras))
            },
            "lexical_units": {
                track: [[group.id for group in unit] for unit in units if len(unit) > 1]
                for track, units in ((PAIRED_CANDIDATE, candidate_units), (PAIRED_OPUS, opus_units))
            },
        }
    return {
        **metadata, "metrics": metrics, "paired": paired, "calibration": calibration or None,
        "judge_status": judge_status, "pinned": pinned,
        "opus_hash": opus_hash, "score_basis": "official_gold",
        "source_safety_basis": "time_aligned_source_meaning_on_scored_windows_with_bounded_readonly_context",
        "judge_context_basis": "fixed_window_plus_direct_source_gold_neighbors_with_explicit_event_body_omissions",
        "delta_basis": "candidate_minus_frozen_opus_on_matching_fixed_groups_and_windows",
    }


def render_paired_md(report: dict) -> list[str]:
    out = [
        "", "## Independent paired evaluation",
        f"Split: `{report.get('split')}`; cycle: `{report.get('cycle_id')}`; "
        f"frozen Opus: `{report['opus_hash']}`.",
        "Official agreement remains the requested score. Source safety is separate. "
        "Main chrF/coverage use track-local lexical overlap units across fixed "
        "window boundaries; chrf_fixed_group retains timing-local agreement. "
        "Unmatched events count against deterministic metrics without covering "
        "gold or redefining the pinned judge sample. Lexical 100 is not a "
        "timing or display-quality verdict. Source safety concerns time-aligned "
        "scored-window content, not video-wide transcript equivalence; wide event "
        "bodies may be explicitly omitted from bounded judge context.",
        "", "| candidate | ΔchrF++ | Δgold score | Δsource meaning | Δsource errors | safe-harbor rate | "
        "Δtiming errors | Δdisplay errors | Δduplicate errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    if report.get("judge_status") == "invalid_calibration":
        out.insert(7, "Pointwise aggregate: **INVALID** — calibration failed; judge deltas are not published.")
    for name, pair in report["paired"].items():
        j = pair.get("judge_delta") or {}
        safety = report["metrics"][name].get("source_safety") or {}
        out.append(
            f"| {name} | {_fmt(pair['deterministic_delta']['chrf'])} | {_fmt(j.get('score'))} | "
            f"{_fmt(j.get('source_meaning_score'))} | {_fmt(j.get('source_error_rate'))} | "
            f"{_fmt(safety.get('safe_harbor_rate'), True)} | {_fmt(j.get('timing_error_rate'))} | "
            f"{_fmt(j.get('display_error_rate'))} | {_fmt(j.get('duplicate_error_rate'))} |"
        )
    return out


def render_judges_md(rows: dict[str, dict[str, dict]], models: list[str]) -> list[str]:
    """Per-judge score tables plus a mean table; ``rows``: row label ->
    model -> summary (``_summarize`` shape). Empty for a single judge."""
    if len(models) < 2:
        return []
    out: list[str] = []
    for model in [*models, MEAN]:
        out += ["", f"### Judge: {model}", "",
                "| candidate | judge | " + " | ".join(ERROR_CATEGORIES) + " | sev sum |",
                "|---" * (3 + len(ERROR_CATEGORIES)) + "|"]
        for label, per in rows.items():
            s = mean_summary([per[m] for m in models]) if model == MEAN else per[model]
            out.append(
                f"| {label} | {_fmt(s['score'])} | "
                + " | ".join(_fmt(s["errors"][c]) for c in ERROR_CATEGORIES)
                + f" | {_fmt(s['severity_sum'])} |"
            )
    return out


def render_calibration_md(calibration: dict[str, dict]) -> list[str]:
    """Calibration section, one table per judge model."""
    out = ["", "## Judge calibration"]
    for model, c in calibration.items():
        out += ["", f"### {model}", "", f"Passed: **{c['passed']}** — checks {c['checks']}", ""]
        out.append("| candidate | score | error rate | sev sum | " + " | ".join(ERROR_CATEGORIES) + " |")
        out.append("|---" * (4 + len(ERROR_CATEGORIES)) + "|")
        for name in (OFFICIAL, DEGRADED, PARAPHRASE) if PARAPHRASE in c else (OFFICIAL, DEGRADED):
            s = c[name]
            out.append(
                f"| {name} | {_fmt(s['score'])} | {_fmt(s['error_rate'], True)} | {s['severity_sum']} | "
                + " | ".join(str(s["errors"][k]) for k in ERROR_CATEGORIES) + " |"
            )
        if PARAPHRASE in c:
            para = c[PARAPHRASE]
            out += [
                "", f"Frozen meaning-preserving paraphrase anchor: `{c['anchors'][PARAPHRASE]}`. "
                f"Source meaning: {_fmt(para['source_meaning_score'])}; "
                f"safe-harbor rate: {_fmt(para['safe_harbor_rate'], True)}.",
            ]
        if "presentation_error_rate" in c[OFFICIAL]:
            out += [
                "", "| anchor | gold score | source meaning | source errors | timing errors | "
                "display errors | duplicate errors | presentation severity |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for name in c["verdicts"]:
                summary = c[name]
                out.append(
                    f"| {name} | {_fmt(summary['score'])} | {_fmt(summary['source_meaning_score'])} | "
                    f"{_fmt(summary['source_error_rate'], True)} | {_fmt(summary['timing_error_rate'], True)} | "
                    f"{_fmt(summary['display_error_rate'], True)} | {_fmt(summary['duplicate_error_rate'], True)} | "
                    f"{_fmt(summary['presentation_severity_sum'])} |"
                )
    if "perturbation" not in next(iter(calibration.values())):
        return out
    p = next(iter(calibration.values()))["perturbation"]
    out += [
        "",
        f"Degraded official: {p['deleted']} of {p['lines']} lines deleted, "
        f"{p['register_swapped']} lines with 해요체↔반말 swapped, {len(p['name_swaps'])} name swaps.",
    ]
    return out



# ---------------------------------------------------------------- decisions pre-benchmark


DECISIONS_AXES = ("meaning", "spoken", "condense", "register", "names", "dash")
_DECISIONS_WEIGHTS = {
    "meaning": 0.30,
    "spoken": 0.25,
    "condense": 0.15,
    "register": 0.15,
    "names": 0.10,
    "dash": 0.05,
}


class _DecisionsCache(BaseModel):
    answers: dict[str, dict[str, Any]] = Field(default_factory=dict)


def evaluate_decisions_prebenchmark(
    ws: Workspace,
    settings: Settings,
    windows: list[Window],
    arms: list[str],
    *,
    model: str = "clef-flash",
    client: ClefDecisionsClient | None = None,
) -> dict[str, dict[str, Any]]:
    """Fast System One 6-axis decomposed pre-benchmark over sampled windows.

    For each arm and each window (batched up to 10 groups * 6 axes = 60
    questions per call), evaluates 6 orthogonal atomic questions comparing the
    candidate against the Official Gold Korean subtitle under
    ``PRO_SUBTITLE_LAWS`` (`output_tokens = 0`).
    """
    clef = client or ClefDecisionsClient(ws, settings, model)
    cp = Checkpoints(ws, "refeval")
    out: dict[str, dict[str, Any]] = {}
    for arm in arms:
        axis_sums = {ax: 0.0 for ax in DECISIONS_AXES}
        total_items = 0
        for w in windows:
            scored_groups = [g for g in w.groups if g.official]
            for idx in range(0, len(scored_groups), 10):
                chunk = scored_groups[idx : idx + 10]
                if not chunk:
                    continue
                cues_state = []
                questions: dict[str, dict[str, Any]] = {}
                for g in chunk:
                    gid = g.id
                    en_t = " / ".join(c.text.replace("\n", " ") for c in g.en)
                    gold_t = " / ".join(l.text.replace("\n", " ") for l in g.official)
                    cand_lines = g.candidates.get(arm) or []
                    cand_t = " / ".join(l.text.replace("\n", " ") for l in cand_lines)
                    cues_state.append({
                        "id": gid,
                        "en": en_t,
                        "gold_ko": gold_t,
                        "candidate_ko": cand_t,
                    })
                    if not cand_t.strip():
                        continue
                    if cand_t.strip() == gold_t.strip():
                        continue
                    questions[f"{gid}_meaning"] = {
                        "type": "noul",
                        "instructions": (
                            f"Group {gid}: Does candidate_ko ({cand_t!r}) convey the same core pragmatic meaning, "
                            f"joke, or speech act as gold_ko ({gold_t!r}) for EN ({en_t!r}) without omission or mistranslation?"
                        ),
                    }
                    questions[f"{gid}_spoken"] = {
                        "type": "noul",
                        "instructions": (
                            f"Group {gid}: Applying PRO_SUBTITLE_LAWS, does candidate_ko ({cand_t!r}) read like natural "
                            f"spoken Korean dialogue (입말) free of literal English translationese (번역투), comparable to gold_ko ({gold_t!r})?"
                        ),
                    }
                    questions[f"{gid}_condense"] = {
                        "type": "noul",
                        "instructions": (
                            f"Group {gid}: Is candidate_ko ({cand_t!r}) briskly condensed for screen reading "
                            f"(dropping filler/vocal noise like gold_ko {gold_t!r})?"
                        ),
                    }
                    questions[f"{gid}_register"] = {
                        "type": "noul",
                        "instructions": (
                            f"Group {gid}: Does candidate_ko ({cand_t!r}) match the Korean speech level (반말/존댓말) "
                            f"and address tone of gold_ko ({gold_t!r})?"
                        ),
                    }
                    questions[f"{gid}_names"] = {
                        "type": "noul",
                        "instructions": (
                            f"Group {gid}: Are proper nouns, nicknames, or cultural terms in candidate_ko ({cand_t!r}) "
                            f"handled accurately relative to gold_ko ({gold_t!r})?"
                        ),
                    }
                    questions[f"{gid}_dash"] = {
                        "type": "noul",
                        "instructions": (
                            f"Group {gid}: Does candidate_ko ({cand_t!r}) use valid multi-speaker '- ' dashes "
                            f"matching gold_ko ({gold_t!r})?"
                        ),
                    }
                state_obj = {
                    "window_id": w.id,
                    "PRO_SUBTITLE_LAWS": PRO_SUBTITLE_LAWS,
                    "items": cues_state,
                }
                answers: dict[str, dict[str, Any]] = {}
                if questions:
                    h = input_hash(
                        "decisions-prebench-v1",
                        clef.model,
                        arm,
                        json.dumps(state_obj, ensure_ascii=False, sort_keys=True),
                        json.dumps(questions, ensure_ascii=False, sort_keys=True),
                    )
                    key = f"clef_pre:{arm}:{w.id}:{idx}"
                    cached = cp.get(key, h, _DecisionsCache)
                    if cached is not None:
                        answers = cached.answers
                    else:
                        answers = clef.decide(stage="refeval-clef", state=state_obj, questions=questions)
                        cp.put(key, h, _DecisionsCache(answers=answers))
                for g in chunk:
                    gid = g.id
                    gold_t = " / ".join(l.text.replace("\n", " ") for l in g.official)
                    cand_t = " / ".join(l.text.replace("\n", " ") for l in (g.candidates.get(arm) or []))
                    total_items += 1
                    if not cand_t.strip():
                        continue
                    if cand_t.strip() == gold_t.strip():
                        for ax in DECISIONS_AXES:
                            axis_sums[ax] += 1.0
                        continue
                    for ax in DECISIONS_AXES:
                        p = float((answers.get(f"{gid}_{ax}") or {}).get("noul", 0.5))
                        axis_sums[ax] += p
        denom = max(1, total_items)
        axes_pct = {ax: round(axis_sums[ax] / denom * 100.0, 2) for ax in DECISIONS_AXES}
        score = round(sum(axes_pct[ax] * _DECISIONS_WEIGHTS[ax] for ax in DECISIONS_AXES), 2)
        out[arm] = {
            "model": clef.model,
            "items": total_items,
            "score": score,
            "axes": axes_pct,
        }
    return out


def render_decisions_md(decisions: dict[str, dict[str, Any]]) -> list[str]:
    if not decisions:
        return []
    first_model = next(iter(decisions.values())).get("model", "clef/clef-omni")
    out = [
        "",
        f"## Decomposed 6-Axis Decisions Pre-Benchmark (`{first_model}`)",
        "",
        "| arm | 6-axis score | meaning | spoken (말맛) | condense (압축) | register | names | dash | items |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, d in decisions.items():
        ax = d.get("axes") or {}
        out.append(
            f"| {arm} | {_fmt(d.get('score'))} | {_fmt(ax.get('meaning'))} | {_fmt(ax.get('spoken'))} | "
            f"{_fmt(ax.get('condense'))} | {_fmt(ax.get('register'))} | {_fmt(ax.get('names'))} | "
            f"{_fmt(ax.get('dash'))} | {d.get('items', 0)} |"
        )
    return out

# ---------------------------------------------------------------- report


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "–"
    return f"{v * 100:.1f}%" if pct else (f"{v:.2f}" if isinstance(v, float) else str(v))


def render_md(report: dict) -> str:
    m = report["metrics"]
    lim = report.get("limits") or {"cps_max": Settings.cps_max, "cpl_limit": Settings.cpl_limit}
    cps_col, cpl_col = f"CPS>{lim['cps_max']:g}", f"CPL>{lim['cpl_limit']}"
    out = [
        f"# Reference evaluation — {report['video_id']}",
        "",
        f"Gold: `{report['ref']}` (official Korean subtitles = correct answer). "
        f"Judge: {', '.join(report['judge_models'])}, {report['n_scenes']} windows, seed {report['seed']}.",
        "",
        f"| arm | chrF++ | coverage | len ratio | level agr | dash agr | {cps_col} | {cpl_col} | "
        "med CPS | med dur s | events/official | judge | "
        + " | ".join(ERROR_CATEGORIES) + " | sev sum |",
        "|---" * (13 + len(ERROR_CATEGORIES)) + "|",
    ]
    for arm, a in m.items():
        j = a.get("judge") or {}
        errs = j.get("errors", {})
        out.append(
            f"| {arm} ({report['arms'][arm]}) | {_fmt(a['chrf'])} | {_fmt(a['coverage'], True)} | "
            f"{_fmt(a['length_ratio'])} | {_fmt(a['speech_level_agreement'], True)} | "
            f"{_fmt(a['dash_agreement'], True)} | {_fmt(a['cps_over_rate'], True)} | "
            f"{_fmt(a['cpl_over_rate'], True)} | {_fmt(a.get('median_cps'))} | "
            f"{_fmt(a.get('median_duration_s'))} | {_fmt(a.get('event_ratio'))} | {_fmt(j.get('score'))} | "
            + " | ".join(_fmt(errs.get(c)) for c in ERROR_CATEGORIES)
            + f" | {_fmt(j.get('severity_sum'))} |"
        )
    o = report["official"]
    out.append(
        f"| official (gold) | – | – | 1.00 | – | – | {_fmt(o['cps_over_rate'], True)} | "
        f"{_fmt(o['cpl_over_rate'], True)} | {_fmt(o.get('median_cps'))} | {_fmt(o.get('median_duration_s'))} | "
        "1.00 | – |" + " – |" * len(ERROR_CATEGORIES) + " – |"
    )
    out += [
        "",
        "coverage = official groups where the arm has any text; len ratio = arm chars / official chars; "
        "level agr = same speech level (hasipsio/haeyo/hae) over groups where both are classified; "
        "dash agr = same dual-speaker '-' use over groups where both have text; "
        f"{cps_col} / {cpl_col} = share of displayed events over {lim['cps_max']:g} chars/s (whitespace not counted) / "
        f"{lim['cpl_limit']} chars/line (the official's own rates are the acceptable range); "
        "med CPS / med dur = median reading speed / display duration per event; "
        "events/official = displayed events per official subtitle. "
        "Error columns: judged items with severity ≥ 1 per category"
        + (" (judge columns = mean over judges)." if len(report["judge_models"]) > 1 else "."),
    ]
    if report.get("calibration"):
        out += render_judges_md(
            {f"{arm} ({report['arms'][arm]})": a["judges"] for arm, a in m.items()}, report["judge_models"]
        )
        out += render_calibration_md(report["calibration"])
    if report.get("paired"):
        out += render_paired_md(report)
    if report.get("decisions"):
        out += render_decisions_md(report["decisions"])
    return "\n".join(out) + "\n"


def run_refeval(
    video_id: str,
    arms: dict[str, str],
    ref_path: Path | str,
    n_scenes: int,
    seed: int,
    out_dir: Path | str,
    *,
    settings: Settings | None = None,
    judge_models: list[str] | None = None,
    judge_factory: JudgeFactory | None = None,
    resolve: Callable[[str], Path] = _default_resolve,
    judge: bool = True,
    evaluation_mode: str = "listwise",
    opus_ref: Path | str | None = None,
    split: str | None = None,
    cycle_id: str | None = None,
    decisions_model: str | None = None,
    decisions_client: ClefDecisionsClient | None = None,
) -> dict:
    """Align, score, judge (every model in ``judge_models``, default
    ``settings.model_main``) and calibrate each judge; write ``refeval.json``
    and ``refeval.md`` to ``out_dir``. Returns the report; ``report["groups"]``
    is the aligned group list (same shape as ``export_groups``)."""
    metadata = evaluation_metadata(evaluation_mode, split=split, cycle_id=cycle_id)
    if evaluation_mode == "paired":
        if opus_ref is None:
            raise ValueError("paired evaluation requires a frozen opus_ref subtitle artifact")
        return run_paired_refeval(
            video_id, arms, ref_path, opus_ref, n_scenes, seed, out_dir, settings=settings,
            judge_models=judge_models, judge_factory=judge_factory, resolve=resolve,
            judge=judge, split=split or "dev", cycle_id=cycle_id,
        )
    if opus_ref is not None:
        raise ValueError("opus_ref requires evaluation_mode='paired'")
    settings = settings or load_settings()
    models = judge_models or [settings.model_main]
    out_dir = Path(out_dir)
    inp = build_groups(video_id, arms, Path(ref_path), resolve=resolve)
    names = list(arms)
    metrics = {arm: arm_metrics(inp.groups, arm, settings) for arm in names}
    report: dict = {
        "video_id": video_id,
        "ref": str(ref_path),
        "arms": dict(arms),
        "seed": seed,
        **({k: v for k, v in metadata.items() if v is not None} if split is not None or cycle_id is not None else {}),
        "n_scenes": n_scenes,
        "judge_models": models,
        "alignment": {
            "groups": len(inp.groups),
            "official_lines": len(inp.official),
            "official_only_groups": sum(1 for g in inp.groups if g.official and not any(g.candidates.values())),
            "max_group_size": max((len(g.en) + len(g.official) for g in inp.groups), default=0),
            "overlap_rule": f">= {MIN_OVERLAP_MS} ms or >= {MIN_OVERLAP_FRAC:.0%} of the shorter",
        },
        "metrics": metrics,
        "official": display_rates(inp.official, settings),
        "limits": {"cps_max": settings.cps_max, "cpl_limit": settings.cpl_limit},
        "calibration": None,
        "scenes": [],
    }
    if judge and n_scenes > 0:
        ws = Workspace(
            video_id=video_id,
            data_dir=inp.base,
            cache_dir=workspace(video_id).cache_dir,
            log_dir=workspace(video_id).log_dir,
        )
        targeted = targeted_cue_ids(inp.context_map, inp.register)
        windows = select_windows(build_windows(inp.groups), n_scenes, seed, targeted)
        results = run_judges(
            ws, settings, models, windows, names, inp.official, name_swaps(inp.context_map),
            seed=seed, factory=judge_factory,
        )
        for arm, s in judge_summaries(results, names).items():
            metrics[arm].update(s)
        report["calibration"] = {m: r["calibration"] for m, r in results.items()}
        report["scenes"] = [
            {
                "id": w.id,
                "reason": w.reason,
                "scene_ids": list(dict.fromkeys(g.scene_id for g in w.groups)),
                "group_ids": [g.id for g in w.groups],
                "start_ms": w.start_ms,
                "end_ms": w.end_ms,
                "verdicts": merge_verdicts({m: r["judged"][i]["verdicts"] for m, r in results.items()}),
                "calibration": merge_verdicts({m: r["calib_judged"][i]["verdicts"] for m, r in results.items()}),
            }
            for i, w in enumerate(windows)
        ]
    if decisions_model and n_scenes > 0:
        ws_dec = Workspace(
            video_id=video_id,
            data_dir=inp.base,
            cache_dir=workspace(video_id).cache_dir,
            log_dir=workspace(video_id).log_dir,
        )
        targeted = targeted_cue_ids(inp.context_map, inp.register)
        dec_windows = select_windows(build_windows(inp.groups), n_scenes, seed, targeted)
        dec_res = evaluate_decisions_prebenchmark(
            ws_dec, settings, dec_windows, names, model=decisions_model, client=decisions_client,
        )
        report["decisions"] = dec_res
        for arm, d in dec_res.items():
            metrics[arm]["decisions"] = d
    report["groups"] = [group_record(g, names) for g in inp.groups]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "refeval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "refeval.md").write_text(render_md(report), encoding="utf-8")
    return report


def run_paired_refeval(
    video_id: str, arms: dict[str, str], ref_path: Path | str, opus_ref: Path | str,
    n_scenes: int, seed: int, out_dir: Path | str, *,
    settings: Settings | None = None, judge_models: list[str] | None = None,
    judge_factory: JudgeFactory | None = None,
    resolve: Callable[[str], Path] = _default_resolve, judge: bool = True,
    split: str = "dev", cycle_id: str | None = None,
) -> dict:
    """Opt-in file/report adapter for ``evaluate_paired``; no Opus generation."""
    settings = settings or load_settings()
    models = list(dict.fromkeys(judge_models or [settings.model_main]))
    base = resolve(video_id)
    en = [c for c in load_models(base / "cues.json", Cue) if c.text.strip()]
    scenes = load_models(base / "scenes.json", Scene) if (base / "scenes.json").is_file() else []
    fm_path, reg_path = base / "context_map.yaml", base / "register.yaml"
    context_map = load_model(fm_path, ContextMap) if fm_path.is_file() else None
    register = load_model(reg_path, RegisterMap) if reg_path.is_file() else None
    official = load_ko(Path(ref_path))
    opus_path = Path(opus_ref)
    opus = load_ko(opus_path)
    candidates = {name: load_ko(arm_srt(resolve(wid))) for name, wid in arms.items()}
    defaults = workspace(video_id)
    ws = Workspace(video_id=video_id, data_dir=base, cache_dir=defaults.cache_dir, log_dir=defaults.log_dir)
    result = evaluate_paired(
        ws, settings, en, official, candidates, opus, scenes=scenes, n_windows=n_scenes, seed=seed,
        judge=judge, judge_models=models, judge_factory=judge_factory,
        targeted=targeted_cue_ids(context_map, register), split=split, cycle_id=cycle_id,
    )
    report = {
        "video_id": video_id, "ref": str(ref_path), "arms": dict(sorted(arms.items())),
        "seed": seed, "n_scenes": n_scenes, "judge_models": models,
        "opus_ref": str(opus_path), "opus_artifact_hash": input_hash(opus_path.read_bytes()),
        "official": display_rates(official, settings),
        "limits": {"cps_max": settings.cps_max, "cpl_limit": settings.cpl_limit},
        "alignment": {
            "scope": "independent_candidate_pairs", "official_lines": len(official),
            "groups": {name: pair["alignment"]["groups"] for name, pair in result["paired"].items()},
        },
        **result,
        "scenes": [
            {**row, "comparison_arm": name}
            for name, pair in result["paired"].items() for row in pair["windows"]
        ],
        "groups": [
            {**row, "comparison_arm": name}
            for name, pair in result["paired"].items() for row in pair["groups"]
        ],
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "refeval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "refeval.md").write_text(render_md(report), encoding="utf-8")
    return report


def default_out_dir(video_id: str) -> Path:
    return workspace(video_id).data_dir / "refeval"


# ---------------------------------------------------------------- pragmatic address (PARA / GRA)


_PARA_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"|‘[^’]*’|“[^”]*”")
_PARA_NEOHUI_RE = re.compile(r"(?<![가-힣])(너희|너네|니들|니네)(?:들)?(?=[은는이가을를도만에게한테의와과랑이나]|\s|$|[^가-힣])")
_PARA_COLLECTIVE_POSS_RE = re.compile(
    r"(?<![가-힣])(너희|너네|니네)\s*(아빠|엄마|아버지|어머니|부모님|집|가족|형|누나|오빠|언니|동생|할머니|할아버지)"
)
_PARA_DANGSIN_RE = re.compile(r"(?<![가-힣])(당신)(?:들)?(?=[은는이가을를도만에게한테의와과랑이나]|\s|$|[^가-힣])")
_PARA_NER_RE = re.compile(r"(?<![가-힣])(너|네가|니가|널|넌)(?=[은는이가을를도만에게한테의와과랑이나]|\s|$|[^가-힣])")
_PARA_HOSTILE_EN_RE = re.compile(
    r"\b(fuck|fucking|motherfucker|motherfuckers|screw|screwed|asshole|bastard|bitch|rodents|limp\s+dick|crush|sue|sued|kill|vultures|traitor)\b",
    re.I,
)
_PARA_VOCATIVE_KIN_EN_RE = re.compile(r"(?:^|[,?!\s])(mom|dad|mother|father|parents)(?=[,?!\s]|$)", re.I)
_PARA_POSS_BEFORE_KIN_RE = re.compile(r"\b(your|my|his|her|their|our|a|the)\s+(mom|dad|mother|father|parents)\b", re.I)
_PARA_UPWARD_KIN_RELATIONS = frozenset({"own_parent", "elder_kin", "surrogate_parent"})
_PARA_UPWARD_CIVIL_RELATIONS = frozenset({"own_parent", "elder_kin", "surrogate_parent", "mentor", "subordinate", "kid_to_adult"})
_PARA_NO_DANGSIN_CIVIL_RELATIONS = frozenset({"own_parent", "elder_kin", "surrogate_parent", "mentor", "protege", "subordinate"})


def evaluate_pragmatic_address(
    cues: Sequence[Cue],
    ko_lines: Sequence[Line] | Mapping[str, str],
    *,
    context_map: ContextMap | None = None,
    register_map: RegisterMap | None = None,
    observations: Sequence[ChunkObservations | CueObservation] = (),
    scenes: Sequence[Scene] = (),
    ep_id: str = "",
) -> dict[str, Any]:
    """Score pragmatic pronoun use and parent-address graph coverage per aligned turn."""
    from . import register as register_mod

    non_ost_cues = [c for c in cues if not c.id.startswith("ost-") and c.source != "ost"]
    if isinstance(ko_lines, Mapping):
        groups_iter = [([c], ko_lines.get(c.id, "")) for c in non_ost_cues]
    else:
        groups_iter = [
            (g.en, " ".join(ln.text for ln in g.official))
            for g in align(non_ost_cues, list(ko_lines), {})
        ]
    edges = {
        (e.speaker, e.addressee): e
        for e in (register_map.edges if register_map else ())
    }
    pairs = (
        register_mod.cue_pairs(context_map, non_ost_cues, observations)
        if context_map is not None else {}
    )
    obs_by_cid = {
        co.cue_id: co
        for ch in observations for co in getattr(ch, "cues", [ch])
        if hasattr(co, "cue_id")
    }
    scene_of = {cid: scene.id for scene in scenes for cid in scene.cue_ids}
    scene_speakers = {
        scene.id: {
            participant for cid in scene.cue_ids
            for participant in pairs.get(cid, ()) if participant != "group"
        }
        for scene in scenes
    }
    n_risk = 0
    n_viol = 0
    gra_triggers = 0
    gra_satisfied = 0
    viols: list[dict[str, Any]] = []
    by_kind: Counter[str] = Counter()
    known_parent_chars = {
        ch.id for ch in (context_map.characters if context_map else ())
        if any(a.lower() in ("mom", "dad", "mother", "father") for a in ch.aliases)
    }
    for c_list, raw_ko in groups_iter:
        if not c_list:
            continue
        ko_txt = _PARA_QUOTED_RE.sub("", raw_ko)
        is_multi_turn = ("-" in raw_ko) or (len(c_list) > 1)
        c = c_list[0]
        en_txt = " ".join(x.text for x in c_list)
        pair = pairs.get(c.id)
        edge = edges.get(pair) if pair else None
        co = obs_by_cid.get(c.id)
        sc_id = scene_of.get(c.id, "")
        sc_participants = scene_speakers.get(sc_id, set())
        spk = pair[0] if pair else None

        has_voc_kin = bool(_PARA_VOCATIVE_KIN_EN_RE.search(en_txt)) and not bool(_PARA_POSS_BEFORE_KIN_RE.search(en_txt))
        if has_voc_kin and spk and (known_parent_chars & sc_participants):
            gra_triggers += 1
            if any(
                (spk, o) in edges and edges[(spk, o)].relation in _PARA_UPWARD_KIN_RELATIONS and edges[(spk, o)].pronoun == "none"
                for o in (known_parent_chars & sc_participants)
            ):
                gra_satisfied += 1

        en_has_2p = bool(re.search(r"\b(you|your|yours|yourself|yourselves)\b", en_txt, re.I))
        en_has_plural_2p = bool(re.search(r"\b(you guys|you all|all of you|you two|both of you)\b", en_txt, re.I))
        en_is_two_only = bool(re.search(r"\b(you two|both of you)\b", en_txt, re.I))
        is_hostile = (
            bool(_PARA_HOSTILE_EN_RE.search(en_txt))
            or bool(re.search(r"(새끼|개새끼|하이에나|병신|자식)", ko_txt))
            or bool(edge and edge.stance in ("open_enmity", "contempt_from_power"))
        )
        has_kin_upward_in_scene = any(
            (spk, other) in edges and edges[(spk, other)].relation in _PARA_UPWARD_KIN_RELATIONS
            for other in sc_participants if other != "group"
        ) if spk else False
        has_upward_in_scene = any(
            (spk, other) in edges and (
                edges[(spk, other)].relation in _PARA_UPWARD_CIVIL_RELATIONS
                or edges[(spk, other)].apparent_generation == "addressee_older"
                or edges[(spk, other)].level in ("haeyo", "hasipsio")
            )
            for other in sc_participants if other != "group"
        ) if spk else False
        has_hae_peer_in_scene = any(
            (spk, other) in edges and edges[(spk, other)].level == "hae" and edges[(spk, other)].pronoun == "너"
            for other in sc_participants if other != "group"
        ) if spk else False
        if en_is_two_only and has_hae_peer_in_scene and not has_kin_upward_in_scene:
            has_upward_in_scene = False

        ko_lvls = [level for level in addressed_levels(ko_txt) if level]
        ko_is_polite = any(level in ("haeyo", "hasipsio") for level in ko_lvls) and not is_multi_turn
        is_upward = (
            (edge and pair[1] != "group" and (edge.relation in _PARA_UPWARD_CIVIL_RELATIONS or edge.apparent_generation == "addressee_older" or edge.level in ("haeyo", "hasipsio")))
            or (en_has_plural_2p and has_upward_in_scene)
            or has_voc_kin
            or ko_is_polite
        )
        is_no_dangsin = bool(
            (edge and edge.relation in _PARA_NO_DANGSIN_CIVIL_RELATIONS and edge.stance == "civil" and not is_hostile)
            or (pair and pair[1] == "group" and edge and edge.relation == "group" and not is_hostile and "당신도 그렇게" in ko_txt)
        )
        is_audience = bool(
            (co and co.addressee in ("audience", "press", "crowd", "viewers", "viewer"))
            or (co and (co.speaker or "").lower() == "narrator")
            or (pair and pair[1] == "group" and not en_is_two_only and not is_hostile and has_upward_in_scene)
        )
        if en_has_2p and (is_upward or is_no_dangsin or is_audience):
            n_risk += 1
            ko_for_neohui = _PARA_COLLECTIVE_POSS_RE.sub("", ko_txt) if (edge and edge.relation == "peer" and edge.level == "hae") else ko_txt
            bad = None
            if _PARA_NEOHUI_RE.search(ko_for_neohui):
                if is_audience:
                    bad = "audience_neohui"
                elif (is_upward or (edge and pair[1] != "group" and edge.pronoun == "none")) and not is_hostile:
                    bad = "upward_or_mixed_neohui"
            elif _PARA_NER_RE.search(ko_txt) and not is_multi_turn:
                if (
                    edge
                    and pair[1] != "group"
                    and edge.relation != "partner"
                    and (
                        edge.relation in _PARA_UPWARD_KIN_RELATIONS
                        or (
                            edge.apparent_generation == "addressee_older"
                            and edge.step not in (4, 10)
                            and edge.relation not in ("child", "younger_kin", "ward", "protege", "subordinate", "peer")
                        )
                    )
                ) or ko_is_polite:
                    bad = "upward_ner"
            elif _PARA_DANGSIN_RE.search(ko_txt) and is_no_dangsin:
                bad = "civil_dangsin"
            if bad:
                n_viol += 1
                by_kind[bad] += 1
                viols.append({
                    "ep": f"{ep_id}:{c.id}" if ep_id else c.id,
                    "kind": bad,
                    "pair": f"{pair[0]}>{pair[1]}" if pair else None,
                    "relation": edge.relation if edge else None,
                    "level": edge.level if edge else None,
                    "en": en_txt,
                    "ko": raw_ko,
                })
    para = 1.0 - n_viol / max(n_risk, 1)
    gra = gra_satisfied / gra_triggers if gra_triggers else 1.0
    return {
        "para": round(para, 4),
        "gra": round(gra, 4),
        "risk_turns": n_risk,
        "violations": viols,
        "by_kind": dict(by_kind),
        "gra_triggers": gra_triggers,
        "gra_satisfied": gra_satisfied,
    }


# ---------------------------------------------------------------- series consistency (SCS)


def _norm_recap_ko(text: str) -> str:
    from .series_state import _strip_recap_ko_header

    s = _strip_recap_ko_header(text or "")
    # Drop separate quoted OST/sign line when stacked with dialogue and deduplicate identical lines
    lines = list(dict.fromkeys(ln.strip() for ln in s.splitlines() if ln.strip()))
    if len(lines) == 2:
        q0 = bool(re.match(r"^[\"'“”‘’].*[\"'“”‘’]$", lines[0]))
        q1 = bool(re.match(r"^[\"'“”‘’].*[\"'“”‘’]$", lines[1]))
        if q0 and not q1:
            lines = [lines[1]]
        elif q1 and not q0:
            lines = [lines[0]]
    joined = " ".join(lines)
    joined = re.sub(r"[^0-9a-zA-Z가-힣]+", "", joined)
    return joined


def evaluate_series_consistency(
    episodes: Sequence[dict[str, Any]],
    *,
    tracked_concepts: Mapping[str, Sequence[str]] | None = None,
    spoiler_probes: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute the 5-axis Series Consistency Score (SCS, 0..100) across an ordered
    list of episode records.

    Each item in `episodes` contains:
      - `ep_id`: str (e.g. 's01e01')
      - `cues`: Sequence[Cue]
      - `ko_lines`: Sequence[Line] or Mapping[str, str] (`cue_id -> ko`)
      - `context_map`: Optional[ContextMap]
      - `register_map`: Optional[RegisterMap]
      - `observations`: Optional[Sequence[ChunkObservations | CueObservation]]
      - `scenes`: Optional[Sequence[Scene]]
      - `pairs_by_cue`: Optional[Mapping[str, tuple[str, str]]]
    """
    import math
    from collections.abc import Mapping
    from .register import binary
    from .series_state import (
        _content_tokens,
        _levenshtein,
        _norm_en_utterance,
        word_boundary_in,
    )

    # Normalize each episode's aligned (Cue, ko_text) pairs
    ep_aligned: list[tuple[str, list[tuple[Cue, str]], ContextMap | None, RegisterMap | None, Mapping[str, tuple[str, str]]]] = []
    pragmatic_results: list[dict[str, Any]] = []
    slash_defects: list[dict[str, Any]] = []
    total_lines = 0
    for ep in episodes:
        ep_id = str(ep["ep_id"])
        cues: list[Cue] = list(ep["cues"])
        raw_ko = ep["ko_lines"]
        cm: ContextMap | None = ep.get("context_map")
        rm: RegisterMap | None = ep.get("register_map")
        pbc: Mapping[str, tuple[str, str]] = ep.get("pairs_by_cue") or {}
        if isinstance(raw_ko, Mapping):
            output_lines = [(cid, str(text)) for cid, text in raw_ko.items()]
            pairs = [(c, str(raw_ko.get(c.id, ""))) for c in cues if (c.text or c.raw).strip()]
        else:
            ko_list = list(raw_ko)
            output_lines = [(str(i), ln.text) for i, ln in enumerate(ko_list, 1)]
            ost_cues = [c for c in cues if c.id.startswith("ost-") and (c.text or c.raw).strip()]
            main_cues = [c for c in cues if not c.id.startswith("ost-") and (c.text or c.raw).strip()]
            pairs = []
            if ost_cues:
                def _is_ost_line(ln: Line) -> bool:
                    if ln.text.startswith("{\\an8}"):
                        return True
                    return any(
                        abs(ln.start_ms - oc.start_ms) <= 250 and abs(ln.end_ms - oc.end_ms) <= 350
                        for oc in ost_cues
                    )

                ost_ko = [ln for ln in ko_list if _is_ost_line(ln)]
                main_ko = [ln for ln in ko_list if not _is_ost_line(ln)]
                for g in align(main_cues, main_ko, {}):
                    if g.en and g.official:
                        if len(g.en) == len(g.official) > 1:
                            for c, ln in zip(g.en, g.official):
                                pairs.append((c, ln.text))
                        else:
                            ko_joined = "\n".join(ln.text for ln in g.official)
                            for c in g.en:
                                pairs.append((c, ko_joined))
                for g in align(ost_cues, ost_ko, {}):
                    if g.en and g.official:
                        ko_joined = "\n".join(ln.text.removeprefix("{\\an8}") for ln in g.official)
                        for c in g.en:
                            pairs.append((c, ko_joined))
                pairs.sort(key=lambda pair: (pair[0].start_ms, pair[0].id))
            else:
                for g in align([c for c in cues if (c.text or c.raw).strip()], ko_list, {}):
                    if g.en and g.official:
                        if len(g.en) == len(g.official) > 1:
                            for c, ln in zip(g.en, g.official):
                                pairs.append((c, ln.text))
                        else:
                            ko_joined = "\n".join(ln.text for ln in g.official)
                            for c in g.en:
                                pairs.append((c, ko_joined))
        aligned_ids = {c.id for c, _ in pairs}
        pairs.extend(
            (c, "") for c in cues if c.id not in aligned_ids and (c.text or c.raw).strip()
        )
        pairs.sort(key=lambda pair: (pair[0].start_ms, pair[0].id))
        for location, text in output_lines:
            for line in text.splitlines():
                total_lines += 1
                if re.search(r"\s+/\s+", line):
                    slash_defects.append({
                        "axis": "slash_separator_artifact",
                        "ep": f"{ep_id}:{location}",
                        "ko": line,
                    })
        ep_aligned.append((ep_id, pairs, cm, rm, pbc))
        if any(key in ep for key in ("register_map", "observations", "scenes")):
            pragmatic_results.append(evaluate_pragmatic_address(
                cues, raw_ko, context_map=cm, register_map=rm,
                observations=ep.get("observations") or (), scenes=ep.get("scenes") or (),
                ep_id=ep_id,
            ))

    # 1. Axis 1: Entity & Concept Conditional Entropy H(T|E) and LTCR
    concepts: dict[str, list[str]] = {}
    if tracked_concepts:
        for en_k, ko_forms in tracked_concepts.items():
            concepts[en_k] = list(ko_forms)
    else:
        # Auto-collect recurring character & glossary English keys from context_maps
        seen_en: Counter[str] = Counter()
        ko_cands: dict[str, list[str]] = {}
        for _ep_id, _pairs, cm, _rm, _pbc in ep_aligned:
            if cm is None:
                continue
            for ch in cm.characters:
                if ch.en_name and ch.ko_name:
                    k = ch.en_name.strip()
                    seen_en[k] += 1
                    ko_cands.setdefault(k, []).append(ch.ko_name.strip())
            for g in cm.glossary:
                if g.en and g.ko:
                    k = g.en.strip()
                    seen_en[k] += 1
                    ko_cands.setdefault(k, []).append(g.ko.strip())
        for k, cands in ko_cands.items():
            if seen_en[k] >= 2 or len(set(cands)) >= 1:
                concepts[k] = list(dict.fromkeys(cands))

    concept_metrics: dict[str, dict[str, Any]] = {}
    ltcr_num = 0.0
    ltcr_den = 0
    entropies: list[float] = []
    defects: list[dict[str, Any]] = list(slash_defects)

    for en_key, ko_forms in sorted(concepts.items()):
        counts: Counter[str] = Counter()
        occ_eps: dict[str, list[str]] = {}
        # Sort forms longest first so full name matches before given/family name
        ordered_forms = sorted(set(ko_forms), key=len, reverse=True)
        for ep_id, pairs, cm, _rm, _pbc in ep_aligned:
            for c, ko_txt in pairs:
                if not word_boundary_in(en_key, f"{c.text} {c.raw}".lower()):
                    continue
                matched = None
                for f in ordered_forms:
                    if f and f in ko_txt:
                        matched = f
                        break
                if matched is not None:
                    counts[matched] += 1
                    occ_eps.setdefault(matched, []).append(f"{ep_id}:{c.id}")
            if not counts and cm is not None:
                for ch in cm.characters:
                    if ch.en_name.strip().lower() == en_key.lower() and ch.ko_name.strip():
                        counts[ch.ko_name.strip()] += 1
                        occ_eps.setdefault(ch.ko_name.strip(), []).append(f"{ep_id}:context_map")
        total = sum(counts.values())
        if total < 2:
            continue
        # Merge pure shortenings (e.g. '신다' inside '신다 캐닝', '퍼타키' inside '새즈 퍼타키')
        clustered: Counter[str] = Counter(counts)
        forms_desc = sorted(clustered.keys(), key=len, reverse=True)
        for short_f in list(forms_desc):
            for long_f in forms_desc:
                if short_f != long_f and short_f in long_f.split():
                    clustered[long_f] += clustered.pop(short_f, 0)
                    break
        dom = max(clustered.values())
        ltcr = dom / total
        h = -sum((cnt / total) * math.log2(cnt / total) for cnt in clustered.values() if cnt > 0)
        ltcr_num += dom
        ltcr_den += total
        entropies.append(h)
        concept_metrics[en_key] = {
            "n": total,
            "ltcr": round(ltcr, 4),
            "entropy_bits": round(h, 4),
            "forms": dict(counts),
            "clustered_forms": dict(clustered),
        }
        if len(clustered) > 1:
            defects.append({
                "axis": "entity_or_concept_drift",
                "key": en_key,
                "forms": dict(clustered),
                "locations": {k: v[:5] for k, v in occ_eps.items()},
            })

    axis1_ltcr = (ltcr_num / ltcr_den) if ltcr_den > 0 else 1.0
    mean_entropy = (sum(entropies) / len(entropies)) if entropies else 0.0

    # 2. Axis 2: Dyadic Register Stability (1 - DFR)
    dyad_history: dict[str, list[tuple[str, str, str, str]]] = {}
    for ep_id, pairs, _cm, rm, pbc in ep_aligned:
        cues_in_ep = {c.id for c, _ in pairs}
        if rm is not None:
            for e in rm.edges:
                if (
                    e.speaker.startswith("ep:")
                    or e.addressee.startswith("ep:")
                    or e.speaker == "group"
                    or e.addressee == "group"
                ):
                    continue
                b_entry = binary(e.level)
                open_shifts = [sh for sh in e.shifts if sh.until_cue is None and sh.at_cue in cues_in_ep]
                b_exit = binary(open_shifts[-1].level) if open_shifts else b_entry
                if b_entry and b_exit:
                    dyad_history.setdefault(f"{e.speaker}>{e.addressee}", []).append((ep_id, b_entry, b_exit, e.relation))
        elif pbc:
            for c, ko_txt in pairs:
                pair = pbc.get(c.id)
                if not pair or "group" in pair:
                    continue
                lvls = [binary(lv) for lv in addressed_levels(ko_txt) if binary(lv)]
                if lvls:
                    b_lv = lvls[-1]
                    dyad_history.setdefault(f"{pair[0]}>{pair[1]}", []).append((f"{ep_id}:{c.id}", b_lv, b_lv, "peer"))  # type: ignore[arg-type]

    dyad_flips = 0
    dyad_transitions = 0
    for pair_key, seq in sorted(dyad_history.items()):
        if len(seq) < 2:
            continue
        for i in range(1, len(seq)):
            dyad_transitions += 1
            prev_ep, _prev_entry, prev_exit, prev_rel = seq[i - 1]
            curr_ep, curr_entry, _curr_exit, curr_rel = seq[i]
            # One-time graduation from first-encounter 'stranger_civil' (J) to permanent crew/partner (B)
            if (
                i == 1
                and prev_exit == "J"
                and curr_entry == "B"
                and prev_rel in ("stranger_civil", "peer")
                and curr_rel in ("crew_buddy", "partner", "mentor", "protege", "enemy", "peer")
                and all(x[1] == "B" for x in seq[i:])
            ):
                continue
            if curr_entry != prev_exit:
                dyad_flips += 1
                defects.append({
                    "axis": "dyad_register_flip",
                    "pair": pair_key,
                    "from": (prev_ep, prev_exit),
                    "to": (curr_ep, curr_entry),
                })
    axis2_register = 1.0 - (dyad_flips / dyad_transitions) if dyad_transitions > 0 else 1.0

    # 3. Axis 3: Recap & High-Information Callback Consistency (1 - mean_NED)
    # Retain source components so a recap split can reuse the displayed prior beads.
    prior_utts: dict[str, tuple[str, str, str, str | None, set[str], tuple[str, ...]]] = {}
    recap_neds: list[float] = []
    recap_flips = 0

    def match_prior(text: str, *, fuzzy: bool, split: bool = False) -> str | None:
        norm = _norm_en_utterance(text)
        if norm in prior_utts:
            return norm
        if not fuzzy:
            return None
        toks = set(_content_tokens(text))
        if not toks:
            return None
        best_norm = None
        best_score = (0.0, 0.0, 0)
        for prev_norm, (_, _, _, _, prev_toks, _) in prior_utts.items():
            shared = len(toks & prev_toks)
            if shared == 0:
                continue
            recap_cov = shared / len(toks)
            jaccard = shared / len(toks | prev_toks)
            if split:
                if not (shared >= 3 and jaccard >= 0.65 and recap_cov >= 0.75):
                    continue
            else:
                substring = len(norm) >= 12 and len(prev_norm) >= 12 and norm in prev_norm and recap_cov >= 0.75
                overlap = recap_cov >= 0.75 and jaccard >= 0.60 and (shared >= 3 or (shared >= 2 and len(norm) >= 15))
                if not (substring or overlap):
                    continue
            score = (jaccard, recap_cov, -abs(len(norm) - len(prev_norm)))
            if score > best_score:
                best_norm, best_score = prev_norm, score
        return best_norm

    for ep_id, pairs, _cm, _rm, _pbc in ep_aligned:
        dialogue = [
            (c, ko_txt) for c, ko_txt in pairs
            if not c.lyric and not re.search(
                r"\b(?:theme\s+song|music|song|noise)\s+(?:playing|plays|continues|stops)\b",
                c.text.lower(),
            )
        ]
        split_targets: dict[str, tuple[str, str]] = {}
        main_dialogue = [(c, ko_txt) for c, ko_txt in dialogue if not c.id.startswith("ost-")]
        for (c0, _), (c1, _) in zip(main_dialogue, main_dialogue[1:]):
            if c0.start_ms > 95_000 or c1.start_ms > 95_000:
                continue
            lines_en = [ln.strip() for ln in c0.text.splitlines() if ln.strip()]
            if len(lines_en) == 2 and all(re.match(r"^[-–—]", ln) for ln in lines_en) and not c1.text.lstrip().startswith("-"):
                norm_b = match_prior(lines_en[-1] + " " + c1.text, fuzzy=True, split=True)
                first_norm = _norm_en_utterance(lines_en[0])
                norm_a = first_norm if first_norm in prior_utts else next(
                    (norm for norm in prior_utts if len(first_norm) >= 6 and norm.startswith(first_norm + " ")),
                    None,
                )
                if norm_a is not None and norm_b is not None:
                    ko_a = prior_utts[norm_a][1].splitlines()[0]
                    ko_b = prior_utts[norm_b][1]
                    rb_lines = [ln for ln in ko_b.splitlines() if ln.strip()]
                    second_ko = rb_lines[0].lstrip("- ") if rb_lines else ko_b.lstrip("- ")
                    rem_ko = rb_lines[-1] if len(rb_lines) > 1 else ko_b
                    split_targets[c0.id] = (norm_a, ko_a + "\n- " + second_ko)
                    split_targets[c1.id] = (norm_b, rem_ko)
                    continue
            if c0.id in split_targets or c1.id in split_targets or c1.text.lstrip().startswith("-"):
                continue
            if _norm_en_utterance(c0.text) in prior_utts or _norm_en_utterance(c1.text) in prior_utts:
                continue
            t0_set = set(_content_tokens(c0.text))
            t1_set = set(_content_tokens(c1.text))
            norm = match_prior(c0.text + " " + c1.text, fuzzy=True, split=True)
            if norm is None:
                continue
            _, prev_raw_ko, _, _, prev_toks, components = prior_utts[norm]
            if not (t0_set & prev_toks) or not (t1_set & prev_toks):
                continue
            ko_parts = list(components) if len(components) == 2 else prev_raw_ko.splitlines()
            if len(ko_parts) == 2:
                ko0, ko1 = ko_parts
            else:
                ko0 = ko1 = prev_raw_ko
            split_targets[c0.id] = (norm, ko0)
            split_targets[c1.id] = (norm, ko1)

        for c, ko_txt in dialogue:
            en_norm = _norm_en_utterance(c.text)
            toks = _content_tokens(c.text)
            is_recap = c.start_ms <= 95_000
            if len(en_norm) < (6 if is_recap else 15) or len(toks) < (1 if is_recap else 4):
                continue
            target = split_targets.get(c.id)
            prev_norm = target[0] if target is not None else match_prior(c.text, fuzzy=is_recap)
            if prev_norm is None:
                continue
            prev_ep, prev_raw_ko, prev_clean, prev_lvl, _, _ = prior_utts[prev_norm]
            full_prev_clean = prev_clean
            prev_lvls = [binary(lv) for lv in addressed_levels(prev_raw_ko) if binary(lv)]
            if target is not None:
                prev_raw_ko = target[1]
                prev_clean = _norm_recap_ko(prev_raw_ko)
                prev_lvls = [binary(lv) for lv in addressed_levels(prev_raw_ko) if binary(lv)]
                prev_lvl = prev_lvls[-1] if prev_lvls else None
            ko_clean = _norm_recap_ko(ko_txt)
            if not ko_clean:
                continue
            lvls = [binary(lv) for lv in addressed_levels(ko_txt) if binary(lv)]
            b_lvl = lvls[-1] if lvls else None
            prev_sub_cleans = [_norm_recap_ko(ln) for ln in prev_raw_ko.splitlines() if _norm_recap_ko(ln)]
            curr_sub_cleans = [_norm_recap_ko(ln) for ln in ko_txt.splitlines() if _norm_recap_ko(ln)]
            shared_sub_lines = [s for s in (set(prev_sub_cleans) & set(curr_sub_cleans)) if len(s) >= 5]
            if (
                ko_clean in prev_sub_cleans
                or prev_clean in curr_sub_cleans
                or bool(shared_sub_lines)
                or (min(len(prev_clean), len(ko_clean)) >= 4 and (prev_clean in ko_clean or ko_clean in prev_clean))
                or (min(len(full_prev_clean), len(ko_clean)) >= 4 and (full_prev_clean in ko_clean or ko_clean in full_prev_clean))
            ):
                ned = 0.0
            else:
                ned = _levenshtein(prev_clean, ko_clean) / max(len(prev_clean), len(ko_clean), 1)
            recap_neds.append(ned)
            lvl_mismatch = bool(ned > 0.0 and prev_lvls and lvls and not (set(prev_lvls) & set(lvls)))
            if lvl_mismatch:
                recap_flips += 1
            if ned > 0.15 or lvl_mismatch:
                defects.append({
                    "axis": "recap_or_callback_drift",
                    "en": c.text,
                    "from_ep": prev_ep,
                    "to_ep": f"{ep_id}:{c.id}",
                    "ko_prev": prev_raw_ko,
                    "ko_curr": ko_txt,
                    "ned": round(ned, 4),
                })

        source_spans: list[tuple[str, str, int, str, tuple[str, ...]]] = [
            (c.id, c.text, c.start_ms, ko_txt, ()) for c, ko_txt in dialogue
        ]
        for (c0, ko0), (c1, ko1) in zip(main_dialogue, main_dialogue[1:]):
            if 0 <= c1.start_ms - c0.end_ms <= 2000:
                source_spans.append((
                    f"{c0.id}+{c1.id}", c0.text + " " + c1.text, c0.start_ms,
                    ko0 if ko0 == ko1 else ko0 + "\n" + ko1, (ko0, ko1),
                ))
        for cid, text, _start_ms, ko_txt, components in source_spans:
            norm = _norm_en_utterance(text)
            toks = set(_content_tokens(text))
            clean = _norm_recap_ko(ko_txt)
            if len(norm) < 6 or not toks or not clean:
                continue
            lvls = [binary(lv) for lv in addressed_levels(ko_txt) if binary(lv)]
            prior_utts.setdefault(
                norm, (f"{ep_id}:{cid}", ko_txt, clean, lvls[-1] if lvls else None, toks, components),
            )
    axis3_recap = 1.0 - (sum(recap_neds) / len(recap_neds)) if recap_neds else 1.0

    # 4. Axis 4: Locked Term Compliance (1 - LTV)
    axis4_ltv = 1.0 - (len([d for d in defects if d["axis"] == "entity_or_concept_drift"]) / max(len(concept_metrics), 1))

    # 5. Axis 5: Causal Spoiler Non-Leakage (1 - Leak)
    leaks = 0
    total_checked = sum(len(p) for _, p, _, _, _ in ep_aligned) or 1
    if spoiler_probes:
        for probe in spoiler_probes:
            reveal_ep = str(probe["reveal_ep"])
            forbidden_ko = [str(x) for x in probe.get("forbidden_ko", ())]
            for ep_id, pairs, _cm, _rm, _pbc in ep_aligned:
                if ep_id >= reveal_ep:
                    break
                for c, ko_txt in pairs:
                    if any(fk in ko_txt for fk in forbidden_ko):
                        leaks += 1
                        defects.append({
                            "axis": "causal_spoiler_leak",
                            "ep": f"{ep_id}:{c.id}",
                            "reveal_ep": reveal_ep,
                            "ko": ko_txt,
                        })
    axis5_spoiler = 1.0 - (leaks / total_checked)

    pragmatic_risks = sum(result["risk_turns"] for result in pragmatic_results)
    pragmatic_violations = [
        violation for result in pragmatic_results for violation in result["violations"]
    ]
    gra_triggers = sum(result["gra_triggers"] for result in pragmatic_results)
    gra_satisfied = sum(result["gra_satisfied"] for result in pragmatic_results)
    para = 1.0 - len(pragmatic_violations) / max(pragmatic_risks, 1)
    gra = gra_satisfied / gra_triggers if gra_triggers else 1.0
    defects.extend(
        {"axis": "pragmatic_address_calque", **violation}
        for violation in pragmatic_violations
    )

    scs = 100.0 * (
        0.30 * axis1_ltcr
        + 0.25 * axis2_register
        + 0.20 * axis3_recap
        + 0.15 * axis4_ltv
        + 0.10 * axis5_spoiler
    )
    return {
        "scs": round(scs, 2),
        "axes": {
            "entity_concept_ltcr": round(axis1_ltcr, 4),
            "mean_entropy_bits": round(mean_entropy, 4),
            "dyadic_register_stability": round(axis2_register, 4),
            "recap_callback_stability": round(axis3_recap, 4),
            "locked_term_compliance": round(axis4_ltv, 4),
            "causal_spoiler_free": round(axis5_spoiler, 4),
            "pragmatic_address_para": round(para, 4),
            "relationship_graph_gra": round(gra, 4),
            "slash_clean_rate": round(1.0 - len(slash_defects) / max(total_lines, 1), 4),
        },
        "counts": {
            "episodes": len(ep_aligned),
            "tracked_concepts": len(concept_metrics),
            "dyad_transitions": dyad_transitions,
            "dyad_flips": dyad_flips,
            "recap_callback_pairs": len(recap_neds),
            "recap_speech_level_flips": recap_flips,
            "spoiler_leaks": leaks,
            "total_defects": len(defects),
        },
        "concepts": concept_metrics,
        "defects": defects,
    }


# ---------------------------------------------------------------- reference-free quality

QUALITY_TARGETS: dict[str, float | int] = {
    "format_score_min": 99.0,
    "collision_free_score_min": 100.0,
    "register_score_min": 97.0,
    "pragmatic_score_min": 99.0,
    "ost_pos_min": 100.0,
    "completeness_score_min": 98.0,
    "sync_err_ms_max": 150.0,
    "quality_score_min": 97.5,
}


def _quality_dirs(video_id: str, ws_dir: Path | str | None) -> list[Path]:
    base = Path(ws_dir) if ws_dir is not None else workspace(video_id).data_dir
    stem = base.name.split("__", 1)[0]
    variants = [base.parent / f"{stem}__{arm}" for arm in ("flash38", "v8", "final")]
    candidates = [base, *variants, base.parent / stem] if ws_dir is not None else [*variants, base]
    return list(dict.fromkeys(candidates))


def _quality_display_events(path: Path, lines: Sequence[Line]) -> list[dict[str, Any]]:
    """Recover positions separately: load_ko intentionally returns plain text."""
    from collections import defaultdict, deque

    if not lines:
        return []
    raw_by_span: dict[tuple[int, int], Any] = defaultdict(deque)
    if path.suffix.lower() not in (".smi", ".sami"):
        subs = pysubs2.SSAFile.from_string(_decode(path))
        for event in subs.events:
            if not event.is_comment and event.end > event.start and event.plaintext.strip():
                style = subs.styles.get(event.style)
                alignment = int(style.alignment) if style is not None else 2
                raw_by_span[(int(event.start), int(event.end))].append((event.text, alignment))
    events = []
    for line in lines:
        raw, alignment = line.text, 2
        if raw_by_span[(line.start_ms, line.end_ms)]:
            raw, alignment = raw_by_span[(line.start_ms, line.end_ms)].popleft()
        positions = re.findall(r"\\an([1-9])\b", raw)
        position = f"an{positions[-1] if positions else alignment}"
        text = re.sub(r"\{[^}]*\}", "", line.text).strip()
        parts = text.splitlines()
        quoted = bool(parts) and all(
            len(part.strip()) >= 2
            and {"'": "'", "‘": "’", "[": "]"}.get(part.strip()[0]) == part.strip()[-1]
            for part in parts
        )
        events.append({
            "idx": line.idx, "start_ms": line.start_ms, "end_ms": line.end_ms,
            "text": text, "position": position, "ost": position == "an8" or quoted,
        })
    return events


def _quality_align(
    cues: Sequence[Cue], events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[int, list[Cue]]]:
    """Sweep time-ordered tracks, retaining split and merged cue associations."""
    by_cue: dict[str, list[Mapping[str, Any]]] = {cue.id: [] for cue in cues}
    by_event: dict[int, list[Cue]] = {int(event["idx"]): [] for event in events}
    left = 0
    for cue in cues:
        while left < len(events) and events[left]["end_ms"] <= cue.start_ms:
            left += 1
        right = left
        while right < len(events) and events[right]["start_ms"] < cue.end_ms:
            event = events[right]
            if event["end_ms"] > cue.start_ms:
                by_cue[cue.id].append(event)
                by_event[int(event["idx"])].append(cue)
            right += 1
    return by_cue, by_event


def _quality_number_signature(text: str) -> tuple[str, ...]:
    ordinals = {
        "첫": "1", "한": "1", "하나": "1", "두": "2", "둘": "2",
        "세": "3", "셋": "3", "네": "4", "넷": "4", "다섯": "5",
        "여섯": "6", "일곱": "7", "여덟": "8", "아홉": "9", "열": "10",
    }
    values = re.findall(r"\d+(?:[.,]\d+)*", text)
    for match in re.finditer(
        r"(?<![가-힣])(첫|한|하나|두|둘|세|셋|네|넷|다섯|여섯|일곱|여덟|아홉|열)\s*(?:번째|째|번)", text,
    ):
        values.append(ordinals[match[1]])
    return tuple(sorted(values))


def _quality_similar(left: str, right: str, minimum: int) -> bool:
    left_numbers, right_numbers = _quality_number_signature(left), _quality_number_signature(right)
    if left_numbers and right_numbers and left_numbers != right_numbers:
        return False
    a, b = _norm_off_surface(left), _norm_off_surface(right)
    aa, bb = "".join(a.split()), "".join(b.split())
    if min(len(aa), len(bb)) < minimum:
        return False
    if aa in bb or bb in aa:
        return True
    def tokens(text: str) -> set[str]:
        result = set()
        for token in text.split():
            for suffix in sorted(_OFF_SIM_JOSA_EOMI, key=len, reverse=True):
                if len(token) > len(suffix) and token.endswith(suffix):
                    token = token[:-len(suffix)]
                    break
            if len(token) >= 2:
                result.add(token)
        return result

    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) >= 2 and len(ta & tb) / max(min(len(ta), len(tb)), 1) >= 0.65


def evaluate_media_quality(
    video_id: str, *, ws_dir: Path | str | None = None,
    srt_path: Path | str | None = None, ref_path: Path | str | None = None,
    sync_delta_ms: float | None = None, settings: Settings | None = None,
) -> dict[str, Any]:
    """Audit displayed subtitles without an official translation or model calls.

    Completeness measures time coverage and untranslated output, not an LLM
    judgment of semantic fidelity. Missing narrative artifacts are disclosed.
    """
    from . import register as register_mod
    from .models import SceneTranslation, Snapshot

    settings = settings or load_settings()
    dirs = _quality_dirs(video_id, ws_dir)
    artifact_names = (
        "cues.json", "scenes.json", "observations.json", "context_map.yaml",
        "register.yaml", "snapshots.json", "translations.json",
    )
    paths = {
        name: next((directory / name for directory in dirs if (directory / name).is_file()), None)
        for name in artifact_names
    }
    if srt_path is None:
        candidates = []
        for directory in dirs:
            candidates.extend((directory / f"{video_id}.ko.srt", directory / f"{directory.name}.ko.srt"))
            candidates.extend(sorted(directory.glob("*.ko.srt")))
        srt = next((
            path for path in candidates
            if path.is_file() and "official" not in path.name.lower()
            and (ref_path is None or path.resolve() != Path(ref_path).resolve())
        ), None)
        if srt is None:
            raise FileNotFoundError(f"no generated Korean subtitle for {video_id}; supply srt_path")
    else:
        srt = Path(srt_path)
    srt_lines = load_ko(srt) if _decode(srt).strip() else []
    cues = sorted(
        load_models(paths["cues.json"], Cue) if paths["cues.json"] else [],
        key=lambda cue: (cue.start_ms, cue.end_ms),
    )
    spoken = [
        cue for cue in cues
        if cue.source != "ost" and not cue.id.startswith("ost-") and not cue.lyric and cue.text.strip()
    ]
    ost_cues_list = [cue for cue in cues if cue.source == "ost" or cue.id.startswith("ost-")]
    events = _quality_display_events(srt, srt_lines)
    if spoken:
        for ev in events:
            if ev["position"] != "an8" and ev["ost"]:
                ov_ost_cue = any(min(ev["end_ms"], oc.end_ms) - max(ev["start_ms"], oc.start_ms) >= 150 for oc in ost_cues_list)
                ov_spk_cue = any(min(ev["end_ms"], sc.end_ms) - max(ev["start_ms"], sc.start_ms) >= 150 for sc in spoken)
                if ov_spk_cue and not ov_ost_cue:
                    ev["ost"] = False
    ost_ev = [event for event in events if event["ost"]]
    dlg_ev = [event for event in events if not event["ost"]]
    scenes = load_models(paths["scenes.json"], Scene) if paths["scenes.json"] else []
    observations: list[ChunkObservations | CueObservation] = []
    if paths["observations.json"]:
        for item in json.loads(paths["observations.json"].read_text(encoding="utf-8")):
            model = ChunkObservations if "cues" in item else CueObservation
            observations.append(model.model_validate(item))
    context_map = load_model(paths["context_map.yaml"], ContextMap) if paths["context_map.yaml"] else None
    register_map = load_model(paths["register.yaml"], RegisterMap) if paths["register.yaml"] else None
    if register_map is not None:
        register_map = register_mod.canonicalize_register_map(register_map, context_map)
    snapshots = load_models(paths["snapshots.json"], Snapshot) if paths["snapshots.json"] else []
    translations = load_models(paths["translations.json"], SceneTranslation) if paths["translations.json"] else []
    translated = {line.cue_id: line for scene in translations for line in scene.cues}
    by_cue, by_event = _quality_align(spoken, dlg_ev)
    ko_by_cue = {cid: "\n".join(event["text"] for event in matches) for cid, matches in by_cue.items()}
    defects: list[dict[str, Any]] = []

    def defect(stage: str, kind: str, event: Mapping[str, Any], **extra: Any) -> None:
        defects.append({
            "stage": stage, "kind": kind, "start_ms": event["start_ms"],
            "end_ms": event["end_ms"], "text": event["text"], **extra,
        })

    cpl_over = cps_over = slash_count = over_two_lines_count = 0
    for event in events:
        text = event["text"]
        for kind, bad in (
            ("cpl_over", any(len(part) > settings.cpl_limit for part in text.splitlines())),
            ("cps_over", cps_chars(text) / max((event["end_ms"] - event["start_ms"]) / 1000, 0.001) > settings.cps_max),
            ("slash_separator", " / " in text),
            ("over_two_lines", len(text.splitlines()) > 2),
        ):
            if bad:
                defect("layout/format", kind, event)
            if kind == "cpl_over":
                cpl_over += bad
            elif kind == "cps_over":
                cps_over += bad
            elif kind == "slash_separator":
                slash_count += text.count(" / ")
            else:
                over_two_lines_count += bad
    count = max(len(events), 1)
    cps_over_rate = cps_over / count
    format_score = max(
        0.0, 100.0 * (1.0 - (cpl_over + slash_count + over_two_lines_count) / count - 0.5 * cps_over_rate),
    )

    collision_counts = {
        "same_channel_overlaps": 0, "ost_duplicate_translations": 0,
        "ost_dialogue_redundancies": 0, "adjacent_dialogue_echoes": 0,
    }

    def collision(kind: str, first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
        collision_counts[kind] += 1
        defect(
            "layout/ost", {
                "same_channel_overlaps": "same_channel_time_overlap",
                "ost_duplicate_translations": "ost_duplicate_translations",
                "ost_dialogue_redundancies": "ost_dialogue_redundancy",
                "adjacent_dialogue_echoes": "adjacent_dialogue_echo",
            }[kind], first,
            other_start_ms=second["start_ms"], other_end_ms=second["end_ms"],
            other_text=second["text"], channel=first["position"],
        )

    active: list[Mapping[str, Any]] = []
    for event in events:
        active = [previous for previous in active if previous["end_ms"] - event["start_ms"] > 50]
        for previous in active:
            if previous["position"] == event["position"] and event["position"] in ("an8", "an2"):
                if min(previous["end_ms"], event["end_ms"]) - max(previous["start_ms"], event["start_ms"]) > 50:
                    collision("same_channel_overlaps", previous, event)
        active.append(event)
    recent_ost: list[Mapping[str, Any]] = []
    for event in ost_ev:
        recent_ost = [previous for previous in recent_ost if event["start_ms"] - previous["end_ms"] <= 12_000]
        for previous in recent_ost:
            if _quality_similar(previous["text"], event["text"], 4):
                collision("ost_duplicate_translations", previous, event)
        recent_ost.append(event)
    bottom = [event for event in events if event["position"] == "an2" and not event["ost"]]
    an8_ost = [event for event in ost_ev if event["position"] == "an8"]
    for event in an8_ost:
        ov_bottom = [
            other for other in bottom
            if min(event["end_ms"], other["end_ms"]) - max(event["start_ms"], other["start_ms"]) > 150
        ]
        if len(ov_bottom) != 1:
            continue
        other = ov_bottom[0]
        ov_top = [
            top_ev for top_ev in an8_ost
            if min(top_ev["end_ms"], other["end_ms"]) - max(top_ev["start_ms"], other["start_ms"]) > 150
        ]
        norm_top = "".join(_norm_off_surface(event["text"]).split())
        norm_bot = "".join(_norm_off_surface(other["text"]).split())
        if (
            len(ov_top) == 1
            and abs(event["start_ms"] - other["start_ms"]) <= 1500
            and abs(event["end_ms"] - other["end_ms"]) <= 1500
            and len(norm_top) >= 2
            and norm_top == norm_bot
        ):
            collision("ost_dialogue_redundancies", event, other)
    for previous, event in zip(bottom, bottom[1:]):
        norm = "".join(_norm_off_surface(previous["text"]).split())
        if len(norm) < 6 or norm != "".join(_norm_off_surface(event["text"]).split()):
            continue
        if event["start_ms"] - previous["end_ms"] > 2000:
            continue
        en_previous = by_event.get(previous["idx"], [])
        en_current = by_event.get(event["idx"], [])
        source_a = re.sub(r"\W+", "", " ".join(cue.text for cue in en_previous)).casefold()
        source_b = re.sub(r"\W+", "", " ".join(cue.text for cue in en_current)).casefold()
        if not (source_a and source_a == source_b):
            collision("adjacent_dialogue_echoes", previous, event)
    collision_free_score = max(0.0, 100.0 - 2.5 * sum(collision_counts.values()))

    alias_collisions = stranger_banmal_count = scene_oscillations = snap_disagree = 0
    if context_map is not None:
        token_owners: dict[str, set[str]] = {}
        for character in context_map.characters:
            for form in (character.id, character.en_name, character.ko_name, *character.aliases):
                for token in _norm_off_surface(form).casefold().split():
                    token_owners.setdefault(token, set()).add(character.id)
        for character in context_map.characters:
            for alias in dict.fromkeys(character.aliases):
                tokens = _norm_off_surface(alias).casefold().split()
                if len(tokens) == 1 and token_owners.get(tokens[0], set()) - {character.id}:
                    alias_collisions += 1
                    defects.append({
                        "stage": "context_map", "kind": "alias_collision",
                        "character": character.id, "text": alias,
                        "other_characters": sorted(token_owners[tokens[0]] - {character.id}),
                    })
    pairs = register_mod.cue_pairs(context_map, spoken, observations) if context_map is not None else {}
    resolver = register_mod.name_resolver(context_map) if context_map is not None else lambda value: value
    for cid, line in translated.items():
        if cid not in pairs and line.speaker and line.addressee:
            spk, adr = resolver(line.speaker), resolver(line.addressee)
            if spk and adr and spk != adr:
                pairs[cid] = (spk, adr)
    cue_order = {cue.id: index for index, cue in enumerate(cues)}
    table = register_mod.RegisterTable(register_map, list(cue_order)) if register_map is not None else None
    scene_of = {cid: scene.id for scene in scenes for cid in scene.cue_ids}
    levels_by_scene: dict[tuple[str, str, str], list[tuple[Cue, str]]] = {}
    for cue in spoken:
        pair = pairs.get(cue.id)
        text = ko_by_cue.get(cue.id, "")
        levels = [register_mod.binary(level) for level in addressed_levels(text) if register_mod.binary(level)]
        if not pair or len(levels) != 1 or any(part.lstrip().startswith(DASHES) for part in text.splitlines()):
            continue
        edge = table.edge(*pair) if table is not None else None
        expected = table.level_at(*pair, cue.id) if table is not None else None
        event = {"start_ms": cue.start_ms, "end_ms": cue.end_ms, "text": text}
        if edge and edge.relation == "stranger_civil" and edge.step == 9 and register_mod.binary(expected) == "J" and levels[0] == "B":
            stranger_banmal_count += 1
            defect("register/translate", "stranger_banmal", event, cue_id=cue.id, pair=list(pair))
        if pair[1] != "group" and cue.id in scene_of:
            levels_by_scene.setdefault((scene_of[cue.id], *pair), []).append((cue, levels[0]))
    for (scene_id, spk, adr), entries in levels_by_scene.items():
        if len(entries) < 3:
            continue
        edge = table.edge(spk, adr) if table is not None else None
        if edge and any(
            table.in_shift(spk, adr, cue.id)
            or shift.at_cue == cue.id or shift.until_cue == cue.id
            for cue, _ in entries for shift in edge.shifts
        ):
            continue
        counts = Counter(level for _, level in entries)
        minority = min(counts, key=counts.get)
        if len(counts) == 2 and counts[minority] / len(entries) <= 0.35:
            for cue, level in entries:
                if level == minority:
                    scene_oscillations += 1
                    defect("register/translate", "scene_oscillation", {
                        "start_ms": cue.start_ms, "end_ms": cue.end_ms, "text": ko_by_cue[cue.id],
                    }, cue_id=cue.id, scene_id=scene_id, pair=[spk, adr])
    scenes_by_id = {scene.id: scene for scene in scenes}
    if table is not None:
        for snapshot in snapshots:
            scene = scenes_by_id.get(snapshot.scene_id)
            if scene is None:
                continue
            for key, entry in snapshot.registers.items():
                if ">" not in key:
                    continue
                spk, adr = key.split(">", 1)
                for cid in scene.cue_ids:
                    expected = table.level_at(spk, adr, cid)
                    if expected is not None and entry.level_at(cid, cue_order) != expected:
                        snap_disagree += 1
                        defects.append({
                            "stage": "snapshots", "kind": "snapshot_register_disagreement",
                            "scene_id": scene.id, "cue_id": cid, "pair": key,
                            "expected": expected, "text": entry.level_at(cid, cue_order),
                        })
    register_score = max(
        0.0, 100.0 - 5.0 * alias_collisions - 2.0 * stranger_banmal_count
        - 1.5 * scene_oscillations - 0.5 * snap_disagree,
    )
    para_res = evaluate_pragmatic_address(
        spoken, ko_by_cue, context_map=context_map, register_map=register_map,
        observations=observations, scenes=scenes, ep_id=video_id,
    )
    pragmatic_score = round(100.0 * para_res["para"], 2)
    defects.extend({"stage": "pragmatic/address", **violation} for violation in para_res["violations"])
    ost_res = evaluate_ost_and_sync(ost_ev, dlg_ev, sync_delta_ms=sync_delta_ms)
    ost_score = (ost_res["ost_pos"] + ost_res["ost_emp"]) / 2
    for event in ost_ev:
        single = evaluate_ost_and_sync([event], dlg_ev)
        if single["ost_emp"] < 100:
            defect("layout/ost", "ost_unquoted", event)
    if ost_res["ost_pos"] < 100:
        defects.append({
            "stage": "layout/ost", "kind": "ost_position",
            "text": "Place entire OST chains at the top only while dialogue overlaps.",
            "ost_pos": ost_res["ost_pos"],
        })
    if ost_res["sync_err_ms"] is not None and ost_res["sync_err_ms"] > QUALITY_TARGETS["sync_err_ms_max"]:
        defects.append({"stage": "sync", "kind": "sync_error", "sync_err_ms": ost_res["sync_err_ms"]})

    missing = [cue for cue in spoken if not by_cue[cue.id]]
    raw_music_count = english_only_count = 0
    for event in events:
        if "♪" in event["text"]:
            raw_music_count += 1
            defect("translate", "raw_music_symbol", event)
        if not event["ost"] and re.search(r"[A-Za-z]", event["text"]) and not re.search(r"[가-힣]", event["text"]):
            english_only_count += 1
            defect("translate", "untranslated_english", event)
    for cue in missing:
        defect("translate", "missing_dialogue", {
            "start_ms": cue.start_ms, "end_ms": cue.end_ms, "text": cue.text,
        }, cue_id=cue.id)
    coverage = (len(spoken) - len(missing)) / len(spoken) if spoken else float(bool(events))
    completeness_score = max(0.0, 100.0 * (coverage - (raw_music_count + english_only_count) / count))
    if not events:
        defect("emit", "empty_subtitles", {"start_ms": 0, "end_ms": 0, "text": ""})
    scores = {
        "format_score": round(format_score, 2), "collision_free_score": collision_free_score,
        "register_score": register_score, "pragmatic_score": pragmatic_score,
        "ost_score": round(ost_score, 2), "completeness_score": round(completeness_score, 2),
    }
    quality_score = round(
        0.25 * format_score + 0.25 * collision_free_score + 0.20 * register_score
        + 0.15 * pragmatic_score + 0.10 * ost_score + 0.05 * completeness_score, 2,
    )
    report: dict[str, Any] = {
        "video_id": video_id, "ws_dir": str(dirs[0]), "srt_path": str(srt),
        **scores, **ost_res, "quality_score": quality_score,
        "passes_targets": (
            quality_score >= QUALITY_TARGETS["quality_score_min"] and collision_free_score == 100.0
            and stranger_banmal_count == alias_collisions == snap_disagree == 0 and ost_res["ost_pos"] == 100.0
        ),
        "targets": dict(QUALITY_TARGETS),
        "counts": {
            "subtitle_events": len(events), "spoken_cues": len(spoken), "missing_dialogue": len(missing),
            "raw_music_count": raw_music_count, "english_only_count": english_only_count,
            "cpl_over": cpl_over, "slash_count": slash_count, "over_two_lines_count": over_two_lines_count,
            **collision_counts, "alias_collisions": alias_collisions, "snap_disagree": snap_disagree,
            "stranger_banmal_count": stranger_banmal_count, "scene_oscillations": scene_oscillations,
        },
        "cpl_over_rate": cpl_over / count, "cps_over_rate": cps_over_rate, "coverage": round(coverage, 4),
        "artifact_paths": {name: str(path) if path is not None else None for name, path in paths.items()},
        "unassessed": [name for name, path in paths.items() if path is None],
        "pragmatic_eval": para_res, "actionable_defects": defects,
    }
    if ref_path is not None and Path(ref_path).is_file():
        gold = load_ko(Path(ref_path))
        groups = align(cues, gold, {"quality": srt_lines})
        metrics = arm_metrics(groups, "quality", settings)
        report["reference_eval"] = {
            "path": str(ref_path), "chrF++": metrics["chrf"],
            "coverage": metrics["coverage"], "level_agreement": metrics["speech_level_agreement"],
            "dash_agreement": metrics["dash_agreement"],
        }
    return report


def evaluate_corpus_quality(
    items: Sequence[str | Mapping[str, Any]], *, settings: Settings | None = None,
) -> dict[str, Any]:
    """Evaluate every media independently; a corpus passes only if all pass."""
    if not items:
        raise ValueError("quality evaluation requires at least one media item")
    settings = settings or load_settings()
    reports = []
    for item in items:
        config = {"video_id": item} if isinstance(item, str) else dict(item)
        reports.append(evaluate_media_quality(**config, settings=settings))
    score_keys = (
        "format_score", "collision_free_score", "register_score", "pragmatic_score",
        "ost_score", "completeness_score", "quality_score",
    )
    summary = {
        key: round(sum(report[key] for report in reports) / len(reports), 2)
        for key in score_keys
    }
    summary.update({
        "media_count": len(reports), "passed_count": sum(report["passes_targets"] for report in reports),
        "passes_targets": all(report["passes_targets"] for report in reports),
        "defect_count": sum(len(report["actionable_defects"]) for report in reports),
    })
    return {
        "media": reports, "summary": summary, "quality_score": summary["quality_score"],
        "passes_targets": summary["passes_targets"], "targets": dict(QUALITY_TARGETS),
    }


def render_quality_md(report: Mapping[str, Any]) -> str:
    """Render reference-free scores and the defects needed for the next cycle."""
    def cell(value: Any) -> str:
        return str(value).replace("|", r"\|").replace("\n", "<br>")

    if "media" in report:
        summary = report["summary"]
        out = [
            "# Subtitle corpus quality", "",
            f"Quality: **{summary['quality_score']:.2f}/100**; targets: **{'PASS' if report['passes_targets'] else 'FAIL'}**.",
            "", "| Media | Quality | Targets | Defects |", "| --- | ---: | --- | ---: |",
        ]
        out.extend(
            f"| {cell(media['video_id'])} | {media['quality_score']:.2f} | "
            f"{'PASS' if media['passes_targets'] else 'FAIL'} | {len(media['actionable_defects'])} |"
            for media in report["media"]
        )
        out.extend(["", *[render_quality_md(media).replace("# Subtitle quality:", "## Subtitle quality:", 1) for media in report["media"]]])
        return "\n".join(out).rstrip() + "\n"
    out = [
        f"# Subtitle quality: {cell(report['video_id'])}", "",
        "Reference-free evaluation; official Korean subtitles are optional.",
        "Completeness checks coverage and untranslated output, not semantic fidelity.", "",
        f"Quality: **{report['quality_score']:.2f}/100**; targets: **{'PASS' if report['passes_targets'] else 'FAIL'}**.",
        "", "| Axis | Score / 100 |", "| --- | ---: |",
    ]
    for name, key in (
        ("NETFLIX-FMT", "format_score"), ("ZERO-DUPLICATE-COLLISION", "collision_free_score"),
        ("REGISTER-CONSISTENCY", "register_score"), ("PRAGMATIC-ADDRESS", "pragmatic_score"),
        ("OST-QUALITY", "ost_score"), ("SEMANTIC-COMPLETENESS", "completeness_score"),
    ):
        out.append(f"| {name} | {report[key]:.2f} |")
    out.extend([
        "", f"OST position: {report['ost_pos']:.2f}%; emphasis: {report['ost_emp']:.2f}%.",
        f"Sync error: {report['sync_err_ms'] if report['sync_err_ms'] is not None else 'unassessed'} ms.",
    ])
    if report.get("unassessed"):
        out.extend(["", "Missing artifacts (related checks are unassessed): " + ", ".join(f"`{name}`" for name in report["unassessed"]) + "."])
    out.extend(["", "## Actionable defects", ""])
    if not report["actionable_defects"]:
        out.append("No detected defects.")
    for defect in report["actionable_defects"]:
        location = mmss(defect["start_ms"]) if "start_ms" in defect else str(defect.get("cue_id", defect.get("scene_id", "")))
        text = defect.get("text", defect.get("ko", defect.get("en", "")))
        other = f" / {cell(defect['other_text'])}" if "other_text" in defect else ""
        out.append(f"- `{cell(defect['stage'])}` / `{cell(defect['kind'])}` {location}: {cell(text)}{other}")
    if report.get("reference_eval"):
        out.extend(["", "## Optional reference agreement", "", "Reference agreement does not affect target gating."])
        out.extend(f"- {cell(key)}: {cell(value)}" for key, value in report["reference_eval"].items())
    return "\n".join(out).rstrip() + "\n"
