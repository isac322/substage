"""Translator memory: aligned (English cue → official Korean line) pairs from
reference videos, retrieved as few-shot house-style examples for S3.

``build_pairs`` reuses refeval's time alignment (connected components of
cross-track overlaps) between a video's English cues and its official Korean
subtitles and keeps groups of 1-3 English cues ↔ 1-2 official lines. Groups
with a lyric cue, no English words or no Hangul are skipped; whitespace is
normalised (Korean display line breaks are kept as ``\\n``).

The corpus is one JSONL file for all videos (``config.memory_path()``); adding a
video replaces its rows. ``retrieve`` scores pairs against a scene's English
cues with BM25 (lowercased word tokens, contractions kept whole), summed over
the scene's cues, and returns the top-k distinct pairs (deterministic
tie-break) with at most ``WINDOW_CAP`` pairs from any ``WINDOW_MS`` window of a
video. The video being translated is always excluded — its rows and its term
statistics — so evaluation against its official subtitles stays honest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config, refeval
from .models import Cue
from .store import _atomic_write

MAX_EN = 3  # English cues per pair
MAX_KO = 2  # official lines per pair
WINDOW_MS = 10_000  # diversity window
WINDOW_CAP = 3  # max pairs from one window of one video
BM25_K1 = 1.2
BM25_B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")
_HANGUL_RE = re.compile(r"[가-힣]")
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})


@dataclass(frozen=True)
class Pair:
    video_id: str
    start_ms: int
    en: str
    ko: str  # official line texts joined by "\n"
    prev_en: str = ""  # previous English cue (context)

    @property
    def id(self) -> str:
        return f"{self.video_id}@{self.start_ms}"


def base_video_id(video_id: str) -> str:
    """Video id without an experiment-arm suffix (``<video_id>__<arm>``)."""
    return video_id.split("__", 1)[0]


def tokens(text: str) -> list[str]:
    """Lowercased word tokens; contractions (don't, you're) stay one token."""
    return _TOKEN_RE.findall(text.lower().translate(_APOSTROPHES))


def _norm_en(text: str) -> str:
    return " ".join(text.split())


def _norm_ko(text: str) -> str:
    rows = (" ".join(line.split()) for line in text.split("\n"))
    return "\n".join(r for r in rows if r)


# ---------------------------------------------------------------- building


def build_pairs(video_id: str, cues: Sequence[Cue], official_lines: Sequence[refeval.Line]) -> list[Pair]:
    """Aligned (EN → official KO) pairs of one video, in time order."""
    en = sorted((c for c in cues if c.text.strip()), key=lambda c: (c.start_ms, c.end_ms))
    prev_of: dict[str, str] = {}
    prev = ""
    for c in en:
        prev_of[c.id] = prev
        if not c.lyric:
            prev = _norm_en(c.text)
    pairs: list[Pair] = []
    for g in refeval.align(en, list(official_lines), {}):
        if not (1 <= len(g.en) <= MAX_EN and 1 <= len(g.official) <= MAX_KO):
            continue
        if any(c.lyric for c in g.en):
            continue
        en_text = _norm_en(" ".join(c.text for c in g.en))
        ko_text = _norm_ko("\n".join(ln.text for ln in g.official))
        if not tokens(en_text) or not _HANGUL_RE.search(ko_text):
            continue
        pairs.append(Pair(video_id, g.en[0].start_ms, en_text, ko_text, prev_of[g.en[0].id]))
    return pairs


# ---------------------------------------------------------------- storage


def load_pairs(path: Path | None = None) -> list[Pair]:
    path = path or config.memory_path()
    if not path.is_file():
        return []
    return [Pair(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def add_video(video_id: str, pairs: Iterable[Pair], path: Path | None = None) -> int:
    """Replace *video_id*'s rows in the corpus with *pairs*; returns the row count written."""
    path = path or config.memory_path()
    new = list(pairs)
    if any(p.video_id != video_id for p in new):
        raise ValueError(f"pairs must all belong to {video_id!r}")
    rows = [p for p in load_pairs(path) if p.video_id != video_id] + new
    rows.sort(key=lambda p: (p.video_id, p.start_ms, p.en))
    data = "".join(json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in rows)
    _atomic_write(path, data.encode("utf-8"))
    return len(new)


def stats(pairs: Sequence[Pair]) -> dict[str, int]:
    """Pair count per video (sorted by video id)."""
    return dict(sorted(Counter(p.video_id for p in pairs).items()))


# ---------------------------------------------------------------- retrieval


class _Index:
    """BM25 inverted index over the English side of a pair list."""

    def __init__(self, pairs: Sequence[Pair]):
        self.pairs = list(pairs)
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, p in enumerate(self.pairs):
            toks = tokens(p.en)
            self.lengths.append(len(toks))
            for t, tf in Counter(toks).items():
                self.postings.setdefault(t, []).append((i, tf))
        n = len(self.pairs)
        self.avgdl = (sum(self.lengths) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - len(ps) + 0.5) / (len(ps) + 0.5)) for t, ps in self.postings.items()}
        h = hashlib.sha256()
        for p in self.pairs:
            h.update(json.dumps(asdict(p), ensure_ascii=False, sort_keys=True).encode("utf-8"))
        self.fingerprint = h.hexdigest()[:16]

    def scores(self, queries: Iterable[str]) -> dict[int, float]:
        out: dict[int, float] = {}
        for q in queries:
            for t, qtf in Counter(tokens(q)).items():
                ps = self.postings.get(t)
                if not ps:
                    continue
                idf = self.idf[t]
                for i, tf in ps:
                    norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[i] / self.avgdl)
                    out[i] = out.get(i, 0.0) + qtf * idf * tf * (BM25_K1 + 1) / norm
        return out


class Corpus:
    """All memory pairs plus lazily built per-exclusion BM25 indexes (thread-safe)."""

    def __init__(self, pairs: Sequence[Pair]):
        self.pairs = list(pairs)
        self._indexes: dict[str, _Index] = {}
        self._lock = threading.Lock()

    def index(self, exclude_video: str) -> _Index:
        with self._lock:
            idx = self._indexes.get(exclude_video)
            if idx is None:
                idx = _Index([p for p in self.pairs if p.video_id != exclude_video])
                self._indexes[exclude_video] = idx
            return idx

    def fingerprint(self, exclude_video: str) -> str:
        """Hash of every pair visible when translating *exclude_video*."""
        return self.index(exclude_video).fingerprint


_cache: dict[Path, tuple[tuple[int, int, int], Corpus]] = {}
_cache_lock = threading.Lock()


def load_corpus(path: Path | None = None) -> Corpus | None:
    """The corpus at *path* (default ``config.memory_path()``), cached by
    mtime/size; None when the file is missing or empty."""
    path = path or config.memory_path()
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    key = (st.st_ino, st.st_mtime_ns, st.st_size)
    with _cache_lock:
        hit = _cache.get(path)
        if hit is not None and hit[0] == key:
            return hit[1]
    pairs = load_pairs(path)
    corpus = Corpus(pairs) if pairs else None
    if corpus is not None:
        with _cache_lock:
            _cache[path] = (key, corpus)
    return corpus


def retrieve(
    corpus: Corpus | Sequence[Pair], scene_en_texts: Sequence[str], exclude_video: str, k: int
) -> list[Pair]:
    """Top-*k* distinct pairs for a scene's English cues, never from
    *exclude_video*, at most ``WINDOW_CAP`` per ``WINDOW_MS`` window of a video.
    Ordered by score (desc), ties by (video_id, start_ms, en)."""
    if k <= 0:
        return []
    if not isinstance(corpus, Corpus):
        corpus = Corpus(corpus)
    idx = corpus.index(exclude_video)
    scored = idx.scores(scene_en_texts)
    ranked = sorted(
        scored.items(),
        key=lambda kv: (-kv[1], idx.pairs[kv[0]].video_id, idx.pairs[kv[0]].start_ms, idx.pairs[kv[0]].en),
    )
    out: list[Pair] = []
    seen: set[tuple[str, str]] = set()
    per_window: Counter[tuple[str, int]] = Counter()
    for i, score in ranked:
        if score <= 0:
            break
        p = idx.pairs[i]
        if p.video_id == exclude_video:
            continue
        text_key = (" ".join(tokens(p.en)), p.ko)
        window = (p.video_id, p.start_ms // WINDOW_MS)
        if text_key in seen or per_window[window] >= WINDOW_CAP:
            continue
        seen.add(text_key)
        per_window[window] += 1
        out.append(p)
        if len(out) == k:
            break
    return out
