"""S0 English subtitle acquisition.

1) Embedded English TEXT track -> extract (prefer SDH over plain; forced
   extracted separately). No verification needed.
2) Otherwise download via the subliminal Python API, filter out wrong
   title/year candidates (subliminal happily returns e.g. Action Series 2002
   subs for Part One), sync each with ffsubsync, then VERIFY CONTENT:
   Gemini transcribes 6 short audio windows and we compare word overlap
   with the synced cue text. ffsubsync's score alone proved unreliable.

External tools: ffmpeg/ffprobe (media.py), ffsubsync (subprocess).
subliminal is imported lazily so the module stays importable without it.
"""

from __future__ import annotations

import difflib
import logging
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

from . import media, subs
from .models import Cue

if TYPE_CHECKING:
    from .config import Workspace
    from .gemini import GeminiClient

logger = logging.getLogger(__name__)

#: Providers we query, in CLI-parity with the original design.
PROVIDERS = ["opensubtitles", "podnapisi", "bsplayer", "subtis"]

_MAX_CANDIDATES = 5
_WINDOW_COUNT = 6
_VERIFY_WINDOW_MS = 20_000
_ACCEPT_RATIO = 0.5


class AcquireError(RuntimeError):
    """No usable English subtitle could be obtained."""


class _WindowTranscript(BaseModel):
    transcript: str


# One clip per call: batching several clips in one request made the model shift
# transcripts between clips (observed on a long video), which looks like a wrong video.
_VERIFY_PROMPT = """\
You are given one short audio clip from a feature video.
Transcribe the English speech verbatim; use an empty string when there is no speech.
Reply with JSON: {"transcript": "..."}"""


# ----------------------------------------------------------------- sync log


_OFFSET_RE = re.compile(
    r"offset[^-+\d]*(?P<offset>[-+]?\d+(?:\.\d+)?)\s*(?:s|sec(?:ond)?s?)?\b",
    re.IGNORECASE,
)
_SCALE_RE = re.compile(
    r"scale(?:\s*factor)?[^\d]*(?P<scale>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_LOW_QUALITY_RE = re.compile(
    r"low[- ]quality|score too low|skipping sync|no alignment|alignment (?:was )?(?:too )?poor"
    r"|unsuccessful sync|unable to sync",
    re.IGNORECASE,
)


def _parse_sync_log(text: str) -> tuple[float | None, float | None, bool]:
    """Extract (offset_s, scale, low_quality_flag) from ffsubsync output."""
    offset = scale = None
    if m := _OFFSET_RE.search(text):
        offset = float(m.group("offset"))
    if m := _SCALE_RE.search(text):
        scale = float(m.group("scale"))
    return offset, scale, bool(_LOW_QUALITY_RE.search(text))


def _run_ffsubsync(video: Path, sub_in: Path, sub_out: Path) -> tuple[bool, float | None, float | None]:
    """Sync *sub_in* against *video* audio; returns (ok, offset_s, scale).

    Low-quality alignments come back as ``ok=False`` (the
    ``--skip-sync-on-low-quality`` flag makes ffsubsync leave the input
    timings untouched and complain in its log).
    """
    proc = subprocess.run(
        [
            "ffsubsync", str(video), "-i", str(sub_in), "-o", str(sub_out),
            "--skip-sync-on-low-quality",
        ],
        capture_output=True,
        text=True,
    )
    log = f"{proc.stdout}\n{proc.stderr}"
    offset, scale, low = _parse_sync_log(log)
    ok = proc.returncode == 0 and not low and sub_out.exists()
    return ok, offset, scale


# ------------------------------------------------------------- candidates


@dataclass
class _Candidate:
    """One downloaded English subtitle candidate."""

    path: Path
    provider: str
    hearing_impaired: bool
    score: int
    desc: str  # human-readable, for error messages


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _title_ok(video: Any, cand_title: str | None) -> bool:
    """Reject only when both titles exist and clearly differ.

    Substring matches must be near-complete: "Action Series" inside
    "Action Series: Part One" is a *different* video (2002 vs 2017).
    """
    vt, ct = getattr(video, "title", None), cand_title
    if not vt or not ct:
        return True
    a = " ".join(_norm_words(vt))
    b = " ".join(_norm_words(ct))
    if not a or not b:
        return True
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if shorter in longer and len(shorter) >= 0.7 * len(longer):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85


def _year_ok(video: Any, cand_year: int | None) -> bool:
    vy = getattr(video, "year", None)
    return not (vy and cand_year) or int(cand_year) == int(vy)


def _candidate_title_year(subtitle: Any) -> tuple[str | None, int | None]:
    """Pull (title, year) metadata from a provider subtitle object."""
    title = (
        getattr(subtitle, "title", None)
        or getattr(subtitle, "series", None)
    )
    year = getattr(subtitle, "year", None)
    try:
        year = int(year) if year else None
    except (TypeError, ValueError):
        year = None
    return title, year


def _media_id_match(video: Any, subtitle: Any) -> bool | None:
    """Whether the candidate's media id matches the video's; None when either
    is missing. A match is authoritative: guessit titles from release names
    ("Action Series - Part Two" -> "Action Series") are not."""
    vid, cid = getattr(video, "media_id", None), getattr(subtitle, "media_id", None)
    if not vid or not cid:
        return None
    return str(vid).lower() == str(cid).lower()


def nfo_media_id(video_path: Path) -> str | None:
    """Media id from a Kodi/Jellyfin .nfo next to the video, if present."""
    for nfo in sorted(Path(video_path).parent.glob("*.nfo")):
        m = re.search(r"\btt\d{7,9}\b", nfo.read_text(encoding="utf-8", errors="replace"))
        if m:
            return m.group(0)
    return None


def _as_standalone_video(video: Any) -> Any:
    """Ensure a standalone video guess with a false episode number is normalized
    to its top-level title."""
    import inspect

    import subliminal

    if not hasattr(video, "series"):
        return video
    cls = getattr(subliminal, "Video", type(video))
    fields = set(inspect.signature(cls.__init__).parameters) - {"self", "name", "title"}
    kwargs = {k: v for k, v in vars(video).items() if k in fields}
    title = getattr(video, "series", None) or getattr(video, "title", None)
    try:
        return cls.fromname(video.name)
    except Exception:
        return video


def _list_candidates(ws: "Workspace", video_path: Path, media_id: str | None = None) -> list[_Candidate]:
    """Download and rank English subtitle candidates via subliminal
    (imported lazily so this module loads without the package).

    Candidates with contradictory title/year evidence (e.g. Action Series 2002
    subs for a Part One rip) are rejected before download. Returns the top
    ``_MAX_CANDIDATES``, ranked score-desc with hearing-impaired preferred.
    Tests inject ``candidate_lister`` and never reach this code.
    """
    from babelfish import Language  # subliminal dependency
    import subliminal

    video = _as_standalone_video(subliminal.scan_video(str(video_path)))
    # We got here because no embedded English TEXT track exists; image tracks that
    # subliminal reports as embedded subtitles must not suppress the search.
    video.subtitles = set()
    # Free-text tag search drowns in other Action Series titles; a media id makes
    # providers return subtitles for exactly this video.
    media_id = media_id or nfo_media_id(video_path)
    if media_id:
        video.media_id = media_id

    with subliminal.AsyncProviderPool(providers=PROVIDERS) as pool:
        listed = pool.list_subtitles(video, {Language("eng")})
        kept: list[tuple[int, bool, Any]] = []
        seen: set[tuple[str, str]] = set()
        for sub in listed:
            key = (sub.provider_name, str(sub.id))
            if key in seen:
                continue  # providers list the same file more than once
            seen.add(key)
            if getattr(sub, "foreign_only", False):
                continue  # forced-only tracks are not dialogue subs
            match = _media_id_match(video, sub)
            if match is False:
                logger.info("reject candidate (media_tag): %s vs video %s", getattr(sub, "media_id", None), video.media_id)
                continue
            if match is None:
                title, year = _candidate_title_year(sub)
                if not _title_ok(video, title):
                    logger.info("reject candidate (title): %s vs video %s", title, video.title)
                    continue
                if not _year_ok(video, year):
                    logger.info("reject candidate (year): %s vs video %s", year, video.year)
                    continue
            try:
                score = subliminal.compute_score(sub, video)
            except Exception:  # noqa: BLE001 - provider-specific scoring quirks
                score = 0
            kept.append((score, bool(getattr(sub, "hearing_impaired", False)), sub))

        kept.sort(key=lambda t: (t[0], t[1]), reverse=True)
        top = kept[:_MAX_CANDIDATES]
        for _, _, sub in top:
            pool.download_subtitle(sub)

    out_dir = ws.cache_dir / "subs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cands: list[_Candidate] = []
    for i, (score, hi, sub) in enumerate(top):
        if not sub.content:
            continue
        try:
            ext = Path(sub.get_path(video)).suffix or ".srt"
        except Exception:  # noqa: BLE001 - unknown subtitle_format
            ext = ".srt"
        path = out_dir / f"cand-{i:02d}-{sub.provider_name}{ext}"
        path.write_bytes(sub.content)
        title, year = _candidate_title_year(sub)
        cands.append(
            _Candidate(
                path=path,
                provider=sub.provider_name,
                hearing_impaired=hi,
                score=score,
                desc=f"{sub.provider_name} ({title or '?'}, {year or '?'})",
            )
        )
    return cands


# ------------------------------------------------------------- verification


def _verify_windows(cues: list[Cue], n: int = _WINDOW_COUNT) -> list[tuple[int, int]]:
    """Pick n windows of ~20 s, each containing at least one cue, spread
    evenly over the subtitled span. Returns (start_ms, end_ms) pairs."""
    if not cues:
        return []
    if n <= 1 or len(cues) == 1:
        anchors = [cues[len(cues) // 2]]
    else:
        last = len(cues) - 1
        anchors = [cues[round(i * last / (n - 1))] for i in range(n)]
    last_end = cues[-1].end_ms
    seen: set[int] = set()
    windows: list[tuple[int, int]] = []
    for anchor in anchors:
        start = max(0, anchor.start_ms - 2_000)
        end = min(start + _VERIFY_WINDOW_MS, last_end)
        start = max(0, end - _VERIFY_WINDOW_MS)
        if start in seen:
            continue
        seen.add(start)
        windows.append((start, end))
    return windows


def _cut_audio(video: Path, audio: media.Stream, start_ms: int, end_ms: int, out: Path) -> Path:
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-ss", f"{start_ms / 1000:.3f}", "-to", f"{end_ms / 1000:.3f}",
            "-i", str(video),
            "-map", f"0:{audio.index}", "-vn",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out.exists():
        raise AcquireError(f"ffmpeg audio cut failed for verify window: {proc.stderr[-300:]}")
    return out


def _words(text: str) -> list[str]:
    return _norm_words(text)


def apply_sync_offset(cues: list[Cue], offset_ms: int) -> list[Cue]:
    """Shift cue timings, clamping at zero and retaining at least 100 ms."""
    shifted: list[Cue] = []
    for cue in cues:
        start_ms = max(0, cue.start_ms + offset_ms)
        end_ms = max(start_ms + 100, cue.end_ms + offset_ms)
        shifted.append(cue.model_copy(update={"start_ms": start_ms, "end_ms": end_ms}))
    return shifted


def _verify_ratio(cues: list[Cue], windows: list[tuple[int, int]], transcripts: list[str]) -> float:
    """Median SequenceMatcher ratio between heard speech and cue text."""
    ratios: list[float] = []
    for (start, end), transcript in zip(windows, transcripts):
        expected = " ".join(
            c.text.replace("\n", " ") for c in cues if c.start_ms < end and c.end_ms > start
        )
        ew, tw = _words(expected), _words(transcript)
        if not ew:
            continue
        ratios.append(difflib.SequenceMatcher(None, ew, tw).ratio())
    return statistics.median(ratios) if ratios else 0.0


def _transcribe_windows(
    gc: "GeminiClient", wavs: list[Path]
) -> list[str]:
    return [
        gc.generate(
            stage="acquire-verify",
            prompt=_VERIFY_PROMPT,
            schema=_WindowTranscript,
            audio=[wav],
            model=gc.settings.model_light,
        ).transcript
        for wav in wavs
    ]


# ------------------------------------------------------------------- main


def acquire_english(
    ws: "Workspace",
    gc: "GeminiClient",
    video: Path,
    streams: list[media.Stream],
    *,
    media_id: str | None = None,
    candidate_lister: Callable[..., list[Any]] | None = None,
    sync_runner: Callable[[Path, Path, Path], tuple[bool, float | None, float | None]] | None = None,
    audio_cutter: Callable[[Path, media.Stream, int, int, Path], Path] | None = None,
) -> tuple[Path, Path | None, str, float | None, float | None]:
    """Return (en_sub_path, forced_path, origin, sync_offset_s, sync_scale).

    Embedded text subs are used as-is. Otherwise candidates are downloaded,
    synced, and content-verified; the first one passing a >=0.5 median
    word-overlap check wins. Raises AcquireError when nothing passes —
    the user may then pass --en-sub manually.

    ``candidate_lister`` may return ``_Candidate`` objects (production path)
    or plain ``Path``s to subtitle files (tests/simple injection).
    """
    video = Path(video)
    picks = media.pick_english_text_subs(streams)

    forced_path: Path | None = None
    if "forced" in picks:
        forced_path = media.extract_subtitle(
            video, picks["forced"], ws.cache("subs/en.forced.srt")
        )

    # Path 1: embedded English text track, SDH preferred.
    for role in ("sdh", "main"):
        if role in picks:
            en_path = media.extract_subtitle(
                video, picks[role], ws.cache(f"subs/en.{role}.srt")
            )
            return en_path, forced_path, f"embedded:{picks[role].index}", None, None

    # Path 2: download + sync + verify.
    try:
        audio = media.pick_dialogue_audio(streams)
    except media.MediaError as exc:
        raise AcquireError(f"{exc}; cannot verify downloaded subtitles") from exc
    lister = candidate_lister or _list_candidates
    sync = sync_runner or _run_ffsubsync
    cut = audio_cutter or _cut_audio

    candidates = list(lister(ws, video, media_id) if media_id else lister(ws, video))

    tried: list[str] = []
    for i, cand in enumerate(candidates):
        cand_path = cand.path if isinstance(cand, _Candidate) else Path(cand)
        desc = cand.desc if isinstance(cand, _Candidate) else f"candidate#{i} {cand_path.name}"
        synced = ws.cache(f"subs/synced-{i:02d}.srt")
        ok, offset, scale = sync(video, cand_path, synced)
        if not ok:
            tried.append(f"{desc}: sync failed/low-quality")
            continue

        cues = subs.load_cues(synced)
        windows = _verify_windows(cues)
        wavs = [
            cut(video, audio, s, e, ws.cache(f"subs/verify-{i:02d}-{j}.wav"))
            for j, (s, e) in enumerate(windows)
        ]
        transcripts = _transcribe_windows(gc, wavs)
        ratio = _verify_ratio(cues, windows, transcripts)
        if ratio >= _ACCEPT_RATIO:
            final = ws.path("en.srt")
            final.write_bytes(synced.read_bytes())
            provider = cand.provider if isinstance(cand, _Candidate) else "unknown"
            return final, forced_path, f"downloaded:{provider}", offset, scale
        tried.append(f"{desc}: content overlap {ratio:.2f} < {_ACCEPT_RATIO}")

    detail = "; ".join(tried) if tried else "no candidates found"
    raise AcquireError(
        f"no verifiable English subtitle for {video.name} ({detail}); "
        "pass --en-sub to supply one manually"
    )
