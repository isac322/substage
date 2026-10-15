"""Application configuration: secrets, settings, and per-video workspace dirs.

All paths follow the XDG Base Directory spec on every OS (macOS included), so the
layout is identical on the dev Mac and the Linux host that runs the pipeline:
  $XDG_CONFIG_HOME/substage/secrets.env   KEY=VALUE lines, loaded into os.environ
  $XDG_CONFIG_HOME/substage/config.toml   optional overrides for Settings
  $XDG_DATA_HOME/substage/<video>          artifacts and outputs
  $XDG_CACHE_HOME/substage/<video>         proxies, audio, downloaded subtitles
  $XDG_STATE_HOME/substage/<video>         usage logs
"""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

APP = "substage"


def _xdg(var: str, default: str) -> Path:
    value = os.environ.get(var)
    return Path(value) if value and os.path.isabs(value) else Path.home() / default


_CONFIG_DIR = _xdg("XDG_CONFIG_HOME", ".config") / APP
_SECRETS_PATH = _CONFIG_DIR / "secrets.env"
_CONFIG_TOML = _CONFIG_DIR / "config.toml"


def _load_env_file(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if not key or key in os.environ:
            continue
        if key == "GOOGLE_APPLICATION_CREDENTIALS" and value:
            cred_path = Path(value).expanduser()
            if not cred_path.is_file():
                alt = Path.home() / ".config" / "gcloud" / cred_path.name
                if alt.is_file():
                    cred_path = alt
            if cred_path.is_file():
                os.environ[key] = str(cred_path)
            continue
        os.environ[key] = value


def load_secrets() -> None:
    """Read ~/.config/substage/secrets.env into os.environ.

    Lines are `KEY=VALUE`; blank lines and `#` comments are skipped. Values may
    be single- or double-quoted. Existing environment variables are NOT
    overwritten — real env always wins over the file.
    """
    _load_env_file(_SECRETS_PATH)
def default_media_bin(tool: str) -> str:
    """``<tool>-rk`` (jellyfin-ffmpeg with Rockchip MPP/RGA) when on PATH on Linux, else *tool*."""
    if sys.platform.startswith("linux") and shutil.which(f"{tool}-rk"):
        return f"{tool}-rk"
    return tool


@dataclass(frozen=True)
class Settings:
    model_main: str = "gemini-3.8-flash"
    model_light: str = "gemini-3.5-flash-lite"  # blind KO read, compression
    # S3 text-mode translator: a plain gemini-* id, anthropic/<model> or
    # codex/<model> (OMP proxies, see llm.py); None = model_main. AV calls always
    # use Gemini. Opus beat Gemini Flash, Sonnet and GPT on chrF and on three
    # cross-family judges, at no cost through the OMP proxy.
    model_translate: str | None = "anthropic/claude-opus-5-5"
    # Text-only stages (S2a video map, S2b timeline, S4 review/finale/regen,
    # S5 compress, S5c condense); None = model_translate (see text_model).
    # Video stages (viewing pass, picture check, AV translation) stay on Gemini.
    model_text: str | None = None
    media_resolution: str = "low"
    # Processing tier for Gemini and OpenAI (codex/) calls: "priority" for
    # lower latency / fewer timeouts; None = provider default. Anthropic has none.
    service_tier: str | None = "priority"
    thinking_level: str = "low"  # default for all calls; stages may override
    # S3 translation: medium beat low (fewer meaning/register errors, chrF +2.2)
    # and high (longer lines, no gain) on a 6-video official-referenced ablation.
    translate_thinking: str = "medium"
    max_cues_per_scene: int = 40
    scene_gap_ms: int = 4000
    viewing_chunk_ms: int = 600_000  # ~10 min viewing-pass chunks, cut at scene boundaries
    # Reading speed in characters per second, whitespace excluded
    # (validate.cps_chars). Calibrated on the official Korean subtitles of six
    # videos (10,697 events): median 4.87 -> cps_target, p94 7.99 -> cps_max.
    cps_target: float = 4.9
    cps_max: float = 8.0
    # Characters per line incl. spaces, measured on 13 official videos: 99.4%
    # of lines are <= 14 chars, and one-line cues stay <= 12 (98.5%); longer
    # one-liners are broken into two lines.
    cpl_limit: int = 14
    cpl_one_line: int = 12
    # S5c layout (official translators' timing habits, measured):
    min_duration_ms: int = 1000  # cues are stretched to ~1 s when there is room
    short_min_ms: int = 500  # ...but >= 0.5 s is fine for text of <= short_text_chars
    short_text_chars: int = 7
    max_duration_ms: int = 5700  # practical display maximum
    extend_ms: int = 250  # end extension into free space past the English end
    min_gap_ms: int = 83  # chained gap (2 frames); events never overlap
    pause_min_ms: int = 250  # gaps between min_gap and this snap to a chained gap
    merge_max_gap_ms: int = 250  # merge only adjacent cues closer than this
    merge_max_span_ms: int = 4500  # combined span of a merged event
    merge_max_chars: int = 20  # merged Korean characters (cps_chars)
    split_min_ms: int = 3000  # dash / multi-sentence cues at least this long split
    split_piece_min_ms: int = 900  # every split piece lasts at least this long
    split_en_chars: int = 60  # one-sentence cues this long (English) may split at a clause
    # S3 per-cue soft length target: aim = floor(max chars * aim_ratio)
    aim_ratio: float = 0.85
    # Narrative context fed to every translation/review/validation prompt:
    # "lite" = story facts only (names, knowledge, recent events, glossary,
    # callbacks, ambiguity); "full" also renders the video map's speech-level
    # matrix, register shifts, voice cards and emotion states.
    narrative_context: str = "lite"
    # S3 translator memory: official-subtitle examples (other videos) retrieved
    # per scene from memory_path(); 0 disables the TRANSLATOR EXAMPLES block.
    memory_k: int = 12
    # S3 firm-register retry: a scene whose lines leave a firm register edge
    # is translated once more. Off: the held-out firm precision (91.7 %)
    # missed the 95 % bar; firm violations are still logged to
    # register_report.json.
    register_retry: bool = False
    # DPAF experiment roles.  These are orchestration metadata only; the
    # existing model_main/model_translate/model_text routing remains unchanged.
    dpaf_perception_model: str = "gemini-3.8-flash"
    dpaf_translation_model: str = "codex/gpt-6.1-sol"
    dpaf_router_model: str = "codex/gpt-6-luna"
    dpaf_comparison_models: tuple[str, ...] = (
        "gemini-3.8-flash",
        "gemini-4-argon",
        "codex/gpt-6-luna",
        "codex/gpt-6.1-sol",
    )
    # Optional Cloudflare Clef / Clef-omni Decisions API model (e.g.
    # "clef-omni", "clef", "clef-flash") for S4b decomposed 6-axis diagnosis;
    # None keeps the default LLM blind_read path.
    decisions_model: str | None = None
    # One explicit ID across every video, workspace and cycle in an experiment.
    # None fails closed for budgeted providers; OpenAI does not require an ID.
    experiment_id: str | None = None
    gemini_budget_usd: float = 500.0
    opus_budget_usd: float = 300.0
    ffmpeg_bin: str = field(default_factory=lambda: default_media_bin("ffmpeg"))
    ffprobe_bin: str = field(default_factory=lambda: default_media_bin("ffprobe"))

    def __post_init__(self) -> None:
        if self.narrative_context not in NARRATIVE_CONTEXTS:
            raise ValueError(f"narrative_context must be one of {NARRATIVE_CONTEXTS}, got {self.narrative_context!r}")
        if self.memory_k < 0:
            raise ValueError(f"memory_k must be >= 0, got {self.memory_k}")
        if self.experiment_id is not None and (
            not isinstance(self.experiment_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.experiment_id) is None
        ):
            raise ValueError("experiment_id must be 1–64 ASCII letters, digits, underscores or hyphens")
        for name, maximum in (("gemini_budget_usd", 500.0), ("opus_budget_usd", 300.0)):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be finite and between 0 and {maximum}")


NARRATIVE_CONTEXTS = ("lite", "full")


def narrative_context(settings) -> str:
    """The narrative level a client's settings select; duck-typed settings
    without the field get the Settings default."""
    return getattr(settings, "narrative_context", Settings.narrative_context)


def text_model(settings) -> str:
    """Model id the text-only stages use: model_text, else model_translate,
    else model_main. Duck-typed settings without the fields get model_main."""
    return (getattr(settings, "model_text", None) or getattr(settings, "model_translate", None)
            or settings.model_main)


_NARRATIVE_TAG = re.compile(r"^(\s*(?:- )?)\[(full|lite)\] ")


def narrative_text(text: str, narrative: str) -> str:
    """Prompt text for one narrative level: lines tagged ``[full] `` or
    ``[lite] `` (after optional indent and ``- `` bullet) are kept, tag
    removed, only for that level; untagged lines are shared."""
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        m = _NARRATIVE_TAG.match(line)
        if m is None:
            out.append(line)
        elif m.group(2) == narrative:
            out.append(m.group(1) + line[m.end() :])
    return "".join(out)


def load_settings() -> Settings:
    """Settings with overrides from ~/.config/substage/config.toml (tomllib).

    Unknown keys are ignored so the file stays forward-compatible.
    """
    try:
        data = tomllib.loads(_CONFIG_TOML.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Settings()
    allowed = {f.name: f.type for f in fields(Settings)}
    overrides = {k: v for k, v in data.items() if k in allowed}
    return Settings(**overrides)


@dataclass(frozen=True)
class Workspace:
    """Per-video directory layout under the XDG base directories."""

    video_id: str
    data_dir: Path  # $XDG_DATA_HOME/substage/video_id
    cache_dir: Path  # $XDG_CACHE_HOME/substage/video_id
    log_dir: Path  # $XDG_STATE_HOME/substage/video_id

    def path(self, name: str) -> Path:
        """data_dir/name, parents created."""
        p = self.data_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def cache(self, name: str) -> Path:
        """cache_dir/name, parents created."""
        p = self.cache_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def workspace(video_id: str) -> Workspace:
    return Workspace(
        video_id=video_id,
        data_dir=_xdg("XDG_DATA_HOME", ".local/share") / APP / video_id,
        cache_dir=_xdg("XDG_CACHE_HOME", ".cache") / APP / video_id,
        log_dir=_xdg("XDG_STATE_HOME", ".local/state") / APP / video_id,
    )


def memory_path() -> Path:
    """Translator-memory corpus shared by all videos: $XDG_DATA_HOME/substage/memory.jsonl."""
    return _xdg("XDG_DATA_HOME", ".local/share") / APP / "memory.jsonl"
