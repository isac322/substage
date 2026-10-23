"""Media probing and extraction via ffprobe/ffmpeg (S0 ingest).

All functions are thin, deterministic wrappers around the external CLIs.
Nothing here touches the network or Gemini.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

from .config import Workspace, load_settings
from .models import VideoInfo, Scene

logger = logging.getLogger(__name__)


@functools.cache
def ffmpeg_bin() -> str:
    """ffmpeg executable from Settings (``ffmpeg-rk`` on the RK3588 host, else ``ffmpeg``)."""
    return load_settings().ffmpeg_bin


@functools.cache
def ffprobe_bin() -> str:
    return load_settings().ffprobe_bin


@functools.cache
def has_rkmpp(ffmpeg: str) -> bool:
    """True when *ffmpeg* lists the ``rkmpp`` hwaccel (jellyfin-ffmpeg on Rockchip)."""
    try:
        proc = subprocess.run([ffmpeg, "-hide_banner", "-hwaccels"], capture_output=True, text=True)
    except OSError:
        return False
    if proc.returncode != 0:
        return False
    return "rkmpp" in proc.stdout.split()


#: Text subtitle codecs we can feed to pysubs2 (no OCR path).
TEXT_SUB_CODECS = frozenset({"subrip", "ass", "ssa", "mov_text", "webvtt"})

#: Words in an audio stream title that mark it as not dialogue.
_NON_DIALOGUE_TITLE = re.compile(r"commentary|description|director", re.IGNORECASE)


class MediaError(RuntimeError):
    """ffprobe/ffmpeg failure or no usable stream."""


@dataclass
class Stream:
    index: int
    codec_type: str
    codec_name: str
    language: str | None
    title: str | None
    channels: int | None
    forced: bool
    hearing_impaired: bool


def _run(cmd: list[str], *, what: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise MediaError(f"{what} failed (exit {proc.returncode}): {' | '.join(tail)}")
    return proc


def probe(video: Path) -> tuple[list[Stream], int]:
    """Return (streams, duration_ms) for *video* via ``ffprobe -show_streams``."""
    video = Path(video)
    proc = _run(
        [ffprobe_bin(), *PROBE_ARGS, str(video)],
        what=f"ffprobe {video.name}",
    )
    return parse_probe(proc.stdout, str(video))


PROBE_ARGS = ["-v", "error", "-print_format", "json", "-show_streams", "-show_format"]


def parse_probe(stdout: str, label: str) -> tuple[list[Stream], int]:
    """Parse ``ffprobe -print_format json -show_streams -show_format`` output."""
    try:
        info = json.loads(stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - ffprobe contract
        raise MediaError(f"ffprobe returned invalid JSON for {label}: {exc}") from exc

    streams: list[Stream] = []
    for st in info.get("streams", []):
        tags = st.get("tags") or {}
        disp = st.get("disposition") or {}
        title = tags.get("title") or st.get("title")
        forced = bool(disp.get("forced", 0)) or bool(
            title and re.search(r"forced", title, re.IGNORECASE)
        )
        hearing_impaired = bool(disp.get("hearing_impaired", 0)) or bool(
            title and re.search(r"\bSDH\b|\bCC\b", title, re.IGNORECASE)
        )
        streams.append(
            Stream(
                index=int(st["index"]),
                codec_type=st.get("codec_type", ""),
                codec_name=st.get("codec_name", ""),
                language=(tags.get("language") or None),
                title=title or None,
                channels=st.get("channels"),
                forced=forced,
                hearing_impaired=hearing_impaired,
            )
        )

    duration_s: float | None = None
    try:
        duration_s = float((info.get("format") or {}).get("duration") or "")
    except ValueError:
        duration_s = None
    if duration_s is None:
        # Fall back to the longest stream duration.
        candidates = [
            float(st["duration"])
            for st in info.get("streams", [])
            if st.get("duration")
        ]
        duration_s = max(candidates) if candidates else 0.0
    return streams, int(round(duration_s * 1000))


def pick_dialogue_audio(streams: list[Stream]) -> Stream:
    """Pick the English dialogue track: never commentary/description, most channels first.
    Untagged tracks (no language / "und") count only when none is tagged English:
    single-audio web/YTS rips often leave the language unset."""
    dialogue = [
        s
        for s in streams
        if s.codec_type == "audio" and not (s.title and _NON_DIALOGUE_TITLE.search(s.title))
    ]
    candidates = [s for s in dialogue if (s.language or "").lower().startswith("en")] or [
        s for s in dialogue if (s.language or "und").lower() == "und"
    ]
    if not candidates:
        raise MediaError("no English non-commentary audio stream found")
    # max() returns the first maximal element -> most channels, then lowest index.
    return max(candidates, key=lambda s: (s.channels or 0))


def pick_english_text_subs(streams: list[Stream]) -> dict[str, Stream]:
    """Classify English text subtitle streams into {"main", "sdh", "forced"}.

    Only real text codecs qualify; image formats (PGS, VobSub) are skipped.
    First stream wins each role; an English forced stream is never 'main'/'sdh'.
    """
    picks: dict[str, Stream] = {}
    for s in sorted(
        (s for s in streams if s.codec_type == "subtitle" and s.codec_name in TEXT_SUB_CODECS),
        key=lambda s: s.index,
    ):
        if (s.language or "").lower()[:2] != "en":
            continue
        if s.forced:
            picks.setdefault("forced", s)
        elif s.hearing_impaired:
            picks.setdefault("sdh", s)
        else:
            picks.setdefault("main", s)
    return picks


def extract_subtitle(video: Path, stream: Stream, out: Path) -> Path:
    """Extract embedded subtitle *stream* to *out* (codec chosen by suffix)."""
    video, out = Path(video), Path(out)
    codec = {".srt": "srt", ".ass": "ass", ".ssa": "ass", ".vtt": "webvtt"}.get(
        out.suffix.lower(), "srt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-i", str(video),
            "-map", f"0:{stream.index}", "-c:s", codec, str(out),
        ],
        what=f"extract subtitle stream {stream.index}",
    )
    return out


def _up_to_date(src: Path, out: Path) -> bool:
    return out.exists() and out.stat().st_mtime >= src.stat().st_mtime and out.stat().st_size > 0


def _partial(out: Path) -> Path:
    """Temp path next to *out*; outputs are renamed into place only when ffmpeg succeeds."""
    return out.with_name(f"{out.stem}.partial{out.suffix}")


def _hwaccel_args() -> list[str]:
    # VideoToolbox decodes 4K HEVC 10-bit in hardware on Apple Silicon. Linux hosts use the
    # rkmpp path (RK_DECODE_ARGS) when the ffmpeg build has it, else decode in software.
    return ["-hwaccel", "videotoolbox"] if sys.platform == "darwin" else []


#: Rockchip MPP hardware decode, frames stay in DRM PRIME buffers for scale_rkrga.
RK_DECODE_ARGS = ["-hwaccel", "rkmpp", "-hwaccel_output_format", "drm_prime"]

PROXY_VIDEO_ARGS = ["-vf", "scale=-2:360,fps=2", "-c:v", "libx264", "-crf", "32", "-preset", "veryfast"]
#: Same output as PROXY_VIDEO_ARGS: drop to 2 fps before RGA scales (10-bit HDR -> 8-bit
#: nv12, no tonemap, same as the software path), download, encode with libx264.
PROXY_RK_VIDEO_ARGS = [
    "-vf", "fps=2,scale_rkrga=w=-2:h=360:format=nv12,hwdownload,format=nv12",
    "-c:v", "libx264", "-crf", "32", "-preset", "veryfast",
]
PROXY_AUDIO_ARGS = ["-c:a", "aac", "-ac", "1", "-ar", "48000"]

REVIEW_VIDEO_ARGS = [
    "-vf", "scale=-2:720", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
    "-pix_fmt", "yuv420p", "-profile:v", "high",
]
REVIEW_RK_VIDEO_ARGS = [
    "-vf", "scale_rkrga=w=-2:h=720:format=nv12",
    "-c:v", "h264_rkmpp", "-rc_mode", "CQP", "-qp_init", "26", "-profile:v", "high", "-level", "4.1",
]
REVIEW_AUDIO_ARGS = ["-c:a", "aac", "-ac", "2", "-b:a", "128k"]
REVIEW_MUX_ARGS = ["-map_chapters", "-1", "-sn", "-dn", "-movflags", "+faststart"]


def _run_hw_or_sw(hw_cmd: list[str] | None, sw_cmd: list[str], *, what: str) -> None:
    """Run *hw_cmd* when given; on failure (e.g. codec rkmpp can't decode) run *sw_cmd*."""
    if hw_cmd is not None:
        try:
            _run(hw_cmd, what=f"{what} (rkmpp)")
            return
        except MediaError as e:
            logger.warning("%s; falling back to software", e)
    _run(sw_cmd, what=what)


def _encode(video: Path, out: Path, maps: list[str], hw: list[str], sw: list[str], *, what: str) -> None:
    """Encode *video* to out's .partial with the rkmpp path when available, then rename into place."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = _partial(out)
    ff = ffmpeg_bin()
    head = [ff, "-nostdin", "-v", "error", "-y"]
    hw_cmd = [*head, *RK_DECODE_ARGS, "-i", str(video), *maps, *hw, str(tmp)] if has_rkmpp(ff) else None
    sw_cmd = [*head, *_hwaccel_args(), "-i", str(video), *maps, *sw, str(tmp)]
    _run_hw_or_sw(hw_cmd, sw_cmd, what=what)
    tmp.replace(out)


def make_proxy(video: Path, audio: Stream, out: Path) -> Path:
    """Build a tiny A/V proxy for Gemini: 360p, 2 fps, mono 48 kHz AAC.

    Skipped (idempotent) when *out* exists and is newer than *video*.
    """
    video, out = Path(video), Path(out)
    if _up_to_date(video, out):
        logger.info("proxy up to date: %s", out)
        return out
    _encode(
        video, out, ["-map", "0:v:0", "-map", f"0:{audio.index}"],
        [*PROXY_RK_VIDEO_ARGS, *PROXY_AUDIO_ARGS],
        [*PROXY_VIDEO_ARGS, *PROXY_AUDIO_ARGS],
        what="make_proxy",
    )
    return out


def make_review_video(video: Path, audio_stream: Stream, out: Path) -> Path:
    """Browser-playable review copy: 720p H.264 at the source frame rate, stereo AAC 128k, mp4 +faststart.

    h264_rkmpp with rkmpp decode + RGA scale when available, else libx264 veryfast crf 26.
    HDR sources are not tonemapped (same as the proxy). Skipped when *out* is up to date.
    """
    video, out = Path(video), Path(out)
    if _up_to_date(video, out):
        logger.info("review video up to date: %s", out)
        return out
    _encode(
        video, out, ["-map", "0:v:0", "-map", f"0:{audio_stream.index}"],
        [*REVIEW_RK_VIDEO_ARGS, *REVIEW_AUDIO_ARGS, *REVIEW_MUX_ARGS],
        [*REVIEW_VIDEO_ARGS, *REVIEW_AUDIO_ARGS, *REVIEW_MUX_ARGS],
        what="make_review_video",
    )
    return out


def make_dialogue_wav(video: Path, audio: Stream, out: Path) -> Path:
    """Extract 16 kHz mono PCM wav of the dialogue track (for ffsubsync)."""
    video, out = Path(video), Path(out)
    if _up_to_date(video, out):
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = _partial(out)
    _run(
        [
            ffmpeg_bin(), "-nostdin", "-v", "error", "-y", "-i", str(video),
            "-map", f"0:{audio.index}", "-vn",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(tmp),
        ],
        what="make_dialogue_wav",
    )
    tmp.replace(out)
    return out


_FRAME_HEIGHT = 360
_FRAME_SW_FILTER = f"scale=-2:{_FRAME_HEIGHT},format=rgb24"
_FRAME_RK_FILTER = (
    f"scale_rkrga=w=-2:h={_FRAME_HEIGHT}:format=nv12,"
    "hwdownload,format=nv12,format=rgb24"
)


def _valid_frame(path: Path) -> bool:
    """Check the complete RGB PNG, including chunk CRCs and decoded scanline lengths."""
    try:
        data = path.read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            return False
        offset = 8
        pixels = bytearray()
        width = height = 0
        while offset + 12 <= len(data):
            length = struct.unpack_from(">I", data, offset)[0]
            end = offset + length + 12
            if end > len(data):
                return False
            kind = data[offset + 4:offset + 8]
            payload = data[offset + 8:end - 4]
            checksum = struct.unpack_from(">I", data, end - 4)[0]
            if zlib.crc32(kind + payload) != checksum:
                return False
            if offset == 8:
                if kind != b"IHDR" or length != 13:
                    return False
                width, height, depth, color, compression, filtering, interlace = struct.unpack(
                    ">IIBBBBB", payload
                )
                if (
                    width <= 0 or height <= 0 or height > _FRAME_HEIGHT
                    or (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0)
                ):
                    return False
            elif kind == b"IHDR":
                return False
            if kind == b"IDAT":
                pixels.extend(payload)
            if kind == b"IEND":
                if length != 0 or end != len(data) or not pixels:
                    return False
                decoded = zlib.decompress(pixels)
                stride = width * 3 + 1
                return (
                    len(decoded) == stride * height
                    and all(decoded[row * stride] <= 4 for row in range(height))
                )
            offset = end
    except (OSError, struct.error, zlib.error):
        return False
    return False


def _frame_source_digest(video: Path) -> str:
    """Read actual source bytes without stat-based digest memoization."""
    with video.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def scene_keyframes(
    ws: Workspace,
    video: VideoInfo,
    scene: Scene,
    *,
    timestamps_ms: list[int] | tuple[int, ...] | None = None,
    max_frames: int = 2,
) -> list[Path]:
    """Extract at most two real, scene-bounded RGB PNGs for text+image translation.

    Default sampling uses the midpoint followed by an early endpoint alternative,
    inset by up to 500 ms. Explicit timestamps are deduplicated in caller order and
    must lie in [start_ms, end_ms). The selected proxy is never silently replaced by
    the original: set ``video.proxy_path=None`` to select the original explicitly.
    Cache keys include the actual source-byte SHA256, path/size/mtime, timestamp,
    scene end bound and output geometry/format. Invalid images are re-extracted;
    extraction failures raise ``MediaError`` rather than returning partial paths.
    Source bytes are hashed afresh on every call and again before publishing new
    frames; unchanged size/mtime alone never permits reuse after a byte change.
    """
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    if not (0 <= scene.start_ms < scene.end_ms <= video.duration_ms):
        raise ValueError("scene bounds must lie within the positive video duration")
    limit = min(max_frames, 2)
    span = scene.end_ms - scene.start_ms
    candidates = timestamps_ms if timestamps_ms is not None else [
        scene.start_ms + span // 2,
        scene.start_ms + min(500, span // 4),
    ]
    timestamps: list[int] = []
    for timestamp in candidates:
        if (
            isinstance(timestamp, bool) or not isinstance(timestamp, int)
            or not scene.start_ms <= timestamp < scene.end_ms
        ):
            raise ValueError("frame timestamps must be integer milliseconds inside the scene")
        if timestamp not in timestamps:
            timestamps.append(timestamp)
    if not timestamps:
        raise ValueError("at least one frame timestamp is required")
    timestamps = timestamps[:limit]
    selected = video.proxy_path if video.proxy_path is not None else video.video_path
    if not selected:
        raise MediaError("no video path selected for scene keyframes")
    try:
        video = Path(selected).expanduser().resolve(strict=True)
        if not video.is_file():
            raise MediaError(f"keyframe source is not a file: {video}")
        source_stat = video.stat()
    except OSError as exc:
        raise MediaError(f"keyframe source unavailable: {selected}: {exc}") from exc
    try:
        source_sha256 = _frame_source_digest(video)
        after_hash = video.stat()
        if (after_hash.st_size, after_hash.st_mtime_ns) != (
            source_stat.st_size, source_stat.st_mtime_ns
        ):
            raise MediaError(f"keyframe source changed while reading identity: {video}")
    except OSError as exc:
        raise MediaError(f"keyframe source identity unavailable: {video}: {exc}") from exc
    source_identity = (str(video), source_stat.st_size, source_stat.st_mtime_ns, source_sha256)
    frames: list[Path] = []
    ff = ffmpeg_bin()
    for timestamp in timestamps:
        identity = [
            "scene-keyframes-v2", *source_identity, timestamp, scene.end_ms,
            _FRAME_SW_FILTER, _FRAME_RK_FILTER, "png",
        ]
        digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        out = ws.cache(f"frames-v2/{timestamp:010d}-{digest}.png")
        if _valid_frame(out):
            frames.append(out)
            continue
        fd, temporary = tempfile.mkstemp(
            prefix=f".{out.stem}.", suffix=".partial.png", dir=out.parent
        )
        os.close(fd)
        tmp = Path(temporary)
        try:
            head = [ff, "-nostdin", "-v", "error", "-y"]
            inputs = ["-ss", f"{timestamp / 1000:.3f}", "-i", str(video)]
            output_args = [
                "-t", f"{(scene.end_ms - timestamp) / 1000:.3f}",
                "-map", "0:v:0", "-frames:v", "1", "-an", "-sn", "-dn",
            ]
            tail = ["-c:v", "png", "-pix_fmt", "rgb24", "-f", "image2", str(tmp)]
            hw_cmd = [
                *head, *RK_DECODE_ARGS, *inputs, *output_args,
                "-vf", _FRAME_RK_FILTER, *tail,
            ] if has_rkmpp(ff) else None
            sw_cmd = [
                *head, *_hwaccel_args(), *inputs, *output_args,
                "-vf", _FRAME_SW_FILTER, *tail,
            ]
            _run_hw_or_sw(hw_cmd, sw_cmd, what=f"scene keyframe {scene.id} at {timestamp}ms")
            if not _valid_frame(tmp):
                raise MediaError(f"ffmpeg produced no valid PNG for {scene.id} at {timestamp}ms")
            current_stat = video.stat()
            if (current_stat.st_size, current_stat.st_mtime_ns) != source_identity[1:3]:
                raise MediaError(f"keyframe source changed during extraction: {video}")
            if _frame_source_digest(video) != source_sha256:
                raise MediaError(f"keyframe source bytes changed during extraction: {video}")
            tmp.replace(out)
        except OSError as exc:
            raise MediaError(f"scene keyframe extraction failed for {video}: {exc}") from exc
        finally:
            tmp.unlink(missing_ok=True)
        frames.append(out)
    return frames


def slice_proxy_clip(ws: Workspace, video: Path, start_s: float, end_s: float) -> Path:
    """Slice [start_s, end_s] from a low-res proxy MP4 into a cached clip file for Vertex AI."""
    video = Path(video).expanduser().resolve(strict=True)
    start = max(0.0, float(start_s))
    end = max(start + 0.1, float(end_s))
    st = video.stat()
    key = hashlib.sha256(
        f"vertex-clip-v2-tc|{video}|{st.st_size}|{st.st_mtime_ns}|{start:.3f}|{end:.3f}".encode()
    ).hexdigest()[:24]
    out = ws.cache(f"vertex-clips/{key}.mp4")
    if out.is_file() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = _partial(out)
    ff = ffmpeg_bin()
    base_cmd = [
        ff, "-nostdin", "-v", "error", "-y",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-copyts",
        "-i", str(video),
    ]
    tail_cmd = [
        "-c:v", "libx264", "-crf", "32", "-preset", "ultrafast",
        "-c:a", "aac", "-ac", "1", "-ar", "48000",
        "-movflags", "+faststart",
    ]
    tc_vf = ["-vf", r"drawtext=text='%{pts\:hms}':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.65"]
    try:
        _run([*base_cmd, *tc_vf, *tail_cmd, str(tmp)], what=f"slice_proxy_clip (timecode) {video.name} {start:.1f}-{end:.1f}s")
    except MediaError as exc:
        logger.warning(
            "Timecode burn-in failed for %s %.1f-%.1fs; using an unburned .notc.mp4 clip: %s",
            video.name, start, end, exc,
        )
        tmp.unlink(missing_ok=True)
        out = out.with_suffix(".notc.mp4")
        if out.is_file() and out.stat().st_size > 0:
            return out
        tmp = _partial(out)
        _run([
            ff, "-nostdin", "-v", "error", "-y",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-i", str(video), *tail_cmd, str(tmp),
        ], what=f"slice_proxy_clip {video.name} {start:.1f}-{end:.1f}s")
    tmp.replace(out)
    return out


def snap_ost_shot(
    video: Path, obs_start_ms: int, obs_end_ms: int, *, frame_exact: bool = False
) -> tuple[int, int]:
    """Snap a coarse VLM on-screen-text interval ``[obs_start_ms, obs_end_ms]`` to
    ``[t_settle, c_out]`` using ``ffmpeg scdet`` shot cuts and ``numpy`` 3x3
    Laplacian sharpness / inter-frame motion settle detection."""
    if frame_exact:
        return obs_start_ms, obs_end_ms
    video = Path(video).expanduser()
    if not video.is_file():
        return obs_start_ms, obs_end_ms
    try:
        import numpy as np
    except ImportError:
        return obs_start_ms, obs_end_ms

    pad_before = 2500
    pad_after = 2500
    w_start_s = max(0.0, (obs_start_ms - pad_before) / 1000.0)
    w_dur_s = max(0.5, (obs_end_ms + pad_after) / 1000.0 - w_start_s)
    ff = ffmpeg_bin()
    try:
        res = subprocess.run(
            [
                ff, "-nostdin", "-hide_banner", "-ss", f"{w_start_s:.3f}", "-t", f"{w_dur_s:.3f}",
                "-i", str(video),
                "-vf", "scale=320:180,scdet=threshold=4.5",
                "-an", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return obs_start_ms, obs_end_ms
    cuts: list[tuple[int, float]] = []
    for m in re.finditer(
        r"lavfi\.scd\.score:\s*([0-9.]+),\s*lavfi\.scd\.time:\s*([0-9.]+)",
        (res.stdout or "") + "\n" + (res.stderr or ""),
    ):
        score = float(m.group(1))
        t_ms = int(round((w_start_s + float(m.group(2))) * 1000))
        cuts.append((t_ms, score))

    if not cuts:
        return obs_start_ms, obs_end_ms
    mid_ms = (obs_start_ms + obs_end_ms) // 2
    before_cuts = [t for t, _ in cuts if t <= mid_ms]
    after_cuts = [t for t, _ in cuts if t > mid_ms]
    c_in = max(before_cuts) if before_cuts else obs_start_ms
    c_out = min(after_cuts) if after_cuts else obs_end_ms
    if c_out - c_in < 700 or c_out - c_in > 9000:
        return obs_start_ms, obs_end_ms

    s_in = c_in / 1000.0
    s_dur = max(0.5, (c_out - c_in) / 1000.0)
    W, H = 160, 90
    try:
        raw_res = subprocess.run(
            [
                ff, "-nostdin", "-hide_banner", "-ss", f"{s_in:.3f}", "-t", f"{s_dur:.3f}",
                "-i", str(video),
                "-vf", f"fps=10,scale={W}:{H},format=gray",
                "-an", "-f", "rawvideo", "-",
            ],
            capture_output=True, timeout=15, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return c_in, c_out
    buf = np.frombuffer(raw_res.stdout, dtype=np.uint8)
    n_frames = len(buf) // (W * H)
    if n_frames < 4:
        return c_in, c_out

    frames = buf[: n_frames * W * H].reshape((n_frames, H, W)).astype(np.float32)
    x0, x1 = int(W * 0.2), int(W * 0.8)
    y0, y1 = int(H * 0.25), int(H * 0.75)
    crop = frames[:, y0:y1, x0:x1]
    lap = (
        crop[:, 1:-1, :-2] + crop[:, 1:-1, 2:] + crop[:, :-2, 1:-1] + crop[:, 2:, 1:-1]
        - 4.0 * crop[:, 1:-1, 1:-1]
    )
    sharp = np.var(lap, axis=(1, 2))
    motion = np.zeros(n_frames, dtype=np.float32)
    motion[1:] = np.mean(np.abs(crop[1:] - crop[:-1]), axis=(1, 2))

    half = max(2, n_frames // 2)
    early_motion = float(np.max(motion[1 : min(18, n_frames)])) if n_frames > 2 else 0.0
    late_motion = float(np.median(motion[half:]))
    plateau_sharp = float(np.percentile(sharp[half:], 75))

    t_settle = max(obs_start_ms, c_in) if abs(obs_start_ms - c_in) <= 600 else c_in
    if early_motion > 7.5 and late_motion < early_motion * 0.65:
        thresh_m = max(6.8, late_motion * 1.6)
        thresh_s = 0.65 * plateau_sharp
        for i in range(1, n_frames - 1):
            if motion[i] <= thresh_m and motion[i + 1] <= thresh_m and sharp[i] >= thresh_s:
                cand_settle = c_in + i * 100
                if c_out - cand_settle >= 800:
                    t_settle = cand_settle
                break
    return t_settle, c_out
