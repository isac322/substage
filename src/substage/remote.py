"""Build the Gemini A/V proxy on this machine for a video that lives on another host.

The pipeline runs where the media library is. With jellyfin-ffmpeg installed as
``ffmpeg-rk`` on a Rockchip Linux host, the host decodes 4K HEVC in hardware (rkmpp,
~5x realtime) and builds the proxy itself; with only stock ffmpeg it decodes in
software. This is the offload path for that case: it streams the file over ssh,
encodes the proxy locally (e.g. VideoToolbox on macOS) with the exact pipeline proxy
settings, and installs it into the host's substage cache so `substage run` on the host
finds it up to date.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from . import media
from .config import workspace


def _ssh(host: str, command: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", host, command], check=True, **kw)


def remote_proxy(host: str, remote_video: str, video_id: str) -> str:
    """Encode the proxy for *remote_video* on *host* and install it there; returns the remote path."""
    q = shlex.quote(remote_video)
    probed = _ssh(host, f"ffprobe {' '.join(media.PROBE_ARGS)} {q}", capture_output=True, text=True)
    streams, _ = media.parse_probe(probed.stdout, f"{host}:{remote_video}")
    audio = media.pick_dialogue_audio(streams)

    local = workspace(video_id).cache("proxy.mp4")
    tmp = local.with_name("proxy.partial.mp4")
    cat = subprocess.Popen(["ssh", host, f"cat {q}"], stdout=subprocess.PIPE)
    try:
        subprocess.run(
            [
                media.ffmpeg_bin(), "-v", "error", "-y", *media._hwaccel_args(), "-i", "pipe:0",
                "-map", "0:v:0", "-map", f"0:{audio.index}",
                *media.PROXY_VIDEO_ARGS, *media.PROXY_AUDIO_ARGS,
                str(tmp),
            ],
            stdin=cat.stdout,
            check=True,
        )
    finally:
        assert cat.stdout is not None
        cat.stdout.close()
        cat.wait()
    if cat.returncode not in (0, None):
        raise media.MediaError(f"streaming {host}:{remote_video} failed (ssh exit {cat.returncode})")
    tmp.replace(local)

    remote_dir = f'"${{XDG_CACHE_HOME:-$HOME/.cache}}/substage/{video_id}"'
    with Path(local).open("rb") as fh:
        _ssh(
            host,
            f"d={remote_dir}; mkdir -p \"$d\" && cat > \"$d/proxy.partial.mp4\" && mv \"$d/proxy.partial.mp4\" \"$d/proxy.mp4\" && echo \"$d/proxy.mp4\"",
            stdin=fh,
            capture_output=True,
            text=True,
        )
    return f"{host}:~/.cache/substage/{video_id}/proxy.mp4"
