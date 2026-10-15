"""Offline tests for ffmpeg binary selection and the rkmpp hardware path (fake subprocess)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from substage import config, media
from substage.config import Settings
from substage.media import Stream

AUDIO = Stream(
    index=1, codec_type="audio", codec_name="truehd", language="eng", title=None,
    channels=8, forced=False, hearing_impaired=False,
)


def _clear_caches() -> None:
    media.ffmpeg_bin.cache_clear()
    media.ffprobe_bin.cache_clear()
    media.has_rkmpp.cache_clear()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    _clear_caches()
    monkeypatch.setattr(media.sys, "platform", "linux")
    yield
    _clear_caches()


class FakeFFmpeg:
    """Stands in for subprocess.run: answers -hwaccels, records encodes, writes their output file."""

    def __init__(self, hwaccels: str, fail_hw: bool = False):
        self.hwaccels = hwaccels
        self.fail_hw = fail_hw
        self.calls: list[list[str]] = []

    def __call__(self, cmd, capture_output=False, text=False):
        self.calls.append(list(cmd))
        if "-hwaccels" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=self.hwaccels, stderr="")
        if self.fail_hw and "rkmpp" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="hevc_rkmpp: init failed")
        Path(cmd[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    @property
    def encodes(self) -> list[list[str]]:
        return [c for c in self.calls if "-hwaccels" not in c]


RK_HWACCELS = "Hardware acceleration methods:\ncuda\ndrm\nopencl\nrkmpp\n"
STOCK_HWACCELS = "Hardware acceleration methods:\nvdpau\ncuda\nvaapi\ndrm\nopencl\nvulkan\n"


def _use(monkeypatch, fake: FakeFFmpeg, ffmpeg: str = "ffmpeg-rk") -> None:
    monkeypatch.setattr(media, "load_settings", lambda: Settings(ffmpeg_bin=ffmpeg, ffprobe_bin="ffprobe-rk"))
    monkeypatch.setattr(media.subprocess, "run", fake)


def _video(tmp_path: Path) -> Path:
    v = tmp_path / "video.mkv"
    v.write_bytes(b"src")
    return v


# --- binary defaults -------------------------------------------------------


@pytest.mark.parametrize(
    ("platform", "on_path", "expected"),
    [
        ("linux", {"ffmpeg-rk", "ffprobe-rk"}, ("ffmpeg-rk", "ffprobe-rk")),
        ("linux", set(), ("ffmpeg", "ffprobe")),
        ("darwin", {"ffmpeg-rk", "ffprobe-rk"}, ("ffmpeg", "ffprobe")),
    ],
)
def test_settings_default_bins(monkeypatch, platform, on_path, expected):
    monkeypatch.setattr(config.sys, "platform", platform)
    monkeypatch.setattr(config.shutil, "which", lambda name: f"/x/{name}" if name in on_path else None)
    s = Settings()
    assert (s.ffmpeg_bin, s.ffprobe_bin) == expected


def test_media_bins_come_from_settings(monkeypatch):
    _use(monkeypatch, FakeFFmpeg(RK_HWACCELS), ffmpeg="/opt/ff")
    assert media.ffmpeg_bin() == "/opt/ff"
    assert media.ffprobe_bin() == "ffprobe-rk"


# --- rkmpp detection -------------------------------------------------------


def test_has_rkmpp_detects_and_caches(monkeypatch):
    fake = FakeFFmpeg(RK_HWACCELS)
    monkeypatch.setattr(media.subprocess, "run", fake)
    assert media.has_rkmpp("ffmpeg-rk") is True
    assert media.has_rkmpp("ffmpeg-rk") is True
    assert fake.calls == [["ffmpeg-rk", "-hide_banner", "-hwaccels"]]


def test_has_rkmpp_false_for_stock_ffmpeg(monkeypatch):
    monkeypatch.setattr(media.subprocess, "run", FakeFFmpeg(STOCK_HWACCELS))
    assert media.has_rkmpp("ffmpeg") is False


def test_has_rkmpp_false_when_binary_missing_or_failing(monkeypatch):
    def missing(cmd, **kw):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(media.subprocess, "run", missing)
    assert media.has_rkmpp("nope") is False

    monkeypatch.setattr(
        media.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="rkmpp", stderr=""),
    )
    assert media.has_rkmpp("broken") is False


# --- make_proxy ------------------------------------------------------------


def test_make_proxy_uses_rkmpp_path(monkeypatch, tmp_path):
    fake = FakeFFmpeg(RK_HWACCELS)
    _use(monkeypatch, fake)
    out = media.make_proxy(_video(tmp_path), AUDIO, tmp_path / "proxy.mp4")

    assert out.read_bytes() == b"mp4"
    assert not (tmp_path / "proxy.partial.mp4").exists()
    [cmd] = fake.encodes
    assert cmd[0] == "ffmpeg-rk"
    assert cmd[cmd.index("-hwaccel") + 1] == "rkmpp"
    assert cmd[cmd.index("-hwaccel_output_format") + 1] == "drm_prime"
    vf = cmd[cmd.index("-vf") + 1]
    assert "fps=2" in vf and "scale_rkrga=w=-2:h=360:format=nv12" in vf and "hwdownload" in vf
    # proxy spec unchanged: libx264 crf 32, mono 48 kHz AAC
    assert cmd[cmd.index("-c:v") + 1] == "libx264" and cmd[cmd.index("-crf") + 1] == "32"
    assert cmd[cmd.index("-ac") + 1] == "1" and cmd[cmd.index("-ar") + 1] == "48000"
    assert cmd[-1] == str(tmp_path / "proxy.partial.mp4")


def test_make_proxy_software_without_rkmpp(monkeypatch, tmp_path):
    fake = FakeFFmpeg(STOCK_HWACCELS)
    _use(monkeypatch, fake, ffmpeg="ffmpeg")
    media.make_proxy(_video(tmp_path), AUDIO, tmp_path / "proxy.mp4")

    [cmd] = fake.encodes
    assert "rkmpp" not in cmd and "-hwaccel" not in cmd
    assert cmd[cmd.index("-vf") + 1] == "scale=-2:360,fps=2"


def test_make_proxy_falls_back_to_software_when_hw_fails(monkeypatch, tmp_path):
    fake = FakeFFmpeg(RK_HWACCELS, fail_hw=True)
    _use(monkeypatch, fake)
    out = media.make_proxy(_video(tmp_path), AUDIO, tmp_path / "proxy.mp4")

    hw, sw = fake.encodes
    assert "rkmpp" in hw
    assert sw[0] == "ffmpeg-rk" and "rkmpp" not in sw
    assert sw[sw.index("-vf") + 1] == "scale=-2:360,fps=2"
    assert out.exists()


# --- make_review_video -----------------------------------------------------


def test_make_review_video_hw(monkeypatch, tmp_path):
    fake = FakeFFmpeg(RK_HWACCELS)
    _use(monkeypatch, fake)
    out = media.make_review_video(_video(tmp_path), AUDIO, tmp_path / "review.mp4")

    assert out.read_bytes() == b"mp4"
    [cmd] = fake.encodes
    assert cmd[cmd.index("-hwaccel") + 1] == "rkmpp"
    assert cmd[cmd.index("-vf") + 1] == "scale_rkrga=w=-2:h=720:format=nv12"
    assert cmd[cmd.index("-c:v") + 1] == "h264_rkmpp"
    assert ["-map", "0:1"] == cmd[cmd.index("0:v:0") + 1 : cmd.index("0:v:0") + 3]
    assert cmd[cmd.index("-ac") + 1] == "2" and cmd[cmd.index("-b:a") + 1] == "128k"
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert "-r" not in cmd  # source frame timing kept
    assert cmd[-1].endswith("review.partial.mp4")


def test_make_review_video_sw(monkeypatch, tmp_path):
    fake = FakeFFmpeg(STOCK_HWACCELS)
    _use(monkeypatch, fake, ffmpeg="ffmpeg")
    media.make_review_video(_video(tmp_path), AUDIO, tmp_path / "review.mp4")

    [cmd] = fake.encodes
    assert "-hwaccel" not in cmd
    assert cmd[cmd.index("-vf") + 1] == "scale=-2:720"
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-preset") + 1] == "veryfast" and cmd[cmd.index("-crf") + 1] == "26"
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"


def test_make_review_video_skips_when_up_to_date(monkeypatch, tmp_path):
    fake = FakeFFmpeg(RK_HWACCELS)
    _use(monkeypatch, fake)
    video = _video(tmp_path)
    out = tmp_path / "review.mp4"
    out.write_bytes(b"done")
    media.make_review_video(video, AUDIO, out)
    assert fake.calls == []
