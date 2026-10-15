"""Offline scene-frame extraction regressions using real RGB PNG bytes and fake ffmpeg."""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

from substage import media
from substage.config import Workspace
from substage.models import VideoInfo, Scene


def _png(rgb: bytes) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00" + rgb))
        + chunk(b"IEND", b"")
    )


def _pixels(path: Path) -> bytes:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">IIBBBBB", data[16:29]) == (1, 1, 8, 2, 0, 0, 0)
    offset = 8
    compressed = bytearray()
    while offset < len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        if data[offset + 4:offset + 8] == b"IDAT":
            compressed.extend(data[offset + 8:offset + 8 + size])
        offset += size + 12
    return zlib.decompress(compressed)[1:]


class FakeFFmpeg:
    def __init__(
        self, *, rkmpp: bool = False, fail_hw: bool = False,
        output: str = "png", mutate_source: bool = False,
    ):
        self.rkmpp = rkmpp
        self.fail_hw = fail_hw
        self.output = output
        self.mutate_source = mutate_source
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        if "-hwaccels" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="rkmpp\n" if self.rkmpp else "", stderr=""
            )
        out = Path(cmd[-1])
        assert out.name.endswith(".partial.png")
        assert out.is_file()  # unique sibling created before ffmpeg runs
        if self.fail_hw and "rkmpp" in cmd:
            out.write_bytes(b"partial failed hardware output")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="decoder failed")
        if self.output == "failure":
            out.write_bytes(b"failed output")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no video stream")
        source = Path(cmd[cmd.index("-i") + 1])
        timestamp = cmd[cmd.index("-ss") + 1]
        if self.output == "png":
            color = hashlib.sha256(source.read_bytes() + timestamp.encode()).digest()[:3]
            out.write_bytes(_png(color))
        elif self.output == "corrupt":
            out.write_bytes(b"\x89PNG\r\n\x1a\nnot a PNG")
        # "missing" leaves the empty mkstemp file in place.
        if self.mutate_source:
            source.write_bytes(b"changed during decode")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    @property
    def extracts(self) -> list[list[str]]:
        return [cmd for cmd in self.calls if "-hwaccels" not in cmd]


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Workspace, VideoInfo, Scene]:
    source = tmp_path / "video.mkv"
    source.write_bytes(b"original video bytes")
    ws = Workspace("video", tmp_path / "data", tmp_path / "cache", tmp_path / "logs")
    video = VideoInfo(
        video_id="video", title="Video", video_path=str(source), duration_ms=20_000,
        audio_stream=1, en_sub_path="en.srt", en_sub_origin="user",
    )
    scene = Scene(id="s001", start_ms=1_000, end_ms=9_000, cue_ids=["c001"])
    return ws, video, scene


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    media.has_rkmpp.cache_clear()
    monkeypatch.setattr(media, "ffmpeg_bin", lambda: "ffmpeg-selected")
    monkeypatch.setattr(media.sys, "platform", "linux")
    yield
    media.has_rkmpp.cache_clear()


def _install(monkeypatch, **kwargs) -> FakeFFmpeg:
    fake = FakeFFmpeg(**kwargs)
    monkeypatch.setattr(media.subprocess, "run", fake)
    return fake


def test_default_frames_have_image_bytes_bound_timestamps_and_source_provenance(
    monkeypatch, inputs,
):
    ws, video, scene = inputs
    fake = _install(monkeypatch)
    original = Path(video.video_path).read_bytes()
    frames = media.scene_keyframes(ws, video, scene)

    assert len(frames) == 2
    assert [cmd[cmd.index("-ss") + 1] for cmd in fake.extracts] == ["5.000", "1.500"]
    for frame, cmd in zip(frames, fake.extracts, strict=True):
        stamp = cmd[cmd.index("-ss") + 1]
        assert _pixels(frame) == hashlib.sha256(original + stamp.encode()).digest()[:3]
        assert cmd[0] == "ffmpeg-selected"
        assert cmd[cmd.index("-i") + 1] == str(Path(video.video_path).resolve())
        assert cmd[cmd.index("-frames:v") + 1] == "1"
        assert cmd[cmd.index("-map") + 1] == "0:v:0"
        assert "-an" in cmd and "-sn" in cmd and "-dn" in cmd
        assert cmd[cmd.index("-vf") + 1] == "scale=-2:360,format=rgb24"
        assert float(stamp) + float(cmd[cmd.index("-t") + 1]) == scene.end_ms / 1000
        assert frame.parent == ws.cache_dir / "frames-v2"
    assert Path(video.video_path).read_bytes() == original
    assert not list((ws.cache_dir / "frames-v2").glob("*.partial.png"))


def test_prefers_selected_proxy_and_original_remains_untouched(monkeypatch, inputs):
    ws, video, scene = inputs
    proxy = Path(video.video_path).with_suffix(".proxy.mp4")
    proxy.write_bytes(b"proxy video bytes")
    video = video.model_copy(update={"proxy_path": str(proxy)})
    fake = _install(monkeypatch)

    frame = media.scene_keyframes(ws, video, scene, max_frames=1)[0]
    cmd = fake.extracts[0]
    assert cmd[cmd.index("-i") + 1] == str(proxy.resolve())
    assert _pixels(frame) == hashlib.sha256(b"proxy video bytes5.000").digest()[:3]
    assert proxy.read_bytes() == b"proxy video bytes"
    assert Path(video.video_path).read_bytes() == b"original video bytes"


def test_configured_missing_proxy_does_not_silently_select_original(monkeypatch, inputs):
    ws, video, scene = inputs
    fake = _install(monkeypatch)
    video = video.model_copy(update={"proxy_path": str(ws.data_dir / "missing.mp4")})
    with pytest.raises(media.MediaError, match="source unavailable"):
        media.scene_keyframes(ws, video, scene)
    assert fake.calls == []


def test_rkmpp_downloads_nv12_before_rgb_png_encoding(monkeypatch, inputs):
    fake = _install(monkeypatch, rkmpp=True)
    frames = media.scene_keyframes(*inputs, max_frames=1)
    assert len(frames) == 1 and len(_pixels(frames[0])) == 3
    cmd = fake.extracts[0]
    assert "rkmpp" in cmd and "drm_prime" in cmd
    assert cmd[cmd.index("-vf") + 1] == (
        "scale_rkrga=w=-2:h=360:format=nv12,hwdownload,format=nv12,format=rgb24"
    )
    assert cmd[cmd.index("-c:v") + 1] == "png"
    assert cmd[cmd.index("-pix_fmt") + 1] == "rgb24"


def test_failed_hardware_decode_uses_software_and_commits_valid_image(monkeypatch, inputs):
    fake = _install(monkeypatch, rkmpp=True, fail_hw=True)
    frames = media.scene_keyframes(*inputs, max_frames=1)
    assert len(fake.extracts) == 2
    assert "rkmpp" in fake.extracts[0]
    assert "rkmpp" not in fake.extracts[1]
    assert fake.extracts[0][-1] == fake.extracts[1][-1]
    assert len(_pixels(frames[0])) == 3


def test_valid_cache_reuse_does_not_run_another_extract(monkeypatch, inputs):
    fake = _install(monkeypatch)
    first = media.scene_keyframes(*inputs)
    call_count = len(fake.calls)
    second = media.scene_keyframes(*inputs)
    assert second == first
    assert len(fake.calls) == call_count
    assert [_pixels(frame) for frame in second] == [_pixels(frame) for frame in first]


@pytest.mark.parametrize("identity_part", ["size", "mtime", "path", "timestamp", "end"])
def test_cache_invalidates_exact_source_and_time_identity(monkeypatch, inputs, identity_part):
    ws, video, scene = inputs
    fake = _install(monkeypatch)
    first = media.scene_keyframes(ws, video, scene, timestamps_ms=[2_000])[0]
    source = Path(video.video_path)
    times = [2_000]
    if identity_part == "size":
        source.write_bytes(b"a differently sized source")
    elif identity_part == "mtime":
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    elif identity_part == "path":
        other = source.with_name("other.mkv")
        other.write_bytes(source.read_bytes())
        stat = source.stat()
        os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        video = video.model_copy(update={"video_path": str(other)})
    elif identity_part == "timestamp":
        times = [2_001]
    else:
        scene = scene.model_copy(update={"end_ms": 8_000})
    second = media.scene_keyframes(ws, video, scene, timestamps_ms=times)[0]
    assert first != second
    assert len(fake.extracts) == 2
    assert len(_pixels(first)) == len(_pixels(second)) == 3
    if identity_part in {"size", "timestamp"}:
        assert _pixels(first) != _pixels(second)


@pytest.mark.parametrize("corruption", ["text", "crc", "missing"])
def test_corrupt_or_missing_cache_is_reextracted(monkeypatch, inputs, corruption):
    fake = _install(monkeypatch)
    frame = media.scene_keyframes(*inputs, max_frames=1)[0]
    expected = _pixels(frame)
    if corruption == "missing":
        frame.unlink()
    elif corruption == "text":
        frame.write_text("not an image")
    else:
        broken = bytearray(frame.read_bytes())
        broken[-1] ^= 1
        frame.write_bytes(broken)
    assert media.scene_keyframes(*inputs, max_frames=1) == [frame]
    assert len(fake.extracts) == 2
    assert _pixels(frame) == expected


def test_atomic_swap_only_exposes_complete_final_image(monkeypatch, inputs):
    ws, _, _ = inputs
    _install(monkeypatch)
    original_replace = Path.replace
    swaps: list[tuple[Path, Path]] = []

    def replace(path: Path, target: Path) -> Path:
        assert path.name.endswith(".partial.png")
        assert not target.exists()
        assert len(_pixels(path)) == 3
        swaps.append((path, target))
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    frames = media.scene_keyframes(*inputs)
    assert [target for _, target in swaps] == frames
    assert all(not temporary.exists() for temporary, _ in swaps)
    assert sorted((ws.cache_dir / "frames-v2").iterdir()) == sorted(frames)


@pytest.mark.parametrize("output", ["failure", "missing", "corrupt"])
def test_no_fabricated_paths_or_committed_images_on_extract_failure(monkeypatch, inputs, output):
    ws, video, _ = inputs
    _install(monkeypatch, output=output)
    with pytest.raises(media.MediaError):
        media.scene_keyframes(*inputs)
    assert not list((ws.cache_dir / "frames-v2").iterdir())
    assert Path(video.video_path).read_bytes() == b"original video bytes"


def test_changed_source_during_decode_is_not_committed_under_old_identity(monkeypatch, inputs):
    ws, _, _ = inputs
    _install(monkeypatch, mutate_source=True)
    with pytest.raises(media.MediaError, match="source changed"):
        media.scene_keyframes(*inputs)
    assert not list((ws.cache_dir / "frames-v2").iterdir())


def test_missing_binary_is_visible_and_partial_is_removed(monkeypatch, inputs):
    ws, _, _ = inputs

    def missing(cmd, **kwargs):
        raise FileNotFoundError("ffmpeg-selected")

    monkeypatch.setattr(media.subprocess, "run", missing)
    with pytest.raises(media.MediaError, match="extraction failed"):
        media.scene_keyframes(*inputs)
    assert not list((ws.cache_dir / "frames-v2").iterdir())


def test_explicit_times_preserve_order_deduplicate_and_hard_cap_two(monkeypatch, inputs):
    fake = _install(monkeypatch)
    frames = media.scene_keyframes(
        *inputs, timestamps_ms=[7_000, 7_000, 1_000, 8_999], max_frames=100,
    )
    assert len(frames) == 2
    assert [cmd[cmd.index("-ss") + 1] for cmd in fake.extracts] == ["7.000", "1.000"]


def test_one_millisecond_scene_deduplicates_default_timestamps(monkeypatch, inputs):
    ws, video, scene = inputs
    scene = scene.model_copy(update={"start_ms": 1_000, "end_ms": 1_001})
    fake = _install(monkeypatch)
    frames = media.scene_keyframes(ws, video, scene)
    assert len(frames) == 1
    assert fake.extracts[0][fake.extracts[0].index("-t") + 1] == "0.001"
    assert len(_pixels(frames[0])) == 3


@pytest.mark.parametrize("max_frames", [0, -1, True, 1.5])
def test_invalid_frame_limits_fail_before_subprocess(monkeypatch, inputs, max_frames):
    fake = _install(monkeypatch)
    with pytest.raises(ValueError, match="max_frames"):
        media.scene_keyframes(*inputs, max_frames=max_frames)
    assert fake.calls == []


@pytest.mark.parametrize("bounds", [(-1, 9_000), (1_000, 1_000), (9_000, 1_000), (1_000, 20_001)])
def test_invalid_scene_bounds_fail_before_subprocess(monkeypatch, inputs, bounds):
    ws, video, scene = inputs
    fake = _install(monkeypatch)
    scene = scene.model_copy(update={"start_ms": bounds[0], "end_ms": bounds[1]})
    with pytest.raises(ValueError, match="scene bounds"):
        media.scene_keyframes(ws, video, scene)
    assert fake.calls == []


@pytest.mark.parametrize("times", [[], [999], [9_000], [True], [2_000.5], [2_000, 3_000, 99_000]])
def test_invalid_explicit_timestamps_fail_before_subprocess(monkeypatch, inputs, times):
    fake = _install(monkeypatch)
    with pytest.raises(ValueError, match="timestamp"):
        media.scene_keyframes(*inputs, timestamps_ms=times)
    assert fake.calls == []


@pytest.mark.parametrize("source_kind", ["absent", "directory", "empty_path"])
def test_no_selected_video_file_is_visible(monkeypatch, inputs, source_kind):
    ws, video, scene = inputs
    source = Path(video.video_path)
    if source_kind == "absent":
        source.unlink()
    elif source_kind == "directory":
        source.unlink()
        source.mkdir()
    else:
        video = video.model_copy(update={"video_path": ""})
    fake = _install(monkeypatch)
    with pytest.raises(media.MediaError):
        media.scene_keyframes(ws, video, scene)
    assert fake.calls == []


@pytest.mark.parametrize("use_proxy", [False, True])
def test_same_size_restored_mtime_new_source_bytes_produce_new_frame(
    monkeypatch, inputs, use_proxy,
):
    ws, video, scene = inputs
    source = Path(video.video_path)
    if use_proxy:
        source = source.with_suffix(".proxy.mp4")
        source.write_bytes(b"proxy video bytes")
        video = video.model_copy(update={"proxy_path": str(source)})
    fake = _install(monkeypatch)
    first = media.scene_keyframes(ws, video, scene, max_frames=1)[0]
    first_pixels = _pixels(first)
    old_stat = source.stat()
    changed = bytes(value ^ 0x01 for value in source.read_bytes())
    source.write_bytes(changed)
    os.utime(source, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert source.stat().st_size == old_stat.st_size
    assert source.stat().st_mtime_ns == old_stat.st_mtime_ns

    second = media.scene_keyframes(ws, video, scene, max_frames=1)[0]
    assert second != first
    assert _pixels(second) == hashlib.sha256(changed + b"5.000").digest()[:3]
    assert _pixels(second) != first_pixels
    assert _pixels(first) == first_pixels  # old cache is not deleted or overwritten
    assert len(fake.extracts) == 2
    assert media.scene_keyframes(ws, video, scene, max_frames=1) == [second]
    assert len(fake.extracts) == 2


def test_v2_cache_does_not_reuse_or_delete_legacy_stat_only_images(monkeypatch, inputs):
    ws, video, scene = inputs
    legacy = ws.cache("frames/legacy-stat-only.png")
    legacy.write_bytes(_png(b"\x01\x02\x03"))
    legacy_bytes = legacy.read_bytes()
    fake = _install(monkeypatch)
    fresh = media.scene_keyframes(ws, video, scene, max_frames=1)[0]
    assert fresh.parent == ws.cache_dir / "frames-v2"
    assert len(fake.extracts) == 1
    assert legacy.read_bytes() == legacy_bytes
    assert _pixels(fresh) != _pixels(legacy)


def test_same_stat_source_mutation_during_decode_is_not_committed(monkeypatch, inputs):
    ws, video, _ = inputs
    fake = FakeFFmpeg()
    source = Path(video.video_path)
    original_stat = source.stat()

    def changing(cmd, **kwargs):
        result = fake(cmd, **kwargs)
        if "-i" in cmd:
            source.write_bytes(bytes(value ^ 0x01 for value in source.read_bytes()))
            os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        return result

    monkeypatch.setattr(media.subprocess, "run", changing)
    with pytest.raises(media.MediaError, match="source bytes changed"):
        media.scene_keyframes(*inputs)
    assert not list((ws.cache_dir / "frames-v2").iterdir())
