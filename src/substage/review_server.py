"""Review web app: watch a video with EN / official KO / substage KO subtitles side by
side, browse a cue table, and leave timestamped notes for fix cycles.

Data root (XDG): ``$XDG_DATA_HOME/substage-review/<video_id>/`` containing
  meta.json           {"title", "year"}
  video.mp4           720p review video (optional → text-only mode)
  en.srt              English subtitles (UTF-8 SRT)
  official.ko.srt     official Korean subtitles, normalized to UTF-8 SRT
  ko.<arm>.srt        one per substage candidate arm
  refeval.json        optional reference evaluation (groups / judged scenes)
  notes.json          reviewer notes appended by the web app

The server is stdlib only (``ThreadingHTTPServer``); ``prepare_video`` uses
pysubs2 to normalize subtitles (SAMI cp949 → UTF-8 SRT). Official subtitles
are evaluation material only and never flow back into the pipeline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pysubs2

from .config import _xdg

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ARM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_CANDIDATE_RE = re.compile(r"^ko\.([A-Za-z0-9][A-Za-z0-9_-]{0,31})\.srt$")
_STATIC = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8",
           "style.css": "text/css; charset=utf-8"}
_ENCODINGS = ("utf-8-sig", "cp949")
_CHUNK = 1 << 20
_MAX_BODY = 64 * 1024
_SAMI_MAX_HOLD_MS = 20_000  # a SAMI cue without a closing &nbsp; holds until the next SYNC, capped

_notes_lock = threading.Lock()


def default_root() -> Path:
    """``$XDG_DATA_HOME/substage-review`` (``~/.local/share/substage-review``)."""
    return _xdg("XDG_DATA_HOME", ".local/share") / "substage-review"


# ---------------------------------------------------------------- subtitles


def _load_subs(path: Path) -> pysubs2.SSAFile:
    last: Exception | None = None
    for enc in _ENCODINGS:
        try:
            subs = pysubs2.load(str(path), encoding=enc)
        except UnicodeDecodeError as e:
            last = e
            continue
        if subs.format == "sami":
            # pysubs2 guesses SAMI end times from text length; SAMI semantics are
            # "shown until the next SYNC" (an &nbsp; SYNC clears the screen).
            ev = sorted(subs.events, key=lambda e: e.start)
            for cur, nxt in zip(ev, ev[1:]):
                cur.end = min(nxt.start, cur.start + _SAMI_MAX_HOLD_MS)
            subs.events = ev
        return subs
    raise ValueError(f"cannot decode {path} as {', '.join(_ENCODINGS)}") from last


def _normalize_to_srt(src: Path, dest: Path) -> int:
    """Load any pysubs2 format, drop blank events, write UTF-8 SRT; returns cue count."""
    subs = _load_subs(src)
    subs.events = sorted(
        (e for e in subs.events if not e.is_comment and e.plaintext.replace("\u00a0", " ").strip()),
        key=lambda e: (e.start, e.end),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    subs.save(str(tmp), encoding="utf-8", format_="srt")
    os.replace(tmp, dest)
    return len(subs.events)


_cue_cache: dict[Path, tuple[float, list[dict]]] = {}
_cue_cache_lock = threading.Lock()


def load_cue_list(path: Path) -> list[dict]:
    """``[{"s": start_ms, "e": end_ms, "t": text}]``; line breaks kept as ``\\n``."""
    mtime = path.stat().st_mtime
    with _cue_cache_lock:
        hit = _cue_cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
    out = []
    for e in _load_subs(path).events:
        text = "\n".join(line.strip() for line in e.plaintext.replace("\u00a0", " ").splitlines()).strip()
        if e.is_comment or not text:
            continue
        out.append({"s": int(e.start), "e": int(max(e.end, e.start)), "t": text})
    out.sort(key=lambda c: (c["s"], c["e"]))
    with _cue_cache_lock:
        _cue_cache[path] = (mtime, out)
    return out


# ---------------------------------------------------------------- data root


def _video_dir(root: Path, video_id: str) -> Path | None:
    """The video folder, or None if the id is invalid or escapes the root."""
    if not _ID_RE.match(video_id) or ".." in video_id:
        return None
    base = root.resolve()
    d = (base / video_id).resolve()
    if d.parent != base or not d.is_dir():
        return None
    return d


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _arms(d: Path) -> list[str]:
    return sorted(m.group(1) for p in d.iterdir() if (m := _CANDIDATE_RE.match(p.name)))


def video_summary(d: Path) -> dict:
    meta = _read_json(d / "meta.json", {})
    return {
        "id": d.name,
        "title": meta.get("title") or d.name,
        "year": meta.get("year"),
        "has_video": (d / "video.mp4").is_file(),
        "arms": _arms(d),
        "has_official": (d / "official.ko.srt").is_file(),
        "has_refeval": (d / "refeval.json").is_file(),
        "n_notes": len(_read_json(d / "notes.json", [])),
    }


def list_videos(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    videos = [video_summary(d) for d in sorted(root.iterdir()) if d.is_dir() and _ID_RE.match(d.name)]
    return [f for f in videos if f["arms"] or f["has_official"] or (root / f["id"] / "en.srt").is_file()]


def video_detail(d: Path) -> dict:
    info = video_summary(d)

    def cues(name: str) -> list[dict]:
        p = d / name
        return load_cue_list(p) if p.is_file() else []

    info.update(
        video=f"/media/{d.name}/video.mp4" if info["has_video"] else None,
        en=cues("en.srt"),
        official=cues("official.ko.srt"),
        candidates={arm: cues(f"ko.{arm}.srt") for arm in info["arms"]},
        refeval=_read_json(d / "refeval.json", None),
        notes=_read_json(d / "notes.json", []),
    )
    return info


def append_note(d: Path, payload: Any) -> dict:
    """Validate and append one note to ``notes.json`` (atomic rewrite)."""
    if not isinstance(payload, dict):
        raise ValueError("body must be a JSON object")
    comment = payload.get("comment")
    if not isinstance(comment, str) or not comment.strip():
        raise ValueError("comment must be a non-empty string")
    time_ms = payload.get("time_ms")
    if isinstance(time_ms, bool) or not isinstance(time_ms, (int, float)) or time_ms < 0:
        raise ValueError("time_ms must be a non-negative number")
    arm = payload.get("arm")
    if arm is not None and not isinstance(arm, str):
        raise ValueError("arm must be a string")
    cue = payload.get("cue")
    if cue is not None and not isinstance(cue, (str, dict)):
        raise ValueError("cue must be a string or object")
    if isinstance(cue, dict):
        cue = {str(k): v for k, v in cue.items() if isinstance(v, (str, int, float)) or v is None}
    note = {
        "time_ms": int(time_ms),
        "cue": cue,
        "arm": arm,
        "comment": comment.strip(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = d / "notes.json"
    with _notes_lock:
        notes = _read_json(path, [])
        note["id"] = max((n.get("id", 0) for n in notes if isinstance(n.get("id"), int)), default=0) + 1
        notes.append(note)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".notes.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    return note


def delete_note(d: Path, note_id: int) -> bool:
    """Remove the note with ``note_id`` from ``notes.json`` (atomic rewrite)."""
    path = d / "notes.json"
    with _notes_lock:
        notes = _read_json(path, [])
        keep = [n for n in notes if not (isinstance(n, dict) and n.get("id") == note_id)]
        if len(keep) == len(notes):
            return False
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".notes.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(keep, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    return True


# ---------------------------------------------------------------- prepare


def _copy_if_changed(src: Path, dest: Path) -> None:
    s = src.stat()
    if dest.is_file():
        d = dest.stat()
        if d.st_size == s.st_size and int(d.st_mtime) == int(s.st_mtime):
            return
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)


def prepare_video(
    review_root: Path,
    video_id: str,
    *,
    title: str,
    year: int | None,
    en_srt: Path,
    official_path: Path | None,
    candidates: dict[str, Path],
    video: Path | None = None,
    refeval: Path | None = None,
) -> Path:
    """Copy/normalize one video's review material into ``review_root/video_id``.

    Subtitles are re-encoded to UTF-8 SRT (SAMI/cp949 included); the video is
    copied only when size or mtime changed; existing ``notes.json`` is kept.
    """
    if not _ID_RE.match(video_id) or ".." in video_id:
        raise ValueError(f"invalid video id {video_id!r}")
    bad = [a for a in candidates if not _ARM_RE.match(a)]
    if bad:
        raise ValueError(f"invalid arm names {bad}")
    d = Path(review_root) / video_id
    d.mkdir(parents=True, exist_ok=True)
    _normalize_to_srt(Path(en_srt), d / "en.srt")
    if official_path is not None:
        _normalize_to_srt(Path(official_path), d / "official.ko.srt")
    for arm, src in candidates.items():
        _normalize_to_srt(Path(src), d / f"ko.{arm}.srt")
    if video is not None:
        _copy_if_changed(Path(video), d / "video.mp4")
    if refeval is not None:
        _copy_if_changed(Path(refeval), d / "refeval.json")
    meta = {"title": title, "year": year, "video_id": video_id,
            "prepared_at": datetime.now(UTC).isoformat(timespec="seconds")}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


# ---------------------------------------------------------------- HTTP


def parse_range(header: str | None, size: int) -> tuple[int, int] | None | str:
    """Single ``bytes=`` range → inclusive (start, end); None → serve whole file;
    ``"unsatisfiable"`` → 416. Malformed or multi-range headers are ignored (RFC 9110)."""
    if not header:
        return None
    m = re.fullmatch(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", header)
    if not m or (not m.group(1) and not m.group(2)):
        return None
    first, last = m.group(1), m.group(2)
    if first:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
        if last and int(last) < start:
            return None
        if start >= size:
            return "unsatisfiable"
        return start, end
    n = int(last)
    if n == 0 or size == 0:
        return "unsatisfiable"
    return max(0, size - n), size - 1


class ReviewHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "substage-review/1"
    root: Path  # set on the subclass made by make_server

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        if os.environ.get("SUBSTAGE_REVIEW_QUIET"):
            return
        super().log_message(format, *args)

    # -- helpers

    def _send_bytes(self, status: int, body: bytes, ctype: str, *, head: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _json(self, status: int, obj: Any, *, head: bool = False) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8", head=head)

    def _error(self, status: int, msg: str, *, head: bool = False) -> None:
        self._json(status, {"error": msg}, head=head)

    def _parts(self) -> list[str] | None:
        raw = urlsplit(self.path).path
        parts = [unquote(p) for p in raw.split("/") if p]
        if any(p in (".", "..") or "/" in p or "\\" in p or "\x00" in p for p in parts):
            return None
        return parts

    # -- routes

    def do_HEAD(self) -> None:
        self._get(head=True)

    def do_GET(self) -> None:
        self._get(head=False)

    def _get(self, *, head: bool) -> None:
        parts = self._parts()
        if parts is None:
            return self._error(HTTPStatus.BAD_REQUEST, "bad path", head=head)
        if not parts:
            return self._static("index.html", head=head)
        match parts:
            case ["static", name]:
                return self._static(name, head=head)
            case ["api", "videos"]:
                return self._json(HTTPStatus.OK, list_videos(self.root), head=head)
            case ["api", "videos", video_id]:
                d = _video_dir(self.root, video_id)
                if d is None:
                    return self._error(HTTPStatus.NOT_FOUND, "no such video", head=head)
                return self._json(HTTPStatus.OK, video_detail(d), head=head)
            case ["api", "notes", video_id]:
                d = _video_dir(self.root, video_id)
                if d is None:
                    return self._error(HTTPStatus.NOT_FOUND, "no such video", head=head)
                return self._json(HTTPStatus.OK, _read_json(d / "notes.json", []), head=head)
            case ["media", video_id, "video.mp4"]:
                d = _video_dir(self.root, video_id)
                if d is None or not (d / "video.mp4").is_file():
                    return self._error(HTTPStatus.NOT_FOUND, "no video", head=head)
                return self._file(d / "video.mp4", "video/mp4", head=head)
        return self._error(HTTPStatus.NOT_FOUND, "not found", head=head)

    def do_POST(self) -> None:
        parts = self._parts()
        if parts is None or len(parts) != 3 or parts[:2] != ["api", "notes"]:
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        d = _video_dir(self.root, parts[2])
        if d is None:
            return self._error(HTTPStatus.NOT_FOUND, "no such video")
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            return self._error(HTTPStatus.BAD_REQUEST, "bad Content-Length")
        if length <= 0 or length > _MAX_BODY:
            self.close_connection = True
            return self._error(HTTPStatus.BAD_REQUEST, f"body must be 1..{_MAX_BODY} bytes")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            note = append_note(d, payload)
        except (ValueError, UnicodeDecodeError) as e:
            return self._error(HTTPStatus.BAD_REQUEST, str(e))
        return self._json(HTTPStatus.CREATED, note)

    def do_DELETE(self) -> None:
        parts = self._parts()
        if parts is None or len(parts) != 4 or parts[:2] != ["api", "notes"]:
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        d = _video_dir(self.root, parts[2])
        if d is None:
            return self._error(HTTPStatus.NOT_FOUND, "no such video")
        try:
            note_id = int(parts[3])
        except ValueError:
            return self._error(HTTPStatus.NOT_FOUND, "no such note")
        if not delete_note(d, note_id):
            return self._error(HTTPStatus.NOT_FOUND, "no such note")
        return self._json(HTTPStatus.OK, {"deleted": note_id})

    def _static(self, name: str, *, head: bool) -> None:
        ctype = _STATIC.get(name)
        if ctype is None:
            return self._error(HTTPStatus.NOT_FOUND, "not found", head=head)
        body = resources.files(__package__).joinpath("review_static", name).read_bytes()
        self._send_bytes(HTTPStatus.OK, body, ctype, head=head)

    def _file(self, path: Path, ctype: str, *, head: bool) -> None:
        size = path.stat().st_size
        rng = parse_range(self.headers.get("Range"), size)
        if rng == "unsatisfiable":
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if rng is None:
            start, end = 0, size - 1
            self.send_response(HTTPStatus.OK)
        else:
            start, end = rng
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        length = max(end - start + 1, 0)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        if head or length == 0:
            return
        try:
            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers abort range requests on every seek; not an error.
            self.close_connection = True


class _AddressFilteredServer(ThreadingHTTPServer):
    """Bound to all interfaces but only answers connections made to `only_addr`.

    macOS lets unprivileged processes bind ports below 1024 only on the wildcard
    address, so port 80 on one interface is served this way.
    """

    only_addr: str = ""

    def verify_request(self, request, client_address) -> bool:  # noqa: ANN001
        return request.getsockname()[0] == self.only_addr


def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, root: Path | None = None) -> ThreadingHTTPServer:
    """A bound, not yet serving, ThreadingHTTPServer (port 0 → ephemeral)."""
    handler = type("BoundReviewHandler", (ReviewHandler,), {"root": Path(root or default_root())})
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except PermissionError:
        if not 0 < port < 1024 or host in ("", "0.0.0.0"):
            raise
        server = _AddressFilteredServer(("0.0.0.0", port), handler)
        server.only_addr = host
    server.daemon_threads = True
    return server


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, root: Path | None = None) -> None:
    """Serve the review app until interrupted."""
    root = Path(root or default_root())
    root.mkdir(parents=True, exist_ok=True)
    server = make_server(host, port, root)
    print(f"substage review: http://{host}:{server.server_address[1]}/  (data {root})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
