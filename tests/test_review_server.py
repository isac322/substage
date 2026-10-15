import http.client
import json
import os
import threading
from pathlib import Path

import pytest

from substage import review_server as rs

os.environ["SUBSTAGE_REVIEW_QUIET"] = "1"

SRT_EN = """1
00:00:01,000 --> 00:00:03,000
- Hello.
- Hi there.

2
00:00:04,000 --> 00:00:06,000
Second line
"""

SRT_KO = """1
00:00:01,000 --> 00:00:03,000
-안녕
-안녕하세요

2
00:00:04,000 --> 00:00:06,000
두 번째 줄
"""

SAMI = """<SAMI>
<HEAD><TITLE>t</TITLE>
<STYLE TYPE="text/css"><!--
P { margin-left:8pt; }
.KRCC { Name:한국어; lang:ko-KR; SAMIType:CC; }
--></STYLE></HEAD>
<BODY>
<SYNC Start=1000><P Class=KRCC>안녕하세요<br>- 반가워요
<SYNC Start=3000><P Class=KRCC>&nbsp;
<SYNC Start=4000><P Class=KRCC><font color="#ffffff">두 번째</font> 자막
<SYNC Start=6500><P Class=KRCC>&nbsp;
<SYNC Start=7000><P Class=KRCC>마지막
</BODY></SAMI>
"""


@pytest.fixture()
def video(tmp_path: Path) -> tuple[Path, Path]:
    src = tmp_path / "src"
    src.mkdir()
    (src / "en.srt").write_text(SRT_EN, encoding="utf-8")
    (src / "ko.srt").write_text(SRT_KO, encoding="utf-8")
    (src / "official.smi").write_bytes(SAMI.encode("cp949"))
    video = src / "video.mp4"
    video.write_bytes(bytes(range(256)) * 40)  # 10240 bytes, distinct pattern
    root = tmp_path / "review"
    rs.prepare_video(root, "demo-2020", title="Demo", year=2020, en_srt=src / "en.srt",
                    official_path=src / "official.smi", candidates={"B": src / "ko.srt"}, video=video)
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")
    return root, video


@pytest.fixture()
def server(video):
    root, _ = video
    srv = rs.make_server("127.0.0.1", 0, root)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def request(srv, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    res = conn.getresponse()
    data = res.read()
    conn.close()
    return res, data


def test_prepare_video_converts_cp949_sami(video):
    root, _ = video
    d = root / "demo-2020"
    off = (d / "official.ko.srt").read_text(encoding="utf-8")
    assert "안녕하세요\n- 반가워요" in off
    assert "두 번째 자막" in off and "font" not in off
    cues = rs.load_cue_list(d / "official.ko.srt")
    # blank (&nbsp;) syncs are dropped; each cue ends at the next SYNC
    assert [(c["s"], c["e"]) for c in cues[:2]] == [(1000, 3000), (4000, 6500)]
    assert [c["t"] for c in cues] == ["안녕하세요\n- 반가워요", "두 번째 자막", "마지막"]
    assert json.loads((d / "meta.json").read_text(encoding="utf-8"))["title"] == "Demo"
    assert (d / "ko.B.srt").is_file() and (d / "video.mp4").stat().st_size == 10240


def test_prepare_video_rejects_bad_ids(tmp_path):
    with pytest.raises(ValueError):
        rs.prepare_video(tmp_path, "../x", title="x", year=None, en_srt=tmp_path / "a.srt",
                        official_path=None, candidates={})
    with pytest.raises(ValueError):
        rs.prepare_video(tmp_path, "ok", title="x", year=None, en_srt=tmp_path / "a.srt",
                        official_path=None, candidates={"../evil": tmp_path / "a.srt"})


@pytest.mark.parametrize(
    ("header", "start", "end"),
    [("bytes=0-99", 0, 99), ("bytes=5000-", 5000, 10239), ("bytes=-100", 10140, 10239),
     ("bytes=10000-99999", 10000, 10239), ("bytes=10239-10239", 10239, 10239)],
)
def test_range_returns_206_with_exact_bytes(server, video, header, start, end):
    _, video = video
    res, data = request(server, "GET", "/media/demo-2020/video.mp4", headers={"Range": header})
    assert res.status == 206
    assert res.getheader("Content-Range") == f"bytes {start}-{end}/10240"
    assert int(res.getheader("Content-Length")) == end - start + 1
    assert res.getheader("Accept-Ranges") == "bytes"
    assert data == video.read_bytes()[start:end + 1]


def test_full_and_unsatisfiable_ranges(server, video):
    _, video = video
    res, data = request(server, "GET", "/media/demo-2020/video.mp4")
    assert res.status == 200 and data == video.read_bytes()
    assert res.getheader("Content-Type") == "video/mp4"
    res, _ = request(server, "GET", "/media/demo-2020/video.mp4", headers={"Range": "bytes=20000-"})
    assert res.status == 416 and res.getheader("Content-Range") == "bytes */10240"
    res, data = request(server, "HEAD", "/media/demo-2020/video.mp4", headers={"Range": "bytes=0-9"})
    assert res.status == 206 and data == b"" and res.getheader("Content-Length") == "10"


def test_parse_range_edge_cases():
    assert rs.parse_range(None, 10) is None
    assert rs.parse_range("bytes=0-1,4-5", 10) is None  # multi-range → whole file
    assert rs.parse_range("items=0-1", 10) is None
    assert rs.parse_range("bytes=5-2", 10) is None
    assert rs.parse_range("bytes=-0", 10) == "unsatisfiable"
    assert rs.parse_range("bytes=-50", 10) == (0, 9)


@pytest.mark.parametrize(
    "path",
    ["/media/..%2Fsecret.txt/video.mp4", "/media/../secret.txt", "/api/videos/..", "/api/videos/%2E%2E",
     "/api/videos/..%2F..%2Fetc", "/static/..%2F..%2Freview_server.py", "/static/../review_server.py",
     "/api/notes/..%2Fsrc", "/media/demo-2020/en.srt", "/demo-2020/video.mp4", "/static/%2e%2e/secret.txt"],
)
def test_traversal_blocked(server, path):
    res, data = request(server, "GET", path)
    assert res.status in (400, 404)
    assert b"top secret" not in data and b"import" not in data


def test_traversal_post_blocked(server):
    res, _ = request(server, "POST", "/api/notes/..%2Fsrc", body=b'{"comment":"x","time_ms":1}',
                     headers={"Content-Type": "application/json"})
    assert res.status == 404


def test_notes_round_trip(server, video):
    root, _ = video
    res, data = request(server, "GET", "/api/notes/demo-2020")
    assert res.status == 200 and json.loads(data) == []
    payload = {"time_ms": 4200, "arm": "B", "comment": "어색한 말투",
               "cue": {"en": "Second line", "official": "두 번째 자막", "substage": "두 번째 줄"}}
    res, data = request(server, "POST", "/api/notes/demo-2020", body=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"})
    assert res.status == 201
    note = json.loads(data)
    assert note["comment"] == "어색한 말투" and note["time_ms"] == 4200 and note["arm"] == "B"
    assert note["cue"]["official"] == "두 번째 자막" and note["created_at"] and note["id"] == 1
    request(server, "POST", "/api/notes/demo-2020",
            body=json.dumps({"time_ms": 1000, "arm": "B", "comment": "two", "cue": "snap"}).encode(),
            headers={"Content-Type": "application/json"})
    res, data = request(server, "GET", "/api/notes/demo-2020")
    notes = json.loads(data)
    assert [n["comment"] for n in notes] == ["어색한 말투", "two"] and notes[1]["id"] == 2
    on_disk = json.loads((root / "demo-2020" / "notes.json").read_text(encoding="utf-8"))
    assert on_disk == notes
    _, data = request(server, "GET", "/api/videos/demo-2020")
    assert len(json.loads(data)["notes"]) == 2


def _post_note(srv, video_id="demo-2020", comment="메모"):
    res, data = request(srv, "POST", f"/api/notes/{video_id}",
                        body=json.dumps({"time_ms": 1000, "comment": comment}).encode(),
                        headers={"Content-Type": "application/json"})
    assert res.status == 201
    return json.loads(data)


def test_notes_delete_round_trip(server, video):
    root, _ = video
    n1 = _post_note(server, comment="첫째")
    n2 = _post_note(server, comment="둘째")
    res, data = request(server, "DELETE", f"/api/notes/demo-2020/{n2['id']}")
    assert res.status == 200 and json.loads(data) == {"deleted": n2["id"]}
    res, data = request(server, "GET", "/api/notes/demo-2020")
    notes = json.loads(data)
    assert [n["id"] for n in notes] == [n1["id"]]
    on_disk = json.loads((root / "demo-2020" / "notes.json").read_text(encoding="utf-8"))
    assert on_disk == notes
    # second delete of the same id is a 404
    res, _ = request(server, "DELETE", f"/api/notes/demo-2020/{n2['id']}")
    assert res.status == 404


def test_notes_delete_unknown(server):
    _post_note(server)
    for path in ["/api/notes/demo-2020/999", "/api/notes/demo-2020/abc",
                 "/api/notes/demo-2020/1/extra", "/api/notes/no-such-video/1"]:
        res, _ = request(server, "DELETE", path)
        assert res.status == 404
    res, data = request(server, "GET", "/api/notes/demo-2020")
    assert len(json.loads(data)) == 1  # nothing was removed


def test_notes_delete_traversal_blocked(server, video):
    root, _ = video
    n = _post_note(server)
    for path in ["/api/notes/..%2Fdemo-2020/1", "/api/notes/../demo-2020/1",
                 "/api/notes/demo-2020/..%2F1", "/api/notes/demo-2020/.."]:
        res, _ = request(server, "DELETE", path)
        assert res.status in (400, 404)
    on_disk = json.loads((root / "demo-2020" / "notes.json").read_text(encoding="utf-8"))
    assert [x["id"] for x in on_disk] == [n["id"]]  # still there


@pytest.mark.parametrize("body", [b"not json", b'{"time_ms": 1}', b'{"comment": "x"}',
                                  b'{"comment": " ", "time_ms": 1}', b'[1]', b'{"comment":"x","time_ms":-5}'])
def test_notes_rejects_invalid(server, body):
    res, _ = request(server, "POST", "/api/notes/demo-2020", body=body,
                     headers={"Content-Type": "application/json"})
    assert res.status == 400


def test_video_api_and_pages(server):
    res, data = request(server, "GET", "/api/videos")
    videos = json.loads(data)
    assert res.status == 200 and videos[0]["id"] == "demo-2020" and videos[0]["arms"] == ["B"]
    assert videos[0]["has_video"] is True
    res, data = request(server, "GET", "/api/videos/demo-2020")
    video = json.loads(data)
    assert video["video"] == "/media/demo-2020/video.mp4"
    assert video["en"][0]["t"] == "- Hello.\n- Hi there."
    assert video["candidates"]["B"][0]["t"] == "-안녕\n-안녕하세요"
    assert video["official"][0] == {"s": 1000, "e": 3000, "t": "안녕하세요\n- 반가워요"}
    assert video["refeval"] is None
    for path, ctype in [("/", "text/html"), ("/static/app.js", "text/javascript"), ("/static/style.css", "text/css")]:
        res, data = request(server, "GET", path)
        assert res.status == 200 and res.getheader("Content-Type").startswith(ctype) and data
    res, _ = request(server, "GET", "/api/videos/nope")
    assert res.status == 404


def test_text_only_video(tmp_path):
    src = tmp_path / "en.srt"
    src.write_text(SRT_EN, encoding="utf-8")
    d = rs.prepare_video(tmp_path / "r", "textonly", title="T", year=None, en_srt=src,
                        official_path=None, candidates={"A": src})
    video = rs.video_detail(d)
    assert video["video"] is None and video["official"] == [] and video["arms"] == ["A"]
