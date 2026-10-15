"""Pull finished videos from the pipeline host and publish them to the review web app.

usage: uv run python scripts/review_sync.py HOST VIDEO_ID[:ARM=WORKSPACE,...] ...
  e.g. uv run python scripts/review_sync.py media-host.local sample-2016:substage=sample-2016__v2a

Per video it copies the English subtitle, the official Korean subtitle (path given by
OFFICIAL_KO below or <video>/official_ko.txt on the host), each arm's .ko.srt, the arm's
refeval.json when present, and the 720p review video, then calls prepare_video().
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from substage.review_server import default_root, prepare_video

REMOTE_DATA = ".local/share/substage"
REMOTE_CACHE = ".cache/substage"


def _ssh_json(host: str, path: str) -> dict:
    out = subprocess.run(["ssh", host, f"cat {path}"], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def _pull(host: str, remote: str, local: Path) -> Path | None:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["rsync", "-a", "--protect-args", f"{host}:{remote}", str(local)], capture_output=True, text=True)
    return local if r.returncode == 0 and local.exists() else None


def sync(host: str, spec: str, official: dict[str, str], tmp: Path) -> None:
    video_id, _, arms_spec = spec.partition(":")
    arms = dict(a.split("=", 1) for a in arms_spec.split(",") if a) or {"substage": video_id}
    video = _ssh_json(host, f"{REMOTE_DATA}/{video_id}/video.json")
    work = tmp / video_id
    en = _pull(host, video["en_sub_path"], work / "en.srt")
    ref_remote = official[video_id]
    ref = _pull(host, ref_remote, work / ("official" + Path(ref_remote).suffix))
    cands: dict[str, Path] = {}
    refeval = None
    for arm, ws in arms.items():
        # emit names the file after VideoInfo.video_id, which is the arm workspace id
        # when the arm re-ran ingest instead of seeding video.json from the base
        got = _pull(host, f"{REMOTE_DATA}/{ws}/{video_id}.ko.srt", work / f"{arm}.ko.srt") or _pull(
            host, f"{REMOTE_DATA}/{ws}/{ws}.ko.srt", work / f"{arm}.ko.srt"
        )
        if got:
            cands[arm] = got
        refeval = refeval or _pull(host, f"{REMOTE_DATA}/{video_id}/refeval_review/refeval.json", work / "refeval.json")
    # Large: keep a persistent local copy so re-syncs only transfer changes.
    video = _pull(host, f"{REMOTE_CACHE}/{video_id}/review.mp4", Path.home() / ".cache/substage-review" / video_id / "video.mp4")
    if en is None or ref is None or not cands:
        raise SystemExit(f"{video_id}: missing inputs (en={en}, official={ref}, arms={list(cands)})")
    prepare_video(
        default_root(), video_id, title=video["title"], year=video.get("year"), en_srt=en,
        official_path=ref, candidates=cands, video=video, refeval=refeval,
    )
    print(f"{video_id}: arms={list(cands)} video={'yes' if video else 'no'}")


def main() -> None:
    host, *specs = sys.argv[1:]
    official = _ssh_json(host, f"{REMOTE_DATA}/official_ko.json")
    with tempfile.TemporaryDirectory() as d:
        for spec in specs:
            sync(host, spec, official, Path(d))


if __name__ == "__main__":
    main()
