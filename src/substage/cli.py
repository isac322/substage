"""Console entry point: `substage run VIDEO ...`, `substage cost VIDEO_ID`, the
evaluation commands `substage quality` / `substage refeval` / `substage ablate`, and the translator
memory corpus `substage memory add|stats`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models import UsageRecord

_JUDGE_MODEL_HELP = (
    "judge model id (gemini-*, anthropic/<m>, codex/<m>); repeatable or comma list — several judges each "
    "score and are calibrated on the same windows, the report adds per-judge tables and a mean "
    "(default: settings model_main)"
)


def _cmd_cost(video_id: str) -> int:
    """Print the per-stage token/cost table from a video's usage log."""
    from .config import workspace
    from .pipeline import _print_rows

    ws = workspace(video_id)
    log = ws.log_dir / "usage.jsonl"
    if not log.exists():
        print(f"substage: no usage recorded for '{video_id}' (looked in {log})", file=sys.stderr)
        return 1
    rows: dict[str, list[float]] = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = UsageRecord.model_validate_json(line)
        tot = rows.setdefault(r.stage, [0.0, 0.0, 0.0, 0.0])
        tot[0] += r.input_tokens
        tot[1] += r.output_tokens
        tot[2] += r.cached_tokens
        tot[3] += r.cost_usd
    table = [(stage, int(v[0]), int(v[1]), int(v[2]), v[3]) for stage, v in rows.items()]
    table.append(
        (
            "TOTAL",
            int(sum(v[0] for v in rows.values())),
            int(sum(v[1] for v in rows.values())),
            int(sum(v[2] for v in rows.values())),
            sum(v[3] for v in rows.values()),
        )
    )
    _print_rows(table)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from . import pipeline

    out = pipeline.run(
        Path(args.video),
        video_id=args.video_id,
        title=args.title,
        year=args.year,
        en_sub=Path(args.en_sub) if args.en_sub else None,
        media_id=args.media_id,
        mode=args.mode,
        stop_after=args.stop_after,
        arm=args.arm,
        no_viewing=args.no_viewing,
        model=args.model,
        architecture=args.architecture,
        experiment_id=args.experiment_id,
    )
    print(f"substage: wrote {out}")
    return 0


def _cmd_quality(args: argparse.Namespace) -> int:
    """Report deterministic quality; references never determine the pass gate."""
    import json

    from . import refeval

    kwargs = {"ws_dir": args.ws_dir, "srt_path": args.srt, "ref_path": args.ref}
    if len(args.video_ids) == 1:
        report = refeval.evaluate_media_quality(args.video_ids[0], **kwargs)
    else:
        report = refeval.evaluate_corpus_quality([
            {"video_id": video_id, **kwargs} for video_id in args.video_ids
        ])
    markdown = refeval.render_quality_md(report)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "quality.json").write_text(serialized, encoding="utf-8")
        (out / "quality.md").write_text(markdown, encoding="utf-8")
    print(serialized if args.json else markdown, end="")
    if args.fail_under is None:
        return 0
    if args.fail_under == 0:
        return 0 if report["passes_targets"] else 1
    return int(report["quality_score"] < args.fail_under)


def _cmd_refeval(args: argparse.Namespace) -> int:
    from . import evaluate, refeval
    from .config import load_settings

    ref = Path(args.ref)
    if not ref.is_file():
        raise FileNotFoundError(f"official subtitle not found: {ref}")
    out = Path(args.out) if args.out else refeval.default_out_dir(args.video_id)
    evaluation_kwargs = {}
    if args.evaluation_mode != "listwise" or any(
        value is not None for value in (args.opus_ref, args.split, args.cycle_id)
    ):
        evaluation_kwargs = {
            "evaluation_mode": args.evaluation_mode,
            "opus_ref": args.opus_ref,
            "split": args.split,
            "cycle_id": args.cycle_id,
        }
    if getattr(args, "decisions_model", None):
        evaluation_kwargs["decisions_model"] = args.decisions_model
    report = refeval.run_refeval(
        args.video_id, evaluate.parse_arms(args.arms), ref, args.n_scenes, args.seed, out, judge=not args.no_judge,
        judge_models=refeval.parse_judge_models(args.judge_model, load_settings().model_main),
        **evaluation_kwargs,
    )
    print((out / "refeval.md").read_text(encoding="utf-8"))
    print(f"substage: wrote {out / 'refeval.json'} ({len(report.get('groups', []))} groups)")
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    from . import ablate, refeval
    from .config import load_settings, workspace

    ref = Path(args.ref)
    if not ref.is_file():
        raise FileNotFoundError(f"official subtitle not found: {ref}")
    configs = [ablate.parse_config(spec) for spec in args.config]
    ws = workspace(args.video_id)
    out = Path(args.out) if args.out else ws.data_dir / "ablate"
    if not args.out and args.sample_from:
        source = Path(args.sample_from)
        source = source / "ablate.json" if source.is_dir() else source
        if source.resolve() == (out / "ablate.json").resolve():
            out = out / "rerun"
    evaluation_kwargs = {}
    if args.evaluation_mode != "listwise" or any(
        value is not None for value in (args.opus_ref, args.split, args.cycle_id)
    ):
        evaluation_kwargs = {
            "evaluation_mode": args.evaluation_mode,
            "opus_ref": args.opus_ref,
            "split": args.split,
            "cycle_id": args.cycle_id,
        }
    extra_kwargs = {}
    if args.sample_from is not None:
        extra_kwargs["sample_from"] = args.sample_from
    if args.experiment_id is not None:
        extra_kwargs["experiment_id"] = args.experiment_id
    if args.declared_architecture is not None:
        extra_kwargs["declared_architecture"] = args.declared_architecture
    if args.effort is not None:
        extra_kwargs["effort"] = args.effort
    if args.role:
        roles = {}
        for raw in args.role:
            key, sep, value = raw.partition("=")
            if not sep or not key.strip() or not value.strip():
                raise ValueError(f"--role must look like NAME=MODEL, got {raw!r}")
            roles[key.strip()] = value.strip()
        extra_kwargs["roles"] = roles
    ablate.run_ablation(
        ws, configs, ref, args.scenes, args.seed, judge=args.judge, out_dir=out,
        judge_models=refeval.parse_judge_models(args.judge_model, load_settings().model_main),
        **evaluation_kwargs, **extra_kwargs,
    )
    print((out / "ablate.md").read_text(encoding="utf-8"))
    print(f"substage: wrote {out / 'ablate.json'}")
    return 0


def _cmd_memory_add(args: argparse.Namespace) -> int:
    from . import memory, refeval
    from .config import memory_path, workspace
    from .models import Cue
    from .store import load_models

    ref = Path(args.ref)
    if not ref.is_file():
        raise FileNotFoundError(f"official subtitle not found: {ref}")
    cues_path = workspace(args.video_id).data_dir / "cues.json"
    if not cues_path.is_file():
        raise FileNotFoundError(f"video workspace has no cues.json (run `substage run --stop-after ingest` first): {cues_path}")
    video_id = memory.base_video_id(args.video_id)
    pairs = memory.build_pairs(video_id, load_models(cues_path, Cue), refeval.load_ko(ref))
    n = memory.add_video(video_id, pairs)
    print(f"substage: {video_id}: {n} pairs → {memory_path()}")
    return 0


def _cmd_memory_stats(args: argparse.Namespace) -> int:
    from . import memory
    from .config import memory_path

    path = memory_path()
    counts = memory.stats(memory.load_pairs(path))
    if not counts:
        print(f"substage: translator memory is empty ({path})")
        return 0
    width = max(len(f) for f in counts)
    for video, n in counts.items():
        print(f"{video:<{width}}  {n:>6}")
    print(f"{'TOTAL':<{width}}  {sum(counts.values()):>6}  ({path})")
    return 0



def _cmd_review_video(args: argparse.Namespace) -> int:
    from . import media
    from .config import workspace

    video = Path(args.video)
    streams, _ = media.probe(video)
    out = workspace(args.video_id).cache("review.mp4")
    media.make_review_video(video, media.pick_dialogue_audio(streams), out)
    print(f"substage: wrote {out}")
    return 0


def _cmd_review_server(args: argparse.Namespace) -> int:
    from .review_server import serve

    serve(args.host, args.port, Path(args.root) if args.root else None)
    return 0


def _cmd_proxy(args: argparse.Namespace) -> int:
    from .remote import remote_proxy

    print(f"substage: installed {remote_proxy(args.host, args.video, args.video_id)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="substage", description="EN subtitles + video + audio → professional Korean subtitles")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="translate a video end to end")
    run_p.add_argument("video", help="path to the video video file")
    run_p.add_argument("--video-id", help="workspace slug (default: derived from --title/filename)")
    run_p.add_argument("--title", help="video title (default: derived from filename)")
    run_p.add_argument("--year", type=int, help="release year (default: parsed from filename)")
    run_p.add_argument("--en-sub", help="use this English subtitle file instead of extracting/downloading")
    run_p.add_argument("--media-id", help="Media id (tt…) used to search English subtitles when none is embedded; read from a .nfo next to the video if omitted")
    run_p.add_argument("--mode", choices=["text", "av", "cascade", "dpaf", "mixed"], default="cascade")
    run_p.add_argument("--model", help="explicit model identity resolved through capability routing")
    run_p.add_argument(
        "--architecture", "--route", dest="architecture",
        choices=("auto", "direct-av", "av-two-pass", "text-snapshot", "dpaf", "cascade", "mixed"),
        help="executed capability route (mixed explicitly permits alternative Korean writers)",
    )
    run_p.add_argument("--experiment-id", help="explicit budget ledger identity for paid model calls")
    run_p.add_argument("--stop-after", metavar="STAGE", help="stop after this stage (ingest, scenes, proxy, observations, context_map, timeline, register, snapshots, ost, translate, review, s5, recheck, layout, emit)")
    run_p.add_argument("--arm", metavar="NAME", help="experiment arm: run in workspace <video_id>__NAME, seeded on first use from <video_id>'s pre-translation checkpoints")
    run_p.add_argument("--no-viewing", action="store_true", help="skip the S2v viewing pass; video map/timeline from text only")
    run_p.set_defaults(func=_cmd_run)

    cost_p = sub.add_parser("cost", help="show the recorded token/cost table for a video")
    cost_p.add_argument("video_id")
    cost_p.set_defaults(func=lambda a: _cmd_cost(a.video_id))

    quality_p = sub.add_parser("quality", help="audit subtitle quality without an official Korean reference")
    quality_p.add_argument("video_ids", nargs="+", metavar="VIDEO_ID")
    quality_p.add_argument("--ws-dir", help="workspace artifact directory")
    quality_p.add_argument("--srt", help="generated Korean subtitle path")
    quality_p.add_argument("--ref", help="optional local reference; never required for target gating")
    quality_p.add_argument("--out", help="write quality.json and quality.md into this directory")
    quality_p.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    quality_p.add_argument("--fail-under", type=float, metavar="SCORE", help="exit 1 below SCORE; 0 checks all pass targets")
    quality_p.set_defaults(func=_cmd_quality)

    ref_p = sub.add_parser("refeval", help="score arms against the official Korean subtitles (gold)")
    ref_p.add_argument("video_id", help="base workspace (cues/scenes/video map come from here)")
    ref_p.add_argument("--arms", required=True, help="NAME=WORKSPACE_ID,... e.g. A=sample-2016__a,B=sample-2016")
    ref_p.add_argument("--ref", required=True, help="official Korean subtitle (.srt/.smi) — the gold answer")
    ref_p.add_argument("--n-scenes", type=int, default=20, help="judge windows (default 20)")
    ref_p.add_argument("--seed", type=int, default=1)
    ref_p.add_argument("--out", help="output dir (default <base data dir>/refeval)")
    ref_p.add_argument("--no-judge", action="store_true", help="deterministic metrics only")
    ref_p.add_argument("--judge-model", action="append", metavar="ID[,ID...]", help=_JUDGE_MODEL_HELP)
    ref_p.add_argument("--evaluation-mode", choices=("listwise", "paired"), default="listwise",
                       help="default legacy listwise; paired scores each arm independently against frozen Opus")
    ref_p.add_argument("--opus-ref", help="frozen Opus Korean subtitle artifact (required only for paired mode)")
    ref_p.add_argument("--split", choices=("dev", "heldout"), help="explicit dataset split (paired default: dev)")
    ref_p.add_argument("--cycle-id", help="experiment cycle identifier stored in the report")
    ref_p.add_argument(
        "--decisions-model",
        metavar="MODEL",
        help="run 6-axis Cloudflare Clef/Clef-omni Decisions pre-benchmark (e.g. clef-omni, clef, clef-flash)",
    )
    ref_p.set_defaults(func=_cmd_refeval)

    ab_p = sub.add_parser("ablate", help="execute production translation architectures on frozen sampled scenes")
    ab_p.add_argument("video_id", help="base workspace (cues/scenes/snapshots/observations come from here)")
    ab_p.add_argument("--ref", required=True, help="official Korean subtitle (.srt/.smi) — scoring gold only")
    ab_p.add_argument("--scenes", type=int, default=40, help="sampled scenes (default 40; half targeted)")
    ab_p.add_argument("--seed", type=int, default=3)
    ab_p.add_argument("--config", action="append", required=True, metavar="NAME:k=v,...",
                      help="model=ID,thinking=low|medium|high|xhigh,mode=auto|text|av|dpaf|cascade|mixed,architecture=auto|direct-av|av-two-pass|text-snapshot|dpaf|cascade|mixed,perception=MODEL,router=MODEL,av=MODEL,observations=on|off,narrative=on|lite|off,aim=RATIO,memory=K (repeatable; fake text model IDs remain supported offline)")
    ab_p.add_argument("--judge", action="store_true", help="also run the blind refeval judge on windows of the sampled scenes")
    ab_p.add_argument("--judge-model", action="append", metavar="ID[,ID...]", help=_JUDGE_MODEL_HELP)
    ab_p.add_argument("--out", help="output dir (default <base data dir>/ablate)")
    ab_p.add_argument("--evaluation-mode", choices=("listwise", "paired"), default="listwise",
                      help="default legacy listwise; paired allows any number of independently judged configs")
    ab_p.add_argument("--opus-ref", help="frozen Opus Korean subtitle artifact (required only for paired mode)")
    ab_p.add_argument("--split", choices=("dev", "heldout"), help="explicit dataset split (paired default: dev)")
    ab_p.add_argument("--cycle-id", help="experiment cycle identifier stored in the report")
    ab_p.add_argument("--sample-from", help="reuse exact scene records from a prior ablate.json; validates video, times, and cue IDs")
    ab_p.add_argument("--experiment-id", help="write an adaptive experiment manifest for this run")
    ab_p.add_argument("--declared-architecture", "--architecture", dest="declared_architecture",
                      choices=("auto", "direct-av", "av-two-pass", "text-snapshot", "dpaf", "cascade", "mixed"),
                      help="execute this architecture for every configuration, not just label the manifest")
    ab_p.add_argument("--effort", choices=("low", "medium", "high", "xhigh"),
                      help="override actual translator and auxiliary role effort")
    ab_p.add_argument("--role", action="append", default=[], metavar="NAME=MODEL",
                      help="execute translator/perception/router/av role (repeatable; unused or incompatible roles fail)")
    ab_p.set_defaults(func=_cmd_ablate)

    mem_p = sub.add_parser("memory", help="translator memory: official (EN → KO) pairs of reference videos used as S3 style examples")
    mem_sub = mem_p.add_subparsers(dest="memory_command", required=True)
    mem_add = mem_sub.add_parser("add", help="align a video's English cues with its official Korean subtitles and store the pairs (replaces the video's rows)")
    mem_add.add_argument("video_id", help="video workspace with cues.json (an arm suffix __NAME is stripped)")
    mem_add.add_argument("--ref", required=True, help="official Korean subtitle (.srt/.smi)")
    mem_add.set_defaults(func=_cmd_memory_add)
    mem_stats = mem_sub.add_parser("stats", help="pair counts per video")
    mem_stats.set_defaults(func=_cmd_memory_stats)

    rv_p = sub.add_parser("review-video", help="encode the 720p H.264 review video (rkmpp hardware path when available)")
    rv_p.add_argument("video")
    rv_p.add_argument("--video-id", required=True)
    rv_p.set_defaults(func=_cmd_review_video)

    rs_p = sub.add_parser("review-server", help="serve the review web app (official vs substage, synced to the video)")
    rs_p.add_argument("--host", default="127.0.0.1")
    rs_p.add_argument("--port", type=int, default=8000)
    rs_p.add_argument("--root", help="data root (default $XDG_DATA_HOME/substage-review)")
    rs_p.set_defaults(func=_cmd_review_server)

    proxy_p = sub.add_parser("proxy", help="encode the A/V proxy locally (hardware decode) for a video on HOST and install it in HOST's cache")
    proxy_p.add_argument("host", help="ssh host that runs the pipeline and stores the video")
    proxy_p.add_argument("video", help="video path on HOST")
    proxy_p.add_argument("--video-id", required=True, help="workspace slug used by `substage run` on HOST")
    proxy_p.set_defaults(func=_cmd_proxy)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
