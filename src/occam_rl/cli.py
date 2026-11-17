"""One readable entry point delegates phases to independently testable modules."""

import argparse
import sys
from pathlib import Path

from .io import ROOT, EventLog, config, now, runs_dir
from .publication import assert_published
from .schema import PROMPT_ARMS, TRAINED_ARMS


def main() -> None:
    """Check publication before importing or executing any experimental phase."""
    parser = argparse.ArgumentParser(description="Local Occam GRPO research pipeline")
    parser.add_argument("--config", help="Published experiment configuration JSON")
    sub = parser.add_subparsers(dest="phase", required=True)
    sub.add_parser("resolve", help="Freeze model/dataset/llama.cpp revisions")
    sub.add_parser("doctor", help="Inspect CUDA, package versions and sandbox image")
    data = sub.add_parser("prepare", help="Generate and freeze tasks, exclusions and split hashes")
    data.add_argument("--procedural-only", action="store_true", help="Exploratory data; not the registered public comparison")
    audit = sub.add_parser("qualify", help="Run the locked constructed judge audit")
    audit.add_argument("role", choices=["qwen", "gemma"])
    sub.add_parser("select-gemma", help="Profile Gemma variants and freeze memory-based choice")
    sub.add_parser("profile", help="Measure full-size GPU passes and role/state preservation")
    sub.add_parser("budget", help="Freeze common updates from Occam pilot timing")
    training = sub.add_parser("train", help="Train one local GRPO adapter")
    training.add_argument("arm", choices=TRAINED_ARMS)
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--pilot", action="store_true")
    training.add_argument("--resume", action="store_true")
    evaluation = sub.add_parser("evaluate", help="Generate, verify and independently judge held-out solutions")
    evaluation.add_argument("arm", choices=(*TRAINED_ARMS, *PROMPT_ARMS))
    evaluation.add_argument("--seed", type=int, default=17)
    evaluation.add_argument("--split", choices=["dev", "calibration", "test", "ood"], default="test")
    evaluation.add_argument("--limit-per-domain", type=int)
    sub.add_parser("calibrate", help="Freeze length normalizers from ordinary-prompt calibration")
    sub.add_parser("report", help="Regenerate reports, figures, paper PDF, README tables and site data")
    dashboard = sub.add_parser("dashboard", help="Serve live local metrics and static explorer")
    dashboard.add_argument("--port", type=int, default=8765)
    publish = sub.add_parser("publish", help="Upload exact source snapshot or one run to Hugging Face")
    publication = publish.add_mutually_exclusive_group(required=True)
    publication.add_argument("--source", action="store_true")
    publication.add_argument("--run")
    args = parser.parse_args()
    if args.config and not Path(args.config).resolve().is_relative_to(ROOT):
        parser.error("Experiment configuration must be in the published checkout")
    cfg = config(args.config)
    commit = assert_published()
    if args.phase == "train":
        name = f"{args.arm}-seed{args.seed}" + ("-pilot" if args.pilot else "")
        log = EventLog(runs_dir() / name / "events.jsonl")
    else:
        stamp = now().replace(":", "-")
        log = EventLog(runs_dir() / "operations" / f"{stamp}-{args.phase}.jsonl")
    log.emit("phase_start", phase=args.phase, git_commit=commit, arguments=vars(args))
    try:
        if args.phase in ("resolve", "doctor", "qualify", "profile", "select-gemma", "budget"):
            from . import preflight
            function = {"resolve": preflight.resolve, "doctor": preflight.doctor, "qualify": preflight.qualify,
                        "profile": preflight.profile, "select-gemma": preflight.select_gemma, "budget": preflight.select_budget}[args.phase]
            function(cfg, log, args.role) if args.phase == "qualify" else function(cfg, log)
        elif args.phase == "prepare":
            from .data import prepare
            prepare(cfg, log, public=not args.procedural_only)
        elif args.phase == "train":
            from .training import train
            train(cfg, log, commit, args.arm, args.seed, pilot=args.pilot, resume=args.resume)
        elif args.phase == "evaluate":
            from .evaluation import evaluate
            evaluate(cfg, log, commit, args.arm, args.seed, args.split, args.limit_per_domain)
        elif args.phase == "calibrate":
            from .evaluation import calibrate
            calibrate(cfg, log)
        elif args.phase == "report":
            from .reporting import report
            report(cfg, log)
        elif args.phase == "dashboard":
            from .dashboard import serve
            serve(args.port)
        elif args.phase == "publish":
            from .publishing import publish_run, publish_source
            publish_source(cfg, log, commit) if args.source else publish_run(cfg, log, commit, args.run)
    except BaseException as exc:
        log.emit("phase_failed", phase=args.phase, error_type=type(exc).__name__, error=str(exc))
        raise
    log.emit("phase_complete", phase=args.phase)


if __name__ == "__main__":
    main()
