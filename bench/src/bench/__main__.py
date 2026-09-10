from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.config import load_config
from bench.paths import bench_root
from bench.runner import run_benchmark


def _csv(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    profiles = _csv(args.profiles) if args.profiles else list(cfg.get("profiles") or {})
    suites = _csv(args.suites) if args.suites else list(cfg.get("suites") or ["tools", "agent", "coding"])
    out = Path(args.out) if args.out else bench_root() / "results"
    run_dir = run_benchmark(
        cfg,
        profiles=profiles,
        suites=suites,
        out_root=out,
        prompt_variants=_csv(args.prompt_variants) if args.prompt_variants else None,
    )
    print(run_dir)
    if (run_dir / "INTERRUPTED.txt").is_file():
        return 130
    return 0


def cmd_check_reference(args: argparse.Namespace) -> int:
    from bench.coding_loop import check_reference

    result = check_reference()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--config", default="config.yaml")
    p_run.add_argument("--profiles", default="greedy")
    p_run.add_argument("--suites", default="tools")
    p_run.add_argument("--prompt-variants", default="neutral", help="neutral,helpful,instructed,harness (empty = config prompt_variants)")
    p_run.add_argument("--out", default="")
    p_run.set_defaults(func=cmd_run)

    p_ref = sub.add_parser("check-reference")
    p_ref.set_defaults(func=cmd_check_reference)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
