from __future__ import annotations

import sys
import time
from typing import Any


def _out(text: str, *, newline: bool = True) -> None:
    sys.stdout.write(text + ("\n" if newline else ""))
    sys.stdout.flush()


def line_start(label: str) -> float:
    _out(f"{label}  ... ", newline=False)
    return time.monotonic()


def line_ok(started: float) -> None:
    _out(f"OK  {time.monotonic() - started:.1f}s")


def line_fail(started: float, reasons: list[str] | None = None) -> None:
    extra = "  " + ",".join(reasons) if reasons else ""
    _out(f"FAIL  {time.monotonic() - started:.1f}s{extra}")


def line_interrupted() -> None:
    _out("INTERRUPTED")


def trial_label(
    *,
    case_id: str,
    suite: str,
    profile: str,
    variant: str,
    repeat: int,
    nrep: int,
    seed: int,
) -> str:
    var = variant if variant else "—"
    return (
        f"{case_id:<8}  {suite:<6}  {profile:<8}  {var:<10}  "
        f"{repeat + 1}/{nrep}  seed={seed}"
    )


def score_reasons(score: dict[str, Any] | None) -> list[str]:
    return [str(v) for v in ((score or {}).get("violations") or [])]


def print_skipping(case_ids: list[str]) -> None:
    if case_ids:
        _out("Skipping test cases: " + ", ".join(case_ids))


def print_header(run_dir: str, model: str) -> None:
    _out(f"bench  {run_dir}")
    _out(f"model  {model}")


def print_totals(*, n_ok: int, n_fail: int, interrupted: bool) -> None:
    extra = "  interrupted" if interrupted else ""
    _out(f"{n_ok} OK, {n_fail} FAIL{extra}")
