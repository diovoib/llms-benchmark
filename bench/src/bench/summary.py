from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from bench.hard_score import DIMENSIONS, SUITE_WEIGHTS, aggregate_trials
from bench.suites.cases import case_stem
from bench.prompts import KNOWN_PROMPT_VARIANTS, POLICY_CONFIRM, variant_system_text
from bench.results_io import (
    copy_judge_bundle,
    write_case_md,
    write_json,
    write_results_readme,
    write_text,
)
from bench.suites.cases import all_cases


STAMP_RE = re.compile(r"^\d{8}T\d{6}Z$")
PROFILE_NAMES = frozenset({"greedy", "real"})


def _rates_block(trials: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list] = defaultdict(list)
    by_suite: dict[str, list] = defaultdict(list)
    by_dim: dict[str, list] = defaultdict(list)
    for t in trials:
        by_case[t["case_id"]].append(t)
        by_suite[t["suite"]].append(t)
        for dim, ids in DIMENSIONS.items():
            if case_stem(t["case_id"]) in ids:
                by_dim[dim].append(t)
    case_agg = {cid: aggregate_trials(rows) for cid, rows in by_case.items()}
    suite_agg = {s: aggregate_trials(rows) for s, rows in by_suite.items()}
    dim_agg = {d: aggregate_trials(rows) for d, rows in by_dim.items()}
    weighted = []
    for suite, agg in suite_agg.items():
        w = SUITE_WEIGHTS.get(suite, 1.0)
        if agg.get("hard_pass_rate") is None or w == 0:
            continue
        weighted.append((w, agg["hard_pass_rate"]))
    overall = None
    if weighted:
        overall = sum(w * r for w, r in weighted) / sum(w for w, _ in weighted)
    return {
        "by_case": case_agg,
        "by_suite": suite_agg,
        "by_dimension": dim_agg,
        "weighted_suite_hard_pass_rate": overall,
        "n_trials": len(trials),
    }


def _rate_delta(instructed: Any, neutral: Any) -> Any:
    if instructed is None or neutral is None:
        return None
    try:
        return instructed - neutral
    except TypeError:
        return None


def _agg_delta(instructed: dict[str, Any] | None, neutral: dict[str, Any] | None) -> dict[str, Any] | None:
    if not instructed or not neutral:
        return None
    keys = set(instructed) | set(neutral)
    out: dict[str, Any] = {}
    for key in sorted(keys):
        a = instructed.get(key) or {}
        b = neutral.get(key) or {}
        out[key] = {
            "hard_pass_rate": _rate_delta(a.get("hard_pass_rate"), b.get("hard_pass_rate")),
            "instructed": a.get("hard_pass_rate"),
            "neutral": b.get("hard_pass_rate"),
            "n_counted_instructed": a.get("n_counted"),
            "n_counted_neutral": b.get("n_counted"),
        }
    return out


def _headline_rate(block: dict[str, Any] | None) -> float | None:
    if not block:
        return None
    rate = block.get("weighted_suite_hard_pass_rate")
    if isinstance(rate, (int, float)):
        return float(rate)
    return None


def build_summary(all_trials: list[dict[str, Any]], coding_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_trials = [t for t in all_trials if not t.get("interrupted") and not t.get("in_progress")]
    coding_rows = [r for r in coding_rows if not r.get("interrupted") and not r.get("in_progress")]
    pool = [t for t in all_trials if t.get("suite") != "coding"]
    by_variant: dict[str, list] = defaultdict(list)
    for t in pool:
        by_variant[str(t.get("prompt_variant") or "unknown")].append(t)
    variant_blocks = {name: _rates_block(rows) for name, rows in by_variant.items()}
    rates = _rates_block(pool)
    inst = variant_blocks.get("instructed")
    neu = variant_blocks.get("neutral")
    delta = None
    if inst and neu:
        delta = {
            "note": "instructed minus neutral. Positive = the scoring rubric prompt fixed a failure.",
            "weighted_suite_hard_pass_rate": _rate_delta(
                inst.get("weighted_suite_hard_pass_rate"),
                neu.get("weighted_suite_hard_pass_rate"),
            ),
            "by_dimension": _agg_delta(inst.get("by_dimension"), neu.get("by_dimension")),
            "by_case": _agg_delta(inst.get("by_case"), neu.get("by_case")),
        }
    return {
        "ground_truth": True,
        "do_not_rejudge": True,
        "by_prompt_variant": variant_blocks,
        "delta_instructed_minus_neutral": delta,
        "by_case": rates.get("by_case"),
        "by_suite": rates.get("by_suite"),
        "by_dimension": rates.get("by_dimension"),
        "weighted_suite_hard_pass_rate": rates.get("weighted_suite_hard_pass_rate"),
        "suite_weights": SUITE_WEIGHTS,
        "coding": [
            {
                "trial_id": r.get("trial_id"),
                "final_review": r.get("final_review"),
                "truncated": r.get("truncated"),
                "n_attempts": r.get("n_attempts"),
                "attempt_files": r.get("attempt_files"),
                "conversation": r.get("conversation"),
                "python_checks_ok": r.get("python_checks_ok"),
            }
            for r in coding_rows
        ],
        "n_trials": rates.get("n_trials"),
    }


def summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "GROUND TRUTH — do not rejudge mechanical fields.",
        f"weighted_suite_hard_pass_rate: {summary.get('weighted_suite_hard_pass_rate')}",
        "",
        "== by prompt variant ==",
    ]
    for name, block in (summary.get("by_prompt_variant") or {}).items():
        rate = block.get("weighted_suite_hard_pass_rate") if isinstance(block, dict) else None
        n = block.get("n_trials") if isinstance(block, dict) else None
        lines.append(f"{name}: weighted={rate} n={n}")
    if summary.get("by_profile"):
        lines.append("")
        lines.append("== by profile ==")
        for name, block in summary["by_profile"].items():
            rate = block.get("weighted_suite_hard_pass_rate") if isinstance(block, dict) else block
            lines.append(f"{name}: weighted={rate}")
    if summary.get("by_model"):
        lines.append("")
        lines.append("== by model ==")
        for name, rate in summary["by_model"].items():
            lines.append(f"{name}: weighted={rate}")
    delta = summary.get("delta_instructed_minus_neutral")
    if delta:
        lines.append("")
        lines.append("== delta instructed − neutral (by dimension) ==")
        for dim, row in (delta.get("by_dimension") or {}).items():
            lines.append(
                f"{dim}: delta={row.get('hard_pass_rate')} instructed={row.get('instructed')} neutral={row.get('neutral')}"
            )
        lines.append(f"weighted delta: {delta.get('weighted_suite_hard_pass_rate')}")
    if summary.get("by_dimension"):
        lines.append("")
        lines.append("== by dimension ==")
        for dim, agg in (summary.get("by_dimension") or {}).items():
            lines.append(
                f"{dim}: hard_pass_rate={agg.get('hard_pass_rate')} mode_agreement={agg.get('mode_agreement')} n={agg.get('n_counted')}"
            )
    if summary.get("by_case"):
        lines.append("")
        lines.append("== by case ==")
        for cid, agg in sorted((summary.get("by_case") or {}).items()):
            lines.append(
                f"{cid}: hard_pass_rate={agg.get('hard_pass_rate')} mode_agreement={agg.get('mode_agreement')} infra={agg.get('n_infra')}"
            )
    lines.append("")
    lines.append("== coding ==")
    for row in summary.get("coding") or []:
        lines.append(
            f"{row.get('trial_id')} verdict={row.get('final_review')} truncated={row.get('truncated')} "
            f"attempts={row.get('n_attempts')} python_checks_ok={row.get('python_checks_ok')} "
            f"conversation={row.get('conversation')}"
        )
    return "\n".join(lines) + "\n"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        sys.stderr.write(f"skipping unparseable JSON: {path}\n")
        sys.stderr.flush()
        return None
    except OSError:
        return None
    return data if isinstance(data, dict) else None


def _write_summary_pair(folder: Path, summary: dict[str, Any]) -> None:
    write_json(folder / "summary.json", summary)
    write_text(folder / "summary.txt", summary_text(summary))


def _load_trial(path: Path) -> dict[str, Any] | None:
    data = _load_json(path)
    if not data or data.get("in_progress"):
        return None
    return data


def write_trial_id_map(root: Path) -> None:
    mapping: dict[str, Any] = {}
    for path in list(root.glob("**/cases/*/trial_*.json")) + list(root.glob("**/coding/*/trial.json")):
        row = _load_json(path)
        if not row or row.get("in_progress"):
            continue
        tid = row.get("trial_id")
        if not tid:
            continue
        mapping[str(tid)] = {
            "case_id": row.get("case_id"),
            "repeat": row.get("repeat"),
            "prompt_variant": row.get("prompt_variant"),
            "path": str(path.relative_to(root)).replace("\\", "/"),
        }
    write_json(root / "trial_id_map.json", mapping)


def write_reconstructed_manifest(root: Path) -> None:
    existing = _load_json(root / "MANIFEST.json") or {}
    model_names: list[str] = []
    for profile_dir in _profile_dirs(root):
        parent = profile_dir.parent
        name = parent.name if parent != root else root.name
        if name not in model_names:
            model_names.append(name)
    profiles = sorted({p.name for p in _profile_dirs(root)})
    variants = sorted({cases_dir.parent.name for cases_dir in root.glob("**/cases") if cases_dir.is_dir()})
    if not existing.get("models"):
        existing["models"] = model_names
    if not existing.get("profiles"):
        existing["profiles"] = profiles
    if not existing.get("prompt_variants"):
        existing["prompt_variants"] = variants
    existing["reconstructed"] = True
    write_json(root / "MANIFEST.json", existing)


def snapshot_prompts(dest: Path) -> None:
    folder = dest / "prompts"
    folder.mkdir(parents=True, exist_ok=True)
    for name in KNOWN_PROMPT_VARIANTS:
        try:
            text = variant_system_text(name, {"harness_system": ""})
        except ValueError:
            continue
        write_text(folder / f"SYSTEM_{name}.txt", text + ("\n" if not text.endswith("\n") else ""))
    write_text(folder / "POLICY.txt", POLICY_CONFIRM + "\n")


def backfill_case_md(root: Path) -> None:
    catalog = all_cases()
    for case_dir in root.glob("**/cases/*"):
        if not case_dir.is_dir():
            continue
        cid = case_dir.name
        case = catalog.get(cid)
        if not case:
            continue
        purpose = case.purpose
        expected = case.expected_result
        if not purpose or not expected:
            continue
        write_case_md(case_dir / "CASE.md", case_id=cid, purpose=str(purpose), expected_result=str(expected))


def backfill_judge_kit(root: Path) -> None:
    copy_judge_bundle(root)
    snapshot_prompts(root)
    write_results_readme(root)
    backfill_case_md(root)


def _profile_dirs(root: Path) -> list[Path]:
    found: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir() and path.name in PROFILE_NAMES:
            found.append(path)
    return sorted(found)


def write_variant_summaries(root: Path) -> None:
    for cases_dir in sorted(root.glob("**/cases")):
        if not cases_dir.is_dir():
            continue
        variant_dir = cases_dir.parent
        trials: list[dict[str, Any]] = []
        for path in sorted(cases_dir.glob("*/trial_*.json")):
            row = _load_trial(path)
            if row:
                trials.append(row)
        summary = build_summary(trials, [])
        summary["prompt_variant"] = variant_dir.name
        if variant_dir.parent.name in PROFILE_NAMES:
            summary["profile"] = variant_dir.parent.name
            summary["model"] = variant_dir.parent.parent.name
        _write_summary_pair(variant_dir, summary)


def write_profile_summaries(root: Path) -> None:
    for profile_dir in _profile_dirs(root):
        by_variant: dict[str, Any] = {}
        rates: list[float] = []
        for child in sorted(profile_dir.iterdir()):
            if not child.is_dir() or child.name == "coding":
                continue
            if not (child / "cases").is_dir():
                continue
            block = _load_json(child / "summary.json")
            rate = _headline_rate(block)
            if rate is None:
                continue
            by_variant[child.name] = {
                "weighted_suite_hard_pass_rate": rate,
                "n_trials": (block or {}).get("n_trials"),
            }
            rates.append(rate)
        coding_rows: list[dict[str, Any]] = []
        coding_root = profile_dir / "coding"
        if coding_root.is_dir():
            for path in sorted(coding_root.glob("*/trial.json")):
                row = _load_trial(path)
                if row:
                    coding_rows.append(row)
        coding_block = build_summary([], coding_rows).get("coding")
        _write_summary_pair(
            profile_dir,
            {
                "weighted_suite_hard_pass_rate": _mean(rates),
                "by_prompt_variant": by_variant,
                "coding": coding_block,
            },
        )


def write_model_summaries(root: Path) -> list[Path]:
    model_dirs: list[Path] = []
    for profile_dir in _profile_dirs(root):
        model_dir = profile_dir.parent
        if model_dir not in model_dirs:
            model_dirs.append(model_dir)
    for model_dir in model_dirs:
        by_profile: dict[str, Any] = {}
        rates: list[float] = []
        for name in ("greedy", "real"):
            block = _load_json(model_dir / name / "summary.json")
            rate = _headline_rate(block)
            if rate is None:
                continue
            by_profile[name] = {"weighted_suite_hard_pass_rate": rate}
            rates.append(rate)
        _write_summary_pair(
            model_dir,
            {
                "weighted_suite_hard_pass_rate": _mean(rates),
                "by_profile": by_profile,
            },
        )
    return model_dirs


def write_root_summary(root: Path, model_dirs: list[Path]) -> None:
    children = [path for path in model_dirs if path != root]
    if not children:
        return
    by_model: dict[str, Any] = {}
    rates: list[float] = []
    for model_dir in children:
        block = _load_json(model_dir / "summary.json")
        rate = _headline_rate(block)
        if rate is None:
            continue
        by_model[model_dir.name] = rate
        rates.append(rate)
    _write_summary_pair(
        root,
        {
            "weighted_suite_hard_pass_rate": _mean(rates),
            "by_model": by_model,
        },
    )


def write_summary_tree(root: Path, *, backfill: bool = False) -> None:
    if backfill:
        backfill_judge_kit(root)
    write_variant_summaries(root)
    write_profile_summaries(root)
    model_dirs = write_model_summaries(root)
    write_root_summary(root, model_dirs)
    write_trial_id_map(root)
    write_reconstructed_manifest(root)


def looks_like_model_tree(path: Path) -> bool:
    for name in PROFILE_NAMES:
        profile = path / name
        if not profile.is_dir():
            continue
        if (profile / "preflight.json").is_file():
            return True
        if (profile / "coding").is_dir():
            return True
        if (profile / "cases").is_dir():
            return True
        if any(child.is_dir() and (child / "cases").is_dir() for child in profile.iterdir()):
            return True
    return False


def summarize_targets(path: Path) -> list[Path]:
    children = [p for p in sorted(path.iterdir()) if p.is_dir()]
    stamps = [p for p in children if STAMP_RE.match(p.name)]
    models = [p for p in children if looks_like_model_tree(p)]
    if stamps and not models:
        return stamps
    if models and not stamps:
        return [path]
    if stamps:
        return stamps
    return [path]


def summarize_path(path: Path) -> list[Path]:
    targets = summarize_targets(path)
    for target in targets:
        write_summary_tree(target, backfill=True)
    return targets
