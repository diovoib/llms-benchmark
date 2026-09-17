from __future__ import annotations

import json
from pathlib import Path

from bench.progress import trial_label
from bench.summary import (
    _is_profile_dir,
    looks_like_model_tree,
    write_summary_tree,
)


def _write_trial(path: Path, *, case_id: str, suite: str, variant: str, hard_pass: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "trial_id": f"t_{case_id}_{variant}",
                "case_id": case_id,
                "suite": suite,
                "repeat": 0,
                "prompt_variant": variant,
                "score": {"hard_pass": hard_pass, "violations": []},
            }
        ),
        encoding="utf-8",
    )


def _plant_profile(root: Path, model: str, profile: str, *, variant: str = "neutral") -> Path:
    profile_dir = root / model / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "preflight.json").write_text('{"ok": true}', encoding="utf-8")
    _write_trial(
        profile_dir / variant / "cases" / "T01_en" / "trial_001.json",
        case_id="T01_en",
        suite="tools",
        variant=variant,
        hard_pass=True,
    )
    return profile_dir


def test_looks_like_model_tree_when_only_agentic_profile_exists(tmp_path: Path) -> None:
    _plant_profile(tmp_path, "m", "agentic")
    assert looks_like_model_tree(tmp_path / "m") is True
    assert looks_like_model_tree(tmp_path) is False


def test_prompt_variant_folder_is_not_a_profile_directory(tmp_path: Path) -> None:
    profile_dir = _plant_profile(tmp_path, "m", "agentic")
    variant_dir = profile_dir / "neutral"
    assert _is_profile_dir(profile_dir) is True
    assert _is_profile_dir(variant_dir) is False


def test_coding_only_creative_folder_is_a_profile_directory(tmp_path: Path) -> None:
    coding = tmp_path / "m" / "creative" / "coding" / "C01" / "trial_001"
    coding.mkdir(parents=True)
    (coding / "trial.json").write_text("{}", encoding="utf-8")
    assert _is_profile_dir(tmp_path / "m" / "creative") is True
    assert looks_like_model_tree(tmp_path / "m") is True


def test_model_summary_includes_agentic_and_creative(tmp_path: Path) -> None:
    _plant_profile(tmp_path, "m", "greedy")
    _plant_profile(tmp_path, "m", "agentic")
    _plant_profile(tmp_path, "m", "creative")
    write_summary_tree(tmp_path, backfill=False)
    summary = json.loads((tmp_path / "m" / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["by_profile"]) == {"agentic", "creative", "greedy"}
    variant = json.loads((tmp_path / "m" / "agentic" / "neutral" / "summary.json").read_text(encoding="utf-8"))
    assert variant["profile"] == "agentic"
    assert variant["model"] == "m"


def test_legacy_real_profile_folder_still_summarizes(tmp_path: Path) -> None:
    _plant_profile(tmp_path, "m", "real")
    write_summary_tree(tmp_path, backfill=False)
    summary = json.loads((tmp_path / "m" / "summary.json").read_text(encoding="utf-8"))
    assert "real" in summary["by_profile"]


def test_trial_label_keeps_full_agentic_and_creative_names() -> None:
    greedy = trial_label(
        case_id="T01_en",
        suite="tools",
        profile="greedy",
        variant="neutral",
        repeat=0,
        nrep=2,
        seed=1,
    )
    agentic = trial_label(
        case_id="T01_en",
        suite="tools",
        profile="agentic",
        variant="neutral",
        repeat=0,
        nrep=5,
        seed=1,
    )
    creative = trial_label(
        case_id="T01_en",
        suite="tools",
        profile="creative",
        variant="neutral",
        repeat=0,
        nrep=2,
        seed=1,
    )
    assert "greedy" in greedy
    assert "agentic" in agentic
    assert "creative" in creative
    assert "agenti " not in agentic
    assert "creativ " not in creative
