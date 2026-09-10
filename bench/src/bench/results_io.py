from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def trial_txt(trial: dict[str, Any]) -> str:
    lines = [
        f"trial_id: {trial.get('trial_id')}",
        f"case_id: {trial.get('case_id')}",
        f"prompt_variant: {trial.get('prompt_variant')}",
        f"repeat: {trial.get('repeat')}",
        f"seed: {trial.get('seed')}",
        f"hard_pass: {trial.get('score', {}).get('hard_pass')}",
        f"violations: {trial.get('score', {}).get('violations')}",
        f"excluded_from_rate: {trial.get('score', {}).get('excluded_from_rate')}",
        f"invalidated: {trial.get('score', {}).get('invalidated')}",
        f"latency_s: {trial.get('latency_s')}",
        f"prompt_tokens: {trial.get('prompt_tokens')}",
        f"completion_tokens: {trial.get('completion_tokens')}",
        f"first_tool_response_latency_s: {trial.get('first_tool_response_latency_s')}",
        f"finish_reason: {trial.get('finish_reason')}",
        "-" * 50,
        "TRANSCRIPT",
        trial.get("transcript_text") or "",
    ]
    return "\n".join(str(x) for x in lines) + "\n"
