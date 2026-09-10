from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_PROFILE_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "seed_policy",
    "thinking",
    "chat_template_kwargs",
)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if len(raw) >= 2 and raw[1] == 0 and raw[0] != 0:
        return raw.decode("utf-16-le")
    return raw.decode("utf-8", errors="replace")


def parse_llama_bat(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    out: dict[str, Any] = {}
    key = re.search(
        r"""--api-key(?:\s+|=)(?:"([^"]*)"|'([^']*)'|(\S+))""",
        text,
        re.IGNORECASE,
    )
    if key:
        out["api_key"] = next(g for g in key.groups() if g is not None)
    return out


def resolve_llama_bat(config_path: Path, data: dict[str, Any]) -> Path | None:
    raw = data.get("llama_bat")
    candidates: list[Path] = []
    if raw:
        p = Path(str(raw))
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(config_path.parent / p)
            candidates.append(config_path.parent.parent / p)
    else:
        candidates.append(config_path.parent.parent / "llama.bat")
        candidates.append(Path.cwd() / "llama.bat")
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    for name, profile in (data.get("profiles") or {}).items():
        missing = [f for f in REQUIRED_PROFILE_FIELDS if f not in profile]
        if missing:
            raise ValueError(f"profile {name} missing fields: {missing}")
        if profile["seed_policy"] not in ("fixed", "unique"):
            raise ValueError(f"profile {name}: seed_policy must be fixed|unique")
        if profile["seed_policy"] == "fixed" and "seed" not in profile:
            raise ValueError(f"profile {name}: fixed seed_policy requires seed")
    if not data.get("models"):
        raise ValueError("config.models is required")

    bat = resolve_llama_bat(config_path, data)
    parsed = parse_llama_bat(bat) if bat else {}
    data["_llama_bat"] = str(bat) if bat else None

    env_key = os.environ.get("BENCH_API_KEY")
    if env_key:
        data["api_key"] = env_key
    elif not data.get("api_key"):
        if parsed.get("api_key"):
            data["api_key"] = parsed["api_key"]
    if not data.get("api_key"):
        raise ValueError("api_key missing: set llama_bat (reads --api-key), env BENCH_API_KEY, or api_key in yaml")
    return data


def snapshot_config(cfg: dict[str, Any]) -> dict[str, Any]:
    snap = deepcopy(cfg)
    if snap.get("api_key"):
        snap["api_key"] = "***"
    return snap


def resolve_temperature(profile: dict[str, Any], model: dict[str, Any]) -> float:
    temp = profile.get("temperature")
    if temp is None:
        return float(model.get("recommended_temperature", 0.7))
    return float(temp)


def model_dir_name(model: dict[str, Any]) -> str:
    return str(model["name"]).replace("/", "_").replace("\\", "_")
