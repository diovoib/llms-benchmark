from __future__ import annotations

import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

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

DEFAULT_CTX_SIZE = 16 * 1024
DEFAULT_LOG_FLUSH_BYTES = 256

_API_KEY_RE = re.compile(
    r"""--api-key(?:\s+|=)(?:"([^"]*)"|'([^']*)'|(\S+))""",
    re.IGNORECASE,
)
_CTX_SIZE_RE = re.compile(
    r"""(?:--ctx-size|--ctx_size)(?:\s+|=)(\d+)""",
    re.IGNORECASE,
)
_CTX_SHORT_RE = re.compile(
    r"""(?:^|\s)-c(?:\s+|=)(\d+)""",
)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if len(raw) >= 2 and raw[1] == 0 and raw[0] != 0:
        return raw.decode("utf-16-le")
    return raw.decode("utf-8", errors="replace")


def _require_number(data: dict[str, Any], dotted: str) -> float:
    cur: Any = data
    parts = dotted.split(".")
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            raise ValueError(f"config missing required key {dotted}")
        cur = cur[part]
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        raise ValueError(f"config {dotted} must be a number")
    return float(cur)


def launcher_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".bat") or name.endswith(".sh"):
        return Path(name).stem.lower()
    return name.lower()


def parse_llama_launcher(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    out: dict[str, Any] = {"kind": "llama", "path": str(path)}
    key = _API_KEY_RE.search(text)
    if key:
        out["api_key"] = next(g for g in key.groups() if g is not None)
    ctx = _CTX_SIZE_RE.search(text)
    if ctx:
        out["ctx_size"] = int(ctx.group(1))
    else:
        short = _CTX_SHORT_RE.search(text)
        if short:
            out["ctx_size"] = int(short.group(1))
        else:
            out["ctx_size"] = DEFAULT_CTX_SIZE
    return out


def _not_implemented(kind: str) -> Callable[[Path], dict[str, Any]]:
    def parse(path: Path) -> dict[str, Any]:
        raise NotImplementedError(f"launcher kind {kind!r} ({path}) is not implemented")

    return parse


LAUNCHER_PARSERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "llama": parse_llama_launcher,
    "ollama": _not_implemented("ollama"),
    "vllm": _not_implemented("vllm"),
    "lmstudio": _not_implemented("lmstudio"),
}


def _with_alt_extension(path: Path) -> Path | None:
    suffix = path.suffix.lower()
    if suffix == ".bat":
        alt = path.with_suffix(".sh")
    elif suffix == ".sh":
        alt = path.with_suffix(".bat")
    else:
        return None
    return alt if alt.is_file() else None


def _default_launcher_name() -> str:
    return "llama.bat" if sys.platform == "win32" else "llama.sh"


def resolve_launcher(config_path: Path, data: dict[str, Any]) -> Path:
    raw = data.get("launcher")
    if not raw:
        raise ValueError("config missing required key launcher")
    p = Path(str(raw))
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(config_path.parent / p)
        candidates.append(config_path.parent.parent / p)
        candidates.append(Path.cwd() / p)

    tried: list[Path] = []
    for cand in candidates:
        tried.append(cand)
        if not cand.suffix:
            plat = cand.with_suffix(".bat" if sys.platform == "win32" else ".sh")
            other = cand.with_suffix(".sh" if sys.platform == "win32" else ".bat")
            if plat.is_file():
                return plat.resolve()
            if other.is_file():
                return other.resolve()
            continue
        if cand.is_file():
            return cand.resolve()
        alt = _with_alt_extension(cand)
        if alt is not None:
            return alt.resolve()
    raise ValueError(f"launcher file not found (config launcher={raw!r}); tried {tried}")


def parse_launcher(path: Path) -> dict[str, Any]:
    kind = launcher_kind(path)
    parser = LAUNCHER_PARSERS.get(kind)
    if parser is None:
        raise NotImplementedError(
            f"unknown launcher kind {kind!r} ({path}); register a parser"
        )
    parsed = parser(path)
    parsed["kind"] = kind
    parsed["path"] = str(path)
    if "ctx_size" not in parsed:
        parsed["ctx_size"] = DEFAULT_CTX_SIZE
    return parsed


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

    _require_number(data, "connect_timeout_s")
    _require_number(data, "suites.tools.request_timeout_s")
    _require_number(data, "suites.agent.request_timeout_s")
    _require_number(data, "suites.coding.request_timeout_s")
    _require_number(data, "suites.tools.max_tokens")
    _require_number(data, "suites.agent.max_tokens")
    if "log_flush_bytes" in data:
        _require_number(data, "log_flush_bytes")
    else:
        data["log_flush_bytes"] = DEFAULT_LOG_FLUSH_BYTES

    launcher = resolve_launcher(config_path, data)
    parsed = parse_launcher(launcher)
    data["_launcher"] = str(launcher)
    data["_launcher_kind"] = parsed.get("kind")
    data["ctx_size"] = int(parsed.get("ctx_size") or DEFAULT_CTX_SIZE)

    env_key = os.environ.get("BENCH_API_KEY")
    if env_key:
        data["api_key"] = env_key
    elif not data.get("api_key"):
        if parsed.get("api_key"):
            data["api_key"] = parsed["api_key"]
    if not data.get("api_key"):
        raise ValueError(
            "api_key missing: set launcher (reads --api-key), env BENCH_API_KEY, or api_key in yaml"
        )
    return data


def snapshot_config(cfg: dict[str, Any]) -> dict[str, Any]:
    snap = deepcopy(cfg)
    if snap.get("api_key"):
        snap["api_key"] = "***"
    snap["launcher"] = {
        "path": snap.get("_launcher") or snap.get("launcher"),
        "kind": snap.get("_launcher_kind"),
        "ctx_size": snap.get("ctx_size"),
        "api_key": "***" if cfg.get("api_key") else None,
    }
    return snap


def resolve_temperature(profile: dict[str, Any], model: dict[str, Any]) -> float:
    temp = profile.get("temperature")
    if temp is None:
        return float(model.get("recommended_temperature", 0.7))
    return float(temp)


def model_dir_name(model: dict[str, Any]) -> str:
    return str(model["name"]).replace("/", "_").replace("\\", "_")
