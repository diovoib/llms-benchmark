from __future__ import annotations

import os
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

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
_LLAMA_CTX_SIZE_RE = re.compile(
    r"""--ctx-size(?:\s+|=)(\d+)""",
    re.IGNORECASE,
)
_LLAMA_API_KEY_RE = re.compile(
    r"""--api-key(?:\s+|=)(?:"([^"]*)"|'([^']*)'|(\S+))""",
    re.IGNORECASE,
)
_OLLAMA_CTX_SIZE_RE = re.compile(
    r"""(?:^|\s)(?:export|set)\s+OLLAMA_CONTEXT_LENGTH\s*=\s*(\d+)""",
    re.IGNORECASE | re.MULTILINE,
)
_OLLAMA_API_KEY_RE = re.compile(
    r"""(?:^|\s)(?:export|set)\s+OLLAMA_API_KEY\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+))""",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class LauncherSettings:
    path: Path
    kind: Literal["llama", "ollama"]
    ctx_size: int
    api_key: str | None = None


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if len(raw) >= 2 and raw[1] == 0 and raw[0] != 0:
        return raw.decode("utf-16-le")
    return raw.decode("utf-8", errors="replace")


def _regex_captured(match: re.Match[str]) -> str:
    return next(g for g in match.groups() if g is not None)


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


def parse_llama_launcher(path: Path) -> LauncherSettings:
    text = _read_text(path)
    ctx_size = DEFAULT_CTX_SIZE
    if match := _LLAMA_CTX_SIZE_RE.search(text):
        ctx_size = int(match.group(1))
    api_key = None
    if match := _LLAMA_API_KEY_RE.search(text):
        api_key = _regex_captured(match)
    return LauncherSettings(
        path=path.resolve(),
        kind="llama",
        ctx_size=ctx_size,
        api_key=api_key,
    )


def parse_ollama_launcher(path: Path) -> LauncherSettings:
    text = _read_text(path)
    ctx_size = DEFAULT_CTX_SIZE
    if match := _OLLAMA_CTX_SIZE_RE.search(text):
        ctx_size = int(match.group(1))
    api_key = None
    if match := _OLLAMA_API_KEY_RE.search(text):
        api_key = _regex_captured(match)
    return LauncherSettings(
        path=path.resolve(),
        kind="ollama",
        ctx_size=ctx_size,
        api_key=api_key,
    )


def _not_implemented(kind: str) -> Callable[[Path], LauncherSettings]:
    def parse(path: Path) -> LauncherSettings:
        raise NotImplementedError(f"launcher kind {kind!r} ({path}) is not implemented")

    return parse


LAUNCHER_PARSERS: dict[str, Callable[[Path], LauncherSettings]] = {
    "llama": parse_llama_launcher,
    "ollama": parse_ollama_launcher,
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


def parse_launcher(path: Path) -> LauncherSettings:
    kind = launcher_kind(path)
    parser = LAUNCHER_PARSERS.get(kind)
    if parser is None:
        raise NotImplementedError(
            f"unknown launcher kind {kind!r} ({path}); register a parser"
        )
    return parser(path)


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
    data["_launcher"] = str(parsed.path)
    data["_launcher_kind"] = parsed.kind
    data["ctx_size"] = parsed.ctx_size

    data.pop("api_key", None)
    env_key = os.environ.get("BENCH_API_KEY")
    if env_key:
        data["api_key"] = env_key
    else:
        data["api_key"] = parsed.api_key or ""
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


def resolve_temperature(profile: dict[str, Any], model: dict[str, Any]) -> float | None:
    temp = profile.get("temperature")
    if temp is not None:
        return float(temp)
    recommended = model.get("recommended_temperature")
    if recommended is None:
        return None
    return float(recommended)


_MODEL_DIR_UNSAFE = str.maketrans({ch: "_" for ch in '\\/:*?"<>|'})


def model_dir_name(model: dict[str, Any]) -> str:
    return str(model["name"]).translate(_MODEL_DIR_UNSAFE)
