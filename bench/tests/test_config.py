from pathlib import Path

import pytest
import yaml

from bench.config import (
    DEFAULT_CTX_SIZE,
    parse_launcher,
    parse_llama_launcher,
    parse_ollama_launcher,
    resolve_temperature,
)
from bench.client import _build_request_body
from bench.preflight import _default_generation_temperature

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "rel",
    ["llama.bat.example", "llama.sh.example"],
)
def test_parse_llama_examples(rel: str) -> None:
    path = ROOT / rel
    settings = parse_llama_launcher(path)
    assert settings.kind == "llama"
    assert settings.ctx_size == 16384
    assert settings.api_key == "<api_key_for_llama>"


@pytest.mark.parametrize(
    "rel",
    ["ollama.bat.example", "ollama.sh.example"],
)
def test_parse_ollama_examples(rel: str) -> None:
    path = ROOT / rel
    settings = parse_ollama_launcher(path)
    assert settings.kind == "ollama"
    assert settings.ctx_size == 16384
    assert settings.api_key == "<api_key_for_ollama>"


def test_parse_launcher_dispatches_by_filename(tmp_path: Path) -> None:
    llama = tmp_path / "llama.bat"
    ollama = tmp_path / "ollama.sh"
    llama.write_text((ROOT / "llama.bat.example").read_text(encoding="utf-8"), encoding="utf-8")
    ollama.write_text((ROOT / "ollama.sh.example").read_text(encoding="utf-8"), encoding="utf-8")
    assert parse_launcher(llama).kind == "llama"
    assert parse_launcher(ollama).kind == "ollama"


def test_ollama_without_context_uses_default(tmp_path: Path) -> None:
    script = tmp_path / "ollama.bat"
    script.write_text("set OLLAMA_API_KEY=secret\nollama serve\n", encoding="utf-8")
    settings = parse_ollama_launcher(script)
    assert settings.api_key == "secret"
    assert settings.ctx_size == DEFAULT_CTX_SIZE


def test_real_profile_uses_numeric_recommended_temperature() -> None:
    assert resolve_temperature({"temperature": None}, {"recommended_temperature": 0.7}) == 0.7


def test_real_profile_omits_temperature_when_recommended_is_absent() -> None:
    assert resolve_temperature({"temperature": None}, {"name": "m"}) is None


def test_real_profile_omits_temperature_when_recommended_is_null() -> None:
    assert resolve_temperature({"temperature": None}, {"recommended_temperature": None}) is None


def test_example_config_profiles_are_greedy_agentic_creative() -> None:
    data = yaml.safe_load((ROOT / "bench" / "config.yaml.example").read_text(encoding="utf-8"))
    assert set(data["profiles"]) == {"greedy", "agentic", "creative"}
    for suite in ("tools", "agent", "coding"):
        assert set(data["suites"][suite]["repeats"]) == {"greedy", "agentic", "creative"}


def test_greedy_profile_keeps_zero_temperature() -> None:
    assert resolve_temperature({"temperature": 0.0}, {}) == 0.0


def test_recommended_zero_is_sent() -> None:
    assert resolve_temperature({"temperature": None}, {"recommended_temperature": 0}) == 0.0


def _chat_body(*, temperature: float | None) -> dict:
    return _build_request_body(
        model="m",
        messages=[],
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
        temperature=temperature,
        top_p=1,
        top_k=0,
        min_p=0,
        repeat_penalty=1,
        seed=1,
        chat_template_kwargs=None,
        max_tokens=None,
        stream=False,
    )


def test_chat_body_includes_numeric_temperature() -> None:
    body = _chat_body(temperature=0.7)
    assert body["temperature"] == 0.7


def test_chat_body_omits_temperature_when_unresolved() -> None:
    body = _chat_body(temperature=None)
    assert "temperature" not in body


def test_llama_props_expose_default_generation_temperature() -> None:
    props = {"default_generation_settings": {"params": {"temperature": 0.8}}}
    assert _default_generation_temperature(props) == 0.8


def test_llama_props_without_temperature_yield_none() -> None:
    assert _default_generation_temperature({"default_generation_settings": {}}) is None


def test_non_object_llama_props_yield_no_default_temperature() -> None:
    assert _default_generation_temperature("broken") is None
