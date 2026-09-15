from pathlib import Path

import pytest

from bench.config import (
    DEFAULT_CTX_SIZE,
    parse_launcher,
    parse_llama_launcher,
    parse_ollama_launcher,
)

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
