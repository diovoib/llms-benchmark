from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
BENCH_ROOT = TESTS_DIR.parent
SRC = BENCH_ROOT / "src"
for path in (SRC, TESTS_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


@pytest.fixture(scope="session")
def bench_root() -> Path:
    return BENCH_ROOT


@pytest.fixture(scope="session")
def weather_tokens(bench_root: Path) -> dict[str, str]:
    path = bench_root / "fixtures" / "tokens.json"
    return json.loads(path.read_text(encoding="utf-8"))
