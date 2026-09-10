from __future__ import annotations

from pathlib import Path


def bench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def src_root() -> Path:
    return Path(__file__).resolve().parents[1]


def coding_assets() -> Path:
    return bench_root() / "coding"
