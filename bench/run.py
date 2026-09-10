"""Windows-friendly launcher: python bench/run.py run --config bench/config.example.yaml"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bench.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
