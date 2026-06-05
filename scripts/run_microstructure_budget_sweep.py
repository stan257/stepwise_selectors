#!/usr/bin/env python3
"""CLI wrapper for exact-k microstructure budget sweeps."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.microstructure_budget_sweep import main


if __name__ == "__main__":
    main()
