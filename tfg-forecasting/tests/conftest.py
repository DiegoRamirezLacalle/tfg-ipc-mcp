"""Make the monorepo root importable so tests can use shared/."""

import sys
from pathlib import Path

MONOREPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MONOREPO))
