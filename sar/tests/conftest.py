import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import os
import pytest


@pytest.fixture(autouse=True)
def _cwd_is_project_root(monkeypatch):
    """Tests reference configs/ and data/ relative to the project root."""
    monkeypatch.chdir(ROOT)
