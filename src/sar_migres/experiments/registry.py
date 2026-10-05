from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
from ..utils.config import Config

def load_experiment_plan(path: str | Path = "configs/experiments.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def baseline_overrides(method: str, *, k: int, run_name: str) -> dict[str, Any]:
    return {
        "retrieval": {"method": method},
        "sar": {"k": k},
        "baseline": {"k": k},
        "run": {"name": run_name, "resume": False},
    }


def ablation_overrides(name: str, weights: tuple[float, float, float], *, k: int, run_name: str) -> dict[str, Any]:
    a, b, g = weights
    return {
        "retrieval": {"method": "sar"},
        "sar": {"alpha": a, "beta": b, "gamma": g, "k": k, "weighting": "static", "dynamic_rule": ""},
        "run": {"name": run_name, "resume": False},
    }
