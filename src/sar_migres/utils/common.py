from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

class Timer:
    def __init__(self):
        import time
        self.elapsed = 0.0
        self._t0 = None
    def __enter__(self):
        import time
        self._t0 = time.perf_counter(); return self
    def __exit__(self, exc_type, exc, tb):
        import time
        self.elapsed = time.perf_counter() - self._t0
        return False

def snippet(text: str, max_chars: int = 240) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= max_chars else text[:max_chars - 1].rstrip() + "…"

def _json_default(o: Any):
    if is_dataclass(o): return asdict(o)
    if isinstance(o, Path): return str(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, set): return sorted(o)
    if hasattr(o, "to_dict"): return o.to_dict()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")

def dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, ensure_ascii=False, allow_nan=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
