"""Small shared helpers (hashing, seeds, run ids, atomic writes)."""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    """Seed Python and NumPy (and torch if it is already imported)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    import sys

    if "torch" in sys.modules:  # never import torch just to seed it
        sys.modules["torch"].manual_seed(seed)


def make_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def snippet(text: str, n: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


class Timer:
    """Context-manager wall-clock timer (perf_counter)."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        return False


def _json_default(o: Any):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return repr(o)


def dumps(obj: Any, **kw) -> str:
    return json.dumps(obj, default=_json_default, ensure_ascii=False, **kw)


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    """Write via temp file + rename so interrupted runs never leave half files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def peak_rss_mb() -> float | None:
    """Peak resident set size in MB (Linux/macOS); None if unavailable."""
    try:
        import resource
        import sys

        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024
    except Exception:  # pragma: no cover
        return None
