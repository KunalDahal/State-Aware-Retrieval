from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def peak_rss_mb() -> float | None:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(rss) / (1024.0 if os.uname().sysname == "Darwin" else 1_048_576.0)
    except Exception:
        return None
