"""Logging with secret redaction (API keys must never reach a log file)."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

_SECRET_ENV_NAMES = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(AIza[0-9A-Za-z_\-]{20,})"),
    re.compile(r"(pcsk_[A-Za-z0-9_\-]{8,})"),
]


class RedactingFilter(logging.Filter):
    """Masks secret-looking tokens and the *values* of secret env vars."""

    def _secret_values(self):
        for k, v in os.environ.items():
            if v and len(v) >= 8 and any(s in k.upper() for s in _SECRET_ENV_NAMES):
                yield v

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for v in self._secret_values():
            msg = msg.replace(v, "***REDACTED***")
        for p in _PATTERNS:
            msg = p.sub("***REDACTED***", msg)
        record.msg, record.args = msg, None
        return True


def setup_logging(level: str = "INFO", log_file: str | os.PathLike | None = None) -> logging.Logger:
    logger = logging.getLogger("sar_migres")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.addFilter(RedactingFilter())
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(RedactingFilter())
        logger.addHandler(fh)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"sar_migres.{name}")
