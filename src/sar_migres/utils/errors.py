"""Explicit exception types + a retry helper (nothing here swallows errors silently)."""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from .logging import get_logger

log = get_logger("errors")
T = TypeVar("T")


class SarMigresError(Exception):
    """Base class."""


class ConfigError(SarMigresError): ...
class MissingCredentialError(SarMigresError): ...
class DependencyMissingError(SarMigresError): ...
class CorpusFormatError(SarMigresError): ...
class DatasetNotFoundError(SarMigresError): ...
class EmbeddingCacheCorruptionError(SarMigresError): ...
class StoreIncompatibleError(SarMigresError): ...
class VectorStoreConnectionError(SarMigresError): ...
class LLMError(SarMigresError): ...
class LLMResponseError(LLMError): ...
class UpstreamMigresError(SarMigresError): ...


_TRANSIENT_HINTS = ("429", "rate", "quota", "timeout", "timed out", "temporar", "unavailable",
                    "connection", "503", "502", "500", "overloaded", "resource_exhausted")


def looks_transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(h in text for h in _TRANSIENT_HINTS)


def retry_call(fn: Callable[[], T], *, attempts: int = 5, base_delay: float = 1.0,
               max_delay: float = 30.0, what: str = "call",
               is_transient: Callable[[BaseException], bool] = looks_transient,
               sleep: Callable[[float], None] = time.sleep) -> T:
    """Retry ``fn`` on transient errors (rate limits / timeouts) with jittered backoff.

    Non-transient errors are re-raised immediately; after the final attempt the last
    exception is re-raised unchanged so callers can classify it.
    """
    last: BaseException | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not is_transient(exc) or i == attempts:
                log.error("%s failed (attempt %d/%d): %s: %s", what, i, attempts, type(exc).__name__, exc)
                raise
            delay = min(max_delay, base_delay * 2 ** (i - 1)) * (0.5 + random.random() / 2)
            log.warning("%s transient failure (attempt %d/%d): %s; retrying in %.1fs", what, i, attempts, exc, delay)
            sleep(delay)
    raise last  # pragma: no cover
