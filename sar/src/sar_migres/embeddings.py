"""Embedding models.

* ``SentenceTransformerEmbedder`` - the model fixed by the methodology
  (``sentence-transformers/multi-qa-mpnet-base-cos-v1``, 768-d).  Used for all real runs.
* ``HashingEmbedder`` - deterministic, dependency-free *lexical* hashing embedder used ONLY
  for smoke tests / unit tests.  It is NOT a semantic model and must not be used for any
  scientific claim; runs using it are tagged ``embedder=hashing-...`` in their metadata.
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from functools import lru_cache
from typing import Sequence

import numpy as np

from .errors import ConfigError, DependencyMissingError
from .logging_utils import get_logger

log = get_logger("embeddings")

DEFAULT_MODEL = "sentence-transformers/multi-qa-mpnet-base-cos-v1"


class Embedder(ABC):
    name: str
    dim: int
    is_semantic: bool = True

    @abstractmethod
    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        """Return float32 array of shape (len(texts), dim)."""


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 batch_size: int = 64, expected_dim: int | None = 768, model=None):
        self.name = model_name
        self.batch_size = batch_size
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - depends on env
                raise DependencyMissingError(
                    "sentence-transformers is not installed; `pip install 'sar-migres[st]'` "
                    "or use embedder.type=hashing for the offline smoke test") from e
            log.info("loading embedding model %s (device=%s)", model_name, device or "auto")
            model = SentenceTransformer(model_name, device=device)
        self._model = model
        self.dim = int(model.get_sentence_embedding_dimension())
        if expected_dim is not None and self.dim != expected_dim:
            raise ConfigError(f"{model_name} produced dim {self.dim}, expected {expected_dim}")

    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self._model.encode(texts, batch_size=batch_size or self.batch_size,
                                 convert_to_numpy=True, show_progress_bar=False,
                                 normalize_embeddings=False)
        return np.asarray(out, dtype=np.float32)


_TOKEN = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=200_000)
def _hash_feature(token: str, dim: int, seed: int) -> tuple[int, float]:
    d = hashlib.blake2b(f"{seed}|{token}".encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(d, "little")
    return v % dim, 1.0 if (v >> 63) & 1 else -1.0


class HashingEmbedder(Embedder):
    """Signed hashing of word uni/bi-grams, sub-linear tf, L2-normalised. Deterministic."""

    is_semantic = False

    def __init__(self, dim: int = 256, seed: int = 0, bigram_weight: float = 0.5):
        self.dim, self.seed, self.bigram_weight = int(dim), int(seed), float(bigram_weight)
        self.name = f"hashing-bow-{self.dim}-s{self.seed}-v1"

    def _one(self, text: str) -> np.ndarray:
        toks = _TOKEN.findall(text.lower())
        feats: Counter = Counter(toks)
        big: Counter = Counter(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
        v = np.zeros(self.dim, dtype=np.float64)
        for t, c in feats.items():
            i, s = _hash_feature(t, self.dim, self.seed)
            v[i] += s * (1.0 + math.log(c))
        for t, c in big.items():
            i, s = _hash_feature(t, self.dim, self.seed)
            v[i] += s * self.bigram_weight * (1.0 + math.log(c))
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)

    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._one(t) for t in texts]).astype(np.float32)


def build_embedder(cfg) -> Embedder:
    """cfg: EmbeddingConfig (see config.py)."""
    if cfg.type == "sentence_transformer":
        return SentenceTransformerEmbedder(cfg.model_name, device=cfg.device or None,
                                           batch_size=cfg.batch_size, expected_dim=cfg.expected_dim)
    if cfg.type == "hashing":
        return HashingEmbedder(dim=cfg.hashing_dim, seed=cfg.hashing_seed)
    raise ConfigError(f"unknown embedder type {cfg.type!r}")
