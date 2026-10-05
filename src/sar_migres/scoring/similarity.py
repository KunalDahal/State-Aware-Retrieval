"""Cosine similarity and the normalised similarity s(x,y) = (1 + cos)/2.

cos(x,y) = phi(x)^T phi(y) / (||phi(x)||_2 ||phi(y)||_2)          (methodology 5.2.4)
s(x,y)   = (1 + cos(x,y)) / 2                                      (Eq. 1)

Zero vectors have no direction; their cosine is defined here as 0 (so s = 0.5)
and a warning is logged by callers that can identify the offending text.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def l2_norms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,ij->i", x, x, dtype=np.float64)).astype(np.float64)


def cosine_matrix(a: np.ndarray, b: np.ndarray, b_norms: np.ndarray | None = None) -> np.ndarray:
    """All-pairs cosine similarity, shape (len(a), len(b)).

    Norms are always computed explicitly (the embedding model is *expected* to
    return unit vectors, but the formula must not depend on that).
    """
    a = np.atleast_2d(np.asarray(a))
    b = np.atleast_2d(np.asarray(b))
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"dimension mismatch: {a.shape[1]} vs {b.shape[1]}")
    an = l2_norms(a)
    bn = l2_norms(b) if b_norms is None else b_norms
    dots = a.astype(np.float32, copy=False) @ b.astype(np.float32, copy=False).T
    denom = np.outer(an, bn)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(denom > _EPS, dots / np.maximum(denom, _EPS), 0.0)
    return np.clip(cos, -1.0, 1.0)


def normalized_similarity_matrix(a: np.ndarray, b: np.ndarray, b_norms: np.ndarray | None = None) -> np.ndarray:
    """s(x,y) = (1 + cos)/2, values in [0, 1]."""
    return (1.0 + cosine_matrix(a, b, b_norms)) / 2.0


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    return float(cosine_matrix(np.asarray(x)[None, :], np.asarray(y)[None, :])[0, 0])


def normalized_similarity(x: np.ndarray, y: np.ndarray) -> float:
    return (1.0 + cosine(x, y)) / 2.0
