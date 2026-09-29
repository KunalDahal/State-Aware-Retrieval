"""State Aware Retrieval scoring - exactly the formulation of methodology.tex Sec. 5.2.

    s(x,y)      = (1 + cos(x,y)) / 2                                   (1)
    R(d,q)      = s(d,q)                                               (2)
    C(d,G_t)    = (1/m) sum_i s(d,g_i);   C = 0 when G_t is empty      (3,4)
    N(d,E_t)    = 1 - max_j s(d,e_j);     N = 1 when E_t is empty      (5,6)
    S_t(d)      = alpha R + beta C + gamma N                           (7)
    alpha,beta,gamma >= 0,  alpha + beta + gamma = 1                   (8)

All functions are vectorised over the document axis; ``score_components`` computes the
three signals with a *single* matrix product against the stacked [q; G_t; E_t] matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .errors import ConfigError
from .similarity import normalized_similarity_matrix

_TOL = 1e-6


@dataclass(frozen=True)
class SARWeights:
    alpha: float  # question relevance
    beta: float  # gap coverage
    gamma: float  # evidence novelty

    def __post_init__(self):
        for n, v in (("alpha", self.alpha), ("beta", self.beta), ("gamma", self.gamma)):
            if not np.isfinite(v) or v < 0:
                raise ConfigError(f"{n} must be a finite value >= 0 (got {v})")
        s = self.alpha + self.beta + self.gamma
        if abs(s - 1.0) > _TOL:
            raise ConfigError(f"alpha + beta + gamma must equal 1 (got {s:.8f})")

    @classmethod
    def from_two(cls, alpha: float, beta: float) -> "SARWeights":
        """Derive gamma = 1 - alpha - beta (two free parameters)."""
        return cls(alpha, beta, 1.0 - alpha - beta)

    @classmethod
    def from_simplex(cls, u: float, v: float) -> "SARWeights":
        """Map (u, v) in [0,1]^2 onto the simplex:
        alpha = u, beta = (1-u) v, gamma = (1-u)(1-v).  Every (u,v) is feasible, which is
        what a box-constrained optimiser (e.g. Bayesian optimisation) needs."""
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise ConfigError("u, v must lie in [0,1]")
        return cls(u, (1.0 - u) * v, (1.0 - u) * (1.0 - v))

    def to_simplex(self) -> tuple[float, float]:
        u = self.alpha
        v = 0.0 if abs(1.0 - u) < 1e-12 else self.beta / (1.0 - u)
        return u, v

    def as_tuple(self) -> tuple[float, float, float]:
        return self.alpha, self.beta, self.gamma


@dataclass
class ScoreComponents:
    relevance: np.ndarray  # R(d,q)   shape (N,)
    gap_coverage: np.ndarray  # C(d,G_t)
    novelty: np.ndarray  # N(d,E_t)


# ---- individual signals (reference implementations, used by tests too) -------------
def question_relevance(doc_emb: np.ndarray, q_emb: np.ndarray, doc_norms=None) -> np.ndarray:
    """R(d,q) = s(d,q)."""
    return normalized_similarity_matrix(doc_emb, np.atleast_2d(q_emb), None)[:, 0] if doc_norms is None \
        else normalized_similarity_matrix(doc_emb, np.atleast_2d(q_emb))[:, 0]


def gap_coverage(doc_emb: np.ndarray, gap_embs: np.ndarray | Sequence) -> np.ndarray:
    """C(d,G_t) = mean_i s(d,g_i); zeros when G_t is empty."""
    n = doc_emb.shape[0]
    g = np.asarray(gap_embs)
    if g.size == 0:
        return np.zeros(n, dtype=np.float64)
    return normalized_similarity_matrix(doc_emb, np.atleast_2d(g)).mean(axis=1)


def evidence_novelty(doc_emb: np.ndarray, ev_embs: np.ndarray | Sequence) -> np.ndarray:
    """N(d,E_t) = 1 - max_j s(d,e_j); ones when E_t is empty."""
    n = doc_emb.shape[0]
    e = np.asarray(ev_embs)
    if e.size == 0:
        return np.ones(n, dtype=np.float64)
    return 1.0 - normalized_similarity_matrix(doc_emb, np.atleast_2d(e)).max(axis=1)


def sar_score(comp: ScoreComponents, w: SARWeights) -> np.ndarray:
    """S_t = alpha R + beta C + gamma N."""
    return w.alpha * comp.relevance + w.beta * comp.gap_coverage + w.gamma * comp.novelty


# ---- fused implementation used by the retriever ------------------------------------
def score_components(doc_emb: np.ndarray, q_emb: np.ndarray, gap_embs: np.ndarray,
                     ev_embs: np.ndarray, doc_norms: np.ndarray | None = None) -> ScoreComponents:
    """R, C, N for a block of documents using ONE matmul against stacked state vectors.

    Mathematically identical to calling the three reference functions above; it avoids
    the O(|D| x |G| x |E|) Python loops the specification warns against.
    """
    q = np.atleast_2d(q_emb)
    m = 0 if gap_embs is None or np.asarray(gap_embs).size == 0 else np.atleast_2d(gap_embs).shape[0]
    n = 0 if ev_embs is None or np.asarray(ev_embs).size == 0 else np.atleast_2d(ev_embs).shape[0]
    blocks = [q]
    if m:
        blocks.append(np.atleast_2d(gap_embs))
    if n:
        blocks.append(np.atleast_2d(ev_embs))
    stacked = np.concatenate(blocks, axis=0)
    s = normalized_similarity_matrix(stacked, doc_emb, None).T if doc_norms is None else \
        (1.0 + _cos_block(doc_emb, doc_norms, stacked)) / 2.0
    R = s[:, 0].astype(np.float64)
    C = s[:, 1:1 + m].mean(axis=1).astype(np.float64) if m else np.zeros(s.shape[0])
    N = (1.0 - s[:, 1 + m:1 + m + n].max(axis=1)).astype(np.float64) if n else np.ones(s.shape[0])
    return ScoreComponents(R, C, N)


def _cos_block(doc_emb: np.ndarray, doc_norms: np.ndarray, stacked: np.ndarray) -> np.ndarray:
    from .similarity import cosine_matrix

    return cosine_matrix(doc_emb, stacked, None) if doc_norms is None else _cos_with_norms(doc_emb, doc_norms, stacked)


def _cos_with_norms(doc_emb: np.ndarray, doc_norms: np.ndarray, stacked: np.ndarray) -> np.ndarray:
    from .similarity import l2_norms

    sn = l2_norms(stacked)
    dots = doc_emb.astype(np.float32, copy=False) @ stacked.astype(np.float32, copy=False).T
    denom = np.outer(doc_norms, sn)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(denom > 1e-12, dots / np.maximum(denom, 1e-12), 0.0)
    return np.clip(cos, -1.0, 1.0)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the K highest scores, descending; ties broken by lower index (deterministic)."""
    scores = np.asarray(scores)
    k = min(int(k), scores.shape[0])
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.arange(scores.shape[0])
    order = np.lexsort((idx, -scores))
    return order[:k]
