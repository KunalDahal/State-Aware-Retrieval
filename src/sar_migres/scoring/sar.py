from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from ..utils.errors import ConfigError
from .similarity import normalized_similarity_matrix
from .weights import SARWeights

@dataclass
class ScoreComponents:
    relevance: np.ndarray  # R(d,q)   shape (N,)
    gap_coverage: np.ndarray  # C(d,G_t)
    novelty: np.ndarray  # N(d,E_t)


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


def top_k_indices(scores: np.ndarray, k: int, ids: Sequence[str] | None = None) -> np.ndarray:
    """Indices of the K highest scores, descending; ties use stable document identity when provided."""
    scores = np.asarray(scores)
    k = min(int(k), scores.shape[0])
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if ids is None:
        ids = [str(i) for i in range(scores.shape[0])]
    order = sorted(range(scores.shape[0]), key=lambda i: (-float(scores[i]), str(ids[i])))
    return np.asarray(order[:k], dtype=np.int64)
