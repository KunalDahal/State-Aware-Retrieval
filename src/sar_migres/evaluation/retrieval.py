from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np

def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant); got = set(retrieved[:k])
    return len(got & rel) / len(rel) if rel else float("nan")


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        return float("nan")
    return len(set(retrieved[:k]) & set(relevant)) / k


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for i, did in enumerate(retrieved, 1):
        if did in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevance: dict[str, float], k: int) -> float:
    def gain(r): return 2.0 ** float(r) - 1.0
    dcg = sum(gain(relevance.get(d, 0.0)) / np.log2(i + 1) for i, d in enumerate(retrieved[:k], 1))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(gain(r) / np.log2(i + 1) for i, r in enumerate(ideal, 1))
    return float(dcg / idcg) if idcg > 0 else float("nan")


def mean_metric(values: Sequence[float]) -> float:
    a = np.asarray(values, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def retrieval_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    u = a | b
    return len(a & b) / len(u) if u else float("nan")

def cumulative_recall(per_iter: Sequence[Sequence[str]], relevant: Iterable[str]) -> list[float]:
    rel, seen, out = set(relevant), set(), []
    for T in per_iter:
        seen |= set(T)
        out.append(len(seen & rel) / len(rel) if rel else float("nan"))
    return out


def marginal_recall_gain(cum: Sequence[float]) -> list[float]:
    return [float(c - (cum[i - 1] if i else 0.0)) for i, c in enumerate(cum)]


def new_document_rate(per_iter: Sequence[Sequence[str]]) -> list[float]:
    seen, out = set(), []
    for T in per_iter:
        Tset = set(T)
        out.append(len(Tset - seen) / len(Tset) if Tset else float("nan"))
        seen |= Tset
    return out
