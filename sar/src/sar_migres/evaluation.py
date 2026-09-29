"""Evaluation interfaces - FUTURE / OCTOBER EXPERIMENTS.

The metric *formulas* of methodology Sec. 5.6.4-5.6.7 are implemented as pure functions (and unit
tested on hand-made inputs).  NOTHING in this repository runs them on benchmark data or reports
numbers from them; that is the October stage.  RGB robustness and the comparative studies are
interfaces only.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

FUTURE_NOTICE = "FUTURE (October) experiment: interface only, no results are produced or claimed by this repository."


# ---- retrieval metrics (5.6.4) -------------------------------------------------------------------
def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    return len(set(retrieved[:k]) & rel) / len(rel) if rel else float("nan")


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(relevant)) / k


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for i, d in enumerate(retrieved, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevance: dict[str, float], k: int) -> float:
    """nDCG@K with gain 2^rel - 1 and log2(i+1) discount (methodology formula); IDCG from the ideal ordering."""
    dcg = sum((2 ** relevance.get(d, 0.0) - 1) / np.log2(i + 1) for i, d in enumerate(retrieved[:k], 1))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / np.log2(i + 1) for i, r in enumerate(ideal, 1))
    return float(dcg / idcg) if idcg > 0 else float("nan")


# ---- iterative-state metrics (5.6.5) -----------------------------------------------------------------
def cumulative_recall(per_iter: Sequence[Sequence[str]], relevant: Iterable[str]) -> list[float]:
    rel, seen, out = set(relevant), set(), []
    for T in per_iter:
        seen |= set(T)
        out.append(len(seen & rel) / len(rel) if rel else float("nan"))
    return out


def marginal_recall_gain(cum: Sequence[float]) -> list[float]:
    return [c - (cum[i - 1] if i else 0.0) for i, c in enumerate(cum)]


def new_document_rate(per_iter: Sequence[Sequence[str]]) -> list[float]:
    seen, out = set(), []
    for T in per_iter:
        out.append(len(set(T) - seen) / len(T) if len(T) else float("nan"))
        seen |= set(T)
    return out


def retrieval_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float("nan")


# ---- answer metrics (5.6.6) --------------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: Sequence[str]) -> float:
    return float(any(_norm(pred) == _norm(g) for g in golds))


def token_f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    P, R = common / len(p), common / len(g)
    return 2 * P * R / (P + R)


# ---- cost (5.6.7) ----------------------------------------------------------------------------------------------
def n_documents_scored(corpus_size: int, iterations: int) -> int:
    return corpus_size * iterations


def n_similarity_calculations(corpus_size: int, m_gaps: int, n_evidence: int) -> int:
    return corpus_size * (1 + m_gaps + n_evidence)


def retrieval_overhead_pct(l_sar: float, l_baseline: float) -> float:
    return (l_sar - l_baseline) / l_baseline * 100.0


class RGBRobustnessEvaluation:
    """FUTURE: noise robustness / negative rejection / information integration / counterfactual robustness
    (Chen et al. 2024).  Data loading exists (``datasets.load_rgb``); the protocol is not implemented."""

    def __init__(self, *_, **__):
        raise NotImplementedError(FUTURE_NOTICE + " RGB protocol (noise ratios, rejection prompts, counterfactual metrics ED/CR) to be added in October.")


class AblationStudy:
    """FUTURE: SAR-Full / SAR-Without-{Relevance,Gap,Evidence} / Question-Only.
    ``ablation_weights`` only lists the weight settings that correspond to those variants."""

    @staticmethod
    def ablation_weights(w) -> dict[str, tuple[float, float, float]]:
        def renorm(a, b, g):
            s = a + b + g
            return (a / s, b / s, g / s)
        return {"SAR-Full": (w.alpha, w.beta, w.gamma),
                "SAR-Without-Relevance": renorm(0, w.beta, w.gamma),
                "SAR-Without-Gap": renorm(w.alpha, 0, w.gamma),
                "SAR-Without-Evidence": renorm(w.alpha, w.beta, 0),
                "Question-Only": (1.0, 0.0, 0.0)}
