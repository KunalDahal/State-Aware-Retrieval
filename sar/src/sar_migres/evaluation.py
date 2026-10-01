"""Operational evaluation helpers matching methodology Sec. 5.6.4-5.6.7."""
from __future__ import annotations

from pathlib import Path

import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


# ---- retrieval metrics -------------------------------------------------------------------------
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


# ---- iterative state metrics ------------------------------------------------------------------
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


def retrieval_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    u = a | b
    return len(a & b) / len(u) if u else float("nan")


# ---- answer metrics ---------------------------------------------------------------------------
_STOP = {"a", "an", "the"}

def _norm(s: Any) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split() if t not in _STOP]
    return " ".join(tokens)


def gold_aliases(gold: Any) -> list[str]:
    if gold is None:
        return []
    if isinstance(gold, dict):
        for key in ("answers", "aliases", "answer", "value", "text"):
            if key in gold:
                return gold_aliases(gold[key])
    if isinstance(gold, (list, tuple, set)):
        out = []
        for x in gold:
            out.extend(gold_aliases(x))
        return out
    return [str(gold)]


def _bool_norm(x: Any) -> bool | None:
    s = _norm(x)
    if s in {"yes", "true", "1", "correct"}:
        return True
    if s in {"no", "false", "0", "incorrect"}:
        return False
    return None


def exact_match(pred: Any, golds: Any) -> float:
    if isinstance(golds, (bool, np.bool_)):
        pb = _bool_norm(pred)
        return float(pb is not None and pb == bool(golds))
    aliases = gold_aliases(golds)
    if not aliases:
        return 0.0
    return float(any(_norm(pred) == _norm(g) for g in aliases))


def answer_accuracy(pred: Any, gold: Any) -> float:
    """Accuracy metric from methodology Sec. 5.6.6.

    For extractive QA this is exact-answer correctness; for boolean/decision QA it also accepts
    normalized yes/no or true/false forms.  Multiple gold aliases are treated as equivalent.
    """
    return exact_match(pred, gold)


def token_f1(pred: Any, gold: Any) -> float:
    aliases = gold_aliases(gold)
    if not aliases:
        return 0.0
    p = _norm(pred).split()
    best = 0.0
    for g0 in aliases:
        g = _norm(g0).split()
        common = sum((Counter(p) & Counter(g)).values())
        if common == 0:
            cur = 1.0 if not p and not g else 0.0
        else:
            P, R = common / len(p), common / len(g)
            cur = 2 * P * R / (P + R)
        best = max(best, cur)
    return best


def evaluate_answers(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(records)
    evaluable = [r for r in rows if r.get("gold", r.get("gold_answer")) is not None]
    if not evaluable:
        return {"exact_match": float("nan"), "accuracy": float("nan"), "token_f1": float("nan"), "n": 0, "n_total": len(rows)}
    em = [exact_match(r.get("prediction", r.get("final_answer")), r.get("gold", r.get("gold_answer"))) for r in evaluable]
    acc = [answer_accuracy(r.get("prediction", r.get("final_answer")), r.get("gold", r.get("gold_answer"))) for r in evaluable]
    f1 = [token_f1(r.get("prediction", r.get("final_answer")), r.get("gold", r.get("gold_answer"))) for r in evaluable]
    return {"exact_match": float(np.mean(em)), "accuracy": float(np.mean(acc)), "token_f1": float(np.mean(f1)), "n": len(evaluable), "n_total": len(rows)}



# ---- paired statistical analysis --------------------------------------------------------------
def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], *, n_resamples: int = 2000,
                        seed: int = 13, confidence: float = 0.95) -> dict[str, float]:
    """Paired bootstrap CI for the mean difference a-b."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return {"n": 0, "mean_difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    diffs = x - y
    samples = np.empty(int(n_resamples), dtype=float)
    for i in range(len(samples)):
        samples[i] = float(np.mean(diffs[rng.integers(0, len(diffs), size=len(diffs))]))
    alpha = 1.0 - confidence
    lo, hi = np.quantile(samples, [alpha/2.0, 1.0-alpha/2.0])
    return {"n": int(len(diffs)), "mean_difference": float(np.mean(diffs)), "ci_low": float(lo), "ci_high": float(hi)}


def paired_permutation_pvalue(a: Sequence[float], b: Sequence[float], *, n_resamples: int = 2000,
                              seed: int = 13) -> dict[str, float]:
    """Two-sided paired randomisation test for the mean difference."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    diffs = (x[mask] - y[mask]).astype(float)
    if len(diffs) == 0:
        return {"n": 0, "observed_difference": float("nan"), "p_value": float("nan")}
    obs = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(int(n_resamples)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(diffs))
        cur = float(np.mean(diffs * signs))
        extreme += int(abs(cur) >= abs(obs))
    p = (extreme + 1) / (int(n_resamples) + 1)
    return {"n": int(len(diffs)), "observed_difference": obs, "p_value": float(p)}


def filter_interaction_metrics(run_dir: str | Path) -> dict[str, float]:
    """Summarise what the common MIGRES filter retained after each retriever's ranking."""
    from .experiment_runner import load_run_traces  # local import: experiment_runner imports this module
    rows = load_run_traces(run_dir)
    total = kept = top1_kept = 0
    for t in rows:
        docs = t.get("documents", [])
        after = t.get("migres_state_after", {})
        if "filter_kept_doc_ids" not in after:
            continue
        ids = [str(d.get("doc_id")) for d in docs]
        kset = set(after.get("filter_kept_doc_ids", []))
        total += len(ids)
        kept += sum(i in kset for i in ids)
        top1_kept += int(bool(ids) and ids[0] in kset)
    return {"retrieved_docs_seen": float(total), "filter_retention_rate": kept / total if total else float("nan"),
            "top1_filter_survival_rate": top1_kept / len(rows) if rows else float("nan")}

# ---- RGB robustness ---------------------------------------------------------------------------
@dataclass
class RGBCase:
    setting: str
    question: str
    gold: Any
    noise_level: float | None = None
    negative: bool = False
    counterfactual: bool = False
    metadata: dict[str, Any] | None = None


class RGBRobustnessEvaluation:
    """Evaluator for the four RGB settings named in methodology Sec. 5.6.6.

    The class consumes *model outputs* rather than making LLM calls itself.  This keeps retrieval,
    MIGRES execution and RGB evaluation separated and allows SAR and every baseline to use the same
    evaluation protocol.
    """

    SETTINGS = {"noise_robustness", "negative_rejection", "information_integration", "counterfactual_robustness"}

    def __init__(self, cases: Sequence[RGBCase]):
        self.cases = list(cases)

    @staticmethod
    def from_records(records: Sequence[dict[str, Any]], setting: str | None = None) -> "RGBRobustnessEvaluation":
        out = []
        for r in records:
            s = setting or str(r.get("setting") or r.get("type") or r.get("task") or "").lower()
            if "noise" in s or "refine" in s:
                s2 = "noise_robustness"
            elif "negative" in s or "reject" in s or "refuse" in s:
                s2 = "negative_rejection"
            elif "integr" in s:
                s2 = "information_integration"
            elif "fact" in s or "counter" in s:
                s2 = "counterfactual_robustness"
            else:
                s2 = s if s in RGBRobustnessEvaluation.SETTINGS else "information_integration"
            nl = r.get("noise_level", r.get("noise_rate"))
            out.append(RGBCase(s2, str(r.get("question", r.get("query", ""))), r.get("answer"),
                               None if nl is None else float(nl), bool(r.get("negative", False)),
                               bool(r.get("counterfactual", False)), dict(r)))
        return RGBRobustnessEvaluation(out)

    def evaluate(self, outputs: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if len(outputs) != len(self.cases):
            raise ValueError(f"RGB outputs={len(outputs)} does not match cases={len(self.cases)}")
        by = defaultdict(list)
        for c, o in zip(self.cases, outputs):
            row = dict(o)
            pred = row.get("prediction", row.get("final_answer", row.get("answer")))
            correct = answer_accuracy(pred, c.gold)
            row["_correct"] = bool(correct)
            by[c.setting].append((c, row))
        result: dict[str, Any] = {}
        for setting, rows in by.items():
            if setting == "noise_robustness":
                levels = defaultdict(list)
                for c, r in rows:
                    levels[c.noise_level].append(float(r["_correct"]))
                result[setting] = {str(level): float(np.mean(vals)) for level, vals in sorted(levels.items(), key=lambda x: str(x[0]))}
            elif setting == "negative_rejection":
                rejected = []
                for c, r in rows:
                    if c.negative or r.get("is_negative"):
                        pred_reject = bool(r.get("rejected", r.get("predicted_reject", False)))
                        rejected.append(float(pred_reject))
                result[setting] = {"RejectionRate": float(np.mean(rejected)) if rejected else float("nan")}
            elif setting == "counterfactual_robustness":
                ed, cr = [], []
                for c, r in rows:
                    if c.counterfactual or r.get("is_counterfactual"):
                        detected = bool(r.get("detected_error", r.get("error_detected", False)))
                        corrected = bool(r.get("corrected", r.get("counterfactual_corrected", False)))
                        ed.append(float(detected)); cr.append(float(corrected and detected))
                result[setting] = {
                    "ED": float(np.mean(ed)) if ed else float("nan"),
                    "CR": float(np.sum(cr) / max(np.sum(ed), 1)) if ed else float("nan"),
                }
            else:
                result[setting] = {"Accuracy": float(np.mean([r["_correct"] for _, r in rows])) if rows else float("nan")}
        return result


# ---- cost ------------------------------------------------------------------------------------
def n_documents_scored(corpus_size: int, iterations: int) -> int:
    return int(corpus_size) * int(iterations)


def n_similarity_calculations(corpus_size: int, m_gaps: int, n_evidence: int) -> int:
    return int(corpus_size) * (1 + int(m_gaps) + int(n_evidence))


def retrieval_overhead_pct(l_sar: float, l_baseline: float) -> float:
    if l_baseline == 0:
        return float("nan")
    return (l_sar - l_baseline) / l_baseline * 100.0


class AblationStudy:
    @staticmethod
    def ablation_weights(w) -> dict[str, tuple[float, float, float]]:
        def renorm(a, b, g):
            s = a + b + g
            if s <= 0:
                raise ValueError("ablation leaves no active SAR signal")
            return a / s, b / s, g / s
        return {
            "SAR-Full": (w.alpha, w.beta, w.gamma),
            "SAR-Without-Relevance": renorm(0, w.beta, w.gamma),
            "SAR-Without-Gap": renorm(w.alpha, 0, w.gamma),
            "SAR-Without-Evidence": renorm(w.alpha, w.beta, 0),
            "Question-Only": (1.0, 0.0, 0.0),
            "Gap-Only": (0.0, 1.0, 0.0),
            "Novelty-Only": (0.0, 0.0, 1.0),
        }
