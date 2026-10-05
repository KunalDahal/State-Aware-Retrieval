from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json
import numpy as np


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
    from ..experiments.iterative import load_run_traces
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
            "SAR-Without-Tuning": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            "Question-Only": (1.0, 0.0, 0.0),
        }
