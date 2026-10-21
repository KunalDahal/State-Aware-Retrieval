from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import json
import math
import yaml

from ..utils.config import Config, load_config, CANONICAL_SAR_K
from ..corpus.models import Corpus
from ..datasets.loader import load_phase_questions, load_questions
from ..retrieval.common import QuestionResult
from ..scoring.weights import SARWeights
from ..utils.errors import DependencyMissingError
from ..datasets.relevance import relevant_doc_ids
from ..evaluation.retrieval import recall_at_k


def validation_mean_recall_objective(config_paths: Sequence[str | Path], items: Sequence[dict[str, Any]], *,
                                     k: int, runs_root: str | Path, run_prefix: str = "bayes"):
    counter = {"n": 0}
    def objective(weights: SARWeights) -> float:
        counter["n"] += 1
        name = f"{run_prefix}_k{k}_{counter['n']:03d}"
        overrides = {"retrieval": {"method": "sar"},
                     "sar": {"alpha": weights.alpha, "beta": weights.beta, "gamma": weights.gamma, "k": k, "weighting": "static"},
                     "run": {"name": name, "output_dir": str(Path(runs_root)), "resume": False}}
        from .iterative import run_pipeline_experiment
        pipe, _ = run_pipeline_experiment(config_paths, items, overrides=overrides, experiment_name=name)
        from .iterative import retrieval_metrics_from_run
        metrics = retrieval_metrics_from_run(pipe.run_dir, items, pipe.corpus)
        value = float(metrics["final_mean"]["recall_at_k"])
        return 0.0 if math.isnan(value) else value
    return objective

def select_sar_weights_and_k(config_paths: Sequence[str | Path], items: Sequence[dict[str, Any]], *,
                             k_grid: Sequence[int], init_points: int, n_iter: int, seed: int,
                             runs_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Select SAR weights while enforcing the protocol's fixed K=50.

    The historical function name is retained for compatibility, but K is not a
    tunable parameter in the canonical scientific protocol.
    """
    ks = [int(k) for k in k_grid]
    if ks != [50]:
        raise ValueError(f"Canonical SAR parameter selection fixes k=50; received k_grid={ks!r}.")
    k = 50
    obj = validation_mean_recall_objective(config_paths, items, k=k, runs_root=runs_root,
                                           run_prefix="parameter_selection")
    best, score, history = optimize_weights(obj, init_points=init_points, n_iter=n_iter, seed=seed)
    candidate = {"k": k, "weights": best.as_tuple(), "objective": score, "history": history}
    result = {
        "objective": "mean_recall_at_k",
        "split": "validation",
        "selected_k": k,
        "selected_weights": best.as_tuple(),
        "candidates": [candidate],
        "note": "K is fixed at 50 by the scientific protocol; Bayesian optimisation tunes only the simplex-constrained SAR weights.",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def frozen_setting(path: str | Path) -> tuple[int, SARWeights]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"frozen parameter artifact is required but missing: {p}")
    row = json.loads(p.read_text(encoding="utf-8"))
    if int(row.get("selected_k", -1)) != CANONICAL_SAR_K:
        raise ValueError(f"frozen parameter artifact must select K={CANONICAL_SAR_K}; got {row.get('selected_k')!r}")
    weights = SARWeights(*row["selected_weights"])
    return CANONICAL_SAR_K, weights

def optimize_weights(objective: Callable[[SARWeights], float], init_points: int = 5, n_iter: int = 20,
                     seed: int = 13, verbose: int = 0) -> tuple[SARWeights, float, list[dict]]:
    try:
        from bayes_opt import BayesianOptimization
    except ImportError as e:
        raise DependencyMissingError("bayesian-optimization not installed: pip install 'sar-migres[optim]'") from e

    def f(u: float, v: float) -> float:
        return float(objective(SARWeights.from_simplex(u, v)))

    opt = BayesianOptimization(f=f, pbounds={"u": (0.0, 1.0), "v": (0.0, 1.0)}, random_state=seed, verbose=verbose)
    opt.maximize(init_points=init_points, n_iter=n_iter)
    best = opt.max
    history = [{"weights": SARWeights.from_simplex(r["params"]["u"], r["params"]["v"]).as_tuple(), "target": r["target"]}
               for r in opt.res]
    return SARWeights.from_simplex(best["params"]["u"], best["params"]["v"]), float(best["target"]), history
