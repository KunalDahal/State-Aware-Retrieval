"""Bayesian optimisation of (alpha, beta, gamma) - FUTURE / OCTOBER EXPERIMENT.

The simplex constraint alpha+beta+gamma = 1, all >= 0, is enforced by construction:
the optimiser searches u, v in [0,1]^2 and ``SARWeights.from_simplex`` maps them to
alpha = u, beta = (1-u) v, gamma = (1-u)(1-v).  No objective is shipped: the user supplies
``objective(weights) -> float`` evaluated on VALIDATION data (the methodology's protocol).
This module is exercised in tests only with a toy analytic function to verify the constraint.
"""
from __future__ import annotations

from typing import Callable

from .errors import DependencyMissingError
from .scoring import SARWeights


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
