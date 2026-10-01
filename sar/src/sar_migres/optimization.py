"""Bayesian optimisation of SAR simplex weights.

The optimisation API takes an explicit validation-only objective.  The supplied methodology
requires Bayesian optimisation but does not state the objective function; the executable
experiment plan therefore records that choice explicitly rather than hiding it in code.
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
