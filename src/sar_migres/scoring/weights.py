from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ..utils.errors import ConfigError

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
