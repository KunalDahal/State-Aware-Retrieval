"""SAR-MIGRES: State Aware Retrieval (SAR) integrated into the MIGRES framework.

Current milestone: DESIGN + CORE IMPLEMENTATION.  Evaluation, baselines comparison,
Bayesian optimisation and RGB robustness are FUTURE (October) modules; nothing in
this package reports experimental results.
"""

__version__ = "0.1.0"

from .schemas import Document, RetrievalState, ScoredDocument  # noqa: F401
from .scoring import SARWeights  # noqa: F401
