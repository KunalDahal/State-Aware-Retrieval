from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Shared retrieval/experiment DTOs. Document remains in sar_migres.schemas for compatibility with the persisted format.
@dataclass
class RetrievalState:
    """(q, E_t, G_t); the corpus D lives in the retriever."""

    question: str
    evidence: list[str] = field(default_factory=list)  # E_t
    gaps: list[str] = field(default_factory=list)  # G_t


@dataclass
class ScoredDocument:
    doc_id: str
    rank: int
    relevance: float  # R(d,q)
    gap_coverage: float  # C(d,G_t)
    novelty: float  # N(d,E_t)
    score: float  # S_t
    title: str = ""
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    documents: list[ScoredDocument]
    alpha: float
    beta: float
    gamma: float
    k: int
    n_docs_scored: int
    n_similarity_calcs: int
    embed_seconds: float
    score_seconds: float
    all_scores: Any = None  # optional np.ndarray (tests / diagnostics only)


@dataclass
class TraceRecord:
    """One SAR retrieval call inside a MIGRES run (one iteration)."""

    run_id: str
    question_id: str
    question: str
    iteration: int
    gaps: list[str]
    evidence: list[str]
    alpha: float
    beta: float
    gamma: float
    k: int
    documents: list[dict[str, Any]]
    n_docs_scored: int
    n_similarity_calcs: int
    embed_seconds: float
    score_seconds: float
    total_seconds: float
    migres_state_after: dict[str, Any] = field(default_factory=dict)
    retriever: str = "sar"
    score_name: str = "sar_score"
    is_sar: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionResult:
    run_id: str
    question_id: str
    question: str
    gold_answer: Any
    final_answer: str | None
    traces: list[TraceRecord]
    status: str  # "ok" | "failed" | "no_answer"
    error: str | None = None
    elapsed_seconds: float = 0.0
    backend: str = ""
    llm_calls: int = 0
    llm_estimated_tokens: int = 0
    migres_record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d
