from __future__ import annotations

def n_documents_scored(corpus_size: int, iterations: int) -> int:
    return int(corpus_size) * int(iterations)


def n_similarity_calculations(corpus_size: int, m_gaps: int, n_evidence: int) -> int:
    return int(corpus_size) * (1 + int(m_gaps) + int(n_evidence))


def retrieval_overhead_pct(l_sar: float, l_baseline: float) -> float:
    if l_baseline == 0:
        return float("nan")
    return (l_sar - l_baseline) / l_baseline * 100.0
