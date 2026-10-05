# Implementation

The supplied implementation specification is the source of truth. Structural refactoring is not allowed to change numerical behavior unless the previous code contradicted that specification.

Canonical semantic boundaries:

- Similarity: embedding normalization, cosine similarity, and normalized similarity.
- Scoring: relevance, gap consistency, evidence novelty, weighted SAR, and static weight validation.
- Retrieval: corpus-wide scoring, block-wise computation, deterministic global Top-K, and identity-based tie-breaking.
- Baselines: ranking/retrieval only, followed by the same downstream MIGRES path.
- MIGRES: upstream integration through an adapter, including the required empty initial evidence state.
- Datasets: loading, normalization, stable corpus identifiers, and relevance mapping.
- Evaluation: retrieval, answer, RGB, iterative, and cost metrics.
- Experiments: orchestration, frozen configuration, ablations, iterative runs, and reproducibility.
