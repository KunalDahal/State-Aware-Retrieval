# Architecture

The only project Python package is `src/sar_migres/`. The package follows the research responsibilities from the implementation specification rather than preserving the legacy flat source layout.

`corpus/` owns document representation, corpus construction and chunking. `datasets/` owns loading, normalization and ground-truth relevance mapping. `embeddings/` owns the project Sentence Transformer and the explicitly offline hashing embedder. `scoring/` owns similarity transforms, SAR component calculations, and simplex-constrained weights. `retrieval/` owns SAR retrieval, deterministic Top-K, and baseline retrievers. `migres/` contains only the project-specific adapter/interface around upstream MIGRES. `experiments/` owns orchestration, frozen parameter selection, registries and iterative analysis. `evaluation/` owns retrieval, answer, RGB, cost and aggregation metrics. `cli/` contains thin package-backed command implementations. `utils/` contains configuration, paths, hashing, logging, errors and shared cross-cutting helpers.

Upstream MIGRES belongs under `external/MIGRES/`. It is not copied into `sar_migres`, and the project does not implement a second MIGRES control loop.

Generated data and artifacts never belong in source directories: raw/intermediate/processed inputs live under `data/`, and experiment outputs live under `outputs/`.
