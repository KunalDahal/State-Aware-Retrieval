# Refactor Report

## 1. Final directory tree

See the repository tree included in the release archive. The canonical source root is `src/sar_migres/`; upstream MIGRES is isolated at `external/MIGRES/`.

## 2. Files moved

The legacy flat implementations were redistributed as follows:

- `corpus.py`, `corpus_builder.py` -> `corpus/models.py`, `corpus/builder.py`, `corpus/chunking.py`
- `datasets.py` -> `datasets/loader.py`, `datasets/normalization.py`, `datasets/relevance.py`
- `embeddings.py` -> `embeddings/encoder.py`
- `similarity.py`, `scoring.py` -> `scoring/similarity.py`, `scoring/sar.py`, `scoring/weights.py`
- `retrieval.py`, `baselines.py` -> `retrieval/sar.py`, `retrieval/baselines.py`, `retrieval/common.py`
- `migres_adapter.py` -> `migres/adapter.py`, `migres/interface.py`
- `experiment_runner.py`, `optimization.py` -> `experiments/runner.py`, `experiments/parameter_selection.py`, `experiments/registry.py`, `experiments/iterative.py`
- `evaluation.py` -> `evaluation/retrieval.py`, `evaluation/answer.py`, `evaluation/rgb.py`, `evaluation/cost.py`, `evaluation/aggregation.py`
- `config.py`, `utils.py`, `errors.py`, `logging_utils.py` -> `utils/config.py`, `utils/common.py`, `utils/hashing.py`, `utils/errors.py`, `utils/paths.py`, `utils/logging.py`
- legacy Python scripts -> package-backed `cli/` modules with tiny root wrappers

## 3. Deleted duplicate/obsolete files

Deleted legacy root algorithm locations after migration, duplicate notebook-local script infrastructure, and generated Python cache files. No second SAR or project-specific MIGRES implementation remains.

## 4. Renamed modules/notebooks

Notebooks were renamed to the canonical seven-file naming scheme from the refactor specification: `01_parameter_hyperparameter_selection`, `02_baseline_comparison`, `03_ablation_study`, `04_retrieval_performance`, `05_iterative_state_analysis`, `06_downstream_rag_robustness`, `07_computational_cost`.

## 5. Import-path changes

Legacy `SAR.*`, notebook-local package, working-directory, and old flat-module imports were replaced with absolute `sar_migres.<module>` imports. CLI entry points now target `sar_migres.cli.*`.

## 6. Scientific contradictions fixed

- MIGRES initial evidence is reset to an empty `E0` in the adapter.
- Baselines retain trace snippets for inspection but pass full passage text downstream to MIGRES.
- MuSiQue relevance maps supporting parents to all matching chunks through stable parent IDs.
- Natural Questions positive contexts are normalized into corpus IDs before evaluation.
- Ablations match the exact specified set; `SAR-Without-Tuning` uses `(1/3, 1/3, 1/3)`.
- Parameter selection fixes `K=50`; only SAR weights are tuned.
- Missing frozen parameter artifacts fail loudly.
- Scientific corpus construction cannot silently fall back to whitespace tokenization.
- Strict scientific MIGRES validation checks the expected upstream commit and pinned file hashes.
- Top-K ties use stable document identity instead of array position.
- Unlabeled retrieval examples are excluded from scored retrieval metrics.
- Dynamic weighting is disabled in the canonical configuration.

## 7. Configuration changes

`pyproject.toml` is authoritative for dependencies and CLI entry points. The canonical default config uses the specified three-dataset pooled corpus, `K=50`, static SAR weighting, strict MIGRES provenance, and the specified tokenizer/model settings. `configs/local.yaml` explicitly selects the offline stand-ins.

## 8. Tests added/changed

Regression coverage now includes similarity normalization, SAR components and empty state, novelty, simplex weights, deterministic identity tie-breaking, block-wise equivalence, stable chunk IDs, MuSiQue/Natural Questions relevance normalization, empty MIGRES state, full-text baseline/MIGRES handoff, frozen settings, config loading, exact ablation definitions, tokenizer fallback rules, and CLI imports.

## 9. Validation executed

- Editable installation: `pip install -e . --no-deps --no-build-isolation` — passed.
- `python -m compileall src tests` — passed.
- `pytest -q` — passed.
- Installation validator with `configs/local.yaml` — passed.
- All package modules imported successfully — passed.
- All five CLI entry points responded to `--help` — passed.
- Notebook format validation and Python AST validation of code cells — passed.
- Synthetic SAR and baseline retrieval smoke tests — passed.
- Repository scan found no stale `SAR.*` or `from code...` Python imports.

## 10. Remaining issue

The uploaded archive did not contain the upstream MIGRES checkout, and the environment could not clone the official upstream repository because outbound DNS/network access was unavailable. Therefore strict live MIGRES execution against the expected upstream commit could not be performed here. The adapter is configured to fail closed rather than silently accepting an unverified upstream version.


## Implementation-alignment audit (2026-10-05)

The implementation audit tightened the repository to the three canonical study datasets only: HotpotQA, MuSiQue, and Natural Questions. Native Natural Questions records using `question_text`, `short_answers`, and `yes_no_answer` are now normalized in the loader, and MuSiQue `answer_aliases` are retained for answer scoring.

The scientific default now resolves to `gemini-3.5-flash-lite` through the OpenAI-compatible provider with provider-default temperature (`null`), while `configs/gemini.yaml` remains an explicit provider overlay. Canonical-protocol validation checks the dataset set, K, iteration budget, embedding model/dimension, chunking, corpus cap, split/seed, reranker, and statistics settings.

The seven notebooks were renamed to the execution-file names specified in the implementation document. Real experiment notebooks default to `RUN_EXPERIMENT=False`, and the parameter-selection notebook no longer installs packages automatically. Retrieval-performance and iterative-state analysis are parameterized for all three datasets rather than hard-coding HotpotQA. The RGB evaluator now imports every runtime dependency it uses.
