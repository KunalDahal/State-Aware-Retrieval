# SAR-MIGRES — State Aware Retrieval with a swappable MIGRES retriever

## Experimental contract

The searchable corpus `D` is a pooled, bounded passage corpus built from the six datasets' training material, deduplicated and chunked to <=350 tokenizer tokens with 50-token overlap, capped at 100,000 passages. Development data are reserved for labeled final evaluation. The vector store is a local disk-backed memmap because SAR must evaluate its composite score over the entire corpus.

Tuning is a deterministic 10% holdout from train. The public `dev` split is the default labeled final split because many benchmark test sets do not expose answer labels. One global `(K, alpha, beta, gamma)` is selected by Bayesian optimization over the six datasets and then frozen.

SAR is `Sentence Transformer -> state-aware algorithm -> SAR score -> Top-K`. Every baseline replaces that entire stack. Only SAR and Sentence Transformer Search construct the project Sentence Transformer. All other baseline methods use native lexical, sparse, Transformer or late-interaction representations.

The common MIGRES filter is deliberately downstream of the retriever. Raw retrieval metrics therefore use the pre-filter ordering; downstream RAG metrics use the post-filter evidence path. Filter retention and top-1 survival are separately recorded because the common filter can change the retriever's ordering before the LLM sees the evidence.

Each experiment from methodology §5.6 has its own notebook.


This repository implements the methodology in the supplied project report and keeps the experimental systems modular.

## Architecture

MIGRES owns the iterative reasoning, evidence extraction, missing-information update, filtering, answer generation, and stopping logic. The retrieval slot is configurable.

For `retrieval.method: sar`, the slot is:

`Sentence Transformer -> SAR state algorithm -> SAR score -> Top-K`

The SAR score is the methodology formulation:

`S_t(d) = alpha R(d,q) + beta C(d,G_t) + gamma N(d,E_t)` with non-negative weights summing to one.

For every other baseline, that *entire SAR retrieval stack* is replaced by the baseline retriever. The Sentence Transformer package is only constructed for the explicit `sentence_transformer_search` baseline and for SAR. BM25, DPR, Contriever, SPLADE and ColBERTv2 do not import `sentence-transformers`.

## Baselines inside the same MIGRES loop

The upstream adapter subclasses the upstream MIGRES `RetGBatched` and overrides the retrieval call. The surrounding MIGRES flow is unchanged. For baselines, each MIGRES-generated query is sent through the selected retriever and the resulting per-document scores are fused by maximum before the same Top-K/filter/forward path. This avoids the previous one-shot-original-question baseline shortcut.

Methods:

* `sar`
* `bm25`
* `mmr`
* `dpr`
* `contriever`
* `splade`
* `colbertv2`
* `sentence_transformer_search`
* `infogain_rag` (optional; requires a compatible published InfoGain-RAG reranker checkpoint)

The non-SAR models use native Transformers / sparse / late-interaction code paths and lazy optional dependencies.

## Memory model

The thesis describes exact full-corpus SAR scoring. Dense 768-dimensional vectors are therefore stored in a disk-backed `DiskMemmapVectorStore` instead of being copied into a Python NumPy matrix. SQLite stores vector IDs and content hashes, while the corpus loader exposes an ID sequence view rather than forcing a full ID list into RAM. SAR iterates the vectors in chunks and never requests the complete dense matrix in RAM.

`vector_store: local_memmap` is the default exact scorer for both small and large corpora. `vector_store: local` is retained for tiny smoke tests only. SAR also synchronizes the index after every completed retrieval iteration, including the final iteration; the corpus source is checked using a cheap size/mtime signature and reloaded only when it changes. BM25 likewise uses an on-disk inverted index so the baseline does not build a Python term-frequency object for the entire corpus.

For a machine with 16 GB RAM, use `corpus_lazy: true`, the disk-backed vector store, and an explicit `data.max_documents` only when a constrained development run is intended. A reduced corpus is **not** a substitute for the final full-corpus experiment; it must be reported as a development setting.

## Experimental notebooks

Each methodology experiment has its own notebook; there is no single all-in-one experiment notebook:

1. `01_parameter_hyperparameter_selection.ipynb`
2. `02_baseline_comparison.ipynb`
3. `03_ablation_study.ipynb`
4. `04_retrieval_performance.ipynb`
5. `05_iterative_state_analysis.ipynb`
6. `06_downstream_rag_robustness.ipynb`
7. `07_computational_cost.ipynb`

Each notebook has its own run directory under `outputs/experiments/<name>/` and defaults to a safe preflight/dry-run until the paths and credentials are supplied.

## Validation and test protocol

The supplied methodology states that validation data are used for parameter selection and the final test configuration is frozen before test evaluation. The code therefore separates `data.split`, filters tagged/split-keyed input files when possible, and provides a parameter-selection notebook that writes a frozen configuration. The parameter notebook tunes `(alpha,beta,gamma)` on validation data and selects `K` from a predefined grid.

One point was missing from the methodology: the optimization **objective** for Bayesian optimization was not stated. The project makes this explicit in `configs/experiments.yaml` as `mean_recall_at_k` and records that choice in experiment metadata. Change the objective there if the report is revised to use a different validation target.

Likewise, the report says dynamic weights may depend on state but does not specify a mathematical rule. The code supports an explicit project extension `state_balanced_v1`; it is never described as part of the supplied methodology and is marked in metadata whenever used.

## RGB

`evaluation.RGBRobustnessEvaluation` now evaluates the four settings named by the report: noise robustness, negative rejection, information integration, and counterfactual robustness. It is intentionally separated from LLM execution: notebooks create the model outputs, then pass those outputs to the evaluator.

## Reproducibility

Every run stores a configuration snapshot, dataset/corpus metadata, retrieval method, score name, package versions, synchronization history, and result/trace JSONL + CSV. API keys are read from environment variables and are never written to run metadata.

## Gemini setup

A ready-to-use Gemini configuration is included at `configs/gemini.yaml`. It uses Gemini through Google's OpenAI-compatible endpoint while preserving the upstream MIGRES chat-completions request shape. The current overlay uses `gemini-3.5-flash-lite`.

Set `GEMINI_API_KEY` and `OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/` in `.env`, then layer the override after the main configuration, for example:

```bash
python scripts/run_single.py --config configs/default.yaml configs/gemini.yaml
```

See `docs/GEMINI_SETUP.md` for the complete setup. The model ID is intentionally kept in configuration so it can be changed when the provider's model availability changes.

## Installation

Offline smoke path:

```bash
pip install -r requirements-offline.txt
pytest
python scripts/validate_installation.py
```

Real baseline/MIGRES experiments can use `requirements.txt`; it intentionally does not install `sentence-transformers`. SAR and the explicit Sentence Transformer Search baseline additionally require `requirements-sar.txt` (or the `st` optional extra). The official MIGRES repository must be configured with `MIGRES_REPO_DIR`.

The model names in the configuration are explicit. The language-model ID remains an environment/configuration choice; the code does not silently assume that a provider supports a particular model string.

## Recommended experiment order

See `EXPERIMENT_PROTOCOL.md` for the validation → frozen-test execution order and the purpose of each notebook.
