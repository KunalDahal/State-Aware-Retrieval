# SAR-MIGRES Research Repository

This repository uses a canonical `src/` Python layout. The implementation specification is authoritative for scientific behavior and module boundaries.

## Repository layout

- `src/sar_migres/`: the only project Python package.
- `configs/`: default, local/offline, provider, dataset, and experiment configuration.
- `notebooks/`: thin experiment interfaces; reusable implementation lives in `src/sar_migres/`.
- `tests/`: unit and integration-style regression tests.
- `data/raw`, `data/interim`, `data/processed`: data lifecycle stages.
- `outputs/`: generated experiment artifacts.
- `external/MIGRES/`: upstream MIGRES checkout; project code integrates it through `sar_migres.migres`.
- `docs/`: architecture and experiment documentation.

## Installation

```bash
pip install -e .
```

Install optional groups as required by the experiment. The `offline` configuration is dependency-light and is intended only for smoke tests; it does not represent the scientific embedding or MIGRES stack.

## Validation

```bash
python -m compileall src tests
pytest -q
sar-validate-installation --config configs/local.yaml
```

The editable install exposes these commands:

```text
sar-run-single
sar-run-batch
sar-build-embeddings
sar-prepare-corpus
sar-validate-installation
```

## Scientific protocol

The canonical study contains exactly three datasets: HotpotQA, MuSiQue, and Natural Questions.

The canonical configuration fixes SAR `K=50`, uses static weights in the main protocol, and tunes only the simplex-constrained `(alpha, beta, gamma)` weights on the deterministic validation holdout. Final labeled experiments consume the frozen parameter-selection artifact and fail when it is absent.

Corpus construction uses the specified tokenizer and 350-token maximum with 50-token overlap. Whitespace tokenization is available only through an explicit offline/test setting.

The ablation set is exactly `SAR-Full`, `SAR-Without-Relevance`, `SAR-Without-Gap`, `SAR-Without-Evidence`, `SAR-Without-Tuning`, and `Question-Only`.

MIGRES is kept outside the project package. The adapter enforces the scientific initial evidence state, passes full retrieved passage text downstream, and validates the expected upstream provenance in strict scientific mode.

## Upstream MIGRES

Populate `external/MIGRES/` with the expected upstream checkout before running the real MIGRES-backed protocol. The current repository intentionally does not silently replace the upstream dependency with a different implementation.

## Run the experiments in Google Colab

Open the notebooks from the `master` branch:

- [00 Colab setup](https://colab.research.google.com/github/KunalDahal/State-Aware-Retrieval/blob/master/src/notebooks/00_colab_setup.ipynb)
- [01 Parameter and hyperparameter selection](https://colab.research.google.com/github/KunalDahal/State-Aware-Retrieval/blob/master/src/notebooks/01_parameter_hyperparameter_selection.ipynb)
- [02 Baseline comparison](https://colab.research.google.com/github/KunalDahal/State-Aware-Retrieval/blob/master/src/notebooks/02_baseline_comparison.ipynb)
- [03 Ablation study](https://colab.research.google.com/github/KunalDahal/State-Aware-Retrieval/blob/master/src/notebooks/03_ablation_study.ipynb)

Run notebook 00 from top to bottom first. It installs the research dependencies, mounts Google Drive, checks the pinned MIGRES checkout, and links `data/` and `outputs/` to persistent Drive storage. Add `GEMINI_API_KEY` in Colab Secrets and enable notebook access before experiments that call the LLM. Notebook 01 prepares the datasets and corpus; set `RUN_EXPERIMENT = True` only when you are ready to run its 10 tuning trials. After the frozen weights are saved, run notebooks 02 and 03. Notebooks 04–07 analyze completed runs and save JSON summaries and PNG plots in their experiment output directories.

Experiment notebooks default to preflight mode. Running real experiments calls the configured LLM and may incur API charges. Keep the canonical protocol settings unchanged when producing research results.
