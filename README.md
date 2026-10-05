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
