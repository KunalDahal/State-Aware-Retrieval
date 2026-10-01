import json
from pathlib import Path

from sar_migres.corpus_builder import chunk_text
from sar_migres.datasets import split_holdout_from_train
from sar_migres.evaluation import paired_bootstrap_ci, paired_permutation_pvalue


def test_chunk_text_respects_token_budget_whitespace():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(text, max_tokens=20, overlap_tokens=5)
    assert chunks
    assert all(len(c.split()) <= 20 for c in chunks)


def test_train_holdout_is_deterministic():
    rows = [{"id": f"q{i}"} for i in range(100)]
    a = [r["id"] for r in split_holdout_from_train(rows, 0.1, 13)]
    b = [r["id"] for r in split_holdout_from_train(rows, 0.1, 13)]
    assert a == b and a


def test_paired_stats_shapes():
    a = [1.0, 0.8, 0.4, 0.9]
    b = [0.8, 0.7, 0.5, 0.9]
    ci = paired_bootstrap_ci(a, b, n_resamples=200, seed=13)
    p = paired_permutation_pvalue(a, b, n_resamples=200, seed=13)
    assert ci["mean_difference"] == sum(x-y for x,y in zip(a,b))/4
    assert 0.0 <= p["p_value"] <= 1.0
