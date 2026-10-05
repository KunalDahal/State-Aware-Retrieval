import numpy as np
import pytest
from sar_migres.scoring.weights import SARWeights
from sar_migres.scoring.sar import score_components, sar_score, top_k_indices

def test_sar_component_calculations_and_empty_state():
    docs = np.array([[1.,0.],[0.,1.]], dtype=float)
    q = np.array([1.,0.])
    c = score_components(docs, q, np.zeros((0,2)), np.zeros((0,2)))
    assert np.allclose(c.gap_coverage, 0.0)
    assert np.allclose(c.novelty, 1.0)
    w = SARWeights(1/3, 1/3, 1/3)
    out = sar_score(c, w)
    assert np.allclose(out, (c.relevance + c.novelty) / 3)

def test_sar_weights_simplex_and_validation():
    w = SARWeights.from_simplex(0.2, 0.5)
    assert np.isclose(sum(w.as_tuple()), 1.0)
    with pytest.raises(Exception): SARWeights(0.5, 0.5, 0.2)

def test_deterministic_top_k_identity_ties():
    scores = np.array([1., 1., 1.])
    ids = ["z", "a", "m"]
    assert top_k_indices(scores, 2, ids).tolist() == [1, 2]

def test_novelty_with_empty_evidence():
    docs = np.array([[1.,0.],[-1.,0.]])
    q = np.array([1.,0.])
    c = score_components(docs, q, np.zeros((0,2)), np.zeros((0,2)))
    assert np.allclose(c.novelty, 1.0)


def test_canonical_config_protocol():
    from sar_migres.utils.config import load_config, validate_canonical_protocol
    cfg=load_config("configs/default.yaml", "configs/gemini.yaml")
    validate_canonical_protocol(cfg)


def test_canonical_dataset_count_is_exactly_three():
    from sar_migres.utils.config import load_config, CANONICAL_STUDY_DATASETS
    cfg=load_config("configs/default.yaml", "configs/gemini.yaml")
    assert tuple(cfg.data.datasets) == CANONICAL_STUDY_DATASETS
    assert set(cfg.data.dataset_paths) == set(CANONICAL_STUDY_DATASETS)
    assert len(cfg.data.datasets) == 3
