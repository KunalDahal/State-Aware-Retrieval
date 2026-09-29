import numpy as np
import pytest

from sar_migres.errors import ConfigError
from sar_migres.scoring import (SARWeights, ScoreComponents, evidence_novelty, gap_coverage, question_relevance,
                                sar_score, score_components, top_k_indices)
from sar_migres.similarity import normalized_similarity


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


@pytest.fixture
def toy():
    D = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([1, 1, 0]), _unit([0, 0, 1])])
    q = _unit([1, 0, 0])[None]
    G = np.stack([_unit([0, 1, 0]), _unit([0, 0, 1])])
    E = np.stack([_unit([1, 0, 0]), _unit([1, 1, 0])])
    return D, q, G, E


def test_question_relevance_is_normalised_similarity(toy):
    D, q, _, _ = toy
    R = question_relevance(D, q)
    for i in range(len(D)):
        assert R[i] == pytest.approx(normalized_similarity(D[i], q[0]))
    assert R[0] == pytest.approx(1.0) and R[1] == pytest.approx(0.5)


def test_gap_coverage_is_mean_over_gaps(toy):
    D, _, G, _ = toy
    C = gap_coverage(D, G)
    for i in range(len(D)):
        assert C[i] == pytest.approx(np.mean([normalized_similarity(D[i], g) for g in G]))


def test_empty_gaps_give_zero_coverage(toy):
    D, *_ = toy
    assert np.all(gap_coverage(D, np.zeros((0, 3))) == 0.0)
    assert np.all(gap_coverage(D, []) == 0.0)


def test_evidence_novelty_uses_max_similarity(toy):
    D, _, _, E = toy
    N = evidence_novelty(D, E)
    for i in range(len(D)):
        assert N[i] == pytest.approx(1 - max(normalized_similarity(D[i], e) for e in E))
    assert N[0] == pytest.approx(0.0)  # document identical to an evidence item -> no novelty


def test_empty_evidence_gives_full_novelty(toy):
    D, *_ = toy
    assert np.all(evidence_novelty(D, np.zeros((0, 3))) == 1.0)


def test_fused_components_equal_reference_functions(toy):
    D, q, G, E = toy
    comp = score_components(D, q, G, E)
    np.testing.assert_allclose(comp.relevance, question_relevance(D, q), atol=1e-6)
    np.testing.assert_allclose(comp.gap_coverage, gap_coverage(D, G), atol=1e-6)
    np.testing.assert_allclose(comp.novelty, evidence_novelty(D, E), atol=1e-6)
    empty = np.zeros((0, 3))
    c0 = score_components(D, q, empty, empty)
    assert np.all(c0.gap_coverage == 0) and np.all(c0.novelty == 1)


def test_sar_score_formula(toy):
    D, q, G, E = toy
    comp = score_components(D, q, G, E)
    w = SARWeights(0.5, 0.3, 0.2)
    np.testing.assert_allclose(sar_score(comp, w), 0.5 * comp.relevance + 0.3 * comp.gap_coverage + 0.2 * comp.novelty)


def test_sar_score_stays_in_unit_interval(toy):
    D, q, G, E = toy
    s = sar_score(score_components(D, q, G, E), SARWeights(0.2, 0.5, 0.3))
    assert s.min() >= 0 and s.max() <= 1


@pytest.mark.parametrize("a,b,g", [(-0.1, 0.6, 0.5), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2), (float("nan"), 0.5, 0.5), (1.2, -0.2, 0.0)])
def test_weight_constraints_rejected(a, b, g):
    with pytest.raises(ConfigError):
        SARWeights(a, b, g)


def test_weights_accept_valid_and_derive_gamma():
    assert SARWeights(1, 0, 0).alpha == 1
    w = SARWeights.from_two(0.5, 0.25)
    assert w.gamma == pytest.approx(0.25)
    with pytest.raises(ConfigError):
        SARWeights.from_two(0.7, 0.6)  # gamma would be negative


def test_simplex_parameterisation_roundtrip_and_feasibility():
    rng = np.random.default_rng(3)
    for u, v in rng.uniform(size=(200, 2)):
        w = SARWeights.from_simplex(u, v)
        assert min(w.as_tuple()) >= 0 and sum(w.as_tuple()) == pytest.approx(1.0)
        u2, v2 = w.to_simplex()
        assert u2 == pytest.approx(u) and v2 == pytest.approx(v)


def test_top_k_ordering_and_tie_break():
    s = np.array([0.1, 0.9, 0.5, 0.9, 0.3])
    assert top_k_indices(s, 3).tolist() == [1, 3, 2]  # equal scores -> lower index first
    assert top_k_indices(s, 99).tolist() == [1, 3, 2, 4, 0]
    assert top_k_indices(s, 0).size == 0
