import numpy as np
import pytest

from sar_migres.similarity import cosine, cosine_matrix, normalized_similarity, normalized_similarity_matrix


def test_cosine_known_values():
    assert cosine(np.array([1.0, 0]), np.array([1.0, 0])) == pytest.approx(1.0)
    assert cosine(np.array([1.0, 0]), np.array([0, 1.0])) == pytest.approx(0.0)
    assert cosine(np.array([1.0, 0]), np.array([-1.0, 0])) == pytest.approx(-1.0)


def test_cosine_is_scale_invariant_and_does_not_assume_unit_norm():
    a, b = np.array([3.0, 4.0]), np.array([10.0, 0.0])
    assert cosine(a, b) == pytest.approx(0.6)
    assert cosine(7 * a, 0.1 * b) == pytest.approx(0.6)


def test_cosine_matches_formula_on_random_vectors():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(5, 16)), rng.normal(size=(4, 16))
    got = cosine_matrix(A, B)
    for i in range(5):
        for j in range(4):
            ref = A[i] @ B[j] / (np.linalg.norm(A[i]) * np.linalg.norm(B[j]))
            assert got[i, j] == pytest.approx(ref, abs=1e-6)


def test_normalized_similarity_range_and_values():
    assert normalized_similarity(np.array([1.0, 0]), np.array([1.0, 0])) == pytest.approx(1.0)
    assert normalized_similarity(np.array([1.0, 0]), np.array([0, 1.0])) == pytest.approx(0.5)
    assert normalized_similarity(np.array([1.0, 0]), np.array([-1.0, 0])) == pytest.approx(0.0)
    rng = np.random.default_rng(1)
    s = normalized_similarity_matrix(rng.normal(size=(50, 8)), rng.normal(size=(30, 8)))
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_zero_vector_has_zero_cosine_not_nan():
    s = cosine_matrix(np.zeros((1, 4)), np.ones((1, 4)))
    assert s[0, 0] == 0.0 and np.isfinite(s).all()


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_matrix(np.ones((1, 3)), np.ones((1, 4)))
