import numpy as np
from sar_migres.scoring.similarity import cosine, normalized_similarity, normalized_similarity_matrix

def test_cosine_similarity():
    assert cosine(np.array([1., 0.]), np.array([1., 0.])) == 1.0
    assert np.isclose(cosine(np.array([1., 0.]), np.array([0., 1.])), 0.0)

def test_similarity_normalization_is_unit_interval():
    a = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    b = np.array([[1., 0.]])
    s = normalized_similarity_matrix(a, b).ravel()
    assert np.all((s >= 0) & (s <= 1))
    assert np.isclose(s[0], 1.0)
    assert np.isclose(s[1], 0.5)
    assert np.isclose(s[2], 0.0)
    assert np.isclose(normalized_similarity(np.array([1.,0.]), np.array([0.,1.])), 0.5)
