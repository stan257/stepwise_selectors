import numpy as np
import pytest

from selection.definitions import GramData


def test_gramdata_rejects_nonsymmetric_gram_with_absolute_tolerance():
    gram = np.array([[1e6, 1.0], [1.0 + 1e-6, 1e6]])
    cov = np.array([1.0, 2.0])
    y_norm = 5.0
    with pytest.raises(ValueError, match="symmetric"):
        GramData(gram=gram, cov=cov, y_norm=y_norm, n_samples=10)


def test_gramdata_rejects_negative_gram_diagonal():
    gram = np.array([[1.0, 0.0], [0.0, -1e-8]])
    cov = np.array([1.0, 2.0])
    y_norm = 5.0
    with pytest.raises(ValueError, match="non-negative"):
        GramData(gram=gram, cov=cov, y_norm=y_norm, n_samples=10)
