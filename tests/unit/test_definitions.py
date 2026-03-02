import warnings

import numpy as np
import pytest

from selection import CrossValGramData, GramData


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


@pytest.mark.parametrize(
    "gram,cov,y_norm,match",
    [
        (
            np.array([[1.0, 0.0], [0.0, np.inf]]),
            np.array([1.0, 2.0]),
            5.0,
            "finite values",
        ),
        (
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([1.0, np.nan]),
            5.0,
            "finite values",
        ),
        (
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([1.0, 2.0]),
            np.inf,
            "must be finite",
        ),
    ],
)
def test_gramdata_rejects_non_finite_inputs(gram, cov, y_norm, match):
    with pytest.raises(ValueError, match=match):
        GramData(gram=gram, cov=cov, y_norm=y_norm, n_samples=10)


def test_cv_derived_data_preserves_warn_if_uncentered_opt_out():
    gram = np.eye(2)
    cov = np.array([1.0, 0.0])
    fold0 = GramData(gram=gram, cov=cov, y_norm=1.0, n_samples=10, warn_if_uncentered=False)
    fold1 = GramData(gram=gram, cov=cov, y_norm=1.0, n_samples=10, warn_if_uncentered=False)
    cv_data = CrossValGramData([fold0, fold1])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train = cv_data.train_data_for_fold(0)
        full = cv_data.make_full_data()

    assert not caught
    assert train.warn_if_uncentered is False
    assert full.warn_if_uncentered is False
