import numpy as np

from selection.criteria import BestRSSCriterion
from selection import CrossValGramData, GramData
from selection import (
    BeamCrossValBackwardSelection,
)


def _make_full_model_optimal_cv(*, folds: int = 3, n: int = 50, p: int = 6):
    """Build diagonal CV data where dropping any feature strictly worsens RSS."""
    fold_data = []
    gram = float(n) * np.eye(p)
    cov = float(n) * np.ones(p, dtype=float)
    y_norm = float(n) * (p + 1.0)
    for _ in range(folds):
        fold_data.append(GramData(gram.copy(), cov.copy(), y_norm, n))
    return CrossValGramData(fold_data)


def test_cv_beam_backward_is_improvement_only_by_default():
    cv_data = _make_full_model_optimal_cv(folds=3, n=40, p=6)

    strict = BeamCrossValBackwardSelection(
        beam_width=2, criterion=BestRSSCriterion
    ).fit(data=cv_data, max_steps=1)
    relaxed = BeamCrossValBackwardSelection(
        beam_width=2, criterion=BestRSSCriterion, allow_worse=True
    ).fit(data=cv_data, max_steps=1)

    assert len(strict.active_set) == cv_data.p
    assert len(relaxed.active_set) == cv_data.p - 1
    assert strict.rss_cv <= relaxed.rss_cv + 1e-12
