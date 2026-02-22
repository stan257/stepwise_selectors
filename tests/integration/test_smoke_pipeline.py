import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import CrossValGramData, GramData
from selection.routines import BackwardSelection, CrossValForwardSelection, ForwardSelection


def make_data(rng: np.random.Generator, n: int, p: int) -> GramData:
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y = X @ beta + 0.05 * rng.standard_normal(n)
    return GramData(X.T @ X, X.T @ y, y @ y, n)


def test_smoke_forward_backward_cv_cycle():
    rng = np.random.default_rng(2027)
    n, p = 60, 8

    # Small sizes keep the test fast while exercising the full pipeline.
    data = make_data(rng, n=n, p=p)

    forward_state = ForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=3
    )
    assert 1 <= len(forward_state.active_set) <= 3
    assert np.isfinite(forward_state.rss)

    backward_state = BackwardSelection(
        criterion_cls=BestRSSCriterion, allow_worse=True
    ).fit(data=data, max_steps=2)
    assert len(backward_state.active_set) == p - 2
    assert np.isfinite(backward_state.rss)

    cv_data = CrossValGramData(
        [make_data(rng, n=40, p=p) for _ in range(3)]
    )
    cv_state = CrossValForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=2
    )
    assert 1 <= len(cv_state.active_set) <= 2
    assert np.isfinite(cv_state.rss_cv)

    # Recompute out-of-sample RSS to confirm CV state consistency.
    recomputed = cv_state.clone().recompute_oos_rss()
    assert pytest.approx(cv_state.rss_cv, rel=1e-9, abs=1e-9) == recomputed
