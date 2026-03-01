import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import GramData
from selection.routines import ForwardSelection
from tests.helpers import explicit_beta_rss


def _make_data_scenario(kind: str, seed: int) -> GramData:
    rng = np.random.default_rng(seed)
    if kind == "p_gt_n":
        n, p = 60, 120
        X = rng.standard_normal((n, p))
    elif kind == "n_gt_p":
        n, p = 400, 20
        X = rng.standard_normal((n, p))
    elif kind == "ill_conditioned":
        n, p = 200, 40
        X = rng.standard_normal((n, p))
        # Introduce strong collinearity in a few columns.
        X[:, 1] = X[:, 0] + 1e-3 * rng.standard_normal(n)
        X[:, 2] = X[:, 0] - 1e-3 * rng.standard_normal(n)
    else:
        raise ValueError(f"Unknown scenario '{kind}'.")

    beta = rng.standard_normal(p)
    y = X @ beta + 0.1 * rng.standard_normal(n)
    return GramData(X.T @ X, X.T @ y, y @ y, n)


@pytest.mark.parametrize("kind", ["p_gt_n", "n_gt_p", "ill_conditioned"])
def test_forward_matches_explicit_across_regimes(kind: str):
    data = _make_data_scenario(kind, seed=2025)
    max_steps = 5

    state = ForwardSelection(criterion=BestRSSCriterion).fit(
        data=data, max_steps=max_steps
    )
    beta, rss = explicit_beta_rss(data, state.active_set)
    np.testing.assert_allclose(state.beta, beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss
