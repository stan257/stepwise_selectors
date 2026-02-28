import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.routines_core import BeamBackwardSelection, BeamMixedSelection
from tests.helpers import explicit_beta_rss, make_regression_gram


def test_fast_beam_backward_best_rss_matches_explicit_solution():
    data = make_regression_gram(777, n=200, p=18)
    state = BeamBackwardSelection(
        beam_width=3, allow_worse=True, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=4)
    beta_expected, rss_expected = explicit_beta_rss(data, state.active_set)

    np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected


def test_fast_beam_mixed_best_rss_matches_explicit_solution():
    data = make_regression_gram(778, n=200, p=18)
    state = BeamMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_forward_steps=3, max_total_steps=6)
    beta_expected, rss_expected = explicit_beta_rss(data, state.active_set)

    np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected
