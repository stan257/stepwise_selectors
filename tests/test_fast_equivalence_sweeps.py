import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import CrossValGramData, GramData
from selection.fast_routines import (
    FastBackwardSelection,
    FastBeamBackwardSelection,
    FastBeamCrossValBackwardSelection,
    FastBeamCrossValForwardSelection,
    FastBeamCrossValMixedSelection,
    FastBeamForwardSelection,
    FastBeamMixedSelection,
    FastCrossValBackwardSelection,
    FastCrossValForwardSelection,
    FastCrossValMixedSelection,
    FastForwardSelection,
    FastMixedSelection,
)
from selection.grouped_routines import FastGroupBackwardSelection, FastGroupForwardSelection
from selection.legacy_grouped_routines import (
    GroupBackwardSelection,
    GroupForwardSelection,
)
from selection.legacy_routines import (
    BackwardSelection,
    BeamBackwardSelection,
    BeamCrossValBackwardSelection,
    BeamCrossValForwardSelection,
    BeamCrossValMixedSelection,
    BeamForwardSelection,
    BeamMixedSelection,
    CrossValBackwardSelection,
    CrossValForwardSelection,
    CrossValMixedSelection,
    ForwardSelection,
    MixedSelection,
)


def _make_regression(seed: int, n: int, p: int) -> GramData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y = X @ beta + 0.1 * rng.standard_normal(n)
    return GramData(X.T @ X, X.T @ y, y @ y, n)


def _make_cv_regression(seed: int, folds: int, n: int, p: int) -> CrossValGramData:
    rng = np.random.default_rng(seed)
    beta = rng.standard_normal(p)
    fold_data = []
    for fold_seed in rng.integers(0, 1_000_000, size=folds):
        r = np.random.default_rng(int(fold_seed))
        X = r.standard_normal((n, p))
        y = X @ beta + 0.1 * r.standard_normal(n)
        fold_data.append(GramData(X.T @ X, X.T @ y, y @ y, n))
    return CrossValGramData(fold_data)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("p", [32, 80])  # include p > 64 to exercise capacity growth
def test_fast_greedy_equivalence_sweep(seed: int, p: int):
    data = _make_regression(seed=seed, n=140, p=p)
    max_steps = min(6, p // 2)

    fast_f = FastForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=max_steps
    )
    ref_f = ForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=max_steps
    )
    assert set(fast_f.active_set) == set(ref_f.active_set)
    np.testing.assert_allclose(fast_f.beta, ref_f.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast_f.rss, rel=1e-8, abs=1e-8) == ref_f.rss

    fast_b = FastBackwardSelection(allow_worse=True).fit(data=data, max_steps=4)
    ref_b = BackwardSelection(allow_worse=True).fit(data=data, max_steps=4)
    assert set(fast_b.active_set) == set(ref_b.active_set)
    np.testing.assert_allclose(fast_b.beta, ref_b.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast_b.rss, rel=1e-8, abs=1e-8) == ref_b.rss

    fast_m = FastMixedSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_forward_steps=3, max_total_steps=5
    )
    ref_m = MixedSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_forward_steps=3, max_total_steps=5
    )
    assert set(fast_m.active_set) == set(ref_m.active_set)
    np.testing.assert_allclose(fast_m.beta, ref_m.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast_m.rss, rel=1e-8, abs=1e-8) == ref_m.rss


@pytest.mark.parametrize("seed", [2, 3])
def test_fast_beam_equivalence_sweep(seed: int):
    data = _make_regression(seed=seed, n=120, p=25)

    fast_f = FastBeamForwardSelection(
        beam_width=3, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=4)
    ref_f = BeamForwardSelection(
        beam_width=3, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=4)
    assert set(fast_f.active_set) == set(ref_f.active_set)
    np.testing.assert_allclose(fast_f.beta, ref_f.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast_f.rss, rel=1e-8, abs=1e-8) == ref_f.rss

    fast_b = FastBeamBackwardSelection(
        beam_width=2, allow_worse=True, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=3)
    ref_b = BeamBackwardSelection(
        beam_width=2, allow_worse=True, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=3)
    assert set(fast_b.active_set) == set(ref_b.active_set)
    np.testing.assert_allclose(fast_b.beta, ref_b.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast_b.rss, rel=1e-8, abs=1e-8) == ref_b.rss

    fast_m = FastBeamMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_forward_steps=3, max_total_steps=6)
    ref_m = BeamMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_forward_steps=3, max_total_steps=6)
    assert set(fast_m.active_set) == set(ref_m.active_set)
    np.testing.assert_allclose(fast_m.beta, ref_m.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast_m.rss, rel=1e-8, abs=1e-8) == ref_m.rss


@pytest.mark.parametrize("seed", [4, 5])
def test_fast_cv_equivalence_sweep(seed: int):
    cv_data = _make_cv_regression(seed=seed, folds=3, n=80, p=18)

    fast_f = FastCrossValForwardSelection(
        criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=4)
    ref_f = CrossValForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    assert set(fast_f.active_set) == set(ref_f.active_set)
    assert pytest.approx(fast_f.rss_cv, rel=1e-8, abs=1e-8) == ref_f.rss_cv

    fast_b = FastCrossValBackwardSelection(
        criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    ref_b = CrossValBackwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=3
    )
    assert set(fast_b.active_set) == set(ref_b.active_set)
    assert pytest.approx(fast_b.rss_cv, rel=1e-8, abs=1e-8) == ref_b.rss_cv

    fast_m = FastCrossValMixedSelection(
        criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=5)
    ref_m = CrossValMixedSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_forward_steps=3, max_total_steps=5
    )
    assert set(fast_m.active_set) == set(ref_m.active_set)
    assert pytest.approx(fast_m.rss_cv, rel=1e-8, abs=1e-8) == ref_m.rss_cv


@pytest.mark.parametrize("seed", [6, 7])
def test_fast_cv_beam_equivalence_sweep(seed: int):
    cv_data = _make_cv_regression(seed=seed, folds=3, n=70, p=16)

    fast_f = FastBeamCrossValForwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    ref_f = BeamCrossValForwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    assert set(fast_f.active_set) == set(ref_f.active_set)
    assert pytest.approx(fast_f.rss_cv, rel=1e-8, abs=1e-8) == ref_f.rss_cv

    fast_b = FastBeamCrossValBackwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    ref_b = BeamCrossValBackwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    assert set(fast_b.active_set) == set(ref_b.active_set)
    assert pytest.approx(fast_b.rss_cv, rel=1e-8, abs=1e-8) == ref_b.rss_cv

    fast_m = FastBeamCrossValMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=5)
    ref_m = BeamCrossValMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=5)
    assert set(fast_m.active_set) == set(ref_m.active_set)
    assert pytest.approx(fast_m.rss_cv, rel=1e-8, abs=1e-8) == ref_m.rss_cv


@pytest.mark.parametrize("seed", [8, 9])
def test_fast_grouped_equivalence_sweep(seed: int):
    data = _make_regression(seed=seed, n=100, p=12)
    groups = [list(range(i, i + 3)) for i in range(0, 12, 3)]

    fast_f = FastGroupForwardSelection(groups, criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=3
    )
    ref_f = GroupForwardSelection(groups, criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=3
    )
    assert set(fast_f.active_groups) == set(ref_f.active_groups)
    np.testing.assert_allclose(fast_f.beta, ref_f.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast_f.rss, rel=1e-8, abs=1e-8) == ref_f.rss

    fast_b = FastGroupBackwardSelection(groups, criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=2
    )
    ref_b = GroupBackwardSelection(groups, criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=2
    )
    assert set(fast_b.active_groups) == set(ref_b.active_groups)
    np.testing.assert_allclose(fast_b.beta, ref_b.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast_b.rss, rel=1e-8, abs=1e-8) == ref_b.rss
