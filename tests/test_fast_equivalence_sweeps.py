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


def _explicit_beta_rss(data: GramData, active_set):
    p = data.gram.shape[0]
    beta = np.zeros(p, dtype=float)
    if not active_set:
        return beta, float(data.y_norm)
    idx = np.array(active_set, dtype=int)
    G_ss = data.gram[np.ix_(idx, idx)]
    cov_s = data.cov[idx]
    beta_s = np.linalg.solve(G_ss, cov_s)
    beta[idx] = beta_s
    rss = float(data.y_norm - cov_s @ beta_s)
    return beta, rss


def _explicit_cv_rss(cv_data: CrossValGramData, active_set):
    if not active_set:
        return float(np.sum(cv_data.y_norm_folds))
    idx = np.array(active_set, dtype=int)
    total = 0.0
    for k in range(cv_data.n_folds):
        train = cv_data.train_data_for_fold(k)
        G_train = train.gram[np.ix_(idx, idx)]
        cov_train = train.cov[idx]
        beta = np.linalg.solve(G_train, cov_train)
        G_val = cv_data.gram_folds[k]
        c_val = cv_data.cov_folds[k]
        y_norm_val = cv_data.y_norm_folds[k]
        G_val_ss = G_val[np.ix_(idx, idx)]
        rss = y_norm_val - 2.0 * float(beta @ c_val[idx]) + float(
            beta @ (G_val_ss @ beta)
        )
        total += rss
    return float(total)


def _assert_state_consistent(state, data: GramData):
    beta, rss = _explicit_beta_rss(data, state.active_set)
    np.testing.assert_allclose(state.beta, beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss


def _assert_group_state_consistent(state, data: GramData, groups):
    active_groups = state.active_groups
    idx = []
    for g in active_groups:
        idx.extend(groups[g])
    beta, rss = _explicit_beta_rss(data, idx)
    np.testing.assert_allclose(state.beta, beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("p", [32, 80])  # include p > 64 to exercise capacity growth
def test_fast_greedy_consistency_sweep(seed: int, p: int):
    data = _make_regression(seed=seed, n=140, p=p)
    max_steps = min(6, p // 2)

    fast_f = FastForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=max_steps
    )
    _assert_state_consistent(fast_f, data)

    fast_b = FastBackwardSelection(allow_worse=True).fit(data=data, max_steps=4)
    _assert_state_consistent(fast_b, data)

    fast_m = FastMixedSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_forward_steps=3, max_total_steps=5
    )
    _assert_state_consistent(fast_m, data)


@pytest.mark.parametrize("seed", [2, 3])
def test_fast_beam_consistency_sweep(seed: int):
    data = _make_regression(seed=seed, n=120, p=25)

    fast_f = FastBeamForwardSelection(
        beam_width=3, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=4)
    _assert_state_consistent(fast_f, data)

    fast_b = FastBeamBackwardSelection(
        beam_width=2, allow_worse=True, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=3)
    _assert_state_consistent(fast_b, data)

    fast_m = FastBeamMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_forward_steps=3, max_total_steps=6)
    _assert_state_consistent(fast_m, data)


@pytest.mark.parametrize("seed", [4, 5])
def test_fast_cv_consistency_sweep(seed: int):
    cv_data = _make_cv_regression(seed=seed, folds=3, n=80, p=18)

    fast_f = FastCrossValForwardSelection(
        criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=4)
    assert pytest.approx(fast_f.rss_cv, rel=1e-8, abs=1e-8) == _explicit_cv_rss(
        cv_data, fast_f.active_set
    )

    fast_b = FastCrossValBackwardSelection(
        criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    assert pytest.approx(fast_b.rss_cv, rel=1e-8, abs=1e-8) == _explicit_cv_rss(
        cv_data, fast_b.active_set
    )

    fast_m = FastCrossValMixedSelection(
        criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=5)
    assert pytest.approx(fast_m.rss_cv, rel=1e-8, abs=1e-8) == _explicit_cv_rss(
        cv_data, fast_m.active_set
    )


@pytest.mark.parametrize("seed", [6, 7])
def test_fast_cv_beam_consistency_sweep(seed: int):
    cv_data = _make_cv_regression(seed=seed, folds=3, n=70, p=16)

    fast_f = FastBeamCrossValForwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    assert pytest.approx(fast_f.rss_cv, rel=1e-8, abs=1e-8) == _explicit_cv_rss(
        cv_data, fast_f.active_set
    )

    fast_b = FastBeamCrossValBackwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    assert pytest.approx(fast_b.rss_cv, rel=1e-8, abs=1e-8) == _explicit_cv_rss(
        cv_data, fast_b.active_set
    )

    fast_m = FastBeamCrossValMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=5)
    assert pytest.approx(fast_m.rss_cv, rel=1e-8, abs=1e-8) == _explicit_cv_rss(
        cv_data, fast_m.active_set
    )


@pytest.mark.parametrize("seed", [8, 9])
def test_fast_grouped_consistency_sweep(seed: int):
    data = _make_regression(seed=seed, n=100, p=12)
    groups = [list(range(i, i + 3)) for i in range(0, 12, 3)]

    fast_f = FastGroupForwardSelection(groups, criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=3
    )
    _assert_group_state_consistent(fast_f, data, groups)

    fast_b = FastGroupBackwardSelection(groups, criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=2
    )
    _assert_group_state_consistent(fast_b, data, groups)
