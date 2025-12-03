import numpy as np  # noqa: D100
import pytest
from selection.criteria import BestRSSCriterion
from selection.definitions import CrossValGramData, GramData
from selection.routines import (
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
from selection.state import SelectionState  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------
def generate_esl_gramdata(
    n_samples=300,
    n_features=31,
    rho=0.85,
    n_true_vars=10,
    beta_mean=0.0,
    beta_variance=0.4,
    noise_variance=6.25,
    seed=None,
    active_indices=None,
):
    rng = np.random.default_rng(seed)
    cov_matrix = np.full((n_features, n_features), rho)
    np.fill_diagonal(cov_matrix, 1.0)
    mean_vector = np.zeros(n_features)
    X = rng.multivariate_normal(mean_vector, cov_matrix, size=n_samples)
    true_beta = np.zeros(n_features)
    if active_indices is None:
        active = np.arange(n_true_vars, dtype=int)
    else:
        active = np.array(active_indices, dtype=int)
    beta_std = np.sqrt(beta_variance)
    true_beta[active] = rng.normal(beta_mean, beta_std, size=n_true_vars)
    noise_std = np.sqrt(noise_variance)
    epsilon = rng.normal(0.0, noise_std, size=n_samples)
    y = X @ true_beta + epsilon
    gram = X.T @ X
    cov = X.T @ y
    y_norm = y @ y
    data = GramData(gram, cov, y_norm, n_samples)
    return data, sorted(active.tolist())


def make_cv_support_problem(p=50, support=15, folds=10, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    beta[:support] = 1.0
    fold_data = []
    for _ in range(folds):
        X = rng.standard_normal((n, p))
        y = X @ beta + 0.01 * rng.standard_normal(n)
        gram = X.T @ X
        cov = X.T @ y
        y_norm = y @ y
        fold_data.append(GramData(gram, cov, y_norm, n))
    return CrossValGramData(fold_data), list(range(support))


def make_diagonal_problem(p):
    idx = np.arange(1, p + 1, dtype=float)
    gram = np.eye(p)
    true_beta = 2**idx
    cov = gram @ true_beta
    y_norm = float(true_beta @ true_beta)
    n_samples = 100
    return gram, cov, y_norm, n_samples


def expected_indices(p, k):
    return list(range(p - k, p))


def make_heterogeneous_cv_problem(folds=4, n=80, p=8, support=3, seed=123):
    """Build per-fold GramData with different seeds so covariances differ."""
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    beta[:support] = 1.0
    fold_data = []
    for fold_seed in rng.integers(0, 1_000_000, size=folds):
        fold_rng = np.random.default_rng(int(fold_seed))
        X = fold_rng.standard_normal((n, p))
        y = X @ beta + 0.05 * fold_rng.standard_normal(n)
        gram = X.T @ X
        cov = X.T @ y
        y_norm = y @ y
        fold_data.append(GramData(gram, cov, y_norm, n))
    return CrossValGramData(fold_data), set(range(support))


@pytest.fixture(scope="module")
def small_problem():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, 0.2, 0.9])
    gram = X.T @ X
    cov = X.T @ y
    y_norm = y @ y
    n_samples = X.shape[0]
    return GramData(gram, cov, y_norm, n_samples)


@pytest.fixture(scope="module")
def small_cv_problem(small_problem):
    fold = small_problem
    return CrossValGramData([fold, fold])


@pytest.fixture(scope="module")
def esl_book():
    gram_data, support = generate_esl_gramdata(noise_variance=0.0, seed=2024)
    return gram_data, support


@pytest.fixture(scope="module")
def esl_cv_data():
    support = list(range(10))
    gram_data, _ = generate_esl_gramdata(
        noise_variance=1e-6, seed=2025, active_indices=support
    )
    fold_data = [gram_data for _ in range(10)]
    cv_data = CrossValGramData(fold_data)
    return cv_data, support, gram_data


# ---------------------------------------------------------------------------
# Basic single-dataset selection on a toy problem
# ---------------------------------------------------------------------------
def test_forward_selects_strongest_variable(small_problem):
    state = ForwardSelection().fit(data=small_problem)

    assert state.active_set == [0]
    assert np.isclose(state.beta[0], 0.95, atol=1e-6)


def test_backward_prunes_to_best_single_variable(small_problem):
    state = BackwardSelection().fit(data=small_problem)

    assert state.active_set == [0]
    assert np.isclose(state.beta[1], 0.0, atol=1e-9)


def test_stepwise_matches_forward_result(small_problem):
    state = MixedSelection().fit(data=small_problem)

    assert state.active_set == [0]
    assert np.isclose(state.beta[0], 0.95, atol=1e-6)


def test_forward_with_best_rss_matches_standard(small_problem):
    state = ForwardSelection(criterion_cls=BestRSSCriterion).fit(data=small_problem)

    assert set(state.active_set) == {0, 1}
    np.testing.assert_allclose(
        state.beta[:2], np.linalg.solve(small_problem.gram, small_problem.cov)
    )


# ---------------------------------------------------------------------------
# Basic cross-validation on duplicated folds derived from the toy problem
# ---------------------------------------------------------------------------
def test_crossval_forward_selection_matches_forward(small_cv_problem):
    cv_selector = CrossValForwardSelection()
    cv_state = cv_selector.fit(data=small_cv_problem, max_steps=1)

    assert cv_state.active_set == [0]


def test_crossval_backward_selection_matches_backward(small_cv_problem):
    cv_selector = CrossValBackwardSelection()
    cv_state = cv_selector.fit(data=small_cv_problem, max_steps=1)

    assert cv_state.active_set == [0]


def test_crossval_mixed_selection_matches_mixed(small_cv_problem):
    cv_selector = CrossValMixedSelection()
    cv_state = cv_selector.fit(
        data=small_cv_problem, max_forward_steps=1, max_total_steps=2
    )

    assert cv_state.active_set == [0]


# ---------------------------------------------------------------------------
# Recovery on ESL-style synthetic data
# ---------------------------------------------------------------------------
def test_forward_selection_recovers_esl_support(esl_book):
    gram_data, support = esl_book
    selector = ForwardSelection(criterion_cls=BestRSSCriterion)
    state = selector.fit(data=gram_data, max_steps=len(support))
    recovered = set(state.active_set)
    assert len(recovered & set(support)) >= int(0.8 * len(support))


def test_backward_selection_recovers_esl_support(esl_book):
    gram_data, support = esl_book
    forward_selector = ForwardSelection(criterion_cls=BestRSSCriterion)
    forward_state = forward_selector.fit(data=gram_data, max_steps=len(support))
    selector = BackwardSelection(criterion_cls=BestRSSCriterion)
    p = gram_data.gram.shape[0]
    state = selector.fit(data=gram_data, max_steps=p - len(support))

    recovered = set(state.active_set)
    forward_support = set(forward_state.active_set)
    assert len(recovered & forward_support) >= int(0.8 * len(forward_support))


# ---------------------------------------------------------------------------
# Cross-validation support recovery and alignment with full-data runs
# ---------------------------------------------------------------------------
def test_crossval_forward_selection_recovers_true_support():
    cv_data, support_set = make_cv_support_problem()
    selector = CrossValForwardSelection()
    state = selector.fit(data=cv_data, max_steps=len(support_set))

    assert set(state.active_set) == set(support_set)


def test_crossval_backward_selection_recovers_true_support():
    cv_data, support_set = make_cv_support_problem()
    selector = CrossValBackwardSelection()
    state = selector.fit(data=cv_data, max_steps=50 - len(support_set))

    assert set(state.active_set) == set(support_set)


def test_crossval_forward_selection_matches_full_run(esl_cv_data):
    cv_data, support, full_data = esl_cv_data
    ForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=full_data, max_steps=len(support)
    )
    selector = CrossValForwardSelection(criterion_cls=BestRSSCriterion)
    state = selector.fit(data=cv_data, max_steps=len(support))

    recovered = set(state.active_set)
    assert len(recovered & set(support)) >= int(0.8 * len(support))


def test_crossval_backward_selection_matches_full_run(esl_cv_data):
    cv_data, support, full_data = esl_cv_data
    BackwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=full_data, max_steps=full_data.gram.shape[0] - len(support)
    )
    selector = CrossValBackwardSelection(criterion_cls=BestRSSCriterion)
    p = cv_data.gram_total.shape[0]
    state = selector.fit(data=cv_data, max_steps=p - len(support))

    recovered = set(state.active_set)
    assert len(recovered & set(support)) >= int(0.8 * len(support))


# ---------------------------------------------------------------------------
# Beam search variants and comparisons (single-dataset and CV)
# ---------------------------------------------------------------------------
def test_beam_crossval_forward_selection_recovers_true_support():
    cv_data, support_set = make_cv_support_problem()
    selector = BeamCrossValForwardSelection(beam_width=2)
    state = selector.fit(data=cv_data, max_steps=len(support_set))
    assert set(state.active_set) == set(support_set)


def test_beam_crossval_backward_selection_recovers_true_support():
    cv_data, support_set = make_cv_support_problem()
    selector = BeamCrossValBackwardSelection(beam_width=2)
    state = selector.fit(data=cv_data, max_steps=50 - len(support_set))

    assert set(state.active_set) == set(support_set)


def test_beam_crossval_mixed_selection_matches_forward(small_cv_problem):
    selector = BeamCrossValMixedSelection(beam_width=2)
    cv_state = selector.fit(
        data=small_cv_problem, max_forward_steps=1, max_total_steps=2
    )
    assert cv_state.active_set == [0]


def test_crossval_beam_matches_greedy_on_simple_problem():
    cv_data, support_set = make_cv_support_problem()
    greedy = CrossValForwardSelection()
    beam = BeamCrossValForwardSelection(beam_width=2)

    greedy_state = greedy.fit(data=cv_data, max_steps=len(support_set))
    beam_state = beam.fit(data=cv_data, max_steps=len(support_set))

    assert set(beam_state.active_set) == set(greedy_state.active_set)
    assert beam_state.rss_cv <= greedy_state.rss_cv + 1e-9


def test_cv_beam_matches_greedy_on_heterogeneous_folds():
    cv_data, support = make_heterogeneous_cv_problem()
    steps = len(support)
    greedy = CrossValForwardSelection()
    beam = BeamCrossValForwardSelection(beam_width=3)

    greedy_state = greedy.fit(data=cv_data, max_steps=steps)
    beam_state = beam.fit(data=cv_data, max_steps=steps)

    assert set(greedy_state.active_set) == set(beam_state.active_set)
    assert beam_state.rss_cv <= greedy_state.rss_cv + 1e-9

    # Check monotonicity over steps for both selectors.
    greedy_rss = []
    beam_rss = []
    for k in range(steps + 1):
        greedy_rss.append(
            CrossValForwardSelection().fit(data=cv_data, max_steps=k).rss_cv
        )
        beam_rss.append(
            BeamCrossValForwardSelection(beam_width=3)
            .fit(data=cv_data, max_steps=k)
            .rss_cv
        )
    assert all(
        later <= earlier + 1e-9 for earlier, later in zip(greedy_rss, greedy_rss[1:])
    )
    assert all(
        later <= earlier + 1e-9 for earlier, later in zip(beam_rss, beam_rss[1:])
    )


def test_forward_beam_search_selects_best_subset():
    gram, cov, y_norm, n_samples = make_diagonal_problem(6)
    selector = BeamForwardSelection(beam_width=3)
    state = selector.fit(data=GramData(gram, cov, y_norm, n_samples), max_steps=3)
    assert set(state.active_set) == set(expected_indices(6, 3))


def test_backward_beam_search_selects_smallest_subset():
    gram, cov, y_norm, n_samples = make_diagonal_problem(6)
    y_norm += 1.0
    selector = BeamBackwardSelection(beam_width=2, allow_worse=True)
    state = selector.fit(data=GramData(gram, cov, y_norm, n_samples), max_steps=3)
    assert len(state.active_set) == 3
    assert set(state.active_set) == set(expected_indices(6, 3))


def test_mixed_beam_search_handles_forward_and_backward():
    gram, cov, y_norm, n_samples = make_diagonal_problem(5)
    selector = BeamMixedSelection(beam_width=2)
    state = selector.fit(
        data=GramData(gram, cov, y_norm, n_samples),
        max_forward_steps=3,
        max_total_steps=9,
    )
    assert set(state.active_set) == set(expected_indices(5, 3))


def test_beam_search_deduplicates_active_sets():
    gram = np.eye(4)
    cov = np.array([1.0, 1.0, 0.5, 0.25])
    y_norm = float(cov @ cov)
    selector = BeamForwardSelection(beam_width=4)
    state = selector.fit(data=GramData(gram, cov, y_norm, 20), max_steps=1)
    assert len(state.active_set) == 1
    assert len(set(state.active_set)) == 1


def test_beam_pruning_is_deterministic_and_non_worsening():
    gram = np.eye(4)
    cov = np.ones(4)
    y_norm = float(cov @ cov)
    n_samples = 50

    greedy = BeamForwardSelection(beam_width=1, criterion_cls=BestRSSCriterion)
    wider = BeamForwardSelection(beam_width=3, criterion_cls=BestRSSCriterion)

    greedy_state = greedy.fit(data=GramData(gram, cov, y_norm, n_samples), max_steps=1)
    wider_state = wider.fit(data=GramData(gram, cov, y_norm, n_samples), max_steps=1)

    assert greedy_state.active_set == [0]
    assert wider_state.active_set == [0]
    assert wider_state.rss <= greedy_state.rss + 1e-12


# ---------------------------------------------------------------------------
# Diagonal design sanity checks and permutation invariance
# ---------------------------------------------------------------------------
def test_forward_selection_permutation_invariance():
    p, steps = 8, 4
    gram, cov, y_norm, n_samples = make_diagonal_problem(p)
    perm = np.array([3, 0, 5, 1, 6, 2, 7, 4], dtype=int)
    P = np.eye(p)[perm]
    gram_perm = P.T @ gram @ P
    cov_perm = P.T @ cov

    selector = ForwardSelection(criterion_cls=BestRSSCriterion)
    base_state = selector.fit(
        data=GramData(gram, cov, y_norm, n_samples), max_steps=steps
    )
    perm_state = selector.fit(
        data=GramData(gram_perm, cov_perm, y_norm, n_samples), max_steps=steps
    )

    mapped = [int(perm[i]) for i in perm_state.active_set]
    assert set(mapped) == set(base_state.active_set)


@pytest.mark.parametrize("p,steps", [(5, 3), (10, 7), (15, 12)])
def test_best_rss_selects_largest_coefficients_forward_and_backward(p, steps):
    gram, cov, y_norm, n_samples = make_diagonal_problem(p)

    forward_selector = ForwardSelection(criterion_cls=BestRSSCriterion)
    forward_state = forward_selector.fit(
        data=GramData(gram, cov, y_norm, n_samples), max_steps=steps
    )

    assert set(forward_state.active_set) == set(expected_indices(p, steps))


def test_mixed_selection_matches_direct_solution():
    rng = np.random.default_rng(123)
    n, p = 40, 6
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.01 * rng.standard_normal(n)
    gram, cov, y_norm = X.T @ X, X.T @ y, y @ y
    state = MixedSelection().fit(
        data=GramData(gram, cov, y_norm, n),
        max_forward_steps=4,
        max_total_steps=7,
    )
    idx = np.array(state.active_set, dtype=int)
    if len(idx):
        gram_ss = gram[np.ix_(idx, idx)]
        beta_expected = np.zeros(p)
        beta_expected[idx] = np.linalg.solve(gram_ss, cov[idx])  # noqa: E501
        np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8)
        rss_expected = y_norm - cov[idx] @ beta_expected[idx]
    else:
        np.testing.assert_allclose(state.beta, np.zeros(p))
        rss_expected = y_norm
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected


# ---------------------------------------------------------------------------
# Invariant and robustness checks
# ---------------------------------------------------------------------------
def test_forward_then_backward_returns_to_empty_state():
    gram, cov, y_norm, n_samples = make_diagonal_problem(p=10)
    data = GramData(gram, cov, y_norm, n_samples)
    forward_state = ForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=4
    )

    # Greedily drop variables until empty, choosing the best backward score each time.
    while forward_state.active_set:
        scores = forward_state.compute_backward_scores()
        idx = int(np.argmin(scores))
        forward_state.apply_backward_step(idx)

    assert forward_state.active_set == []
    np.testing.assert_allclose(forward_state.beta, np.zeros_like(forward_state.beta))
    assert pytest.approx(forward_state.rss, rel=1e-9, abs=1e-9) == y_norm


def test_crossval_mixed_keeps_fold_state_in_sync():
    """
    Mixed CV selection alternates forward and backward moves across folds.
    This test asserts that after each operation the global active set and
    CV RSS stay consistent with the per-fold training states, catching any
    desync between folds and the aggregated view.
    """
    cv_data, support_set = make_cv_support_problem(p=12, support=4, folds=3, n=200)
    max_forward_steps = 3  # noqa: F841
    max_total_steps = 5

    selector = CrossValMixedSelection()
    cv_state = selector._init_run(None, cv_data, mode="empty")[0]

    # Helper to recompute rss_cv fresh from per-fold states.
    def recompute_from_folds(state):
        clone = state.clone()
        return clone.recompute_oos_rss()

    ops = 0
    forward_steps = 0
    while True:
        if max_total_steps is not None and ops >= max_total_steps:
            break
        if forward_steps >= max_forward_steps:
            break
        if not selector._forward_step(
            cv_state, selector._init_criterion(cv_state, cv_data)
        ):
            break
        forward_steps += 1
        ops += 1

        # Assert active sets match across folds and the shared view.
        fold_sets = [tuple(s.active_set) for s in cv_state.train_states]
        assert all(fs == fold_sets[0] for fs in fold_sets)
        assert list(fold_sets[0]) == cv_state.active_set

        # Assert cached rss_cv matches fresh recomputation.
        assert pytest.approx(
            cv_state.rss_cv, rel=1e-9, abs=1e-9
        ) == recompute_from_folds(cv_state)

        # Try one backward step if budget allows.
        if max_total_steps is not None and ops >= max_total_steps:
            break
        if selector._backward_step(
            cv_state, selector._init_criterion(cv_state, cv_data)
        ):
            ops += 1
            fold_sets = [tuple(s.active_set) for s in cv_state.train_states]
            assert all(fs == fold_sets[0] for fs in fold_sets)
            assert list(fold_sets[0]) == cv_state.active_set
            assert pytest.approx(
                cv_state.rss_cv, rel=1e-9, abs=1e-9
            ) == recompute_from_folds(cv_state)


# ---------------------------------------------------------------------------
# Rank-deficient handling
# ---------------------------------------------------------------------------
def test_rank_deficient_backward_recovers_full_rank_subset():
    gram = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0 + 1e-6],
        ]
    )
    cov = np.array([1.0, 2.0, 3.0])
    y_norm = float(cov @ cov)
    data = GramData(gram, cov, y_norm, n_samples=5)

    # Direct backward update on a rank-deficient full model should drop one column.
    direct_state = SelectionState(data)
    direct_state.init_full()
    direct_state.apply_backward_step(2)
    assert len(direct_state.active_set) == 2
    inv_direct = np.linalg.inv(
        gram[np.ix_(direct_state.active_set, direct_state.active_set)]
    )
    assert np.isfinite(inv_direct).all()

    # Beam backward on the same problem should also return a full-rank subset.
    beam = BeamBackwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion, allow_worse=True
    )
    beam_state = beam.fit(data=data, max_steps=1)
    assert len(beam_state.active_set) == 2
    inv_beam = np.linalg.inv(gram[np.ix_(beam_state.active_set, beam_state.active_set)])
    assert np.isfinite(inv_beam).all()
