"""Targeted behavioral tests for cross-cutting properties.

Each test validates a property that, if violated, would indicate a real
algorithmic bug rather than a mere code-path exercise.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from selection.criteria import AICCriterion, BestRSSCriterion
from selection.definitions import CrossValGramData, GramData
from selection.routines_core import (
    BackwardSelection,
    CrossValForwardSelection,
    ForwardState,
    MixedSelection,
)
from selection.grouped_routines import (
    GroupBackwardSelection,
    GroupForwardSelection,
)
from selection.routines import ForwardSelection
from selection.state import SelectionState
from tests.helpers import explicit_beta_rss, explicit_cv_rss, make_regression_gram


# ---------------------------------------------------------------------------
# Test 1: ForwardState and SelectionState pick identical features
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [10, 11, 12, 13, 14, 15])
def test_fast_and_reference_select_same_features_stepwise(seed: int):
    """Both implementations must choose the same feature at every forward step
    and produce identical backward scores from the full model."""
    n, p = 80, 10
    data = make_regression_gram(seed, n=n, p=p)

    fast = ForwardState.create(data, tol=1e-10)
    ref = SelectionState(data)
    ref.init_empty()

    max_k = 6
    for _ in range(max_k):
        # Forward candidates from both implementations.
        fast_scored = fast.candidate_scores()
        ref_cache = ref.compute_forward_deltas()
        assert fast_scored is not None and ref_cache is not None

        fast_candidates, fast_rss_new = fast_scored
        ref_candidates = ref_cache.candidates
        ref_rss_new = ref_cache.rss_new

        # Both should agree on the best candidate (minimum RSS).
        fast_best_pos = int(np.argmin(fast_rss_new))
        fast_best_feat = int(fast_candidates[fast_best_pos])

        ref_best_pos = int(np.argmin(ref_rss_new))
        ref_best_feat = int(ref_candidates[ref_best_pos])

        assert fast_best_feat == ref_best_feat, (
            f"Step {len(fast.active_set)+1}: fast chose {fast_best_feat}, "
            f"ref chose {ref_best_feat}"
        )

        # Apply the step in both.
        fast.apply_forward(fast_best_feat)
        ref.apply_forward_step(ref_cache, ref_best_pos)

    # From the full model, backward scores should match.
    full_fast = ForwardState.from_active_set(data, list(range(p)), tol=1e-10)
    full_ref = SelectionState(data)
    full_ref.init_full()

    fast_bw = full_fast.backward_scores()
    ref_bw = full_ref.compute_backward_scores()
    assert fast_bw is not None and ref_bw is not None
    np.testing.assert_allclose(fast_bw, ref_bw, atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------------------
# Test 2: Backward RSS is monotone increasing and matches OLS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(24)))
def test_backward_rss_monotone_and_matches_ols(seed: int):
    """Backward selection with BestRSS must produce monotonically increasing
    RSS, and each intermediate model must match direct OLS on its support."""
    n, p = 80, 8
    data = make_regression_gram(seed, n=n, p=p)

    rss_path: list[float] = []
    for max_steps in range(p + 1):
        state = BackwardSelection(
            criterion_cls=BestRSSCriterion, allow_worse=True
        ).fit(data=data, max_steps=max_steps)

        # Verify beta/RSS match direct OLS on the active support.
        beta_ols, rss_ols = explicit_beta_rss(data, state.active_set)
        np.testing.assert_allclose(state.beta, beta_ols, atol=1e-8, rtol=1e-8)
        assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_ols

        rss_path.append(state.rss)

    # RSS must be monotonically non-decreasing (backward removes features).
    for i in range(len(rss_path) - 1):
        assert rss_path[i + 1] >= rss_path[i] - 1e-9, (
            f"RSS decreased at step {i+1}: {rss_path[i+1]:.12f} < {rss_path[i]:.12f}"
        )


# ---------------------------------------------------------------------------
# Test 3: CV forward finds optimal subset on diagonal Gram
# ---------------------------------------------------------------------------


def test_cv_forward_finds_optimal_subset_on_diagonal():
    """On a diagonal Gram matrix, CV forward selection must pick the globally
    optimal k-subset (the k features with largest |cov|), verified by
    exhaustive enumeration."""
    p = 8
    n_folds = 3
    n_per_fold = 40

    rng = np.random.default_rng(42)

    # Diagonal Gram: features are independent.
    # Covariances are distinct so that rankings are unambiguous.
    cov_signal = np.array([4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5])

    fold_data = []
    for _ in range(n_folds):
        gram = float(n_per_fold) * np.eye(p)
        noise = rng.standard_normal(p) * 0.1
        cov = float(n_per_fold) * cov_signal + noise
        y_norm = float(cov @ np.linalg.solve(gram, cov)) + float(n_per_fold) * 2.0
        fold_data.append(GramData(gram, cov, y_norm, n_per_fold))

    cv_data = CrossValGramData(fold_data)

    for k in range(1, 5):
        # Find the globally optimal k-subset by exhaustive enumeration.
        best_cv_rss = float("inf")
        best_subset = None
        for subset in itertools.combinations(range(p), k):
            cv_rss = explicit_cv_rss(cv_data, list(subset))
            if cv_rss < best_cv_rss:
                best_cv_rss = cv_rss
                best_subset = set(subset)

        # Run CV forward selection.
        state = CrossValForwardSelection(criterion_cls=BestRSSCriterion).fit(
            data=cv_data, max_steps=k
        )

        selected_cv_rss = explicit_cv_rss(cv_data, state.active_set)
        assert selected_cv_rss <= best_cv_rss + 1e-6, (
            f"k={k}: selected CV RSS {selected_cv_rss:.8f} > "
            f"optimal {best_cv_rss:.8f}"
        )
        assert set(state.active_set) == best_subset, (
            f"k={k}: selected {set(state.active_set)}, optimal {best_subset}"
        )


# ---------------------------------------------------------------------------
# Test 4: CV fold partition is exact
# ---------------------------------------------------------------------------


def test_cv_fold_partition_is_exact():
    """For every fold k, train + val Gram statistics must sum to the total."""
    rng = np.random.default_rng(77)
    p = 6

    # Use distinct fold sizes for generality.
    fold_sizes = [20, 25, 30, 18, 27]
    fold_data = []
    for n_k in fold_sizes:
        X = rng.standard_normal((n_k, p))
        y = rng.standard_normal(n_k)
        fold_data.append(GramData(X.T @ X, X.T @ y, y @ y, n_k))

    cv_data = CrossValGramData(fold_data)

    for k in range(cv_data.n_folds):
        train = cv_data.train_data_for_fold(k)
        val = cv_data.val_data_for_fold(k)

        np.testing.assert_allclose(
            train.gram + val.gram, cv_data.gram_total, atol=1e-12
        )
        np.testing.assert_allclose(
            train.cov + val.cov, cv_data.cov_total, atol=1e-12
        )
        assert abs(
            (train.y_norm + val.y_norm) - cv_data.y_norm_total
        ) < 1e-12, (
            f"Fold {k}: y_norm mismatch "
            f"{train.y_norm + val.y_norm} != {cv_data.y_norm_total}"
        )
        assert train.n_samples + val.n_samples == cv_data.n_samples_total


# ---------------------------------------------------------------------------
# Test 5: Grouped active set is always union of complete groups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [20, 21, 22])
def test_grouped_active_set_is_always_union_of_complete_groups(seed: int):
    """At every intermediate step, the active set must be exactly the union
    of complete groups — never a partial group."""
    n, p = 100, 12
    groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    data = make_regression_gram(seed, n=n, p=p)

    # Forward: start empty, add groups one by one.
    for max_steps in range(1, len(groups) + 1):
        result = GroupForwardSelection(
            groups, criterion_cls=BestRSSCriterion
        ).fit(data=data, max_steps=max_steps)

        expected_features = set()
        for g in result.active_groups:
            expected_features.update(groups[g])

        # Verify through beta: nonzero entries should match expected_features.
        nonzero = set(int(i) for i in np.nonzero(result.beta)[0])
        assert nonzero == expected_features, (
            f"Forward step {max_steps}: nonzero features {nonzero} "
            f"!= expected {expected_features}"
        )
        assert len(result.active_groups) == max_steps

    # Backward: use AIC so removing groups can improve the criterion.
    # Start full, remove groups one by one.
    for max_steps in range(1, len(groups)):
        result = GroupBackwardSelection(
            groups, criterion_cls=AICCriterion
        ).fit(data=data, max_steps=max_steps)

        expected_features = set()
        for g in result.active_groups:
            expected_features.update(groups[g])

        # Check completeness: no partial groups.
        nonzero = set(int(i) for i in np.nonzero(result.beta)[0])
        assert nonzero == expected_features, (
            f"Backward step {max_steps}: nonzero features {nonzero} "
            f"!= expected {expected_features}"
        )
        # The number of active groups must decrease by exactly the number
        # of steps taken (or fewer if the criterion stops early).
        assert len(result.active_groups) <= len(groups)
        assert len(result.active_groups) >= len(groups) - max_steps


# ---------------------------------------------------------------------------
# Test 6: Mixed selection improves over pure forward
# ---------------------------------------------------------------------------


def test_mixed_selection_improves_over_pure_forward():
    """On a problem where greedy forward gets trapped, mixed selection
    (forward + backward refinement) must find a strictly better AIC than
    pure forward.

    With n=10 and p=6, the AIC penalty (2 per feature) is significant.
    Forward greedily adds features that later become redundant once better
    features enter the model. Mixed selection's backward step can remove
    those redundant features, opening the door for superior alternatives.
    """
    n_samples = 10
    p = 6

    # Generate a regression problem where the forward path includes a feature
    # that becomes redundant (removable by backward step) once a later,
    # stronger feature is added.
    rng = np.random.default_rng(117)
    X = rng.standard_normal((n_samples, p))
    beta_true = np.zeros(p)
    support = rng.choice(p, size=2, replace=False)
    beta_true[support] = rng.standard_normal(2) * 3
    y = X @ beta_true + 0.3 * rng.standard_normal(n_samples)

    data = GramData(X.T @ X, X.T @ y, y @ y, n_samples)

    # Pure forward with AIC.
    forward_state = ForwardSelection(criterion_cls=AICCriterion).fit(data=data)

    # Mixed with AIC: backward refinement after each forward step can remove
    # features that became redundant, allowing a strictly better model.
    mixed_state = MixedSelection(criterion_cls=AICCriterion).fit(
        data=data, max_total_steps=30
    )

    # Compute AIC for both.
    aic_crit = AICCriterion(n_samples=n_samples)
    forward_aic = float(
        np.asarray(aic_crit.evaluate(forward_state.rss, len(forward_state.active_set)))
    )
    mixed_aic = float(
        np.asarray(aic_crit.evaluate(mixed_state.rss, len(mixed_state.active_set)))
    )

    assert mixed_aic < forward_aic - 1e-9, (
        f"Mixed AIC ({mixed_aic:.8f}) should be strictly better than "
        f"forward AIC ({forward_aic:.8f})"
    )
