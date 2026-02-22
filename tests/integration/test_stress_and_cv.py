import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import CrossValGramData, GramData
from selection.routines import BackwardSelection, CrossValForwardSelection, ForwardSelection


def _ill_conditioned_gram(p: int, cond_exponent: float = 6.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    eigvals = np.logspace(0.0, -cond_exponent, p)
    gram = Q @ np.diag(eigvals) @ Q.T
    gram = 0.5 * (gram + gram.T)
    return gram


def test_ill_conditioned_forward_and_backward_are_finite():
    p = 60
    gram = _ill_conditioned_gram(p, cond_exponent=6.0, seed=42)
    rng = np.random.default_rng(42)
    cov = rng.standard_normal(p)
    quadratic = float(cov @ np.linalg.solve(gram, cov))
    y_norm = quadratic + 10.0

    data = GramData(gram, cov, y_norm, n_samples=500)

    forward_state = ForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=10
    )
    assert np.isfinite(forward_state.rss)
    assert np.isfinite(forward_state.beta).all()

    backward_state = BackwardSelection(
        criterion_cls=BestRSSCriterion, allow_worse=True
    ).fit(data=data, max_steps=5)
    assert np.isfinite(backward_state.rss)
    assert np.isfinite(backward_state.beta).all()


def test_cv_forward_with_variable_fold_sizes_matches_manual_rss():
    rng = np.random.default_rng(7)
    p = 6
    fold_sizes = [35, 50, 25]
    fold_data = []
    beta_true = rng.standard_normal(p)

    for n in fold_sizes:
        X = rng.standard_normal((n, p))
        y = X @ beta_true + 0.05 * rng.standard_normal(n)
        fold_data.append(GramData(X.T @ X, X.T @ y, y @ y, n))

    cv_data = CrossValGramData(fold_data)
    assert cv_data.n_samples_total == sum(fold_sizes)

    cv_state = CrossValForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=3
    )

    idx = np.array(cv_state.active_set, dtype=int)
    manual_sum = 0.0
    for k, fold in enumerate(fold_data):
        train = cv_data.train_data_for_fold(k)
        if idx.size:
            beta_train = np.linalg.solve(train.gram[np.ix_(idx, idx)], train.cov[idx])
            G_val = fold.gram[np.ix_(idx, idx)]
            c_val = fold.cov[idx]
            y_norm = fold.y_norm
            rss_k = y_norm - 2.0 * float(beta_train @ c_val) + float(
                beta_train @ (G_val @ beta_train)
            )
        else:
            rss_k = fold.y_norm
        manual_sum += rss_k

    assert pytest.approx(cv_state.rss_cv, rel=1e-8, abs=1e-8) == manual_sum
