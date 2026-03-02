import numpy as np
import pytest

from selection import CrossValGramData, GramData


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


def make_cv_beam_trap_problem(folds=4, n_samples=50):
    """CV analogue of the 3-feature trap where beam(2) beats greedy in 2 steps."""
    rho = 0.6
    c0 = 1.2
    gram = np.array([[1.0, rho, rho], [rho, 1.0, 0.0], [rho, 0.0, 1.0]])
    cov = np.array([c0, 1.0, 1.0])
    y_norm = float(cov @ np.linalg.solve(gram, cov) + 1.0)
    fold = GramData(gram, cov, y_norm, n_samples=n_samples)
    return CrossValGramData([fold for _ in range(folds)])


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

