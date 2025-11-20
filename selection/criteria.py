import numpy as np

from .constants import ABS_TOL


class SelectionCriterion:
    """Pure criterion evaluator with optional score tracking."""

    def __init__(
        self,
        *,
        minimize: bool = True,
        abs_tol: float = ABS_TOL,
        rel_tol: float = 1e-8,
    ):
        self.minimize = minimize
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.current_value: float = float("inf") if minimize else float("-inf")

    def evaluate(self, rss, k: int):
        raise NotImplementedError

    def is_improvement(self, candidate: float, incumbent=None) -> bool:
        ref = self.current_value if incumbent is None else incumbent
        tol = self.abs_tol + abs(ref) * self.rel_tol
        if self.minimize:
            return candidate < ref - tol
        return candidate > ref + tol

    def update_current(self, value: float):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Cannot record non-finite criterion value: {value}")
        self.current_value = value

    def clone(self) -> "SelectionCriterion":
        from copy import deepcopy

        return deepcopy(self)

    def best_candidate(self, rss, k: int):
        scores = np.asarray(self.evaluate(rss, k))
        flat = scores.reshape(-1)
        if not flat.size:
            raise ValueError("No candidate scores provided.")
        if self.minimize:
            idx = int(np.argmin(flat))
        else:
            idx = int(np.argmax(flat))
        return idx, float(flat[idx])


class AICCriterion(SelectionCriterion):
    """Akaike information criterion."""

    def __init__(self, *, n_samples: int, **kwargs):
        super().__init__(minimize=True, **kwargs)
        self.n_samples = int(n_samples)

    def evaluate(self, rss, k: int):
        rss_arr = np.asarray(rss, dtype=float)
        if np.any(rss_arr <= 0):
            raise ValueError("RSS must be positive to compute AIC.")
        return self.n_samples * np.log(rss_arr / self.n_samples) + 2 * k

    def best_candidate(self, rss, k: int):
        rss_arr = np.asarray(rss, dtype=float).reshape(-1)
        if not rss_arr.size:
            raise ValueError("No candidate RSS values provided.")
        idx = int(np.argmin(rss_arr))
        best_rss = float(rss_arr[idx])
        best_value = float(self.evaluate(best_rss, k))
        return idx, best_value


class BestRSSCriterion(SelectionCriterion):
    """Criterion that greedily minimizes RSS without explicit complexity penalty."""

    def __init__(self, **kwargs):
        super().__init__(minimize=True, **kwargs)

    def evaluate(self, rss, k: int):
        rss_arr = np.asarray(rss, dtype=float)
        if __debug__ and np.any(rss_arr < 0):
            raise ValueError("RSS must be non-negative.")
        return rss_arr

    def best_candidate(self, rss, k: int):
        rss_arr = np.asarray(rss, dtype=float).reshape(-1)
        if not rss_arr.size:
            raise ValueError("No candidate RSS values provided.")
        idx = int(np.argmin(rss_arr))
        return idx, float(rss_arr[idx])
