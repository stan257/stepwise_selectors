import math
from numbers import Integral, Real
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .constants import ABS_TOL


@runtime_checkable
class CriterionProtocol(Protocol):
    """Structural contract expected by selector routines."""

    minimize: bool
    current_value: float

    def evaluate(self, rss: Any, k: int): ...

    def is_improvement(self, candidate: float, incumbent: float | None = None) -> bool: ...

    def update_current(self, value: float) -> None: ...

    def clone(self) -> "CriterionProtocol": ...

    def best_candidate(self, rss: Any, k: int) -> tuple[int, float]: ...


class SelectionCriterion:
    """Pure criterion evaluator with optional score tracking."""

    cv_compatible: bool = True

    def __init__(
        self,
        *,
        minimize: bool = True,
        abs_tol: float = ABS_TOL,
        rel_tol: float = 1e-8,
    ):
        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a bool.")
        self.minimize = minimize
        self.abs_tol = _validate_non_negative_finite_float(abs_tol, name="abs_tol")
        self.rel_tol = _validate_non_negative_finite_float(rel_tol, name="rel_tol")
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

    cv_compatible = False

    def __init__(self, *, n_samples: int, **kwargs):
        super().__init__(minimize=True, **kwargs)
        self.n_samples = _validate_integral_param(n_samples, name="n_samples")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive for AIC.")

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


class BICCriterion(SelectionCriterion):
    """Bayesian information criterion."""

    cv_compatible = False

    def __init__(self, *, n_samples: int, **kwargs):
        super().__init__(minimize=True, **kwargs)
        self.n_samples = _validate_integral_param(n_samples, name="n_samples")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive for BIC.")

    def evaluate(self, rss, k: int):
        rss_arr = np.asarray(rss, dtype=float)
        if np.any(rss_arr <= 0):
            raise ValueError("RSS must be positive to compute BIC.")
        n = self.n_samples
        return n * np.log(rss_arr / n) + k * np.log(n)


class AICcCriterion(SelectionCriterion):
    """Small-sample corrected Akaike information criterion."""

    cv_compatible = False

    def __init__(self, *, n_samples: int, **kwargs):
        super().__init__(minimize=True, **kwargs)
        self.n_samples = _validate_integral_param(n_samples, name="n_samples")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive for AICc.")

    def evaluate(self, rss, k: int):
        rss_arr = np.asarray(rss, dtype=float)
        if np.any(rss_arr <= 0):
            raise ValueError("RSS must be positive to compute AICc.")
        n = self.n_samples
        aic = n * np.log(rss_arr / n) + 2 * k
        denom = n - k - 1
        if denom <= 0:
            return np.full_like(rss_arr, np.inf, dtype=float)
        correction = (2 * k * (k + 1)) / denom
        return aic + correction


class HQICCriterion(SelectionCriterion):
    """Hannan-Quinn information criterion."""

    cv_compatible = False

    def __init__(self, *, n_samples: int, **kwargs):
        super().__init__(minimize=True, **kwargs)
        self.n_samples = _validate_integral_param(n_samples, name="n_samples")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive for HQIC.")
        if self.n_samples < 3:
            raise ValueError(
                "HQIC requires n_samples >= 3 (log(log(n)) must be positive)."
            )

    def evaluate(self, rss, k: int):
        rss_arr = np.asarray(rss, dtype=float)
        if np.any(rss_arr <= 0):
            raise ValueError("RSS must be positive to compute HQIC.")
        n = self.n_samples
        penalty = 2.0 * k * np.log(np.log(n))
        return n * np.log(rss_arr / n) + penalty


class EBICCriterion(SelectionCriterion):
    """Extended BIC with a combinatorial penalty term."""

    cv_compatible = False

    def __init__(self, *, n_samples: int, p: int, gamma: float = 0.5, **kwargs):
        super().__init__(minimize=True, **kwargs)
        self.n_samples = _validate_integral_param(n_samples, name="n_samples")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive for EBIC.")
        self.p = _validate_integral_param(p, name="p")
        self.gamma = _validate_finite_float(gamma, name="gamma")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1 for EBIC.")
        if self.p <= 0:
            raise ValueError("p must be positive for EBIC.")

    def evaluate(self, rss, k: int):
        rss_arr = np.asarray(rss, dtype=float)
        if np.any(rss_arr <= 0):
            raise ValueError("RSS must be positive to compute EBIC.")
        if k < 0 or k > self.p:
            raise ValueError("k must be between 0 and p for EBIC.")
        n = self.n_samples
        base = n * np.log(rss_arr / n) + k * np.log(n)
        if k == 0:
            return base
        log_choose = (
            math.lgamma(self.p + 1)
            - math.lgamma(k + 1)
            - math.lgamma(self.p - k + 1)
        )
        return base + 2.0 * self.gamma * log_choose


class GCVCriterion(SelectionCriterion):
    """Generalized cross-validation score."""

    cv_compatible = False

    def __init__(self, *, n_samples: int, **kwargs):
        super().__init__(minimize=True, **kwargs)
        self.n_samples = _validate_integral_param(n_samples, name="n_samples")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive for GCV.")

    def evaluate(self, rss, k: int):
        rss_arr = np.asarray(rss, dtype=float)
        if np.any(rss_arr < 0):
            raise ValueError("RSS must be non-negative to compute GCV.")
        denom = self.n_samples - k
        if denom <= 0:
            return np.full_like(rss_arr, np.inf, dtype=float)
        scale = (denom / self.n_samples) ** 2
        return (rss_arr / self.n_samples) / scale


class BestRSSCriterion(SelectionCriterion):
    """Criterion that greedily minimizes RSS without explicit complexity penalty."""

    def __init__(self, **kwargs):
        super().__init__(minimize=True, **kwargs)

    def evaluate(self, rss, k: int):
        rss_arr = np.asarray(rss, dtype=float)
        if np.any(rss_arr < 0):
            raise ValueError("RSS must be non-negative.")
        return rss_arr

    def best_candidate(self, rss, k: int):
        rss_arr = np.asarray(rss, dtype=float).reshape(-1)
        if not rss_arr.size:
            raise ValueError("No candidate RSS values provided.")
        idx = int(np.argmin(rss_arr))
        return idx, float(rss_arr[idx])


def _validate_integral_param(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


def _validate_finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite.")
    return value_float


def _validate_non_negative_finite_float(value: float, *, name: str) -> float:
    value_float = _validate_finite_float(value, name=name)
    if value_float < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return value_float
