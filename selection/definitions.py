from dataclasses import dataclass, field

import numpy as np
from numbers import Real, Integral


@dataclass(frozen=True)
class GramData:
    gram: np.ndarray
    cov: np.ndarray
    y_norm: float
    n_samples: int
    dtype: np.dtype | None = None

    def __post_init__(self):
        if not isinstance(self.gram, np.ndarray):
            raise TypeError("Gram matrix must be a numpy array.")
        if not isinstance(self.cov, np.ndarray):
            raise TypeError("cov vector must be a numpy array.")
        # Ensure contiguous storage (and optional casting) for BLAS-friendly access.
        dtype = None if self.dtype is None else np.dtype(self.dtype)
        object.__setattr__(self, "gram", np.ascontiguousarray(self.gram, dtype=dtype))
        object.__setattr__(self, "cov", np.ascontiguousarray(self.cov, dtype=dtype))
        if not isinstance(self.y_norm, Real):
            raise TypeError("y_norm must be a real number.")
        if not isinstance(self.n_samples, Integral):
            raise TypeError("n_samples must be an integer.")
        if not np.all(np.isfinite(self.gram)):
            raise ValueError("Gram matrix must contain only finite values.")
        if not np.all(np.isfinite(self.cov)):
            raise ValueError("cov vector must contain only finite values.")
        if not np.isfinite(self.y_norm):
            raise ValueError("y_norm must be finite.")
        if self.gram.ndim != 2 or self.gram.shape[0] != self.gram.shape[1]:
            raise ValueError("Gram matrix must be square.")
        if not np.allclose(self.gram, self.gram.T, atol=1e-10, rtol=0.0):
            raise ValueError("Gram matrix must be symmetric.")
        if np.any(np.diag(self.gram) < 0):
            raise ValueError("Gram diagonal entries must be non-negative.")
        if self.cov.ndim != 1 or self.cov.shape[0] != self.gram.shape[0]:
            raise ValueError("cov vector must match Gram dimensions.")
        if self.y_norm < 0:
            raise ValueError("y_norm must be non-negative.")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")


@dataclass
class CrossValGramData:
    """
    Container for full-data and per-fold Gram statistics for CV / LOO selection.

    Assumes that, for each fold k, we have a GramData object computed on that
    fold's samples only. The "all-data" Gram triple is then the sum over folds.
    """

    folds: list[GramData]

    gram_total: np.ndarray = field(init=False)
    cov_total: np.ndarray = field(init=False)
    y_norm_total: float = field(init=False)

    n_folds: int = field(init=False)
    p: int = field(init=False)
    fold_sizes: np.ndarray = field(init=False)
    n_samples_total: int | None = field(init=False, default=None)

    gram_folds: list[np.ndarray] = field(init=False)
    cov_folds: list[np.ndarray] = field(init=False)
    y_norm_folds: list[float] = field(init=False)
    cov_folds_arr: np.ndarray = field(init=False)
    y_norm_folds_arr: np.ndarray = field(init=False)

    def check_data_validity(self) -> int:
        if len(self.folds) < 2:
            raise ValueError("CrossValidationGramData requires at least two folds.")
        first = self.folds[0]
        if not isinstance(first, GramData):
            raise TypeError("All folds must be GramData instances.")
        p = first.gram.shape[0]
        for k, fd in enumerate(self.folds):
            if not isinstance(fd, GramData):
                raise TypeError(f"Fold {k} must be a GramData instance.")
            if fd.gram.shape != (p, p):
                raise ValueError(
                    f"Fold {k} has incompatible gram shape {fd.gram.shape}; "
                    f"expected ({p}, {p})."
                )
            if fd.cov.shape != (p,):
                raise ValueError(
                    f"Fold {k} has incompatible cov shape {fd.cov.shape}; "
                    f"expected ({p},)."
                )
        return p

    def __post_init__(self):
        if len(self.folds) < 2:
            raise ValueError("CrossValidationGramData requires at least two folds.")

        self.n_folds = len(self.folds)
        self.p = self.check_data_validity()

        gram_sum = np.zeros_like(self.folds[0].gram)
        cov_sum = np.zeros_like(self.folds[0].cov)
        y_norm_sum = 0.0
        fold_sizes = np.empty(self.n_folds, dtype=int)

        for k, fd in enumerate(self.folds):
            gram_sum += fd.gram
            cov_sum += fd.cov
            y_norm_sum += fd.y_norm
            fold_sizes[k] = fd.n_samples

        self.gram_total = gram_sum
        self.cov_total = cov_sum
        self.y_norm_total = y_norm_sum
        self.fold_sizes = fold_sizes
        self.n_samples_total = np.sum(fold_sizes)

        self.gram_folds = [fd.gram for fd in self.folds]
        self.cov_folds = [fd.cov for fd in self.folds]
        self.y_norm_folds = [fd.y_norm for fd in self.folds]
        # Stack per-fold covariances for faster vectorized CV computations.
        self.cov_folds_arr = np.stack(self.cov_folds, axis=0)
        self.y_norm_folds_arr = np.array(self.y_norm_folds, dtype=float)

    def val_data_for_fold(self, k: int) -> GramData:
        if not (0 <= k < self.n_folds):
            raise IndexError("Fold index out of range.")
        return self.folds[k]

    def train_data_for_fold(self, k: int) -> GramData:
        if not (0 <= k < self.n_folds):
            raise IndexError("Fold index out of range.")

        fd = self.folds[k]

        gram_train = self.gram_total - fd.gram
        cov_train = self.cov_total - fd.cov
        y_norm_train = self.y_norm_total - fd.y_norm

        if self.n_samples_total is None:
            raise ValueError(
                "n_samples information is required to build training GramData."
            )
        n_samples_train = self.n_samples_total - self.fold_sizes[k]
        return GramData(
            gram=gram_train,
            cov=cov_train,
            y_norm=y_norm_train,
            n_samples=n_samples_train,
        )

    def make_full_data(self) -> GramData:
        """
        Convenience method to get a GramData view of the full dataset,
        using the aggregated sums.
        """
        if self.n_samples_total is None:
            raise ValueError(
                "Fold GramData objects must include n_samples to recover full data."
            )
        return GramData(
            gram=self.gram_total,
            cov=self.cov_total,
            y_norm=self.y_norm_total,
            n_samples=int(self.n_samples_total),
        )
