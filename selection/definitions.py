from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class GramData:
    gram: np.ndarray
    cov: np.ndarray
    y_norm: float
    n_samples: int

    def __post_init__(self):
        gram = np.asarray(self.gram, dtype=float)
        cov = np.asarray(self.cov, dtype=float)
        object.__setattr__(self, "y_norm", float(self.y_norm))
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
            raise ValueError("Gram matrix must be square.")
        if cov.ndim != 1 or cov.shape[0] != gram.shape[0]:
            raise ValueError("cov vector must match Gram dimensions.")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")


@dataclass
class CrossValGramData:
    """
    Container for full-data and per-fold Gram statistics for CV / LOO selection.

    Assumes that, for each fold k, we have a GramData object computed on that
    fold's samples only. The "all-data" Gram triple is then the sum over folds.
    """

    folds: List[GramData]

    gram_total: np.ndarray = field(init=False)
    cov_total: np.ndarray = field(init=False)
    y_norm_total: float = field(init=False)

    n_folds: int = field(init=False)
    p: int = field(init=False)
    fold_sizes: np.ndarray = field(init=False)
    n_samples_total: Optional[int] = field(init=False, default=None)

    gram_folds: List[np.ndarray] = field(init=False)
    cov_folds: List[np.ndarray] = field(init=False)
    y_norm_folds: List[float] = field(init=False)

    def __post_init__(self):
        if not self.folds:
            raise ValueError("CrossValidationGramData requires at least one fold.")

        self.n_folds = len(self.folds)
        first = self.folds[0]
        self.p = first.gram.shape[0]

        gram_sum = np.zeros_like(first.gram)
        cov_sum = np.zeros_like(first.cov)
        y_norm_sum = 0.0
        fold_sizes = np.empty(self.n_folds, dtype=int)

        for k, fd in enumerate(self.folds):
            if fd.gram.shape != (self.p, self.p):
                raise ValueError(
                    f"Fold {k} has incompatible gram shape {fd.gram.shape}; "
                    f"expected ({self.p}, {self.p})."
                )
            if fd.cov.shape != (self.p,):
                raise ValueError(
                    f"Fold {k} has incompatible cov shape {fd.cov.shape}; "
                    f"expected ({self.p},)."
                )

            gram_sum += fd.gram
            cov_sum += fd.cov
            y_norm_sum += float(fd.y_norm)
            fold_sizes[k] = getattr(fd, "n_samples", 0)

        self.gram_total = gram_sum
        self.cov_total = cov_sum
        self.y_norm_total = y_norm_sum
        self.fold_sizes = fold_sizes
        if np.any(fold_sizes):
            self.n_samples_total = int(np.sum(fold_sizes))
        else:
            self.n_samples_total = None

        self.gram_folds = [fd.gram for fd in self.folds]
        self.cov_folds = [fd.cov for fd in self.folds]
        self.y_norm_folds = [float(fd.y_norm) for fd in self.folds]

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
        y_norm_train = self.y_norm_total - float(fd.y_norm)

        if self.n_samples_total is None:
            raise ValueError(
                "n_samples information is required to build training GramData."
            )
        n_samples_train = self.n_samples_total - int(self.fold_sizes[k])
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
        n_samples_total = None
        if np.any(self.fold_sizes):
            n_samples_total = int(np.sum(self.fold_sizes))

        if n_samples_total is None:
            raise ValueError(
                "Fold GramData objects must include n_samples to recover full data."
            )
        return GramData(
            gram=self.gram_total,
            cov=self.cov_total,
            y_norm=self.y_norm_total,
            n_samples=n_samples_total,
        )
