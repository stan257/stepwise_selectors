"""Shared NPZ serialization helpers for chunked GramData benchmark datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

from selection import GramData

_DATASET_VERSION = 1

T = TypeVar("T")



def save_chunk_dataset(dataset: object, path: str | Path) -> None:
    """Persist a chunk dataset with the standard benchmark NPZ schema."""
    out_path = Path(path)
    gram_chunks = list(getattr(dataset, "gram_chunks"))
    n_chunks = len(gram_chunks)
    if n_chunks == 0:
        raise ValueError("dataset.gram_chunks must contain at least one chunk.")

    feature_names = tuple(str(name) for name in getattr(dataset, "feature_names"))
    if not feature_names:
        raise ValueError("dataset.feature_names must be non-empty.")

    p = len(feature_names)
    gram_stack = np.stack(
        [np.asarray(chunk.gram, dtype=float) for chunk in gram_chunks], axis=0
    )
    cov_stack = np.stack(
        [np.asarray(chunk.cov, dtype=float) for chunk in gram_chunks], axis=0
    )
    y_norm = np.asarray([float(chunk.y_norm) for chunk in gram_chunks], dtype=float)
    n_samples = np.asarray(
        [int(chunk.n_samples) for chunk in gram_chunks], dtype=np.int64
    )

    if gram_stack.shape != (n_chunks, p, p):
        raise ValueError("Each chunk gram matrix must have shape (p, p).")
    if cov_stack.shape != (n_chunks, p):
        raise ValueError("Each chunk cov vector must have shape (p,).")

    chunk_ranges_raw = tuple(getattr(dataset, "chunk_ranges"))
    if len(chunk_ranges_raw) != n_chunks:
        raise ValueError("chunk_ranges length must match number of chunks.")
    chunk_ranges = np.asarray(chunk_ranges_raw, dtype=np.int64)
    if chunk_ranges.shape != (n_chunks, 2):
        raise ValueError("chunk_ranges must have shape (n_chunks, 2).")

    regime_by_chunk_raw = tuple(getattr(dataset, "regime_by_chunk"))
    if len(regime_by_chunk_raw) != n_chunks:
        raise ValueError("regime_by_chunk length must match number of chunks.")
    regime_by_chunk = np.asarray(regime_by_chunk_raw, dtype=np.int64)

    support_by_chunk = getattr(dataset, "support_by_chunk")
    if support_by_chunk is None:
        support_lengths = np.asarray([], dtype=np.int64)
        support_matrix = np.asarray([], dtype=np.int64).reshape(0, 0)
    else:
        support_by_chunk = tuple(support_by_chunk)
        if len(support_by_chunk) != n_chunks:
            raise ValueError("support_by_chunk length must match number of chunks.")
        support_lengths = np.asarray(
            [len(support) for support in support_by_chunk], dtype=np.int64
        )
        width = int(np.max(support_lengths)) if support_lengths.size > 0 else 0
        support_matrix = np.full((n_chunks, width), -1, dtype=np.int64)
        for chunk_idx, support in enumerate(support_by_chunk):
            vals = np.asarray(support, dtype=np.int64)
            if vals.ndim != 1:
                raise ValueError("Each support entry must be a 1-D sequence of indices.")
            if np.any(vals < 0) or np.any(vals >= p):
                raise ValueError("Support indices must be in [0, p).")
            k = vals.shape[0]
            if k > 0:
                support_matrix[chunk_idx, :k] = vals

    beta_by_chunk = getattr(dataset, "beta_by_chunk")
    if beta_by_chunk is None:
        beta_stack = np.asarray([], dtype=float).reshape(0, p)
    else:
        beta_by_chunk = tuple(beta_by_chunk)
        if len(beta_by_chunk) != n_chunks:
            raise ValueError("beta_by_chunk length must match number of chunks.")
        beta_stack = np.stack(
            [np.asarray(beta, dtype=float) for beta in beta_by_chunk], axis=0
        )
        if beta_stack.shape != (n_chunks, p):
            raise ValueError("Each beta vector must have shape (p,).")

    feature_names_arr = np.asarray(feature_names, dtype=str)
    meta_json = json.dumps(getattr(dataset, "meta"), sort_keys=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        version=np.asarray([_DATASET_VERSION], dtype=np.int64),
        gram=gram_stack,
        cov=cov_stack,
        y_norm=y_norm,
        n_samples=n_samples,
        feature_names=feature_names_arr,
        chunk_ranges=chunk_ranges,
        regime_by_chunk=regime_by_chunk,
        support_lengths=support_lengths,
        support_matrix=support_matrix,
        beta=beta_stack,
        meta_json=np.asarray(meta_json, dtype=str),
    )



def load_chunk_dataset(
    path: str | Path,
    *,
    dataset_factory: Callable[..., T],
    invalid_version_message: str,
) -> T:
    """Load a standard chunk dataset and construct it via ``dataset_factory``."""
    in_path = Path(path)
    with np.load(in_path, allow_pickle=False) as npz:
        version = np.asarray(npz["version"], dtype=np.int64)
        if version.shape != (1,) or int(version[0]) != _DATASET_VERSION:
            raise ValueError(invalid_version_message)

        gram_stack = np.asarray(npz["gram"], dtype=float)
        cov_stack = np.asarray(npz["cov"], dtype=float)
        y_norm = np.asarray(npz["y_norm"], dtype=float)
        n_samples = np.asarray(npz["n_samples"], dtype=np.int64)
        feature_names_arr = np.asarray(npz["feature_names"], dtype=str)
        chunk_ranges = np.asarray(npz["chunk_ranges"], dtype=np.int64)
        regime_by_chunk = np.asarray(npz["regime_by_chunk"], dtype=np.int64)
        support_lengths = np.asarray(npz["support_lengths"], dtype=np.int64)
        support_matrix = np.asarray(npz["support_matrix"], dtype=np.int64)
        beta_stack = np.asarray(npz["beta"], dtype=float)
        meta_json = str(np.asarray(npz["meta_json"], dtype=str).item())

    if gram_stack.ndim != 3:
        raise ValueError("Stored gram array must have shape (n_chunks, p, p).")
    n_chunks, p, p2 = gram_stack.shape
    if p != p2:
        raise ValueError("Stored gram matrices must be square.")
    if cov_stack.shape != (n_chunks, p):
        raise ValueError("Stored cov array must have shape (n_chunks, p).")
    if y_norm.shape != (n_chunks,):
        raise ValueError("Stored y_norm must have shape (n_chunks,).")
    if n_samples.shape != (n_chunks,):
        raise ValueError("Stored n_samples must have shape (n_chunks,).")
    if feature_names_arr.shape != (p,):
        raise ValueError("Stored feature_names must have shape (p,).")
    if chunk_ranges.shape != (n_chunks, 2):
        raise ValueError("Stored chunk_ranges must have shape (n_chunks, 2).")
    if regime_by_chunk.shape != (n_chunks,):
        raise ValueError("Stored regime_by_chunk must have shape (n_chunks,).")

    gram_chunks = [
        GramData(
            gram=np.ascontiguousarray(gram_stack[idx]),
            cov=np.ascontiguousarray(cov_stack[idx]),
            y_norm=float(y_norm[idx]),
            n_samples=int(n_samples[idx]),
            warn_if_uncentered=False,
        )
        for idx in range(n_chunks)
    ]

    if support_lengths.size == 0:
        support_by_chunk: tuple[tuple[int, ...], ...] | None = None
    else:
        if support_lengths.shape != (n_chunks,):
            raise ValueError("Stored support_lengths must have shape (n_chunks,).")
        if support_matrix.shape[0] != n_chunks:
            raise ValueError("Stored support_matrix has incompatible first dimension.")
        support_list: list[tuple[int, ...]] = []
        for chunk_idx, length in enumerate(support_lengths.tolist()):
            k = int(length)
            if k < 0 or k > support_matrix.shape[1]:
                raise ValueError("Stored support length is out of range.")
            if k == 0:
                support_list.append(tuple())
                continue
            support_vals = support_matrix[chunk_idx, :k]
            if np.any(support_vals < 0) or np.any(support_vals >= p):
                raise ValueError("Stored support contains out-of-range indices.")
            support_list.append(tuple(int(v) for v in support_vals.tolist()))
        support_by_chunk = tuple(support_list)

    if beta_stack.shape == (0, p):
        beta_by_chunk: tuple[np.ndarray, ...] | None = None
    else:
        if beta_stack.shape != (n_chunks, p):
            raise ValueError("Stored beta array must have shape (n_chunks, p).")
        beta_by_chunk = tuple(
            np.ascontiguousarray(beta_stack[idx].copy()) for idx in range(n_chunks)
        )

    meta = json.loads(meta_json)
    return dataset_factory(
        gram_chunks=gram_chunks,
        feature_names=tuple(str(v) for v in feature_names_arr.tolist()),
        chunk_ranges=tuple((int(s), int(e)) for s, e in chunk_ranges.tolist()),
        regime_by_chunk=tuple(int(v) for v in regime_by_chunk.tolist()),
        support_by_chunk=support_by_chunk,
        beta_by_chunk=beta_by_chunk,
        meta=meta,
    )


__all__ = ["load_chunk_dataset", "save_chunk_dataset"]
