"""Target builders for the microstructure benchmark pipeline."""

from __future__ import annotations

import numpy as np

from .types import MicrostructureObservables



def build_microstructure_target(
    observables: MicrostructureObservables,
    *,
    horizon_events: int,
) -> np.ndarray:
    if horizon_events <= 0:
        raise ValueError("horizon_events must be > 0.")
    if observables.mid_log.shape[0] <= horizon_events:
        raise ValueError("Not enough events for requested horizon_events.")
    return np.asarray(
        observables.mid_log[horizon_events:] - observables.mid_log[:-horizon_events],
        dtype=float,
    )


__all__ = ["build_microstructure_target"]
