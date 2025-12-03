"""Selection routines orchestrating forward, backward, and mixed strategies."""

import inspect
import numpy as np

from .constants import ABS_TOL
from .criteria import AICCriterion, BestRSSCriterion, SelectionCriterion
from .beam_search import Beam, beam_backward_children, beam_forward_children, beam_best_backward_child
from .beam_utils import (
    cv_beam_backward_children,
    cv_beam_best_backward_child,
    cv_beam_forward_children,
    run_beam_mixed,
    run_beam_search,
)
from .cv_utils import cv_backward_scores, cv_forward_scores
from .definitions import GramData, CrossValGramData
from .state import SelectionState, CrossValSelectionState


class BaseSingleSelectionRoutine:
    """Stores configuration shared by greedy selection routines."""

    def __init__(
        self,
        *,
        tol=ABS_TOL,
        criterion_cls=None,
        criterion_kwargs=None,
    ):
        self.tol = tol
        self.criterion_cls = criterion_cls or AICCriterion
        self.criterion_kwargs = dict(criterion_kwargs or {})

    def _init_run(
        self,
        state: SelectionState | None,
        data: GramData,
        *,
        mode: str,
    ) -> tuple[SelectionState, SelectionCriterion]:
        working_state = state.clone() if state is not None else SelectionState(data)
        params = dict(self.criterion_kwargs)
        init_params = inspect.signature(self.criterion_cls.__init__).parameters
        if "n_samples" in init_params and "n_samples" not in params:
            params["n_samples"] = data.n_samples
        criterion = self.criterion_cls(**params)
        if mode == "empty":
            working_state.init_empty()
        elif mode == "full":
            working_state.init_full()
        else:
            raise ValueError(f"Unsupported session mode '{mode}'.")
        initial_score = float(
            np.asarray(
                criterion.evaluate(working_state.rss, len(working_state.active_set))
            )
        )
        criterion.update_current(initial_score)
        return working_state, criterion

    @staticmethod
    def _select_forward_candidate(state: SelectionState, criterion) -> tuple | None:
        cache = state.compute_forward_deltas()
        if cache is None or not cache.candidates.size:
            return None
        scores = cache.rss_new
        best_idx, best_value = criterion.best_candidate(scores, cache.active_rk + 1)
        if not criterion.is_improvement(best_value):
            return None
        return cache, best_idx, best_value

    def _forward_step(self, state: SelectionState, criterion) -> bool:
        best = self._select_forward_candidate(state, criterion)
        if best is None:
            return False
        cache, idx, score = best
        state.apply_forward_step(cache, idx)
        criterion.update_current(score)
        return True

    @staticmethod
    def _select_backward_candidate(
        state: SelectionState, criterion, allow_worse: bool = False
    ) -> tuple | None:
        if not state.active_set:
            return None
        rss_values = state.compute_backward_scores()
        if rss_values is None or not len(rss_values):
            return None
        scores = rss_values
        best_idx, best_value = criterion.best_candidate(
            scores, len(state.active_set) - 1
        )
        if not allow_worse and not criterion.is_improvement(best_value):
            return None
        return best_idx, best_value

    def _backward_step(
        self, state: SelectionState, criterion, *, allow_worse: bool = False
    ) -> bool:
        best = self._select_backward_candidate(
            state, criterion, allow_worse=allow_worse
        )
        if best is None:
            return False
        idx, score = best
        try:
            state.apply_backward_step(idx)
        except np.linalg.LinAlgError:
            return False
        criterion.update_current(score)
        return True


class ForwardSelection(BaseSingleSelectionRoutine):
    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: GramData,
        max_steps=None,
    ) -> SelectionState:
        run_state, criterion = self._init_run(state, data, mode="empty")
        steps = 0
        while max_steps is None or steps < max_steps:
            if not self._forward_step(run_state, criterion):
                break
            steps += 1
        return run_state


class BackwardSelection(BaseSingleSelectionRoutine):
    """Greedy backward selection.

    Parameters
    ----------
    allow_worse : bool, default=False
        If True, allows the selection to make a non-improving move at each step.
        This can help escape local optima, especially when `max_steps` is limited.
    """

    def __init__(self, *, allow_worse: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.allow_worse = allow_worse

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: GramData,
        max_steps=None,
    ) -> SelectionState:
        run_state, criterion = self._init_run(state, data, mode="full")
        steps = 0
        while max_steps is None or steps < max_steps:
            if not self._backward_step(run_state, criterion, allow_worse=self.allow_worse):
                break
            steps += 1
        return run_state


class MixedSelection(BaseSingleSelectionRoutine):
    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: GramData,
        max_forward_steps=None,
        max_total_steps=None,
    ) -> SelectionState:
        run_state, criterion = self._init_run(state, data, mode="empty")
        forward_steps = 0
        total_operations = 0
        while True:
            if max_total_steps is not None and total_operations >= max_total_steps:
                break
            if not self._forward_step(run_state, criterion):
                break
            forward_steps += 1
            total_operations += 1
            if max_forward_steps is not None and forward_steps >= max_forward_steps:
                break
            while True:
                if max_total_steps is not None and total_operations >= max_total_steps:
                    break
                if not self._backward_step(run_state, criterion):
                    break
                total_operations += 1
        return run_state


class BaseBeamSelectionRoutine(BaseSingleSelectionRoutine):
    """Shared machinery for beam-search variants."""

    def __init__(
        self,
        *,
        beam_width: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))


class BeamForwardSelection(BaseBeamSelectionRoutine):
    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: GramData,
        max_steps=None,
    ) -> SelectionState:
        run_state, criterion = self._init_run(state, data, mode="empty")
        initial = Beam(run_state, criterion, criterion.current_value)
        return run_beam_search(
            initial,
            beam_width=self.beam_width,
            max_steps=max_steps,
            expand_fn=lambda beam: beam_forward_children(
                beam, self.beam_width, self.tol
            ),
        )


class BeamBackwardSelection(BaseBeamSelectionRoutine):
    """Backward selection with beam search.

    Parameters
    ----------
    allow_worse : bool, default=False
        If True, allows the search to explore non-improving moves. This can
        help escape local optima, especially when `max_steps` is limited.
    """

    def __init__(self, *, allow_worse: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.allow_worse = allow_worse

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: GramData,
        max_steps=None,
    ) -> SelectionState:
        run_state, criterion = self._init_run(state, data, mode="full")
        initial = Beam(run_state, criterion, criterion.current_value)
        return run_beam_search(
            initial,
            beam_width=self.beam_width,
            max_steps=max_steps,
            expand_fn=lambda beam: beam_backward_children(
                beam, self.beam_width, self.tol, allow_worse=self.allow_worse
            ),
        )


class BeamMixedSelection(BaseBeamSelectionRoutine):
    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: GramData,
        max_forward_steps=None,
        max_total_steps=None,
    ) -> SelectionState:
        run_state, criterion = self._init_run(state, data, mode="empty")
        initial = Beam(run_state, criterion, criterion.current_value)
        return run_beam_mixed(
            initial,
            beam_width=self.beam_width,
            max_forward_steps=max_forward_steps,
            max_total_steps=max_total_steps,
            forward_expand=lambda beam: beam_forward_children(
                beam, self.beam_width, self.tol
            ),
            backward_improve=lambda beam: beam_best_backward_child(beam, self.tol),
        )


class BaseCrossValSelection:
    def __init__(
        self,
        *,
        tol: float = ABS_TOL,
        criterion_cls=None,
        criterion_kwargs=None,
    ):
        self.tol = tol
        self.criterion_cls = criterion_cls or BestRSSCriterion
        self.criterion_kwargs = dict(criterion_kwargs or {})

    def _init_run(
        self,
        state: CrossValSelectionState | None,
        data: CrossValGramData,
        *,
        mode: str,
    ) -> tuple[CrossValSelectionState, SelectionCriterion]:
        cv_state = state.clone() if state is not None else CrossValSelectionState(data)
        if mode == "empty":
            cv_state.init_empty()
        elif mode == "full":
            cv_state.init_full()
        else:
            raise ValueError(f"Unsupported session mode '{mode}'.")
        criterion = self._init_criterion(cv_state, data)
        return cv_state, criterion

    def _init_criterion(
        self, cv_state: CrossValSelectionState, data: CrossValGramData
    ) -> SelectionCriterion:
        params = dict(self.criterion_kwargs)
        init_params = inspect.signature(self.criterion_cls.__init__).parameters
        if "n_samples" in init_params and "n_samples" not in params:
            if data.n_samples_total is None:
                raise ValueError(
                    "CrossValGramData must include n_samples to instantiate the criterion."
                )
            params["n_samples"] = data.n_samples_total
        criterion = self.criterion_cls(**params)
        initial_score = float(
            np.asarray(criterion.evaluate(cv_state.rss_cv, len(cv_state.active_set)))
        )
        criterion.update_current(initial_score)
        return criterion

    def _forward_step(self, cv_state: CrossValSelectionState, criterion) -> bool:
        forward_data = cv_forward_scores(cv_state, self.tol)
        if forward_data is None:
            return False
        fold_caches = forward_data.fold_caches
        candidate_maps = forward_data.candidate_maps
        candidates = forward_data.candidates
        aggregated = forward_data.aggregated_rss
        best_idx, best_value = criterion.best_candidate(
            aggregated, len(cv_state.active_set) + 1
        )
        if not criterion.is_improvement(best_value):
            return False
        chosen = candidates[best_idx]
        for fold_idx, fold_state in enumerate(cv_state.train_states):
            cache = fold_caches[fold_idx]
            idx_local = candidate_maps[fold_idx][chosen]
            fold_state.apply_forward_step(cache, idx_local)
        cv_state._sync_active_set()
        cv_state.recompute_oos_rss()
        criterion.update_current(best_value)
        return True

    def _backward_step(self, cv_state: CrossValSelectionState, criterion) -> bool:
        backward_data = cv_backward_scores(cv_state, self.tol)
        if backward_data is None:
            return False
        aggregated_rss = backward_data.aggregated_rss
        best_local_idx, best_score = criterion.best_candidate(
            aggregated_rss, len(cv_state.active_set) - 1
        )
        try:
            cv_state.apply_backward_step(best_local_idx, self.tol)
            criterion.update_current(best_score)
            return True
        except np.linalg.LinAlgError:
            return False


class CrossValForwardSelection(BaseCrossValSelection):
    """Forward selection driven by cross-validated RSS via CrossValSelectionState."""

    def fit(
        self,
        state: CrossValSelectionState | None = None,
        *,
        data: CrossValGramData,
        max_steps=None,
    ) -> CrossValSelectionState:
        cv_state, criterion = self._init_run(state, data, mode="empty")
        steps = 0
        while max_steps is None or steps < max_steps:
            if not self._forward_step(cv_state, criterion):
                break
            steps += 1
        return cv_state


class CrossValBackwardSelection(BaseCrossValSelection):
    """Backward selection driven by cross-validated RSS via CrossValSelectionState."""

    def fit(
        self,
        state: CrossValSelectionState | None = None,
        *,
        data: CrossValGramData,
        max_steps=None,
    ) -> CrossValSelectionState:
        cv_state, criterion = self._init_run(state, data, mode="full")
        steps = 0
        while max_steps is None or steps < max_steps:
            if not self._backward_step(cv_state, criterion):
                break
            steps += 1
        return cv_state


class CrossValMixedSelection(BaseCrossValSelection):
    """Mixed forward/backward selection driven by cross-validated RSS."""

    def fit(
        self,
        state: CrossValSelectionState | None = None,
        *,
        data: CrossValGramData,
        max_forward_steps=None,
        max_total_steps=None,
    ) -> CrossValSelectionState:
        cv_state, criterion = self._init_run(state, data, mode="empty")
        forward_steps = 0
        total_ops = 0
        while True:
            if max_total_steps is not None and total_ops >= max_total_steps:
                break
            if not self._forward_step(cv_state, criterion):
                break
            total_ops += 1
            forward_steps += 1
            if max_forward_steps is not None and forward_steps >= max_forward_steps:
                break
            while True:
                if max_total_steps is not None and total_ops >= max_total_steps:
                    break
                if not self._backward_step(cv_state, criterion):
                    break
                total_ops += 1
        return cv_state


class BaseBeamCrossValSelection(BaseCrossValSelection):
    def __init__(
        self,
        *,
        beam_width: int = 1,
        tol: float = ABS_TOL,
        criterion_cls=None,
        criterion_kwargs=None,
    ):
        super().__init__(
            tol=tol, criterion_cls=criterion_cls, criterion_kwargs=criterion_kwargs
        )
        self.beam_width = max(1, int(beam_width))


class BeamCrossValForwardSelection(BaseBeamCrossValSelection):
    """Beam-search forward selection for cross-validated RSS."""

    def fit(
        self,
        state: CrossValSelectionState | None = None,
        *,
        data: CrossValGramData,
        max_steps=None,
    ) -> CrossValSelectionState:
        cv_state, criterion = self._init_run(state, data, mode="empty")
        initial = Beam(cv_state, criterion, criterion.current_value)
        return run_beam_search(
            initial,
            beam_width=self.beam_width,
            max_steps=max_steps,
            expand_fn=lambda beam: cv_beam_forward_children(
                beam, self.beam_width, self.tol
            ),
        )


class BeamCrossValBackwardSelection(BaseBeamCrossValSelection):
    """Beam-search backward selection for cross-validated RSS."""

    def fit(
        self,
        state: CrossValSelectionState | None = None,
        *,
        data: CrossValGramData,
        max_steps=None,
    ) -> CrossValSelectionState:
        cv_state, criterion = self._init_run(state, data, mode="full")
        allow_worse = max_steps is not None
        initial = Beam(cv_state, criterion, criterion.current_value)
        return run_beam_search(
            initial,
            beam_width=self.beam_width,
            max_steps=max_steps,
            expand_fn=lambda beam: cv_beam_backward_children(
                beam, self.beam_width, self.tol, allow_worse=allow_worse
            ),
        )


class BeamCrossValMixedSelection(BaseBeamCrossValSelection):
    """Beam-search mixed selection for cross-validated RSS."""

    def fit(
        self,
        state: CrossValSelectionState | None = None,
        *,
        data: CrossValGramData,
        max_forward_steps=None,
        max_total_steps=None,
    ) -> CrossValSelectionState:
        cv_state, criterion = self._init_run(state, data, mode="empty")
        initial = Beam(cv_state, criterion, criterion.current_value)
        return run_beam_mixed(
            initial,
            beam_width=self.beam_width,
            max_forward_steps=max_forward_steps,
            max_total_steps=max_total_steps,
            forward_expand=lambda beam: cv_beam_forward_children(
                beam, self.beam_width, self.tol
            ),
            backward_improve=lambda beam: cv_beam_best_backward_child(beam, self.tol),
        )
