import numpy as np

from selection.criteria import BestRSSCriterion
from selection.definitions import GramData
from selection.routines import BeamBackwardSelection, BeamForwardSelection, BeamMixedSelection
from selection.state_single import SelectionState
from tests.integration._selection_routines_helpers import (
    expected_indices,
    make_diagonal_problem,
)


def test_forward_beam_search_selects_best_subset():
    gram, cov, y_norm, n_samples = make_diagonal_problem(6)
    selector = BeamForwardSelection(beam_width=3)
    state = selector.fit(data=GramData(gram, cov, y_norm, n_samples), max_steps=3)
    assert set(state.active_set) == set(expected_indices(6, 3))


def test_backward_beam_search_selects_smallest_subset():
    gram, cov, y_norm, n_samples = make_diagonal_problem(6)
    y_norm += 1.0
    selector = BeamBackwardSelection(beam_width=2, allow_worse=True)
    state = selector.fit(data=GramData(gram, cov, y_norm, n_samples), max_steps=3)
    assert len(state.active_set) == 3
    assert set(state.active_set) == set(expected_indices(6, 3))


def test_mixed_beam_search_handles_forward_and_backward():
    gram, cov, y_norm, n_samples = make_diagonal_problem(5)
    selector = BeamMixedSelection(beam_width=2)
    state = selector.fit(
        data=GramData(gram, cov, y_norm, n_samples),
        max_forward_steps=3,
        max_total_steps=9,
    )
    assert set(state.active_set) == set(expected_indices(5, 3))


def test_beam_search_deduplicates_active_sets():
    gram = np.eye(4)
    cov = np.array([1.0, 1.0, 0.5, 0.25])
    y_norm = float(cov @ cov)
    selector = BeamForwardSelection(beam_width=4)
    state = selector.fit(data=GramData(gram, cov, y_norm, 20), max_steps=1)
    assert len(state.active_set) == 1
    assert len(set(state.active_set)) == 1


def test_beam_pruning_is_deterministic_and_non_worsening():
    gram = np.eye(4)
    cov = np.ones(4)
    y_norm = float(cov @ cov)
    n_samples = 50

    greedy = BeamForwardSelection(beam_width=1, criterion=BestRSSCriterion)
    wider = BeamForwardSelection(beam_width=3, criterion=BestRSSCriterion)

    greedy_state = greedy.fit(data=GramData(gram, cov, y_norm, n_samples), max_steps=1)
    wider_state = wider.fit(data=GramData(gram, cov, y_norm, n_samples), max_steps=1)

    assert greedy_state.active_set == [0]
    assert wider_state.active_set == [0]
    assert wider_state.rss <= greedy_state.rss + 1e-12


def test_rank_deficient_backward_recovers_full_rank_subset():
    gram = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0 + 1e-6],
        ]
    )
    cov = np.array([1.0, 2.0, 3.0])
    y_norm = float(cov @ cov)
    data = GramData(gram, cov, y_norm, n_samples=5)

    direct_state = SelectionState(data)
    direct_state.init_full()
    direct_state.apply_backward_step(2)
    assert len(direct_state.active_set) == 2
    inv_direct = np.linalg.inv(
        gram[np.ix_(direct_state.active_set, direct_state.active_set)]
    )
    assert np.isfinite(inv_direct).all()

    beam = BeamBackwardSelection(
        beam_width=2, criterion=BestRSSCriterion, allow_worse=True
    )
    beam_state = beam.fit(data=data, max_steps=1)
    assert len(beam_state.active_set) == 2
    inv_beam = np.linalg.inv(gram[np.ix_(beam_state.active_set, beam_state.active_set)])
    assert np.isfinite(inv_beam).all()
