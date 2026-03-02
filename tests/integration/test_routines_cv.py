import numpy as np

from selection.criteria import BestRSSCriterion
from selection.routines import (
    BackwardSelection,
    BeamCrossValBackwardSelection,
    BeamCrossValForwardSelection,
    CrossValBackwardSelection,
    CrossValForwardSelection,
    ForwardSelection,
)
from tests.integration._selection_routines_helpers import (
    esl_cv_data,
    make_cv_beam_trap_problem,
    make_cv_support_problem,
    make_heterogeneous_cv_problem,
)


def test_crossval_forward_selection_recovers_true_support():
    cv_data, support_set = make_cv_support_problem()
    selector = CrossValForwardSelection()
    state = selector.fit(data=cv_data, max_steps=len(support_set))

    assert set(state.active_set) == set(support_set)


def test_crossval_backward_selection_recovers_true_support():
    cv_data, support_set = make_cv_support_problem()
    selector = CrossValBackwardSelection()
    state = selector.fit(data=cv_data, max_steps=50 - len(support_set))

    recovered = set(state.active_set)
    assert set(support_set).issubset(recovered)
    assert len(recovered) < cv_data.gram_total.shape[0]


def test_crossval_forward_selection_matches_full_run(esl_cv_data):
    cv_data, support, full_data = esl_cv_data
    ForwardSelection(criterion=BestRSSCriterion).fit(
        data=full_data, max_steps=len(support)
    )
    selector = CrossValForwardSelection(criterion=BestRSSCriterion)
    state = selector.fit(data=cv_data, max_steps=len(support))

    recovered = set(state.active_set)
    assert len(recovered & set(support)) >= int(0.8 * len(support))


def test_crossval_backward_selection_matches_full_run(esl_cv_data):
    cv_data, support, full_data = esl_cv_data
    BackwardSelection(criterion=BestRSSCriterion).fit(
        data=full_data, max_steps=full_data.gram.shape[0] - len(support)
    )
    selector = CrossValBackwardSelection(criterion=BestRSSCriterion)
    p = cv_data.gram_total.shape[0]
    state = selector.fit(data=cv_data, max_steps=p - len(support))

    recovered = set(state.active_set)
    assert len(recovered & set(support)) >= int(0.8 * len(support))


def test_beam_crossval_forward_selection_recovers_true_support():
    cv_data, support_set = make_cv_support_problem()
    selector = BeamCrossValForwardSelection(beam_width=2)
    state = selector.fit(data=cv_data, max_steps=len(support_set))
    assert set(state.active_set) == set(support_set)


def test_beam_crossval_backward_selection_recovers_true_support():
    cv_data, support_set = make_cv_support_problem()
    selector = BeamCrossValBackwardSelection(beam_width=2, allow_worse=True)
    state = selector.fit(data=cv_data, max_steps=50 - len(support_set))

    assert set(state.active_set) == set(support_set)


def test_cv_beam_matches_greedy_support_on_heterogeneous_folds():
    cv_data, support = make_heterogeneous_cv_problem()
    steps = len(support)
    greedy = CrossValForwardSelection()
    beam = BeamCrossValForwardSelection(beam_width=3)

    greedy_state = greedy.fit(data=cv_data, max_steps=steps)
    beam_state = beam.fit(data=cv_data, max_steps=steps)

    assert set(greedy_state.active_set) == set(beam_state.active_set)

    greedy_rss = []
    beam_rss = []
    for k in range(steps + 1):
        greedy_rss.append(CrossValForwardSelection().fit(data=cv_data, max_steps=k).rss_cv)
        beam_rss.append(
            BeamCrossValForwardSelection(beam_width=3)
            .fit(data=cv_data, max_steps=k)
            .rss_cv
        )
    assert all(
        later <= earlier + 1e-9 for earlier, later in zip(greedy_rss, greedy_rss[1:])
    )
    assert all(
        later <= earlier + 1e-9 for earlier, later in zip(beam_rss, beam_rss[1:])
    )


def test_crossval_beam_can_beat_greedy_in_two_steps():
    cv_data = make_cv_beam_trap_problem()
    greedy_state = CrossValForwardSelection(criterion=BestRSSCriterion).fit(
        data=cv_data, max_steps=2
    )
    beam_state = BeamCrossValForwardSelection(
        beam_width=2, criterion=BestRSSCriterion
    ).fit(data=cv_data, max_steps=2)

    assert 0 in greedy_state.active_set
    assert set(beam_state.active_set) == {1, 2}
    assert beam_state.rss_cv < greedy_state.rss_cv - 1e-9
