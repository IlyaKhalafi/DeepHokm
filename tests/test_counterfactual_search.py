"""Tests for the exhaustive-action counterfactual rollout diagnostic."""

from __future__ import annotations

import copy
import math

import pytest

from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies.counterfactual_search import (
    CounterfactualSearchPolicy,
    play_match,
)
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase, team_of


@pytest.mark.parametrize("bad_max_p_value", [-0.1, 1.1, math.nan, math.inf])
def test_rejects_an_out_of_range_max_p_value(bad_max_p_value: float) -> None:
    with pytest.raises(ValueError, match="max_p_value"):
        CounterfactualSearchPolicy(n_samples=4, max_p_value=bad_max_p_value)


def _play_to_card_play(engine: HokmEngine, seed: int) -> None:
    """Advance a fresh match past the trump call using GreedyPolicy."""
    engine.start_match(seed=seed)
    greedy = GreedyPolicy()
    while engine.state.hands.phase is Phase.TRUMP_CALL:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        obs = observation_for(engine.state.hands, seat, engine.state.game_points)
        mask = mask_for(legal)
        engine.apply_action(greedy.act(obs, mask), seat=seat)


def test_decide_returns_a_legal_action_and_never_mutates_the_real_engine() -> None:
    search = CounterfactualSearchPolicy(n_samples=3, seed=0)
    engine = HokmEngine()
    _play_to_card_play(engine, seed=2)
    search.reset_hand()

    before = copy.deepcopy(engine.state)
    seat = engine.current_seat()
    legal = engine.legal_actions(seat)

    action = search.decide(engine)

    assert action in legal
    assert engine.state == before


def test_decide_handles_a_forced_single_legal_action_without_searching() -> None:
    search = CounterfactualSearchPolicy(n_samples=3, seed=0)
    engine = HokmEngine()
    _play_to_card_play(engine, seed=3)
    search.reset_hand()

    while len(engine.state.hands.hands[engine.current_seat()]) > 1:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        outcome = engine.apply_action(legal[0], seat=seat)
        if outcome.hand_complete:
            return  # hand ended early (a team already reached 7 tricks)

    seat = engine.current_seat()
    legal = engine.legal_actions(seat)
    assert len(legal) == 1
    assert search.decide(engine) == legal[0]
    # A forced single-action decision is not "searched" -- nothing to compare.
    assert search.decisions_searched == 0


def test_decide_defers_the_trump_call_to_greedy_policy() -> None:
    search = CounterfactualSearchPolicy(n_samples=3, seed=0)
    engine = HokmEngine()
    engine.start_match(seed=4)
    assert engine.state.hands.phase is Phase.TRUMP_CALL

    seat = engine.current_seat()
    obs = observation_for(engine.state.hands, seat, engine.state.game_points)
    mask = mask_for(engine.legal_actions(seat))
    expected = GreedyPolicy().act(obs, mask)

    assert search.decide(engine) == expected
    assert search.decisions_searched == 0


def test_decide_records_a_correctly_shaped_decision() -> None:
    search = CounterfactualSearchPolicy(n_samples=4, seed=0)
    engine = HokmEngine()
    _play_to_card_play(engine, seed=5)
    search.reset_hand()

    seat = engine.current_seat()
    legal = engine.legal_actions(seat)
    if len(legal) <= 1:
        return  # rare unlucky seed; the property is checked elsewhere too

    search.decide(engine)

    assert search.decisions_searched == 1
    assert len(search.records) == 1
    record = search.records[0]
    assert record.root_seat == seat
    assert record.n_legal == len(legal)
    assert record.greedy_action in legal
    assert record.best_action in legal
    assert record.chosen_action in legal
    if record.best_action == record.greedy_action:
        # Nothing was independently evaluated; both scores are the same
        # single (winner's-curse-biased, but irrelevant here) selection
        # batch estimate, and there is no advantage to report.
        assert record.best_score == record.greedy_score
        assert record.p_value == 1.0
    elif record.chosen_action == record.best_action:
        # best_score/greedy_score come from a fresh, independent evaluation
        # batch, so best_score is NOT guaranteed to beat greedy_score here
        # (that would defeat the point of re-evaluating independently) --
        # only the chosen/switch decision is guaranteed consistent with it.
        assert record.p_value <= search.max_p_value


def test_max_p_value_threshold_prevents_switching_on_noise() -> None:
    """An unreachable (zero) threshold must never switch away from
    GreedyPolicy: the exact sign-test p-value is always strictly positive
    for any finite sample, so requiring p <= 0.0 can never be satisfied.
    """
    search = CounterfactualSearchPolicy(n_samples=4, max_p_value=0.0, seed=0)
    engine = HokmEngine()
    _play_to_card_play(engine, seed=6)
    search.reset_hand()

    for _ in range(5):
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        action = search.decide(engine)
        if len(legal) > 1:
            assert action == search.records[-1].greedy_action
        outcome = engine.apply_action(action, seat=seat)
        if outcome.hand_complete or engine.state.winner is not None:
            break


def test_decide_never_reads_hidden_information_from_other_seats() -> None:
    """Swapping other seats' real (hidden) hands must not change the decision.

    ``decide()`` must derive every non-root seat's simulated cards solely
    from ``sample_determinized_hands`` (rng + public sizes/voids), never
    from the real, live ``HandState`` -- the one invariant ``search.py``
    already established that this module must preserve exactly. A
    regression in ``_clone_for_simulation`` that copied a real non-root hand
    into the simulation would still pass every legality/non-mutation test
    above, but would leak information here: this swap changes nothing this
    policy is allowed to see (root's own hand, public sizes, voids are all
    identical) while changing what a leaking simulation would see.
    """
    engine_a = HokmEngine()
    _play_to_card_play(engine_a, seed=12)
    root_seat = engine_a.current_seat()
    legal = engine_a.legal_actions(root_seat)
    if len(legal) <= 1:
        return  # rare unlucky seed; other tests cover the forced-action path

    other_seats = [s for s in range(4) if s != root_seat]
    a, b = other_seats[0], other_seats[1]
    hand_a = list(engine_a.state.hands.hands[a])
    hand_b = list(engine_a.state.hands.hands[b])
    if len(hand_a) < 1 or len(hand_b) < 1:
        return  # need at least one card each to swap

    engine_b = copy.deepcopy(engine_a)
    engine_b.state.hands.hands[a] = [hand_b[0]] + hand_a[1:]
    engine_b.state.hands.hands[b] = [hand_a[0]] + hand_b[1:]
    assert engine_b.state.hands.hands[a] != hand_a  # sanity: a real change
    assert len(engine_b.state.hands.hands[a]) == len(hand_a)
    assert len(engine_b.state.hands.hands[b]) == len(hand_b)

    search_a = CounterfactualSearchPolicy(n_samples=8, seed=99)
    search_a.reset_hand()
    action_a = search_a.decide(engine_a)

    search_b = CounterfactualSearchPolicy(n_samples=8, seed=99)
    search_b.reset_hand()
    action_b = search_b.decide(engine_b)

    assert action_a == action_b
    assert search_a.records[0] == search_b.records[0]


def test_play_match_returns_a_valid_outcome_and_used_the_search() -> None:
    won, search = play_match(seed=7, controlled_team=0, n_samples=4)
    assert isinstance(won, bool)
    assert search.decisions_searched > 0
    # Every recorded decision must belong to the controlled team's seats.
    for record in search.records:
        assert team_of(record.root_seat) == 0


def test_play_match_controlled_team_one_uses_the_other_seats() -> None:
    _, search = play_match(seed=8, controlled_team=1, n_samples=4)
    for record in search.records:
        assert team_of(record.root_seat) == 1


def test_play_match_hand_only_stops_after_the_first_hand() -> None:
    _, search = play_match(seed=11, controlled_team=0, n_samples=4, hand_only=True)
    # A single hand is at most 13 tricks; each controlled seat acts on at
    # most half the plies of a trick it's involved in, so way under 30
    # searched decisions -- this is the actual regression this test guards:
    # hand_only must not silently play the whole multi-hand match anyway.
    assert search.decisions_searched < 30


def test_play_match_hand_only_reports_the_first_hands_own_winner() -> None:
    """The project has a documented history of exactly this bug class in a
    different module (``HokmEnv``'s ``hand_only`` flag: an automatic redeal
    after a hand completes leaked a later hand's outcome). Guard it here by
    independently computing the first hand's winner from a pure-Greedy
    engine on the same seed and requiring an exact match -- not just a
    loose ply-count bound.
    """
    seed = 11
    # An unreachable threshold makes the search never switch away from
    # GreedyPolicy, so it must reproduce a pure-Greedy first hand exactly.
    won, search = play_match(
        seed=seed, controlled_team=0, n_samples=4, max_p_value=0.0, hand_only=True
    )

    engine = HokmEngine()
    engine.start_match(seed=seed)
    greedy = GreedyPolicy()
    outcome = None
    while True:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        obs = observation_for(engine.state.hands, seat, engine.state.game_points)
        mask = mask_for(legal)
        outcome = engine.apply_action(greedy.act(obs, mask), seat=seat)
        if outcome.hand_complete:
            break

    assert outcome.hand_winner_team is not None
    assert won == (outcome.hand_winner_team == 0)
    assert all(record.hand_number == 1 for record in search.records)


def test_two_sample_evaluation_batch_cannot_clear_the_default_threshold() -> None:
    """Regression for a real bug an adversarial review found in the
    previous (Gaussian z-score) implementation: with n_samples=4 (n_eval=2),
    a fully-agreeing 2-sample evaluation batch has zero variance, and the
    old code computed z=inf from that -- unconditionally switching even
    though a 2-for-2 result has a 25% probability under the null (nowhere
    near the intended 5%). The exact sign test cannot make this mistake:
    its minimum possible p-value at n_eval=2 is 0.5**2 = 0.25, always above
    the default max_p_value=0.05, so it must never switch here regardless
    of how one-sided the two draws happen to be.
    """
    _, search = play_match(seed=23, controlled_team=0, n_samples=4, search_seed=0, hand_only=True)
    for record in search.records:
        assert record.chosen_action == record.greedy_action
        assert record.p_value >= 0.25


def test_search_is_deterministic_given_the_same_seed() -> None:
    """Same engine state, same search seed -> same decision (reproducibility)."""
    engine = HokmEngine()
    _play_to_card_play(engine, seed=9)

    search_a = CounterfactualSearchPolicy(n_samples=4, seed=42)
    search_a.reset_hand()
    action_a = search_a.decide(engine)

    search_b = CounterfactualSearchPolicy(n_samples=4, seed=42)
    search_b.reset_hand()
    action_b = search_b.decide(engine)

    assert action_a == action_b
