"""Tests for the legal (information-respecting) K=1 search policy."""

from __future__ import annotations

import copy

from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.legal_oracle_search import LegalOracleSearchPolicy, play_hand
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase, team_of


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
    search = LegalOracleSearchPolicy(depth=2, seed=0)
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
    search = LegalOracleSearchPolicy(depth=2, seed=0)
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


def test_decide_defers_the_trump_call_to_greedy_policy() -> None:
    search = LegalOracleSearchPolicy(depth=2, seed=0)
    engine = HokmEngine()
    engine.start_match(seed=4)
    assert engine.state.hands.phase is Phase.TRUMP_CALL

    seat = engine.current_seat()
    obs = observation_for(engine.state.hands, seat, engine.state.game_points)
    mask = mask_for(engine.legal_actions(seat))
    expected = GreedyPolicy().act(obs, mask)

    assert search.decide(engine) == expected


def test_decide_never_reads_hidden_information_from_other_seats() -> None:
    """Swapping other seats' real (hidden) hands must not change the
    decision -- the same invariant already established for
    counterfactual_search.py's CounterfactualSearchPolicy, re-verified here
    since this module's ``_clone_for_simulation`` usage is new code, not a
    reused, already-covered call site.
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

    search_a = LegalOracleSearchPolicy(depth=2, seed=99)
    search_a.reset_hand()
    action_a = search_a.decide(engine_a)

    search_b = LegalOracleSearchPolicy(depth=2, seed=99)
    search_b.reset_hand()
    action_b = search_b.decide(engine_b)

    assert action_a == action_b


def test_play_hand_returns_a_valid_result() -> None:
    result = play_hand(seed=7, controlled_team=0, depth=2)
    assert isinstance(result.won, bool)
    assert isinstance(result.team_is_hakem, bool)


def test_play_hand_controlled_team_one_uses_the_other_seats() -> None:
    """Sanity check both team labels run without error and agree with the
    real, independently-known hakem team.
    """
    result = play_hand(seed=8, controlled_team=1, depth=2)
    engine = HokmEngine()
    engine.start_match(seed=8)
    assert result.team_is_hakem == (team_of(engine.state.hands.hakem) == 1)


def test_search_is_deterministic_given_the_same_seed() -> None:
    """Same engine state, same search seed -> same decision (reproducibility)."""
    engine = HokmEngine()
    _play_to_card_play(engine, seed=9)

    search_a = LegalOracleSearchPolicy(depth=2, seed=42)
    search_a.reset_hand()
    action_a = search_a.decide(engine)

    search_b = LegalOracleSearchPolicy(depth=2, seed=42)
    search_b.reset_hand()
    action_b = search_b.decide(engine)

    assert action_a == action_b
