"""Tests for the clairvoyant depth-D ceiling (oracle_search.py)."""

from __future__ import annotations

import copy

import pytest

from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.oracle_search import oracle_best_action, oracle_ceiling
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


def _advance_n_greedy_plies(engine: HokmEngine, seed: int, n_plies: int) -> None:
    """Deterministically replay ``n_plies`` GreedyPolicy-vs-GreedyPolicy card
    plays after the trump call, to reach a specific, reproducible state.
    """
    _play_to_card_play(engine, seed)
    greedy = GreedyPolicy()
    for _ in range(n_plies):
        seat = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(seat)
        obs = observation_for(hands, seat, engine.state.game_points)
        action = legal[0] if len(legal) == 1 else greedy.act(obs, mask_for(legal))
        engine.apply_action(action, seat=seat)


def _play_to_a_real_choice(engine: HokmEngine, seed: int) -> int | None:
    """Advance to the first card-play decision seat 0 faces with >1 legal
    action, or return None if none arises before the hand ends (rare).
    """
    _play_to_card_play(engine, seed)
    greedy = GreedyPolicy()
    for _ in range(60):
        seat = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(seat)
        if seat == 0 and hands.phase is Phase.CARD_PLAY and len(legal) > 1:
            return seat
        obs = observation_for(hands, seat, engine.state.game_points)
        mask = mask_for(legal)
        outcome = engine.apply_action(greedy.act(obs, mask), seat=seat)
        if outcome.hand_complete:
            return None
    return None


def test_oracle_ceiling_never_mutates_the_real_engine() -> None:
    engine = HokmEngine()
    seat = _play_to_a_real_choice(engine, seed=1)
    if seat is None:
        pytest.skip("no multi-option decision arose for this seed")
    before = copy.deepcopy(engine.state)

    oracle_ceiling(engine, team_of(seat), depth=2)

    assert engine.state == before


def test_oracle_ceiling_returns_plus_or_minus_one() -> None:
    engine = HokmEngine()
    seat = _play_to_a_real_choice(engine, seed=2)
    if seat is None:
        pytest.skip("no multi-option decision arose for this seed")

    for depth in (0, 1, 2):
        value = oracle_ceiling(engine, team_of(seat), depth=depth)
        assert value in (-1.0, 1.0)


def test_oracle_ceiling_at_depth_zero_matches_the_actual_greedy_outcome() -> None:
    """With no branching budget, the oracle must reproduce exactly what
    really happens when both teams just keep playing GreedyPolicy from
    here -- i.e. the true, eventual outcome of this exact real game.
    """
    engine = HokmEngine()
    seat = _play_to_a_real_choice(engine, seed=3)
    if seat is None:
        pytest.skip("no multi-option decision arose for this seed")
    team = team_of(seat)

    depth_zero = oracle_ceiling(engine, team, depth=0)

    # Independently continue the SAME real game with GreedyPolicy for
    # everyone (exactly what the generation trajectory does) and see who
    # actually wins the hand.
    greedy = GreedyPolicy()
    while True:
        s = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(s)
        obs = observation_for(hands, s, engine.state.game_points)
        mask = mask_for(legal)
        outcome = engine.apply_action(greedy.act(obs, mask), seat=s)
        if outcome.hand_complete:
            break

    assert outcome.hand_winner_team is not None
    actual = 1.0 if outcome.hand_winner_team == team else -1.0
    assert depth_zero == actual


def test_oracle_ceiling_is_monotonically_nondecreasing_in_depth() -> None:
    """A deeper budget can never do worse: GreedyPolicy's own line at any
    depth is always one of the branches a deeper search also considers.
    """
    for seed in range(15):
        engine = HokmEngine()
        seat = _play_to_a_real_choice(engine, seed=seed)
        if seat is None:
            continue
        team = team_of(seat)
        values = [oracle_ceiling(engine, team, depth=d) for d in (0, 1, 2)]
        assert values == sorted(values)


def test_oracle_ceiling_rejects_a_non_card_play_state() -> None:
    engine = HokmEngine()
    engine.start_match(seed=4)
    assert engine.state.hands.phase is Phase.TRUMP_CALL
    with pytest.raises(ValueError, match="CARD_PLAY"):
        oracle_ceiling(engine, controlled_team=0, depth=2)


def test_oracle_ceiling_handles_a_forced_single_legal_action() -> None:
    engine = HokmEngine()
    _play_to_card_play(engine, seed=5)
    greedy = GreedyPolicy()
    while len(engine.state.hands.hands[engine.current_seat()]) > 1:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        obs = observation_for(engine.state.hands, seat, engine.state.game_points)
        mask = mask_for(legal)
        outcome = engine.apply_action(greedy.act(obs, mask), seat=seat)
        if outcome.hand_complete:
            pytest.skip("hand ended early for this seed")

    seat = engine.current_seat()
    assert len(engine.legal_actions(seat)) == 1
    value = oracle_ceiling(engine, team_of(seat), depth=2)
    assert value in (-1.0, 1.0)


def test_oracle_ceiling_is_deterministic() -> None:
    """No randomness anywhere in this module -- repeat calls must agree."""
    engine = HokmEngine()
    seat = _play_to_a_real_choice(engine, seed=6)
    if seat is None:
        pytest.skip("no multi-option decision arose for this seed")
    team = team_of(seat)

    first = oracle_ceiling(engine, team, depth=2)
    second = oracle_ceiling(engine, team, depth=2)
    assert first == second


# The tests above only ever check aggregate properties (legality,
# monotonicity, never-mutates); an adversarial review found none of them
# would actually fail if the controlled-team branching were disabled
# entirely (every depth silently collapsing to the depth-0 rollout still
# satisfies "non-decreasing"). These three use concrete, mined real-game
# fixtures to prove the search actually does what it claims: the ROOT
# seat's own branching finds a strictly better line, the PARTNER seat's
# branching does too (not just the root's), and a forced single-legal-
# action decision truly consumes none of the depth budget.


def test_oracle_ceiling_root_branching_finds_a_strictly_better_line() -> None:
    """Seed 14, ply 0: root seat 0's own choice, depth=1 beats depth=0."""
    engine = HokmEngine()
    _advance_n_greedy_plies(engine, seed=14, n_plies=0)
    seat = engine.current_seat()
    assert seat == 0
    assert len(engine.legal_actions(seat)) > 1
    team = team_of(seat)

    depth_zero = oracle_ceiling(engine, team, depth=0)
    depth_one = oracle_ceiling(engine, team, depth=1)

    assert depth_zero == -1.0
    assert depth_one == 1.0


def test_oracle_ceiling_partner_branching_finds_a_strictly_better_line() -> None:
    """Seed 14, ply 14: seat 2 (seat 0's partner) is acting, depth=1 beats
    depth=0 -- proving the search genuinely branches over the PARTNER's
    own decisions too, not only the root seat that initiated the search.
    """
    engine = HokmEngine()
    _advance_n_greedy_plies(engine, seed=14, n_plies=14)
    seat = engine.current_seat()
    assert seat == 2
    assert len(engine.legal_actions(seat)) > 1
    team = team_of(seat)

    depth_zero = oracle_ceiling(engine, team, depth=0)
    depth_one = oracle_ceiling(engine, team, depth=1)

    assert depth_zero == -1.0
    assert depth_one == 1.0


def test_oracle_ceiling_forced_action_does_not_consume_depth_budget() -> None:
    """Seed 0, ply 13: the acting seat has only one legal action. depth=1
    measured HERE must equal depth=1 measured at the very next real
    decision, right after that forced action is applied -- proving the
    forced move consumed none of the depth budget (a real, later decision
    still has the full budget available, not one less).
    """
    engine = HokmEngine()
    _advance_n_greedy_plies(engine, seed=0, n_plies=13)
    seat = engine.current_seat()
    legal = engine.legal_actions(seat)
    assert len(legal) == 1
    team = team_of(seat)

    before = oracle_ceiling(engine, team, depth=1)

    after_engine = copy.deepcopy(engine)
    obs = observation_for(after_engine.state.hands, seat, after_engine.state.game_points)
    action = GreedyPolicy().act(obs, mask_for(legal))
    outcome = after_engine.apply_action(action, seat=seat)
    assert not outcome.hand_complete
    assert after_engine.state.hands.phase is Phase.CARD_PLAY

    after = oracle_ceiling(after_engine, team, depth=1)

    assert before == after == 1.0


def test_oracle_best_action_never_mutates_the_real_engine_and_returns_legal() -> None:
    engine = HokmEngine()
    seat = _play_to_a_real_choice(engine, seed=7)
    if seat is None:
        pytest.skip("no multi-option decision arose for this seed")
    before = copy.deepcopy(engine.state)
    legal = engine.legal_actions(seat)

    action = oracle_best_action(engine, team_of(seat), depth=2)

    assert action in legal
    assert engine.state == before


def test_oracle_best_action_achieves_the_ceiling_value() -> None:
    """The action oracle_best_action returns, once applied, must lead into
    a subtree whose value (continuing the search one level shallower)
    equals oracle_ceiling's own reported best value at the original depth.
    """
    engine = HokmEngine()
    seat = _play_to_a_real_choice(engine, seed=14)
    if seat is None:
        pytest.skip("no multi-option decision arose for this seed")
    team = team_of(seat)
    depth = 2

    best_value = oracle_ceiling(engine, team, depth=depth)
    action = oracle_best_action(engine, team, depth=depth)

    child = copy.deepcopy(engine)
    outcome = child.apply_action(action, seat=seat)
    if outcome.hand_complete:
        assert outcome.hand_winner_team is not None
        achieved = 1.0 if outcome.hand_winner_team == team else -1.0
    else:
        achieved = oracle_ceiling(child, team, depth=depth - 1)
    assert achieved == best_value


def test_oracle_best_action_handles_a_forced_single_legal_action() -> None:
    engine = HokmEngine()
    _play_to_card_play(engine, seed=5)
    greedy = GreedyPolicy()
    while len(engine.state.hands.hands[engine.current_seat()]) > 1:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        obs = observation_for(engine.state.hands, seat, engine.state.game_points)
        mask = mask_for(legal)
        outcome = engine.apply_action(greedy.act(obs, mask), seat=seat)
        if outcome.hand_complete:
            pytest.skip("hand ended early for this seed")

    seat = engine.current_seat()
    legal = engine.legal_actions(seat)
    assert len(legal) == 1
    assert oracle_best_action(engine, team_of(seat), depth=2) == legal[0]


def test_oracle_best_action_rejects_a_non_controlled_seats_turn() -> None:
    engine = HokmEngine()
    seat = _play_to_a_real_choice(engine, seed=8)
    if seat is None:
        pytest.skip("no multi-option decision arose for this seed")
    other_team = 1 - team_of(seat)
    with pytest.raises(ValueError, match="controlled_team"):
        oracle_best_action(engine, other_team, depth=2)


def test_oracle_best_action_is_deterministic() -> None:
    engine = HokmEngine()
    seat = _play_to_a_real_choice(engine, seed=9)
    if seat is None:
        pytest.skip("no multi-option decision arose for this seed")
    team = team_of(seat)

    first = oracle_best_action(engine, team, depth=2)
    second = oracle_best_action(engine, team, depth=2)
    assert first == second
