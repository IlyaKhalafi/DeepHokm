"""Tests for the legal, depth-D, K-sample search with statistical caution."""

from __future__ import annotations

import copy

import pytest

from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies import legal_depth_search
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.legal_depth_search import LegalDepthSearchPolicy, play_hand
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
    search = LegalDepthSearchPolicy(n_samples=4, search_depth=2, seed=0)
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
    search = LegalDepthSearchPolicy(n_samples=4, search_depth=2, seed=0)
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
    search = LegalDepthSearchPolicy(n_samples=4, search_depth=2, seed=0)
    engine = HokmEngine()
    engine.start_match(seed=4)
    assert engine.state.hands.phase is Phase.TRUMP_CALL

    seat = engine.current_seat()
    obs = observation_for(engine.state.hands, seat, engine.state.game_points)
    mask = mask_for(engine.legal_actions(seat))
    expected = GreedyPolicy().act(obs, mask)

    assert search.decide(engine) == expected


def test_min_p_value_threshold_prevents_switching_on_noise() -> None:
    """An unreachable (zero) threshold must never switch away from
    GreedyPolicy: the exact sign-test p-value is always strictly positive
    for any finite sample.
    """
    search = LegalDepthSearchPolicy(n_samples=4, search_depth=2, max_p_value=0.0, seed=0)
    engine = HokmEngine()
    _play_to_card_play(engine, seed=6)
    search.reset_hand()

    for _ in range(5):
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        obs = observation_for(engine.state.hands, seat, engine.state.game_points)
        expected = GreedyPolicy().act(obs, mask_for(legal))
        action = search.decide(engine)
        if len(legal) > 1:
            assert action == expected
        outcome = engine.apply_action(action, seat=seat)
        if outcome.hand_complete or engine.state.winner is not None:
            break


def test_decide_never_reads_hidden_information_from_other_seats() -> None:
    """Swapping other seats' real (hidden) hands must not change the
    decision -- the central invariant this project has repeatedly had to
    defend, re-verified here since this module's sampling/scoring call
    sites are new code.
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

    search_a = LegalDepthSearchPolicy(n_samples=4, search_depth=2, seed=99)
    search_a.reset_hand()
    action_a = search_a.decide(engine_a)

    search_b = LegalDepthSearchPolicy(n_samples=4, search_depth=2, seed=99)
    search_b.reset_hand()
    action_b = search_b.decide(engine_b)

    assert action_a == action_b


def test_play_hand_returns_a_valid_result() -> None:
    result = play_hand(seed=7, controlled_team=0, n_samples=4, search_depth=2)
    assert isinstance(result.won, bool)
    assert isinstance(result.team_is_hakem, bool)


def test_play_hand_controlled_team_one_agrees_with_the_real_hakem_team() -> None:
    result = play_hand(seed=8, controlled_team=1, n_samples=4, search_depth=2)
    engine = HokmEngine()
    engine.start_match(seed=8)
    assert result.team_is_hakem == (team_of(engine.state.hands.hakem) == 1)


def test_search_is_deterministic_given_the_same_seed() -> None:
    engine = HokmEngine()
    _play_to_card_play(engine, seed=9)

    search_a = LegalDepthSearchPolicy(n_samples=4, search_depth=2, seed=42)
    search_a.reset_hand()
    action_a = search_a.decide(engine)

    search_b = LegalDepthSearchPolicy(n_samples=4, search_depth=2, seed=42)
    search_b.reset_hand()
    action_b = search_b.decide(engine)

    assert action_a == action_b


def test_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        LegalDepthSearchPolicy(n_samples=2)


def test_rejects_a_zero_search_depth() -> None:
    with pytest.raises(ValueError, match="search_depth"):
        LegalDepthSearchPolicy(search_depth=0)


@pytest.mark.parametrize("bad_max_p_value", [-0.1, 1.1, float("nan"), float("inf")])
def test_rejects_an_out_of_range_max_p_value(bad_max_p_value: float) -> None:
    with pytest.raises(ValueError, match="max_p_value"):
        LegalDepthSearchPolicy(max_p_value=bad_max_p_value)


def test_play_hand_rejects_an_invalid_controlled_team() -> None:
    with pytest.raises(ValueError, match="controlled_team"):
        play_hand(seed=1, controlled_team=2, n_samples=4, search_depth=1)


def test_decide_switches_when_the_score_clearly_favors_a_challenger(monkeypatch) -> None:
    """A deterministic, mocked score matrix that unambiguously favors one
    non-greedy action must make decide() actually switch to it. None of the
    other (real, unmocked) tests can force this: a real position rarely
    has an advantage large enough to clear the significance bar within a
    small, fast test sample, so without this test the switching logic
    could be entirely disabled (or broken) and nothing here would notice.
    """
    search = LegalDepthSearchPolicy(n_samples=24, search_depth=1, seed=0)
    engine = HokmEngine()
    _play_to_card_play(engine, seed=2)
    search.reset_hand()

    seat = engine.current_seat()
    legal = engine.legal_actions(seat)
    if len(legal) <= 1:
        pytest.skip("no multi-option decision arose for this seed")
    obs = observation_for(engine.state.hands, seat, engine.state.game_points)
    greedy_action = GreedyPolicy().act(obs, mask_for(legal))
    challenger = next(a for a in legal if a != greedy_action)

    def fake_score(self, engine, seat, action, team, sampled_hands):  # noqa: PLR0917
        return 1.0 if action == challenger else -1.0

    monkeypatch.setattr(LegalDepthSearchPolicy, "_score_action", fake_score)

    assert search.decide(engine) == challenger


def test_score_action_uses_search_depth_minus_one_on_the_post_action_state(monkeypatch) -> None:
    """_score_action must score the state AFTER `action` is applied, for
    `team`, via exactly `search_depth - 1` -- not the original engine's
    state, not the wrong team, not an off-by-one depth.
    """
    search = LegalDepthSearchPolicy(n_samples=4, search_depth=3, seed=0)
    engine = HokmEngine()
    _play_to_card_play(engine, seed=5)  # first decision of the hand: far from hand completion
    seat = engine.current_seat()
    legal = engine.legal_actions(seat)
    if len(legal) <= 1:
        pytest.skip("no multi-option decision arose for this seed")
    team = team_of(seat)
    action = legal[0]

    captured = {}

    def fake_oracle_ceiling(clone, controlled_team, *, depth, max_rollout_plies):
        captured["controlled_team"] = controlled_team
        captured["depth"] = depth
        captured["acting_seat"] = clone.current_seat()
        captured["own_hand_size"] = len(clone.state.hands.hands[seat])
        return 0.0

    monkeypatch.setattr(legal_depth_search, "oracle_ceiling", fake_oracle_ceiling)

    sampled_hands = [list(h) for h in engine.state.hands.hands]
    search._score_action(engine, seat, action, team, sampled_hands)

    assert captured["depth"] == search.search_depth - 1
    assert captured["controlled_team"] == team
    # One fewer card than before: `action` was actually applied to reach
    # the state being scored, not the pre-action state.
    assert captured["own_hand_size"] == len(engine.state.hands.hands[seat]) - 1
