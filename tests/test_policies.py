"""Tests for opponent policies."""

from __future__ import annotations

import random
import subprocess
import sys

import numpy as np
import pytest

from deephokm.env.hokm_env import HokmEnv
from deephokm.env.spaces import (
    Observation,
    empty_observation,
    mask_for,
    observation_for,
)
from deephokm.policies import greedy_policy as greedy_module
from deephokm.policies.greedy_policy import GreedyPolicy, PublicKnowledge
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules import HokmEngine
from deephokm.rules.legality import NUM_ACTIONS
from deephokm.rules.state import team_of


def _mask(actions: list[int]) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    mask[actions] = 1
    return mask


def _obs() -> Observation:
    return empty_observation()


def test_random_policy_returns_legal_actions() -> None:
    policy = RandomPolicy(0)
    mask = _mask([3, 17, 52])
    for _ in range(100):
        assert policy.act(_obs(), mask) in {3, 17, 52}


def test_random_policy_rejects_empty_mask() -> None:
    policy = RandomPolicy(1)
    with np.errstate(all="ignore"):
        try:
            policy.act(_obs(), _mask([]))
        except ValueError as e:
            assert "no legal actions" in str(e)
        else:
            raise AssertionError("empty mask must raise")


def test_random_policy_is_uniform() -> None:
    policy = RandomPolicy(2)
    mask = _mask([10, 20, 30])
    counts = {10: 0, 20: 0, 30: 0}
    for _ in range(3000):
        counts[policy.act(_obs(), mask)] += 1
    for action, count in counts.items():
        assert 800 <= count <= 1200, f"action {action} chosen {count}/3000"


def test_random_policy_reproducible() -> None:
    a = RandomPolicy(42)
    b = RandomPolicy(42)
    mask = _mask(list(range(NUM_ACTIONS)))
    assert [a.act(_obs(), mask) for _ in range(50)] == [b.act(_obs(), mask) for _ in range(50)]


def test_random_policy_reset_restores_stream() -> None:
    policy = RandomPolicy(7)
    mask = _mask(list(range(NUM_ACTIONS)))
    first = [policy.act(_obs(), mask) for _ in range(10)]
    policy.reset()
    again = [policy.act(_obs(), mask) for _ in range(10)]
    assert first == again


def test_random_policy_reset_with_new_seed() -> None:
    policy = RandomPolicy(7)
    mask = _mask(list(range(NUM_ACTIONS)))
    before = [policy.act(_obs(), mask) for _ in range(10)]
    policy.reset(seed=99)
    after = [policy.act(_obs(), mask) for _ in range(10)]
    assert before != after or len(set(after)) <= 1


def test_policy_uses_only_masked_actions_under_engine() -> None:
    """End-to-end: the policy drives all four seats legally for a full match."""
    engine = HokmEngine(random.Random(5))
    engine.start_match()
    policies = [RandomPolicy(100 + i) for i in range(4)]
    while engine.state.winner is None:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        mask[legal] = 1
        action = policies[seat].act(_obs(), mask)  # type: ignore[arg-type]
        assert action in legal
        engine.apply_action(action)


def test_random_policy_default_seed_unpredictable() -> None:
    """Two unseeded policies produce different streams (time-based seeding)."""
    a = RandomPolicy()
    b = RandomPolicy()
    mask = _mask(list(range(NUM_ACTIONS)))
    a_actions = [a.act(_obs(), mask) for _ in range(20)]
    b_actions = [b.act(_obs(), mask) for _ in range(20)]
    assert a_actions != b_actions


def test_policy_modules_import_without_env_first() -> None:
    """Importing policies before env must not hit a circular import."""
    code = "from deephokm.policies.base import HokmPolicy; print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def _greedy_obs(
    hand: list[int],
    *,
    seat: int = 0,
    trump: int | None = None,
    trick_play: dict[int, int] | None = None,
    history: list[tuple[int, int]] | None = None,
) -> Observation:
    """Build a minimal observation for the greedy policy's decisions."""
    obs = empty_observation()
    obs["hand"][hand] = 1
    obs["seen"][hand] = 1
    obs["seat"][seat] = 1
    obs["phase"][1] = 1
    if trump is not None:
        obs["trump"][trump] = 1
    for played_seat, card in (trick_play or {}).items():
        obs["trick_play"][played_seat] = card
        obs["trick"][card] = 1
        obs["seen"][card] = 1
    for slot, (played_seat, card) in enumerate(reversed(history or [])):
        obs["history"][slot] = card
        obs["history_role"][slot] = (played_seat - seat) % 4
        obs["seen"][card] = 1
    return obs


def test_greedy_policy_only_returns_masked_actions() -> None:
    policy = GreedyPolicy()
    engine = HokmEngine()
    engine.start_match(seed=11)
    steps = 0
    while engine.state.winner is None and steps < 400:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        obs = observation_for(engine.state.hands, seat, engine.state.game_points)
        action = policy.act(obs, _mask(legal)) if legal else None
        assert action is None or action in legal
        engine.apply_action(action if action is not None else legal[0])
        steps += 1


def test_greedy_policy_calls_its_longest_suit() -> None:
    """Six clubs against three of everything else: clubs is the call."""
    policy = GreedyPolicy()
    hand = [0, 1, 2, 3, 4, 5] + [13, 14, 15] + [26, 27, 28] + [39]
    obs = _greedy_obs(hand)
    obs["phase"][:] = 0
    obs["phase"][0] = 1
    action = policy.act(obs, _mask([52, 53, 54, 55]))
    assert action == 52, "clubs (suit 0) is the longest suit held"


def test_greedy_policy_wins_the_trick_as_cheaply_as_possible() -> None:
    """Holding the 10 and the ace of the led suit, play the 10."""
    policy = GreedyPolicy()
    # Led: clubs 9 (rank index 7 -> card 7) by seat 3, so seat 0 is next and
    # its partner (seat 2) has not played.
    hand = [8, 12, 20]  # clubs 10, clubs ace, diamonds 9
    obs = _greedy_obs(hand, seat=0, trump=3, trick_play={3: 7})
    action = policy.act(obs, _mask([8, 12]))
    assert action == 8, "the cheapest winner, not the ace"


def test_greedy_policy_lets_its_partner_win() -> None:
    """The partner already leads the trick: throw the lowest card."""
    policy = GreedyPolicy()
    hand = [8, 12]
    # Seat 2 (the partner) led clubs ace; seat 3 followed with clubs 3.
    obs = _greedy_obs(hand, seat=0, trump=3, trick_play={2: 12 - 0, 3: 1})
    obs["trick_play"][2] = 11  # clubs king leads the trick
    action = policy.act(obs, _mask([8, 12]))
    assert action == 8, "keep the ace while the partner is winning"


def test_greedy_policy_beats_random_by_a_wide_margin() -> None:
    """The scripted baseline must be a meaningfully stronger yardstick."""
    wins = 0
    games = 40
    learner = GreedyPolicy()
    for game in range(games):
        env = HokmEnv(seat=0, opponents=[RandomPolicy(200 + i) for i in range(4)])
        obs, info = env.reset(seed=game)
        done = False
        while not done:
            action = learner.act(obs, np.asarray(info["action_mask"], dtype=bool))
            obs, _reward, terminated, truncated, info = env.step(np.int64(action))
            done = terminated or truncated
        wins += int(env.engine.state.winner == team_of(0))
        env.close()
    assert wins / games > 0.8, f"greedy won only {wins}/{games} against random play"


def test_greedy_trump_call_always_prefers_the_longer_suit() -> None:
    """Four honors cannot outweigh a genuinely longer five-card suit."""
    policy = GreedyPolicy()
    hand = [0, 1, 2, 3, 4, 22, 23, 24, 25]
    obs = _greedy_obs(hand)
    obs["phase"][:] = 0
    obs["phase"][0] = 1
    assert policy.act(obs, _mask([52, 53, 54, 55])) == 52


def test_greedy_trump_call_respects_a_restricted_mask() -> None:
    policy = GreedyPolicy()
    obs = _greedy_obs([0, 1, 2, 22, 23, 24])
    obs["phase"][:] = 0
    obs["phase"][0] = 1
    assert policy.act(obs, _mask([53, 54])) == 53


def test_public_knowledge_reconstructs_history_and_void_suits() -> None:
    history = [
        (3, 8),
        (0, 13),
        (1, 1),
        (2, 2),
        (1, 42),
        (2, 43),
        (3, 26),
        (0, 14),
    ]
    obs = _greedy_obs([12], seat=2, trump=3, history=history)
    knowledge = PublicKnowledge.from_observation(obs)
    assert knowledge.played_cards == tuple(card for _, card in history)
    assert knowledge.void_suits[0] == frozenset({0, 3})
    assert knowledge.void_suits[3] == frozenset({3})
    assert knowledge.void_suits[1] == frozenset()


def test_public_knowledge_infers_current_trick_void_across_seat_wrap() -> None:
    obs = _greedy_obs([1, 2], seat=1, trump=3, trick_play={3: 8, 0: 13})
    knowledge = PublicKnowledge.from_observation(obs)
    assert knowledge.current_trick == ((3, 8), (0, 13))
    assert knowledge.void_suits[0] == frozenset({0})
    assert knowledge.void_suits[3] == frozenset()


def test_greedy_discards_low_when_opponent_winner_is_unbeatable() -> None:
    policy = GreedyPolicy()
    obs = _greedy_obs([26, 37], seat=0, trump=3, trick_play={3: 38})
    assert policy.act(obs, _mask([26, 37])) == 26


def test_greedy_does_not_try_to_beat_an_opponent_trump_with_led_ace() -> None:
    policy = GreedyPolicy()
    obs = _greedy_obs(
        [26, 38],
        seat=0,
        trump=3,
        trick_play={2: 34, 3: 39},
    )
    assert policy.act(obs, _mask([26, 38])) == 26


def test_greedy_uses_the_cheapest_overtrump_when_playing_last() -> None:
    policy = GreedyPolicy()
    obs = _greedy_obs(
        [43, 51],
        seat=0,
        trump=3,
        trick_play={1: 38, 2: 0, 3: 42},
    )
    assert policy.act(obs, _mask([43, 51])) == 43


def test_greedy_preserves_trump_when_a_higher_trump_already_wins() -> None:
    policy = GreedyPolicy()
    obs = _greedy_obs(
        [0, 39],
        seat=0,
        trump=3,
        trick_play={2: 26, 3: 51},
    )
    assert policy.act(obs, _mask([0, 39])) == 0


def test_greedy_protects_a_threatened_partner_with_a_master_trump() -> None:
    policy = GreedyPolicy()
    obs = _greedy_obs(
        [39, 51],
        seat=0,
        trump=3,
        trick_play={2: 47, 3: 46},
    )
    assert policy.act(obs, _mask([39, 51])) == 51


def test_greedy_uses_remembered_void_to_avoid_overtaking_partner() -> None:
    policy = GreedyPolicy()
    history = [(0, 40), (1, 0), (2, 41), (3, 42)]
    obs = _greedy_obs(
        [39, 51],
        seat=0,
        trump=3,
        trick_play={2: 47, 3: 46},
        history=history,
    )
    assert policy.act(obs, _mask([39, 51])) == 39


def test_greedy_preserves_winner_when_overtake_still_is_not_safe() -> None:
    policy = GreedyPolicy()
    obs = _greedy_obs(
        [2, 11],
        seat=0,
        trump=3,
        trick_play={2: 6, 3: 1},
    )
    assert policy.act(obs, _mask([2, 11])) == 2


def test_greedy_spends_stronger_winner_when_next_trick_decides_hand() -> None:
    policy = GreedyPolicy()
    obs = _greedy_obs([8, 12], seat=0, trump=3, trick_play={3: 7})
    obs["tricks_won"][:] = [6, 5]
    assert policy.act(obs, _mask([8, 12])) == 12


def test_greedy_partially_protects_partner_when_next_trick_decides_hand() -> None:
    policy = GreedyPolicy()
    obs = _greedy_obs([2, 11], seat=0, trump=3, trick_play={2: 6, 3: 1})
    obs["tricks_won"][:] = [5, 6]
    assert policy.act(obs, _mask([2, 11])) == 11


def test_greedy_third_hand_spends_a_high_card_to_avoid_an_easy_final_seat_counter() -> None:
    hand = [3, 12, *range(14, 25)]
    obs = _greedy_obs(hand, seat=2, trump=3, trick_play={0: 0, 1: 2})
    assert GreedyPolicy().act(obs, _mask([3, 12])) == 12


def test_greedy_fourth_hand_keeps_the_cheapest_winner_without_future_opponents() -> None:
    hand = [3, 12, *range(14, 25)]
    obs = _greedy_obs(hand, seat=3, trump=3, trick_play={0: 0, 1: 1, 2: 2})
    assert GreedyPolicy().act(obs, _mask([3, 12])) == 3


def test_greedy_third_hand_does_not_replace_its_partner_protection_rules() -> None:
    hand = [3, 12, *range(14, 25)]
    obs = _greedy_obs(hand, seat=2, trump=3, trick_play={0: 11, 1: 2})
    assert GreedyPolicy().act(obs, _mask([3, 12])) == 3


@pytest.mark.parametrize(
    ("unseen", "voids", "expected"),
    [
        (frozenset({14, 42}), frozenset(), 40),  # forced side-suit follow
        (frozenset({26, 27, 42}), frozenset({1}), 45),  # led suit known void
        (frozenset({26, 27, 42}), frozenset({3}), 40),  # no trump possible
    ],
)
def test_greedy_third_hand_uses_follow_and_void_facts_before_overtrumping(
    unseen: frozenset[int],
    voids: frozenset[int],
    expected: int,
) -> None:
    knowledge = PublicKnowledge(
        seat=2,
        trump=3,
        hand=frozenset({40, 45}),
        played_cards=(),
        current_trick=((0, 13), (1, 25)),
        void_suits=(frozenset(), frozenset(), frozenset(), voids),
        unseen_cards=unseen,
    )
    assert GreedyPolicy()._play_with_knowledge(knowledge, [40, 45]) == expected


@pytest.mark.parametrize(("score", "expected"), [((0, 0), 40), ((6, 5), 45), ((5, 6), 45)])
def test_greedy_odds_failure_keeps_the_previous_decisive_trick_rule(
    score: tuple[int, int],
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(greedy_module, "opponent_beating_probability", lambda **kwargs: None)
    knowledge = PublicKnowledge(
        seat=2,
        trump=3,
        hand=frozenset({40, 45}),
        played_cards=(),
        current_trick=((0, 13), (1, 25)),
        void_suits=(frozenset(),) * 4,
        unseen_cards=frozenset({26, 27, 42}),
        tricks_won=score,
    )
    assert GreedyPolicy()._play_with_knowledge(knowledge, [40, 45]) == expected


def test_public_knowledge_maps_absolute_scores_to_acting_team() -> None:
    knowledge = PublicKnowledge.from_state(
        seat=1,
        trump=3,
        hand=[0],
        played=[],
        played_by=[],
        current_trick=[],
        tricks_won=[5, 6],
    )
    assert knowledge.tricks_won == (6, 5)


def test_public_knowledge_infers_score_when_state_caller_omits_it() -> None:
    knowledge = PublicKnowledge.from_state(
        seat=1,
        trump=3,
        hand=[14],
        played=[12, 13, 1, 2],
        played_by=[0, 1, 2, 3],
        current_trick=[],
    )
    assert knowledge.tricks_won == (0, 1)


def test_greedy_does_not_lead_a_master_into_a_known_opponent_ruff() -> None:
    policy = GreedyPolicy()
    history = [(0, 12), (1, 13), (2, 10), (3, 9)]
    obs = _greedy_obs([11, 38], seat=0, trump=3, history=history)
    assert policy.act(obs, _mask([11, 38])) == 38


def test_greedy_develops_a_short_side_suit_early() -> None:
    hand = [3, 8, 13, 14, 25, 39, 50]
    obs = _greedy_obs(hand, trump=3)
    assert GreedyPolicy().act(obs, _mask(hand)) == 3


@pytest.mark.parametrize("score", [(4, 3), (6, 0), (0, 6)])
def test_greedy_cashes_controls_after_development_or_on_decisive_trick(
    score: tuple[int, int],
) -> None:
    hand = [3, 8, 13, 14, 25, 39, 50]
    obs = _greedy_obs(hand, trump=3)
    obs["tricks_won"][:] = score
    assert GreedyPolicy().act(obs, _mask(hand)) == 25


def test_greedy_does_not_develop_a_short_suit_into_a_known_ruff() -> None:
    hand = [3, 8, 14, 15, 25, 40, 50]
    obs = _greedy_obs(hand, trump=3, history=[(0, 12), (1, 13), (2, 11), (3, 10)])
    assert GreedyPolicy().act(obs, _mask(hand)) == 25


def test_greedy_knows_a_void_opponent_cannot_ruff_when_trump_is_exhausted() -> None:
    played = tuple(range(39, 52))
    knowledge = PublicKnowledge(
        seat=0,
        trump=3,
        hand=frozenset({11}),
        played_cards=played,
        current_trick=(),
        void_suits=(frozenset(), frozenset({0}), frozenset(), frozenset()),
        unseen_cards=frozenset(range(39)),
    )
    assert not GreedyPolicy._can_ruff(1, 0, knowledge)


def test_greedy_preserves_king_backed_by_own_ace_on_discard() -> None:
    """Higher cards in our hand cannot be held by an opponent."""
    policy = GreedyPolicy()
    obs = _greedy_obs(
        [11, 12, 39],
        seat=0,
        trump=3,
        trick_play={3: 25},
    )
    assert policy.act(obs, _mask([11, 12, 39])) == 39


def test_greedy_act_and_public_state_fast_path_match() -> None:
    policy = GreedyPolicy()
    for seed in (37, 81, 134, 233, 377):
        engine = HokmEngine()
        engine.start_match(seed=seed)
        while engine.state.winner is None:
            seat = engine.current_seat()
            hands = engine.state.hands
            legal = engine.legal_actions(seat)
            obs = observation_for(hands, seat, engine.state.game_points)
            action = policy.act(obs, mask_for(legal))
            assert action in legal
            if action < 52:
                fast_action = policy.play_from_state(
                    hands.trump,
                    hands.current_trick,
                    seat,
                    legal,
                    hand=hands.hands[seat],
                    played=hands.played,
                    played_by=hands.played_by,
                    void_suits=hands.void_suits,
                    tricks_won=hands.tricks_won,
                )
                assert fast_action == action
            engine.apply_action(action, seat=seat)


def test_greedy_rejects_incomplete_history() -> None:
    obs = _greedy_obs([1], seat=0, history=[(3, 8)])
    with pytest.raises(ValueError, match="complete four-card tricks"):
        GreedyPolicy().act(obs, _mask([1]))


def test_current_best_rejects_a_full_trick_instead_of_guessing() -> None:
    """A partial trick always has a seat with no predecessor in it -- that
    seat is the leader. A full 4-entry trick has none (every predecessor has
    played too), which never happens through a real call site (the acting
    seat's own card is never in ``played`` yet), so this must fail loudly
    rather than silently default to seat 0.
    """
    played = [(0, 5), (1, 6), (2, 7), (3, 8)]
    with pytest.raises(ValueError, match="no leader"):
        GreedyPolicy._current_best(played, trump=None)
