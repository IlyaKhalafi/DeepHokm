"""Tests for opponent policies."""

from __future__ import annotations

import random
import subprocess
import sys

import numpy as np
import pytest

from deephokm.env.hokm_env import HokmEnv
from deephokm.env.spaces import Observation, empty_observation
from deephokm.policies.greedy_policy import GreedyPolicy
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
    return obs


def test_greedy_policy_only_returns_masked_actions() -> None:
    policy = GreedyPolicy()
    engine = HokmEngine()
    engine.start_match(seed=11)
    steps = 0
    while engine.state.winner is None and steps < 400:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        obs = _obs()
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
