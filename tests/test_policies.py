"""Tests for opponent policies."""

from __future__ import annotations

import random
import subprocess
import sys

import numpy as np

from deephokm.env.spaces import Observation, empty_observation
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules import HokmEngine
from deephokm.rules.legality import NUM_ACTIONS


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
