"""Vectorized environment construction for training and evaluation."""

from __future__ import annotations

from typing import Any

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from deephokm.env import HokmEnv
from deephokm.env.hokm_env import HokmEnv as _HokmEnv
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.state import NUM_SEATS


def make_env(
    cfg: Any = None,
    rank: int = 0,
    seed: int = 0,
    opponent_provider: Any = None,
    trick_reward: float = 0.0,
) -> HokmEnv:
    """Build one learning environment for worker ``rank``.

    The learner sits at ``rank % 4`` so seat rotation across workers exposes
    the policy to every seat's perspective (symmetry). Opponents come from
    ``opponent_provider(observation_space, action_space)`` when given, else
    uniform random policies.

    Args:
        cfg: Reserved config object (unused fields are allowed).
        rank: Worker index; determines the learner seat and seed offset.
        seed: Base seed; the worker seed is ``seed + rank``.
        opponent_provider: Callable returning the 4 opponent policies.
        trick_reward: Optional per-trick shaping magnitude.

    Returns:
        The (unwrapped) environment.
    """
    del cfg  # configuration flows through explicit arguments
    seat = rank % NUM_SEATS
    if opponent_provider is not None:
        opponents = opponent_provider(None, None)
    else:
        opponents = [RandomPolicy(seed + rank + i) for i in range(NUM_SEATS)]
    env = _HokmEnv(seat=seat, opponents=opponents, trick_reward=trick_reward)
    # Seed the engine now so the very first reset() (which SubprocVecEnv
    # issues without a seed) is already reproducible per worker.
    env.reset(seed=seed + rank)
    return env


def make_vec_env(
    n_envs: int = 8,
    seed: int = 0,
    opponent_provider: Any = None,
    trick_reward: float = 0.0,
    start_method: str | None = None,
) -> VecMonitor:
    """Build the vectorized training environment.

    Args:
        n_envs: Number of parallel workers.
        seed: Base seed (worker ``i`` uses ``seed + i``).
        opponent_provider: Per-worker opponent factory.
        trick_reward: Optional per-trick shaping.
        start_method: Subprocess start method (default fork).

    Returns:
        A ``VecMonitor(SubprocVecEnv)`` stack.
    """

    def _make_env_fn(rank: int) -> Any:
        # Zero-arg callable per worker: SubprocVecEnv invokes it inside the
        # subprocess. VecMonitor at the stack level provides episode
        # statistics; an inner Monitor per worker would only duplicate them.
        def _thunk() -> Any:
            return make_env(
                rank=rank,
                seed=seed,
                opponent_provider=opponent_provider,
                trick_reward=trick_reward,
            )

        return _thunk

    return VecMonitor(
        SubprocVecEnv(
            [_make_env_fn(i) for i in range(n_envs)],
            start_method=start_method or "fork",
        )
    )
