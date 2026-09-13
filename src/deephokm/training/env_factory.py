"""Vectorized environment construction for training and evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from deephokm.env import HokmEnv
from deephokm.env.hokm_env import HokmEnv as _HokmEnv
from deephokm.policies.base import HokmPolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.state import NUM_SEATS

OpponentProvider = Callable[[], list[HokmPolicy]]


def make_env(
    rank: int = 0,
    seed: int = 0,
    opponent_provider: OpponentProvider | None = None,
    trick_reward: float = 0.0,
    hand_reward: float = 0.0,
) -> HokmEnv:
    """Build one learning environment for worker ``rank``.

    The learner sits at ``rank % 4`` so seat rotation across workers exposes
    the policy to every seat's perspective (symmetry). When an opponent
    provider is given it is handed to the environment, which re-draws the
    opponent seats on every ``reset()``; otherwise the three opponent seats
    are fixed random policies.

    Args:
        rank: Worker index; determines the learner seat and seed offset.
        seed: Base seed; the worker seed is ``seed + rank``.
        opponent_provider: Callable returning the 4 seat policies per episode.
        trick_reward: Optional per-trick shaping magnitude.
        hand_reward: Optional per-hand shaping magnitude.

    Returns:
        The (unwrapped) environment.
    """
    seat = rank % NUM_SEATS
    reseed = getattr(opponent_provider, "reseed", None)
    if callable(reseed):
        # A provider shared across SubprocVecEnv workers is forked into each
        # subprocess at the same RNG state; without this every worker's
        # first draw (before any per-episode reset() advances it) would pick
        # the identical opponent mix.
        reseed(seed + rank)
    env = _HokmEnv(
        seat=seat,
        opponents=[RandomPolicy(seed + rank + i) for i in range(NUM_SEATS)],
        trick_reward=trick_reward,
        hand_reward=hand_reward,
        opponent_provider=opponent_provider,
    )
    # Seed the engine now so the very first reset() (which SubprocVecEnv
    # issues without a seed) is already reproducible per worker.
    env.reset(seed=seed + rank)
    return env


def make_vec_env(
    n_envs: int = 8,
    seed: int = 0,
    *,
    opponent_provider: OpponentProvider | None = None,
    trick_reward: float = 0.0,
    hand_reward: float = 0.0,
    start_method: str | None = None,
) -> VecMonitor:
    """Build the vectorized training environment.

    Args:
        n_envs: Number of parallel workers.
        seed: Base seed (worker ``i`` uses ``seed + i``).
        opponent_provider: Per-episode opponent factory, re-drawn on reset.
        trick_reward: Optional per-trick shaping.
        hand_reward: Optional per-hand shaping.
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
                hand_reward=hand_reward,
            )

        return _thunk

    return VecMonitor(
        SubprocVecEnv(
            [_make_env_fn(i) for i in range(n_envs)],
            start_method=start_method or "fork",
        )
    )


def make_team_vs_greedy_env(
    rank: int = 0,
    seed: int = 0,
    trick_reward: float = 0.0,
    hand_reward: float = 0.0,
    hand_only: bool = False,
) -> HokmEnv:
    """Build one team-controlled environment with both opposing seats Greedy.

    Used for a dedicated best-response fine-tune: the model controls both
    seats of one team (``control_partner=True``), and the opposing team is
    always the fixed, deterministic :class:`GreedyPolicy` — no self-play
    dilution, no opponent-mix draw. ``rank % 2`` picks which team the model
    controls so parallel workers cover both team labels evenly, matching the
    rotation rationale :func:`make_env` uses for seats.

    Args:
        rank: Worker index; determines the controlled team and seed offset.
        seed: Base seed; the worker seed is ``seed + rank``.
        trick_reward: Optional per-trick shaping magnitude.
        hand_reward: Optional per-hand shaping magnitude.
        hand_only: See :class:`~deephokm.env.hokm_env.HokmEnv`; a curriculum
            knob that ends each episode at the first hand instead of playing
            a full match out.

    Returns:
        The (unwrapped) environment.
    """
    seat = rank % 2
    # The controlled team's two entries are never consulted (control_partner
    # skips them); a real GreedyPolicy placeholder keeps the list uniformly
    # typed rather than reaching for a sentinel None.
    opponents: list[HokmPolicy] = [GreedyPolicy() for _ in range(NUM_SEATS)]
    env = _HokmEnv(
        seat=seat,
        opponents=opponents,
        trick_reward=trick_reward,
        hand_reward=hand_reward,
        control_partner=True,
        hand_only=hand_only,
    )
    env.reset(seed=seed + rank)
    return env


def make_team_vs_greedy_vec_env(
    n_envs: int = 8,
    seed: int = 0,
    *,
    trick_reward: float = 0.0,
    hand_reward: float = 0.0,
    hand_only: bool = False,
    start_method: str | None = None,
) -> VecMonitor:
    """Build the vectorized team-vs-Greedy training environment.

    See :func:`make_team_vs_greedy_env`. GreedyPolicy is stateless and
    deterministic, so unlike :func:`make_vec_env` there is no snapshot pool
    to share across forked workers — each worker's fixed opponents are
    constructed once, in-process, at startup.

    Args:
        n_envs: Number of parallel workers.
        seed: Base seed (worker ``i`` uses ``seed + i``).
        trick_reward: Optional per-trick shaping.
        hand_reward: Optional per-hand shaping.
        hand_only: See :func:`make_team_vs_greedy_env`.
        start_method: Subprocess start method (default fork).

    Returns:
        A ``VecMonitor(SubprocVecEnv)`` stack.
    """

    def _make_env_fn(rank: int) -> Any:
        def _thunk() -> Any:
            return make_team_vs_greedy_env(
                rank=rank,
                seed=seed,
                trick_reward=trick_reward,
                hand_reward=hand_reward,
                hand_only=hand_only,
            )

        return _thunk

    return VecMonitor(
        SubprocVecEnv(
            [_make_env_fn(i) for i in range(n_envs)],
            start_method=start_method or "fork",
        )
    )
