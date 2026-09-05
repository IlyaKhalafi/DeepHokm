"""Training callbacks: self-play snapshots and evaluation gauntlet logging.

The self-play callback periodically saves the current policy into the
:class:`~deephokm.training.selfplay.SelfPlayPool`. Evaluation against the
gauntlet (random policy, earliest snapshot, latest snapshot) is handled by
SB3's ``EvalCallback`` plus :class:`GauntletCallback`, which runs round-robin
evaluations and writes win rates to the TensorBoard log.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from deephokm.env.hokm_env import HokmEnv
from deephokm.env.spaces import Observation
from deephokm.rules.state import team_of
from deephokm.training.gauntlet_workers import run_gauntlet_shards
from deephokm.training.selfplay import SelfPlayPool, SnapshotPolicy


class SelfPlayCallback(BaseCallback):
    """Snapshot the current policy into the opponent pool on a step cadence.

    Attributes:
        pool: The pool receiving snapshots.
        save_freq: Environment steps between snapshots.
        save_path: Directory the snapshot zips are written to.
    """

    def __init__(
        self,
        pool: SelfPlayPool,
        *,
        save_freq: int,
        save_path: Path,
        verbose: int = 0,
    ) -> None:
        """Create the callback.

        Args:
            pool: The self-play pool to feed.
            save_freq: Steps between snapshots.
            save_path: Directory for snapshot files.
            verbose: SB3 verbosity.
        """
        super().__init__(verbose)
        self.pool = pool
        self.save_freq = save_freq
        self.save_path = save_path
        self._last_snapshot_step = 0

    def _on_step(self) -> bool:
        """Snapshot when the step cadence elapses."""
        if self.num_timesteps - self._last_snapshot_step >= self.save_freq:
            self._snapshot()
        return True

    def _snapshot(self) -> None:
        """Save the current policy into the pool."""
        self.save_path.mkdir(parents=True, exist_ok=True)
        path = self.save_path / f"snapshot_{self.num_timesteps:012d}.zip"
        # Save the policy only (the full model carries the optimizer state,
        # which opponents never need and which triples the snapshot size).
        self.model.policy.save(str(path))
        self.pool.add(path, self.num_timesteps)
        self._last_snapshot_step = self.num_timesteps
        self.logger.record("selfplay/pool_size", len(self.pool))
        if self.verbose:
            print(f"[selfplay] snapshot at step {self.num_timesteps} -> {path.name}")


class GauntletCallback(BaseCallback):
    """Evaluate the policy against fixed opponents and log win rates.

    Games run in parallel batches: ``batch_size`` environments advance
    together and their observations are stacked into a single policy forward
    pass per decision round, which is an order of magnitude faster than
    one-game-at-a-time inference on this network.
    """

    def __init__(
        self,
        eval_env_factory: Callable[[], HokmEnv],
        *,
        opponents: dict[str, Callable[[], list[Any]]],
        n_games: int,
        eval_freq: int,
        batch_size: int = 32,
        n_workers: int = 8,
        verbose: int = 0,
    ) -> None:
        """Create the callback.

        Args:
            eval_env_factory: Zero-arg callable producing a fresh eval env.
            opponents: Gauntlet name -> factory producing the opponent list.
            n_games: Games per gauntlet opponent per round.
            eval_freq: Steps between evaluation rounds.
            batch_size: Environments advanced concurrently per worker.
            n_workers: Processes sharing a gauntlet round.
            verbose: SB3 verbosity.
        """
        super().__init__(verbose)
        self.eval_env_factory = eval_env_factory
        self.opponents = opponents
        self.n_games = n_games
        self.eval_freq = eval_freq
        self.batch_size = batch_size
        self.n_workers = n_workers
        self._last_eval_step = 0
        self._eval_count = 0

    def _on_step(self) -> bool:
        """Run a gauntlet round when the cadence elapses."""
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._run_gauntlet()
        return True

    def _on_training_end(self) -> None:
        """Run a final round so the log ends on fresh numbers."""
        self._run_gauntlet()

    def _run_gauntlet(self) -> None:
        """Play every gauntlet opponent and log win rates.

        Rounds run in worker processes (each plays a shard of games with its
        own batched loop and model copy) so a 100-game gauntlet costs a
        fraction of the wall time it would take inside the training loop.
        """
        with tempfile.TemporaryDirectory() as tmp:
            model_path = str(Path(tmp) / "eval_model.zip")
            self.model.save(model_path)
            for name, make_opponents in self.opponents.items():
                snapshot_path = _snapshot_path_of(make_opponents())
                wins = run_gauntlet_shards(
                    model_path=model_path,
                    snapshot_path=snapshot_path,
                    n_games=self.n_games,
                    batch_size=self.batch_size,
                    n_workers=self.n_workers,
                    seed=self._eval_count,
                )
                win_rate = wins / self.n_games
                self.logger.record(f"gauntlet/{name}", win_rate)
                if self.verbose:
                    print(f"[gauntlet] {name}: {win_rate:.3f} over {self.n_games} games")
        self._last_eval_step = self.num_timesteps
        self._eval_count += 1

    def _play_batch(self, name: str, make_opponents: Callable[[], list[Any]]) -> int:
        """Play all games against one opponent, in parallel batches."""
        del name
        envs: list[HokmEnv] = []
        results = [False] * self.n_games
        active: list[int] = []
        obs_by_env: dict[HokmEnv, Observation] = {}
        info_by_env: dict[HokmEnv, dict[str, Any]] = {}
        next_game = 0
        try:
            while next_game < self.n_games or active:
                while len(active) < self.batch_size and next_game < self.n_games:
                    env = self.eval_env_factory()
                    env.opponents = make_opponents()
                    obs, info = env.reset(seed=next_game)
                    envs.append(env)
                    active.append(next_game)
                    obs_by_env[env] = obs
                    info_by_env[env] = info
                    next_game += 1
                # One stacked forward pass for every live env.
                first = obs_by_env[envs[0]]
                stacked = {
                    "hand": np.stack([obs_by_env[env]["hand"] for env in envs]),
                    "seen": np.stack([obs_by_env[env]["seen"] for env in envs]),
                    "trick": np.stack([obs_by_env[env]["trick"] for env in envs]),
                    "trick_play": np.stack([obs_by_env[env]["trick_play"] for env in envs]),
                    "trump": np.stack([obs_by_env[env]["trump"] for env in envs]),
                    "phase": np.stack([obs_by_env[env]["phase"] for env in envs]),
                    "tricks_won": np.stack([obs_by_env[env]["tricks_won"] for env in envs]),
                    "game_points": np.stack([obs_by_env[env]["game_points"] for env in envs]),
                    "seat": np.stack([obs_by_env[env]["seat"] for env in envs]),
                }
                del first
                masks = np.stack(
                    [np.asarray(info_by_env[env]["action_mask"], dtype=bool) for env in envs]
                )
                actions, _ = self.model.predict(stacked, action_masks=masks, deterministic=True)  # type: ignore[call-arg]
                # Step each env; collect finished games.
                finished: list[int] = []
                for i, env in enumerate(envs):
                    obs, _reward, terminated, truncated, info = env.step(
                        np.int64(int(np.asarray(actions[i]).reshape(-1)[0]))
                    )
                    if terminated or truncated:
                        winner = env._engine.state.winner
                        results[active[i]] = winner == team_of(env.seat)
                        finished.append(i)
                    else:
                        obs_by_env[env] = obs
                        info_by_env[env] = info
                for i in reversed(finished):
                    envs[i].close()
                    envs.pop(i)
                    active.pop(i)
        finally:
            for env in envs:
                env.close()
        return sum(results)


def _snapshot_path_of(opponents: list[Any]) -> str | None:
    """Return the snapshot file the opponent list was built from, if any."""
    for opponent in opponents:
        if isinstance(opponent, SnapshotPolicy):
            return opponent.path_str
    return None
