"""Training callbacks: self-play snapshots and evaluation gauntlet logging.

:class:`SelfPlayCallback` periodically saves the current policy into the
snapshot directory that the worker environments read (see
:class:`~deephokm.training.selfplay.PoolOpponentProvider`).
:class:`GauntletCallback` plays the current policy against a fixed set of
opponents and writes the win rates to the TensorBoard log.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from deephokm.training.gauntlet_workers import (
    OpponentSpec,
    run_gauntlet_shards,
    run_team_gauntlet_shards,
)
from deephokm.training.selfplay import SNAPSHOT_GLOB, snapshot_step


class SelfPlayCallback(BaseCallback):
    """Snapshot the current policy into the opponent pool on a step cadence.

    Snapshots are written atomically (temporary name, then rename) because
    the environment worker processes read the same directory concurrently
    and must never load a half-written zip. Files beyond ``capacity`` are
    deleted so the directory the workers scan is exactly the live pool.

    Attributes:
        save_freq: Environment steps between snapshots.
        save_path: Directory the snapshot zips are written to.
        capacity: Number of snapshots retained on disk.
    """

    def __init__(
        self,
        *,
        save_freq: int,
        save_path: Path,
        capacity: int = 10,
        verbose: int = 0,
    ) -> None:
        """Create the callback.

        Args:
            save_freq: Environment steps between snapshots.
            save_path: Directory for snapshot files.
            capacity: Snapshots kept on disk (oldest deleted beyond it).
            verbose: SB3 verbosity.
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.capacity = capacity
        self._last_snapshot_step = 0

    def _on_step(self) -> bool:
        """Snapshot when the step cadence elapses."""
        if self.num_timesteps - self._last_snapshot_step >= self.save_freq:
            self.snapshot()
        return True

    def snapshot(self) -> Path:
        """Save the current policy into the pool directory atomically."""
        self.save_path.mkdir(parents=True, exist_ok=True)
        path = self.save_path / f"snapshot_{self.num_timesteps:012d}.zip"
        staging = path.with_suffix(".zip.partial")
        # Save the policy only (the full model carries the optimizer state,
        # which opponents never need and which triples the snapshot size).
        self.model.policy.save(str(staging))
        staging.replace(path)
        self._prune()
        self._last_snapshot_step = self.num_timesteps
        self.logger.record("selfplay/pool_size", self._pool_size())
        if self.verbose:
            print(f"[selfplay] snapshot at step {self.num_timesteps} -> {path.name}")
        return path

    def _pool_size(self) -> int:
        """Return the number of snapshots the workers currently draw from."""
        return min(len(list(self.save_path.glob(SNAPSHOT_GLOB))), self.capacity)

    def _prune(self) -> None:
        """Delete all but the newest ``capacity`` snapshots.

        The earliest snapshot is kept regardless: the gauntlet measures
        progress against it, so it is a fixed reference rather than pool
        membership.
        """
        existing = sorted(self.save_path.glob(SNAPSHOT_GLOB), key=snapshot_step)
        if len(existing) <= self.capacity:
            return
        keep = set(existing[-self.capacity :]) | {existing[0]}
        for path in existing:
            if path not in keep:
                path.unlink(missing_ok=True)


class RollingCheckpointCallback(CheckpointCallback):
    """``CheckpointCallback`` that retains only the newest ``keep`` archives.

    A full model archive carries the optimizer state, so an unbounded series
    fills the run directory with tens of gigabytes over a long run for no
    benefit — only the newest few are ever resumed from.

    Attributes:
        keep: Number of checkpoint archives retained.
    """

    def __init__(self, *args: Any, keep: int = 5, **kwargs: Any) -> None:
        """Create the callback.

        Args:
            *args: Forwarded to ``CheckpointCallback``.
            keep: Number of newest checkpoints to retain.
            **kwargs: Forwarded to ``CheckpointCallback``.
        """
        super().__init__(*args, **kwargs)
        self.keep = keep

    def _on_step(self) -> bool:
        """Save on cadence, then drop everything but the newest ``keep``."""
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            self._prune()
        return result

    def _prune(self) -> None:
        """Delete all but the newest ``keep`` checkpoint archives."""
        directory = Path(self.save_path)
        existing = sorted(
            directory.glob(f"{self.name_prefix}_*_steps.zip"),
            key=lambda path: int(path.stem.rsplit("_", 2)[-2]),
        )
        for path in existing[: -self.keep] if self.keep > 0 else existing:
            path.unlink(missing_ok=True)


class GauntletCallback(BaseCallback):
    """Evaluate the policy against fixed opponents and log win rates.

    Each round saves the live model once, then plays every gauntlet opponent
    across worker processes, which cuts the wall time of a round to a
    fraction of what it costs inside the training loop. Training does block
    for the round: the point is that the round is short, not that it is
    asynchronous.

    Attributes:
        opponents: Gauntlet name -> opponent specification.
        n_games: Games per opponent per round.
        eval_freq: Environment steps between rounds.
    """

    def __init__(
        self,
        *,
        opponents: dict[str, OpponentSpec],
        n_games: int,
        eval_freq: int,
        n_workers: int = 8,
        verbose: int = 0,
    ) -> None:
        """Create the callback.

        Args:
            opponents: Gauntlet name -> opponent specification. Entries whose
                spec resolves to ``None`` at round time are skipped.
            n_games: Games per gauntlet opponent per round.
            eval_freq: Environment steps between evaluation rounds.
            n_workers: Processes sharing a gauntlet round.
            verbose: SB3 verbosity.
        """
        super().__init__(verbose)
        self.opponents = opponents
        self.n_games = n_games
        self.eval_freq = eval_freq
        self.n_workers = n_workers
        self._last_eval_step = 0
        self._eval_count = 0
        self._snapshot_dir: Path | None = None

    def bind_snapshot_dir(self, snapshot_dir: Path) -> None:
        """Point the callback at the live snapshot directory.

        The first and latest snapshot rungs are resolved per round from this
        directory, so they track the pool the training process is writing.
        """
        self._snapshot_dir = snapshot_dir

    def _on_step(self) -> bool:
        """Run a gauntlet round when the step cadence elapses."""
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self.run_gauntlet()
        return True

    def _on_training_end(self) -> None:
        """Run a final round so the log ends on fresh numbers."""
        self.run_gauntlet()

    def _resolve(self, name: str, spec: OpponentSpec) -> OpponentSpec | None:
        """Resolve a snapshot rung against the live snapshot directory."""
        if spec.kind != "snapshot":
            return spec
        if self._snapshot_dir is None:
            return None
        found = sorted(self._snapshot_dir.glob(SNAPSHOT_GLOB), key=snapshot_step)
        if not found:
            return None
        chosen = found[0] if name == "first_snapshot" else found[-1]
        return OpponentSpec(kind="snapshot", path=str(chosen))

    def run_gauntlet(self) -> dict[str, float]:
        """Play every gauntlet opponent and log win rates."""
        rates: dict[str, float] = {}
        with tempfile.TemporaryDirectory() as tmp:
            model_path = str(Path(tmp) / "eval_model.zip")
            self.model.save(model_path)
            for name, spec in self.opponents.items():
                resolved = self._resolve(name, spec)
                if resolved is None:
                    continue
                wins = run_gauntlet_shards(
                    model_path=model_path,
                    spec=resolved,
                    n_games=self.n_games,
                    n_workers=self.n_workers,
                    seed=self._eval_count,
                )
                win_rate = wins / self.n_games
                rates[name] = win_rate
                self.logger.record(f"gauntlet/{name}", win_rate)
                if self.verbose:
                    print(f"[gauntlet] {name}: {win_rate:.3f} over {self.n_games} games")
        self._last_eval_step = self.num_timesteps
        self._eval_count += 1
        return rates


class TeamGauntletCallback(BaseCallback):
    """Log the model's win rate controlling BOTH team seats against a fixed
    opponent (see :func:`~deephokm.training.gauntlet_workers.run_team_gauntlet_shards`).

    This is the metric a dedicated best-response fine-tune against a single
    fixed opponent actually cares about: :class:`GauntletCallback` plays the
    model in one seat only, with the opponent policy also filling the
    learner's own partner seat, which understates a policy trained to control
    a whole team.

    Attributes:
        opponent: The fixed opposing-team opponent specification.
        n_games: Games per round.
        eval_freq: Environment steps between rounds.
    """

    def __init__(
        self,
        *,
        opponent: OpponentSpec,
        name: str,
        n_games: int,
        eval_freq: int,
        n_workers: int = 8,
        verbose: int = 0,
    ) -> None:
        """Create the callback.

        Args:
            opponent: Opponent specification for the opposing team's two seats.
            name: TensorBoard tag suffix (``gauntlet/team_vs_<name>``).
            n_games: Games per round.
            eval_freq: Environment steps between rounds.
            n_workers: Processes sharing a round.
            verbose: SB3 verbosity.

        Raises:
            ValueError: If ``n_games`` is not positive (a zero-game round
                would divide by zero when computing the win rate).
        """
        if n_games <= 0:
            raise ValueError(f"n_games must be positive, got {n_games}")
        super().__init__(verbose)
        self.opponent = opponent
        self.name = name
        self.n_games = n_games
        self.eval_freq = eval_freq
        self.n_workers = n_workers
        self._last_eval_step = 0
        self._eval_count = 0

    def _on_step(self) -> bool:
        """Run a round when the step cadence elapses."""
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self.run_gauntlet()
        return True

    def _on_training_end(self) -> None:
        """Run a final round so the log ends on fresh numbers."""
        self.run_gauntlet()

    def run_gauntlet(self) -> float:
        """Play one round and log the team win rate."""
        with tempfile.TemporaryDirectory() as tmp:
            model_path = str(Path(tmp) / "eval_model.zip")
            self.model.save(model_path)
            wins = run_team_gauntlet_shards(
                model_path=model_path,
                spec=self.opponent,
                n_games=self.n_games,
                n_workers=self.n_workers,
                seed=self._eval_count,
            )
        win_rate = wins / self.n_games
        self.logger.record(f"gauntlet/team_vs_{self.name}", win_rate)
        if self.verbose:
            print(f"[team-gauntlet] vs {self.name}: {win_rate:.3f} over {self.n_games} games")
        self._last_eval_step = self.num_timesteps
        self._eval_count += 1
        return win_rate
