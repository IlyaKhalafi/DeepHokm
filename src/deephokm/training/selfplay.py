"""The opponent pool for self-play training.

Snapshot policies are :class:`~deephokm.nn.policy.HokmMaskablePolicy`
instances saved from an earlier training state. Per episode the environment
draws one policy per opponent team (see :data:`P_LATEST`, :data:`P_POOL`,
:data:`P_GREEDY`, :data:`P_RANDOM` for the mix): predominantly the latest
snapshot or the retained pool, with a small share of the scripted
:class:`~deephokm.policies.greedy_policy.GreedyPolicy` baseline and a small
share of fresh random play.

Training runs the environments in forked worker processes, so the pool
cannot be shared as a Python object: the workers would keep the snapshot
list they were forked with and self-play would silently degenerate into
play against fixed random opponents. The snapshot directory on disk is the
shared channel instead — :class:`PoolOpponentProvider` rescans it inside the
worker, so snapshots written by the training process are picked up on the
next episode.
"""

from __future__ import annotations

import os
import pickle
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch as th

from deephokm.env.spaces import Observation
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.policies.base import HokmPolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.random_policy import RandomPolicy

P_LATEST = 0.30
P_POOL = 0.20
P_GREEDY = 0.50
P_RANDOM = 0.00

SNAPSHOT_GLOB = "snapshot_*.zip"

# Everything torch/zip deserialization raises for a file that is not (yet) a
# complete policy archive. Snapshots are written atomically, so this is a
# belt-and-braces guard rather than the expected path.
UNREADABLE_SNAPSHOT = (
    OSError,
    EOFError,
    ValueError,
    KeyError,
    RuntimeError,
    BadZipFile,
    pickle.UnpicklingError,
)


def _thread_env_unset() -> bool:
    """Return True unless the user pinned CPU thread counts via the env."""
    return not any(
        os.environ.get(var) for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS")
    )


@dataclass
class SnapshotInfo:
    """Metadata for one stored opponent snapshot.

    Attributes:
        path: Filesystem path of the saved policy zip.
        global_step: Training step at which the snapshot was taken.
    """

    path: Path
    global_step: int


def snapshot_step(path: Path) -> int:
    """Return the training step encoded in a snapshot filename.

    Snapshots are named ``snapshot_<zero-padded step>.zip``; the step is what
    orders the pool, so it is parsed rather than relying on mtime.
    """
    return int(path.stem.split("_")[-1])


class SelfPlayPool:
    """A bounded pool of opponent snapshots with weighted sampling.

    Attributes:
        capacity: Maximum number of snapshots retained (oldest evicted).
        rng: Seeded generator for opponent assignment.
    """

    def __init__(self, capacity: int = 10, seed: int = 0) -> None:
        """Create an empty pool.

        Args:
            capacity: Maximum snapshots kept.
            seed: Seed for the sampling RNG.
        """
        self.capacity = capacity
        self.rng = random.Random(seed)
        self._snapshots: deque[SnapshotInfo] = deque(maxlen=capacity)
        self._cache: dict[Path, SnapshotPolicy] = {}

    def __len__(self) -> int:
        """Return the number of stored snapshots."""
        return len(self._snapshots)

    @property
    def snapshots(self) -> list[SnapshotInfo]:
        """Return the stored snapshots in insertion order."""
        return list(self._snapshots)

    def add(self, path: Path, global_step: int) -> None:
        """Store a snapshot, evicting the oldest when at capacity."""
        self._snapshots.append(SnapshotInfo(path=path, global_step=global_step))
        # Evicted entries may still be cached; drop cache entries whose file
        # is no longer part of the pool.
        live = {info.path for info in self._snapshots}
        for cached in list(self._cache):
            if cached not in live:
                self._cache.pop(cached)

    def load(self, path: Path) -> SnapshotPolicy:
        """Load (and cache) a snapshot policy from disk."""
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        policy = SnapshotPolicy.from_file(path)
        self._cache[path] = policy
        return policy

    def latest(self) -> SnapshotInfo | None:
        """Return the most recently added snapshot, if any."""
        return self._snapshots[-1] if self._snapshots else None


class PoolOpponentProvider:
    """Per-episode opponent draw backed by a snapshot directory.

    The provider is handed to :class:`~deephokm.env.hokm_env.HokmEnv` as its
    ``opponent_provider`` and called once per ``reset()``. It rescans the
    snapshot directory (cheaply, and at most every ``refresh_every`` calls)
    so that snapshots written by the training process reach the forked
    workers, then draws one policy **per team**, not per seat: ``P_LATEST``
    the newest snapshot, ``P_POOL`` uniform over the retained pool,
    ``P_GREEDY`` the scripted, non-learning :class:`GreedyPolicy` baseline,
    ``P_RANDOM`` a fresh random policy. Both seats of a team share the same
    draw, so the learner's partner plays one coherent style for the whole
    episode instead of an independent per-seat coin flip -- with a naive
    per-seat draw, roughly 1 - (1 - P_RANDOM)^3 of tables would carry at
    least one uniformly-random seat among the other three, which teaches
    the learner to exploit noise rather than play alongside a consistent
    partner. ``GreedyPolicy`` is drawn during training, not just held back
    for evaluation: a policy that never sees disciplined, non-self-play
    opponents during training has no pressure to learn the tactics that
    beat one.

    Attributes:
        snapshot_dir: Directory the training process writes snapshots to.
        capacity: Number of most-recent snapshots kept in the pool.
        refresh_every: Episodes between directory rescans.
    """

    def __init__(
        self,
        snapshot_dir: Path,
        *,
        capacity: int = 10,
        seed: int = 0,
        refresh_every: int = 1,
    ) -> None:
        """Create the provider.

        Args:
            snapshot_dir: Directory holding ``snapshot_*.zip`` files.
            capacity: Pool size (newest ``capacity`` snapshots).
            seed: Seed for the draw RNG.
            refresh_every: Episodes between directory rescans (1 = every
                episode; a rescan is one ``listdir``).
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.capacity = capacity
        self.refresh_every = max(1, refresh_every)
        self.rng = random.Random(seed)
        self._seed = seed
        self._paths: list[Path] = []
        self._cache: dict[Path, SnapshotPolicy] = {}
        self._calls = 0

    def __getstate__(self) -> dict[str, object]:
        """Drop loaded policies before pickling into a worker process."""
        state = dict(self.__dict__)
        state["_cache"] = {}
        return state

    def reseed(self, seed: int) -> None:
        """Reseed the draw RNG (e.g. with a rank-derived seed per worker).

        A provider constructed once and handed to every ``SubprocVecEnv``
        worker is forked into each subprocess at the *same* RNG state, so
        every worker's very first draw (before any per-episode ``reset()``
        advances it) would otherwise pick the identical opponent mix. Called
        once per worker, right after the fork, to decorrelate that startup
        draw; per-episode draws are unaffected once training is underway.
        """
        self.rng = random.Random(seed)

    def refresh(self) -> list[Path]:
        """Rescan the snapshot directory and return the current pool paths."""
        try:
            found = sorted(self.snapshot_dir.glob(SNAPSHOT_GLOB), key=snapshot_step)
        except (OSError, ValueError):
            return self._paths
        self._paths = found[-self.capacity :]
        live = set(self._paths)
        for cached in list(self._cache):
            if cached not in live:
                self._cache.pop(cached)
        return self._paths

    def _load(self, path: Path) -> SnapshotPolicy | None:
        """Load and cache a snapshot, or return None if it is unreadable.

        A snapshot can briefly be unreadable while the training process is
        still writing it; the draw falls back to a random policy rather than
        crashing the worker.
        """
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        try:
            policy = SnapshotPolicy.from_file(
                path, deterministic=False, seed=self.rng.randrange(2**31)
            )
        except UNREADABLE_SNAPSHOT:
            return None
        self._cache[path] = policy
        return policy

    def __call__(self) -> list[HokmPolicy]:
        """Draw one policy per team, applied to both of that team's seats.

        Returns a 4-entry list indexed by seat (matching
        :class:`~deephokm.env.hokm_env.HokmEnv`'s ``opponents`` contract,
        which ignores the learner's own seat), but only 2 independent draws
        happen: seats {0, 2} share one, seats {1, 3} share the other.
        """
        if self._calls % self.refresh_every == 0:
            self.refresh()
        self._calls += 1
        team_a = self._draw_team()
        team_b = self._draw_team()
        return [team_a, team_b, team_a, team_b]

    def _draw_team(self) -> HokmPolicy:
        """Draw one team's shared policy from the weighted mix."""
        if self._paths:
            roll = self.rng.random()
            if roll < P_LATEST:
                policy = self._load(self._paths[-1])
                if policy is not None:
                    return policy
            elif roll < P_LATEST + P_POOL:
                policy = self._load(self.rng.choice(self._paths))
                if policy is not None:
                    return policy
            elif roll < P_LATEST + P_POOL + P_GREEDY:
                return GreedyPolicy()
        elif self.rng.random() < P_GREEDY:
            # No snapshots yet (run just started): still give GreedyPolicy
            # its share so training sees a disciplined opponent from step
            # zero, not only once the pool has something to sample.
            return GreedyPolicy()
        return RandomPolicy(self.rng.randrange(2**31))


class SnapshotPolicy:
    """A frozen HokmMaskablePolicy loaded from a checkpoint, acting greedily.

    Inference runs under ``torch.no_grad()`` on CPU with a single intra-op
    thread: benchmarks on this machine show CPU beats GPU for batch-1 calls
    (2.9 ms vs 13.4 ms) and extra threads only oversubscribe (160-core box,
    batch-1 tensors are latency-bound, not compute-bound).
    """

    def __init__(
        self,
        policy: HokmMaskablePolicy,
        path_str: str = "",
        *,
        deterministic: bool = True,
        seed: int | None = None,
    ) -> None:
        self.policy = policy
        self.policy.eval()
        self.path_str = path_str
        self.deterministic = deterministic
        self._seed = seed
        self._rng = random.Random(seed)

    @staticmethod
    def _pin_single_thread() -> None:
        """Limit torch CPU intra-op threads to one for this process.

        Only applied when the process has not chosen a thread count itself.
        """
        if th.get_num_threads() > 1 and _thread_env_unset():
            th.set_num_threads(1)

    @classmethod
    def from_file(
        cls, path: Path, *, deterministic: bool = True, seed: int | None = None
    ) -> SnapshotPolicy:
        """Load a saved policy zip into a snapshot.

        Args:
            path: The saved policy zip.
            deterministic: Whether the snapshot plays its argmax action.
                Evaluation wants the argmax; self-play opponents sample, so
                the learner meets varied lines instead of one frozen script.
            seed: Seed for the stochastic-sampling stream (see :meth:`act`);
                ignored when ``deterministic=True``, since argmax play has no
                randomness to seed.
        """
        cls._pin_single_thread()
        policy = HokmMaskablePolicy.load(str(path), device="cpu")
        return cls(policy, path_str=str(path), deterministic=deterministic, seed=seed)

    def reset(self, seed: int | None = None) -> None:
        """Restart the stochastic-sampling stream (optionally from a new seed).

        A deterministic (argmax) snapshot has no randomness to reset, so this
        is a no-op in that mode; a sampling snapshot's action stream is fully
        seeded by :meth:`act`'s scoped reseed below, so a seeded ``reset()``
        replays an episode against it exactly, matching every other
        :class:`~deephokm.policies.base.HokmPolicy` in this codebase.
        """
        self._rng = random.Random(self._seed if seed is None else seed)

    def act(self, observation: Observation, action_mask: np.ndarray) -> int:
        """Return the snapshot's action for the observation."""
        obs = {key: np.asarray(value)[None, ...] for key, value in observation.items()}
        mask = np.asarray(action_mask, dtype=bool)[None, ...]
        with th.no_grad():
            if self.deterministic:
                actions, _ = self.policy.predict(obs, action_masks=mask, deterministic=True)
            else:
                # predict(deterministic=False) samples from torch's global
                # RNG, which this policy does not own (the calling process
                # might be the training loop itself, mid-rollout). Save and
                # restore the global state around a per-instance reseed so
                # this snapshot's own play is reproducible under a seeded
                # env.reset() without perturbing anyone else's draws.
                global_state = th.random.get_rng_state()
                th.manual_seed(self._rng.randrange(2**31))
                try:
                    actions, _ = self.policy.predict(obs, action_masks=mask, deterministic=False)
                finally:
                    th.random.set_rng_state(global_state)
        return int(np.asarray(actions).reshape(-1)[0])


def opponent_probability_check() -> tuple[float, float, float, float]:
    """Return the (latest, pool, greedy, random) team-draw probabilities."""
    return P_LATEST, P_POOL, P_GREEDY, P_RANDOM
