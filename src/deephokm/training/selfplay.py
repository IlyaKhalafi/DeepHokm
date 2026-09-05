"""The opponent pool for self-play training.

Snapshot policies are :class:`~deephokm.nn.policy.HokmMaskablePolicy`
instances saved from an earlier training state. Per episode the environment
draws which policy plays each opponent seat: 60% the latest snapshot, 30%
uniform over the pool, 10% the random policy.

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
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.state import NUM_SEATS

P_LATEST = 0.6
P_POOL = 0.3
P_RANDOM = 0.1

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
    workers, then draws one policy per seat: ``P_LATEST`` the newest
    snapshot, ``P_POOL`` uniform over the retained pool, ``P_RANDOM`` a fresh
    random policy. Drawing per seat rather than per episode means the learner
    meets mixed-strength tables, which is what keeps the partner seat from
    co-adapting to a single opponent generation.

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
        self._paths: list[Path] = []
        self._cache: dict[Path, SnapshotPolicy] = {}
        self._calls = 0

    def __getstate__(self) -> dict[str, object]:
        """Drop loaded policies before pickling into a worker process."""
        state = dict(self.__dict__)
        state["_cache"] = {}
        return state

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
            policy = SnapshotPolicy.from_file(path, deterministic=False)
        except UNREADABLE_SNAPSHOT:
            return None
        self._cache[path] = policy
        return policy

    def __call__(self) -> list[HokmPolicy]:
        """Draw one policy per seat for the next episode."""
        if self._calls % self.refresh_every == 0:
            self.refresh()
        self._calls += 1
        return [self._draw() for _ in range(NUM_SEATS)]

    def _draw(self) -> HokmPolicy:
        """Draw a single seat's policy from the weighted mix."""
        if self._paths:
            roll = self.rng.random()
            path = None
            if roll < P_LATEST:
                path = self._paths[-1]
            elif roll < P_LATEST + P_POOL:
                path = self.rng.choice(self._paths)
            if path is not None:
                policy = self._load(path)
                if policy is not None:
                    return policy
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
    ) -> None:
        self.policy = policy
        self.policy.eval()
        self.path_str = path_str
        self.deterministic = deterministic

    @staticmethod
    def _pin_single_thread() -> None:
        """Limit torch CPU intra-op threads to one for this process.

        Only applied when the process has not chosen a thread count itself.
        """
        if th.get_num_threads() > 1 and _thread_env_unset():
            th.set_num_threads(1)

    @classmethod
    def from_file(cls, path: Path, *, deterministic: bool = True) -> SnapshotPolicy:
        """Load a saved policy zip into a snapshot.

        Args:
            path: The saved policy zip.
            deterministic: Whether the snapshot plays its argmax action.
                Evaluation wants the argmax; self-play opponents sample, so
                the learner meets varied lines instead of one frozen script.
        """
        cls._pin_single_thread()
        policy = HokmMaskablePolicy.load(str(path), device="cpu")
        return cls(policy, path_str=str(path), deterministic=deterministic)

    def reset(self, seed: int | None = None) -> None:
        """No-op: the snapshot policy is stateless across episodes."""

    def act(self, observation: Observation, action_mask: np.ndarray) -> int:
        """Return the snapshot's action for the observation."""
        obs = {key: np.asarray(value)[None, ...] for key, value in observation.items()}
        mask = np.asarray(action_mask, dtype=bool)[None, ...]
        with th.no_grad():
            actions, _ = self.policy.predict(
                obs, action_masks=mask, deterministic=self.deterministic
            )
        return int(np.asarray(actions).reshape(-1)[0])


def opponent_probability_check() -> tuple[float, float, float]:
    """Return the (latest, pool, random) sampling probabilities."""
    return P_LATEST, P_POOL, P_RANDOM
