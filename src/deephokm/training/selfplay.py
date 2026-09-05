"""The opponent pool for self-play training.

Snapshot policies are :class:`~deephokm.nn.policy.HokmMaskablePolicy`
instances (a policy saved from an earlier training state). Per episode the
environment samples which snapshot plays the three opponent seats:
60% the latest snapshot, 30% uniform over the pool, 10% the random policy.
"""

from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th

from deephokm.env.spaces import Observation
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.policies.base import HokmPolicy
from deephokm.policies.random_policy import RandomPolicy

P_LATEST = 0.6
P_POOL = 0.3
P_RANDOM = 0.1


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

    def sample_opponents(
        self,
        observation_space: object,
        action_space: object,
        *,
        n_seats: int = 4,
    ) -> list[HokmPolicy]:
        """Return opponent policies for one episode (one per seat, 4 entries).

        The mix per seat is drawn independently: 60% latest snapshot, 30%
        uniform over the pool, 10% uniform random policy. The learner's own
        entry is ignored by the environment.
        """
        opponents: list[HokmPolicy] = []
        for _seat in range(n_seats):
            roll = self.rng.random()
            latest = self.latest()
            if self._snapshots and roll < P_LATEST and latest is not None:
                opponents.append(self.load(latest.path))
            elif self._snapshots and roll < P_LATEST + P_POOL:
                info = self.rng.choice(self._snapshots)
                opponents.append(self.load(info.path))
            else:
                opponents.append(RandomPolicy(self.rng.randrange(2**31)))
        return opponents


class SnapshotPolicy:
    """A frozen HokmMaskablePolicy loaded from a checkpoint, acting greedily.

    Inference runs under ``torch.no_grad()`` on CPU with a single intra-op
    thread: benchmarks on this machine show CPU beats GPU for batch-1 calls
    (2.9 ms vs 13.4 ms) and extra threads only oversubscribe (160-core box,
    batch-1 tensors are latency-bound, not compute-bound).
    """

    def __init__(self, policy: HokmMaskablePolicy) -> None:
        self.policy = policy
        self.policy.eval()

    @staticmethod
    def _pin_single_thread() -> None:
        """Limit torch CPU intra-op threads to one for this process.

        Only applied when the process has not chosen a thread count itself.
        """
        if th.get_num_threads() > 1 and _thread_env_unset():
            th.set_num_threads(1)

    @classmethod
    def from_file(cls, path: Path) -> SnapshotPolicy:
        """Load a saved policy zip into a snapshot."""
        cls._pin_single_thread()
        policy = HokmMaskablePolicy.load(str(path), device="cpu")
        return cls(policy)

    def reset(self, seed: int | None = None) -> None:
        """No-op: the snapshot policy is stateless across episodes."""

    def act(self, observation: Observation, action_mask: np.ndarray) -> int:
        """Return the snapshot's action for the observation."""
        obs = {key: np.asarray(value)[None, ...] for key, value in observation.items()}
        mask = np.asarray(action_mask, dtype=bool)[None, ...]
        with th.no_grad():
            actions, _ = self.policy.predict(obs, action_masks=mask, deterministic=True)
        return int(np.asarray(actions).reshape(-1)[0])


def opponent_probability_check() -> tuple[float, float, float]:
    """Return the (latest, pool, random) sampling probabilities."""
    return P_LATEST, P_POOL, P_RANDOM
