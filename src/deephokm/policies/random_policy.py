"""A uniform random legal-action policy, the weakest baseline."""

from __future__ import annotations

import random

import numpy as np

from deephokm.env.spaces import Observation


class RandomPolicy:
    """Chooses uniformly among the legal actions.

    The generator is seeded per policy instance so evaluation runs are
    reproducible; instances are cheap to construct per worker.

    Attributes:
        rng: The policy's random generator.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        """Restart the policy's stream (optionally from a new seed)."""
        self.rng = random.Random(self._seed if seed is None else seed)

    def act(self, observation: Observation, action_mask: np.ndarray) -> int:
        """Return a uniformly random legal action."""
        legal = np.flatnonzero(action_mask)
        if legal.size == 0:
            raise ValueError("action mask has no legal actions")
        return int(self.rng.choice(legal))
