"""The HokmPolicy protocol implemented by every opponent of the learner."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from deephokm.env.spaces import Observation

ObservationDict = Observation


@runtime_checkable
class HokmPolicy(Protocol):
    """An agent that acts on Hokm observations.

    Implementations must only use the observation and action mask passed to
    ``act`` — hidden information must never leak through any other channel.
    Stateful policies (e.g. seeded random policies) should implement
    ``reset(seed)`` so the environment can restore per-episode determinism.
    """

    def act(self, observation: ObservationDict, action_mask: np.ndarray) -> int:
        """Return a legal action id for the observation.

        Args:
            observation: The acting player's observation dict.
            action_mask: Boolean array of shape ``(56,)``; only actions with a
                true entry may be returned.

        Returns:
            An action id in ``0..55`` that must be legal under the mask.
        """
        ...

    def reset(self, seed: int | None = None) -> None:
        """Reset any per-episode state; called by the environment on reset.

        Args:
            seed: Seed derived from the environment's reset seed; ``None``
                continues the policy's own stream.
        """
        ...
