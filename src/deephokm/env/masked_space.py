"""An action space whose ``sample()`` only returns currently-legal actions.

Gymnasium's env checker and other generic tooling call
``env.action_space.sample()`` without consulting action masks. For a
masked-action environment that produces illegal actions, so the environment
exposes this space: sampling is drawn from the live legal-action set while
``contains`` still accepts every action id (legality is a property of the
state, not the space).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

LegalProvider = Callable[[], list[int]]


class MaskedDiscrete(spaces.Discrete[np.integer]):
    """A :class:`~gymnasium.spaces.Discrete` with state-aware sampling.

    Attributes:
        legal_provider: Zero-argument callable returning the list of currently
            legal action ids.
    """

    def __init__(self, n: int, legal_provider: LegalProvider) -> None:
        """Create the space.

        Args:
            n: Number of actions.
            legal_provider: Callable returning the current legal action ids;
                called on every ``sample()``.
        """
        super().__init__(n)
        self.legal_provider: LegalProvider | None = legal_provider

    def sample(
        self,
        mask: NDArray[np.int8] | None = None,
        probability: NDArray[np.int8] | None = None,
    ) -> np.integer:
        """Return a uniformly random action among the currently legal ones.

        Args:
            mask: Ignored; legality comes from the live provider.
            probability: Ignored; sampling is uniform over legal actions.
        """
        provider = self.legal_provider
        if provider is None:
            raise RuntimeError(
                "MaskedDiscrete lost its legal-action provider (unpickled?); "
                "re-attach one before sampling"
            )
        legal = list(provider())
        if not legal:
            # Terminal state: no legal action exists; return any well-formed id
            # so generic tooling keeps working.
            return super().sample()
        chosen: np.integer = np.int64(legal[int(self.np_random.integers(0, len(legal)))])
        return chosen

    def __getstate__(self) -> dict[str, Any]:
        """Drop the env-bound provider before pickling (SubprocVecEnv forks)."""
        state = self.__dict__.copy()
        state["legal_provider"] = None
        return state

    def __setstate__(self, state: Iterable[tuple[str, Any]] | Mapping[str, Any]) -> None:
        """Restore pickled state; sampling falls back to plain Discrete."""
        if isinstance(state, Mapping):
            self.__dict__.update(state)
        else:
            self.__dict__.update(dict(state))
        self.legal_provider = None
