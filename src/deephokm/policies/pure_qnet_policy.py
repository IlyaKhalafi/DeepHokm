"""The Q-network alone, with no live search: argmax over legal actions.

Every other served policy in this package pairs the network with a
determinized search that samples hidden worlds and verifies each candidate
by rollout -- that pairing is what reaches 0.855 against a greedy opposing
team (see :mod:`deephokm.policies.numpy_hybrid`), but it costs seconds per
decision because it runs real rollouts at play time. Search is also how the
teacher labels this network trained on were generated in the first place
(:mod:`deephokm.policies.legal_depth_search` at K up to 6144) -- that use of
search is entirely offline, before this network ever plays a card.

This class is the other half of that split: a serving mode with no search
at all, just the trained network's own forward pass. Measured at 0.6625
against a greedy opposing team (80 held-out matches, 95% CI
[0.559, 0.766]), at a median 18 ms per decision. Weaker than the search-paired
policy by design -- section 19 of ``docs/METHODS_AND_RESULTS.md`` measures why
a distilled student cannot exceed the search it trained on -- but roughly
200x faster, which matters when K is raised for strength in the offline
teacher rather than kept low for a live decision budget.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deephokm.env.spaces import Observation, mask_for, observation_for
from deephokm.nn.feature_contract import resolve_feature_mode
from deephokm.nn.features import build_features
from deephokm.nn.numpy_qnet import NumpyQNet, load_weights
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.rules.engine import HokmEngine
from deephokm.rules.legality import TRUMP_ACTION_OFFSET
from deephokm.rules.state import Phase


class PureQNetPolicy:
    """Argmax over the network's legal-action values; no rollouts.

    Attributes:
        net: The numpy action-value network.
        greedy: Trump-call fallback -- the network has no trained signal at
            the trump decision point, exactly as in
            :class:`~deephokm.policies.numpy_hybrid.NumpyHybridPolicy`.
    """

    def __init__(
        self, weights: Path | dict[str, np.ndarray], *, feature_mode: str | None = None
    ) -> None:
        """Create the policy.

        Args:
            weights: A ``.npz`` path, or an already-loaded weight dict.
        """
        params = load_weights(weights) if isinstance(weights, Path) else weights
        self.net = NumpyQNet(params)
        self.feature_mode = resolve_feature_mode(weights, self.net.input_planes, feature_mode)
        self.greedy = GreedyPolicy()

    def reset_hand(self) -> None:
        """No-op: this policy tracks no per-hand state.

        Present so callers written against the search policies (which do
        track inferred voids) can treat every served policy identically.
        """

    def observe(self, seat: int, card: int, led_suit: int | None) -> None:
        """No-op, for the same reason as :meth:`reset_hand`."""

    def close(self) -> None:
        """No-op: this policy owns no process pool to release."""

    def decide(self, engine: HokmEngine) -> int:
        """Choose the acting seat's action.

        Args:
            engine: The live engine; read-only here (no simulated clones).
        """
        seat = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(seat)
        if len(legal) == 1:
            return legal[0]
        observation = observation_for(hands, seat, engine.state.game_points)
        mask = mask_for(legal)
        if hands.phase is not Phase.CARD_PLAY:
            return self.greedy.act(observation, mask)
        planes, scalars = build_features(observation, mask, feature_mode=self.feature_mode)
        values = self.net(planes[None], scalars[None])[0]
        return max(legal, key=lambda a: values[a])

    def act(self, observation: Observation, action_mask: np.ndarray) -> int:
        """``HokmPolicy`` entry point for direct (non-search-aware) use.

        Needs nothing from ``engine`` history: it reads only the
        observation and mask, like
        :class:`~deephokm.policies.greedy_policy.GreedyPolicy`. Phase is
        read off the mask the same way that class does: the trump-call
        actions are a fixed, disjoint numeric range, so a legal action in
        that range means this is the trump decision, not a card play.
        """
        legal = np.flatnonzero(action_mask)
        if legal.size == 1:
            return int(legal[0])
        if legal[0] >= TRUMP_ACTION_OFFSET:
            return self.greedy.act(observation, action_mask)
        planes, scalars = build_features(observation, action_mask, feature_mode=self.feature_mode)
        values = self.net(planes[None], scalars[None])[0]
        return int(max(legal, key=lambda a: values[a]))


__all__ = ["PureQNetPolicy"]
