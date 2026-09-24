"""Nominate with the numpy Q-network, confirm with a search sign test.

The network is fast but imperfect; the search is slow but trustworthy. This
policy spends the network's speed to shrink the search's work:

1. The Q-network ranks the legal actions (one numpy forward pass, ~6 ms).
2. Its top candidates that differ from :class:`GreedyPolicy`'s choice are
   proposed, best first.
3. Each proposal is checked against greedy's action over ``verify_samples``
   determinized worlds, and played only if it wins an exact one-sided sign
   test at ``max_p_value``.

Searching every legal action costs ``K x |legal|`` rollouts; checking ``M``
proposals costs ``K x 2M``, which at the measured mean of ~5 legal actions and
``M = 3`` is comparable per decision while concentrating the rollouts on the
actions worth arguing about. Proposing more than one candidate is deliberate:
the network's top choice matches the teacher's best on ~65% of decisive
decisions, but its top three contain it far more often, and the sign test is
exactly the mechanism that can tell which of the three is right.

Nothing here sees hidden cards. The network reads only the observation, and
the sampled worlds are guesses constrained by the voids inferred from public
play.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from deephokm.cards import NUM_CARDS
from deephokm.env.spaces import Observation, mask_for, observation_for
from deephokm.nn.features import build_features
from deephokm.nn.numpy_qnet import NumpyQNet, load_weights
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.legal_depth_search import _sign_test_p_value
from deephokm.policies.oracle_search import oracle_ceiling
from deephokm.policies.search import (
    VoidTracker,
    _clone_for_simulation,
    sample_determinized_hands,
)
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import NUM_SEATS, Phase, team_of

DEFAULT_VERIFY_SAMPLES = 192
DEFAULT_MAX_P_VALUE = 0.05
DEFAULT_TOP_M = 3


class NumpyHybridPolicy:
    """Greedy by default; network proposals confirmed by a sign test.

    Attributes:
        net: The numpy action-value network used to rank actions.
        verify_samples: Determinized worlds drawn per proposal.
        top_m: How many of the network's best actions to propose.
        max_p_value: One-sided sign-test threshold for accepting a proposal.
        greedy: The scripted baseline that proposals must beat.
        voids: Suit voids inferred from public play.
    """

    def __init__(
        self,
        weights: Path | dict[str, np.ndarray],
        *,
        verify_samples: int = DEFAULT_VERIFY_SAMPLES,
        top_m: int = DEFAULT_TOP_M,
        max_p_value: float = DEFAULT_MAX_P_VALUE,
        seed: int = 0,
    ) -> None:
        """Create the policy.

        Args:
            weights: A ``.npz`` path, or an already-loaded weight dict.
            verify_samples: Worlds sampled per proposal.
            top_m: Number of network proposals to consider.
            max_p_value: Sign-test threshold.
            seed: Seed for the world sampler.
        """
        params = load_weights(weights) if isinstance(weights, Path) else weights
        self.net = NumpyQNet(params)
        self.verify_samples = verify_samples
        self.top_m = top_m
        self.max_p_value = max_p_value
        self.greedy = GreedyPolicy()
        self.voids = VoidTracker()
        self._rng = random.Random(seed)

    def reset_hand(self) -> None:
        """Forget inferred voids; call at the start of every hand."""
        self.voids.reset()

    def observe(self, seat: int, card: int, led_suit: int | None) -> None:
        """Record one publicly played card."""
        self.voids.observe(seat, card, led_suit)

    def decide(self, engine: HokmEngine) -> int:
        """Choose the acting seat's action."""
        seat = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(seat)
        if len(legal) == 1:
            return legal[0]

        observation = observation_for(hands, seat, engine.state.game_points)
        mask = mask_for(legal)
        if hands.phase is not Phase.CARD_PLAY:
            # Trump declaration: the network has no trained signal here and
            # both the scripted baseline and the pure search defer to greedy.
            return self.greedy.act(observation, mask)

        baseline = self.greedy.act(observation, mask)
        for candidate in self._proposals(observation, mask, legal, baseline):
            if self._beats(engine, seat, candidate, baseline):
                return candidate
        return baseline

    def _proposals(
        self,
        observation: Observation,
        mask: np.ndarray,
        legal: list[int],
        baseline: int,
    ) -> list[int]:
        """The network's ``top_m`` legal actions, best first, minus greedy's."""
        planes, scalars = build_features(observation, mask)
        values = self.net(planes[None], scalars[None])[0]
        ranked = [int(legal[i]) for i in np.argsort(values[legal])[::-1]]
        return [a for a in ranked[: self.top_m] if a != baseline]

    def _beats(self, engine: HokmEngine, seat: int, candidate: int, baseline: int) -> bool:
        """Sign test: does ``candidate`` beat ``baseline`` across sampled worlds?"""
        hands = engine.state.hands
        own_hand = hands.hands[seat]
        seen = set(hands.played) | set(own_hand)
        unseen = [card for card in range(NUM_CARDS) if card not in seen]
        sizes = [len(hands.hands[s]) for s in range(NUM_SEATS)]
        team = team_of(seat)
        # One clone RNG per decision; clones only consume it for a simulated
        # redeal, into state that is discarded.
        clone_rng = random.Random()
        wins_candidate = wins_baseline = 0
        for _ in range(self.verify_samples):
            world = sample_determinized_hands(
                seat, own_hand, unseen, sizes, voids=self.voids.voids, rng=self._rng
            )
            value_c = self._value(
                engine, seat, candidate, team=team, world=world, clone_rng=clone_rng
            )
            value_b = self._value(
                engine, seat, baseline, team=team, world=world, clone_rng=clone_rng
            )
            if value_c > value_b:
                wins_candidate += 1
            elif value_b > value_c:
                wins_baseline += 1
        discordant = wins_candidate + wins_baseline
        return _sign_test_p_value(wins_candidate, discordant) <= self.max_p_value

    def _value(
        self,
        engine: HokmEngine,
        seat: int,
        action: int,
        *,
        team: int,
        world: list[list[int]],
        clone_rng: random.Random,
    ) -> float:
        """Outcome of ``action`` in one sampled world."""
        clone = _clone_for_simulation(engine, seat, world, rng=clone_rng)
        outcome = clone.apply_action(action, seat=seat)
        if outcome.hand_complete:
            assert outcome.hand_winner_team is not None
            return 1.0 if outcome.hand_winner_team == team else -1.0
        return oracle_ceiling(clone, team, depth=1, rng=clone_rng)


__all__ = ["DEFAULT_TOP_M", "DEFAULT_VERIFY_SAMPLES", "NumpyHybridPolicy"]
