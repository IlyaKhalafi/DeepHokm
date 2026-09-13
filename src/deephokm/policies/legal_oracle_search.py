"""A legal (information-respecting) bounded-depth search policy.

The clairvoyant oracle (:mod:`deephokm.policies.oracle_search`) found a
real, large gap between GreedyPolicy's actual hand win rate and what a
FULLY-INFORMED team could force within a bounded lookahead: at depth=6,
starting right after the trump call, 0.755 vs. 0.525 (0.645 vs. 0.525 at
the much cheaper depth=2) -- but that used the real hidden hands, an
oracle no real player or policy could have.

This module answers the actual, deployable question: how much of that gap
survives when the search must guess the hidden hands instead of seeing
them? The simplest possible legal policy -- and the natural K=1 starting
point of the K-determinization sweep recommended after the oracle
result -- samples a *single* determinized world consistent with public
information (:func:`deephokm.policies.search.sample_determinized_hands`,
the same rejection-sampling/max-flow-fallback sampler already reviewed
three times for :mod:`counterfactual_search`), pretends it is the true
deal, and runs the exact same bounded-depth oracle search
(:func:`deephokm.policies.oracle_search.oracle_best_action`) against it --
then executes the resulting action in the real, hidden game.

This is a real, legal, randomized policy: at no point does it read another
seat's actual hidden cards (only its own hand and public information --
sizes, voids, played cards -- ever inform the determinization), and the
action it commits to is actually played in the real hidden deal, not
merely scored in a hypothetical one. It is not claimed to be optimal --
the single sampled world can simply be wrong, and (per the design
consult behind this module) its own internal continuation values still
assume that one guessed world is certain, which is not itself a fully
information-set-correct policy -- but its end-to-end win rate, measured by
actually playing hands with it, is a real number regardless of any flaw in
how it arrived at its choices.

Scope, like the sibling modules: card-play decisions only; the trump call
defers to :class:`GreedyPolicy` directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from deephokm.cards import NUM_CARDS, NUM_RANKS
from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies.counterfactual_search import MAX_ROLLOUT_PLIES_DEFAULT
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.oracle_search import DEPTH_DEFAULT, oracle_best_action
from deephokm.policies.search import VoidTracker, _clone_for_simulation, sample_determinized_hands
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import NUM_SEATS, Phase, team_of


@dataclass
class LegalOracleSearchPolicy:
    """K=1 single-determinization legal search.

    Attributes:
        depth: Forwarded to :func:`oracle_best_action`.
        max_rollout_plies: Forwarded to :func:`oracle_best_action`.
        seed: Seeds this policy's own determinization RNG.
    """

    depth: int = DEPTH_DEFAULT
    max_rollout_plies: int = MAX_ROLLOUT_PLIES_DEFAULT
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.voids = VoidTracker()
        self.greedy = GreedyPolicy()

    def reset_hand(self) -> None:
        """Clear void state; call at the start of every new hand."""
        self.voids.reset()

    def observe(self, seat: int, card: int, led_suit: int | None) -> None:
        """Record one publicly-played card from anywhere at the table."""
        self.voids.observe(seat, card, led_suit)

    def decide(self, engine: HokmEngine) -> int:
        """Choose the acting seat's next action, searching only card plays.

        Args:
            engine: The real, live match engine (read, never mutated).

        Returns:
            A legal action id for the engine's current seat.
        """
        root_seat = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(root_seat)
        if len(legal) <= 1:
            return legal[0]
        if hands.phase is not Phase.CARD_PLAY:
            # Scope: the trump call defers to GreedyPolicy directly.
            obs = observation_for(hands, root_seat, engine.state.game_points)
            return self.greedy.act(obs, mask_for(legal))

        own_hand = hands.hands[root_seat]
        seen = set(hands.played) | set(own_hand)
        unseen_pool = [c for c in range(NUM_CARDS) if c not in seen]
        remaining_sizes = [len(hands.hands[s]) for s in range(NUM_SEATS)]
        sampled_hands = sample_determinized_hands(
            root_seat,
            own_hand,
            unseen_pool,
            remaining_sizes,
            voids=self.voids.voids,
            rng=self._rng,
        )
        determinized = _clone_for_simulation(engine, root_seat, sampled_hands)
        team = team_of(root_seat)
        return oracle_best_action(
            determinized, team, depth=self.depth, max_rollout_plies=self.max_rollout_plies
        )


@dataclass
class HandResult:
    """One played hand's outcome, for aggregate reporting."""

    won: bool
    team_is_hakem: bool


def play_hand(
    seed: int,
    controlled_team: int,
    *,
    depth: int = DEPTH_DEFAULT,
    search_seed: int = 0,
    max_rollout_plies: int = MAX_ROLLOUT_PLIES_DEFAULT,
) -> HandResult:
    """Play a single seeded match's first hand; ``controlled_team`` uses
    :class:`LegalOracleSearchPolicy`, the other team plays :class:`GreedyPolicy`.

    Mirrors :func:`deephokm.policies.counterfactual_search.play_match`'s
    structure (real void observation fed to the search policy after every
    ply, reset at the start of the hand -- here always just the first hand,
    since this module's evaluation only ever samples one hand per match
    seed, matching ``scripts/oracle_hand_ceiling.py``'s regime for a fair,
    paired comparison against the oracle numbers).
    """
    if controlled_team not in (0, 1):
        raise ValueError(f"controlled_team must be 0 or 1, got {controlled_team}")
    engine = HokmEngine()
    engine.start_match(seed=seed)
    controlled_seats = {controlled_team, controlled_team + 2}
    search = LegalOracleSearchPolicy(
        depth=depth, max_rollout_plies=max_rollout_plies, seed=search_seed
    )
    greedy = GreedyPolicy()

    while engine.state.hands.phase is Phase.TRUMP_CALL:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        obs = observation_for(engine.state.hands, seat, engine.state.game_points)
        engine.apply_action(greedy.act(obs, mask_for(legal)), seat=seat)

    team_is_hakem = team_of(engine.state.hands.hakem) == controlled_team

    while True:
        seat = engine.current_seat()
        hands = engine.state.hands
        if seat in controlled_seats:
            action = search.decide(engine)
        else:
            legal = engine.legal_actions(seat)
            obs = observation_for(hands, seat, engine.state.game_points)
            action = greedy.act(obs, mask_for(legal))
        led_suit = None
        if hands.phase is Phase.CARD_PLAY and hands.current_trick:
            led_suit = hands.current_trick[0][1] // NUM_RANKS
        outcome = engine.apply_action(action, seat=seat)
        if outcome.card is not None:
            search.observe(seat, outcome.card, led_suit)
        if outcome.hand_complete:
            assert outcome.hand_winner_team is not None
            return HandResult(
                won=outcome.hand_winner_team == controlled_team, team_is_hakem=team_is_hakem
            )
