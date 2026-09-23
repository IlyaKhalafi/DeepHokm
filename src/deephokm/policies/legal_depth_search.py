"""A legal, depth-D, K-sample search with statistical caution.

Two things were learned earlier this session, each in isolation:

- The clairvoyant oracle (:mod:`oracle_search`) found a real, large gap
  between GreedyPolicy's actual hand win rate and what a fully-informed
  team could force within a bounded depth-D lookahead over its own future
  decisions -- the gap grows with D (0.12 at D=2, 0.23 at D=6, in the
  "team 0, first hand" regime).
- A naive K=1 legal version (:mod:`legal_oracle_search`: sample ONE
  determinized world, trust its depth-D analysis completely, act on it)
  was not merely unhelpful but actively harmful (0.19 hand win rate vs. a
  ~0.48-0.53 do-nothing baseline, at BOTH D=0 and D=2 -- ruling out
  "assumed future coordination" as the cause). The real cause, confirmed
  by comparing against the already-reviewed depth-1 K-many
  ``CounterfactualSearchPolicy`` run in the identical regime (0.48, right
  at baseline): committing hard to one near-certainly-wrong random guess
  about the entire hidden deal, with no statistical caution and no
  comparison against GreedyPolicy's own action, deviates from a safe,
  robust heuristic far too often, on unreliable grounds.

This module combines the two lessons: ``CounterfactualSearchPolicy``'s
already-reviewed statistical machinery (average over K sampled
determinized worlds, then only switch away from GreedyPolicy's own action
if an INDEPENDENT evaluation batch clears an exact one-sided sign-test
p-value -- never trust a single noisy or single-guess estimate) combined
with ``oracle_search``'s depth-D branching (each sampled world's candidate
action is scored not by a plain GreedyPolicy rollout, as in
``CounterfactualSearchPolicy``, but by ``oracle_best_action``'s own
depth-(D-1) single-guess continuation WITHIN that same sampled world --
this is the only way to add real multi-ply lookahead without an
exponential blowup: K samples X depth-1's branching, not K samples at
EVERY node of a depth-D tree).

This is still, honestly, only a partial fix for the imperfect-information
problem depth>0 introduces (each sampled world's own depth-(D-1)
continuation still assumes that ONE guessed world for its own internal
value estimate -- the "strategy fusion" caveat from the original oracle
design consult). What changes is that the ROOT decision itself is no
longer staked on a single guess: it only acts on an average over K
independent guesses, gated by the same statistical caution that kept
``CounterfactualSearchPolicy`` safe.

Scope, like the sibling modules: card-play decisions only; the trump call
defers to :class:`GreedyPolicy` directly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from math import comb

from deephokm.cards import NUM_CARDS, NUM_RANKS
from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies.counterfactual_search import MAX_P_VALUE_DEFAULT, MAX_ROLLOUT_PLIES_DEFAULT
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.oracle_search import oracle_ceiling
from deephokm.policies.search import VoidTracker, _clone_for_simulation, sample_determinized_hands
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import NUM_SEATS, Phase, team_of

N_SAMPLES_DEFAULT = 24
SEARCH_DEPTH_DEFAULT = 2
MIN_N_SAMPLES = 3


def _sign_test_p_value(wins: int, n_discordant: int) -> float:
    """Exact one-sided sign-test p-value: ``P(Binomial(n_discordant, 0.5) >= wins)``."""
    if n_discordant == 0:
        return 1.0
    favorable = sum(comb(n_discordant, k) for k in range(wins, n_discordant + 1))
    return favorable / (1 << n_discordant)


@dataclass
class LegalDepthSearchPolicy:
    """K-sample, depth-D legal search with an exact sign-test switch gate.

    Attributes:
        n_samples: Determinized worlds sampled per real decision, split
            into a selection batch and an independent evaluation batch
            (see ``CounterfactualSearchPolicy`` for why this split matters).
        search_depth: Forwarded to ``oracle_best_action``/``oracle_ceiling``
            as each sample's own depth-D continuation budget.
        max_rollout_plies: Safety bound on the post-depth rollout.
        max_p_value: Maximum one-sided sign-test p-value before switching
            away from GreedyPolicy's own action.
        seed: Seeds this policy's own determinization RNG.
    """

    n_samples: int = N_SAMPLES_DEFAULT
    search_depth: int = SEARCH_DEPTH_DEFAULT
    max_rollout_plies: int = MAX_ROLLOUT_PLIES_DEFAULT
    max_p_value: float = MAX_P_VALUE_DEFAULT
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_samples < MIN_N_SAMPLES:
            raise ValueError(f"n_samples must be >= {MIN_N_SAMPLES}, got {self.n_samples}")
        if self.search_depth < 1:
            raise ValueError(f"search_depth must be >= 1, got {self.search_depth}")
        if not (math.isfinite(self.max_p_value) and 0.0 <= self.max_p_value <= 1.0):
            raise ValueError(f"max_p_value must be finite and in [0, 1], got {self.max_p_value}")
        if self.max_rollout_plies <= 0:
            raise ValueError(f"max_rollout_plies must be positive, got {self.max_rollout_plies}")
        self._rng = random.Random(self.seed)
        self.voids = VoidTracker()
        self.greedy = GreedyPolicy()

    def reset_hand(self) -> None:
        """Clear void state; call at the start of every new hand."""
        self.voids.reset()

    def observe(self, seat: int, card: int, led_suit: int | None) -> None:
        """Record one publicly-played card from anywhere at the table."""
        self.voids.observe(seat, card, led_suit)

    def _score_action(
        self,
        engine: HokmEngine,
        seat: int,
        action: int,
        team: int,
        sampled_hands: list[list[int]],
        clone_rng: random.Random,
    ) -> float:
        """Score ``action`` in one sampled world via its own depth-(D-1)
        single-guess continuation (not a plain GreedyPolicy rollout, unlike
        ``CounterfactualSearchPolicy`` -- this is what adds real multi-ply
        lookahead within each sample).
        """
        clone = _clone_for_simulation(engine, seat, sampled_hands, rng=clone_rng)
        outcome = clone.apply_action(action, seat=seat)
        if outcome.hand_complete:
            assert outcome.hand_winner_team is not None
            return 1.0 if outcome.hand_winner_team == team else -1.0
        # A card-play action never re-enters TRUMP_CALL mid-hand, so
        # `clone` is guaranteed still in Phase.CARD_PLAY here -- the only
        # way to leave it is `outcome.hand_complete`, already handled above.
        assert clone.state.hands.phase is Phase.CARD_PLAY
        return oracle_ceiling(
            clone,
            team,
            depth=self.search_depth - 1,
            max_rollout_plies=self.max_rollout_plies,
            rng=clone_rng,
        )

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
        game_points = engine.state.game_points
        if len(legal) <= 1:
            return legal[0]
        if hands.phase is not Phase.CARD_PLAY:
            obs = observation_for(hands, root_seat, game_points)
            return self.greedy.act(obs, mask_for(legal))

        team = team_of(root_seat)
        obs = observation_for(hands, root_seat, game_points)
        greedy_action = self.greedy.act(obs, mask_for(legal))
        # One hoisted clone RNG per decision: the clones only ever consume
        # it via a simulated hand's redeal, into state that is discarded
        # the instant the hand completes, so a single shared (fresh,
        # unseeded -- as the per-clone RNGs used to be) instance per
        # decide() is behaviorally identical and skips one construction
        # per simulated clone.
        clone_rng = random.Random()

        own_hand = hands.hands[root_seat]
        seen = set(hands.played) | set(own_hand)
        unseen_pool = [c for c in range(NUM_CARDS) if c not in seen]
        remaining_sizes = [len(hands.hands[s]) for s in range(NUM_SEATS)]

        def sample_world() -> list[list[int]]:
            return sample_determinized_hands(
                root_seat,
                own_hand,
                unseen_pool,
                remaining_sizes,
                voids=self.voids.voids,
                rng=self._rng,
            )

        n_select = self.n_samples // 2
        n_eval = self.n_samples - n_select

        select_totals = dict.fromkeys(legal, 0.0)
        for _ in range(n_select):
            sampled_hands = sample_world()
            for action in legal:
                select_totals[action] += self._score_action(
                    engine, root_seat, action, team, sampled_hands, clone_rng
                )
        best_action = max(legal, key=lambda a: select_totals[a])

        if best_action == greedy_action or self.max_p_value <= 0.0:
            # max_p_value <= 0.0 makes switching impossible by definition
            # (an exact p-value is always > 0 for a finite sample) -- this
            # is checked directly rather than via the float comparison
            # below, which can underflow to exactly 0.0 at a large enough
            # evaluation batch and wrongly compare equal to a 0.0 threshold.
            return greedy_action

        wins = losses = 0
        for _ in range(n_eval):
            sampled_hands = sample_world()
            b = self._score_action(engine, root_seat, best_action, team, sampled_hands, clone_rng)
            g = self._score_action(engine, root_seat, greedy_action, team, sampled_hands, clone_rng)
            if b > g:
                wins += 1
            elif g > b:
                losses += 1
        p_value = _sign_test_p_value(wins, wins + losses)
        return best_action if p_value <= self.max_p_value else greedy_action


@dataclass
class HandResult:
    """One played hand's outcome, for aggregate reporting."""

    won: bool
    team_is_hakem: bool


def play_hand(
    seed: int,
    controlled_team: int,
    *,
    n_samples: int = N_SAMPLES_DEFAULT,
    search_depth: int = SEARCH_DEPTH_DEFAULT,
    max_p_value: float = MAX_P_VALUE_DEFAULT,
    search_seed: int = 0,
    max_rollout_plies: int = MAX_ROLLOUT_PLIES_DEFAULT,
) -> HandResult:
    """Play a single seeded match's first hand; ``controlled_team`` uses
    :class:`LegalDepthSearchPolicy`, the other team plays :class:`GreedyPolicy`.
    """
    if controlled_team not in (0, 1):
        raise ValueError(f"controlled_team must be 0 or 1, got {controlled_team}")
    engine = HokmEngine()
    engine.start_match(seed=seed)
    controlled_seats = {controlled_team, controlled_team + 2}
    search = LegalDepthSearchPolicy(
        n_samples=n_samples,
        search_depth=search_depth,
        max_p_value=max_p_value,
        max_rollout_plies=max_rollout_plies,
        seed=search_seed,
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
