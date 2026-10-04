"""Exhaustive-action counterfactual rollout diagnostic against GreedyPolicy.

Every fine-tuning attempt in this project's session-N+1 log (see
``PROGRESS.md``) trained a policy starting from a near-perfect GreedyPolicy
clone, then let PPO's own exploration try to discover something better. That
never produced more than noise around the clone's own baseline. This module
answers a narrower, more decisive question first, independent of PPO
entirely: **does a real, exploitable advantage over GreedyPolicy's own
choice exist at a given decision at all**, before spending more GPU-hours
hoping policy-gradient exploration stumbles onto it.

Unlike :mod:`deephokm.policies.search` (which searches only a trained
policy's own top-k actions and rolls out with that same trained policy,
bootstrapping off its critic), this module:

- Enumerates **every** legal action at the decision point, not a policy's
  own top-k -- GreedyPolicy's choice is not assumed to even be among the
  "reasonable" candidates.
- Rolls every simulated seat -- including the root seat's own later turns
  in the same hand -- forward with plain :class:`GreedyPolicy`, to exact
  hand completion. No critic, no neural network anywhere in this module.
  GreedyPolicy is a cheap, accurate continuation model for a policy this
  project already showed can imitate it at 99%+ action agreement, and hand
  completion (rather than stopping at the root's next turn) needs no value
  function at all -- the hand's winner is exact ground truth.
- Uses the *same* determinized worlds for every candidate action at a given
  decision (sampled once per determinization, reused across all actions),
  so comparisons between actions are not contaminated by additional sampling
  noise on top of the real variance across worlds.

The same information-correctness invariant as ``search.py`` applies: nothing
here may read another seat's actual hidden cards for use in sampling or
search. ``sample_determinized_hands`` remains the sole place other seats'
cards are invented for a simulation, never read from live state.

Scope: card-play decisions only, exactly as ``search.py`` documents; the
trump call defers to :class:`GreedyPolicy` directly.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from math import comb

from deephokm.cards import NUM_CARDS, NUM_RANKS
from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.search import VoidTracker, _clone_for_simulation, sample_determinized_hands
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import NUM_SEATS, HandState, Phase, team_of

N_SAMPLES_DEFAULT = 48
# A hand has at most TRICKS_PER_HAND * NUM_SEATS = 52 remaining plies from
# its very first decision; a generous margin above that catches a real bug
# (an infinite loop) as a loud RuntimeError instead of a silent bad score.
MAX_ROLLOUT_PLIES_DEFAULT = 60
# Only switch away from GreedyPolicy's own action when the independently
# re-evaluated advantage (see `decide`'s selection/evaluation split) clears
# this one-sided exact sign-test p-value threshold (see `_sign_test_p_value`
# and `_evaluate_advantage`). A Gaussian z-test was tried first and rejected:
# per-sample paired differences are discrete (-2, 0, or +2, not continuous),
# so at small n the normal approximation is miscalibrated (an empirically
# checked n=24 case with 16 "+2"s and 8 "-2"s computed a z clearing the 95%
# bound while its true one-sided binomial probability was ~0.076, not
# ~0.05), and an all-agreeing batch (e.g. 2-for-2 favoring the challenger)
# produces zero variance and therefore an infinite z that bypasses every
# finite threshold regardless of sample size (a symmetric-null event with
# 25% probability at n=2, not remotely rare). The exact sign test has
# neither problem: it is calibrated at any n, and a small n simply cannot
# produce a small p-value no matter how one-sided the (few) samples are.
MAX_P_VALUE_DEFAULT = 0.05
# Below this, splitting into a selection batch and a >= 2-sample evaluation
# batch (needed for a standard error) is impossible.
MIN_N_SAMPLES = 3


def rollout_to_hand_end(
    engine: HokmEngine,
    root_team: int,
    greedy: GreedyPolicy,
    max_plies: int = MAX_ROLLOUT_PLIES_DEFAULT,
) -> float:
    """Roll ``engine`` forward with GreedyPolicy for every seat, in place.

    Every seat -- including the root team's own later turns in this hand --
    plays GreedyPolicy until the hand completes; returns the exact outcome
    from ``root_team``'s perspective. No critic, no bootstrap: a hand is at
    most 13 tricks, cheap enough to always run to completion. Shared by
    :class:`CounterfactualSearchPolicy` and the oracle-ceiling module
    (:mod:`deephokm.policies.oracle_search`), which both need this exact
    "GreedyPolicy from here to hand end" continuation.
    """
    for _ in range(max_plies):
        hands = engine.state.hands
        # CARD_PLAY only (both call sites enter right after a card play in a
        # non-completed hand), so the acting seat is the current trick's
        # follower, or the trick's leader when it is empty -- inlined rather
        # than a current_seat() call per ply.
        seat = (hands.current_trick[-1][0] + 1) % NUM_SEATS if hands.current_trick else hands.leader
        legal = engine.legal_actions(seat)
        action = greedy.play_from_state(
            hands.trump,
            hands.current_trick,
            seat,
            legal,
            hand=hands.hands[seat],
            played=hands.played,
            played_by=hands.played_by,
            void_suits=hands.void_suits,
            tricks_won=hands.tricks_won,
        )
        outcome = engine.apply_action(action, seat=seat, _legal=legal)
        # None: mid-trick, non-terminal play on the _legal fast path.
        if outcome is not None and outcome.hand_complete:
            assert outcome.hand_winner_team is not None
            return 1.0 if outcome.hand_winner_team == root_team else -1.0
    raise RuntimeError(f"rollout exceeded {max_plies} plies without completing the hand")


def _sign_test_p_value(wins: int, n_discordant: int) -> float:
    """Exact one-sided sign-test p-value: ``P(Binomial(n_discordant, 0.5) >= wins)``.

    Ties (neither action beating the other in a given sampled world) are
    excluded from ``n_discordant`` entirely, as in the standard sign/McNemar
    test -- they carry no evidence either way. ``1.0`` (never enough
    evidence to switch) when there is no discordant evidence at all.
    """
    if n_discordant == 0:
        return 1.0
    favorable = sum(comb(n_discordant, k) for k in range(wins, n_discordant + 1))
    return favorable / (1 << n_discordant)


@dataclass
class DecisionRecord:
    """One searched decision's outcome, for post-hoc diagnostic analysis.

    Attributes:
        hand_number: The match's 1-based hand index.
        root_seat: The seat that made this decision.
        n_legal: Number of legal actions considered.
        greedy_action: The action :class:`GreedyPolicy` would have played.
        chosen_action: The action this policy actually returned (may equal
            ``greedy_action`` if no alternative cleared ``max_p_value``).
        greedy_score: ``greedy_action``'s average hand-outcome score (+1
            root's team wins the hand, -1 loses). Equal to ``best_score``
            when ``best_action == greedy_action`` (a single selection-batch
            score, since there is nothing to independently re-evaluate);
            otherwise from the independent evaluation batch.
        best_score: ``best_action``'s score, from the same batch as
            ``greedy_score`` (see above) -- so the two are always directly
            comparable.
        best_action: The action with the highest selection-batch score
            (may equal ``greedy_action``, if it was already best).
        p_value: The evaluation batch's exact one-sided sign-test p-value
            for ``best_action`` beating ``greedy_action`` more often than
            chance across discordant paired outcomes (1.0 when
            ``best_action == greedy_action``, since nothing was evaluated).
    """

    hand_number: int
    root_seat: int
    n_legal: int
    greedy_action: int
    chosen_action: int
    greedy_score: float
    best_score: float
    best_action: int
    p_value: float

    @property
    def raw_advantage(self) -> float:
        """``best_score - greedy_score``, uncapped by the switch threshold.

        Both scores come from the independent evaluation batch (see
        ``CounterfactualSearchPolicy.decide``), not the selection batch that
        picked ``best_action`` -- so this is an unbiased estimate, not one
        inflated by winner's-curse from maximizing over every legal action
        on a single noisy sample batch.
        """
        return self.best_score - self.greedy_score


@dataclass
class CounterfactualSearchPolicy:
    """Exhaustive one-ply IIMC search with a GreedyPolicy rollout continuation.

    Not a :class:`~deephokm.policies.base.HokmPolicy`: like
    :class:`~deephokm.policies.search.IIMCSearchPolicy`, it needs direct
    engine access, so it is driven by a bespoke match runner (see
    :func:`play_match`) that calls :meth:`observe` after every ply at the
    table and :meth:`reset_hand` at the start of every hand.

    Attributes:
        n_samples: Determinized worlds sampled per real decision.
        max_rollout_plies: Safety bound on plies simulated per candidate
            action before raising (a hand has at most 52 remaining plies).
        max_p_value: Maximum exact one-sided sign-test p-value (see
            `_sign_test_p_value`) the best action's advantage over
            GreedyPolicy's own action must clear before switching.
        seed: Seeds this policy's own determinization RNG.
        record_decisions: Keep a :class:`DecisionRecord` per real decision
            in :attr:`records` (small overhead; disable for a pure
            match-outcome run over many seeds if memory matters).
    """

    n_samples: int = N_SAMPLES_DEFAULT
    max_rollout_plies: int = MAX_ROLLOUT_PLIES_DEFAULT
    max_p_value: float = MAX_P_VALUE_DEFAULT
    seed: int = 0
    record_decisions: bool = True

    def __post_init__(self) -> None:
        if self.n_samples < MIN_N_SAMPLES:
            raise ValueError(f"n_samples must be >= {MIN_N_SAMPLES}, got {self.n_samples}")
        if not (math.isfinite(self.max_p_value) and 0.0 <= self.max_p_value <= 1.0):
            raise ValueError(f"max_p_value must be finite and in [0, 1], got {self.max_p_value}")
        self._rng = random.Random(self.seed)
        self.voids = VoidTracker()
        self.greedy = GreedyPolicy()
        self.decisions_searched = 0
        self.decisions_switched = 0
        self.records: list[DecisionRecord] = []

    def reset_hand(self) -> None:
        """Clear void state; call at the start of every new hand."""
        self.voids.reset()

    def observe(self, seat: int, card: int, led_suit: int | None) -> None:
        """Record one publicly-played card from anywhere at the table."""
        self.voids.observe(seat, card, led_suit)

    def _greedy_action(
        self, hands: HandState, seat: int, game_points: list[int], legal: list[int]
    ) -> int:
        obs = observation_for(hands, seat, game_points)
        mask = mask_for(legal)
        return self.greedy.act(obs, mask)

    def _rollout_to_hand_end(self, sim_engine: HokmEngine, root_team: int) -> float:
        return rollout_to_hand_end(sim_engine, root_team, self.greedy, self.max_rollout_plies)

    def _samplers(
        self,
        engine: HokmEngine,
        root_seat: int,
        root_team: int,
        clone_rng: random.Random,
    ) -> tuple[Callable[[], list[list[int]]], Callable[[int, list[list[int]]], float]]:
        """Build this decision's ``sample_world``/``score_action`` closures."""
        hands = engine.state.hands
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

        def score_action(action: int, sampled_hands: list[list[int]]) -> float:
            sim_engine = _clone_for_simulation(engine, root_seat, sampled_hands, rng=clone_rng)
            outcome = sim_engine.apply_action(action, seat=root_seat)
            if outcome.hand_complete:
                assert outcome.hand_winner_team is not None
                return 1.0 if outcome.hand_winner_team == root_team else -1.0
            return self._rollout_to_hand_end(sim_engine, root_team)

        return sample_world, score_action

    def _select_best_action(
        self,
        legal: list[int],
        n_select: int,
        sample_world: Callable[[], list[list[int]]],
        score_action: Callable[[int, list[list[int]]], float],
    ) -> tuple[int, dict[int, float]]:
        """Selection batch: the action that looks best across every legal
        action. Winner's-curse-biased (the max over many noisy options is
        upward-biased even with no real advantage) -- the winning action is
        used only to pick a *candidate*, never reported or compared against
        the switch threshold directly; the per-action averages are still
        returned so a non-switching decision can report an averaged score
        instead of a single noisy sample.
        """
        totals = dict.fromkeys(legal, 0.0)
        for _ in range(n_select):
            sampled_hands = sample_world()
            for action in legal:
                totals[action] += score_action(action, sampled_hands)
        averages = {action: total / n_select for action, total in totals.items()}
        return max(legal, key=lambda a: averages[a]), averages

    def _evaluate_advantage(
        self,
        best_action: int,
        greedy_action: int,
        n_eval: int,
        sample_world: Callable[[], list[list[int]]],
        score_action: Callable[[int, list[list[int]]], float],
    ) -> tuple[float, float, float]:
        """Evaluation batch: fresh, independent worlds, scoring only these
        two actions. This is what actually removes the winner's-curse bias
        -- an action can look best purely by sampling luck when many
        actions are compared, but that same luck does not, in expectation,
        repeat on an independent draw. Returns ``(best_score, greedy_score,
        p_value)``, the last from an exact one-sided sign test over
        discordant paired outcomes (see ``_sign_test_p_value``) rather than
        a Gaussian z -- the per-sample paired difference is discrete
        (-2, 0, or +2), so a normal approximation is miscalibrated at the
        sample sizes this module actually runs at, and degenerates to an
        unconditionally-passing infinite z whenever every sample agrees.
        """
        best_scores = []
        greedy_scores = []
        wins = losses = 0
        for _ in range(n_eval):
            sampled_hands = sample_world()
            b = score_action(best_action, sampled_hands)
            g = score_action(greedy_action, sampled_hands)
            best_scores.append(b)
            greedy_scores.append(g)
            if b > g:
                wins += 1
            elif g > b:
                losses += 1
        p_value = _sign_test_p_value(wins, wins + losses)
        return sum(best_scores) / n_eval, sum(greedy_scores) / n_eval, p_value

    def decide(self, engine: HokmEngine) -> int:
        """Choose the acting seat's next action, searching only card plays.

        Args:
            engine: The real, live match engine (read, never mutated).

        Returns:
            A legal action id for the engine's current seat.
        """
        root_seat = engine.current_seat()
        legal = engine.legal_actions(root_seat)
        hands = engine.state.hands
        game_points = engine.state.game_points
        if len(legal) <= 1:
            return legal[0]
        if hands.phase is not Phase.CARD_PLAY:
            # Scope: the trump call defers to GreedyPolicy directly.
            return self._greedy_action(hands, root_seat, game_points, legal)

        root_team = team_of(root_seat)
        greedy_action = self._greedy_action(hands, root_seat, game_points, legal)
        # One hoisted clone RNG per decision (a fresh, unseeded instance is
        # what the clones used to get each -- and clones only ever consume
        # it via a simulated hand's redeal, into state the caller discards),
        # so the determinization stream on self._rng is untouched.
        clone_rng = random.Random()
        sample_world, score_action = self._samplers(engine, root_seat, root_team, clone_rng)

        n_select = self.n_samples // 2
        n_eval = self.n_samples - n_select
        best_action, select_averages = self._select_best_action(
            legal, n_select, sample_world, score_action
        )

        if best_action == greedy_action:
            score = select_averages[greedy_action]
            chosen, best_score, greedy_score, p_value = greedy_action, score, score, 1.0
        else:
            best_score, greedy_score, p_value = self._evaluate_advantage(
                best_action, greedy_action, n_eval, sample_world, score_action
            )
            # max_p_value <= 0.0 is checked directly, rather than relying
            # only on the float comparison below, which can underflow to
            # exactly 0.0 at a large enough evaluation batch and wrongly
            # compare equal to a 0.0 threshold (switching impossible by
            # definition -- an exact p-value is always > 0 for a finite
            # sample).
            chosen = (
                best_action
                if self.max_p_value > 0.0 and p_value <= self.max_p_value
                else greedy_action
            )

        self.decisions_searched += 1
        if chosen != greedy_action:
            self.decisions_switched += 1
        if self.record_decisions:
            self.records.append(
                DecisionRecord(
                    hand_number=engine.state.hand_number,
                    root_seat=root_seat,
                    n_legal=len(legal),
                    greedy_action=greedy_action,
                    chosen_action=chosen,
                    greedy_score=greedy_score,
                    best_score=best_score,
                    best_action=best_action,
                    p_value=p_value,
                )
            )
        return chosen


def play_match(
    seed: int,
    controlled_team: int,
    *,
    n_samples: int = N_SAMPLES_DEFAULT,
    max_p_value: float = MAX_P_VALUE_DEFAULT,
    search_seed: int = 0,
    hand_only: bool = False,
) -> tuple[bool, CounterfactualSearchPolicy]:
    """Play one seeded match; both of ``controlled_team``'s seats search.

    The opposing team is always two fresh :class:`GreedyPolicy` instances.
    This is the bespoke match runner :class:`CounterfactualSearchPolicy`
    needs: it feeds every real ply at the table to the *same* search policy
    instance's :meth:`~CounterfactualSearchPolicy.observe` (voids are
    public, table-wide information) and resets it at every new hand.

    Args:
        seed: Match seed (deal reproducibility).
        controlled_team: 0 (seats 0, 2) or 1 (seats 1, 3) -- the team the
            search policy controls.
        n_samples: Forwarded to :class:`CounterfactualSearchPolicy`.
        max_p_value: Forwarded to :class:`CounterfactualSearchPolicy`.
        search_seed: Seeds the search policy's own determinization RNG.
        hand_only: Stop after the first hand instead of playing the whole
            match out, returning that hand's own winner. A full match
            multiplies per-decision search cost by every hand it takes to
            reach 7 points (~150-300 searched decisions observed in
            practice); a single hand is 10-20x cheaper and is the more
            fundamental calibration number in any case (see
            ``PROGRESS.md``: ~61-62% per-hand win rate already implies an
            ~80% match win rate under a simplified independent-hand model).

    Returns:
        ``(controlled_team_won, search_policy)`` -- the policy instance is
        returned so its accumulated ``records``/counters can be inspected.
        ``controlled_team_won`` reflects the single hand's winner when
        ``hand_only=True``, the match's winner otherwise.
    """
    engine = HokmEngine()
    engine.start_match(seed=seed)
    controlled_seats = {controlled_team, controlled_team + 2}
    search = CounterfactualSearchPolicy(
        n_samples=n_samples, max_p_value=max_p_value, seed=search_seed
    )
    opponents = {s: GreedyPolicy() for s in range(NUM_SEATS) if s not in controlled_seats}

    current_hand = engine.state.hand_number
    while engine.state.winner is None:
        if engine.state.hand_number != current_hand:
            search.reset_hand()
            current_hand = engine.state.hand_number
        seat = engine.current_seat()
        hands = engine.state.hands
        if seat in controlled_seats:
            action = search.decide(engine)
        else:
            legal = engine.legal_actions(seat)
            obs = observation_for(hands, seat, engine.state.game_points)
            mask = mask_for(legal)
            action = opponents[seat].act(obs, mask)
        led_suit = None
        if hands.phase is Phase.CARD_PLAY and hands.current_trick:
            led_suit = hands.current_trick[0][1] // NUM_RANKS
        outcome = engine.apply_action(action, seat=seat)
        if outcome.card is not None:
            search.observe(seat, outcome.card, led_suit)
        if hand_only and outcome.hand_complete:
            assert outcome.hand_winner_team is not None
            return outcome.hand_winner_team == controlled_team, search

    assert engine.state.winner is not None
    return engine.state.winner == controlled_team, search
