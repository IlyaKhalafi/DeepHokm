"""Clairvoyant depth-D ceiling: how much value depends on hidden information?

The one-ply counterfactual search (:mod:`deephokm.policies.counterfactual_search`)
found that only ~1% of real decisions have a statistically real advantage
over GreedyPolicy's own choice, and playing that way scores 0.52 hand win
rate -- indistinguishable from doing nothing. That search has a specific,
narrow blind spot: it only ever looks one ply ahead for the root seat, then
falls back to GreedyPolicy for literally everyone (including the root
team's own later turns) for the rest of the hand. It cannot see an exploit
that requires a *coordinated sequence* of non-greedy choices by the
controlled team across several of its own turns in one hand (e.g.
sacrificing a trick now to set up a stronger position two tricks later).

Before spending engineering effort on a *legal*, information-set-correct
multi-ply search (which has to solve a real new correctness problem: a
future controlled decision must not implicitly depend on hidden
information that seat could not actually have -- "strategy fusion" in
imperfect-information game AI), this module answers a cheaper, prior
question: even with **full knowledge of every hidden hand** (an oracle no
real player or policy could have), how much better can the controlled team
do than GreedyPolicy's own realized outcome, within a bounded D-ply
lookahead over its own decisions? If this ceiling is itself close to zero,
there is no legal exploit at this depth worth searching for, and building
the harder, correctness-sensitive legal version is not worth it.

This is dramatically simpler to get right than the one-ply search's own
history (which took three rounds of adversarial review to fix real
statistical bugs): with the true hand plugged in, there is no
determinization sampling, no noise, no significance test at all. A given
real decision either has a strictly better forced line within D plies or
it does not -- an exact, deterministic answer.

Scope, like ``counterfactual_search.py``: card-play decisions only. Every
branch here terminates at-or-before the current hand's completion (a
terminal outcome is returned the instant ``apply_action`` reports one), so
this never touches a subsequent hand's trump call or the engine's
automatic redeal -- callers must start from a state already in
``Phase.CARD_PLAY`` (:func:`oracle_ceiling` asserts this).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from deephokm.policies.counterfactual_search import (
    MAX_ROLLOUT_PLIES_DEFAULT,
    rollout_to_hand_end,
)
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.search import _clone_for_simulation
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import NUM_SEATS, HandState, Phase

DEPTH_DEFAULT = 2


@dataclass
class OracleCeilingResult:
    """One real decision's exact, clairvoyant depth-D ceiling.

    Attributes:
        hand_number: The match's 1-based hand index.
        root_seat: The seat that made this decision in the real game.
        depth: The lookahead depth (controlled-team decisions) searched.
        actual_outcome: What GreedyPolicy's own choice actually led to in
            the real game this decision was sampled from (+1/-1).
        oracle_outcome: The best outcome achievable within ``depth`` plies
            of the controlled team's own decisions, given full knowledge of
            every hand. Always >= ``actual_outcome`` (GreedyPolicy's own
            line is one of the branches considered).
    """

    hand_number: int
    root_seat: int
    depth: int
    actual_outcome: float
    oracle_outcome: float

    @property
    def improved(self) -> bool:
        """Whether the oracle found a strictly better forced line."""
        return self.oracle_outcome > self.actual_outcome


@dataclass(frozen=True)
class _SearchContext:
    """The parts of an oracle search that never change across recursion."""

    controlled_seats: set[int]
    root_team: int
    greedy: GreedyPolicy
    max_rollout_plies: int
    # One hoisted RNG for the whole search: clones only consume it when a
    # simulated hand re-deals, and a fresh, never-seeded random.Random per
    # clone costs more than it does here.
    rng: random.Random


def _greedy_action(hands: HandState, seat: int, legal: list[int], ctx: _SearchContext) -> int:
    # play_from_state is the same decision act() makes from the equivalent
    # observation (cheapest win, lowest discard, lead), minus the numpy
    # observation build; a non-empty hand always has a legal card action, so
    # no mask is needed.
    return ctx.greedy.play_from_state(hands.trump, hands.current_trick, seat, legal)


def _step(
    engine: HokmEngine, seat: int, action: int, ctx: _SearchContext
) -> tuple[float | None, HokmEngine | None]:
    """Clone ``engine`` (every seat's *real* hand, not a determinization),
    apply ``action``, and report a terminal outcome if the hand just ended.
    """
    clone = _clone_for_simulation(engine, seat, engine.state.hands.hands, rng=ctx.rng)
    outcome = clone.apply_action(action, seat=seat)
    if outcome.hand_complete:
        assert outcome.hand_winner_team is not None
        return (1.0 if outcome.hand_winner_team == ctx.root_team else -1.0), None
    return None, clone


def _oracle_value(engine: HokmEngine, remaining_depth: int, ctx: _SearchContext) -> float:
    """Exact best achievable hand outcome for ``ctx.root_team`` from ``engine``.

    Branches only at a controlled seat's own card-play decision while
    ``remaining_depth`` budget remains -- an opponent's turn, and a forced
    single-legal-action turn for anyone, is always one deterministic
    action, never a branch point. Once the depth budget is exhausted at a
    real controlled-team choice, the rest of the hand is played out by a
    single GreedyPolicy-for-everyone rollout (matching the one-ply search's
    own continuation model exactly, just with the true hand instead of a
    determinization).
    """
    hands = engine.state.hands
    # Always CARD_PLAY here (callers assert it; a card play can only exit it
    # via a completed hand, which returns before recursing). Inline the same
    # next-seat rule engine.current_seat() applies in CARD_PLAY.
    if hands.current_trick:
        seat = (hands.current_trick[-1][0] + 1) % NUM_SEATS
    else:
        seat = hands.leader
    legal = engine.legal_actions(seat)
    controlled = seat in ctx.controlled_seats

    if controlled and remaining_depth > 0 and len(legal) > 1:
        best = -1.0
        for action in legal:
            outcome, child = _step(engine, seat, action, ctx)
            if outcome is not None:
                value = outcome
            else:
                assert child is not None
                value = _oracle_value(child, remaining_depth - 1, ctx)
            best = max(best, value)
            if best == 1.0:
                break  # can't do better than a guaranteed win
        return best

    if controlled and remaining_depth <= 0 and len(legal) > 1:
        clone = _clone_for_simulation(engine, seat, hands.hands, rng=ctx.rng)
        return rollout_to_hand_end(clone, ctx.root_team, ctx.greedy, ctx.max_rollout_plies)

    action = legal[0] if len(legal) == 1 else _greedy_action(hands, seat, legal, ctx)
    outcome, child = _step(engine, seat, action, ctx)
    if outcome is not None:
        return outcome
    assert child is not None
    return _oracle_value(child, remaining_depth, ctx)


def oracle_ceiling(
    engine: HokmEngine,
    controlled_team: int,
    *,
    depth: int = DEPTH_DEFAULT,
    max_rollout_plies: int = MAX_ROLLOUT_PLIES_DEFAULT,
    rng: random.Random | None = None,
) -> float:
    """The exact best achievable outcome for ``controlled_team`` from ``engine``.

    Args:
        engine: The real, live match engine (read, never mutated).
        controlled_team: 0 (seats 0, 2) or 1 (seats 1, 3).
        depth: How many of the controlled team's own future decisions to
            branch over before falling back to a GreedyPolicy rollout.
        max_rollout_plies: Forwarded to the post-depth rollout.
        rng: Shared RNG for the simulated clones (only consumed by a
            simulated hand's redeal, which callers discard); pass a
            hoisted instance when calling this repeatedly in one loop.

    Returns:
        +1.0 if the controlled team can force a win within ``depth`` plies
        of its own decisions (given full information), -1.0 otherwise.
    """
    if engine.state.hands.phase is not Phase.CARD_PLAY:
        raise ValueError("oracle_ceiling requires a state already in Phase.CARD_PLAY")
    if controlled_team not in (0, 1):
        raise ValueError(f"controlled_team must be 0 or 1, got {controlled_team}")
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    if max_rollout_plies <= 0:
        raise ValueError(f"max_rollout_plies must be positive, got {max_rollout_plies}")
    ctx = _SearchContext(
        controlled_seats={controlled_team, controlled_team + 2},
        root_team=controlled_team,
        greedy=GreedyPolicy(),
        max_rollout_plies=max_rollout_plies,
        # Hoisted: one clone RNG per search call instead of one fresh,
        # never-seeded random.Random per cloned engine.
        rng=rng if rng is not None else random.Random(),
    )
    return _oracle_value(engine, depth, ctx)


def oracle_best_action(
    engine: HokmEngine,
    controlled_team: int,
    *,
    depth: int = DEPTH_DEFAULT,
    max_rollout_plies: int = MAX_ROLLOUT_PLIES_DEFAULT,
    rng: random.Random | None = None,
) -> int:
    """The action achieving :func:`oracle_ceiling`'s value at the ROOT decision.

    Unlike :func:`oracle_ceiling` (which only returns the best achievable
    *value*), this is meant to be executed -- as the one real move an
    (oracle or oracle-informed) policy actually makes right now. Requires
    ``engine.current_seat()`` to be one of ``controlled_team``'s own seats;
    :func:`oracle_ceiling` itself has no such restriction (any seat may be
    acting when it is called, since it recurses through every seat's
    turns), but *choosing* a controlled team's action from a non-controlled
    seat's turn is not a meaningful request.

    Args:
        engine: The real, live match engine (read, never mutated).
        controlled_team: 0 (seats 0, 2) or 1 (seats 1, 3).
        depth: Forwarded to the underlying search.
        max_rollout_plies: Forwarded to the underlying search.
        rng: Shared RNG for the simulated clones (see
            :func:`oracle_ceiling`).

    Returns:
        A legal action id for the engine's current seat.
    """
    if engine.state.hands.phase is not Phase.CARD_PLAY:
        raise ValueError("oracle_best_action requires a state already in Phase.CARD_PLAY")
    if controlled_team not in (0, 1):
        raise ValueError(f"controlled_team must be 0 or 1, got {controlled_team}")
    seat = engine.current_seat()
    if seat not in (controlled_team, controlled_team + 2):
        raise ValueError(
            f"engine.current_seat() ({seat}) is not one of "
            f"controlled_team {controlled_team}'s seats"
        )
    legal = engine.legal_actions(seat)
    if len(legal) == 1:
        return legal[0]

    ctx = _SearchContext(
        controlled_seats={controlled_team, controlled_team + 2},
        root_team=controlled_team,
        greedy=GreedyPolicy(),
        max_rollout_plies=max_rollout_plies,
        # Hoisted: one clone RNG per search call instead of one fresh,
        # never-seeded random.Random per cloned engine.
        rng=rng if rng is not None else random.Random(),
    )
    best_action, best_value = legal[0], -2.0
    for action in legal:
        outcome, child = _step(engine, seat, action, ctx)
        if outcome is not None:
            value = outcome
        else:
            assert child is not None
            # `depth - 1` may go negative; `_oracle_value` treats any
            # remaining_depth <= 0 as "budget exhausted" uniformly, so this
            # is correct whether depth was 0 (evaluate each candidate by a
            # single GreedyPolicy-continued rollout) or positive.
            value = _oracle_value(child, depth - 1, ctx)
        if value > best_value:
            best_action, best_value = action, value
        if best_value == 1.0:
            break
    return best_action
