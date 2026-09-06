"""The full Hokm state machine.

:class:`HokmEngine` advances a :class:`~deephokm.rules.state.MatchState` through
trump declaration and card play, applying the pure rule functions from the
sibling modules. It is the single authority on turn order; the Gymnasium
environment and the web UI both sit on top of it.

Public events are returned from :meth:`HokmEngine.apply_action` so callers can
derive observations, rewards, and rendering without re-deriving state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from deephokm.rules import dealing, legality, scoring, tricks
from deephokm.rules.state import (
    NUM_SEATS,
    TRICKS_PER_HAND,
    HandState,
    MatchState,
    Phase,
    team_of,
    teammate,
)

ActionType = int


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """The state transition produced by one applied action.

    Attributes:
        seat: The seat that acted.
        action: The action id applied.
        card: The card id played, or ``None`` for a trump declaration.
        trump: The newly declared trump suit, or ``None``.
        trick_complete: Whether this action completed a trick.
        trick_winner: The seat that won the completed trick, if any.
        tricks_won: Per-team trick tally *of the hand this trick belonged
            to*, captured at the moment the trick completed. A completed
            hand's 13th trick immediately deals the next hand (zeroing the
            live ``HandState.tricks_won``), so a caller wanting the final
            tally of the hand that just ended must read it from here, not
            from the engine's current state.
        hand_complete: Whether this action completed the hand.
        hand_winner_team: Team index that won the completed hand, if any.
        match_complete: Whether this action completed the match.
        match_winner_team: Team index that won the match, if any.
    """

    seat: int
    action: int
    card: int | None = None
    trump: int | None = None
    trick_complete: bool = False
    trick_winner: int | None = None
    tricks_won: tuple[int, int] | None = None
    hand_complete: bool = False
    hand_winner_team: int | None = None
    match_complete: bool = False
    match_winner_team: int | None = None


@dataclass
class HokmEngine:
    """Authoritative Hokm state machine for one match.

    The engine owns a seeded RNG used for the first hakem choice and every
    shuffle; identical seeds reproduce identical matches.

    Attributes:
        rng: The seeded random generator.
        state: The live match state.
    """

    rng: random.Random = field(default_factory=random.Random)
    state: MatchState = field(default_factory=lambda: MatchState(hands=_empty_hand()))

    def start_match(self, seed: int | None = None) -> None:
        """Reset to a fresh match with a new deal.

        Args:
            seed: Optional explicit seed; when ``None`` the engine keeps its
                current RNG stream (so ``random.Random(s)`` + ``start_match()``
                is fully reproducible).
        """
        if seed is not None:
            self.rng = random.Random(seed)
        self.state.reset_points()
        self.state.hakem = dealing.first_hakem(self.rng)
        self._deal_hand()

    def _deal_hand(self) -> None:
        """Deal a new hand for the current hakem."""
        deck = dealing.shuffle_deck(self.rng)
        self.state.hands = dealing.new_hand(deck, self.state.hakem)

    def current_seat(self) -> int:
        """Return the seat that must act next.

        Raises:
            RuntimeError: If the match is over (no seat acts).
        """
        hands = self.state.hands
        if hands.phase is Phase.TRUMP_CALL:
            return hands.hakem
        if hands.phase is Phase.CARD_PLAY:
            if not hands.current_trick:
                return hands.leader
            last_seat = hands.current_trick[-1][0]
            return (last_seat + 1) % NUM_SEATS
        raise RuntimeError(f"no acting seat in phase {hands.phase.name}")

    def legal_actions(self, seat: int | None = None) -> list[int]:
        """Return the legal actions for ``seat`` (default: current seat)."""
        if seat is None:
            seat = self.current_seat()
        return legality.legal_actions(self.state.hands, seat)

    def is_legal(self, action: int, seat: int | None = None) -> bool:
        """Return whether ``action`` is legal for ``seat`` (default: current)."""
        if seat is None:
            seat = self.current_seat()
        return action in self.legal_actions(seat)

    def apply_action(self, action: int, seat: int | None = None) -> ActionOutcome:
        """Apply an action and return the resulting transition.

        Args:
            action: Action id in ``0..55``.
            seat: The acting seat; defaults to the current seat. Passing a
                seat that is not the current one is an error.

        Returns:
            The :class:`ActionOutcome` describing what happened.

        Raises:
            ValueError: If the action is illegal for the acting seat.
            RuntimeError: If the match is already over.
        """
        if self.state.winner is not None:
            raise RuntimeError("match is over; call start_match() to start a new one")
        if seat is None:
            seat = self.current_seat()
        elif seat != self.current_seat():
            raise ValueError(f"seat {seat} cannot act; it is seat {self.current_seat()}'s turn")
        if not self.is_legal(action, seat):
            raise ValueError(
                f"illegal action {action} for seat {seat} in phase {self.state.hands.phase.name}"
            )

        hands = self.state.hands
        if legality.is_trump_action(action):
            suit = legality.trump_action_to_suit(action)
            hands.trump = suit
            # The hakem calls trump on the opening 5 alone; the other 47
            # cards were held back exactly for this moment.
            dealing.deal_remaining(hands.pending_deck, hands.hakem, hands.hands)
            hands.pending_deck = []
            hands.phase = Phase.CARD_PLAY
            hands.leader = hands.hakem
            return ActionOutcome(seat=seat, action=action, trump=suit)

        card = action
        hands.remove_card(seat, card)
        hands.current_trick.append((seat, card))
        hands.played.append(card)

        if len(hands.current_trick) < NUM_SEATS:
            return ActionOutcome(seat=seat, action=action, card=card)

        winner = tricks.resolve_trick(hands.current_trick, hands.trump)
        hands.tricks_won[team_of(winner)] += 1
        hands.trick_winners.append(winner)
        hands.current_trick = []
        hands.leader = winner
        # Snapshot now: a 13th trick redeals the next hand below, zeroing
        # HandState.tricks_won before a caller can read the final tally.
        tricks_won_snapshot = (hands.tricks_won[0], hands.tricks_won[1])

        trick_complete = True
        hand_complete = False
        hand_winner_team: int | None = None
        match_complete = False
        match_winner_team: int | None = None

        if len(hands.trick_winners) == TRICKS_PER_HAND:
            hands.phase = Phase.HAND_OVER
            hand_complete = True
            hand_winner_team = scoring.score_hand(hands.tricks_won)
            scoring.award_game_point(self.state, hand_winner_team)
            if self.state.winner is not None:
                match_complete = True
                match_winner_team = self.state.winner
                hands.phase = Phase.MATCH_OVER
            else:
                self.state.hakem = scoring.next_hakem(self.state.hakem, hand_winner_team)
                self.state.hand_number += 1
                self._deal_hand()

        return ActionOutcome(
            seat=seat,
            action=action,
            card=card,
            trick_complete=trick_complete,
            trick_winner=winner,
            tricks_won=tricks_won_snapshot,
            hand_complete=hand_complete,
            hand_winner_team=hand_winner_team,
            match_complete=match_complete,
            match_winner_team=match_winner_team,
        )


def _empty_hand() -> HandState:
    """Return a placeholder hand state (no cards, used before the first deal)."""
    return HandState(hands=[[] for _ in range(NUM_SEATS)], hakem=0, phase=Phase.MATCH_OVER)


def partner_of(seat: int) -> int:
    """Return the partner seat of ``seat`` (re-exported convenience)."""
    return teammate(seat)


__all__ = [
    "ActionOutcome",
    "ActionType",
    "HokmEngine",
    "partner_of",
]
