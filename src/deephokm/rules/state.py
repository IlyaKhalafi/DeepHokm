"""Core game state and phase tracking for the Hokm engine.

The state is a plain mutable structure owned by the engine. Cards are held as
per-player sorted lists of integer card ids; played cards accumulate in a
history list (most recent last).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from deephokm.cards import NUM_CARDS

NUM_SEATS = 4
TRICKS_PER_HAND = NUM_CARDS // NUM_SEATS  # 13
CARDS_PER_PLAYER = TRICKS_PER_HAND  # 13
HAKEM_FIRST_BATCH = 5
POINTS_TO_WIN_MATCH = 7
TRICKS_TO_WIN_HAND = (TRICKS_PER_HAND // 2) + 1  # 7 of 13

TEAM_A = (0, 2)
TEAM_B = (1, 3)


def teammate(seat: int) -> int:
    """Return the partner seat of ``seat``."""
    return (seat + 2) % NUM_SEATS


def team_of(seat: int) -> int:
    """Return the team index (0 for seats {0, 2}, 1 for seats {1, 3}) of ``seat``."""
    return seat % 2


class Phase(IntEnum):
    """Game phases, exactly one active at a time."""

    TRUMP_CALL = 0
    CARD_PLAY = 1
    HAND_OVER = 2
    MATCH_OVER = 3


@dataclass
class HandState:
    """Mutable state of a single hand (deal through the 13th trick).

    Attributes:
        hands: Per-seat sorted lists of card ids currently held.
        hakem: The seat that declares trump for this hand.
        trump: The declared trump suit id, or ``None`` before declaration.
        leader: The seat leading the current trick.
        current_trick: Cards played so far in the current trick, in play order,
            as ``(seat, card_id)`` pairs.
        played: All cards played this hand, in play order.
        tricks_won: Per-team trick counts for this hand, index 0 = team A.
        trick_winners: Seat that won each completed trick, in order.
        phase: The current phase.
    """

    hands: list[list[int]]
    hakem: int
    trump: int | None = None
    leader: int = -1
    current_trick: list[tuple[int, int]] = field(default_factory=list)
    played: list[int] = field(default_factory=list)
    tricks_won: list[int] = field(default_factory=lambda: [0, 0])
    trick_winners: list[int] = field(default_factory=list)
    phase: Phase = Phase.TRUMP_CALL

    def remove_card(self, seat: int, card_id: int) -> None:
        """Remove ``card_id`` from ``seat``'s hand, raising if absent."""
        hand = self.hands[seat]
        try:
            hand.remove(card_id)
        except ValueError:
            raise ValueError(f"seat {seat} does not hold card {card_id} (hand: {hand})") from None

    def on_table(self) -> list[int]:
        """Return the card ids currently on the table."""
        return [card for _, card in self.current_trick]


@dataclass
class MatchState:
    """Mutable state of a full match (hands until one team reaches 7 points).

    Attributes:
        hands: The current hand state (replaced on each new deal).
        game_points: Per-team match points, index 0 = team A.
        hakem: The current hakem seat.
        hand_number: 1-based index of the current hand.
        winner: Winning team index once the match is over, else ``None``.
    """

    hands: HandState
    game_points: list[int] = field(default_factory=lambda: [0, 0])
    hakem: int = 0
    hand_number: int = 1
    winner: int | None = None

    def reset_points(self) -> None:
        """Zero the match score at the start of a new match."""
        self.game_points = [0, 0]
        self.winner = None
        self.hand_number = 1
