"""Trick resolution: who wins a completed trick of Hokm."""

from __future__ import annotations

from deephokm.cards import NUM_RANKS
from deephokm.rules.state import NUM_SEATS


def resolve_trick(trick: list[tuple[int, int]], trump: int | None) -> int:
    """Return the seat that wins a completed trick.

    The highest trump played wins the trick; if no trump was played, the
    highest card of the led suit wins. The first entry of ``trick`` led.

    Args:
        trick: The four ``(seat, card_id)`` plays in order.
        trump: The declared trump suit id, or ``None`` (treated as no trump,
            which cannot occur in a real Hokm hand but keeps the function total).

    Raises:
        ValueError: If the trick does not hold exactly four cards or seats repeat.
    """
    if len(trick) != NUM_SEATS:
        raise ValueError(f"trick holds {len(trick)} cards, expected 4")
    seats = [seat for seat, _ in trick]
    if len(set(seats)) != NUM_SEATS:
        raise ValueError(f"trick seats repeat: {seats}")

    winner_seat, winner_card = trick[0]
    winner_suit = winner_card // NUM_RANKS
    winner_is_trump = trump is not None and winner_suit == trump
    for seat, card in trick[1:]:
        suit = card // NUM_RANKS
        is_trump = trump is not None and suit == trump
        beats = False
        if is_trump and not winner_is_trump:
            # Any trump beats a non-trump winner.
            beats = True
        elif is_trump == winner_is_trump and suit == winner_suit and card > winner_card:
            # Same suit and same trump-ness as the current winner: higher wins.
            beats = True
        if beats:
            winner_seat, winner_card = seat, card
            winner_suit = suit
            winner_is_trump = is_trump
    return winner_seat
