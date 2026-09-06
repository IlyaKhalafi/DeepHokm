"""Dealing: hakem selection and uniform card distribution.

The shuffle is a uniform Fisher-Yates driven by a caller-supplied
:class:`random.Random` instance. The hakem receives 5 cards first, declares
trump, then the remaining 47 cards are dealt so that every player ends with
exactly 13 (the hakem 5 + 8). Dealing proceeds around the table clockwise
starting left of the hakem.
"""

from __future__ import annotations

import random

from deephokm.cards import NUM_CARDS
from deephokm.rules.state import (
    CARDS_PER_PLAYER,
    HAKEM_FIRST_BATCH,
    NUM_SEATS,
    HandState,
)

# After the hakem's opening 5 cards, the remaining 47 split as 8 more to the
# hakem and 13 to each other seat. Cards flow one at a time clockwise from the
# seat left of the hakem, skipping seats whose quota is exhausted.


def shuffle_deck(rng: random.Random) -> list[int]:
    """Return a uniformly shuffled permutation of all 52 card ids."""
    deck = list(range(NUM_CARDS))
    # Fisher-Yates: swap each position with a uniformly chosen later-or-equal one.
    for i in range(len(deck) - 1, 0, -1):
        j = rng.randrange(i + 1)
        deck[i], deck[j] = deck[j], deck[i]
    return deck


def first_hakem(rng: random.Random) -> int:
    """Return a uniformly random first hakem seat."""
    return rng.randrange(NUM_SEATS)


def deal_initial(deck: list[int], hakem: int) -> list[list[int]]:
    """Deal the hakem's opening 5 cards from the front of the deck.

    Args:
        deck: The shuffled deck (consumed from the front).
        hakem: The hakem seat.

    Returns:
        Per-seat hands; only the hakem is non-empty (5 cards).
    """
    hands: list[list[int]] = [[] for _ in range(NUM_SEATS)]
    hands[hakem] = sorted(deck[:HAKEM_FIRST_BATCH])
    return hands


def remaining_quota(hakem: int) -> list[int]:
    """Return how many more cards each seat receives after the opening 5.

    The hakem needs ``13 - 5 = 8`` more; every other seat needs 13.
    """
    return [CARDS_PER_PLAYER - (HAKEM_FIRST_BATCH if s == hakem else 0) for s in range(NUM_SEATS)]


def deal_remaining(deck: list[int], hakem: int, hands: list[list[int]]) -> None:
    """Deal the remaining 47 cards around the table clockwise, left of the hakem.

    Cards flow one at a time in seat order starting left of the hakem, with
    each seat skipped once its remaining quota is exhausted. The schedule is an
    internal detail; the guarantees are that the remaining deck is exhausted,
    every seat ends with exactly :data:`CARDS_PER_PLAYER` cards, and the deal
    starts left of the hakem proceeding clockwise.

    Args:
        deck: The shuffled deck with the first 5 cards already consumed.
        hakem: The hakem seat.
        hands: Per-seat hands to extend (mutated in place). The hakem already
            holds the opening 5 cards.
    """
    quota = remaining_quota(hakem)
    seat = (hakem + 1) % NUM_SEATS
    pos = 0
    while pos < len(deck):
        if quota[seat] > 0:
            hands[seat].append(deck[pos])
            quota[seat] -= 1
            pos += 1
        seat = (seat + 1) % NUM_SEATS
    for s, hand in enumerate(hands):
        if len(hand) != CARDS_PER_PLAYER:
            raise AssertionError(f"seat {s} holds {len(hand)} cards, expected {CARDS_PER_PLAYER}")
    for hand in hands:
        hand.sort()


def new_hand(deck: list[int], hakem: int) -> HandState:
    """Create a fresh :class:`HandState` with only the hakem's opening 5 dealt.

    The other 47 cards stay in ``pending_deck`` — undealt to any seat — until
    the hakem declares trump; the engine calls :func:`deal_remaining` on that
    pending deck once the trump action is applied. Trump is left undeclared
    (``TRUMP_CALL`` phase).
    """
    hands = deal_initial(deck, hakem)
    return HandState(hands=hands, hakem=hakem, pending_deck=deck[HAKEM_FIRST_BATCH:])
