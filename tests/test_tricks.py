"""Tests for trick resolution."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from deephokm.cards import NUM_RANKS
from deephokm.rules import tricks


def cid(suit: int, rank: int) -> int:
    return suit * NUM_RANKS + rank


def test_highest_of_led_suit_wins_no_trump() -> None:
    trick = [(0, cid(0, 5)), (1, cid(0, 12)), (2, cid(1, 11)), (3, cid(0, 9))]
    assert tricks.resolve_trick(trick, trump=None) == 1


def test_highest_trump_wins() -> None:
    trick = [(0, cid(0, 12)), (1, cid(3, 0)), (2, cid(1, 11)), (3, cid(3, 5))]
    assert tricks.resolve_trick(trick, trump=3) == 3


def test_first_trump_beats_high_non_trump() -> None:
    trick = [(0, cid(0, 12)), (1, cid(2, 0)), (2, cid(1, 12)), (3, cid(1, 5))]
    assert tricks.resolve_trick(trick, trump=2) == 1


def test_led_trump_still_requires_highest_trump() -> None:
    """When trump is led, the highest trump wins (not the first played)."""
    trick = [(0, cid(2, 3)), (1, cid(2, 12)), (2, cid(0, 12)), (3, cid(2, 7))]
    assert tricks.resolve_trick(trick, trump=2) == 1


def test_trump_led_ace_beats_lower_trump() -> None:
    trick = [(0, cid(1, 12)), (1, cid(1, 0)), (2, cid(1, 5)), (3, cid(0, 12))]
    assert tricks.resolve_trick(trick, trump=1) == 0


def test_discard_never_wins() -> None:
    """A card off the led suit that is not trump can never win."""
    trick = [(0, cid(0, 2)), (1, cid(1, 12)), (2, cid(0, 3)), (3, cid(2, 12))]
    assert tricks.resolve_trick(trick, trump=3) == 2


def test_leader_wins_tie_impossible_but_low_led() -> None:
    """Leader with the highest led card wins even when others discard high."""
    trick = [(0, cid(0, 8)), (1, cid(1, 12)), (2, cid(3, 12)), (3, cid(2, 12))]
    assert tricks.resolve_trick(trick, trump=None) == 0


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5])
def test_resolve_rejects_wrong_trick_size(n: int) -> None:
    trick = [(i % 4, i) for i in range(n)]
    with pytest.raises(ValueError, match="expected 4"):
        tricks.resolve_trick(trick, trump=0)


def test_resolve_rejects_repeated_seats() -> None:
    trick = [(0, 1), (0, 2), (2, 3), (3, 4)]
    with pytest.raises(ValueError, match="seats repeat"):
        tricks.resolve_trick(trick, trump=0)


@given(
    suits=st.lists(st.integers(min_value=0, max_value=3), min_size=4, max_size=4),
    ranks=st.lists(st.integers(min_value=0, max_value=12), min_size=4, max_size=4),
    trump=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=300, deadline=None)
def test_winner_is_always_trump_or_led_suit_card(
    suits: list[int], ranks: list[int], trump: int
) -> None:
    """The winning card must be the highest trump or the highest led-suit card."""
    # Force distinct cards: tweak ranks until all four card ids differ.
    cards: list[int] = []
    for i in range(4):
        card = suits[i] * NUM_RANKS + ranks[i]
        while card in cards:
            card = (card + NUM_RANKS) % 52
        cards.append(card)
    # Re-derive suits/ranks from the deduplicated cards.
    trick = [(i, cards[i]) for i in range(4)]
    winner = tricks.resolve_trick(trick, trump)
    winning_card = cards[winner]
    winning_suit = winning_card // NUM_RANKS
    led_suit = cards[0] // NUM_RANKS

    trump_cards = [c for c in cards if c // NUM_RANKS == trump]
    if trump_cards:
        assert winning_suit == trump
        assert winning_card == max(trump_cards)
    else:
        assert winning_suit == led_suit
        led_cards = [c for c in cards if c // NUM_RANKS == led_suit]
        assert winning_card == max(led_cards)


@given(
    base=st.integers(min_value=0, max_value=11),
    trump=st.integers(min_value=0, max_value=3),
    high_first=st.booleans(),
)
@settings(max_examples=200, deadline=None)
def test_higher_trump_always_wins_over_lower(base: int, trump: int, high_first: bool) -> None:
    """For any adjacent pair of trumps, the higher wins regardless of order."""
    low = base + trump * NUM_RANKS
    high = low + 1
    filler = ((trump + 1) % 4) * NUM_RANKS
    first, second = (high, low) if high_first else (low, high)
    plays = [(0, first), (1, second), (2, filler), (3, filler + 1)]
    winner = tricks.resolve_trick(plays, trump)
    assert winner in (0, 1)
    assert plays[winner][1] == high
