"""Tests for dealing: hakem selection, shuffle uniformity, hand totals."""

from __future__ import annotations

import random
from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from deephokm.cards import NUM_CARDS
from deephokm.rules import dealing
from deephokm.rules.state import CARDS_PER_PLAYER, HAKEM_FIRST_BATCH, NUM_SEATS


def test_shuffle_is_permutation() -> None:
    deck = dealing.shuffle_deck(random.Random(0))
    assert sorted(deck) == list(range(NUM_CARDS))


def test_shuffle_deterministic_under_seed() -> None:
    assert dealing.shuffle_deck(random.Random(7)) == dealing.shuffle_deck(random.Random(7))


@given(st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=50, deadline=None)
def test_shuffle_permutation_property(seed: int) -> None:
    deck = dealing.shuffle_deck(random.Random(seed))
    assert sorted(deck) == list(range(NUM_CARDS))


def test_shuffle_uniformity_chi_square() -> None:
    """Each card should land in each position with near-equal frequency."""
    rng = random.Random(1234)
    counts = Counter()
    n = 400
    for _ in range(n):
        deck = dealing.shuffle_deck(rng)
        for pos, card in enumerate(deck):
            counts[(card, pos)] += 1
    expected = n / NUM_CARDS
    # With 400 trials per cell the mean is ~7.7; allow generous slack for a
    # uniformity smoke check (a broken shuffle would concentrate mass).
    for key, count in counts.items():
        assert count < expected * 4, f"card {key[0]} over-concentrated in position {key[1]}"


def test_first_hakem_uniform() -> None:
    rng = random.Random(99)
    counts = Counter(dealing.first_hakem(rng) for _ in range(4000))
    assert set(counts) == set(range(NUM_SEATS))
    for seat in range(NUM_SEATS):
        assert abs(counts[seat] - 1000) < 150, f"hakem seat {seat} not uniform: {counts}"


@pytest.mark.parametrize("hakem", range(NUM_SEATS))
def test_new_hand_totals(hakem: int) -> None:
    deck = dealing.shuffle_deck(random.Random(hakem))
    hand = dealing.new_hand(deck, hakem)
    for seat in range(NUM_SEATS):
        assert len(hand.hands[seat]) == CARDS_PER_PLAYER
    all_cards = [c for h in hand.hands for c in h]
    assert sorted(all_cards) == list(range(NUM_CARDS))


@pytest.mark.parametrize("hakem", range(NUM_SEATS))
def test_hakem_receives_opening_five(hakem: int) -> None:
    deck = dealing.shuffle_deck(random.Random(5))
    hands = dealing.deal_initial(deck, hakem)
    assert hands[hakem] == sorted(deck[:HAKEM_FIRST_BATCH])
    assert len(hands[hakem]) == HAKEM_FIRST_BATCH
    for seat in range(NUM_SEATS):
        if seat != hakem:
            assert hands[seat] == []


def test_hands_are_sorted() -> None:
    deck = dealing.shuffle_deck(random.Random(3))
    hand = dealing.new_hand(deck, 2)
    for h in hand.hands:
        assert h == sorted(h)


def test_remaining_quota_sums_to_47() -> None:
    quota = dealing.remaining_quota(1)
    assert sum(quota) == 47
    assert quota[1] == 8
    assert quota[0] == quota[2] == quota[3] == 13


def test_deal_uses_deck_exactly() -> None:
    """Every card id in the deck must appear exactly once across the hands."""
    deck = dealing.shuffle_deck(random.Random(11))
    hand = dealing.new_hand(deck, 3)
    dealt = sorted(c for h in hand.hands for c in h)
    assert dealt == list(range(NUM_CARDS))


def test_deal_rejects_inconsistent_deck() -> None:
    deck = dealing.shuffle_deck(random.Random(0))
    hands = dealing.deal_initial(deck, 0)
    # A short remaining deck must be caught (exhaustion check fires via totals).
    with pytest.raises(AssertionError):
        dealing.deal_remaining(deck[HAKEM_FIRST_BATCH:-1], 0, hands)
