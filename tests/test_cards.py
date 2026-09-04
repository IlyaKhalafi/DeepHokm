"""Tests for card primitives."""

from __future__ import annotations

import pytest

from deephokm.cards import (
    NUM_CARDS,
    NUM_RANKS,
    NUM_SUITS,
    Card,
    Rank,
    Suit,
    all_card_ids,
    card_from_id,
    card_id,
    card_name,
    rank_of,
    suit_of,
)


def test_suit_ids_are_zero_to_three() -> None:
    assert list(Suit) == [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
    assert [int(s) for s in Suit] == [0, 1, 2, 3]
    assert NUM_SUITS == 4


def test_rank_ordering_two_lowest_ace_highest() -> None:
    assert Rank.TWO == 0
    assert Rank.ACE == 12
    assert Rank.TWO < Rank.THREE < Rank.ACE
    assert NUM_RANKS == 13


def test_rank_symbols() -> None:
    assert Rank.TWO.symbol == "2"
    assert Rank.TEN.symbol == "10"
    assert Rank.JACK.symbol == "J"
    assert Rank.ACE.symbol == "A"


def test_card_id_roundtrip() -> None:
    for cid in all_card_ids():
        card = card_from_id(cid)
        assert card.id == cid
        assert card_id(card) == cid
        assert suit_of(cid) == card.suit
        assert rank_of(cid) == cid % NUM_RANKS


def test_card_id_formula() -> None:
    assert Card(Suit.SPADES, Rank.ACE).id == 13 * 3 + 12
    assert Card(Suit.CLUBS, Rank.TWO).id == 0


def test_card_from_id_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        card_from_id(-1)
    with pytest.raises(ValueError, match="out of range"):
        card_from_id(NUM_CARDS)


def test_all_card_ids_are_unique_and_total() -> None:
    ids = list(all_card_ids())
    assert len(ids) == NUM_CARDS == 52
    assert len(set(ids)) == 52


def test_card_str_and_name() -> None:
    card = Card(Suit.HEARTS, Rank.QUEEN)
    assert str(card) == "Q♥"
    assert card_name(card.id) == "Q♥"


def test_suit_colors() -> None:
    assert Suit.HEARTS.color == "red"
    assert Suit.DIAMONDS.color == "red"
    assert Suit.CLUBS.color == "black"
    assert Suit.SPADES.color == "black"


def test_cards_are_hashable_and_immutable() -> None:
    a = Card(Suit.CLUBS, Rank.TWO)
    b = Card(Suit.CLUBS, Rank.TWO)
    c = Card(Suit.CLUBS, Rank.THREE)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert len({a, b, c}) == 2
