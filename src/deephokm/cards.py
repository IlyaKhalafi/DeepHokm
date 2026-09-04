"""Card, suit, rank and deck primitives for Hokm.

Cards are represented at the API surface by the :class:`Card` dataclass and
internally, in the hot paths, by integer ids in ``0..51`` where
``card_id = 13 * suit_id + rank_index``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

NUM_SUITS = 4
NUM_RANKS = 13
NUM_CARDS = NUM_SUITS * NUM_RANKS  # 52

PAD_TOKEN = NUM_CARDS  # embedding id 52 reserved for padding


class Suit(IntEnum):
    """The four suits, ordered clubs, diamonds, hearts, spades (ids 0-3)."""

    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    @property
    def symbol(self) -> str:
        """Unicode suit symbol used by renderers."""
        return _SUIT_SYMBOLS[self]

    @property
    def color(self) -> str:
        """Display color class of the suit."""
        return _SUIT_COLORS[self]


_SUIT_SYMBOLS = {
    Suit.CLUBS: "\N{BLACK CLUB SUIT}",
    Suit.DIAMONDS: "\N{BLACK DIAMOND SUIT}",
    Suit.HEARTS: "\N{BLACK HEART SUIT}",
    Suit.SPADES: "\N{BLACK SPADE SUIT}",
}

_SUIT_COLORS = {
    Suit.CLUBS: "black",
    Suit.DIAMONDS: "red",
    Suit.HEARTS: "red",
    Suit.SPADES: "black",
}


class Rank(IntEnum):
    """Card ranks with 2 lowest and ace highest (rank index 0-12)."""

    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12

    @property
    def symbol(self) -> str:
        """Human-readable rank label (2-10, J, Q, K, A)."""
        return _RANK_SYMBOLS[self]


_RANK_SYMBOLS = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "10",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}


@dataclass(frozen=True, slots=True)
class Card:
    """An immutable playing card.

    Attributes:
        suit: The card suit.
        rank: The card rank.
    """

    suit: Suit
    rank: Rank

    @property
    def id(self) -> int:
        """Integer id in ``0..51`` (``13 * suit + rank``)."""
        return NUM_RANKS * self.suit + self.rank

    def __str__(self) -> str:
        return f"{self.rank.symbol}{self.suit.symbol}"


def card_id(card: Card) -> int:
    """Return the integer id of a card."""
    return card.id


def card_from_id(card_id: int) -> Card:
    """Return the :class:`Card` for an integer id in ``0..51``."""
    if not 0 <= card_id < NUM_CARDS:
        raise ValueError(f"card id {card_id} out of range [0, {NUM_CARDS})")
    return Card(suit=Suit(card_id // NUM_RANKS), rank=Rank(card_id % NUM_RANKS))


def suit_of(card_id: int) -> Suit:
    """Return the suit of a card id without allocating a :class:`Card`."""
    return Suit(validate_card_id(card_id) // NUM_RANKS)


def rank_of(card_id: int) -> Rank:
    """Return the rank of a card id without allocating a :class:`Card`."""
    return Rank(validate_card_id(card_id) % NUM_RANKS)


def validate_card_id(card_id: int) -> int:
    """Return ``card_id`` if it names a real card, else raise ``ValueError``."""
    if not 0 <= card_id < NUM_CARDS:
        raise ValueError(f"card id {card_id} out of range [0, {NUM_CARDS})")
    return card_id


def card_name(card_id: int) -> str:
    """Return the human-readable name of a card id, e.g. ``'Q♠'``."""
    return f"{Rank(card_id % NUM_RANKS).symbol}{Suit(card_id // NUM_RANKS).symbol}"


def all_card_ids() -> range:
    """Return an iterable over all 52 card ids in id order."""
    return range(NUM_CARDS)
