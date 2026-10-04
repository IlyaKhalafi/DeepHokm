"""Compare algebraic trick risks to independent legal-hand enumeration."""

from itertools import combinations

import pytest

from deephokm.cards import NUM_CARDS, NUM_RANKS, SUIT_OF
from deephokm.policies.trick_odds import opponent_beating_probability
from deephokm.rules.tricks import resolve_trick


def _beats_in_resolved_trick(card: int, winner: int, led_suit: int, trump: int | None) -> bool:
    """Independent full-trick oracle, with fillers unable to beat the winner."""
    fillers = [
        other
        for other in range(NUM_CARDS)
        if SUIT_OF[other] not in (led_suit, trump) and other not in (card, winner)
    ]
    if SUIT_OF[winner] == led_suit:
        return resolve_trick([(0, winner), (1, card), (2, fillers[0]), (3, fillers[1])], trump) == 1
    lead = next(
        other for other in range(led_suit * NUM_RANKS, (led_suit + 1) * NUM_RANKS) if other != card
    )
    return resolve_trick([(0, lead), (1, winner), (2, card), (3, fillers[0])], trump) == 2


@pytest.mark.parametrize("hand_size", [1, 2, 3])
@pytest.mark.parametrize("voids", [frozenset(), frozenset({0}), frozenset({3})])
@pytest.mark.parametrize(
    ("winner", "led_suit", "trump"),
    [(5, 0, 3), (5, 0, None), (44, 0, 3), (44, 3, 3), (20, 1, 3)],
)
def test_estimate_matches_exhaustive_uniform_legal_hands(
    winner: int,
    led_suit: int,
    trump: int | None,
    voids: frozenset[int],
    hand_size: int,
) -> None:
    unseen = frozenset({0, 3, 8, 12, 13, 25, 26, 39, 41, 49, 51})
    pool = sorted(card for card in unseen if SUIT_OF[card] not in voids)
    possibilities = list(combinations(pool, hand_size))
    beaten = 0
    for held in possibilities:
        following = [card for card in held if SUIT_OF[card] == led_suit]
        legal = following or held
        beaten += any(_beats_in_resolved_trick(card, winner, led_suit, trump) for card in legal)
    probability = opponent_beating_probability(
        winner=winner,
        led_suit=led_suit,
        trump=trump,
        unseen_cards=unseen,
        void_suits=voids,
        hand_size=hand_size,
    )
    assert probability == pytest.approx(beaten / len(possibilities), abs=1e-12)


def test_higher_trump_cannot_overruff_when_the_opponent_must_follow() -> None:
    assert (
        opponent_beating_probability(
            winner=40,
            led_suit=0,
            trump=3,
            unseen_cards=frozenset({0, 51}),
            void_suits=frozenset(),
            hand_size=2,
        )
        == 0
    )


@pytest.mark.parametrize("hand_size", [0, 3])
def test_inconsistent_remaining_capacity_declines_estimate(hand_size: int) -> None:
    assert (
        opponent_beating_probability(
            winner=5,
            led_suit=0,
            trump=3,
            unseen_cards=frozenset({0, 1}),
            void_suits=frozenset(),
            hand_size=hand_size,
        )
        is None
    )
