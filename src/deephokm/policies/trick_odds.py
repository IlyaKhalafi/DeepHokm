"""Small algebraic risk estimates, not sampled worlds or hidden-hand reads."""

from __future__ import annotations

from math import comb

from deephokm.cards import SUIT_OF


def opponent_beating_probability(
    *,
    winner: int,
    led_suit: int,
    trump: int | None,
    unseen_cards: frozenset[int],
    void_suits: frozenset[int],
    hand_size: int,
) -> float | None:
    """Estimate whether one opponent could legally beat a winning card.

    Assume a uniform hand of ``hand_size`` from unseen cards compatible with
    this seat's proven voids. Other seats' ownership correlations are ignored;
    this is a heuristic probability, not a public-information guarantee.

    Higher led-suit cards can beat immediately. Ruffing requires *no* led-suit
    card. Likewise, overruffing a trump played on a side-suit lead requires
    the opponent to be void in that side suit. These disjoint events have
    exact hypergeometric probabilities under the stated approximation.
    """
    pool = tuple(card for card in unseen_cards if SUIT_OF[card] not in void_suits)
    count = len(pool)
    winner_suit = SUIT_OF[winner]
    if hand_size <= 0 or count < hand_size or winner_suit not in (led_suit, trump):
        return None
    denominator = comb(count, hand_size)

    def probability_from(available: int) -> float:
        return comb(available, hand_size) / denominator if available >= hand_size else 0.0

    led_count = sum(SUIT_OF[card] == led_suit for card in pool)
    higher_count = sum(SUIT_OF[card] == winner_suit and card > winner for card in pool)
    if winner_suit == trump and led_suit != trump:
        # A higher trump is unusable while the seat can follow the side suit.
        beating = probability_from(count - led_count) - probability_from(
            count - led_count - higher_count
        )
    else:
        beating = 1.0 - probability_from(count - higher_count)
        if trump is not None and winner_suit != trump:
            trump_count = sum(SUIT_OF[card] == trump for card in pool)
            beating += probability_from(count - led_count) - probability_from(
                count - led_count - trump_count
            )
    return max(0.0, min(1.0, beating))
