"""Action legality: which card plays and trump declarations are legal.

Actions are integers in ``0..55``: 0-51 play the card with that id, 52-55
declare trump suit 0-3. Trump declaration is legal exactly at the hakem's
trump-call decision point.
"""

from __future__ import annotations

from deephokm.cards import NUM_CARDS, NUM_RANKS, NUM_SUITS
from deephokm.rules.state import HandState, Phase

NUM_ACTIONS = NUM_CARDS + NUM_SUITS  # 56
TRUMP_ACTION_OFFSET = NUM_CARDS  # 52


def trump_action(suit: int) -> int:
    """Return the action id that declares trump suit ``suit``."""
    return TRUMP_ACTION_OFFSET + suit


def trump_action_to_suit(action: int) -> int:
    """Return the suit declared by trump action id ``action``."""
    return action - TRUMP_ACTION_OFFSET


def is_trump_action(action: int) -> bool:
    """Return whether ``action`` is a trump-declaration action (52-55)."""
    return TRUMP_ACTION_OFFSET <= action < NUM_ACTIONS


def legal_actions(hand: HandState, seat: int) -> list[int]:
    """Return the list of legal action ids for ``seat`` in a hand state.

    At the trump-call point only the hakem may act, and only with the four
    trump actions. During card play the mask enforces cards-in-hand and
    follow-suit: a player must follow the led suit when able, otherwise any
    card in hand is legal. Outside the acting phases nothing is legal.

    Args:
        hand: The live hand state.
        seat: The acting seat.
    """
    if hand.phase is Phase.TRUMP_CALL:
        if seat != hand.hakem or hand.trump is not None:
            return []
        return [trump_action(s) for s in range(NUM_SUITS)]
    if hand.phase is not Phase.CARD_PLAY:
        return []
    cards = hand.hands[seat]
    if not hand.current_trick:
        # Leading: any card in hand.
        return list(cards)
    led_suit = hand.current_trick[0][1] // NUM_RANKS
    followers = [c for c in cards if c // NUM_RANKS == led_suit]
    if followers:
        return followers
    return list(cards)


def legal_actions_mask(hand: HandState, seat: int) -> list[bool]:
    """Return a 56-entry boolean mask of legal actions for ``seat``.

    See :func:`legal_actions` for the semantics.
    """
    mask = [False] * NUM_ACTIONS
    for action in legal_actions(hand, seat):
        mask[action] = True
    return mask


def is_legal(action: int, hand: HandState, seat: int) -> bool:
    """Return whether ``action`` is legal for ``seat`` in the hand state."""
    if not 0 <= action < NUM_ACTIONS:
        return False
    return action in legal_actions(hand, seat)
