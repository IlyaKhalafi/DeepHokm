"""Tests for action legality and masking."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from deephokm.cards import NUM_CARDS, NUM_SUITS
from deephokm.rules import legality
from deephokm.rules.legality import NUM_ACTIONS
from deephokm.rules.state import HandState, Phase


def make_hand(
    hands: list[list[int]],
    *,
    phase: Phase = Phase.CARD_PLAY,
    hakem: int = 0,
    trump: int | None = None,
    current_trick: list[tuple[int, int]] | None = None,
    leader: int = -1,
) -> HandState:
    """Build a HandState for legality tests."""
    return HandState(
        hands=hands,
        hakem=hakem,
        trump=trump,
        phase=phase,
        current_trick=current_trick or [],
        leader=leader,
    )


def test_trump_action_encoding() -> None:
    for suit in range(NUM_SUITS):
        action = legality.trump_action(suit)
        assert action == NUM_CARDS + suit
        assert legality.is_trump_action(action)
        assert legality.trump_action_to_suit(action) == suit
    assert not legality.is_trump_action(NUM_CARDS - 1)
    assert NUM_ACTIONS == 56


def test_trump_call_only_hakem_acts() -> None:
    for seat in range(4):
        hand = make_hand([[0, 1, 2]] * 4, phase=Phase.TRUMP_CALL, hakem=0)
        actions = legality.legal_actions(hand, seat)
        if seat == 0:
            assert actions == [52, 53, 54, 55]
        else:
            assert actions == []


def test_trump_call_blocked_after_declaration() -> None:
    hand = make_hand([[0, 1, 2]] * 4, phase=Phase.TRUMP_CALL, hakem=0, trump=2)
    assert legality.legal_actions(hand, 0) == []


def test_leading_any_card_in_hand() -> None:
    cards = [5, 17, 30]
    hand = make_hand([cards, [], [], []], trump=1, leader=1)
    assert legality.legal_actions(hand, 0) == sorted(cards)


def test_follow_suit_enforced() -> None:
    cards = [0, 1, 13, 26]  # 2♣, 3♣, 2♦, 2♥
    hand = make_hand([cards, [], [], []], trump=1, current_trick=[(2, 0)])
    assert legality.legal_actions(hand, 0) == [0, 1]


def test_any_card_when_void_in_led_suit() -> None:
    cards = [13, 14, 26]  # diamonds and hearts only
    hand = make_hand([cards, [], [], []], trump=1, current_trick=[(2, 0)])
    assert legality.legal_actions(hand, 0) == sorted(cards)


def test_follow_suit_even_when_trump_available() -> None:
    """Holding the led suit forces following it; trump cannot be played off-suit."""
    cards = [13, 51]  # 2♦, A♠
    hand = make_hand([cards, [], [], []], trump=3, current_trick=[(2, 13)])
    assert legality.legal_actions(hand, 0) == [13]


def test_trump_playable_when_void_in_led_suit() -> None:
    cards = [39, 40]  # spades only
    hand = make_hand([cards, [], [], []], trump=3, current_trick=[(2, 0)])
    assert legality.legal_actions(hand, 0) == sorted(cards)


def test_no_actions_when_hand_empty_or_terminal() -> None:
    empty = make_hand([[], [], [], []], trump=1, current_trick=[(1, 5)])
    assert legality.legal_actions(empty, 0) == []
    over = make_hand([[1, 2]] * 4, phase=Phase.HAND_OVER)
    assert legality.legal_actions(over, 0) == []
    match_over = make_hand([[1, 2]] * 4, phase=Phase.MATCH_OVER)
    assert legality.legal_actions(match_over, 0) == []


def test_mask_matches_action_list() -> None:
    hand = make_hand([[3, 14, 27], [], [], []], trump=0, current_trick=[(0, 1)])
    mask = legality.legal_actions_mask(hand, 2)
    actions = legality.legal_actions(hand, 2)
    assert len(mask) == NUM_ACTIONS
    assert [i for i, m in enumerate(mask) if m] == actions


def test_is_legal_rejects_out_of_range() -> None:
    hand = make_hand([[0], [], [], []], phase=Phase.TRUMP_CALL)
    assert not legality.is_legal(-1, hand, 0)
    assert not legality.is_legal(NUM_ACTIONS, hand, 0)


@given(
    cards=st.lists(st.integers(min_value=0, max_value=51), min_size=0, max_size=13),
    trump=st.one_of(st.none(), st.integers(min_value=0, max_value=3)),
    trick=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=3),
            st.integers(min_value=0, max_value=51),
        ),
        min_size=0,
        max_size=4,
    ),
)
@settings(max_examples=200, deadline=None)
def test_legal_card_actions_are_subset_of_hand(
    cards: list[int], trump: int | None, trick: list[tuple[int, int]]
) -> None:
    """Every card action must be a card in the acting seat's hand."""
    hand = make_hand([sorted(set(cards)), [], [], []], trump=trump, current_trick=trick)
    for action in legality.legal_actions(hand, 0):
        assert action in hand.hands[0]
