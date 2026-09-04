"""Tests for the HokmEngine state machine."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from deephokm.cards import NUM_CARDS, NUM_SUITS
from deephokm.rules import HokmEngine, legality
from deephokm.rules.state import (
    CARDS_PER_PLAYER,
    NUM_SEATS,
    TRICKS_PER_HAND,
    TRICKS_TO_WIN_HAND,
    Phase,
    team_of,
)


def make_engine(seed: int = 0) -> HokmEngine:
    eng = HokmEngine(random.Random(seed))
    eng.start_match()
    return eng


def play_random_match(seed: int) -> HokmEngine:
    """Play a full random-legal match and return the engine."""
    eng = make_engine(seed)
    while eng.state.winner is None:
        action = eng.rng.choice(eng.legal_actions())
        eng.apply_action(action)
    return eng


def test_start_match_deals_and_sets_trump_call() -> None:
    eng = make_engine(0)
    assert eng.state.hands.phase is Phase.TRUMP_CALL
    assert eng.state.hakem == eng.state.hands.hakem
    for hand in eng.state.hands.hands:
        assert len(hand) == CARDS_PER_PLAYER
    assert eng.current_seat() == eng.state.hakem


def test_first_seat_is_hakem_for_trump() -> None:
    eng = make_engine(1)
    assert eng.current_seat() == eng.state.hands.hakem


def test_trump_declaration_transitions_phase() -> None:
    eng = make_engine(2)
    hakem = eng.current_seat()
    suit = 3
    outcome = eng.apply_action(legality.trump_action(suit), hakem)
    assert eng.state.hands.trump == suit
    assert eng.state.hands.phase is Phase.CARD_PLAY
    assert eng.state.hands.leader == hakem
    assert outcome.trump == suit
    assert eng.current_seat() == hakem  # hakem leads the first trick


def test_illegal_action_raises() -> None:
    eng = make_engine(3)
    # Card play during trump call is illegal.
    card = eng.state.hands.hands[eng.current_seat()][0]
    with pytest.raises(ValueError, match="illegal action"):
        eng.apply_action(card)
    # Trump action by the non-hakem seat (wrong turn) is rejected.
    with pytest.raises(ValueError, match="cannot act"):
        eng.apply_action(52, seat=(eng.current_seat() + 1) % NUM_SEATS)


def test_card_not_in_hand_raises() -> None:
    eng = make_engine(4)
    eng.apply_action(52)
    hand = eng.state.hands.hands[eng.current_seat()]
    missing = next(c for c in range(NUM_CARDS) if c not in hand)
    with pytest.raises(ValueError, match="illegal action"):
        eng.apply_action(missing)


def test_remove_card_direct_reports_missing_card() -> None:
    """The HandState.remove_card guard fires when bypassing the mask."""
    eng = make_engine(4)
    hand_state = eng.state.hands
    held = hand_state.hands[0][0]
    hand_state.remove_card(0, held)
    with pytest.raises(ValueError, match="does not hold"):
        hand_state.remove_card(0, held)


def test_playing_removes_card_from_hand() -> None:
    eng = make_engine(5)
    eng.apply_action(52)
    seat = eng.current_seat()
    card = eng.state.hands.hands[seat][0]
    hand_len = len(eng.state.hands.hands[seat])
    eng.apply_action(card)
    assert len(eng.state.hands.hands[seat]) == hand_len - 1
    assert card not in eng.state.hands.hands[seat]
    assert eng.state.hands.on_table() == [card]


def test_actions_after_match_over_raise() -> None:
    eng = play_random_match(6)
    assert eng.state.winner is not None
    with pytest.raises(RuntimeError, match="match is over"):
        eng.apply_action(52)


def test_engine_random_match_completes() -> None:
    for seed in range(20):
        eng = play_random_match(seed)
        assert eng.state.winner in (0, 1)
        assert max(eng.state.game_points) == 7
        assert min(eng.state.game_points) < 7


def test_match_reproducible_from_seed() -> None:
    def trace(seed: int) -> list[tuple[int, int]]:
        eng = HokmEngine(random.Random(seed))
        eng.start_match()
        steps = []
        while eng.state.winner is None:
            seat = eng.current_seat()
            action = eng.rng.choice(eng.legal_actions())
            steps.append((seat, action))
            eng.apply_action(action)
        return steps

    assert trace(1234) == trace(1234)
    assert trace(1234) != trace(1235)


def test_trick_winner_leads_next() -> None:
    eng = make_engine(7)
    eng.apply_action(52)
    # Play out one trick, tracking who won.
    winner = None
    for _ in range(NUM_SEATS):
        action = eng.rng.choice(eng.legal_actions())
        outcome = eng.apply_action(action)
        if outcome.trick_complete:
            winner = outcome.trick_winner
    assert winner is not None
    assert eng.current_seat() == winner
    assert eng.state.hands.leader == winner


def test_hand_completion_transitions_and_deals() -> None:
    eng = make_engine(8)
    outcome = None
    tricks_before: list[int] = []
    while eng.state.winner is None:
        # Snapshot the live hand's trick tally so post-hand assertions can
        # inspect it even after the engine swaps in a fresh deal.
        if len(eng.state.hands.trick_winners) == TRICKS_PER_HAND - 1:
            tricks_before = list(eng.state.hands.tricks_won)
        outcome = eng.apply_action(eng.rng.choice(eng.legal_actions()))
        if outcome is not None and outcome.hand_complete:
            break
    assert outcome is not None and outcome.hand_complete
    assert outcome.hand_winner_team is not None
    # The final trick decided the hand: the winner reached at least 7 tricks
    # and the loser stayed below it.
    final_tally = list(tricks_before)
    final_tally[outcome.hand_winner_team] += 1
    assert sum(final_tally) == TRICKS_PER_HAND
    assert final_tally[outcome.hand_winner_team] >= TRICKS_TO_WIN_HAND
    assert final_tally[1 - outcome.hand_winner_team] < TRICKS_TO_WIN_HAND
    if eng.state.winner is None:
        assert eng.state.hands.phase is Phase.TRUMP_CALL
        for hand in eng.state.hands.hands:
            assert len(hand) == CARDS_PER_PLAYER
        assert eng.state.hand_number == 2


def test_hakem_rotation_matches_rules() -> None:
    """Hakem stays if their team won the hand, else passes to the next seat."""
    for seed in range(30):
        eng = make_engine(seed)
        old_hakem = eng.state.hakem
        eng.apply_action(legality.trump_action(eng.rng.randrange(NUM_SUITS)))
        while eng.state.hands.phase is Phase.CARD_PLAY and eng.state.winner is None:
            outcome = eng.apply_action(eng.rng.choice(eng.legal_actions()))
            if outcome.hand_complete:
                assert outcome.hand_winner_team is not None
                winner_team = outcome.hand_winner_team
                if eng.state.winner is None:
                    expected = (
                        old_hakem
                        if team_of(old_hakem) == winner_team
                        else (old_hakem + 1) % NUM_SEATS
                    )
                    assert eng.state.hakem == expected
                break


def test_engine_after_hand_over_phase_is_immediately_replaced() -> None:
    """The HAND_OVER phase is transient: the engine re-deals within apply_action."""
    eng = make_engine(9)
    eng.apply_action(52)
    seen_hand_over = False
    while eng.state.winner is None:
        outcome = eng.apply_action(eng.rng.choice(eng.legal_actions()))
        if outcome.hand_complete:
            seen_hand_over = True
            if eng.state.winner is None:
                assert eng.state.hands.phase is Phase.TRUMP_CALL
            break
    assert seen_hand_over


def test_match_winner_receives_exactly_seven_points() -> None:
    for seed in (10, 11, 12):
        eng = play_random_match(seed)
        winner = eng.state.winner
        assert eng.state.game_points[winner] == 7
        assert eng.state.game_points[1 - winner] < 7


def test_outcome_reports_card_and_seat() -> None:
    eng = make_engine(13)
    eng.apply_action(52)
    seat = eng.current_seat()
    card = eng.state.hands.hands[seat][-1]
    outcome = eng.apply_action(card)
    assert outcome.seat == seat
    assert outcome.card == card
    assert outcome.trick_complete is False


def test_start_match_with_explicit_seed_matches_constructed_rng() -> None:
    """start_match(seed) must equal HokmEngine(Random(seed)).start_match()."""
    a = HokmEngine(random.Random(1))
    a.start_match(seed=777)
    b = HokmEngine(random.Random(777))
    b.start_match()
    assert a.state.hakem == b.state.hakem
    assert [h[:] for h in a.state.hands.hands] == [h[:] for h in b.state.hands.hands]


def test_start_match_reuses_rng_stream_without_seed() -> None:
    eng = HokmEngine(random.Random(5))
    eng.start_match()
    first_hakem = eng.state.hakem
    eng.start_match()
    # The second deal draws from the same stream; determinism is preserved.
    again = HokmEngine(random.Random(5))
    again.start_match()
    again.start_match()
    assert eng.state.hakem == again.state.hakem
    assert eng.state.hakem is not None
    del first_hakem


def test_is_legal_reflects_mask() -> None:
    eng = make_engine(14)
    legal = eng.legal_actions()
    for action in range(56):
        assert eng.is_legal(action) == (action in legal)
    # Explicit non-current seat is judged against that seat's options.
    other = (eng.current_seat() + 1) % NUM_SEATS
    assert eng.is_legal(52, seat=other) is False


def test_current_seat_raises_when_match_over() -> None:
    eng = play_random_match(15)
    with pytest.raises(RuntimeError, match="no acting seat"):
        eng.current_seat()


def test_partner_of_reexport() -> None:
    from deephokm.rules.engine import partner_of  # noqa: PLC0415

    assert partner_of(0) == 2
    assert partner_of(3) == 1


def test_first_hakem_is_uniformly_random_across_seeds() -> None:
    """The first hakem of a match must be a uniform random seat per seed."""
    counts: Counter[int] = Counter()
    for seed in range(400):
        eng = HokmEngine(random.Random(seed))
        eng.start_match()
        counts[eng.state.hakem] += 1
    assert set(counts) == {0, 1, 2, 3}
    for seat, count in counts.items():
        assert 70 <= count <= 130, f"seat {seat} selected {count}/400 times"
