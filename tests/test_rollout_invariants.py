"""Seeded full-match rollout invariants under masked random policies.

1000 matches are played; every engine invariant is asserted continuously:
each card played at most once per hand, a hand ends the moment a team takes
7 tricks, follow-suit never violated, trick counts stay consistent with the
hands actually completed, termination exactly when a team reaches 7 points,
and seeded reproducibility.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from deephokm.cards import NUM_CARDS, NUM_RANKS
from deephokm.rules import HokmEngine
from deephokm.rules.state import (
    HAKEM_FIRST_BATCH,
    POINTS_TO_WIN_MATCH,
    TRICKS_PER_HAND,
    TRICKS_TO_WIN_HAND,
    Phase,
)

N_MATCHES = 1000


def _check_card_play_invariants(seed: int, eng: HokmEngine, legal: list[int]) -> None:
    """Assert follow-suit and cards-in-hand on the legal action set."""
    hands = eng.state.hands
    seat = eng.current_seat()
    if hands.current_trick:
        led_suit = hands.current_trick[0][1] // NUM_RANKS
        own = [a for a in hands.hands[seat] if a // NUM_RANKS == led_suit]
        if own:
            assert set(legal) == set(own), (
                f"seed {seed}: follow-suit violation; legal={legal} expected={own}"
            )
        else:
            assert set(legal) == set(hands.hands[seat])
    else:
        assert set(legal) == set(hands.hands[seat])


def _check_hand_completion(
    seed: int,
    eng: HokmEngine,
    hand_cards: set[int],
    final_tally: list[int],
) -> set[int]:
    """Assert hand-end invariants; return a fresh empty card set.

    ``final_tally`` is the completed hand's trick tally, captured before the
    engine swapped in the next deal.
    """
    assert len(hand_cards) == 4 * sum(final_tally), (
        f"seed {seed}: hand ended with {len(hand_cards)} cards played "
        f"for tally {final_tally}"
    )
    assert TRICKS_TO_WIN_HAND <= sum(final_tally) <= TRICKS_PER_HAND
    assert max(final_tally) >= TRICKS_TO_WIN_HAND
    assert sum(1 for t in final_tally if t >= TRICKS_TO_WIN_HAND) == 1
    if eng.state.winner is not None:
        assert eng.state.game_points[eng.state.winner] == POINTS_TO_WIN_MATCH
        assert min(eng.state.game_points) < POINTS_TO_WIN_MATCH
    else:
        new_hakem = eng.state.hands.hakem
        for seat, hand in enumerate(eng.state.hands.hands):
            expected = HAKEM_FIRST_BATCH if seat == new_hakem else 0
            assert len(hand) == expected
        assert eng.state.hands.phase is Phase.TRUMP_CALL
    return set()


def play_match_with_invariants(seed: int) -> dict[str, float]:
    """Play one match under a masked random policy, asserting invariants.

    Returns a summary dict with hands played and final game points.
    """
    eng = HokmEngine(random.Random(seed))
    eng.start_match()

    hand_cards: set[int] = set()
    hands_played = 0
    tricks_this_hand = 0
    while eng.state.winner is None:
        hands = eng.state.hands
        # Snapshot the tally before acting: a hand-completing action replaces
        # the hand state with a fresh deal.
        tally_snapshot = list(hands.tricks_won)
        seat = eng.current_seat()
        legal = eng.legal_actions()
        assert legal, f"seed {seed}: no legal actions in phase {hands.phase.name}"

        if hands.phase is Phase.TRUMP_CALL:
            assert seat == hands.hakem
            assert all(52 <= a < 56 for a in legal)
        else:
            _check_card_play_invariants(seed, eng, legal)
        action = eng.rng.choice(legal)
        outcome = eng.apply_action(action)

        if outcome.card is not None:
            assert outcome.card not in hand_cards, (
                f"seed {seed}: card {outcome.card} played twice in one hand"
            )
            hand_cards.add(outcome.card)

        if outcome.trick_complete:
            tricks_this_hand += 1
            winner = outcome.trick_winner
            assert winner is not None
            if not outcome.hand_complete:
                # Mid-hand the trick winner leads the next trick; a
                # hand-completing trick triggers an immediate re-deal.
                assert sum(hands.tricks_won) == tricks_this_hand
                assert eng.current_seat() == winner

        if outcome.hand_complete:
            hands_played += 1
            assert outcome.hand_winner_team is not None
            final_tally = list(tally_snapshot)
            assert outcome.trick_winner is not None
            final_tally[outcome.hand_winner_team] += 1
            hand_cards = _check_hand_completion(seed, eng, hand_cards, final_tally)
            tricks_this_hand = 0

    assert eng.state.winner in (0, 1)
    assert hands_played >= POINTS_TO_WIN_MATCH, (
        f"seed {seed}: match ended after {hands_played} hands"
    )
    return {
        "hands": hands_played,
        "team0_points": eng.state.game_points[0],
        "team1_points": eng.state.game_points[1],
    }


def test_1000_seeded_rollouts_hold_all_invariants() -> None:
    summaries = [play_match_with_invariants(seed) for seed in range(N_MATCHES)]
    hands = [s["hands"] for s in summaries]
    assert all(h >= POINTS_TO_WIN_MATCH for h in hands)
    # A random-vs-random match needs on average more than 7 hands; sanity-check
    # the distribution is neither degenerate nor runaway.
    assert min(hands) >= POINTS_TO_WIN_MATCH
    assert max(hands) < 200, f"suspiciously long matches: {max(hands)}"


def test_rolled_out_matches_terminate_correctly() -> None:
    """Termination occurs exactly when a team reaches 7 game points."""
    for seed in range(100):
        eng = HokmEngine(random.Random(seed))
        eng.start_match()
        previous_points = (0, 0)
        while eng.state.winner is None:
            assert max(eng.state.game_points) < POINTS_TO_WIN_MATCH
            assert tuple(eng.state.game_points) >= previous_points
            previous_points = tuple(eng.state.game_points)
            eng.apply_action(eng.rng.choice(eng.legal_actions()))
        assert max(eng.state.game_points) == POINTS_TO_WIN_MATCH
        assert eng.state.winner == (0 if eng.state.game_points[0] == POINTS_TO_WIN_MATCH else 1)


def test_rollout_reproducibility() -> None:
    """Same seed => identical action sequence across independent engines."""

    def trace(seed: int) -> list[int]:
        eng = HokmEngine(random.Random(seed))
        eng.start_match()
        actions = []
        while eng.state.winner is None:
            action = eng.rng.choice(eng.legal_actions())
            actions.append(action)
            eng.apply_action(action)
        return actions

    for seed in (0, 1, 42, 999):
        assert trace(seed) == trace(seed)


def test_team_win_rates_are_balanced_under_random_play() -> None:
    """Random-vs-random wins should be near 50/50 across many matches."""
    wins = Counter()
    for seed in range(200):
        eng = HokmEngine(random.Random(seed))
        eng.start_match()
        while eng.state.winner is None:
            eng.apply_action(eng.rng.choice(eng.legal_actions()))
        assert eng.state.winner is not None
        wins[eng.state.winner] += 1
    # Two-sided check at a generous band: a broken engine would skew heavily.
    assert 70 <= wins[0] <= 130, f"team win rates skewed: {dict(wins)}"


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_every_card_appears_in_exactly_one_hand_per_match(seed: int) -> None:
    """Across all hands of a match, each deal partitions the full deck."""
    eng = HokmEngine(random.Random(seed))
    eng.start_match()
    deals_seen = 0
    while eng.state.winner is None:
        if eng.state.hands.phase is Phase.TRUMP_CALL:
            deals_seen += 1
            # Before trump is declared only the hakem's opening 5 are dealt;
            # the other 47 sit in pending_deck. The full deck must still
            # partition exactly, across hands + pending_deck together.
            all_cards = [c for h in eng.state.hands.hands for c in h]
            all_cards += eng.state.hands.pending_deck
            assert len(all_cards) == NUM_CARDS
            assert len(set(all_cards)) == NUM_CARDS
            new_hakem = eng.state.hands.hakem
            for seat, hand in enumerate(eng.state.hands.hands):
                expected = HAKEM_FIRST_BATCH if seat == new_hakem else 0
                assert len(hand) == expected
        eng.apply_action(eng.rng.choice(eng.legal_actions()))
    assert deals_seen >= POINTS_TO_WIN_MATCH
