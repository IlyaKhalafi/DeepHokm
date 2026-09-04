"""Tests for hand scoring, match termination, and hakem rotation."""

from __future__ import annotations

import pytest

from deephokm.rules import scoring
from deephokm.rules.state import (
    POINTS_TO_WIN_MATCH,
    TRICKS_PER_HAND,
    TRICKS_TO_WIN_HAND,
    HandState,
    MatchState,
    team_of,
    teammate,
)


def test_hand_winner_at_seven_tricks() -> None:
    assert scoring.hand_winner_team([7, 6]) == 0
    assert scoring.hand_winner_team([6, 7]) == 1
    assert scoring.hand_winner_team([13, 0]) == 0
    assert scoring.hand_winner_team([0, 13]) == 1


def test_hand_winner_undecided_below_seven() -> None:
    assert scoring.hand_winner_team([6, 6]) is None
    assert scoring.hand_winner_team([0, 0]) is None


def test_score_hand_returns_winning_team() -> None:
    assert scoring.score_hand([7, 6]) == 0
    assert scoring.score_hand([5, 8]) == 1


def test_score_hand_rejects_bad_totals() -> None:
    with pytest.raises(ValueError, match="do not sum"):
        scoring.score_hand([7, 7])
    with pytest.raises(ValueError, match="do not sum"):
        scoring.score_hand([6, 6])


def test_score_hand_no_winner_guard_is_unreachable_with_valid_sum() -> None:
    """With 13 tricks split between two teams, one team always has >= 7.

    The no-winner ValueError guard in score_hand is therefore defensive only;
    hand_winner_team is the function that exposes the undecided state.
    """
    for a in range(14):
        if scoring.hand_winner_team([a, 13 - a]) is None:
            raise AssertionError(f"split [{a}, {13 - a}] has no winner")
    # The guard still fires for the malformed-then-undecided path: a total of
    # 13 is required first, so a no-winner split can never reach it.
    assert scoring.hand_winner_team([6, 6]) is None


def _empty_match() -> MatchState:
    """A MatchState with a placeholder hand (scoring tests never touch it)."""
    return MatchState(hands=HandState(hands=[[], [], [], []], hakem=0))


def test_award_game_point_accumulates() -> None:
    match = _empty_match()
    scoring.award_game_point(match, 0)
    scoring.award_game_point(match, 0)
    assert match.game_points == [2, 0]
    assert match.winner is None


def test_award_game_point_finalizes_at_seven() -> None:
    match = _empty_match()
    for _ in range(POINTS_TO_WIN_MATCH - 1):
        scoring.award_game_point(match, 1)
    assert match.winner is None
    scoring.award_game_point(match, 1)
    assert match.winner == 1
    assert scoring.match_is_over(match)


def test_award_game_point_rejects_bad_team() -> None:
    match = _empty_match()
    with pytest.raises(ValueError, match="team must be 0 or 1"):
        scoring.award_game_point(match, 2)


@pytest.mark.parametrize(
    ("old", "winner", "expected"),
    [
        (0, 0, 0),  # hakem's team won: hakem stays
        (0, 1, 1),  # hakem's team lost: next seat
        (1, 1, 1),
        (1, 0, 2),
        (2, 0, 2),
        (2, 1, 3),
        (3, 1, 3),
        (3, 0, 0),
    ],
)
def test_next_hakem_rotation(old: int, winner: int, expected: int) -> None:
    assert scoring.next_hakem(old, winner) == expected


def test_constants() -> None:
    assert TRICKS_PER_HAND == 13
    assert TRICKS_TO_WIN_HAND == 7
    assert POINTS_TO_WIN_MATCH == 7


def test_team_and_teammate() -> None:
    assert team_of(0) == team_of(2) == 0
    assert team_of(1) == team_of(3) == 1
    assert teammate(0) == 2
    assert teammate(1) == 3
    assert teammate(teammate(0)) == 0
