"""Hand scoring, match termination, and hakem rotation."""

from __future__ import annotations

from deephokm.rules.state import (
    NUM_SEATS,
    POINTS_TO_WIN_MATCH,
    TRICKS_PER_HAND,
    TRICKS_TO_WIN_HAND,
    MatchState,
)


def hand_winner_team(tricks_won: list[int]) -> int | None:
    """Return the team index that won the hand, or ``None`` while undecided.

    A team wins the hand as soon as it captures
    :data:`TRICKS_TO_WIN_HAND` tricks; with 13 tricks total exactly one team
    can reach 7.
    """
    for team in range(2):
        if tricks_won[team] >= TRICKS_TO_WIN_HAND:
            return team
    return None


def score_hand(tricks_won: list[int]) -> int:
    """Return the game point awarded for a completed hand (0 or 1 per team).

    A hand ends the moment a team reaches ``TRICKS_TO_WIN_HAND``, so the
    tally can sum to anything from 7 to 13; what is required is that the
    played tricks are a plausible prefix of a hand (no more than 13) and
    that a team has actually crossed the line.

    Raises:
        ValueError: If more than 13 tricks are recorded or neither team won.
    """
    if sum(tricks_won) > TRICKS_PER_HAND:
        raise ValueError(f"trick counts {tricks_won} exceed {TRICKS_PER_HAND}")
    winner = hand_winner_team(tricks_won)
    if winner is None:
        raise ValueError(f"no team reached {TRICKS_TO_WIN_HAND} tricks: {tricks_won}")
    return winner


def award_game_point(match: MatchState, team: int) -> None:
    """Add one game point to ``team`` and finalize the match at 7 points."""
    if team not in (0, 1):
        raise ValueError(f"team must be 0 or 1, got {team}")
    match.game_points[team] += 1
    if match.game_points[team] >= POINTS_TO_WIN_MATCH:
        match.winner = team


def next_hakem(old_hakem: int, hand_winner_team_idx: int) -> int:
    """Return the next hakem seat.

    If the hakem's team won the hand the hakem stays; otherwise the hakem
    passes to ``(old_hakem + 1) % 4``.
    """
    hakem_team = old_hakem % 2
    if hand_winner_team_idx == hakem_team:
        return old_hakem
    return (old_hakem + 1) % NUM_SEATS


def match_is_over(match: MatchState) -> bool:
    """Return whether a team has reached the match-winning point total."""
    return match.winner is not None
