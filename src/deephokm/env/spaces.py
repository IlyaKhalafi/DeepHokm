"""Observation and action space definitions for HokmEnv.

The observation is a Dict space from the acting player's perspective. The
network's tokenizer and the tests both depend on the layout, so any change
here must update the feature extractor and tests in the same commit.

``history`` is the one addition to the base layout: the binary ``seen``
vector records *which* cards have been played but not *when*, and a
trick-taking policy needs play order to read the table. It carries the
completed tricks of the current hand in reverse play order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from gymnasium import spaces

from deephokm.cards import NUM_CARDS, NUM_SUITS
from deephokm.rules.legality import NUM_ACTIONS
from deephokm.rules.state import NUM_SEATS, TRICKS_PER_HAND, Phase, team_of

if TYPE_CHECKING:  # pragma: no cover
    from deephokm.rules.state import HandState

# Cards from completed tricks in one hand: 12 finished tricks at most, because
# the thirteenth ends the hand. The current trick has its own observation slot.
HISTORY_SLOTS = (TRICKS_PER_HAND - 1) * NUM_SEATS


class Observation(TypedDict):
    """The observation dict handed to the acting player.

    Attributes:
        hand: MultiBinary(52) — cards currently in the acting player's hand.
        seen: MultiBinary(52) — own hand plus every card played this hand.
        trick: MultiBinary(52) — cards currently on the table.
        trick_play: Box(4) — card id (or -1) played by each seat in the
            current trick, in seat order; -1 for seats yet to play. The leader
            is the first seat with a card.
        history: Box(48) — card ids from this hand's completed tricks in
            reverse play order (most recent first), -1 padded. Play order is
            not recoverable from ``seen``, so it is carried explicitly.
        trump: MultiBinary(4) — one-hot trump suit; all zeros before declared.
        phase: MultiBinary(2) — [trump-call, card-play].
        tricks_won: Box(2) — [own team, opposing team] tricks this hand.
        game_points: Box(2) — [own team, opposing team] match points.
        seat: MultiBinary(4) — one-hot acting seat.
    """

    hand: np.ndarray
    seen: np.ndarray
    trick: np.ndarray
    trick_play: np.ndarray
    history: np.ndarray
    trump: np.ndarray
    phase: np.ndarray
    tricks_won: np.ndarray
    game_points: np.ndarray
    seat: np.ndarray


def observation_space() -> spaces.Dict:
    """Return the observation space (identical for every seat)."""
    return spaces.Dict(
        {
            "hand": spaces.MultiBinary(NUM_CARDS),
            "seen": spaces.MultiBinary(NUM_CARDS),
            "trick": spaces.MultiBinary(NUM_CARDS),
            "trick_play": spaces.Box(
                low=-1, high=NUM_CARDS - 1, shape=(NUM_SEATS,), dtype=np.int64
            ),
            "history": spaces.Box(
                low=-1, high=NUM_CARDS - 1, shape=(HISTORY_SLOTS,), dtype=np.int64
            ),
            "trump": spaces.MultiBinary(NUM_SUITS),
            "phase": spaces.MultiBinary(2),
            "tricks_won": spaces.Box(low=0, high=13, shape=(2,), dtype=np.int64),
            "game_points": spaces.Box(low=0, high=7, shape=(2,), dtype=np.int64),
            "seat": spaces.MultiBinary(NUM_SEATS),
        }
    )


def observation_for(hands: HandState, seat: int, game_points: Sequence[int]) -> Observation:
    """Build ``seat``'s observation from hand-state public info + its own hand.

    This is the single source of truth for the acting-player's-perspective
    observation: :class:`~deephokm.env.hokm_env.HokmEnv` uses it during
    normal play, and anything that drives the rules engine directly for
    an arbitrary seat -- the web UI's spectate mode, the behavioral-cloning
    dataset collector, the search-augmented policy -- must go through this
    function too rather than re-deriving the layout, so every consumer sees
    exactly the same fields the network was trained on and nothing a real
    player at that seat could not see.

    Args:
        hands: The current hand's state.
        seat: The seat to build the observation for.
        game_points: The match's ``[team_a, team_b]`` game point tally.

    Returns:
        The observation for ``seat``.
    """
    own_team = team_of(seat)

    hand = np.zeros(NUM_CARDS, dtype=np.int8)
    hand[hands.hands[seat]] = 1

    seen = hand.copy()
    seen[hands.played] = 1

    trick = np.zeros(NUM_CARDS, dtype=np.int8)
    trick_play = np.full(NUM_SEATS, -1, dtype=np.int64)
    for played_seat, card in hands.current_trick:
        trick[card] = 1
        trick_play[played_seat] = card

    # Completed tricks only: the current trick has its own slot. Reverse
    # play order puts the most recent card first, so a fixed slot always
    # means the same recency regardless of how far the hand has run.
    completed = len(hands.played) - len(hands.current_trick)
    history = np.full(HISTORY_SLOTS, -1, dtype=np.int64)
    recent = hands.played[:completed][::-1][:HISTORY_SLOTS]
    history[: len(recent)] = recent

    trump = np.zeros(NUM_SUITS, dtype=np.int8)
    if hands.trump is not None:
        trump[hands.trump] = 1

    phase = np.zeros(2, dtype=np.int8)
    if hands.phase is Phase.TRUMP_CALL:
        phase[0] = 1
    elif hands.phase is Phase.CARD_PLAY:
        phase[1] = 1

    tricks_won = np.array(
        [hands.tricks_won[own_team], hands.tricks_won[1 - own_team]], dtype=np.int64
    )
    game_points_arr = np.array(
        [game_points[own_team], game_points[1 - own_team]], dtype=np.int64
    )

    seat_onehot = np.zeros(NUM_SEATS, dtype=np.int8)
    seat_onehot[seat] = 1

    return Observation(
        hand=hand,
        seen=seen,
        trick=trick,
        trick_play=trick_play,
        history=history,
        trump=trump,
        phase=phase,
        tricks_won=tricks_won,
        game_points=game_points_arr,
        seat=seat_onehot,
    )


def mask_for(legal_actions: Sequence[int]) -> np.ndarray:
    """Return an action mask with 1s at the given legal action ids.

    Args:
        legal_actions: Legal action ids, e.g. from
            ``HokmEngine.legal_actions(seat)``.

    Returns:
        A ``(NUM_ACTIONS,)`` int8 array.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    mask[list(legal_actions)] = 1
    return mask


def empty_observation() -> Observation:
    """Return an all-zero observation (used before the first reset)."""
    return Observation(
        hand=np.zeros(NUM_CARDS, dtype=np.int8),
        seen=np.zeros(NUM_CARDS, dtype=np.int8),
        trick=np.zeros(NUM_CARDS, dtype=np.int8),
        trick_play=np.full(NUM_SEATS, -1, dtype=np.int64),
        history=np.full(HISTORY_SLOTS, -1, dtype=np.int64),
        trump=np.zeros(NUM_SUITS, dtype=np.int8),
        phase=np.zeros(2, dtype=np.int8),
        tricks_won=np.zeros(2, dtype=np.int64),
        game_points=np.zeros(2, dtype=np.int64),
        seat=np.zeros(NUM_SEATS, dtype=np.int8),
    )
