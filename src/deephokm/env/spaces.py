"""Observation and action space definitions for HokmEnv.

The observation is a Dict space from the acting player's perspective. The
layout is frozen by the project specification; the network's tokenizer and
the tests both depend on it. Any additive change must update the feature
extractor and tests in the same commit.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from gymnasium import spaces

from deephokm.cards import NUM_CARDS, NUM_SUITS
from deephokm.rules.state import NUM_SEATS


class Observation(TypedDict):
    """The observation dict handed to the acting player.

    Attributes:
        hand: MultiBinary(52) — cards currently in the acting player's hand.
        seen: MultiBinary(52) — own hand plus every card played this hand.
        trick: MultiBinary(52) — cards currently on the table.
        trick_play: Box(4) — card id (or -1) played by each seat in the
            current trick, in seat order; -1 for seats yet to play. The leader
            is the first seat with a card.
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
            "trump": spaces.MultiBinary(NUM_SUITS),
            "phase": spaces.MultiBinary(2),
            "tricks_won": spaces.Box(low=0, high=13, shape=(2,), dtype=np.int64),
            "game_points": spaces.Box(low=0, high=7, shape=(2,), dtype=np.int64),
            "seat": spaces.MultiBinary(NUM_SEATS),
        }
    )


def empty_observation() -> Observation:
    """Return an all-zero observation (used before the first reset)."""
    return Observation(
        hand=np.zeros(NUM_CARDS, dtype=np.int8),
        seen=np.zeros(NUM_CARDS, dtype=np.int8),
        trick=np.zeros(NUM_CARDS, dtype=np.int8),
        trick_play=np.full(NUM_SEATS, -1, dtype=np.int64),
        trump=np.zeros(NUM_SUITS, dtype=np.int8),
        phase=np.zeros(2, dtype=np.int8),
        tricks_won=np.zeros(2, dtype=np.int64),
        game_points=np.zeros(2, dtype=np.int64),
        seat=np.zeros(NUM_SEATS, dtype=np.int8),
    )
