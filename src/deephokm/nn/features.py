"""Grid features for the convolutional Q-network.

The observation is rendered as a stack of ``4 x 13`` planes indexed by
(suit, rank). That layout is what lets the network share weights across suits
while convolving along ranks: suits are exchangeable labels with no ordering,
whereas ranks are ordered, so rank adjacency is real structure and suit
adjacency is not.

Plane layout (14 planes):

====  ===========================================================
  0   cards in the acting player's hand
  1   cards seen (own hand plus everything played this hand)
  2   cards currently on the table
  3   the trump suit, broadcast across ranks
  4   play recency, 1.0 for the most recent completed-trick card
5-8   who played each historical card, by seat relative to the actor
9-12  who played each card in the current trick, same convention
 13   cards that are legal to play right now
====  ===========================================================

Scalars (10): phase one-hot (2), tricks won normalised (2), game points
normalised (2), acting seat one-hot (4).
"""

from __future__ import annotations

import numpy as np

from deephokm.cards import NUM_CARDS, NUM_RANKS, NUM_SUITS
from deephokm.env.spaces import Observation
from deephokm.rules.state import TRICKS_PER_HAND

NUM_PLANES = 14
NUM_SCALARS = 10
MAX_GAME_POINTS = 7

PLANE_HAND = 0
PLANE_SEEN = 1
PLANE_TRICK = 2
PLANE_TRUMP = 3
PLANE_RECENCY = 4
PLANE_HISTORY_ROLE = 5  # ... through 8
PLANE_TRICK_ROLE = 9  # ... through 12
PLANE_LEGAL = 13


def build_features(
    observation: Observation, action_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Render one observation as ``(planes, scalars)``.

    Args:
        observation: The acting player's observation.
        action_mask: The current action mask; its first 52 entries mark the
            legal cards and become the legality plane.

    Returns:
        ``planes`` of shape ``(14, 4, 13)`` and ``scalars`` of shape ``(10,)``,
        both float32.
    """
    planes = np.zeros((NUM_PLANES, NUM_SUITS, NUM_RANKS), dtype=np.float32)
    planes[PLANE_HAND] = np.asarray(observation["hand"]).reshape(NUM_SUITS, NUM_RANKS)
    planes[PLANE_SEEN] = np.asarray(observation["seen"]).reshape(NUM_SUITS, NUM_RANKS)
    planes[PLANE_TRICK] = np.asarray(observation["trick"]).reshape(NUM_SUITS, NUM_RANKS)
    planes[PLANE_TRUMP] = np.asarray(observation["trump"]).reshape(NUM_SUITS, 1)

    history = np.asarray(observation["history"])
    history_role = np.asarray(observation["history_role"])
    slots = len(history)
    for position, (card, role) in enumerate(zip(history, history_role, strict=True)):
        if card < 0:
            continue
        suit, rank = divmod(int(card), NUM_RANKS)
        planes[PLANE_RECENCY, suit, rank] = 1.0 - position / slots
        planes[PLANE_HISTORY_ROLE + int(role), suit, rank] = 1.0

    trick_play = np.asarray(observation["trick_play"])
    seat = int(np.argmax(np.asarray(observation["seat"])))
    for played_seat in range(NUM_SUITS):
        card = int(trick_play[played_seat])
        if card < 0:
            continue
        suit, rank = divmod(card, NUM_RANKS)
        role = (played_seat - seat) % NUM_SUITS
        planes[PLANE_TRICK_ROLE + role, suit, rank] = 1.0

    planes[PLANE_LEGAL] = (
        np.asarray(action_mask)[:NUM_CARDS].reshape(NUM_SUITS, NUM_RANKS).astype(np.float32)
    )

    scalars = np.concatenate(
        [
            np.asarray(observation["phase"], dtype=np.float32),
            np.asarray(observation["tricks_won"], dtype=np.float32) / TRICKS_PER_HAND,
            np.asarray(observation["game_points"], dtype=np.float32) / MAX_GAME_POINTS,
            np.asarray(observation["seat"], dtype=np.float32),
        ]
    ).astype(np.float32)
    return planes, scalars


__all__ = ["NUM_PLANES", "NUM_SCALARS", "build_features"]
