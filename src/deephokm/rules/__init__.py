"""The pure Hokm rules engine.

Modules:
    :mod:`deephokm.rules.state` — core state types and phase tracking.
    :mod:`deephokm.rules.dealing` — hakem selection and dealing.
    :mod:`deephokm.rules.legality` — which actions are legal in a state.
    :mod:`deephokm.rules.tricks` — trick resolution.
    :mod:`deephokm.rules.scoring` — hand and match scoring, hakem rotation.
    :mod:`deephokm.rules.engine` — the full state machine tying them together.
"""

from __future__ import annotations

from deephokm.rules.engine import ActionOutcome, HokmEngine
from deephokm.rules.legality import (
    NUM_ACTIONS,
    TRUMP_ACTION_OFFSET,
    is_trump_action,
    legal_actions,
    trump_action,
    trump_action_to_suit,
)
from deephokm.rules.state import (
    CARDS_PER_PLAYER,
    HAKEM_FIRST_BATCH,
    NUM_SEATS,
    POINTS_TO_WIN_MATCH,
    TRICKS_PER_HAND,
    TRICKS_TO_WIN_HAND,
    HandState,
    MatchState,
    Phase,
    team_of,
    teammate,
)

__all__ = [
    "ActionOutcome",
    "CARDS_PER_PLAYER",
    "HAKEM_FIRST_BATCH",
    "HandState",
    "HokmEngine",
    "MatchState",
    "NUM_ACTIONS",
    "NUM_SEATS",
    "POINTS_TO_WIN_MATCH",
    "Phase",
    "TRICKS_PER_HAND",
    "TRICKS_TO_WIN_HAND",
    "TRUMP_ACTION_OFFSET",
    "is_trump_action",
    "legal_actions",
    "team_of",
    "teammate",
    "trump_action",
    "trump_action_to_suit",
]
