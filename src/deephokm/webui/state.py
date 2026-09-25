"""Game state management for the web UI.

The server owns every game and drives the same rules engine used in
training; the frontend is a thin renderer and action submitter. Games are
identified by an opaque id and hold a HokmEngine plus the acting policies.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np

from deephokm.cards import card_name
from deephokm.env.hokm_env import HokmEnv
from deephokm.rules.state import NUM_SEATS


@dataclass
class GameRecord:
    """One live web game.

    Attributes:
        id: Opaque game identifier.
        mode: ``"human"`` (one human seat) or ``"spectate"`` (all AI).
        seed: Match seed.
        env: The environment driving the match (learner seat = human seat in
            human mode, seat 0 in spectate mode).
        viewer_seat: The seat whose perspective API responses render.
        created_at: Monotonic creation order.
    """

    id: str
    mode: str
    seed: int
    env: HokmEnv
    viewer_seat: int
    created_at: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    moves: int = 0


class GameStore:
    """Thread-safe registry of live games."""

    def __init__(self, max_games: int = 256) -> None:
        """Create the store.

        Args:
            max_games: Maximum concurrent games; oldest evicted.
        """
        self._games: dict[str, GameRecord] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._max_games = max_games

    def create(self, mode: str, seed: int, opponents: list[Any]) -> GameRecord:
        """Create a new game and return its record."""
        with self._lock:
            self._counter += 1
            game_id = uuid4().hex[:12]
            viewer_seat = 0
            env = HokmEnv(
                seat=viewer_seat,
                opponents=opponents,
            )
            # Attach before reset: reset() already asks opponents to act when
            # the viewer is not first to move, and an unattached policy would
            # otherwise decide against the previous game's finished engine.
            for opponent in opponents:
                attach = getattr(opponent, "attach", None)
                if attach is not None:
                    attach(env.engine)
            env.reset(seed=seed)
            record = GameRecord(
                id=game_id,
                mode=mode,
                seed=seed,
                env=env,
                viewer_seat=viewer_seat,
                created_at=self._counter,
            )
            self._games[game_id] = record
            self._evict_locked()
            return record

    def get(self, game_id: str) -> GameRecord | None:
        """Return the game record, or None if unknown."""
        with self._lock:
            return self._games.get(game_id)

    def _evict_locked(self) -> None:
        """Drop the oldest games when over capacity (caller holds the lock)."""
        while len(self._games) > self._max_games:
            oldest = min(self._games.values(), key=lambda r: r.created_at)
            del self._games[oldest.id]


def public_state(record: GameRecord) -> dict[str, Any]:
    """Render the viewer's public game state as JSON.

    Derives everything from the engine's public state plus the viewer's own
    hand — never another player's private cards.
    """
    env = record.env
    engine = env.engine
    hands = engine.state.hands
    seat = record.viewer_seat
    current_seat = engine.current_seat() if engine.state.winner is None else -1
    phase = hands.phase.name

    # The table: cards in the current trick with seat attribution, in play order.
    leader = hands.leader
    table: list[dict[str, Any]] = []
    if hands.current_trick:
        # Play order rotates from the leader.
        order = []
        for offset in range(NUM_SEATS):
            s = (leader + offset) % NUM_SEATS
            for ps, card in hands.current_trick:
                if ps == s:
                    order.append((ps, card))
                    break
        for played_seat, card in order:
            table.append({"seat": played_seat, "card": int(card), "name": card_name(card)})

    # In spectate mode no seat is the viewer's: expose neither a private hand
    # nor seat-specific legal actions; the spectator sees card backs, the
    # table, and counts only.
    spectating = record.mode == "spectate"
    legal_actions: list[int] = (
        []
        if spectating
        else (
            engine.legal_actions(seat)
            if engine.state.winner is None and current_seat == seat
            else []
        )
    )
    visible_hand = [] if spectating else list(hands.hands[seat])

    return {
        "game_id": record.id,
        "mode": record.mode,
        # The seed is deliberately NOT exposed: the engine is deterministic,
        # so a client knowing the seed could reconstruct every private hand.
        "viewer_seat": seat,
        "current_seat": current_seat,
        "phase": phase,
        "trump": hands.trump,
        "hand": [{"card": int(c), "name": card_name(c)} for c in visible_hand],
        "hand_counts": [len(h) for h in hands.hands],
        "table": table,
        "tricks_won": list(hands.tricks_won),
        "game_points": list(engine.state.game_points),
        "hakem": hands.hakem,
        "hand_number": engine.state.hand_number,
        "legal_actions": legal_actions,
        "winner": engine.state.winner,
        "moves": record.moves,
        "terminal": engine.state.winner is not None,
    }


def apply_human_action(record: GameRecord, action: int) -> None:
    """Apply the viewer's action; the AI seats respond inside step()."""
    env = record.env
    mask = env.action_masks()
    if not mask[action]:
        raise ValueError(
            f"illegal action {action} for seat {record.viewer_seat}; this indicates a client bug"
        )
    env.step(np.int64(action))
    record.moves += 1
