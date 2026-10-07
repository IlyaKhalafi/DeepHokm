"""Game state management for the web UI.

The server owns every game and drives the same rules engine used in
training; the frontend is a thin renderer and action submitter. Games are
identified by an opaque id and hold a HokmEngine plus the acting policies.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from deephokm.cards import card_name
from deephokm.env.hokm_env import HokmEnv
from deephokm.rules.state import NUM_SEATS

GameDifficulty = Literal["fast", "hard"]


@dataclass
class GameRecord:
    """One live web game.

    Attributes:
        id: Opaque game identifier.
        mode: ``"human"`` (one human seat) or ``"spectate"`` (all AI).
        difficulty: ``"fast"`` (Q-pure) or ``"hard"`` (Q-hybrid).
        policy: Stable identifier for the policy actually serving the game.
        seed: Match seed.
        env: The environment driving the match (learner seat = human seat in
            human mode, seat 0 in spectate mode).
        viewer_seat: The seat whose perspective API responses render.
        created_at: Monotonic creation order.
    """

    id: str
    mode: str
    difficulty: GameDifficulty
    policy: str
    seed: int
    env: HokmEnv
    viewer_seat: int
    created_at: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    moves: int = 0
    # The engine clears a trick the instant its fourth card lands, so without
    # keeping a copy the winning play is never visible to the player.
    last_trick: list[tuple[int, int]] = field(default_factory=list)
    last_trick_winner: int | None = None


def _close_opponents(opponents: list[Any]) -> None:
    """Release any per-game resources the acting policies were holding.

    ``build_opponents`` puts the same served-policy object in all four
    seats, so this closes each distinct object once rather than four times.
    A policy without a ``close`` (the memoized RL checkpoint, RandomPolicy)
    is simply skipped -- only the search policy owns a process pool.
    """
    for policy in {id(p): p for p in opponents}.values():
        close = getattr(policy, "close", None)
        if close is not None:
            close()


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

    def create(
        self,
        mode: str,
        seed: int,
        opponents: list[Any],
        difficulty: GameDifficulty = "fast",
        policy: str = "random-baseline",
    ) -> GameRecord:
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
                difficulty=difficulty,
                policy=policy,
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
            _close_opponents(oldest.env.opponents)
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
        "difficulty": record.difficulty,
        "policy": record.policy,
        # The seed is deliberately NOT exposed: the engine is deterministic,
        # so a client knowing the seed could reconstruct every private hand.
        "viewer_seat": seat,
        "current_seat": current_seat,
        "phase": phase,
        "trump": hands.trump,
        "hand": [{"card": int(c), "name": card_name(c)} for c in visible_hand],
        "hand_counts": [len(h) for h in hands.hands],
        "table": table,
        # The trick that just finished, shown while the table is empty so the
        # winning play does not vanish the instant it resolves.
        "last_trick": [
            {"seat": s, "card": c, "name": card_name(c)} for s, c in record.last_trick
        ],
        "last_trick_winner": record.last_trick_winner,
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
    """Apply the viewer's action and nothing else.

    This deliberately does not go through ``HokmEnv.step()``. That call is
    built around the single-learner Gym API and auto-plays every other seat
    before returning, so the player's card and all three replies resolved in
    one request and the trick was already cleared by the time the response
    arrived -- the player never saw what anyone else played. The AI seats are
    advanced one ply at a time through ``advance_one_ply`` instead.
    """
    engine = record.env.engine
    if engine.state.winner is not None:
        raise ValueError("the match is over")
    seat = record.viewer_seat
    current = engine.current_seat()
    if current != seat:
        raise ValueError(f"it is seat {current}'s turn, not yours")
    # Validate against the engine, not HokmEnv.action_masks(): that returns a
    # mask cached by step()/reset(), and this path drives the engine directly
    # so the cache is never refreshed. Using it rejected every legal card after
    # the opening move.
    legal = engine.legal_actions(seat)
    if action not in legal:
        raise ValueError(f"illegal action {action} for seat {seat}; legal actions are {legal}")
    _apply_and_record(record, action, seat)


def advance_one_ply(record: GameRecord) -> None:
    """Play exactly one AI seat's action.

    Raises:
        ValueError: If it is the viewer's turn; the server must never play
            the player's card for them.
    """
    env = record.env
    engine = env.engine
    if engine.state.winner is not None:
        return
    seat = engine.current_seat()
    if record.mode != "spectate" and seat == record.viewer_seat:
        raise ValueError("it is the viewer's turn; the server does not act for them")
    action = env.opponents[seat].act(env._observation_for(seat), env._mask_for(seat))
    _apply_and_record(record, action, seat)


def _apply_and_record(record: GameRecord, action: int, seat: int) -> None:
    """Apply one action, keeping the trick that just completed for display.

    The completed trick is held until the first card of the *next* trick is
    played, and the client shows it whenever the live table is empty. That is
    the window in which the player would otherwise see nothing at all.
    """
    engine = record.env.engine
    completed = list(engine.state.hands.current_trick)
    outcome = engine.apply_action(action, seat=seat)

    if outcome.trick_complete:
        completed.append((seat, action))
        record.last_trick = [(int(s), int(c)) for s, c in completed]
        record.last_trick_winner = outcome.trick_winner
    elif outcome.card is not None and len(engine.state.hands.current_trick) == 1:
        # First card of a new trick: the previous one is no longer current.
        record.last_trick = []
        record.last_trick_winner = None
    record.moves += 1
