"""The Hokm Gymnasium environment.

One learning seat plays a full match against three opponents that implement
the :class:`~deephokm.policies.base.HokmPolicy` protocol. The opponents act
inside :meth:`HokmEnv.step` before control returns to the learner, so one
environment step equals one learner decision.

Rewards are sparse by default: +1 when the learner's team wins the match, -1
when it loses, 0 otherwise. ``trick_reward`` optionally adds per-trick
shaping; ``hand_reward`` optionally adds per-hand shaping (the coarser,
usually preferable alternative -- see the constructor docstring).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

from deephokm.cards import NUM_CARDS, NUM_SUITS, Suit
from deephokm.cards import card_name as _card_name
from deephokm.env.masked_space import MaskedDiscrete
from deephokm.env.spaces import (
    HISTORY_SLOTS,
    Observation,
    observation_space,
)
from deephokm.rules import HokmEngine
from deephokm.rules.legality import NUM_ACTIONS, is_trump_action, trump_action_to_suit
from deephokm.rules.state import NUM_SEATS, Phase, team_of

if TYPE_CHECKING:  # pragma: no cover
    from deephokm.policies.base import HokmPolicy


class HokmEnv(gym.Env[Observation, np.integer]):
    """A single-seat Hokm environment over the shared rules engine.

    Attributes:
        seat: The learning seat.
        opponents: Policies for the other three seats, indexed by seat.
        opponent_provider: Optional callable re-drawn on every ``reset()`` to
            replace ``opponents``; self-play training uses it to pick a fresh
            opponent mix per episode.
        trick_reward: Optional per-trick shaping magnitude (default 0).
        hand_reward: Optional per-hand shaping magnitude (default 0).
        render_mode: ``"human"`` prints a trick log; ``None`` is silent.
    """

    metadata: dict[str, Any] = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        seat: int = 0,
        opponents: list[HokmPolicy] | None = None,
        *,
        trick_reward: float = 0.0,
        render_mode: str | None = None,
        opponent_provider: Callable[[], list[HokmPolicy]] | None = None,
        hand_reward: float = 0.0,
    ) -> None:
        """Create the environment.

        Args:
            seat: The seat controlled by the learning agent (0-3).
            opponents: A length-4 list of policies indexed by seat id; the
                learner's entry is ignored. Defaults to uniform random
                policies.
            trick_reward: Magnitude of optional +/- shaping per trick won/lost.
                Every trick pays the same, whether it decides a close hand or
                mops up an already-settled one, so a large value can dominate
                the sparse match outcome and reward tactically meaningless
                tricks; ``hand_reward`` is the coarser, usually preferable
                alternative.
            render_mode: ``"human"`` or ``None``.
            opponent_provider: Optional callable invoked on every ``reset()``
                to draw a fresh length-4 opponent list.
            hand_reward: Magnitude of optional +/- shaping per hand won/lost
                (a hand is 13 tricks; a match is played to 7 hand-level game
                points). Coarser than ``trick_reward`` and closer to what
                actually matters -- it rewards winning the hand, not padding
                a trick count that is already decided.

        Raises:
            ValueError: If the seat is out of range or a mask bug surfaces.
        """
        super().__init__()
        if not 0 <= seat < NUM_SEATS:
            raise ValueError(f"seat must be in [0, {NUM_SEATS}), got {seat}")
        self.seat = seat
        if opponents is None:
            # Imported lazily: policies depend on env.spaces, so a top-level
            # import here would create an import cycle.
            from deephokm.policies.random_policy import (  # noqa: PLC0415
                RandomPolicy,
            )

            opponents = [RandomPolicy() for _ in range(NUM_SEATS)]
        if len(opponents) != NUM_SEATS:
            raise ValueError(
                f"opponents must have {NUM_SEATS} entries (one per seat, the "
                f"learner's entry is ignored), got {len(opponents)}"
            )
        self.opponents = list(opponents)
        self.opponent_provider = opponent_provider
        self.trick_reward = trick_reward
        self.hand_reward = hand_reward
        self.render_mode = render_mode
        self.action_space = MaskedDiscrete(NUM_ACTIONS, legal_provider=self._sample_legal_actions)
        # spaces.Dict is invariant; the TypedDict Observation describes the
        # same structure (this is the pattern gymnasium itself uses for
        # structured observations).
        self.observation_space = observation_space()  # type: ignore[assignment]

        self._engine = HokmEngine()
        self._action_mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self._last_info: dict[str, Any] = {}
        self._pending_reward = 0.0
        self._terminated = False
        self._render_lines: list[str] = []

    # ------------------------------------------------------------------ API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Start a fresh match.

        Args:
            seed: Seed for the engine RNG; identical seeds replay identical
                shuffles and action offers.
            options: Unused (accepted for API compliance).

        Returns:
            The learner's first observation and an info dict with the action
            mask. If the learner's seat acts first, the observation is the
            trump-call decision; otherwise opponents act until it is the
            learner's turn.
        """
        super().reset(seed=seed)
        self._engine.start_match(seed=seed)
        self._draw_opponents()
        self._reset_opponents(seed)
        self._terminated = False
        self._pending_reward = 0.0
        self._render_lines = [self._render_header()]
        obs, info = self._advance_to_learner()
        self._maybe_render_obs(obs)
        return obs, info

    def _draw_opponents(self) -> None:
        """Re-draw the opponent seats from the provider, when one is set.

        Self-play training rotates the opponent mix per episode; without this
        the policies chosen when the worker was created would be frozen for
        the whole run.
        """
        if self.opponent_provider is None:
            return
        drawn = list(self.opponent_provider())
        if len(drawn) != NUM_SEATS:
            raise ValueError(
                f"opponent_provider must return {NUM_SEATS} policies, got {len(drawn)}"
            )
        self.opponents = drawn

    def _reset_opponents(self, seed: int | None) -> None:
        """Restore per-episode determinism for stateful opponents.

        Policies exposing ``reset(seed)`` are reseeded with a seat-derived
        offset of the environment seed so that a seeded ``reset`` replays the
        entire episode, opponents included. Policies without the hook are
        assumed stateless across episodes.
        """
        if seed is None:
            for policy in self.opponents:
                if hasattr(policy, "reset"):
                    policy.reset()
            return
        for seat, policy in enumerate(self.opponents):
            if hasattr(policy, "reset"):
                policy.reset(seed * NUM_SEATS + seat)

    def step(self, action: np.integer) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply the learner's action and let the opponents respond.

        Args:
            action: Action id in ``0..55``.

        Returns:
            ``(observation, reward, terminated, truncated, info)`` where the
            observation is the learner's next decision point (or the terminal
            observation).

        Raises:
            ValueError: If the action is illegal for the learner's seat.
        """
        if self._terminated:
            raise RuntimeError("episode is over; call reset() first")
        if not self._action_mask[int(action)]:
            raise ValueError(
                f"illegal action {action} for seat {self.seat}; this indicates an action-mask bug"
            )

        self._apply_learner_action(int(action))
        obs, info = self._advance_to_learner()
        reward = self._pending_reward
        self._pending_reward = 0.0
        terminated = self._engine.state.winner is not None
        self._terminated = terminated
        self._maybe_render_obs(obs)
        return obs, reward, terminated, False, info

    @property
    def engine(self) -> HokmEngine:
        """Return the rules engine driving this environment.

        The web UI and the evaluation harness read match state (score, phase,
        winner) straight off the engine rather than reconstructing it from
        observations; this is the supported accessor for that.
        """
        return self._engine

    def action_masks(self) -> np.ndarray:
        """Return the current action mask (sb3-contrib protocol)."""
        mask: np.ndarray = self._action_mask.copy()
        return mask

    def render(self) -> None:
        """Print the accumulated trick log (``render_mode='human'``)."""
        if self.render_mode != "human":
            return
        print("\n".join(self._render_lines))

    # ---------------------------------------------------------- internals

    def _apply_learner_action(self, action: int) -> None:
        """Apply the learner's action and accumulate any immediate reward."""
        outcome = self._engine.apply_action(action)
        self._render_action(outcome)
        if is_trump_action(action):
            self._render_trump(trump_action_to_suit(action))
        self._accumulate_rewards(outcome)

    def _advance_to_learner(self) -> tuple[Observation, dict[str, Any]]:
        """Run opponents until the learner's turn or the match ends."""
        while self._engine.state.winner is None:
            seat = self._engine.current_seat()
            if seat == self.seat:
                break
            mask = self._mask_for(seat)
            obs = self._observation_for(seat)
            action = self.opponents[seat].act(obs, mask)
            if not mask[action]:
                raise RuntimeError(f"opponent at seat {seat} returned illegal action {action}")
            outcome = self._engine.apply_action(action)
            self._render_action(outcome)
            if is_trump_action(action):
                self._render_trump(trump_action_to_suit(action))
            self._accumulate_rewards(outcome)
        self._refresh_mask()
        obs = self._observation_for(self.seat)
        return obs, self._info()

    def _accumulate_rewards(self, outcome: Any) -> None:
        """Fold an action outcome into the pending reward."""
        own = team_of(self.seat)
        if outcome.trick_complete and self.trick_reward:
            winner = outcome.trick_winner
            assert winner is not None
            if team_of(winner) == own:
                self._pending_reward += self.trick_reward
            else:
                self._pending_reward -= self.trick_reward
        if outcome.hand_complete and self.hand_reward:
            hand_winner = outcome.hand_winner_team
            assert hand_winner is not None
            if hand_winner == own:
                self._pending_reward += self.hand_reward
            else:
                self._pending_reward -= self.hand_reward
        if outcome.match_complete:
            match_winner = outcome.match_winner_team
            assert match_winner is not None
            self._pending_reward += 1.0 if match_winner == own else -1.0

    def _refresh_mask(self) -> None:
        """Recompute the learner's action mask from the live state."""
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        if self._engine.state.winner is None:
            legal = self._engine.legal_actions(self.seat)
            mask[legal] = 1
        self._action_mask = mask

    def _sample_legal_actions(self) -> list[int]:
        """Legal action ids for the action space's masked sampler."""
        if self._engine.state.winner is not None:
            return []
        return self._engine.legal_actions(self.seat)

    def _mask_for(self, seat: int) -> np.ndarray:
        """Return the action mask for an arbitrary seat."""
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        mask[self._engine.legal_actions(seat)] = 1
        return mask

    def _info(self) -> dict[str, Any]:
        """Build the info dict returned with every observation."""
        return {
            "action_mask": self._action_mask.copy(),
            "hand_number": self._engine.state.hand_number,
            "phase": self._engine.state.hands.phase.name,
            "current_seat": self._engine.current_seat()
            if self._engine.state.winner is None
            else -1,
        }

    def _observation_for(self, seat: int) -> Observation:
        """Build the observation for ``seat`` from public state + own hand."""
        hands = self._engine.state.hands
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
        game_points = np.array(
            [
                self._engine.state.game_points[own_team],
                self._engine.state.game_points[1 - own_team],
            ],
            dtype=np.int64,
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
            game_points=game_points,
            seat=seat_onehot,
        )

    # ---------------------------------------------------------- rendering

    def _render_header(self) -> str:
        """First line of the render log: hakem for the opening hand."""
        return f"=== Match start; first hakem: seat {self._engine.state.hakem} ==="

    def _render_trump(self, suit: int) -> None:
        """Append the trump declaration to the render log."""
        self._render_lines.append(f"Trump declared: {Suit(suit).name}")

    def _render_action(self, outcome: Any) -> None:
        """Append a played card / trick result to the render log."""
        if outcome.card is not None:
            self._render_lines.append(f"seat {outcome.seat} plays {_card_name(outcome.card)}")
        if outcome.trick_complete:
            assert outcome.trick_winner is not None
            assert outcome.tricks_won is not None
            self._render_lines.append(
                f"--> seat {outcome.trick_winner} wins the trick "
                f"({outcome.tricks_won[0]}-{outcome.tricks_won[1]})"
            )
        if outcome.hand_complete:
            assert outcome.hand_winner_team is not None
            self._render_lines.append(
                f"=== Hand over; team {outcome.hand_winner_team} wins "
                f"({self._engine.state.game_points[0]}-"
                f"{self._engine.state.game_points[1]}) ==="
            )
        if outcome.match_complete:
            assert outcome.match_winner_team is not None
            self._render_lines.append(f"=== Match over; team {outcome.match_winner_team} wins ===")

    def _maybe_render_obs(self, obs: Observation) -> None:
        """Flush the render log when in human mode."""
        if self.render_mode == "human" and self._render_lines:
            print("\n".join(self._render_lines))
            self._render_lines = []
