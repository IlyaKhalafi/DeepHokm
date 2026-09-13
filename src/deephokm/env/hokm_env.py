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

from deephokm.cards import Suit
from deephokm.cards import card_name as _card_name
from deephokm.env.masked_space import MaskedDiscrete
from deephokm.env.spaces import (
    Observation,
    empty_observation,
    mask_for,
    observation_for,
    observation_space,
)
from deephokm.rules import HokmEngine
from deephokm.rules.legality import NUM_ACTIONS, is_trump_action, trump_action_to_suit
from deephokm.rules.state import NUM_SEATS, Phase, team_of, teammate

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
        control_partner: When ``True``, the learner also controls ``seat``'s
            partner: ``step()`` accepts an action for whichever of the two
            controlled seats is next to act, and observations/masks switch to
            match. Both entries of ``opponents`` for the controlled team are
            ignored (as the learner's own entry already is). Rewards are
            unaffected — they were already computed per-team, not per-seat.
            Off by default, preserving the original one-seat contract.
        hand_only: When ``True``, the episode ends at the first completed
            hand rather than playing out the whole match. A curriculum knob:
            a full match is a long, sparse-reward episode (~13 tricks per
            hand times however many hands it takes one team to reach 7
            points) that is a much noisier credit-assignment problem than a
            single hand. The underlying engine still deals the next hand
            internally when one finishes (it has no "pause here" mode), but
            the environment simply never plays into it -- the episode is
            already over from the caller's perspective, and the next
            ``reset()`` starts a fresh match regardless.
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
        control_partner: bool = False,
        hand_only: bool = False,
    ) -> None:
        """Create the environment.

        Args:
            seat: The seat controlled by the learning agent (0-3).
            opponents: A length-4 list of policies indexed by seat id; the
                learner's entry (and its partner's, when ``control_partner``)
                is ignored. Defaults to uniform random policies.
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
            control_partner: See the class docstring.
            hand_only: See the class docstring.

        Raises:
            ValueError: If the seat is out of range or a mask bug surfaces.
        """
        super().__init__()
        if not 0 <= seat < NUM_SEATS:
            raise ValueError(f"seat must be in [0, {NUM_SEATS}), got {seat}")
        self.seat = seat
        self.control_partner = control_partner
        self._controlled_seats = {seat, teammate(seat)} if control_partner else {seat}
        self._acting_seat = seat
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
        self.hand_only = hand_only
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
        self._hand_over_early = False
        self._hand_over_early_hand_number: int | None = None
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
        self._hand_over_early = False
        self._hand_over_early_hand_number = None
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
                f"illegal action {action} for seat {self._acting_seat}; "
                "this indicates an action-mask bug"
            )

        self._apply_learner_action(int(action))
        # _advance_to_learner()'s loop already no-ops when the match or (with
        # hand_only) the hand just ended -- its while condition is false from
        # the start, and it falls straight through to refreshing the mask and
        # building the (unused, since terminated=True) terminal observation.
        obs, info = self._advance_to_learner()
        reward = self._pending_reward
        self._pending_reward = 0.0
        terminated = self._engine.state.winner is not None or self._hand_over_early
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
        """Run opponents until a controlled seat's turn, a hand ends (when
        ``hand_only``), or the match ends."""
        while self._engine.state.winner is None and not self._hand_over_early:
            seat = self._engine.current_seat()
            if seat in self._controlled_seats:
                self._acting_seat = seat
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
        if self._hand_over_early:
            # The engine already redealt internally by this point (it has no
            # "pause after one hand" mode): self._engine.state.hands is a
            # *different*, freshly-started hand, not the one that just ended.
            # Returning an observation/mask built from it would silently
            # leak a wrong hand_number and a stale-acting-seat mask into
            # what is supposed to be a terminal transition. Since terminated
            # transitions are never bootstrapped off their observation (SB3
            # only does that for truncated ones, and this env always reports
            # truncated=False), the content is inert for training, but
            # honesty matters for every other consumer (logging, callbacks,
            # future features) -- report an explicit empty/terminal state
            # instead of fabricating one from the wrong hand.
            self._action_mask = mask_for([])
            return empty_observation(), self._info()
        self._refresh_mask()
        obs = self._observation_for(self._acting_seat)
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
        if self.hand_only and outcome.hand_complete:
            self._hand_over_early = True
            # The engine has already incremented hand_number and dealt the
            # next hand by the time this outcome is processed (unless the
            # match itself just ended too, which cannot happen on hand_only's
            # first hand -- a hand awards exactly one game point, nowhere
            # near the 7 needed to end a fresh match). Recover the number of
            # the hand that actually just ended for _info() to report.
            self._hand_over_early_hand_number = (
                self._engine.state.hand_number - 1
                if not outcome.match_complete
                else self._engine.state.hand_number
            )

    def _refresh_mask(self) -> None:
        """Recompute the acting seat's action mask from the live state."""
        legal = (
            self._engine.legal_actions(self._acting_seat)
            if self._engine.state.winner is None
            else []
        )
        self._action_mask = mask_for(legal)

    def _sample_legal_actions(self) -> list[int]:
        """Legal action ids for the action space's masked sampler."""
        if self._engine.state.winner is not None or self._hand_over_early:
            return []
        return self._engine.legal_actions(self._acting_seat)

    def _mask_for(self, seat: int) -> np.ndarray:
        """Return the action mask for an arbitrary seat."""
        return mask_for(self._engine.legal_actions(seat))

    def _info(self) -> dict[str, Any]:
        """Build the info dict returned with every observation."""
        if self._hand_over_early:
            # The engine has already moved on to a new hand internally; none
            # of these should describe it (see _advance_to_learner).
            assert self._hand_over_early_hand_number is not None
            return {
                "action_mask": self._action_mask.copy(),
                "hand_number": self._hand_over_early_hand_number,
                "phase": Phase.HAND_OVER.name,
                "current_seat": -1,
            }
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
        return observation_for(self._engine.state.hands, seat, self._engine.state.game_points)

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
