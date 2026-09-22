"""Q-hybrid policy: neural nomination with search verification.

Two-tier strategy:

- When PyTorch reports an available CUDA device, a small action-value
  network (the 256-d transformer encoder plus a 56-way Q head, trained
  supervised on per-action search estimates) nominates a candidate action
  whenever it disagrees with :class:`GreedyPolicy`, and only that single
  candidate is verified against greedy's action by the same K-sample
  sign-test machinery the legal depth search uses. This cuts the rollout
  count per decision from ``K x |legal|`` to ``K x 2`` — roughly an order
  of magnitude less search at similar decision quality.
- Without a usable CUDA device the network is skipped entirely and the
  policy falls back to :class:`LegalDepthSearchPolicy`, because batch-1
  CPU inference of the transformer costs more than the search it saves.

Like every policy in this package, decisions use only the observation and
action mask plus the public history of played cards (via the void tracker);
the sampled worlds are guesses about hidden cards, never knowledge of them.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

from deephokm.cards import NUM_CARDS, NUM_RANKS
from deephokm.env.spaces import Observation, mask_for, observation_for
from deephokm.policies.base import HokmPolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.legal_depth_search import (
    LegalDepthSearchPolicy,
    _sign_test_p_value,
    VoidTracker,
)
from deephokm.policies.search import _clone_for_simulation, sample_determinized_hands
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase, team_of

QNET_PATH = Path(__file__).resolve().parents[3] / "checkpoints" / "qnet.pt"
DEFAULT_VERIFY_K = 48
DEFAULT_MAX_P = 0.05


def _cuda_available() -> bool:
    """Return True only when torch can actually use a CUDA device.

    torch.cuda.is_available() alone is not enough in stripped-down
    environments: it can report True while every runtime call fails. The
    device-count check plus a tiny guarded allocation makes the
    availability test robust; if anything raises, treat CUDA as absent.
    """
    try:
        import torch as th

        if not th.cuda.is_available() or th.cuda.device_count() == 0:
            return False
        th.zeros(1, device="cuda")
        return True
    except Exception:
        return False


def _load_qnet():
    """Load the action-value network onto the first CUDA device.

    Returns a callable mapping an observation dict to a 56-vector of
    Q values, or None when loading is not possible (no torch, missing
    checkpoint, no CUDA). The caller treats None as "use the pure search".
    """
    if not _cuda_available() or not QNET_PATH.exists():
        return None
    try:
        import torch as th
        import torch.nn as nn

        from deephokm.env.hokm_env import HokmEnv
        from deephokm.nn.extractor import HokmTransformerExtractor
        from deephokm.policies.greedy_policy import GreedyPolicy
        from deephokm.rules.state import NUM_SEATS

        class _QNet(nn.Module):
            def __init__(self, obs_space):
                super().__init__()
                self.extractor = HokmTransformerExtractor(obs_space)
                self.head = nn.Sequential(
                    nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 56)
                )

            def forward(self, ob):
                return self.head(self.extractor(ob))

        probe_env = HokmEnv(
            seat=0, opponents=[GreedyPolicy() for _ in range(NUM_SEATS)]
        )
        net = _QNet(probe_env.observation_space)
        net.load_state_dict(th.load(QNET_PATH, weights_only=True))
        net.to("cuda").eval()

        @th.no_grad()
        def predict(obs: Observation) -> np.ndarray:
            batch = {
                k: th.as_tensor(np.asarray(v)[None, ...], device="cuda")
                for k, v in obs.items()
            }
            return net(batch)[0].cpu().numpy()

        return predict
    except Exception:
        return None


class HybridQSearchPolicy:
    """Greedy by default; neural nomination verified by a sign test.

    Attributes:
        verify_k: Sampled worlds used to verify a nominated candidate
            against greedy's action (split evenly between the two actions).
        max_p_value: One-sided sign-test threshold for accepting the
            nominated action.
        search: Fallback (and trump-call) policy when no usable Q-net
            exists — a plain :class:`LegalDepthSearchPolicy`.
    """

    def __init__(
        self,
        *,
        verify_k: int = DEFAULT_VERIFY_K,
        max_p_value: float = DEFAULT_MAX_P,
        seed: int = 0,
        use_qnet: bool | None = None,
    ) -> None:
        self.verify_k = verify_k
        self.max_p_value = max_p_value
        self._rng = random.Random(seed)
        self.greedy = GreedyPolicy()
        self.voids = VoidTracker()
        self._predict = _load_qnet() if use_qnet is None else (
            _load_qnet() if use_qnet else None
        )
        self.search = LegalDepthSearchPolicy(
            n_samples=verify_k, search_depth=2, max_p_value=max_p_value, seed=seed
        )

    def reset_hand(self) -> None:
        """Clear void state; call at the start of every new hand."""
        self.voids.reset()
        self.search.reset_hand()

    def observe(self, seat: int, card: int, led_suit: int | None) -> None:
        """Record one publicly-played card from anywhere at the table."""
        self.voids.observe(seat, card, led_suit)
        self.search.observe(seat, card, led_suit)

    def decide(self, engine: HokmEngine) -> int:
        """Choose the acting seat's next action (see module docstring)."""
        root_seat = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(root_seat)
        if len(legal) <= 1:
            return legal[0]
        if hands.phase is not Phase.CARD_PLAY:
            # Trump call: no trained head for it; the scripted baseline and
            # the pure search both defer to greedy here anyway.
            obs = observation_for(hands, root_seat, engine.state.game_points)
            return self.greedy.act(obs, mask_for(legal))

        if self._predict is None:
            return self.search.decide(engine)

        obs = observation_for(hands, root_seat, engine.state.game_points)
        greedy_action = self.greedy.act(obs, mask_for(legal))
        q = self._predict(obs)
        nominated = int(legal[int(np.argmax(q[legal]))])
        if nominated == greedy_action:
            return greedy_action

        team = team_of(root_seat)
        own_hand = hands.hands[root_seat]
        seen = set(hands.played) | set(own_hand)
        unseen = [c for c in range(NUM_CARDS) if c not in seen]
        sizes = [len(hands.hands[s]) for s in range(4)]
        wins_nominated = wins_greedy = 0
        for _ in range(self.verify_k):
            samp = sample_determinized_hands(
                root_seat, own_hand, unseen, sizes, voids=self.voids.voids, rng=self._rng
            )
            n_val = self._rollout_value(engine, root_seat, nominated, team, samp)
            g_val = self._rollout_value(engine, root_seat, greedy_action, team, samp)
            if n_val > g_val:
                wins_nominated += 1
            elif g_val > n_val:
                wins_greedy += 1
        p = _sign_test_p_value(wins_nominated, wins_nominated + wins_greedy)
        return nominated if p <= self.max_p_value else greedy_action

    def _rollout_value(
        self,
        engine: HokmEngine,
        seat: int,
        action: int,
        team: int,
        sampled_hands: list[list[int]],
    ) -> float:
        """One sampled world's outcome for ``action`` (depth-1 continuation)."""
        clone = _clone_for_simulation(engine, seat, sampled_hands)
        outcome = clone.apply_action(action, seat=seat)
        if outcome.hand_complete:
            assert outcome.hand_winner_team is not None
            return 1.0 if outcome.hand_winner_team == team else -1.0
        assert clone.state.hands.phase is Phase.CARD_PLAY
        # Depth-1 single-guess continuation inside the sampled world, the
        # same evaluation the legal depth search's score function uses.
        from deephokm.policies.oracle_search import oracle_ceiling

        return oracle_ceiling(clone, team, depth=1)
