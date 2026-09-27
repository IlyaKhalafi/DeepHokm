"""Serve the strongest measured policy to the web UI.

The UI previously served a ``MaskablePPO`` checkpoint, which is the approach
that plateaued at the level of a greedy clone, and fell back to random play when
no checkpoint was present. This serves what actually measures well: the numpy
action-value network nominating and ordering, with the determinized search
confirming, which reaches 0.841 against a greedy opposing team at high search
budgets.

Search strength is a latency trade, so it is configurable. At the default
budget a decision costs well under a second, which keeps the UI responsive;
raising ``DEEPHOKM_SEARCH_K`` buys strength at proportional cost.

The environment hands opponents only an observation and a mask, but the search
needs the live engine, so the game store attaches it after construction. Voids
are rebuilt from the public play record on every decision rather than tracked
incrementally: the environment does not notify opponents of other seats' cards,
and re-deriving is both simpler and impossible to desynchronise.
"""

from __future__ import annotations

import os
import threading
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np

from deephokm.cards import NUM_RANKS
from deephokm.env.spaces import Observation
from deephokm.nn.numpy_qnet import load_weights
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.numpy_hybrid import NumpyHybridPolicy
from deephokm.policies.pure_qnet_policy import PureQNetPolicy
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import NUM_SEATS, TRICKS_PER_HAND, Phase

DEFAULT_WEIGHTS = "checkpoints/qnet_numpy_full.npz"
# 0: serve the network alone, no live search (see resolve_search_k). Search
# is what the offline teacher labels this network trained on used, not what
# runs at play time; raising this past 0 opts into live search instead.
DEFAULT_SEARCH_K = 0


class SearchServedPolicy:
    """Network-guided search, exposed through the ``HokmPolicy`` interface.

    Attributes:
        policy: The underlying network-plus-search policy.
        engine: The live engine, attached by the game store; without it the
            policy degrades to the scripted baseline rather than failing.
    """

    def __init__(
        self,
        weights: dict[str, np.ndarray],
        *,
        search_k: int = DEFAULT_SEARCH_K,
        eliminate: bool = True,
        seed: int = 0,
        workers: int = 1,
        pool: Pool | None = None,
    ) -> None:
        """Build the policy over an already-loaded weight set.

        Args:
            weights: Weight arrays, shared read-only across games.
            search_k: Sampled worlds per decision; higher is stronger and
                slower. Zero serves the network alone -- no live search, no
                rollouts, a plain argmax over the legal actions' values.
                Measured at 0.6625 against a greedy opposing team (well below
                the search-paired policy's 0.855) but a median 18 ms per
                decision. Raising K past zero always means live search: the
                offline search used to generate this network's own training
                labels (K up to 6144, see
                ``docs/METHODS_AND_RESULTS.md``) is a separate, one-time cost
                that never runs at serve time.
            eliminate: Score every legal action and drop only those the
                evidence rules out. Pruning to the network's favourites
                measures worse once the search is strong, because an action
                that is never scored can never be chosen. Unused when
                ``search_k`` is 0.
            seed: Seed for the world sampler. Unused when ``search_k`` is 0.
            workers: Parallel scorer processes for the search rounds; the
                network forward pass itself stays a single-threaded numpy
                call. Large search budgets stay interactive only with this.
                Unused when ``search_k`` is 0.
            pool: An app-owned scorer pool, forked once at server startup
                before any request thread exists (see
                :func:`NumpyHybridPolicy.__init__`'s ``pool`` argument for
                why that timing matters). When given, every per-game policy
                shares it and ``close()`` on this object is a no-op -- the
                app closes it once at shutdown instead. Unused when
                ``search_k`` is 0.

        """
        self.policy: NumpyHybridPolicy | PureQNetPolicy
        if search_k <= 0:
            self.policy = PureQNetPolicy(weights)
        else:
            self.policy = NumpyHybridPolicy(
                weights, verify_samples=search_k, eliminate=eliminate, seed=seed,
                workers=workers, pool=pool,
            )
        self.greedy = GreedyPolicy()
        self.engine: HokmEngine | None = None

    def attach(self, engine: HokmEngine) -> None:
        """Give the policy the live engine the search needs."""
        self.engine = engine

    def close(self) -> None:
        """Release the underlying policy's scorer pool, if it owns one.

        The web app shares one pool across every game (see the ``pool``
        argument above), so this is normally a no-op -- the app closes that
        pool once at shutdown. It only does real work for a policy built
        without an injected pool, where the game store must still call this
        on eviction or a ``workers > 1`` policy leaks its own worker
        processes for as long as the pool object survives.
        """
        self.policy.close()

    def act(self, observation: Observation, action_mask: np.ndarray) -> int:
        """Return the policy's action for the acting seat.

        Falls back to the scripted baseline whenever the engine is absent or
        not at a live card-play decision. The environment asks opponents to act
        during ``reset()`` too, so an engine that is finished or mid-deal must
        degrade rather than raise.
        """
        engine = self.engine
        if engine is None or engine.state.winner is not None:
            return self.greedy.act(observation, action_mask)
        if engine.state.hands.phase not in (Phase.CARD_PLAY, Phase.TRUMP_CALL):
            return self.greedy.act(observation, action_mask)
        self._rebuild_voids()
        return self.policy.decide(engine)

    def _rebuild_voids(self) -> None:
        """Re-derive each seat's known voids from this hand's public play.

        A seat that fails to follow the led suit has shown, to everyone at the
        table, that it holds no more of that suit. Replaying the hand's
        completed and current tricks reconstructs exactly that, using only
        information a human at the table also has.
        """
        assert self.engine is not None
        hands = self.engine.state.hands
        self.policy.reset_hand()
        played, played_by = hands.played, hands.played_by
        for index, (card, seat) in enumerate(zip(played, played_by, strict=True)):
            position = index % NUM_SEATS
            if position == 0:
                continue  # the leader sets the suit and reveals nothing
            led = played[index - position] // NUM_RANKS
            self.policy.observe(seat, card, led)

    def reset(self, seed: int | None = None) -> None:
        """No-op: per-hand state is rebuilt from the engine on every decision."""


_weights_lock = threading.Lock()
_weights_cache: dict[str, dict[str, np.ndarray]] = {}


def load_shared_weights(path: str) -> dict[str, np.ndarray]:
    """Load a weight archive once per process and share it across games.

    The arrays are only read during inference, so one copy serves every game;
    each game still gets its own policy object, because a policy holds the
    engine it is deciding for and a shared instance would let concurrent games
    overwrite each other's reference.

    Raises:
        FileNotFoundError: If the archive is missing.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"numpy weights not found at {path}; set DEEPHOKM_QNET")
    with _weights_lock:
        cached = _weights_cache.get(path)
        if cached is None:
            cached = load_weights(Path(path))
            _weights_cache[path] = cached
    return cached


def resolve_weights_path() -> str:
    """Return the numpy weight path from the environment (or the default)."""
    return os.environ.get("DEEPHOKM_QNET", DEFAULT_WEIGHTS)


def resolve_search_k() -> int:
    """Return the per-decision world count from the environment.

    Zero (the default) means the served policy is the network alone, with
    no live search; see :class:`SearchServedPolicy`'s ``search_k`` argument.
    """
    raw = os.environ.get("DEEPHOKM_SEARCH_K", str(DEFAULT_SEARCH_K))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SEARCH_K
    return max(0, value)


def resolve_workers() -> int:
    """Return the parallel scorer process count from the environment.

    Defaults to 1 (serial rollouts). Capped at the machine's core count so a
    typo cannot fork a swarm.
    """
    raw = os.environ.get("DEEPHOKM_SEARCH_WORKERS", "1")
    try:
        value = int(raw)
    except ValueError:
        return 1
    return max(1, min(value, os.cpu_count() or 1))


def describe_policy(policy: SearchServedPolicy | None) -> dict[str, object]:
    """Summarise the served policy for the health endpoint."""
    if policy is None:
        return {"policy": "greedy-baseline", "search_k": 0}
    inner = policy.policy
    if isinstance(inner, PureQNetPolicy):
        payload: dict[str, object] = {
            "policy": "numpy-qnet-only",
            "search_k": 0,
            "search_workers": 0,
        }
    else:
        payload = {
            "policy": "numpy-qnet+elimination-search",
            "search_k": inner.verify_samples,
            "search_workers": inner.workers,
        }
    # A hand stops at seven tricks for either team; 13 is the cap, not the
    # length.
    payload["max_tricks_per_hand"] = TRICKS_PER_HAND
    return payload


__all__ = [
    "DEFAULT_SEARCH_K",
    "SearchServedPolicy",
    "describe_policy",
    "resolve_search_k",
    "resolve_weights_path",
    "resolve_workers",
]
