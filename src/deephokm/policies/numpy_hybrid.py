"""Nominate with the numpy Q-network, confirm with a search sign test.

The network is fast but imperfect; the search is slow but trustworthy. This
policy spends the network's speed to shrink the search's work:

1. The Q-network ranks the legal actions (one numpy forward pass, ~6 ms).
2. Its top candidates that differ from :class:`GreedyPolicy`'s choice are
   proposed, best first.
3. Each proposal is checked against greedy's action over ``verify_samples``
   determinized worlds, and played only if it wins an exact one-sided sign
   test at ``max_p_value``.

Searching every legal action costs ``K x |legal|`` rollouts; checking ``M``
proposals costs ``K x 2M``, which at the measured mean of ~5 legal actions and
``M = 3`` is comparable per decision while concentrating the rollouts on the
actions worth arguing about. Proposing more than one candidate is deliberate:
the network's top choice matches the teacher's best on ~65% of decisive
decisions, but its top three contain it far more often, and the sign test is
exactly the mechanism that can tell which of the three is right.

Nothing here sees hidden cards. The network reads only the observation, and
the sampled worlds are guesses constrained by the voids inferred from public
play.
"""

from __future__ import annotations

import multiprocessing
import pickle
import random
from pathlib import Path

import numpy as np

from deephokm.cards import NUM_CARDS
from deephokm.env.spaces import Observation, mask_for, observation_for
from deephokm.nn.feature_contract import resolve_feature_mode
from deephokm.nn.features import build_features
from deephokm.nn.numpy_qnet import NumpyQNet, load_weights
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.legal_depth_search import _sign_test_p_value
from deephokm.policies.oracle_search import oracle_ceiling
from deephokm.policies.search import (
    VoidTracker,
    _clone_for_simulation,
    sample_determinized_hands,
)
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import NUM_SEATS, Phase, team_of

DEFAULT_VERIFY_SAMPLES = 192
DEFAULT_MAX_P_VALUE = 0.05
DEFAULT_TOP_M = 3
DEFAULT_TEMPERATURE = 0.1
# Sequential elimination: worlds are drawn in rounds, every surviving action is
# scored in each world, and actions far enough behind the leader are dropped.
ELIMINATION_ROUNDS = 6
ELIMINATION_Z = 2.0  # drop an action this many standard errors behind the best
MIN_DRAWS_TO_ELIMINATE = 2  # a variance estimate needs at least two samples
# Every legal action keeps at least this share of the mean per-action budget,
# so a confident-but-wrong network can never starve the right move entirely.
MIN_BUDGET_SHARE = 0.25


def _score_pair(
    task: tuple[bytes, int, int, int, int, list[list[int]], int],
) -> tuple[float, float]:
    """Score (candidate, baseline) in one sampled world (worker process).

    Returns the paired outcomes the sign test compares, so the test's
    world-level pairing is preserved exactly as in the serial loop.
    """
    engine_bytes, seat, team, candidate, baseline, world, seed = task
    engine = pickle.loads(engine_bytes)
    clone_rng = random.Random(seed)
    value_c = _action_value(
        engine, seat, candidate, team=team, world=world, rng=clone_rng
    )
    value_b = _action_value(
        engine, seat, baseline, team=team, world=world, rng=clone_rng
    )
    return value_c, value_b


def _score_batch(
    task: tuple[bytes, int, int, list[list[int]], list[int], int],
) -> list[float]:
    """Score every listed action in one sampled world (worker process).

    The engine travels as a pickle because the worker is a fresh process;
    the clone built from it is thrown away with the result. ``seed`` keeps
    the clone's rollout RNG independent across workers without any shared
    state.
    """
    engine_bytes, seat, team, world, actions, seed = task
    engine = pickle.loads(engine_bytes)
    clone_rng = random.Random(seed)
    return [
        _action_value(engine, seat, action, team=team, world=world, rng=clone_rng)
        for action in actions
    ]


def _action_value(
    engine: HokmEngine,
    seat: int,
    action: int,
    *,
    team: int,
    world: list[list[int]],
    rng: random.Random,
) -> float:
    """Outcome of ``action`` in one sampled world (module level, picklable)."""
    clone = _clone_for_simulation(engine, seat, world, rng=rng)
    outcome = clone.apply_action(action, seat=seat)
    if outcome.hand_complete:
        return 1.0 if outcome.hand_winner_team == team else -1.0
    return oracle_ceiling(clone, team, depth=1, rng=rng)


class NumpyHybridPolicy:
    """Greedy by default; network proposals confirmed by a sign test.

    Attributes:
        net: The numpy action-value network used to rank actions.
        verify_samples: Determinized worlds drawn per proposal.
        top_m: How many of the network's best actions to propose.
        max_p_value: One-sided sign-test threshold for accepting a proposal.
        workers: Parallel scorer processes for elimination rounds; 1 keeps
            the serial loop (the rollouts are pure Python, so this is the
            only way a large verify budget stays interactive).
        greedy: The scripted baseline that proposals must beat.
        voids: Suit voids inferred from public play.
    """

    def __init__(
        self,
        weights: Path | dict[str, np.ndarray],
        *,
        verify_samples: int = DEFAULT_VERIFY_SAMPLES,
        workers: int = 1,
        top_m: int = DEFAULT_TOP_M,
        max_p_value: float = DEFAULT_MAX_P_VALUE,
        seed: int = 0,
        allocate: bool = False,
        eliminate: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
        pool: multiprocessing.pool.Pool | None = None,
        feature_mode: str | None = None,
    ) -> None:
        """Create the policy.

        Args:
            weights: A ``.npz`` path, or an already-loaded weight dict.
            verify_samples: Worlds sampled per proposal.
            top_m: Number of network proposals to consider.
            max_p_value: Sign-test threshold.
            seed: Seed for the world sampler.
            allocate: Steer the rollout budget with the network instead of
                pruning to its top ``top_m`` actions.
            eliminate: Score every action in every sampled world and drop the
                ones that fall statistically behind, using the network only to
                order the field. Unlike pruning this cannot discard the action
                full search would choose before it has been scored.
            temperature: Softmax temperature for that split; lower
                concentrates more budget on the network's favourites.
            pool: An externally owned scorer pool to reuse instead of
                creating a private one. Pass this from a long-lived,
                multithreaded host (the web server): forking a pool lazily,
                the first time a request thread happens to call ``decide``,
                forks the whole process from inside that thread, and any
                lock another request thread holds at that instant (loggers,
                malloc arenas, C-extension globals) is inherited already
                held and never released in the child. A pool forked once
                from the main thread before request handling starts has no
                such thread to race against. Callers that pass a pool own
                its lifetime; ``close()`` on this policy is then a no-op.
        """
        params = load_weights(weights) if isinstance(weights, Path) else weights
        self.net = NumpyQNet(params)
        self.feature_mode = resolve_feature_mode(weights, self.net.input_planes, feature_mode)
        self.verify_samples = verify_samples
        self.top_m = top_m
        self.max_p_value = max_p_value
        self.allocate = allocate
        self.eliminate = eliminate
        self.workers = workers
        # Worker-process rollout seeds. Deliberately NOT self._rng: the
        # serial path consumes no randomness beyond world sampling, so the
        # parallel path must not either, or the two paths diverge in the
        # worlds they draw from the same seed.
        self._worker_seed_rng = random.Random(seed ^ 0x5EED)
        self.temperature = temperature
        self.greedy = GreedyPolicy()
        self.voids = VoidTracker()
        self._rng = random.Random(seed)
        # A lazily self-created pool (used by every harness and test script,
        # which are single-threaded top-level programs with no fork-safety
        # concern) is closed on close(); an injected one is not -- its
        # owner controls that.
        self._pool = pool
        self._owns_pool = pool is None
        self._pool_ctx = multiprocessing.get_context("fork")

    def close(self) -> None:
        """Release the scorer pool, if this policy owns one."""
        if self._owns_pool and self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def _scorer_pool(self) -> multiprocessing.pool.Pool:
        """The scorer pool for this policy: injected, or created on demand."""
        if self._pool is None:
            self._pool = self._pool_ctx.Pool(self.workers)
        return self._pool

    def reset_hand(self) -> None:
        """Forget inferred voids; call at the start of every hand."""
        self.voids.reset()

    def observe(self, seat: int, card: int, led_suit: int | None) -> None:
        """Record one publicly played card."""
        self.voids.observe(seat, card, led_suit)

    def decide(self, engine: HokmEngine) -> int:
        """Choose the acting seat's action.

        With ``allocate=True`` the network steers how the rollout budget is
        divided across every legal action instead of pruning to its favourites.
        Measured, top-M pruning *loses* to plain search once the search is
        strong (lift +0.143 at K=48 decaying to -0.029 at K=384 on held-out
        seeds): pruning sometimes removes the action full search would have
        chosen, and no amount of extra search recovers an action that was never
        scored. Allocation keeps every action in play, so a wrong prior costs
        precision rather than the answer.
        """
        seat = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(seat)
        if len(legal) == 1:
            return legal[0]

        observation = observation_for(hands, seat, engine.state.game_points)
        mask = mask_for(legal)
        if hands.phase is not Phase.CARD_PLAY:
            # Trump declaration: the network has no trained signal here and
            # both the scripted baseline and the pure search defer to greedy.
            return self.greedy.act(observation, mask)

        baseline = self.greedy.act(observation, mask)
        if self.eliminate:
            return self._decide_by_elimination(
                engine, seat, observation=observation, mask=mask,
                legal=legal, baseline=baseline,
            )
        if self.allocate:
            return self._decide_by_allocation(
                engine, seat, observation=observation, mask=mask,
                legal=legal, baseline=baseline,
            )
        for candidate in self._proposals(observation, mask, legal, baseline):
            if self._beats(engine, seat, candidate, baseline):
                return candidate
        return baseline

    def _budget(self, values: np.ndarray, legal: list[int], total: int) -> dict[int, int]:
        """Split ``total`` rollouts across legal actions by the network's prior.

        The split is a temperature-softened softmax over the network's values,
        floored so no action drops below ``MIN_BUDGET_SHARE`` of an even split.
        The floor is what makes this safe: the prior can concentrate effort but
        never silently eliminate a candidate.
        """
        scores = values[legal].astype(np.float64)
        weights = np.exp((scores - scores.max()) / max(self.temperature, 1e-6))
        weights /= weights.sum()
        floor = MIN_BUDGET_SHARE / len(legal)
        weights = np.maximum(weights, floor)
        weights /= weights.sum()
        counts = np.maximum((weights * total).astype(int), 1)
        return {int(a): int(c) for a, c in zip(legal, counts, strict=True)}

    def _decide_by_allocation(
        self,
        engine: HokmEngine,
        seat: int,
        *,
        observation: Observation,
        mask: np.ndarray,
        legal: list[int],
        baseline: int,
    ) -> int:
        """Score every legal action, with the network deciding sample counts."""
        planes, scalars = build_features(observation, mask, feature_mode=self.feature_mode)
        values = self.net(planes[None], scalars[None])[0]
        budget = self._budget(values, legal, self.verify_samples * len(legal))

        team = team_of(seat)
        hands = engine.state.hands
        own_hand = hands.hands[seat]
        seen = set(hands.played) | set(own_hand)
        unseen = [card for card in range(NUM_CARDS) if card not in seen]
        sizes = [len(hands.hands[s]) for s in range(NUM_SEATS)]
        clone_rng = random.Random()

        means: dict[int, float] = {}
        for action, samples in budget.items():
            total = 0.0
            for _ in range(samples):
                world = sample_determinized_hands(
                    seat, own_hand, unseen, sizes, voids=self.voids.voids, rng=self._rng
                )
                total += self._value(
                    engine, seat, action, team=team, world=world, clone_rng=clone_rng
                )
            means[action] = total / samples

        best = max(means, key=lambda a: means[a])
        # Ties go to the scripted baseline: deviating on a dead heat is how the
        # earlier raw-argmax policy lost to greedy.
        return baseline if means[best] <= means.get(baseline, -2.0) else best

    def _proposals(
        self,
        observation: Observation,
        mask: np.ndarray,
        legal: list[int],
        baseline: int,
    ) -> list[int]:
        """The network's ``top_m`` legal actions, best first, minus greedy's."""
        planes, scalars = build_features(observation, mask, feature_mode=self.feature_mode)
        values = self.net(planes[None], scalars[None])[0]
        ranked = [int(legal[i]) for i in np.argsort(values[legal])[::-1]]
        return [a for a in ranked[: self.top_m] if a != baseline]

    def _beats(self, engine: HokmEngine, seat: int, candidate: int, baseline: int) -> bool:
        """Sign test: does ``candidate`` beat ``baseline`` across sampled worlds?"""
        hands = engine.state.hands
        own_hand = hands.hands[seat]
        seen = set(hands.played) | set(own_hand)
        unseen = [card for card in range(NUM_CARDS) if card not in seen]
        sizes = [len(hands.hands[s]) for s in range(NUM_SEATS)]
        team = team_of(seat)
        # One clone RNG per decision; clones only consume it for a simulated
        # redeal, into state that is discarded.
        clone_rng = random.Random()
        wins_candidate = wins_baseline = 0
        if self.workers > 1:
            # Same paired sign test, scored in parallel: one task per world,
            # both actions inside the task so each worker's two rollouts see
            # the identical determinization the serial loop would.
            engine_bytes = pickle.dumps(engine)
            worlds = [
                sample_determinized_hands(
                    seat, own_hand, unseen, sizes, voids=self.voids.voids, rng=self._rng
                )
                for _ in range(self.verify_samples)
            ]
            seed_base = self._worker_seed_rng.randrange(2**30)
            tasks = [
                (engine_bytes, seat, team, candidate, baseline, world, seed_base + i)
                for i, world in enumerate(worlds)
            ]
            pairs = self._scorer_pool().map(_score_pair, tasks, chunksize=1)
            for value_c, value_b in pairs:
                if value_c > value_b:
                    wins_candidate += 1
                elif value_b > value_c:
                    wins_baseline += 1
        else:
            for _ in range(self.verify_samples):
                world = sample_determinized_hands(
                    seat, own_hand, unseen, sizes, voids=self.voids.voids, rng=self._rng
                )
                value_c = self._value(
                    engine, seat, candidate, team=team, world=world, clone_rng=clone_rng
                )
                value_b = self._value(
                    engine, seat, baseline, team=team, world=world, clone_rng=clone_rng
                )
                if value_c > value_b:
                    wins_candidate += 1
                elif value_b > value_c:
                    wins_baseline += 1
        discordant = wins_candidate + wins_baseline
        return _sign_test_p_value(wins_candidate, discordant) <= self.max_p_value


    def _decide_by_elimination(
        self,
        engine: HokmEngine,
        seat: int,
        *,
        observation: Observation,
        mask: np.ndarray,
        legal: list[int],
        baseline: int,
    ) -> int:
        """Full-search quality at lower cost, via sequential elimination.

        Pruning to the network's favourites cannot beat plain search, because
        it maximises over a subset of what plain search considers: measured
        lift decays from +0.143 at K=48 to -0.060 at K=3072. This keeps every
        action in the field and removes only those the evidence has already
        ruled out.

        Each round draws worlds and scores *every survivor in the same world*,
        so world-level variance cancels the way the paired sign test does -- the
        mistake that sank the allocation variant was comparing actions across
        different worlds. After each round an action is dropped when it trails
        the leader by more than ``ELIMINATION_Z`` standard errors of the paired
        difference. The network supplies the initial ordering, which decides
        only who is measured first, never who is excluded.

        Greedy's action is never eliminated, and is returned unless a survivor
        beats it on the same sign test the pure search uses, so the accept gate
        is unchanged.
        """
        planes, scalars = build_features(observation, mask, feature_mode=self.feature_mode)
        values = self.net(planes[None], scalars[None])[0]
        survivors = sorted(legal, key=lambda a: -values[a])
        if baseline not in survivors:  # defensive: greedy must stay in the field
            survivors.append(baseline)

        team = team_of(seat)
        hands = engine.state.hands
        own_hand = hands.hands[seat]
        seen = set(hands.played) | set(own_hand)
        unseen = [card for card in range(NUM_CARDS) if card not in seen]
        sizes = [len(hands.hands[s]) for s in range(NUM_SEATS)]
        clone_rng = random.Random()

        per_round = max(1, self.verify_samples // ELIMINATION_ROUNDS)
        totals = dict.fromkeys(survivors, 0.0)
        squares = dict.fromkeys(survivors, 0.0)
        # Per-action sample counts, because an eliminated action stops
        # accumulating: comparing raw totals would rank actions by how long they
        # survived rather than by how well they scored.
        counts = dict.fromkeys(survivors, 0)
        n_workers = self.workers
        use_pool = n_workers > 1 and per_round >= n_workers
        engine_bytes = pickle.dumps(engine) if use_pool else b""
        for _ in range(ELIMINATION_ROUNDS):
            if len(survivors) <= 1:
                break
            worlds = [
                sample_determinized_hands(
                    seat, own_hand, unseen, sizes, voids=self.voids.voids, rng=self._rng
                )
                for _ in range(per_round)
            ]
            if use_pool:
                # One task per world; each worker scores every surviving
                # action in its world, so world-level variance cancels
                # exactly as it does in the serial loop.
                seed_base = self._worker_seed_rng.randrange(2**30)
                tasks = [
                    (engine_bytes, seat, team, world, list(survivors), seed_base + i)
                    for i, world in enumerate(worlds)
                ]
                for values in self._scorer_pool().map(
                    _score_batch, tasks, chunksize=1
                ):
                    for action, value in zip(survivors, values, strict=True):
                        totals[action] += value
                        squares[action] += value * value
                        counts[action] += 1
            else:
                for world in worlds:
                    for action in survivors:
                        value = self._value(
                            engine, seat, action, team=team, world=world,
                            clone_rng=clone_rng,
                        )
                        totals[action] += value
                        squares[action] += value * value
                        counts[action] += 1
            survivors = self._survivors(survivors, totals, squares, counts, baseline)

        best = max(survivors, key=lambda a: totals[a] / counts[a])
        if best == baseline:
            return baseline
        return best if self._beats(engine, seat, best, baseline) else baseline

    @staticmethod
    def _survivors(
        actions: list[int],
        totals: dict[int, float],
        squares: dict[int, float],
        counts: dict[int, int],
        baseline: int,
    ) -> list[int]:
        """Keep the leader, greedy's action, and everything not yet ruled out.

        Each action is judged on its own mean over its own sample count, so an
        action that has been measured fewer times is not penalised for it.
        """
        means = {a: totals[a] / counts[a] for a in actions if counts[a] > 0}
        if not means:
            return actions
        leader = max(means, key=lambda a: means[a])
        if min(counts[a] for a in actions) < MIN_DRAWS_TO_ELIMINATE:
            return actions
        kept = []
        for a in actions:
            n = counts[a]
            variance = max(squares[a] / n - means[a] ** 2, 0.0)
            stderr = (variance / n) ** 0.5
            gap = means[leader] - means[a]
            if a in (leader, baseline) or gap <= ELIMINATION_Z * max(stderr, 1e-9):
                kept.append(a)
        return kept

    def _value(
        self,
        engine: HokmEngine,
        seat: int,
        action: int,
        *,
        team: int,
        world: list[list[int]],
        clone_rng: random.Random,
    ) -> float:
        """Outcome of ``action`` in one sampled world."""
        clone = _clone_for_simulation(engine, seat, world, rng=clone_rng)
        outcome = clone.apply_action(action, seat=seat)
        if outcome.hand_complete:
            assert outcome.hand_winner_team is not None
            return 1.0 if outcome.hand_winner_team == team else -1.0
        return oracle_ceiling(clone, team, depth=1, rng=clone_rng)


__all__ = ["DEFAULT_TOP_M", "DEFAULT_VERIFY_SAMPLES", "NumpyHybridPolicy"]
