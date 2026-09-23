"""Root-sampled imperfect-information Monte Carlo search (test-time only).

Chess engines like Stockfish combine game-tree search with a learned
evaluator because chess is perfect information: the whole board is visible,
so "if I play X, they play Y, ..." can be searched directly. Hokm is
imperfect information -- opponents' hands are hidden -- so a naive
minimax/MCTS tree does not apply; the tree would have to branch over every
possible hidden deal.

This module implements the standard fix used by strong trick-taking-game
programs (bridge's GIB and relatives): at each real decision, sample several
full "determinized" worlds (plausible complete deals consistent with
everything actually observable so far), then evaluate each candidate action
against the trained policy's own rollout in each sampled world. This is
root-sampled imperfect-information Monte Carlo (IIMC), not classic PIMC:
every simulated seat still only ever sees its own legal observation and
mask, never another seat's hand or which determinized world is "true", so
the simulated future play stays information-correct instead of quietly
assuming full visibility (the failure mode classic PIMC is known for --
"strategy fusion").

The single correctness invariant every function here must preserve: nothing
in this module may read another seat's actual card ids out of a live
``HandState`` for use in sampling or search. The only things drawn from
live state for seats other than the acting one are quantities a real player
at the table can already see -- how many cards a seat holds, and which
cards have been publicly played -- never the hidden hand contents
themselves. ``sample_determinized_hands`` is the sole place hidden cards for
other seats are invented, and it invents them, never reads them.

Scope: card-play decisions only. The trump call is left to the wrapped
policy's own choice -- searching it would require determinizing the 47
still-undealt cards under a very different (and much weaker) constraint set,
which is a larger, separate piece of work.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from deephokm.cards import NUM_CARDS, NUM_RANKS, NUM_SUITS
from deephokm.env.spaces import Observation, mask_for, observation_for
from deephokm.rules.engine import ActionOutcome, HokmEngine
from deephokm.rules.state import NUM_SEATS, HandState, MatchState, Phase, team_of

TOP_K_DEFAULT = 3
N_SAMPLES_DEFAULT = 8
MAX_ROLLOUT_PLIES_DEFAULT = 12
MAX_SAMPLE_ATTEMPTS = 200


def _suit_of(card: int) -> int:
    """Return the suit id of a card id."""
    return card // NUM_RANKS


def _build_suit_seat_flow_graph(
    suit_counts: dict[int, int],
    seats: list[int],
    voids: list[set[int]],
    remaining_sizes: list[int],
    rng: random.Random,
) -> dict[object, dict[object, int]]:
    """Build the source -> suit -> seat -> sink flow graph for one split.

    No edge exists for a seat's void suits, so any flow the graph admits is
    void-respecting by construction. Edges are inserted in an ``rng``-shuffled
    order (consumed by ``_bfs_augmenting_path`` iterating dict insertion
    order) so that different calls can land on different vertex solutions
    of the underlying transportation polytope -- see
    ``_feasible_suit_seat_counts`` for why that matters.
    """
    graph: dict[object, dict[object, int]] = {}

    def add_edge(u: object, v: object, cap: int) -> None:
        graph.setdefault(u, {})[v] = graph.get(u, {}).get(v, 0) + cap
        graph.setdefault(v, {}).setdefault(u, 0)

    suit_items = list(suit_counts.items())
    rng.shuffle(suit_items)
    for suit, count in suit_items:
        if count:
            add_edge("source", ("suit", suit), count)
    seat_order = list(seats)
    rng.shuffle(seat_order)
    for seat in seat_order:
        if remaining_sizes[seat]:
            add_edge(("seat", seat), "sink", remaining_sizes[seat])
    for suit, count in suit_items:
        if not count:
            continue
        seat_order = list(seats)
        rng.shuffle(seat_order)
        for seat in seat_order:
            if suit not in voids[seat]:
                add_edge(("suit", suit), ("seat", seat), count)
    return graph


def _bfs_augmenting_path(
    graph: dict[object, dict[object, int]], source: object, sink: object
) -> list[tuple[object, object]] | None:
    parent: dict[object, object | None] = {source: None}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if u == sink:
            path = []
            v: object = sink
            while v != source:
                p = parent[v]
                path.append((p, v))
                v = p
            return list(reversed(path))
        for v, cap in graph.get(u, {}).items():
            if cap > 0 and v not in parent:
                parent[v] = u
                queue.append(v)
    return None


def _feasible_suit_seat_counts(
    suit_counts: dict[int, int],
    seats: list[int],
    voids: list[set[int]],
    remaining_sizes: list[int],
    rng: random.Random,
) -> dict[int, dict[int, int]]:
    """Find a void-respecting split of each suit's card count across seats.

    Modeled as max-flow rather than a single greedy pass: a feasible split
    is guaranteed to exist (the real, hidden deal this is standing in for
    is itself a witness), but a greedy top-to-bottom assignment can miss
    one even when it exists -- filling an unconstrained seat first can
    strand a later, more constrained seat with only cards of a suit it
    cannot hold. Max-flow finds a feasible split whenever one exists.

    With no randomness anywhere, the *same* inputs would always produce
    the *same* split -- repeated calls for one decision (which share the
    same voids/sizes) would all collapse onto one effective world instead
    of the independent determinizations this module relies on. Augmenting
    a network flow along a different order of paths can land on a
    different basic feasible solution (vertex) of the transportation
    polytope, so ``_build_suit_seat_flow_graph`` inserts edges in an
    ``rng``-shuffled order. This samples among the (generally few) *basic*
    feasible vertices reachable this way, not uniformly over the full
    feasible region (an interior mixed-suit split is not a vertex a
    network-flow solution can land on at all) -- a real, disclosed
    limitation, but a large improvement over a single deterministic split,
    and this fallback triggers only in the rare cases dense-enough voids
    make plain rejection sampling fail (see the caller).
    """
    source, sink = "source", "sink"
    graph = _build_suit_seat_flow_graph(suit_counts, seats, voids, remaining_sizes, rng)

    total_flow = 0
    while (path := _bfs_augmenting_path(graph, source, sink)) is not None:
        bottleneck = min(graph[u][v] for u, v in path)
        for u, v in path:
            graph[u][v] -= bottleneck
            graph[v][u] += bottleneck
        total_flow += bottleneck

    total_needed = sum(remaining_sizes[s] for s in seats)
    if total_flow != total_needed:
        raise RuntimeError(
            f"no void-respecting hand assignment exists: max flow {total_flow} != "
            f"{total_needed} cards needed -- voids are inconsistent with the public state"
        )

    counts: dict[int, dict[int, int]] = {seat: dict.fromkeys(suit_counts, 0) for seat in seats}
    for suit, count in suit_counts.items():
        if not count:
            continue
        for seat in seats:
            if suit in voids[seat]:
                continue
            used = count - graph[("suit", suit)][("seat", seat)]
            counts[seat][suit] = used
    return counts


@dataclass
class VoidTracker:
    """Per-hand record of which suits each seat is known not to hold.

    A seat that fails to follow the led suit reveals -- publicly, to every
    player at the table, not just to whoever led -- that it holds no more
    of that suit for the rest of the hand. This is exactly the kind of
    information a real player uses and a fair search must use too; it must
    be rebuilt from scratch every new hand.
    """

    voids: list[set[int]] = field(default_factory=lambda: [set() for _ in range(NUM_SEATS)])

    def reset(self) -> None:
        """Clear all voids (call at the start of every new hand)."""
        self.voids = [set() for _ in range(NUM_SEATS)]

    def observe(self, seat: int, card: int, led_suit: int | None) -> None:
        """Record one publicly-played card.

        Args:
            seat: The seat that played ``card``.
            card: The card id played.
            led_suit: The suit led in this trick, or ``None`` if ``seat`` is
                itself the leader (leading is never evidence of a void).
        """
        if led_suit is not None and _suit_of(card) != led_suit:
            self.voids[seat].add(led_suit)


def sample_determinized_hands(
    root_seat: int,
    root_hand: list[int],
    unseen_pool: list[int],
    remaining_sizes: list[int],
    *,
    voids: list[set[int]],
    rng: random.Random,
    max_attempts: int = MAX_SAMPLE_ATTEMPTS,
) -> list[list[int]]:
    """Sample one plausible full hand assignment for the other three seats.

    Uses whole-deal rejection sampling: shuffle the unseen cards uniformly,
    slice them into per-seat chunks of the correct (publicly known)
    remaining size, and reject the whole sample if it assigns a
    known-void suit to a seat. This keeps the sampler unbiased -- a greedy
    "avoid voids while assigning" scheme would not sample uniformly among
    the deals actually consistent with the voids.

    Args:
        root_seat: The seat whose hand is already known (kept as given).
        root_hand: ``root_seat``'s actual hand.
        unseen_pool: All card ids not in ``root_hand`` and not yet played.
        remaining_sizes: Each seat's true remaining hand size this hand
            (public: 13 minus how many cards that seat has played).
        voids: Per-seat known-void suit sets (see :class:`VoidTracker`).
        rng: Source of randomness for this sample.
        max_attempts: Rejection-sampling attempts before falling back to an
            unconstrained (void-ignoring) assignment.

    Returns:
        A length-``NUM_SEATS`` list of hands; ``result[root_seat]`` is
        ``root_hand`` unchanged.
    """
    other_seats = [s for s in range(NUM_SEATS) if s != root_seat]
    expected_total = sum(remaining_sizes[s] for s in other_seats)
    if expected_total != len(unseen_pool):
        raise ValueError(
            f"remaining_sizes for the other seats sum to {expected_total}, "
            f"but the unseen pool has {len(unseen_pool)} cards"
        )

    pool = list(unseen_pool)
    assignment: dict[int, list[int]] = {}
    for _ in range(max_attempts):
        rng.shuffle(pool)
        idx = 0
        ok = True
        assignment = {}
        for seat in other_seats:
            size = remaining_sizes[seat]
            chunk = pool[idx : idx + size]
            idx += size
            if voids[seat] and any(_suit_of(c) in voids[seat] for c in chunk):
                ok = False
                break
            assignment[seat] = chunk
        if ok:
            break
    else:
        # Exhausted every attempt without a fully void-respecting whole-deal
        # split from plain rejection sampling. Falling back to an assignment
        # that ignores voids entirely -- or a single greedy top-to-bottom
        # pass -- would let the rollout simulate a seat following a suit it
        # has publicly proven it does not hold: an impossible world, not
        # just a low-probability one, and a single greedy pass can miss a
        # feasible split that exists (filling an unconstrained seat first
        # can strand a later, voided seat with only cards of its void
        # suit). Solve for a feasible suit-count split directly via
        # max-flow instead -- guaranteed to find one whenever it exists,
        # which it always does here (the real, hidden deal is a witness).
        suit_counts: dict[int, int] = dict.fromkeys(range(NUM_SUITS), 0)
        for c in pool:
            suit_counts[_suit_of(c)] += 1
        per_seat_suit_counts = _feasible_suit_seat_counts(
            suit_counts, other_seats, voids, remaining_sizes, rng
        )
        cards_by_suit: dict[int, list[int]] = {suit: [] for suit in suit_counts}
        for c in pool:
            cards_by_suit[_suit_of(c)].append(c)
        for cards in cards_by_suit.values():
            rng.shuffle(cards)
        cursor = dict.fromkeys(suit_counts, 0)
        assignment = {}
        for seat in other_seats:
            hand: list[int] = []
            for suit, count in per_seat_suit_counts[seat].items():
                if count:
                    start = cursor[suit]
                    hand.extend(cards_by_suit[suit][start : start + count])
                    cursor[suit] += count
            assignment[seat] = hand
    return [
        list(root_hand) if seat == root_seat else assignment[seat] for seat in range(NUM_SEATS)
    ]


def _clone_for_simulation(
    engine: HokmEngine,
    visible_seat: int,
    sampled_hands: list[list[int]],
    rng: random.Random | None = None,
) -> HokmEngine:
    """Return a fresh, independent engine for simulating from a determinized world.

    Copies every mutable field so that driving the clone through
    ``apply_action`` never touches ``engine``'s real state. ``visible_seat``
    is the only seat whose real hand is copied from ``engine``; every other
    seat's hand comes from ``sampled_hands`` (never from ``engine`` itself),
    so a real hidden hand can never even transiently exist in the clone --
    this is enforced structurally rather than by remembering to overwrite it
    afterwards. The clone's own RNG is only ever consumed if a simulated
    hand runs to completion (which triggers an automatic re-deal for a hand
    the caller discards anyway).

    Args:
        engine: The real, live match engine (read, never mutated).
        visible_seat: The one seat whose real hand may be copied.
        sampled_hands: A length-``NUM_SEATS`` determinized hand assignment;
            only the entries for seats other than ``visible_seat`` are used.
        rng: RNG for the clone (consumed only if a simulated hand re-deals);
            pass a hoisted shared instance in rollout loops to avoid paying
            for a fresh, never-seeded :class:`random.Random` per clone.
    """
    hands = engine.state.hands
    cloned_hands = HandState(
        hands=[
            list(hands.hands[visible_seat]) if seat == visible_seat else list(sampled_hands[seat])
            for seat in range(NUM_SEATS)
        ],
        hakem=hands.hakem,
        trump=hands.trump,
        leader=hands.leader,
        current_trick=list(hands.current_trick),
        played=list(hands.played),
        played_by=list(hands.played_by),
        tricks_won=list(hands.tricks_won),
        trick_winners=list(hands.trick_winners),
        phase=hands.phase,
        pending_deck=list(hands.pending_deck),
    )
    cloned_state = MatchState(
        hands=cloned_hands,
        game_points=list(engine.state.game_points),
        hakem=engine.state.hakem,
        hand_number=engine.state.hand_number,
        winner=None,
    )
    return HokmEngine(rng=rng if rng is not None else random.Random(), state=cloned_state)


@dataclass
class IIMCSearchPolicy:
    """A test-time search wrapper around a trained masked actor-critic policy.

    Not a :class:`~deephokm.policies.base.HokmPolicy`: it needs direct
    engine access (to build determinized samples and simulate forward), not
    just an isolated ``(observation, mask)`` pair, so it is driven by a
    bespoke match runner rather than plugged into ``HokmEnv``. That runner
    must call :meth:`observe` after *every* ply of the real match (not just
    this policy's own) and :meth:`reset_hand` at the start of every new
    hand -- the void tracker depends on seeing the whole table.

    Args:
        policy: The trained policy (``model.policy`` of a ``MaskablePPO``).
        n_samples: Determinized worlds sampled per real decision.
        top_k: Number of the wrapped policy's own top legal actions to
            actually search; the rest are assumed dominated.
        max_rollout_plies: Hard cap on plies simulated per sample before
            giving up and bootstrapping anyway (safety bound; the rollout
            normally stops much sooner, at the root seat's next turn).
        seed: Seeds this policy's own determinization RNG.
    """

    policy: object
    n_samples: int = N_SAMPLES_DEFAULT
    top_k: int = TOP_K_DEFAULT
    max_rollout_plies: int = MAX_ROLLOUT_PLIES_DEFAULT
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.voids = VoidTracker()
        self.decisions_searched = 0
        self.decisions_changed = 0

    def reset_hand(self) -> None:
        """Clear void state; call at the start of every new hand."""
        self.voids.reset()

    def observe(self, seat: int, card: int, led_suit: int | None) -> None:
        """Record one publicly-played card from anywhere at the table."""
        self.voids.observe(seat, card, led_suit)

    def _policy_action(self, obs: Observation, mask: np.ndarray) -> int:
        """The wrapped policy's own deterministic choice, no search."""
        policy: Any = self.policy
        action, _ = policy.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    def _value(self, obs: Observation) -> float:
        """The wrapped policy's critic estimate for ``obs``."""
        policy: Any = self.policy
        obs_tensor, _ = policy.obs_to_tensor(obs)
        return float(policy.predict_values(obs_tensor).item())

    def _top_k_legal_actions(
        self, obs: Observation, mask: np.ndarray, legal: list[int]
    ) -> list[int]:
        """The wrapped policy's ``top_k`` legal actions by its own probability."""
        policy: Any = self.policy
        obs_tensor, _ = policy.obs_to_tensor(obs)
        distribution = policy.get_distribution(obs_tensor, action_masks=mask)
        probs = distribution.distribution.probs.detach().cpu().numpy().reshape(-1)
        ranked = sorted(legal, key=lambda a: probs[a], reverse=True)
        return ranked[: self.top_k]

    def _simulate_rollout(self, sim_engine: HokmEngine, root_seat: int) -> float:
        """Roll out ``sim_engine`` (already past the candidate action) and score it.

        Stops -- and returns a value from ``root_seat``'s own perspective --
        as soon as either the hand ends (exact scoring) or it is
        ``root_seat``'s turn again (critic bootstrap). Every intervening
        seat acts only on its own ``observation_for``/``mask_for`` pair, so
        the rollout can never condition on which determinized world it is
        actually in.
        """
        for _ in range(self.max_rollout_plies):
            if sim_engine.state.winner is not None:
                break
            seat = sim_engine.current_seat()
            if seat == root_seat:
                points = sim_engine.state.game_points
                return self._value(observation_for(sim_engine.state.hands, root_seat, points))
            legal = sim_engine.legal_actions(seat)
            obs = observation_for(sim_engine.state.hands, seat, sim_engine.state.game_points)
            mask = mask_for(legal)
            action = self._policy_action(obs, mask)
            outcome = sim_engine.apply_action(action, seat=seat)
            if outcome.hand_complete:
                assert outcome.hand_winner_team is not None
                return 1.0 if outcome.hand_winner_team == team_of(root_seat) else -1.0
        # Safety-bound exit: neither stopping condition fired within the ply
        # cap (should not happen in practice -- a hand has 13 tricks and the
        # root seat is never more than ~6 plies from its own next turn).
        obs = observation_for(sim_engine.state.hands, root_seat, sim_engine.state.game_points)
        return self._value(obs)

    def decide(self, engine: HokmEngine) -> int:
        """Choose the acting seat's next action, searching only card plays.

        Args:
            engine: The real, live match engine (never mutated).

        Returns:
            A legal action id for the engine's current seat.
        """
        root_seat = engine.current_seat()
        legal = engine.legal_actions(root_seat)
        if len(legal) <= 1:
            return legal[0]

        hands = engine.state.hands
        obs = observation_for(hands, root_seat, engine.state.game_points)
        mask = mask_for(legal)
        if hands.phase is not Phase.CARD_PLAY:
            # MVP scope: the trump call is left to the wrapped policy.
            return self._policy_action(obs, mask)

        candidates = self._top_k_legal_actions(obs, mask, legal)
        if len(candidates) <= 1:
            return candidates[0]

        own_hand = hands.hands[root_seat]
        seen = set(hands.played) | set(own_hand)
        unseen_pool = [c for c in range(NUM_CARDS) if c not in seen]
        remaining_sizes = [len(hands.hands[s]) for s in range(NUM_SEATS)]

        # One hoisted clone RNG per decision (fresh, unseeded -- as the
        # per-clone RNGs used to be; a clone only consumes its RNG via a
        # simulated hand's redeal, into state the caller discards).
        clone_rng = random.Random()
        scores = dict.fromkeys(candidates, 0.0)
        for _ in range(self.n_samples):
            sampled_hands = sample_determinized_hands(
                root_seat,
                own_hand,
                unseen_pool,
                remaining_sizes,
                voids=self.voids.voids,
                rng=self._rng,
            )
            for action in candidates:
                sim_engine = _clone_for_simulation(engine, root_seat, sampled_hands, rng=clone_rng)
                outcome: ActionOutcome = sim_engine.apply_action(action, seat=root_seat)
                if outcome.hand_complete:
                    assert outcome.hand_winner_team is not None
                    score = 1.0 if outcome.hand_winner_team == team_of(root_seat) else -1.0
                else:
                    score = self._simulate_rollout(sim_engine, root_seat)
                scores[action] += score / self.n_samples

        best_action = max(candidates, key=lambda a: scores[a])
        self.decisions_searched += 1
        if best_action != candidates[0]:
            self.decisions_changed += 1
        return best_action
