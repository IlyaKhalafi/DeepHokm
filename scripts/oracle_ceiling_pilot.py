"""Clairvoyant depth-D ceiling pilot: how much value depends on hidden info?

Samples real card-play decisions from GreedyPolicy-vs-GreedyPolicy games
(the same generation harness as ``counterfactual_diagnostic.py``) and, at
each, computes the exact best achievable hand outcome for the acting
team within a bounded lookahead over its own future decisions -- given
full knowledge of every hand (an oracle no real policy could have). See
``deephokm.policies.oracle_search`` for why this cheap, exact check comes
before building a legal, information-set-correct multi-ply search.

Usage::

    uv run python scripts/oracle_ceiling_pilot.py --n-decisions 300 --depth 2
"""

from __future__ import annotations

import argparse
import random
import time

from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.oracle_search import DEPTH_DEFAULT, OracleCeilingResult, oracle_ceiling
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase, team_of


def collect_results(
    n_decisions: int, *, depth: int, sample_rate: float, base_seed: int
) -> list[OracleCeilingResult]:
    """Play GreedyPolicy-vs-GreedyPolicy matches, oracle-searching a random
    subset of seat 0's card-play decisions along the way (same rationale as
    ``counterfactual_diagnostic.collect_decisions``: any seat is
    representative, since Hokm has no seat asymmetry).
    """
    root_seat = 0
    results: list[OracleCeilingResult] = []
    match_seed = base_seed
    sampler_rng = random.Random(base_seed * 7919 + 1)
    greedy = GreedyPolicy()

    while len(results) < n_decisions:
        engine = HokmEngine()
        engine.start_match(seed=match_seed)
        match_seed += 1

        while engine.state.winner is None and len(results) < n_decisions:
            seat = engine.current_seat()
            hands = engine.state.hands
            legal = engine.legal_actions(seat)

            do_search = (
                seat == root_seat
                and hands.phase is Phase.CARD_PLAY
                and len(legal) > 1
                and sampler_rng.random() < sample_rate
            )
            if do_search:
                team = team_of(root_seat)
                actual = oracle_ceiling(engine, team, depth=0)
                best = oracle_ceiling(engine, team, depth=depth)
                results.append(
                    OracleCeilingResult(
                        hand_number=engine.state.hand_number,
                        root_seat=root_seat,
                        depth=depth,
                        actual_outcome=actual,
                        oracle_outcome=best,
                    )
                )

            obs = observation_for(hands, seat, engine.state.game_points)
            action = greedy.act(obs, mask_for(legal))
            engine.apply_action(action, seat=seat)

    return results[:n_decisions]


def summarize(results: list[OracleCeilingResult]) -> None:
    n = len(results)
    improved = [r for r in results if r.improved]
    print(f"decisions checked: {n} (depth={results[0].depth if results else '-'})")
    print(f"decisions where the oracle forces a strictly better outcome: "
          f"{len(improved)} ({len(improved) / n:.1%})")
    if improved:
        print(f"  (of which, actual outcome was a loss the oracle turns into a win: "
              f"{sum(1 for r in improved if r.actual_outcome < 0)})")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def _sample_rate(raw: str) -> float:
    value = float(raw)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-decisions", type=_positive_int, default=300)
    parser.add_argument("--depth", type=int, default=DEPTH_DEFAULT)
    parser.add_argument("--sample-rate", type=_sample_rate, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.time()
    results = collect_results(
        args.n_decisions, depth=args.depth, sample_rate=args.sample_rate, base_seed=args.seed
    )
    elapsed = time.time() - t0
    per_decision = elapsed / len(results)
    print(f"collected {len(results)} decisions in {elapsed:.1f}s ({per_decision:.3f}s/decision)")
    summarize(results)


if __name__ == "__main__":
    main()
