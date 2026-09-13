"""Counterfactual-search diagnostic: does GreedyPolicy have real, exploitable
weaknesses at all, independent of whether PPO can find them?

Plays GreedyPolicy-vs-GreedyPolicy matches (the actual matchup this project
cares about: two greedy opponents, and a greedy partner, at the seats a
trained policy would occupy) and, at a random sample of card-play decisions
for a designated seat, runs the exhaustive one-ply counterfactual search
(:mod:`deephokm.policies.counterfactual_search`) *once* -- not embedded into
full match play, which multiplies the cost by the number of decisions in a
match and made a single 500-game evaluation impractical (~97s/match at
n_samples=48, ~266 searched decisions per match).

Usage::

    uv run python scripts/counterfactual_diagnostic.py --n-decisions 300
"""

from __future__ import annotations

import argparse
import math
import random
import time

from deephokm.cards import NUM_RANKS
from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies.counterfactual_search import (
    MAX_P_VALUE_DEFAULT,
    MIN_N_SAMPLES,
    N_SAMPLES_DEFAULT,
    CounterfactualSearchPolicy,
    DecisionRecord,
)
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.search import VoidTracker
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import NUM_SEATS, HandState, Phase


def _led_suit(hands: HandState) -> int | None:
    """The suit led in the current trick, or ``None`` if empty (leading)."""
    if hands.phase is not Phase.CARD_PLAY or not hands.current_trick:
        return None
    return hands.current_trick[0][1] // NUM_RANKS


def collect_decisions(
    n_decisions: int,
    *,
    sample_rate: float,
    n_samples: int,
    max_p_value: float,
    base_seed: int,
) -> list[DecisionRecord]:
    """Play GreedyPolicy-vs-GreedyPolicy matches, searching a random subset
    of seat 0's card-play decisions along the way.

    Seat 0 is an arbitrary but fixed choice -- Hokm's rules have no seat
    asymmetry (the first hakem is drawn uniformly), so any seat is
    representative of what a trained policy at that table would face. The
    generation trajectory always plays GreedyPolicy's own action (never the
    search's suggestion), so sampled decisions stay i.i.d. draws from the
    real GreedyPolicy-vs-GreedyPolicy matchup rather than a trajectory the
    search itself has started to alter.
    """
    root_seat = 0
    records: list[DecisionRecord] = []
    match_seed = base_seed
    sampler_rng = random.Random(base_seed * 7919 + 1)  # independent of match_seed's stream

    while len(records) < n_decisions:
        engine = HokmEngine()
        engine.start_match(seed=match_seed)
        match_seed += 1
        opponents = [GreedyPolicy() for _ in range(NUM_SEATS)]
        voids = VoidTracker()
        current_hand = engine.state.hand_number

        while engine.state.winner is None and len(records) < n_decisions:
            if engine.state.hand_number != current_hand:
                voids.reset()
                current_hand = engine.state.hand_number
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
                search = CounterfactualSearchPolicy(
                    n_samples=n_samples,
                    max_p_value=max_p_value,
                    seed=sampler_rng.randrange(2**31),
                )
                search.voids = voids  # reuse the true, live void state
                search.decide(engine)  # clones internally; never mutates engine
                record = search.records[0]
                records.append(record)
                action = record.greedy_action  # keep the generation trajectory pure-greedy
            else:
                obs = observation_for(hands, seat, engine.state.game_points)
                mask = mask_for(legal)
                action = opponents[seat].act(obs, mask)

            led_suit = _led_suit(hands)
            outcome = engine.apply_action(action, seat=seat)
            if outcome.card is not None:
                voids.observe(seat, outcome.card, led_suit)

    return records[:n_decisions]


def summarize(records: list[DecisionRecord]) -> None:
    epsilon = 1e-9
    n = len(records)
    switched = [r for r in records if r.chosen_action != r.greedy_action]
    positive_raw = [r for r in records if r.raw_advantage > epsilon]
    advantages = [r.raw_advantage for r in records]
    mean_adv = sum(advantages) / n
    max_adv = max(advantages)

    print(f"decisions searched: {n}")
    print(f"switched from GreedyPolicy's action (p-value <= max_p_value): "
          f"{len(switched)} ({len(switched) / n:.1%})")
    print(f"decisions with ANY positive raw advantage (before threshold): "
          f"{len(positive_raw)} ({len(positive_raw) / n:.1%})")
    print(f"mean raw advantage: {mean_adv:+.4f}")
    print(f"max raw advantage: {max_adv:+.4f}")
    if switched:
        switched_adv = [r.raw_advantage for r in switched]
        mean_switched_adv = sum(switched_adv) / len(switched_adv)
        print(f"mean advantage among switched decisions: {mean_switched_adv:+.4f}")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def _n_samples(raw: str) -> int:
    value = int(raw)
    if value < MIN_N_SAMPLES:
        raise argparse.ArgumentTypeError(f"must be >= {MIN_N_SAMPLES}, got {value}")
    return value


def _sample_rate(raw: str) -> float:
    value = float(raw)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {value}")
    return value


def _p_value(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be a finite value in [0, 1], got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-decisions", type=_positive_int, default=300)
    parser.add_argument("--sample-rate", type=_sample_rate, default=0.5)
    parser.add_argument("--n-samples", type=_n_samples, default=N_SAMPLES_DEFAULT)
    parser.add_argument("--max-p-value", type=_p_value, default=MAX_P_VALUE_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.time()
    records = collect_decisions(
        args.n_decisions,
        sample_rate=args.sample_rate,
        n_samples=args.n_samples,
        max_p_value=args.max_p_value,
        base_seed=args.seed,
    )
    elapsed = time.time() - t0
    per_decision = elapsed / len(records)
    print(f"collected {len(records)} decisions in {elapsed:.1f}s ({per_decision:.3f}s/decision)")
    summarize(records)


if __name__ == "__main__":
    main()
