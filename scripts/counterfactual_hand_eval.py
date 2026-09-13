"""Hand-level win rate of the counterfactual search policy against GreedyPolicy.

This is the decisive follow-up to ``counterfactual_diagnostic.py``: that
script showed individual decisions have real, non-noise counterfactual
advantage over GreedyPolicy's own choice; this script checks whether
actually *playing* that way, hand after hand, turns into a real hand win
rate above the ~50% baseline -- and specifically whether it clears the
~61-62% per-hand threshold that a simplified independent-hand model implies
is enough for an ~80% first-to-7 match win rate (see ``PROGRESS.md``).

Usage::

    uv run python scripts/counterfactual_hand_eval.py --n-seeds 150 --n-samples 32
"""

from __future__ import annotations

import argparse
import math
import time

from deephokm.policies.counterfactual_search import (
    MAX_P_VALUE_DEFAULT,
    MIN_N_SAMPLES,
    N_SAMPLES_DEFAULT,
    play_match,
)


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


def _p_value(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be a finite value in [0, 1], got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=_positive_int, default=150)
    parser.add_argument("--n-samples", type=_n_samples, default=N_SAMPLES_DEFAULT)
    parser.add_argument("--max-p-value", type=_p_value, default=MAX_P_VALUE_DEFAULT)
    parser.add_argument("--seed-offset", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    wins = 0
    games = 0
    t0 = time.time()
    for i in range(args.n_seeds):
        seed = args.seed_offset + i
        for controlled_team in (0, 1):
            won, _ = play_match(
                seed=seed,
                controlled_team=controlled_team,
                n_samples=args.n_samples,
                max_p_value=args.max_p_value,
                search_seed=seed * 2 + controlled_team,
                hand_only=True,
            )
            wins += int(won)
            games += 1
        elapsed = time.time() - t0
        print(
            f"seed {seed}: running {wins}/{games} = {wins / games:.3f} "
            f"({elapsed / games:.2f}s/hand)",
            flush=True,
        )
    print(f"FINAL: {wins}/{games} = {wins / games:.4f}")


if __name__ == "__main__":
    main()
