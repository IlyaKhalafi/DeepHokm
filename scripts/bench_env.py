"""Benchmark the Hokm environment's throughput.

Measures learner steps per second for a single worker (and optionally several
processes) under a masked random policy, printing the aggregate rate. Used to
verify the >= 1000 env steps/s/worker requirement.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from deephokm.env import HokmEnv
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.state import NUM_SEATS


def run_single(seat: int, seed: int, steps: int) -> int:
    """Play masked-random for exactly ``steps`` learner steps; return count."""
    env = HokmEnv(seat=seat, opponents=[RandomPolicy(seed + i) for i in range(NUM_SEATS)])
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    done = False
    taken = 0
    while taken < steps:
        if done:
            obs, info = env.reset()
            done = False
        legal = np.flatnonzero(info["action_mask"])
        _, _, term, trunc, info = env.step(int(rng.choice(legal)))
        done = term or trunc
        taken += 1
    return taken


MIN_STEPS_PER_SECOND = 1000  # per-worker throughput requirement


def main() -> None:
    """Run the benchmark and print steps/second."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20000, help="learner steps to time")
    parser.add_argument("--seat", type=int, default=None, help="learner seat (default: 0)")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    args = parser.parse_args()

    seat = args.seat if args.seat is not None else args.seed % NUM_SEATS
    start = time.perf_counter()
    taken = run_single(seat, args.seed, args.steps)
    elapsed = time.perf_counter() - start
    rate = taken / elapsed
    print(f"seat={seat} steps={taken} elapsed={elapsed:.2f}s rate={rate:,.0f} steps/s")
    if rate < MIN_STEPS_PER_SECOND:
        raise SystemExit(f"throughput below {MIN_STEPS_PER_SECOND} steps/s per worker: {rate:,.0f}")


if __name__ == "__main__":
    main()
