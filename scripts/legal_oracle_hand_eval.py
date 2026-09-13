"""Hand win rate of the legal K=1 search policy against GreedyPolicy.

The decisive follow-up to the oracle ceiling
(``scripts/oracle_hand_ceiling.py``): that showed a real, large gap
between GreedyPolicy's actual hand win rate and what a fully-informed team
could force. This script measures how much of that gap survives when the
policy must guess the hidden hands instead of seeing them (see
``deephokm.policies.legal_oracle_search`` for the design). Same regime as
``oracle_hand_ceiling.py`` (team 0 fixed, each match's first hand only,
hakem-stratified) for a fair, paired comparison.

Usage::

    uv run python scripts/legal_oracle_hand_eval.py --n-hands 200 --depth 2
"""

from __future__ import annotations

import argparse
import time

from deephokm.policies.legal_oracle_search import play_hand
from deephokm.policies.oracle_search import DEPTH_DEFAULT


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def collect_results(
    n_hands: int, *, depth: int, base_seed: int, controlled_team: int = 0
) -> list[tuple[bool, bool]]:
    """Returns a list of ``(won, team_is_hakem)`` tuples, one per hand."""
    results = []
    for i in range(n_hands):
        seed = base_seed + i
        result = play_hand(
            seed=seed, controlled_team=controlled_team, depth=depth, search_seed=seed
        )
        results.append((result.won, result.team_is_hakem))
    return results


def _report(label: str, rows: list[tuple[bool, bool]]) -> None:
    n = len(rows)
    if n == 0:
        print(f"  {label}: no hands in this stratum")
        return
    wins = sum(1 for won, _ in rows if won)
    print(f"  {label} (n={n}): {wins}/{n} = {wins / n:.3f}")


def summarize(results: list[tuple[bool, bool]], depth: int) -> None:
    print(f"hands checked: {len(results)} (depth={depth}, legal K=1 search, team 0)")
    _report("overall", results)
    _report("team 0 IS hakem", [r for r in results if r[1]])
    _report("team 0 is NOT hakem", [r for r in results if not r[1]])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-hands", type=_positive_int, default=200)
    parser.add_argument("--depth", type=int, default=DEPTH_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.time()
    results = collect_results(args.n_hands, depth=args.depth, base_seed=args.seed)
    elapsed = time.time() - t0
    per_hand = elapsed / len(results)
    print(f"collected {len(results)} hands in {elapsed:.1f}s ({per_hand:.2f}s/hand)", flush=True)
    summarize(results, args.depth)


if __name__ == "__main__":
    main()
