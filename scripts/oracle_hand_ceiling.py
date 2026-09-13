"""Hand-level clairvoyant ceiling: starting right after the trump call of
each hand (not a random mid-hand point), how many hands can a fully-
informed TEAM 0 win within a bounded lookahead over its own future
decisions, versus how many it actually wins playing GreedyPolicy?

Team 0 is fixed regardless of who holds hakem: ``oracle_ceiling`` already
fast-forwards through every non-team-0 turn (the hakem's own turns
included, whichever team that is) via plain GreedyPolicy and only branches
once it reaches one of team 0's own decisions, so passing it the state
right after the trump call is resolved -- with no manual "skip to team 0's
first turn" loop needed -- is correct and simpler. (An earlier version of
this script instead used ``engine.current_seat()`` right after the trump
call, which is *always the hakem's seat* per the rules -- "the hakem leads
the first trick" -- silently measuring the hakem's own team's ceiling
every time, not a representative sample of team 0. A large, cheap,
oracle-free check confirmed this: unconditional team-0 hand win rate is
~48% as expected, but the hakem's team specifically wins ~57.5% -- a real
effect, but well short of the ~64% the old script reported, the rest being
ordinary sampling noise on top of that selection bug.)

``depth`` does not map onto "tricks of the hand": it counts every one of
team 0's own *non-forced* decisions (both seats combined), and a team can
face on the order of 20+ raw turns across a 13-trick hand (many forced
once void constraints bite, especially late) -- so ``--depth 6`` should be
read as "the team's first 6 real choices", not "the whole hand". See
``scripts/oracle_depth_coverage.py`` for how much of a hand a given depth
typically covers.

Usage::

    uv run python scripts/oracle_hand_ceiling.py --n-hands 50 --depth 6
"""

from __future__ import annotations

import argparse
import time

from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.oracle_search import DEPTH_DEFAULT, oracle_ceiling
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase, team_of


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def _play_to_first_card_play_decision(engine: HokmEngine, seed: int, greedy: GreedyPolicy) -> None:
    """Advance a fresh match past the trump call to the first hand's first
    card-play decision (the hakem's), in place.
    """
    engine.start_match(seed=seed)
    while engine.state.hands.phase is Phase.TRUMP_CALL:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        obs = observation_for(engine.state.hands, seat, engine.state.game_points)
        engine.apply_action(greedy.act(obs, mask_for(legal)), seat=seat)


def collect_hand_ceilings(
    n_hands: int, *, depth: int, base_seed: int, controlled_team: int = 0
) -> list[tuple[float, float, bool]]:
    """Returns a list of ``(actual_outcome, oracle_outcome, team_is_hakem)``
    tuples, one per hand, from ``controlled_team``'s perspective, sampled
    across distinct match seeds (each match's first hand only, to keep
    every sample statistically independent and cheap to reach).
    """
    greedy = GreedyPolicy()
    results: list[tuple[float, float, bool]] = []
    seed = base_seed
    while len(results) < n_hands:
        engine = HokmEngine()
        _play_to_first_card_play_decision(engine, seed, greedy)
        seed += 1
        team_is_hakem = team_of(engine.state.hands.hakem) == controlled_team
        actual = oracle_ceiling(engine, controlled_team, depth=0)
        best = oracle_ceiling(engine, controlled_team, depth=depth)
        results.append((actual, best, team_is_hakem))
    return results


def _report(label: str, rows: list[tuple[float, float, bool]]) -> None:
    n = len(rows)
    if n == 0:
        print(f"  {label}: no hands in this stratum")
        return
    actual_wins = sum(1 for a, _, _ in rows if a > 0)
    oracle_wins = sum(1 for _, b, _ in rows if b > 0)
    flipped = sum(1 for a, b, _ in rows if b > a)
    print(f"  {label} (n={n}): actual {actual_wins}/{n} = {actual_wins / n:.3f}, "
          f"oracle {oracle_wins}/{n} = {oracle_wins / n:.3f}, "
          f"flipped {flipped}/{n} = {flipped / n:.3f}")


def summarize(results: list[tuple[float, float, bool]], depth: int) -> None:
    print(f"hands checked: {len(results)} (depth={depth}, team 0 fixed, "
          f"from right after each hand's trump call)")
    _report("overall", results)
    _report("team 0 IS hakem", [r for r in results if r[2]])
    _report("team 0 is NOT hakem", [r for r in results if not r[2]])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-hands", type=_positive_int, default=50)
    parser.add_argument("--depth", type=int, default=DEPTH_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.time()
    results = collect_hand_ceilings(args.n_hands, depth=args.depth, base_seed=args.seed)
    elapsed = time.time() - t0
    per_hand = elapsed / len(results)
    print(f"collected {len(results)} hands in {elapsed:.1f}s ({per_hand:.2f}s/hand)", flush=True)
    summarize(results, args.depth)


if __name__ == "__main__":
    main()
