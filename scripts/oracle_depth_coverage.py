"""How many of a depth budget's units does a hand actually need?

``oracle_ceiling``'s ``depth`` counts team 0's own *non-forced* card-play
decisions (both seats combined), not tricks. This script estimates how
many such decisions a typical hand actually presents, by replaying real
GreedyPolicy-vs-GreedyPolicy hands and counting team 0's own turns where
more than one action is legal -- so a chosen ``--depth`` can be read
against a real distribution instead of guessed at.

Usage::

    uv run python scripts/oracle_depth_coverage.py --n-hands 200
"""

from __future__ import annotations

import argparse
import statistics

from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase, team_of


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def count_real_decisions(n_hands: int, *, controlled_team: int, base_seed: int) -> list[int]:
    greedy = GreedyPolicy()
    counts: list[int] = []
    seed = base_seed
    while len(counts) < n_hands:
        engine = HokmEngine()
        engine.start_match(seed=seed)
        seed += 1
        while engine.state.hands.phase is Phase.TRUMP_CALL:
            seat = engine.current_seat()
            legal = engine.legal_actions(seat)
            obs = observation_for(engine.state.hands, seat, engine.state.game_points)
            engine.apply_action(greedy.act(obs, mask_for(legal)), seat=seat)

        real_decisions = 0
        while True:
            seat = engine.current_seat()
            hands = engine.state.hands
            legal = engine.legal_actions(seat)
            if team_of(seat) == controlled_team and len(legal) > 1:
                real_decisions += 1
            obs = observation_for(hands, seat, engine.state.game_points)
            action = legal[0] if len(legal) == 1 else greedy.act(obs, mask_for(legal))
            outcome = engine.apply_action(action, seat=seat)
            if outcome.hand_complete:
                break
        counts.append(real_decisions)
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-hands", type=_positive_int, default=200)
    parser.add_argument("--controlled-team", type=int, default=0, choices=(0, 1))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    counts = count_real_decisions(
        args.n_hands, controlled_team=args.controlled_team, base_seed=args.seed
    )
    print(f"hands checked: {len(counts)}")
    print(f"real (non-forced) team-{args.controlled_team} decisions per hand: "
          f"mean={statistics.mean(counts):.2f}, median={statistics.median(counts)}, "
          f"min={min(counts)}, max={max(counts)}")
    for d in (2, 3, 4, 6, 8, 10):
        covered = sum(1 for c in counts if c <= d)
        print(f"  depth={d} fully covers the hand's real decisions in "
              f"{covered}/{len(counts)} = {covered / len(counts):.3f} of hands")


if __name__ == "__main__":
    main()
