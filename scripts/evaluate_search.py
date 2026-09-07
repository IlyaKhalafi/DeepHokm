"""Evaluate the IIMC search wrapper against a fixed non-searching opponent.

The searching team runs the trained network through :class:`IIMCSearchPolicy`;
the other team runs one of three baselines picked by ``--opponent``:
``plain`` (the *same* network, deterministic, no search -- isolates the
effect of search alone), ``greedy`` (the scripted baseline), or ``random``.
Each seed is played twice with the searching team swapped (paired
comparison), which cancels out any seed-specific advantage and halves the
variance for a given game budget.

Usage::

    uv run python scripts/evaluate_search.py --model checkpoints/m8_main/final.zip \
        --opponent plain --n-seeds 50 --n-samples 8 --top-k 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO

from deephokm.cards import NUM_RANKS
from deephokm.env.spaces import Observation, mask_for, observation_for
from deephokm.policies.base import HokmPolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.random_policy import RandomPolicy
from deephokm.policies.search import IIMCSearchPolicy
from deephokm.rules.engine import HokmEngine
from deephokm.rules.legality import is_trump_action
from deephokm.rules.state import NUM_SEATS, Phase, team_of


class _PlainPolicyOpponent:
    """Wraps the trained network's own deterministic choice as a HokmPolicy."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def act(self, observation: Observation, action_mask: Any) -> int:
        action, _ = self._model.predict(observation, action_masks=action_mask, deterministic=True)
        return int(action)

    def reset(self, seed: int | None = None) -> None:
        del seed


def _make_opponent(kind: str, model: Any, seed: int) -> HokmPolicy:
    """Build the non-searching team's per-seat opponent policy."""
    if kind == "plain":
        return _PlainPolicyOpponent(model)
    if kind == "greedy":
        policy: HokmPolicy = GreedyPolicy()
    elif kind == "random":
        policy = RandomPolicy(seed)
    else:
        raise ValueError(f"unknown opponent kind {kind!r}")
    policy.reset(seed)
    return policy


def _suit_of(card: int) -> int:
    return card // NUM_RANKS


def _broadcast(agents: list[IIMCSearchPolicy | None], fn_name: str, *args: object) -> None:
    """Call ``fn_name(*args)`` on every non-``None`` agent."""
    for agent in agents:
        if agent is not None:
            getattr(agent, fn_name)(*args)


def _play_match(
    policy: Any,
    seed: int,
    searching_team: int,
    *,
    n_samples: int,
    top_k: int,
    opponent_kind: str,
) -> tuple[int, int, int]:
    """Play one match; ``searching_team`` (0 or 1) uses search on both its seats.

    The other team is built fresh from ``opponent_kind`` (see
    :func:`_make_opponent`) for every match, matching how ``HokmEnv``
    reseeds stateful opponents per episode.

    Returns:
        ``(winning_team, decisions_searched, decisions_changed)`` summed
        across both of the searching team's agents.
    """
    search_agents: list[IIMCSearchPolicy | None] = [None] * NUM_SEATS
    opponents: list[HokmPolicy | None] = [None] * NUM_SEATS
    for seat in range(NUM_SEATS):
        if team_of(seat) == searching_team:
            search_agents[seat] = IIMCSearchPolicy(
                policy, n_samples=n_samples, top_k=top_k, seed=seed * NUM_SEATS + seat
            )
        else:
            opponents[seat] = _make_opponent(opponent_kind, policy, seed * NUM_SEATS + seat)

    engine = HokmEngine()
    engine.start_match(seed=seed)
    _broadcast(search_agents, "reset_hand")

    while engine.state.winner is None:
        seat = engine.current_seat()
        agent = search_agents[seat]
        if agent is not None:
            action = agent.decide(engine)
        else:
            hands = engine.state.hands
            obs = observation_for(hands, seat, engine.state.game_points)
            mask = mask_for(engine.legal_actions(seat))
            opponent = opponents[seat]
            assert opponent is not None
            action = opponent.act(obs, mask)

        led_suit = None
        current_trick = engine.state.hands.current_trick
        if engine.state.hands.phase is Phase.CARD_PLAY and current_trick:
            led_suit = _suit_of(current_trick[0][1])
        card = action if not is_trump_action(action) else None

        outcome = engine.apply_action(action, seat=seat)

        if card is not None:
            _broadcast(search_agents, "observe", seat, card, led_suit)
        if outcome.hand_complete:
            _broadcast(search_agents, "reset_hand")

    searched = sum(a.decisions_searched for a in search_agents if a is not None)
    changed = sum(a.decisions_changed for a in search_agents if a is not None)
    assert engine.state.winner is not None
    return engine.state.winner, searched, changed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/m8_main/final.zip"))
    parser.add_argument("--opponent", choices=["plain", "greedy", "random"], default="plain")
    parser.add_argument("--n-seeds", type=int, default=25)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the paired search-vs-opponent evaluation and print a summary."""
    args = parse_args(argv)
    model = MaskablePPO.load(str(args.model), device=args.device)
    policy = model.policy
    policy.set_training_mode(False)

    search_wins = 0
    opponent_wins = 0
    total_searched = 0
    total_changed = 0
    start = time.monotonic()

    for i in range(args.n_seeds):
        seed = args.seed_offset + i
        for searching_team in (0, 1):
            winner, searched, changed = _play_match(
                policy,
                seed,
                searching_team,
                n_samples=args.n_samples,
                top_k=args.top_k,
                opponent_kind=args.opponent,
            )
            total_searched += searched
            total_changed += changed
            if winner == searching_team:
                search_wins += 1
            else:
                opponent_wins += 1
        elapsed = time.monotonic() - start
        played = 2 * (i + 1)
        print(
            f"[eval] {played} games: search {search_wins} - {args.opponent} {opponent_wins} "
            f"({elapsed / played:.1f}s/game avg)"
        )

    total = search_wins + opponent_wins
    print(f"\nFinal: search team won {search_wins}/{total} ({search_wins / total:.3f})")
    print(f"Decisions actually searched: {total_searched}")
    if total_searched:
        print(f"Fraction where search changed the action: {total_changed / total_searched:.3f}")


if __name__ == "__main__":
    main()
