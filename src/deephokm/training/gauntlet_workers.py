"""Multiprocess gauntlet evaluation workers.

Each worker process plays a shard of the gauntlet games with its own copy of
the evaluation model and its own opponents, so a 100-game gauntlet round
costs a fraction of the wall time it would take inside the training process.
Games stay deterministic: game ``i`` always uses seed ``i`` and the same
opponent configuration, regardless of which worker plays it.

The learner occupies one seat and the named opponent policy occupies the
other three, its own partner seat included. That is the honest reading of
"win rate against X": a lone learner steering a table of X.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from deephokm.env.hokm_env import HokmEnv
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.state import NUM_SEATS, team_of
from deephokm.training.env_factory import make_env
from deephokm.training.selfplay import SnapshotPolicy

# Match seeds are laid out round by round with this stride, so two rounds
# never share a deal however many games a round plays.
GAMES_PER_ROUND_STRIDE = 100_000

RANDOM_SEED_BASE = 5


@dataclass(frozen=True)
class OpponentSpec:
    """A picklable description of one gauntlet opponent.

    Attributes:
        kind: ``"random"``, ``"greedy"`` or ``"snapshot"``.
        path: Snapshot zip path, required for ``"snapshot"``.
    """

    kind: str
    path: str | None = None

    def build(self) -> list[Any]:
        """Instantiate the four seat policies this spec describes."""
        if self.kind == "random":
            return [RandomPolicy(RANDOM_SEED_BASE + i) for i in range(NUM_SEATS)]
        if self.kind == "greedy":
            return [GreedyPolicy() for _ in range(NUM_SEATS)]
        if self.kind == "snapshot":
            if self.path is None:
                raise ValueError("snapshot opponent spec needs a path")
            snap = SnapshotPolicy.from_file(Path(self.path))
            return [snap] * NUM_SEATS
        raise ValueError(f"unknown opponent kind {self.kind!r}")


def match_seed(round_seed: int, index: int) -> int:
    """Return the match seed for game ``index`` of gauntlet round ``round_seed``.

    Rounds are laid out on disjoint seed ranges so two rounds never replay the
    same deals, while a game's deal is independent of which worker drew it.
    """
    return round_seed * GAMES_PER_ROUND_STRIDE + index


def _play_shard(
    model_path: str,
    spec: OpponentSpec,
    game_indices: list[int],
    round_seed: int,
) -> int:
    """Play the given games; return how many the learner won.

    Runs inside a worker process: loads its own model and opponents, then
    plays each assigned game to completion. Game ``i`` of round ``r`` always
    uses match seed ``r * GAMES_PER_ROUND_STRIDE + i``, so a game's deal
    depends on the round and the game index but never on which worker drew
    it — rounds are independent samples and shard layout does not change the
    result.
    """
    torch.set_num_threads(1)

    model = MaskablePPO.load(model_path, device="cpu")
    model.set_random_seed(0)

    env = make_env(rank=0, seed=10_000 + round_seed)
    env.opponents = spec.build()

    wins = 0
    for index in game_indices:
        wins += int(_play_game(model, env, match_seed(round_seed, index)))
    env.close()
    return wins


def _play_game(model: MaskablePPO, env: HokmEnv, seed: int) -> bool:
    """Play one seeded match; return whether the learner's team won."""
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        mask = np.asarray(info["action_mask"], dtype=bool)
        action, _ = model.predict(obs, action_masks=mask[None, ...], deterministic=True)  # type: ignore[arg-type]
        obs, _reward, terminated, truncated, info = env.step(
            np.int64(int(np.asarray(action).reshape(-1)[0]))
        )
        done = terminated or truncated
    winner = env.engine.state.winner
    assert winner is not None
    return bool(winner == team_of(env.seat))


def _shard_indices(n_games: int, n_workers: int) -> list[list[int]]:
    """Split game indices into near-equal shards."""
    indices = list(range(n_games))
    if n_workers <= 1:
        return [indices]
    size = (n_games + n_workers - 1) // n_workers
    return [indices[i : i + size] for i in range(0, n_games, size)]


def run_gauntlet_shards(
    model_path: str,
    spec: OpponentSpec,
    *,
    n_games: int,
    n_workers: int,
    seed: int,
) -> int:
    """Play ``n_games`` across worker processes; return total learner wins.

    Args:
        model_path: Saved evaluation model (the current policy).
        spec: The opponent configuration for every seat but the learner's.
        n_games: Total games in the round.
        n_workers: Worker processes to split the games across.
        seed: Round index; offsets the game seeds so rounds are independent.

    Returns:
        The number of games the learner won.
    """
    shards = [shard for shard in _shard_indices(n_games, max(1, n_workers)) if shard]
    if not shards:
        return 0
    if len(shards) == 1:
        return _play_shard(model_path, spec, shards[0], round_seed=seed)
    # "spawn", not "fork": the training process has an initialized CUDA
    # context, and forking one is undefined behaviour in torch. The workers
    # are CPU-only, and rounds are rare enough that a fresh interpreter per
    # round costs nothing measurable.
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=len(shards)) as pool:
        results = pool.starmap(
            _play_shard,
            [(model_path, spec, shard, seed) for shard in shards],
        )
    return sum(results)
