"""Multiprocess gauntlet evaluation workers.

Each worker process plays a shard of the gauntlet games with its own batched
game loop and its own copy of the evaluation model, so a 100-game gauntlet
round costs a fraction of the wall time it would take inside the training
process. Games stay deterministic: game ``i`` always uses seed ``i`` and the
same opponent configuration, regardless of which worker plays it.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from deephokm.env.hokm_env import HokmEnv
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.state import NUM_SEATS, team_of
from deephokm.training.env_factory import make_env
from deephokm.training.selfplay import SelfPlayPool

MAX_GAMES_PER_BATCH = 32


def _play_shard(
    model_path: str,
    snapshot_path: str | None,
    game_indices: list[int],
    batch_size: int,
    seed_offset: int,
) -> int:
    """Play the given games; return how many the learner won.

    Runs inside a worker process: loads its own model and opponent snapshot,
    then advances ``batch_size`` games at a time with stacked forward passes.
    """
    torch.set_num_threads(1)

    model = MaskablePPO.load(model_path, device="cpu")
    model.set_random_seed(0)

    opponents: list[Any]
    if snapshot_path is None:
        opponents = [RandomPolicy(5 + i) for i in range(NUM_SEATS)]
    else:
        pool = SelfPlayPool(capacity=1, seed=0)
        snap = pool.load(Path(snapshot_path))
        opponents = [snap, snap, snap, snap]

    env = make_env(rank=0, seed=10_000 + seed_offset)
    env.opponents = opponents

    wins = 0
    pending = list(game_indices)
    while pending:
        shard = pending[:batch_size]
        pending = pending[batch_size:]
        wins += _play_batched_games(model, env, shard)
    env.close()
    return wins


def _play_batched_games(model: MaskablePPO, env: HokmEnv, seeds: list[int]) -> int:
    """Play the shard's games against a fixed env; return learner wins.

    Games run sequentially; the learner predicts once per decision and the
    opponents act inside ``env.step()``.
    """
    wins = 0
    # Games run sequentially within the shard but each uses the batched
    # learner predict for its steps; opponents run inside env.step().
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            mask = np.asarray(info["action_mask"], dtype=bool)
            action, _ = model.predict(obs, action_masks=mask[None, ...], deterministic=True)  # type: ignore[arg-type]
            obs, _reward, terminated, truncated, info = env.step(
                np.int64(int(np.asarray(action).reshape(-1)[0]))
            )
            done = terminated or truncated
        winner = env._engine.state.winner
        assert winner is not None
        wins += int(winner == team_of(env.seat))
    return wins


def _shard_indices(n_games: int, n_workers: int) -> list[list[int]]:
    """Split game indices into near-equal shards."""
    indices = list(range(n_games))
    if n_workers <= 1:
        return [indices]
    size = (n_games + n_workers - 1) // n_workers
    return [indices[i : i + size] for i in range(0, n_games, size)]


def run_gauntlet_shards(
    model_path: str,
    snapshot_path: str | None,
    *,
    n_games: int,
    batch_size: int,
    n_workers: int,
    seed: int,
) -> int:
    """Play ``n_games`` across worker processes; return total learner wins.

    Args:
        model_path: Saved evaluation model (the current policy).
        snapshot_path: Opponent snapshot file, or None for random opponents.
        n_games: Total games in the round.
        batch_size: Games advanced concurrently per worker.
        n_workers: Worker processes to split the games across.
        seed: Round seed (offsets game seeds so rounds are independent).

    Returns:
        The number of games the learner won.
    """
    shards = _shard_indices(n_games, max(1, n_workers))
    shards = [shard for shard in shards if shard]
    if not shards:
        return 0
    if len(shards) == 1:
        return _play_shard(
            model_path,
            snapshot_path,
            shards[0],
            batch_size=batch_size,
            seed_offset=seed * 100_000,
        )
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=len(shards)) as pool:
        results = pool.starmap(
            _play_shard,
            [
                (
                    model_path,
                    snapshot_path,
                    shard,
                    batch_size,
                    seed * 100_000 + w,
                )
                for w, shard in enumerate(shards)
            ],
        )
    return sum(results)
