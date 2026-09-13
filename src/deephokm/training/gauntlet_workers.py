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
import os
import random
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


def _pin_thread_if_unset() -> None:
    """Limit torch to one CPU thread unless the user pinned a count.

    ``run_gauntlet_shards`` calls ``_play_shard`` directly, in-process,
    whenever a round collapses to a single shard (``--eval-workers 1``, or
    fewer games than workers). Unlike the spawned-subprocess path, that
    process may be the training process itself, so an unconditional
    ``torch.set_num_threads(1)`` would silently override whatever thread
    count the user configured for training — pinned env vars always win.
    """
    if torch.get_num_threads() > 1 and not any(
        os.environ.get(var) for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS")
    ):
        torch.set_num_threads(1)


def _load_model_preserving_global_rng(model_path: str) -> MaskablePPO:
    """Load a checkpoint without perturbing the caller's global RNG state.

    ``MaskablePPO.load`` reconstructs the model via the algorithm's own
    ``__init__``, which calls ``set_random_seed(self.seed)`` during
    ``_setup_model`` -- and every checkpoint this project trains carries an
    explicit ``seed`` (``train.py`` always passes ``--seed``), so loading one
    deterministically resets python's, numpy's and torch's *global* RNGs to a
    fixed state tied to that seed, not just the loaded model's own state.
    Confirmed directly: ``torch.manual_seed(111); MaskablePPO.load(path);
    torch.rand(4)`` equals the same sequence started from
    ``torch.manual_seed(222)`` whenever the checkpoint has a seed. Harmless
    in a spawned subprocess (nothing else depends on its RNG afterwards), but
    ``run_gauntlet_shards``/``run_team_gauntlet_shards`` call the ``_shard``
    functions directly, in-process, whenever a round collapses to a single
    shard -- and that process may be the training loop itself, which would
    silently collapse its own rollout action sampling onto a fixed, repeating
    stream after every such round. This is the same failure mode the
    now-removed explicit ``model.set_random_seed(0)`` call caused (see
    ``test_single_shard_gauntlet_does_not_reseed_the_global_rng``), just
    reached through the SB3 API's own ``load()`` instead of an extra call
    this code used to make.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        return MaskablePPO.load(model_path, device="cpu")
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


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

    A single-shard round runs this function directly in the caller's own
    process rather than a spawned subprocess (see ``run_gauntlet_shards``),
    so nothing here may touch process-global state the caller depends on:
    every prediction is deterministic (argmax, no sampling) so the model's
    own RNG is never seeded here, loading is wrapped to leave the caller's
    global RNG untouched (see :func:`_load_model_preserving_global_rng`), and
    the thread count is only pinned when unset.
    """
    _pin_thread_if_unset()

    model = _load_model_preserving_global_rng(model_path)

    opponents = spec.build()
    env = HokmEnv(seat=0, opponents=opponents)

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


def _team_env_for(spec: OpponentSpec, controlled_team: int) -> HokmEnv:
    """Build a team-controlled env with ``spec`` filling the other team."""
    opponents = spec.build()
    controlled_placeholder = opponents[0]  # never consulted; controlled seats are skipped
    team_opponents = list(opponents)
    for seat in (controlled_team, controlled_team + 2):
        team_opponents[seat] = controlled_placeholder
    return HokmEnv(seat=controlled_team, opponents=team_opponents, control_partner=True)


def _play_team_shard(
    model_path: str,
    spec: OpponentSpec,
    game_indices: list[int],
    round_seed: int,
    controlled_team: int | None = None,
) -> int:
    """Play ``game_indices`` with the model controlling BOTH seats of one team.

    Unlike :func:`_play_shard` (one learner seat, the opponent policy fills
    every other seat including the learner's own partner), this is the
    honest reading of "our team versus their team": both of the controlled
    team's seats act through the same model, both of the other team's seats
    run ``spec``. The network sees an absolute one-hot seat token, so nothing
    guarantees it plays identically on both team labels: when
    ``controlled_team`` is ``None`` (the default), each game index alternates
    between controlling team 0 (even index) and team 1 (odd index), so a
    round always covers both labels instead of risking a blind spot on the
    one never evaluated; passing 0 or 1 pins a single label (used by tests
    that need one deterministic matchup). See :func:`run_team_gauntlet_shards`
    for why this is a separate function rather than a flag on ``_play_shard``
    -- it shares that function's single-shard in-process / spawned-subprocess
    split and RNG/thread-count care, so it is kept structurally parallel
    rather than branchy.
    """
    _pin_thread_if_unset()
    model = _load_model_preserving_global_rng(model_path)
    envs = {
        team: _team_env_for(spec, team)
        for team in ({0, 1} if controlled_team is None else {controlled_team})
    }
    wins = 0
    for index in game_indices:
        team = index % 2 if controlled_team is None else controlled_team
        wins += int(_play_game(model, envs[team], match_seed(round_seed, index)))
    for env in envs.values():
        env.close()
    return wins


def run_team_gauntlet_shards(
    model_path: str,
    spec: OpponentSpec,
    *,
    n_games: int,
    n_workers: int,
    seed: int,
    controlled_team: int | None = None,
) -> int:
    """Team-vs-team counterpart of :func:`run_gauntlet_shards`.

    The model controls both seats of one team; ``spec`` fills both seats of
    the other. No engine asymmetry favours either team label (the first
    hakem is drawn uniformly per match), so a single team label would be an
    unbiased estimate of "our team's win rate against a team of X" *if* the
    policy played identically on both labels -- but it sees an absolute
    one-hot seat token, so nothing guarantees that. ``controlled_team=None``
    (the default) alternates team 0 and team 1 by game index within each
    shard, so the round always covers both; passing 0 or 1 pins one label
    (used by tests that need a single deterministic matchup).

    Args:
        model_path: Saved evaluation model (the current policy).
        spec: The opponent configuration for the opposing team's two seats.
        n_games: Total games in the round.
        n_workers: Worker processes to split the games across.
        seed: Round index; offsets the game seeds so rounds are independent.
        controlled_team: Fix the controlled team to 0 or 1, or alternate by
            game index when ``None``.

    Returns:
        The number of games the model's team won.
    """
    shards = [shard for shard in _shard_indices(n_games, max(1, n_workers)) if shard]
    if not shards:
        return 0
    if len(shards) == 1:
        return _play_team_shard(
            model_path, spec, shards[0], round_seed=seed, controlled_team=controlled_team
        )
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=len(shards)) as pool:
        results = pool.starmap(
            _play_team_shard,
            [(model_path, spec, shard, seed, controlled_team) for shard in shards],
        )
    return sum(results)


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
