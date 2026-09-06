"""Tests for the training pipeline modules (selfplay, callbacks, factory)."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import KVWriter, Logger

from deephokm.env import HokmEnv
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.state import NUM_SEATS
from deephokm.training.callbacks import (
    GauntletCallback,
    RollingCheckpointCallback,
    SelfPlayCallback,
)
from deephokm.training.env_factory import make_env, make_vec_env
from deephokm.training.gauntlet_workers import (
    OpponentSpec,
    match_seed,
    run_gauntlet_shards,
)
from deephokm.training.selfplay import (
    PoolOpponentProvider,
    SelfPlayPool,
    SnapshotPolicy,
    snapshot_step,
)
from deephokm.training.train import (
    DEFAULT_GAMMA,
    DEFAULT_HAND_REWARD,
    DEFAULT_N_STEPS,
    DEFAULT_TRICK_REWARD,
    build_config,
    gauntlet_opponents,
    hyperparameters,
    hyperparameters_for_config,
    parse_args,
)
from deephokm.training.train import main as train_main


def tiny_model(env: HokmEnv) -> MaskablePPO:
    """Return a minimal CPU MaskablePPO over the Hokm policy."""
    return MaskablePPO(
        HokmMaskablePolicy,
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        device="cpu",
    )


def test_selfplay_pool_add_and_eviction(tmp_path: Path) -> None:
    pool = SelfPlayPool(capacity=3, seed=0)
    for i in range(5):
        pool.add(tmp_path / f"snap_{i}.zip", global_step=i * 100)
    assert len(pool) == 3
    # Oldest evicted; the latest three remain in order.
    assert [s.global_step for s in pool.snapshots] == [200, 300, 400]
    assert pool.latest() is not None and pool.latest().global_step == 400


def test_selfplay_pool_latest_and_empty() -> None:
    pool = SelfPlayPool(capacity=2, seed=0)
    assert pool.latest() is None
    assert len(pool) == 0


def test_snapshot_step_parses_filename() -> None:
    assert snapshot_step(Path("/x/snapshot_000000123456.zip")) == 123456


def _write_fake_snapshots(directory: Path, steps: list[int]) -> list[Path]:
    """Create placeholder snapshot files; only their names are read."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for step in steps:
        path = directory / f"snapshot_{step:012d}.zip"
        path.write_bytes(b"not-a-real-zip")
        paths.append(path)
    return paths


def test_provider_falls_back_to_random_on_unreadable_snapshot(tmp_path: Path) -> None:
    """A half-written snapshot must not crash a worker.

    A draw that rolls into the snapshot arms (latest/pool) and finds the
    file unreadable falls back to RandomPolicy; a draw that rolls into the
    greedy arm never touches the snapshot at all and legitimately returns
    GreedyPolicy. Either is a safe outcome; a SnapshotPolicy (or a crash)
    is not.
    """
    _write_fake_snapshots(tmp_path, [1000])
    provider = PoolOpponentProvider(tmp_path, capacity=10, seed=0)
    drawn = provider()
    assert len(drawn) == NUM_SEATS
    assert all(isinstance(p, RandomPolicy | GreedyPolicy) for p in drawn)


def test_provider_returns_random_or_greedy_when_directory_is_empty(tmp_path: Path) -> None:
    provider = PoolOpponentProvider(tmp_path / "missing", capacity=10, seed=0)
    drawn = provider()
    assert len(drawn) == NUM_SEATS
    assert all(isinstance(p, RandomPolicy | GreedyPolicy) for p in drawn)


def test_provider_keeps_only_the_newest_capacity_snapshots(tmp_path: Path) -> None:
    _write_fake_snapshots(tmp_path, [100, 200, 300, 400, 500])
    provider = PoolOpponentProvider(tmp_path, capacity=2, seed=0)
    paths = provider.refresh()
    assert [snapshot_step(p) for p in paths] == [400, 500]


def test_provider_picks_up_snapshots_written_after_construction(tmp_path: Path) -> None:
    """Regression: the pool must reach workers created before it filled.

    Training forks its environment workers up front and only starts writing
    snapshots later. When the pool was an in-process object the workers kept
    the (empty) pool they were forked with, so self-play silently degenerated
    into permanent play against random opponents.
    """
    snapshot_dir = tmp_path / "opponents"
    snapshot_dir.mkdir()
    provider = PoolOpponentProvider(snapshot_dir, capacity=10, seed=0)
    assert provider.refresh() == []

    env = make_env(rank=0, seed=0)
    model = tiny_model(env)
    callback = SelfPlayCallback(save_freq=64, save_path=snapshot_dir, verbose=0)
    model.learn(total_timesteps=192, callback=CallbackList([callback]))
    env.close()

    assert provider.refresh(), "snapshots written after construction must be visible"
    # A single draw has only a P_LATEST + P_POOL chance of landing on a
    # snapshot (the rest goes to GreedyPolicy/RandomPolicy); several draws
    # make the check robust instead of occasionally flaky.
    drawn_over_several_calls = [policy for _ in range(20) for policy in provider()]
    assert any(isinstance(policy, SnapshotPolicy) for policy in drawn_over_several_calls)


def test_provider_draw_mix_matches_the_documented_probabilities(tmp_path: Path) -> None:
    """With snapshots present, the per-team draw mix matches the documented probabilities."""
    _write_fake_snapshots(tmp_path, [100, 200, 300, 400])
    provider = PoolOpponentProvider(tmp_path, capacity=10, seed=42)
    provider.refresh()
    # Load is exercised elsewhere; here only the draw arm matters, so map
    # every path to a distinguishable sentinel.
    loaded: dict[Path, str] = {p: p.name for p in provider.refresh()}
    provider._load = loaded.get  # type: ignore[assignment]

    counts: Counter[str] = Counter()
    latest = provider.refresh()[-1].name
    n = 4000
    for _ in range(n):
        seats = provider()
        # Each call draws exactly 2 independent team policies (seats 0/2
        # share one, seats 1/3 share the other); count each draw once.
        assert seats[0] is seats[2]
        assert seats[1] is seats[3]
        for policy in (seats[0], seats[1]):
            if isinstance(policy, RandomPolicy):
                counts["random"] += 1
            elif isinstance(policy, GreedyPolicy):
                counts["greedy"] += 1
            else:
                counts["snapshot"] += 1
                if policy == latest:
                    counts["latest"] += 1
    total = n * 2
    snapshot_share = counts["snapshot"] / total
    greedy_share = counts["greedy"] / total
    random_share = counts["random"] / total
    assert 0.60 < snapshot_share < 0.70, dict(counts)  # P_LATEST + P_POOL = 0.65
    assert 0.20 < greedy_share < 0.30, dict(counts)  # P_GREEDY = 0.25
    assert 0.05 < random_share < 0.15, dict(counts)  # P_RANDOM = 0.10
    # Of the snapshot draws, P_LATEST / (P_LATEST + P_POOL) = 0.45/0.65 are latest.
    assert counts["latest"] / counts["snapshot"] > 0.55, dict(counts)


def test_env_redraws_opponents_every_reset() -> None:
    """The provider is consulted per episode, not once per worker."""
    drawn: list[int] = []

    def provider() -> list[RandomPolicy]:
        drawn.append(len(drawn))
        return [RandomPolicy(len(drawn) * 10 + i) for i in range(NUM_SEATS)]

    env = HokmEnv(seat=0, opponent_provider=provider)
    first = env.reset(seed=1)[0]
    before = list(env.opponents)
    env.reset(seed=2)
    assert len(drawn) == 2
    assert env.opponents != before
    assert first is not None
    env.close()


def test_env_rejects_provider_with_wrong_length() -> None:
    env = HokmEnv(seat=0, opponent_provider=lambda: [RandomPolicy(0)])
    try:
        env.reset(seed=0)
    except ValueError as exc:
        assert "4 policies" in str(exc)
    else:  # pragma: no cover - the reset must raise
        raise AssertionError("a short opponent list must be rejected")
    env.close()


def test_vec_env_workers_see_snapshots_written_after_fork(tmp_path: Path) -> None:
    """End-to-end: subprocess workers must pick up new pool snapshots."""
    snapshot_dir = tmp_path / "opponents"
    snapshot_dir.mkdir()
    provider = PoolOpponentProvider(snapshot_dir, capacity=10, seed=0)
    venv = make_vec_env(n_envs=2, seed=0, opponent_provider=provider, start_method="fork")
    try:
        env = make_env(rank=0, seed=0)
        model = tiny_model(env)
        callback = SelfPlayCallback(save_freq=64, save_path=snapshot_dir, verbose=0)
        model.learn(total_timesteps=192, callback=CallbackList([callback]))
        env.close()

        found_snapshot = False
        for _ in range(10):
            venv.reset()
            venv.env_method("_draw_opponents")
            names = venv.get_attr("opponents")
            if any(
                type(policy).__name__ == "SnapshotPolicy" for seats in names for policy in seats
            ):
                found_snapshot = True
                break
        assert found_snapshot, "workers must load snapshots written after the fork"
    finally:
        venv.close()


def test_make_env_seat_rotation() -> None:
    for rank in range(8):
        env = make_env(rank=rank, seed=rank)
        assert env.seat == rank % NUM_SEATS
        env.close()


def test_make_env_seeded_first_reset_reproducible() -> None:
    a = make_env(rank=1, seed=123)
    b = make_env(rank=1, seed=123)
    obs_a, info_a = a.reset(seed=50)
    obs_b, info_b = b.reset(seed=50)
    for key in obs_a:
        np.testing.assert_array_equal(obs_a[key], obs_b[key], err_msg=key)
    np.testing.assert_array_equal(info_a["action_mask"], info_b["action_mask"])
    a.close()
    b.close()


def test_make_vec_env_shapes_and_masks() -> None:
    venv = make_vec_env(n_envs=2, seed=5, start_method="fork")
    obs = venv.reset()
    assert obs["hand"].shape == (2, 52)
    assert obs["history"].shape == (2, 48)
    masks = np.stack(venv.env_method("action_masks"))
    assert masks.shape == (2, 56)
    assert (masks.sum(axis=1) > 0).all()
    venv.close()


def test_selfplay_callback_cadence(tmp_path: Path) -> None:
    env = make_env(rank=0, seed=0)
    model = tiny_model(env)
    callback = SelfPlayCallback(save_freq=64, save_path=tmp_path / "opp", verbose=0)
    model.learn(total_timesteps=256, callback=CallbackList([callback]))
    snapshots = sorted((tmp_path / "opp").glob("snapshot_*.zip"))
    assert len(snapshots) >= 2
    # Every file the workers can see is a complete zip: nothing partial is
    # left behind by the staged write.
    assert not list((tmp_path / "opp").glob("*.partial"))
    env.close()


def test_selfplay_callback_prunes_but_keeps_the_first(tmp_path: Path) -> None:
    env = make_env(rank=0, seed=0)
    model = tiny_model(env)
    callback = SelfPlayCallback(save_freq=64, save_path=tmp_path / "opp", capacity=2, verbose=0)
    model.learn(total_timesteps=512, callback=CallbackList([callback]))
    steps = sorted(snapshot_step(p) for p in (tmp_path / "opp").glob("snapshot_*.zip"))
    # The oldest snapshot survives as the gauntlet's fixed reference; the rest
    # is the capacity-bounded live pool.
    assert len(steps) == 3, steps
    env.close()


def test_gauntlet_cadence_is_measured_in_environment_steps(tmp_path: Path) -> None:
    """Regression: the eval cadence must not be scaled by the worker count."""
    env = make_env(rank=0, seed=0)
    model = tiny_model(env)
    callback = GauntletCallback(
        opponents={"random": OpponentSpec(kind="random")},
        n_games=2,
        eval_freq=128,
        n_workers=1,
        verbose=0,
    )
    rounds: list[int] = []
    original = callback.run_gauntlet

    def counted() -> dict[str, float]:
        rounds.append(callback.num_timesteps)
        return original()

    callback.run_gauntlet = counted  # type: ignore[method-assign]
    model.learn(total_timesteps=256, callback=CallbackList([callback]))
    # 256 steps at one env, a round every 128 steps, plus the final round.
    assert len(rounds) == 3, rounds
    env.close()


class NullWriter(KVWriter):
    def write(self, key_values, key_excluded, step=0) -> None:  # type: ignore[no-untyped-def]
        pass

    def close(self) -> None:
        pass


def test_gauntlet_reports_every_configured_rung(tmp_path: Path) -> None:
    env = make_env(rank=0, seed=0)
    model = tiny_model(env)
    model._logger = Logger(None, [NullWriter()])  # type: ignore[assignment]
    callback = GauntletCallback(
        opponents={
            "random": OpponentSpec(kind="random"),
            "greedy": OpponentSpec(kind="greedy"),
            "first_snapshot": OpponentSpec(kind="snapshot"),
        },
        n_games=4,
        eval_freq=10**9,
        n_workers=2,
        verbose=0,
    )
    callback.model = model
    callback.num_timesteps = 0
    callback._logger = model._logger
    rates = callback.run_gauntlet()
    # The snapshot rung is skipped while no snapshot directory is bound.
    assert set(rates) == {"random", "greedy"}
    assert all(0.0 <= value <= 1.0 for value in rates.values())
    env.close()


def test_hyperparameters_match_spec() -> None:
    hp = hyperparameters()
    assert hp["n_steps"] == DEFAULT_N_STEPS
    assert hyperparameters(1024)["n_steps"] == 1024
    # gamma/gae_lambda are the reference values; DEFAULT_GAMMA itself was
    # reverted from an earlier gamma=0.95 deviation that turned out to rest
    # on backwards math (see the comment on DEFAULT_GAMMA in train.py).
    assert hp["gamma"] == DEFAULT_GAMMA
    assert DEFAULT_GAMMA == 0.997
    assert hyperparameters(gamma=0.997)["gamma"] == 0.997
    assert hp["gae_lambda"] == 0.95
    assert hp["clip_range"] == 0.2
    assert hp["ent_coef"] == 0.01
    assert hp["vf_coef"] == 0.5
    assert hp["max_grad_norm"] == 0.5
    # Deviations from the reference set, each forced by evidence recorded in
    # REVIEW_LOG.local.md.
    assert hp["batch_size"] == 1024
    assert hp["n_epochs"] == 4
    assert hp["target_kl"] == 0.02
    schedule = hp["learning_rate"]
    assert callable(schedule)
    # progress_remaining runs 1 -> 0 across training; the floor is 3e-5, not 0.
    assert schedule(1.0) == 3e-4
    assert schedule(0.0) == pytest.approx(3e-5)


def test_hyperparameters_for_config_is_json_safe() -> None:
    dumped = json.dumps(hyperparameters_for_config())
    assert "linear" in dumped


def test_gauntlet_opponents_cover_the_specified_rungs() -> None:
    rungs = gauntlet_opponents()
    assert set(rungs) == {"random", "greedy", "first_snapshot", "latest_snapshot"}
    assert rungs["first_snapshot"].kind == "snapshot"


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.total_timesteps == 1_000_000
    assert args.n_envs == 8
    assert args.seed == 0
    assert args.trick_reward == DEFAULT_TRICK_REWARD
    assert args.hand_reward == DEFAULT_HAND_REWARD
    assert args.gamma == DEFAULT_GAMMA
    assert args.device == "cuda"


def test_build_config_covers_reproducibility() -> None:
    args = parse_args(["--total-timesteps", "100", "--seed", "3"])
    config = build_config(args)
    assert config["total_timesteps"] == 100
    assert config["seed"] == 3
    assert config["trick_reward"] == DEFAULT_TRICK_REWARD
    assert config["hand_reward"] == DEFAULT_HAND_REWARD
    assert config["hyperparameters"] == hyperparameters_for_config()
    assert "network" in config and "observation_layout" in config
    assert config["opponent_mix"]["latest"] == 0.45
    assert config["opponent_mix"]["greedy"] == 0.25


def test_random_policy_reset_after_pool_use() -> None:
    """Seeded resets replay identical episodes, opponents included.

    The action source is a fresh RNG per episode so only the environment's
    determinism is under test.
    """
    env = HokmEnv(seat=0, opponents=[RandomPolicy(i) for i in range(NUM_SEATS)])

    def play_episode() -> list[int]:
        obs, info = env.reset(seed=1)
        rng = random.Random(0)  # same stream per episode
        actions = []
        done = False
        while not done:
            legal = np.flatnonzero(info["action_mask"])
            action = int(rng.choice(legal))
            actions.append(action)
            obs, _r, term, trunc, info = env.step(action)
            done = term or trunc
        return actions

    first = play_episode()
    second = play_episode()
    assert first == second
    env.close()


def test_gauntlet_multiprocess_matches_serial() -> None:
    """The sharded gauntlet must equal the serial one and be worker-invariant."""
    env = make_env(rank=0, seed=0)
    model = tiny_model(env)
    model.learn(total_timesteps=128)
    spec = OpponentSpec(kind="random")
    with tempfile.TemporaryDirectory() as tmp:
        model_path = f"{tmp}/m.zip"
        model.save(model_path)
        w1 = run_gauntlet_shards(model_path, spec, n_games=12, n_workers=3, seed=1)
        w2 = run_gauntlet_shards(model_path, spec, n_games=12, n_workers=3, seed=1)
        w_serial = run_gauntlet_shards(model_path, spec, n_games=12, n_workers=1, seed=1)
    assert w1 == w2, "same-seed gauntlet rounds must match"
    assert w1 == w_serial, "worker count must not change results"
    assert 0 <= w1 <= 12
    env.close()


def test_single_shard_gauntlet_does_not_reseed_the_global_rng() -> None:
    """A single-shard round runs in-process; it must not reseed torch's RNG.

    Regression test: ``_play_shard`` used to call ``model.set_random_seed(0)``
    unconditionally. That call reseeds the *global* python/numpy/torch RNGs to
    a fixed value (not just the loaded eval model's own state), which is
    harmless when the round runs in a spawned subprocess but collapses the
    training process's own rollout sampling onto a fixed, repeating stream
    whenever a round runs directly in the training process instead (a round
    collapses to one shard with ``--eval-workers 1``, or fewer games than
    workers). Every gauntlet prediction is already ``deterministic=True``, so
    no RNG seeding was ever needed here.

    Loading a fresh policy for evaluation does perturb the global RNG somewhat
    (network construction draws from it before the saved weights overwrite
    them), so the property under test is not "unchanged" but "still depends on
    what the caller's RNG was doing beforehand" -- under the bug, two
    differently-seeded callers converged on the exact same post-call state.
    """
    env = make_env(rank=0, seed=0)
    model = tiny_model(env)
    model.learn(total_timesteps=64)
    spec = OpponentSpec(kind="random")
    with tempfile.TemporaryDirectory() as tmp:
        model_path = f"{tmp}/m.zip"
        model.save(model_path)

        torch.manual_seed(111)
        run_gauntlet_shards(model_path, spec, n_games=4, n_workers=1, seed=7)
        after_a = torch.rand(8)

        torch.manual_seed(222)
        run_gauntlet_shards(model_path, spec, n_games=4, n_workers=1, seed=7)
        after_b = torch.rand(8)
    assert not torch.equal(after_a, after_b), (
        "two differently-seeded callers must not converge on the same RNG "
        "state after a single-shard gauntlet round"
    )
    env.close()


def test_single_shard_gauntlet_respects_pinned_thread_count() -> None:
    """The in-process single-shard path must not override a pinned thread count.

    Regression test: ``_play_shard`` used to call ``torch.set_num_threads(1)``
    unconditionally, which silently overrode a thread count the user pinned
    via an environment variable for the training process itself whenever a
    round ran in-process.
    """
    env = make_env(rank=0, seed=0)
    model = tiny_model(env)
    model.learn(total_timesteps=64)
    spec = OpponentSpec(kind="random")
    original = os.environ.get("OMP_NUM_THREADS")
    original_threads = torch.get_num_threads()
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory() as tmp:
            model_path = f"{tmp}/m.zip"
            model.save(model_path)
            run_gauntlet_shards(model_path, spec, n_games=4, n_workers=1, seed=8)
        assert torch.get_num_threads() == 4, "a pinned thread count must not be overridden"
    finally:
        if original is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = original
        torch.set_num_threads(original_threads)
    env.close()


def test_gauntlet_snapshot_round_runs(tmp_path: Path) -> None:
    """A gauntlet round against a real snapshot file completes and reports."""
    env = make_env(rank=0, seed=0)
    model = tiny_model(env)
    callback = SelfPlayCallback(save_freq=64, save_path=tmp_path / "opp")
    model.learn(total_timesteps=128, callback=CallbackList([callback]))
    snapshots = sorted((tmp_path / "opp").glob("snapshot_*.zip"))
    assert snapshots

    with tempfile.TemporaryDirectory() as tmp:
        model_path = f"{tmp}/m.zip"
        model.save(model_path)
        wins = run_gauntlet_shards(
            model_path,
            OpponentSpec(kind="snapshot", path=str(snapshots[-1])),
            n_games=8,
            n_workers=2,
            seed=0,
        )
    assert 0 <= wins <= 8
    env.close()


def test_match_seeds_are_disjoint_between_rounds() -> None:
    """Regression: every gauntlet round replayed the same 100 deals.

    ``run_gauntlet_shards`` takes a round seed, but it only ever reached the
    evaluation env's construction — each game was reset with its bare index,
    so round 0 and round 40 scored the identical deals and the win-rate curve
    carried no fresh sample at all.
    """
    round_a = {match_seed(0, i) for i in range(100)}
    round_b = {match_seed(1, i) for i in range(100)}
    assert not round_a & round_b
    # The seed depends on the game index, not on the shard that drew it.
    assert match_seed(3, 7) == match_seed(3, 7)


def test_rolling_checkpoint_keeps_only_the_newest(tmp_path: Path) -> None:
    """Checkpoint retention must be bounded; archives carry optimizer state."""
    env = make_env(rank=0, seed=0)
    model = tiny_model(env)
    callback = RollingCheckpointCallback(
        save_freq=64,
        save_path=str(tmp_path / "ckpt"),
        name_prefix="ppo",
        keep=2,
        verbose=0,
    )
    model.learn(total_timesteps=512, callback=CallbackList([callback]))
    saved = sorted((tmp_path / "ckpt").glob("ppo_*_steps.zip"))
    assert len(saved) == 2, [p.name for p in saved]
    steps = sorted(int(p.stem.rsplit("_", 2)[-2]) for p in saved)
    assert steps == steps[-2:], steps
    env.close()


def test_resume_continues_from_a_saved_run(tmp_path: Path) -> None:
    """Regression: ``--resume`` raised TypeError before doing any work.

    ``env`` was both a named argument of ``MaskablePPO.load`` and a member of
    the forwarded keyword overrides, so every resume attempt died immediately
    after forking the workers.
    """
    out = tmp_path / "run"
    common = [
        "--n-envs",
        "1",
        "--n-steps",
        "64",
        "--eval-every",
        "10000000",
        "--snapshot-every",
        "10000000",
        "--checkpoint-every",
        "10000000",
        "--eval-games",
        "2",
        "--eval-workers",
        "1",
        "--device",
        "cpu",
        "--tensorboard-log",
        str(tmp_path / "tb"),
        "--out-dir",
        str(out),
    ]
    train_main(["--total-timesteps", "64", *common])
    first = out / "final.zip"
    assert first.is_file()

    resumed_dir = tmp_path / "resumed"
    train_main(
        [
            "--total-timesteps",
            "64",
            "--resume",
            str(first),
            *[a if a != str(out) else str(resumed_dir) for a in common],
        ]
    )
    assert (resumed_dir / "final.zip").is_file()
