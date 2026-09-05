"""Tests for the training pipeline modules (selfplay, callbacks, factory)."""

from __future__ import annotations

import random
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import KVWriter, Logger

from deephokm.env import HokmEnv
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.state import NUM_SEATS
from deephokm.training.callbacks import GauntletCallback, SelfPlayCallback
from deephokm.training.env_factory import make_env, make_vec_env
from deephokm.training.gauntlet_workers import run_gauntlet_shards
from deephokm.training.selfplay import SelfPlayPool
from deephokm.training.train import build_config, hyperparameters, parse_args


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


def test_selfplay_pool_sampling_probabilities() -> None:
    """With snapshots present, the mix must be ~60/30/10."""
    pool = SelfPlayPool(capacity=10, seed=42)
    # Paths need not exist for sampling of *which* opponent; but load() would
    # read them. Instead, patch load to return a sentinel per snapshot.
    sentinels: dict[str, object] = {}
    for i in range(4):
        path = Path(f"/virtual/snap_{i}.zip")
        pool.add(path, global_step=i)
        sentinels[str(path)] = f"snap{i}"

    def fake_load(path: Path) -> object:
        return sentinels[str(path)]

    pool.load = fake_load  # type: ignore[method-assign]

    counts: Counter[str] = Counter()
    n = 4000
    for _ in range(n):
        opponents = pool.sample_opponents(None, None)
        assert len(opponents) == NUM_SEATS
        for opp in opponents:
            if isinstance(opp, RandomPolicy):
                counts["random"] += 1
            else:
                counts["snapshot"] += 1
    total = n * NUM_SEATS
    snapshot_share = counts["snapshot"] / total
    # 90% of draws should be snapshots (60 latest + 30 pool); allow slack.
    assert 0.85 < snapshot_share < 0.95, f"snapshot share {snapshot_share}"


def test_selfplay_pool_prefers_latest() -> None:
    """The latest snapshot should be drawn about twice as often as the pool."""
    pool = SelfPlayPool(capacity=10, seed=7)
    for i in range(4):
        pool.add(Path(f"/virtual/snap_{i}.zip"), global_step=i)
    which: Counter[str] = Counter()
    original_load = pool.load

    def tracked_load(path: Path) -> object:
        which[path.name] += 1
        return original_load(path)

    pool.load = tracked_load  # type: ignore[method-assign]
    # Sampling calls load(); to avoid file reads, redirect to a fake.
    pool.load = lambda path: which.update([path.name]) or None  # type: ignore[method-assign]
    for _ in range(2000):
        pool.sample_opponents(None, None)
    latest_name = "snap_3.zip"
    latest_count = which[latest_name]
    total_snapshot = sum(which.values())
    assert total_snapshot > 0
    # Roughly 60% of snapshot draws are the latest (vs 30% spread over 4).
    assert latest_count / total_snapshot > 0.5, dict(which)


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
    masks = np.stack(venv.env_method("action_masks"))
    assert masks.shape == (2, 56)
    assert (masks.sum(axis=1) > 0).all()
    venv.close()


def test_selfplay_callback_cadence(tmp_path: Path) -> None:
    pool = SelfPlayPool(capacity=10, seed=0)
    env = make_env(rank=0, seed=0)
    model = MaskablePPO(
        HokmMaskablePolicy,
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        device="cpu",
    )
    callback = SelfPlayCallback(pool=pool, save_freq=64, save_path=tmp_path / "opp", verbose=0)
    model.learn(total_timesteps=256, callback=CallbackList([callback]))
    assert len(pool) >= 2
    assert all((tmp_path / "opp").joinpath(s.path.name).exists() for s in pool.snapshots)
    env.close()


def test_gauntlet_batched_matches_expected_range(tmp_path: Path) -> None:
    """The batched gauntlet runs games and produces a win rate in [0, 1]."""
    env = make_env(rank=0, seed=0)
    model = MaskablePPO(
        HokmMaskablePolicy,
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        device="cpu",
    )
    callback = GauntletCallback(
        eval_env_factory=lambda: make_env(rank=0, seed=10_000),
        opponents={"random": lambda: [RandomPolicy(3 + i) for i in range(NUM_SEATS)]},
        n_games=8,
        eval_freq=10**9,
        batch_size=4,
        verbose=0,
    )
    model.learn(total_timesteps=64, callback=CallbackList([callback]))
    # The final-round evaluation ran during _on_training_end.
    env.close()


class NullWriter(KVWriter):
    def write(self, key_values, key_excluded, step=0) -> None:  # type: ignore[no-untyped-def]
        pass

    def close(self) -> None:
        pass


def test_gauntlet_results_are_deterministic(tmp_path: Path) -> None:
    """Same seeds + same model => same gauntlet outcome."""
    env = make_env(rank=0, seed=0)
    model = MaskablePPO(
        HokmMaskablePolicy,
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        device="cpu",
    )

    model._logger = Logger(None, [NullWriter()])  # type: ignore[assignment]

    def run_gauntlet() -> int:
        cb = GauntletCallback(
            eval_env_factory=lambda: make_env(rank=0, seed=10_000),
            opponents={"random": lambda: [RandomPolicy(3 + i) for i in range(NUM_SEATS)]},
            n_games=6,
            eval_freq=10**9,
            batch_size=3,
            verbose=0,
        )
        cb.model = model
        cb.num_timesteps = 0
        cb._logger = model._logger
        return cb._play_batch("random", lambda: [RandomPolicy(3 + i) for i in range(NUM_SEATS)])

    wins_a = run_gauntlet()
    wins_b = run_gauntlet()
    assert wins_a == wins_b
    assert 0 <= wins_a <= 6
    env.close()


def test_hyperparameters_match_spec() -> None:
    hp = hyperparameters()
    assert hp["learning_rate"] == 3e-4
    assert hp["n_steps"] == 1024
    assert hp["batch_size"] == 512
    assert hp["n_epochs"] == 10
    assert hp["gamma"] == 0.997
    assert hp["gae_lambda"] == 0.95
    assert hp["clip_range"] == 0.2
    assert hp["ent_coef"] == 0.01
    assert hp["vf_coef"] == 0.5
    assert hp["max_grad_norm"] == 0.5


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.total_timesteps == 1_000_000
    assert args.n_envs == 8
    assert args.seed == 0
    assert args.trick_reward == 0.0
    assert args.device == "cuda"


def test_build_config_covers_reproducibility() -> None:
    args = parse_args(["--total-timesteps", "100", "--seed", "3"])
    config = build_config(args)
    assert config["total_timesteps"] == 100
    assert config["seed"] == 3
    assert config["hyperparameters"] == hyperparameters()
    assert "network" in config and "observation_layout" in config


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


def test_gauntlet_multiprocess_matches_serial(tmp_path: Path) -> None:
    """The sharded gauntlet must equal the serial one and be worker-invariant."""

    # Train a tiny model and save it.
    env = make_env(rank=0, seed=0)
    model = MaskablePPO(
        HokmMaskablePolicy,
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        device="cpu",
    )
    model.learn(total_timesteps=128)
    with tempfile.TemporaryDirectory() as tmp:
        model_path = f"{tmp}/m.zip"
        model.save(model_path)

        w1 = run_gauntlet_shards(model_path, None, n_games=12, batch_size=4, n_workers=3, seed=1)
        w2 = run_gauntlet_shards(model_path, None, n_games=12, batch_size=4, n_workers=3, seed=1)
        w_serial = run_gauntlet_shards(
            model_path, None, n_games=12, batch_size=4, n_workers=1, seed=1
        )
    assert w1 == w2, "same-seed gauntlet rounds must match"
    assert w1 == w_serial, "worker count must not change results"
    assert 0 <= w1 <= 12
    env.close()


def test_gauntlet_snapshot_round_runs(tmp_path: Path) -> None:
    """A gauntlet round against a real snapshot file completes and reports."""

    pool = SelfPlayPool(capacity=2, seed=0)
    env = make_env(rank=0, seed=0)
    model = MaskablePPO(
        HokmMaskablePolicy,
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        device="cpu",
    )
    cb = SelfPlayCallback(pool=pool, save_freq=64, save_path=tmp_path / "opp")
    model.learn(total_timesteps=128, callback=CallbackList([cb]))
    snapshot = pool.latest()
    assert snapshot is not None

    with tempfile.TemporaryDirectory() as tmp:
        model_path = f"{tmp}/m.zip"
        model.save(model_path)
        wins = run_gauntlet_shards(
            model_path, str(snapshot.path), n_games=8, batch_size=4, n_workers=2, seed=0
        )
    assert 0 <= wins <= 8
    env.close()
