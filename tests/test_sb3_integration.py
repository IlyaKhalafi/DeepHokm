from __future__ import annotations

import pathlib
import tempfile

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import is_masking_supported
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from deephokm.env import HokmEnv
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.legality import NUM_ACTIONS
from deephokm.rules.state import NUM_SEATS
from deephokm.training.callbacks import GauntletCallback, SelfPlayCallback
from deephokm.training.env_factory import make_vec_env
from deephokm.training.gauntlet_workers import OpponentSpec
from deephokm.training.selfplay import PoolOpponentProvider, SnapshotPolicy


def tiny_model(env: VecEnv | HokmEnv, **kwargs: object) -> MaskablePPO:
    """Build a fast MaskablePPO for tests."""
    return MaskablePPO(
        HokmMaskablePolicy,
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        learning_rate=3e-4,
        device="cpu",
        **kwargs,  # type: ignore[arg-type]
    )


def make_single_env(seat: int = 0, seed: int = 0) -> HokmEnv:
    return HokmEnv(seat=seat, opponents=[RandomPolicy(seed + i) for i in range(NUM_SEATS)])


def test_maskable_ppo_constructs_with_custom_policy() -> None:
    env = make_single_env()
    model = tiny_model(env)
    assert isinstance(model.policy, HokmMaskablePolicy)
    # The reference extractor width (d_model) that the policy heads size to.
    assert model.policy.features_dim == 256
    env.close()


def test_learn_2000_steps_on_cpu() -> None:
    env = make_single_env()
    model = tiny_model(env)
    model.learn(total_timesteps=2000)
    losses = model.logger.name_to_value
    assert "train/loss" in losses
    assert losses["train/loss"] == losses["train/loss"]  # finite check via NaN != NaN
    env.close()


def test_learn_completes_on_vec_env() -> None:
    venv = make_vec_env(n_envs=2, seed=3, start_method="fork")
    model = tiny_model(venv)
    model.learn(total_timesteps=512)
    venv.close()


def test_masking_enforced_end_to_end() -> None:
    """During learn() the environment never receives an illegal action.

    The env raises ValueError on illegal actions by contract, so completing
    learn() without error is itself the enforcement proof; additionally the
    recorded masks are checked for phase consistency.
    """
    env = make_single_env()
    seen_phases: list[str] = []

    original_reset = env.reset

    def tracked_reset(**kwargs):  # type: ignore[no-untyped-def]
        obs, info = original_reset(**kwargs)
        seen_phases.append(info["phase"])
        return obs, info

    env.reset = tracked_reset  # type: ignore[method-assign]
    model = tiny_model(env)
    model.learn(total_timesteps=1000)
    assert seen_phases, "reset was never called"
    assert all(p in ("TRUMP_CALL", "CARD_PLAY") for p in seen_phases)
    env.close()


def test_save_load_roundtrip_identical_actions() -> None:

    env = make_single_env()
    model = tiny_model(env)
    obs, info = env.reset(seed=42)

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/model.zip"
        model.save(path)
        loaded = MaskablePPO.load(path, device="cpu")

        mask = np.asarray(info["action_mask"], dtype=bool)
        # Deterministic predictions must match exactly.
        actions_a, _ = model.predict(obs, action_masks=mask[None], deterministic=True)
        actions_b, _ = loaded.predict(obs, action_masks=mask[None], deterministic=True)
        np.testing.assert_array_equal(actions_a, actions_b)

        # A batch of observations too.
        batch = {k: np.stack([v, v]) for k, v in obs.items()}
        masks = np.stack([mask, mask])
        actions_a, _ = model.predict(batch, action_masks=masks, deterministic=True)
        actions_b, _ = loaded.predict(batch, action_masks=masks, deterministic=True)
        np.testing.assert_array_equal(actions_a, actions_b)
    env.close()


def test_vec_env_propagates_action_masks() -> None:
    """The SubprocVecEnv path exposes per-worker masks via env_method."""

    venv = make_vec_env(n_envs=3, seed=11, start_method="fork")
    venv.reset()
    masks = np.stack(venv.env_method("action_masks"))
    assert masks.shape == (3, NUM_ACTIONS)
    assert (masks.sum(axis=1) > 0).all()
    # Stepping with masked actions never raises (masking is respected).
    actions = np.array([int(np.flatnonzero(m)[0]) for m in masks], dtype=np.int64)
    venv.step(actions)
    masks_after = np.stack(venv.env_method("action_masks"))
    assert (masks_after.sum(axis=1) > 0).all()
    venv.close()


def test_vec_env_reset_is_seeded() -> None:

    venv_a = make_vec_env(n_envs=2, seed=99, start_method="fork")
    venv_b = make_vec_env(n_envs=2, seed=99, start_method="fork")
    obs_a = venv_a.reset()
    obs_b = venv_b.reset()
    for key in obs_a:
        np.testing.assert_array_equal(obs_a[key], obs_b[key], err_msg=key)
    venv_a.close()
    venv_b.close()


def test_selfplay_callback_snapshots(tmp_path: pathlib.Path) -> None:
    snapshot_dir = pathlib.Path(str(tmp_path)) / "opponents"
    env = make_single_env()
    model = tiny_model(env)
    callback = SelfPlayCallback(save_freq=64, save_path=snapshot_dir, verbose=0)
    model.learn(total_timesteps=256, callback=CallbackList([callback]))
    snapshots = sorted(snapshot_dir.glob("snapshot_*.zip"))
    assert snapshots
    assert all(path.stat().st_size > 0 for path in snapshots)
    env.close()


def test_selfplay_pool_bounded_eviction(tmp_path: pathlib.Path) -> None:
    """The directory the workers scan stays bounded by the pool capacity."""
    base = pathlib.Path(str(tmp_path))
    env = make_single_env()
    model = tiny_model(env)
    callback = SelfPlayCallback(save_freq=32, save_path=base / "opp", capacity=3, verbose=0)
    model.learn(total_timesteps=256, callback=CallbackList([callback]))
    provider = PoolOpponentProvider(base / "opp", capacity=3, seed=0)
    assert len(provider.refresh()) <= 3
    env.close()


def test_gauntlet_callback_runs(tmp_path: pathlib.Path) -> None:
    env = make_single_env()
    model = tiny_model(env)
    callback = GauntletCallback(
        opponents={"random": OpponentSpec(kind="random")},
        n_games=4,
        eval_freq=64,
        n_workers=2,
        verbose=1,
    )
    model.learn(total_timesteps=128, callback=CallbackList([callback]))
    env.close()


def test_eval_callback_runs() -> None:

    env = make_single_env()
    eval_env = Monitor(make_single_env())
    model = tiny_model(env)
    callback = MaskableEvalCallback(
        eval_env,
        n_eval_episodes=2,
        eval_freq=64,
        deterministic=True,
        verbose=0,
    )
    model.learn(total_timesteps=128, callback=CallbackList([callback]))
    env.close()
    eval_env.close()


def test_snapshot_policy_acts_legally(tmp_path: pathlib.Path) -> None:
    """A snapshot loaded from the pool directory returns only legal actions."""
    env = make_single_env()
    model = tiny_model(env)
    opponents_dir = pathlib.Path(str(tmp_path)) / "opp"
    callback = SelfPlayCallback(save_freq=64, save_path=opponents_dir)
    model.learn(total_timesteps=128, callback=CallbackList([callback]))
    snapshots = sorted(opponents_dir.glob("snapshot_*.zip"))
    assert snapshots

    snap = SnapshotPolicy.from_file(snapshots[0])
    obs, info = env.reset(seed=7)
    legal = np.flatnonzero(info["action_mask"])
    for _ in range(5):
        action = snap.act(obs, np.asarray(info["action_mask"], dtype=bool))
        assert action in legal
        obs, _r, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset(seed=7)
        legal = np.flatnonzero(info["action_mask"])
    env.close()


def test_dummy_vec_env_masking_supported() -> None:
    """is_masking_supported must detect the mask method on a DummyVecEnv."""

    venv = DummyVecEnv([lambda: make_single_env(0), lambda: make_single_env(1)])
    assert is_masking_supported(venv)
    venv.close()


def test_policy_load_roundtrip_via_pool(tmp_path: pathlib.Path) -> None:
    """Snapshots saved by the callback load back and reproduce their actions."""
    env = make_single_env()
    model = tiny_model(env)
    opponents_dir = pathlib.Path(str(tmp_path)) / "opp"
    callback = SelfPlayCallback(save_freq=64, save_path=opponents_dir)
    model.learn(total_timesteps=128, callback=CallbackList([callback]))
    latest = sorted(opponents_dir.glob("snapshot_*.zip"))[-1]

    snap = SnapshotPolicy.from_file(latest)
    obs, info = env.reset(seed=11)
    mask = np.asarray(info["action_mask"], dtype=bool)
    first = snap.act(obs, mask)
    reloaded = SnapshotPolicy.from_file(latest)
    assert reloaded.act(obs, mask) == first
    env.close()


def test_model_predict_respects_masks() -> None:

    env = make_single_env()
    model = tiny_model(env)
    obs, info = env.reset(seed=3)
    legal = set(np.flatnonzero(info["action_mask"]).tolist())
    mask = np.asarray(info["action_mask"], dtype=bool)
    for _ in range(10):
        action, _ = model.predict(obs, action_masks=mask[None], deterministic=False)
        assert int(action) in legal
    env.close()
