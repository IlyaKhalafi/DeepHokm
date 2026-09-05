"""The self-play training CLI.

Runs MaskablePPO with the Hokm transformer policy over a vectorized
self-play environment, snapshotting opponents into a bounded pool,
evaluating against a fixed gauntlet, and dumping a reproducible
``config.json`` next to the checkpoints.

Usage::

    uv run python -m deephokm.training.train --total-timesteps 1000000
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from deephokm.env import HokmEnv
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.state import NUM_SEATS
from deephokm.training.callbacks import GauntletCallback, SelfPlayCallback
from deephokm.training.env_factory import make_env, make_vec_env
from deephokm.training.selfplay import SelfPlayPool

DEFAULT_OUT_DIR = Path("checkpoints")
SNAPSHOT_EVERY = 100_000
POOL_CAPACITY = 10
CHECKPOINT_EVERY = 100_000
CHECKPOINTS_TO_KEEP = 5
EVAL_FREQ = 100_000
EVAL_GAMES = 100


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trick-reward", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--eval-every", type=int, default=EVAL_FREQ)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tensorboard-log", type=Path, default=Path("logs/tb"))
    return parser.parse_args(argv)


def hyperparameters() -> dict[str, Any]:
    """Return the PPO hyperparameters (the reference starting set)."""
    return {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": 0.997,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the reproducible config dump."""
    return {
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "trick_reward": args.trick_reward,
        "out_dir": str(args.out_dir),
        "resume": str(args.resume) if args.resume else None,
        "snapshot_every": args.snapshot_every,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "device": args.device,
        "pool_capacity": POOL_CAPACITY,
        "eval_freq": EVAL_FREQ,
        "eval_games": EVAL_GAMES,
        "hyperparameters": hyperparameters(),
        "observation_layout": "hand(13) trick(4) history(13) context(4)",
        "network": {
            "extractor": "HokmTransformerExtractor",
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 512,
            "dropout": 0.0,
        },
    }


def opponent_provider(pool: SelfPlayPool) -> Callable[[Any, Any], list[Any]]:
    """Return a per-worker opponent factory bound to the pool."""

    def provide(_obs_space: Any, _action_space: Any) -> list[Any]:
        return pool.sample_opponents(_obs_space, _action_space)

    return provide


def gauntlet_opponents(pool: SelfPlayPool) -> dict[str, Callable[[], list[Any]]]:
    """Return the gauntlet: random, earliest snapshot, latest snapshot."""

    def random_opponents() -> list[Any]:
        return [RandomPolicy(777 + i) for i in range(NUM_SEATS)]

    def earliest_opponents() -> list[Any]:
        snaps = pool.snapshots
        if not snaps:
            return random_opponents()
        snap = pool.load(snaps[0].path)
        return [snap, snap, snap, snap]

    def latest_opponents() -> list[Any]:
        info = pool.latest()
        if info is None:
            return random_opponents()
        snap = pool.load(info.path)
        return [snap, snap, snap, snap]

    return {
        "random": random_opponents,
        "first_snapshot": earliest_opponents,
        "latest_snapshot": latest_opponents,
    }


def _pin_cpu_threads() -> None:
    """Cap CPU thread pools to one when the user has not chosen a count.

    The machine exposes 160 cores; torch's default intra-op pool then spins
    80+ threads per process on batch-1 tensors, which is pure
    oversubscription. Pinned env vars always win.
    """
    if not any(
        os.environ.get(var) for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS")
    ):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        torch.set_num_threads(1)


def main(argv: list[str] | None = None) -> None:
    """Run training."""
    args = parse_args(argv)
    _pin_cpu_threads()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    pool = SelfPlayPool(capacity=POOL_CAPACITY, seed=args.seed)

    train_env = make_vec_env(
        n_envs=args.n_envs,
        seed=args.seed,
        opponent_provider=opponent_provider(pool),
        trick_reward=args.trick_reward,
    )

    def eval_env_factory() -> HokmEnv:
        return make_env(rank=0, seed=args.seed + 10_000)

    model_kwargs: dict[str, Any] = dict(
        policy=HokmMaskablePolicy,
        env=train_env,
        tensorboard_log=str(args.tensorboard_log),
        seed=args.seed,
        device=args.device,
        verbose=1,
        **hyperparameters(),
    )

    if args.resume is not None:
        model = MaskablePPO.load(
            str(args.resume),
            env=train_env,
            **{k: v for k, v in model_kwargs.items() if k not in ("policy",)},
        )
        print(f"Resumed from {args.resume}")
    else:
        model = MaskablePPO(**model_kwargs)

    callbacks = [
        SelfPlayCallback(
            pool=pool,
            save_freq=args.snapshot_every,
            save_path=out_dir / "opponents",
            verbose=1,
        ),
        CheckpointCallback(
            # CheckpointCallback counts _on_step calls, one per rollout step
            # across ALL parallel envs; convert the env-step cadence.
            save_freq=max(1, args.checkpoint_every // args.n_envs),
            save_path=str(out_dir / "checkpoints"),
            name_prefix="ppo",
            save_replay_buffer=False,
            save_vecnormalize=False,
            verbose=1,
        ),
        MaskableEvalCallback(
            Monitor(eval_env_factory()),
            best_model_save_path=str(out_dir / "best"),
            eval_freq=max(1, args.eval_every // args.n_envs),
            n_eval_episodes=32,
            deterministic=True,
            verbose=1,
        ),
        GauntletCallback(
            eval_env_factory=eval_env_factory,
            opponents=gauntlet_opponents(pool),
            n_games=EVAL_GAMES,
            eval_freq=max(1, args.eval_every // args.n_envs),
            verbose=1,
        ),
    ]

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=False,
    )

    final_path = out_dir / "final.zip"
    model.save(str(final_path))
    print(f"Training complete; final model at {final_path}")
    train_env.close()


if __name__ == "__main__":
    main()
