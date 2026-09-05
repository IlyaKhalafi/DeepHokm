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
from pathlib import Path
from typing import Any

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import LinearSchedule

from deephokm.env import HokmEnv
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.training.callbacks import (
    GauntletCallback,
    RollingCheckpointCallback,
    SelfPlayCallback,
)
from deephokm.training.env_factory import make_env, make_vec_env
from deephokm.training.gauntlet_workers import OpponentSpec
from deephokm.training.selfplay import P_LATEST, P_POOL, P_RANDOM, PoolOpponentProvider

DEFAULT_OUT_DIR = Path("checkpoints")
SNAPSHOT_EVERY = 100_000
POOL_CAPACITY = 10
CHECKPOINT_EVERY = 100_000
CHECKPOINTS_TO_KEEP = 5
EVAL_FREQ = 250_000
EVAL_GAMES = 100
EVAL_WORKERS = 10

# Rollout length *per worker*. The reference value of 1024 was written for a
# single environment; with N parallel workers the rollout is N x n_steps
# samples, and at 32 workers that is 32k samples per policy update — a few
# hundred updates for a whole multi-million-step run. 256 keeps the rollout
# near the reference size (8k samples at 32 workers) so the update budget
# scales with the run instead of shrinking as workers are added.
DEFAULT_N_STEPS = 256

# Trick-level shaping and a shorter discount horizon, not the sparse-reward
# gamma=0.997 reference default. A controlled 400k-step comparison (recorded
# in REVIEW_LOG.local.md) found the sparse default still at chance (0.48-0.57
# win rate vs random) while trick_reward=0.05 with gamma=0.95 reached 0.58-0.67
# in the same budget: a +/-1 match outcome roughly 150 steps away carries
# almost no gradient at gamma=0.997, so the dense per-trick signal is what
# actually teaches trick-taking within a reasonable step budget.
DEFAULT_TRICK_REWARD = 0.05
DEFAULT_GAMMA = 0.95


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trick-reward", type=float, default=DEFAULT_TRICK_REWARD)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--eval-every", type=int, default=EVAL_FREQ)
    parser.add_argument("--eval-games", type=int, default=EVAL_GAMES)
    parser.add_argument("--eval-workers", type=int, default=EVAL_WORKERS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tensorboard-log", type=Path, default=Path("logs/tb"))
    return parser.parse_args(argv)


def hyperparameters(n_steps: int = DEFAULT_N_STEPS, gamma: float = DEFAULT_GAMMA) -> dict[str, Any]:
    """Return the PPO hyperparameters actually used for training.

    These start from the project's reference set. Three deviations are
    deliberate, each forced by evidence recorded in ``REVIEW_LOG.local.md``:

    - ``target_kl`` stops the epoch loop once an update has moved the policy
      by more than 0.03 nats. The first full run climbed to about 68% against
      random play and then collapsed back to chance while ``approx_kl`` grew
      past 0.4 — ten epochs at a fixed 3e-4 over a 56-action masked space
      walks the policy far outside the trust region every update.
    - ``learning_rate`` decays linearly to zero across the run, so late
      updates refine instead of overwriting.
    - ``gamma`` (see :data:`DEFAULT_GAMMA`) and the paired ``trick_reward``
      shaping default: at the reference gamma=0.997 a +/-1 match outcome some
      150 steps away carries almost no gradient, so a controlled comparison
      found the sparse-reward default still at chance after 400k steps.

    ``n_steps`` is the fourth: see :data:`DEFAULT_N_STEPS`.

    Args:
        n_steps: Rollout length per worker.
        gamma: Discount factor.
    """
    return {
        "learning_rate": LinearSchedule(3e-4, 0.0, 1.0),
        "n_steps": n_steps,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": gamma,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.03,
    }


def hyperparameters_for_config(
    n_steps: int = DEFAULT_N_STEPS, gamma: float = DEFAULT_GAMMA
) -> dict[str, Any]:
    """Return the hyperparameters in a JSON-serializable form."""
    values = dict(hyperparameters(n_steps, gamma))
    values["learning_rate"] = "linear 3e-4 -> 0"
    return values


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
        "eval_games": args.eval_games,
        "hyperparameters": hyperparameters_for_config(args.n_steps, args.gamma),
        "observation_layout": "hand(13) trick(4) history(48) context(5)",
        "opponent_mix": {"latest": P_LATEST, "pool": P_POOL, "random": P_RANDOM},
        "network": {
            "extractor": "HokmTransformerExtractor",
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 512,
            "dropout": 0.0,
        },
    }


def gauntlet_opponents() -> dict[str, OpponentSpec]:
    """Return the gauntlet rungs.

    Beyond the specified random / earliest-snapshot / latest-snapshot rungs,
    the scripted greedy baseline is included: uniform random play is a very
    low bar, and greedy gives the win-rate curve a fixed, non-trivial
    reference that does not move as the pool evolves.
    """
    return {
        "random": OpponentSpec(kind="random"),
        "greedy": OpponentSpec(kind="greedy"),
        "first_snapshot": OpponentSpec(kind="snapshot"),
        "latest_snapshot": OpponentSpec(kind="snapshot"),
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
    snapshot_dir = out_dir / "opponents"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # The workers are forked, so the opponent pool cannot be a shared Python
    # object: they read the snapshot directory instead and pick up whatever
    # the training process has written by the time an episode starts.
    provider = PoolOpponentProvider(
        snapshot_dir,
        capacity=POOL_CAPACITY,
        seed=args.seed,
    )

    train_env = make_vec_env(
        n_envs=args.n_envs,
        seed=args.seed,
        opponent_provider=provider,
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
        **hyperparameters(args.n_steps, args.gamma),
    )

    if args.resume is not None:
        # ``load`` takes the policy class and the environment from the saved
        # archive and its own named arguments, so neither may be repeated in
        # the overrides — passing them again is a TypeError.
        overrides = {
            key: value
            for key, value in model_kwargs.items()
            if key not in ("policy", "env", "device")
        }
        model = MaskablePPO.load(
            str(args.resume),
            env=train_env,
            device=args.device,
            **overrides,
        )
        print(f"Resumed from {args.resume}")
    else:
        model = MaskablePPO(**model_kwargs)

    gauntlet = GauntletCallback(
        opponents=gauntlet_opponents(),
        n_games=args.eval_games,
        eval_freq=args.eval_every,
        n_workers=args.eval_workers,
        verbose=1,
    )
    gauntlet.bind_snapshot_dir(snapshot_dir)

    callbacks = [
        SelfPlayCallback(
            save_freq=args.snapshot_every,
            save_path=snapshot_dir,
            capacity=POOL_CAPACITY,
            verbose=1,
        ),
        RollingCheckpointCallback(
            # CheckpointCallback counts _on_step calls, one per rollout step
            # across ALL parallel envs; convert the env-step cadence.
            save_freq=max(1, args.checkpoint_every // args.n_envs),
            save_path=str(out_dir / "checkpoints"),
            name_prefix="ppo",
            save_replay_buffer=False,
            save_vecnormalize=False,
            keep=CHECKPOINTS_TO_KEEP,
            verbose=1,
        ),
        MaskableEvalCallback(
            Monitor(eval_env_factory()),
            best_model_save_path=str(out_dir / "best"),
            # EvalCallback counts its own _on_step calls (one per vectorized
            # step), so the env-step cadence is divided by the worker count.
            eval_freq=max(1, args.eval_every // args.n_envs),
            n_eval_episodes=32,
            deterministic=True,
            verbose=1,
        ),
        gauntlet,
    ]

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=False,
        reset_num_timesteps=args.resume is None,
    )

    final_path = out_dir / "final.zip"
    model.save(str(final_path))
    print(f"Training complete; final model at {final_path}")
    train_env.close()


if __name__ == "__main__":
    main()
