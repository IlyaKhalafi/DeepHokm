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
from deephokm.training.selfplay import P_GREEDY, P_LATEST, P_POOL, P_RANDOM, PoolOpponentProvider

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

# The earlier deviation to gamma=0.95 (paired with trick_reward=0.05) was a
# mistake: it was reasoned from a claim that a +/-1 match outcome ~150 steps
# away carries "almost no gradient" at gamma=0.997. That claim was checked
# directly (0.997**150 ~= 0.64 of the reward retained, vs 0.95**150 ~= 0.0005)
# and is backwards — 0.997 keeps most of the credit over that horizon, 0.95
# destroys nearly all of it. Reverted to the reference gamma=0.997.
#
# Per-trick shaping still has the flaw described in HokmEnv's docstring: it
# pays the same for a trick that decided a close hand as for one that mops
# up an already-settled one, so it can teach the policy to chase tricks that
# no longer matter. hand_reward is the coarser, better-aligned signal — it
# only pays out at the point a hand's outcome is actually decided. Both
# shaping terms are configurable and are disabled for evaluation.
DEFAULT_TRICK_REWARD = 0.10
DEFAULT_HAND_REWARD = 0.30
DEFAULT_GAMMA = 0.997


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trick-reward", type=float, default=DEFAULT_TRICK_REWARD)
    parser.add_argument("--hand-reward", type=float, default=DEFAULT_HAND_REWARD)
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

    ``gamma`` and ``gae_lambda`` are back at the reference values (see the
    comment on :data:`DEFAULT_GAMMA` for why the earlier gamma=0.95 deviation
    was reverted). The remaining deviations from the reference set are each
    motivated by observed training instability. The current experimental
    defaults use two epochs, a 5120-sample minibatch, a 0.02 target-KL
    safety limit, and a fixed 0.005 entropy coefficient. The learning rate
    decays linearly from 1e-3 to a nonzero 1e-5 floor.

    ``n_steps`` is unchanged from the vec-env scaling rationale in
    :data:`DEFAULT_N_STEPS`.

    Args:
        n_steps: Rollout length per worker.
        gamma: Discount factor.
    """
    return {
        "learning_rate": LinearSchedule(1e-3, 1e-5, 1.0),
        "n_steps": n_steps,
        "batch_size": 5120,
        "n_epochs": 2,
        "gamma": gamma,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.02,
    }


def hyperparameters_for_config(
    n_steps: int = DEFAULT_N_STEPS, gamma: float = DEFAULT_GAMMA
) -> dict[str, Any]:
    """Return the hyperparameters in a JSON-serializable form."""
    values = dict(hyperparameters(n_steps, gamma))
    values["learning_rate"] = "linear 1e-3 -> 1e-5"
    return values


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the reproducible config dump."""
    return {
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "trick_reward": args.trick_reward,
        "hand_reward": args.hand_reward,
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
        "opponent_mix": {
            "latest": P_LATEST,
            "pool": P_POOL,
            "greedy": P_GREEDY,
            "random": P_RANDOM,
        },
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
        hand_reward=args.hand_reward,
    )

    def eval_env_factory() -> HokmEnv:
        # Deliberately unshaped: EvalCallback's mean reward is the metric used
        # for best-model selection, and it should reflect the actual +/-1
        # match outcome, not training-time shaping.
        return make_env(rank=0, seed=args.seed + 10_000)

    policy_kwargs: dict[str, object] = {}

    model_kwargs: dict[str, Any] = dict(
        policy=HokmMaskablePolicy,
        env=train_env,
        tensorboard_log=str(args.tensorboard_log),
        seed=args.seed,
        device=args.device,
        verbose=1,
        policy_kwargs=policy_kwargs,
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
