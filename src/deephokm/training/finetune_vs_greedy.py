"""Dedicated best-response fine-tuning against GreedyPolicy.

The self-play recipe in ``train.py`` mixes GreedyPolicy into the training
population at a minority share (see :mod:`deephokm.training.selfplay`) and
always evaluates the learner alone against three copies of the named
opponent, including the learner's own partner seat. Two 8M-step self-play
runs and a near-lossless behavioral-cloning warm start of GreedyPolicy's own
decision rule all plateaued around a 42% single-seat win rate against
GreedyPolicy specifically, with no trend over the whole run -- self-play
never actually ran a training population that was purely, and always,
GreedyPolicy, and never trained one policy to control both team seats
jointly against it.

This CLI runs that missing experiment directly: ``MaskablePPO`` controls
both seats of one team (:class:`~deephokm.env.hokm_env.HokmEnv`'s
``control_partner=True``), the opposing team is always two fresh
``GreedyPolicy`` instances, and the recipe is intended to resume from an
existing checkpoint (the behavioral-cloning warm start or a self-play
checkpoint) rather than train from scratch.

Usage::

    uv run python -m deephokm.training.finetune_vs_greedy \\
        --resume checkpoints/bc_pretrained.zip --total-timesteps 4000000
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

from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.training.callbacks import RollingCheckpointCallback, TeamGauntletCallback
from deephokm.training.env_factory import make_team_vs_greedy_env, make_team_vs_greedy_vec_env
from deephokm.training.gauntlet_workers import OpponentSpec

DEFAULT_OUT_DIR = Path("checkpoints")
CHECKPOINT_EVERY = 100_000
CHECKPOINTS_TO_KEEP = 5
EVAL_FREQ = 100_000
EVAL_GAMES = 200
EVAL_WORKERS = 10

# Both opponent seats are the same fixed, deterministic policy every episode
# (no self-play non-stationarity to guard against), and the warm start
# already imitates it almost losslessly, so the learning rate is lower than
# self-play's 3e-4 -> 3e-5 -- large steps off a good starting point risk
# unlearning it before the (much sparser) team-coordination signal has a
# chance to shape it. gae_lambda is raised from self-play's 0.95 to 0.97:
# gamma * gae_lambda = 0.997 * 0.97 ~= 0.967 versus 0.947, retaining more
# credit across matches that can run 100+ learner decisions.
DEFAULT_GAMMA = 0.997
DEFAULT_GAE_LAMBDA = 0.97
DEFAULT_HAND_REWARD = 0.15
DEFAULT_TRICK_REWARD = 0.0
DEFAULT_N_STEPS = 256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=4_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trick-reward", type=float, default=DEFAULT_TRICK_REWARD)
    parser.add_argument("--hand-reward", type=float, default=DEFAULT_HAND_REWARD)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--gae-lambda", type=float, default=DEFAULT_GAE_LAMBDA)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--n-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--lr-start", type=float, default=1e-4)
    parser.add_argument("--lr-end", type=float, default=1e-5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--eval-every", type=int, default=EVAL_FREQ)
    parser.add_argument("--eval-games", type=int, default=EVAL_GAMES)
    parser.add_argument("--eval-workers", type=int, default=EVAL_WORKERS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tensorboard-log", type=Path, default=Path("logs/tb"))
    parser.add_argument(
        "--hand-only",
        action="store_true",
        help=(
            "Curriculum knob: end each episode at the first completed hand "
            "instead of playing a full match. A full match is a long "
            "(~hundreds of learner decisions), sparse-reward episode; "
            "single-hand episodes give PPO a much shorter horizon and a "
            "denser, lower-variance reward signal to learn joint two-seat "
            "coordination from before tackling full matches. hand_reward "
            "becomes the episode's only reward source in this mode (there "
            "is no match-level reward to shape on top of), so pass a "
            "meaningful --hand-reward (e.g. 1.0), not the small match-level "
            "shaping magnitude used otherwise."
        ),
    )
    return parser.parse_args(argv)


def hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    """Return the PPO hyperparameters for this run."""
    return {
        "learning_rate": LinearSchedule(args.lr_start, args.lr_end, 1.0),
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": 0.2,
        "ent_coef": args.ent_coef,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": args.target_kl,
    }


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the reproducible config dump."""
    values = dict(hyperparameters(args))
    values["learning_rate"] = f"linear {args.lr_start:g} -> {args.lr_end:g}"
    return {
        "resume": str(args.resume),
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "trick_reward": args.trick_reward,
        "hand_reward": args.hand_reward,
        "out_dir": str(args.out_dir),
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "eval_games": args.eval_games,
        "device": args.device,
        "hyperparameters": values,
        "opponent": "GreedyPolicy (both seats, every episode; control_partner=True)",
        "hand_only": args.hand_only,
    }


def _pin_cpu_threads() -> None:
    """Cap CPU thread pools to one when the user has not chosen a count."""
    if not any(
        os.environ.get(var) for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS")
    ):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        torch.set_num_threads(1)


def main(argv: list[str] | None = None) -> None:
    """Run the fine-tune."""
    args = parse_args(argv)
    _pin_cpu_threads()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(build_config(args), indent=2))

    train_env = make_team_vs_greedy_vec_env(
        n_envs=args.n_envs,
        seed=args.seed,
        trick_reward=args.trick_reward,
        hand_reward=args.hand_reward,
        hand_only=args.hand_only,
    )

    def eval_env_factory() -> Any:
        # Deliberately unshaped, matching train.py's EvalCallback: the mean
        # reward used for best-model selection should reflect the real
        # +/-1 match outcome, not training-time shaping.
        return make_team_vs_greedy_env(rank=0, seed=args.seed + 10_000)

    overrides = {
        key: value
        for key, value in dict(
            policy=HokmMaskablePolicy,
            env=train_env,
            tensorboard_log=str(args.tensorboard_log),
            seed=args.seed,
            device=args.device,
            verbose=1,
            **hyperparameters(args),
        ).items()
        if key not in ("policy", "env", "device")
    }
    model = MaskablePPO.load(
        str(args.resume),
        env=train_env,
        device=args.device,
        **overrides,
    )
    print(f"Resumed from {args.resume} for a dedicated GreedyPolicy fine-tune")

    team_gauntlet = TeamGauntletCallback(
        opponent=OpponentSpec(kind="greedy"),
        name="greedy",
        n_games=args.eval_games,
        eval_freq=args.eval_every,
        n_workers=args.eval_workers,
        verbose=1,
    )

    callbacks = [
        RollingCheckpointCallback(
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
            eval_freq=max(1, args.eval_every // args.n_envs),
            n_eval_episodes=32,
            deterministic=True,
            verbose=1,
        ),
        team_gauntlet,
    ]

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=False,
        reset_num_timesteps=True,
    )

    final_path = out_dir / "final.zip"
    model.save(str(final_path))
    print(f"Fine-tune complete; final model at {final_path}")
    train_env.close()


if __name__ == "__main__":
    main()
