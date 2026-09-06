"""Supervised behavioral-cloning warm start against the scripted baseline.

Two straight from-scratch self-play runs (see ``REVIEW_LOG.local.md`` and the
README's *Results* section) plateaued at essentially the same win rate
against :class:`~deephokm.policies.greedy_policy.GreedyPolicy` (roughly
0.10-0.24) regardless of gamma, reward shaping, or how much training-time
exposure to it the self-play mix carried. An earlier probe already showed
the network can represent good play (0.79 held-out accuracy imitating
greedy from a small dataset); this module turns that probe into an actual
initialization instead of a diagnostic. It collects a dataset of
(observation, action) pairs from greedy-vs-greedy-vs-greedy-vs-greedy
matches -- every seat played by an independent, freshly seeded
``GreedyPolicy`` -- and fits a fresh :class:`~sb3_contrib.MaskablePPO`
policy to imitate those actions before any RL. The saved checkpoint is a
normal ``MaskablePPO.save()`` archive, loadable through ``train.py``'s
``--resume`` flag exactly like a mid-run checkpoint; PPO fine-tuning then
starts from a policy that already plays a disciplined game.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from deephokm.env.hokm_env import HokmEnv
from deephokm.env.spaces import Observation
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.rules.state import NUM_SEATS

DEFAULT_N_MATCHES = 3000
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_EPOCHS = 8
DEFAULT_BATCH_SIZE = 512
DEFAULT_LR = 1e-3


@dataclass
class BCDataset:
    """A flat dataset of (observation, mask, action) plies from scripted play.

    ``match_ids`` records which match each ply came from (plies from the same
    match share an id, in non-decreasing order) so a held-out split can cut
    at a match boundary instead of an arbitrary ply count -- splitting mid
    match would otherwise leak part of a training match's play into the
    "held-out" set.
    """

    observations: list[Observation]
    masks: np.ndarray
    actions: np.ndarray
    match_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.observations)


def collect_dataset(n_matches: int, seed: int) -> BCDataset:
    """Roll out greedy-vs-greedy-vs-greedy-vs-greedy matches and record every ply.

    Every seat is played by its own :class:`GreedyPolicy` instance, so the
    recorded action at each ply is exactly what a disciplined, non-learning
    player would do in that state. The rules engine is driven directly
    (``env.engine.start_match`` / ``apply_action``), not through
    ``HokmEnv.reset()``/``step()``: those are built around a single learner
    seat and would auto-advance -- and silently drop from the dataset --
    every ply before that seat's first turn.

    Args:
        n_matches: Number of full matches to roll out.
        seed: Base seed; match ``i`` uses ``seed + i``.

    Returns:
        The flat dataset across every match.
    """
    env = HokmEnv(seat=0, opponents=[GreedyPolicy() for _ in range(NUM_SEATS)])
    greedy_by_seat = [GreedyPolicy() for _ in range(NUM_SEATS)]
    observations: list[Observation] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    match_ids: list[int] = []

    for match_idx in range(n_matches):
        match_seed = seed + match_idx
        env.engine.start_match(seed=match_seed)
        for seat, greedy in enumerate(greedy_by_seat):
            greedy.reset(match_seed * NUM_SEATS + seat)
        while env.engine.state.winner is None:
            seat = env.engine.current_seat()
            obs = env._observation_for(seat)  # noqa: SLF001
            mask = env._mask_for(seat)  # noqa: SLF001
            action = greedy_by_seat[seat].act(obs, mask)
            observations.append(obs)
            masks.append(mask)
            actions.append(action)
            match_ids.append(match_idx)
            env.engine.apply_action(action, seat=seat)

    return BCDataset(
        observations=observations,
        masks=np.stack(masks),
        actions=np.array(actions, dtype=np.int64),
        match_ids=np.array(match_ids, dtype=np.int64),
    )


def _stack_observations(observations: list[Observation]) -> dict[str, np.ndarray]:
    """Stack a list of per-ply observation dicts into batched arrays."""
    raw = [cast("dict[str, np.ndarray]", dict(obs)) for obs in observations]
    keys = raw[0].keys()
    return {key: np.stack([obs[key] for obs in raw]) for key in keys}


def _batch_loss_and_correct(
    policy: MaskableActorCriticPolicy,
    batch_obs: dict[str, np.ndarray],
    batch_masks: np.ndarray,
    batch_actions: np.ndarray,
) -> tuple[th.Tensor, int]:
    """Return (loss, correct_count) for one batch, without taking an optimizer step.

    ``correct_count`` (not a per-batch accuracy fraction) is returned
    deliberately: batches vary in size (the last batch of an epoch is
    typically short), so callers must weight by batch size when aggregating
    accuracy across batches rather than averaging per-batch fractions.
    """
    obs_tensor, _ = policy.obs_to_tensor(batch_obs)
    distribution = policy.get_distribution(obs_tensor, action_masks=batch_masks)
    target = th.as_tensor(batch_actions, device=policy.device)
    loss = -distribution.log_prob(target).mean()
    with th.no_grad():
        predicted = distribution.mode()
        correct = int((predicted == target).sum().item())
    return loss, correct


def train_bc(
    policy: MaskableActorCriticPolicy,
    dataset: BCDataset,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
) -> float:
    """Fit ``policy`` to imitate the dataset's actions; return final val accuracy.

    The held-out split is drawn from the *last* ``val_fraction`` of matches
    (by match id, not by ply count): cutting at an arbitrary ply index would
    routinely split a single match's plies across both sets, so its
    early-hand plies would train the network on exactly the match whose
    later plies "held-out" accuracy is then measured against.

    Args:
        policy: The policy to train in place (its parameters are updated).
        dataset: The flat (observation, mask, action) dataset.
        epochs: Number of passes over the training split.
        batch_size: Minibatch size.
        lr: Adam learning rate.
        val_fraction: Fraction of matches held out.
        seed: Shuffling seed for the training split.

    Returns:
        The held-out accuracy after the final epoch.
    """
    match_ids = np.unique(dataset.match_ids)
    n_val_matches = max(1, round(len(match_ids) * val_fraction))
    val_match_ids = set(match_ids[-n_val_matches:].tolist())
    is_val = np.array([m in val_match_ids for m in dataset.match_ids])
    train_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)
    n_train = len(train_idx)

    stacked = _stack_observations(dataset.observations)
    train_obs = {k: v[train_idx] for k, v in stacked.items()}
    train_masks = dataset.masks[train_idx]
    train_actions = dataset.actions[train_idx]
    val_obs = {k: v[val_idx] for k, v in stacked.items()}
    val_masks = dataset.masks[val_idx]
    val_actions = dataset.actions[val_idx]

    optimizer = th.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    val_accuracy = 0.0

    for epoch in range(epochs):
        order = rng.permutation(n_train)
        policy.set_training_mode(True)
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            batch_obs = {k: v[idx] for k, v in train_obs.items()}
            loss, _ = _batch_loss_and_correct(
                policy, batch_obs, train_masks[idx], train_actions[idx]
            )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        policy.set_training_mode(False)
        correct_total = 0
        with th.no_grad():
            for start in range(0, len(val_actions), batch_size):
                batch_obs = {k: v[start : start + batch_size] for k, v in val_obs.items()}
                _, correct = _batch_loss_and_correct(
                    policy,
                    batch_obs,
                    val_masks[start : start + batch_size],
                    val_actions[start : start + batch_size],
                )
                correct_total += correct
        val_accuracy = correct_total / len(val_actions)
        print(f"[bc] epoch {epoch + 1}/{epochs}: val_accuracy={val_accuracy:.4f}")

    return val_accuracy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-matches", type=int, default=DEFAULT_N_MATCHES)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-path", type=Path, default=Path("checkpoints/bc_pretrained.zip"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Collect the dataset, run BC, and save a MaskablePPO-loadable checkpoint."""
    args = parse_args(argv)

    print(f"[bc] collecting {args.n_matches} greedy-vs-greedy matches (seed={args.seed})")
    dataset = collect_dataset(args.n_matches, args.seed)
    print(f"[bc] collected {len(dataset)} plies")

    env = HokmEnv(seat=0, opponents=[GreedyPolicy() for _ in range(NUM_SEATS)])
    model = MaskablePPO(
        policy=HokmMaskablePolicy,
        env=env,
        device=args.device,
        seed=args.seed,
        verbose=0,
    )

    val_accuracy = train_bc(
        model.policy,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"[bc] final held-out accuracy imitating GreedyPolicy: {val_accuracy:.4f}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out_path))
    print(f"[bc] saved warm-started checkpoint to {args.out_path}")


if __name__ == "__main__":
    main()
