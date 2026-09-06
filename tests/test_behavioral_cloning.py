"""Tests for the behavioral-cloning warm-start pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from deephokm.env.hokm_env import HokmEnv
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.rules.state import NUM_SEATS
from deephokm.training.behavioral_cloning import collect_dataset, main, train_bc
from deephokm.training.train import hyperparameters


def test_collect_dataset_records_only_legal_actions() -> None:
    dataset = collect_dataset(n_matches=3, seed=0)
    assert len(dataset) > 0
    assert len(dataset.observations) == len(dataset.masks) == len(dataset.actions)
    for mask, action in zip(dataset.masks, dataset.actions, strict=True):
        assert mask[action] == 1


def test_collect_dataset_is_seed_reproducible() -> None:
    a = collect_dataset(n_matches=2, seed=7)
    b = collect_dataset(n_matches=2, seed=7)
    assert len(a) == len(b)
    assert np.array_equal(a.actions, b.actions)
    assert np.array_equal(a.masks, b.masks)


def test_val_split_never_straddles_a_match() -> None:
    """Regression: splitting by ply count could cut a match in half.

    With 8 matches, seed 1, and val_fraction=0.2 the ply-count cut used to
    land 75 plies into the 7th match, putting that match's early plies in
    the training set and its later plies in the "held-out" set -- so the
    reported val accuracy was partly measuring memorization of a match the
    network had already trained on.
    """
    dataset = collect_dataset(n_matches=8, seed=1)
    match_ids = np.unique(dataset.match_ids)
    n_val_matches = max(1, round(len(match_ids) * 0.2))
    val_match_ids = set(match_ids[-n_val_matches:].tolist())
    train_match_ids = set(match_ids.tolist()) - val_match_ids
    assert train_match_ids.isdisjoint(val_match_ids)
    assert val_match_ids, "the held-out set must not be empty"


def test_train_bc_produces_a_valid_accuracy_and_updates_weights() -> None:
    dataset = collect_dataset(n_matches=8, seed=1)
    env = HokmEnv(seat=0, opponents=[GreedyPolicy() for _ in range(NUM_SEATS)])
    model = MaskablePPO(policy=HokmMaskablePolicy, env=env, device="cpu", seed=0, verbose=0)
    before = [p.clone() for p in model.policy.parameters()]

    accuracy = train_bc(model.policy, dataset, epochs=2, batch_size=32, val_fraction=0.2, seed=0)

    assert 0.0 <= accuracy <= 1.0
    after = list(model.policy.parameters())
    assert any(not (a == b).all() for a, b in zip(before, after, strict=True))


def test_bc_cli_writes_a_checkpoint_loadable_and_resumable(tmp_path: Path) -> None:
    """The saved archive must be a normal, train.py-``--resume``-compatible checkpoint."""
    out_path = tmp_path / "bc.zip"
    main(
        [
            "--n-matches",
            "5",
            "--epochs",
            "1",
            "--batch-size",
            "64",
            "--device",
            "cpu",
            "--out-path",
            str(out_path),
        ]
    )
    assert out_path.is_file()

    env = HokmEnv(seat=0, opponents=[GreedyPolicy() for _ in range(NUM_SEATS)])
    # Loading with explicit hyperparameter overrides (train.py's --resume path)
    # must apply those overrides, not the throwaway defaults the BC model was
    # constructed with.
    overrides = hyperparameters(n_steps=64, gamma=0.997)
    resumed = MaskablePPO.load(str(out_path), env=env, device="cpu", **overrides)
    assert resumed.gamma == 0.997
    assert resumed.n_epochs == overrides["n_epochs"]

    obs, _ = env.reset(seed=0)
    mask = env.action_masks()
    action, _ = resumed.predict(obs, action_masks=mask, deterministic=True)
    assert mask[int(action)] == 1
