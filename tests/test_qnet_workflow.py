"""The public training CLI works without local scratch files or job launchers."""

import hashlib
import json
import pickle
import sys

import numpy as np
from scripts import qnet_features, train_qnet

from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies.pure_qnet_policy import PureQNetPolicy
from deephokm.rules.engine import HokmEngine


def test_train_and_serve_from_explicit_data_directory(tmp_path, monkeypatch):
    data = tmp_path / "teacher-data"
    data.mkdir()
    monkeypatch.setattr(qnet_features, "ROOT", str(data))
    engine = HokmEngine()
    engine.start_match(seed=0)
    engine.apply_action(engine.legal_actions()[0])
    actor = engine.current_seat()
    obs = observation_for(engine.state.hands, actor, engine.state.game_points)
    legal = engine.legal_actions(actor)
    mask = mask_for(legal)
    seeds = {}
    for seed in range(100):
        validation = hashlib.sha256(f"seed:{seed}".encode()).digest()[0] % 4 == 0
        seeds.setdefault(validation, seed)
    for index, seed in enumerate(seeds.values()):
        shard = {
            "seed": seed,
            "obs": {key: value[None] for key, value in obs.items()},
            "masks": mask[None],
            "legals": [legal],
            "qvals": [np.linspace(-1, 1, len(legal), dtype=np.float32)],
        }
        (data / f"qdata_{index}.pkl").write_bytes(pickle.dumps(shard))
    checkpoint, export = tmp_path / "best.pt", tmp_path / "qnet_numpy.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_qnet",
            "rank_cnn",
            "1",
            "0.125",
            "0.013",
            "0",
            "qdata_*.pkl",
            "--data-dir",
            str(data),
            "--device",
            "cpu",
            "--feature-mode",
            "public_strength",
            "--checkpoint",
            str(checkpoint),
            "--export-numpy",
            str(export),
        ],
    )
    train_qnet.main()
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["decisive_training_examples"] == 1
    assert metadata["decisive_validation_examples"] == 1
    assert all(str(data) in row["path"] for row in metadata["dataset"])
    policy = PureQNetPolicy(export)
    assert policy.feature_mode == "public_strength"
    assert policy.act(obs, mask) in legal
