"""Public void inference, matched feature controls, and export parity."""

import json

import numpy as np
import pytest
import torch
from scripts import qnet_features as arch_sweep
from scripts import train_qnet as qnet_soft

from deephokm.env.spaces import empty_observation, mask_for, observation_for
from deephokm.nn.numpy_qnet import NumpyQNet
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.search import VoidTracker
from deephokm.rules.engine import HokmEngine


def test_reverse_history_marks_followers_not_leader():
    obs = empty_observation()
    obs["history"][:4] = [40, 27, 14, 1]
    obs["history_role"][:4] = [3, 2, 1, 0]
    expected = np.zeros((4, 4))
    expected[1:, 0] = 1
    np.testing.assert_array_equal(arch_sweep.public_voids(obs), expected)


def test_current_trick_leader_wraps_across_seat_zero():
    obs = empty_observation()
    obs["seat"][2] = 1
    obs["trick_play"][:] = [14, 2, -1, 0]
    expected = np.zeros((4, 4))
    expected[2, 0] = 1  # Seat zero is the acting player's partner.
    np.testing.assert_array_equal(arch_sweep.public_voids(obs), expected)


@pytest.mark.parametrize(
    "kind", ["partial_history", "history_hole", "bad_role", "bad_order", "bad_table", "bad_seat"]
)
def test_malformed_history_is_rejected_instead_of_guessing(kind):
    obs = empty_observation()
    obs["history"][:4] = [3, 2, 1, 0]
    obs["history_role"][:4] = [3, 2, 1, 0]
    if kind == "partial_history":
        obs["history"][3] = -1
    elif kind == "history_hole":
        obs["history"][0], obs["history"][4] = -1, 5
    elif kind == "bad_role":
        obs["history_role"][0] = -1
    elif kind == "bad_order":
        obs["history_role"][:4] = [0, 1, 2, 3]
    elif kind == "bad_table":
        obs["seat"][0] = 1
        obs["trick_play"][1] = 5
    else:
        obs["trick_play"][1] = 5
    with pytest.raises(ValueError):
        arch_sweep.public_voids(obs)


def test_public_voids_match_search_tracker_through_full_match_and_resets():
    engine, tracker, greedy = HokmEngine(), VoidTracker(), GreedyPolicy()
    engine.start_match(seed=430071)
    checked, resets, positions = 0, 0, set()
    while engine.state.winner is None:
        seat, hands = engine.current_seat(), engine.state.hands
        obs = observation_for(hands, seat, engine.state.game_points)
        expected = np.zeros((4, 4), dtype=np.float32)
        for absolute, suits in enumerate(tracker.voids):
            expected[(absolute - seat) % 4, list(suits)] = 1
        np.testing.assert_array_equal(arch_sweep.public_voids(obs), expected)
        positions.add(len(hands.current_trick))
        checked += 1
        legal = engine.legal_actions(seat)
        action = greedy.act(obs, mask_for(legal))
        led = hands.current_trick[0][1] // 13 if hands.current_trick else None
        outcome = engine.apply_action(action, seat=seat)
        if outcome.card is not None:
            tracker.observe(seat, outcome.card, led)
        if outcome.hand_complete:
            tracker.reset()
            resets += 1
    assert checked > 100 and resets > 1 and positions == {0, 1, 2, 3}


def test_batch_live_feature_parity_and_zero_control(monkeypatch):
    obs = empty_observation()
    obs["history"][:4] = [40, 27, 14, 1]
    obs["history_role"][:4] = [3, 2, 1, 0]
    mask = mask_for([2, 5])
    shard = {
        "obs": {key: value[None] for key, value in obs.items()},
        "masks": mask[None],
        "qvals": [np.asarray([0.1, 0.2])],
        "legals": [np.asarray([2, 5])],
    }
    monkeypatch.setattr(arch_sweep, "load_split_shards", lambda _: ([shard], 0))
    baseline = arch_sweep.build_row_features(obs, mask)
    for mode in ("voids", "voids_zero"):
        batch = arch_sweep.build_features(files=["fixture"], feature_mode=mode)
        row = arch_sweep.build_row_features(obs, mask, feature_mode=mode)
        np.testing.assert_array_equal(batch[0][0], row[0])
        np.testing.assert_array_equal(batch[1][0], row[1])
        np.testing.assert_array_equal(row[0][:14], baseline[0])
        assert row[0].shape == (18, 4, 13)
        assert bool(row[0][14:].any()) == (mode == "voids")


def test_extra_features_remain_suit_equivariant_and_numpy_compatible():
    torch.manual_seed(0)
    model = arch_sweep.RankCNN(ch=8, layers=2, input_planes=18).eval()
    planes = np.random.default_rng(0).normal(size=(2, 18, 4, 13)).astype(np.float32)
    scalars = np.zeros((2, 10), dtype=np.float32)
    permutation = [3, 1, 0, 2]
    with torch.inference_mode():
        expected = model(torch.from_numpy(planes), torch.from_numpy(scalars)).numpy()
        permuted = model(
            torch.from_numpy(planes[:, :, permutation]), torch.from_numpy(scalars)
        ).numpy()
    np.testing.assert_allclose(
        permuted[:, :52].reshape(2, 4, 13),
        expected[:, :52].reshape(2, 4, 13)[:, permutation],
        atol=1e-6,
    )
    np.testing.assert_allclose(permuted[:, 52:], expected[:, 52:], atol=1e-6)
    params = {key: value.numpy() for key, value in model.state_dict().items()}
    np.testing.assert_allclose(NumpyQNet(params)(planes, scalars), expected, atol=1e-6)


def test_training_mode_is_explicit_and_other_architectures_are_rejected():
    args = qnet_soft.parse_args(["rank_cnn", "80", "1.5", "--feature-mode", "voids"])
    assert args.feature_mode == "voids"
    with pytest.raises(SystemExit):
        qnet_soft.parse_args(["mlp", "80", "1", "--feature-mode", "voids"])
    with pytest.raises(ValueError):
        arch_sweep.build_features(files=[], feature_mode="typo")


def test_expanded_model_control_has_identical_initial_weights():
    states = []
    for _mode in ("voids_zero", "voids"):
        torch.manual_seed(0)
        model = arch_sweep._build("rank_cnn", arch_sweep.RankCNN, 0.1, input_planes=18)
        states.append(model.state_dict())
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])


@pytest.mark.parametrize(
    "mode",
    [
        "voids",
        "voids_zero",
        "trick_context",
        "trick_context_zero",
        "public_strength",
        "public_strength_zero",
    ],
)
def test_expanded_training_cpu_smoke_exports_and_records_feature_mode(tmp_path, monkeypatch, mode):
    n, n_train = 4, 2
    input_planes = (
        48
        if mode.startswith("public_strength")
        else (36 if mode.startswith("trick_context") else 18)
    )
    planes = np.zeros((n, input_planes, 4, 13), dtype=np.float32)
    if mode == "voids":
        planes[:, 14, 1] = 1
    scalars = np.zeros((n, 10), dtype=np.float32)
    masks = np.zeros((n, 56), dtype=np.float32)
    masks[:, :2] = 1
    targets = np.zeros((n, 56), dtype=np.float32)
    targets[:, 1] = 0.2
    shard = tmp_path / "shard.pkl"
    shard.write_bytes(b"test frozen dataset signature")
    checkpoint, export = tmp_path / "best.pt", tmp_path / "qnet.npz"

    def load(_pattern, *, feature_mode):
        assert feature_mode == mode
        return planes, scalars, masks, targets, n_train

    monkeypatch.setattr(qnet_soft, "load_features", load)
    monkeypatch.setattr(qnet_soft, "matching_files", lambda _: [str(shard)])
    monkeypatch.setattr(
        qnet_soft.sys,
        "argv",
        [
            "qnet_soft.py",
            "rank_cnn",
            "1",
            ".04",
            "--device",
            "cpu",
            "--cpu-threads",
            "1",
            "--feature-mode",
            mode,
            "--seed",
            "2",
            "--checkpoint",
            str(checkpoint),
            "--export-numpy",
            str(export),
        ],
    )
    qnet_soft.main()
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["config"]["feature_mode"] == mode
    assert metadata["config"]["seed"] == 2
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["blocks.0.weight"].shape[1] == 2 * input_planes
    with np.load(export) as archive:
        for key, value in state.items():
            np.testing.assert_array_equal(archive[key], value.numpy())
