"""Feature contracts, public-only strength and production policy compatibility."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts import qnet_features as arch_sweep

from deephokm.env.spaces import empty_observation, mask_for, observation_for
from deephokm.nn import public_features
from deephokm.nn.features import build_features
from deephokm.nn.numpy_qnet import save_weights
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.numpy_hybrid import NumpyHybridPolicy
from deephokm.policies.pure_qnet_policy import PureQNetPolicy
from deephokm.rules.engine import HokmEngine


def weights(planes=36):
    torch.manual_seed(0)
    model = arch_sweep.RankCNN(ch=8, layers=2, input_planes=planes)
    return {key: value.detach().numpy() for key, value in model.state_dict().items()}


def test_expanded_weights_require_unambiguous_schema():
    with pytest.raises(ValueError, match="requires feature metadata"):
        PureQNetPolicy(weights())
    with pytest.raises(ValueError, match="channels"):
        PureQNetPolicy(weights(), feature_mode="voids")
    for cls in (PureQNetPolicy, NumpyHybridPolicy):
        assert cls(weights(), feature_mode="trick_context").feature_mode == "trick_context"


def test_new_contract_checks_weights_feature_code_and_channels(tmp_path):
    path = tmp_path / "qnet.npz"
    save_weights(weights(), path)
    data = {
        "schema_version": 1,
        "feature_mode": "trick_context",
        "input_planes": 36,
        "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "public_features_sha256": hashlib.sha256(
            Path(public_features.__file__).read_bytes()
        ).hexdigest(),
    }
    contract = path.with_suffix(".features.json")
    contract.write_text(json.dumps(data))
    assert PureQNetPolicy(path).feature_mode == "trick_context"
    with pytest.raises(ValueError, match="disagrees"):
        PureQNetPolicy(path, feature_mode="trick_context_zero")
    for key, value in (
        ("weights_sha256", "wrong"),
        ("public_features_sha256", "wrong"),
        ("input_planes", 48),
        ("schema_version", 99),
    ):
        changed = {**data, key: value}
        contract.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            PureQNetPolicy(path)


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
def test_production_and_training_features_identical_for_full_match(mode):
    engine, greedy = HokmEngine(), GreedyPolicy()
    engine.start_match(seed=1701)
    while engine.state.winner is None:
        seat = engine.current_seat()
        obs = observation_for(engine.state.hands, seat, engine.state.game_points)
        mask = mask_for(engine.legal_actions(seat))
        expected = arch_sweep.build_row_features(obs, mask, feature_mode=mode)
        actual = build_features(obs, mask, feature_mode=mode)
        for old, new in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(old, new)
        engine.apply_action(greedy.act(obs, mask), seat=seat)


def test_no_higher_unseen_is_not_highest_remaining_when_we_hold_a_higher_card():
    obs = empty_observation()
    obs["seat"][0] = obs["phase"][1] = obs["trump"][3] = 1
    obs["hand"][[0, 1, 12, 39]] = 1
    obs["seen"][:] = 1
    extra = public_features.strength_planes(obs)
    assert extra[1, 0, 0] == 1 and extra[2, 0, 0] == 0
    assert extra[2, 0, 12] == 1
    assert extra[7, 0, 0] == pytest.approx(2 / 13)
    assert extra[3, 0, 0] == pytest.approx(3 / 13)
    assert not extra[4:6].any()
    obs["seen"][11] = 0
    changed = public_features.strength_planes(obs)
    assert changed[0, 0, 0] == pytest.approx(1 / 13)
    assert changed[1, 0, 0] == 0 and changed[1, 0, 12] == 1


def test_strength_control_retains_all_previous_inputs_and_is_suit_equivariant():
    engine = HokmEngine()
    engine.start_match(seed=1802)
    engine.apply_action(engine.legal_actions()[0])
    actor = engine.current_seat()
    obs = observation_for(engine.state.hands, actor, engine.state.game_points)
    mask = mask_for(engine.legal_actions(actor))
    actual = build_features(obs, mask, feature_mode="public_strength")[0]
    control = build_features(obs, mask, feature_mode="public_strength_zero")[0]
    assert actual.shape == control.shape == (48, 4, 13)
    np.testing.assert_array_equal(actual[:36], control[:36])
    assert not control[36:].any() and actual[36:].any()
    permutation = np.array([2, 0, 3, 1])
    changed = copy.deepcopy(obs)
    for key in ("hand", "seen", "trick", "trump"):
        changed[key].reshape(4, -1)[permutation] = obs[key].reshape(4, -1)
    np.testing.assert_array_equal(
        public_features.strength_planes(changed)[:, permutation], actual[36:]
    )


def test_production_pure_and_hybrid_proposals_use_expanded_features_legally():
    engine, greedy = HokmEngine(), GreedyPolicy()
    pure = PureQNetPolicy(weights(), feature_mode="trick_context")
    hybrid = NumpyHybridPolicy(weights(), feature_mode="trick_context", verify_samples=3)
    engine.start_match(seed=1803)
    for _ in range(24):
        seat = engine.current_seat()
        obs = observation_for(engine.state.hands, seat, engine.state.game_points)
        legal = engine.legal_actions(seat)
        mask = mask_for(legal)
        assert pure.decide(engine) == pure.act(obs, mask)
        if obs["phase"][1]:
            proposals = hybrid._proposals(obs, mask, legal, greedy.act(obs, mask))
            assert set(proposals).issubset(legal)
        else:
            assert hybrid.decide(engine) in legal
        outcome = engine.apply_action(pure.decide(engine), seat=seat)
        if outcome.hand_complete:
            hybrid.reset_hand()


def test_hidden_hands_do_not_affect_strength_features():
    engine = HokmEngine()
    engine.start_match(seed=1804)
    engine.apply_action(engine.legal_actions()[0])
    actor, state = engine.current_seat(), engine.state.hands
    before = public_features.strength_planes(
        observation_for(state, actor, engine.state.game_points)
    )
    changed = copy.deepcopy(state)
    others = [seat for seat in range(4) if seat != actor]
    hidden = [card for seat in others for card in changed.hands[seat]]
    hidden.reverse()
    offset = 0
    for seat in others:
        size = len(changed.hands[seat])
        changed.hands[seat] = hidden[offset : offset + size]
        offset += size
    after = public_features.strength_planes(
        observation_for(changed, actor, engine.state.game_points)
    )
    np.testing.assert_array_equal(before, after)
