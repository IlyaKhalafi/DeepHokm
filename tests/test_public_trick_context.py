"""Legal public context, counterexamples, equivariance and inference parity."""

import copy

import numpy as np
import pytest
import torch
from scripts import qnet_features as arch_sweep

from deephokm.env.spaces import empty_observation, mask_for, observation_for
from deephokm.nn.numpy_qnet import NumpyQNet
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.rules.engine import HokmEngine
from deephokm.rules.tricks import resolve_trick


def position(*, seat=0, table=(-1, -1, -1, -1), hand=(2, 12, 39), trump=3):
    obs = empty_observation()
    obs["seat"][seat] = obs["phase"][1] = obs["trump"][trump] = 1
    obs["trick_play"][:] = table
    obs["hand"][list(hand)] = 1
    obs["seen"][:] = obs["hand"]
    cards = np.asarray(table)[np.asarray(table) >= 0]
    obs["trick"][cards] = obs["seen"][cards] = 1
    return obs, mask_for(hand)


def context(obs, mask):
    return arch_sweep.build_row_features(obs, mask, feature_mode="trick_context")[0][18:]


def test_leading_and_action_flags_are_not_fictitious_trick_winners():
    obs, mask = position()
    extra = context(obs, mask)
    assert not extra[:7].any() and not extra[14:16].any()
    assert extra[7].all() and extra[11:14].all()
    assert extra[16, 3, 0] == 1 and extra[17, 3, 0] == 1
    assert not extra[17, 0].any()  # Two cards in suit zero: neither creates a void.
    assert extra[14:].sum() == 2


@pytest.mark.parametrize(
    "table,hand,trump,winner,beats,partner",
    [
        ((-1, 0, 11, 4), (2, 12), 3, 11, [12], True),
        ((-1, 12, 40, 14), (39, 41, 51), 3, 40, [41, 51], True),
        ((-1, 12, 40, 14), (25, 38), 3, 40, [], True),
        ((-1, 39, 40, 51), (41, 50), 3, 51, [], False),
        ((-1, -1, -1, 12), (14, 38, 39), 3, 12, [39], False),
        ((-1, -1, -1, 25), (14, 24), 1, 25, [], False),
    ],
)
def test_trumps_off_suit_discards_and_partner_overtakes(
    *, table, hand, trump, winner, beats, partner
):
    obs, mask = position(table=table, hand=hand, trump=trump)
    extra = context(obs, mask)
    assert np.flatnonzero(extra[1].ravel()).tolist() == [winner]
    assert np.flatnonzero(extra[14].ravel()).tolist() == beats
    assert np.flatnonzero(extra[15].ravel()).tolist() == (beats if partner else [])
    assert bool(extra[6].all()) == partner
    assert not extra[14:][..., mask[:52].reshape(4, 13) == 0].any()


@pytest.mark.parametrize("seat", range(4))
@pytest.mark.parametrize("played", range(4))
def test_all_seat_wraps_positions_and_remaining_actors(seat, played):
    seats = [(seat - played + i) % 4 for i in range(played)]
    table = [-1] * 4
    for i, played_seat in enumerate(seats):
        table[played_seat] = i
    obs, mask = position(seat=seat, table=table, hand=(10, 12))
    extra = context(obs, mask)
    assert extra[7 + played].all()
    assert extra[7:11].sum() == 52
    for relative in range(1, 4):
        assert bool(extra[10 + relative].all()) == (relative < 4 - played)
    if played:
        assert extra[5].all()  # Highest led-suit card belongs to previous player.


@pytest.mark.parametrize("mode", ["trick_context", "trick_context_zero"])
def test_batch_live_parity_and_matched_control(monkeypatch, mode):
    obs, mask = position(table=(-1, 0, 11, 4), hand=(2, 12))
    shard = {
        "obs": {key: value[None] for key, value in obs.items()},
        "masks": mask[None],
        "qvals": [np.asarray([0.1, 0.2])],
        "legals": [np.asarray([2, 12])],
    }
    monkeypatch.setattr(arch_sweep, "load_split_shards", lambda _: ([shard], 0))
    batch = arch_sweep.build_features(files=["fixture"], feature_mode=mode)
    row = arch_sweep.build_row_features(obs, mask, feature_mode=mode)
    np.testing.assert_array_equal(batch[0][0], row[0])
    np.testing.assert_array_equal(batch[1][0], row[1])
    np.testing.assert_array_equal(
        row[0][:18], arch_sweep.build_row_features(obs, mask, feature_mode="voids")[0]
    )
    assert row[0].shape == (36, 4, 13)
    assert bool(row[0][18:].any()) == (mode == "trick_context")


def test_features_are_suit_equivariant_and_seat_rotation_relative():
    obs, mask = position(seat=2, table=(14, 2, -1, 0), hand=(5, 12))
    expected = context(obs, mask)
    permutation = np.array([2, 0, 3, 1])  # Old suit -> new suit.
    changed = copy.deepcopy(obs)
    for key in ("hand", "seen", "trick", "trump"):
        changed[key] = np.empty_like(obs[key])
        changed[key].reshape(4, -1)[permutation] = obs[key].reshape(4, -1)
    for key in ("trick_play", "history"):
        valid = changed[key] >= 0
        cards = changed[key][valid]
        changed[key][valid] = permutation[cards // 13] * 13 + cards % 13
    changed_mask = np.zeros_like(mask)
    changed_mask[:52].reshape(4, 13)[permutation] = mask[:52].reshape(4, 13)
    changed_mask[52:][permutation] = mask[52:]
    actual = context(changed, changed_mask)
    np.testing.assert_array_equal(actual[:, permutation], expected)
    changed = copy.deepcopy(obs)
    changed["trick_play"] = np.roll(changed["trick_play"], 1)
    changed["seat"] = np.roll(changed["seat"], 1)
    np.testing.assert_array_equal(context(changed, mask), expected)


def test_hidden_hands_never_change_features_through_full_match():
    engine, greedy, rng = HokmEngine(), GreedyPolicy(), np.random.default_rng(19)
    engine.start_match(seed=430070)
    positions, checked = set(), 0
    while engine.state.winner is None:
        actor, state = engine.current_seat(), engine.state.hands
        obs = observation_for(state, actor, engine.state.game_points)
        mask = mask_for(engine.legal_actions(actor))
        before = arch_sweep.build_row_features(obs, mask, feature_mode="trick_context")
        changed = copy.deepcopy(state)
        other = [seat for seat in range(4) if seat != actor]
        hidden = [card for seat in other for card in changed.hands[seat]]
        rng.shuffle(hidden)
        offset = 0
        for seat in other:
            size = len(changed.hands[seat])
            changed.hands[seat] = hidden[offset : offset + size]
            offset += size
        after = arch_sweep.build_row_features(
            observation_for(changed, actor, engine.state.game_points),
            mask,
            feature_mode="trick_context",
        )
        for old, new in zip(before, after, strict=True):
            np.testing.assert_array_equal(old, new)
        positions.add(len(state.current_trick))
        checked += 1
        if obs["phase"][0]:
            assert not before[0][18:].any()
        if len(state.current_trick) == 3:
            for action in np.flatnonzero(mask):
                winner = resolve_trick([*state.current_trick, (actor, int(action))], state.trump)
                assert bool(before[0][32].ravel()[action]) == (winner == actor)
        engine.apply_action(greedy.act(obs, mask), seat=actor)
    assert positions == {0, 1, 2, 3} and checked > 100


@pytest.mark.parametrize(
    "kind", ["full", "gap", "duplicate", "invalid", "missing_hand", "trump", "phase"]
)
def test_malformed_context_fails_closed(kind):
    obs, mask = position()
    if kind == "full":
        obs["trick_play"][:] = [0, 1, 3, 4]
    elif kind == "gap":
        obs["trick_play"][1] = 0
    elif kind == "duplicate":
        obs["trick_play"][2:] = 0
    elif kind == "invalid":
        obs["trick_play"][3] = -2
    elif kind == "missing_hand":
        mask[4] = 1
    elif kind == "trump":
        obs["trump"][:] = 0
    else:
        obs["phase"][:] = 0
    with pytest.raises(ValueError):
        context(obs, mask)


def test_matched_seed_initialization_and_numpy_parity():
    states = []
    for _mode in ("trick_context_zero", "trick_context"):
        torch.manual_seed(2)
        model = arch_sweep.RankCNN(ch=8, layers=2, input_planes=36).eval()
        states.append(model.state_dict())
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
    planes = np.random.default_rng(2).normal(size=(2, 36, 4, 13)).astype(np.float32)
    scalars = np.zeros((2, 10), dtype=np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(planes), torch.from_numpy(scalars)).numpy()
    params = {key: value.numpy() for key, value in model.state_dict().items()}
    np.testing.assert_allclose(NumpyQNet(params)(planes, scalars), expected, atol=1e-6)
    torch.manual_seed(1)
    other = arch_sweep.RankCNN(ch=8, layers=2, input_planes=36)
    assert not torch.equal(other.state_dict()["blocks.0.weight"], states[0]["blocks.0.weight"])
