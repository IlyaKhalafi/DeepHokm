"""Checks for long-running Q-label collection and its incremental cache."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from scripts import generate_q_data
from scripts import qnet_features as arch_sweep
from scripts.generate_q_data import score_decision

from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.legal_depth_search import LegalDepthSearchPolicy
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase


@pytest.mark.parametrize("k", [3, 4])
def test_collected_actions_and_rng_match_teacher_across_match(k: int) -> None:
    engine = HokmEngine()
    engine.start_match(seed=43_000_123)
    greedy = GreedyPolicy()
    teacher = LegalDepthSearchPolicy(n_samples=k, search_depth=2, seed=7)
    collector = LegalDepthSearchPolicy(n_samples=k, search_depth=2, seed=7)
    decisions = 0
    while engine.state.winner is None:
        seat = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(seat)
        if hands.phase is Phase.CARD_PLAY and seat % 2 == 1 and len(legal) > 1:
            action, record = score_decision(engine, collector, greedy, legal)
            assert action == teacher.decide(engine)
            assert collector._rng.getstate() == teacher._rng.getstate()
            assert np.array_equal(record[2], legal)
            assert record[3].shape == (len(legal),)
            assert np.all(np.isfinite(record[3]))
            assert np.all(np.abs(record[3]) <= 1)
            decisions += 1
        else:
            obs = observation_for(hands, seat, engine.state.game_points)
            action = greedy.act(obs, mask_for(legal))
        led = hands.current_trick[0][1] // 13 if hands.current_trick else None
        outcome = engine.apply_action(action, seat=seat)
        for policy in (teacher, collector):
            if outcome.card is not None:
                policy.observe(seat, outcome.card, led)
            if outcome.hand_complete:
                policy.reset_hand()
    assert decisions > 0


def test_interrupted_match_resumes_with_identical_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed, k = 43_000_456, 3
    expected = generate_q_data.play_match(seed, k)
    original = score_decision
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == generate_q_data.CHECKPOINT_DECISIONS + 2:
            raise InterruptedError("simulated worker interruption")
        return original(*args, **kwargs)

    checkpoint = tmp_path / ".resume"
    monkeypatch.setattr(generate_q_data, "score_decision", interrupt)
    with pytest.raises(InterruptedError):
        generate_q_data.play_match(seed, k, checkpoint_path=checkpoint)
    assert checkpoint.exists()
    monkeypatch.setattr(generate_q_data, "score_decision", original)
    resumed = generate_q_data.play_match(seed, k, checkpoint_path=checkpoint)
    assert np.array_equal(expected["teacher_actions"], resumed["teacher_actions"])
    for key in expected["obs"]:
        assert np.array_equal(expected["obs"][key], resumed["obs"][key])
    for expected_q, resumed_q in zip(expected["qvals"], resumed["qvals"], strict=True):
        assert np.array_equal(expected_q, resumed_q)


def test_feature_cache_refreshes_as_shards_arrive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arch_sweep, "ROOT", str(tmp_path))
    calls: list[int] = []

    def fake_build_features(pattern: str, *, files: list[str] | None = None):
        count = len(arch_sweep.matching_files(pattern))
        calls.append(count)
        value = np.asarray([count], dtype=np.float32)
        return value, value, value, value, count

    monkeypatch.setattr(arch_sweep, "build_features", fake_build_features)
    first = tmp_path / "q6144_test_0.pkl"
    first.write_bytes(b"a")
    assert arch_sweep.load_features("q6144_test_*.pkl")[-1] == 1
    assert arch_sweep.load_features("q6144_test_*.pkl")[-1] == 1
    assert calls == [1]

    second = tmp_path / "q6144_test_1.pkl"
    second.write_bytes(b"b")
    assert arch_sweep.load_features("q6144_test_*.pkl")[-1] == 2
    assert calls == [1, 2]

    second.write_bytes(b"longer")
    assert arch_sweep.load_features("q6144_test_*.pkl")[-1] == 2
    assert calls == [1, 2, 2]


def test_overlapping_dataset_patterns_do_not_duplicate_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arch_sweep, "ROOT", str(tmp_path))
    shard = tmp_path / "q6144_test_0.pkl"
    shard.write_bytes(b"a")
    assert arch_sweep.matching_files("q6144_test_*.pkl,q6144_test_0.pkl") == [str(shard)]


def test_validation_matches_stay_held_out_as_dataset_grows(tmp_path: Path) -> None:
    validation_seed = next(
        seed for seed in range(100) if hashlib.sha256(f"seed:{seed}".encode()).digest()[0] % 4 == 0
    )
    training_seeds = [
        seed for seed in range(100) if hashlib.sha256(f"seed:{seed}".encode()).digest()[0] % 4 != 0
    ][:3]
    paths = []
    for index, seed in enumerate((validation_seed, *training_seeds)):
        path = tmp_path / f"shard_{index}.pkl"
        generate_q_data.write_atomic(path, {"seed": seed, "qvals": [np.asarray([0.0])]})
        paths.append(str(path))
    for files in (paths[:2], paths):
        data, n_train = arch_sweep.load_split_shards(files)
        assert {shard["seed"] for shard in data[n_train:]} == {validation_seed}
        assert all(shard["seed"] != validation_seed for shard in data[:n_train])


def test_two_shards_of_the_same_seed_are_rejected(tmp_path: Path) -> None:
    paths = [tmp_path / "first.pkl", tmp_path / "second.pkl"]
    for path in paths:
        generate_q_data.write_atomic(path, {"seed": 1, "qvals": [np.asarray([0.0])]})
    with pytest.raises(ValueError, match="duplicate match seed"):
        arch_sweep.load_split_shards([str(path) for path in paths])
