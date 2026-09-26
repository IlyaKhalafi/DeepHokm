"""The numpy backend must agree with torch, and be fast enough to ship.

Playing does not require torch, so the numpy forward pass is the code that
actually runs in production. Two things can go wrong silently: the port can
drift from the trained weights (changing how the model plays without any
error), and it can be too slow for interactive play. Both are gated here.
"""

from __future__ import annotations

import random
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest
import torch as th

from deephokm.env import HokmEnv
from deephokm.env.spaces import mask_for
from deephokm.nn.features import NUM_PLANES, NUM_SCALARS, build_features
from deephokm.nn.numpy_qnet import NumpyQNet, load_weights, save_weights
from deephokm.nn.rank_cnn import RankCNN, export_weights
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.state import NUM_SEATS

# float32 accumulation order differs between the two implementations; this is
# orders of magnitude below any gap that could reorder two action values, and
# the argmax check is the real gate.
MAX_ABS_DIFF = 1e-4
LATENCY_BUDGET_S = 1.0
N_LATENCY_REPS = 20


def collect_states(n: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Gather real ``(planes, scalars)`` features from a random rollout."""
    env = HokmEnv(seat=0, opponents=[RandomPolicy(seed + i) for i in range(NUM_SEATS)])
    rng = random.Random(seed)
    obs, info = env.reset(seed=seed)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    done = False
    while not done and len(out) < n:
        legal = np.flatnonzero(info["action_mask"])
        out.append(build_features(obs, mask_for([int(a) for a in legal])))
        obs, _, term, trunc, info = env.step(int(rng.choice(legal)))
        done = term or trunc
    env.close()
    return out


def stack(states: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([p for p, _ in states]),
        np.stack([s for _, s in states]),
    )


def test_feature_shapes_and_range() -> None:
    states = collect_states(12)
    assert states, "no states collected"
    for planes, scalars in states:
        assert planes.shape == (NUM_PLANES, 4, 13)
        assert scalars.shape == (NUM_SCALARS,)
        assert planes.dtype == np.float32
        assert np.isfinite(planes).all() and np.isfinite(scalars).all()
        assert planes.min() >= 0.0 and planes.max() <= 1.0


@pytest.mark.parametrize("channels", [32, 128])
def test_numpy_matches_torch(channels: int) -> None:
    th.manual_seed(0)
    model = RankCNN(channels=channels).eval()
    numpy_model = NumpyQNet(export_weights(model))
    planes, scalars = stack(collect_states(16, seed=3))

    with th.no_grad():
        reference = model(th.as_tensor(planes), th.as_tensor(scalars)).numpy()
    got = numpy_model(planes, scalars)

    assert got.shape == reference.shape
    assert np.abs(reference - got).max() < MAX_ABS_DIFF
    # The decision the model makes must be identical, not merely close.
    assert (reference.argmax(axis=1) == got.argmax(axis=1)).all()


def test_numpy_is_suit_equivariant() -> None:
    """Relabelling suits must permute the card values the same way.

    This is the property the architecture exists for, and it holds only if
    every layer shares weights across suits and mixes them symmetrically.
    """
    th.manual_seed(0)
    numpy_model = NumpyQNet(export_weights(RankCNN(channels=32).eval()))
    planes, scalars = stack(collect_states(8, seed=5))
    perm = [2, 0, 3, 1]

    base = numpy_model(planes, scalars)
    permuted = numpy_model(planes[:, :, perm, :], scalars)

    base_cards = base[:, :52].reshape(-1, 4, 13)[:, perm, :].reshape(-1, 52)
    assert np.abs(base_cards - permuted[:, :52]).max() < MAX_ABS_DIFF
    # Trump-declaration values are pooled symmetrically, so they are invariant.
    assert np.abs(base[:, 52:] - permuted[:, 52:]).max() < MAX_ABS_DIFF


def test_single_decision_within_latency_budget() -> None:
    """One decision is the unit the player actually pays for."""
    th.manual_seed(0)
    numpy_model = NumpyQNet(export_weights(RankCNN().eval()))
    planes, scalars = stack(collect_states(1, seed=7))

    numpy_model(planes, scalars)  # warm up
    start = time.perf_counter()
    for _ in range(N_LATENCY_REPS):
        numpy_model(planes, scalars)
    per_call = (time.perf_counter() - start) / N_LATENCY_REPS
    assert per_call < LATENCY_BUDGET_S, f"{per_call:.3f}s exceeds the {LATENCY_BUDGET_S}s budget"


def test_weight_roundtrip_needs_no_torch(tmp_path) -> None:  # noqa: ANN001
    """Saved weights reload into an identical network without torch."""
    th.manual_seed(0)
    params = export_weights(RankCNN(channels=32).eval())
    path = tmp_path / "qnet.npz"
    save_weights(params, path)
    reloaded = NumpyQNet(load_weights(path))

    planes, scalars = stack(collect_states(4, seed=11))
    assert np.array_equal(NumpyQNet(params)(planes, scalars), reloaded(planes, scalars))


def test_rejects_weights_that_are_not_a_rank_cnn() -> None:
    with pytest.raises(ValueError, match="RankCNN"):
        NumpyQNet({"something.weight": np.zeros((2, 2), dtype=np.float32)})


def test_play_path_does_not_import_torch() -> None:
    """Playing must not pull in PyTorch.

    The README states that a deployment installs numpy without a deep-learning
    framework. That held only by accident until ``deephokm.nn``'s eager imports
    were made lazy: importing the feature builder used to drag in the
    transformer extractor and therefore torch. This runs a subprocess with the
    torch import blocked and plays part of a match, so the claim fails loudly
    if any play-time module reaches for it again.
    """
    script = textwrap.dedent(
        """
        import sys

        class Blocker:
            def find_module(self, name, path=None):
                if name == "torch" or name.startswith("torch."):
                    return self

            def load_module(self, name):
                raise ImportError("torch imported at play time: " + name)

        sys.meta_path.insert(0, Blocker())

        from deephokm.env.spaces import mask_for, observation_for
        from deephokm.policies.greedy_policy import GreedyPolicy
        from deephokm.policies.numpy_hybrid import NumpyHybridPolicy
        from deephokm.rules.engine import HokmEngine

        engine = HokmEngine()
        engine.start_match(seed=5)
        greedy = GreedyPolicy()
        for _ in range(12):
            if engine.state.winner is not None:
                break
            seat = engine.current_seat()
            hands = engine.state.hands
            legal = engine.legal_actions(seat)
            obs = observation_for(hands, seat, engine.state.game_points)
            engine.apply_action(greedy.act(obs, mask_for(legal)), seat=seat)

        assert "torch" not in sys.modules, "torch was imported by the play path"
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, f"play path imported torch:\n{result.stderr[-1500:]}"
    assert "OK" in result.stdout
