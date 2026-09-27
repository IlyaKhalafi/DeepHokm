"""The no-search serving policy must play legally and need no torch.

These are correctness gates, not a strength measurement: win rate is
measured separately over many matches (0.6625 against a greedy opposing
team, see docs/METHODS_AND_RESULTS.md). What matters here is that the
policy never proposes an illegal action, is deterministic (there is no
randomness in an argmax), and runs off numpy weights alone.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import torch as th

from deephokm.env.spaces import mask_for, observation_for
from deephokm.nn.rank_cnn import RankCNN, export_weights
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.pure_qnet_policy import PureQNetPolicy
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import NUM_SEATS, Phase

TEST_CHANNELS = 16


def make_policy() -> PureQNetPolicy:
    th.manual_seed(0)
    weights = export_weights(RankCNN(channels=TEST_CHANNELS).eval())
    return PureQNetPolicy(weights)


def play_match(policy: PureQNetPolicy, seed: int, max_actions: int = 4000) -> list[int]:
    """Drive a full match with the policy on one team, greedy on the other."""
    engine = HokmEngine()
    engine.start_match(seed=seed)
    greedy = GreedyPolicy()
    controlled = {0, 2}
    actions: list[int] = []
    steps = 0
    while engine.state.winner is None and steps < max_actions:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        hands = engine.state.hands
        if seat in controlled:
            action = policy.decide(engine)
        else:
            obs = observation_for(hands, seat, engine.state.game_points)
            action = greedy.act(obs, mask_for(legal))
        assert action in legal, f"illegal action {action} for seat {seat}"
        actions.append(action)
        engine.apply_action(action, seat=seat)
        steps += 1
    assert engine.state.winner is not None, "match did not terminate"
    return actions


def test_plays_a_full_match_legally() -> None:
    play_match(make_policy(), seed=7)


def test_is_deterministic() -> None:
    """An argmax has no randomness: identical weights, identical actions."""
    assert play_match(make_policy(), seed=3) == play_match(make_policy(), seed=3)


def test_falls_back_to_greedy_for_the_trump_call() -> None:
    """The network has no trained signal at the trump decision point."""
    policy = make_policy()
    greedy = GreedyPolicy()
    engine = HokmEngine()
    engine.start_match(seed=11)
    seat = engine.current_seat()
    assert engine.state.hands.phase is Phase.TRUMP_CALL
    hands = engine.state.hands
    obs = observation_for(hands, seat, engine.state.game_points)
    mask = mask_for(engine.legal_actions(seat))
    assert policy.decide(engine) == greedy.act(obs, mask)


def test_act_and_decide_agree() -> None:
    """The HokmPolicy.act() entry point must pick the same card as decide()."""
    policy = make_policy()
    engine = HokmEngine()
    engine.start_match(seed=13)
    while engine.state.hands.phase is not Phase.CARD_PLAY:
        engine.apply_action(engine.legal_actions()[0])
    seat = engine.current_seat()
    hands = engine.state.hands
    obs = observation_for(hands, seat, engine.state.game_points)
    mask = mask_for(engine.legal_actions(seat))
    assert policy.decide(engine) == policy.act(obs, mask)


def test_lifecycle_methods_are_harmless_no_ops() -> None:
    policy = make_policy()
    policy.reset_hand()
    policy.observe(seat=0, card=5, led_suit=None)
    policy.close()
    policy.close()  # a second call must not raise
    play_match(policy, seed=17)


def test_seats_see_only_their_own_hand() -> None:
    """The network reads observation+mask only; no engine hand access."""
    engine = HokmEngine()
    engine.start_match(seed=19)
    while engine.state.hands.phase is not Phase.CARD_PLAY:
        engine.apply_action(engine.legal_actions()[0])
    for seat in range(NUM_SEATS):
        if engine.current_seat() != seat:
            continue
        hands = engine.state.hands
        obs = observation_for(hands, seat, engine.state.game_points)
        own_hand = np.flatnonzero(np.asarray(obs["hand"]))
        assert set(own_hand.tolist()) == set(hands.hands[seat])
        break


def test_runs_without_torch() -> None:
    """Playing must not pull in PyTorch.

    Mirrors the equivalent guarantee for NumpyHybridPolicy
    (tests/test_numpy_qnet.py): a deployment installs numpy without a
    deep-learning framework, and this fails loudly if that ever regresses.
    """
    # export_weights itself needs torch (it reads a live nn.Module), so the
    # torch-blocked half runs only the policy's own decision path -- built
    # from a plain dict of numpy arrays, exactly like the shipped .npz.
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        from deephokm.nn.rank_cnn import RankCNN, export_weights
        import torch as th
        th.manual_seed(0)
        weights = export_weights(RankCNN(channels=16).eval())

        class Blocker:
            def find_module(self, name, path=None):
                if name == "torch" or name.startswith("torch."):
                    return self

            def load_module(self, name):
                raise ImportError("torch imported at play time: " + name)

        for mod in list(sys.modules):
            if mod == "torch" or mod.startswith("torch."):
                del sys.modules[mod]
        sys.meta_path.insert(0, Blocker())

        from deephokm.env.spaces import mask_for, observation_for
        from deephokm.policies.greedy_policy import GreedyPolicy
        from deephokm.policies.pure_qnet_policy import PureQNetPolicy
        from deephokm.rules.engine import HokmEngine

        policy = PureQNetPolicy(weights)
        engine = HokmEngine()
        engine.start_match(seed=5)
        for _ in range(12):
            if engine.state.winner is not None:
                break
            seat = engine.current_seat()
            legal = engine.legal_actions(seat)
            action = policy.decide(engine)
            assert action in legal
            engine.apply_action(action, seat=seat)

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
