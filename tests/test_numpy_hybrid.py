"""The shipped policy must play legally and need no torch at decision time.

These are correctness gates, not strength measurements: win rate is measured
separately over hundreds of matches. What matters here is that the policy
never proposes an illegal action, never sees hidden cards, and runs off numpy
weights alone.
"""

from __future__ import annotations

import numpy as np
import torch as th

from deephokm.env.spaces import mask_for, observation_for
from deephokm.nn.features import PLANE_HAND, PLANE_SEEN, build_features
from deephokm.nn.numpy_qnet import save_weights
from deephokm.nn.rank_cnn import RankCNN, export_weights
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.numpy_hybrid import NumpyHybridPolicy
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import NUM_SEATS, Phase

# Tiny network and sample count: these tests check legality and wiring, and a
# realistic verify_samples would make them minutes long for no extra coverage.
TEST_CHANNELS = 16
TEST_SAMPLES = 4


def make_policy(seed: int = 0) -> NumpyHybridPolicy:
    th.manual_seed(0)
    weights = export_weights(RankCNN(channels=TEST_CHANNELS).eval())
    return NumpyHybridPolicy(
        weights, verify_samples=TEST_SAMPLES, top_m=3, seed=seed
    )


def play_match(policy: NumpyHybridPolicy, seed: int, max_actions: int = 4000) -> list[int]:
    """Drive a full match with the policy on one team, greedy on the other."""
    engine = HokmEngine()
    engine.start_match(seed=seed)
    greedy = GreedyPolicy()
    controlled = {0, 2}
    actions: list[int] = []
    policy.reset_hand()
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
        led = hands.current_trick[0][1] // 13 if hands.current_trick else None
        outcome = engine.apply_action(action, seat=seat)
        if outcome.card is not None:
            policy.observe(seat, outcome.card, led)
        if outcome.hand_complete:
            policy.reset_hand()
        steps += 1
    assert engine.state.winner is not None, "match did not terminate"
    return actions


def test_plays_a_full_match_legally() -> None:
    actions = play_match(make_policy(), seed=7)
    assert len(actions) > 13, "match ended implausibly early"


def test_is_deterministic_for_a_fixed_seed() -> None:
    """Same weights and seed must reproduce the same match exactly."""
    first = play_match(make_policy(seed=5), seed=11)
    second = play_match(make_policy(seed=5), seed=11)
    assert first == second


def test_falls_back_to_greedy_for_the_trump_call() -> None:
    """The network has no trained trump-declaration signal, so greedy decides."""
    policy = make_policy()
    engine = HokmEngine()
    engine.start_match(seed=3)
    assert engine.state.hands.phase is Phase.TRUMP_CALL
    seat = engine.current_seat()
    legal = engine.legal_actions(seat)


    obs = observation_for(engine.state.hands, seat, engine.state.game_points)
    assert policy.decide(engine) == GreedyPolicy().act(obs, mask_for(legal))


def test_runs_from_saved_weights_without_touching_torch(tmp_path) -> None:  # noqa: ANN001
    th.manual_seed(0)
    path = tmp_path / "qnet.npz"
    save_weights(export_weights(RankCNN(channels=TEST_CHANNELS).eval()), path)
    policy = NumpyHybridPolicy(path, verify_samples=TEST_SAMPLES, seed=0)
    assert isinstance(policy.net.params["card_head.weight"], np.ndarray)
    play_match(policy, seed=13)


def test_never_deviates_without_clearing_the_sign_test() -> None:
    """With an impossible threshold the policy must reproduce greedy exactly.

    max_p_value = 0 can never be met, so every proposal is rejected and the
    policy has to fall back to the baseline on every decision. If it ever
    returns something else, the gate is not actually controlling deviation.
    """
    th.manual_seed(0)
    weights = export_weights(RankCNN(channels=TEST_CHANNELS).eval())
    gated = NumpyHybridPolicy(weights, verify_samples=TEST_SAMPLES, max_p_value=0.0, seed=0)
    greedy = GreedyPolicy()

    engine = HokmEngine()
    engine.start_match(seed=17)

    checked = 0
    while engine.state.winner is None and checked < 60:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        obs = observation_for(engine.state.hands, seat, engine.state.game_points)
        expected = greedy.act(obs, mask_for(legal))
        if seat in {0, 2} and len(legal) > 1:
            assert gated.decide(engine) == expected
            checked += 1
        hands = engine.state.hands
        led = hands.current_trick[0][1] // 13 if hands.current_trick else None
        outcome = engine.apply_action(expected, seat=seat)
        if outcome.card is not None:
            gated.observe(seat, outcome.card, led)
        if outcome.hand_complete:
            gated.reset_hand()
    assert checked > 5, "too few multi-choice decisions exercised"


def test_seats_see_only_their_own_hand() -> None:
    """The features must never encode another seat's cards.

    The hand plane may only contain cards the acting seat holds, and the seen
    plane may only contain its hand plus cards actually played.
    """

    engine = HokmEngine()
    engine.start_match(seed=23)
    engine.apply_action(engine.legal_actions()[0])  # declare trump, deal the rest
    for seat in range(NUM_SEATS):
        obs = observation_for(engine.state.hands, seat, engine.state.game_points)
        planes, _ = build_features(obs, mask_for(engine.legal_actions(engine.current_seat())))
        own = set(engine.state.hands.hands[seat])
        rows, cols = np.nonzero(planes[PLANE_HAND])
        encoded = {int(a) * 13 + int(b) for a, b in zip(rows, cols, strict=True)}
        assert encoded == own
        rows, cols = np.nonzero(planes[PLANE_SEEN])
        seen = {int(a) * 13 + int(b) for a, b in zip(rows, cols, strict=True)}
        assert seen <= own | set(engine.state.hands.played)
