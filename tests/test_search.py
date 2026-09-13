"""Tests for the root-sampled IIMC search wrapper."""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest
from sb3_contrib import MaskablePPO

from deephokm.env.hokm_env import HokmEnv
from deephokm.env.spaces import mask_for, observation_for
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.policies.random_policy import RandomPolicy
from deephokm.policies.search import (
    IIMCSearchPolicy,
    VoidTracker,
    sample_determinized_hands,
)
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import NUM_SEATS, Phase


def _tiny_model() -> MaskablePPO:
    env = HokmEnv(seat=0, opponents=[RandomPolicy(i) for i in range(NUM_SEATS)])
    return MaskablePPO(policy=HokmMaskablePolicy, env=env, device="cpu", seed=0, verbose=0)


def _play_to_card_play(engine: HokmEngine, seed: int) -> None:
    """Advance a fresh match past the trump call using a random legal action."""
    engine.start_match(seed=seed)
    rng = random.Random(seed)
    while engine.state.hands.phase is Phase.TRUMP_CALL:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        engine.apply_action(rng.choice(legal), seat=seat)


def test_void_tracker_flags_a_seat_that_fails_to_follow_suit() -> None:
    tracker = VoidTracker()
    # Suit 0 led; seat 1 plays suit 1 instead -> seat 1 is void in suit 0.
    tracker.observe(seat=1, card=13, led_suit=0)
    assert 0 in tracker.voids[1]
    assert tracker.voids[0] == set()

    tracker.reset()
    assert all(v == set() for v in tracker.voids)


def test_void_tracker_leader_is_never_marked_void() -> None:
    tracker = VoidTracker()
    tracker.observe(seat=2, card=5, led_suit=None)
    assert tracker.voids[2] == set()


def test_sample_determinized_hands_respects_sizes_and_voids() -> None:
    root_seat = 0
    root_hand = [0, 1, 2]
    # 49 unseen cards split across the other 3 seats: sizes chosen freely.
    unseen_pool = list(range(3, 52))
    remaining_sizes = [len(root_hand), 16, 16, 17]
    voids: list[set[int]] = [set(), {0}, set(), set()]  # seat 1 void in suit 0 (cards 0-12)

    rng = random.Random(0)
    for _ in range(20):
        # A high attempt budget: with a 16-card chunk drawn from 49 cards
        # (10 of them void for seat 1), a single shuffle satisfies the void
        # constraint only ~2.6% of the time, so the production default of
        # 200 attempts occasionally (rarely) falls back within just this
        # test's 20 repeated calls -- not a bug, but this test is about the
        # void constraint actually holding, so it needs enough budget that
        # falling back here would indicate a real regression instead of bad
        # luck (dedicated test below covers the fallback path itself).
        hands = sample_determinized_hands(
            root_seat,
            root_hand,
            unseen_pool,
            remaining_sizes,
            voids=voids,
            rng=rng,
            max_attempts=5000,
        )
        assert hands[0] == root_hand
        assert [len(h) for h in hands] == remaining_sizes
        # Every unseen card is assigned to exactly one seat.
        all_assigned = sorted(c for seat in range(1, NUM_SEATS) for c in hands[seat])
        assert all_assigned == sorted(unseen_pool)
        # Seat 1's void in suit 0 (cards 0-12) must be respected.
        assert all(c >= 13 for c in hands[1])


def test_sample_determinized_hands_rejects_mismatched_sizes() -> None:
    rng = random.Random(0)
    # remaining_sizes for seats 1-3 sum to 15, but the pool only has 3 cards.
    try:
        sample_determinized_hands(
            0, [0], [1, 2, 3], [1, 5, 5, 5], voids=[set()] * NUM_SEATS, rng=rng
        )
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a size/pool mismatch")


def test_sample_determinized_hands_raises_when_voids_make_it_truly_infeasible() -> None:
    """A truly infeasible void set must raise, never silently cross a void.

    All of the unseen pool is suit 0; every other seat is marked void in
    suit 0, so no assignment -- constrained or not -- can ever satisfy
    every seat's voids simultaneously. This can never happen for a real,
    live game state (whatever the actual hidden deal is, it is itself a
    witness that a feasible assignment exists), so this construction is
    deliberately artificial. The old fallback used to paper over exactly
    this case by silently assigning a void-suit card anyway -- simulating
    a seat following a suit it has publicly proven it does not hold, an
    impossible world. The fix must fail loudly instead.
    """
    root_seat = 0
    root_hand = [40, 41]  # suit 3 cards, irrelevant to the pool
    unseen_pool = list(range(0, 13))  # all of suit 0
    remaining_sizes = [len(root_hand), 4, 4, 5]
    voids: list[set[int]] = [set(), {0}, {0}, {0}]

    rng = random.Random(0)
    with pytest.raises(RuntimeError, match="no void-respecting hand assignment exists"):
        sample_determinized_hands(
            root_seat, root_hand, unseen_pool, remaining_sizes, voids=voids, rng=rng, max_attempts=5
        )


def test_fallback_still_respects_voids_when_feasible() -> None:
    """The greedy fallback must not gratuitously violate a satisfiable void.

    With ``max_attempts=1`` the whole-deal rejection sampler almost always
    exhausts its budget and falls back (a single random split respects this
    test's void constraint only ~2.6% of the time -- see the comment in
    ``test_sample_determinized_hands_respects_sizes_and_voids``), but unlike
    the infeasible-by-construction test above, a void-respecting assignment
    genuinely exists here. The fallback must find one instead of shrugging
    and assigning a card of the void suit anyway.
    """
    root_seat = 0
    root_hand = [0, 1, 2]
    unseen_pool = list(range(3, 52))
    remaining_sizes = [len(root_hand), 16, 16, 17]
    voids: list[set[int]] = [set(), {0}, set(), set()]

    rng = random.Random(0)
    for _ in range(20):
        hands = sample_determinized_hands(
            root_seat, root_hand, unseen_pool, remaining_sizes, voids=voids, rng=rng, max_attempts=1
        )
        assert [len(h) for h in hands] == remaining_sizes
        assert all(c >= 13 for c in hands[1])


def test_observation_for_never_depends_on_other_seats_hands() -> None:
    """The single biggest correctness risk: no hidden-info leakage.

    Two engines that differ only in seats other than the acting one must
    produce an identical observation for the acting seat -- otherwise the
    search could be conditioning a decision on a determinized world instead
    of only on genuinely public information.
    """
    engine = HokmEngine()
    _play_to_card_play(engine, seed=1)
    acting_seat = engine.current_seat()

    baseline = observation_for(engine.state.hands, acting_seat, engine.state.game_points)

    # Exchange one card between two *other* seats -- actual card membership
    # changes (unlike reordering a hand list, which a set-like binary vector
    # cannot even represent), while both seats' hand sizes are preserved.
    other_seats = [s for s in range(NUM_SEATS) if s != acting_seat]
    seat_a, seat_b = other_seats[0], other_seats[1]
    perturbed = copy.deepcopy(engine.state.hands)
    assert perturbed.hands[seat_a] and perturbed.hands[seat_b]
    card_a, card_b = perturbed.hands[seat_a][0], perturbed.hands[seat_b][0]
    assert card_a != card_b
    perturbed.hands[seat_a][0] = card_b
    perturbed.hands[seat_b][0] = card_a
    perturbed_obs = observation_for(perturbed, acting_seat, engine.state.game_points)

    for key in baseline:
        assert np.array_equal(baseline[key], perturbed_obs[key]), key


def test_decide_returns_a_legal_action_and_never_mutates_the_real_engine() -> None:
    model = _tiny_model()
    search = IIMCSearchPolicy(model.policy, n_samples=2, top_k=2, seed=0)

    engine = HokmEngine()
    _play_to_card_play(engine, seed=2)
    search.reset_hand()

    before = copy.deepcopy(engine.state)
    seat = engine.current_seat()
    legal = engine.legal_actions(seat)

    action = search.decide(engine)

    assert action in legal
    assert engine.state == before


def test_decide_handles_a_forced_single_legal_action_without_searching() -> None:
    model = _tiny_model()
    search = IIMCSearchPolicy(model.policy, n_samples=2, top_k=2, seed=0)

    engine = HokmEngine()
    _play_to_card_play(engine, seed=3)
    search.reset_hand()

    # Drive to a state with a single legal action by exhausting a hand down
    # to each seat's last card (the last trick always has exactly one legal
    # card per seat).
    while len(engine.state.hands.hands[engine.current_seat()]) > 1:
        seat = engine.current_seat()
        legal = engine.legal_actions(seat)
        outcome = engine.apply_action(legal[0], seat=seat)
        if outcome.hand_complete:
            return  # hand ended early (a team already reached 7 tricks)

    seat = engine.current_seat()
    legal = engine.legal_actions(seat)
    assert len(legal) == 1
    assert search.decide(engine) == legal[0]


def test_decide_skips_trump_call_search_and_defers_to_the_wrapped_policy() -> None:
    model = _tiny_model()
    search = IIMCSearchPolicy(model.policy, n_samples=2, top_k=2, seed=0)

    engine = HokmEngine()
    engine.start_match(seed=4)
    assert engine.state.hands.phase is Phase.TRUMP_CALL

    seat = engine.current_seat()
    obs = observation_for(engine.state.hands, seat, engine.state.game_points)
    mask = mask_for(engine.legal_actions(seat))
    action, _ = model.policy.predict(obs, action_masks=mask, deterministic=True)

    assert search.decide(engine) == int(action)
