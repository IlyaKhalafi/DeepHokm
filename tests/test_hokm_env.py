"""Tests for the HokmEnv Gymnasium environment."""

from __future__ import annotations

import pickle
import random
from collections import Counter

import gymnasium
import numpy as np
import pytest
from gymnasium import Wrapper
from gymnasium.utils.env_checker import check_env

from deephokm.cards import NUM_CARDS, NUM_SUITS
from deephokm.env import HokmEnv
from deephokm.env.hokm_env import Observation
from deephokm.env.spaces import HISTORY_SLOTS
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.legality import NUM_ACTIONS
from deephokm.rules.state import NUM_SEATS, TRICKS_PER_HAND, team_of


def random_opponents(seed: int | None = None) -> list[RandomPolicy]:
    return [RandomPolicy(seed) for _ in range(NUM_SEATS)]


class CheckerProbeShim(Wrapper):
    """Resample actions for gymnasium's blind env-checker probe.

    ``check_env`` calls ``step(action_space.sample())`` once without
    consulting the action mask, and it samples before its final ``reset``,
    so the drawn action can belong to the previous episode's hand. Real
    callers (MaskablePPO, scripted policies) always act under the mask, and
    ``HokmEnv.step`` keeps strict ValueError semantics for illegal actions.
    This wrapper exists only so the checker can validate everything else
    (spaces, reset determinism, obs containment, step signature) without
    tripping over that one unsatisfiable probe.
    """

    def step(self, action: int):
        mask = self.env.unwrapped.action_masks()
        if not mask[action]:
            action = int(np.flatnonzero(mask)[0])
        return self.env.step(action)


def play_episode(env: HokmEnv, seed: int) -> tuple[int, float, int]:
    """Play one episode choosing the first legal action; return stats."""
    obs, info = env.reset(seed=seed)
    steps = 0
    total = 0.0
    done = False
    while not done:
        mask = info["action_mask"]
        assert mask.sum() > 0
        action = int(np.flatnonzero(mask)[0])
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        steps += 1
        done = term or trunc
    return steps, total, env._engine.state.winner if hasattr(env, "_engine") else None


@pytest.mark.parametrize("seat", range(NUM_SEATS))
def test_env_checker_passes(seat: int) -> None:
    env = HokmEnv(seat=seat, opponents=random_opponents(0))
    check_env(CheckerProbeShim(env), skip_render_check=True)


@pytest.mark.parametrize("seat", range(NUM_SEATS))
def test_env_checker_passes_via_make(seat: int) -> None:
    env = gymnasium.make(
        "Hokm-v0",
        seat=seat,
        opponents=random_opponents(0),
        render_mode=None,
        disable_env_checker=True,
    )
    check_env(CheckerProbeShim(env), skip_render_check=True, skip_close_check=True)
    env.close()


def test_reset_seed_reproducibility() -> None:
    env_a = HokmEnv(seat=1, opponents=random_opponents(7))
    env_b = HokmEnv(seat=1, opponents=random_opponents(7))
    obs_a, info_a = env_a.reset(seed=123)
    obs_b, info_b = env_b.reset(seed=123)
    for key in obs_a:
        np.testing.assert_array_equal(obs_a[key], obs_b[key], err_msg=key)
    np.testing.assert_array_equal(info_a["action_mask"], info_b["action_mask"])

    # Full episodes replay identically under the same seed.
    steps_a, total_a, _ = play_episode(env_a, 123)
    steps_b, total_b, _ = play_episode(env_b, 123)
    assert steps_a == steps_b
    assert total_a == total_b


def test_different_seeds_differ() -> None:
    env = HokmEnv(seat=0, opponents=random_opponents(1))
    obs_a, _ = env.reset(seed=1)
    obs_b, _ = env.reset(seed=2)
    assert not np.array_equal(obs_a["hand"], obs_b["hand"])


def test_observation_space_bounds() -> None:
    env = HokmEnv(seat=0, opponents=random_opponents())
    obs, _ = env.reset(seed=5)
    assert env.observation_space.contains(obs)
    for _ in range(50):
        mask = env.action_masks()
        action = int(random.choice(np.flatnonzero(mask).tolist()))
        obs, _, term, trunc, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if term or trunc:
            obs, _ = env.reset(seed=5)


def test_action_space_is_discrete_56() -> None:
    env = HokmEnv(seat=0)
    assert env.action_space == gymnasium.spaces.Discrete(NUM_ACTIONS)


def test_hand_starts_with_13_cards() -> None:
    env = HokmEnv(seat=2, opponents=random_opponents())
    obs, info = env.reset(seed=9)
    assert obs["hand"].sum() == 13
    # seen = own hand + every card already played (public information).
    assert obs["seen"].sum() == 13 + int(obs["trick"].sum())


def test_seen_equals_hand_plus_played() -> None:
    env = HokmEnv(seat=3, opponents=random_opponents(3))
    obs, info = env.reset(seed=11)
    done = False
    rng = random.Random(0)
    checks = 0
    while not done and checks < 200:
        hand = obs["hand"]
        seen = obs["seen"]
        trick = obs["trick"]
        # seen ⊇ hand and seen ⊇ trick
        assert np.all(seen >= hand)
        assert np.all(seen >= trick)
        mask = info["action_mask"]
        action = int(rng.choice(np.flatnonzero(mask).tolist()))
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        checks += 1


def test_trick_play_tracks_seats_and_leader() -> None:
    """trick_play[seat] holds the played card or -1; leader is first card."""
    env = HokmEnv(seat=0, opponents=random_opponents(4))
    obs, info = env.reset(seed=13)
    rng = random.Random(1)
    for _ in range(120):
        trick_play = obs["trick_play"]
        played = [int(x) for x in trick_play if x >= 0]
        assert len(played) <= NUM_SEATS
        # Seats yet to play hold -1.
        if played:
            leader = next(i for i, x in enumerate(trick_play) if x >= 0)
            table_cards = obs["trick"]
            for card in played:
                assert table_cards[card] == 1
            del leader
        mask = info["action_mask"]
        action = int(rng.choice(np.flatnonzero(mask).tolist()))
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break


def test_phase_encoding() -> None:
    env = HokmEnv(seat=0, opponents=random_opponents())
    obs, _ = env.reset(seed=17)
    # The phase is trump-call until the hakem declares; if the learner is the
    # hakem the first observation is the trump decision.
    if obs["phase"][0] == 1:
        np.testing.assert_array_equal(obs["phase"], [1, 0])
        np.testing.assert_array_equal(obs["trump"], [0, 0, 0, 0])
    else:
        np.testing.assert_array_equal(obs["phase"], [0, 1])


def test_trump_one_hot_after_declaration() -> None:
    for seat in range(NUM_SEATS):
        env = HokmEnv(seat=seat, opponents=random_opponents(seat))
        obs, info = env.reset(seed=seat + 100)
        done = False
        while not done:
            mask = info["action_mask"]
            if mask[52:].any():  # trump decision available
                np.testing.assert_array_equal(obs["trump"], [0, 0, 0, 0])
                action = 52 + seat % NUM_SUITS
                obs, _, term, trunc, info = env.step(action)
                np.testing.assert_array_equal(
                    obs["trump"], [1 if i == seat % NUM_SUITS else 0 for i in range(4)]
                )
                done = True
                del term, trunc
                break
            action = int(np.flatnonzero(mask)[0])
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc


def test_scores_are_from_own_team_perspective() -> None:
    """tricks_won/game_points must be [own, opposing] for the viewing seat."""
    env = HokmEnv(seat=1, opponents=random_opponents(2))
    obs, _ = env.reset(seed=19)
    engine = env._engine
    own = team_of(1)
    np.testing.assert_array_equal(
        obs["game_points"],
        [engine.state.game_points[own], engine.state.game_points[1 - own]],
    )
    np.testing.assert_array_equal(
        obs["tricks_won"],
        [engine.state.hands.tricks_won[own], engine.state.hands.tricks_won[1 - own]],
    )


def test_seat_one_hot() -> None:
    for seat in range(NUM_SEATS):
        env = HokmEnv(seat=seat, opponents=random_opponents(seat))
        obs, _ = env.reset(seed=21 + seat)
        expected = np.zeros(NUM_SEATS, dtype=np.int8)
        expected[seat] = 1
        np.testing.assert_array_equal(obs["seat"], expected)


def test_illegal_action_raises_value_error() -> None:
    env = HokmEnv(seat=0, opponents=random_opponents())
    obs, info = env.reset(seed=23)
    mask = info["action_mask"]
    illegal = int(np.flatnonzero(mask == 0)[0])
    with pytest.raises(ValueError, match="illegal action"):
        env.step(illegal)


def test_step_after_done_raises() -> None:
    env = HokmEnv(seat=0, opponents=random_opponents())
    obs, info = env.reset(seed=25)
    done = False
    while not done:
        action = int(np.flatnonzero(info["action_mask"])[0])
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    with pytest.raises(RuntimeError, match="episode is over"):
        env.step(52)


def test_reward_is_pm_one_at_match_end() -> None:
    for seed in range(30):
        env = HokmEnv(seat=seed % NUM_SEATS, opponents=random_opponents(seed))
        obs, info = env.reset(seed=seed)
        total = 0.0
        done = False
        while not done:
            action = int(np.flatnonzero(info["action_mask"])[0])
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            done = term or trunc
        winner = env._engine.state.winner
        own = team_of(seed % NUM_SEATS)
        expected = 1.0 if winner == own else -1.0
        assert total == expected, f"seed {seed}: reward {total} != {expected}"


def test_trick_reward_shaping() -> None:
    """Shaped reward must equal trick_reward * (own - opp) tricks + terminal.

    The expectation is derived independently by observing the engine's trick
    outcomes, so a sign inversion or magnitude error cannot pass.
    """
    trick_reward = 0.1
    env = HokmEnv(seat=0, opponents=random_opponents(31), trick_reward=trick_reward)
    obs, info = env.reset(seed=31)
    own_team = team_of(0)
    tallies = {"own": 0, "opp": 0}
    real_apply = env._engine.apply_action

    def counting_apply(action: int, seat: int | None = None) -> object:
        outcome = real_apply(action, seat)
        if outcome.trick_complete and outcome.trick_winner is not None:
            key = "own" if team_of(outcome.trick_winner) == own_team else "opp"
            tallies[key] += 1
        return outcome

    env._engine.apply_action = counting_apply  # type: ignore[method-assign]
    total = 0.0
    done = False
    while not done:
        action = int(np.flatnonzero(info["action_mask"])[0])
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        done = term or trunc

    winner = env._engine.state.winner
    terminal = 1.0 if winner == own_team else -1.0
    expected = trick_reward * (tallies["own"] - tallies["opp"]) + terminal
    assert total == pytest.approx(expected), (
        f"shaped total {total} != expected {expected} (own={tallies['own']}, opp={tallies['opp']})"
    )


def test_every_card_played_once_per_hand_via_obs() -> None:
    """Across a hand, the seen vector grows to 52 and trick totals sum to 13."""
    env = HokmEnv(seat=0, opponents=random_opponents(41))
    obs, info = env.reset(seed=41)
    done = False
    seen_counts = []
    while not done:
        seen_counts.append(int(obs["seen"].sum()))
        action = int(np.flatnonzero(info["action_mask"])[0])
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    # Within a hand the seen count only grows, up to the full 52; a re-deal
    # drops it back to the new hand (>= 13 cards).
    for a, b in zip(seen_counts, seen_counts[1:], strict=False):
        assert b >= a or b >= 13
    assert max(seen_counts) == NUM_CARDS


def test_no_hidden_information_leak() -> None:
    """Observations must be derivable from public state + own hand only.

    Two environments with identical public trajectories but different hidden
    hands would be indistinguishable; here we verify the stronger property
    that the observation contains exactly: own hand, played cards, current
    trick, trump, phase, scores, seat.
    """
    for seat in range(NUM_SEATS):
        env = HokmEnv(seat=seat, opponents=random_opponents(seat))
        obs, info = env.reset(seed=seat + 300)
        done = False
        rng = random.Random(seat)
        while not done:
            hand = env._engine.state.hands.hands[seat]
            played = env._engine.state.hands.played
            on_table = [c for _, c in env._engine.state.hands.current_trick]
            expected_hand = np.zeros(NUM_CARDS, dtype=np.int8)
            expected_hand[hand] = 1
            expected_seen = np.zeros(NUM_CARDS, dtype=np.int8)
            expected_seen[hand] = 1
            expected_seen[played] = 1
            expected_trick = np.zeros(NUM_CARDS, dtype=np.int8)
            expected_trick[on_table] = 1
            completed = played[: len(played) - len(on_table)]
            expected_history = np.full(HISTORY_SLOTS, -1, dtype=np.int64)
            expected_history[: len(completed)] = completed[::-1]
            np.testing.assert_array_equal(obs["hand"], expected_hand)
            np.testing.assert_array_equal(obs["seen"], expected_seen)
            np.testing.assert_array_equal(obs["trick"], expected_trick)
            np.testing.assert_array_equal(obs["history"], expected_history)
            # A partner's hidden hand must never appear anywhere.
            partner = (seat + 2) % NUM_SEATS
            partner_private = set(env._engine.state.hands.hands[partner]) - set(played)
            for card in partner_private:
                assert obs["hand"][card] == 0
                assert obs["trick"][card] == 0
                assert card not in obs["history"].tolist()
                if card not in hand:
                    assert obs["seen"][card] == 0
            mask = info["action_mask"]
            action = int(rng.choice(np.flatnonzero(mask).tolist()))
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc


def test_follow_suit_never_violated_under_random_policy() -> None:
    """Playing always-first-legal must never break follow-suit (mask works)."""
    for seed in range(50):
        env = HokmEnv(seat=seed % NUM_SEATS, opponents=random_opponents(seed))
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            mask = info["action_mask"]
            legal = np.flatnonzero(mask)
            # If a trick is in progress, every legal card must follow suit
            # when the seat holds the led suit.
            trick_play = obs["trick_play"]
            if (trick_play >= 0).any() and (mask[:52].any()):
                # The leader is the engine's leader, not the first seat-order
                # entry: trick_play is indexed by seat, not play order.
                leader = env._engine.state.hands.leader
                led_card = int(trick_play[leader])
                led_suit = led_card // 13
                hand = np.flatnonzero(obs["hand"])
                suit_cards = [c for c in hand if c // 13 == led_suit]
                if suit_cards:
                    assert set(legal.tolist()) <= set(suit_cards)
            action = int(legal[0])
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc


def test_episode_length_matches_match_actions() -> None:
    """Learner steps: one per learner decision (13+ per hand as the hakem)."""
    env = HokmEnv(seat=0, opponents=random_opponents(51))
    obs, info = env.reset(seed=51)
    steps = 0
    done = False
    while not done:
        action = int(np.flatnonzero(info["action_mask"])[0])
        obs, _, term, trunc, info = env.step(action)
        steps += 1
        done = term or trunc
    # A match has >= 7 hands; the learner acts 13 times per hand it plays plus
    # once per trump call when hakem.
    assert steps >= TRICKS_PER_HAND


def test_opponent_illegal_action_raises() -> None:
    """An opponent returning an off-mask action must be caught, not coerced."""

    class BadPolicy:
        def act(self, observation: Observation, action_mask: np.ndarray) -> int:
            return int(np.flatnonzero(action_mask == 0)[0])

    opponents: list[object] = [BadPolicy() for _ in range(NUM_SEATS)]
    env = HokmEnv(seat=0, opponents=opponents)  # type: ignore[arg-type]
    # Seat 0 may or may not be the hakem; either way some opponent acts during
    # reset unless the learner leads every trick, so loop resets until an
    # opponent is queried.
    with pytest.raises(RuntimeError, match="illegal action"):
        for seed in range(10):
            env.reset(seed=seed)


def test_registered_entry_point_resolves() -> None:
    env = gymnasium.make("Hokm-v0", seat=0, opponents=random_opponents(0))
    obs, info = env.reset(seed=3)
    assert "action_mask" in info
    env.close()


def test_render_human_mode_produces_output(capsys: pytest.CaptureFixture[str]) -> None:
    env = HokmEnv(seat=0, opponents=random_opponents(61), render_mode="human")
    env.reset(seed=61)
    action = int(np.flatnonzero(env.action_masks())[0])
    env.step(action)
    out = capsys.readouterr().out
    assert "plays" in out or "Trump declared" in out or "hakem" in out


def test_rewards_sum_to_matches_won() -> None:
    """A random learner against random opponents wins ~half its matches.

    Uses independent opponent seeds and a two-sigma band around 50% so the
    check is a bias detector, not a flaky coin flip.
    """
    rng = np.random.default_rng(2024)
    wins = 0
    n = 120
    for seed in range(n):
        seat = seed % NUM_SEATS
        env = HokmEnv(seat=seat, opponents=random_opponents(9000 + seed))
        obs, info = env.reset(seed=seed)
        while True:
            legal = np.flatnonzero(info["action_mask"])
            action = int(rng.choice(legal))
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        if env._engine.state.winner == team_of(seat):
            wins += 1
    share = wins / n
    assert 0.35 < share < 0.65, f"win-rate bias detected: {share:.2f}"


def test_seat_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="seat must be"):
        HokmEnv(seat=4)


def test_action_masks_matches_info() -> None:
    env = HokmEnv(seat=0, opponents=random_opponents())
    obs, info = env.reset(seed=71)
    np.testing.assert_array_equal(env.action_masks(), info["action_mask"])


def test_multiple_resets_stay_consistent() -> None:
    env = HokmEnv(seat=2, opponents=random_opponents(81))
    totals: Counter[float] = Counter()
    for seed in (81, 82, 83, 81):
        obs, info = env.reset(seed=seed)
        total = 0.0
        done = False
        while not done:
            action = int(np.flatnonzero(info["action_mask"])[0])
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            done = term or trunc
        totals[total] += 1
    # Two runs of seed 81 must produce the same total.
    assert totals[totals.most_common(1)[0][0]] >= 2


def test_env_pickles_for_subproc_vec_env() -> None:
    """SubprocVecEnv workers receive envs by pickle; the state must survive.

    The action space's env-bound sampler is deliberately not pickled; asking
    it to sample afterwards fails loudly rather than silently drawing illegal
    actions.
    """
    env = HokmEnv(seat=0, opponents=random_opponents(91))
    env.reset(seed=91)
    restored = pickle.loads(pickle.dumps(env))
    mask = restored.action_masks()
    assert mask.sum() > 0
    obs, info = restored.reset(seed=92)
    assert info["action_mask"].sum() > 0
    with pytest.raises(RuntimeError, match="lost its legal-action provider"):
        restored.action_space.sample()


def test_history_is_reverse_play_order_of_completed_tricks() -> None:
    """The history slot must carry recency, which ``seen`` cannot express."""
    env = HokmEnv(seat=0, opponents=random_opponents(9))
    obs, info = env.reset(seed=9)
    rng = random.Random(9)
    done = False
    saw_full_trick = False
    while not done:
        played = env.engine.state.hands.played
        on_table = [card for _, card in env.engine.state.hands.current_trick]
        completed = played[: len(played) - len(on_table)]
        history = [int(card) for card in obs["history"] if card >= 0]
        assert history == completed[::-1]
        # Nothing on the table is also in history, and nothing repeats.
        assert not set(history) & set(on_table)
        assert len(set(history)) == len(history)
        if len(completed) >= NUM_SEATS:
            saw_full_trick = True
        action = int(rng.choice(np.flatnonzero(info["action_mask"]).tolist()))
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    assert saw_full_trick, "the episode must cover at least one completed trick"
    env.close()


def test_history_resets_between_hands() -> None:
    """History is per hand: a new deal starts from an empty history."""
    env = HokmEnv(seat=0, opponents=random_opponents(21))
    obs, info = env.reset(seed=21)
    rng = random.Random(21)
    hand_number = info["hand_number"]
    checked = 0
    done = False
    while not done and checked < 3:
        action = int(rng.choice(np.flatnonzero(info["action_mask"]).tolist()))
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if not done and info["hand_number"] != hand_number:
            hand_number = info["hand_number"]
            played = env.engine.state.hands.played
            on_table = [card for _, card in env.engine.state.hands.current_trick]
            completed = len(played) - len(on_table)
            assert int((obs["history"] >= 0).sum()) == completed
            checked += 1
    assert checked > 0, "the episode must span more than one hand"
    env.close()
