"""Tests for the transformer feature extractor and policy wiring."""

from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

import deephokm.nn.extractor as extractor_mod
from deephokm.cards import NUM_RANKS
from deephokm.env import HokmEnv
from deephokm.env.spaces import empty_observation, observation_space
from deephokm.nn.extractor import HokmTransformerExtractor
from deephokm.nn.policy import POLICY_HEAD_GAIN, VALUE_HEAD_GAIN, HokmMaskablePolicy
from deephokm.nn.tokenizer import (
    CTX_COLUMNS,
    NUM_HAND_SLOTS,
    NUM_TRICK_SLOTS,
    ROLE_NA,
    tokenize,
    tokenize_tensor_batch,
)
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.state import NUM_SEATS

# The d_model=256/6-layer model sits near 8M parameters; the cap guards
# against accidental capacity regressions, not the reference 128-wide model.
MAX_PARAMS = 10_000_000


def make_env(seed: int = 0, seat: int = 0) -> HokmEnv:
    env = HokmEnv(seat=seat, opponents=[RandomPolicy(seed + i) for i in range(NUM_SEATS)])
    env.reset(seed=seed)
    return env


def obs_batch(env: HokmEnv, n: int) -> dict[str, th.Tensor]:
    """Stack the current observation ``n`` times into a tensor batch."""
    obs, _ = env.reset(seed=5)
    return {k: th.tensor(np.stack([v] * n)) for k, v in obs.items()}


def gather_states(n: int, seed: int = 0, seat: int | None = None) -> list[dict[str, np.ndarray]]:
    """Collect ``n`` diverse observation dicts from a random rollout.

    Rotates the learner seat with the seed unless ``seat`` is pinned, so the
    collected states cover every seat's perspective.
    """
    chosen_seat = seat if seat is not None else seed % NUM_SEATS
    env = make_env(seed, seat=chosen_seat)
    rng = random.Random(seed)
    states: list[dict[str, np.ndarray]] = []
    obs, info = env.reset(seed=seed)
    done = False
    while not done and len(states) < n:
        states.append(dict(obs))
        legal = np.flatnonzero(info["action_mask"])
        obs, _, term, trunc, info = env.step(int(rng.choice(legal)))
        done = term or trunc
    return states


def stack(states: list[dict[str, np.ndarray]]) -> dict[str, th.Tensor]:
    return {k: th.tensor(np.stack([s[k] for s in states])) for k in states[0]}


def test_forward_shapes_on_padded_batch() -> None:
    ext = HokmTransformerExtractor(observation_space())
    states = gather_states(8)
    out = ext(stack(states))
    assert out.shape == (8, ext.d_model)
    assert th.isfinite(out).all()


def test_forward_edge_case_empty_hand() -> None:
    """The terminal observation (hand fully played out) must forward cleanly."""
    ext = HokmTransformerExtractor(observation_space())
    env = make_env(3)
    rng = random.Random(0)
    obs, info = env.reset(seed=3)
    done = False
    while not done:
        legal = np.flatnonzero(info["action_mask"])
        obs, _, term, trunc, info = env.step(int(rng.choice(legal)))
        done = term or trunc
    assert obs["hand"].sum() == 0, "terminal observation should hold no cards"
    batch = {k: th.tensor(np.stack([v, v])) for k, v in obs.items()}
    out = ext(batch)
    assert out.shape == (2, ext.d_model)
    assert th.isfinite(out).all()


def test_forward_edge_case_empty_trick() -> None:
    ext = HokmTransformerExtractor(observation_space())
    states = gather_states(30)
    no_trick = [s for s in states if (s["trick_play"] < 0).all()]
    assert no_trick, "no lead state collected"
    out = ext(stack(no_trick))
    assert out.shape == (len(no_trick), ext.d_model)
    assert th.isfinite(out).all()


def test_forward_edge_case_pre_trump_phase() -> None:
    """The trump-call decision: all-zero trump vector and phase=[1,0]."""
    ext = HokmTransformerExtractor(observation_space())
    env = HokmEnv(seat=None or 0, opponents=[RandomPolicy(i) for i in range(NUM_SEATS)])
    obs, info = env.reset(seed=1)
    while info["phase"] != "TRUMP_CALL":
        obs, info = env.reset(seed=random.randint(0, 10_000))
    batch = {k: th.tensor(np.stack([v, v])) for k, v in obs.items()}
    out = ext(batch)
    assert out.shape == (2, ext.d_model)
    assert th.isfinite(out).all()


def test_parameter_count_capped() -> None:
    ext = HokmTransformerExtractor(observation_space())
    n = sum(p.numel() for p in ext.parameters())
    print(f"extractor parameters: {n:,}")
    assert n < MAX_PARAMS


def test_gradient_flow_to_every_parameter() -> None:
    ext = HokmTransformerExtractor(observation_space())
    states = gather_states(4)
    out = ext(stack(states))
    out.sum().backward()
    missing = [name for name, p in ext.named_parameters() if p.grad is None]
    zeroed = [
        name
        for name, p in ext.named_parameters()
        if p.grad is not None and p.grad.abs().sum().item() == 0.0
    ]
    assert not missing, f"parameters with no grad: {missing}"
    # Positional/type embeddings for groups present in the batch must move;
    # hand-position slots are unused by design (positions are all 0 there).
    assert not zeroed, f"parameters with zero grad: {zeroed}"


def test_no_nan_after_multiple_forwards() -> None:
    ext = HokmTransformerExtractor(observation_space())
    states = gather_states(16)
    for _ in range(5):
        out = ext(stack(states))
        assert th.isfinite(out).all()


def test_padding_invariance() -> None:
    """Padded slots must not influence the pooled features.

    Every token position that the padding mask marks as unreal is corrupted
    with arbitrary card ids; the output must be identical. This catches
    padding leaking into attention (encoder mask), pooling (attn mask), or
    the embeddings (PAD routed through the card table).
    """
    ext = HokmTransformerExtractor(observation_space())
    ext.eval()
    states = gather_states(4)
    batch = stack(states)
    with th.no_grad():
        base = ext(batch)

    class PadScrambler:
        """Tokenize, then overwrite PAD slots with arbitrary card ids."""

        def __init__(self, inner):  # noqa: ANN001
            self.inner = inner

        def __call__(self, obs):  # noqa: ANN001
            tokens = self.inner(obs)
            pad = ~tokens.padding_mask
            if pad.any():
                noise = th.randint(0, 52, tokens.tokens.shape, device=tokens.tokens.device)
                noise_rank = noise % NUM_RANKS
                shape = tokens.tokens.shape
                dev = tokens.tokens.device
                noise_trump = th.randint(0, 2, shape, device=dev).bool()
                noise_slot = th.randint(0, 4, shape, device=dev)
                tokens = dataclasses.replace(
                    tokens,
                    tokens=th.where(pad, noise, tokens.tokens),
                    ranks=th.where(pad, noise_rank, tokens.ranks),
                    is_trump=th.where(pad, noise_trump, tokens.is_trump),
                    suit_slots=th.where(pad, noise_slot, tokens.suit_slots),
                )
            return tokens

    original = extractor_mod.tokenize_tensor_batch
    extractor_mod.tokenize_tensor_batch = PadScrambler(original)
    try:
        with th.no_grad():
            scrambled = ext(batch)
    finally:
        extractor_mod.tokenize_tensor_batch = original
    assert th.allclose(base, scrambled, atol=1e-5), "padded slots leaked into features"


def test_hand_order_invariance() -> None:
    """Permuting the hand tokens must not change the pooled features.

    The hand is a set: no positional embedding is applied to hand slots and
    attention is permutation-equivariant, so scrambling the order of the hand
    token columns (and their validity/type flags with them) must leave the
    pooled output identical. This catches accidental order-sensitivity in the
    hand encoding.
    """
    ext = HokmTransformerExtractor(observation_space())
    ext.eval()
    states = gather_states(4)
    batch = stack(states)
    with th.no_grad():
        base = ext(batch)

    class HandShuffler:
        """Tokenize, then reverse the hand token columns (a permutation)."""

        def __init__(self, inner):  # noqa: ANN001
            self.inner = inner

        def __call__(self, obs):  # noqa: ANN001
            tokens = self.inner(obs)
            hand = slice(0, 13)
            updates = {}
            for field in (
                "tokens",
                "is_card",
                "ranks",
                "is_trump",
                "suit_slots",
                "type_ids",
                "padding_mask",
                "positions",
                "roles",
            ):
                value = getattr(tokens, field)
                updates[field] = th.cat([value[:, hand].flip(dims=[1]), value[:, 13:]], dim=1)
            return dataclasses.replace(tokens, **updates)

    original = extractor_mod.tokenize_tensor_batch
    extractor_mod.tokenize_tensor_batch = HandShuffler(original)
    try:
        with th.no_grad():
            shuffled = ext(batch)
    finally:
        extractor_mod.tokenize_tensor_batch = original
    assert th.allclose(base, shuffled, atol=1e-5), "hand order changed features"


def test_tokenizer_layout_is_fixed() -> None:
    """Hand, then the trick, then history, then the context slots."""
    states = gather_states(6)
    hist_start = NUM_HAND_SLOTS + NUM_TRICK_SLOTS
    for state in states:
        t = tokenize(state)
        tokens = t.tokens[0].tolist()
        hand = sorted(int(c) for c in np.flatnonzero(state["hand"]))
        assert tokens[: len(hand)] == hand
        n_played = len(_played_cards(state))
        assert tokens[NUM_HAND_SLOTS + n_played : hist_start] == [52] * (
            NUM_TRICK_SLOTS - n_played
        )
        history = [int(c) for c in state["history"] if c >= 0]
        assert tokens[hist_start : hist_start + len(history)] == history
        # Context slots always present.
        assert t.padding_mask[0][list(CTX_COLUMNS)].all()


def _played_cards(state: dict[str, np.ndarray]) -> list[int]:
    return [int(c) for c in state["trick_play"] if c >= 0]


@pytest.mark.parametrize("seat", range(NUM_SEATS))
def test_trick_tokens_follow_play_order(seat: int) -> None:
    """Trick tokens must be in true play order for every learner seat.

    Regression test for the leader-inference bug: when the played run wraps
    past seat 0 (leader 2 with seats {2,3,0} played), the lowest played seat
    is NOT the leader. The leader is the played seat whose counterclockwise
    neighbor has not played.
    """
    states = gather_states(40, seed=seat + 1, seat=seat)
    checked = 0
    for state in states:
        trick_play = np.asarray(state["trick_play"])
        played_mask = trick_play >= 0
        if not played_mask.any() or played_mask.all():
            continue
        # Derive the true leader independently.
        leader = None
        for s in range(NUM_SEATS):
            if played_mask[s] and not played_mask[(s - 1) % NUM_SEATS]:
                leader = s
                break
        assert leader is not None
        expected = []
        for offset in range(NUM_SEATS):
            s = (leader + offset) % NUM_SEATS
            if played_mask[s]:
                expected.append(int(trick_play[s]))
        t = tokenize(state)
        got = [c for c in t.tokens[0, 13:17].tolist() if c != 52]
        assert got == expected, (
            f"seat {seat}: trick_play={trick_play.tolist()} leader={leader} "
            f"expected={expected} got={got}"
        )
        checked += 1
    assert checked > 5, "no mid-trick states collected"


def test_full_trick_tokenizers_agree_on_zero_tokens() -> None:
    """A fully-played trick yields zero trick tokens in both tokenizers.

    Completed tricks never appear in learner observations (the env resets the
    trick first); both paths must agree on the synthetic input regardless.
    """
    obs = empty_observation()
    obs["trick_play"] = np.array([10, 20, 30, 40])
    obs["hand"][5] = 1
    single = tokenize(obs)
    assert single.tokens[0, 13:17].tolist() == [52] * 4
    batch = {k: th.tensor(np.stack([v])) for k, v in obs.items()}
    vectorized = tokenize_tensor_batch(batch)
    for field in ("tokens", "is_card", "ranks", "is_trump", "suit_slots",
                  "type_ids", "positions", "roles", "padding_mask"):
        assert th.equal(getattr(single, field)[0], getattr(vectorized, field)[0]), (
            f"{field} mismatch"
        )


def test_trick_order_wrapping_case() -> None:
    """Direct check of the wrap-around leader case."""
    obs = empty_observation()
    obs["trick_play"] = np.array([32, -1, 33, 27])  # leader=2, wraps past seat 0
    obs["hand"][10] = 1
    t = tokenize(obs)
    assert t.tokens[0, 13:17].tolist() == [33, 27, 32, 52]


def test_trick_and_hand_roles_are_relative_to_the_acting_seat() -> None:
    """Role tokens must encode who played each card, relative to the actor.

    Reusing the wrap-around trick from ``test_trick_order_wrapping_case``
    (leader 2, play order seats 2, 3, 0) with the acting seat set to 1: role
    is ``(played_seat - acting_seat) % NUM_SEATS``, so the expected roles in
    play order are 1 (seat 2), 2 (seat 3), 3 (seat 0), then ROLE_NA padding.
    Hand tokens (always the acting seat's own cards) must be role 0.
    """
    obs = empty_observation()
    obs["trick_play"] = np.array([32, -1, 33, 27])
    obs["hand"][10] = 1
    obs["seat"][1] = 1
    t = tokenize(obs)
    trick_roles = t.roles[0, NUM_HAND_SLOTS : NUM_HAND_SLOTS + NUM_TRICK_SLOTS].tolist()
    assert trick_roles == [1, 2, 3, ROLE_NA]
    assert t.roles[0, 0].item() == 0  # the one hand card


def test_history_roles_are_relative_to_the_acting_seat() -> None:
    """History roles must track who actually played each historical card."""
    env = HokmEnv(seat=2, opponents=[RandomPolicy(i) for i in range(NUM_SEATS)])
    obs, info = env.reset(seed=17)
    rng = np.random.default_rng(17)
    done = False
    saw_history = False
    while not done:
        history_role = np.asarray(obs["history_role"])
        history = np.asarray(obs["history"])
        real = history >= 0
        if real.any():
            saw_history = True
            played = env.engine.state.hands.played
            played_by = env.engine.state.hands.played_by
            on_table = len(env.engine.state.hands.current_trick)
            completed_cards = played[: len(played) - on_table]
            completed_seats = played_by[: len(played_by) - on_table]
            expected_cards = completed_cards[::-1]
            expected_roles = [(s - env.seat) % NUM_SEATS for s in completed_seats[::-1]]
            n = int(real.sum())
            assert history[:n].tolist() == expected_cards
            assert history_role[:n].tolist() == expected_roles
        action = int(rng.choice(np.flatnonzero(info["action_mask"])))
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    assert saw_history, "the episode must reach at least one completed trick"
    env.close()


def test_seat_context_token_present() -> None:
    """The acting seat must be tokenized as a context value."""
    for seat in range(NUM_SEATS):
        states = gather_states(3, seed=seat + 20, seat=seat)
        for state in states:
            t = tokenize(state)
            assert t.context_values[0, 4].item() == seat


def test_tokenizers_agree() -> None:
    """The vectorized batch tokenizer must match the per-row tokenizer."""
    states = gather_states(32)
    batch = stack(states)
    vectorized = tokenize_tensor_batch(batch)
    for i, state in enumerate(states):
        single = tokenize(state)
        for field in (
            "tokens",
            "is_card",
            "ranks",
            "is_trump",
            "suit_slots",
            "type_ids",
            "positions",
            "roles",
            "context_values",
            "padding_mask",
        ):
            a = getattr(single, field)[0]
            b = getattr(vectorized, field)[i]
            assert th.equal(a, b), f"{field} mismatch at sample {i}"


def test_suit_permutation_invariance() -> None:
    """Swapping two non-trump suits changes nothing observable by extractor.

    Cards differ only by rank + trump flag. Permuting suit labels among
    non-trump suits leaves every rank identical and every trump flag False;
    the only change is hand-slot order, absorbed by pooling up to float
    rounding (same tolerance as the hand-order test). Trump suit swap would
    flip flags, so non-trump suits only.
    """
    ext = HokmTransformerExtractor(observation_space())
    ext.eval()
    states = gather_states(12)
    declared = [s for s in states if int(np.asarray(s["trump"]).sum()) > 0]
    assert declared, "no post-trump states collected"
    trump0 = int(np.asarray(declared[0]["trump"]).argmax())
    keep = [s for s in declared if int(np.asarray(s["trump"]).argmax()) == trump0]
    batch = stack(keep)

    def swap_nontrump(obs: dict[str, th.Tensor]) -> dict[str, th.Tensor]:
        trump = obs["trump"]
        trump_suit = int(trump.argmax(dim=1)[0])
        others = [s for s in range(4) if s != trump_suit]
        a, b = others[0], others[1]
        out = dict(obs)
        for key in ("hand", "seen", "trick"):
            m = obs[key].clone()
            ca = m[:, a * 13 : (a + 1) * 13].clone()
            m[:, a * 13 : (a + 1) * 13] = m[:, b * 13 : (b + 1) * 13]
            m[:, b * 13 : (b + 1) * 13] = ca
            out[key] = m
        for key in ("trick_play", "history"):
            m = obs[key].clone()
            orig = obs[key]
            is_a = (orig >= a * 13) & (orig < (a + 1) * 13)
            is_b = (orig >= b * 13) & (orig < (b + 1) * 13)
            m = th.where(is_a, orig + (b - a) * 13, m)
            m = th.where(is_b, orig - (b - a) * 13, m)
            out[key] = m
        return out

    swapped_batch = swap_nontrump(batch)
    assert (
        (swapped_batch["hand"] != batch["hand"]).any()
        or (swapped_batch["trick_play"] != batch["trick_play"]).any()
        or (swapped_batch["history"] != batch["history"]).any()
    ), "swap was a no-op: no swapped-suit card in batch"
    with th.no_grad():
        base = ext(batch)
        swapped = ext(swapped_batch)
    assert th.allclose(base, swapped, atol=1e-5), "non-trump suit swap changed features"


def test_policy_constructs_and_predicts() -> None:

    def lr_schedule(progress: float) -> float:
        return 3e-4

    policy = HokmMaskablePolicy(
        observation_space(),
        spaces.Discrete(56),
        lr_schedule,
    )
    assert isinstance(policy.features_extractor, HokmTransformerExtractor)
    states = gather_states(4)
    batch = stack(states)
    with th.no_grad():
        actions, values, log_probs = policy(batch)
    assert actions.shape == (4,)
    assert values.shape == (4, 1)
    assert log_probs.shape == (4,)
    assert th.isfinite(values).all() and th.isfinite(log_probs).all()


def test_policy_head_gains_are_orthogonal_spec() -> None:
    """The action/value heads must use the specified orthogonal init gains.

    sb3-contrib applies orthogonal init with gain 0.01 to action_net and 1.0
    to value_net; verify by re-running the initializer and comparing against
    the documented behavior (the parent's gain table).
    """

    def lr_schedule(progress: float) -> float:
        return 3e-4

    policy = HokmMaskablePolicy(
        observation_space(),
        spaces.Discrete(56),
        lr_schedule,
    )
    # Orthogonal init with gain g gives rows with norm ~g.
    action_rows = policy.action_net.weight.norm(dim=1)
    value_rows = policy.value_net.weight.norm(dim=1)
    assert th.allclose(action_rows, th.full_like(action_rows, POLICY_HEAD_GAIN), atol=1e-3)
    assert th.allclose(value_rows, th.full_like(value_rows, VALUE_HEAD_GAIN), atol=1e-3)


def test_masking_applied_in_forward() -> None:
    """Passing an action mask must zero out masked logits in the sample."""

    def lr_schedule(progress: float) -> float:
        return 3e-4

    policy = HokmMaskablePolicy(
        observation_space(),
        spaces.Discrete(56),
        lr_schedule,
    )
    states = gather_states(2)
    batch = stack(states)
    mask = np.zeros((2, 56), dtype=bool)
    mask[:, 52:] = True  # only trump actions legal
    with th.no_grad():
        actions, _, _ = policy(batch, action_masks=mask)
    assert ((actions >= 52) & (actions < 56)).all()


def test_extractor_rejects_mismatched_features_dim() -> None:
    """A features_dim that disagrees with d_model must fail loudly at init.

    ``_pool`` always returns a ``d_model``-wide vector; silently accepting a
    different ``features_dim`` would make SB3 build policy/value heads sized
    for an input the forward pass never produces, failing opaquely deep
    inside the first training step instead of here.
    """
    with pytest.raises(ValueError, match="features_dim"):
        HokmTransformerExtractor(observation_space(), d_model=128, features_dim=256)
    # The matching value and the default (None) must both still work.
    HokmTransformerExtractor(observation_space(), d_model=128, features_dim=128)
    HokmTransformerExtractor(observation_space(), d_model=128, features_dim=None)
