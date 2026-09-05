"""Tests for the transformer feature extractor and policy wiring."""

from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

import deephokm.nn.extractor as extractor_mod
from deephokm.env import HokmEnv
from deephokm.env.spaces import empty_observation, observation_space
from deephokm.nn.extractor import HokmTransformerExtractor
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.nn.tokenizer import (
    tokenize,
    tokenize_tensor_batch,
)
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.state import NUM_SEATS

MAX_PARAMS = 2_000_000


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
    assert out.shape == (8, 128)
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
    assert out.shape == (2, 128)
    assert th.isfinite(out).all()


def test_forward_edge_case_empty_trick() -> None:
    ext = HokmTransformerExtractor(observation_space())
    states = gather_states(30)
    no_trick = [s for s in states if (s["trick_play"] < 0).all()]
    assert no_trick, "no lead state collected"
    out = ext(stack(no_trick))
    assert out.shape == (len(no_trick), 128)
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
    assert out.shape == (2, 128)
    assert th.isfinite(out).all()


def test_parameter_count_under_2m() -> None:
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
                tokens = dataclasses.replace(tokens, tokens=th.where(pad, noise, tokens.tokens))
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
            for field in ("tokens", "is_card", "type_ids", "padding_mask", "positions"):
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
    """Hand at 0-12, trick at 13-16, history at 17-29, context at 30-34."""
    states = gather_states(6)
    for state in states:
        t = tokenize(state)
        tokens = t.tokens[0].tolist()
        hand = sorted(int(c) for c in np.flatnonzero(state["hand"]))
        assert tokens[: len(hand)] == hand
        assert tokens[13 + len(_played_cards(state)) : 17] == [52] * (4 - len(_played_cards(state)))
        # Context slots always present.
        assert t.padding_mask[0][30:35].all()


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


def test_trick_order_wrapping_case() -> None:
    """Direct check of the wrap-around leader case."""
    obs = empty_observation()
    obs["trick_play"] = np.array([32, -1, 33, 27])  # leader=2, wraps past seat 0
    obs["hand"][10] = 1
    t = tokenize(obs)
    assert t.tokens[0, 13:17].tolist() == [33, 27, 32, 52]


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
            "type_ids",
            "positions",
            "context_values",
            "padding_mask",
        ):
            a = getattr(single, field)[0]
            b = getattr(vectorized, field)[i]
            assert th.equal(a, b), f"{field} mismatch at sample {i}"


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
    assert th.allclose(action_rows, th.full_like(action_rows, 0.01), atol=1e-3)
    assert th.allclose(value_rows, th.full_like(value_rows, 1.0), atol=1e-3)


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
