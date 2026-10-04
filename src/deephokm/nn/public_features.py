"""Public-information feature extensions shared by training and serving."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

NUM_SUITS, NUM_RANKS, NUM_CARDS, NUM_SEATS = 4, 13, 52, 4
PARTNER_ROLE = 2
FEATURE_MODES = (
    "baseline",
    "voids",
    "voids_zero",
    "trick_context",
    "trick_context_zero",
    "public_strength",
    "public_strength_zero",
)

CONTEXT_NAMES = (
    "led_suit",
    "winning_card",
    "winner_self",
    "winner_next",
    "winner_partner",
    "winner_previous",
    "partner_winning",
    "position_first",
    "position_second",
    "position_third",
    "position_fourth",
    "next_still_to_act",
    "partner_still_to_act",
    "previous_still_to_act",
    "beats_winner",
    "overtakes_partner",
    "spends_trump",
    "creates_own_void",
)


def public_voids(obs: Mapping[str, object]) -> np.ndarray:
    """Proven suit voids, indexed by relative seat and suit; public information only.

    History contains complete tricks in reverse play order. The current
    trick ends immediately before the acting seat, including across seat 3/0.
    Never infer a leader by sorting absolute seat IDs. Rebuilding per observation
    naturally resets memory between hands and works on shuffled training rows.
    """
    history = np.asarray(obs["history"])
    roles = np.asarray(obs["history_role"])
    if history.shape != roles.shape or history.ndim != 1:
        raise ValueError("history and roles must be matching vectors")
    valid = history >= 0
    count = int(valid.sum())
    if count % NUM_SEATS or not np.array_equal(valid, np.arange(len(history)) < count):
        raise ValueError("history must contain complete, prefix-packed tricks")
    if (
        np.any(history[valid] >= NUM_CARDS)
        or np.any(roles[valid] < 0)
        or np.any(roles[valid] >= NUM_SEATS)
    ):
        raise ValueError("invalid public history card or seat")
    voids = np.zeros((NUM_SEATS, NUM_SUITS), dtype=np.float32)
    for start in range(0, count, NUM_SEATS):
        cards, seats = history[start : start + NUM_SEATS], roles[start : start + NUM_SEATS]
        leader = int(seats[-1])
        if not np.array_equal(
            seats, [(leader + NUM_SEATS - 1 - i) % NUM_SEATS for i in range(NUM_SEATS)]
        ):
            raise ValueError("history seats are not in reverse clockwise order")
        led = int(cards[-1]) // NUM_RANKS
        for card, role in zip(cards[:-1], seats[:-1], strict=True):
            if int(card) // NUM_RANKS != led:
                voids[int(role), led] = 1
    _mark_current_voids(obs, voids)
    return voids


def current_trick(obs: Mapping[str, object]) -> tuple[int, list[int], np.ndarray]:
    """Validate public play order and return actor, chronological seats and cards."""
    table = np.asarray(obs["trick_play"])
    if (
        table.shape != (NUM_SEATS,)
        or not np.issubdtype(table.dtype, np.integer)
        or np.any((table < -1) | (table >= NUM_CARDS))
    ):
        raise ValueError("invalid current trick")
    seat_vector = np.asarray(obs["seat"])
    if (
        seat_vector.shape != (NUM_SEATS,)
        or not np.isin(seat_vector, (0, 1)).all()
        or seat_vector.sum() != 1
    ):
        raise ValueError("acting seat must be one-hot")
    seat = int(seat_vector.argmax())
    occupied = np.flatnonzero(table >= 0)
    leader = (seat - len(occupied)) % NUM_SEATS
    expected = [(leader + i) % NUM_SEATS for i in range(len(occupied))]
    if len(occupied) == NUM_SEATS or set(expected) != set(occupied):
        raise ValueError("current trick must immediately precede acting seat")
    if len(np.unique(table[occupied])) != len(occupied):
        raise ValueError("duplicate card in current trick")
    return seat, expected, table


def _mark_current_voids(obs: Mapping[str, object], voids: np.ndarray) -> None:
    # Preserve empty/trump-call observations accepted by legacy void features.
    if not np.any(np.asarray(obs["trick_play"]) >= 0):
        return
    seat, expected, table = current_trick(obs)
    occupied = expected
    if len(occupied):
        led = int(table[expected[0]]) // NUM_RANKS
        for played_seat in expected[1:]:
            if int(table[played_seat]) // NUM_RANKS != led:
                voids[(played_seat - seat) % NUM_SEATS, led] = 1


def trick_context_planes(obs: Mapping[str, object], legal_cards: np.ndarray) -> np.ndarray:
    """Explicit tactics derived solely from the player's observation and mask.

    A current lead is not a guarantee of winning the completed trick. Future
    actors are encoded, never their hidden cards. All action flags are masked.
    Trump selection has no card-play context and therefore returns zeros.
    """
    extra = np.zeros((len(CONTEXT_NAMES), NUM_SUITS, NUM_RANKS), dtype=np.float32)
    phase = np.asarray(obs["phase"])
    if phase.shape != (2,) or not np.isin(phase, (0, 1)).all() or phase.sum() != 1:
        raise ValueError("phase must be one-hot")
    if phase[0]:
        return extra
    actor, seats, table = current_trick(obs)
    trump_vector = np.asarray(obs["trump"])
    if (
        trump_vector.shape != (NUM_SUITS,)
        or not np.isin(trump_vector, (0, 1)).all()
        or trump_vector.sum() != 1
    ):
        raise ValueError("card play requires a one-hot trump")
    trump = int(trump_vector.argmax())
    hand = np.asarray(obs["hand"]).reshape(NUM_SUITS, NUM_RANKS)
    legal = np.asarray(legal_cards).reshape(NUM_SUITS, NUM_RANKS)
    if not np.isin(legal, (0, 1)).all() or not np.isin(hand, (0, 1)).all():
        raise ValueError("hand and legality must be binary")
    if np.any((legal > 0) & (hand == 0)) or np.any(hand.ravel()[table[seats]]):
        raise ValueError("legal cards and table must be consistent with own hand")
    extra[7 + len(seats)] = 1
    for relative in range(1, NUM_SEATS - len(seats)):
        extra[10 + relative] = 1
    extra[16, trump] = legal[trump]
    extra[17] = legal * (hand.sum(1) == 1)[:, None]
    if not seats:
        return extra
    led = int(table[seats[0]]) // NUM_RANKS
    extra[0, led] = 1

    def strength(card: int) -> tuple[int, int]:
        suit, rank = divmod(int(card), NUM_RANKS)
        return (2 if suit == trump else 1 if suit == led else 0, rank)

    winner_seat = max(seats, key=lambda seat: strength(int(table[seat])))
    winner = int(table[winner_seat])
    relative_winner = (winner_seat - actor) % NUM_SEATS
    extra[1, winner // NUM_RANKS, winner % NUM_RANKS] = 1
    extra[2 + relative_winner] = 1
    partner_winning = relative_winner == PARTNER_ROLE
    extra[6] = float(partner_winning)
    for card in np.flatnonzero(legal.ravel()):
        if strength(int(card)) > strength(winner):
            suit, rank = divmod(int(card), NUM_RANKS)
            extra[14, suit, rank] = 1
            extra[15, suit, rank] = float(partner_winning)
    return extra


STRENGTH_NAMES = (
    "higher_unseen_count",
    "no_higher_unseen",
    "highest_remaining",
    "own_suit_length",
    "unseen_suit_length",
    "unseen_trumps",
    "own_trumps",
    "own_rank_sequence",
    "highest_own_rank",
    "own_suit_rank_sum",
    "higher_unseen_above_own_top",
    "opponents_proven_void",
)


def strength_planes(obs: Mapping[str, object]) -> np.ndarray:
    """Public rank/suit strength, without guessing actual card ownership.

    'No higher unseen' means no opponent holds a higher card of this suit;
    it is not a guaranteed trick win (trumps and later play still matter).
    'Highest remaining' also accounts for higher cards in our own hand.
    """
    extra = np.zeros((len(STRENGTH_NAMES), NUM_SUITS, NUM_RANKS), dtype=np.float32)
    if np.asarray(obs["phase"])[0]:
        return extra
    hand = np.asarray(obs["hand"]).reshape(NUM_SUITS, NUM_RANKS)
    seen = np.asarray(obs["seen"]).reshape(NUM_SUITS, NUM_RANKS)
    if not np.isin(seen, (0, 1)).all() or np.any(hand > seen):
        raise ValueError("seen must be binary and include our own hand")
    unseen = 1 - seen
    remaining = np.maximum(unseen, hand)
    higher_unseen = np.cumsum(unseen[:, ::-1], axis=1)[:, ::-1] - unseen
    higher_remaining = np.cumsum(remaining[:, ::-1], axis=1)[:, ::-1] - remaining
    extra[0] = hand * higher_unseen / NUM_RANKS
    extra[1] = hand * (higher_unseen == 0)
    extra[2] = hand * (higher_remaining == 0)
    extra[3] = hand.sum(1)[:, None] / NUM_RANKS
    extra[4] = unseen.sum(1)[:, None] / NUM_RANKS
    trump = int(np.asarray(obs["trump"]).argmax())
    extra[5] = unseen[trump].sum() / NUM_RANKS
    extra[6] = hand[trump].sum() / NUM_RANKS
    for suit in range(NUM_SUITS):
        held = np.flatnonzero(hand[suit])
        if not len(held):
            continue
        highest = int(held[-1])
        extra[8, suit] = highest / (NUM_RANKS - 1)
        extra[9, suit] = held.sum() / sum(range(NUM_RANKS))
        extra[10, suit] = higher_unseen[suit, highest] / NUM_RANKS
        run = 0
        for rank in range(NUM_RANKS - 1, -1, -1):
            run = run + 1 if hand[suit, rank] else 0
            extra[7, suit, rank] = run / NUM_RANKS
    extra[11] = public_voids(obs)[[1, 3]].sum(0)[:, None] / 2
    return extra


def feature_plane_count(mode: str) -> int:
    if mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature mode: {mode}")
    if mode == "baseline":
        return 14
    if mode.startswith("voids"):
        return 18
    return 48 if mode.startswith("public_strength") else 36


def append_void_planes(
    planes: np.ndarray, obs: Mapping[str, object], feature_mode: str
) -> np.ndarray:
    """Share the exact same feature transform between batch and live inference."""
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature mode: {feature_mode}")
    if feature_mode == "baseline":
        return planes
    voids = public_voids(obs) if feature_mode != "voids_zero" else np.zeros((4, NUM_SUITS))
    extra = np.broadcast_to(voids[:, :, None], (4, NUM_SUITS, NUM_RANKS))
    result = np.concatenate((planes, extra.astype(np.float32)), axis=0)
    if feature_mode.startswith(("trick_context", "public_strength")):
        context = trick_context_planes(obs, planes[13])
        if feature_mode == "trick_context_zero":
            context = np.zeros_like(context)
        result = np.concatenate((result, context), axis=0)
    if feature_mode.startswith("public_strength"):
        strength = strength_planes(obs)
        if feature_mode == "public_strength_zero":
            strength = np.zeros_like(strength)
        result = np.concatenate((result, strength), axis=0)
    return result
