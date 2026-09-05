"""Tokenization of Hokm observations for the transformer extractor.

The Dict observation becomes a card-token sequence plus context tokens:

- up to 13 hand tokens (order-free; the hand is a set),
- up to 4 current-trick tokens in play order,
- up to 13 recently played cards from completed tricks, ordered by
  descending card id (true play recency is not recoverable from the public
  observation vectors; the id ordering is a deterministic stand-in),
- 5 context tokens (trump, phase, scores, points) plus the acting seat
  carried as a context value.

Card tokens index a shared ``nn.Embedding(53, d_model)`` (52 cards + PAD).
Context tokens are not cards: each carries a bounded integer feature value
that the extractor maps through a dedicated per-slot embedding. Learned type
embeddings distinguish hand/trick/history/context, and learned positional
embeddings mark order-sensitive slots (trick play order, history position).
Hand tokens get no positional embedding — the hand is a set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch as th

from deephokm.cards import NUM_CARDS, PAD_TOKEN
from deephokm.env.spaces import Observation
from deephokm.rules.state import NUM_SEATS, TRICKS_PER_HAND

NUM_HAND_SLOTS = 13
NUM_TRICK_SLOTS = NUM_SEATS
NUM_HISTORY_SLOTS = 13
NUM_CONTEXT_TOKENS = 5
MAX_TOKENS = NUM_HAND_SLOTS + NUM_TRICK_SLOTS + NUM_HISTORY_SLOTS + NUM_CONTEXT_TOKENS

TYPE_HAND = 0
TYPE_TRICK = 1
TYPE_HISTORY = 2
TYPE_CONTEXT = 3
NUM_TYPES = 4

# Context token slots, in fixed order.
CTX_TRUMP = 0
CTX_PHASE = 1
CTX_SCORES = 2
CTX_POINTS = 3
CTX_SEAT = 4

# Bounded value ranges for the four context features.
TRUMP_VOCAB = 5  # undeclared or suit 0-3
PHASE_VOCAB = 3  # trump-call / card-play / pre-deal
SCORES_VOCAB = (TRICKS_PER_HAND + 1) * (TRICKS_PER_HAND + 1)  # tricks pair
POINTS_VOCAB = 8 * 8  # game-points pair
SEAT_VOCAB = NUM_SEATS + 1  # acting seat, or "none" pre-deal
CONTEXT_VOCABS = (TRUMP_VOCAB, PHASE_VOCAB, SCORES_VOCAB, POINTS_VOCAB, SEAT_VOCAB)

NUM_CARD_TOKENS = NUM_CARDS + 1  # 52 cards + PAD
NUM_POSITION_SLOTS = max(NUM_HAND_SLOTS, NUM_TRICK_SLOTS, NUM_HISTORY_SLOTS)


@dataclass(frozen=True, slots=True)
class TokenizedObservation:
    """Padded token batch for the transformer.

    Attributes:
        tokens: ``(batch, MAX_TOKENS)`` int64 card ids for card slots; the
            context slots hold that slot's value index instead (the extractor
            routes them through per-slot embeddings, never the card table).
        is_card: ``(batch, MAX_TOKENS)`` bool; True where ``tokens`` holds a
            card id.
        type_ids: ``(batch, MAX_TOKENS)`` int64 token-type ids.
        positions: ``(batch, MAX_TOKENS)`` int64 slot positions within a type
            (always 0 for hand tokens: the hand is a set).
        context_values: ``(batch, NUM_CONTEXT_TOKENS)`` int64 non-negative
            indices into each context slot's value range.
        padding_mask: ``(batch, MAX_TOKENS)`` bool; True where the token is
            real (context tokens are always real).
    """

    tokens: th.Tensor
    is_card: th.Tensor
    type_ids: th.Tensor
    positions: th.Tensor
    context_values: th.Tensor
    padding_mask: th.Tensor


def _set_ids(observation: Observation, key: Literal["hand", "seen"]) -> list[int]:
    """Return the card ids where the binary vector ``key`` is set."""
    vec = np.asarray(observation[key])
    ids: list[int] = np.flatnonzero(vec).astype(np.int64).tolist()
    return ids


def _trick_play_order(observation: Observation) -> list[int]:
    """Return the current trick's cards in play order.

    ``trick_play`` is indexed by seat; play order rotates clockwise from the
    leader. The leader is the played seat whose counterclockwise neighbor
    (``seat - 1``) has not played: the played set is a clockwise run starting
    at the leader, so only the leader's predecessor is unplayed.
    """
    trick_play = np.asarray(observation["trick_play"])
    played_mask = trick_play >= 0
    if not played_mask.any():
        return []
    leader = -1
    for seat in range(NUM_SEATS):
        if played_mask[seat] and not played_mask[(seat - 1) % NUM_SEATS]:
            leader = seat
            break
    assert leader >= 0, "no leader found among played seats"
    ordered: list[int] = []
    for offset in range(NUM_SEATS):
        seat = (leader + offset) % NUM_SEATS
        card = int(trick_play[seat])
        if card >= 0:
            ordered.append(card)
    return ordered


def _context_values(observation: Observation) -> list[int]:
    """Return the five context values as non-negative slot indices."""
    trump = np.asarray(observation["trump"])
    phase = np.asarray(observation["phase"])
    tricks = np.asarray(observation["tricks_won"])
    points = np.asarray(observation["game_points"])
    seat = np.asarray(observation["seat"])
    # Trump: suit id, or the last bucket when undeclared.
    trump_value = int(np.argmax(trump)) if trump.sum() else TRUMP_VOCAB - 1
    # Phase: one-hot index (0 trump-call, 1 card-play), or 2 pre-deal.
    phase_value = int(np.argmax(phase)) if phase.sum() else 2
    scores_index = int(tricks[0]) * (TRICKS_PER_HAND + 1) + int(tricks[1])
    points_index = int(points[0]) * 8 + int(points[1])
    seat_value = int(np.argmax(seat)) if seat.sum() else NUM_SEATS
    return [trump_value, phase_value, scores_index, points_index, seat_value]


def tokenize(observation: Observation) -> TokenizedObservation:
    """Tokenize a single observation into a batch of size 1.

    Uses the same fixed slot layout as :func:`tokenize_tensor_batch`: hand at
    columns 0-12, trick at 13-16, history at 17-29, context at 30-33.

    Args:
        observation: The acting player's observation dict.

    Returns:
        The padded token batch.
    """
    hand = sorted(_set_ids(observation, "hand"))[:NUM_HAND_SLOTS]
    trick = _trick_play_order(observation)[:NUM_TRICK_SLOTS]
    seen = _set_ids(observation, "seen")
    # History holds cards from completed tricks only; the current trick has
    # its own token group. Recency is approximated by descending card id.
    trick_set = set(trick)
    history = sorted(set(seen) - set(hand) - trick_set, reverse=True)[:NUM_HISTORY_SLOTS]
    context = _context_values(observation)

    tokens = [PAD_TOKEN] * MAX_TOKENS
    is_card = [False] * MAX_TOKENS
    type_ids = [TYPE_CONTEXT] * MAX_TOKENS
    positions = [0] * MAX_TOKENS
    padding = [False] * MAX_TOKENS

    for i, card in enumerate(hand):
        tokens[i] = card
        is_card[i] = True
        type_ids[i] = TYPE_HAND
        padding[i] = True
    for i, card in enumerate(trick):
        col = NUM_HAND_SLOTS + i
        tokens[col] = card
        is_card[col] = True
        type_ids[col] = TYPE_TRICK
        positions[col] = i
        padding[col] = True
    for i, card in enumerate(history):
        col = NUM_HAND_SLOTS + NUM_TRICK_SLOTS + i
        tokens[col] = card
        is_card[col] = True
        type_ids[col] = TYPE_HISTORY
        positions[col] = i
        padding[col] = True
    for slot, column in enumerate(CTX_COLUMNS):
        tokens[column] = context[slot]
        type_ids[column] = TYPE_CONTEXT
        positions[column] = slot
        padding[column] = True

    return TokenizedObservation(
        tokens=th.tensor([tokens], dtype=th.int64),
        is_card=th.tensor([is_card], dtype=th.bool),
        type_ids=th.tensor([type_ids], dtype=th.int64),
        positions=th.tensor([positions], dtype=th.int64),
        context_values=th.tensor([context], dtype=th.int64),
        padding_mask=th.tensor([padding], dtype=th.bool),
    )


# Fixed column layout: hand at 0-12, trick at 13-16, history at 17-29, and
# the context slots are always the last NUM_CONTEXT_TOKENS columns.
HAND_SLICE = slice(0, NUM_HAND_SLOTS)
TRICK_SLICE = slice(NUM_HAND_SLOTS, NUM_HAND_SLOTS + NUM_TRICK_SLOTS)
HISTORY_SLICE = slice(
    NUM_HAND_SLOTS + NUM_TRICK_SLOTS,
    NUM_HAND_SLOTS + NUM_TRICK_SLOTS + NUM_HISTORY_SLOTS,
)
CTX_COLUMNS = tuple(MAX_TOKENS - NUM_CONTEXT_TOKENS + slot for slot in range(NUM_CONTEXT_TOKENS))


def _compact_left(ids: th.Tensor, valid: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """Left-pack rows of ``ids`` keeping only ``valid`` entries, PAD the rest.

    Stable order: preserved entries keep their relative order.
    """
    n, width = ids.shape
    # Sort invalid entries to the end, keeping relative order among valid.
    order = th.argsort((~valid).to(th.int64), dim=1, stable=True)
    packed = th.gather(ids, 1, order)
    positions = th.arange(width, device=ids.device).unsqueeze(0).expand(n, -1)
    counts = valid.sum(dim=1, keepdim=True)
    result = th.where(positions < counts, packed, PAD_TOKEN)
    return result, counts.squeeze(1)


def _binary_to_ids(mask: th.Tensor, limit: int | None) -> tuple[th.Tensor, th.Tensor]:
    """Left-pack the set column indices of a binary mask, PAD-padded.

    Args:
        mask: ``(batch, 52)`` binary tensor.
        limit: Optional cap on kept indices per row.

    Returns:
        ``(ids, counts)`` where ids is ``(batch, limit or width)`` int64 with
        PAD_TOKEN in unused slots, ascending card order preserved.
    """
    width = mask.shape[1]
    present = mask > 0
    ids = th.arange(width, device=mask.device).unsqueeze(0).expand(mask.shape[0], -1)
    ids = th.where(present, ids, PAD_TOKEN)
    ids, counts = _compact_left(ids, present)
    if limit is not None:
        ids = ids[:, :limit]
        counts = counts.clamp(max=limit)
    return ids, counts


def _batch_card_groups(
    hand: th.Tensor, seen: th.Tensor, trick_play: th.Tensor
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Compute per-row hand/trick/history card id tensors.

    Returns:
        ``(hand_ids, n_hand, trick_ids, n_trick, history)`` where all id
        tensors are left-packed with PAD_TOKEN in unused slots.
    """
    device = hand.device
    # Hand: binary vector; the card id is the column index of each set bit.
    hand_ids, n_hand = _binary_to_ids(hand, NUM_HAND_SLOTS)

    # Trick: play order rotates from the leader. The played set is a
    # clockwise run from the leader, so the leader is the played seat whose
    # counterclockwise neighbor is unplayed (argmax picks it; exactly one
    # such seat exists whenever fewer than four seats have played).
    played = trick_play >= 0
    neighbor_unplayed = th.roll(~played, shifts=1, dims=1)
    leader = (played & neighbor_unplayed).to(th.int64).argmax(dim=1)
    offsets = th.arange(NUM_SEATS, device=device).view(1, -1)
    seats = (leader.view(-1, 1) + offsets) % NUM_SEATS
    seq = th.gather(trick_play, 1, seats)  # play order; -1 where absent
    seq_valid = seq >= 0
    trick_ids, n_trick = _compact_left(th.where(seq_valid, seq, PAD_TOKEN), seq_valid)

    # History: cards from completed tricks only (the current trick has its own
    # token group). Recency is approximated by descending card id.
    played_not_held = (seen > 0) & (hand == 0)
    # Knock out the current trick's cards: a card column is a trick card
    # when it equals some played entry. (scatter_ with clamped -1 indices is
    # order-dependent and can erase a genuine card 0.)
    card_columns = th.arange(seen.shape[1], device=device).view(1, -1)
    in_trick = (seq.unsqueeze(-1) == card_columns.unsqueeze(1)) & seq_valid.unsqueeze(-1)
    history_mask = played_not_held & ~in_trick.any(dim=1)
    older, _ = _binary_to_ids(history_mask, None)
    order = th.argsort(th.where(older == PAD_TOKEN, -1, older), dim=1, descending=True, stable=True)
    history = th.gather(older, 1, order)[:, :NUM_HISTORY_SLOTS]
    return hand_ids, n_hand, trick_ids, n_trick, history


def _batch_context(
    trump: th.Tensor,
    phase: th.Tensor,
    tricks_won: th.Tensor,
    game_points: th.Tensor,
    seat: th.Tensor,
) -> th.Tensor:
    """Compute the five context value indices for each row."""
    n = trump.shape[0]
    device = trump.device
    trump_value = th.where(trump.sum(dim=1) > 0, trump.argmax(dim=1), TRUMP_VOCAB - 1)
    phase_value = th.where(
        phase.sum(dim=1) > 0, phase.argmax(dim=1), th.full((n,), 2, device=device)
    )
    scores_index = tricks_won[:, 0] * (TRICKS_PER_HAND + 1) + tricks_won[:, 1]
    points_index = game_points[:, 0] * 8 + game_points[:, 1]
    seat_value = th.where(
        seat.sum(dim=1) > 0, seat.argmax(dim=1), th.full((n,), NUM_SEATS, device=device)
    )
    return th.stack([trump_value, phase_value, scores_index, points_index, seat_value], dim=1)


def tokenize_tensor_batch(observations: dict[str, th.Tensor]) -> TokenizedObservation:
    """Tokenize a dict of stacked observation tensors (the training hot path).

    Produces exactly the layout of :func:`tokenize` — hand cards ascending,
    trick cards in play order, history most-recent-first, four context
    values — without per-row Python work.

    Args:
        observations: Preprocessed observation dict of stacked tensors, as
            produced by SB3's Dict-observation preprocessing.

    Returns:
        The padded token batch.
    """
    hand = observations["hand"].to(th.int64)
    seen = observations["seen"].to(th.int64)
    trick_play = observations["trick_play"].to(th.int64)
    trump = observations["trump"].to(th.int64)
    phase = observations["phase"].to(th.int64)
    tricks_won = observations["tricks_won"].to(th.int64)
    game_points = observations["game_points"].to(th.int64)

    n = hand.shape[0]
    device = hand.device
    hand_ids, n_hand, trick_ids, n_trick, history = _batch_card_groups(hand, seen, trick_play)
    seat = observations["seat"].to(th.int64)
    context_values = _batch_context(trump, phase, tricks_won, game_points, seat)

    tokens = th.full((n, MAX_TOKENS), PAD_TOKEN, dtype=th.int64, device=device)
    is_card = th.zeros((n, MAX_TOKENS), dtype=th.bool, device=device)
    type_ids = th.full((n, MAX_TOKENS), TYPE_CONTEXT, dtype=th.int64, device=device)
    positions = th.zeros((n, MAX_TOKENS), dtype=th.int64, device=device)
    padding = th.zeros((n, MAX_TOKENS), dtype=th.bool, device=device)

    tokens[:, HAND_SLICE] = hand_ids
    hand_valid = th.arange(NUM_HAND_SLOTS, device=device).unsqueeze(0) < n_hand.unsqueeze(1)
    is_card[:, HAND_SLICE] = hand_valid
    type_ids[:, HAND_SLICE] = th.where(hand_valid, TYPE_HAND, TYPE_CONTEXT)

    tokens[:, TRICK_SLICE] = trick_ids[:, :NUM_TRICK_SLOTS]
    trick_valid = th.arange(NUM_TRICK_SLOTS, device=device).unsqueeze(0) < n_trick.unsqueeze(1)
    is_card[:, TRICK_SLICE] = trick_valid
    type_ids[:, TRICK_SLICE] = th.where(trick_valid, TYPE_TRICK, TYPE_CONTEXT)
    positions[:, TRICK_SLICE] = th.where(trick_valid, th.arange(NUM_TRICK_SLOTS, device=device), 0)

    hist_valid = history != PAD_TOKEN
    tokens[:, HISTORY_SLICE] = history
    is_card[:, HISTORY_SLICE] = hist_valid
    type_ids[:, HISTORY_SLICE] = th.where(hist_valid, TYPE_HISTORY, TYPE_CONTEXT)
    positions[:, HISTORY_SLICE] = th.where(
        hist_valid, th.arange(NUM_HISTORY_SLOTS, device=device), 0
    )

    for slot, column in enumerate(CTX_COLUMNS):
        tokens[:, column] = context_values[:, slot]
        type_ids[:, column] = TYPE_CONTEXT
        positions[:, column] = slot
        padding[:, column] = True

    padding[:, HAND_SLICE] = is_card[:, HAND_SLICE]
    padding[:, TRICK_SLICE] = is_card[:, TRICK_SLICE]
    padding[:, HISTORY_SLICE] = is_card[:, HISTORY_SLICE]

    return TokenizedObservation(
        tokens=tokens,
        is_card=is_card,
        type_ids=type_ids,
        positions=positions,
        context_values=context_values,
        padding_mask=padding,
    )
