"""Tokenization of Hokm observations for the transformer extractor.

The Dict observation becomes a card-token sequence plus context tokens:

- up to 13 hand tokens (order-free; the hand is a set),
- up to 4 current-trick tokens in play order,
- up to 48 history tokens: every card from this hand's completed tricks in
  reverse play order (most recent first), straight from the observation's
  ``history`` slot,
- 5 context tokens (trump, phase, tricks, points, seat).

The history group deliberately spans the whole hand rather than the 13 most
recent plays: which cards are gone is the central read in a trick-taking
game, and truncating the group hides most of it. Forty-eight extra tokens
cost nothing at this model size.

Card tokens are factorized into a rank (``card_id % 13``, the token's
``ranks`` entry) and a trump flag (``is_trump``: the card's suit matches the
declared trump suit; always False while the trump is still all-zero) that the
extractor maps through a shared rank embedding and a two-row trump-flag
embedding, plus a canonical suit slot (``suit_slots``) that preserves which
cards share a suit without revealing which suit it is. Context tokens are not
cards: each carries a bounded integer
feature value that the extractor maps through a dedicated per-slot embedding.
Learned type
embeddings distinguish hand/trick/history/context, and learned positional
embeddings mark order-sensitive slots (trick play order, history position).
Hand tokens get no positional embedding — the hand is a set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch as th

from deephokm.cards import NUM_RANKS, NUM_SUITS, PAD_TOKEN
from deephokm.env.spaces import HISTORY_SLOTS, Observation
from deephokm.rules.state import NUM_SEATS, TRICKS_PER_HAND

NUM_HAND_SLOTS = 13
NUM_TRICK_SLOTS = NUM_SEATS
NUM_HISTORY_SLOTS = HISTORY_SLOTS
NUM_CONTEXT_TOKENS = 5
MAX_TOKENS = NUM_HAND_SLOTS + NUM_TRICK_SLOTS + NUM_HISTORY_SLOTS + NUM_CONTEXT_TOKENS

TYPE_HAND = 0
TYPE_TRICK = 1
TYPE_HISTORY = 2
TYPE_CONTEXT = 3
NUM_TYPES = 4

# Relative-seat role of a card token: 0 self, 1 the next seat clockwise, 2
# partner, 3 the previous seat clockwise; ROLE_NA for tokens with no owning
# seat (the context slots). Hand tokens are always role 0 (they are always
# the acting seat's own cards); trick/history tokens carry the relative role
# of whoever actually played that card, letting the network condition on
# partner-vs-opponent play directly instead of inferring it from card-id
# patterns alone.
ROLE_NA = NUM_SEATS
ROLE_VOCAB = NUM_SEATS + 1

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
        ranks: ``(batch, MAX_TOKENS)`` int64 rank index (``card_id % 13``)
            for card slots; 0 elsewhere (the extractor only indexes its rank
            embedding on card slots).
        is_trump: ``(batch, MAX_TOKENS)`` bool; True for card slots whose
            card's suit matches the declared trump suit; False while the
            trump is still all-zero.
        suit_slots: ``(batch, MAX_TOKENS)`` int64 canonical suit slot
            (``0..3``) for card slots; 0 elsewhere. The slot is the card's
            suit's rank in a canonical, relabeling-invariant ordering of the
            four suits (see :func:`_canonical_suit_slots`), so two cards share
            a slot exactly when they share a suit — the same-suit partition
            follow-suit reasoning needs — without the absolute suit label
            ever entering the input.
        type_ids: ``(batch, MAX_TOKENS)`` int64 token-type ids.
        positions: ``(batch, MAX_TOKENS)`` int64 slot positions within a type
            (always 0 for hand tokens: the hand is a set).
        roles: ``(batch, MAX_TOKENS)`` int64 relative-seat role per token (see
            :data:`ROLE_NA`): 0 for hand tokens (always the acting seat's own
            cards), the played seat's relative role for trick/history
            tokens, ``ROLE_NA`` for context tokens and unused padding.
        context_values: ``(batch, NUM_CONTEXT_TOKENS)`` int64 non-negative
            indices into each context slot's value range.
        padding_mask: ``(batch, MAX_TOKENS)`` bool; True where the token is
            real (context tokens are always real).
    """

    tokens: th.Tensor
    is_card: th.Tensor
    ranks: th.Tensor
    is_trump: th.Tensor
    suit_slots: th.Tensor
    type_ids: th.Tensor
    positions: th.Tensor
    roles: th.Tensor
    context_values: th.Tensor
    padding_mask: th.Tensor


def _hand_ids(observation: Observation) -> list[int]:
    """Return the card ids held in the acting player's hand."""
    ids: list[int] = np.flatnonzero(np.asarray(observation["hand"])).astype(np.int64).tolist()
    return ids


def _trick_play_order(observation: Observation) -> list[tuple[int, int]]:
    """Return the current trick's ``(card, role)`` pairs in play order.

    ``trick_play`` is indexed by seat; play order rotates clockwise from the
    leader. The leader is the played seat whose counterclockwise neighbor
    (``seat - 1``) has not played: the played set is a clockwise run starting
    at the leader, so only the leader's predecessor is unplayed. ``role`` is
    the played seat relative to the acting seat (see :data:`ROLE_NA`'s
    docstring for the convention).
    """
    trick_play = np.asarray(observation["trick_play"])
    seat_onehot = np.asarray(observation["seat"])
    acting_seat = int(np.argmax(seat_onehot)) if seat_onehot.sum() else 0
    played_mask = trick_play >= 0
    if not played_mask.any():
        return []
    if played_mask.all():
        # A completed trick never appears in a learner observation (HokmEnv
        # resets it before handing control over); emit zero trick tokens so
        # this path agrees with the batch tokenizer on synthetic inputs.
        return []
    leader = -1
    for seat in range(NUM_SEATS):
        if played_mask[seat] and not played_mask[(seat - 1) % NUM_SEATS]:
            leader = seat
            break
    assert leader >= 0, "no leader found among played seats"
    ordered: list[tuple[int, int]] = []
    for offset in range(NUM_SEATS):
        seat = (leader + offset) % NUM_SEATS
        card = int(trick_play[seat])
        if card >= 0:
            ordered.append((card, (seat - acting_seat) % NUM_SEATS))
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


def _rank_trump_flags(card_ids: list[int], trump_suit: int) -> tuple[list[int], list[bool]]:
    """Per-card rank indices and trump flags for a list of card ids.

    ``trump_suit`` is the declared trump suit id, or -1 while the trump is
    still all-zero (in which case every flag is False).
    """
    ranks = [c % NUM_RANKS for c in card_ids]
    is_trump = [trump_suit >= 0 and c // NUM_RANKS == trump_suit for c in card_ids]
    return ranks, is_trump


@dataclass(slots=True)
class _RowBuffers:
    """The eight per-row token lists that :func:`tokenize` fills."""

    tokens: list[int]
    is_card: list[bool]
    ranks: list[int]
    is_trump: list[bool]
    type_ids: list[int]
    positions: list[int]
    roles: list[int]
    padding: list[bool]


@dataclass(slots=True)
class _BatchBuffers:
    """The eight batched token tensors that :func:`tokenize_tensor_batch` fills."""

    tokens: th.Tensor
    is_card: th.Tensor
    ranks: th.Tensor
    is_trump: th.Tensor
    type_ids: th.Tensor
    positions: th.Tensor
    roles: th.Tensor
    padding: th.Tensor


def _fill_card_group(
    buffers: _RowBuffers,
    *,
    offset: int,
    cards: list[int],
    group_ranks: list[int],
    group_trumps: list[bool],
    type_id: int,
    with_positions: bool,
    roles_values: list[int] | None,
) -> None:
    """Write one hand/trick/history card group into the flat token lists.

    Args:
        buffers: The per-row token lists (modified in place).
        offset: First column of this group.
        cards: Card ids for this group, in slot order.
        group_ranks: Precomputed rank index per card.
        group_trumps: Precomputed trump flag per card.
        type_id: Token-type id for this group (TYPE_HAND/TYPE_TRICK/TYPE_HISTORY).
        with_positions: Write the slot index into ``positions`` (trick and
            history are order-sensitive; the hand is a set, so no position).
        roles_values: One role per card, or None for the fixed role 0 (hand).
    """
    for i, card in enumerate(cards):
        col = offset + i
        buffers.tokens[col] = card
        buffers.is_card[col] = True
        buffers.ranks[col] = group_ranks[i]
        buffers.is_trump[col] = group_trumps[i]
        buffers.type_ids[col] = type_id
        if with_positions:
            buffers.positions[col] = i
        if roles_values is not None:
            buffers.roles[col] = roles_values[i]
        else:
            buffers.roles[col] = 0
        buffers.padding[col] = True


def _canonical_suit_slots(
    tokens: th.Tensor,
    is_card: th.Tensor,
    *,
    hand: th.Tensor,
    seen: th.Tensor,
    trick: th.Tensor,
    trump: th.Tensor,
) -> th.Tensor:
    """Per-token canonical suit slot, invariant to relabeling the suits.

    Collapsing a card to rank plus a trump flag throws away too much: two
    non-trump cards of different suits become indistinguishable, so the
    network cannot tell whether a card in hand shares a suit with the led
    card, which is what follow-suit legality and void reasoning are built
    on. What the rules actually license is discarding the suits' *names*,
    not the partition they induce.

    So each suit is scored by a key derived only from suit-symmetric
    observables — is it trump, which of its ranks are in hand, which are
    seen, which are on the table — and the suits are ordered by that key.
    A suit's slot is its position in that order, so the encoding carries
    "these two cards are the same suit" while never seeing which suit that
    is. Trump sorts first (its key's top bit), giving it slot 0 whenever it
    is declared.

    Two suits with byte-identical observables tie; the tie breaks on the
    lower suit id. Such suits are interchangeable in everything the
    observation reveals, so the choice cannot change decision quality, but
    it does mean strict bit-invariance holds only for untied suits.

    Args:
        tokens: ``(batch, MAX_TOKENS)`` int64 token values (card ids where
            ``is_card``).
        is_card: ``(batch, MAX_TOKENS)`` bool card-slot mask.
        hand: ``(batch, 52)`` int64 own-hand indicator.
        seen: ``(batch, 52)`` int64 own hand plus every card played.
        trick: ``(batch, 52)`` int64 cards currently on the table.
        trump: ``(batch, 4)`` int64 one-hot trump suit; all zeros undeclared.

    Returns:
        ``(batch, MAX_TOKENS)`` int64 slot in ``0..3`` on card slots, 0
        elsewhere.
    """
    n = hand.shape[0]
    device = hand.device
    bits = (1 << th.arange(NUM_RANKS, device=device, dtype=th.int64)).view(1, 1, NUM_RANKS)
    # Per-suit 13-bit signatures over the three suit-symmetric card sets.
    by_suit = [x.view(n, NUM_SUITS, NUM_RANKS).to(th.int64) for x in (hand, seen, trick)]
    hand_sig, seen_sig, trick_sig = ((x * bits).sum(dim=2) for x in by_suit)
    declared = (trump.sum(dim=1) > 0).view(-1, 1)
    is_trump_suit = declared & (
        th.arange(NUM_SUITS, device=device).view(1, -1) == trump.argmax(dim=1, keepdim=True)
    )
    key = (is_trump_suit.to(th.int64) << 39) | (hand_sig << 26) | (seen_sig << 13) | trick_sig
    # Descending on the key; ties fall to the lower suit id. Shifting the key
    # left by two and subtracting the suit id folds both into one sort value.
    sort_value = (key << 2) - th.arange(NUM_SUITS, device=device).view(1, -1)
    order = th.argsort(sort_value, dim=1, descending=True, stable=True)
    slot_of_suit = th.empty((n, NUM_SUITS), dtype=th.int64, device=device)
    slot_of_suit.scatter_(
        1, order, th.arange(NUM_SUITS, device=device).view(1, -1).expand(n, NUM_SUITS)
    )
    card_suit = th.where(is_card, tokens // NUM_RANKS, th.zeros_like(tokens))
    return th.where(is_card, th.gather(slot_of_suit, 1, card_suit), th.zeros_like(tokens))


def tokenize(observation: Observation) -> TokenizedObservation:
    """Tokenize a single observation into a batch of size 1.

    Uses the same fixed slot layout as :func:`tokenize_tensor_batch`: hand
    first, then the current trick, then history, then the context slots.

    Args:
        observation: The acting player's observation dict.

    Returns:
        The padded token batch.
    """
    hand = sorted(_hand_ids(observation))[:NUM_HAND_SLOTS]
    trump_vec = np.asarray(observation["trump"])
    trump_suit = int(np.argmax(trump_vec)) if trump_vec.sum() else -1
    trick = _trick_play_order(observation)[:NUM_TRICK_SLOTS]
    # History arrives already ordered (most recent completed-trick play
    # first) and -1 padded; history_role is padded in lockstep.
    raw_history = np.asarray(observation["history"])
    raw_history_role = np.asarray(observation["history_role"])
    history_valid = raw_history >= 0
    history = [int(c) for c in raw_history[history_valid][:NUM_HISTORY_SLOTS]]
    history_roles = [int(r) for r in raw_history_role[history_valid][:NUM_HISTORY_SLOTS]]
    context = _context_values(observation)

    buffers = _RowBuffers(
        tokens=[PAD_TOKEN] * MAX_TOKENS,
        is_card=[False] * MAX_TOKENS,
        ranks=[0] * MAX_TOKENS,
        is_trump=[False] * MAX_TOKENS,
        type_ids=[TYPE_CONTEXT] * MAX_TOKENS,
        positions=[0] * MAX_TOKENS,
        roles=[ROLE_NA] * MAX_TOKENS,
        padding=[False] * MAX_TOKENS,
    )

    hand_ranks, hand_trumps = _rank_trump_flags(hand, trump_suit)
    _fill_card_group(
        buffers,
        offset=0,
        cards=hand,
        group_ranks=hand_ranks,
        group_trumps=hand_trumps,
        type_id=TYPE_HAND,
        with_positions=False,
        roles_values=None,
    )
    trick_cards = [card for card, _ in trick]
    trick_ranks, trick_trumps = _rank_trump_flags(trick_cards, trump_suit)
    _fill_card_group(
        buffers,
        offset=NUM_HAND_SLOTS,
        cards=trick_cards,
        group_ranks=trick_ranks,
        group_trumps=trick_trumps,
        type_id=TYPE_TRICK,
        with_positions=True,
        roles_values=[role for _, role in trick],
    )
    hist_ranks, hist_trumps = _rank_trump_flags(history, trump_suit)
    _fill_card_group(
        buffers,
        offset=NUM_HAND_SLOTS + NUM_TRICK_SLOTS,
        cards=history,
        group_ranks=hist_ranks,
        group_trumps=hist_trumps,
        type_id=TYPE_HISTORY,
        with_positions=True,
        roles_values=history_roles,
    )
    for slot, column in enumerate(CTX_COLUMNS):
        buffers.tokens[column] = context[slot]
        buffers.type_ids[column] = TYPE_CONTEXT
        buffers.positions[column] = slot
        buffers.padding[column] = True

    tokens_t = th.tensor([buffers.tokens], dtype=th.int64)
    is_card_t = th.tensor([buffers.is_card], dtype=th.bool)
    suit_slots = _canonical_suit_slots(
        tokens_t,
        is_card_t,
        hand=th.as_tensor(np.asarray(observation["hand"])[None], dtype=th.int64),
        seen=th.as_tensor(np.asarray(observation["seen"])[None], dtype=th.int64),
        trick=th.as_tensor(np.asarray(observation["trick"])[None], dtype=th.int64),
        trump=th.as_tensor(np.asarray(trump_vec)[None], dtype=th.int64),
    )

    return TokenizedObservation(
        tokens=tokens_t,
        is_card=is_card_t,
        ranks=th.tensor([buffers.ranks], dtype=th.int64),
        is_trump=th.tensor([buffers.is_trump], dtype=th.bool),
        suit_slots=suit_slots,
        type_ids=th.tensor([buffers.type_ids], dtype=th.int64),
        positions=th.tensor([buffers.positions], dtype=th.int64),
        roles=th.tensor([buffers.roles], dtype=th.int64),
        context_values=th.tensor([context], dtype=th.int64),
        padding_mask=th.tensor([buffers.padding], dtype=th.bool),
    )


# Fixed column layout: hand, then the current trick, then history; the
# context slots are always the last NUM_CONTEXT_TOKENS columns.
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


def _is_trump(card_ids: th.Tensor, trump: th.Tensor, valid: th.Tensor) -> th.Tensor:
    """Per-card trump flags: the card's suit matches the declared trump suit.

    Args:
        card_ids: ``(batch, width)`` int64 card ids (PAD in invalid slots).
        trump: ``(batch, 4)`` int64 one-hot trump suit; all zeros while
            undeclared.
        valid: ``(batch, width)`` bool; False where the slot holds PAD.

    Returns:
        ``(batch, width)`` bool.
    """
    declared = trump.sum(dim=1) > 0
    suit = card_ids // NUM_RANKS
    return valid & (suit == trump.argmax(dim=1, keepdim=True)) & declared.view(-1, 1)


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
    hand: th.Tensor,
    trick_play: th.Tensor,
    history: th.Tensor,
    history_role: th.Tensor,
    seat: th.Tensor,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Compute per-row hand/trick/history card id (and role) tensors.

    Returns:
        ``(hand_ids, n_hand, trick_ids, trick_roles, n_trick, history_ids,
        history_roles)`` where all id/role tensors are left-packed with
        ``PAD_TOKEN``/``ROLE_NA`` respectively in unused slots.
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
    # A fully-played trick never appears in a learner observation; zero it so
    # this path agrees with the single-row tokenizer (which returns no trick
    # tokens for the same synthetic input) instead of emitting seat order.
    seq_valid = seq_valid & ~played.all(dim=1, keepdim=True)
    trick_ids, n_trick = _compact_left(th.where(seq_valid, seq, PAD_TOKEN), seq_valid)
    # _compact_left's packing permutation is a deterministic function of
    # `valid` alone (stable-sorts on ~valid, never on the ids' own values),
    # so packing the role tensor with the identical `seq_valid` mask lands
    # each role in the same column as its card in trick_ids.
    acting_seat = seat.argmax(dim=1)
    trick_role_raw = (seats - acting_seat.view(-1, 1)) % NUM_SEATS
    trick_roles, _ = _compact_left(trick_role_raw, seq_valid)

    # History arrives already ordered (most recent completed-trick play
    # first) with -1 padding; only the padding value has to be remapped.
    hist_valid = history >= 0
    history_ids = th.where(hist_valid, history, PAD_TOKEN)[:, :NUM_HISTORY_SLOTS]
    history_roles = th.where(hist_valid, history_role, ROLE_NA)[:, :NUM_HISTORY_SLOTS]
    return hand_ids, n_hand, trick_ids, trick_roles, n_trick, history_ids, history_roles


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


def _fill_batch_card_group(
    buffers: _BatchBuffers,
    *,
    col_slice: slice,
    card_ids: th.Tensor,
    valid: th.Tensor,
    trump: th.Tensor,
    type_id: int,
    positions_values: th.Tensor | None,
    roles_values: th.Tensor | None,
) -> None:
    """Write one hand/trick/history card group into the batched token tensors.

    Args:
        buffers: The batched token tensors (modified in place).
        col_slice: Column range of this group.
        card_ids: Left-packed card ids for this group, ``(batch, group width)``.
        valid: ``(batch, group width)`` bool; True where the slot holds a card.
        trump: ``(batch, 4)`` one-hot trump suit.
        type_id: Token-type id for this group (TYPE_HAND/TYPE_TRICK/TYPE_HISTORY).
        positions_values: Position index per slot for this group's width, or
            None for a group with no positions (the hand is a set).
        roles_values: Per-card roles, or None for the fixed role 0 (hand).
    """
    buffers.tokens[:, col_slice] = card_ids
    buffers.is_card[:, col_slice] = valid
    buffers.ranks[:, col_slice] = th.where(valid, card_ids % NUM_RANKS, 0)
    buffers.is_trump[:, col_slice] = _is_trump(card_ids, trump, valid)
    buffers.type_ids[:, col_slice] = th.where(valid, type_id, TYPE_CONTEXT)
    if positions_values is not None:
        buffers.positions[:, col_slice] = th.where(valid, positions_values, 0)
    if roles_values is not None:
        buffers.roles[:, col_slice] = th.where(valid, roles_values, ROLE_NA)
    else:
        buffers.roles[:, col_slice] = th.where(valid, 0, ROLE_NA)


def tokenize_tensor_batch(observations: dict[str, th.Tensor]) -> TokenizedObservation:
    """Tokenize a dict of stacked observation tensors (the training hot path).

    Produces exactly the layout of :func:`tokenize` — hand cards ascending,
    trick cards in play order, history most-recent-first, five context
    values — without per-row Python work.

    Args:
        observations: Preprocessed observation dict of stacked tensors, as
            produced by SB3's Dict-observation preprocessing.

    Returns:
        The padded token batch.
    """
    hand = observations["hand"].to(th.int64)
    trick_play = observations["trick_play"].to(th.int64)
    history = observations["history"].to(th.int64)
    history_role = observations["history_role"].to(th.int64)
    trump = observations["trump"].to(th.int64)
    phase = observations["phase"].to(th.int64)
    tricks_won = observations["tricks_won"].to(th.int64)
    game_points = observations["game_points"].to(th.int64)
    seat = observations["seat"].to(th.int64)

    n = hand.shape[0]
    device = hand.device
    hand_ids, n_hand, trick_ids, trick_roles, n_trick, history_ids, history_roles = (
        _batch_card_groups(hand, trick_play, history, history_role, seat)
    )
    context_values = _batch_context(trump, phase, tricks_won, game_points, seat)

    buffers = _BatchBuffers(
        tokens=th.full((n, MAX_TOKENS), PAD_TOKEN, dtype=th.int64, device=device),
        is_card=th.zeros((n, MAX_TOKENS), dtype=th.bool, device=device),
        ranks=th.zeros((n, MAX_TOKENS), dtype=th.int64, device=device),
        is_trump=th.zeros((n, MAX_TOKENS), dtype=th.bool, device=device),
        type_ids=th.full((n, MAX_TOKENS), TYPE_CONTEXT, dtype=th.int64, device=device),
        positions=th.zeros((n, MAX_TOKENS), dtype=th.int64, device=device),
        roles=th.full((n, MAX_TOKENS), ROLE_NA, dtype=th.int64, device=device),
        padding=th.zeros((n, MAX_TOKENS), dtype=th.bool, device=device),
    )

    hand_valid = th.arange(NUM_HAND_SLOTS, device=device).unsqueeze(0) < n_hand.unsqueeze(1)
    _fill_batch_card_group(
        buffers,
        col_slice=HAND_SLICE,
        card_ids=hand_ids,
        valid=hand_valid,
        trump=trump,
        type_id=TYPE_HAND,
        positions_values=None,
        roles_values=None,
    )
    trick_ids = trick_ids[:, :NUM_TRICK_SLOTS]
    trick_roles = trick_roles[:, :NUM_TRICK_SLOTS]
    trick_valid = th.arange(NUM_TRICK_SLOTS, device=device).unsqueeze(0) < n_trick.unsqueeze(1)
    _fill_batch_card_group(
        buffers,
        col_slice=TRICK_SLICE,
        card_ids=trick_ids,
        valid=trick_valid,
        trump=trump,
        type_id=TYPE_TRICK,
        positions_values=th.arange(NUM_TRICK_SLOTS, device=device),
        roles_values=trick_roles,
    )
    hist_valid = history_ids != PAD_TOKEN
    _fill_batch_card_group(
        buffers,
        col_slice=HISTORY_SLICE,
        card_ids=history_ids,
        valid=hist_valid,
        trump=trump,
        type_id=TYPE_HISTORY,
        positions_values=th.arange(NUM_HISTORY_SLOTS, device=device),
        roles_values=history_roles,
    )

    for slot, column in enumerate(CTX_COLUMNS):
        buffers.tokens[:, column] = context_values[:, slot]
        buffers.type_ids[:, column] = TYPE_CONTEXT
        buffers.positions[:, column] = slot
        buffers.padding[:, column] = True

    buffers.padding[:, HAND_SLICE] = buffers.is_card[:, HAND_SLICE]
    buffers.padding[:, TRICK_SLICE] = buffers.is_card[:, TRICK_SLICE]
    buffers.padding[:, HISTORY_SLICE] = buffers.is_card[:, HISTORY_SLICE]

    suit_slots = _canonical_suit_slots(
        buffers.tokens,
        buffers.is_card,
        hand=hand,
        seen=observations["seen"].to(th.int64),
        trick=observations["trick"].to(th.int64),
        trump=trump,
    )

    return TokenizedObservation(
        tokens=buffers.tokens,
        is_card=buffers.is_card,
        ranks=buffers.ranks,
        is_trump=buffers.is_trump,
        suit_slots=suit_slots,
        type_ids=buffers.type_ids,
        positions=buffers.positions,
        roles=buffers.roles,
        context_values=context_values,
        padding_mask=buffers.padding,
    )
