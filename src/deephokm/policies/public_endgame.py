"""Bounded endgame proofs across every publicly possible remaining deal.

The shortcut activates only when the acting hand, played cards, void suits,
and remaining capacities admit a complete, bounded enumeration of hidden
deals. It changes a move only if the same card forces a hand win in *every*
deal, regardless of all other legal plays, including its partner's. The root
has only one card after this move, so its future play cannot depend on hidden
ownership. This avoids both clairvoyant continuations and an assumption that
the partner shares the acting player's private deductions.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

from deephokm.cards import NUM_RANKS, NUM_SUITS, SUIT_OF
from deephokm.rules.state import NUM_SEATS, TRICKS_PER_HAND, TRICKS_TO_WIN_HAND
from deephokm.rules.tricks import resolve_trick

if TYPE_CHECKING:
    from deephokm.policies.greedy_policy import PublicKnowledge

MAX_HAND_CARDS = 2
MAX_INFERENCE_NODES = 4096
MAX_SOLVER_NODES = 10_000
MAX_CACHED_POSITIONS = 8192
_SUIT_MASKS = tuple(
    sum(1 << card for card in range(suit * NUM_RANKS, (suit + 1) * NUM_RANKS))
    for suit in range(NUM_SUITS)
)


class _BudgetExceeded(Exception):
    """A proof that exceeds its budget falls back to the normal heuristic."""


def _feasible_hands(
    knowledge: PublicKnowledge,
    *,
    max_nodes: int | None = None,
) -> tuple[tuple[int, ...], ...] | None:
    """Enumerate complete feasible deals, or decline if the budget runs out.

    With two cards in each hand there are at most 6! / (2!)**3 = 90
    assignments. A partial trick or known voids reduce that count further.
    No sample or truncated enumeration can establish a universal proof.
    """
    count = len(knowledge.hand)
    budget = MAX_INFERENCE_NODES if max_nodes is None else max_nodes
    played_seats = {seat for seat, _ in knowledge.current_trick}
    capacities = [count - int(seat in played_seats) for seat in range(NUM_SEATS)]
    capacities[knowledge.seat] = 0
    unseen = tuple(sorted(knowledge.unseen_cards))
    if sum(capacities) != len(unseen):
        return None
    held = [0] * NUM_SEATS
    held[knowledge.seat] = sum(1 << card for card in knowledge.hand)
    solutions: list[tuple[int, ...]] = []
    nodes = 0

    def visit(remaining: tuple[int, ...]) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > budget:
            raise _BudgetExceeded
        if not remaining:
            solutions.append(tuple(held))
            return
        options = {
            card: tuple(
                seat
                for seat in range(NUM_SEATS)
                if capacities[seat] > 0 and SUIT_OF[card] not in knowledge.void_suits[seat]
            )
            for card in remaining
        }
        if any(not owners for owners in options.values()):
            return
        for seat, capacity in enumerate(capacities):
            if capacity > sum(seat in owners for owners in options.values()):
                return
        card = min(remaining, key=lambda card: (len(options[card]), card))
        rest = tuple(other for other in remaining if other != card)
        for seat in options[card]:
            capacities[seat] -= 1
            held[seat] |= 1 << card
            visit(rest)
            held[seat] ^= 1 << card
            capacities[seat] += 1

    try:
        visit(unseen)
    except _BudgetExceeded:
        return None
    return tuple(solutions) if solutions else None


def _eligible_position(knowledge: PublicKnowledge, legal: list[int], baseline: int) -> bool:
    """Reject unsupported positions and inconsistent root action masks."""
    if (
        not 1 < len(knowledge.hand) <= MAX_HAND_CARDS
        or len(legal) != len(knowledge.hand)
        or len(knowledge.tricks_won) != NUM_SEATS // 2
        or len(knowledge.void_suits) != NUM_SEATS
        or baseline not in legal
        or not set(legal).issubset(knowledge.hand)
        or sum(knowledge.tricks_won) != TRICKS_PER_HAND - len(knowledge.hand)
        or any(not 0 <= count < TRICKS_TO_WIN_HAND for count in knowledge.tricks_won)
    ):
        return False
    choices = set(knowledge.hand)
    if knowledge.current_trick:
        following = {
            card for card in choices if SUIT_OF[card] == SUIT_OF[knowledge.current_trick[0][1]]
        }
        choices = following or choices
    return set(legal) == choices


def _already_secures_hand(knowledge: PublicKnowledge, baseline: int) -> bool:
    """Cheap sufficient proof when this trick alone wins the hand."""
    if knowledge.tricks_won[0] != TRICKS_TO_WIN_HAND - 1:
        return False
    played = (*knowledge.current_trick, (knowledge.seat, baseline))
    led_suit = SUIT_OF[played[0][1]]

    def strength(card: int) -> tuple[bool, bool, int]:
        return SUIT_OF[card] == knowledge.trump, SUIT_OF[card] == led_suit, card % NUM_RANKS

    winner_seat, winner = max(played, key=lambda item: strength(item[1]))
    if winner_seat % 2 != knowledge.seat % 2:
        return False
    # Consider every unseen counter, even counters that may be forced to
    # follow suit instead. Overestimating threats keeps this check sound.
    return not any(
        strength(card) > strength(winner) and SUIT_OF[card] not in knowledge.void_suits[opponent]
        for offset in range(1, NUM_SEATS - len(knowledge.current_trick))
        if (opponent := (knowledge.seat + offset) % NUM_SEATS) % 2 != knowledge.seat % 2
        for card in knowledge.unseen_cards
    )


def _interchangeable_actions(knowledge: PublicKnowledge, legal: list[int]) -> bool:
    """Skip ranks whose swap preserves every possible live-card comparison."""
    low, high = sorted(legal)
    suit = SUIT_OF[low]
    return SUIT_OF[high] == suit and not any(
        SUIT_OF[card] == suit and low < card < high
        for card in (*knowledge.unseen_cards, *(card for _, card in knowledge.current_trick))
    )


def guaranteed_endgame_action(
    knowledge: PublicKnowledge,
    legal: list[int],
    baseline: int,
) -> int | None:
    """Find a strictly stronger forced win, or leave the heuristic unchanged."""
    if (
        not _eligible_position(knowledge, legal, baseline)
        or _already_secures_hand(knowledge, baseline)
        or _interchangeable_actions(knowledge, legal)
    ):
        return None
    # Past play order is irrelevant once unseen cards and voids are known.
    # Canonicalizing it lets repeated rollout states reuse the same proof.
    # Include both budgets so cached proofs never bypass a changed limit.
    return _cached_proof(
        replace(knowledge, played_cards=()),
        tuple(sorted(legal)),
        baseline,
        MAX_INFERENCE_NODES,
        MAX_SOLVER_NODES,
    )


@lru_cache(maxsize=MAX_CACHED_POSITIONS)
def _cached_proof(
    knowledge: PublicKnowledge,
    legal: tuple[int, ...],
    baseline: int,
    inference_budget: int,
    solver_budget: int,
) -> int | None:
    """Cache only the public facts relevant to a deterministic bounded proof."""
    deals = _feasible_hands(knowledge, max_nodes=inference_budget)
    if deals is None:
        return None
    root_seat = knowledge.seat
    root_team = root_seat % 2
    score = knowledge.tricks_won if root_team == 0 else knowledge.tricks_won[::-1]
    nodes = 0

    def play(
        masks: tuple[int, ...],
        seat: int,
        trick: tuple[tuple[int, int], ...],
        counts: tuple[int, int],
        card: int,
    ) -> int:
        remaining = list(masks)
        remaining[seat] &= ~(1 << card)
        played = (*trick, (seat, card))
        if len(played) == NUM_SEATS:
            winner = resolve_trick(list(played), knowledge.trump)
            updated = list(counts)
            updated[winner % 2] += 1
            return solve(tuple(remaining), winner, (), (updated[0], updated[1]))
        return solve(tuple(remaining), (seat + 1) % NUM_SEATS, played, counts)

    @lru_cache(maxsize=solver_budget)
    def solve(
        masks: tuple[int, ...],
        seat: int,
        trick: tuple[tuple[int, int], ...],
        counts: tuple[int, int],
    ) -> int:
        nonlocal nodes
        if counts[root_team] >= TRICKS_TO_WIN_HAND:
            return 1
        if counts[1 - root_team] >= TRICKS_TO_WIN_HAND:
            return -1
        nodes += 1
        if nodes > solver_budget:
            raise _BudgetExceeded
        choices = masks[seat]
        if trick:
            following = choices & _SUIT_MASKS[SUIT_OF[trick[0][1]]]
            choices = following or choices
        # Missing cards or inconsistent terminal scores are not a proof.
        if not choices:
            raise _BudgetExceeded
        maximizing = seat == root_seat
        best = -1 if maximizing else 1
        while choices:
            bit = choices & -choices
            choices ^= bit
            value = play(masks, seat, trick, counts, bit.bit_length() - 1)
            best = max(best, value) if maximizing else min(best, value)
            if (maximizing and best == 1) or (not maximizing and best == -1):
                break
        return best

    try:

        def guarantees_win(card: int) -> bool:
            return all(
                play(hands, root_seat, knowledge.current_trick, score, card) == 1 for hands in deals
            )

        if guarantees_win(baseline):
            return None
        for card in legal:
            if card != baseline and guarantees_win(card):
                return card
    except _BudgetExceeded:
        return None
    return None
