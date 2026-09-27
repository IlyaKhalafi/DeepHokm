"""A scripted greedy Hokm baseline.

The policy plays the obvious lines a competent human plays without any
lookahead, which makes it a far more informative yardstick than uniform
random play: it calls its longest, strongest suit for trump, wins tricks as
cheaply as it can, and throws its lowest card when it cannot win or when its
partner is already winning.

It reads only the observation and the action mask, exactly like every other
:class:`~deephokm.policies.base.HokmPolicy`, so it can play any seat without
seeing hidden hands.
"""

from __future__ import annotations

import numpy as np

from deephokm.cards import NUM_RANKS, NUM_SUITS, RANK_OF, SUIT_OF
from deephokm.env.spaces import Observation
from deephokm.rules.legality import TRUMP_ACTION_OFFSET
from deephokm.rules.state import NUM_SEATS


class GreedyPolicy:
    """Scripted greedy play: cheapest win, lowest discard, longest trump.

    Trump call: score each suit by its length plus a small bonus for high
    cards, and declare the best. Card play: if a partner is already winning
    the trick, discard the lowest legal card; otherwise play the cheapest
    card that beats the current best, and the lowest legal card when nothing
    beats it. Leading, play the highest card of the longest suit held.

    The policy is deterministic: every choice is a total order over the legal
    cards, so there is nothing to seed and four greedy seats play identically.
    """

    def reset(self, seed: int | None = None) -> None:
        """No-op: the policy is deterministic and carries no episode state."""

    def act(self, observation: Observation, action_mask: np.ndarray) -> int:
        """Return the greedy legal action for the observation."""
        legal = np.flatnonzero(np.asarray(action_mask))
        if legal.size == 0:
            raise ValueError("action mask has no legal actions")
        if legal[0] >= TRUMP_ACTION_OFFSET:
            return self._call_trump(observation)
        return self._play_card(observation, [int(a) for a in legal])

    def _call_trump(self, observation: Observation) -> int:
        """Declare the longest suit, tie-broken by high-card strength."""
        hand = np.flatnonzero(np.asarray(observation["hand"]))
        strength = [0.0] * NUM_SUITS
        for card in hand:
            # Length dominates; the rank term only separates equal lengths.
            strength[SUIT_OF[int(card)]] += 1.0 + RANK_OF[int(card)] / (2 * NUM_RANKS)
        best = int(np.argmax(strength))
        return TRUMP_ACTION_OFFSET + best

    def _play_card(self, observation: Observation, legal: list[int]) -> int:
        """Choose a card: cheapest win, or lowest discard."""
        trump_vec = np.asarray(observation["trump"])
        trump = int(np.argmax(trump_vec)) if trump_vec.sum() else None
        trick_play = np.asarray(observation["trick_play"])
        seat = int(np.argmax(np.asarray(observation["seat"])))

        played = [(s, int(trick_play[s])) for s in range(NUM_SEATS) if trick_play[s] >= 0]
        if not played:
            return self._lead(legal)

        best_seat, best_card = self._current_best(played, trump)
        if (best_seat % 2) == (seat % 2):
            # The partner is winning: keep high cards, throw the lowest.
            return self._cheapest(legal, trump)
        winners = [c for c in legal if self._beats(c, best_card, trump)]
        if winners:
            return self._cheapest(winners, trump)
        return self._cheapest(legal, trump)

    def play_from_state(
        self,
        trump: int | None,
        current_trick: list[tuple[int, int]],
        seat: int,
        legal: list[int],
    ) -> int:
        """Choose a card directly from engine state, no observation built.

        Identical decisions to :meth:`_play_card`: cheapest win, lowest
        discard, highest of longest suit on lead. Simulation-only fast path
        for rollouts that otherwise rebuild a full observation dict per play.
        """
        if not current_trick:
            return self._lead(legal)
        best_seat, best_card = self._current_best(current_trick, trump)
        if (best_seat % 2) == (seat % 2):
            return self._cheapest(legal, trump)
        winners = [c for c in legal if self._beats(c, best_card, trump)]
        if winners:
            return self._cheapest(winners, trump)
        return self._cheapest(legal, trump)

    def _lead(self, legal: list[int]) -> int:
        """Lead the highest card of the longest suit held."""
        counts = [0] * NUM_SUITS
        for card in legal:
            counts[SUIT_OF[card]] += 1
        best_suit = max(range(NUM_SUITS), key=lambda s: (counts[s], s))
        candidates = [c for c in legal if SUIT_OF[c] == best_suit]
        return max(candidates, key=lambda c: RANK_OF[c])

    @staticmethod
    def _current_best(played: list[tuple[int, int]], trump: int | None) -> tuple[int, int]:
        """Return the ``(seat, card)`` currently winning the partial trick.

        ``played`` is in seat order, which is not play order, so the led suit
        is taken from the seat that led — the played seat whose predecessor
        has not played. ``played`` has at most 4 entries, so both the leader
        check and the play-order walk are plain loops; no set or sort.

        Every real caller passes a partial trick (at most 3 entries: the
        acting seat's own card is not yet in ``played``), so some seat's
        predecessor is always missing from the list and a leader is always
        found. A full 4-entry trick has no such seat -- every predecessor is
        present -- so that case raises rather than silently guessing seat 0,
        matching what the original set-based lookup (``next(...)`` with no
        default) did before this loop replaced it.
        """
        leader = None
        for s, _ in played:
            is_leader = True
            for s2, _ in played:
                if s2 == (s - 1) % NUM_SEATS:
                    is_leader = False
                    break
            if is_leader:
                leader = s
                break
        if leader is None:
            raise ValueError(
                f"no leader in {played}: every predecessor seat has played, "
                "which means this is a full trick, not a partial one"
            )
        by_seat = dict(played)
        best_seat, best_card = leader, by_seat[leader]
        for step in range(1, len(played)):
            s = (leader + step) % NUM_SEATS
            card = by_seat[s]
            if GreedyPolicy._beats(card, best_card, trump):
                best_seat, best_card = s, card
        return best_seat, best_card

    @staticmethod
    def _beats(card: int, best: int, trump: int | None) -> bool:
        """Return whether ``card`` beats the trick's current best card."""
        suit, best_suit = SUIT_OF[card], SUIT_OF[best]
        is_trump = trump is not None and suit == trump
        best_is_trump = trump is not None and best_suit == trump
        if is_trump and not best_is_trump:
            return True
        if is_trump == best_is_trump and suit == best_suit:
            return card > best
        return False

    @staticmethod
    def _cheapest(legal: list[int], trump: int | None) -> int:
        """Return the cheapest card in ``legal``: lowest rank first, trumps last.

        The (is_trump, rank, card) key list is built once per call and min'd,
        the same total order the old per-element key imposed (the card id
        breaks same-rank ties deterministically).
        """
        keys = [
            (1 if trump is not None and SUIT_OF[c] == trump else 0, RANK_OF[c], c) for c in legal
        ]
        return min(keys)[2]
