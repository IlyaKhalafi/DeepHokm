"""A public-information heuristic Hokm baseline.

The policy plays the obvious lines a competent human plays, which makes
it a far more informative yardstick than uniform
random play: it calls its longest, strongest suit for trump, wins tricks as
cheaply as it can, and throws its lowest card when it cannot win or when its
partner is already winning.

It reads only the observation and the action mask, exactly like every other
:class:`~deephokm.policies.base.HokmPolicy`, so it can play any seat without
seeing hidden hands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deephokm.cards import NUM_CARDS, NUM_RANKS, NUM_SUITS, RANK_OF, SUIT_OF
from deephokm.env.spaces import Observation
from deephokm.policies.public_endgame import guaranteed_endgame_action
from deephokm.policies.trick_odds import opponent_beating_probability
from deephokm.rules.legality import TRUMP_ACTION_OFFSET
from deephokm.rules.state import NUM_SEATS, TRICKS_TO_WIN_HAND
from deephokm.rules.tricks import resolve_trick

_ALL_CARDS = frozenset(range(NUM_CARDS))
_NUM_TEAMS = NUM_SEATS // 2
_DEVELOP_THROUGH_TRICK = 6
_SHORT_SUIT_LIMIT = 2
_THIRD_HAND_RANK_COST = 0.04


@dataclass(frozen=True, slots=True)
class PublicKnowledge:
    """Information a player can prove from its observation.

    Rebuilding this value on every decision gives the policy complete memory
    of the hand without retaining private information when one policy object
    is shared by several seats or reused by search rollouts.
    """

    seat: int
    trump: int | None
    hand: frozenset[int]
    played_cards: tuple[int, ...]
    current_trick: tuple[tuple[int, int], ...]
    void_suits: tuple[frozenset[int], ...]
    unseen_cards: frozenset[int]
    tricks_won: tuple[int, int] = (0, 0)

    @classmethod
    def from_observation(cls, observation: Observation) -> PublicKnowledge:
        """Reconstruct ordered play and proven void suits."""
        seat_vec = np.asarray(observation["seat"])
        if np.count_nonzero(seat_vec) != 1:
            raise ValueError("observation seat must be one-hot")
        seat = int(np.argmax(seat_vec))

        trump_vec = np.asarray(observation["trump"])
        if np.count_nonzero(trump_vec) > 1:
            raise ValueError("observation trump must be zero-hot or one-hot")
        trump = int(np.argmax(trump_vec)) if np.any(trump_vec) else None
        hand = frozenset(int(card) for card in np.flatnonzero(observation["hand"]))

        history = np.asarray(observation["history"], dtype=np.int64)
        history_role = np.asarray(observation["history_role"], dtype=np.int64)
        card_slots = history >= 0
        role_slots = history_role >= 0
        if not np.array_equal(card_slots, role_slots):
            raise ValueError("history and history_role padding must match")
        history_size = int(np.count_nonzero(card_slots))
        if history_size and not np.all(card_slots[:history_size]):
            raise ValueError("history entries must precede padding")
        if history_size % NUM_SEATS:
            raise ValueError("history must contain complete four-card tricks")

        # Both history arrays are newest-first. Reverse them together and
        # turn relative roles back into absolute seat ids.
        past_cards = [int(card) for card in history[:history_size][::-1]]
        past_roles = [int(role) for role in history_role[:history_size][::-1]]
        past_plays = [
            ((seat + role) % NUM_SEATS, card)
            for role, card in zip(past_roles, past_cards, strict=True)
        ]

        trick_play = np.asarray(observation["trick_play"], dtype=np.int64)
        occupied = [s for s in range(NUM_SEATS) if trick_play[s] >= 0]
        current: list[tuple[int, int]] = []
        if occupied:
            leaders = [s for s in occupied if (s - 1) % NUM_SEATS not in occupied]
            if len(leaders) != 1:
                raise ValueError("current trick seats are not one consecutive partial trick")
            leader = leaders[0]
            expected = [(leader + step) % NUM_SEATS for step in range(len(occupied))]
            if set(expected) != set(occupied):
                raise ValueError("current trick seats are not consecutive")
            current = [(s, int(trick_play[s])) for s in expected]

        counts = observation["tricks_won"]
        if len(counts) != _NUM_TEAMS:
            raise ValueError("tricks_won must contain one count per team")

        return cls._build(
            seat=seat,
            trump=trump,
            hand=hand,
            past_plays=past_plays,
            current=current,
            tricks_won=(int(counts[0]), int(counts[1])),
        )

    @classmethod
    def from_state(
        cls,
        *,
        seat: int,
        trump: int | None,
        hand: list[int],
        played: list[int],
        played_by: list[int],
        current_trick: list[tuple[int, int]],
        void_suits: list[set[int]] | None = None,
        tricks_won: list[int] | tuple[int, int] | None = None,
    ) -> PublicKnowledge:
        """Build the same snapshot from public engine state for rollouts."""
        if len(played) != len(played_by):
            raise ValueError("played and played_by lengths must match")
        if not 0 <= seat < NUM_SEATS:
            raise ValueError(f"seat must be in [0, {NUM_SEATS})")
        completed = len(played) - len(current_trick)
        if completed < 0 or completed % NUM_SEATS:
            raise ValueError("played history must contain complete tricks before current_trick")
        expected_current = list(zip(played_by[completed:], played[completed:], strict=True))
        if expected_current != current_trick:
            raise ValueError("current_trick must be the suffix of played history")
        if tricks_won is not None and len(tricks_won) != _NUM_TEAMS:
            raise ValueError("tricks_won must contain one count per team")
        if tricks_won is None:
            # Compatibility callers need the same score-aware decision as
            # act(). Rollout callers pass the cached engine tally and skip
            # this reconstruction on their hot path.
            inferred = [0] * _NUM_TEAMS
            for start in range(0, completed, NUM_SEATS):
                past_trick = list(
                    zip(
                        played_by[start : start + NUM_SEATS],
                        played[start : start + NUM_SEATS],
                        strict=True,
                    )
                )
                inferred[resolve_trick(past_trick, trump) % 2] += 1
            tricks_won = inferred
        own_team = seat % 2
        own_tricks = (int(tricks_won[own_team]), int(tricks_won[1 - own_team]))
        if void_suits is not None:
            if len(void_suits) != NUM_SEATS:
                raise ValueError(f"void_suits must have {NUM_SEATS} entries")
            hand_cards = frozenset(hand)
            played_cards = tuple(played)
            return cls(
                seat=seat,
                trump=trump,
                hand=hand_cards,
                played_cards=played_cards,
                current_trick=tuple(current_trick),
                void_suits=tuple(frozenset(suits) for suits in void_suits),
                unseen_cards=_ALL_CARDS.difference(hand_cards, played_cards),
                tricks_won=own_tricks,
            )
        past_plays = list(zip(played_by[:completed], played[:completed], strict=True))
        return cls._build(
            seat=seat,
            trump=trump,
            hand=frozenset(hand),
            past_plays=past_plays,
            current=list(current_trick),
            known_void_suits=void_suits,
            tricks_won=own_tricks,
        )

    @classmethod
    def _build(
        cls,
        *,
        seat: int,
        trump: int | None,
        hand: frozenset[int],
        past_plays: list[tuple[int, int]],
        current: list[tuple[int, int]],
        known_void_suits: list[set[int]] | None = None,
        tricks_won: tuple[int, int] = (0, 0),
    ) -> PublicKnowledge:
        """Validate public plays and infer facts shared by both input paths."""
        if not 0 <= seat < NUM_SEATS:
            raise ValueError(f"seat must be in [0, {NUM_SEATS})")
        if len(current) >= NUM_SEATS:
            raise ValueError("current trick must be partial")
        if current and current[-1][0] != (seat - 1) % NUM_SEATS:
            raise ValueError("acting seat must follow the final current-trick seat")

        voids: list[set[int]] = (
            [set() for _ in range(NUM_SEATS)]
            if known_void_suits is None
            else [set(suits) for suits in known_void_suits]
        )
        if len(voids) != NUM_SEATS:
            raise ValueError(f"void_suits must have {NUM_SEATS} entries")

        def remember_voids(trick: list[tuple[int, int]]) -> None:
            if len({played_seat for played_seat, _ in trick}) != len(trick):
                raise ValueError("a seat cannot play twice in one trick")
            led_suit = SUIT_OF[trick[0][1]]
            for played_seat, card in trick[1:]:
                if SUIT_OF[card] != led_suit:
                    voids[played_seat].add(led_suit)

        for start in range(0, len(past_plays), NUM_SEATS):
            trick = past_plays[start : start + NUM_SEATS]
            expected = [(trick[0][0] + step) % NUM_SEATS for step in range(NUM_SEATS)]
            if [played_seat for played_seat, _ in trick] != expected:
                raise ValueError("completed trick seats must be in play order")
            if known_void_suits is None:
                remember_voids(trick)
        if current and known_void_suits is None:
            remember_voids(current)

        played_cards = tuple(card for _, card in past_plays) + tuple(card for _, card in current)
        if len(set(played_cards)) != len(played_cards):
            raise ValueError("a card cannot be played twice")
        if hand.intersection(played_cards):
            raise ValueError("played cards cannot remain in hand")
        unseen = _ALL_CARDS.difference(hand, played_cards)
        return cls(
            seat=seat,
            trump=trump,
            hand=hand,
            played_cards=played_cards,
            current_trick=tuple(current),
            void_suits=tuple(frozenset(suits) for suits in voids),
            unseen_cards=unseen,
            tricks_won=tricks_won,
        )


class GreedyPolicy:
    """Deterministic heuristic play using all legally public information.

    The policy remembers cards and proven void suits by rebuilding an
    immutable knowledge snapshot from the complete observation each turn.
    It wins as cheaply as possible, avoids overtaking a safe partner, protects
    a threatened partner when it can make the trick secure, spends stronger
    cards when the next trick decides the hand, develops short side suits
    early to create ruffing opportunities, preserves master cards and trumps
    on discards, and avoids leading into known opponent ruffs. Third-hand
    winners balance algebraic final-seat counter odds against card cost,
    accounting for the opponent's obligation to follow suit. With two
    cards left, a bounded proof can replace the heuristic only when the same
    move wins across every deal allowed by public constraints. No hidden hand
    is ever inspected, and no Monte Carlo rollouts are performed.
    """

    def reset(self, seed: int | None = None) -> None:
        """No-op: decisions are deterministic snapshots of each observation."""

    def act(self, observation: Observation, action_mask: np.ndarray) -> int:
        """Return a legal heuristic action for the observation."""
        legal = [int(action) for action in np.flatnonzero(np.asarray(action_mask))]
        if not legal:
            raise ValueError("action mask has no legal actions")
        trump_actions = [action for action in legal if action >= TRUMP_ACTION_OFFSET]
        if trump_actions:
            if len(trump_actions) != len(legal):
                raise ValueError("action mask cannot mix card plays and trump calls")
            return self._call_trump(observation, trump_actions)
        knowledge = PublicKnowledge.from_observation(observation)
        return self._play_with_knowledge(knowledge, legal)

    def _call_trump(self, observation: Observation, legal: list[int]) -> int:
        """Declare the longest legal suit, then prefer high-card strength."""
        hand = [int(card) for card in np.flatnonzero(observation["hand"])]
        legal_suits = [action - TRUMP_ACTION_OFFSET for action in legal]

        def suit_strength(suit: int) -> tuple[int, int, tuple[int, ...], int]:
            ranks = sorted((RANK_OF[card] for card in hand if SUIT_OF[card] == suit), reverse=True)
            return len(ranks), sum(ranks), tuple(ranks), suit

        best = max(legal_suits, key=suit_strength)
        return TRUMP_ACTION_OFFSET + best

    def _play_with_knowledge(self, knowledge: PublicKnowledge, legal: list[int]) -> int:
        """Choose a card from an immutable public-information snapshot."""
        baseline = self._heuristic_with_knowledge(knowledge, legal)
        proved = guaranteed_endgame_action(knowledge, legal, baseline)
        return baseline if proved is None else proved

    def _heuristic_with_knowledge(self, knowledge: PublicKnowledge, legal: list[int]) -> int:
        """Apply the normal lead, teamwork, winning, and discard rules."""
        if not knowledge.current_trick:
            return self._lead_with_knowledge(legal, knowledge)

        trick = list(knowledge.current_trick)
        best_seat, best_card = self._winner_in_order(trick, knowledge.trump)
        remaining = NUM_SEATS - len(trick) - 1
        future_seats = [(knowledge.seat + offset) % NUM_SEATS for offset in range(1, remaining + 1)]
        future_opponents = [seat for seat in future_seats if seat % 2 != knowledge.seat % 2]
        winners = [card for card in legal if self._beats(card, best_card, knowledge.trump)]
        decisive_trick = max(knowledge.tricks_won) == TRICKS_TO_WIN_HAND - 1

        if best_seat % 2 == knowledge.seat % 2:
            underplays = [
                card for card in legal if not self._beats(card, best_card, knowledge.trump)
            ]
            current_threats = self._threat_count(best_card, future_opponents, knowledge)
            if current_threats == 0:
                return self._discard(underplays or legal, knowledge)
            # Overtaking spends a winning card and steals the lead from our
            # partner. A merely *less vulnerable* card still risks losing
            # both cards to the same opponent; spend it only if it secures
            # the trick against every publicly plausible counter.
            protectors = []
            for card in winners:
                threats = self._threat_count(card, future_opponents, knowledge)
                if threats == 0 or (decisive_trick and threats < current_threats):
                    protectors.append(card)
            if protectors:
                return self._best_protector(protectors, future_opponents, knowledge)
            return self._discard(underplays or legal, knowledge)

        if not winners:
            return self._discard(legal, knowledge)
        return self._winning_response(winners, future_opponents, knowledge)

    def _winning_response(
        self,
        candidates: list[int],
        opponents: list[int],
        knowledge: PublicKnowledge,
    ) -> int:
        """Choose the cost of winning according to who can still counter."""
        if len(knowledge.current_trick) == NUM_SEATS - 2:
            return self._third_hand_winner(candidates, opponents[0], knowledge)
        return self._fallback_winner(candidates, opponents, knowledge)

    def _fallback_winner(
        self,
        candidates: list[int],
        opponents: list[int],
        knowledge: PublicKnowledge,
    ) -> int:
        """Preserve the original decisive-trick rule without a usable estimate."""
        return (
            self._best_protector(candidates, opponents, knowledge)
            if max(knowledge.tricks_won) == TRICKS_TO_WIN_HAND - 1 and opponents
            else self._cheapest(candidates, knowledge.trump)
        )

    def _third_hand_winner(
        self,
        candidates: list[int],
        opponent: int,
        knowledge: PublicKnowledge,
    ) -> int:
        """Balance final-seat counters against spending a higher winning card.

        Only the last opponent remains. A high card can secure this trick,
        but a higher trump cannot counter us if that opponent must follow a
        side suit. Approximate ownership odds respect both cases, while the
        empirical rank cost preserves useful cards for later tricks.
        """
        if len(candidates) == 1:
            return candidates[0]
        led_suit = SUIT_OF[knowledge.current_trick[0][1]]
        values: dict[int, float] = {}
        for card in candidates:
            risk = opponent_beating_probability(
                winner=card,
                led_suit=led_suit,
                trump=knowledge.trump,
                unseen_cards=knowledge.unseen_cards,
                void_suits=knowledge.void_suits[opponent],
                hand_size=len(knowledge.hand),
            )
            if risk is None:
                return self._fallback_winner(candidates, [opponent], knowledge)
            values[card] = 1.0 - risk - _THIRD_HAND_RANK_COST * RANK_OF[card]
        return max(candidates, key=lambda card: (values[card], -RANK_OF[card], -card))

    def _lead_with_knowledge(self, legal: list[int], knowledge: PublicKnowledge) -> int:
        """Develop short side suits early, then cash controls or a long suit."""
        counts = [0] * NUM_SUITS
        for card in legal:
            counts[SUIT_OF[card]] += 1
        opponents = [seat for seat in range(NUM_SEATS) if seat % 2 != knowledge.seat % 2]
        if (
            sum(knowledge.tricks_won) <= _DEVELOP_THROUGH_TRICK
            and max(knowledge.tricks_won) < TRICKS_TO_WIN_HAND - 1
        ):
            short_suits = {
                suit
                for suit in range(NUM_SUITS)
                if suit != knowledge.trump
                and 0 < counts[suit] <= _SHORT_SUIT_LIMIT
                and not any(self._can_ruff(seat, suit, knowledge) for seat in opponents)
            }
            if short_suits:
                return min(
                    (card for card in legal if SUIT_OF[card] in short_suits),
                    key=lambda card: (counts[SUIT_OF[card]], RANK_OF[card], card),
                )
        high_by_suit = {
            suit: max(
                (card for card in legal if SUIT_OF[card] == suit),
                key=lambda card: RANK_OF[card],
            )
            for suit in range(NUM_SUITS)
            if counts[suit]
        }
        partner = (knowledge.seat + 2) % NUM_SEATS

        def lead_key(suit: int) -> tuple[int, int, int, int, int, int, int]:
            card = high_by_suit[suit]
            ruff_danger = sum(self._can_ruff(seat, suit, knowledge) for seat in opponents)
            partner_can_ruff = int(self._can_ruff(partner, suit, knowledge))
            non_trump = int(knowledge.trump is None or suit != knowledge.trump)
            return (
                -ruff_danger,
                int(self._is_master(card, knowledge)),
                counts[suit],
                partner_can_ruff,
                non_trump,
                RANK_OF[card],
                suit,
            )

        suit = max(high_by_suit, key=lead_key)
        return high_by_suit[suit]

    def _discard(self, legal: list[int], knowledge: PublicKnowledge) -> int:
        """Shed the least useful card while preserving control cards."""
        counts = [0] * NUM_SUITS
        for card in knowledge.hand:
            counts[SUIT_OF[card]] += 1

        def discard_key(card: int) -> tuple[int, int, int, int]:
            is_trump = int(knowledge.trump is not None and SUIT_OF[card] == knowledge.trump)
            strategic_value = is_trump + 2 * int(self._is_master(card, knowledge))
            return strategic_value, RANK_OF[card], counts[SUIT_OF[card]], card

        return min(legal, key=discard_key)

    def _threat_count(
        self,
        winner: int,
        opponents: list[int],
        knowledge: PublicKnowledge,
    ) -> int:
        """Count unseen opponent-card placements that could beat ``winner``."""
        return sum(
            1
            for seat in opponents
            for card in knowledge.unseen_cards
            if SUIT_OF[card] not in knowledge.void_suits[seat]
            and self._beats(card, winner, knowledge.trump)
        )

    def _best_protector(
        self,
        candidates: list[int],
        opponents: list[int],
        knowledge: PublicKnowledge,
    ) -> int:
        """Minimize plausible counters, then spend the cheapest winning card."""
        trump = knowledge.trump
        return min(
            candidates,
            key=lambda card: (
                self._threat_count(card, opponents, knowledge),
                int(trump is not None and SUIT_OF[card] == trump),
                RANK_OF[card],
                card,
            ),
        )

    @staticmethod
    def _is_master(card: int, knowledge: PublicKnowledge) -> bool:
        """Return whether no other player can hold a higher card of this suit.

        A higher card in our own hand is just as unavailable to opponents as
        one already played. This matters when discarding: a king backed by
        our ace is a future winner, not an ordinary losing card.
        """
        suit = SUIT_OF[card]
        return all(
            suit * NUM_RANKS + rank not in knowledge.unseen_cards
            for rank in range(RANK_OF[card] + 1, NUM_RANKS)
        )

    @staticmethod
    def _can_ruff(seat: int, led_suit: int, knowledge: PublicKnowledge) -> bool:
        """Return whether public facts leave ``seat`` able to ruff this suit."""
        trump = knowledge.trump
        if trump is None or led_suit == trump:
            return False
        return (
            led_suit in knowledge.void_suits[seat]
            and trump not in knowledge.void_suits[seat]
            and any(SUIT_OF[card] == trump for card in knowledge.unseen_cards)
        )

    @classmethod
    def _winner_in_order(
        cls,
        played: list[tuple[int, int]],
        trump: int | None,
    ) -> tuple[int, int]:
        """Return the winner of plays already arranged chronologically."""
        best_seat, best_card = played[0]
        for seat, card in played[1:]:
            if cls._beats(card, best_card, trump):
                best_seat, best_card = seat, card
        return best_seat, best_card

    def play_from_state(
        self,
        trump: int | None,
        current_trick: list[tuple[int, int]],
        seat: int,
        legal: list[int],
        *,
        hand: list[int] | None = None,
        played: list[int] | None = None,
        played_by: list[int] | None = None,
        void_suits: list[set[int]] | None = None,
        tricks_won: list[int] | tuple[int, int] | None = None,
    ) -> int:
        """Choose directly from engine state without allocating an observation.

        Rollout callers pass the optional hand, play history, void suits, and
        trick score to get decisions identical to :meth:`act`. The original
        four-argument API is retained as a conservative compatibility fallback.
        """
        supplied = (hand is not None, played is not None, played_by is not None)
        if any(supplied):
            if not all(supplied):
                raise ValueError("hand, played, and played_by must be supplied together")
            assert hand is not None and played is not None and played_by is not None
            knowledge = PublicKnowledge.from_state(
                seat=seat,
                trump=trump,
                hand=hand,
                played=played,
                played_by=played_by,
                current_trick=current_trick,
                void_suits=void_suits,
                tricks_won=tricks_won,
            )
            return self._play_with_knowledge(knowledge, legal)

        if not current_trick:
            return self._lead(legal)
        best_seat, best_card = self._current_best(current_trick, trump)
        if best_seat % 2 == seat % 2:
            return self._cheapest(legal, trump)
        winners = [card for card in legal if self._beats(card, best_card, trump)]
        return self._cheapest(winners or legal, trump)

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
        ordered: list[tuple[int, int]] = []
        for step in range(len(played)):
            seat = (leader + step) % NUM_SEATS
            if seat not in by_seat:
                raise ValueError(f"non-consecutive seats in partial trick: {played}")
            ordered.append((seat, by_seat[seat]))
        return GreedyPolicy._winner_in_order(ordered, trump)

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
