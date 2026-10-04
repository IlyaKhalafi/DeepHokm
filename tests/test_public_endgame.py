"""Public ownership deductions and conservative endgame proofs."""

from copy import deepcopy
from dataclasses import replace
from itertools import combinations
from random import Random

import pytest

from deephokm.cards import NUM_CARDS, SUIT_OF
from deephokm.policies import public_endgame
from deephokm.policies.greedy_policy import GreedyPolicy, PublicKnowledge
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import HandState, Phase


def _forced_position(seat: int = 0) -> PublicKnowledge:
    hand = frozenset({12, 51})
    unseen = frozenset({0, 1, 13, 26, 27, 39})
    relative_voids = (frozenset(), frozenset({2}), frozenset({0, 1, 3}), frozenset({1, 2, 3}))
    voids = tuple(relative_voids[(absolute - seat) % 4] for absolute in range(4))
    return PublicKnowledge(
        seat=seat,
        trump=3,
        hand=hand,
        played_cards=tuple(sorted(set(range(NUM_CARDS)) - hand - unseen)),
        current_trick=(),
        void_suits=voids,
        unseen_cards=unseen,
        tricks_won=(5, 6),
    )


@pytest.mark.parametrize("seat", range(4))
def test_forced_ownership_uses_capacities_to_find_a_winning_trump_lead(seat: int) -> None:
    knowledge = _forced_position(seat)
    policy = GreedyPolicy()
    assert policy._lead_with_knowledge([12, 51], knowledge) == 12
    # The next enemy must hold the only unseen diamond and trump. Public
    # capacity then assigns both clubs to the last enemy, even though the
    # next enemy has never explicitly shown a club void.
    expected = ((12, 51), (13, 39), (26, 27), (0, 1))
    deals = public_endgame._feasible_hands(knowledge)
    assert deals is not None and len(deals) == 1
    masks = deals[0]
    for role, cards in enumerate(expected):
        assert masks[(seat + role) % 4] == sum(1 << card for card in cards)
    # Leading the club ace loses immediately to a ruff. Drawing the sole
    # opposing trump first wins both remaining tricks regardless of others.
    assert public_endgame.guaranteed_endgame_action(knowledge, [12, 51], 12) == 51
    assert policy._play_with_knowledge(knowledge, [12, 51]) == 51


def test_endgame_does_not_change_an_already_guaranteed_winner() -> None:
    knowledge = _forced_position()
    assert public_endgame.guaranteed_endgame_action(knowledge, [12, 51], 51) is None


def test_universal_win_does_not_require_any_known_void_suits() -> None:
    knowledge = _forced_position()
    ambiguous = PublicKnowledge(
        seat=knowledge.seat,
        trump=knowledge.trump,
        hand=knowledge.hand,
        played_cards=knowledge.played_cards,
        current_trick=(),
        void_suits=(frozenset(),) * 4,
        unseen_cards=knowledge.unseen_cards,
        tricks_won=knowledge.tricks_won,
    )
    deals = public_endgame._feasible_hands(ambiguous)
    assert deals is not None and len(deals) == 90
    assert public_endgame.guaranteed_endgame_action(ambiguous, [12, 51], 12) == 51


def test_one_counterexample_deal_prevents_a_clairvoyant_move() -> None:
    knowledge = _forced_position()
    # With two enemy trumps possibly together, drawing one does not secure
    # the subsequent club ace. A favorable sample is not a guarantee.
    unseen = (knowledge.unseen_cards - {13}) | {40}
    ambiguous = replace(
        knowledge,
        unseen_cards=unseen,
        played_cards=tuple(sorted(set(range(NUM_CARDS)) - knowledge.hand - unseen)),
        void_suits=(frozenset(),) * 4,
    )
    assert public_endgame.guaranteed_endgame_action(ambiguous, [12, 51], 12) is None


@pytest.mark.parametrize("seat", range(4))
def test_same_trump_lead_wins_across_all_six_ambiguous_enemy_deals(seat: int) -> None:
    knowledge = _forced_position(seat)
    voids = list(knowledge.void_suits)
    voids[(seat + 3) % 4] = frozenset({2})
    ambiguous = replace(knowledge, void_suits=tuple(voids))
    deals = public_endgame._feasible_hands(ambiguous)
    assert deals is not None and len(deals) == 6
    assert public_endgame.guaranteed_endgame_action(ambiguous, [12, 51], 12) == 51


@pytest.mark.parametrize("seat", range(4))
@pytest.mark.parametrize("n_played", [1, 2, 3])
def test_unique_ownership_accounts_for_partial_tricks_and_seat_wrap(
    seat: int,
    n_played: int,
) -> None:
    knowledge = _forced_position(seat)
    role_cards = ((1, 13), (2, 26), (3, 0))[-n_played:]
    current = tuple(((seat + role) % 4, card) for role, card in role_cards)
    partial = replace(
        knowledge,
        current_trick=current,
        played_cards=(*knowledge.played_cards, *(card for _, card in current)),
        unseen_cards=knowledge.unseen_cards - {card for _, card in current},
    )
    original = public_endgame._feasible_hands(knowledge)
    inferred = public_endgame._feasible_hands(partial)
    assert original is not None and inferred is not None
    assert len(original) == len(inferred) == 1
    expected = list(original[0])
    for played_seat, card in current:
        expected[played_seat] &= ~(1 << card)
    assert inferred[0] == tuple(expected)


def test_endgame_rejects_actions_that_violate_follow_suit() -> None:
    knowledge = _forced_position()
    partial = replace(
        knowledge,
        current_trick=((3, 0),),
        unseen_cards=knowledge.unseen_cards - {0},
        played_cards=(*knowledge.played_cards, 0),
    )
    assert public_endgame.guaranteed_endgame_action(partial, [12, 51], 12) is None


def test_endgame_rejects_duplicate_action_entries() -> None:
    assert public_endgame.guaranteed_endgame_action(_forced_position(), [12, 51, 51], 12) is None


def test_endgame_rejects_inconsistent_score_and_unheld_action() -> None:
    knowledge = _forced_position()
    assert public_endgame.guaranteed_endgame_action(knowledge, [0, 51], 0) is None
    assert (
        public_endgame.guaranteed_endgame_action(
            replace(knowledge, tricks_won=(6, 6)),
            [12, 51],
            12,
        )
        is None
    )


@pytest.mark.parametrize("budget_name", ["MAX_INFERENCE_NODES", "MAX_SOLVER_NODES"])
def test_endgame_exhausted_budget_falls_back(
    budget_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(public_endgame, budget_name, 0)
    assert public_endgame.guaranteed_endgame_action(_forced_position(), [12, 51], 12) is None


def _reference_deals(knowledge: PublicKnowledge) -> list[list[list[int]]]:
    """Independent combinations-based enumeration, without production bitmasks."""
    seats = [seat for seat in range(4) if seat != knowledge.seat]
    played = {seat for seat, _ in knowledge.current_trick}
    hands = [[] for _ in range(4)]
    hands[knowledge.seat] = sorted(knowledge.hand)
    deals = []

    def assign(index: int, cards: frozenset[int]) -> None:
        if index == len(seats):
            if not cards:
                deals.append(deepcopy(hands))
            return
        seat = seats[index]
        capacity = len(knowledge.hand) - int(seat in played)
        for held in combinations(sorted(cards), capacity):
            if any(SUIT_OF[card] in knowledge.void_suits[seat] for card in held):
                continue
            hands[seat] = list(held)
            assign(index + 1, cards - set(held))

    assign(0, knowledge.unseen_cards)
    return deals


def _reference_value(
    engine: HokmEngine,
    root: int,
    action: int,
    *,
    cooperative_partner: bool = False,
) -> int:
    """Use actual engine transitions; everyone but root is adversarial."""
    engine = deepcopy(engine)
    outcome = engine.apply_action(action, seat=engine.current_seat())
    if outcome.hand_complete:
        return 1 if outcome.hand_winner_team == root % 2 else -1
    seat = engine.current_seat()
    values = (
        _reference_value(engine, root, card, cooperative_partner=cooperative_partner)
        for card in engine.legal_actions()
    )
    maximizing = seat == root or (cooperative_partner and seat == (root + 2) % 4)
    return max(values) if maximizing else min(values)


def test_endgame_does_not_assume_partner_chooses_a_cooperative_card() -> None:
    hands = [[44, 46], [26, 31], [3, 25], [47, 49]]
    hand = frozenset(hands[0])
    unseen = frozenset(card for held in hands[1:] for card in held)
    knowledge = replace(
        _forced_position(),
        trump=1,
        hand=hand,
        unseen_cards=unseen,
        played_cards=tuple(sorted(set(range(NUM_CARDS)) - hand - unseen)),
        void_suits=(
            frozenset({0, 1, 2}),
            frozenset({0, 1, 3}),
            frozenset({2, 3}),
            frozenset({0, 1, 2}),
        ),
    )
    deals = public_endgame._feasible_hands(knowledge)
    assert deals is not None and len(deals) == 1
    engine = HokmEngine()
    engine.state.hands = HandState(
        hands=hands,
        hakem=0,
        leader=0,
        trump=1,
        tricks_won=[5, 6],
        phase=Phase.CARD_PLAY,
    )
    for action in hands[0]:
        assert _reference_value(engine, 0, action, cooperative_partner=True) == 1
        assert _reference_value(engine, 0, action) == -1
    assert public_endgame.guaranteed_endgame_action(knowledge, [44, 46], 44) is None


@pytest.mark.parametrize("budget_name", ["MAX_INFERENCE_NODES", "MAX_SOLVER_NODES"])
def test_endgame_budget_after_partial_progress_cannot_prove_a_sample(
    budget_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge = replace(_forced_position(), void_suits=(frozenset(),) * 4)
    assert public_endgame.guaranteed_endgame_action(knowledge, [12, 51], 12) == 51
    monkeypatch.setattr(public_endgame, budget_name, 10)
    assert public_endgame.guaranteed_endgame_action(knowledge, [12, 51], 12) is None


def test_proof_cache_ignores_history_order_but_not_public_voids() -> None:
    public_endgame._cached_proof.cache_clear()
    knowledge = _forced_position()
    assert public_endgame.guaranteed_endgame_action(knowledge, [12, 51], 12) == 51
    changed_order = replace(knowledge, played_cards=knowledge.played_cards[::-1])
    assert public_endgame.guaranteed_endgame_action(changed_order, [51, 12], 12) == 51
    assert public_endgame._cached_proof.cache_info().hits == 1
    changed_voids = replace(knowledge, void_suits=(frozenset(),) * 4)
    assert public_endgame.guaranteed_endgame_action(changed_voids, [12, 51], 12) == 51
    assert public_endgame._cached_proof.cache_info().misses == 2


def test_immediate_seventh_trick_needs_no_inference_or_remaining_card_play(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge = replace(_forced_position(), tricks_won=(6, 5))
    assert public_endgame._already_secures_hand(knowledge, 51)

    def unexpected_enumeration(*args, **kwargs):
        raise AssertionError("a secured seventh trick needs no hidden-deal proof")

    monkeypatch.setattr(public_endgame, "_feasible_hands", unexpected_enumeration)
    assert public_endgame.guaranteed_endgame_action(knowledge, [12, 51], 51) is None


def test_root_rank_equivalence_includes_cards_already_in_the_current_trick() -> None:
    knowledge = replace(
        _forced_position(),
        hand=frozenset({3, 5}),
        unseen_cards=frozenset({0, 1, 13, 26, 27, 39}),
    )
    assert public_endgame._interchangeable_actions(knowledge, [3, 5])
    assert not public_endgame._interchangeable_actions(
        replace(knowledge, current_trick=((3, 4),)), [3, 5]
    )
    assert not public_endgame._interchangeable_actions(
        replace(knowledge, unseen_cards=knowledge.unseen_cards | {4}), [3, 5]
    )


@pytest.mark.parametrize("seed", range(32))
def test_public_proof_matches_independent_engine_and_deal_enumeration(seed: int) -> None:
    rng = Random(seed)
    cards = rng.sample(range(NUM_CARDS), 8)
    hands = [sorted(cards[index : index + 2]) for index in range(0, 8, 2)]
    leader = rng.randrange(4)
    trump = rng.choice([None, 0, 1, 2, 3])
    score = [5, 6] if seed % 2 else [6, 5]
    engine = HokmEngine()
    engine.state.hands = HandState(
        hands=hands,
        hakem=leader,
        leader=leader,
        trump=trump,
        tricks_won=score,
        phase=Phase.CARD_PLAY,
    )
    # Exercise every partial-trick length, including wrapped seat order.
    for _ in range(seed % 4):
        engine.apply_action(rng.choice(engine.legal_actions()), seat=engine.current_seat())
    state = engine.state.hands
    root = engine.current_seat()
    hand = frozenset(state.hands[root])
    played = frozenset(range(NUM_CARDS)) - {card for held in state.hands for card in held}
    unseen = frozenset(range(NUM_CARDS)) - hand - played
    # Occasionally add truthful public voids; never inspect ownership in
    # production. Otherwise enumerate all capacity-consistent assignments.
    voids = tuple(
        frozenset(suit for suit in range(4) if all(SUIT_OF[c] != suit for c in held))
        if seed % 3 == 0
        else frozenset()
        for held in state.hands
    )
    knowledge = PublicKnowledge(
        seat=root,
        trump=trump,
        hand=hand,
        played_cards=tuple(sorted(played)),
        current_trick=tuple(state.current_trick),
        void_suits=voids,
        unseen_cards=unseen,
        tricks_won=(score[root % 2], score[1 - root % 2]),
    )
    reference = _reference_deals(knowledge)
    inferred = public_endgame._feasible_hands(knowledge)
    assert inferred is not None
    assert set(inferred) == {
        tuple(sum(1 << card for card in held) for held in deal) for deal in reference
    }
    legal = engine.legal_actions()
    baseline = GreedyPolicy()._heuristic_with_knowledge(knowledge, legal)
    guaranteed = {}
    for action in legal:
        guaranteed[action] = True
        for deal in reference:
            variant = deepcopy(engine)
            variant.state.hands.hands = deal
            if _reference_value(variant, root, action) != 1:
                guaranteed[action] = False
                break
    expected = (
        None
        if guaranteed[baseline]
        else next((action for action in legal if guaranteed[action]), None)
    )
    assert public_endgame.guaranteed_endgame_action(knowledge, legal, baseline) == expected
