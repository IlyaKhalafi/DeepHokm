"""How much of the teacher's own choice is sampling noise?

Replays the same deals the Q-label data came from and, at each controlled
decision, computes the teacher's per-action Q values TWICE with independent
determinization RNG streams. Two runs of the same teacher on the same state
disagreeing puts a hard ceiling on how well any student can match it: a
student cannot reproduce a coin flip.

Reports, per K:
  argmax agreement      run A's top action == run B's top action
  optimal-set agreement run A's top action is within run B's tied-max set
  tie fraction          how often the teacher's own top action is tied

argv: worker shard, total shards, matches, K
"""
import random
import sys

import numpy as np

sys.path.insert(0, "/home/ubuntu8/ilya/research/DeepHokm/src")
from deephokm.cards import NUM_CARDS
from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.legal_depth_search import LegalDepthSearchPolicy
from deephokm.policies.search import sample_determinized_hands
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase

TIE_EPS = 1e-9
MIN_CHOICES = 2

w, W, N, K = (int(x) for x in sys.argv[1:5])
greedy = GreedyPolicy()
engine = HokmEngine()

stats = {"n": 0, "argmax": 0, "optset": 0, "tie_a": 0, "all_tied": 0,
         "dn": 0, "dargmax": 0, "doptset": 0}


def teacher_q(search, seat, team, legal, rng_seed):
    """Per-action mean value over K determinized worlds, own RNG stream."""
    hands = engine.state.hands
    own = hands.hands[seat]
    seen = set(hands.played) | set(own)
    unseen = [c for c in range(NUM_CARDS) if c not in seen]
    sizes = [len(hands.hands[s]) for s in range(4)]
    rng = random.Random(rng_seed)
    totals = dict.fromkeys(legal, 0.0)
    for _ in range(K):
        samp = sample_determinized_hands(
            seat, own, unseen, sizes, voids=search.voids.voids, rng=rng
        )
        for a in legal:
            totals[a] += search._score_action(engine, seat, a, team, samp, random.Random())
    return np.array([totals[a] / K for a in legal], dtype=np.float64)


for i in range(w, N, W):
    seed = 28000000 + i          # same deals the K=192 label data used
    team = i % 2
    search = LegalDepthSearchPolicy(n_samples=K, search_depth=2, seed=seed)
    engine.start_match(seed=seed)
    controlled = {team, team + 2}
    while engine.state.winner is None:
        seat = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(seat)
        obs = observation_for(hands, seat, engine.state.game_points)
        if hands.phase is Phase.TRUMP_CALL:
            engine.apply_action(greedy.act(obs, mask_for(legal)), seat=seat)
            continue
        if seat in controlled and len(legal) >= MIN_CHOICES:
            qa = teacher_q(search, seat, team, legal, rng_seed=7_000_000 + i)
            qb = teacher_q(search, seat, team, legal, rng_seed=9_000_000 + i)
            a_top, b_top = int(np.argmax(qa)), int(np.argmax(qb))
            stats["n"] += 1
            stats["argmax"] += a_top == b_top
            stats["optset"] += qb[a_top] >= qb.max() - TIE_EPS
            stats["tie_a"] += int((qa >= qa.max() - TIE_EPS).sum()) > 1
            all_tied = int((qa >= qa.max() - TIE_EPS).sum()) == len(legal)
            stats["all_tied"] += all_tied
            if not all_tied:
                # Decisive decisions only. When every action ties, both runs
                # return identical Q arrays and argmax picks the same index
                # for free, which would inflate the agreement figure.
                stats["dn"] += 1
                stats["dargmax"] += a_top == b_top
                stats["doptset"] += qb[a_top] >= qb.max() - TIE_EPS
            action = legal[a_top]
        else:
            action = greedy.act(obs, mask_for(legal)) if len(legal) > 1 else legal[0]
        led = hands.current_trick[0][1] // 13 if hands.current_trick else None
        outcome = engine.apply_action(action, seat=seat)
        if outcome.card is not None:
            search.observe(seat, outcome.card, led)
        if outcome.hand_complete:
            search.reset_hand()
    n, dn = stats["n"], max(stats["dn"], 1)
    print(
        f"shard {w} K={K} after match {i}: n={n} "
        f"argmax_agree={stats['argmax'] / n:.4f} "
        f"optset_agree={stats['optset'] / n:.4f} "
        f"tie={stats['tie_a'] / n:.4f} all_tied={stats['all_tied'] / n:.4f} "
        f"| DECISIVE n={stats['dn']} "
        f"argmax_agree={stats['dargmax'] / dn:.4f} "
        f"optset_agree={stats['doptset'] / dn:.4f}",
        flush=True,
    )
