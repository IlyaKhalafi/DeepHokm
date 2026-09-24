"""Win rate of the RankCNN hybrid against a two-seat greedy opposing team.

Pipeline under test (the deployed shape):
  1. The numpy RankCNN scores every legal action.
  2. Its top-M candidates that differ from greedy's pick are proposed.
  3. Each candidate is verified against greedy's action by the same K-sample
     sign test the pure search uses; the first to clear p <= MAX_P is played.

Top-M rather than top-1 because the verifier exists precisely to tolerate a
wrong nomination: measured top-1 optimal-set accuracy is ~0.61 while top-3 is
far higher, so proposing three candidates converts recall into win rate at
roughly M times the verification cost -- still an order of magnitude below
full search over every legal action.

argv: shard n_shards matches K top_m checkpoint
"""
import random
import sys

import numpy as np

sys.path.insert(0, "/home/ubuntu8/ilya/research/DeepHokm/src")
sys.path.insert(0, "/home/ubuntu8/ilya/research/DeepHokm/scratch")
from arch_sweep import build_row_features  # noqa: E402
from rankcnn_numpy import RankCNNNumpy, load_numpy_params  # noqa: E402

from deephokm.cards import NUM_CARDS  # noqa: E402
from deephokm.env.spaces import mask_for, observation_for  # noqa: E402
from deephokm.policies.greedy_policy import GreedyPolicy  # noqa: E402
from deephokm.policies.legal_depth_search import (  # noqa: E402
    VoidTracker,
    _sign_test_p_value,
)
from deephokm.policies.oracle_search import oracle_ceiling  # noqa: E402
from deephokm.policies.search import (  # noqa: E402
    _clone_for_simulation,
    sample_determinized_hands,
)
from deephokm.rules.engine import HokmEngine  # noqa: E402
from deephokm.rules.state import Phase, team_of  # noqa: E402

w, W, N, K, TOP_M = (int(x) for x in sys.argv[1:6])
CKPT = sys.argv[6]
MAX_P = 0.05

net = RankCNNNumpy(load_numpy_params(CKPT))
greedy = GreedyPolicy()


def nominate(obs, legal):
    """Return the net's legal actions, best first."""
    planes, scalars = build_row_features(obs, mask_for(legal))
    logits = net(planes[None], scalars[None])[0]
    order = np.argsort(logits[legal])[::-1]
    return [int(legal[i]) for i in order]


def rollout_value(engine, seat, action, *, team, samp, rng):
    clone = _clone_for_simulation(engine, seat, samp, rng=rng)
    outcome = clone.apply_action(action, seat=seat)
    if outcome.hand_complete:
        return 1.0 if outcome.hand_winner_team == team else -1.0
    return oracle_ceiling(clone, team, depth=1, rng=rng)


def verify(engine, seat, cand, *, base, team, voids, rng):
    """Sign test: is ``cand`` better than ``base`` over K sampled worlds?"""
    hands = engine.state.hands
    own = hands.hands[seat]
    seen = set(hands.played) | set(own)
    unseen = [c for c in range(NUM_CARDS) if c not in seen]
    sizes = [len(hands.hands[s]) for s in range(4)]
    clone_rng = random.Random()
    w_c = w_b = 0
    for _ in range(K):
        samp = sample_determinized_hands(seat, own, unseen, sizes, voids=voids, rng=rng)
        vc = rollout_value(engine, seat, cand, team=team, samp=samp, rng=clone_rng)
        vb = rollout_value(engine, seat, base, team=team, samp=samp, rng=clone_rng)
        if vc > vb:
            w_c += 1
        elif vb > vc:
            w_b += 1
    return _sign_test_p_value(w_c, w_c + w_b) <= MAX_P


wins = 0
played = 0
for i in range(w, N, W):
    seed = 41000000 + i
    team = i % 2
    rng = random.Random(seed)
    voids = VoidTracker()
    engine = HokmEngine()
    engine.start_match(seed=seed)
    controlled = {team, team + 2}
    while engine.state.winner is None:
        seat = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(seat)
        obs = observation_for(hands, seat, engine.state.game_points)
        if hands.phase is Phase.TRUMP_CALL or len(legal) == 1:
            action = greedy.act(obs, mask_for(legal)) if len(legal) > 1 else legal[0]
        elif seat in controlled:
            g = greedy.act(obs, mask_for(legal))
            action = g
            ranked = nominate(obs, legal)
            for cand in [c for c in ranked[:TOP_M] if c != g]:
                if verify(
                    engine, seat, cand, base=g, team=team_of(seat),
                    voids=voids.voids, rng=rng,
                ):
                    action = cand
                    break
        else:
            action = greedy.act(obs, mask_for(legal))
        led = hands.current_trick[0][1] // 13 if hands.current_trick else None
        outcome = engine.apply_action(action, seat=seat)
        if outcome.card is not None:
            voids.observe(seat, outcome.card, led)
        if outcome.hand_complete:
            voids.reset()
    wins += int(engine.state.winner == team)
    played += 1  # noqa: SIM113 (loop strides shards, not 0..n)
    print(
        f"shard {w} K={K} M={TOP_M}: {wins}/{played} = {wins / played:.3f}",
        flush=True,
    )
