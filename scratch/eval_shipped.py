"""Win rate of the SHIPPED policy, on seeds never used for tuning.

scratch/eval_rankcnn_hybrid.py and deephokm.policies.numpy_hybrid implement the
same idea in separate code paths. A win rate measured only by the harness could
be an artifact of the harness, so the number that gets reported has to come
from the module that actually ships, loading the actual .npz weights.

Seeds are disjoint from every earlier evaluation: reusing seeds that guided the
choice of checkpoint and verify-K would report a tuned number as a held-out one.

argv: shard n_shards matches verify_k top_m weights_npz [seed_base] [mode] [start]

``mode`` is 0 for top-M pruning, 1 for budget allocation, 2 for sequential
elimination.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu8/ilya/research/DeepHokm/src")

from deephokm.env.spaces import mask_for, observation_for  # noqa: E402
from deephokm.policies.greedy_policy import GreedyPolicy  # noqa: E402
from deephokm.policies.numpy_hybrid import NumpyHybridPolicy  # noqa: E402
from deephokm.rules.engine import HokmEngine  # noqa: E402
from deephokm.rules.state import Phase  # noqa: E402

DEFAULT_SEED_BASE = 77_000_000  # disjoint from all tuning evaluations
ARG_SEED_BASE = 6   # argv index past which an explicit base was given
ARG_MODE = 7        # ... and a decision-mode selector
ARG_START = 8       # ... and a first match index, for extending a finished run
MODE_ALLOCATE, MODE_ELIMINATE = 1, 2

shard, n_shards, matches, verify_k, top_m = (int(x) for x in sys.argv[1:6])
weights = Path(sys.argv[6])
seed_base = (
    int(sys.argv[ARG_SEED_BASE + 1])
    if len(sys.argv) > ARG_SEED_BASE + 1
    else DEFAULT_SEED_BASE
)
mode = int(sys.argv[ARG_MODE + 1]) if len(sys.argv) > ARG_MODE + 1 else 0
allocate = mode == MODE_ALLOCATE
eliminate = mode == MODE_ELIMINATE

greedy = GreedyPolicy()
wins = 0
played = 0

start = int(sys.argv[ARG_START + 1]) if len(sys.argv) > ARG_START + 1 else 0
for index in range(start + shard, matches, n_shards):
    seed = seed_base + index
    team = index % 2
    policy = NumpyHybridPolicy(
        weights, verify_samples=verify_k, top_m=top_m, seed=seed,
        allocate=allocate, eliminate=eliminate,
        workers=int(os.environ.get("DEEPHOKM_SEARCH_WORKERS", "1")),
    )
    engine = HokmEngine()
    engine.start_match(seed=seed)
    controlled = {team, team + 2}
    policy.reset_hand()

    while engine.state.winner is None:
        seat = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(seat)
        if seat in controlled:
            action = policy.decide(engine)
        elif hands.phase is Phase.TRUMP_CALL or len(legal) > 1:
            obs = observation_for(hands, seat, engine.state.game_points)
            action = greedy.act(obs, mask_for(legal))
        else:
            action = legal[0]
        assert action in legal, f"illegal action {action}"
        led = hands.current_trick[0][1] // 13 if hands.current_trick else None
        outcome = engine.apply_action(action, seat=seat)
        if outcome.card is not None:
            policy.observe(seat, outcome.card, led)
        if outcome.hand_complete:
            policy.reset_hand()

    wins += int(engine.state.winner == team)
    played += 1  # noqa: SIM113 (the loop strides shards, not 0..n)
    print(
        f"shard {shard} K={verify_k} M={top_m} mode={mode}: "
        f"{wins}/{played} = {wins / played:.3f}",
        flush=True,
    )
