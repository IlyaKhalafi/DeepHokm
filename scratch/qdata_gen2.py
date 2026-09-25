"""Q-data generation: for each controlled decision, record per-action mean
outcome over K sampled worlds (near-exact Q labels) alongside the observation.

Shard worker: worker w of W over N matches.

Checkpoints its shard every CHECKPOINT_EVERY matches. At K=3072 a shard needs
~36h, so an all-or-nothing write at the end means days with nothing usable and
total loss if the run is interrupted; incremental dumps make the dataset grow
continuously and survive a restart."""
import os, sys, pickle, random
sys.path.insert(0, '/home/ubuntu8/ilya/research/DeepHokm/src')
import numpy as np
from deephokm.policies.legal_depth_search import LegalDepthSearchPolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.env.spaces import mask_for, observation_for
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase

w, W, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
CHECKPOINT_EVERY = 5
# Optional 5th/6th args let a second pool run on disjoint seeds with its own
# output prefix, so every core can generate without two jobs colliding.
PREFIX = sys.argv[5] if len(sys.argv) > 5 else "q3072"
SEED_BASE = int(sys.argv[6]) if len(sys.argv) > 6 else 28000000
OUT = f"/home/ubuntu8/ilya/research/DeepHokm/scratch/{PREFIX}_{w}.pkl"


def dump(records, done):
    """Write the shard so far; atomic via rename so a reader never sees half."""
    if not records:
        return
    keys = records[0][0].keys()
    payload = {
        "obs": {k: np.stack([r[0][k] for r in records]) for k in keys},
        "masks": np.stack([r[1] for r in records]),
        "legals": [r[2] for r in records],
        "qvals": [r[3] for r in records],
        "matches_done": done,
    }
    with open(OUT + ".tmp", "wb") as f:
        pickle.dump(payload, f)
    os.replace(OUT + ".tmp", OUT)
engine = HokmEngine()
greedy = GreedyPolicy()
done = 0
records = []  # (obs, mask, q_values list aligned to legal actions, legal actions)

for i in range(w, N, W):
    seed = SEED_BASE + i
    team = i % 2
    search = LegalDepthSearchPolicy(n_samples=K, search_depth=2, seed=seed)
    engine.start_match(seed=seed)
    controlled = {team, team + 2}
    while engine.state.winner is None:
        seat = engine.current_seat()
        hands = engine.state.hands
        if hands.phase is Phase.TRUMP_CALL:
            legal = engine.legal_actions(seat)
            obs = observation_for(hands, seat, engine.state.game_points)
            engine.apply_action(greedy.act(obs, mask_for(legal)), seat=seat)
            continue
        if seat in controlled:
            legal = engine.legal_actions(seat)
            if len(legal) > 1:
                obs = observation_for(hands, seat, engine.state.game_points)
                # Re-implement the scoring loop to capture per-action values
                from deephokm.policies.search import sample_determinized_hands
                from deephokm.cards import NUM_CARDS
                from deephokm.rules.state import team_of
                own_hand = hands.hands[seat]
                seen = set(hands.played) | set(own_hand)
                unseen = [c for c in range(NUM_CARDS) if c not in seen]
                sizes = [len(hands.hands[s]) for s in range(4)]
                totals = dict.fromkeys(legal, 0.0)
                n_samp = K
                for _ in range(n_samp):
                    samp = sample_determinized_hands(
                        seat, own_hand, unseen, sizes,
                        voids=search.voids.voids, rng=search._rng)
                    for a in legal:
                        totals[a] += search._score_action(
                            engine, seat, a, team, samp, random.Random())
                qvals = np.array([totals[a] / n_samp for a in legal], dtype=np.float32)
                records.append((obs, mask_for(legal), np.array(legal), qvals))
                action = max(legal, key=lambda a: totals[a])
            else:
                action = legal[0]
        else:
            legal = engine.legal_actions(seat)
            obs = observation_for(hands, seat, engine.state.game_points)
            action = greedy.act(obs, mask_for(legal))
        led = hands.current_trick[0][1] // 13 if hands.current_trick else None
        outcome = engine.apply_action(action, seat=seat)
        if outcome.card is not None:
            search.observe(seat, outcome.card, led)
        if outcome.hand_complete:
            search.reset_hand()
    done += 1
    if done % CHECKPOINT_EVERY == 0:
        dump(records, done)
        print(f"shard {w}: {done} matches, {len(records)} decisions (checkpointed)", flush=True)

dump(records, done)
print(f"shard {w}: saved {len(records)} decisions from {done} matches", flush=True)
