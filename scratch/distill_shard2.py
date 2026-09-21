"""Shard v2: D=3 search (deeper, more deviations), full matches."""
import sys, pickle
sys.path.insert(0, '/home/ubuntu8/ilya/research/DeepHokm/src')
import numpy as np
from deephokm.policies.legal_depth_search import LegalDepthSearchPolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.env.spaces import mask_for, observation_for
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase

w, W, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
engine = HokmEngine()
greedy = GreedyPolicy()
obs_list, mask_list, act_list = [], [], []
for i in range(w, N, W):
    seed = 400000 + i
    team = i % 2
    search = LegalDepthSearchPolicy(n_samples=48, search_depth=3, seed=seed)
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
            action = search.decide(engine)
            obs_list.append(observation_for(hands, seat, engine.state.game_points))
            mask_list.append(mask_for(engine.legal_actions(seat)))
            act_list.append(action)
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
raw = [dict(o) for o in obs_list]
keys = raw[0].keys()
stacked = {k: np.stack([o[k] for o in raw]) for k in keys}
with open(f"/home/ubuntu8/ilya/research/DeepHokm/scratch/shard2_{w}.pkl", "wb") as f:
    pickle.dump({"obs": stacked, "masks": np.stack(mask_list),
                 "actions": np.array(act_list, dtype=np.int64)}, f)
print(f"shard2 {w}: {len(act_list)} decisions", flush=True)
