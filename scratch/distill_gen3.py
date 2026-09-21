"""Full-match distill data: search policy controls both seats vs greedy,
all hands until match end."""
import sys, pickle
sys.path.insert(0, '/home/ubuntu8/ilya/research/DeepHokm/src')
import numpy as np
from deephokm.policies.legal_depth_search import LegalDepthSearchPolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.env.spaces import mask_for, observation_for
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase
from deephokm.cards import NUM_RANKS

N = 700
TARGET = 60_000
engine = HokmEngine()
greedy = GreedyPolicy()
obs_list, mask_list, act_list = [], [], []
for i in range(N):
    seed = 130000 + i
    team = i % 2
    search = LegalDepthSearchPolicy(n_samples=48, search_depth=2, seed=seed)
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
            legal = engine.legal_actions(seat)
            obs_list.append(observation_for(hands, seat, engine.state.game_points))
            mask_list.append(mask_for(legal))
            act_list.append(action)
        else:
            legal = engine.legal_actions(seat)
            obs = observation_for(hands, seat, engine.state.game_points)
            action = greedy.act(obs, mask_for(legal))
        led = None
        if hands.current_trick:
            led = hands.current_trick[0][1] // NUM_RANKS
        outcome = engine.apply_action(action, seat=seat)
        if outcome.card is not None:
            search.observe(seat, outcome.card, led)
        if outcome.hand_complete:
            search.reset_hand()
    if (i + 1) % 25 == 0:
        print(f"{i+1}/{N} matches, {len(act_list)} decisions", flush=True)
    if len(act_list) >= TARGET:
        break
raw = [dict(o) for o in obs_list]
keys = raw[0].keys()
stacked = {k: np.stack([o[k] for o in raw]) for k in keys}
with open("/home/ubuntu8/ilya/research/DeepHokm/scratch/distill_full.pkl", "wb") as f:
    pickle.dump({"obs": stacked, "masks": np.stack(mask_list),
                 "actions": np.array(act_list, dtype=np.int64),
                 "match_ids": np.repeat(np.arange(len(act_list)//200+1), 200)[:len(act_list)]}, f)
print(f"saved {len(act_list)} decisions", flush=True)
