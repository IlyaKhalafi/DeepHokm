"""Shard: K=768 teacher matches, worker w of W."""
import sys
sys.path.insert(0, '/home/ubuntu8/ilya/research/DeepHokm/src')
from deephokm.policies.legal_depth_search import LegalDepthSearchPolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.env.spaces import mask_for, observation_for
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase

w, W, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
wins = 0
n = 0
for i in range(w, N, W):
    seed = 900000 + i
    team = i % 2
    engine = HokmEngine()
    search = LegalDepthSearchPolicy(n_samples=768, search_depth=2, seed=seed)
    greedy = GreedyPolicy()
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
    wins += int(engine.state.winner == team)
    n += 1
    if n % 5 == 0: print(f"shard {w}: {n} matches, {wins}/{n} = {wins/n:.3f}", flush=True)
print(f"SHARD {w} FINAL: {wins}/{n} = {wins/max(1,n):.3f}", flush=True)
