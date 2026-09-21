"""Late-hand hybrid: search only when <=5 tricks remain, clone (greedy) elsewhere.
Measures how much of teacher's 0.570 survives with ~60% less search."""
import sys
sys.path.insert(0, '/home/ubuntu8/ilya/research/DeepHokm/src')
from deephokm.policies.legal_depth_search import LegalDepthSearchPolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.env.spaces import mask_for, observation_for
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase

N = 200
TRIGGER = 4  # search when tricks_played >= 13 - TRIGGER
wins = 0
for i in range(N):
    seed = 500000 + i
    team = i % 2
    engine = HokmEngine()
    search = LegalDepthSearchPolicy(n_samples=48, search_depth=2, seed=seed)
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
        late = 13 - len(hands.hands[seat]) >= 13 - TRIGGER
        if seat in controlled and late:
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
    if (i+1) % 20 == 0: print(f"{i+1}/{N}: {wins/(i+1):.3f}", flush=True)
print(f"late-hand hybrid: {wins}/{N} = {wins/N:.3f}", flush=True)
