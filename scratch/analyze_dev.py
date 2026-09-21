"""Count how often search deviates from greedy in distill data."""
import sys, pickle
sys.path.insert(0, '/home/ubuntu8/ilya/research/DeepHokm/src')
import numpy as np
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.env.spaces import mask_for

with open("/home/ubuntu8/ilya/research/DeepHokm/scratch/distill_full.pkl", "rb") as f:
    d = pickle.load(f)
greedy = GreedyPolicy()
obs_list = [dict(zip(d["obs"].keys(), vals)) for vals in zip(*d["obs"].values())]
dev = 0
n = len(obs_list)
for i in range(n):
    g = greedy.act(obs_list[i], d["masks"][i])
    dev += int(g != d["actions"][i])
print(f"total {n}, search deviates from greedy: {dev} ({dev/n:.3f})")
