import sys, pickle
sys.path.insert(0, '/home/ubuntu8/ilya/research/DeepHokm/src')
import numpy as np
from sb3_contrib import MaskablePPO
from deephokm.env.hokm_env import HokmEnv
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.rules.state import NUM_SEATS
from deephokm.training.behavioral_cloning import train_bc, BCDataset
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

with open("/home/ubuntu8/ilya/research/DeepHokm/scratch/distill_full.pkl", "rb") as f:
    d = pickle.load(f)
obs_list = [dict(zip(d["obs"].keys(), vals)) for vals in zip(*d["obs"].values())]
ds = BCDataset(observations=obs_list, masks=d["masks"], actions=d["actions"],
               match_ids=d["match_ids"])
print(f"dataset: {len(ds)} decisions", flush=True)

env = HokmEnv(seat=0, opponents=[GreedyPolicy() for _ in range(NUM_SEATS)])
model = MaskablePPO(policy=HokmMaskablePolicy, env=env, device="cuda", seed=42, verbose=0)
acc = train_bc(model.policy, ds, epochs=10, batch_size=1024, lr=5e-4, val_fraction=0.1, seed=42)
print(f"final val accuracy: {acc:.4f}", flush=True)
model.save("checkpoints/distill_full.zip")
print("saved checkpoints/distill_full.zip", flush=True)
