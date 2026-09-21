"""Deviation-only BC: train ONLY on the 2081 deviation examples, val split
by deviation index (shuffled), starting from bc_256d weights (greedy core already known)."""
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

def load(path):
    with open(path, "rb") as f: return pickle.load(f)

greedy = GreedyPolicy()
obs_all, mask_all, act_all = [], [], []
for p in ("/home/ubuntu8/ilya/research/DeepHokm/scratch/distill_full.pkl",
          "/home/ubuntu8/ilya/research/DeepHokm/scratch/distill_d3.pkl"):
    d = load(p)
    obs_list = [dict(zip(d["obs"].keys(), vals)) for vals in zip(*d["obs"].values())]
    for i in range(len(obs_list)):
        if greedy.act(obs_list[i], d["masks"][i]) != d["actions"][i]:
            obs_all.append(obs_list[i]); mask_all.append(d["masks"][i]); act_all.append(d["actions"][i])
print(f"deviations: {len(obs_all)}", flush=True)

# pseudo match ids interleaved so train/val split sees mixed deviation types
ids = np.arange(len(obs_all))
rng = np.random.default_rng(0)
rng.shuffle(ids)
ds = BCDataset(observations=obs_all, masks=np.stack(mask_all),
               actions=np.array(act_all, dtype=np.int64), match_ids=ids % 200)

env = HokmEnv(seat=0, opponents=[GreedyPolicy() for _ in range(NUM_SEATS)])
model = MaskablePPO.load("checkpoints/bc_256d.zip", env=env, device="cuda")  # greedy core intact
acc = train_bc(model.policy, ds, epochs=40, batch_size=256, lr=2e-4, val_fraction=0.15, seed=42)
print(f"deviation val accuracy: {acc:.4f}", flush=True)
model.save("checkpoints/dev_only.zip")
print("saved checkpoints/dev_only.zip", flush=True)
