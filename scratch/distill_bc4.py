import sys, pickle
sys.path.insert(0, '/home/ubuntu8/ilya/research/DeepHokm/src')
import numpy as np
from sb3_contrib import MaskablePPO
from deephokm.env.hokm_env import HokmEnv
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.rules.state import NUM_SEATS
from deephokm.training.behavioral_cloning import collect_dataset, train_bc, BCDataset
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

def load(path):
    with open(path, "rb") as f: return pickle.load(f)

d2 = load("/home/ubuntu8/ilya/research/DeepHokm/scratch/distill_full.pkl")
d3 = load("/home/ubuntu8/ilya/research/DeepHokm/scratch/distill_d3.pkl")
greedy = GreedyPolicy()

obs_all, mask_all, act_all, mid_all = [], [], [], []
mid = 0
for d in (d2, d3):
    obs_list = [dict(zip(d["obs"].keys(), vals)) for vals in zip(*d["obs"].values())]
    for i in range(len(obs_list)):
        if greedy.act(obs_list[i], d["masks"][i]) != d["actions"][i]:
            obs_all.append(obs_list[i])
            mask_all.append(d["masks"][i])
            act_all.append(d["actions"][i])
            mid_all.append(mid); mid += 1
print(f"deviation-only examples: {len(obs_all)}", flush=True)

print("collecting bulk greedy data...", flush=True)
bulk = collect_dataset(3000, seed=888)
print(f"bulk: {len(bulk)} plies", flush=True)

# deviation-only (repeated 30x) + bulk
comb_obs = list(bulk.observations)
comb_masks = [m for m in bulk.masks]
comb_actions = [a for a in bulk.actions]
comb_match = list(bulk.match_ids)
for rep in range(30):
    for o, m, a, mi in zip(obs_all, mask_all, act_all, mid_all):
        comb_obs.append(o); comb_masks.append(m); comb_actions.append(a)
        comb_match.append(10_000 + mi)
print(f"combined: {len(comb_obs)}", flush=True)

ds = BCDataset(observations=comb_obs, masks=np.stack(comb_masks),
               actions=np.array(comb_actions, dtype=np.int64),
               match_ids=np.array(comb_match, dtype=np.int64))
env = HokmEnv(seat=0, opponents=[GreedyPolicy() for _ in range(NUM_SEATS)])
model = MaskablePPO(policy=HokmMaskablePolicy, env=env, device="cuda", seed=42, verbose=0)
acc = train_bc(model.policy, ds, epochs=12, batch_size=1024, lr=5e-4, val_fraction=0.05, seed=42)
print(f"final val accuracy: {acc:.4f}", flush=True)
model.save("checkpoints/distill_v4.zip")
print("saved checkpoints/distill_v4.zip", flush=True)
