"""Distill v3: mix full-match search data (with rare deviations) + greedy-vs-greedy
BC data (bulk), longer training. Deviations upweighted 20x."""
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

with open("/home/ubuntu8/ilya/research/DeepHokm/scratch/distill_full.pkl", "rb") as f:
    d = pickle.load(f)
obs_list = [dict(zip(d["obs"].keys(), vals)) for vals in zip(*d["obs"].values())]
greedy = GreedyPolicy()
dev_idx = [i for i in range(len(obs_list))
           if greedy.act(obs_list[i], d["masks"][i]) != d["actions"][i]]
print(f"deviations: {len(dev_idx)}", flush=True)

print("collecting greedy-vs-greedy bulk data...", flush=True)
bulk = collect_dataset(3000, seed=777)
print(f"bulk: {len(bulk)} plies", flush=True)

# combine: bulk + distill data, deviations repeated 20x
comb_obs = list(bulk.observations) + obs_list
comb_masks = np.concatenate([bulk.masks, d["masks"]])
comb_actions = np.concatenate([bulk.actions, d["actions"]])
comb_match = np.concatenate([bulk.match_ids, d["match_ids"] + 10_000])
for _ in range(20):
    for i in dev_idx:
        comb_obs.append(obs_list[i])
        comb_masks = np.concatenate([comb_masks, d["masks"][i][None]])
        comb_actions = np.concatenate([comb_actions, [d["actions"][i]]])
        comb_match = np.concatenate([comb_match, [-1]])
print(f"combined: {len(comb_obs)} plies", flush=True)

ds = BCDataset(observations=comb_obs, masks=comb_masks, actions=comb_actions,
               match_ids=comb_match)
env = HokmEnv(seat=0, opponents=[GreedyPolicy() for _ in range(NUM_SEATS)])
model = MaskablePPO(policy=HokmMaskablePolicy, env=env, device="cuda", seed=42, verbose=0)
acc = train_bc(model.policy, ds, epochs=12, batch_size=1024, lr=5e-4, val_fraction=0.05, seed=42)
print(f"final val accuracy: {acc:.4f}", flush=True)
model.save("checkpoints/distill_v3.zip")
print("saved checkpoints/distill_v3.zip", flush=True)
