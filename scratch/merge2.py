import pickle, glob
import numpy as np
files = sorted(glob.glob("/home/ubuntu8/ilya/research/DeepHokm/scratch/shard2_*.pkl"),
               key=lambda p: int(p.split("_")[-1].split(".")[0]))
obs_acc = {k: [] for k in pickle.load(open(files[0],"rb"))["obs"].keys()}
masks, acts = [], []
n = 0
for f in files:
    d = pickle.load(open(f, "rb"))
    for k in obs_acc: obs_acc[k].append(d["obs"][k])
    masks.append(d["masks"]); acts.append(d["actions"]); n += len(d["actions"])
out = {"obs": {k: np.concatenate(v) for k, v in obs_acc.items()},
       "masks": np.concatenate(masks), "actions": np.concatenate(acts)}
with open("/home/ubuntu8/ilya/research/DeepHokm/scratch/distill_d3.pkl", "wb") as f:
    pickle.dump(out, f)
print(f"merged {n} D=3 decisions")
