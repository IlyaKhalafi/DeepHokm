"""Score the student against the teacher's OPTIMAL SET, not its tie-break.

60% of decisions have two or more actions sharing the teacher's maximum Q.
Argmax accuracy grades the student on which of several equally-valued actions
the teacher happened to list first, which is arbitrary. The honest question is
whether the student picks an action the teacher values at (or near) the max.
"""
import glob
import pickle
import sys

import numpy as np
import torch as th
from torch import nn

sys.path.insert(0, "/home/ubuntu8/ilya/research/DeepHokm/src")
from deephokm.env.hokm_env import HokmEnv
from deephokm.nn.extractor import HokmTransformerExtractor
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.rules.state import NUM_SEATS

MIN_CHOICES = 2
TIE_EPS = 1e-9

files = sorted(
    glob.glob("/home/ubuntu8/ilya/research/DeepHokm/scratch/qdata_*.pkl"),
    key=lambda p: int(p.split("_")[-1].split(".")[0]),
)
data = []
for f in files:
    with open(f, "rb") as fh:
        data.append(pickle.load(fh))
obs = {k: np.concatenate([d["obs"][k] for d in data]) for k in data[0]["obs"]}
qvals = [q for d in data for q in d["qvals"]]
legals = [lg for d in data for lg in d["legals"]]
n = len(qvals)
n_train = sum(len(d["qvals"]) for d in data[:6])

sizes = np.array([int((q >= q.max() - TIE_EPS).sum()) for q in qvals])
nleg = np.array([len(lg) for lg in legals])
print(f"mean |optimal set|={sizes.mean():.2f}  mean |legal|={nleg.mean():.2f}")
print(f"decisions where EVERY legal action is optimal: {(sizes == nleg).mean():.3f}")
print(f"random-legal chance of hitting the optimal set (val)="
      f"{np.mean((sizes / nleg)[n_train:]):.4f}")

env = HokmEnv(seat=0, opponents=[GreedyPolicy() for _ in range(NUM_SEATS)])


class QNet(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        self.extractor = HokmTransformerExtractor(obs_space)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 56))

    def forward(self, ob):
        return self.head(self.extractor(ob))


device = "cuda" if th.cuda.is_available() else "cpu"
model = QNet(env.observation_space).to(device)
model.load_state_dict(th.load(sys.argv[1], weights_only=True))
model.eval()

obs_t = {k: th.as_tensor(v) for k, v in obs.items()}
preds = np.zeros((n, 56), dtype=np.float32)
with th.no_grad():
    for s in range(0, n, 512):
        idx = th.arange(s, min(s + 512, n))
        preds[s : s + len(idx)] = model({k: v[idx].to(device) for k, v in obs_t.items()}).cpu()

for name, lo, hi in (("TRAIN", 0, n_train), ("VAL", n_train, n)):
    hit = chance = tot = 0
    hit_decisive = tot_decisive = 0
    for i in range(lo, hi):
        lg, q = legals[i], qvals[i]
        if len(lg) < MIN_CHOICES:
            continue
        pick = int(np.argmax(preds[i][lg]))
        in_opt = q[pick] >= q.max() - TIE_EPS
        hit += in_opt
        chance += sizes[i] / nleg[i]
        tot += 1
        if sizes[i] < nleg[i]:          # at least one action is strictly worse
            hit_decisive += in_opt
            tot_decisive += 1
    print(
        f"{name}: optimal-set accuracy={hit / tot:.4f} (random-legal={chance / tot:.4f})"
        f"   on decisions that are NOT all-tied: {hit_decisive / tot_decisive:.4f}"
        f" (n={tot_decisive})"
    )
