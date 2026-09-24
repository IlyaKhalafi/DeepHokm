"""Is 0.35 teacher agreement a model limit or a label-noise/metric artifact?

Answers, in order:
  - what does random-legal score (the floor the 0.35 must be judged against)
  - how often is the teacher's own top action a near-tie
  - train vs val argmax accuracy (data-limited or optimization-broken)
  - top-1/2/3 accuracy and Spearman rho over legal actions
  - accuracy stratified by the teacher's Q-gap (Q_1 - Q_2)
  - normalized regret: how much teacher-Q the student actually gives up
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

MIN_CHOICES = 2       # a decision with one legal action carries no signal
TIE_EPS = 1e-9        # teacher Q values closer than this are an exact tie


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy dependency."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan

files = sorted(
    glob.glob("/home/ubuntu8/ilya/research/DeepHokm/scratch/qdata_*.pkl"),
    key=lambda p: int(p.split("_")[-1].split(".")[0]),
)
data = []
for f in files:
    with open(f, "rb") as fh:
        data.append(pickle.load(fh))
obs = {k: np.concatenate([d["obs"][k] for d in data]) for k in data[0]["obs"]}
masks = np.concatenate([d["masks"] for d in data])
qvals = [q for d in data for q in d["qvals"]]
legals = [lg for d in data for lg in d["legals"]]
n = len(qvals)
n_train = sum(len(d["qvals"]) for d in data[:6])

n_legal = np.array([len(lg) for lg in legals])
print(f"decisions={n}  train={n_train}  val={n - n_train}")
print(f"mean legal actions={n_legal.mean():.2f}  median={np.median(n_legal):.0f}")
print(f"RANDOM-LEGAL baseline (val)={np.mean(1.0 / n_legal[n_train:]):.4f}")

gaps, ties = [], 0
for q in qvals:
    if len(q) < MIN_CHOICES:
        gaps.append(np.nan)
        continue
    srt = np.sort(q)[::-1]
    gaps.append(srt[0] - srt[1])
    if srt[0] - srt[1] < TIE_EPS:
        ties += 1
gaps = np.array(gaps)
print(f"exact ties for the teacher's top action: {ties}/{n} = {ties / n:.3f}")
spread = np.array([np.ptp(q) if len(q) >= MIN_CHOICES else np.nan for q in qvals])
with np.errstate(invalid="ignore"):
    rel = gaps / np.where(spread > 0, spread, np.nan)
print(f"median Q-gap={np.nanmedian(gaps):.4f}  median gap/spread={np.nanmedian(rel):.3f}")

# --- model side ---
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
        batch = {k: v[idx].to(device) for k, v in obs_t.items()}
        preds[s : s + len(idx)] = model(batch).cpu().numpy()


def report(lo, hi, name):
    top1 = top2 = top3 = tot = 0
    regrets, rhos = [], []
    for i in range(lo, hi):
        lg, q, p = legals[i], qvals[i], preds[i][legals[i]]
        if len(lg) < MIN_CHOICES:
            continue
        order = np.argsort(p)[::-1]
        best = int(np.argmax(q))
        top1 += order[0] == best
        top2 += best in order[:2]
        top3 += best in order[:3]
        sp = np.ptp(q)
        regrets.append((q[best] - q[order[0]]) / sp if sp > 0 else 0.0)
        if len(lg) > MIN_CHOICES:
            rhos.append(_spearman(p, q))
        tot += 1
    print(
        f"{name}: n={tot} top1={top1 / tot:.4f} top2={top2 / tot:.4f} "
        f"top3={top3 / tot:.4f} mean_norm_regret={np.mean(regrets):.4f} "
        f"spearman={np.nanmean(rhos):.4f}"
    )


report(0, n_train, "TRAIN")
report(n_train, n, "VAL  ")

print("\nval accuracy stratified by teacher Q-gap decile (gap/spread):")
vi = np.arange(n_train, n)
vi = vi[[len(legals[i]) >= MIN_CHOICES for i in vi]]
rv = rel[vi]
edges = np.nanpercentile(rv, np.arange(0, 101, 20))
for b in range(5):
    sel = vi[(rv >= edges[b]) & (rv <= edges[b + 1])]
    if len(sel) == 0:
        continue
    acc = np.mean(
        [int(np.argmax(preds[i][legals[i]])) == int(np.argmax(qvals[i])) for i in sel]
    )
    rnd = np.mean([1.0 / len(legals[i]) for i in sel])
    print(
        f"  quintile {b + 1} (gap/spread {edges[b]:.3f}-{edges[b + 1]:.3f}): "
        f"n={len(sel):5d} acc={acc:.4f} random-legal={rnd:.4f}"
    )
