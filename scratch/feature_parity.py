"""The live feature builder must match the training one exactly.

Training features come from the label pickles in bulk; play builds them one
decision at a time. Any mismatch means the net is fed a different input
distribution at play time than it trained on, which would silently invalidate
every win-rate measurement, so this compares them row by row on real data.
"""
import glob
import pickle
import sys

import numpy as np

sys.path.insert(0, "/home/ubuntu8/ilya/research/DeepHokm/scratch")
from arch_sweep import build_row_features, load_features  # noqa: E402

planes, scalars, masks, _, _ = load_features()

files = sorted(
    glob.glob("/home/ubuntu8/ilya/research/DeepHokm/scratch/qdata_*.pkl"),
    key=lambda p: int(p.split("_")[-1].split(".")[0]),
)
data = []
for f in files:
    with open(f, "rb") as fh:
        data.append(pickle.load(fh))
obs = {k: np.concatenate([d["obs"][k] for d in data]) for k in data[0]["obs"]}

rng = np.random.default_rng(0)
rows = rng.choice(len(planes), size=400, replace=False)
worst_p = worst_s = 0.0
for i in rows:
    row = {k: v[i] for k, v in obs.items()}
    p_, s_ = build_row_features(row, masks[i])
    worst_p = max(worst_p, float(np.abs(p_ - planes[i]).max()))
    worst_s = max(worst_s, float(np.abs(s_ - scalars[i]).max()))
print(f"checked {len(rows)} rows")
print(f"max plane diff  = {worst_p:.3e}")
print(f"max scalar diff = {worst_s:.3e}")
assert worst_p == 0.0, "live planes differ from training planes"
assert worst_s == 0.0, "live scalars differ from training scalars"
print("FEATURE PARITY OK")
