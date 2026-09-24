"""Numpy RankCNN must match torch bit-for-bit-ish, and fit the 1s budget.

Parity is the gate: a numpy port that silently disagrees with the trained
weights would quietly change play quality. Latency is the second gate: the
shipped player runs this per decision.
"""
import sys
import time

import numpy as np
import torch as th

sys.path.insert(0, "/home/ubuntu8/ilya/research/DeepHokm/scratch")
from arch_sweep import ARMS, _build, load_features  # noqa: E402
from rankcnn_numpy import RankCNNNumpy  # noqa: E402

SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
# float32 conv order differs between backends; 1e-3 is far below the gap that
# could flip an argmax, and the argmax check below is the real gate.
MAX_ABS_DIFF = 1e-3
LATENCY_BUDGET_S = 1.0

th.manual_seed(0)
torch_model = _build("rank_cnn", ARMS["rank_cnn"], SCALE).eval()
params = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}
np_model = RankCNNNumpy(params)

planes, scalars, _, _, _ = load_features()
batch_planes = planes[:64]
batch_scalars = scalars[:64]

with th.no_grad():
    ref = torch_model(th.as_tensor(batch_planes), th.as_tensor(batch_scalars)).numpy()
got = np_model(batch_planes, batch_scalars)

gap = np.abs(ref - got).max()
rel = gap / max(np.abs(ref).max(), 1e-9)
print(f"scale={SCALE} params={sum(v.size for v in params.values()):,}")
print(f"max abs diff = {gap:.3e}   relative = {rel:.3e}")
agree = (ref.argmax(1) == got.argmax(1)).mean()
print(f"argmax agreement over {len(ref)} states = {agree:.4f}")
assert gap < MAX_ABS_DIFF, f"numpy port disagrees with torch: {gap}"
assert agree == 1.0, "argmax differs between backends"

# Latency: one decision = batch of 1, which is how the player calls it.
one_p, one_s = batch_planes[:1], batch_scalars[:1]
np_model(one_p, one_s)                      # warm up
t0 = time.perf_counter()
REPS = 50
for _ in range(REPS):
    np_model(one_p, one_s)
per = (time.perf_counter() - t0) / REPS
print(f"numpy latency = {per * 1000:.2f} ms per forward pass (budget 1000 ms)")
assert per < LATENCY_BUDGET_S, "over the 1s budget"
print("PARITY + LATENCY OK")
