"""Pure-numpy forward pass for the suit-equivariant RankCNN Q-network.

Lets the shipped player run without PyTorch: torch stays a development
dependency for training, numpy alone is needed to play. The network is small
(377k params at the baseline width), and every layer here is a matmul over a
4x13 grid, so a single decision costs well under the 1s budget.

Layer-for-layer mirror of :class:`scratch.arch_sweep.RankCNN`:
  for each block:  h <- gelu(conv_{1x3}([h ; mean_over_suits(h)]))
  h <- h + scalar_proj(scalars)                      (broadcast over the grid)
  card logits <- conv_{1x1}(h)                       -> 52
  trump logits <- mlp([mean(h) ; max(h) ; scalars])  -> 4
"""
from __future__ import annotations

import math

import numpy as np

_ERF = np.vectorize(math.erf, otypes=[np.float32])

NUM_SUITS, NUM_RANKS, NUM_CARDS, NUM_ACTIONS = 4, 13, 52, 56


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU, matching torch's default erf formulation.

    A tanh approximation drifts from torch by ~1e-3, which is enough to flip
    an argmax between near-equal actions, so this uses the true erf.
    """
    return 0.5 * x * (1.0 + _ERF(x / math.sqrt(2.0)))


def _conv1x3(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Convolve along the rank axis only, weights shared across suits.

    Args:
        x: ``(B, Cin, 4, 13)``.
        weight: ``(Cout, Cin, 1, 3)``.
        bias: ``(Cout,)``.

    Returns:
        ``(B, Cout, 4, 13)``.
    """
    b, cin, ns, nr = x.shape
    cout = weight.shape[0]
    padded = np.zeros((b, cin, ns, nr + 2), dtype=x.dtype)
    padded[:, :, :, 1:-1] = x
    # Sliding windows over ranks: (B, Cin, 4, 13, 3)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 3, axis=3)
    # (B, 4, 13, Cin*3) @ (Cin*3, Cout)
    cols = windows.transpose(0, 2, 3, 1, 4).reshape(b * ns * nr, cin * 3)
    w = weight.reshape(cout, cin * 3).T
    out = cols @ w + bias
    return out.reshape(b, ns, nr, cout).transpose(0, 3, 1, 2)


def _conv1x1(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """1x1 convolution: a per-cell linear map. ``weight`` is ``(Cout,Cin,1,1)``."""
    b, cin, ns, nr = x.shape
    cout = weight.shape[0]
    flat = x.transpose(0, 2, 3, 1).reshape(-1, cin)
    out = flat @ weight.reshape(cout, cin).T + bias
    return out.reshape(b, ns, nr, cout).transpose(0, 3, 1, 2)


class RankCNNNumpy:
    """Numpy RankCNN built from a state dict of numpy arrays."""

    def __init__(self, params: dict[str, np.ndarray]) -> None:
        self.p = params
        self.n_blocks = sum(1 for k in params if k.startswith("blocks.") and k.endswith(".weight"))

    def __call__(self, planes: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        """Return ``(B, 56)`` action logits for float32 inputs."""
        h = planes.astype(np.float32)
        for i in range(self.n_blocks):
            ctx = h.mean(axis=2, keepdims=True)
            ctx = np.broadcast_to(ctx, h.shape)
            both = np.concatenate([h, ctx], axis=1)
            h = gelu(_conv1x3(both, self.p[f"blocks.{i}.weight"], self.p[f"blocks.{i}.bias"]))
        proj = scalars.astype(np.float32) @ self.p["scalar_proj.weight"].T + self.p[
            "scalar_proj.bias"
        ]
        h = h + proj[:, :, None, None]
        card = _conv1x1(h, self.p["card_head.weight"], self.p["card_head.bias"])
        card_logits = card.reshape(h.shape[0], -1)

        pooled = np.concatenate([h.mean(axis=(2, 3)), h.max(axis=(2, 3))], axis=1)
        t = np.concatenate([pooled, scalars.astype(np.float32)], axis=1)
        t = gelu(t @ self.p["trump_head.0.weight"].T + self.p["trump_head.0.bias"])
        trump_logits = t @ self.p["trump_head.2.weight"].T + self.p["trump_head.2.bias"]
        return np.concatenate([card_logits, trump_logits], axis=1)


def load_numpy_params(path: str) -> dict[str, np.ndarray]:
    """Load a torch checkpoint into plain numpy arrays.

    The only torch touch point, and it runs at conversion time, not at play
    time: the resulting dict is saved with ``np.savez`` and the player loads
    that instead.
    """
    import torch as th  # noqa: PLC0415  (conversion-time only, not a play-time dep)

    sd = th.load(path, map_location="cpu", weights_only=True)
    return {k: v.numpy() for k, v in sd.items()}
