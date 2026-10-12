"""Pure-numpy inference for the suit-equivariant Q-network.

Playing against the trained model must not require PyTorch. Training and
weight conversion use torch; this module needs nothing but numpy, so a
deployment installs numpy alone.

The forward pass mirrors :class:`~deephokm.nn.rank_cnn.RankCNN` layer for
layer, and :mod:`tests.test_numpy_qnet` gates the two against each other --
a port that drifts would silently change how the model plays.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from deephokm.cards import NUM_RANKS, NUM_SUITS
from deephokm.rules.legality import NUM_ACTIONS

_SQRT2 = math.sqrt(2.0)
_KERNEL_WIDTH = 3
GRID_INPUT_DIMENSIONS = 4

# numpy ships no erf, and calling math.erf once per activation was over half of
# a forward pass. The table holds exact math.erf values on a grid fine enough
# that linear interpolation is off by under 5e-9 relative -- an order of
# magnitude below what a float32 can represent. The step is a power of two so
# scaling a float32 onto the grid is exact, and erf(6) is already 1.0 in
# float64, so clamping there loses nothing.
_ERF_STEPS_PER_UNIT = 8192
_ERF_LIMIT = 6 * _ERF_STEPS_PER_UNIT
_ERF_TABLE = np.array([math.erf(i / _ERF_STEPS_PER_UNIT) for i in range(_ERF_LIMIT + 2)])


def _erf(x: np.ndarray) -> np.ndarray:
    """Elementwise erf, interpolated from ``_ERF_TABLE`` (odd, so |x| suffices)."""
    # fmin, not minimum: a NaN must still index the table. It reaches the
    # output regardless, through gelu's own multiplication by x.
    position = np.fmin(np.abs(x) * _ERF_STEPS_PER_UNIT, _ERF_LIMIT)
    index = position.astype(np.intp)
    low = _ERF_TABLE[index]
    value = low + (position - index) * (_ERF_TABLE[index + 1] - low)
    out: np.ndarray = np.copysign(value, x).astype(np.float32)
    return out


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU in torch's exact (erf) form.

    The tanh approximation differs by ~1e-3, which is enough to reorder two
    near-equal action values, so the exact form is used to keep the numpy and
    torch paths interchangeable.
    """
    out: np.ndarray = 0.5 * x * (1.0 + _erf(x / _SQRT2))
    return out


def _conv_over_ranks(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Convolve along the rank axis with weights shared across suits.

    Args:
        x: ``(batch, in_channels, 4, 13)``.
        weight: ``(out_channels, in_channels, 1, 3)``.
        bias: ``(out_channels,)``.

    Returns:
        ``(batch, out_channels, 4, 13)``.
    """
    batch, in_ch, suits, ranks = x.shape
    out_ch = weight.shape[0]
    padded = np.zeros((batch, in_ch, suits, ranks + 2), dtype=np.float32)
    padded[:, :, :, 1:-1] = x
    windows = np.lib.stride_tricks.sliding_window_view(padded, _KERNEL_WIDTH, axis=3)
    columns = windows.transpose(0, 2, 3, 1, 4).reshape(batch * suits * ranks, in_ch * _KERNEL_WIDTH)
    flat = columns @ weight.reshape(out_ch, in_ch * _KERNEL_WIDTH).T + bias
    out: np.ndarray = flat.reshape(batch, suits, ranks, out_ch).transpose(0, 3, 1, 2)
    return out


def _conv_pointwise(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """1x1 convolution: an independent linear map at every grid cell."""
    batch, in_ch, suits, ranks = x.shape
    out_ch = weight.shape[0]
    flat = x.transpose(0, 2, 3, 1).reshape(-1, in_ch)
    mapped = flat @ weight.reshape(out_ch, in_ch).T + bias
    out: np.ndarray = mapped.reshape(batch, suits, ranks, out_ch).transpose(0, 3, 1, 2)
    return out


class NumpyQNet:
    """Numpy forward pass over a converted RankCNN weight dictionary.

    Attributes:
        params: Layer name to weight array, as produced by
            :func:`convert_checkpoint`.
    """

    def __init__(self, params: dict[str, np.ndarray]) -> None:
        """Create the network from converted weights.

        Args:
            params: Weight arrays keyed by the torch parameter names.

        Raises:
            ValueError: If no convolution blocks are present, which means the
                weights are not a RankCNN checkpoint.
        """
        self.params = params
        self._n_blocks = sum(
            1 for key in params if key.startswith("blocks.") and key.endswith(".weight")
        )
        if self._n_blocks == 0:
            raise ValueError("weights contain no 'blocks.*' convolutions; not a RankCNN checkpoint")
        first_channels = params["blocks.0.weight"].shape[1]
        if first_channels % 2:
            raise ValueError("first convolution requires paired card/context channels")
        self.input_planes = first_channels // 2

    def __call__(self, planes: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        """Return action values.

        Args:
            planes: ``(batch, 14, 4, 13)`` grid features.
            scalars: ``(batch, 10)`` scalar features.

        Returns:
            ``(batch, NUM_ACTIONS)`` action values; the first 52 entries are
            per-card, the last four are trump declarations.
        """
        h = np.ascontiguousarray(planes, dtype=np.float32)
        if h.ndim != GRID_INPUT_DIMENSIONS or h.shape[1:] != (
            self.input_planes,
            NUM_SUITS,
            NUM_RANKS,
        ):
            raise ValueError(f"network expects {self.input_planes} input planes on a 4x13 grid")
        for i in range(self._n_blocks):
            context = np.broadcast_to(h.mean(axis=2, keepdims=True), h.shape)
            h = gelu(
                _conv_over_ranks(
                    np.concatenate([h, context], axis=1),
                    self.params[f"blocks.{i}.weight"],
                    self.params[f"blocks.{i}.bias"],
                )
            )
        scalars = np.ascontiguousarray(scalars, dtype=np.float32)
        projected = scalars @ self.params["scalar_proj.weight"].T + self.params["scalar_proj.bias"]
        h = h + projected[:, :, None, None]

        cards = _conv_pointwise(h, self.params["card_head.weight"], self.params["card_head.bias"])
        card_values = cards.reshape(h.shape[0], NUM_SUITS * NUM_RANKS)

        pooled = np.concatenate([h.mean(axis=(2, 3)), h.max(axis=(2, 3))], axis=1)
        hidden = gelu(
            np.concatenate([pooled, scalars], axis=1) @ self.params["trump_head.0.weight"].T
            + self.params["trump_head.0.bias"]
        )
        trump_values = (
            hidden @ self.params["trump_head.2.weight"].T + self.params["trump_head.2.bias"]
        )
        values: np.ndarray = np.concatenate([card_values, trump_values], axis=1)
        assert values.shape[1] == NUM_ACTIONS
        return values


def save_weights(params: dict[str, np.ndarray], path: Path) -> None:
    """Write converted weights to a ``.npz`` archive."""
    np.savez(str(path), **params)  # type: ignore[arg-type]


def load_weights(path: Path) -> dict[str, np.ndarray]:
    """Read weights written by :func:`save_weights`. No torch involved."""
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


__all__ = ["NumpyQNet", "gelu", "load_weights", "save_weights"]
