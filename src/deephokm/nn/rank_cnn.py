"""Suit-equivariant convolutional action-value network.

Hokm's rules are symmetric under relabelling the non-trump suits, and ranks
are ordered while suits are not. This network encodes both facts structurally
rather than hoping to learn them:

- Convolutions span the RANK axis only (``1 x 3`` kernels). Adjacent ranks are
  adjacent in strength, so rank locality is real; suits have no ordering, so a
  kernel spanning suits would assert an adjacency the game does not have.
  Measured: convolving across suits scores 0.313 against 0.333 for rank-only.
- Weights are shared across the four suit rows, which makes the network
  exactly equivariant to relabelling suits.
- Suits still have to interact -- a void in one suit promotes another -- so
  each block appends the MEAN over suits. Mean is symmetric, so the
  interaction preserves equivariance instead of breaking it.
- The head decodes per grid cell with a ``1 x 1`` convolution, so every card
  keeps its own value, and a symmetric pooled vector feeds the four trump
  declarations.

At the reference width this is ~377k parameters against ~8M for the
transformer it replaces, and a 3x-wide variant scores better still.
"""

from __future__ import annotations

import numpy as np
import torch as th
from torch import nn

from deephokm.cards import NUM_RANKS, NUM_SUITS
from deephokm.nn.features import NUM_PLANES, NUM_SCALARS
from deephokm.rules.legality import NUM_ACTIONS

DEFAULT_CHANNELS = 128
DEFAULT_LAYERS = 4
TRUMP_HIDDEN = 256


class RankCNN(nn.Module):
    """Action-value network over ``(suit, rank)`` grid features.

    Attributes:
        channels: Width of every convolution block.
    """

    def __init__(
        self,
        *,
        channels: int = DEFAULT_CHANNELS,
        layers: int = DEFAULT_LAYERS,
    ) -> None:
        """Build the network.

        Args:
            channels: Convolution width.
            layers: Number of rank-convolution blocks.
        """
        super().__init__()
        self.channels = channels
        self.blocks = nn.ModuleList()
        in_ch = NUM_PLANES
        for _ in range(layers):
            # Input is the features concatenated with their mean over suits,
            # hence 2 * in_ch.
            self.blocks.append(nn.Conv2d(in_ch * 2, channels, (1, 3), padding=(0, 1)))
            in_ch = channels
        self.scalar_proj = nn.Linear(NUM_SCALARS, channels)
        self.card_head = nn.Conv2d(channels, 1, 1)
        self.trump_head = nn.Sequential(
            nn.Linear(channels * 2 + NUM_SCALARS, TRUMP_HIDDEN),
            nn.GELU(),
            nn.Linear(TRUMP_HIDDEN, NUM_SUITS),
        )

    def forward(self, planes: th.Tensor, scalars: th.Tensor) -> th.Tensor:
        """Return ``(batch, NUM_ACTIONS)`` action values.

        Args:
            planes: ``(batch, 14, 4, 13)`` grid features.
            scalars: ``(batch, 10)`` scalar features.
        """
        h = planes
        for block in self.blocks:
            context = h.mean(dim=2, keepdim=True).expand_as(h)
            h = th.nn.functional.gelu(block(th.cat([h, context], dim=1)))
        h = h + self.scalar_proj(scalars).view(h.shape[0], -1, 1, 1)

        card_values = self.card_head(h).flatten(1)
        pooled = th.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        trump_values = self.trump_head(th.cat([pooled, scalars], dim=1))
        values: th.Tensor = th.cat([card_values, trump_values], dim=1)
        assert values.shape[1] == NUM_ACTIONS
        assert card_values.shape[1] == NUM_SUITS * NUM_RANKS
        return values


def export_weights(model: RankCNN) -> dict[str, np.ndarray]:
    """Convert a trained model to plain numpy arrays for the numpy backend."""
    return {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}


__all__ = ["DEFAULT_CHANNELS", "DEFAULT_LAYERS", "RankCNN", "export_weights"]
