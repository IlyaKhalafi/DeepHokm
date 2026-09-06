"""The transformer feature extractor for Hokm observations.

Tokenizes the Dict observation (see :mod:`deephokm.nn.tokenizer`) into a
card-token sequence plus context tokens, runs a small pre-LayerNorm
transformer encoder over them, and attention-pools the real tokens into a
fixed-size feature vector for the policy and value heads.

Reference defaults (deviations must be justified in the docstring and
reflected in the dumped training config): d_model=128, nhead=4, num_layers=3,
dim_feedforward=512, GELU, dropout=0.0 (dropout hurts RL), bidirectional
attention with a padding mask, single learned attention-pooling query.
"""

from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from deephokm.cards import PAD_TOKEN
from deephokm.nn.tokenizer import (
    CONTEXT_VOCABS,
    CTX_COLUMNS,
    NUM_CARD_TOKENS,
    NUM_CONTEXT_TOKENS,
    NUM_POSITION_SLOTS,
    NUM_TYPES,
    TokenizedObservation,
    tokenize_tensor_batch,
)


class HokmTransformerExtractor(BaseFeaturesExtractor):
    """Transformer feature extractor over the Hokm Dict observation.

    The observation is tokenized into up to 13 hand tokens, up to 4 trick
    tokens in play order, up to 48 history tokens in recency order, and 5
    context tokens (trump, phase, tricks, points, seat). Card tokens share one
    embedding table (52 cards + PAD); each context slot has its own embedding
    over its bounded value range. Learned type embeddings separate the four
    groups; learned positional embeddings mark order-sensitive slots (trick
    play order, history recency). Hand tokens carry no positional embedding —
    the hand is a set, so permuting hand tokens cannot change the pooled
    features.

    The history group spans the whole hand rather than the 13 most recent
    plays: knowing which cards are gone is the central read in a
    trick-taking game, and truncation would hide most of it.

    The backbone is a pre-LayerNorm (``norm_first``) transformer encoder with
    bidirectional attention and a padding mask, followed by attention pooling
    with a single learned query. ``features_dim`` equals ``d_model`` by
    default.

    Defaults: d_model=128, nhead=4, num_layers=3, dim_feedforward=512, GELU,
    dropout=0.0.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        features_dim: int | None = None,
    ) -> None:
        """Create the extractor.

        Args:
            observation_space: The environment's Dict observation space.
            d_model: Token embedding width.
            nhead: Attention heads.
            num_layers: Transformer encoder layers.
            dim_feedforward: Width of the feed-forward blocks.
            features_dim: Output width; defaults to ``d_model``. Any other
                value is rejected: ``_pool`` always returns a ``d_model``-wide
                vector, so a mismatched ``features_dim`` would make SB3 build
                policy/value heads sized for an input the forward pass never
                produces, failing opaquely on the first call instead of here.

        Raises:
            ValueError: If ``features_dim`` is given and differs from
                ``d_model``.
        """
        if features_dim is not None and features_dim != d_model:
            raise ValueError(
                f"features_dim ({features_dim}) must equal d_model ({d_model}); "
                "the pooled output is always d_model wide"
            )
        super().__init__(observation_space, features_dim or d_model)
        self.d_model = d_model

        self.card_embedding = th.nn.Embedding(NUM_CARD_TOKENS, d_model)
        self.type_embedding = th.nn.Embedding(NUM_TYPES, d_model)
        self.position_embedding = th.nn.Embedding(NUM_POSITION_SLOTS, d_model)
        self.context_embeddings = th.nn.ModuleList(
            [th.nn.Embedding(vocab, d_model) for vocab in CONTEXT_VOCABS]
        )

        self.encoder = th.nn.TransformerEncoder(
            th.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation="gelu",
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=th.nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.pool_query = th.nn.Parameter(th.zeros(1, 1, d_model))
        th.nn.init.normal_(self.pool_query, std=0.02)
        # Context slot columns are fixed; keep them on-device as a buffer so
        # the hot path never rebuilds the tensor.
        self.register_buffer("ctx_columns", th.tensor(CTX_COLUMNS, dtype=th.long), persistent=False)

    def forward(self, observations: dict[str, th.Tensor] | th.Tensor) -> th.Tensor:
        """Extract pooled features for a batch of observations.

        Args:
            observations: A dict of tensors as produced by SB3's preprocessing
                of the Dict observation space.

        Returns:
            ``(batch, features_dim)`` pooled features.
        """
        if not isinstance(observations, dict):
            raise TypeError(
                "HokmTransformerExtractor expects the Dict observation; got "
                f"{type(observations).__name__}"
            )
        tokens = tokenize_tensor_batch(observations)

        x = self._embed(tokens)
        x = self.encoder(x, src_key_padding_mask=~tokens.padding_mask)
        return self._pool(x, tokens.padding_mask)

    def _embed(self, tokens: TokenizedObservation) -> th.Tensor:
        """Combine card, type, position, and context embeddings.

        Card slots take the shared card embedding; context slots (whose
        ``tokens`` entry is a value index, not a card id) take their per-slot
        embedding. Type and positional embeddings are added on top.
        """
        # Non-card slots must not index the card table (their token entry is a
        # context value index, not a card id); route them to PAD first.
        card_ids = th.where(tokens.is_card, tokens.tokens, PAD_TOKEN)
        x = self.card_embedding(card_ids)

        # Context values replace the (zeroed) card embedding at their slots.
        context_stack: th.Tensor = th.stack(
            [
                self.context_embeddings[slot](tokens.context_values[:, slot])
                for slot in range(NUM_CONTEXT_TOKENS)
            ],
            dim=1,
        )  # (batch, NUM_CONTEXT_TOKENS, d_model)
        with_context: th.Tensor = x.index_copy(1, self.ctx_columns, context_stack)

        typed = with_context + self.type_embedding(tokens.type_ids)
        positioned: th.Tensor = typed + self.position_embedding(tokens.positions)
        return positioned

    def _pool(self, x: th.Tensor, padding_mask: th.Tensor) -> th.Tensor:
        """Attention-pool the real tokens with a single learned query."""
        query = self.pool_query.expand(x.shape[0], -1, -1)
        attn = th.nn.functional.scaled_dot_product_attention(
            query,
            x,
            x,
            attn_mask=padding_mask.unsqueeze(1),
        )
        return attn.squeeze(1)
