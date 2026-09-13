"""Alternative architecture extractors for comparative benchmark."""

from __future__ import annotations

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from deephokm.nn.tokenizer import TokenizedObservation, tokenize_tensor_batch


class HokmLSTMExtractor(BaseFeaturesExtractor):
    """BiLSTM + attention pooling."""

    def __init__(self, observation_space, d_hidden: int = 256, num_layers: int = 2):
        super().__init__(observation_space, features_dim=d_hidden)
        self.d_hidden = d_hidden
        self.embeddings = nn.Embedding(134, d_hidden)  # tokenizer vocab size
        self.lstm = nn.LSTM(d_hidden, d_hidden, num_layers, batch_first=True, bidirectional=True)
        self.attn_query = nn.Parameter(th.randn(1, 1, 2 * d_hidden))
        self.proj = nn.Linear(2 * d_hidden, d_hidden)

    def forward(self, observations: dict[str, th.Tensor] | th.Tensor) -> th.Tensor:
        if isinstance(observations, dict):
            obs = tokenize_tensor_batch(observations)
        else:
            obs = observations
        tokens = obs.tokens  # (B, T)
        padding_mask = obs.padding_mask  # (B, T) boolean
        # Ensure tokens are on same device as model
        tokens = tokens.to(next(self.parameters()).device)
        padding_mask = padding_mask.to(next(self.parameters()).device)
        x = self.embeddings(tokens)  # (B, T, d)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)  # (B, T, 2*d)
        # Mask: padding_mask is True for valid tokens, False for pads
        mask_value = (~padding_mask).unsqueeze(2).float() * -1e9
        attn_weights = th.softmax(
            th.matmul(lstm_out, self.attn_query.transpose(1, 2)) + mask_value,
            dim=1
        )  # (B, T, 1)
        pooled = th.sum(lstm_out * attn_weights, dim=1)  # (B, 2*d)
        return self.proj(pooled)


class HokmHybridExtractor(BaseFeaturesExtractor):
    """LSTM trunk + Transformer head."""

    def __init__(self, observation_space, d_hidden: int = 256):
        super().__init__(observation_space, features_dim=d_hidden)
        self.d_hidden = d_hidden
        self.embeddings = nn.Embedding(134, d_hidden)  # tokenizer vocab size
        self.lstm = nn.LSTM(d_hidden, d_hidden, 1, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * d_hidden, nhead=4, dim_feedforward=512, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn_query = nn.Parameter(th.randn(1, 1, 2 * d_hidden))
        self.proj = nn.Linear(2 * d_hidden, d_hidden)

    def forward(self, observations: dict[str, th.Tensor] | th.Tensor) -> th.Tensor:
        if isinstance(observations, dict):
            obs = tokenize_tensor_batch(observations)
        else:
            obs = observations
        tokens = obs.tokens
        padding_mask = obs.padding_mask
        # Ensure tensors are on same device as model
        tokens = tokens.to(next(self.parameters()).device)
        padding_mask = padding_mask.to(next(self.parameters()).device)
        x = self.embeddings(tokens)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        src_key_padding_mask = ~padding_mask.bool()
        transformer_out = self.transformer(lstm_out, src_key_padding_mask=src_key_padding_mask)
        mask_value = (~padding_mask).unsqueeze(2).float() * -1e9
        attn_weights = th.softmax(
            th.matmul(transformer_out, self.attn_query.transpose(1, 2)) + mask_value,
            dim=1
        )
        pooled = th.sum(transformer_out * attn_weights, dim=1)
        return self.proj(pooled)


class HokmGNNExtractor(BaseFeaturesExtractor):
    """Graph neural net over card relationships."""

    def __init__(self, observation_space, d_hidden: int = 256):
        super().__init__(observation_space, features_dim=d_hidden)
        self.d_hidden = d_hidden
        self.embeddings = nn.Embedding(134, d_hidden)  # tokenizer vocab size
        self.gnn_layers = nn.ModuleList([
            nn.Linear(d_hidden, d_hidden) for _ in range(2)
        ])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_hidden, nhead=4, dim_feedforward=512, batch_first=True, norm_first=True),
            num_layers=2
        )
        self.attn_query = nn.Parameter(th.randn(1, 1, d_hidden))
        self.proj = nn.Linear(d_hidden, d_hidden)

    def forward(self, observations: dict[str, th.Tensor] | th.Tensor) -> th.Tensor:
        if isinstance(observations, dict):
            obs = tokenize_tensor_batch(observations)
        else:
            obs = observations
        tokens = obs.tokens
        padding_mask = obs.padding_mask
        # Ensure tensors are on same device as model
        tokens = tokens.to(next(self.parameters()).device)
        padding_mask = padding_mask.to(next(self.parameters()).device)
        x = self.embeddings(tokens)
        for layer in self.gnn_layers:
            x = layer(x)
            x = th.relu(x)
        src_key_padding_mask = ~padding_mask.bool()
        transformer_out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        mask_value = (~padding_mask).unsqueeze(2).float() * -1e9
        attn_weights = th.softmax(
            th.matmul(transformer_out, self.attn_query.transpose(1, 2)) + mask_value,
            dim=1
        )
        pooled = th.sum(transformer_out * attn_weights, dim=1)
        return self.proj(pooled)
