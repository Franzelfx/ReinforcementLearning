"""Transformer-based discrete policy network for market trading.

Architecture
------------
TemporalTransformerEncoder
    Treats each timestep in the observation window as a token.
    Projects ``n_features`` → ``embed_dim``, adds learnable positional
    embeddings, runs a multi-layer Transformer encoder, then mean-pools
    across the window dimension.

TradingPolicyNetwork
    1. TemporalTransformerEncoder  (temporal attention over the feature window)
    2. MLP policy head             (outputs logits over discrete positions)

Actions are sampled from a Categorical distribution over the position list
(e.g. [-1, -0.5, 0, 0.5, 1]), so the action index selects a portfolio weight.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class TemporalTransformerEncoder(nn.Module):
    """Encodes a window of feature vectors via a Transformer.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    window : int
        Number of timesteps (tokens) in the observation.
    embed_dim : int
        Transformer embedding dimension.
    num_layers : int
        Number of TransformerEncoder layers.
    nhead : int
        Number of attention heads (must divide ``embed_dim``).
    dropout : float
        Dropout rate inside the Transformer.
    """

    def __init__(
        self,
        n_features: int,
        window:     int,
        embed_dim:  int   = 64,
        num_layers: int   = 4,
        nhead:      int   = 4,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj    = nn.Linear(n_features, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(window, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = embed_dim,
            nhead          = nhead,
            dim_feedforward= 4 * embed_dim,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.score_head  = nn.Linear(embed_dim, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : (B, window, n_features)

        Returns
        -------
        pooled   : (B, embed_dim)   — mean-pooled representation
        weights  : (B, window)      — softmax attention scores (for logging)
        """

        tokens  = self.input_proj(x) + self.pos_embedding.unsqueeze(0)
        encoded = self.transformer(tokens)                           # (B, T, E)
        weights = torch.softmax(self.score_head(encoded).squeeze(-1), dim=1)  # (B, T)
        pooled  = (encoded * weights.unsqueeze(-1)).sum(dim=1)       # (B, E)
        return pooled, weights


class TradingPolicyNetwork(nn.Module):
    """Discrete policy for position-based trading via Categorical distribution."""

    def __init__(
        self,
        n_features:            int,
        num_positions:         int,
        window:                int,
        hidden_dim:            int = 256,
        transformer_embed_dim: int = 64,
        transformer_num_layers:int = 4,
    ) -> None:
        super().__init__()
        self.encoder = TemporalTransformerEncoder(
            n_features = n_features,
            window     = window,
            embed_dim  = transformer_embed_dim,
            num_layers = transformer_num_layers,
        )
        self.last_attention_weights: Tensor | None = None

        self.policy_head = nn.Sequential(
            nn.Linear(transformer_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_positions),
        )

    def forward(self, observation: Tensor) -> Tensor:
        """
        Parameters
        ----------
        observation : (B, window, n_features)

        Returns
        -------
        logits : (B, num_positions)
        """

        pooled, weights = self.encoder(observation)
        self.last_attention_weights = weights.detach()
        return self.policy_head(pooled)


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------

def choose_action(
    policy:      TradingPolicyNetwork,
    observation: Tensor,
) -> tuple[np.ndarray, Tensor, Tensor]:
    """Single-env action sampling.

    Returns
    -------
    action_np : (1,) int32 — position index for env.step()
    log_prob  : scalar tensor (gradient attached)
    entropy   : scalar tensor
    """

    logits = policy(observation)
    dist   = Categorical(logits=logits)
    action = dist.sample()
    return (
        action.detach().cpu().numpy().astype(np.int32),
        dist.log_prob(action).squeeze(),
        dist.entropy().squeeze(),
    )


def choose_action_batched(
    policy:       TradingPolicyNetwork,
    observations: Tensor,
) -> tuple[np.ndarray, Tensor, Tensor, Tensor | None]:
    """Batched action sampling for N environments in one GPU pass.

    Matches the ``SampleFn`` protocol expected by ``collect_tabular_rollouts``.

    Returns
    -------
    actions_np  : (N,) int32  — position indices for venv.step()
    log_probs   : (N,) tensor (gradient attached)
    entropies   : (N,) tensor
    attn_weights: (N, window) detached tensor or None
    """

    logits   = policy(observations)
    dist     = Categorical(logits=logits)
    actions  = dist.sample()
    return (
        actions.detach().cpu().numpy().astype(np.int32),
        dist.log_prob(actions),
        dist.entropy(),
        policy.last_attention_weights,
    )
