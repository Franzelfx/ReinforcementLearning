"""Policy network and action-sampling functions for CarRacing-v3.

Architecture
------------
FrameTransformerAttention
    Treats each grayscale frame in the stack as a token, runs a multi-layer
    Transformer encoder to compute per-frame attention weights, and returns a
    weighted frame stack.

PolicyNetwork
    1. FrameTransformerAttention  (temporal attention over the frame stack)
    2. CNN encoder                (spatial feature extraction)
    3. Linear policy head         (outputs mean of a Normal distribution)
    4. Learnable log_std          (one scalar per action dimension)

Actions are sampled from Normal(mean, exp(log_std)) and clipped to their
valid ranges via tanh / sigmoid before being sent to the environment.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal

from .config import FRAME_STACK_SIZE

# CarRacing-v3 continuous action dimensions:
#   [0] steering  in [-1, 1]   (tanh)
#   [1] gas       in [ 0, 1]   (sigmoid)
#   [2] brake     in [ 0, 1]   (sigmoid)
ACTION_DIM = 3

# Expected observation resolution after grayscale conversion.
_OBS_H = 96
_OBS_W = 96


class FrameTransformerAttention(nn.Module):
    """Multi-layer Transformer encoder over the frame stack.

    Each frame is projected to ``embed_dim`` dimensions and treated as a
    sequence token.  Learned positional embeddings encode temporal order.
    The encoder outputs per-frame attention weights that modulate the stack
    before it enters the CNN.
    """

    def __init__(
        self,
        frame_count: int,
        embed_dim:   int   = 64,
        num_layers:  int   = 4,
        nhead:       int   = 4,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.input_proj    = nn.Linear(1, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(frame_count, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model       = embed_dim,
            nhead         = nhead,
            dim_feedforward= 4 * embed_dim,
            dropout       = dropout,
            batch_first   = True,
            norm_first    = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head  = nn.Linear(embed_dim, 1)

    def forward(self, frame_stack: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        frame_stack : (B, T, H, W)

        Returns
        -------
        weighted_stack : (B, T, H, W)  — frame stack scaled by attention
        weights        : (B, T)        — softmax attention weights
        """

        frame_summary = frame_stack.mean(dim=(2, 3)).unsqueeze(-1)   # (B, T, 1)
        tokens        = self.input_proj(frame_summary) + self.pos_embedding.unsqueeze(0)
        encoded       = self.transformer(tokens)
        weights       = torch.softmax(self.score_head(encoded).squeeze(-1), dim=1)
        weighted_stack = frame_stack * weights.unsqueeze(-1).unsqueeze(-1)
        return weighted_stack, weights


class PolicyNetwork(nn.Module):
    """CNN policy with Transformer frame-attention and continuous action output."""

    def __init__(
        self,
        hidden_dim:             int,
        transformer_embed_dim:  int = 64,
        transformer_num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.frame_attention = FrameTransformerAttention(
            frame_count = FRAME_STACK_SIZE,
            embed_dim   = transformer_embed_dim,
            num_layers  = transformer_num_layers,
        )
        self.last_attention_weights: Tensor | None = None

        self.encoder = nn.Sequential(
            nn.Conv2d(FRAME_STACK_SIZE, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        encoder_out = self._infer_encoder_out()

        self.policy_head = nn.Sequential(
            nn.Linear(encoder_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM))

    def _infer_encoder_out(self) -> int:
        dummy = torch.zeros(1, FRAME_STACK_SIZE, _OBS_H, _OBS_W)
        with torch.no_grad():
            return int(self.encoder(dummy).shape[1])

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        attended, weights          = self.frame_attention(observation)
        self.last_attention_weights = weights.detach()
        encoded                    = self.encoder(attended)
        mean                       = self.policy_head(encoded)
        log_std                    = self.log_std.expand_as(mean)
        return mean, log_std


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------

def _sample_from_policy(
    policy:      PolicyNetwork,
    observation: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Shared sampling logic for single-env and batched inference.

    Returns ``(action_t, log_probs, entropies)`` all on the same device as
    ``observation``.  ``action_t`` has bounds applied (tanh/sigmoid) but still
    has gradients attached.
    """

    mean, log_std = policy(observation)
    std           = log_std.clamp(min=-4.0, max=1.0).exp()
    dist          = Normal(mean, std)

    raw      = dist.rsample()
    steering = torch.tanh(raw[..., 0:1])
    gas      = torch.sigmoid(raw[..., 1:2])
    brake    = torch.sigmoid(raw[..., 2:3])
    action_t = torch.cat([steering, gas, brake], dim=-1)

    return action_t, dist.log_prob(raw).sum(dim=-1), dist.entropy().sum(dim=-1)


def choose_action(
    policy:      PolicyNetwork,
    observation: Tensor,
) -> tuple[np.ndarray, Tensor, Tensor]:
    """Single-env action sampling (batch size 1).

    Returns
    -------
    action_np : (3,) float32 numpy array for ``env.step()``
    log_prob  : scalar tensor (gradient attached)
    entropy   : scalar tensor
    """

    action_t, log_probs, entropies = _sample_from_policy(policy, observation)
    action_np = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return action_np, log_probs.squeeze(), entropies.squeeze()


def choose_action_batched(
    policy:       PolicyNetwork,
    observations: Tensor,
) -> tuple[np.ndarray, Tensor, Tensor, Tensor | None]:
    """Batched action sampling for N environments in one GPU pass.

    Returns
    -------
    actions_np  : (N, 3) float32 numpy array for ``venv.step()``
    log_probs   : (N,) tensor (gradient attached)
    entropies   : (N,) tensor
    attn_weights: (N, T) detached tensor or None
    """

    action_t, log_probs, entropies = _sample_from_policy(policy, observations)
    actions_np   = action_t.detach().cpu().numpy().astype(np.float32)
    attn_weights = policy.last_attention_weights   # set by forward(), already detached
    return actions_np, log_probs, entropies, attn_weights
