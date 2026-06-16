"""REINFORCE algorithm primitives.

These functions are pure tensor operations — they have no dependency on any
specific environment or model architecture.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .rollout import EpisodeBatch


def discounted_returns(
    rewards: list[float],
    gamma:   float,
    device:  torch.device,
) -> Tensor:
    """Computes discounted returns G_t for a sequence of rewards.

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Returns a 1-D float32 tensor on ``device``.
    """

    running = 0.0
    returns: list[float] = []
    for r in reversed(rewards):
        running = r + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=device)


def reinforce_loss(
    episode:          EpisodeBatch,
    gamma:            float,
    device:           torch.device,
    normalize_returns: bool  = True,
    entropy_coef:     float = 0.01,
) -> Tensor:
    """Computes the REINFORCE loss for one episode.

    loss = -sum(log_pi(a_t | s_t) * G_t)  - entropy_coef * H(pi)

    Parameters
    ----------
    normalize_returns:
        Normalize G_t to zero mean and unit variance within the episode.
        Dramatically reduces gradient variance — recommended to keep True.
    entropy_coef:
        Weight of the entropy bonus. Encourages exploration and prevents
        premature collapse to a deterministic policy.
    """

    returns = discounted_returns(episode.rewards, gamma, device)
    if normalize_returns and returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs     = torch.stack(episode.log_probs)
    entropies     = torch.stack(episode.entropies)
    policy_loss   = -(log_probs * returns).sum()
    entropy_bonus = entropies.sum()
    return policy_loss - entropy_coef * entropy_bonus
