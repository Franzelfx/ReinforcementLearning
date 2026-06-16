"""Visual observation preprocessing helpers for frame-stacked environments.

All functions are stateless — callers manage their own frame-stack deques.
This makes the functions safe to use in parallel (multiple envs, no shared
global state).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import Tensor


def obs_to_grayscale_frame(obs: np.ndarray) -> Tensor:
    """(H, W, C) uint8 RGB -> (1, H, W) float32 CPU tensor."""

    rgb  = obs.astype(np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return torch.from_numpy(gray).unsqueeze(0)


def push_obs_to_stack(obs: np.ndarray, frame_stack: deque[Tensor]) -> None:
    """Converts one observation to grayscale and appends it to the frame stack.

    If the stack is empty (start of episode), it is filled with the first frame
    so the downstream model always receives a full stack.
    """

    frame = obs_to_grayscale_frame(obs)
    if not frame_stack:
        for _ in range(frame_stack.maxlen):  # type: ignore[arg-type]
            frame_stack.append(frame)
    else:
        frame_stack.append(frame)


def stacks_to_batch(frame_stacks: list[deque[Tensor]], device: torch.device) -> Tensor:
    """Converts N per-env frame stacks into a single (N, T, H, W) batch tensor."""

    return torch.stack(
        [torch.cat(list(fs), dim=0) for fs in frame_stacks], dim=0
    ).to(device)


def preprocess_observation(
    observation: np.ndarray,
    device: torch.device,
    frame_stack: deque[Tensor],
) -> Tensor:
    """Full preprocessing pipeline for a single sequential observation.

    Converts the observation to grayscale, pushes it into the caller-owned
    frame stack, and returns the stacked tensor on ``device``.

    Returns:
        Tensor of shape (1, T, H, W) on ``device``.
    """

    push_obs_to_stack(observation, frame_stack)
    stacked = torch.cat(list(frame_stack), dim=0)   # (T, H, W)
    return stacked.unsqueeze(0).to(device)           # (1, T, H, W)
