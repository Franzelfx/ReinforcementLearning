"""Parallel rollout collection using gymnasium.vector.AsyncVectorEnv.

Architecture
------------
- AsyncVectorEnv steps N environments in persistent subprocesses.
- The caller supplies a ``sample_fn`` that runs a single batched forward pass
  on the GPU/MPS in the main process.
- Only small observation/action arrays cross the process boundary — no model
  weights or gradients are ever serialised.

Two public collectors are provided:

collect_parallel_rollouts  — visual envs (H, W, C) RGB obs with frame stacking
collect_tabular_rollouts   — tabular envs (flat/windowed feature vectors), no stacking
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor
import gymnasium as gym

from .preprocessing import obs_to_grayscale_frame, push_obs_to_stack, stacks_to_batch


# Signature of the function the caller provides for action sampling.
# Input:  (N, T, H, W) batch state tensor on the main-process device.
# Output: (actions_np, log_probs, entropies, attn_weights_or_None)
SampleFn = Callable[
    [Tensor],
    tuple[np.ndarray, Tensor, Tensor, Tensor | None],
]


@dataclass(slots=True)
class EpisodeBatch:
    """Stores one complete episode's data needed for a policy update."""

    log_probs:             list[Tensor]
    rewards:               list[float]
    entropies:             list[Tensor]
    attention_mean_weights: list[float]
    episode_reward:        float
    steps:                 int


@dataclass
class _EnvTrajectory:
    """Accumulates data for one in-progress episode inside a vectorised env."""

    log_probs:     list[Tensor]
    rewards:       list[float]
    entropies:     list[Tensor]
    attn_frames:   list[Tensor]
    episode_reward: float = 0.0


def collect_parallel_rollouts(
    sample_fn:             SampleFn,
    device:                torch.device,
    venv:                  gym.vector.VectorEnv,
    num_envs:              int,
    min_steps:             int,
    action_repeat:         int,
    max_steps_per_episode: int,
    frame_stack_size:      int,
    base_seed:             int,
) -> list[EpisodeBatch]:
    """Collects complete episodes from N parallel envs with batched GPU inference.

    Parameters
    ----------
    sample_fn:
        Callable ``(states) -> (actions_np, log_probs, entropies, attn_or_None)``.
        Runs on the main process; typically wraps ``choose_action_batched``.
    device:
        Target device for state tensors (MPS / CUDA / CPU).
    venv:
        A persistent ``AsyncVectorEnv`` (or any VectorEnv) with ``num_envs`` envs.
    num_envs:
        Number of parallel environments.
    min_steps:
        Keep collecting until this many env steps have been logged.
    action_repeat:
        Number of consecutive env steps per policy decision.
    max_steps_per_episode:
        Episode hard-cap (steps, not policy decisions).
    frame_stack_size:
        Number of frames to stack per observation.
    base_seed:
        Seeds envs as ``base_seed + i`` for reproducibility.

    Returns
    -------
    list[EpisodeBatch]
        Completed (and any partial) episodes, each with gradient-attached
        log_probs on ``device``.
    """

    frame_stacks: list[deque[Tensor]] = [
        deque(maxlen=frame_stack_size) for _ in range(num_envs)
    ]
    trajs: list[_EnvTrajectory] = [_EnvTrajectory([], [], [], []) for _ in range(num_envs)]
    completed: list[EpisodeBatch] = []
    total_steps = 0
    ep_steps    = np.zeros(num_envs, dtype=np.int32)

    def _reset_stack(i: int, obs: np.ndarray) -> None:
        frame_stacks[i].clear()
        push_obs_to_stack(obs, frame_stacks[i])

    def _finalize(i: int) -> None:
        nonlocal total_steps
        if not trajs[i].rewards:
            return
        mean_attn = (
            [float(v) for v in torch.stack(trajs[i].attn_frames).mean(0).tolist()]
            if trajs[i].attn_frames else [0.0] * frame_stack_size
        )
        completed.append(EpisodeBatch(
            log_probs             = trajs[i].log_probs,
            rewards               = trajs[i].rewards,
            entropies             = trajs[i].entropies,
            attention_mean_weights= mean_attn,
            episode_reward        = trajs[i].episode_reward,
            steps                 = int(ep_steps[i]),
        ))
        total_steps  += int(ep_steps[i])
        trajs[i]      = _EnvTrajectory([], [], [], [])
        ep_steps[i]   = 0

    obs, _ = venv.reset(seed=[base_seed + i for i in range(num_envs)])
    for i, o in enumerate(obs):
        _reset_stack(i, o)

    latest_obs = obs
    while total_steps < min_steps:
        states = stacks_to_batch(frame_stacks, device)
        actions_np, log_probs, entropies, attn = sample_fn(states)

        step_rewards = np.zeros(num_envs, dtype=np.float64)
        done         = np.zeros(num_envs, dtype=bool)

        for _ in range(action_repeat):
            latest_obs, r, terminated, truncated, _ = venv.step(actions_np)
            active        = ~done
            ep_steps[active] += 1
            step_rewards  += r * active.astype(np.float64)
            done          |= terminated | truncated | (ep_steps >= max_steps_per_episode)
            if done.all():
                break

        for i in range(num_envs):
            trajs[i].log_probs.append(log_probs[i])
            trajs[i].rewards.append(float(step_rewards[i]))
            trajs[i].entropies.append(entropies[i])
            trajs[i].episode_reward += float(step_rewards[i])
            if attn is not None:
                trajs[i].attn_frames.append(attn[i])

            if done[i]:
                _finalize(i)
                _reset_stack(i, latest_obs[i])
            else:
                push_obs_to_stack(latest_obs[i], frame_stacks[i])

    for i in range(num_envs):
        _finalize(i)

    return completed


def collect_tabular_rollouts(
    sample_fn:             SampleFn,
    device:                torch.device,
    venv:                  gym.vector.VectorEnv,
    num_envs:              int,
    min_steps:             int,
    max_steps_per_episode: int,
    base_seed:             int,
) -> list[EpisodeBatch]:
    """Collects complete episodes from N parallel tabular envs with batched inference.

    Analogous to ``collect_parallel_rollouts`` but for environments whose
    observations are numeric feature vectors (flat or windowed), not images.
    No frame stacking is performed — observations are converted to float tensors
    and passed directly to ``sample_fn``.

    Parameters
    ----------
    sample_fn:
        Callable ``(states) -> (actions_np, log_probs, entropies, attn_or_None)``.
        ``states`` shape: ``(N, *obs_shape)`` on ``device``.
    device:
        Target device for state tensors.
    venv:
        A persistent ``VectorEnv`` with ``num_envs`` environments.
    num_envs:
        Number of parallel environments.
    min_steps:
        Keep collecting until this many env steps have been logged.
    max_steps_per_episode:
        Episode hard-cap.
    base_seed:
        Seeds envs as ``base_seed + i`` for reproducibility.

    Returns
    -------
    list[EpisodeBatch]
        Completed (and any partial) episodes with gradient-attached log_probs.
    """

    trajs: list[_EnvTrajectory] = [_EnvTrajectory([], [], [], []) for _ in range(num_envs)]
    completed: list[EpisodeBatch] = []
    total_steps = 0
    ep_steps    = np.zeros(num_envs, dtype=np.int32)

    def _finalize(i: int) -> None:
        nonlocal total_steps
        if not trajs[i].rewards:
            return
        mean_attn = (
            [float(v) for v in torch.stack(trajs[i].attn_frames).mean(0).tolist()]
            if trajs[i].attn_frames else []
        )
        completed.append(EpisodeBatch(
            log_probs             = trajs[i].log_probs,
            rewards               = trajs[i].rewards,
            entropies             = trajs[i].entropies,
            attention_mean_weights= mean_attn,
            episode_reward        = trajs[i].episode_reward,
            steps                 = int(ep_steps[i]),
        ))
        total_steps  += int(ep_steps[i])
        trajs[i]      = _EnvTrajectory([], [], [], [])
        ep_steps[i]   = 0

    obs, _ = venv.reset(seed=[base_seed + i for i in range(num_envs)])
    latest_obs = obs

    while total_steps < min_steps:
        states = torch.from_numpy(np.array(latest_obs, dtype=np.float32)).to(device)
        actions_np, log_probs, entropies, attn = sample_fn(states)

        latest_obs, rewards, terminated, truncated, _ = venv.step(actions_np)

        done = terminated | truncated | (ep_steps + 1 >= max_steps_per_episode)
        ep_steps[~done] += 1
        ep_steps[done]  += 1  # count the final step too

        for i in range(num_envs):
            trajs[i].log_probs.append(log_probs[i])
            trajs[i].rewards.append(float(rewards[i]))
            trajs[i].entropies.append(entropies[i])
            trajs[i].episode_reward += float(rewards[i])
            if attn is not None:
                trajs[i].attn_frames.append(attn[i])

            if done[i]:
                _finalize(i)
                # venv auto-resets on termination; latest_obs[i] is already the
                # first obs of the next episode — no manual reset needed.

    for i in range(num_envs):
        _finalize(i)

    return completed
