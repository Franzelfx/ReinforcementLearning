"""Training and evaluation entry point for the CarRacing project.

Run from the src/ directory:
    python -m car_racing.train [--mode train|evaluate] [options]
"""

from __future__ import annotations

import math
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv
from torch import nn

from rl_core.device import select_device, set_global_seed
from rl_core.persistence import load_model_if_available, save_model
from rl_core.plotting import save_attention_plot, save_reward_plot
from rl_core.preprocessing import preprocess_observation
from rl_core.reinforce import reinforce_loss
from rl_core.rollout import EpisodeBatch, collect_parallel_rollouts

from .config import FRAME_STACK_SIZE, CarRacingConfig, build_arg_parser, config_from_args
from .model import PolicyNetwork, choose_action, choose_action_batched


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(config: CarRacingConfig, render_mode: str | None = None) -> gym.Env:
    return gym.make(
        config.env_id,
        continuous      = True,
        render_mode     = render_mode,
        domain_randomize= config.domain_randomize,
    )


# ---------------------------------------------------------------------------
# Sequential single-env episode (used by evaluate())
# ---------------------------------------------------------------------------

def run_episode(
    env:          gym.Env,
    policy:       PolicyNetwork,
    config:       CarRacingConfig,
    device:       torch.device,
    episode_seed: int | None = None,
) -> EpisodeBatch:
    """Runs one episode sequentially; used for evaluation (no parallelism)."""

    observation, _ = env.reset(seed=episode_seed)
    frame_stack    = deque(maxlen=FRAME_STACK_SIZE)

    log_probs:        list[torch.Tensor] = []
    rewards:          list[float]        = []
    entropies:        list[torch.Tensor] = []
    attention_weights:list[torch.Tensor] = []
    episode_reward = 0.0
    steps          = 0

    while steps < config.max_steps_per_episode:
        state  = preprocess_observation(observation, device, frame_stack)
        action, log_prob, entropy = choose_action(policy, state)

        if policy.last_attention_weights is not None:
            attention_weights.append(policy.last_attention_weights.squeeze(0))

        repeated_reward = 0.0
        terminated = truncated = False
        for _ in range(config.action_repeat):
            observation, reward, terminated, truncated, _ = env.step(action)
            repeated_reward += reward
            steps += 1
            if terminated or truncated or steps >= config.max_steps_per_episode:
                break

        log_probs.append(log_prob)
        rewards.append(repeated_reward)
        entropies.append(entropy)
        episode_reward += repeated_reward

        if terminated or truncated:
            break

    if attention_weights:
        mean_attn = [float(v) for v in torch.stack(attention_weights).mean(0).tolist()]
    else:
        mean_attn = [0.0] * FRAME_STACK_SIZE

    return EpisodeBatch(
        log_probs             = log_probs,
        rewards               = rewards,
        entropies             = entropies,
        attention_mean_weights= mean_attn,
        episode_reward        = episode_reward,
        steps                 = steps,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _print_device_info(device: torch.device) -> None:
    print(f"Device  : {device}  |  PyTorch {torch.__version__}")
    if device.type == "mps":
        print(
            "Apple Silicon MPS active.\n"
            "  -> Batched inference for all parallel envs runs on GPU.\n"
            "  -> Env-stepping runs in persistent AsyncVectorEnv subprocesses."
        )
    elif device.type == "cuda":
        print(f"CUDA    : {torch.cuda.get_device_name(device)}")
    else:
        print("No accelerator found — training on CPU.")


def train(config: CarRacingConfig) -> tuple[list[float], list[list[float]]]:
    """Trains the policy using REINFORCE with vectorised parallel environments."""

    set_global_seed(config.seed)
    device = torch.device(config.device)
    _print_device_info(device)

    policy = PolicyNetwork(
        hidden_dim             = config.hidden_dim,
        transformer_embed_dim  = config.transformer_embed_dim,
        transformer_num_layers = config.transformer_num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    reward_history:    list[float]       = []
    attention_history: list[list[float]] = []

    num_envs = max(
        config.num_workers,
        math.ceil(config.min_steps_per_update / config.max_steps_per_episode),
    )
    print(
        f"Vectorised training: {num_envs} envs, "
        f"min {config.min_steps_per_update} steps/update."
    )

    # sample_fn wraps choose_action_batched and captures `policy` from the
    # enclosing scope — no serialisation of weights across process boundaries.
    def sample_fn(states: torch.Tensor):
        return choose_action_batched(policy, states)

    venv = AsyncVectorEnv([lambda: make_env(config, render_mode=None)] * num_envs)
    try:
        for update_idx in range(1, config.episodes + 1):
            episodes = collect_parallel_rollouts(
                sample_fn             = sample_fn,
                device                = device,
                venv                  = venv,
                num_envs              = num_envs,
                min_steps             = config.min_steps_per_update,
                action_repeat         = config.action_repeat,
                max_steps_per_episode = config.max_steps_per_episode,
                frame_stack_size      = FRAME_STACK_SIZE,
                base_seed             = config.seed + (update_idx - 1) * num_envs,
            )

            # REINFORCE update — average loss over all collected episodes.
            optimizer.zero_grad()
            losses = [
                reinforce_loss(
                    ep,
                    gamma             = config.gamma,
                    device            = device,
                    normalize_returns = config.normalize_returns,
                    entropy_coef      = config.entropy_coef,
                )
                for ep in episodes if ep.log_probs
            ]
            if losses:
                loss = sum(losses) / len(losses)   # type: ignore[arg-type]
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

            if device.type == "mps":
                torch.mps.empty_cache()

            # Logging
            rewards     = [ep.episode_reward for ep in episodes]
            total_steps = sum(ep.steps for ep in episodes)
            all_attn    = [ep.attention_mean_weights for ep in episodes]
            avg_reward  = float(np.mean(rewards))
            avg_attn    = [
                float(np.mean([w[i] for w in all_attn]))
                for i in range(FRAME_STACK_SIZE)
            ]

            reward_history.append(avg_reward)
            attention_history.append(avg_attn)

            mean_recent   = float(np.mean(reward_history[-10:]))
            attention_str = " | ".join(f"F{i}:{w:.3f}" for i, w in enumerate(avg_attn))
            print(
                f"Update {update_idx:03d} | "
                f"Reward (avg): {avg_reward:8.2f} | "
                f"Steps: {total_steps:5d} | "
                f"Episodes: {len(episodes):2d} | "
                f"Mean(10): {mean_recent:8.2f} | "
                f"Attn: {attention_str}"
            )
    finally:
        venv.close()

    save_model(policy, config.model_path)
    print(f"Model saved to: {config.model_path}")
    return reward_history, attention_history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(config: CarRacingConfig) -> None:
    """Runs sequential evaluation episodes with the saved policy."""

    set_global_seed(config.seed)
    device = torch.device(config.device)
    print(f"Evaluating on: {device}  |  PyTorch {torch.__version__}")

    env    = make_env(config, render_mode=config.render_mode)
    policy = PolicyNetwork(
        hidden_dim             = config.hidden_dim,
        transformer_embed_dim  = config.transformer_embed_dim,
        transformer_num_layers = config.transformer_num_layers,
    ).to(device)
    load_model_if_available(policy, config.model_path, device)

    try:
        for ep_idx in range(1, config.episodes + 1):
            episode = run_episode(
                env,
                policy,
                config,
                device,
                episode_seed=config.seed + ep_idx - 1,
            )
            attn_str = " | ".join(
                f"F{i}:{w:.3f}" for i, w in enumerate(episode.attention_mean_weights)
            )
            print(
                f"Eval {ep_idx:03d} | "
                f"Reward: {episode.episode_reward:8.2f} | "
                f"Steps: {episode.steps:4d} | "
                f"Attn: {attn_str}"
            )
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()
    config = config_from_args(args)

    if args.mode == "train":
        rewards, attention_history = train(config)
        plot_path   = save_reward_plot(rewards, config.save_plot_path)
        attn_path   = save_attention_plot(attention_history, config.attention_plot_path)
        print(f"Reward plot  : {plot_path}")
        print(f"Attention map: {attn_path}")
    else:
        evaluate(config)


if __name__ == "__main__":
    main()
