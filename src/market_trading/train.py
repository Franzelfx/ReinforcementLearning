"""Training and evaluation entry point for the MarketTrading project.

Run from the src/ directory:
    python -m market_trading.train [--mode train|evaluate] [options]

Quick start with synthetic data (no CSV required):
    python -m market_trading.train --synthetic --episodes 5 --min-steps 200
"""

from __future__ import annotations

import functools
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from torch import nn

from rl_core.device import select_device, set_global_seed
from rl_core.persistence import load_model_if_available, save_model
from rl_core.plotting import save_reward_plot
from rl_core.reinforce import reinforce_loss
from rl_core.rollout import EpisodeBatch, collect_tabular_rollouts

from .config import MarketTradingConfig, build_arg_parser, config_from_args
from .data import add_features, ensure_data, load_dataframe, make_synthetic_dataframe
from .model import TradingPolicyNetwork, choose_action, choose_action_batched

import gym_trading_env  # noqa: F401 — registers "TradingEnv" with gymnasium


# ---------------------------------------------------------------------------
# Hold wrapper — prevents overtrading by enforcing a minimum hold period
# ---------------------------------------------------------------------------

class HoldWrapper(gym.Wrapper):
    """Forces the agent to keep each position for at least `min_hold` steps.

    Without this constraint an untrained policy trades nearly every bar,
    losing ~14% per episode purely to fees before learning anything useful.
    """

    def __init__(self, env: gym.Env, min_hold: int = 5) -> None:
        super().__init__(env)
        self._min_hold    = min_hold
        self._steps_held  = 0
        self._last_action = None

    def reset(self, **kwargs):
        self._steps_held  = 0
        self._last_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._last_action is None or self._steps_held >= self._min_hold:
            self._last_action = action
            self._steps_held  = 0
        self._steps_held += 1
        return self.env.step(self._last_action)


def _make_single_env(config: MarketTradingConfig, df) -> gym.Env:
    """Creates one TradingEnv with the HoldWrapper applied (used by AsyncVectorEnv)."""
    env = gym.make(
        config.env_id,
        name                    = "MarketTrading",
        df                      = df,
        windows                 = config.windows,
        positions               = config.positions,
        initial_position        = 0,
        trading_fees            = config.trading_fees,
        borrow_interest_rate    = config.borrow_interest_rate,
        portfolio_initial_value = config.portfolio_initial_value,
    )
    if config.min_hold_steps > 1:
        env = HoldWrapper(env, config.min_hold_steps)
    return env


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _build_df(config: MarketTradingConfig) -> "pd.DataFrame":
    if config.use_synthetic:
        df = make_synthetic_dataframe(n_bars=max(config.max_steps_per_episode * 2, 4000))
    else:
        path = ensure_data(
            data_dir  = config.data_dir,
            exchange  = config.exchange,
            symbol    = config.symbol,
            timeframe = config.timeframe,
            since     = config.download_since,
        )
        df = load_dataframe(path)
    return add_features(df)


def make_env(config: MarketTradingConfig, render_mode: str = "logs") -> gym.Env:
    df  = _build_df(config)
    env = gym.make(
        config.env_id,
        name                    = "MarketTrading",
        df                      = df,
        windows                 = config.windows,
        positions               = config.positions,
        initial_position        = 0,
        trading_fees            = config.trading_fees,
        borrow_interest_rate    = config.borrow_interest_rate,
        portfolio_initial_value = config.portfolio_initial_value,
        render_mode             = render_mode,
    )
    if config.min_hold_steps > 1:
        env = HoldWrapper(env, config.min_hold_steps)
    return env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _n_features(config: MarketTradingConfig) -> int:
    """Returns the number of feature columns in the DataFrame."""
    df = _build_df(config)
    env = gym.make(
        config.env_id,
        name    = "probe",
        df      = df,
        windows = config.windows,
        positions = config.positions,
    )
    obs, _ = env.reset()
    env.close()
    return int(obs.shape[-1])


def _print_device_info(device: torch.device) -> None:
    print(f"Device  : {device}  |  PyTorch {torch.__version__}")
    if device.type == "mps":
        print("Apple Silicon MPS active — batched inference on GPU.")
    elif device.type == "cuda":
        print(f"CUDA    : {torch.cuda.get_device_name(device)}")
    else:
        print("No accelerator found — training on CPU.")


# ---------------------------------------------------------------------------
# Sequential single-env episode (used by evaluate())
# ---------------------------------------------------------------------------

def run_episode(
    env:          gym.Env,
    policy:       TradingPolicyNetwork,
    config:       MarketTradingConfig,
    device:       torch.device,
    episode_seed: int | None = None,
) -> EpisodeBatch:
    """Runs one episode sequentially; used for evaluation."""

    observation, _ = env.reset(seed=episode_seed)
    log_probs:  list[torch.Tensor] = []
    rewards:    list[float]        = []
    entropies:  list[torch.Tensor] = []
    episode_reward = 0.0
    steps = 0

    while steps < config.max_steps_per_episode:
        state = torch.from_numpy(np.array(observation, dtype=np.float32)).unsqueeze(0).to(device)
        action_np, log_prob, entropy = choose_action(policy, state)

        observation, reward, terminated, truncated, _ = env.step(action_np[0])
        log_probs.append(log_prob)
        rewards.append(float(reward))
        entropies.append(entropy)
        episode_reward += float(reward)
        steps += 1

        if terminated or truncated:
            break

    return EpisodeBatch(
        log_probs             = log_probs,
        rewards               = rewards,
        entropies             = entropies,
        attention_mean_weights= [],
        episode_reward        = episode_reward,
        steps                 = steps,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: MarketTradingConfig) -> list[float]:
    """Trains the trading policy using REINFORCE with vectorised parallel environments."""

    set_global_seed(config.seed)
    device = torch.device(config.device)
    _print_device_info(device)

    n_features    = _n_features(config)
    num_positions = len(config.positions)
    print(
        f"Observation : window={config.windows}, n_features={n_features}\n"
        f"Action space: {num_positions} positions — {config.positions}"
    )

    policy = TradingPolicyNetwork(
        n_features             = n_features,
        num_positions          = num_positions,
        window                 = config.windows,
        hidden_dim             = config.hidden_dim,
        transformer_embed_dim  = config.transformer_embed_dim,
        transformer_num_layers = config.transformer_num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    reward_history: list[float] = []

    def sample_fn(states: torch.Tensor):
        return choose_action_batched(policy, states)

    df       = _build_df(config)
    factories = [functools.partial(_make_single_env, config, df)
                 for _ in range(config.num_workers)]
    venv = gym.vector.AsyncVectorEnv(factories)
    print(
        f"Vectorised training: {config.num_workers} envs, "
        f"min {config.min_steps_per_update} steps/update, "
        f"min_hold={config.min_hold_steps} bars."
    )

    try:
        for update_idx in range(1, config.episodes + 1):
            episodes = collect_tabular_rollouts(
                sample_fn             = sample_fn,
                device                = device,
                venv                  = venv,
                num_envs              = config.num_workers,
                min_steps             = config.min_steps_per_update,
                max_steps_per_episode = config.max_steps_per_episode,
                base_seed             = config.seed + (update_idx - 1) * config.num_workers,
            )

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

            rewards     = [ep.episode_reward for ep in episodes]
            total_steps = sum(ep.steps for ep in episodes)
            avg_reward  = float(np.mean(rewards))
            mean_recent = float(np.mean(reward_history[-10:])) if reward_history else avg_reward
            reward_history.append(avg_reward)

            print(
                f"Update {update_idx:03d} | "
                f"Reward (avg): {avg_reward:8.4f} | "
                f"Steps: {total_steps:5d} | "
                f"Episodes: {len(episodes):2d} | "
                f"Mean(10): {mean_recent:8.4f}"
            )
    finally:
        venv.close()

    save_model(policy, config.model_path)
    print(f"Model saved to: {config.model_path}")
    return reward_history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(config: MarketTradingConfig) -> None:
    """Runs sequential evaluation episodes and saves render logs for the viewer."""

    set_global_seed(config.seed)
    device = torch.device(config.device)

    n_features    = _n_features(config)
    num_positions = len(config.positions)
    policy = TradingPolicyNetwork(
        n_features             = n_features,
        num_positions          = num_positions,
        window                 = config.windows,
        hidden_dim             = config.hidden_dim,
        transformer_embed_dim  = config.transformer_embed_dim,
        transformer_num_layers = config.transformer_num_layers,
    ).to(device)
    load_model_if_available(policy, config.model_path, device)

    env = make_env(config, render_mode="logs")
    try:
        for ep_idx in range(1, config.episodes + 1):
            episode = run_episode(
                env, policy, config, device,
                episode_seed=config.seed + ep_idx - 1,
            )
            # Absolute path so save_for_render writes to the right place
            # regardless of the process working directory.
            logs_dir = str(Path(config.render_logs_dir).resolve())
            env.unwrapped.save_for_render(dir=logs_dir)
            print(
                f"Eval {ep_idx:03d} | "
                f"Reward: {episode.episode_reward:8.4f} | "
                f"Steps: {episode.steps:4d} | "
                f"Render log saved to {config.render_logs_dir}/"
            )
    finally:
        env.close()

    print(
        f"\nAll render logs saved to '{config.render_logs_dir}/'.\n"
        f"Launch the interactive chart with:\n"
        f"  python -m market_trading.train --mode render"
    )


def render_chart(config: MarketTradingConfig) -> None:
    """Starts the interactive Flask chart server and opens it in the browser."""
    from .renderer import serve
    serve(config.render_logs_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()
    config = config_from_args(args)

    if args.mode == "train":
        rewards   = train(config)
        plot_path = save_reward_plot(rewards, config.save_plot_path)
        print(f"Reward plot: {plot_path}")
    elif args.mode == "evaluate":
        evaluate(config)
    else:  # render
        render_chart(config)


if __name__ == "__main__":
    main()
