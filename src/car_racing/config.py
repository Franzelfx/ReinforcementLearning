"""CarRacing project configuration and CLI argument parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from rl_core.device import select_device

# Number of grayscale frames stacked into the CNN input channel dimension.
# Changing this requires rebuilding the model (affects Conv2d input channels
# and the FrameTransformerAttention token count).
FRAME_STACK_SIZE: int = 32


@dataclass(slots=True)
class CarRacingConfig:
    """All hyperparameters and paths for a CarRacing training or eval run.

    Centralised here so workshop participants don't need to hunt across files.
    """

    env_id:                str       = "CarRacing-v3"
    episodes:              int       = 500
    max_steps_per_episode: int       = 600
    gamma:                 float     = 0.99
    learning_rate:         float     = 1e-4
    action_repeat:         int       = 4
    seed:                  int       = 42
    hidden_dim:            int       = 256
    device:                str       = select_device()
    render_mode:           str | None= "rgb_array"
    domain_randomize:      bool      = False
    save_plot_path:        str       = "training_rewards.png"
    attention_plot_path:   str       = "attention_weights.png"
    model_path:            str       = "car_racing_policy.pt"
    transformer_embed_dim: int       = 128
    transformer_num_layers:int       = 8
    num_workers:           int       = 12
    min_steps_per_update:  int       = 1000
    normalize_returns:     bool      = True
    entropy_coef:          float     = 0.01


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI that exposes every CarRacingConfig field as a flag."""

    p = argparse.ArgumentParser(description="REINFORCE on CarRacing-v3")
    p.add_argument("--mode", choices=["train", "evaluate"], default="train")
    p.add_argument("--episodes",              type=int,   default=500)
    p.add_argument("--gamma",                 type=float, default=0.99)
    p.add_argument("--learning-rate",         type=float, default=1e-4)
    p.add_argument("--action-repeat",         type=int,   default=4)
    p.add_argument("--seed",                  type=int,   default=42)
    p.add_argument("--render",
                   choices=["none", "human", "rgb_array"], default="rgb_array")
    p.add_argument("--domain-randomize",      action="store_true")
    p.add_argument("--save-plot",             default="training_rewards.png")
    p.add_argument("--attention-plot",        default="attention_weights.png")
    p.add_argument("--model-path",            default="car_racing_policy.pt")
    p.add_argument("--transformer-embed-dim", type=int,   default=128)
    p.add_argument("--transformer-num-layers",type=int,   default=8)
    p.add_argument("--num-workers",           type=int,   default=12)
    p.add_argument("--min-steps",             type=int,   default=1000)
    p.add_argument("--device",                default=None,
                   help="mps | cuda | cpu  (default: auto-detect)")
    p.add_argument("--no-normalize-returns",  action="store_true",
                   help="Disable return normalisation (for comparison experiments)")
    p.add_argument("--entropy-coef",          type=float, default=0.01,
                   help="Entropy bonus weight (0 = off, ~0.01 recommended)")
    return p


def config_from_args(args: argparse.Namespace) -> CarRacingConfig:
    """Builds a CarRacingConfig from parsed CLI arguments."""

    return CarRacingConfig(
        episodes              = args.episodes,
        gamma                 = args.gamma,
        learning_rate         = args.learning_rate,
        action_repeat         = args.action_repeat,
        seed                  = args.seed,
        render_mode           = None if args.render == "none" else args.render,
        domain_randomize      = args.domain_randomize,
        save_plot_path        = args.save_plot,
        attention_plot_path   = args.attention_plot,
        model_path            = args.model_path,
        transformer_embed_dim = args.transformer_embed_dim,
        transformer_num_layers= args.transformer_num_layers,
        num_workers           = args.num_workers,
        min_steps_per_update  = args.min_steps,
        device                = select_device(args.device),
        normalize_returns     = not args.no_normalize_returns,
        entropy_coef          = args.entropy_coef,
    )
