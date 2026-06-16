"""MarketTrading project configuration and CLI argument parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from rl_core.device import select_device

# Default file path matches the gym-trading-env downloader naming convention:
# {data_dir}/{exchange}-{symbol_no_slash}-{timeframe}.pkl
_DEFAULT_DATA_DIR  = "market_trading/data"
_DEFAULT_EXCHANGE  = "binance"
_DEFAULT_SYMBOL    = "BTC/USDT"
_DEFAULT_TIMEFRAME = "5m"
_DEFAULT_DATA_PATH = f"{_DEFAULT_DATA_DIR}/{_DEFAULT_EXCHANGE}-BTCUSDT-{_DEFAULT_TIMEFRAME}.pkl"


@dataclass(slots=True)
class MarketTradingConfig:
    """All hyperparameters and paths for a market-trading training or eval run."""

    env_id:                  str         = "TradingEnv"
    # --- data / download ---
    data_dir:                str         = _DEFAULT_DATA_DIR
    data_path:               str         = _DEFAULT_DATA_PATH
    exchange:                str         = _DEFAULT_EXCHANGE
    symbol:                  str         = _DEFAULT_SYMBOL
    timeframe:               str         = _DEFAULT_TIMEFRAME
    download_since:          str         = "2022-01-01"
    use_synthetic:           bool        = False
    # --- environment ---
    episodes:                int         = 1000
    max_steps_per_episode:   int         = 2000
    gamma:                   float       = 0.99
    learning_rate:           float       = 1e-4
    seed:                    int         = 42
    device:                  str         = field(default_factory=select_device)
    positions:               list[float] = field(
                                 default_factory=lambda: [-1.0, 0.0, 1.0]
                             )
    windows:                 int         = 30
    trading_fees:            float       = 0.01 / 100
    borrow_interest_rate:    float       = 0.0003 / 100
    portfolio_initial_value: float       = 1000.0
    min_hold_steps:          int         = 5
    # --- model ---
    hidden_dim:              int         = 256
    transformer_embed_dim:   int         = 64
    transformer_num_layers:  int         = 2
    # --- training ---
    num_workers:             int         = 4
    min_steps_per_update:    int         = 1000
    normalize_returns:       bool        = True
    entropy_coef:            float       = 0.001
    # --- output ---
    model_path:              str         = "market_trading/market_trading_policy.pt"
    save_plot_path:          str         = "market_trading/market_training_rewards.png"
    render_logs_dir:         str         = "market_trading/render_logs"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="REINFORCE on gym-trading-env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",           choices=["train", "evaluate", "render"], default="train")
    # data / download
    p.add_argument("--data-dir",       default="market_trading/data",
                   help="Directory where downloaded pickle files are stored")
    p.add_argument("--exchange",       default=_DEFAULT_EXCHANGE,
                   help="ccxt exchange id (binance | huobi | bitfinex2)")
    p.add_argument("--symbol",         default=_DEFAULT_SYMBOL,
                   help="Market symbol in ccxt format, e.g. BTC/USDT")
    p.add_argument("--timeframe",      default=_DEFAULT_TIMEFRAME,
                   help="Bar size: 1m | 5m | 15m | 1h | 4h | 1d")
    p.add_argument("--since",          default="2022-01-01",
                   help="Download start date YYYY-MM-DD")
    p.add_argument("--synthetic",      action="store_true",
                   help="Use synthetic price data instead of downloading (for testing)")
    # environment
    p.add_argument("--episodes",       type=int,   default=1000,
                   help="Training updates (train) or number of eval episodes (evaluate)")
    p.add_argument("--max-steps",      type=int,   default=2000)
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--learning-rate",  type=float, default=1e-4)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         default=None,
                   help="mps | cuda | cpu  (default: auto-detect)")
    p.add_argument("--windows",        type=int,   default=30,
                   help="Number of historical bars in each observation")
    p.add_argument("--trading-fees",   type=float, default=0.01/100)
    p.add_argument("--borrow-rate",    type=float, default=0.0003/100)
    p.add_argument("--portfolio-value",type=float, default=1000.0)
    p.add_argument("--min-hold-steps", type=int,   default=5,
                   help="Minimum bars to hold each position before switching (prevents overtrading)")
    # model
    p.add_argument("--hidden-dim",     type=int,   default=256)
    p.add_argument("--transformer-embed-dim",  type=int, default=64)
    p.add_argument("--transformer-num-layers", type=int, default=2)
    # training
    p.add_argument("--num-workers",    type=int,   default=4)
    p.add_argument("--min-steps",      type=int,   default=1000)
    p.add_argument("--no-normalize-returns", action="store_true")
    p.add_argument("--entropy-coef",   type=float, default=0.001)
    # output
    p.add_argument("--model-path",        default="market_trading/market_trading_policy.pt")
    p.add_argument("--save-plot",         default="market_trading/market_training_rewards.png")
    p.add_argument("--render-logs-dir",   default="market_trading/render_logs",
                   help="Directory for render log pickles (used by evaluate and render modes)")
    return p


def config_from_args(args: argparse.Namespace) -> MarketTradingConfig:
    exchange  = args.exchange
    symbol    = args.symbol
    timeframe = args.timeframe
    data_dir  = args.data_dir
    symbol_slug = symbol.replace("/", "")
    data_path   = f"{data_dir}/{exchange}-{symbol_slug}-{timeframe}.pkl"

    return MarketTradingConfig(
        data_dir                = data_dir,
        data_path               = data_path,
        exchange                = exchange,
        symbol                  = symbol,
        timeframe               = timeframe,
        download_since          = args.since,
        use_synthetic           = args.synthetic,
        episodes                = args.episodes,
        max_steps_per_episode   = args.max_steps,
        gamma                   = args.gamma,
        learning_rate           = args.learning_rate,
        seed                    = args.seed,
        device                  = select_device(args.device),
        windows                 = args.windows,
        trading_fees            = args.trading_fees,
        borrow_interest_rate    = args.borrow_rate,
        portfolio_initial_value = args.portfolio_value,
        hidden_dim              = args.hidden_dim,
        transformer_embed_dim   = args.transformer_embed_dim,
        transformer_num_layers  = args.transformer_num_layers,
        num_workers             = args.num_workers,
        min_steps_per_update    = args.min_steps,
        normalize_returns       = not args.no_normalize_returns,
        entropy_coef            = args.entropy_coef,
        min_hold_steps          = args.min_hold_steps,
        model_path              = args.model_path,
        save_plot_path          = args.save_plot,
        render_logs_dir         = args.render_logs_dir,
    )
