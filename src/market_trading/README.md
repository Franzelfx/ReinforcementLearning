# market_trading

REINFORCE policy-gradient agent on BTC/USDT 5-minute bars using
[gym-trading-env](https://pypi.org/project/gym-trading-env/0.1.6/).

The policy is a Transformer encoder over a sliding window of OHLCV features
followed by a discrete categorical action head (portfolio position fractions).
Training is parallelised over multiple environment workers via `gymnasium`'s
`AsyncVectorEnv`.

## Folder layout

All artifacts are written inside this folder:

```
src/market_trading/
├── data/                        # downloaded OHLCV pickle(s)
│   └── binance-BTCUSDT-5m.pkl
├── render_logs/                 # evaluation episode logs (*.pkl)
├── market_trading_policy.pt     # saved model weights
├── market_training_rewards.png  # reward-per-update plot
├── __init__.py
├── config.py                    # MarketTradingConfig + CLI
├── data.py                      # download / feature engineering
├── model.py                     # TradingPolicyNetwork (Transformer)
├── renderer.py                  # Flask chart server
├── train.py                     # train / evaluate / render entry point
└── README.md
```

## Quick start

All commands are run from the `src/` directory.

### 1 — Install dependencies

```bash
pip install -r ../requirements.txt
```

### 2 — Train

Market data is downloaded automatically on first run (requires internet).

```bash
cd src
python3 -m market_trading.train
```

The default run trains for 300 updates with 4 parallel workers and at least
1 000 steps per update.  A reward plot is saved when training finishes.

Common overrides:

```bash
python3 -m market_trading.train \
    --episodes 500 \
    --learning-rate 1e-4 \
    --num-workers 8 \
    --min-steps 2000
```

Use synthetic data (no download, fast sanity check):

```bash
python3 -m market_trading.train --synthetic --episodes 5 --min-steps 200
```

### 3 — Evaluate

Runs sequential episodes and saves render logs for the interactive viewer.

```bash
python3 -m market_trading.train --mode evaluate --episodes 5
```

### 4 — Render (interactive chart)

Starts a local Flask server and opens the chart viewer in your browser.

```bash
python3 -m market_trading.train --mode render
```

Navigate between saved episodes using the dropdown on the right.
Stop the server with `Ctrl+C`.

> **macOS note:** the built-in gym-trading-env renderer is blocked by AirPlay
> Receiver on port 5000.  This project ships its own renderer (`renderer.py`)
> that auto-selects a free port starting at 5001.

## Configuration reference

All options can be listed with:

```bash
python3 -m market_trading.train --help
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `train` | `train` / `evaluate` / `render` |
| `--exchange` | `binance` | ccxt exchange id |
| `--symbol` | `BTC/USDT` | Market pair |
| `--timeframe` | `5m` | Bar size (`1m`, `5m`, `1h`, …) |
| `--since` | `2022-01-01` | Download start date |
| `--episodes` | `300` | Training updates (or eval episodes) |
| `--num-workers` | `4` | Parallel env workers |
| `--windows` | `30` | Observation window (bars) |
| `--synthetic` | off | Use random price data instead of download |
| `--device` | auto | `mps` / `cuda` / `cpu` |

## Architecture

```
Observation (B, window=30, n_features=8)
  │
  ├── Linear(n_features → embed_dim=64) + positional embeddings
  ├── TransformerEncoder (4 layers, nhead=4)
  └── Attention-weighted mean pool → (B, 64)
        │
        ├── Linear(64 → 256) → ReLU → Linear(256 → 5)
        └── Categorical over positions [-1, -0.5, 0, 0.5, 1]
```

Features added on top of raw OHLCV: `pct_change`, `oc_ratio`, `hl_ratio`,
`vol_norm`, `rsi_14`, `ma_ratio_20`.

Training uses REINFORCE with return normalisation, gradient clipping (norm 1.0),
and an entropy bonus (`--entropy-coef 0.01`).

## Shared `rl_core` components used

| Module | What is reused |
|--------|---------------|
| `rl_core.device` | `select_device()`, `set_global_seed()` |
| `rl_core.reinforce` | `reinforce_loss()` |
| `rl_core.rollout` | `collect_tabular_rollouts()`, `EpisodeBatch` |
| `rl_core.persistence` | `save_model()`, `load_model_if_available()` |
| `rl_core.plotting` | `save_reward_plot()` |
