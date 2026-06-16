"""Data loading, downloading, and feature engineering for market-trading environments."""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def ensure_data(
    data_dir:  str,
    exchange:  str = "binance",
    symbol:    str = "BTC/USDT",
    timeframe: str = "5m",
    since:     str = "2022-01-01",
) -> str:
    """Downloads OHLCV data if not already present and returns the file path.

    The downloader (gym-trading-env / ccxt) saves files as:
        {data_dir}/{exchange}-{symbol_no_slash}-{timeframe}.pkl

    Parameters
    ----------
    data_dir:  Directory where the pickle file will be stored.
    exchange:  ccxt exchange id (e.g. "binance", "huobi", "bitfinex2").
    symbol:    Market symbol in ccxt format (e.g. "BTC/USDT").
    timeframe: Bar size string (e.g. "5m", "1h", "1d").
    since:     Download start date as "YYYY-MM-DD" string.

    Returns
    -------
    Absolute path to the pickle file.
    """

    symbol_slug = symbol.replace("/", "")
    file_path   = Path(data_dir) / f"{exchange}-{symbol_slug}-{timeframe}.pkl"

    if file_path.exists():
        print(f"[data] Found existing data at {file_path}")
        return str(file_path)

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"[data] Downloading {symbol} {timeframe} from {exchange} "
        f"since {since} → {file_path}"
    )

    from gym_trading_env.downloader import download

    since_dt = datetime.datetime.strptime(since, "%Y-%m-%d")
    download(
        exchange_names = [exchange],
        symbols        = [symbol],
        timeframe      = timeframe,
        dir            = data_dir,
        since          = since_dt,
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Download completed but expected file not found: {file_path}"
        )

    print(f"[data] Download complete: {file_path}")
    return str(file_path)


def load_dataframe(path: str) -> pd.DataFrame:
    """Loads an OHLCV file (pickle or CSV) and validates required columns.

    Accepts both:
    - .pkl  files produced by gym-trading-env's downloader
    - .csv  files with a datetime index in the first column

    Columns are lowercased automatically.

    Raises
    ------
    ValueError
        If required OHLCV columns are missing after normalisation.
    """

    p = Path(path)
    if p.suffix == ".pkl":
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

    df.columns = [c.lower().strip() for c in df.columns]

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required OHLCV columns: {sorted(missing)}"
        )

    df.sort_index(inplace=True)
    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — manual implementation (no external TA lib)."""

    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l  = loss.ewm(com=period - 1, min_periods=period).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes technical indicators and adds them as 'feature_*' columns.

    gym-trading-env auto-detects any column whose name contains 'feature' as
    an input feature to the agent.

    Features added
    --------------
    feature_pct_change   : close-to-close return
    feature_hl_ratio     : high / low ratio (bar range)
    feature_vol_norm     : volume normalised by 24-period rolling max
    feature_rsi_14       : 14-period RSI, normalised to [0, 1]
    feature_ma_ratio_20  : close / 20-period SMA (trend direction)
    feature_ma_ratio_50  : close / 50-period SMA (longer trend)
    feature_bb_pct       : Bollinger Band %B — price position in recent range
    feature_momentum_5   : 5-bar return (short-term momentum)
    feature_volatility   : 20-period rolling std of returns, normalised

    Returns a copy with NaN rows removed.
    """

    df = df.copy()
    ret = df["close"].pct_change()

    df["feature_pct_change"]  = ret
    df["feature_hl_ratio"]    = df["high"] / df["low"]
    df["feature_vol_norm"]    = df["volume"] / df["volume"].rolling(24).max()
    df["feature_rsi_14"]      = _rsi(df["close"], 14) / 100.0

    # Trend
    df["feature_ma_ratio_20"] = df["close"] / df["close"].rolling(20).mean()
    df["feature_ma_ratio_50"] = df["close"] / df["close"].rolling(50).mean()

    # Bollinger Band %B: 0 = at lower band, 1 = at upper band
    bb_mid   = df["close"].rolling(20).mean()
    bb_std   = df["close"].rolling(20).std()
    df["feature_bb_pct"] = ((df["close"] - (bb_mid - 2 * bb_std)) /
                             (4 * bb_std + 1e-10)).clip(0.0, 1.0)

    # Short-term momentum
    df["feature_momentum_5"] = df["close"].pct_change(5)

    # Volatility regime — how noisy the current window is
    rv = ret.rolling(20).std()
    df["feature_volatility"] = rv / (rv.rolling(200).mean() + 1e-10)

    df.dropna(inplace=True)
    return df


def make_synthetic_dataframe(n_bars: int = 2000) -> pd.DataFrame:
    """Generates a synthetic OHLCV DataFrame for smoke-testing without real data."""

    rng   = np.random.default_rng(42)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n_bars)))
    noise = rng.uniform(0.995, 1.005, n_bars)
    df = pd.DataFrame(
        {
            "open":   close * rng.uniform(0.99, 1.01, n_bars),
            "high":   close * rng.uniform(1.00, 1.02, n_bars),
            "low":    close * rng.uniform(0.98, 1.00, n_bars),
            "close":  close,
            "volume": rng.uniform(1e5, 1e6, n_bars) * noise,
        },
        index=pd.date_range("2020-01-01", periods=n_bars, freq="1h"),
    )
    return df
