"""Regime-branched signal computation: mean reversion (ranging) + momentum (trending)."""

import numpy as np
import pandas as pd


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP, reset at the start of each trading day."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical * df["volume"]
    dates = df.index.date
    vwap = pd.Series(index=df.index, dtype=float)
    for date in np.unique(dates):
        mask = np.array(dates) == date
        cum_tp_vol = tp_vol[mask].cumsum()
        cum_vol = df["volume"][mask].cumsum()
        vwap[mask] = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap


def compute_bollinger(df: pd.DataFrame, window: int, std_dev: float) -> pd.DataFrame:
    """Rolling Bollinger Bands on close. Returns bb_mid, bb_upper, bb_lower, bb_width."""
    mid = df["close"].rolling(window, min_periods=window).mean()
    std = df["close"].rolling(window, min_periods=window).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": upper - lower},
        index=df.index,
    )


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Average True Range using Wilder smoothing (alpha = 1/window)."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def compute_adx(df: pd.DataFrame, window: int, atr: pd.Series = None) -> pd.Series:
    """
    Average Directional Index using Wilder smoothing (alpha = 1/window).
    Accepts a precomputed ATR series to avoid recomputing it.
    Returns ADX in range [0, 100]. < threshold = ranging, >= threshold = trending.
    """
    high, low, close = df["high"], df["low"], df["close"]

    if atr is None:
        atr = compute_atr(df, window)

    up_move = high.diff()
    down_move = -low.diff()
    pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    alpha = 1 / window
    pos_di = 100 * pos_dm.ewm(alpha=alpha, min_periods=window, adjust=False).mean() / atr
    neg_di = 100 * neg_dm.ewm(alpha=alpha, min_periods=window, adjust=False).mean() / atr

    di_sum = (pos_di + neg_di).replace(0, np.nan)
    dx = 100 * (pos_di - neg_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    return adx.fillna(0)


def compute_rsi(df: pd.DataFrame, window: int) -> pd.Series:
    """RSI using Wilder smoothing (EWM with com = window - 1). Returns series in [0, 100]."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def compute_ema(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Fast and slow EMA on close. Returns DataFrame with ema_fast, ema_slow."""
    return pd.DataFrame({
        "ema_fast": df["close"].ewm(span=fast, min_periods=fast, adjust=False).mean(),
        "ema_slow": df["close"].ewm(span=slow, min_periods=slow, adjust=False).mean(),
    }, index=df.index)


def compute_signals(
    df: pd.DataFrame,
    # Ranging (mean reversion) params
    bb_window: int = 20,
    bb_std: float = 2.0,
    vwap_threshold: float = 0.005,
    # Regime detection params
    adx_window: int = 14,
    adx_threshold: float = 25.0,
    min_trending_adx: float = 30.0,  # trending entry requires ADX >= this (>= adx_threshold)
    # Trending (momentum) params
    fast_ema: int = 9,
    slow_ema: int = 20,
    rsi_window: int = 14,
    rsi_overbought: int = 70,
    rsi_oversold: int = 30,
    # Medium-term trend filter — only take ranging entries above this SMA
    trend_filter_window: int = 50,
    # Disable trending (EMA crossover) branch entirely
    trending_enabled: bool = True,
) -> pd.DataFrame:
    """
    Regime-branched entry/exit signals.

    Ranging regime (ADX < adx_threshold):
      Entry: close < bb_lower AND close < vwap * (1 - vwap_threshold)
             AND close > trend_filter_window-bar SMA (avoids catching falling knives)
      Exit:  close >= bb_mid  OR  close >= vwap
      Strength: normalised distance below bb_lower

    Trending regime (ADX >= adx_threshold):
      Entry: fast EMA crosses above slow EMA
      Exit:  fast EMA crosses below slow EMA  OR  RSI >= rsi_overbought
      Strength: ADX / 100

    Note: exit signal reflects the regime at each bar, not the regime at entry.
    If ADX crosses the threshold while a position is open, the exit logic silently
    switches to match the new regime. The engine's stop-loss and timeout are the
    hard backstop regardless of regime. Locking exit to entry regime would require
    per-position state in the engine, which is out of scope here.

    Returns original columns plus all indicator columns and:
        signal_entry (bool), signal_exit (bool), signal_strength (float),
        regime (str: 'ranging' | 'trending')
    """
    close = df["close"]

    bb = compute_bollinger(df, bb_window, bb_std)
    vwap = compute_vwap(df)
    atr = compute_atr(df, adx_window)          # shared by ADX and stop sizing
    adx = compute_adx(df, adx_window, atr=atr)
    ema = compute_ema(df, fast_ema, slow_ema)
    rsi = compute_rsi(df, rsi_window)

    # --- Regime ---
    trending = adx >= adx_threshold

    # --- Medium-term trend filter (0 = disabled) ---
    if trend_filter_window > 0:
        sma_mid = close.rolling(trend_filter_window, min_periods=trend_filter_window).mean()
        above_sma = close > sma_mid
    else:
        sma_mid = pd.Series(np.nan, index=df.index)
        above_sma = pd.Series(True, index=df.index, dtype=bool)

    # --- Ranging signals ---
    ranging_entry   = (~trending) & (close < bb["bb_lower"]) & (close < vwap * (1 - vwap_threshold)) & above_sma
    ranging_exit    = (close >= bb["bb_mid"]) | (close >= vwap)
    ranging_strength = ((bb["bb_lower"] - close) / bb["bb_width"].replace(0, np.nan)).clip(lower=0).fillna(0)

    # --- Trending signals ---
    if trending_enabled:
        ema_cross_up      = (ema["ema_fast"] > ema["ema_slow"]) & (ema["ema_fast"].shift(1) <= ema["ema_slow"].shift(1))
        ema_cross_down    = (ema["ema_fast"] < ema["ema_slow"]) & (ema["ema_fast"].shift(1) >= ema["ema_slow"].shift(1))
        strong_trend      = adx >= max(adx_threshold, min_trending_adx)
        trending_entry    = strong_trend & ema_cross_up
        trending_exit     = ema_cross_down | (rsi >= rsi_overbought)
        trending_strength = (adx / 100).clip(0, 1)
    else:
        trending_entry    = pd.Series(False, index=df.index, dtype=bool)
        trending_exit     = pd.Series(False, index=df.index, dtype=bool)
        trending_strength = pd.Series(0.0,   index=df.index, dtype=float)

    # --- Combined ---
    signal_entry = ranging_entry | trending_entry
    signal_exit = pd.Series(
        np.where(trending, trending_exit, ranging_exit),
        index=df.index,
        dtype=bool,
    )
    # On non-entry bars: trending_entry=False so falls through to ranging_strength,
    # which is 0 above the lower band. Strength is only meaningful at entry bars.
    signal_strength = pd.Series(
        np.where(trending_entry, trending_strength, ranging_strength),
        index=df.index,
        dtype=float,
    )

    out = df.copy()
    out["vwap"] = vwap
    out["sma_mid"] = sma_mid
    out["atr"] = atr
    out = out.join(bb)
    out["adx"] = adx
    out["ema_fast"] = ema["ema_fast"]
    out["ema_slow"] = ema["ema_slow"]
    out["rsi"] = rsi
    out["regime"] = np.where(trending, "trending", "ranging")
    out["signal_entry"] = signal_entry
    out["signal_exit"] = signal_exit
    out["signal_strength"] = signal_strength
    return out
