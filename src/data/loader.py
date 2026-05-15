"""ASX intraday data loader using yfinance."""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List


# ASX session hours by interval.
# 5m and finer: start at 10:15 to skip opening auction noise.
# 1h and coarser: start at 10:00 — the 10:00 bar is the first full hourly bar
#   and the auction noise is diluted across the whole hour.
_SESSION_START = {"1m": "10:15", "5m": "10:15", "15m": "10:15", "30m": "10:00", "1h": "10:00"}
SESSION_END = "16:00"
ASX_TIMEZONE = "Australia/Sydney"

# yfinance per-request limits by interval
_CHUNK_DAYS = {"1m": 7, "5m": 30, "15m": 30, "30m": 30, "1h": 90}


def fetch_bars(ticker: str, days: int = 58, interval: str = "5m") -> pd.DataFrame:
    """
    Download OHLCV bars for an ASX ticker by fetching in chunks sized to yfinance's
    per-request limit for the given interval.

    Args:
        ticker: ASX ticker without suffix, e.g. "CBA" — .AX appended automatically.
        days: total calendar days of history to fetch.
        interval: bar size — "1m", "5m", "15m", "30m", "1h".

    Returns:
        DataFrame with columns [open, high, low, close, volume],
        DatetimeIndex in ASX_TIMEZONE, restricted to session hours.
    """
    # Indices (^AXJO) and already-suffixed tickers don't need .AX appended
    symbol = ticker if (ticker.endswith(".AX") or ticker.startswith("^")) else f"{ticker}.AX"
    chunk_size = _CHUNK_DAYS.get(interval, 30)
    end = datetime.now()
    chunks = []

    cursor = end
    while cursor > end - timedelta(days=days):
        chunk_start = max(cursor - timedelta(days=chunk_size), end - timedelta(days=days))
        raw = yf.download(
            symbol,
            start=chunk_start.strftime("%Y-%m-%d"),
            end=cursor.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if not raw.empty:
            chunks.append(raw)
        cursor = chunk_start

    if not chunks:
        raise ValueError(f"No data returned for {symbol}. Check ticker or network.")

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Flatten multi-level columns yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]

    # Localise index to ASX timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ASX_TIMEZONE)

    # Filter to clean session hours only
    session_start = _SESSION_START.get(interval, "10:15")
    df = df.between_time(session_start, SESSION_END)
    df = df.dropna()
    return df


def fetch_universe(tickers: List[str], days: int = 58, interval: str = "5m") -> Dict[str, pd.DataFrame]:
    """
    Fetch bars for a list of ASX tickers. Returns dict keyed by ticker (no .AX suffix).
    Silently skips tickers that return no data, printing a warning.
    """
    universe: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            bars = fetch_bars(ticker, days=days, interval=interval)
            universe[ticker] = bars
            print(f"  {ticker}: {len(bars)} bars loaded")
        except ValueError as e:
            print(f"  WARNING — skipping {ticker}: {e}")
    return universe
