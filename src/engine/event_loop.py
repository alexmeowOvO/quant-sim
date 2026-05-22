"""Event-driven simulation engine."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class Position:
    ticker: str
    entry_price: float          # filled at next-bar open after signal
    entry_bar: int              # bar index when entered
    size: float                 # AUD dollar size allocated
    shares: float               # shares held (size / entry_price, adjusted for slippage)
    stop_price: float
    max_hold_bars: int
    entry_regime: str           # "ranging" or "trending"


@dataclass
class Trade:
    ticker: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    shares: float
    pnl: float                  # AUD, net of commission
    pnl_gross: float            # AUD, before commission
    exit_reason: str            # "signal", "stop", "timeout"
    entry_regime: str           # "ranging" or "trending"


@dataclass
class TierParams:
    atr_multiplier: float = 2.0   # stop = fill_price - atr_multiplier * ATR
    max_hold_bars: int = 10


@dataclass
class SimConfig:
    initial_capital: float = 100_000.0
    position_size_pct: float = 0.10
    max_concurrent_positions: int = 5
    slippage_pct: float = 0.0005
    commission: float = 10.0
    # Fallback stop used when ATR is unavailable (NaN at start of series)
    stop_loss_pct: float = 0.010
    # Fallback hold bars used when ticker has no tier assignment
    max_hold_bars: int = 10
    # Per-ticker tier assignment: ticker → tier name ("stable"/"medium"/"high")
    ticker_tiers: Dict[str, str] = field(default_factory=dict)
    # Per-tier params: tier name → TierParams
    tier_params: Dict[str, "TierParams"] = field(default_factory=dict)


def run_simulation(
    universe: Dict[str, pd.DataFrame],
    cfg: SimConfig,
    market_filter: Optional[pd.Series] = None,
    return_open: bool = False,
) -> tuple:
    """
    Event-driven simulation over a pre-signalled universe.

    universe: dict of ticker → DataFrame with signal columns already computed
              (output of compute_signals). Must share a common DatetimeIndex.
    cfg:      SimConfig

    Returns:
        trades:      list of completed Trade records
        equity_curve: pd.Series of portfolio value at each bar, indexed by timestamp
    """
    # Build unified sorted timeline of all bar timestamps
    all_timestamps = sorted(
        set(ts for df in universe.values() for ts in df.index)
    )

    # Align market filter to simulation timeline via forward-fill.
    # True = market above SMA (entries allowed), False = below (no new entries).
    # Missing leading bars (before SMA warmup) default to True.
    if market_filter is not None:
        market_filter = (
            market_filter
            .reindex(all_timestamps)
            .ffill()
            .infer_objects(copy=False)
            .fillna(True)
            .astype(bool)
        )

    capital = cfg.initial_capital
    positions: Dict[str, Position] = {}     # ticker → open Position
    trades: List[Trade] = []
    equity_history: List[tuple] = []
    missing_bar_log: List[dict] = []        # forward-fill events for data-quality audit

    for ts in all_timestamps:
        # --- 1. Process exits first (use current bar's open as fill price) ---
        for ticker in list(positions.keys()):
            df = universe[ticker]
            if ts not in df.index:
                continue
            pos = positions[ticker]
            bar = df.loc[ts]
            current_bar_num = df.index.get_loc(ts)
            bars_held = current_bar_num - pos.entry_bar

            fill_open = bar["open"] * (1 - cfg.slippage_pct)   # sell side slippage

            exit_reason: Optional[str] = None

            # Use previous bar's signal_exit so decision (close) and execution (open) are
            # on different bars — mirrors how entries work (signal at bar N, fill at N+1).
            prev_signal_exit = (
                df.iloc[current_bar_num - 1]["signal_exit"]
                if current_bar_num > 0 else False
            )

            # Stop loss — check against bar low (could have been triggered intrabar)
            if bar["low"] <= pos.stop_price:
                fill_open = pos.stop_price * (1 - cfg.slippage_pct)
                exit_reason = "stop"
            elif prev_signal_exit and bars_held > 0:
                exit_reason = "signal"
            elif bars_held >= pos.max_hold_bars:
                exit_reason = "timeout"

            if exit_reason:
                proceeds = fill_open * pos.shares - cfg.commission
                pnl = proceeds - (pos.entry_price * pos.shares + cfg.commission)
                pnl_gross = (fill_open - pos.entry_price) * pos.shares
                capital += pos.entry_price * pos.shares + cfg.commission + pnl
                trades.append(Trade(
                    ticker=ticker,
                    entry_bar=pos.entry_bar,
                    exit_bar=current_bar_num,
                    entry_price=pos.entry_price,
                    exit_price=fill_open,
                    shares=pos.shares,
                    pnl=pnl,
                    pnl_gross=pnl_gross,
                    exit_reason=exit_reason,
                    entry_regime=pos.entry_regime,
                ))
                del positions[ticker]

        # --- 2. Collect and rank entry signals ---
        market_open = market_filter is None or bool(market_filter.loc[ts])
        if len(positions) < cfg.max_concurrent_positions and market_open:
            slots_available = cfg.max_concurrent_positions - len(positions)
            candidates = []

            for ticker, df in universe.items():
                if ticker in positions:
                    continue
                if ts not in df.index:
                    continue
                bar = df.loc[ts]
                if bar["signal_entry"] and bar["signal_strength"] > 0:
                    candidates.append((ticker, bar["signal_strength"]))

            # Rank by signal strength, take top N
            candidates.sort(key=lambda x: x[1], reverse=True)
            for ticker, _ in candidates[:slots_available]:
                df = universe[ticker]
                bar_num = df.index.get_loc(ts)

                # Entry fills at NEXT bar's open — peek ahead if possible
                if bar_num + 1 >= len(df):
                    continue
                next_bar = df.iloc[bar_num + 1]
                fill_price = next_bar["open"] * (1 + cfg.slippage_pct)

                position_size = capital * cfg.position_size_pct
                if position_size < fill_price:
                    continue  # can't afford even one share

                shares = position_size / fill_price
                cost = fill_price * shares + cfg.commission

                if cost > capital:
                    continue

                # ATR-based stop: use tier's multiplier, fall back to fixed pct if ATR is NaN
                tier_name = cfg.ticker_tiers.get(ticker, "")
                tp = cfg.tier_params.get(tier_name, TierParams(
                    max_hold_bars=cfg.max_hold_bars,
                ))
                atr_val = next_bar["atr"] if "atr" in next_bar.index else np.nan
                if pd.notna(atr_val) and atr_val > 0:
                    stop_price = fill_price - tp.atr_multiplier * atr_val
                    # Floor: stop must be at least stop_loss_pct below fill (prevents
                    # ultra-tight ATR stops on very low-volatility bars)
                    floor = fill_price * (1 - cfg.stop_loss_pct)
                    stop_price = min(stop_price, floor)
                else:
                    stop_price = fill_price * (1 - cfg.stop_loss_pct)

                capital -= cost
                positions[ticker] = Position(
                    ticker=ticker,
                    entry_price=fill_price,
                    entry_bar=bar_num + 1,
                    size=position_size,
                    shares=shares,
                    stop_price=stop_price,
                    max_hold_bars=tp.max_hold_bars,
                    entry_regime=df.loc[ts]["regime"],
                )

        # --- 3. Mark-to-market equity ---
        # Forward-fill price for open positions whose stock has no bar at this
        # timestamp (e.g. holiday gaps, cross-listed stocks with different
        # calendars).  Silently dropping the position's value would produce a
        # spurious spike down in the equity curve and inflate max_drawdown.
        open_value = 0.0
        for t, p in positions.items():
            df = universe[t]
            if ts in df.index:
                price = float(df.loc[ts, "close"])
            else:
                earlier = df.index[df.index < ts]
                price = float(df.loc[earlier[-1], "close"]) if len(earlier) > 0 else p.entry_price
                # Data-quality warning: log whenever forward-fill fires so we
                # can distinguish one-off Yahoo glitches from structural
                # calendar mismatches (e.g. NZ stocks vs ASX holidays).
                missing_bar_log.append({
                    "timestamp": str(ts),
                    "ticker":    t,
                    "last_known_close": round(price, 4),
                    "reason": "missing_bar_forward_fill",
                })
            open_value += price * p.shares
        equity_history.append((ts, capital + open_value))

    equity_curve = pd.Series(
        {ts: val for ts, val in equity_history},
        name="equity",
    )
    if return_open:
        return trades, equity_curve, positions, missing_bar_log
    return trades, equity_curve, missing_bar_log
