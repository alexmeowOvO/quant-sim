"""
paper_trade.py — Daily paper trading runner.

Run once after ASX market close (~4:15 PM Sydney time), or automatically
via the LaunchAgent at 5 PM. Re-runs the simulation on a 60-day rolling
window, then reports today's entries, exits, and open positions.

Usage:
    python paper_trade.py
    python paper_trade.py --config config/default.yaml
    python paper_trade.py --force   # bypass time/date guards (for testing)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, date

import pytz
import yaml

warnings.filterwarnings("ignore")

from src.data.loader import fetch_universe, fetch_bars, ASX_TIMEZONE
from src.engine.event_loop import run_simulation, SimConfig
from src.reporting.metrics import compute_metrics
from src.strategy.signals import compute_signals
from main import build_tier_maps, load_config, tier_signal_params

STATE_FILE = "results/paper_trade_state.json"
LOG_FILE   = "results/paper_trade_log.csv"

LOG_FIELDS = [
    "date", "portfolio_value", "total_return_pct",
    "open_positions", "entries_today", "exits_today",
    "total_trades", "sharpe", "win_rate_pct",
]


def _notify(message: str) -> None:
    subprocess.run(
        ["osascript", "-e", f'display notification "{message}" with title "Paper Trader"'],
        check=False, capture_output=True,
    )


def _append_log(row: dict) -> None:
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_paper_trade(cfg: dict, force: bool = False) -> None:
    os.makedirs("results", exist_ok=True)

    asx_tz  = pytz.timezone(ASX_TIMEZONE)
    now     = datetime.now(tz=asx_tz)
    today   = now.date()

    # ── Guard: only run between 4 PM and midnight Sydney time ────────────────
    if not force and not (16 <= now.hour < 24):
        print(f"Outside window ({now.strftime('%H:%M')} AEST) — skipping. Use --force to override.")
        sys.exit(0)

    tiers_cfg = cfg["universe"]["tiers"]
    data_cfg  = cfg["data"]
    pt_cfg    = cfg.get("paper_trade", {})
    strat     = cfg["strategy"]
    exec_cfg  = cfg["execution"]
    port_cfg  = cfg["portfolio"]
    risk_cfg  = cfg["risk"]

    days = pt_cfg.get("days", 60)

    ticker_list, ticker_tier_map, tier_params_map = build_tier_maps(cfg)

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    print(f"Fetching {len(ticker_list)} symbols ({days}d of {data_cfg['interval']} bars)...")
    raw_universe = fetch_universe(ticker_list, days=days, interval=data_cfg["interval"])
    if not raw_universe:
        print("ERROR: no data loaded.")
        sys.exit(1)

    # ── Guard: skip non-trading days ──────────────────────────────────────────
    latest_bar_date = max(df.index[-1].date() for df in raw_universe.values())
    if not force and latest_bar_date < today:
        print(f"No new data (latest bar: {latest_bar_date}) — market closed today.")
        sys.exit(0)

    # ── 2. Market regime filter ───────────────────────────────────────────────
    market_ok = None
    mf_cfg    = cfg.get("market_filter", {})
    mf_window = mf_cfg.get("sma_window", 0)
    if mf_window > 0:
        mf_ticker = mf_cfg.get("ticker", "^AXJO")
        try:
            mf_df     = fetch_bars(mf_ticker, days=days, interval=data_cfg["interval"])
            mf_sma    = mf_df["close"].rolling(mf_window, min_periods=mf_window).mean()
            market_ok = mf_df["close"] > mf_sma
        except Exception as e:
            print(f"  WARNING: market filter fetch failed ({e})")

    # ── 3. Compute signals ────────────────────────────────────────────────────
    signalled_universe = {}
    for ticker, df in raw_universe.items():
        tier_name = ticker_tier_map.get(ticker, "medium")
        sig_params = tier_signal_params(tiers_cfg[tier_name], strat)
        signalled_universe[ticker] = compute_signals(df, **sig_params)

    # ── 4. Run simulation ─────────────────────────────────────────────────────
    sim_cfg = SimConfig(
        initial_capital          = port_cfg["initial_capital"],
        position_size_pct        = port_cfg["position_size_pct"],
        max_concurrent_positions = strat.get("max_concurrent_positions", 5),
        slippage_pct             = exec_cfg["slippage_pct"],
        commission               = exec_cfg["commission_per_trade"],
        stop_loss_pct            = risk_cfg["stop_loss_pct"],
        ticker_tiers             = ticker_tier_map,
        tier_params              = tier_params_map,
    )
    trades, equity_curve, open_positions = run_simulation(
        signalled_universe, sim_cfg, market_filter=market_ok, return_open=True
    )

    # ── 5. Today's activity ───────────────────────────────────────────────────
    # Use bar dates directly — handles same-day open-and-close correctly.
    def bar_date(ticker: str, bar_idx: int) -> date:
        return signalled_universe[ticker].index[bar_idx].date()

    entries_today = sorted({
        ticker
        for ticker, pos in open_positions.items()
        if bar_date(ticker, pos.entry_bar) == today
    } | {
        t.ticker for t in trades
        if bar_date(t.ticker, t.entry_bar) == today
    })

    exits_today = [t for t in trades if bar_date(t.ticker, t.exit_bar) == today]

    # ── 6. Open position snapshot ─────────────────────────────────────────────
    open_pos_data = {}
    for ticker, pos in open_positions.items():
        df            = signalled_universe[ticker]
        current_price = df["close"].iloc[-1]
        # Entry commission already paid; deduct anticipated exit commission too
        unrealised    = (current_price - pos.entry_price) * pos.shares - 2 * sim_cfg.commission
        open_pos_data[ticker] = {
            "entry_date":     df.index[pos.entry_bar].strftime("%Y-%m-%d %H:%M"),
            "entry_price":    round(pos.entry_price, 4),
            "current_price":  round(current_price, 4),
            "shares":         round(pos.shares, 4),
            "unrealised_pnl": round(unrealised, 2),
            "stop_price":     round(pos.stop_price, 4),
            "regime":         pos.entry_regime,
            "bars_held":      len(df) - 1 - pos.entry_bar,
            "max_hold_bars":  pos.max_hold_bars,
        }

    # ── 7. Metrics ────────────────────────────────────────────────────────────
    metrics         = compute_metrics(trades, equity_curve, sim_cfg.initial_capital) if trades else {}
    portfolio_value = equity_curve.iloc[-1]
    total_return    = (portfolio_value - sim_cfg.initial_capital) / sim_cfg.initial_capital * 100

    # ── 8. Print summary ──────────────────────────────────────────────────────
    now_str = now.strftime("%Y-%m-%d %H:%M %Z")
    print(f"\n{'='*62}")
    print(f"  PAPER TRADING DAILY SUMMARY — {now_str}")
    print(f"{'='*62}")

    print(f"\n  Portfolio value   ${portfolio_value:>12,.2f}")
    print(f"  Initial capital   ${sim_cfg.initial_capital:>12,.2f}")
    print(f"  Total return      {total_return:>+12.2f}%")
    if "sharpe_ratio" in metrics:
        print(f"  Sharpe ratio      {metrics['sharpe_ratio']:>12.3f}")
        print(f"  Max drawdown      {metrics['max_drawdown_pct']:>+12.2f}%")
        print(f"  Win rate          {metrics['win_rate_pct']:>12.1f}%")
        print(f"  Total trades      {metrics['total_trades']:>12d}")

    if market_ok is not None:
        regime_now = "OPEN (above SMA)" if bool(market_ok.iloc[-1]) else "BLOCKED (below SMA)"
        print(f"  Market regime     {regime_now}")

    if entries_today:
        print(f"\n  ── NEW ENTRIES TODAY ({len(entries_today)}) ──")
        for ticker in entries_today:
            if ticker in open_pos_data:
                p = open_pos_data[ticker]
                print(f"    {ticker:4s}  entry={p['entry_price']:.3f}  "
                      f"stop={p['stop_price']:.3f}  regime={p['regime']}")
            else:
                # Same-day open and close
                t = next(tr for tr in exits_today if tr.ticker == ticker)
                print(f"    {ticker:4s}  entry={t.entry_price:.3f}  "
                      f"closed same day  pnl={t.pnl:+.2f}")

    if exits_today:
        print(f"\n  ── EXITS TODAY ({len(exits_today)}) ──")
        for t in exits_today:
            print(f"    {t.ticker:4s}  pnl={t.pnl:+.2f}  "
                  f"reason={t.exit_reason}  exit={t.exit_price:.3f}")

    if open_pos_data:
        print(f"\n  ── OPEN POSITIONS ({len(open_pos_data)}) ──")
        for ticker, p in sorted(open_pos_data.items()):
            print(f"    {ticker:4s}  entry={p['entry_price']:.3f}  "
                  f"now={p['current_price']:.3f}  "
                  f"unrealised={p['unrealised_pnl']:+.2f}  "
                  f"stop={p['stop_price']:.3f}  "
                  f"bars={p['bars_held']}/{p['max_hold_bars']}")
    else:
        print("\n  OPEN POSITIONS: none")

    if not entries_today and not exits_today and not open_pos_data:
        print("\n  No activity — waiting for signals.")

    print(f"\n{'='*62}\n")

    # ── 9. Notification ───────────────────────────────────────────────────────
    if entries_today or exits_today:
        parts = []
        if entries_today:
            parts.append(f"Entries: {', '.join(entries_today)}")
        if exits_today:
            parts.append(f"Exits: {', '.join(t.ticker for t in exits_today)}")
        _notify("  |  ".join(parts))

    # ── 10. Persist state ─────────────────────────────────────────────────────
    state = {
        "last_run":         now_str,
        "last_data_date":   str(latest_bar_date),
        "portfolio_value":  round(portfolio_value, 2),
        "total_return_pct": round(total_return, 2),
        "open_positions":   open_pos_data,
        "metrics":          metrics if "error" not in metrics else {},
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    # ── 11. Append to CSV log ─────────────────────────────────────────────────
    today_str = today.strftime("%Y-%m-%d")
    _append_log({
        "date":             today_str,
        "portfolio_value":  round(portfolio_value, 2),
        "total_return_pct": round(total_return, 2),
        "open_positions":   len(open_pos_data),
        "entries_today":    " ".join(entries_today),
        "exits_today":      " ".join(t.ticker for t in exits_today),
        "total_trades":     metrics.get("total_trades", 0),
        "sharpe":           metrics.get("sharpe_ratio", ""),
        "win_rate_pct":     metrics.get("win_rate_pct", ""),
    })

    print(f"  State → {STATE_FILE}")
    print(f"  Log   → {LOG_FILE}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASX paper trader — daily runner")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Bypass time and date guards (for testing)")
    args = parser.parse_args()
    run_paper_trade(load_config(args.config), force=args.force)
