"""
signal_logger.py — Daily observation logger (NOT a paper trader).

Runs after ASX close (5pm Sydney time). Fetches latest bars, computes signals
using the frozen config, and logs:
  - ENTRY signals (conditions fully met)
  - NEAR signals (1 condition short of entry)

Does NOT track portfolio state, positions, cash, or PnL.
Purpose: observe whether the frozen config produces sane live signals.

Output:
  results/daily_signals.csv   — one row per signal per day
  results/daily_signals.log   — human-readable summary

Usage:
  python3 signal_logger.py                  # use default frozen config
  python3 signal_logger.py --config config/experiment_100stocks.yaml
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")
import pandas as pd
import yaml

sys.path.insert(0, "/Users/alex/quant-sim")
from main import load_config, build_tier_maps, tier_signal_params
from src.data.loader import fetch_universe, fetch_bars
from src.strategy.signals import compute_signals

SIGNALS_CSV = "results/daily_signals.csv"
SIGNALS_LOG = "results/daily_signals.log"
FROZEN_CONFIG = "config/baseline_73stocks_sharpe1975.yaml"
SYDNEY_TZ = ZoneInfo("Australia/Sydney")

# How close to a signal counts as "near miss" (fraction of bb_lower distance)
NEAR_MISS_THRESHOLD = 0.995   # price within 0.5% of bb_lower


def compute_signal_state(sig: pd.DataFrame, ticker: str, tier_cfg: dict) -> dict:
    """
    Inspect the most recent completed bar and return signal state.
    Returns dict with all observable dimensions.
    """
    if sig is None or len(sig) < 2:
        return None

    # Use the last completed bar (index -1 is current partial bar if intraday)
    row = sig.iloc[-1]

    close       = row.get("close",       float("nan"))
    bb_lower    = row.get("bb_lower",    float("nan"))
    vwap        = row.get("vwap",        float("nan"))
    adx         = row.get("adx",         float("nan"))
    atr         = row.get("atr",         float("nan"))
    regime      = row.get("regime",      "unknown")
    signal_entry = bool(row.get("signal_entry", False))

    vwap_threshold = tier_cfg.get("vwap_threshold", 0.005)
    adx_threshold  = tier_cfg.get("adx_threshold",  20)

    # Individual condition checks
    cond_bb     = close <= bb_lower if pd.notna(close) and pd.notna(bb_lower) else False
    cond_vwap   = close < vwap * (1 - vwap_threshold) if pd.notna(close) and pd.notna(vwap) else False
    cond_adx    = (adx < adx_threshold) if pd.notna(adx) else False
    cond_regime = regime == "ranging"

    conditions_met = sum([cond_bb, cond_vwap, cond_adx, cond_regime])

    # Near miss: price within NEAR_MISS_THRESHOLD of bb_lower (not yet touching)
    near_bb = (
        pd.notna(close) and pd.notna(bb_lower) and
        close > bb_lower and
        close <= bb_lower / NEAR_MISS_THRESHOLD
    )

    # Reason string
    reasons = []
    if signal_entry:
        reasons.append("ENTRY")
    else:
        if not cond_bb:
            reasons.append("bb_not_touched" + (" (near)" if near_bb else ""))
        if not cond_vwap:
            reasons.append("vwap_not_breached")
        if not cond_adx:
            reasons.append("adx_too_high")
        if not cond_regime:
            reasons.append("not_ranging")

    return {
        "close":           round(float(close), 4)      if pd.notna(close)    else None,
        "bb_lower":        round(float(bb_lower), 4)   if pd.notna(bb_lower) else None,
        "vwap":            round(float(vwap), 4)        if pd.notna(vwap)     else None,
        "adx":             round(float(adx), 2)         if pd.notna(adx)      else None,
        "atr":             round(float(atr), 4)         if pd.notna(atr)      else None,
        "regime":          regime,
        "signal_entry":    signal_entry,
        "near_entry":      near_bb and not signal_entry,
        "cond_bb":         cond_bb,
        "cond_vwap":       cond_vwap,
        "cond_adx":        cond_adx,
        "cond_regime":     cond_regime,
        "conditions_met":  conditions_met,
        "reason":          "; ".join(reasons) if reasons else "no_signal",
    }


def run_logger(config_path: str) -> None:
    run_time  = datetime.now(SYDNEY_TZ)
    today     = run_time.date().isoformat()
    now       = run_time.strftime("%H:%M")
    cfg       = load_config(config_path)
    tiers_cfg = cfg["universe"]["tiers"]
    data_cfg  = cfg["data"]
    strat     = cfg["strategy"]

    ticker_list, ticker_tier_map, _ = build_tier_maps(cfg)

    print(f"\n{'='*60}")
    print(f"  SIGNAL LOGGER  —  {today}  {now} Sydney")
    print(f"  Config: {config_path}")
    print(f"  Universe: {len(ticker_list)} stocks")
    print(f"{'='*60}")

    # Fetch only last 30 days to keep it fast (enough for indicators to settle)
    print("\nFetching data (30d)...")
    raw_universe = fetch_universe(ticker_list, days=30, interval="1h")
    if not raw_universe:
        print("ERROR: no data loaded.")
        return

    print(f"  Loaded {len(raw_universe)} tickers\n")

    rows = []
    entries    = []
    near_misses = []

    for ticker, df in sorted(raw_universe.items()):
        tier_name = ticker_tier_map.get(ticker, "tier_a")
        tier_cfg  = tiers_cfg[tier_name]
        sp        = tier_signal_params(tier_cfg, strat)
        sig       = compute_signals(df, **sp)
        state     = compute_signal_state(sig, ticker, tier_cfg)
        if state is None:
            continue

        row = {
            "date":          today,
            "time":          now,
            "ticker":        ticker,
            "tier":          tier_name,
            "signal_entry":  state["signal_entry"],
            "near_entry":    state["near_entry"],
            "close":         state["close"],
            "bb_lower":      state["bb_lower"],
            "vwap":          state["vwap"],
            "adx":           state["adx"],
            "atr":           state["atr"],
            "regime":        state["regime"],
            "conditions_met": state["conditions_met"],
            "reason":        state["reason"],
        }
        rows.append(row)

        if state["signal_entry"]:
            entries.append(row)
        elif state["near_entry"]:
            near_misses.append(row)

    # Print summary
    print(f"  ENTRY SIGNALS ({len(entries)}):")
    if entries:
        for r in sorted(entries, key=lambda x: x["tier"]):
            print(f"    ✓  {r['ticker']:<5} [{r['tier']}]  "
                  f"close={r['close']}  bb={r['bb_lower']}  "
                  f"vwap={r['vwap']}  adx={r['adx']}  regime={r['regime']}")
    else:
        print("    (none today)")

    print(f"\n  NEAR MISSES ({len(near_misses)}):")
    if near_misses:
        for r in sorted(near_misses, key=lambda x: x["tier"]):
            pct_away = ""
            if r["close"] and r["bb_lower"]:
                pct = 100 * (r["close"] / r["bb_lower"] - 1)
                pct_away = f"  ({pct:.2f}% above bb)"
            print(f"    ~  {r['ticker']:<5} [{r['tier']}]  "
                  f"close={r['close']}  bb={r['bb_lower']}{pct_away}")
    else:
        print("    (none today)")

    print(f"\n  Total: {len(rows)} stocks checked  |  "
          f"{len(entries)} entries  |  {len(near_misses)} near misses")
    print(f"{'='*60}\n")

    # Append to CSV
    os.makedirs("results", exist_ok=True)
    df_new = pd.DataFrame(rows)
    if os.path.exists(SIGNALS_CSV):
        df_new.to_csv(SIGNALS_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(SIGNALS_CSV, index=False)

    # Append to log
    with open(SIGNALS_LOG, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"  {today}  {now}  |  {len(entries)} entries  |  {len(near_misses)} near\n")
        f.write(f"{'='*60}\n")
        if entries:
            f.write("ENTRIES:\n")
            for r in entries:
                f.write(f"  {r['ticker']:<5} [{r['tier']}]  close={r['close']}  "
                        f"bb={r['bb_lower']}  vwap={r['vwap']}  adx={r['adx']}\n")
        if near_misses:
            f.write("NEAR MISSES:\n")
            for r in near_misses:
                f.write(f"  {r['ticker']:<5} [{r['tier']}]  close={r['close']}  bb={r['bb_lower']}\n")

    print(f"Saved → {SIGNALS_CSV}")
    print(f"Saved → {SIGNALS_LOG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=FROZEN_CONFIG)
    args = parser.parse_args()
    os.chdir("/Users/alex/quant-sim")
    run_logger(args.config)
