"""Entry point — loads data, runs signals, executes simulation, prints report."""

import argparse
import os
from typing import Optional
import yaml
from src.data.loader import fetch_universe, fetch_bars
from src.strategy.signals import compute_signals
from src.engine.event_loop import run_simulation, SimConfig, TierParams
from src.reporting.metrics import compute_metrics, per_symbol_breakdown, print_report
from src.reporting.charts import plot_equity_curve, plot_signals


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_tier_maps(cfg: dict) -> tuple[list, dict, dict]:
    """
    Parse universe.tiers config into:
      ticker_list:      flat list of all tickers in tier order
      ticker_tier_map:  ticker → tier name
      tier_params_map:  tier name → TierParams
    """
    tiers = cfg["universe"]["tiers"]
    ticker_list = [t for tier in tiers.values() for t in tier["tickers"]]
    ticker_tier_map = {t: name for name, tier in tiers.items() for t in tier["tickers"]}
    tier_params_map = {
        name: TierParams(
            atr_multiplier=tier["atr_multiplier"],
            max_hold_bars=tier["max_hold_bars"],
        )
        for name, tier in tiers.items()
    }
    return ticker_list, ticker_tier_map, tier_params_map


def tier_signal_params(tier_cfg: dict, global_strat: dict) -> dict:
    """Extract compute_signals kwargs from a tier config block, falling back to global_strat."""
    return dict(
        bb_window=tier_cfg.get("bb_window", 20),
        bb_std=tier_cfg["bb_std"],
        vwap_threshold=tier_cfg.get("vwap_threshold", 0.005),
        adx_window=tier_cfg.get("adx_window", 14),
        adx_threshold=tier_cfg.get("adx_threshold", 25),
        min_trending_adx=tier_cfg.get("min_trending_adx", 40),
        fast_ema=global_strat["ema"]["fast"],
        slow_ema=global_strat["ema"]["slow"],
        rsi_window=global_strat["rsi"]["window"],
        rsi_overbought=global_strat["rsi"]["overbought"],
        rsi_oversold=global_strat["rsi"]["oversold"],
        trend_filter_window=tier_cfg.get("trend_filter_window", 100),
        trending_enabled=global_strat.get("trending_enabled", True),
    )


def main(cfg: dict, charts: bool = True, chart_ticker: Optional[str] = None) -> None:
    tiers_cfg   = cfg["universe"]["tiers"]
    data_cfg    = cfg["data"]
    strat       = cfg["strategy"]
    exec_cfg    = cfg["execution"]
    port_cfg    = cfg["portfolio"]
    risk_cfg    = cfg["risk"]

    os.makedirs("results", exist_ok=True)

    ticker_list, ticker_tier_map, tier_params_map = build_tier_maps(cfg)

    # 1. Load data
    print(f"Loading {len(ticker_list)} ASX symbols ({data_cfg['days']}d of {data_cfg['interval']} bars)...\n")
    raw_universe = fetch_universe(ticker_list, days=data_cfg["days"], interval=data_cfg["interval"])
    if not raw_universe:
        print("ERROR: no data loaded.")
        return

    # 1b. Market regime filter (^AXJO SMA — blocks entries in downtrends)
    market_ok = None
    mf_cfg = cfg.get("market_filter", {})
    mf_window = mf_cfg.get("sma_window", 0)
    if mf_window > 0:
        mf_ticker = mf_cfg.get("ticker", "^AXJO")
        print(f"Fetching market filter ({mf_ticker}, SMA={mf_window} bars)...")
        try:
            mf_df = fetch_bars(mf_ticker, days=data_cfg["days"], interval=data_cfg["interval"])
            mf_sma = mf_df["close"].rolling(mf_window, min_periods=mf_window).mean()
            market_ok = (mf_df["close"] > mf_sma)
            pct_open = market_ok.sum() / len(market_ok) * 100
            print(f"  {mf_ticker}: {len(mf_df)} bars — market above SMA({mf_window}): "
                  f"{pct_open:.0f}% of bars  (half-size entries {100-pct_open:.0f}% of the time)")
        except Exception as e:
            print(f"  WARNING: market filter fetch failed ({e}), filter disabled")

    # 2. Compute signals — each ticker uses its tier's own signal params
    print("\nComputing signals...")
    signalled_universe = {}
    for ticker, df in raw_universe.items():
        tier_name = ticker_tier_map.get(ticker, "medium")
        sig_params = tier_signal_params(tiers_cfg[tier_name], strat)
        signalled_universe[ticker] = compute_signals(df, **sig_params)
        sig = signalled_universe[ticker]
        ranging  = sig["signal_entry"] & (sig["regime"] == "ranging")
        trending = sig["signal_entry"] & (sig["regime"] == "trending")
        print(f"  {ticker:4s} [{tier_name:6s}] bb={sig_params['bb_window']}/std={sig_params['bb_std']}: "
              f"{sig['signal_entry'].sum()} entries "
              f"({ranging.sum()} ranging, {trending.sum()} trending)")

    # 3. Run simulation
    print("\nRunning simulation...")
    sim_cfg = SimConfig(
        initial_capital=port_cfg["initial_capital"],
        position_size_pct=port_cfg["position_size_pct"],
        max_concurrent_positions=strat.get("max_concurrent_positions", 5),
        slippage_pct=exec_cfg["slippage_pct"],
        commission=exec_cfg["commission_per_trade"],
        stop_loss_pct=risk_cfg["stop_loss_pct"],
        ticker_tiers=ticker_tier_map,
        tier_params=tier_params_map,
    )
    trades, equity_curve, missing_bar_log = run_simulation(signalled_universe, sim_cfg, market_filter=market_ok)
    print(f"  {len(trades)} trades completed.")
    if missing_bar_log:
        print(f"  ⚠  {len(missing_bar_log)} missing-bar forward-fill event(s) — see results/missing_bar_log.csv")

    # 4. Report
    metrics   = compute_metrics(trades, equity_curve, sim_cfg.initial_capital)
    breakdown = per_symbol_breakdown(trades)
    print_report(trades, metrics, breakdown)

    # 4b. Save trades CSV
    import pandas as pd
    if trades:
        ref_idx = next(iter(signalled_universe.values())).index
        def bar_to_ts(bar):
            return str(ref_idx[bar]) if bar < len(ref_idx) else str(bar)
        trades_df = pd.DataFrame([{
            "ticker":       t.ticker,
            "tier":         ticker_tier_map.get(t.ticker, ""),
            "entry_date":   bar_to_ts(t.entry_bar),
            "exit_date":    bar_to_ts(t.exit_bar),
            "entry_price":  round(t.entry_price, 4),
            "exit_price":   round(t.exit_price,  4),
            "shares":       round(t.shares, 2),
            "pnl":          round(t.pnl, 2),
            "exit_reason":  t.exit_reason,
        } for t in trades])
        trades_df.to_csv("results/trades.csv", index=False)

        # Stock breakdown CSV (breakdown is already a DataFrame)
        bd_df = breakdown.copy()
        bd_df.insert(1, "tier", bd_df["ticker"].map(lambda t: ticker_tier_map.get(t, "")))
        bd_df.to_csv("results/stock_breakdown.csv", index=False)

    # 4c. Missing-bar forward-fill log
    if missing_bar_log:
        mb_df = pd.DataFrame(missing_bar_log)
        mb_df.to_csv("results/missing_bar_log.csv", index=False)

    # 5. Tier summary + tier_breakdown.csv
    print("\nTIER BREAKDOWN (entry regime × volatility tier)")
    tier_rows = []
    for tier_name in cfg["universe"]["tiers"]:
        ts = [t for t in trades if ticker_tier_map.get(t.ticker) == tier_name]
        if not ts:
            print(f"  {tier_name:6s}: 0 trades")
            continue
        wins = [t for t in ts if t.pnl > 0]
        tp   = tier_params_map[tier_name]
        total_pnl = sum(t.pnl for t in ts)
        print(f"  {tier_name:6s}: {len(ts):3d} trades  "
              f"win={100*len(wins)//len(ts):2d}%  "
              f"atr_mult={tp.atr_multiplier}  hold≤{tp.max_hold_bars}h  "
              f"total_pnl={total_pnl:+.0f}")
        tier_rows.append({
            "tier":           tier_name,
            "trades":         len(ts),
            "win_rate_pct":   round(100 * len(wins) / len(ts), 1),
            "total_pnl":      round(total_pnl, 2),
            "atr_multiplier": tp.atr_multiplier,
            "max_hold_bars":  tp.max_hold_bars,
        })
    if tier_rows:
        pd.DataFrame(tier_rows).to_csv("results/tier_breakdown.csv", index=False)

    # 5b. summary_metrics.json
    import json, datetime as _dt
    summary = {**metrics,
               "date":    str(_dt.date.today()),
               "universe": f"{len(ticker_list)}_stocks",
               "tier_a_bb_std": cfg["universe"]["tiers"].get("tier_a", {}).get("bb_std"),
               "tier_b_bb_std": cfg["universe"]["tiers"].get("tier_b", {}).get("bb_std"),
    }
    with open("results/summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 6. Charts
    if charts:
        print("\nGenerating charts...")
        plot_equity_curve(
            equity_curve,
            sim_cfg.initial_capital,
            title="ASX Volatility-Tier Strategy — Equity Curve",
            save_path="results/equity_curve.png",
        )
        target = chart_ticker or (list(raw_universe.keys())[0] if raw_universe else None)
        if target and target in signalled_universe:
            plot_signals(
                signalled_universe[target],
                ticker=target,
                trades=trades,
                save_path=f"results/signals_{target}.png",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASX quant sim — single run")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--chart-ticker", default=None, help="Ticker to plot signal chart for")
    args = parser.parse_args()

    main(load_config(args.config), charts=not args.no_charts, chart_ticker=args.chart_ticker)
