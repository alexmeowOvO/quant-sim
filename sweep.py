"""Parameter sweep — finds best config by Sharpe ratio."""

import argparse
import itertools
import math
import os
import yaml
import pandas as pd
from src.data.loader import fetch_universe, fetch_bars
from src.strategy.signals import compute_signals
from src.engine.event_loop import run_simulation, SimConfig, TierParams
from src.reporting.metrics import compute_metrics
from main import build_tier_maps


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_sweep(cfg: dict) -> pd.DataFrame:
    tiers_cfg   = cfg["universe"]["tiers"]
    data_cfg    = cfg["data"]
    exec_cfg    = cfg["execution"]
    port_cfg    = cfg["portfolio"]
    risk_cfg    = cfg["risk"]
    strat       = cfg["strategy"]
    sweep_cfg   = cfg["sweep"]

    ticker_list, ticker_tier_map, tier_params_base = build_tier_maps(cfg)

    # Load data once — reused across all sweep runs
    print(f"Loading data for {len(ticker_list)} symbols...\n")
    raw_universe = fetch_universe(ticker_list, days=data_cfg["days"], interval=data_cfg["interval"])
    if not raw_universe:
        raise RuntimeError("No data loaded.")

    # Pre-fetch market index for filter (reused across combos)
    mf_cfg = cfg.get("market_filter", {})
    mf_ticker = mf_cfg.get("ticker", "^AXJO")
    print(f"Fetching market filter data ({mf_ticker})...")
    try:
        mf_df = fetch_bars(mf_ticker, days=data_cfg["days"], interval=data_cfg["interval"])
        print(f"  {mf_ticker}: {len(mf_df)} bars loaded\n")
    except Exception as e:
        print(f"  WARNING: market filter fetch failed ({e}), filter will be disabled\n")
        mf_df = None

    # Build parameter grid
    bb_windows          = sweep_cfg["bollinger_window"]
    atr_mult_scales     = sweep_cfg["atr_multiplier_scale"]
    hold_scales         = sweep_cfg["max_hold_bars_scale"]
    adx_windows         = sweep_cfg["adx_window"]
    adx_thresholds      = sweep_cfg["adx_threshold"]
    min_trending_adxs   = sweep_cfg["min_trending_adx"]
    fast_emas           = sweep_cfg["fast_ema"]
    slow_emas           = sweep_cfg["slow_ema"]
    rsi_windows         = sweep_cfg["rsi_window"]
    rsi_ob_os            = sweep_cfg["rsi_ob_os"]
    trend_filter_windows = sweep_cfg["trend_filter_window"]
    market_sma_windows   = sweep_cfg.get("market_sma_window", [0])
    # vwap_threshold: use the first tier's value as the shared global default
    first_tier = next(iter(tiers_cfg.values()))
    vwap_threshold = first_tier.get("vwap_threshold", 0.005)

    combos = list(itertools.product(
        bb_windows, atr_mult_scales, hold_scales,
        adx_windows, adx_thresholds, min_trending_adxs,
        fast_emas, slow_emas,
        rsi_windows, rsi_ob_os,
        trend_filter_windows,
        market_sma_windows,
    ))
    print(f"Running {len(combos)} parameter combinations...\n")

    sim_cfg_base = dict(
        initial_capital=port_cfg["initial_capital"],
        position_size_pct=port_cfg["position_size_pct"],
        max_concurrent_positions=strat.get("max_concurrent_positions", 5),
        slippage_pct=exec_cfg["slippage_pct"],
        commission=exec_cfg["commission_per_trade"],
        stop_loss_pct=risk_cfg["stop_loss_pct"],
        ticker_tiers=ticker_tier_map,
    )

    results = []
    for i, (bb_w, atr_scale, hold_scale, adx_w, adx_t, min_adx,
            f_ema, s_ema, rsi_w, ob_os, tf_win, mkt_win) in enumerate(combos, 1):
        rsi_ob, rsi_os = ob_os

        # Build market filter for this combo's SMA window
        market_ok = None
        if mkt_win > 0 and mf_df is not None:
            mf_sma = mf_df["close"].rolling(mkt_win, min_periods=mkt_win).mean()
            market_ok = (mf_df["close"] > mf_sma)

        # Scale each tier's ATR multiplier and max_hold_bars by the sweep factors
        scaled_tier_params = {
            name: TierParams(
                atr_multiplier=round(tp.atr_multiplier * atr_scale, 4),
                max_hold_bars=max(1, math.ceil(tp.max_hold_bars * hold_scale)),
            )
            for name, tp in tier_params_base.items()
        }

        # Compute signals per ticker using tier-specific bb_std
        signalled = {
            t: compute_signals(
                df,
                bb_window=bb_w,
                bb_std=tiers_cfg[ticker_tier_map[t]]["bb_std"],
                vwap_threshold=vwap_threshold,
                adx_window=adx_w,
                adx_threshold=adx_t,
                min_trending_adx=min_adx,
                fast_ema=f_ema,
                slow_ema=s_ema,
                rsi_window=rsi_w,
                rsi_overbought=rsi_ob,
                rsi_oversold=rsi_os,
                trend_filter_window=tf_win,
                trending_enabled=strat.get("trending_enabled", True),
            )
            for t, df in raw_universe.items()
        }

        sim_cfg = SimConfig(**sim_cfg_base, tier_params=scaled_tier_params)
        trades, equity = run_simulation(signalled, sim_cfg, market_filter=market_ok)
        m = compute_metrics(trades, equity, sim_cfg.initial_capital)

        if "error" not in m:
            results.append({
                "bb_window":           bb_w,
                "atr_multiplier_scale": atr_scale,
                "max_hold_bars_scale":  hold_scale,
                "adx_window":          adx_w,
                "adx_threshold":       adx_t,
                "min_trending_adx":    min_adx,
                "fast_ema":            f_ema,
                "slow_ema":            s_ema,
                "rsi_window":          rsi_w,
                "rsi_ob":              rsi_ob,
                "trend_filter_window": tf_win,
                "market_sma_window":   mkt_win,
                "sharpe":              m["sharpe_ratio"],
                "sortino":             m["sortino_ratio"],
                "total_return_%":      m["total_return_pct"],
                "max_drawdown_%":      m["max_drawdown_pct"],
                "win_rate_%":          m["win_rate_pct"],
                "trades":              m["total_trades"],
                "profit_factor":       m["profit_factor"],
            })
            tier_stop_str = "  ".join(
                f"{name}={round(tp.atr_multiplier * atr_scale, 2)}"
                for name, tp in tier_params_base.items()
            )
            print(f"  [{i:3d}/{len(combos)}] BB({bb_w}) atr×{atr_scale} [{tier_stop_str}] "
                  f"hold×{hold_scale} tf={tf_win} mkt={mkt_win} → "
                  f"sharpe={m['sharpe_ratio']:+.3f}  ret={m['total_return_pct']:+.1f}%  "
                  f"trades={m['total_trades']}")
        else:
            print(f"  [{i:3d}/{len(combos)}] → no trades")

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASX quant sim — parameter sweep")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--top", type=int, default=10, help="Number of top results to display")
    parser.add_argument("--out", default="results/sweep_results.csv", help="CSV output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = run_sweep(cfg)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results.to_csv(args.out, index=False)
    print(f"\nFull results saved → {args.out}")

    print("\n" + "=" * 80)
    print(f"TOP {args.top} CONFIGURATIONS (ranked by Sharpe ratio)")
    print("=" * 80)
    print(results.head(args.top).to_string(index=False))

    best = results.iloc[0]
    tiers_cfg = cfg["universe"]["tiers"]
    print(f"\nBEST → Sharpe {best.sharpe:+.3f}, return {best['total_return_%']:+.1f}%")
    print("Update config/default.yaml with:")
    print(f"  strategy.bollinger.window:   {int(best.bb_window)}")
    print(f"  strategy.adx.window:         {int(best.adx_window)}")
    print(f"  strategy.adx.threshold:      {best.adx_threshold}")
    print(f"  strategy.adx.min_trending:   {best.min_trending_adx}")
    print(f"  strategy.trend_filter.window: {int(best.trend_filter_window)}")
    print(f"  strategy.ema.fast:           {int(best.fast_ema)}")
    print(f"  strategy.ema.slow:           {int(best.slow_ema)}")
    print(f"  strategy.rsi.overbought:     {int(best.rsi_ob)}")
    print(f"  Per-tier ATR multipliers (scale={best.atr_multiplier_scale}):")
    for name, tier in tiers_cfg.items():
        print(f"    {name}: {tier['atr_multiplier']} × {best.atr_multiplier_scale}"
              f" = {tier['atr_multiplier'] * best.atr_multiplier_scale:.3f}")
    print(f"  market_filter.sma_window:    {int(best.market_sma_window)}")
    print(f"  Per-tier max_hold_bars (scale={best.max_hold_bars_scale}):")
    for name, tier in tiers_cfg.items():
        print(f"    {name}: {tier['max_hold_bars']} × {best.max_hold_bars_scale}"
              f" = {max(1, math.ceil(tier['max_hold_bars'] * best.max_hold_bars_scale))}")
