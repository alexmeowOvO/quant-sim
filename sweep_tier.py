"""Per-tier parameter sweep — optimises one tier in isolation, then reports combined results."""

import argparse
import itertools
import os
import warnings
import yaml
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from src.data.loader import fetch_universe, fetch_bars
from src.strategy.signals import compute_signals
from src.engine.event_loop import run_simulation, SimConfig, TierParams
from src.reporting.metrics import compute_metrics
from main import build_tier_maps, tier_signal_params


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_tier_sweep(cfg: dict, target_tier: str) -> pd.DataFrame:
    tiers_cfg  = cfg["universe"]["tiers"]
    data_cfg   = cfg["data"]
    exec_cfg   = cfg["execution"]
    port_cfg   = cfg["portfolio"]
    risk_cfg   = cfg["risk"]
    strat      = cfg["strategy"]

    if target_tier not in tiers_cfg:
        raise ValueError(f"Tier '{target_tier}' not found. Available: {list(tiers_cfg)}")

    tier_sweep_cfg = cfg.get("tier_sweep", {}).get(target_tier, {})
    if not tier_sweep_cfg:
        raise ValueError(f"No tier_sweep config found for tier '{target_tier}'")

    ticker_list, ticker_tier_map, tier_params_map = build_tier_maps(cfg)
    target_tickers = tiers_cfg[target_tier]["tickers"]

    # Load data for target tier only
    print(f"Loading data for {len(target_tickers)} {target_tier}-tier symbols...")
    raw_universe = fetch_universe(target_tickers, days=data_cfg["days"], interval=data_cfg["interval"])
    if not raw_universe:
        raise RuntimeError("No data loaded.")

    # Pre-fetch market filter
    mf_cfg = cfg.get("market_filter", {})
    mf_ticker = mf_cfg.get("ticker", "^AXJO")
    mf_window = mf_cfg.get("sma_window", 0)
    mf_df = None
    if mf_window > 0:
        print(f"Fetching market filter ({mf_ticker}, SMA={mf_window})...")
        try:
            mf_df = fetch_bars(mf_ticker, days=data_cfg["days"], interval=data_cfg["interval"])
            print(f"  {mf_ticker}: {len(mf_df)} bars loaded\n")
        except Exception as e:
            print(f"  WARNING: market filter fetch failed ({e}), filter disabled\n")

    # Build sweep grid
    bb_windows          = tier_sweep_cfg["bb_window"]
    bb_stds             = tier_sweep_cfg["bb_std"]
    vwap_thresholds     = tier_sweep_cfg["vwap_threshold"]
    adx_thresholds      = tier_sweep_cfg["adx_threshold"]
    trend_filter_windows = tier_sweep_cfg["trend_filter_window"]
    atr_multipliers     = tier_sweep_cfg["atr_multiplier"]
    max_hold_bars_list  = tier_sweep_cfg["max_hold_bars"]

    combos = list(itertools.product(
        bb_windows, bb_stds, vwap_thresholds,
        adx_thresholds, trend_filter_windows,
        atr_multipliers, max_hold_bars_list,
    ))
    print(f"Running {len(combos)} combinations for '{target_tier}' tier...\n")

    # Fixed params for the target tier (adx_window, EMA/RSI stay at config defaults)
    base_tier_cfg = tiers_cfg[target_tier]
    adx_window = base_tier_cfg.get("adx_window", 14)
    min_trending_adx = base_tier_cfg.get("min_trending_adx", 40)
    trending_enabled = strat.get("trending_enabled", True)

    # Tier params for simulation (only this tier is being swept)
    fixed_tier_params = {
        name: TierParams(
            atr_multiplier=tiers_cfg[name]["atr_multiplier"],
            max_hold_bars=tiers_cfg[name]["max_hold_bars"],
        )
        for name in tiers_cfg
        if name != target_tier
    }

    sim_cfg_base = dict(
        initial_capital=port_cfg["initial_capital"],
        position_size_pct=port_cfg["position_size_pct"],
        max_concurrent_positions=strat.get("max_concurrent_positions", 5),
        slippage_pct=exec_cfg["slippage_pct"],
        commission=exec_cfg["commission_per_trade"],
        stop_loss_pct=risk_cfg["stop_loss_pct"],
        ticker_tiers={t: target_tier for t in raw_universe},
    )

    results = []
    for i, (bb_w, bb_s, vwap_t, adx_t, tf_win, atr_mult, hold_bars) in enumerate(combos, 1):
        # Build market filter for this simulation
        market_ok = None
        if mf_window > 0 and mf_df is not None:
            all_timestamps = sorted(set(ts for df in raw_universe.values() for ts in df.index))
            mf_sma = mf_df["close"].rolling(mf_window, min_periods=mf_window).mean()
            market_ok = (mf_df["close"] > mf_sma)

        # Compute signals with swept params
        signalled = {
            t: compute_signals(
                df,
                bb_window=bb_w,
                bb_std=bb_s,
                vwap_threshold=vwap_t,
                adx_window=adx_window,
                adx_threshold=adx_t,
                min_trending_adx=min_trending_adx,
                fast_ema=strat["ema"]["fast"],
                slow_ema=strat["ema"]["slow"],
                rsi_window=strat["rsi"]["window"],
                rsi_overbought=strat["rsi"]["overbought"],
                rsi_oversold=strat["rsi"]["oversold"],
                trend_filter_window=tf_win,
                trending_enabled=trending_enabled,
            )
            for t, df in raw_universe.items()
        }

        swept_tp = TierParams(atr_multiplier=atr_mult, max_hold_bars=hold_bars)
        tier_params = {**fixed_tier_params, target_tier: swept_tp}

        sim_cfg = SimConfig(**sim_cfg_base, tier_params=tier_params)
        trades, equity = run_simulation(signalled, sim_cfg, market_filter=market_ok)
        m = compute_metrics(trades, equity, sim_cfg.initial_capital)

        if "error" not in m:
            results.append({
                "bb_window":           bb_w,
                "bb_std":              bb_s,
                "vwap_threshold":      vwap_t,
                "adx_threshold":       adx_t,
                "trend_filter_window": tf_win,
                "atr_multiplier":      atr_mult,
                "max_hold_bars":       hold_bars,
                "sharpe":              m["sharpe_ratio"],
                "sortino":             m["sortino_ratio"],
                "total_return_%":      m["total_return_pct"],
                "max_drawdown_%":      m["max_drawdown_pct"],
                "win_rate_%":          m["win_rate_pct"],
                "trades":              m["total_trades"],
                "profit_factor":       m["profit_factor"],
            })
            print(f"  [{i:4d}/{len(combos)}] bb={bb_w}/std={bb_s} vwap={vwap_t} "
                  f"adx_t={adx_t} tf={tf_win} atr={atr_mult} hold={hold_bars} → "
                  f"sharpe={m['sharpe_ratio']:+.3f}  ret={m['total_return_pct']:+.1f}%  "
                  f"trades={m['total_trades']}")
        else:
            print(f"  [{i:4d}/{len(combos)}] → no trades")

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASX quant sim — per-tier parameter sweep")
    parser.add_argument("--tier", required=True, help="Tier to sweep (e.g. 'medium' or 'stable')")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--top", type=int, default=10, help="Number of top results to display")
    parser.add_argument("--out", default=None, help="CSV output path (default: results/sweep_<tier>.csv)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = run_tier_sweep(cfg, args.tier)

    out_path = args.out or f"results/sweep_{args.tier}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nFull results saved → {out_path}")

    print("\n" + "=" * 80)
    print(f"TOP {args.top} CONFIGURATIONS FOR '{args.tier}' TIER (ranked by Sharpe)")
    print("=" * 80)
    print(results.head(args.top).to_string(index=False))

    best = results.iloc[0]
    print(f"\nBEST → Sharpe {best.sharpe:+.3f}, return {best['total_return_%']:+.1f}%, "
          f"win_rate {best['win_rate_%']:.1f}%, trades {int(best.trades)}")
    print(f"\nUpdate config/default.yaml universe.tiers.{args.tier} with:")
    print(f"  bb_window:           {int(best.bb_window)}")
    print(f"  bb_std:              {best.bb_std}")
    print(f"  vwap_threshold:      {best.vwap_threshold}")
    print(f"  adx_threshold:       {best.adx_threshold}")
    print(f"  trend_filter_window: {int(best.trend_filter_window)}")
    print(f"  atr_multiplier:      {best.atr_multiplier}")
    print(f"  max_hold_bars:       {int(best.max_hold_bars)}")
