"""Performance metrics and equity curve reporting."""

from typing import List
import pandas as pd
import numpy as np
from src.engine.event_loop import Trade


def compute_metrics(trades: List[Trade], equity_curve: pd.Series, initial_capital: float) -> dict:
    if not trades:
        return {"error": "no trades executed"}

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    returns = equity_curve.pct_change().dropna()

    # Infer bar duration from median inter-bar gap, then annualise.
    # Uses median (not mean) to ignore weekend/holiday gaps.
    # ASX trades ~5.75 h/day (10:00–15:45 effective); use 6 h as a round figure.
    if len(equity_curve.index) > 1 and hasattr(equity_curve.index, "to_series"):
        median_bar_secs = equity_curve.index.to_series().diff().dropna().median().total_seconds()
        bars_per_year = int(252 * 6 * 3600 / median_bar_secs) if median_bar_secs > 0 else 252 * 66
    else:
        bars_per_year = 252 * 66  # fallback

    sharpe = (
        (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
        if returns.std() > 0 else 0.0
    )
    downside = returns[returns < 0].std()
    sortino = (
        (returns.mean() / downside) * np.sqrt(bars_per_year)
        if downside > 0 else 0.0
    )

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    calmar = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_commission = len(trades) * 20.0    # $10 each side
    avg_pnl_net = np.mean(pnls)
    avg_pnl_gross = np.mean([t.pnl_gross for t in trades])

    return {
        "total_return_pct":    round(total_return * 100, 2),
        "initial_capital":     round(initial_capital, 2),
        "final_equity":        round(equity_curve.iloc[-1], 2),
        "sharpe_ratio":        round(sharpe, 3),
        "sortino_ratio":       round(sortino, 3),
        "calmar_ratio":        round(calmar, 3),
        "max_drawdown_pct":    round(max_drawdown * 100, 2),
        "profit_factor":       round(profit_factor, 3),
        "total_trades":        len(trades),
        "win_rate_pct":        round(len(wins) / len(trades) * 100, 1),
        "avg_win":             round(np.mean(wins), 2) if wins else 0,
        "avg_loss":            round(np.mean(losses), 2) if losses else 0,
        "avg_pnl_net":         round(avg_pnl_net, 2),
        "avg_pnl_gross":       round(avg_pnl_gross, 2),
        "total_commission":    round(total_commission, 2),
        "exit_reasons":        pd.Series([t.exit_reason for t in trades]).value_counts().to_dict(),
    }


def regime_breakdown(trades: List[Trade]) -> pd.DataFrame:
    """Trades, win rate, gross/net PnL, and commission drag split by entry regime."""
    rows = []
    for regime in ["ranging", "trending"]:
        ts = [t for t in trades if t.entry_regime == regime]
        if not ts:
            rows.append({"regime": regime, "trades": 0, "pct_%": 0.0,
                         "win_rate_%": 0.0, "avg_pnl_gross": 0.0,
                         "avg_pnl_net": 0.0, "commission_drag": 0.0})
            continue
        pnls = [t.pnl for t in ts]
        gross = [t.pnl_gross for t in ts]
        wins = [p for p in pnls if p > 0]
        rows.append({
            "regime":           regime,
            "trades":           len(ts),
            "pct_%":            round(len(ts) / len(trades) * 100, 1),
            "win_rate_%":       round(len(wins) / len(ts) * 100, 1),
            "avg_pnl_gross":    round(np.mean(gross), 2),
            "avg_pnl_net":      round(np.mean(pnls), 2),
            "commission_drag":  round(np.mean(pnls) - np.mean(gross), 2),
        })
    return pd.DataFrame(rows)


def exit_breakdown(trades: List[Trade]) -> pd.DataFrame:
    """Trades, win rate, and avg gross/net PnL split by exit reason."""
    rows = []
    for reason in ["signal", "stop", "timeout"]:
        ts = [t for t in trades if t.exit_reason == reason]
        if not ts:
            rows.append({"exit_reason": reason, "trades": 0, "win_rate_%": 0.0,
                         "avg_pnl_gross": 0.0, "avg_pnl_net": 0.0, "total_pnl": 0.0})
            continue
        pnls = [t.pnl for t in ts]
        gross = [t.pnl_gross for t in ts]
        wins = [p for p in pnls if p > 0]
        rows.append({
            "exit_reason":   reason,
            "trades":        len(ts),
            "win_rate_%":    round(len(wins) / len(ts) * 100, 1),
            "avg_pnl_gross": round(np.mean(gross), 2),
            "avg_pnl_net":   round(np.mean(pnls), 2),
            "total_pnl":     round(sum(pnls), 2),
        })
    return pd.DataFrame(rows)


def per_symbol_breakdown(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = []
    by_ticker: dict = {}
    for t in trades:
        by_ticker.setdefault(t.ticker, []).append(t)
    for ticker, ts in by_ticker.items():
        pnls = [t.pnl for t in ts]
        wins = [p for p in pnls if p > 0]
        rows.append({
            "ticker":     ticker,
            "trades":     len(ts),
            "win_rate_%": round(len(wins) / len(ts) * 100, 1),
            "total_pnl":  round(sum(pnls), 2),
            "avg_pnl":    round(np.mean(pnls), 2),
        })
    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)


def print_report(trades: List[Trade], metrics: dict, symbol_breakdown: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("SIMULATION RESULTS")
    print("=" * 55)
    skip = {"exit_reasons", "avg_pnl_net", "avg_pnl_gross", "total_commission"}
    for k, v in metrics.items():
        if k not in skip:
            print(f"  {k:<26} {v}")
    print(f"  {'exit_reasons':<26} {metrics.get('exit_reasons', {})}")

    print("\nCOMMISSION DRAG")
    print(f"  avg_pnl_gross (before)  {metrics['avg_pnl_gross']:>8.2f}")
    print(f"  avg_pnl_net   (after)   {metrics['avg_pnl_net']:>8.2f}")
    print(f"  total_commission        {metrics['total_commission']:>8.2f}")
    loss = abs(metrics["final_equity"] - metrics["initial_capital"])
    drag_pct = metrics["total_commission"] / loss * 100 if loss > 0 else 0
    print(f"  commission as % of loss {drag_pct:>7.1f}%")

    print("\nEXIT BREAKDOWN")
    print(exit_breakdown(trades).to_string(index=False))

    print("\nREGIME BREAKDOWN")
    print(regime_breakdown(trades).to_string(index=False))

    print("\nPER-SYMBOL BREAKDOWN")
    print(symbol_breakdown.to_string(index=False))
    print("=" * 55)
