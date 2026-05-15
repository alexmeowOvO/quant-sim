"""Equity curve and trade visualisation."""

from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_equity_curve(
    equity_curve: pd.Series,
    initial_capital: float,
    title: str = "Equity Curve",
    save_path: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Top: equity curve vs flat capital line
    ax1 = axes[0]
    ax1.plot(equity_curve.index, equity_curve.values, color="#2196F3", linewidth=1.2, label="Portfolio")
    ax1.axhline(initial_capital, color="#9E9E9E", linewidth=0.8, linestyle="--", label="Initial capital")
    ax1.set_ylabel("Portfolio Value (AUD)")
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax1.grid(axis="y", alpha=0.3)

    # Bottom: drawdown
    ax2 = axes[1]
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.5)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_signals(
    df: pd.DataFrame,
    ticker: str,
    trades: List,
    save_path: Optional[str] = None,
) -> None:
    """Plot price with BB bands, VWAP, and trade entry/exit markers for one symbol."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{ticker} — Price, Bollinger Bands & VWAP", fontsize=13, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(df.index, df["close"], color="#212121", linewidth=0.8, label="Close")
    ax1.plot(df.index, df["bb_mid"], color="#1565C0", linewidth=0.7, linestyle="--", label="BB mid")
    ax1.fill_between(df.index, df["bb_lower"], df["bb_upper"], alpha=0.08, color="#1565C0", label="BB band")
    ax1.plot(df.index, df["vwap"], color="#F57C00", linewidth=0.7, linestyle=":", label="VWAP")

    # Mark entry/exit points from trades for this ticker
    ticker_trades = [t for t in trades if t.ticker == ticker]
    if ticker_trades and not df.empty:
        timestamps = df.index
        for t in ticker_trades:
            if t.entry_bar < len(timestamps):
                entry_ts = timestamps[t.entry_bar]
                exit_ts = timestamps[min(t.exit_bar, len(timestamps) - 1)]
                color = "#4CAF50" if t.pnl > 0 else "#F44336"
                ax1.axvline(entry_ts, color=color, linewidth=0.5, alpha=0.4)
                ax1.axvline(exit_ts, color="#9E9E9E", linewidth=0.5, alpha=0.3)

    ax1.set_ylabel("Price (AUD)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax1.grid(alpha=0.2)

    # Bottom: volume
    ax2 = axes[1]
    ax2.bar(df.index, df["volume"], color="#90A4AE", width=0.0005)
    ax2.set_ylabel("Volume")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.grid(axis="y", alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved → {save_path}")
    else:
        plt.show()
    plt.close()
