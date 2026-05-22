"""
measure_behaviour.py — Score each stock on three behavioural dimensions:

  1. ranging_pct   — % of bars where ADX(14) < 20  (higher = better fit)
  2. atr_pct       — median ATR(14) / close × 100   (volatility level)
  3. vwap_dev_freq — % of bars where close < vwap × (1 - 0.003)
                     (how often price trades far enough below VWAP for a signal)

Usage:
    python measure_behaviour.py                    # active stocks only
    python measure_behaviour.py --deleted          # deleted stocks only
    python measure_behaviour.py --candidates       # 130 rotation candidates
    python measure_behaviour.py --all              # all three lists
    python measure_behaviour.py --tickers CBA MQG  # specific tickers
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Optional

sys.path.insert(0, "/Users/alex/quant-sim")
from src.data.loader import fetch_bars
from src.strategy.signals import compute_atr, compute_adx, compute_vwap

ADX_WINDOW      = 14
ADX_RANGING_THR = 20       # bars below this count as "ranging"
VWAP_DEV_THR    = 0.003    # 0.3% below VWAP counts as a qualifying deviation
DAYS            = 730


ACTIVE_STOCKS = {
    "CBA": "banking",    "NAB": "banking",    "MQG": "banking",  "CGF": "banking",
    "PPT": "banking",    "GQG": "banking",    "IFL": "banking",  "SDF": "banking",
    "EVN": "mining",     "IGO": "mining",
    "WDS": "energy",     "AGL": "energy",
    "SHL": "healthcare", "COH": "healthcare", "ANN": "healthcare",
    "FPH": "healthcare", "EBO": "healthcare",
    "LOV": "consumer",   "PMV": "consumer",   "SUL": "consumer",
    "ARB": "consumer",   "NCK": "consumer",
    "GMG": "realestate", "VCX": "realestate", "CHC": "realestate",
    "DXS": "realestate", "BWP": "realestate", "NSR": "realestate",
    "TLS": "technology", "NXT": "technology", "XRO": "technology",
    "WTC": "technology", "CPU": "technology", "ASX": "technology",
    "REA": "technology", "SEK": "technology", "CAR": "technology",
    "TNE": "technology", "HUB": "technology", "TPG": "technology",
    "NWL": "technology",
    "JHX": "industrial", "BXB": "industrial", "SVW": "industrial",
    "DOW": "industrial",
}

# 130 rotation candidates — not yet in active universe
CANDIDATES = {
    # ── Previously measured, good scores, deleted for non-behaviour reasons ──
    "WBC": "banking",    "ANZ": "banking",    "QBE": "banking",    "IAG": "banking",
    "SUN": "banking",    "MPL": "healthcare", "NHF": "healthcare", "RHC": "healthcare",
    "TCL": "industrial", "ALX": "industrial", "NWH": "industrial",
    "WPR": "realestate", "CLW": "realestate", "CIP": "realestate", "SCG": "realestate",
    "APA": "realestate",
    "MTS": "consumer",   "HVN": "consumer",
    "RIO": "mining",     "FMG": "mining",     "BHP": "mining",     "S32": "mining",
    "SFR": "mining",     "STO": "energy",     "ORG": "energy",
    # ── Banking / Insurance / Wealth ──
    "BEN": "banking",    "BOQ": "banking",    "AMP": "banking",    "AUB": "banking",
    "HLI": "banking",    "PNI": "banking",    "MFG": "banking",    "PTM": "banking",
    "EQT": "banking",    "PAC": "banking",    "JHG": "banking",
    # ── Infrastructure ──
    "AZJ": "industrial", "CWY": "industrial",
    # ── REITs (new) ──
    "GPT": "realestate", "SGP": "realestate", "LLC": "realestate", "ABP": "realestate",
    "CQR": "realestate", "HDN": "realestate", "ARF": "realestate", "CNI": "realestate",
    "COF": "realestate", "HPI": "realestate", "URW": "realestate", "HMC": "realestate",
    # ── Consumer Staples ──
    "TWE": "consumer",   "GNC": "consumer",   "BGA": "consumer",
    "ING": "consumer",   "ELD": "consumer",
    # ── Healthcare (defensive) ──
    "SIG": "healthcare", "HLS": "healthcare", "CAJ": "healthcare",
    "IDX": "healthcare", "MVF": "healthcare", "NAN": "healthcare",
    # ── Media / Telco ──
    "NEC": "technology", "HT1": "technology", "OML": "technology",
    "SWM": "technology", "SPK": "technology", "NWS": "technology",
    # ── Travel / Consumer Services ──
    "FLT": "consumer",   "WEB": "consumer",   "CTD": "consumer",   "QAN": "industrial",
    # ── Retail ──
    "AX1": "consumer",   "UNI": "consumer",   "KMD": "consumer",   "APE": "consumer",
    # ── Industrials (steady) ──
    "GWA": "industrial", "MND": "industrial", "IPH": "industrial",
    "WOR": "industrial", "DGL": "industrial",
    # ── Gaming / Leisure ──
    "ALL": "consumer",   "TAH": "consumer",
    # ── Healthcare (growth) ──
    "CSL": "healthcare", "PME": "healthcare", "PNV": "healthcare",
    "MSB": "healthcare", "IMM": "healthcare",
    # ── Technology / SaaS ──
    "APX": "technology", "MP1": "technology", "IDP": "technology", "JIN": "technology",
    "SQ2": "technology", "ZIP": "technology", "TYR": "technology", "SLC": "technology",
    "EML": "technology", "DTL": "technology", "RDY": "technology", "MAQ": "technology",
    "360": "technology", "SKO": "technology", "LNK": "technology", "PPS": "technology",
    "BTH": "technology", "GTK": "technology", "GDG": "banking",
    # ── Mining (new) ──
    "NST": "mining",     "WHC": "mining",     "NHC": "mining",     "YAL": "mining",
    "CRN": "mining",     "GOR": "mining",     "RRL": "mining",     "CMM": "mining",
    "OGC": "mining",     "NIC": "mining",     "MIN": "mining",     "PLS": "mining",
    "LTR": "mining",     "AKE": "mining",
    # ── Growth Consumer / Cyclical ──
    "CCX": "consumer",   "KGN": "technology", "THL": "consumer",
    "OCA": "healthcare", "AIZ": "industrial", "SXL": "technology",
    "GEM": "consumer",   "VGL": "technology",
    # ── Re-test flat stocks (expect to confirm bad) ──
    "WES": "consumer",   "COL": "consumer",   "WOW": "consumer",
}

# All deleted stocks (both rotation rounds)
DELETED_STOCKS = {
    # Initial cleanup
    "ANZ": "banking",   "WBC": "banking",   "SUN": "banking",   "QBE": "banking",
    "MPL": "healthcare","RHC": "healthcare","NHF": "healthcare",
    "WES": "consumer",  "JBH": "consumer",  "HVN": "consumer",  "MTS": "consumer",
    "TCL": "industrial","AMC": "industrial","ALD": "industrial", "SKI": "industrial",
    "ORG": "energy",    "STO": "energy",
    "BHP": "mining",    "RIO": "mining",    "FMG": "mining",     "S32": "mining",
    "NCM": "mining",    "OZL": "mining",
    "WPR": "realestate","CIP": "realestate","CLW": "realestate",
    "ALX": "industrial","IAG": "banking",
    # Rotation 1
    "SFR": "mining",    "RMD": "healthcare","COL": "consumer",   "WOW": "consumer",
    "BRG": "consumer",  "MGR": "realestate","SCG": "realestate","APA": "realestate",
    "ORI": "industrial","REH": "industrial","IEL": "industrial",
    # Rotation 2 (zero-trade)
    "SUL": "consumer",  "NCK": "consumer",
    "BWP": "realestate","NSR": "realestate",
    "TLS": "technology","CAR": "technology","NWL": "technology",
    "JHX": "industrial","SVW": "industrial",
}


def measure(ticker: str) -> Optional[dict]:
    try:
        df = fetch_bars(ticker, days=DAYS, interval="1h")
    except Exception as e:
        print(f"  {ticker}: fetch error — {e}")
        return None

    if df is None or len(df) < ADX_WINDOW * 3:
        print(f"  {ticker}: insufficient data ({len(df) if df is not None else 0} bars)")
        return None

    atr  = compute_atr(df, ADX_WINDOW)
    adx  = compute_adx(df, ADX_WINDOW, atr=atr)
    vwap = compute_vwap(df)

    valid = adx.notna() & (adx > 0)
    ranging_pct   = (adx[valid] < ADX_RANGING_THR).mean() * 100

    atr_pct_series = (atr / df["close"]).replace([np.inf, -np.inf], np.nan).dropna() * 100
    atr_pct        = atr_pct_series.median()

    below_vwap_thr = df["close"] < vwap * (1 - VWAP_DEV_THR)
    vwap_dev_freq  = below_vwap_thr.mean() * 100

    return {
        "ranging_pct":   round(ranging_pct,   1),
        "atr_pct":       round(atr_pct,        3),
        "vwap_dev_freq": round(vwap_dev_freq,  1),
        "bars":          int(valid.sum()),
    }


def score_tier(ranging_pct: float, atr_pct: float, vwap_dev_freq: float) -> str:
    """
    Classify into behavioural tier.
    Thresholds are intentionally loose — the scatter will show where to tighten them.
    """
    if ranging_pct < 40 and atr_pct > 1.0:
        return "EXCLUDE"
    if atr_pct < 0.55 and ranging_pct > 55:
        return "calm"
    if atr_pct > 0.55 or ranging_pct < 55:
        return "mid"
    return "calm"


def run(tickers: "dict[str, str]") -> None:
    results = []
    total = len(tickers)
    for i, (ticker, sector) in enumerate(tickers.items(), 1):
        print(f"  [{i:>2}/{total}] {ticker}...", end=" ", flush=True)
        m = measure(ticker)
        if m:
            tier = score_tier(m["ranging_pct"], m["atr_pct"], m["vwap_dev_freq"])
            results.append({
                "ticker":        ticker,
                "sector":        sector,
                "ranging_pct":   m["ranging_pct"],
                "atr_pct":       m["atr_pct"],
                "vwap_dev_freq": m["vwap_dev_freq"],
                "bars":          m["bars"],
                "tier":          tier,
            })
            print(f"ranging={m['ranging_pct']:5.1f}%  atr={m['atr_pct']:.3f}%  vwap_dev={m['vwap_dev_freq']:5.1f}%  → {tier}")
        else:
            print("SKIP")

    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results).sort_values("ranging_pct", ascending=False)

    print(f"\n{'─'*78}")
    print(f"{'ticker':<6}  {'sector':<12}  {'ranging%':>9}  {'atr%':>6}  {'vwap_dev%':>9}  {'bars':>5}  tier")
    print(f"{'─'*78}")
    for _, r in df.iterrows():
        print(f"{r['ticker']:<6}  {r['sector']:<12}  {r['ranging_pct']:>9.1f}  "
              f"{r['atr_pct']:>6.3f}  {r['vwap_dev_freq']:>9.1f}  {r['bars']:>5}  {r['tier']}")
    print(f"{'─'*78}")

    calm  = df[df["tier"] == "calm"]
    mid   = df[df["tier"] == "mid"]
    excl  = df[df["tier"] == "EXCLUDE"]
    print(f"\nSummary:  calm={len(calm)}  mid={len(mid)}  exclude={len(excl)}")
    if len(calm):
        print(f"  calm   avg ranging={calm['ranging_pct'].mean():.1f}%  avg atr={calm['atr_pct'].mean():.3f}%")
    if len(mid):
        print(f"  mid    avg ranging={mid['ranging_pct'].mean():.1f}%  avg atr={mid['atr_pct'].mean():.3f}%")
    if len(excl):
        print(f"  excl   tickers: {', '.join(excl['ticker'].tolist())}")

    out_path = "/Users/alex/quant-sim/results/behaviour_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deleted",    action="store_true", help="Measure deleted stocks only")
    parser.add_argument("--candidates", action="store_true", help="Measure 130 rotation candidates")
    parser.add_argument("--all",        action="store_true", help="Measure all stocks")
    parser.add_argument("--tickers",    nargs="+",           help="Specific tickers to measure")
    args = parser.parse_args()

    if args.tickers:
        all_known = {**ACTIVE_STOCKS, **DELETED_STOCKS, **CANDIDATES}
        targets = {t: all_known.get(t, "unknown") for t in args.tickers}
    elif args.deleted:
        targets = DELETED_STOCKS
    elif args.candidates:
        targets = CANDIDATES
    elif args.all:
        targets = {**ACTIVE_STOCKS, **DELETED_STOCKS, **CANDIDATES}
    else:
        targets = ACTIVE_STOCKS

    print(f"Measuring {len(targets)} stocks over {DAYS} days of 1h bars...\n")
    run(targets)
