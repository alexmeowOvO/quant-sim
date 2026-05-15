# quant-sim

An event-driven backtester for ASX stocks on 1-hour bars.

## Strategy

Mean reversion for ranging stocks, with two independent entry signals:

- **Bollinger Band + VWAP** — enter when price touches the lower BB (1.2σ) while trading at a VWAP discount
- **Volume climax** — enter on selling exhaustion: high-volume red candle with a small body below VWAP

Both entries require:
- ADX < 20 (ranging regime — trending stocks are skipped)
- Price above the 50-bar SMA (no catching falling knives)
- ASX 200 (`^AXJO`) above its 50-bar SMA — no entries during market downtrends

Exit on signal reversal (price returns to BB midline or VWAP), ATR-based stop, or hold timeout.

## Volatility tiers

Stocks are split into two tiers with independent signal and risk parameters:

| Tier | Stocks | ATR stop mult | Max hold |
|------|--------|---------------|----------|
| Stable | CBA, NAB, WBC, ANZ, WES, WOW, COL | 2.0× | 12 bars |
| Medium | MQG, BHP, RIO, WDS, STO, CSL, SHL, GMG, EVN | 5.0× | 18 bars |

## Usage

```bash
pip install -r requirements.txt

# Single run
python main.py

# Per-tier parameter sweep
python sweep_tier.py --tier medium
python sweep_tier.py --tier stable

# Global sweep (scales ATR and hold params across both tiers uniformly)
python sweep.py
```

Results and charts are saved to `results/`.

## Files

```
config/default.yaml     — per-tier signal params, market filter, sweep ranges
main.py                 — entry point
sweep_tier.py           — per-tier parameter sweep (sweeps each tier in isolation)
sweep.py                — global parameter sweep (uniform scaling across tiers)
src/
  data/loader.py        — yfinance 1h bar fetcher, session-hour filtering
  strategy/signals.py   — BB, VWAP, ADX, ATR, RSI, EMA, volume climax
  engine/event_loop.py  — simulation engine: ATR stops, entry ranking, market filter
  reporting/
    metrics.py          — Sharpe, Sortino, Calmar, profit factor, breakdowns
    charts.py           — equity curve + per-symbol signal chart
```
