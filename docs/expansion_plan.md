# Universe Expansion Plan — 68 → 100+ Stocks

**Baseline frozen:** `config/baseline_69stocks_sharpe1928.yaml` (68 stocks after IFL delisted)
**Expansion branch:** `config/experiment_100stocks.yaml`
**Signal logger:** runs Mon-Fri at 5pm Sydney time (cron installed 2026-05-22)

---

## Principles

1. **Never touch the baseline.** All experiments run against `experiment_100stocks.yaml`.
   Merge back only when a new config beats Sharpe ≥ 1.5 with max_dd < 12% and stops < 5%.
2. **Behaviour-fit over sector diversity.** A stock earns a slot by ranging cleanly, not
   by filling a sector gap.
3. **Batches of 15-20.** Add, sweep, backtest, decide. Never mass-add untested stocks.
4. **Screen first, backtest second.** Only stocks that pass `auto_rotate.py`'s screen
   enter the backtest. Don't waste sweep compute on REJECT candidates.
5. **bb_std stays at 1.0 (tier_a) / 1.5 (tier_b)** until a future robustness analysis
   with ≥ 6 months of live signal data justifies a change.

---

## Screen thresholds (from `auto_rotate.py`)

| Check | tier_a | tier_b |
|---|---|---|
| min bars | ≥ 3000 | ≥ 3000 |
| ranging_pct | — | ≥ 15% |
| vwap_dev_freq | ≥ 15% | ≥ 15% |
| ATR floor | ≥ 0.45% | ≥ 0.45% |
| ATR ceiling | ≤ 0.95% | ≤ 1.00% |
| PASS_WITH_WARNING | vwap_dev/ranging < 0.80 | vwap_dev/ranging < 0.80 |

Stocks in `MANUAL_EXCLUDE`: FMG, PMV, PPT, SIG, MND, FLT, IFL, CTD — blocked permanently
(or until quarantine lifted).

---

## Batch 1 — best candidates from current behaviour_scores.csv

Run `auto_rotate.py` to screen these. Stocks that get PASS or PASS_WITH_WARNING go into
`experiment_100stocks.yaml` for testing.

Top screened candidates (as of 2026-05-22):

| Ticker | Ranging% | VwapDev% | ATR% | Bars | Notes |
|--------|----------|----------|------|------|-------|
| NEC | 30.8 | 23.2 | 0.952 | 3434 | tier_a candidate |
| GEM | 29.7 | 27.8 | 0.976 | 3448 | tier_a candidate |
| CNI | 27.1 | 21.3 | 0.997 | 3436 | tier_a/b borderline |
| EQT | 41.6 | 23.7 | 1.029 | 3315 | REJECT_HIGH_ATR (atr > 1.0) |
| OML | 24.6 | 23.7 | 1.034 | 3448 | REJECT_HIGH_ATR |
| KGN | 24.2 | 27.1 | 1.075 | 3440 | REJECT_HIGH_ATR |

**Action:** Run `python3 auto_rotate.py` with a broader candidate list to find more passers.

---

## Finding new candidates

The current `behaviour_scores.csv` covers a limited universe. To find more:

### Option A — Widen the score run
Edit `auto_rotate.py`'s candidate fetch to include more ASX200/300 tickers:
```python
CANDIDATE_TICKERS = [
    # Add from ASX200 names not yet scored:
    "ALQ","ASK","BAP","BEN","CAR","CCX","CIA","CLW","CMW","CNU",
    "EVN","GDF","GOR","HMC","HVN","IGO","ILU","KLS","LNK","LYC",
    "MIN","NCM","NIC","NUF","OFX","OZL","PBH","PLT","RFF","RRL",
    "SBM","SDR","SKI","SPT","SQ2","STO","SUL","SWM","TAH","TPW",
    "VGL","WGX","ZIP",
]
```
Then run: `python3 auto_rotate.py` (dry-run) to score and screen them all.

### Option B — Pull from ASX indices
Use yfinance to fetch all ASX200 constituents and score them in a batch job.

### Recommended next step
Run auto_rotate.py with ~50 new candidates from the ASX200/300 not yet in the universe.
Expect ~15-25% pass rate → 8-12 new passers per batch.

---

## Expansion workflow (each batch)

```
1. Identify 20-30 new candidates (not already in universe, not in MANUAL_EXCLUDE)
2. Run behaviour scoring on them:
      python3 auto_rotate.py   (scores + screens all candidates)
3. Take the PASS/PASS_WITH_WARNING stocks → add to experiment_100stocks.yaml
4. Run tier sweep on the modified tiers:
      python3 sweep_tier.py --tier tier_a --config config/experiment_100stocks.yaml
      python3 sweep_tier.py --tier tier_b --config config/experiment_100stocks.yaml
5. Update experiment_100stocks.yaml with best sweep params
6. Full backtest:
      python3 main.py --config config/experiment_100stocks.yaml --no-charts
7. Compare against baseline (Sharpe 1.928):
      - Sharpe ≥ 1.5?  →  acceptable
      - max_dd < 12%?  →  acceptable
      - stops < 5% of trades?  →  acceptable
      - If all pass: keep the batch
      - If any fail: remove worst offenders, re-run from step 4
8. After 3-4 successful batches reaching ~100 stocks:
      cp config/experiment_100stocks.yaml config/baseline_100stocks_<sharpe>.yaml
      Update signal_logger.py default config if desired
```

---

## Gate criteria for merging experiment → new baseline

| Metric | Min threshold | Notes |
|--------|---------------|-------|
| Sharpe ratio | ≥ 1.50 | Lower OK given larger universe (more diversification) |
| Max drawdown | < 12% | Tighter than current 10.5% budget |
| Stop loss trades | < 5% of total | Currently 0/115 = 0% |
| Win rate | > 55% | Currently 58.3% |
| Profit factor | > 1.05 | Currently implied ~1.1 |
| Zero-trade stocks | 0 | All new stocks must fire at least once |

---

## Current universe size

| Config | Stocks | Sharpe | Status |
|--------|--------|--------|--------|
| `baseline_69stocks_sharpe1928.yaml` | 68 (IFL delisted) | 1.928 | ✅ frozen baseline |
| `experiment_100stocks.yaml` | 68 | 1.928 | working branch |

---

## Paper trading milestone

After 4 weeks of clean daily signal logs (`results/daily_signals.csv`):
- Assess signal rate: are we getting 2-5 entries/week?
- Check near-miss accuracy: do near misses convert within 1-2 days?
- If signals look sane: build `paper_trader.py` on top of signal_logger.py

Paper trader will track: virtual positions, entry/exit prices, hold duration, virtual PnL.
It will NOT execute real trades.

---

## Files

| File | Purpose |
|------|---------|
| `config/baseline_69stocks_sharpe1928.yaml` | Frozen. Do not edit. |
| `config/experiment_100stocks.yaml` | Expansion branch — edit freely |
| `config/default.yaml` | Mirrors baseline; used by main.py default |
| `results/daily_signals.csv` | Live signal log (appended daily at 5pm) |
| `results/signal_logger_cron.log` | Cron stdout/stderr log |
| `results/behaviour_scores.csv` | Candidate behaviour scores |
| `results/deleted_stocks.csv` | Quarantine/delisted log |
| `results/trades.csv` | Last backtest trades |
| `results/stock_breakdown.csv` | Per-stock summary |
| `results/summary_metrics.json` | Baseline metrics snapshot |
