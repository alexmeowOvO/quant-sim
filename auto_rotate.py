"""
auto_rotate.py — Post-measurement rotation automation.

Reads results/candidates_measurement.log, screens each stock with a flag system,
updates config/default.yaml with new tier assignments, then launches the sweep.

Screen flags (in priority order):
  MANUAL_EXCLUDE_BAD_HISTORY  — proven bad backtest record; blocked regardless of scores
  REJECT_INSUFFICIENT_DATA    — bars < MIN_BARS (sparse data, CTD-style)
  REJECT_HIGH_ATR             — atr_pct > ATR_CEIL for tier (too volatile to mean-revert)
  REJECT_LOW_ATR              — atr_pct < ATR_FLOOR (too flat, never reaches signal thresholds)
  REJECT_LOW_VWAP_DEV         — vwap_dev_freq < VWAP_DEV_MIN (never strays below VWAP)
  REJECT_LOW_RANGING          — ranging_pct < RANGING_MIN for tier_b (no mean-reversion regime)
  PASS_WITH_WARNING           — passes hard filters but vwap_dev/ranging ratio < RATIO_WARN
                                 (high VWAP deviation relative to ranging may reflect trending
                                  dips rather than genuine mean reversion)
  PASS                        — meets all criteria cleanly

Tier assignment (from behaviour score):
  ranging_pct >= 25  → tier_a   (ATR ceiling 0.95)
  ranging_pct <  25  → tier_b   (ATR ceiling 1.00, ranging floor 15%)

Caps: PASS stocks fill first; PASS_WITH_WARNING fill remaining slots.
      Max cap_per_tier candidates added per run (default 25).
"""

import os
import subprocess
import sys
from datetime import date

import yaml

SCORES_LOG  = "results/candidates_measurement.log"
CONFIG_PATH = "config/default.yaml"
TODAY       = date.today().isoformat()

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_BARS        = 3000    # minimum hourly bars for reliable statistics
ATR_FLOOR       = 0.45    # below this: stock too flat to generate meaningful signals
ATR_CEIL_A      = 1.00    # tier_a ceiling — same as tier_b; tier split is about ranging not volatility
ATR_CEIL_B      = 1.00    # tier_b ceiling
VWAP_DEV_MIN    = 15.0    # minimum vwap_dev_freq % for both tiers
RANGING_MIN_B   = 15.0    # minimum ranging_pct for tier_b (floor, not tier boundary)
RATIO_WARN      = 0.80    # vwap_dev/ranging below this → PASS_WITH_WARNING
                           # ratio < 1 means VWAP deviations happen mostly during trends

# ── Protected active tickers ─────────────────────────────────────────────────
# These are always written to config unchanged. Not subject to screen filters.
ACTIVE_TICKERS = {
    "tier_a": ["EBO","FPH","SDF","GMG","SEK","TPG","ALD","DXS","CBA","BXB",
               "DOW","NAB","JBH","AGL","CGF","VCX","SHL","ASX"],
    "tier_b": ["ARB","CPU","XRO","NXT","ANN","REA","MQG","TNE",
               "WTC","CHC","HUB","COH","WDS"],
}

# ── Manual exclusion list ─────────────────────────────────────────────────────
# Stocks that pass the behaviour screen but are blocked due to proven bad
# backtest history. Update this list after each rotation cycle.
# Format: ticker → reason string (shown in summary).
MANUAL_EXCLUDE = {
    "FMG": "repeated_zero_win_rate",       # 0% win rate across multiple runs
    "PMV": "chronic_negative_pnl",         # consistently negative across 3+ runs
    "PPT": "chronic_negative_pnl",
    "SIG": "chronic_negative_pnl",
    "MND": "chronic_negative_pnl",         # 16% win rate, -$673 in best run
    "FLT": "quarantine_cyclical_event_risk",  # one stop but high event sensitivity; re-evaluate 2026-08
    "IFL": "delisted",                        # IFL.AX no data as of 2026-05-22; possibly delisted
    # CTD: would also fail REJECT_INSUFFICIENT_DATA (bars=2148).
}

# ── Stock name lookup ─────────────────────────────────────────────────────────
NAMES = {
    "WBC":"Westpac","ANZ":"ANZ Group","BEN":"Bendigo Bank","BOQ":"Bank of Qld",
    "AMP":"AMP","AUB":"AUB Group","HLI":"Helia Group","PNI":"Pinnacle Invest",
    "MFG":"Magellan Financial","PTM":"Platinum Asset","EQT":"EQT Holdings",
    "PAC":"Pacific Current","JHG":"Janus Henderson","AZJ":"Aurizon",
    "CWY":"Cleanaway","GPT":"GPT Group","SGP":"Stockland","LLC":"Lendlease",
    "ABP":"Abacus Property","CQR":"Charter Hall Retail","HDN":"HomeCo Daily Needs",
    "ARF":"Arena REIT","CNI":"Centuria Industrial","COF":"Centuria Office",
    "HPI":"Hotel Property","URW":"Unibail-Rodamco","HMC":"HMC Capital",
    "WPR":"Waypoint REIT","CLW":"Charter Hall Long WALE","CIP":"Centuria Ind REIT",
    "SCG":"Scentre Group","APA":"APA Group","TWE":"Treasury Wine",
    "GNC":"GrainCorp","BGA":"Bega Cheese","ING":"Inghams Group","ELD":"Elders",
    "SIG":"Sigma Healthcare","HLS":"Healius","CAJ":"Capitol Health",
    "IDX":"Integral Diagnostics","MVF":"Monash IVF","NAN":"Nanosonics",
    "SUN":"Suncorp","MPL":"Medibank","NHF":"nib Holdings","RHC":"Ramsay Health",
    "NEC":"Nine Entertainment","HT1":"HT&E","OML":"oOh!media","SWM":"Seven West",
    "SPK":"Spark NZ","NWS":"News Corp","FLT":"Flight Centre","WEB":"Webjet",
    "CTD":"Corporate Travel","QAN":"Qantas","AX1":"Accent Group",
    "UNI":"Universal Store","KMD":"KMD Brands","APE":"Eagers Automotive",
    "GWA":"GWA Group","MND":"Monadelphous","IPH":"IPH Limited","WOR":"Worley",
    "NWH":"NRW Holdings","DGL":"DGL Group","ALL":"Aristocrat Leisure","TAH":"Tabcorp",
    "CSL":"CSL Limited","PME":"Pro Medicus","PNV":"Polynovo","MSB":"Mesoblast",
    "IMM":"Immutep","APX":"Appen","MP1":"Megaport","IDP":"IDP Education",
    "JIN":"Jumbo Interactive","SQ2":"Block","ZIP":"Zip Co","TYR":"Tyro Payments",
    "SLC":"Superloop","EML":"EML Payments","DTL":"Data#3","RDY":"ReadyTech",
    "MAQ":"Macquarie Telecom","SKO":"Serko","LNK":"Link Admin","PPS":"Praemium",
    "BTH":"Bigtincan","GTK":"Gentrack","GDG":"Generation Dev","NST":"Northern Star",
    "WHC":"Whitehaven","NHC":"New Hope Coal","YAL":"Yancoal","CRN":"Coronado",
    "GOR":"Gold Road","RRL":"Regis Resources","CMM":"Capricorn Metals",
    "OGC":"OceanaGold","NIC":"Nickel Industries","MIN":"Mineral Resources",
    "PLS":"Pilbara Minerals","LTR":"Liontown","AKE":"Arcadium Lithium",
    "CCX":"City Chic","KGN":"Kogan","THL":"Tourism Holdings","OCA":"Oceania HC",
    "AIZ":"Air NZ","SXL":"Southern Cross","GEM":"G8 Education","VGL":"Vista Group",
    "WES":"Wesfarmers","COL":"Coles","WOW":"Woolworths",
    "TCL":"Transurban","ALX":"Atlas Arteria","QBE":"QBE Insurance","IAG":"IAG",
    "SFR":"Sandfire","STO":"Santos","ORG":"Origin Energy","RIO":"Rio Tinto",
    "FMG":"Fortescue","BHP":"BHP Group","S32":"South32","MTS":"Metcash",
    "HVN":"Harvey Norman","NWH":"NRW Holdings","WPR":"Waypoint REIT",
}


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_scores(log_path: str) -> list:
    """
    Table format: ── / header / ── / data rows / ──
    Data rows start after the SECOND ── line.
    """
    rows = []
    dash_count = 0
    with open(log_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("─"):
                dash_count += 1
                continue
            if dash_count < 2:
                continue
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            if parts[0] == "ticker":
                continue
            try:
                rows.append({
                    "ticker":        parts[0],
                    "sector":        parts[1],
                    "ranging_pct":   float(parts[2]),
                    "atr_pct":       float(parts[3]),
                    "vwap_dev_freq": float(parts[4]),
                    "bars":          int(parts[5]),
                    "tier_label":    parts[6] if len(parts) > 6 else "mid",
                })
            except (ValueError, IndexError):
                continue
    return rows


# ── Screening ─────────────────────────────────────────────────────────────────

def screen(row: dict) -> tuple:
    """
    Returns (flag, detail) for a single candidate row.

    Flags (in priority order):
      MANUAL_EXCLUDE_BAD_HISTORY
      REJECT_INSUFFICIENT_DATA
      REJECT_HIGH_ATR
      REJECT_LOW_ATR
      REJECT_LOW_VWAP_DEV
      REJECT_LOW_RANGING          (tier_b only)
      PASS_WITH_WARNING
      PASS
    """
    ticker  = row["ticker"]
    ranging = row["ranging_pct"]
    atr     = row["atr_pct"]
    vwap    = row["vwap_dev_freq"]
    bars    = row["bars"]
    tier    = "tier_a" if ranging >= 25 else "tier_b"

    if ticker in MANUAL_EXCLUDE:
        return "MANUAL_EXCLUDE_BAD_HISTORY", MANUAL_EXCLUDE[ticker]

    if bars < MIN_BARS:
        return "REJECT_INSUFFICIENT_DATA", f"bars={bars} < {MIN_BARS}"

    atr_ceil = ATR_CEIL_A if tier == "tier_a" else ATR_CEIL_B
    if atr > atr_ceil:
        return "REJECT_HIGH_ATR", f"atr={atr:.3f} > {atr_ceil}"

    if atr < ATR_FLOOR:
        return "REJECT_LOW_ATR", f"atr={atr:.3f} < {ATR_FLOOR}"

    if vwap < VWAP_DEV_MIN:
        return "REJECT_LOW_VWAP_DEV", f"vwap_dev={vwap:.1f}% < {VWAP_DEV_MIN}%"

    if tier == "tier_b" and ranging < RANGING_MIN_B:
        return "REJECT_LOW_RANGING", f"ranging={ranging:.1f}% < {RANGING_MIN_B}%"

    ratio = vwap / ranging if ranging > 0 else 0
    if ratio < RATIO_WARN:
        return "PASS_WITH_WARNING", (
            f"dev_ratio={ratio:.2f} < {RATIO_WARN} "
            f"(vwap_dev={vwap:.1f}% but ranging={ranging:.1f}% — trending-dipper risk)"
        )

    return "PASS", ""


def select_candidates(rows: list, cap_per_tier: int = 25) -> tuple:
    """
    Screen all candidate rows, exclude actives and pure-numeric tickers.
    Returns (tier_a_list, tier_b_list, screened_all).

    Within each tier:
      - PASS stocks fill first, sorted by ranging_pct descending
      - PASS_WITH_WARNING fill remaining slots
      - All REJECT* and MANUAL_EXCLUDE are dropped
    Each tier capped at cap_per_tier.
    """
    all_active = set(ACTIVE_TICKERS["tier_a"] + ACTIVE_TICKERS["tier_b"])

    screened = []
    for r in rows:
        if r["ticker"] in all_active:
            continue
        if r["ticker"].isdigit():
            r = dict(r)
            r["flag"], r["flag_detail"] = "REJECT_INSUFFICIENT_DATA", "pure-numeric ticker (YAML int risk)"
            screened.append(r)
            continue
        flag, detail = screen(r)
        r = dict(r)
        r["flag"] = flag
        r["flag_detail"] = detail
        screened.append(r)

    def pick(candidates, tier_key):
        in_tier = [r for r in candidates if ("tier_a" if r["ranging_pct"] >= 25 else "tier_b") == tier_key]
        passing  = sorted([r for r in in_tier if r["flag"] == "PASS"],
                          key=lambda x: -x["ranging_pct"])
        warnings = sorted([r for r in in_tier if r["flag"] == "PASS_WITH_WARNING"],
                          key=lambda x: -x["ranging_pct"])
        combined = passing + warnings
        return combined[:cap_per_tier]

    tier_a = pick(screened, "tier_a")
    tier_b = pick(screened, "tier_b")
    return tier_a, tier_b, screened


# ── Config writer ─────────────────────────────────────────────────────────────

def update_config(tier_a_new: list, tier_b_new: list):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    all_active_a = set(ACTIVE_TICKERS["tier_a"])
    all_active_b = set(ACTIVE_TICKERS["tier_b"])
    scores_map   = {r["ticker"]: r for r in tier_a_new + tier_b_new}

    new_a_tickers = ACTIVE_TICKERS["tier_a"] + [r["ticker"] for r in tier_a_new]
    new_b_tickers = ACTIVE_TICKERS["tier_b"] + [r["ticker"] for r in tier_b_new]

    ta_params  = {k: v for k, v in cfg["universe"]["tiers"]["tier_a"].items() if k != "tickers"}
    tb_params  = {k: v for k, v in cfg["universe"]["tiers"]["tier_b"].items() if k != "tickers"}
    tier_sweep = cfg.get("tier_sweep", {})
    rest       = {k: v for k, v in cfg.items() if k not in ("universe", "tier_sweep")}

    def ticker_block(tickers, active_set, scores):
        lines = []
        for t in tickers:
            name = NAMES.get(t, t)
            if t in active_set:
                comment = "(active)"
            elif t in scores:
                r = scores[t]
                warn = " ⚠" if r.get("flag") == "PASS_WITH_WARNING" else ""
                comment = f"{r['ranging_pct']:.1f}%{warn}"
            else:
                comment = ""
            lines.append(f"        - {t:<5} # {name:<26} {comment}")
        return "\n".join(lines)

    def params_block(params):
        return "\n".join(f"      {k}: {v}" for k, v in params.items())

    def tier_sweep_block(ts):
        lines = ["tier_sweep:"]
        for tier_name, sweep in ts.items():
            lines.append(f"  {tier_name}:")
            for k, v in sweep.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    def rest_block(r):
        return yaml.dump(r, default_flow_style=False, sort_keys=False).rstrip()

    content = f"""universe:
  tiers:
    # Behaviour tier A — ranging% > 25%  ({len(new_a_tickers)} stocks)
    tier_a:
      tickers:
{ticker_block(new_a_tickers, all_active_a, scores_map)}
{params_block(ta_params)}

    # Behaviour tier B — ranging% <= 25%  ({len(new_b_tickers)} stocks)
    tier_b:
      tickers:
{ticker_block(new_b_tickers, all_active_b, scores_map)}
{params_block(tb_params)}

{rest_block(rest)}

{tier_sweep_block(tier_sweep)}
"""

    with open(CONFIG_PATH, "w") as f:
        f.write(content)

    print(f"Config written: tier_a={len(new_a_tickers)}, tier_b={len(new_b_tickers)} stocks")


# ── Summary printer ───────────────────────────────────────────────────────────

_FLAG_ORDER = [
    "PASS",
    "PASS_WITH_WARNING",
    "REJECT_LOW_RANGING",
    "REJECT_LOW_VWAP_DEV",
    "REJECT_LOW_ATR",
    "REJECT_HIGH_ATR",
    "REJECT_INSUFFICIENT_DATA",
    "MANUAL_EXCLUDE_BAD_HISTORY",
]

def print_summary(tier_a_new: list, tier_b_new: list, screened: list):
    total_a = len(ACTIVE_TICKERS["tier_a"]) + len(tier_a_new)
    total_b = len(ACTIVE_TICKERS["tier_b"]) + len(tier_b_new)

    print(f"\n{'='*72}")
    print(f"  ROTATION SUMMARY  —  {TODAY}")
    print(f"{'='*72}")

    for tier_name, new_stocks, total in [
        ("A", tier_a_new, total_a),
        ("B", tier_b_new, total_b),
    ]:
        n_active = len(ACTIVE_TICKERS[f"tier_{tier_name.lower()}"])
        print(f"\n  Tier {tier_name}: {n_active} active + {len(new_stocks)} new = {total} total")
        for r in new_stocks:
            warn = " ⚠ WARNING" if r["flag"] == "PASS_WITH_WARNING" else ""
            print(f"    + {r['ticker']:<5}  ranging={r['ranging_pct']:.1f}%  "
                  f"atr={r['atr_pct']:.3f}  vwap_dev={r['vwap_dev_freq']:.1f}%"
                  f"  ratio={r['vwap_dev_freq']/r['ranging_pct']:.2f}{warn}")
            if r["flag"] == "PASS_WITH_WARNING":
                print(f"           {r['flag_detail']}")

    # Group rejected candidates by flag
    rejected = [r for r in screened if r["flag"] not in ("PASS", "PASS_WITH_WARNING")]
    by_flag = {}
    for r in rejected:
        by_flag.setdefault(r["flag"], []).append(r)

    print(f"\n  Screened out ({len(rejected)} candidates):")
    for flag in _FLAG_ORDER:
        group = by_flag.get(flag, [])
        if not group:
            continue
        group_sorted = sorted(group, key=lambda x: -x["ranging_pct"])
        print(f"\n    [{flag}]  ({len(group)} stocks)")
        for r in group_sorted[:12]:
            print(f"      {r['ticker']:<5}  ranging={r['ranging_pct']:.1f}%  "
                  f"atr={r['atr_pct']:.3f}  vwap_dev={r['vwap_dev_freq']:.1f}%"
                  f"  — {r['flag_detail']}")
        if len(group) > 12:
            print(f"      ... and {len(group)-12} more")

    print(f"\n  Total universe after rotation: {total_a + total_b} stocks")
    print(f"{'='*72}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(SCORES_LOG):
        print(f"ERROR: {SCORES_LOG} not found.")
        sys.exit(1)

    print(f"Parsing {SCORES_LOG}...")
    rows = parse_scores(SCORES_LOG)
    if not rows:
        print("ERROR: no scores found in log.")
        sys.exit(1)
    print(f"  {len(rows)} stocks scored\n")

    tier_a_new, tier_b_new, screened = select_candidates(rows, cap_per_tier=25)

    print_summary(tier_a_new, tier_b_new, screened)

    print("Updating config/default.yaml...")
    update_config(tier_a_new, tier_b_new)

    print("\nLaunching sweep (bash run_all_sweeps.sh)...")
    result = subprocess.run(
        ["bash", "run_all_sweeps.sh"],
        cwd="/Users/alex/quant-sim",
    )
    sys.exit(result.returncode)
