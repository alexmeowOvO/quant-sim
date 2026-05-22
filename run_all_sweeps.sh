#!/bin/bash
cd /Users/alex/quant-sim
for tier in tier_a tier_b; do
    echo "========================================"
    echo "Starting sweep: $tier ($(date))"
    echo "========================================"
    python3 sweep_tier.py --tier $tier --out results/sweep_${tier}.csv
    echo "Done: $tier ($(date))"
done
echo "All sweeps complete."
