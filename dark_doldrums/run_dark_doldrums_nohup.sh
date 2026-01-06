#!/bin/bash
# Run dark_doldrums CTRL vs SSP585 analysis with nohup
# Usage: nohup ./run_dark_doldrums_nohup.sh > dark_doldrums.log 2>&1 &

SCRIPT_DIR="/mnt/lustre/home/awiiccp2/software/GlobalLab_hackathon_2025"
cd $SCRIPT_DIR

DB_PATH="${SCRIPT_DIR}/dark_doldrums_results/dark_doldrums_results.db"

echo "=== Dark_Doldrums CTRL vs SSP585 ==="
echo "Started: $(date)"
echo ""

# CTRL: 1950C, 1995-2014 (20 years)
echo "Running CTRL (1950C) 1995-2014..."
python3 dark_doldrums_analysis.py \
    --source tco1279 --scenario 1950C \
    --start 1995 --end 2014 \
    --output-dir ./dark_doldrums_results/tco1279_CTRL \
    --region Germany --db $DB_PATH

echo ""
echo "CTRL done: $(date)"
echo ""

# SSP585: 2080C, 2091-2099 (9 years - 3h data only available from 2091)
echo "Running SSP585 (2080C) 2091-2099..."
python3 dark_doldrums_analysis.py \
    --source tco1279 --scenario 2080C \
    --start 2091 --end 2099 \
    --output-dir ./dark_doldrums_results/tco1279_SSP585 \
    --region Germany --db $DB_PATH

echo ""
echo "=== All done: $(date) ==="
