#!/bin/sh
#PBS -q iccp
#PBS -l walltime=24:00:00
#PBS -l nodes=1
#PBS -j oe
#PBS -N dunkelflaute_comparison
#PBS -A awiiccp

# Dunkelflaute analysis for CTRL (1950C) and SSP585 (2080C) periods
# CTRL: 20 years (1995-2014)
# SSP585: 14 years (2086-2099)

SCRIPT_DIR="/mnt/lustre/home/awiiccp2/software/GlobalLab_hackathon_2025"
cd $SCRIPT_DIR

echo "=============================================="
echo "Dunkelflaute CTRL vs SSP585 Comparison"
echo "=============================================="
echo "Start time: $(date)"

# CTRL period: 1950C scenario, 1995-2014 (20 years)
echo ""
echo "=== Running CTRL (1950C) 1995-2014 ==="
OUTPUT_CTRL="${SCRIPT_DIR}/dunkelflaute_results/tco1279_CTRL"
mkdir -p $OUTPUT_CTRL

python3 dunkelflaute_analysis.py \
    --source tco1279 \
    --scenario 1950C \
    --start 1995 \
    --end 2014 \
    --output-dir $OUTPUT_CTRL

echo "CTRL finished: $(date)"

# SSP585 period: 2080C scenario, 2086-2099 (14 years)
echo ""
echo "=== Running SSP585 (2080C) 2086-2099 ==="
OUTPUT_SSP="${SCRIPT_DIR}/dunkelflaute_results/tco1279_SSP585"
mkdir -p $OUTPUT_SSP

python3 dunkelflaute_analysis.py \
    --source tco1279 \
    --scenario 2080C \
    --start 2086 \
    --end 2099 \
    --output-dir $OUTPUT_SSP

echo "SSP585 finished: $(date)"

echo ""
echo "=============================================="
echo "All done: $(date)"
echo "=============================================="
