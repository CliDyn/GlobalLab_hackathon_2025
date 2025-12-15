#!/bin/sh
#PBS -q iccp
#PBS -l walltime=08:00:00
#PBS -l nodes=1
#PBS -j oe
#PBS -N dunkelflaute_tco
#PBS -A awiiccp

# Dunkelflaute analysis using TCo1279-DART data
# Usage: qsub -v SCENARIO=1950C,START_YEAR=1950,END_YEAR=2014 submit_dunkelflaute_tco1279.sh
# Example: qsub -v SCENARIO=2080C,START_YEAR=2065,END_YEAR=2100 submit_dunkelflaute_tco1279.sh

# Default values if not passed via -v
SCENARIO=${SCENARIO:-1950C}
START_YEAR=${START_YEAR:-1950}
END_YEAR=${END_YEAR:-2014}

# Path to the script (adjust as needed for target HPC)
SCRIPT_DIR="/scratch/awiiccp2/dunkelflaute"
cd $SCRIPT_DIR

echo "Starting TCo1279-DART Dunkelflaute analysis"
echo "Scenario: $SCENARIO"
echo "Years: $START_YEAR to $END_YEAR"
echo "Start time: $(date)"

# Create output directory for this scenario
OUTPUT_DIR="${SCRIPT_DIR}/dunkelflaute_results/tco1279_${SCENARIO}"
mkdir -p $OUTPUT_DIR

python3 dunkelflaute_era5.py \
    --source tco1279 \
    --scenario $SCENARIO \
    --start $START_YEAR \
    --end $END_YEAR \
    --output-dir $OUTPUT_DIR

echo ""
echo "Finished: $(date)"
