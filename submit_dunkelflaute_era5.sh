#!/bin/bash
#SBATCH --job-name=dunkelflaute
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=ab0246
#SBATCH --output=dunkelflaute_%j.out
#SBATCH --error=dunkelflaute_%j.err

# Dunkelflaute analysis using ERA5 data
# Usage: sbatch submit_dunkelflaute_era5.sh [year]

YEAR=${1:-1997}

cd /work/ab0246/a270092/software/GlobalLab_hackathon_2025

echo "Starting Dunkelflaute analysis for year $YEAR"
echo "Time: $(date)"

python3 dunkelflaute_analysis.py --year $YEAR --output-dir ./dunkelflaute_results

echo "Finished: $(date)"
