#!/bin/bash
#SBATCH --job-name=dark_doldrums
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=ab0246
#SBATCH --output=dark_doldrums_%j.out
#SBATCH --error=dark_doldrums_%j.err

# Dark_Doldrums analysis using ERA5 data
# Usage: sbatch submit_dark_doldrums_era5.sh [year]

YEAR=${1:-1997}

cd /work/ab0246/a270092/software/GlobalLab_hackathon_2025

echo "Starting Dark_Doldrums analysis for year $YEAR"
echo "Time: $(date)"

python3 dark_doldrums/dark_doldrums_analysis.py --year $YEAR --output-dir ./dark_doldrums/dark_doldrums_results

echo "Finished: $(date)"
