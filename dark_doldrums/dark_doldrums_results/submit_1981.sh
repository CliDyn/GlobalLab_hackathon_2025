#!/bin/bash
#SBATCH --job-name=dd_1981
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=ab0246
#SBATCH --output=/work/ab0246/a270092/software/GlobalLab_hackathon_2025/dark_doldrums_results/job_1981.out
#SBATCH --error=/work/ab0246/a270092/software/GlobalLab_hackathon_2025/dark_doldrums_results/job_1981.err

module load python3

cd /work/ab0246/a270092/software/GlobalLab_hackathon_2025

python3 /work/ab0246/a270092/software/GlobalLab_hackathon_2025/dark_doldrums_analysis.py --year 1981 --output /work/ab0246/a270092/software/GlobalLab_hackathon_2025/dark_doldrums_results \
    --threshold 0.06 --moving-avg 24 \
    --source era5

echo "DONE: 1981"
