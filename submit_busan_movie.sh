#!/bin/bash
#SBATCH --account=ab0246
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=busan_movie
#SBATCH --output=busan_movie_%j.out
#SBATCH --error=busan_movie_%j.err

cd /work/ab0246/a270092/software/GlobalLab_hackathon_2025

echo "Starting Busan Typhoon Movie Generation"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
echo "Start time: $(date)"

python3 busan_movie.py

echo "End time: $(date)"
echo "Done!"
