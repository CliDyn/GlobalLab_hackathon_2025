#!/bin/bash
#SBATCH --account=ab0246
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=max_all_years
#SBATCH --output=/work/ab0246/a270092/software/GlobalLab_hackathon_2025/max_all_years_%j.out

cd /work/ab0246/a270092/software/GlobalLab_hackathon_2025
python process_all_experiments.py
