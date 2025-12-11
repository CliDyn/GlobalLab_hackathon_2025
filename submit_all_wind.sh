#!/bin/bash
#SBATCH --account=ab0246
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=max_wind
#SBATCH --output=/work/ab0246/a270092/software/GlobalLab_hackathon_2025/max_wind_%j.out

cd /work/ab0246/a270092/software/GlobalLab_hackathon_2025
python process_all_wind.py
