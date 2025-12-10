#!/bin/bash
#SBATCH --account=ab0246
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --job-name=city_temps
#SBATCH --output=/work/ab0246/a270092/software/GlobalLab_hackathon_2025/city_temps_%j.out

cd /work/ab0246/a270092/software/GlobalLab_hackathon_2025
python process_city_temps.py
