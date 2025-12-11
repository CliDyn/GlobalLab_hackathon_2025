#!/bin/bash
#SBATCH --account=ab0246
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --job-name=city_wind_ctrl
#SBATCH --output=city_wind_ctrl_%j.out

cd /work/ab0246/a270092/software/GlobalLab_hackathon_2025
python process_city_v2.py --var wind --exp ctrl --all
