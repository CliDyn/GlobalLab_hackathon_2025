#!/bin/bash
#SBATCH --account=ab0246
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --job-name=city_temp
#SBATCH --output=/work/ab0246/a270092/software/GlobalLab_hackathon_2025/city_extremes/city_temp_%j.out

cd /work/ab0246/a270092/software/GlobalLab_hackathon_2025/city_extremes
python process_city.py --var temp --all
