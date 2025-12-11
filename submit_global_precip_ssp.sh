#!/bin/bash
#SBATCH --account=ab0246
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --job-name=global_precip_ssp
#SBATCH --output=global_precip_ssp_%j.out

cd /work/ab0246/a270092/software/GlobalLab_hackathon_2025
python process_global_max.py --var precip --exp ssp585
