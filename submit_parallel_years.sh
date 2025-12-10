#!/bin/bash
# Submit two parallel jobs for years 1950 and 1951

SCRIPT_DIR="/work/ab0246/a270092/software/GlobalLab_hackathon_2025"

# Submit job for 1950
sbatch --account=ab0246 --partition=compute --time=00:10:00 \
    --cpus-per-task=128 --mem=64G \
    --job-name=max_1950 \
    --output=${SCRIPT_DIR}/max_1950_%j.out \
    --wrap="python ${SCRIPT_DIR}/find_max_temp_db.py 1950 --threads 128"

# Submit job for 1951
sbatch --account=ab0246 --partition=compute --time=00:10:00 \
    --cpus-per-task=128 --mem=64G \
    --job-name=max_1951 \
    --output=${SCRIPT_DIR}/max_1951_%j.out \
    --wrap="python ${SCRIPT_DIR}/find_max_temp_db.py 1951 --threads 128"

echo "Jobs submitted. Check status with: squeue -u \$USER"
