#!/bin/bash
#SBATCH --job-name=dd_orchestrate
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --account=ab0246
#SBATCH --output=dark_doldrums_orchestrate_%j.out
#SBATCH --error=dark_doldrums_orchestrate_%j.err

# Orchestration script: submits yearly jobs, waits, joins results

cd /work/ab0246/a270092/software/GlobalLab_hackathon_2025

RESULTS_DIR="dark_doldrums_results"
START_YEAR=${1:-1979}
END_YEAR=${2:-2024}

# Parameters - use literature-aligned (24h moving avg) or original (48h)
MOVING_AVG=${3:-24}
THRESHOLD=${4:-0.06}

echo "=============================================="
echo "Dark Doldrums Analysis Orchestrator"
echo "=============================================="
echo "Years: ${START_YEAR}-${END_YEAR}"
echo "Moving average: ${MOVING_AVG}h"
echo "Threshold: ${THRESHOLD}"
echo "Start time: $(date)"
echo ""

mkdir -p ${RESULTS_DIR}

# Submit individual year jobs
declare -a JOB_IDS
for YEAR in $(seq ${START_YEAR} ${END_YEAR}); do
    JOB_ID=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --job-name=dd_${YEAR}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=ab0246
#SBATCH --output=${RESULTS_DIR}/job_${YEAR}_%j.out
#SBATCH --error=${RESULTS_DIR}/job_${YEAR}_%j.err

module load python3
cd /work/ab0246/a270092/software/GlobalLab_hackathon_2025

python3 dark_doldrums_analysis.py --year ${YEAR} --output ${RESULTS_DIR} \
    --threshold ${THRESHOLD} --moving-avg ${MOVING_AVG} --source era5

echo "COMPLETED: ${YEAR}"
EOF
)
    echo "Submitted year ${YEAR}: job ${JOB_ID}"
    JOB_IDS+=("${JOB_ID}")
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs"
echo "Job IDs: ${JOB_IDS[@]}"
echo ""

# Wait for all jobs to complete
echo "Waiting for jobs to complete..."
for JOB_ID in "${JOB_IDS[@]}"; do
    while squeue -j ${JOB_ID} -h 2>/dev/null | grep -q ${JOB_ID}; do
        sleep 60
    done
    echo "  Job ${JOB_ID} completed"
done

echo ""
echo "All jobs completed at: $(date)"
echo ""

# Join results
echo "Joining results..."
python3 << 'PYEOF'
import pandas as pd
from pathlib import Path
import sys

results_dir = Path("dark_doldrums_results")
start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 1979
end_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2024

all_events = []
for year in range(start_year, end_year + 1):
    f = results_dir / f"dark_doldrums_events_{year}.csv"
    if f.exists():
        df = pd.read_csv(f)
        df['year'] = year
        all_events.append(df)
        print(f"  {year}: {len(df)} events")
    else:
        print(f"  {year}: MISSING")

if all_events:
    combined = pd.concat(all_events, ignore_index=True)
    combined = combined.sort_values('severity', ascending=False)
    out = results_dir / f"all_events_{start_year}_{end_year}.csv"
    combined.to_csv(out, index=False)
    print(f"\nCombined {len(combined)} events -> {out}")
    
    # Summary stats
    print(f"\nTop 10 events by severity:")
    print(combined[['year', 'start', 'duration_hours', 'severity']].head(10).to_string())
PYEOF

# Run validation metrics
echo ""
echo "Running validation metrics..."
cd validation_data
python3 quality_score.py
python3 visualize_validation.py
cd ..

echo ""
echo "=============================================="
echo "COMPLETE at: $(date)"
echo "=============================================="
