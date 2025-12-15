#!/bin/bash
#SBATCH --job-name=dunkelflaute_all
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=ab0246
#SBATCH --output=dunkelflaute_multiyear_%j.out
#SBATCH --error=dunkelflaute_multiyear_%j.err

# Multi-year Dunkelflaute analysis using ERA5 data (1979-2024)

cd /work/ab0246/a270092/software/GlobalLab_hackathon_2025

echo "Starting multi-year Dunkelflaute analysis"
echo "Years: 1979-2024"
echo "Start time: $(date)"

for year in $(seq 1979 2024); do
    echo ""
    echo "=========================================="
    echo "Processing year $year"
    echo "=========================================="
    python3 dunkelflaute_analysis.py --year $year --output-dir ./dunkelflaute_results
done

echo ""
echo "All years complete: $(date)"

# Combine all events into one file
echo "Combining results..."
python3 -c "
import pandas as pd
from pathlib import Path
import glob

# Combine all event files
event_files = sorted(glob.glob('dunkelflaute_results/dunkelflaute_events_*.csv'))
all_events = []
for f in event_files:
    year = f.split('_')[-1].replace('.csv', '')
    df = pd.read_csv(f)
    df['year'] = int(year)
    all_events.append(df)

combined = pd.concat(all_events, ignore_index=True)
combined = combined.sort_values('severity', ascending=False)
combined.to_csv('dunkelflaute_results/all_events_1979_2024.csv', index=False)
print(f'Combined {len(combined)} events from {len(event_files)} years')
print('Top 20 events by severity:')
print(combined.head(20)[['year', 'start', 'end', 'duration_hours', 'mean_cf', 'severity']].to_string())
"
