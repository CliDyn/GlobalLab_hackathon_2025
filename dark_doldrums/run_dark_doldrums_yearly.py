#!/usr/bin/env python3
"""
Parent script to orchestrate yearly dark doldrums analysis via SLURM.

Features:
- Submits one sbatch job per year
- Monitors job completion
- Joins individual CSV results
- Runs validation metrics and plots

Literature-aligned parameters (Kittel & Schill 2024):
- Relative threshold: 20% of mean CF (instead of absolute 0.06)
- 24h moving average (literature: single events typically ≤24h)
- Min duration: 12h (to catch shorter events)
"""
import subprocess
import time
import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Configuration
WORK_DIR = Path("/work/ab0246/a270092/software/GlobalLab_hackathon_2025")
RESULTS_DIR = WORK_DIR / "dark_doldrums_results"
SCRIPT_PATH = WORK_DIR / "dark_doldrums_analysis.py"

# Literature-aligned parameters
LITERATURE_PARAMS = {
    'moving_avg_hours': 24,      # Literature: events typically ≤24h
    'threshold_relative': 0.20,  # 20% of mean CF (Kittel: 10-100% range)
    'threshold_absolute': 0.06,  # Our original absolute threshold
    'min_duration_hours': 12,    # Shorter minimum to catch more events
}

# SLURM job template
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=dd_{year}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=ab0246
#SBATCH --output={results_dir}/job_{year}.out
#SBATCH --error={results_dir}/job_{year}.err

module load python3

cd {work_dir}

python3 {script} --year {year} --output {results_dir} \\
    --threshold {threshold} --moving-avg {moving_avg} \\
    --source era5

echo "DONE: {year}"
"""


def submit_job(year, params, dry_run=False):
    """Submit a single year job to SLURM."""
    job_script = SBATCH_TEMPLATE.format(
        year=year,
        work_dir=WORK_DIR,
        script=SCRIPT_PATH,
        results_dir=RESULTS_DIR,
        threshold=params['threshold'],
        moving_avg=params['moving_avg_hours'],
    )
    
    script_file = RESULTS_DIR / f"submit_{year}.sh"
    with open(script_file, 'w') as f:
        f.write(job_script)
    
    if dry_run:
        print(f"  [DRY RUN] Would submit: {script_file}")
        return f"DRY_{year}"
    
    result = subprocess.run(
        ['sbatch', str(script_file)],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted year {year}: job {job_id}")
        return job_id
    else:
        print(f"  ERROR submitting {year}: {result.stderr}")
        return None


def check_jobs_status(job_ids):
    """Check status of submitted jobs."""
    if not job_ids:
        return True, []
    
    result = subprocess.run(
        ['squeue', '-u', os.environ['USER'], '-h', '-o', '%i %t'],
        capture_output=True, text=True
    )
    
    running_jobs = {}
    for line in result.stdout.strip().split('\n'):
        if line:
            parts = line.split()
            if len(parts) >= 2:
                running_jobs[parts[0]] = parts[1]
    
    pending = []
    for jid in job_ids:
        if jid in running_jobs:
            pending.append((jid, running_jobs[jid]))
    
    return len(pending) == 0, pending


def wait_for_jobs(job_ids, check_interval=30):
    """Wait for all jobs to complete."""
    print("\nWaiting for jobs to complete...")
    
    while True:
        done, pending = check_jobs_status(job_ids)
        
        if done:
            print("All jobs completed!")
            return True
        
        status_str = ', '.join([f"{jid}({st})" for jid, st in pending[:5]])
        if len(pending) > 5:
            status_str += f" ... +{len(pending)-5} more"
        print(f"  {len(pending)} jobs pending: {status_str}")
        
        time.sleep(check_interval)


def join_results(years, results_dir):
    """Join individual year CSV files into combined dataset."""
    print("\nJoining results...")
    
    all_events = []
    all_stats = []
    
    for year in years:
        events_file = results_dir / f"dark_doldrums_events_{year}.csv"
        
        if events_file.exists():
            df = pd.read_csv(events_file)
            df['year'] = year
            all_events.append(df)
            print(f"  {year}: {len(df)} events")
        else:
            print(f"  {year}: MISSING")
    
    if all_events:
        combined = pd.concat(all_events, ignore_index=True)
        combined = combined.sort_values('severity', ascending=False)
        
        output_file = results_dir / f"all_events_{years[0]}_{years[-1]}.csv"
        combined.to_csv(output_file, index=False)
        print(f"\nCombined {len(combined)} events -> {output_file}")
        return combined
    
    return None


def run_validation(results_dir):
    """Run validation metrics and generate plots."""
    print("\nRunning validation...")
    
    validation_dir = WORK_DIR / "validation_data"
    
    # Run quality score
    subprocess.run(
        ['python3', str(validation_dir / 'quality_score.py')],
        cwd=validation_dir
    )
    
    # Run visualization
    subprocess.run(
        ['python3', str(validation_dir / 'visualize_validation.py')],
        cwd=validation_dir
    )
    
    print("Validation complete!")


def main():
    parser = argparse.ArgumentParser(description='Orchestrate yearly dark doldrums analysis')
    parser.add_argument('--start-year', type=int, default=1979, help='First year')
    parser.add_argument('--end-year', type=int, default=2024, help='Last year')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--use-literature-params', action='store_true', 
                        help='Use literature-aligned parameters')
    parser.add_argument('--threshold', type=float, default=0.06, 
                        help='CF threshold (default: 0.06)')
    parser.add_argument('--moving-avg', type=int, default=48,
                        help='Moving average window hours (default: 48)')
    parser.add_argument('--join-only', action='store_true',
                        help='Only join existing results, no new jobs')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only run validation on existing results')
    
    args = parser.parse_args()
    
    years = list(range(args.start_year, args.end_year + 1))
    
    # Set parameters
    if args.use_literature_params:
        params = {
            'threshold': LITERATURE_PARAMS['threshold_absolute'],
            'moving_avg_hours': LITERATURE_PARAMS['moving_avg_hours'],
        }
        print("Using LITERATURE-ALIGNED parameters:")
    else:
        params = {
            'threshold': args.threshold,
            'moving_avg_hours': args.moving_avg,
        }
        print("Using CUSTOM parameters:")
    
    print(f"  Threshold: CF < {params['threshold']}")
    print(f"  Moving average: {params['moving_avg_hours']}h")
    print(f"  Years: {args.start_year}-{args.end_year} ({len(years)} years)")
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)
    
    if args.validate_only:
        run_validation(RESULTS_DIR)
        return
    
    if args.join_only:
        join_results(years, RESULTS_DIR)
        run_validation(RESULTS_DIR)
        return
    
    # Submit jobs
    print(f"\nSubmitting {len(years)} jobs...")
    job_ids = []
    for year in years:
        jid = submit_job(year, params, dry_run=args.dry_run)
        if jid:
            job_ids.append(jid)
    
    if args.dry_run:
        print("\n[DRY RUN] Would wait for jobs and join results")
        return
    
    # Wait for completion
    wait_for_jobs(job_ids)
    
    # Join results
    join_results(years, RESULTS_DIR)
    
    # Run validation
    run_validation(RESULTS_DIR)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
