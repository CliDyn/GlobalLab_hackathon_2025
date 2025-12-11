#!/usr/bin/env python
"""Debug script to test loading a specific year with verbose output."""
import intake
import numpy as np
import traceback

CATALOG_PATH = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/catalog/main.yaml"

# Config
EXP = 'TCo1279-DART-2080C'  # SSP585
YEAR_IDX = 6  # 2086 = 2080 + 6
VAR = '2t'

print(f"Loading catalog...", flush=True)
gl_cat = intake.open_catalog(CATALOG_PATH)

print(f"Opening dataset: {EXP}", flush=True)
ds = gl_cat['ICCP'][EXP]['atmos']['native']['atmos_3h'].to_dask()

print(f"Dataset info:", flush=True)
print(f"  dims: {dict(ds.dims)}", flush=True)
print(f"  {VAR} shape: {ds[VAR].shape}", flush=True)
print(f"  {VAR} chunks: {ds[VAR].chunks}", flush=True)

# Calculate time range for year
t_start = YEAR_IDX * 2920
t_end = min(t_start + 2920, ds.dims['time_counter'])
print(f"\nYear 2086: timesteps {t_start} to {t_end}", flush=True)

# Try loading in smaller chunks to find the bad one
chunk_size = 100
for chunk_start in range(t_start, t_end, chunk_size):
    chunk_end = min(chunk_start + chunk_size, t_end)
    print(f"  Loading timesteps {chunk_start}-{chunk_end}...", end=" ", flush=True)
    try:
        data = ds[VAR].isel(time_counter=slice(chunk_start, chunk_end)).values
        print(f"OK, shape={data.shape}, max={data.max()-273.15:.1f}°C", flush=True)
    except Exception as e:
        print(f"FAILED!", flush=True)
        print(f"  Error: {e}", flush=True)
        print(f"\nFull traceback:", flush=True)
        traceback.print_exc()
        
        # Try even smaller to pinpoint
        print(f"\nPinpointing bad timestep in range {chunk_start}-{chunk_end}:", flush=True)
        for t in range(chunk_start, chunk_end):
            try:
                data = ds[VAR].isel(time_counter=t).values
                print(f"    t={t}: OK", flush=True)
            except Exception as e2:
                print(f"    t={t}: FAILED - {e2}", flush=True)
                
                # Get timestamp info
                try:
                    ts = ds['time_counter'].isel(time_counter=t).values
                    print(f"    Timestamp: {ts}", flush=True)
                except:
                    pass
                break
        break

print("\nDone.", flush=True)
