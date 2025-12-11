#!/usr/bin/env python
"""Process all years from both CTRL (1950C) and SSP585 (2080C) experiments for precipitation."""
import intake
import time as timer
import dask
import sqlite3
import numpy as np

DB_PATH = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/max_precip.db"
CATALOG_PATH = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/catalog/main.yaml"
N_THREADS = 32

def init_db():
    """Initialize database with experiment column."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS max_precipitation (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment TEXT,
        year INTEGER,
        max_precip_m REAL,
        max_precip_mm REAL,
        time_idx INTEGER,
        cell_idx INTEGER,
        timestamp TEXT,
        lat REAL,
        lon REAL,
        elapsed_s REAL,
        n_threads INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(experiment, year)
    )''')
    conn.commit()
    conn.close()

def process_year(ds, year, experiment, n_threads=32):
    """Find max precipitation (cp + lsp) and its location for a given year."""
    time_coord = ds['time_counter']
    year_mask = time_coord.dt.year == year
    
    cp_year = ds['cp'].sel(time_counter=year_mask)
    lsp_year = ds['lsp'].sel(time_counter=year_mask)
    
    n_timesteps = cp_year.shape[0]
    if n_timesteps == 0:
        print(f"  No data for year {year}")
        return None
    
    # Total precipitation = convective + large-scale
    precip_year = cp_year + lsp_year
    
    print(f"  Year {year}: {n_timesteps} timesteps", end=" ", flush=True)
    
    with dask.config.set(scheduler='threads', num_workers=n_threads):
        start = timer.time()
        
        # Find max value
        max_val = float(precip_year.max().compute())
        
        # Find index of max
        flat_idx = int(precip_year.argmax().compute())
        time_idx = flat_idx // precip_year.shape[1]
        cell_idx = flat_idx % precip_year.shape[1]
        
        # Get timestamp
        timestamp = precip_year.time_counter.isel(time_counter=time_idx).compute()
        
        # Get lat/lon
        lat = float(ds['lat'].isel(cell=cell_idx).compute())
        lon = float(ds['lon'].isel(cell=cell_idx).compute())
        
        elapsed = timer.time() - start
    
    max_mm = max_val * 1000  # Convert m to mm
    print(f"-> {max_mm:.1f}mm at ({lat:.1f}°N, {lon:.1f}°E) [t={time_idx}, c={cell_idx}] [{elapsed:.0f}s]")
    
    return {
        'experiment': experiment,
        'year': year,
        'max_precip_m': max_val,
        'max_precip_mm': max_mm,
        'time_idx': time_idx,
        'cell_idx': cell_idx,
        'timestamp': str(timestamp.values),
        'lat': lat,
        'lon': lon,
        'elapsed_s': elapsed,
        'n_threads': n_threads,
    }

def save_result(result):
    """Save single result to database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO max_precipitation 
        (experiment, year, max_precip_m, max_precip_mm, time_idx, cell_idx, 
         timestamp, lat, lon, elapsed_s, n_threads)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (result['experiment'], result['year'], result['max_precip_m'], 
         result['max_precip_mm'], result['time_idx'], result['cell_idx'],
         result['timestamp'], result['lat'], result['lon'], 
         result['elapsed_s'], result['n_threads']))
    conn.commit()
    conn.close()

def main():
    init_db()
    gl_cat = intake.open_catalog(CATALOG_PATH)
    
    experiments = [
        ('TCo1279-DART-1950C', 'CTRL'),
        ('TCo1279-DART-2080C', 'SSP585'),
    ]
    
    total_start = timer.time()
    total_years = 0
    
    for exp_id, exp_name in experiments:
        print(f"\n{'='*60}")
        print(f"Processing {exp_name} ({exp_id})")
        print('='*60)
        
        ds = gl_cat['ICCP'][exp_id]['atmos']['native']['atmos_3h'].to_dask()
        years = sorted(set(ds['time_counter'].dt.year.compute().values))
        print(f"Years: {years[0]} to {years[-1]} ({len(years)} years)\n")
        
        for year in years:
            result = process_year(ds, year, exp_name, N_THREADS)
            if result:
                save_result(result)
                total_years += 1
    
    total_elapsed = timer.time() - total_start
    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_years} years in {total_elapsed/60:.1f} min")
    print(f"Average: {total_elapsed/total_years:.1f}s per year")
    print(f"Database: {DB_PATH}")
    print('='*60)

if __name__ == '__main__':
    main()
