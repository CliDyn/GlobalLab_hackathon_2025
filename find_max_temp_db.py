#!/usr/bin/env python
"""Find max temperature for a given year and save to SQLite database."""
import intake
import time as timer
import dask
import sqlite3
import numpy as np
import argparse
import os

def find_max_temp_year(year, n_threads=32):
    """Find max temperature and its location for a given year."""
    gl_cat = intake.open_catalog("/work/ab0246/a270092/software/GlobalLab_hackathon_2025/catalog/main.yaml")
    ds = gl_cat['ICCP']['TCo1279-DART-1950C']['atmos']['native']['atmos_3h'].to_dask()
    
    time_coord = ds['time_counter']
    t2m_year = ds['2t'].sel(time_counter=time_coord.dt.year == year)
    
    print(f"Year {year}: {t2m_year.shape[0]} timesteps × {t2m_year.shape[1]:,} cells")
    
    with dask.config.set(scheduler='threads', num_workers=n_threads):
        start = timer.time()
        
        # Find max value
        max_val = float(t2m_year.max().compute())
        
        # Find index of max (argmax)
        flat_idx = int(t2m_year.argmax().compute())
        time_idx = flat_idx // t2m_year.shape[1]
        cell_idx = flat_idx % t2m_year.shape[1]
        
        # Get the timestamp
        timestamp = t2m_year.time_counter.isel(time_counter=time_idx).compute()
        
        # Get lat/lon for this cell
        lat = float(ds['lat'].isel(cell=cell_idx).compute())
        lon = float(ds['lon'].isel(cell=cell_idx).compute())
        
        elapsed = timer.time() - start
    
    result = {
        'year': year,
        'max_temp_k': max_val,
        'max_temp_c': max_val - 273.15,
        'time_idx': time_idx,
        'cell_idx': cell_idx,
        'timestamp': str(timestamp.values),
        'lat': lat,
        'lon': lon,
        'elapsed_s': elapsed,
        'n_threads': n_threads,
    }
    
    print(f"Max: {max_val:.2f} K ({max_val-273.15:.2f} °C)")
    print(f"Location: lat={lat:.2f}, lon={lon:.2f}")
    print(f"Time: {timestamp.values}")
    print(f"Elapsed: {elapsed:.1f}s")
    
    return result

def save_to_db(result, db_path):
    """Save result to SQLite database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS max_temperatures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        year INTEGER UNIQUE,
        max_temp_k REAL,
        max_temp_c REAL,
        time_idx INTEGER,
        cell_idx INTEGER,
        timestamp TEXT,
        lat REAL,
        lon REAL,
        elapsed_s REAL,
        n_threads INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Insert or replace
    c.execute('''INSERT OR REPLACE INTO max_temperatures 
        (year, max_temp_k, max_temp_c, time_idx, cell_idx, timestamp, lat, lon, elapsed_s, n_threads)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (result['year'], result['max_temp_k'], result['max_temp_c'],
         result['time_idx'], result['cell_idx'], result['timestamp'],
         result['lat'], result['lon'], result['elapsed_s'], result['n_threads']))
    
    conn.commit()
    conn.close()
    print(f"Saved to {db_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int, help='Year to process')
    parser.add_argument('--threads', type=int, default=128, help='Number of threads')
    parser.add_argument('--db', default='/work/ab0246/a270092/software/GlobalLab_hackathon_2025/max_temps.db',
                        help='Database path')
    args = parser.parse_args()
    
    print(f"=== Processing year {args.year} with {args.threads} threads ===")
    result = find_max_temp_year(args.year, args.threads)
    save_to_db(result, args.db)
