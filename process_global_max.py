#!/usr/bin/env python
"""
Find global max per year for a single experiment.
Usage:
  python process_global_max.py --var temp --exp ctrl
  python process_global_max.py --var precip --exp ssp585
  python process_global_max.py --var wind --exp ctrl
"""
import intake
import time as timer
import dask
import sqlite3
import numpy as np
import argparse

CATALOG_PATH = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/catalog/main.yaml"
OUTPUT_DIR = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025"
N_THREADS = 32

EXPERIMENTS = {
    'ctrl': ('TCo1279-DART-1950C', 'CTRL', range(1950, 1971)),
    'ssp585': ('TCo1279-DART-2080C', 'SSP585', range(2080, 2101)),
}

VAR_CONFIG = {
    'temp': {
        'table': 'max_temperatures',
        'vars': ['2t'],
        'compute': lambda ds, t0, t1: ds['2t'].isel(time_counter=slice(t0, t1)),
        'format': lambda v: f"{v-273.15:.2f}°C ({v:.2f}K)",
    },
    'precip': {
        'table': 'max_precipitation', 
        'vars': ['cp', 'lsp'],
        'compute': lambda ds, t0, t1: ds['cp'].isel(time_counter=slice(t0, t1)) + ds['lsp'].isel(time_counter=slice(t0, t1)),
        'format': lambda v: f"{v*1000:.1f}mm",
    },
    'wind': {
        'table': 'max_wind',
        'vars': ['10u', '10v'],
        'compute': lambda ds, t0, t1: np.sqrt(ds['10u'].isel(time_counter=slice(t0, t1))**2 + ds['10v'].isel(time_counter=slice(t0, t1))**2),
        'format': lambda v: f"{v:.1f} m/s ({v*3.6:.1f} km/h)",
    },
}


def init_db(db_path, table_name, var_type):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(f'''CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment TEXT,
        year INTEGER,
        max_value REAL,
        time_idx INTEGER,
        cell_idx INTEGER,
        timestamp TEXT,
        lat REAL,
        lon REAL,
        elapsed_s REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(experiment, year)
    )''')
    conn.commit()
    conn.close()


def load_chunk_with_retry(ds, t_start, t_end, var_cfg):
    """Load a chunk with retry. Returns None on failure."""
    for attempt in range(3):
        try:
            return var_cfg['compute'](ds, t_start, t_end).values
        except Exception as e:
            if attempt < 2:
                timer.sleep(2)
    return None


def process_year(ds, year_idx, year, lats, lons, var_cfg):
    """Process one year of data, falling back to month-by-month if year fails."""
    t_start = year_idx * 2920
    t_end = min(t_start + 2920, ds.dims['time_counter'])
    n_timesteps = t_end - t_start
    
    print(f"  Year {year}: {n_timesteps} timesteps", end=" ", flush=True)
    start = timer.time()
    
    # Try loading whole year first
    data = load_chunk_with_retry(ds, t_start, t_end, var_cfg)
    
    if data is None:
        # Fall back to month-by-month loading
        print("(month-by-month)", end=" ", flush=True)
        MONTH_SIZE = 248
        max_val = -np.inf
        best_time_idx = 0
        best_cell_idx = 0
        skipped = 0
        
        for m in range(12):
            m_start = t_start + m * MONTH_SIZE
            m_end = min(m_start + MONTH_SIZE, t_end)
            if m_start >= t_end:
                break
            
            month_data = load_chunk_with_retry(ds, m_start, m_end, var_cfg)
            if month_data is None:
                skipped += 1
                continue
            
            month_max = float(month_data.max())
            if month_max > max_val:
                max_val = month_max
                flat_idx = int(month_data.argmax())
                best_time_idx = m_start + flat_idx // month_data.shape[1]
                best_cell_idx = flat_idx % month_data.shape[1]
            del month_data
        
        if max_val == -np.inf:
            print(f"FAILED (all months)", flush=True)
            return None
        
        if skipped > 0:
            print(f"(skipped {skipped} months)", end=" ", flush=True)
        
        time_idx = best_time_idx
        cell_idx = best_cell_idx
    else:
        # Year loaded successfully
        max_val = float(data.max())
        flat_idx = int(data.argmax())
        time_idx = t_start + flat_idx // data.shape[1]
        cell_idx = flat_idx % data.shape[1]
        del data
    
    # Get timestamp and location
    timestamp = ds['time_counter'].isel(time_counter=time_idx).values
    lat, lon = float(lats[cell_idx]), float(lons[cell_idx])
    
    elapsed = timer.time() - start
    print(f"-> {var_cfg['format'](max_val)} at ({lat:.1f}°N, {lon:.1f}°E) [t={time_idx}, c={cell_idx}] [{elapsed:.0f}s]", flush=True)
    
    return {
        'max_value': max_val,
        'time_idx': time_idx,
        'cell_idx': cell_idx,
        'timestamp': str(timestamp),
        'lat': lat,
        'lon': lon,
        'elapsed_s': elapsed,
    }


def save_result(db_path, table_name, experiment, year, result):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(f'''INSERT OR REPLACE INTO {table_name} 
        (experiment, year, max_value, time_idx, cell_idx, timestamp, lat, lon, elapsed_s)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (experiment, year, result['max_value'], result['time_idx'], result['cell_idx'],
         result['timestamp'], result['lat'], result['lon'], result['elapsed_s']))
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--var', choices=['temp', 'precip', 'wind'], required=True)
    parser.add_argument('--exp', choices=['ctrl', 'ssp585'], required=True)
    args = parser.parse_args()
    
    var_type = args.var
    var_cfg = VAR_CONFIG[var_type]
    exp_dataset, exp_name, years = EXPERIMENTS[args.exp]
    
    db_path = f"{OUTPUT_DIR}/max_{var_type}_{args.exp}.db"
    table_name = var_cfg['table']
    
    print(f"\n*** Global Max {var_type.upper()} - {exp_name} ***", flush=True)
    print(f"Database: {db_path}\n", flush=True)
    
    init_db(db_path, table_name, var_type)
    
    # Load dataset
    gl_cat = intake.open_catalog(CATALOG_PATH)
    ds = gl_cat['ICCP'][exp_dataset]['atmos']['native']['atmos_3h'].to_dask()
    lats = ds['lat'].values
    lons = ds['lon'].values
    
    print(f"Processing {exp_name}: {len(list(years))} years\n", flush=True)
    
    total_start = timer.time()
    for year_idx, year in enumerate(years):
        result = process_year(ds, year_idx, year, lats, lons, var_cfg)
        if result:
            save_result(db_path, table_name, exp_name, year, result)
    
    elapsed = timer.time() - total_start
    print(f"\n{'='*60}", flush=True)
    print(f"COMPLETE: {exp_name} in {elapsed/60:.1f} min", flush=True)
    print(f"Saved to: {db_path}", flush=True)
    print('='*60, flush=True)


if __name__ == '__main__':
    main()
