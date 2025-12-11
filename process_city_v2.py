#!/usr/bin/env python
"""
Find max values near populated places - single experiment version.
Usage:
  python process_city_v2.py --var temp --exp ctrl --all
  python process_city_v2.py --var temp --exp ssp585 --all
"""
import intake
import time as timer
import numpy as np
import pandas as pd
import geopandas as gpd
from cartopy.io import shapereader
from tqdm import tqdm
import argparse

CATALOG_PATH = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/catalog/main.yaml"
OUTPUT_DIR = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025"

EXPERIMENTS = {
    'ctrl': ('TCo1279-DART-1950C', 'CTRL (1950-1970)'),
    'ssp585': ('TCo1279-DART-2080C', 'SSP585 (2080-2100)'),
}

VAR_CONFIG = {
    'temp': {'unit': 'C', 'unit_raw': 'K', 'convert': lambda x: x - 273.15, 'label': 'Temperature'},
    'precip': {'unit': 'mm', 'unit_raw': 'm', 'convert': lambda x: x * 1000, 'label': 'Precipitation'},
    'wind': {'unit': 'km/h', 'unit_raw': 'm/s', 'convert': lambda x: x * 3.6, 'label': 'Wind Speed'},
}


def get_radius_km(population):
    if population >= 10_000_000: return 50
    elif population >= 1_000_000: return 25
    elif population >= 100_000: return 15
    else: return 8


def load_cities(n_cities=50):
    resolution = '10m' if n_cities == 0 else '50m'
    print(f"Loading populated places ({resolution})...", flush=True)
    shapefile = shapereader.natural_earth(resolution=resolution, category='cultural', name='populated_places')
    cities_gdf = gpd.read_file(shapefile)
    pop_col = 'POP_MAX' if 'POP_MAX' in cities_gdf.columns else 'pop_max'
    cities_gdf['population'] = cities_gdf[pop_col].fillna(0).astype(int)
    if n_cities > 0:
        cities_gdf = cities_gdf.nlargest(n_cities, 'population')
    print(f"  {len(cities_gdf)} cities loaded", flush=True)
    return cities_gdf


def find_city_cells(cities_gdf, lats, lons):
    print("Finding cells with population-scaled radius...", flush=True)
    all_cells = set()
    city_cell_map = {}
    radii = []
    for idx, row in tqdm(cities_gdf.iterrows(), total=len(cities_gdf), desc="Mapping"):
        city_lat, city_lon = row.geometry.y, row.geometry.x
        if city_lon < 0: city_lon += 360
        radius_km = get_radius_km(row['population'])
        radii.append(radius_km)
        lat_r = radius_km / 111.0
        lon_r = radius_km / (111.0 * np.cos(np.radians(city_lat)))
        mask = ((lats >= city_lat - lat_r) & (lats <= city_lat + lat_r) & 
                (lons >= city_lon - lon_r) & (lons <= city_lon + lon_r))
        nearby = np.where(mask)[0]
        if len(nearby) > 0:
            city_cell_map[idx] = nearby
            all_cells.update(nearby)
    cell_indices = np.array(sorted(all_cells))
    print(f"  Radius: {min(radii)}-{max(radii)}km (avg {np.mean(radii):.1f}km)", flush=True)
    print(f"  {len(cell_indices):,} unique cells", flush=True)
    return cell_indices, city_cell_map


def load_chunk(ds, t_start, t_end, cell_indices, var_type):
    """Load one chunk of data with retry logic. Returns None if all retries fail."""
    for attempt in range(3):
        try:
            if var_type == 'temp':
                data_full = ds['2t'].isel(time_counter=slice(t_start, t_end)).values
                data = data_full[:, cell_indices]
                del data_full
            elif var_type == 'precip':
                cp_full = ds['cp'].isel(time_counter=slice(t_start, t_end)).values
                cp_data = cp_full[:, cell_indices]
                del cp_full
                lsp_full = ds['lsp'].isel(time_counter=slice(t_start, t_end)).values
                lsp_data = lsp_full[:, cell_indices]
                del lsp_full
                data = cp_data + lsp_data
                del cp_data, lsp_data
            elif var_type == 'wind':
                u_full = ds['10u'].isel(time_counter=slice(t_start, t_end)).values
                u_data = u_full[:, cell_indices]
                del u_full
                v_full = ds['10v'].isel(time_counter=slice(t_start, t_end)).values
                v_data = v_full[:, cell_indices]
                del v_full
                data = np.sqrt(u_data**2 + v_data**2)
                del u_data, v_data
            return data
        except Exception as e:
            if attempt < 2:
                timer.sleep(2)
            else:
                return None
    return None


def process_experiment(ds, exp_name, cell_indices, var_type):
    """Load data month-by-month, filter to city cells in numpy."""
    print(f"\n{'='*60}", flush=True)
    print(f"Processing {exp_name} - {var_type}", flush=True)
    print('='*60, flush=True)
    
    n_times = ds.dims['time_counter']
    # ~248 timesteps per month (31 days × 8 per day)
    MONTH_SIZE = 248
    n_months = (n_times + MONTH_SIZE - 1) // MONTH_SIZE
    print(f"  {n_times} timesteps, {n_months} months, {len(cell_indices)} cells", flush=True)
    
    max_per_cell = None
    argmax_per_cell = None
    time_offset = 0
    skipped = 0
    
    for month_idx in tqdm(range(n_months), desc=f"  {exp_name[:4]}"):
        t_start = month_idx * MONTH_SIZE
        t_end = min(t_start + MONTH_SIZE, n_times)
        
        data = load_chunk(ds, t_start, t_end, cell_indices, var_type)
        
        if data is None:
            tqdm.write(f"  WARNING: Month {month_idx} failed, skipping...")
            skipped += 1
            time_offset += (t_end - t_start)
            continue
        
        month_max = data.max(axis=0)
        month_argmax = data.argmax(axis=0) + time_offset
        
        if max_per_cell is None:
            max_per_cell = month_max
            argmax_per_cell = month_argmax
        else:
            better = month_max > max_per_cell
            max_per_cell = np.where(better, month_max, max_per_cell)
            argmax_per_cell = np.where(better, month_argmax, argmax_per_cell)
        
        time_offset += (t_end - t_start)
        del data
    
    if skipped > 0:
        print(f"  Skipped {skipped} months due to errors", flush=True)
    
    timestamps = ds['time_counter'].values
    return {'max_values': max_per_cell, 'time_indices': argmax_per_cell, 
            'timestamps': timestamps, 'cell_indices': cell_indices}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--var', choices=['temp', 'precip', 'wind'], default='temp')
    parser.add_argument('--exp', choices=['ctrl', 'ssp585'], required=True)
    parser.add_argument('--top', type=int, default=50)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    
    var_type = args.var
    var_cfg = VAR_CONFIG[var_type]
    exp_key = args.exp
    exp_dataset, exp_name = EXPERIMENTS[exp_key]
    n_cities = 0 if args.all else args.top
    
    print(f"\n*** {var_cfg['label']} - {exp_name} ***\n", flush=True)
    total_start = timer.time()
    
    # Load
    gl_cat = intake.open_catalog(CATALOG_PATH)
    cities_gdf = load_cities(n_cities=n_cities)
    
    print("\nLoading grid...", flush=True)
    ds = gl_cat['ICCP'][exp_dataset]['atmos']['native']['atmos_3h'].to_dask()
    lats, lons = ds['lat'].values, ds['lon'].values
    
    cell_indices, city_cell_map = find_city_cells(cities_gdf, lats, lons)
    result = process_experiment(ds, exp_name, cell_indices, var_type)
    
    # Extract max per city
    print("\nExtracting max per city...", flush=True)
    cell_to_idx = {c: i for i, c in enumerate(cell_indices)}
    
    rows = []
    for city_idx, city_cells in city_cell_map.items():
        subset_idx = np.array([cell_to_idx[c] for c in city_cells])
        local_max_idx = np.argmax(result['max_values'][subset_idx])
        best_idx = subset_idx[local_max_idx]
        cell_idx = cell_indices[best_idx]
        time_idx = result['time_indices'][best_idx]
        max_val = result['max_values'][best_idx]
        timestamp = str(result['timestamps'][time_idx])
        
        row = cities_gdf.loc[city_idx]
        val_converted = var_cfg['convert'](max_val)
        
        rows.append({
            'city_name': row.get('NAME', row.get('name', 'Unknown')),
            'country': row.get('ADM0NAME', row.get('SOV0NAME', 'Unknown')),
            'city_lat': row.geometry.y,
            'city_lon': row.geometry.x,
            'population': row['population'],
            f'max_{var_type}_raw': float(max_val),
            f'max_{var_type}': float(val_converted),
            'cell_idx': int(cell_idx),
            'time_idx': int(time_idx),
            'timestamp': timestamp,
            'lat': float(lats[cell_idx]),
            'lon': float(lons[cell_idx]),
        })
    
    df = pd.DataFrame(rows).sort_values(f'max_{var_type}', ascending=False)
    
    suffix = 'all' if args.all else f'top{n_cities}'
    output_csv = f"{OUTPUT_DIR}/city_max_{var_type}_{exp_key}_{suffix}.csv"
    df.to_csv(output_csv, index=False)
    
    elapsed = timer.time() - total_start
    print(f"\n{'='*60}", flush=True)
    print(f"COMPLETE: {exp_name} in {elapsed/60:.1f} min", flush=True)
    print(f"Saved: {output_csv}", flush=True)
    print(f"Cities: {len(df)}, Max {var_type}: {df[f'max_{var_type}'].max():.1f} {var_cfg['unit']}", flush=True)
    print('='*60, flush=True)


if __name__ == '__main__':
    main()
