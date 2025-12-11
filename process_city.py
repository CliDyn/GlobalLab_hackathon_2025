#!/usr/bin/env python
"""
Find max values near populated places for CTRL vs SSP585.
Supports: temp (2t), precip (cp+lsp), wind (sqrt(10u²+10v²))

Usage:
  python process_city.py --var temp              # 2m temperature (default)
  python process_city.py --var precip            # precipitation
  python process_city.py --var wind              # 10m wind magnitude
  python process_city.py --var temp --all        # all 7342 cities
  python process_city.py --var temp --top 50     # top 50 cities
"""
import intake
import time as timer
import numpy as np
import pandas as pd
import geopandas as gpd
from cartopy.io import shapereader
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import dask
import argparse

# Use synchronous scheduler to avoid parallel decompression issues
dask.config.set(scheduler='synchronous')

CATALOG_PATH = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/catalog/main.yaml"
OUTPUT_DIR = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025"


def get_radius_km(population):
    """Scale radius with city population."""
    if population >= 10_000_000:
        return 50  # Megacities (19 cities)
    elif population >= 1_000_000:
        return 25  # Large cities (486 cities)
    elif population >= 100_000:
        return 15  # Medium cities (2584 cities)
    else:
        return 8   # Small cities (4253 cities)


def load_cities(n_cities=50):
    """Load populated places. Top N by default, all with n_cities=0."""
    resolution = '10m' if n_cities == 0 else '50m'
    print(f"Loading populated places ({resolution})...", flush=True)
    
    shapefile = shapereader.natural_earth(resolution=resolution, category='cultural', name='populated_places')
    cities_gdf = gpd.read_file(shapefile)
    
    pop_col = 'POP_MAX' if 'POP_MAX' in cities_gdf.columns else 'pop_max'
    cities_gdf['population'] = cities_gdf[pop_col].fillna(0).astype(int)
    
    if n_cities > 0:
        cities_gdf = cities_gdf.nlargest(n_cities, 'population')
    
    cities_gdf = cities_gdf.reset_index(drop=True)
    print(f"  {len(cities_gdf)} cities, pop range: {cities_gdf['population'].min():,} - {cities_gdf['population'].max():,}", flush=True)
    return cities_gdf


def find_city_cells(cities_gdf, lats, lons):
    """Find grid cells near each city with population-scaled radius."""
    print(f"Finding cells with population-scaled radius...", flush=True)
    
    all_cells = set()
    city_cell_map = {}
    radii = []
    
    for idx, row in tqdm(cities_gdf.iterrows(), total=len(cities_gdf), desc="Mapping"):
        city_lat, city_lon = row.geometry.y, row.geometry.x
        if city_lon < 0:
            city_lon += 360
        
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
    mem_gb = len(cell_indices) * 58440 * 4 / 1e9
    print(f"  Radius: {min(radii)}-{max(radii)}km (avg {np.mean(radii):.1f}km)", flush=True)
    print(f"  {len(cell_indices):,} unique cells, ~{mem_gb:.1f} GB to load", flush=True)
    return cell_indices, city_cell_map


def process_experiment(ds, exp_name, cell_indices, var_type):
    """Load data year-by-year (full slices), filter to city cells in numpy."""
    print(f"\n{'='*60}", flush=True)
    print(f"Processing {exp_name} - {var_type}", flush=True)
    print('='*60, flush=True)
    
    n_times = ds.dims['time_counter']
    n_years = (n_times + 2919) // 2920  # ~2920 timesteps per year
    print(f"  {n_times} timesteps, {n_years} years, {len(cell_indices)} cells", flush=True)
    
    max_per_cell = None
    argmax_per_cell = None
    time_offset = 0
    
    for year_idx in tqdm(range(n_years), desc=f"  {exp_name[:4]}"):
        t_start = year_idx * 2920
        t_end = min(t_start + 2920, n_times)
        
        # Load FULL slice (all cells), then filter - avoids kerchunk cell selection bug
        if var_type == 'temp':
            data_full = ds['2t'].isel(time_counter=slice(t_start, t_end)).values
            data = data_full[:, cell_indices]
            del data_full  # Free 77GB immediately
        elif var_type == 'precip':
            cp_full = ds['cp'].isel(time_counter=slice(t_start, t_end)).values
            cp_data = cp_full[:, cell_indices]
            del cp_full  # Free 77GB immediately
            lsp_full = ds['lsp'].isel(time_counter=slice(t_start, t_end)).values
            lsp_data = lsp_full[:, cell_indices]
            del lsp_full  # Free 77GB immediately
            data = cp_data + lsp_data
            del cp_data, lsp_data
        elif var_type == 'wind':
            u_full = ds['10u'].isel(time_counter=slice(t_start, t_end)).values
            u_data = u_full[:, cell_indices]
            del u_full  # Free 77GB immediately
            v_full = ds['10v'].isel(time_counter=slice(t_start, t_end)).values
            v_data = v_full[:, cell_indices]
            del v_full  # Free 77GB immediately
            data = np.sqrt(u_data**2 + v_data**2)
            del u_data, v_data
        
        # Track max and when it occurred
        year_max = data.max(axis=0)
        year_argmax = data.argmax(axis=0) + time_offset
        
        if max_per_cell is None:
            max_per_cell = year_max
            argmax_per_cell = year_argmax
        else:
            better = year_max > max_per_cell
            max_per_cell = np.where(better, year_max, max_per_cell)
            argmax_per_cell = np.where(better, year_argmax, argmax_per_cell)
        
        time_offset += (t_end - t_start)
    
    # Load timestamps for later lookup
    print("  Loading timestamps...", flush=True)
    timestamps = ds['time_counter'].values
    
    return {
        'max_values': max_per_cell,
        'time_indices': argmax_per_cell,
        'timestamps': timestamps,
        'cell_indices': cell_indices,
    }


# Variable configuration
VAR_CONFIG = {
    'temp': {'unit': 'C', 'unit_raw': 'K', 'convert': lambda x: x - 273.15, 'label': 'Temperature'},
    'precip': {'unit': 'mm', 'unit_raw': 'm', 'convert': lambda x: x * 1000, 'label': 'Precipitation'},
    'wind': {'unit': 'km/h', 'unit_raw': 'm/s', 'convert': lambda x: x * 3.6, 'label': 'Wind Speed'},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--var', choices=['temp', 'precip', 'wind'], default='temp', help='Variable to process')
    parser.add_argument('--top', type=int, default=50, help='Number of top cities (default 50)')
    parser.add_argument('--all', action='store_true', help='Use all 7342 cities (needs ~14GB RAM)')
    args = parser.parse_args()
    
    var_type = args.var
    var_cfg = VAR_CONFIG[var_type]
    n_cities = 0 if args.all else args.top
    
    print(f"\n*** Processing {var_cfg['label']} for cities ***\n", flush=True)
    
    total_start = timer.time()
    
    # Load catalog and cities
    gl_cat = intake.open_catalog(CATALOG_PATH)
    cities_gdf = load_cities(n_cities=n_cities)
    
    # Load grid
    print("\nLoading TCo1279 grid...", flush=True)
    ds_ctrl = gl_cat['ICCP']['TCo1279-DART-1950C']['atmos']['native']['atmos_3h'].to_dask()
    lats, lons = ds_ctrl['lat'].values, ds_ctrl['lon'].values
    print(f"  {len(lats):,} cells", flush=True)
    
    # Find city cells (radius scaled by population)
    cell_indices, city_cell_map = find_city_cells(cities_gdf, lats, lons)
    
    # Process experiments
    result_ctrl = process_experiment(ds_ctrl, 'CTRL (1950-1970)', cell_indices, var_type)
    
    ds_ssp = gl_cat['ICCP']['TCo1279-DART-2080C']['atmos']['native']['atmos_3h'].to_dask()
    result_ssp = process_experiment(ds_ssp, 'SSP585 (2080-2100)', cell_indices, var_type)
    
    # Extract max per city with full traceability
    print("\nExtracting max per city...", flush=True)
    cell_to_idx = {c: i for i, c in enumerate(cell_indices)}
    
    results = []
    for city_idx, city_cells in city_cell_map.items():
        subset_idx = np.array([cell_to_idx[c] for c in city_cells])
        
        # CTRL: find which cell has max and get its details
        ctrl_local_max_idx = np.argmax(result_ctrl['max_values'][subset_idx])
        ctrl_best_subset_idx = subset_idx[ctrl_local_max_idx]
        ctrl_cell_idx = cell_indices[ctrl_best_subset_idx]  # Original grid index
        ctrl_time_idx = result_ctrl['time_indices'][ctrl_best_subset_idx]
        ctrl_max_val = result_ctrl['max_values'][ctrl_best_subset_idx]
        ctrl_timestamp = str(result_ctrl['timestamps'][ctrl_time_idx])
        
        # SSP585: find which cell has max and get its details
        ssp_local_max_idx = np.argmax(result_ssp['max_values'][subset_idx])
        ssp_best_subset_idx = subset_idx[ssp_local_max_idx]
        ssp_cell_idx = cell_indices[ssp_best_subset_idx]  # Original grid index
        ssp_time_idx = result_ssp['time_indices'][ssp_best_subset_idx]
        ssp_max_val = result_ssp['max_values'][ssp_best_subset_idx]
        ssp_timestamp = str(result_ssp['timestamps'][ssp_time_idx])
        
        row = cities_gdf.loc[city_idx]
        
        # Convert values
        ctrl_val_converted = var_cfg['convert'](ctrl_max_val)
        ssp_val_converted = var_cfg['convert'](ssp_max_val)
        change = ssp_val_converted - ctrl_val_converted
        
        results.append({
            'city_name': row.get('NAME', row.get('name', 'Unknown')),
            'country': row.get('ADM0NAME', row.get('SOV0NAME', 'Unknown')),
            'city_lat': row.geometry.y,
            'city_lon': row.geometry.x,
            'population': row['population'],
            'radius_km': get_radius_km(row['population']),
            # CTRL results
            f'max_{var_type}_CTRL_raw': float(ctrl_max_val),
            f'max_{var_type}_CTRL': float(ctrl_val_converted),
            'CTRL_cell_idx': int(ctrl_cell_idx),
            'CTRL_time_idx': int(ctrl_time_idx),
            'CTRL_timestamp': ctrl_timestamp,
            'CTRL_lat': float(lats[ctrl_cell_idx]),
            'CTRL_lon': float(lons[ctrl_cell_idx]),
            # SSP585 results
            f'max_{var_type}_SSP585_raw': float(ssp_max_val),
            f'max_{var_type}_SSP585': float(ssp_val_converted),
            'SSP585_cell_idx': int(ssp_cell_idx),
            'SSP585_time_idx': int(ssp_time_idx),
            'SSP585_timestamp': ssp_timestamp,
            'SSP585_lat': float(lats[ssp_cell_idx]),
            'SSP585_lon': float(lons[ssp_cell_idx]),
            # Change
            f'{var_type}_change': float(change),
        })
    
    change_col = f'{var_type}_change'
    result_df = pd.DataFrame(results).sort_values(change_col, ascending=False)
    
    # Save
    suffix = 'all' if args.all else f'top{n_cities}'
    output_csv = f"{OUTPUT_DIR}/city_max_{var_type}_{suffix}.csv"
    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}", flush=True)
    
    # Summary
    ctrl_col = f'max_{var_type}_CTRL'
    ssp_col = f'max_{var_type}_SSP585'
    
    print("\n" + "="*60, flush=True)
    print(f"TOP 10 LARGEST {var_cfg['label'].upper()} INCREASES:", flush=True)
    print("="*60, flush=True)
    cols = ['city_name', 'country', ctrl_col, ssp_col, change_col]
    print(result_df[cols].head(10).to_string(index=False))
    
    print("\n" + "="*60, flush=True)
    print("TOP 10 SMALLEST CHANGES:", flush=True)
    print("="*60, flush=True)
    print(result_df[cols].tail(10).to_string(index=False))
    
    total_elapsed = timer.time() - total_start
    print(f"\n{'='*60}", flush=True)
    print(f"COMPLETE in {total_elapsed/60:.1f} min", flush=True)
    print(f"Cities: {len(result_df)}, Avg change: {result_df[change_col].mean():.2f} {var_cfg['unit']}", flush=True)
    print('='*60, flush=True)


if __name__ == '__main__':
    main()
