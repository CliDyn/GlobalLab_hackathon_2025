#!/usr/bin/env python
"""
Find max temperatures near population centers for CTRL and SSP585 experiments.
Compare climate change impact on extreme temperatures by city.
"""
import intake
import time as timer
import dask
import numpy as np
import pandas as pd
import geopandas as gpd
from cartopy.io import shapereader
from shapely.geometry import box, Point
import sys

CATALOG_PATH = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/catalog/main.yaml"
OUTPUT_CSV = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/city_max_temps.csv"
MASK_FILE = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/city_mask_10m_25km.pkl"
N_THREADS = 32
CITY_RADIUS_KM = 25


def load_populated_places(use_fine=False):
    """Load Natural Earth populated places data."""
    resolution = '10m' if use_fine else '50m'
    print(f"Loading Natural Earth populated places ({resolution} resolution)...")
    
    shapefile = shapereader.natural_earth(
        resolution=resolution,
        category='cultural',
        name='populated_places'
    )
    cities_gdf = gpd.read_file(shapefile)
    print(f"Loaded {len(cities_gdf)} populated places")
    return cities_gdf


def find_nearby_cells_for_city(args):
    """Find cells near a single city (for parallel processing)."""
    city_idx, city_lat, city_lon, lats, lons, size_km = args
    
    # Convert city lon from -180/180 to 0/360 if needed
    if city_lon < 0:
        city_lon = city_lon + 360
    
    lat_deg_per_km = 1.0 / 111.0
    lon_deg_per_km = 1.0 / (111.0 * np.cos(np.radians(city_lat)))
    
    lat_radius = size_km * lat_deg_per_km
    lon_radius = size_km * lon_deg_per_km
    
    lat_mask = (lats >= city_lat - lat_radius) & (lats <= city_lat + lat_radius)
    
    # Handle wraparound at 0/360
    if city_lon - lon_radius < 0:
        lon_mask = (lons >= city_lon - lon_radius + 360) | (lons <= city_lon + lon_radius)
    elif city_lon + lon_radius > 360:
        lon_mask = (lons >= city_lon - lon_radius) | (lons <= city_lon + lon_radius - 360)
    else:
        lon_mask = (lons >= city_lon - lon_radius) & (lons <= city_lon + lon_radius)
    
    nearby_mask = lat_mask & lon_mask
    
    nearby_indices = np.where(nearby_mask)[0]
    return city_idx, nearby_indices


def create_city_mask(cities_gdf, lats, lons, size_km=25):
    """
    Create a mask of grid cells within size_km of any city.
    Uses parallel processing with 32 threads.
    
    Returns:
        - cell_indices: array of cell indices near cities
        - city_cell_map: dict mapping city index to list of nearby cell indices
    """
    from concurrent.futures import ThreadPoolExecutor
    
    print(f"Creating city mask (radius={size_km}km) with parallel processing...")
    
    # Prepare arguments for parallel processing
    args_list = [
        (idx, row.geometry.y, row.geometry.x, lats, lons, size_km)
        for idx, row in cities_gdf.iterrows()
    ]
    
    city_cell_map = {}
    all_nearby_cells = set()
    
    # Process in parallel with 32 threads
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(find_nearby_cells_for_city, args_list))
    
    for city_idx, nearby_indices in results:
        if len(nearby_indices) > 0:
            city_cell_map[city_idx] = nearby_indices
            all_nearby_cells.update(nearby_indices)
    
    cell_indices = np.array(sorted(all_nearby_cells))
    print(f"Found {len(cell_indices)} unique cells near {len(city_cell_map)} cities")
    
    return cell_indices, city_cell_map


def process_experiment(gl_cat, exp_id, exp_name, cell_indices, city_cell_map, cities_gdf):
    """Process one experiment and find max temp per city."""
    print(f"\n{'='*60}")
    print(f"Processing {exp_name} ({exp_id})")
    print('='*60)
    
    ds = gl_cat['ICCP'][exp_id]['atmos']['native']['atmos_3h'].to_dask()
    
    # Get time range
    years = ds['time_counter'].dt.year.compute()
    year_min, year_max = int(years.min()), int(years.max())
    print(f"Years: {year_min} to {year_max}")
    
    # Select only cells near cities
    print(f"Selecting {len(cell_indices)} cells near cities...")
    t2m_subset = ds['2t'].isel(cell=cell_indices)
    
    print(f"Computing max temperature across all timesteps...")
    from dask.diagnostics import ProgressBar
    with dask.config.set(scheduler='threads', num_workers=N_THREADS):
        start = timer.time()
        # Max over time for each cell
        with ProgressBar():
            max_per_cell = t2m_subset.max(dim='time_counter').compute()
        elapsed = timer.time() - start
    
    print(f"Computed in {elapsed:.1f}s")
    
    # Map back to original cell indices
    cell_to_subset_idx = {cell: i for i, cell in enumerate(cell_indices)}
    
    # Find max temp per city
    results = []
    for city_idx, city_cells in city_cell_map.items():
        # Get subset indices for this city's cells
        subset_indices = [cell_to_subset_idx[c] for c in city_cells if c in cell_to_subset_idx]
        
        if len(subset_indices) > 0:
            city_max = float(max_per_cell.isel(cell=subset_indices).max())
            city_max_c = city_max - 273.15
            
            row = cities_gdf.loc[city_idx]
            results.append({
                'city_idx': city_idx,
                'city_name': row.get('NAME', row.get('name', 'Unknown')),
                'country': row.get('ADM0NAME', row.get('SOV0NAME', 'Unknown')),
                'lat': row.geometry.y,
                'lon': row.geometry.x,
                'pop_est': row.get('POP_MAX', row.get('pop_max', 0)),
                f'max_temp_{exp_name}_K': city_max,
                f'max_temp_{exp_name}_C': city_max_c,
            })
    
    print(f"Found max temps for {len(results)} cities")
    return pd.DataFrame(results)


def load_city_mask():
    """Load pre-computed city mask from file."""
    import pickle
    print(f"Loading city mask from {MASK_FILE}...")
    with open(MASK_FILE, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded: {len(data['city_cell_map'])} cities, {len(data['cell_indices'])} cells")
    return data['cell_indices'], data['city_cell_map'], data['cities_gdf']


def main():
    total_start = timer.time()
    
    # Load catalog
    gl_cat = intake.open_catalog(CATALOG_PATH)
    
    # Load pre-computed city mask
    cell_indices, city_cell_map, cities_gdf = load_city_mask()
    
    # Process both experiments
    experiments = [
        ('TCo1279-DART-1950C', 'CTRL'),
        ('TCo1279-DART-2080C', 'SSP585'),
    ]
    
    dfs = []
    for exp_id, exp_name in experiments:
        df = process_experiment(gl_cat, exp_id, exp_name, cell_indices, city_cell_map, cities_gdf)
        dfs.append(df)
    
    # Merge results
    print("\n" + "="*60)
    print("Merging results...")
    
    # Merge on city info columns
    merge_cols = ['city_idx', 'city_name', 'country', 'lat', 'lon', 'pop_est']
    result_df = dfs[0][merge_cols + ['max_temp_CTRL_K', 'max_temp_CTRL_C']].merge(
        dfs[1][['city_idx', 'max_temp_SSP585_K', 'max_temp_SSP585_C']],
        on='city_idx'
    )
    
    # Calculate change
    result_df['temp_change_C'] = result_df['max_temp_SSP585_C'] - result_df['max_temp_CTRL_C']
    
    # Sort by change (largest increase first, then largest drop)
    result_df = result_df.sort_values('temp_change_C', ascending=False)
    
    # Save to CSV
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")
    
    # Print summary
    print("\n" + "="*60)
    print("TOP 10 LARGEST TEMPERATURE INCREASES:")
    print("="*60)
    print(result_df[['city_name', 'country', 'max_temp_CTRL_C', 'max_temp_SSP585_C', 'temp_change_C']].head(10).to_string())
    
    print("\n" + "="*60)
    print("TOP 10 LARGEST TEMPERATURE DECREASES:")
    print("="*60)
    print(result_df[['city_name', 'country', 'max_temp_CTRL_C', 'max_temp_SSP585_C', 'temp_change_C']].tail(10).to_string())
    
    total_elapsed = timer.time() - total_start
    print(f"\n{'='*60}")
    print(f"COMPLETE in {total_elapsed/60:.1f} min")
    print(f"Cities analyzed: {len(result_df)}")
    print(f"Average change: {result_df['temp_change_C'].mean():.2f}°C")
    print(f"Max increase: {result_df['temp_change_C'].max():.2f}°C")
    print(f"Max decrease: {result_df['temp_change_C'].min():.2f}°C")
    print('='*60)


if __name__ == '__main__':
    main()
