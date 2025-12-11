#!/usr/bin/env python
"""
Diagnostic script to understand city temp processing bottlenecks.
"""
import intake
import time as timer
import numpy as np
import geopandas as gpd
from cartopy.io import shapereader
import dask
import argparse

dask.config.set(scheduler='synchronous')

CATALOG_PATH = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/catalog/main.yaml"


def timed(name):
    """Context manager for timing blocks."""
    class Timer:
        def __enter__(self):
            self.start = timer.time()
            print(f"[START] {name}...", flush=True)
            return self
        def __exit__(self, *args):
            elapsed = timer.time() - self.start
            print(f"[DONE]  {name}: {elapsed:.1f}s", flush=True)
    return Timer()


def get_radius_km(population):
    if population >= 10_000_000: return 50
    elif population >= 1_000_000: return 25
    elif population >= 100_000: return 15
    else: return 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=3, help='Number of cities')
    args = parser.parse_args()
    
    print(f"\n{'='*60}", flush=True)
    print(f"TIMING TEST: {args.top} cities", flush=True)
    print('='*60, flush=True)
    
    # 1. Load catalog
    with timed("Load catalog"):
        gl_cat = intake.open_catalog(CATALOG_PATH)
    
    # 2. Load cities
    with timed(f"Load {args.top} cities"):
        sf = shapereader.natural_earth(resolution='50m', category='cultural', name='populated_places')
        cities = gpd.read_file(sf)
        cities['pop'] = cities['POP_MAX'].fillna(0).astype(int)
        cities = cities.nlargest(args.top, 'pop')
    
    # 3. Open dataset (lazy)
    with timed("Open dataset (lazy)"):
        ds = gl_cat['ICCP']['TCo1279-DART-1950C']['atmos']['native']['atmos_3h'].to_dask()
    
    # 4. Load grid coordinates
    with timed("Load grid coordinates"):
        lats = ds['lat'].values
        lons = ds['lon'].values
    print(f"    Grid: {len(lats):,} cells", flush=True)
    
    # 5. Find city cells
    with timed("Find city cells"):
        all_cells = set()
        for idx, row in cities.iterrows():
            city_lat, city_lon = row.geometry.y, row.geometry.x
            if city_lon < 0: city_lon += 360
            radius_km = get_radius_km(row['pop'])
            lat_r = radius_km / 111.0
            lon_r = radius_km / (111.0 * np.cos(np.radians(city_lat)))
            mask = ((lats >= city_lat - lat_r) & (lats <= city_lat + lat_r) & 
                    (lons >= city_lon - lon_r) & (lons <= city_lon + lon_r))
            all_cells.update(np.where(mask)[0])
        cell_indices = np.array(sorted(all_cells))
    print(f"    Cells: {len(cell_indices):,}", flush=True)
    
    # 6. Select data (lazy)
    with timed("Select data (lazy isel)"):
        t2m = ds['2t'].isel(cell=cell_indices)
    print(f"    Shape: {t2m.shape}", flush=True)
    print(f"    Size: {t2m.nbytes/1e9:.2f} GB", flush=True)
    print(f"    Chunks: {t2m.chunks}", flush=True)
    
    # 7. Load data - THE SLOW PART
    print("\n--- DATA LOADING (the slow part) ---", flush=True)
    
    # Try loading just 1 year first
    with timed("Load 1 year (2920 timesteps)"):
        data_1yr = t2m.isel(time_counter=slice(0, 2920)).values
    print(f"    1 year shape: {data_1yr.shape}", flush=True)
    
    # Try loading 1 timestep
    with timed("Load 1 timestep"):
        data_1t = t2m.isel(time_counter=0).values
    print(f"    1 timestep shape: {data_1t.shape}", flush=True)
    
    # 8. Compute max (should be fast once data is loaded)
    with timed("Compute max on 1 year"):
        max_val = data_1yr.max()
    print(f"    Max: {max_val - 273.15:.1f}°C", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("SUMMARY: Most time should be in 'Load data' step", flush=True)
    print("If 1-year load is slow, kerchunk access pattern is the bottleneck", flush=True)
    print("="*60, flush=True)


if __name__ == '__main__':
    main()
