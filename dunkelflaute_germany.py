#!/usr/bin/env python3
"""
Compute longest Dunkelflaute events over Germany.

Dunkelflaute = sustained periods with low combined wind + solar power generation.

Based on literature (Kittel & Schill 2024, Li et al.):
- Use MOVING AVERAGE over minimum duration (not instantaneous values)
- Compute wind capacity factor using simplified power curve
- Compute solar capacity factor using cloud cover as proxy
- Threshold: Moving average of VRE CF < 20% of long-term mean

Key insight: Duration matters! A 3-hour dip is not a Dunkelflaute.
We use moving averages over different windows (12h, 24h, 48h) to identify
sustained low-VRE periods.
"""

import intake
import numpy as np
import xarray as xr
from pathlib import Path
import argparse
from scipy.ndimage import uniform_filter1d

# Germany bounding box (approximate)
GERMANY_BOUNDS = {
    'lon_min': 5.8,
    'lon_max': 15.1,
    'lat_min': 47.2,
    'lat_max': 55.1
}

# Dunkelflaute thresholds (absolute capacity factors, as in Li et al.)
# Literature: both wind AND solar below 20% of capacity
WIND_CF_THRESHOLD = 0.10     # Wind CF below 10%
SOLAR_CF_THRESHOLD = 0.05   # Solar CF below 5% (accounting for night)

# Minimum event durations to analyze (in hours)
MIN_DURATIONS = [12, 24, 48, 72]

# Wind turbine parameters (for capacity factor calculation)
WIND_CUT_IN = 3.0    # m/s
WIND_RATED = 12.0    # m/s  
WIND_CUT_OUT = 25.0  # m/s

CATALOG_PATH = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/catalog/main.yaml"
GRID_FILE = "/work/ab0246/a270092/input/oasis/cy43r3/TCO1279-DART/grids.nc"


def wind_capacity_factor(wind_speed):
    """
    Convert wind speed to capacity factor using simplified power curve.
    
    - Below cut-in (3 m/s): CF = 0
    - Between cut-in and rated: cubic relationship
    - Above rated (12 m/s): CF = 1
    - Above cut-out (25 m/s): CF = 0
    """
    cf = np.zeros_like(wind_speed)
    
    # Cubic region between cut-in and rated
    mask_cubic = (wind_speed >= WIND_CUT_IN) & (wind_speed < WIND_RATED)
    cf[mask_cubic] = ((wind_speed[mask_cubic] - WIND_CUT_IN) / (WIND_RATED - WIND_CUT_IN)) ** 3
    
    # Rated region
    mask_rated = (wind_speed >= WIND_RATED) & (wind_speed < WIND_CUT_OUT)
    cf[mask_rated] = 1.0
    
    # Above cut-out: shutdown
    cf[wind_speed >= WIND_CUT_OUT] = 0.0
    
    return cf


def solar_capacity_factor(cloud_cover, hour_of_day):
    """
    Estimate solar capacity factor from cloud cover.
    
    - Base clear-sky CF depends on hour (0 at night, peak ~0.8 at noon)
    - Cloud cover reduces this proportionally
    """
    # Simple diurnal cycle (peaks at hour 12)
    # Assuming ~6am sunrise, ~6pm sunset for mid-latitudes
    solar_angle = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
    clear_sky_cf = 0.8 * solar_angle  # Peak CF ~0.8 at noon
    
    # Cloud cover reduces solar (linear approximation)
    # High cloud transmits more than low cloud, but we simplify
    cf = clear_sky_cf * (1 - 0.9 * cloud_cover)  # 90% reduction at full cloud
    
    return np.maximum(0, cf)


def get_germany_cells(grid_file):
    """Find grid cells within Germany bounds."""
    grid = xr.open_dataset(grid_file)
    lon = grid['A128.lon'].values.flatten()
    lat = grid['A128.lat'].values.flatten()
    
    mask = ((lon >= GERMANY_BOUNDS['lon_min']) & 
            (lon <= GERMANY_BOUNDS['lon_max']) &
            (lat >= GERMANY_BOUNDS['lat_min']) & 
            (lat <= GERMANY_BOUNDS['lat_max']))
    
    cell_indices = np.where(mask)[0]
    print(f"Found {len(cell_indices)} cells covering Germany")
    return cell_indices, lon[mask], lat[mask]


def compute_vre_capacity_factors(ds, year_idx, cell_indices, year):
    """Compute wind and solar capacity factors for one year, processing in monthly chunks."""
    # 2920 timesteps per year (365 days * 8 per day, 3h resolution)
    t_start = year_idx * 2920
    t_end = min(t_start + 2920, ds.dims['time_counter'])
    n_timesteps = t_end - t_start
    
    print(f"  Processing year {year} ({n_timesteps} timesteps)...")
    
    # Process in monthly chunks (~248 timesteps each)
    CHUNK_SIZE = 248
    
    wind_cf_list = []
    solar_cf_list = []
    timestamps_list = []
    
    for chunk_start in range(t_start, t_end, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, t_end)
        chunk_num = (chunk_start - t_start) // CHUNK_SIZE + 1
        total_chunks = (n_timesteps + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"    Chunk {chunk_num}/{total_chunks}...", end=" ", flush=True)
        
        # Load wind - compute speed immediately and take spatial mean
        u10 = ds['10u'].isel(time_counter=slice(chunk_start, chunk_end)).values
        v10 = ds['10v'].isel(time_counter=slice(chunk_start, chunk_end)).values
        wind_speed = np.sqrt(u10[:, cell_indices]**2 + v10[:, cell_indices]**2)
        wind_cf_chunk = wind_capacity_factor(wind_speed).mean(axis=1)
        del u10, v10, wind_speed
        
        # Load cloud cover and compute total
        hcc = ds['hcc'].isel(time_counter=slice(chunk_start, chunk_end)).values[:, cell_indices]
        mcc = ds['mcc'].isel(time_counter=slice(chunk_start, chunk_end)).values[:, cell_indices]
        lcc = ds['lcc'].isel(time_counter=slice(chunk_start, chunk_end)).values[:, cell_indices]
        total_cloud = 1 - (1 - hcc) * (1 - mcc) * (1 - lcc)
        del hcc, mcc, lcc
        
        # Get timestamps and hours
        timestamps_chunk = ds['time_counter'].isel(time_counter=slice(chunk_start, chunk_end)).values
        hours = np.array([np.datetime64(t, 'h').astype(int) % 24 for t in timestamps_chunk])
        
        # Solar CF
        solar_cf_chunk = np.array([
            solar_capacity_factor(total_cloud[i], hours[i]).mean() 
            for i in range(len(hours))
        ])
        del total_cloud
        
        wind_cf_list.append(wind_cf_chunk)
        solar_cf_list.append(solar_cf_chunk)
        timestamps_list.append(timestamps_chunk)
        print("done", flush=True)
    
    # Concatenate
    wind_cf = np.concatenate(wind_cf_list)
    solar_cf = np.concatenate(solar_cf_list)
    timestamps = np.concatenate(timestamps_list)
    
    # Combined VRE (assume 60% wind, 40% solar mix for Germany)
    combined_cf = 0.6 * wind_cf + 0.4 * solar_cf
    
    return wind_cf, solar_cf, combined_cf, timestamps


def find_dunkelflaute_events(wind_cf, solar_cf, min_duration_hours, timestep_hours=3):
    """
    Find Dunkelflaute events: sustained periods where BOTH wind AND solar are low.
    
    Uses moving averages over the minimum duration window to identify
    sustained low-VRE periods (not just instantaneous dips).
    
    Args:
        wind_cf: Wind capacity factor time series
        solar_cf: Solar capacity factor time series
        min_duration_hours: Minimum event duration to consider
        timestep_hours: Hours per timestep (3 for our data)
    
    Returns:
        List of events with start/end indices and duration
    """
    window_size = max(1, min_duration_hours // timestep_hours)
    
    # Compute moving averages
    wind_ma = uniform_filter1d(wind_cf, size=window_size, mode='nearest')
    solar_ma = uniform_filter1d(solar_cf, size=window_size, mode='nearest')
    
    # Dunkelflaute: BOTH wind AND solar below thresholds (sustained)
    is_drought = (wind_ma < WIND_CF_THRESHOLD) & (solar_ma < SOLAR_CF_THRESHOLD)
    
    # Find contiguous drought periods
    events = []
    start_idx = None
    
    for i, is_d in enumerate(is_drought):
        if is_d and start_idx is None:
            start_idx = i
        elif not is_d and start_idx is not None:
            duration_hours = (i - start_idx) * timestep_hours
            if duration_hours >= min_duration_hours:
                events.append({
                    'start_idx': start_idx,
                    'end_idx': i - 1,
                    'duration_hours': duration_hours,
                    'mean_wind_cf': wind_cf[start_idx:i].mean(),
                    'mean_solar_cf': solar_cf[start_idx:i].mean()
                })
            start_idx = None
    
    # Handle event at end
    if start_idx is not None:
        duration_hours = (len(is_drought) - start_idx) * timestep_hours
        if duration_hours >= min_duration_hours:
            events.append({
                'start_idx': start_idx,
                'end_idx': len(is_drought) - 1,
                'duration_hours': duration_hours,
                'mean_wind_cf': wind_cf[start_idx:].mean(),
                'mean_solar_cf': solar_cf[start_idx:].mean()
            })
    
    # Sort by duration (longest first)
    events.sort(key=lambda x: x['duration_hours'], reverse=True)
    
    return events


def main():
    parser = argparse.ArgumentParser(description='Compute Dunkelflaute events over Germany')
    parser.add_argument('--exp', choices=['ctrl', 'ssp585'], default='ctrl')
    parser.add_argument('--year', type=int, default=1950, help='Year to analyze')
    args = parser.parse_args()
    
    exp_map = {
        'ctrl': ('TCo1279-DART-1950C', 1950),
        'ssp585': ('TCo1279-DART-2080C', 2080)
    }
    exp_name, start_year = exp_map[args.exp]
    year_idx = args.year - start_year
    
    print(f"{'='*60}")
    print(f"Dunkelflaute Analysis for Germany")
    print(f"{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Year: {args.year}")
    print(f"Wind CF threshold: < {WIND_CF_THRESHOLD*100:.0f}%")
    print(f"Solar CF threshold: < {SOLAR_CF_THRESHOLD*100:.0f}%")
    print(f"Min durations analyzed: {MIN_DURATIONS} hours")
    print()
    
    # Get Germany cells
    cell_indices, lons, lats = get_germany_cells(GRID_FILE)
    
    # Load data
    print("\nLoading catalog...")
    cat = intake.open_catalog(CATALOG_PATH)
    ds = cat['ICCP'][exp_name]['atmos']['native']['atmos_3h'].to_dask()
    
    # Compute capacity factors
    wind_cf, solar_cf, combined_cf, timestamps = compute_vre_capacity_factors(
        ds, year_idx, cell_indices, args.year
    )
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"Capacity Factor Statistics for {args.year}")
    print(f"{'='*60}")
    print(f"Wind CF:     mean={wind_cf.mean():.3f}, std={wind_cf.std():.3f}, max={wind_cf.max():.3f}")
    print(f"Solar CF:    mean={solar_cf.mean():.3f}, std={solar_cf.std():.3f}, max={solar_cf.max():.3f}")
    print(f"Combined CF: mean={combined_cf.mean():.3f}, std={combined_cf.std():.3f}")
    
    # Count hours below threshold (instantaneous)
    wind_low = (wind_cf < WIND_CF_THRESHOLD).sum() * 3
    solar_low = (solar_cf < SOLAR_CF_THRESHOLD).sum() * 3
    both_low = ((wind_cf < WIND_CF_THRESHOLD) & (solar_cf < SOLAR_CF_THRESHOLD)).sum() * 3
    print(f"\nInstantaneous low-VRE hours:")
    print(f"  Wind < {WIND_CF_THRESHOLD*100:.0f}%:  {wind_low}h ({wind_low/87.6:.1f}% of year)")
    print(f"  Solar < {SOLAR_CF_THRESHOLD*100:.0f}%: {solar_low}h ({solar_low/87.6:.1f}% of year)")
    print(f"  Both low:    {both_low}h ({both_low/87.6:.1f}% of year)")
    
    # Find events for different minimum durations
    print(f"\n{'='*60}")
    print(f"Dunkelflaute Events (sustained low wind AND solar)")
    print(f"{'='*60}")
    
    for min_dur in MIN_DURATIONS:
        events = find_dunkelflaute_events(wind_cf, solar_cf, min_dur)
        
        total_hours = sum(e['duration_hours'] for e in events)
        print(f"\n--- Minimum duration: {min_dur}h ---")
        print(f"Number of events: {len(events)}")
        print(f"Total Dunkelflaute hours: {total_hours}")
        
        if events:
            print(f"Longest event: {events[0]['duration_hours']}h ({events[0]['duration_hours']/24:.1f} days)")
            print(f"\nTop 5 events:")
            for i, evt in enumerate(events[:5], 1):
                start = str(timestamps[evt['start_idx']])[:16]
                end = str(timestamps[evt['end_idx']])[:16]
                print(f"  {i}. {evt['duration_hours']:3d}h (wind={evt['mean_wind_cf']:.3f}, solar={evt['mean_solar_cf']:.3f}): {start} → {end}")


if __name__ == '__main__':
    main()
