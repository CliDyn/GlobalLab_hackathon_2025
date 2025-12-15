#!/usr/bin/env python3
"""
Compute Dunkelflaute events for Germany using ERA5 data.

Based on Mockert et al. (2023) and Kittel & Schill (2024) methodology:
- 48-hour moving average of combined capacity factor
- Threshold: combined CF < 0.06 (6% of installed capacity)
- Capacity weights for Germany 2024: 58% solar, 37% onshore wind, 5% offshore wind

This script is designed to work with ERA5 data but can be adapted to TCo1279-DART
by changing the data source configuration.

Variables needed:
- 10m u-component of wind (10u, code 165)
- 10m v-component of wind (10v, code 166)
- Surface solar radiation downwards (ssrd, code 169)
- Optionally: 100m wind for offshore (100u/100v, codes 246/247)

References:
- Mockert et al. (2023): "A Brief Climatology of Dunkelflaute Events"
- Kittel & Schill (2024): "Measuring the Dunkelflaute"
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')
import os
# Set cfgrib to use a writable directory for index files
os.environ['GRIB_INDEX_PATH'] = '/tmp'

# ============================================================================
# CONFIGURATION - Change this section to switch between ERA5 and TCo1279-DART
# ============================================================================

class ERA5Config:
    """Configuration for ERA5 data."""
    name = "ERA5"
    base_path = Path("/pool/data/ERA5/E5/sf")
    temporal_resolution_hours = 1  # ERA5 is hourly
    
    # Variable paths (relative to base_path)
    var_paths = {
        '10u': 'an/1H/165',   # 10m u-wind
        '10v': 'an/1H/166',   # 10m v-wind
        'ssrd': 'fc/1H/169',  # Surface solar radiation downwards
        '100u': 'an/1H/246',  # 100m u-wind (for offshore, if available)
        '100v': 'an/1H/247',  # 100m v-wind (for offshore, if available)
    }
    
    # File pattern: E5sf00_1H_YYYY-MM-DD_VAR.grb for analysis
    #               E5sf12_1H_YYYY-MM-DD_VAR.grb for forecast
    def get_file_pattern(self, var, year, month=None):
        if var in ['ssrd']:
            prefix = 'E5sf12'
        else:
            prefix = 'E5sf00'
        if month:
            return f"{prefix}_1H_{year}-{month:02d}-*_{self.var_paths[var].split('/')[-1]}.grb"
        return f"{prefix}_1H_{year}-*_{self.var_paths[var].split('/')[-1]}.grb"
    
    # Spatial resolution
    lat_res = 0.25
    lon_res = 0.25


class TCo1279Config:
    """Configuration for TCo1279-DART data (template for future use)."""
    name = "TCo1279-DART"
    base_path = Path("/work/ab0246/a270092/data/TCo1279-DART")  # Adjust as needed
    temporal_resolution_hours = 3  # TCo1279 is 3-hourly
    
    # Variable mapping (adjust based on actual variable names in TCo1279)
    var_names = {
        '10u': '10u',
        '10v': '10v', 
        'ssrd': 'ssrd',  # or alternative like 'ssr' or computed from cloud cover
        '100u': '100u',
        '100v': '100v',
    }
    
    # Native grid - will need grid file for lat/lon
    grid_file = "/work/ab0246/a270092/input/oasis/cy43r3/TCO1279-DART/grids.nc"


# ============================================================================
# GERMANY REGION DEFINITION
# ============================================================================

GERMANY_BOUNDS = {
    'lon_min': 5.8,
    'lon_max': 15.1,
    'lat_min': 47.2,
    'lat_max': 55.1
}

# ============================================================================
# CAPACITY FACTOR PARAMETERS
# ============================================================================

# Germany 2024 installed capacity shares (Bundesnetzagentur, 2025)
CAPACITY_WEIGHTS = {
    'solar': 0.577,      # 99.3 GW
    'onshore_wind': 0.369,  # 63.5 GW
    'offshore_wind': 0.054  # 9.2 GW
}

# Wind turbine power curve parameters
WIND_PARAMS = {
    'onshore': {
        'cut_in': 3.0,    # m/s
        'rated': 12.0,    # m/s
        'cut_out': 25.0,  # m/s
        'hub_height': 100  # m (typical modern turbine)
    },
    'offshore': {
        'cut_in': 3.0,
        'rated': 13.0,
        'cut_out': 25.0,
        'hub_height': 100
    }
}

# Dunkelflaute detection parameters (Mockert et al. 2023)
DUNKELFLAUTE_PARAMS = {
    'moving_avg_hours': 48,     # 48-hour moving average
    'threshold': 0.06,          # Combined CF threshold (6%)
    'min_duration_hours': 24,   # Minimum event duration to report
}


# ============================================================================
# CAPACITY FACTOR CALCULATIONS
# ============================================================================

def wind_speed_at_hub_height(u10, v10, hub_height=100, z_ref=10, roughness=0.03):
    """
    Extrapolate 10m wind to hub height using log wind profile.
    
    Args:
        u10, v10: Wind components at 10m
        hub_height: Target height (m)
        z_ref: Reference height (10m)
        roughness: Surface roughness length (0.03 for open terrain)
    
    Returns:
        Wind speed at hub height
    """
    ws_10m = np.sqrt(u10**2 + v10**2)
    # Log wind profile extrapolation
    scale_factor = np.log(hub_height / roughness) / np.log(z_ref / roughness)
    ws_hub = ws_10m * scale_factor
    return ws_hub


def wind_capacity_factor(wind_speed, params):
    """
    Convert wind speed to capacity factor using IEC power curve.
    
    Follows atlite methodology: cubic relationship between cut-in and rated.
    """
    cf = np.zeros_like(wind_speed, dtype=np.float32)
    
    cut_in = params['cut_in']
    rated = params['rated']
    cut_out = params['cut_out']
    
    # Cubic region between cut-in and rated
    mask_cubic = (wind_speed >= cut_in) & (wind_speed < rated)
    cf[mask_cubic] = ((wind_speed[mask_cubic] - cut_in) / (rated - cut_in)) ** 3
    
    # Rated region
    mask_rated = (wind_speed >= rated) & (wind_speed < cut_out)
    cf[mask_rated] = 1.0
    
    # Above cut-out: shutdown
    cf[wind_speed >= cut_out] = 0.0
    
    return cf


def solar_capacity_factor_from_ssrd(ssrd, timestep_hours=1):
    """
    Convert surface solar radiation downwards (SSRD) to capacity factor.
    
    SSRD is accumulated J/m² - need to convert to power and normalize.
    Reference: typical peak irradiance ~1000 W/m², panel efficiency ~15-20%
    
    Args:
        ssrd: Surface solar radiation downwards (J/m² accumulated over timestep)
        timestep_hours: Hours per timestep
    
    Returns:
        Solar capacity factor [0, 1]
    """
    # Convert J/m² to W/m² (average power over timestep)
    irradiance = ssrd / (timestep_hours * 3600)  # W/m²
    
    # Reference irradiance for CF=1 (Standard Test Conditions: 1000 W/m²)
    # Account for typical system efficiency (~15-18% for real installations)
    reference_irradiance = 1000.0  # W/m²
    
    # Capacity factor (capped at 1.0)
    cf = np.minimum(irradiance / reference_irradiance, 1.0)
    cf = np.maximum(cf, 0.0)
    
    return cf.astype(np.float32)


def solar_capacity_factor_from_cloud(tcc, hour_of_day, lat):
    """
    Alternative: Estimate solar CF from total cloud cover (for TCo1279 if no ssrd).
    
    Args:
        tcc: Total cloud cover [0, 1]
        hour_of_day: Hour of day (0-23)
        lat: Latitude (for day length estimation)
    
    Returns:
        Solar capacity factor [0, 1]
    """
    # Simple clear-sky model based on solar zenith angle
    # This is a fallback if SSRD is not available
    solar_hour_angle = (hour_of_day - 12) * 15 * np.pi / 180  # radians
    declination = 0  # Simplification - should vary with day of year
    
    lat_rad = np.deg2rad(lat)
    cos_zenith = np.sin(lat_rad) * np.sin(declination) + \
                 np.cos(lat_rad) * np.cos(declination) * np.cos(solar_hour_angle)
    cos_zenith = np.maximum(cos_zenith, 0)
    
    # Clear-sky CF (peak ~0.8)
    clear_sky_cf = 0.8 * cos_zenith
    
    # Cloud attenuation (simple linear model)
    cf = clear_sky_cf * (1 - 0.75 * tcc)
    
    return np.maximum(cf, 0).astype(np.float32)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_era5_year(config, year, variables=['10u', '10v', 'ssrd']):
    """
    Load ERA5 data for a specific year and Germany region.
    
    Returns:
        xarray.Dataset with requested variables, subset to Germany
    """
    datasets = {}
    
    for var in variables:
        var_path = config.base_path / config.var_paths[var]
        
        # Find all files for this year
        if var == 'ssrd':
            pattern = f"E5sf12_1H_{year}-*_{config.var_paths[var].split('/')[-1]}.grb"
        else:
            pattern = f"E5sf00_1H_{year}-*_{config.var_paths[var].split('/')[-1]}.grb"
        
        files = sorted(var_path.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No files found for {var} in {year} at {var_path}")
        
        print(f"  Loading {var}: {len(files)} files...")
        
        # Load and concatenate
        ds = xr.open_mfdataset(
            files, 
            engine='cfgrib',
            combine='by_coords',
            parallel=True
        )
        
        # Subset to Germany
        ds = ds.sel(
            latitude=slice(GERMANY_BOUNDS['lat_max'], GERMANY_BOUNDS['lat_min']),
            longitude=slice(GERMANY_BOUNDS['lon_min'], GERMANY_BOUNDS['lon_max'])
        )
        
        datasets[var] = ds
    
    return datasets


def get_germany_mask_era5(lats, lons):
    """
    Create a boolean mask for Germany grid cells.
    
    ERA5 uses reduced Gaussian grid where lat/lon are 1D arrays over 'values' dimension.
    """
    mask = ((lons >= GERMANY_BOUNDS['lon_min']) & 
            (lons <= GERMANY_BOUNDS['lon_max']) &
            (lats >= GERMANY_BOUNDS['lat_min']) & 
            (lats <= GERMANY_BOUNDS['lat_max']))
    return mask


def load_era5_data_efficient(year, months=None):
    """
    Efficiently load ERA5 data for dunkelflaute calculation.
    
    Loads only what's needed and computes spatial averages early to reduce memory.
    ERA5 uses reduced Gaussian grid - lat/lon are coordinates over 'values' dimension.
    
    Args:
        year: Year to load
        months: List of months (1-12) or None for full year
    
    Returns:
        DataFrame with hourly time series of Germany-average capacity factors
    """
    config = ERA5Config()
    base_path = config.base_path
    
    if months is None:
        months = range(1, 13)
    
    results = []
    germany_mask = None
    
    for month in months:
        print(f"  Processing {year}-{month:02d}... ({month}/12 = {month*100//12}%)", flush=True)
        
        # Load 10m wind components
        u10_path = base_path / 'an/1H/165'
        v10_path = base_path / 'an/1H/166'
        ssrd_path = base_path / 'fc/1H/169'
        
        # File patterns for this month
        u10_files = sorted(u10_path.glob(f"E5sf00_1H_{year}-{month:02d}-*_165.grb"))
        v10_files = sorted(v10_path.glob(f"E5sf00_1H_{year}-{month:02d}-*_166.grb"))
        ssrd_files = sorted(ssrd_path.glob(f"E5sf12_1H_{year}-{month:02d}-*_169.grb"))
        
        if not u10_files or not v10_files or not ssrd_files:
            print(f"    Warning: Missing data for {year}-{month:02d}")
            continue
        
        # Load data file by file to handle reduced Gaussian grid
        for day_idx, (u10_file, v10_file, ssrd_file) in enumerate(zip(u10_files, v10_files, ssrd_files)):
            try:
                ds_u10 = xr.open_dataset(u10_file, engine='cfgrib')
                ds_v10 = xr.open_dataset(v10_file, engine='cfgrib')
                ds_ssrd = xr.open_dataset(ssrd_file, engine='cfgrib')
            except Exception as e:
                print(f"    Error loading day {day_idx}: {e}")
                continue
            
            # Get Germany mask on first iteration
            if germany_mask is None:
                lats = ds_u10['latitude'].values
                lons = ds_u10['longitude'].values
                germany_mask = get_germany_mask_era5(lats, lons)
                n_germany_cells = germany_mask.sum()
                print(f"    Found {n_germany_cells} grid cells for Germany")
            
            # Get variable names (cfgrib uses shortName)
            u10_var = list(ds_u10.data_vars)[0]
            v10_var = list(ds_v10.data_vars)[0]
            ssrd_var = list(ds_ssrd.data_vars)[0]
            
            # Process each timestep in this file
            n_times = ds_u10.dims['time']
            
            # SSRD has different structure: (time, step, values) for forecast
            # Need to flatten it to match wind data times
            # valid_time = time + step gives actual datetime
            ssrd_valid_times = ds_ssrd['valid_time'].values.flatten()
            ssrd_data = ds_ssrd[ssrd_var].values.reshape(-1, ds_ssrd.dims['values'])
            
            for i in range(n_times):
                t = ds_u10['time'].values[i]
                
                # Extract values for Germany only
                u10 = ds_u10[u10_var].isel(time=i).values[germany_mask]
                v10 = ds_v10[v10_var].isel(time=i).values[germany_mask]
                
                # Wind at hub height (100m)
                ws_hub = wind_speed_at_hub_height(u10, v10, hub_height=100)
                
                # Onshore wind CF (spatial average)
                onshore_cf = wind_capacity_factor(ws_hub, WIND_PARAMS['onshore']).mean()
                
                # Offshore wind CF (use same for now)
                offshore_cf = wind_capacity_factor(ws_hub, WIND_PARAMS['offshore']).mean()
                
                # Solar CF - find matching valid_time in SSRD
                try:
                    # Find closest SSRD time to wind time
                    time_diffs = np.abs(ssrd_valid_times - t)
                    closest_idx = np.argmin(time_diffs)
                    ssrd = ssrd_data[closest_idx, germany_mask]
                    solar_cf = solar_capacity_factor_from_ssrd(ssrd, timestep_hours=1).mean()
                except Exception as e:
                    solar_cf = 0.0
                
                # Combined CF (weighted average)
                combined_cf = (
                    CAPACITY_WEIGHTS['solar'] * solar_cf +
                    CAPACITY_WEIGHTS['onshore_wind'] * onshore_cf +
                    CAPACITY_WEIGHTS['offshore_wind'] * offshore_cf
                )
                
                results.append({
                    'time': pd.Timestamp(t),
                    'solar_cf': solar_cf,
                    'onshore_wind_cf': onshore_cf,
                    'offshore_wind_cf': offshore_cf,
                    'combined_cf': combined_cf
                })
            
            # Close datasets to free memory
            ds_u10.close()
            ds_v10.close()
            ds_ssrd.close()
        
        print(f"    Processed {len(results)} timesteps so far")
    
    return pd.DataFrame(results).set_index('time').sort_index()


# ============================================================================
# DUNKELFLAUTE DETECTION
# ============================================================================

def detect_dunkelflaute_events(df, moving_avg_hours=48, threshold=0.06, 
                               min_duration_hours=24, timestep_hours=1):
    """
    Detect Dunkelflaute events using the Mockert et al. (2023) methodology.
    
    Args:
        df: DataFrame with 'combined_cf' column and datetime index
        moving_avg_hours: Window for moving average (default 48h)
        threshold: CF threshold for drought (default 0.06)
        min_duration_hours: Minimum event duration to report
        timestep_hours: Hours per timestep
    
    Returns:
        List of event dictionaries with start, end, duration, severity
    """
    # Compute moving average
    window_size = max(1, moving_avg_hours // timestep_hours)
    cf_ma = uniform_filter1d(df['combined_cf'].values, size=window_size, mode='nearest')
    
    # Add to dataframe
    df = df.copy()
    df['cf_ma'] = cf_ma
    df['is_drought'] = cf_ma < threshold
    
    # Find contiguous drought periods
    events = []
    start_idx = None
    
    for i in range(len(df)):
        is_d = df['is_drought'].iloc[i]
        
        if is_d and start_idx is None:
            start_idx = i
        elif not is_d and start_idx is not None:
            # End of event
            duration_hours = (i - start_idx) * timestep_hours
            
            if duration_hours >= min_duration_hours:
                event_data = df.iloc[start_idx:i]
                events.append({
                    'start': df.index[start_idx],
                    'end': df.index[i-1],
                    'duration_hours': duration_hours,
                    'mean_cf': event_data['combined_cf'].mean(),
                    'min_cf': event_data['combined_cf'].min(),
                    'mean_cf_ma': event_data['cf_ma'].mean(),
                    'mean_solar_cf': event_data['solar_cf'].mean(),
                    'mean_wind_cf': (
                        CAPACITY_WEIGHTS['onshore_wind'] * event_data['onshore_wind_cf'].mean() +
                        CAPACITY_WEIGHTS['offshore_wind'] * event_data['offshore_wind_cf'].mean()
                    ) / (CAPACITY_WEIGHTS['onshore_wind'] + CAPACITY_WEIGHTS['offshore_wind']),
                    'severity': (threshold - event_data['combined_cf'].mean()) * duration_hours
                })
            start_idx = None
    
    # Handle event at end of time series
    if start_idx is not None:
        duration_hours = (len(df) - start_idx) * timestep_hours
        if duration_hours >= min_duration_hours:
            event_data = df.iloc[start_idx:]
            events.append({
                'start': df.index[start_idx],
                'end': df.index[-1],
                'duration_hours': duration_hours,
                'mean_cf': event_data['combined_cf'].mean(),
                'min_cf': event_data['combined_cf'].min(),
                'mean_cf_ma': event_data['cf_ma'].mean(),
                'mean_solar_cf': event_data['solar_cf'].mean(),
                'mean_wind_cf': (
                    CAPACITY_WEIGHTS['onshore_wind'] * event_data['onshore_wind_cf'].mean() +
                    CAPACITY_WEIGHTS['offshore_wind'] * event_data['offshore_wind_cf'].mean()
                ) / (CAPACITY_WEIGHTS['onshore_wind'] + CAPACITY_WEIGHTS['offshore_wind']),
                'severity': (threshold - event_data['combined_cf'].mean()) * duration_hours
            })
    
    # Sort by severity (most severe first)
    events.sort(key=lambda x: x['severity'], reverse=True)
    
    return events, df


def compute_annual_statistics(df, events):
    """Compute annual dunkelflaute statistics."""
    stats = {
        'n_events': len(events),
        'total_hours': sum(e['duration_hours'] for e in events),
        'mean_cf': df['combined_cf'].mean(),
        'std_cf': df['combined_cf'].std(),
        'min_cf': df['combined_cf'].min(),
        'hours_below_threshold': (df['combined_cf'] < DUNKELFLAUTE_PARAMS['threshold']).sum(),
    }
    
    if events:
        stats['max_duration_hours'] = max(e['duration_hours'] for e in events)
        stats['max_severity'] = max(e['severity'] for e in events)
        stats['longest_event_start'] = max(events, key=lambda x: x['duration_hours'])['start']
    else:
        stats['max_duration_hours'] = 0
        stats['max_severity'] = 0
        stats['longest_event_start'] = None
    
    return stats


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_year(year, save_results=True, output_dir=None):
    """
    Run full dunkelflaute analysis for a single year.
    
    Args:
        year: Year to analyze
        save_results: Whether to save results to CSV
        output_dir: Directory for output files
    
    Returns:
        events: List of detected events
        stats: Annual statistics dictionary
        df: Full time series DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Dunkelflaute Analysis for Germany - {year}")
    print(f"{'='*60}")
    print(f"Method: {DUNKELFLAUTE_PARAMS['moving_avg_hours']}h moving average")
    print(f"Threshold: CF < {DUNKELFLAUTE_PARAMS['threshold']}")
    print(f"Capacity weights: Solar {CAPACITY_WEIGHTS['solar']*100:.1f}%, "
          f"Onshore {CAPACITY_WEIGHTS['onshore_wind']*100:.1f}%, "
          f"Offshore {CAPACITY_WEIGHTS['offshore_wind']*100:.1f}%")
    print()
    
    # Load data
    print("Loading ERA5 data...")
    df = load_era5_data_efficient(year)
    print(f"  Loaded {len(df)} timesteps")
    
    # Detect events
    print("\nDetecting Dunkelflaute events...")
    events, df = detect_dunkelflaute_events(
        df,
        moving_avg_hours=DUNKELFLAUTE_PARAMS['moving_avg_hours'],
        threshold=DUNKELFLAUTE_PARAMS['threshold'],
        min_duration_hours=DUNKELFLAUTE_PARAMS['min_duration_hours'],
        timestep_hours=1  # ERA5 is hourly
    )
    
    # Statistics
    stats = compute_annual_statistics(df, events)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {year}")
    print(f"{'='*60}")
    print(f"Mean combined CF: {stats['mean_cf']:.3f} ± {stats['std_cf']:.3f}")
    print(f"Min combined CF: {stats['min_cf']:.3f}")
    print(f"Hours below threshold: {stats['hours_below_threshold']} ({stats['hours_below_threshold']/87.6:.1f}%)")
    print(f"\nDunkelflaute events: {stats['n_events']}")
    print(f"Total Dunkelflaute hours: {stats['total_hours']}")
    
    if events:
        print(f"Longest event: {stats['max_duration_hours']}h ({stats['max_duration_hours']/24:.1f} days)")
        print(f"\nTop 10 events by severity:")
        for i, evt in enumerate(events[:10], 1):
            print(f"  {i:2d}. {evt['start'].strftime('%Y-%m-%d')} to {evt['end'].strftime('%Y-%m-%d')}: "
                  f"{evt['duration_hours']:3d}h ({evt['duration_hours']/24:.1f}d), "
                  f"CF={evt['mean_cf']:.3f}, severity={evt['severity']:.2f}")
    
    # Save results
    if save_results and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save time series
        df.to_csv(output_dir / f"dunkelflaute_timeseries_{year}.csv")
        
        # Save events
        if events:
            events_df = pd.DataFrame(events)
            events_df.to_csv(output_dir / f"dunkelflaute_events_{year}.csv", index=False)
        
        print(f"\nResults saved to {output_dir}")
    
    return events, stats, df


def analyze_multi_year(start_year, end_year, output_dir=None):
    """
    Run dunkelflaute analysis for multiple years.
    
    Args:
        start_year: First year
        end_year: Last year (inclusive)
        output_dir: Directory for output files
    
    Returns:
        all_events: List of all events across years
        annual_stats: Dict of annual statistics
    """
    all_events = []
    annual_stats = {}
    
    for year in range(start_year, end_year + 1):
        try:
            events, stats, _ = analyze_year(year, save_results=True, output_dir=output_dir)
            annual_stats[year] = stats
            for evt in events:
                evt['year'] = year
            all_events.extend(events)
        except Exception as e:
            print(f"Error processing {year}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Multi-Year Summary ({start_year}-{end_year})")
    print(f"{'='*60}")
    
    if annual_stats:
        years = sorted(annual_stats.keys())
        total_events = sum(s['n_events'] for s in annual_stats.values())
        total_hours = sum(s['total_hours'] for s in annual_stats.values())
        
        print(f"Total years analyzed: {len(years)}")
        print(f"Total events: {total_events}")
        print(f"Total Dunkelflaute hours: {total_hours}")
        print(f"Average events/year: {total_events/len(years):.1f}")
        print(f"Average Dunkelflaute hours/year: {total_hours/len(years):.1f}")
        
        # Find worst years
        worst_years = sorted(annual_stats.items(), key=lambda x: x[1]['total_hours'], reverse=True)[:5]
        print(f"\nWorst years (by total hours):")
        for year, s in worst_years:
            print(f"  {year}: {s['total_hours']}h in {s['n_events']} events")
        
        # Find worst events overall
        if all_events:
            worst_events = sorted(all_events, key=lambda x: x['severity'], reverse=True)[:10]
            print(f"\nWorst events overall:")
            for i, evt in enumerate(worst_events, 1):
                print(f"  {i:2d}. {evt['start'].strftime('%Y-%m-%d')} to {evt['end'].strftime('%Y-%m-%d')}: "
                      f"{evt['duration_hours']:3d}h, severity={evt['severity']:.2f}")
    
    # Save summary
    if output_dir:
        output_dir = Path(output_dir)
        
        # Save annual stats
        stats_df = pd.DataFrame(annual_stats).T
        stats_df.index.name = 'year'
        stats_df.to_csv(output_dir / f"dunkelflaute_annual_stats_{start_year}_{end_year}.csv")
        
        # Save all events
        if all_events:
            events_df = pd.DataFrame(all_events)
            events_df.to_csv(output_dir / f"dunkelflaute_all_events_{start_year}_{end_year}.csv", index=False)
    
    return all_events, annual_stats


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute Dunkelflaute events for Germany using ERA5 data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --year 1997                    # Analyze single year
  %(prog)s --start 1979 --end 2024        # Analyze multiple years
  %(prog)s --year 1997 --threshold 0.10   # Use different threshold
        """
    )
    
    parser.add_argument('--year', type=int, help='Single year to analyze')
    parser.add_argument('--start', type=int, help='Start year for multi-year analysis')
    parser.add_argument('--end', type=int, help='End year for multi-year analysis')
    parser.add_argument('--threshold', type=float, default=0.06,
                        help='Dunkelflaute threshold (default: 0.06)')
    parser.add_argument('--moving-avg', type=int, default=48,
                        help='Moving average window in hours (default: 48)')
    parser.add_argument('--output-dir', type=str, 
                        default='/work/ab0246/a270092/software/GlobalLab_hackathon_2025/dunkelflaute_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Update parameters
    DUNKELFLAUTE_PARAMS['threshold'] = args.threshold
    DUNKELFLAUTE_PARAMS['moving_avg_hours'] = args.moving_avg
    
    # Run analysis
    if args.year:
        analyze_year(args.year, save_results=True, output_dir=args.output_dir)
    elif args.start and args.end:
        analyze_multi_year(args.start, args.end, output_dir=args.output_dir)
    else:
        parser.print_help()
        print("\nError: Must specify either --year or both --start and --end")


if __name__ == '__main__':
    main()
