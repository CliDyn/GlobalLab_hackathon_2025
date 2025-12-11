#!/usr/bin/env python
"""Plot regional temperature maps around top extreme cities."""
import intake
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

CATALOG_PATH = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/catalog/main.yaml"
OUTPUT_DIR = Path("/work/ab0246/a270092/software/GlobalLab_hackathon_2025/plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load catalog
print("Loading catalog...")
gl_cat = intake.open_catalog(CATALOG_PATH)

# Load both datasets
print("Loading datasets...")
ds_ctrl = gl_cat['ICCP']['TCo1279-DART-1950C']['atmos']['native']['atmos_3h'].to_dask()
ds_ssp = gl_cat['ICCP']['TCo1279-DART-2080C']['atmos']['native']['atmos_3h'].to_dask()

# Get grid coordinates
lats = ds_ctrl['lat'].values
lons = ds_ctrl['lon'].values


def plot_regional_temp(ds, time_idx, city_lat, city_lon, city_name, timestamp, 
                       title_suffix, output_name, radius_deg=5):
    """Plot temperature around a city at a specific timestep."""
    print(f"  Plotting {city_name} at t={time_idx} ({timestamp})...")
    
    # Find cells within radius
    mask = (np.abs(lats - city_lat) < radius_deg) & (np.abs(lons - city_lon) < radius_deg)
    cell_indices = np.where(mask)[0]
    
    if len(cell_indices) == 0:
        print(f"    No cells found for {city_name}")
        return
    
    # Load temperature for this timestep and region
    try:
        temp_full = ds['2t'].isel(time_counter=time_idx).values
        temp_region = temp_full[cell_indices] - 273.15  # Convert to Celsius
        lats_region = lats[cell_indices]
        lons_region = lons[cell_indices]
    except Exception as e:
        print(f"    Error loading data: {e}")
        return
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set extent
    ax.set_extent([city_lon - radius_deg, city_lon + radius_deg, 
                   city_lat - radius_deg, city_lat + radius_deg])
    
    # Add features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.add_feature(cfeature.LAND, alpha=0.1)
    
    # Scatter plot of temperatures
    sc = ax.scatter(lons_region, lats_region, c=temp_region, 
                    cmap='RdYlBu_r', s=2, vmin=20, vmax=60,
                    transform=ccrs.PlateCarree())
    
    # Mark city location
    ax.plot(city_lon, city_lat, 'k*', markersize=15, transform=ccrs.PlateCarree())
    ax.plot(city_lon, city_lat, 'w*', markersize=10, transform=ccrs.PlateCarree())
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Temperature (°C)')
    
    # Title
    ax.set_title(f"{city_name} - {title_suffix}\n{timestamp[:10]} | Max: {temp_region.max():.1f}°C")
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    # Save
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_name}")


# Load sorted CSVs
ctrl_sorted = pd.read_csv('city_max_temp_ctrl_sorted.csv')
ssp_sorted = pd.read_csv('city_max_temp_ssp585_sorted.csv')
diff_sorted = pd.read_csv('city_temp_diff_sorted.csv')

# Plot top 10 CTRL
print("\n=== Top 10 Hottest Cities (CTRL 1950-1970) ===")
for i, row in ctrl_sorted.head(10).iterrows():
    plot_regional_temp(ds_ctrl, int(row['time_idx']), 
                       row['city_lat'], row['city_lon'], row['city_name'],
                       row['timestamp'], 
                       f"CTRL ({row['max_temp']:.1f}°C)",
                       f"ctrl_top{i+1:02d}_{row['city_name'].replace(' ', '_')}.png")

# Plot top 10 SSP585
print("\n=== Top 10 Hottest Cities (SSP585 2080-2100) ===")
for i, row in ssp_sorted.head(10).iterrows():
    plot_regional_temp(ds_ssp, int(row['time_idx']), 
                       row['city_lat'], row['city_lon'], row['city_name'],
                       row['timestamp'],
                       f"SSP585 ({row['max_temp']:.1f}°C)",
                       f"ssp_top{i+1:02d}_{row['city_name'].replace(' ', '_')}.png")

# Plot top 10 by difference (comparing both periods)
print("\n=== Top 10 Cities by Temperature Increase ===")
for i, row in diff_sorted.head(10).iterrows():
    # Plot CTRL
    plot_regional_temp(ds_ctrl, int(row['time_idx_ctrl']),
                       row['city_lat'], row['city_lon'], row['city_name'],
                       row['timestamp_ctrl'],
                       f"CTRL ({row['max_temp_ctrl']:.1f}°C)",
                       f"diff_top{i+1:02d}_{row['city_name'].replace(' ', '_')}_ctrl.png")
    # Plot SSP585
    plot_regional_temp(ds_ssp, int(row['time_idx_ssp']),
                       row['city_lat'], row['city_lon'], row['city_name'],
                       row['timestamp_ssp'],
                       f"SSP585 ({row['max_temp_ssp']:.1f}°C) [+{row['temp_diff']:.1f}°C]",
                       f"diff_top{i+1:02d}_{row['city_name'].replace(' ', '_')}_ssp.png")

print(f"\nDone! Plots saved to {OUTPUT_DIR}")
