#!/usr/bin/env python
"""
Plot regional temperature maps on native OIFS grid (no interpolation).
Uses quadrilateral cells from TCO1279 grid file.
"""
import intake
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

CATALOG_PATH = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/catalog/main.yaml"
GRID_FILE = "/work/ab0246/a270092/input/oasis/cy43r3/TCO1279-DART/grids.nc"
BASE_DIR = Path("/work/ab0246/a270092/software/GlobalLab_hackathon_2025/plots/2t")
CTRL_DIR = BASE_DIR / "ctrl_records"
SSP_DIR = BASE_DIR / "ssp585_records"
DIFF_DIR = BASE_DIR / "difference_records"
for d in [CTRL_DIR, SSP_DIR, DIFF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load catalog
print("Loading catalog...")
gl_cat = intake.open_catalog(CATALOG_PATH)

# Load datasets
print("Loading datasets...")
ds_ctrl = gl_cat['ICCP']['TCo1279-DART-1950C']['atmos']['native']['atmos_3h'].to_dask()
ds_ssp = gl_cat['ICCP']['TCo1279-DART-2080C']['atmos']['native']['atmos_3h'].to_dask()

# Load grid (cell corners)
print("Loading OIFS grid...")
grid = xr.open_dataset(GRID_FILE)
clo = grid['A128.clo'].values  # (4, 1, 6599680) - corner longitudes
cla = grid['A128.cla'].values  # (4, 1, 6599680) - corner latitudes
lon_centers = grid['A128.lon'].values.flatten()
lat_centers = grid['A128.lat'].values.flatten()

# Reshape corners to (n_cells, 4, 2) for PolyCollection
clo = clo.squeeze().T  # (6599680, 4)
cla = cla.squeeze().T  # (6599680, 4)
cell_verts = np.stack([clo, cla], axis=-1)  # (6599680, 4, 2)

# Precompute mask for wrap-around cells (cells that span the date line)
lon_range = np.max(clo, axis=1) - np.min(clo, axis=1)
valid_global_mask = lon_range < 180  # Skip cells that wrap around

print(f"Grid loaded: {len(lon_centers):,} cells, {np.sum(valid_global_mask):,} valid")


def plot_regional_temp_native(ds, time_idx, city_lat, city_lon, city_name, timestamp,
                               title_suffix, output_dir, output_name, radius_deg=5, vmin=20, vmax=60):
    """Plot temperature on native grid around a city."""
    print(f"  Plotting {city_name} at t={time_idx}...")
    
    # Define region bounds
    lon_min, lon_max = city_lon - radius_deg, city_lon + radius_deg
    lat_min, lat_max = city_lat - radius_deg, city_lat + radius_deg
    
    # Find cells in region (use centers for filtering)
    region_mask = ((lon_centers >= lon_min - 0.5) & (lon_centers <= lon_max + 0.5) &
                   (lat_centers >= lat_min - 0.5) & (lat_centers <= lat_max + 0.5) &
                   valid_global_mask)
    cell_indices = np.where(region_mask)[0]
    
    if len(cell_indices) == 0:
        print(f"    No cells found for {city_name}")
        return
    
    print(f"    {len(cell_indices):,} cells in region")
    
    # Load temperature for this timestep (full grid, then filter)
    try:
        temp_full = ds['2t'].isel(time_counter=time_idx).values
        temp_region = temp_full[cell_indices] - 273.15  # Convert to Celsius
    except Exception as e:
        print(f"    Error loading data: {e}")
        return
    
    # Get cell vertices for this region
    verts_region = cell_verts[cell_indices]
    
    # Create figure (smaller = bigger fonts)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set extent
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Create PolyCollection for native grid cells
    collection = PolyCollection(
        verts_region,
        array=temp_region,
        cmap='RdYlBu_r',
        edgecolors='none',
        linewidths=0,
        transform=ccrs.PlateCarree(),
    )
    collection.set_clim(vmin, vmax)
    ax.add_collection(collection)
    
    # Add features - clear outlines
    ax.coastlines(linewidth=1.5, color='black', zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=1.0, linestyle='-', color='#333333', zorder=4)
    ax.add_feature(cfeature.RIVERS, linewidth=0.8, edgecolor='#0066cc', zorder=3)
    ax.add_feature(cfeature.LAKES, facecolor='#99ccff', edgecolor='#0066cc', linewidth=0.5, zorder=3)
    
    # Mark city location
    ax.plot(city_lon, city_lat, 'k*', markersize=20, transform=ccrs.PlateCarree(), zorder=10)
    ax.plot(city_lon, city_lat, 'w*', markersize=12, transform=ccrs.PlateCarree(), zorder=11)
    
    # Colorbar
    cbar = plt.colorbar(collection, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Temperature (°C)', fontsize=11)
    
    # Title
    max_temp = temp_region.max()
    ax.set_title(f"{city_name} - {title_suffix}\n{timestamp[:10]} | Regional max: {max_temp:.1f}°C", fontsize=12)
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Save
    plt.tight_layout()
    plt.savefig(output_dir / output_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_dir.name}/{output_name}")


# Load sorted CSVs
ctrl_sorted = pd.read_csv('city_max_temp_ctrl_sorted.csv')
ssp_sorted = pd.read_csv('city_max_temp_ssp585_sorted.csv')
diff_sorted = pd.read_csv('city_temp_diff_sorted.csv')

# Plot top 10 CTRL record holders
print("\n=== Top 10 Hottest Cities (CTRL 1950-1970) ===")
for i, row in ctrl_sorted.head(10).iterrows():
    city_safe = row['city_name'].replace(' ', '_').replace('/', '_')
    plot_regional_temp_native(
        ds_ctrl, int(row['time_idx']),
        row['city_lat'], row['city_lon'], row['city_name'],
        row['timestamp'],
        f"CTRL Record ({row['max_temp']:.1f}°C)",
        CTRL_DIR,
        f"ctrl_record_{i+1:02d}_{city_safe}_{row['max_temp']:.0f}C.png",
        vmin=30, vmax=55
    )

# Plot top 10 SSP585 record holders
print("\n=== Top 10 Hottest Cities (SSP585 2080-2100) ===")
for i, row in ssp_sorted.head(10).iterrows():
    city_safe = row['city_name'].replace(' ', '_').replace('/', '_')
    plot_regional_temp_native(
        ds_ssp, int(row['time_idx']),
        row['city_lat'], row['city_lon'], row['city_name'],
        row['timestamp'],
        f"SSP585 Record ({row['max_temp']:.1f}°C)",
        SSP_DIR,
        f"ssp585_record_{i+1:02d}_{city_safe}_{row['max_temp']:.0f}C.png",
        vmin=35, vmax=65
    )

# Plot top 10 by difference (biggest warming)
print("\n=== Top 10 Cities by Temperature Increase ===")
for i, row in diff_sorted.head(10).iterrows():
    city_safe = row['city_name'].replace(' ', '_').replace('/', '_')
    diff_val = row['temp_diff']
    # Plot CTRL
    plot_regional_temp_native(
        ds_ctrl, int(row['time_idx_ctrl']),
        row['city_lat'], row['city_lon'], row['city_name'],
        row['timestamp_ctrl'],
        f"CTRL ({row['max_temp_ctrl']:.1f}°C)",
        DIFF_DIR,
        f"diff_record_{i+1:02d}_{city_safe}_ctrl_{row['max_temp_ctrl']:.0f}C.png",
        vmin=10, vmax=50
    )
    # Plot SSP585
    plot_regional_temp_native(
        ds_ssp, int(row['time_idx_ssp']),
        row['city_lat'], row['city_lon'], row['city_name'],
        row['timestamp_ssp'],
        f"SSP585 ({row['max_temp_ssp']:.1f}°C) [+{diff_val:.1f}°C]",
        DIFF_DIR,
        f"diff_record_{i+1:02d}_{city_safe}_ssp585_{row['max_temp_ssp']:.0f}C.png",
        vmin=10, vmax=50
    )

print(f"\nDone! Plots saved to {BASE_DIR}")
