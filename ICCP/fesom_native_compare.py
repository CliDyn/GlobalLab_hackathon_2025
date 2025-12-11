#!/usr/bin/env python
"""
FESOM native mesh comparison: TCO95 vs TCO1279
Plots any variable on native triangular mesh (no interpolation) side by side
Supports custom derived variables computed from multiple fields

Usage:
    python fesom_native_compare.py -d 1950-07-15 -r korea -v temp
    python fesom_native_compare.py -d 1950-07-15 -r baltic -v salt
    python fesom_native_compare.py -d 1950-07-15 -r mediterranean -v custom
"""

import argparse
import xarray as xr
import numpy as np
import intake
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from tqdm import tqdm

# =============================================================================
# CUSTOM VARIABLE DEFINITIONS
# Users can define derived variables here by specifying:
#   - 'datasets': list of dataset names to load
#   - 'variables': list of variable names from those datasets
#   - 'compute': function that takes dict of {varname: data} and returns derived field
#   - 'units': units string for colorbar
#   - 'cmap': colormap name
# =============================================================================
def compute_sst(data_dict):
    """Extract sea surface temperature (top level of temp)."""
    return data_dict['temp']

def compute_sss(data_dict):
    """Extract sea surface salinity (top level of salt)."""
    return data_dict['salt']

def compute_density_anomaly(data_dict):
    """Compute simple density anomaly from T and S (linear approximation)."""
    T = data_dict['temp']
    S = data_dict['salt']
    # Simple linear equation of state (approximate)
    rho0 = 1025.0  # reference density
    alpha = 0.2    # thermal expansion coeff (kg/m³/°C)
    beta = 0.8     # haline contraction coeff (kg/m³/psu)
    return -alpha * (T - 10) + beta * (S - 35)

CUSTOM_VARIABLES = {
    'temp': {
        'datasets': {'TCO95': 'ocean_daily', 'TCO1279': 'ocean_daily_temp'},
        'variables': ['temp1-31'],
        'compute': compute_sst,
        'units': '°C',
        'cmap': 'RdYlBu_r',
        'long_name': 'Sea Surface Temperature',
    },
    'salt': {
        'datasets': {'TCO95': 'ocean_daily', 'TCO1279': 'ocean_daily_salt'},
        'variables': ['salt1-31'],
        'compute': compute_sss,
        'units': 'PSU',
        'cmap': 'viridis',
        'long_name': 'Sea Surface Salinity',
    },
    # Example of a derived variable using multiple fields:
    # 'density': {
    #     'datasets': {'TCO95': ['ocean_daily', 'ocean_daily'], 
    #                  'TCO1279': ['ocean_daily_temp', 'ocean_daily_salt']},
    #     'variables': ['temp1-31', 'salt1-31'],
    #     'compute': compute_density_anomaly,
    #     'units': 'kg/m³',
    #     'cmap': 'coolwarm',
    #     'long_name': 'Density Anomaly',
    # },
}

# =============================================================================
# Parse arguments
# =============================================================================
parser = argparse.ArgumentParser(
    description='Compare FESOM variables across resolutions on native mesh',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  %(prog)s -d 1950-07-15 -r korea -v temp
  %(prog)s -d 1950-04-11 -r baltic -v salt
  %(prog)s -d 1960-01-15 -r arctic -v temp

Available variables: """ + ', '.join(CUSTOM_VARIABLES.keys())
)
parser.add_argument('--date', '-d', type=str, default='1950-04-11', 
                    help='Date to plot (YYYY-MM-DD format, default: 1950-04-11)')
parser.add_argument('--region', '-r', type=str, default='global', 
                    choices=['global', 'korea', 'baltic', 'mediterranean', 'arctic', 'antarctic'],
                    help='Region to plot (default: global)')
parser.add_argument('--variable', '-v', type=str, default='temp',
                    choices=list(CUSTOM_VARIABLES.keys()),
                    help='Variable to plot (default: temp)')
parser.add_argument('--output', '-o', type=str, default=None,
                    help='Output filename (default: auto-generated)')
args = parser.parse_args()

target_date = pd.Timestamp(args.date)
var_config = CUSTOM_VARIABLES[args.variable]
print(f"Target date: {target_date.strftime('%Y-%m-%d')}")
print(f"Variable: {var_config['long_name']}")

# =============================================================================
# Configuration for each resolution
# =============================================================================
configs = {
    'TCO95': {
        'expid': 'TCO95L91-CORE2-ctl1950d',
        'mesh_file': '/work/ab0995/ICCP_AWI_hackthon_2025/core2_griddes_elements.nc',
        'mesh_diag': '/work/ab0995/a270088/meshes/core_2.7/fesom.mesh.diag.nc',
        'title': 'TCO95 (CORE2)\n126K nodes, 245K triangles'
    },
    'TCO1279': {
        'expid': 'TCo1279-DART-1950C',
        'mesh_file': '/work/ba1264/a270210/model/input/fesom2/dart/dart_griddes_elems.nc',
        'mesh_diag': '/work/ba1264/a270210/model/input/fesom2/dart/fesom.mesh.diag.nc',
        'title': 'TCO1279 (DART)\n3.2M nodes, 6.3M triangles'
    }
}

# =============================================================================
# Region configurations
# =============================================================================
REGIONS = {
    'global': {'extent': None, 'title': 'Global', 'projection': 'platecarree'},
    'korea': {'extent': [120, 135, 30, 45], 'title': 'Korean Peninsula', 'projection': 'platecarree'},
    'baltic': {'extent': [9, 31, 53, 66], 'title': 'Baltic Sea', 'projection': 'platecarree'},
    'mediterranean': {'extent': [-6, 37, 30, 46], 'title': 'Mediterranean Sea', 'projection': 'platecarree'},
    'arctic': {'extent': None, 'title': 'Arctic Ocean', 'projection': 'north_polar', 'lat_limit': 65},
    'antarctic': {'extent': None, 'title': 'Antarctic Ocean', 'projection': 'south_polar', 'lat_limit': -60},
}

# =============================================================================
# Load catalog
# =============================================================================
print("Loading catalog...")
gl_cat = intake.open_catalog(
    "https://raw.githubusercontent.com/CliDyn/GlobalLab_hackathon_2025/refs/heads/main/catalog/main.yaml"
)

# =============================================================================
# Function to load and process data for one resolution
# =============================================================================
def load_resolution_data(res_name, config, var_config, target_date):
    """Load variable data and mesh for one resolution."""
    
    print(f"\n--- Loading {config['expid']} ---")
    
    # Get dataset name for this resolution
    dataset_name = var_config['datasets'][res_name]
    
    # Load data from catalog
    cat_data = gl_cat['ICCP'][config['expid']]['ocean']['native']
    data = cat_data[dataset_name].to_dask()
    
    # Select by date - handle different time coordinate structures
    time_vals = data['time'].values
    
    # Check if time is datetime or integer (broken kerchunk)
    if np.issubdtype(time_vals.dtype, np.datetime64):
        data_sel = data.sel(time=target_date, method='nearest')
        actual_time = pd.Timestamp(data_sel['time'].values)
    else:
        # Time is integer indices - assume daily from 1950-01-01
        start_date = pd.Timestamp('1950-01-01')
        idx = (target_date - start_date).days
        idx = max(0, min(idx, len(time_vals) - 1))
        data_sel = data.isel(time=idx)
        actual_time = start_date + pd.Timedelta(days=idx)
        print(f"  (Using day index: {idx})")
    print(f"  Selected time: {actual_time}")
    
    # Load all required variables
    data_dict = {}
    for var_name in var_config['variables']:
        print(f"  Extracting {var_name}...")
        var_data = data_sel[var_name]
        
        # Select surface level if 3D
        if 'nz1' in var_data.dims:
            var_data = var_data.isel(nz1=0)
        elif 'nz' in var_data.dims:
            var_data = var_data.isel(nz=0)
        
        data_dict[var_name.replace('1-31', '')] = var_data.compute().values
    
    # Compute derived variable
    field = var_config['compute'](data_dict)
    print(f"  Field shape: {field.shape}, range: [{np.nanmin(field):.2f}, {np.nanmax(field):.2f}]")
    
    # Load mesh (triangle corners)
    print(f"  Loading mesh...")
    mesh = xr.open_dataset(config['mesh_file'])
    lon_bnds = mesh['lon_bnds'].values
    lat_bnds = mesh['lat_bnds'].values
    
    print(f"  Triangles: {len(lon_bnds)}")
    
    # Filter wrap-around triangles
    lon_diff = np.max(lon_bnds, axis=1) - np.min(lon_bnds, axis=1)
    valid_mask = lon_diff < 180
    lon_bnds = lon_bnds[valid_mask]
    lat_bnds = lat_bnds[valid_mask]
    
    # Build triangle vertices: (n_triangles, 3, 2)
    tri_verts = np.stack([lon_bnds, lat_bnds], axis=-1)
    
    # Handle node vs element centered data
    n_data = len(field)
    n_elem = mesh.dims['ntriags']
    n_nodes = mesh.dims['ncells']
    
    if n_data == n_elem:
        field_triangles = field[valid_mask]
    elif n_data == n_nodes:
        mesh_diag = xr.open_dataset(config['mesh_diag'])
        if 'face_nodes' in mesh_diag:
            face_nodes = mesh_diag['face_nodes'].values.T - 1
        else:
            face_nodes = mesh_diag['elem'].values.T - 1
        face_nodes_filtered = face_nodes[valid_mask]
        field_triangles = np.mean(field[face_nodes_filtered], axis=1)
    else:
        raise ValueError(f"Data size {n_data} doesn't match elements {n_elem} or nodes {n_nodes}")
    
    lon_centers = mesh['lon'].values[valid_mask]
    lat_centers = mesh['lat'].values[valid_mask]
    
    return tri_verts, field_triangles, lon_centers, lat_centers, actual_time, config['title']

# =============================================================================
# Load data for all resolutions
# =============================================================================
results = {}
for res_name, config in tqdm(configs.items(), desc="Loading models"):
    try:
        results[res_name] = load_resolution_data(res_name, config, var_config, target_date)
    except Exception as e:
        print(f"  ERROR loading {res_name}: {e}")
        import traceback
        traceback.print_exc()
        results[res_name] = None

# =============================================================================
# Create comparison plot
# =============================================================================
print("\nCreating plot...")

region_cfg = REGIONS[args.region]
fig_title = region_cfg['title']

# Determine projection
if region_cfg['projection'] == 'north_polar':
    proj = ccrs.NorthPolarStereo()
elif region_cfg['projection'] == 'south_polar':
    proj = ccrs.SouthPolarStereo()
else:
    proj = ccrs.PlateCarree()

n_plots = sum(1 for r in results.values() if r is not None)
if n_plots == 0:
    print("ERROR: No data loaded successfully!")
    exit(1)

fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 7), 
                          subplot_kw={'projection': proj})

if n_plots == 1:
    axes = [axes]

# Collect all visible values to compute shared colorbar range
all_visible_vals = []

plot_idx = 0
collections = []
for res_name, result in results.items():
    if result is None:
        continue
    
    tri_verts, field_triangles, lon_centers, lat_centers, actual_time, title = result
    ax = axes[plot_idx]
    
    # Filter triangles to visible region (CRITICAL for performance!)
    if region_cfg['extent']:
        lon_min, lon_max, lat_min, lat_max = region_cfg['extent']
        visible_mask = ((lon_centers >= lon_min - 1) & (lon_centers <= lon_max + 1) &
                        (lat_centers >= lat_min - 1) & (lat_centers <= lat_max + 1))
    elif region_cfg['projection'] == 'north_polar':
        visible_mask = lat_centers >= region_cfg['lat_limit'] - 2
    elif region_cfg['projection'] == 'south_polar':
        visible_mask = lat_centers <= region_cfg['lat_limit'] + 2
    else:
        visible_mask = np.ones(len(field_triangles), dtype=bool)
    
    tri_verts_vis = tri_verts[visible_mask]
    field_vis = field_triangles[visible_mask]
    
    print(f"  {res_name}: {np.sum(visible_mask):,} / {len(visible_mask):,} triangles in view")
    
    visible_vals = field_vis[~np.isnan(field_vis)]
    if len(visible_vals) > 0:
        all_visible_vals.extend(visible_vals)
    
    # Create PolyCollection with ONLY visible triangles
    collection = PolyCollection(
        tri_verts_vis,
        array=field_vis,
        cmap=var_config['cmap'],
        edgecolors='none',
        linewidths=0,
        transform=ccrs.PlateCarree(),
    )
    ax.add_collection(collection)
    collections.append(collection)
    
    # Set extent
    if region_cfg['extent']:
        ax.set_extent(region_cfg['extent'], crs=ccrs.PlateCarree())
    elif region_cfg['projection'] in ['north_polar', 'south_polar']:
        ax.set_extent([-180, 180, region_cfg['lat_limit'], 
                       90 if 'north' in region_cfg['projection'] else -90], 
                      crs=ccrs.PlateCarree())
    else:
        ax.set_global()
    
    ax.coastlines(linewidth=0.5, color='black')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    if args.region in ['korea', 'baltic', 'mediterranean']:
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    
    ax.set_title(title, fontsize=11)
    plot_idx += 1

# Compute colorbar range from visible data
if len(all_visible_vals) > 0:
    vmin = np.percentile(all_visible_vals, 1)
    vmax = np.percentile(all_visible_vals, 99)
    vmin = np.floor(vmin * 10) / 10
    vmax = np.ceil(vmax * 10) / 10
else:
    vmin, vmax = 0, 1

print(f"Colorbar range: {vmin:.1f} to {vmax:.1f} {var_config['units']}")

for coll in collections:
    coll.set_clim(vmin, vmax)

# Layout
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.85, wspace=0.02)

cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
cbar = fig.colorbar(collections[0], cax=cbar_ax, orientation='horizontal')
cbar.set_label(f"{var_config['long_name']} ({var_config['units']})", fontsize=11)

time_str = actual_time.strftime('%Y-%m-%d')
fig.suptitle(f"FESOM {var_config['long_name']} - {fig_title}\n{time_str}", fontsize=13, y=0.92)

if args.output:
    output_file = args.output
else:
    output_file = f'fesom_{args.variable}_{args.region}.png'
    
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlot saved to: {output_file}")
