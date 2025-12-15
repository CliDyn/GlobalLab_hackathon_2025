#!/usr/bin/env python3
"""
Busan Typhoon Movie Generator (Parallel Version)
Creates an animated comparison of typhoon events between CTRL and SSP585 scenarios.
Shows SST change (5-day difference), low clouds, and wind speed.
Uses multiprocessing for parallel frame generation.
"""

import intake
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for multiprocessing
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation
from matplotlib.patches import Rectangle, ConnectionPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
import imageio
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Paths
CATALOG_PATH = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/catalog/main.yaml"
ATMOS_GRID = "/work/ab0246/a270092/input/oasis/cy43r3/TCO1279-DART/grids.nc"
FESOM_MESH_DIAG = "/work/ba1264/a270210/model/input/fesom2/dart/fesom.mesh.diag.nc"
FESOM_MESH_NODES = "/work/ba1264/a270210/model/input/fesom2/dart/mesh_griddes_nodes.nc"
BM_IMG = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/blue_marble.jpg"
OUTPUT_DIR = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025/plots/busan_movie"
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")

# Movie settings
FPS = 10  # frames per second
N_FRAMES = 80  # number of frames (10 days at 8 timesteps/day = 80 frames, 3-hourly data)

# Colormaps
sst_colors = ['#1a2f98', '#346ab0', '#569bc5', '#8ad1d5', '#c9e8c8', '#cdcb7e', '#b38f43', '#985c28', '#792211']
sst_cmap = LinearSegmentedColormap.from_list('sst_ref', sst_colors, N=256)
wind_main_colors = ['#1a1a1a', '#3d3d00', '#8b8b00', '#c9a030', '#e0a0a0', '#f0c0c0', '#fae8e8']
wind_main_cmap = LinearSegmentedColormap.from_list('wind_main', wind_main_colors, N=256)
inset_colors = ['#000066', '#333399', '#6666cc', '#9999ff', '#cccc00', '#ffff00']
inset_cmap = LinearSegmentedColormap.from_list('inset', inset_colors, N=256)

# Location settings
busan_lat, busan_lon = 35.1, 129.0
radius = 12
inset_radius = 2.0
extent = [busan_lon-radius, busan_lon+radius, busan_lat-radius, busan_lat+radius]
extent_inset = [busan_lon-inset_radius, busan_lon+inset_radius, busan_lat-inset_radius, busan_lat+inset_radius]


def lonlat_to_pixel(lon, lat, w, h):
    x = int((lon + 180) / 360 * w)
    y = int((90 - lat) / 180 * h)
    return x, y


def get_mask(lon_arr, lat_arr, clat, clon, r, valid=None):
    m = ((lon_arr >= clon-r) & (lon_arr <= clon+r) & (lat_arr >= clat-r) & (lat_arr <= clat+r))
    if valid is not None:
        m &= valid
    return m


def plot_frame(ax1, ax2, ax1_inset, ax2_inset, fig,
               sst_ctrl_diff, lcc_ctrl, wind_ctrl, ctrl_time,
               sst_ssp_diff, lcc_ssp, wind_ssp, ssp_time,
               atmos_verts, atmos_idx, atmos_idx_inset, triang, fesom_idx,
               bm_region, bm_extent):
    """Plot a single frame with both CTRL and SSP585 panels."""
    
    def plot_main(ax, sst_diff_data, lcc_data, wind_data, title):
        ax.clear()
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        verts = atmos_verts[atmos_idx]
        
        # Background
        ax.imshow(bm_region, origin='upper', extent=bm_extent, transform=ccrs.PlateCarree(), zorder=0)
        
        # SST difference
        ax.tripcolor(triang, sst_diff_data[fesom_idx], cmap=sst_cmap,
                     vmin=-5, vmax=5, transform=ccrs.PlateCarree(), zorder=1)
        
        # Clouds - white with alpha
        cloud_vals = lcc_data[atmos_idx]
        cloud_colors_arr = np.zeros((len(atmos_idx), 4))
        cloud_colors_arr[:, 0] = 1.0
        cloud_colors_arr[:, 1] = 1.0
        cloud_colors_arr[:, 2] = 1.0
        cloud_colors_arr[:, 3] = cloud_vals * 0.7
        cloud_coll = PolyCollection(verts, facecolors=cloud_colors_arr,
                                     edgecolors='none', transform=ccrs.PlateCarree(), zorder=2)
        ax.add_collection(cloud_coll)
        
        # Wind
        wind_vals = wind_data[atmos_idx]
        max_wind = wind_vals.max()
        wind_norm_vals = np.clip((wind_vals - 20) / (50 - 20), 0, 1)
        wind_rgba = wind_main_cmap(wind_norm_vals)
        wind_alphas = np.where(wind_vals >= 20,
                               np.clip(0.6 + 0.35 * (wind_vals - 20) / 30, 0.6, 0.95), 0)
        wind_rgba[:, 3] = wind_alphas
        wind_coll = PolyCollection(verts, facecolors=wind_rgba,
                                    edgecolors='none', transform=ccrs.PlateCarree(), zorder=3)
        ax.add_collection(wind_coll)
        
        ax.coastlines(resolution='10m', linewidth=0.8, color='black', zorder=7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='#333333', zorder=7)
        ax.plot(busan_lon, busan_lat, 'w*', markersize=14, markeredgecolor='k',
                markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=10)
        
        rect = Rectangle((extent_inset[0], extent_inset[2]),
                          extent_inset[1]-extent_inset[0], extent_inset[3]-extent_inset[2],
                          fill=False, edgecolor='black', linewidth=2,
                          transform=ccrs.PlateCarree(), zorder=8)
        ax.add_patch(rect)
        
        ax.set_title(f'{title}\nMax Wind: {max_wind:.0f} m/s ({max_wind*3.6:.0f} km/h)', fontsize=12, fontweight='bold')
    
    def plot_inset(ax_in, wind_data, title_short):
        ax_in.clear()
        ax_in.set_extent(extent_inset, crs=ccrs.PlateCarree())
        verts = atmos_verts[atmos_idx_inset]
        
        wind_vals = wind_data[atmos_idx_inset]
        wind_norm = np.clip((wind_vals - 5) / 40, 0, 1)
        wind_rgba = inset_cmap(wind_norm)
        
        wind_coll = PolyCollection(verts, facecolors=wind_rgba,
                                    edgecolors='none', transform=ccrs.PlateCarree(), zorder=1)
        ax_in.add_collection(wind_coll)
        
        ax_in.coastlines(linewidth=0.5, color='black', zorder=5)
        ax_in.plot(busan_lon, busan_lat, 'k*', markersize=8, transform=ccrs.PlateCarree(), zorder=10)
        
        for spine in ax_in.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        
        ax_in.set_title(f'{title_short}\n{wind_vals.max():.0f} m/s', fontsize=9)
    
    plot_main(ax1, sst_ctrl_diff, lcc_ctrl, wind_ctrl, f'CTRL ({ctrl_time})')
    plot_main(ax2, sst_ssp_diff, lcc_ssp, wind_ssp, f'SSP585 ({ssp_time})')
    plot_inset(ax1_inset, wind_ctrl, 'CTRL')
    plot_inset(ax2_inset, wind_ssp, 'SSP585')


# Global variables for parallel processing (set in main, used in worker)
SHARED_DATA = {}


def generate_single_frame(frame_idx):
    """Generate a single frame - called in parallel by worker processes."""
    # Unpack shared data
    atmos_verts = SHARED_DATA['atmos_verts']
    atmos_idx = SHARED_DATA['atmos_idx']
    atmos_idx_inset = SHARED_DATA['atmos_idx_inset']
    fesom_idx = SHARED_DATA['fesom_idx']
    fesom_lon = SHARED_DATA['fesom_lon']
    fesom_lat = SHARED_DATA['fesom_lat']
    tri_regional = SHARED_DATA['tri_regional']
    bm_region = SHARED_DATA['bm_region']
    bm_extent = SHARED_DATA['bm_extent']
    ctrl_atmos_center = SHARED_DATA['ctrl_atmos_center']
    ssp_atmos_center = SHARED_DATA['ssp_atmos_center']
    ctrl_ocean_center = SHARED_DATA['ctrl_ocean_center']
    ssp_ocean_center = SHARED_DATA['ssp_ocean_center']
    start_offset = SHARED_DATA['start_offset']
    
    # Open datasets in this worker (each worker needs its own handle)
    gl_cat = intake.open_catalog(CATALOG_PATH)
    ds_atmos_ctrl = gl_cat['ICCP']['TCo1279-DART-1950C']['atmos']['native']['atmos_3h'].to_dask()
    ds_atmos_ssp = gl_cat['ICCP']['TCo1279-DART-2080C']['atmos']['native']['atmos_3h'].to_dask()
    ds_ocean_ctrl = gl_cat['ICCP']['TCo1279-DART-1950C']['ocean']['native']['ocean_daily_temp'].to_dask()
    ds_ocean_ssp = gl_cat['ICCP']['TCo1279-DART-2080C']['ocean']['native']['ocean_daily_temp'].to_dask()
    
    # FESOM triangulation (recreate in worker)
    triang = Triangulation(fesom_lon[fesom_idx], fesom_lat[fesom_idx], np.array(tri_regional))
    
    # Time indices
    ctrl_atmos_idx = ctrl_atmos_center + start_offset + frame_idx
    ssp_atmos_idx = ssp_atmos_center + start_offset + frame_idx
    ctrl_ocean_day = ctrl_ocean_center + (start_offset + frame_idx) // 8
    ssp_ocean_day = ssp_ocean_center + (start_offset + frame_idx) // 8
    ctrl_ocean_day_5d = ctrl_ocean_day - 5
    ssp_ocean_day_5d = ssp_ocean_day - 5
    
    # Load atmospheric data
    u_ctrl = ds_atmos_ctrl['10u'].isel(time_counter=ctrl_atmos_idx).values
    v_ctrl = ds_atmos_ctrl['10v'].isel(time_counter=ctrl_atmos_idx).values
    wind_ctrl = np.sqrt(u_ctrl**2 + v_ctrl**2)
    lcc_ctrl = ds_atmos_ctrl['lcc'].isel(time_counter=ctrl_atmos_idx).values
    
    u_ssp = ds_atmos_ssp['10u'].isel(time_counter=ssp_atmos_idx).values
    v_ssp = ds_atmos_ssp['10v'].isel(time_counter=ssp_atmos_idx).values
    wind_ssp = np.sqrt(u_ssp**2 + v_ssp**2)
    lcc_ssp = ds_atmos_ssp['lcc'].isel(time_counter=ssp_atmos_idx).values
    
    # Load SST and compute 5-day difference
    sst_ctrl_now = ds_ocean_ctrl['temp1-31'].isel(time=ctrl_ocean_day, nz1=0).values
    sst_ctrl_5d = ds_ocean_ctrl['temp1-31'].isel(time=ctrl_ocean_day_5d, nz1=0).values
    sst_ctrl_diff = sst_ctrl_now - sst_ctrl_5d
    
    sst_ssp_now = ds_ocean_ssp['temp1-31'].isel(time=ssp_ocean_day, nz1=0).values
    sst_ssp_5d = ds_ocean_ssp['temp1-31'].isel(time=ssp_ocean_day_5d, nz1=0).values
    sst_ssp_diff = sst_ssp_now - sst_ssp_5d
    
    # Get timestamps
    ctrl_time = ds_atmos_ctrl['time_counter'].isel(time_counter=ctrl_atmos_idx).values
    ssp_time = ds_atmos_ssp['time_counter'].isel(time_counter=ssp_atmos_idx).values
    ctrl_time_str = str(ctrl_time)[:10]
    ssp_time_str = str(ssp_time)[:10]
    
    # Create figure
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_axes([0.05, 0.15, 0.42, 0.75], projection=ccrs.PlateCarree())
    ax2 = fig.add_axes([0.53, 0.15, 0.42, 0.75], projection=ccrs.PlateCarree())
    ax1_inset = fig.add_axes([0.32, 0.62, 0.14, 0.26], projection=ccrs.PlateCarree())
    ax2_inset = fig.add_axes([0.80, 0.62, 0.14, 0.26], projection=ccrs.PlateCarree())
    
    # Plot
    plot_frame(ax1, ax2, ax1_inset, ax2_inset, fig,
               sst_ctrl_diff, lcc_ctrl, wind_ctrl, ctrl_time_str,
               sst_ssp_diff, lcc_ssp, wind_ssp, ssp_time_str,
               atmos_verts, atmos_idx, atmos_idx_inset, triang, fesom_idx,
               bm_region, bm_extent)
    
    # Connection lines
    for ax_main, ax_in in [(ax1, ax1_inset), (ax2, ax2_inset)]:
        con1 = ConnectionPatch(
            xyA=(extent_inset[0], extent_inset[3]), coordsA=ax_main.transData,
            xyB=(0, 0), coordsB=ax_in.transAxes, color='black', linewidth=1.5)
        fig.add_artist(con1)
        con2 = ConnectionPatch(
            xyA=(extent_inset[1], extent_inset[3]), coordsA=ax_main.transData,
            xyB=(1, 0), coordsB=ax_in.transAxes, color='black', linewidth=1.5)
        fig.add_artist(con2)
    
    # Colorbars
    cbar_ax1 = fig.add_axes([0.08, 0.08, 0.24, 0.025])
    sm_sst = plt.cm.ScalarMappable(cmap=sst_cmap, norm=Normalize(vmin=-5, vmax=5))
    fig.colorbar(sm_sst, cax=cbar_ax1, orientation='horizontal').set_label('SST Change (°C, now - 5d)', fontsize=10)
    
    cbar_ax2 = fig.add_axes([0.38, 0.08, 0.24, 0.025])
    sm_cloud = plt.cm.ScalarMappable(cmap='Greys_r', norm=Normalize(vmin=0, vmax=1.0))
    fig.colorbar(sm_cloud, cax=cbar_ax2, orientation='horizontal').set_label('Low Clouds (Fraction)', fontsize=10)
    
    cbar_ax3 = fig.add_axes([0.68, 0.08, 0.24, 0.025])
    sm_wind = plt.cm.ScalarMappable(cmap=wind_main_cmap, norm=Normalize(vmin=20, vmax=50))
    fig.colorbar(sm_wind, cax=cbar_ax3, orientation='horizontal').set_label('Wind Speed (m/s)', fontsize=10)
    
    fig.suptitle('Busan Typhoon — CTRL vs SSP585\nSST Change (5-day), Low Clouds, and Wind Speed',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save frame
    frame_path = os.path.join(FRAME_DIR, f'frame_{frame_idx:04d}.png')
    plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return frame_path


def init_worker(shared_data):
    """Initialize worker process with shared data."""
    global SHARED_DATA
    SHARED_DATA = shared_data


def main():
    global SHARED_DATA
    os.makedirs(FRAME_DIR, exist_ok=True)
    
    print("Loading Blue Marble image...")
    bm_full = np.array(Image.open(BM_IMG))
    bm_h, bm_w = bm_full.shape[:2]
    
    x0, y0 = lonlat_to_pixel(extent[0]-1, extent[3]+1, bm_w, bm_h)
    x1, y1 = lonlat_to_pixel(extent[1]+1, extent[2]-1, bm_w, bm_h)
    bm_region = bm_full[y0:y1, x0:x1]
    bm_extent_local = [extent[0]-1, extent[1]+1, extent[2]-1, extent[3]+1]
    
    print("Loading grid data...")
    atmos_grid = xr.open_dataset(ATMOS_GRID)
    fesom_diag = xr.open_dataset(FESOM_MESH_DIAG)
    fesom_nodes = xr.open_dataset(FESOM_MESH_NODES)
    
    clo = atmos_grid['A128.clo'].values.squeeze().T
    cla = atmos_grid['A128.cla'].values.squeeze().T
    atmos_verts = np.stack([clo, cla], axis=-1)
    atmos_lon = atmos_grid['A128.lon'].values.flatten()
    atmos_lat = atmos_grid['A128.lat'].values.flatten()
    atmos_valid = (np.max(clo, axis=1) - np.min(clo, axis=1)) < 180
    
    fesom_lon = fesom_nodes['lon'].values
    fesom_lat = fesom_nodes['lat'].values
    fesom_tri = fesom_diag['elem'].values.T - 1
    
    atmos_idx = np.where(get_mask(atmos_lon, atmos_lat, busan_lat, busan_lon, radius, atmos_valid))[0]
    atmos_idx_inset = np.where(get_mask(atmos_lon, atmos_lat, busan_lat, busan_lon, inset_radius, atmos_valid))[0]
    fesom_idx = np.where(get_mask(fesom_lon, fesom_lat, busan_lat, busan_lon, radius))[0]
    
    # FESOM triangulation indices
    node_map = {old: new for new, old in enumerate(fesom_idx)}
    tri_regional = [([node_map[n] for n in tri]) for tri in fesom_tri if all(n in node_map for n in tri)]
    
    # Time indices
    ctrl_atmos_center = 22367
    ssp_atmos_center = 28316
    ctrl_ocean_center = (1957-1950)*365 + 240
    ssp_ocean_center = (2089-2080)*365 + 252
    start_offset = -5 * 8
    
    # Shared data for workers
    SHARED_DATA = {
        'atmos_verts': atmos_verts,
        'atmos_idx': atmos_idx,
        'atmos_idx_inset': atmos_idx_inset,
        'fesom_idx': fesom_idx,
        'fesom_lon': fesom_lon,
        'fesom_lat': fesom_lat,
        'tri_regional': tri_regional,
        'bm_region': bm_region,
        'bm_extent': bm_extent_local,
        'ctrl_atmos_center': ctrl_atmos_center,
        'ssp_atmos_center': ssp_atmos_center,
        'ctrl_ocean_center': ctrl_ocean_center,
        'ssp_ocean_center': ssp_ocean_center,
        'start_offset': start_offset,
    }
    
    # Parallel frame generation
    n_workers = min(cpu_count(), 16)  # Cap at 16 workers
    print(f"Generating {N_FRAMES} frames using {n_workers} parallel workers...")
    
    with Pool(processes=n_workers, initializer=init_worker, initargs=(SHARED_DATA,)) as pool:
        frame_files = list(tqdm(
            pool.imap(generate_single_frame, range(N_FRAMES)),
            total=N_FRAMES,
            desc="Frames"
        ))
    
    # Sort frame files by index
    frame_files = sorted(frame_files)
    
    # Compile frames into video using OpenCV
    import cv2
    print(f"Compiling video at {FPS} fps...")
    output_path = os.path.join(OUTPUT_DIR, 'busan_typhoon.mp4')
    
    first_frame = cv2.imread(frame_files[0])
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
    
    for frame_path in tqdm(frame_files, desc="Writing video"):
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()
    
    print(f"Video saved: {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()
