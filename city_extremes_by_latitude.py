#!/usr/bin/env python3
"""
Plot percentage of cities with more/less extreme weather by 5° latitude bands.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
BASE_DIR = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025"
OUTPUT_DIR = f"{BASE_DIR}/plots"

# Load diff data (already has the difference computed)
print("Loading data...")
temp_diff = pd.read_csv(f"{BASE_DIR}/city_temp_diff_sorted.csv")
wind_diff = pd.read_csv(f"{BASE_DIR}/city_wind_diff_sorted.csv")
precip_diff = pd.read_csv(f"{BASE_DIR}/city_precip_diff_sorted.csv")

print(f"Cities: temp={len(temp_diff)}, wind={len(wind_diff)}, precip={len(precip_diff)}")

def create_lat_band_plot(data, diff_col, title, filename):
    """Create a bar plot showing % more/less extreme by 5° latitude bands."""
    
    # Create 10° latitude bins from -90 to 90
    bins = np.arange(-90, 100, 10)
    bin_labels = [f"{b}°" for b in bins[:-1]]
    
    # Assign each city to a latitude band
    data = data.copy()
    data['lat_band'] = pd.cut(data['city_lat'], bins=bins, labels=bin_labels, right=False)
    
    # Calculate stats per band (exclude bands with < 20 cities)
    stats = []
    for band in bin_labels:
        band_data = data[data['lat_band'] == band]
        n_total = len(band_data)
        if n_total < 20:
            continue
        n_more = (band_data[diff_col] > 0).sum()
        n_less = (band_data[diff_col] < 0).sum()
        pct_more = 100 * n_more / n_total
        pct_less = 100 * n_less / n_total
        lat_center = float(band.replace('°', '')) + 5
        stats.append({
            'band': band,
            'lat_center': lat_center,
            'n_total': n_total,
            'pct_more': pct_more,
            'pct_less': pct_less
        })
    
    stats_df = pd.DataFrame(stats)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    width = 8  # bar width in degrees
    
    # Plot stacked bars
    ax.bar(stats_df['lat_center'], stats_df['pct_more'], width=width, 
           label='More extreme', color='#d73027', alpha=0.8)
    ax.bar(stats_df['lat_center'], -stats_df['pct_less'], width=width,
           label='Less extreme', color='#4575b4', alpha=0.8)
    
    # Reference line at 0
    ax.axhline(y=0, color='black', linewidth=1)
    
    # Styling
    ax.set_xlim(-90, 90)
    ax.set_ylim(-100, 100)
    ax.set_xlabel('Latitude (°)', fontsize=14)
    ax.set_ylabel('Percentage of cities (%)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # X-axis ticks
    ax.set_xticks(np.arange(-90, 91, 15))
    ax.set_xticklabels([f'{x}°' for x in np.arange(-90, 91, 15)])
    
    # Y-axis: show absolute values
    ax.set_yticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
    ax.set_yticklabels(['100%', '75%', '50%', '25%', '0%', '25%', '50%', '75%', '100%'])
    
    # Add text labels for more/less
    ax.text(0.02, 0.98, 'More extreme →', transform=ax.transAxes, fontsize=12,
            va='top', ha='left', color='#d73027', fontweight='bold')
    ax.text(0.02, 0.02, '← Less extreme', transform=ax.transAxes, fontsize=12,
            va='bottom', ha='left', color='#4575b4', fontweight='bold')
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Add city count annotation
    total_more = (data[diff_col] > 0).sum()
    total_less = (data[diff_col] < 0).sum()
    ax.text(0.98, 0.98, f'Total: {total_more} more, {total_less} less\n({len(data)} cities)',
            transform=ax.transAxes, fontsize=11, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")

# Create plots
print("\nCreating latitude band plots...")

create_lat_band_plot(
    temp_diff, 'temp_diff',
    'Change in Maximum Temperature by Latitude (SSP585 − CTRL)',
    'city_extremes_latitude_temperature.png'
)

create_lat_band_plot(
    wind_diff, 'wind_diff',
    'Change in Maximum Wind Speed by Latitude (SSP585 − CTRL)',
    'city_extremes_latitude_wind.png'
)

create_lat_band_plot(
    precip_diff, 'precip_diff',
    'Change in Maximum Precipitation by Latitude (SSP585 − CTRL)',
    'city_extremes_latitude_precipitation.png'
)

print("\nDone!")
