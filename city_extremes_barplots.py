#!/usr/bin/env python3
"""
Bar plots comparing extreme weather between CTRL and SSP585 for all cities.
- Divergent colorbar: blue (less extreme) → yellow (no change) → red (more extreme)
- Highlights 19 most populous cities per continent with names
- Simple bars with zero line
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
import matplotlib.cm as cm
from adjustText import adjust_text

# Increase default font sizes
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# Paths
BASE_DIR = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025"
OUTPUT_DIR = f"{BASE_DIR}/plots"

# Country to continent mapping
COUNTRY_TO_CONTINENT = {
    # Africa
    'Algeria': 'Africa', 'Angola': 'Africa', 'Benin': 'Africa', 'Botswana': 'Africa',
    'Burkina Faso': 'Africa', 'Burundi': 'Africa', 'Cameroon': 'Africa', 'Cape Verde': 'Africa',
    'Central African Republic': 'Africa', 'Chad': 'Africa', 'Comoros': 'Africa',
    'Democratic Republic of the Congo': 'Africa', 'Republic of the Congo': 'Africa',
    'Congo (Kinshasa)': 'Africa', 'Congo (Brazzaville)': 'Africa',
    'Ivory Coast': 'Africa', "Côte d'Ivoire": 'Africa', 'Djibouti': 'Africa', 'Egypt': 'Africa',
    'Equatorial Guinea': 'Africa', 'Eritrea': 'Africa', 'Eswatini': 'Africa', 'eSwatini': 'Africa',
    'Ethiopia': 'Africa', 'Gabon': 'Africa', 'Gambia': 'Africa', 'The Gambia': 'Africa',
    'Ghana': 'Africa', 'Guinea': 'Africa', 'Guinea-Bissau': 'Africa', 'Guinea Bissau': 'Africa',
    'Kenya': 'Africa', 'Lesotho': 'Africa', 'Liberia': 'Africa', 'Libya': 'Africa',
    'Madagascar': 'Africa', 'Malawi': 'Africa', 'Mali': 'Africa', 'Mauritania': 'Africa',
    'Mauritius': 'Africa', 'Morocco': 'Africa', 'Mozambique': 'Africa', 'Namibia': 'Africa',
    'Niger': 'Africa', 'Nigeria': 'Africa', 'Rwanda': 'Africa', 'Senegal': 'Africa',
    'Sierra Leone': 'Africa', 'Somalia': 'Africa', 'Somaliland': 'Africa',
    'South Africa': 'Africa', 'South Sudan': 'Africa', 'Sudan': 'Africa', 'Tanzania': 'Africa',
    'Togo': 'Africa', 'Tunisia': 'Africa', 'Uganda': 'Africa', 'Zambia': 'Africa',
    'Zimbabwe': 'Africa', 'Western Sahara': 'Africa', 'Sao Tome and Principe': 'Africa',
    'Seychelles': 'Africa',
    
    # Asia
    'Afghanistan': 'Asia', 'Armenia': 'Asia', 'Azerbaijan': 'Asia', 'Bahrain': 'Asia',
    'Bangladesh': 'Asia', 'Bhutan': 'Asia', 'Brunei': 'Asia', 'Cambodia': 'Asia',
    'China': 'Asia', 'Hong Kong S.A.R.': 'Asia', 'Macau S.A.R': 'Asia',
    'Cyprus': 'Asia', 'Georgia': 'Asia', 'India': 'Asia', 'Indonesia': 'Asia',
    'Iran': 'Asia', 'Iraq': 'Asia', 'Israel': 'Asia', 'Japan': 'Asia', 'Jordan': 'Asia',
    'Kazakhstan': 'Asia', 'Kuwait': 'Asia', 'Kyrgyzstan': 'Asia', 'Laos': 'Asia',
    'Lebanon': 'Asia', 'Malaysia': 'Asia', 'Maldives': 'Asia', 'Mongolia': 'Asia',
    'Myanmar': 'Asia', 'Nepal': 'Asia', 'North Korea': 'Asia', 'Oman': 'Asia',
    'Pakistan': 'Asia', 'Palestine': 'Asia', 'Philippines': 'Asia', 'Qatar': 'Asia',
    'Saudi Arabia': 'Asia', 'Singapore': 'Asia', 'South Korea': 'Asia', 'Sri Lanka': 'Asia',
    'Syria': 'Asia', 'Taiwan': 'Asia', 'Tajikistan': 'Asia', 'Thailand': 'Asia',
    'Timor-Leste': 'Asia', 'Turkey': 'Asia', 'Turkmenistan': 'Asia',
    'United Arab Emirates': 'Asia', 'Uzbekistan': 'Asia', 'Vietnam': 'Asia', 'Yemen': 'Asia',
    
    # Europe
    'Albania': 'Europe', 'Andorra': 'Europe', 'Austria': 'Europe', 'Belarus': 'Europe',
    'Belgium': 'Europe', 'Bosnia and Herzegovina': 'Europe', 'Bulgaria': 'Europe',
    'Croatia': 'Europe', 'Czech Republic': 'Europe', 'Czechia': 'Europe', 'Denmark': 'Europe',
    'Estonia': 'Europe', 'Finland': 'Europe', 'France': 'Europe', 'Germany': 'Europe',
    'Greece': 'Europe', 'Hungary': 'Europe', 'Iceland': 'Europe', 'Ireland': 'Europe',
    'Italy': 'Europe', 'Kosovo': 'Europe', 'Latvia': 'Europe', 'Liechtenstein': 'Europe',
    'Lithuania': 'Europe', 'Luxembourg': 'Europe', 'Malta': 'Europe', 'Moldova': 'Europe',
    'Monaco': 'Europe', 'Montenegro': 'Europe', 'Netherlands': 'Europe',
    'North Macedonia': 'Europe', 'Norway': 'Europe', 'Poland': 'Europe', 'Portugal': 'Europe',
    'Romania': 'Europe', 'Russia': 'Europe', 'San Marino': 'Europe', 'Serbia': 'Europe',
    'Slovakia': 'Europe', 'Slovenia': 'Europe', 'Spain': 'Europe', 'Sweden': 'Europe',
    'Switzerland': 'Europe', 'Ukraine': 'Europe', 'United Kingdom': 'Europe',
    'Vatican City': 'Europe', 'Vatican': 'Europe', 'Gibraltar': 'Europe',
    
    # North America
    'Antigua and Barbuda': 'North America', 'Bahamas': 'North America', 'Barbados': 'North America',
    'Belize': 'North America', 'Canada': 'North America', 'Costa Rica': 'North America',
    'Cuba': 'North America', 'Dominica': 'North America', 'Dominican Republic': 'North America',
    'El Salvador': 'North America', 'Grenada': 'North America', 'Guatemala': 'North America',
    'Haiti': 'North America', 'Honduras': 'North America', 'Jamaica': 'North America',
    'Mexico': 'North America', 'Nicaragua': 'North America', 'Panama': 'North America',
    'Saint Kitts and Nevis': 'North America', 'Saint Lucia': 'North America',
    'Saint Vincent and the Grenadines': 'North America', 'Trinidad and Tobago': 'North America',
    'United States of America': 'North America', 'United States': 'North America',
    'Puerto Rico': 'North America',
    
    # South America
    'Argentina': 'South America', 'Bolivia': 'South America', 'Brazil': 'South America',
    'Chile': 'South America', 'Colombia': 'South America', 'Ecuador': 'South America',
    'Guyana': 'South America', 'Paraguay': 'South America', 'Peru': 'South America',
    'Suriname': 'South America', 'Uruguay': 'South America', 'Venezuela': 'South America',
    
    # Oceania
    'Australia': 'Oceania', 'Fiji': 'Oceania', 'Kiribati': 'Oceania',
    'Marshall Islands': 'Oceania', 'Micronesia': 'Oceania', 'Nauru': 'Oceania',
    'New Zealand': 'Oceania', 'Palau': 'Oceania', 'Papua New Guinea': 'Oceania',
    'Samoa': 'Oceania', 'Solomon Islands': 'Oceania', 'Tonga': 'Oceania',
    'Tuvalu': 'Oceania', 'Vanuatu': 'Oceania',
    
    # Antarctica
    'Antarctica': 'Antarctica',
}

# Load data
print("Loading city data...")
temp_ctrl = pd.read_csv(f"{BASE_DIR}/city_max_temp_ctrl_sorted.csv")
temp_ssp = pd.read_csv(f"{BASE_DIR}/city_max_temp_ssp585_sorted.csv")
wind_ctrl = pd.read_csv(f"{BASE_DIR}/city_max_wind_ctrl_sorted.csv")
wind_ssp = pd.read_csv(f"{BASE_DIR}/city_max_wind_ssp585_sorted.csv")
precip_ctrl = pd.read_csv(f"{BASE_DIR}/city_max_precip_ctrl_sorted.csv")
precip_ssp = pd.read_csv(f"{BASE_DIR}/city_max_precip_ssp585_sorted.csv")

print(f"Cities: temp={len(temp_ctrl)}, wind={len(wind_ctrl)}, precip={len(precip_ctrl)}")

# Merge CTRL and SSP585 data
def merge_and_diff(ctrl, ssp, value_col, suffix):
    """Merge CTRL and SSP585 data, compute difference, add continent."""
    ctrl_sub = ctrl[['city_name', 'country', 'city_lat', 'city_lon', 'population', value_col]].copy()
    ctrl_sub.columns = ['city_name', 'country', 'lat', 'lon', 'population', f'{suffix}_ctrl']
    
    ssp_sub = ssp[['city_name', 'country', value_col]].copy()
    ssp_sub.columns = ['city_name', 'country', f'{suffix}_ssp']
    
    merged = ctrl_sub.merge(ssp_sub, on=['city_name', 'country'], how='inner')
    merged[f'{suffix}_diff'] = merged[f'{suffix}_ssp'] - merged[f'{suffix}_ctrl']
    
    # Add continent
    merged['continent'] = merged['country'].map(COUNTRY_TO_CONTINENT)
    return merged

def get_top_cities_per_continent(data, n=19):
    """Get set of city names that are in top n by population per continent."""
    top_cities = set()
    for continent in data['continent'].dropna().unique():
        cont_data = data[data['continent'] == continent]
        top_n = cont_data.nlargest(n, 'population')['city_name'].tolist()
        top_cities.update(top_n)
    return top_cities

temp_merged = merge_and_diff(temp_ctrl, temp_ssp, 'max_temp', 'temp')
wind_merged = merge_and_diff(wind_ctrl, wind_ssp, 'max_wind', 'wind')
precip_merged = merge_and_diff(precip_ctrl, precip_ssp, 'max_precip', 'precip')

print(f"Merged: temp={len(temp_merged)}, wind={len(wind_merged)}, precip={len(precip_merged)}")

# Divergent colormap: blue → light yellow → red
div_colors = ['#2166ac', '#67a9cf', '#d1e5f0', '#ffffbf', '#fddbc7', '#ef8a62', '#b2182b']
div_cmap = LinearSegmentedColormap.from_list('divergent', div_colors, N=256)

def create_bar_plot(data, diff_col, title, ylabel, filename, ylim_min, ylim_max):
    """Create a bar plot with symlog colorbar and non-overlapping labels."""
    # Sort by difference value
    data_sorted = data.sort_values(diff_col).reset_index(drop=True)
    n = len(data_sorted)
    
    # Create figure with extra right margin and more height for labels
    fig, ax = plt.subplots(figsize=(20, 10))
    fig.subplots_adjust(right=0.88, top=0.88)  # Leave room for labels
    
    # Symmetric log normalization using y-limits for colorbar
    vmax = max(abs(ylim_min), abs(ylim_max))
    norm = SymLogNorm(linthresh=0.1, vmin=-vmax, vmax=vmax)
    colors = div_cmap(norm(data_sorted[diff_col].values))
    
    # Find top 19 cities per continent for special treatment
    big_city_names = get_top_cities_per_continent(data_sorted, n=19)
    print(f"  Highlighting {len(big_city_names)} cities (19 per continent)")
    
    # Plot bars - regular cities first (zorder=1), mega cities with outline (zorder=2)
    x = np.arange(n)
    for i, (idx, row) in enumerate(data_sorted.iterrows()):
        is_mega = row['city_name'] in big_city_names
        ax.bar(i, row[diff_col], color=colors[i], width=1.0,
               edgecolor='black' if is_mega else 'none',
               linewidth=1.5 if is_mega else 0,
               zorder=2 if is_mega else 1)
    
    # Zero line
    ax.axhline(y=0, color='black', linewidth=2, zorder=3)
    
    # Styling - symlog y-axis (set BEFORE colorbar and labels)
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_yscale('symlog', linthresh=0.1)
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    
    # Add colorbar BEFORE labels so labels are on top
    sm = cm.ScalarMappable(cmap=div_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label(ylabel)
    cbar.ax.set_zorder(1)  # Lower zorder for colorbar
    
    # Find top cities per continent and create text objects for adjustText
    big_cities = data_sorted[data_sorted['city_name'].isin(big_city_names)].copy()
    
    # Sort big cities by x position (index) and stagger heights
    big_cities_sorted = big_cities.copy()
    big_cities_sorted['x_idx'] = big_cities_sorted['city_name'].apply(
        lambda name: data_sorted[data_sorted['city_name'] == name].index[0]
    )
    big_cities_sorted = big_cities_sorted.sort_values('x_idx')
    
    texts = []
    prev_x = -1000
    stagger_level = 0
    for i, (_, city) in enumerate(big_cities_sorted.iterrows()):
        idx = city['x_idx']
        y_val = city[diff_col]
        
        # Stagger labels if they're close together
        if idx - prev_x < n * 0.08:  # Within 8% of plot width
            stagger_level = (stagger_level + 1) % 3
        else:
            stagger_level = 0
        prev_x = idx
        
        # Offset y position based on stagger level
        y_offset_mult = 1.0 + stagger_level * 0.4
        
        txt = ax.text(
            idx, y_val, city['city_name'],
            fontsize=10,
            ha='center',
            va='bottom' if y_val >= 0 else 'top',
            fontweight='bold',
            color='black',
            zorder=10,  # Labels on top
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.85, edgecolor='none')
        )
        texts.append(txt)
    
    # Adjust text positions to avoid overlap - allow full movement
    if texts:
        adjust_text(texts, ax=ax, 
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                    expand_points=(3.0, 3.0), 
                    expand_text=(3.0, 3.0),
                    force_text=(3.0, 4.0),
                    force_points=(1.0, 1.0),
                    iter_lim=3000,
                    avoid_self=True)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Stats annotation
    n_more = (data_sorted[diff_col] > 0).sum()
    n_less = (data_sorted[diff_col] < 0).sum()
    ax.text(0.02, 0.98, f'More extreme: {n_more} cities\nLess extreme: {n_less} cities',
            transform=ax.transAxes, fontsize=16, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/{filename}")

# Create plots
print("\nCreating temperature bar plot...")
create_bar_plot(
    temp_merged, 'temp_diff',
    'Change in Maximum Temperature (SSP585 − CTRL)',
    'Temperature Change (°C)',
    'city_extremes_temperature.png',
    ylim_min=-5, ylim_max=15
)

print("\nCreating wind bar plot...")
create_bar_plot(
    wind_merged, 'wind_diff',
    'Change in Maximum Wind Speed (SSP585 − CTRL)',
    'Wind Speed Change (km/h)',
    'city_extremes_wind.png',
    ylim_min=-50, ylim_max=50
)

print("\nCreating precipitation bar plot...")
create_bar_plot(
    precip_merged, 'precip_diff',
    'Change in Maximum Precipitation (SSP585 − CTRL)',
    'Precipitation Change (mm)',
    'city_extremes_precipitation.png',
    ylim_min=-100, ylim_max=200
)

print("\nDone!")
