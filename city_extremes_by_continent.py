#!/usr/bin/env python3
"""
Bar plots comparing extreme weather between CTRL and SSP585 by continent.
Shows 19 largest cities per continent.
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
    'Ivory Coast': 'Africa', 'Djibouti': 'Africa', 'Egypt': 'Africa', 'Equatorial Guinea': 'Africa',
    'Eritrea': 'Africa', 'Eswatini': 'Africa', 'Ethiopia': 'Africa', 'Gabon': 'Africa',
    'Gambia': 'Africa', 'Ghana': 'Africa', 'Guinea': 'Africa', 'Guinea-Bissau': 'Africa',
    'Kenya': 'Africa', 'Lesotho': 'Africa', 'Liberia': 'Africa', 'Libya': 'Africa',
    'Madagascar': 'Africa', 'Malawi': 'Africa', 'Mali': 'Africa', 'Mauritania': 'Africa',
    'Mauritius': 'Africa', 'Morocco': 'Africa', 'Mozambique': 'Africa', 'Namibia': 'Africa',
    'Niger': 'Africa', 'Nigeria': 'Africa', 'Rwanda': 'Africa', 'Senegal': 'Africa',
    'Sierra Leone': 'Africa', 'Somalia': 'Africa', 'South Africa': 'Africa', 'South Sudan': 'Africa',
    'Sudan': 'Africa', 'Tanzania': 'Africa', 'Togo': 'Africa', 'Tunisia': 'Africa',
    'Uganda': 'Africa', 'Zambia': 'Africa', 'Zimbabwe': 'Africa', 'Western Sahara': 'Africa',
    "Côte d'Ivoire": 'Africa', 'Sao Tome and Principe': 'Africa', 'Seychelles': 'Africa',
    
    # Asia
    'Afghanistan': 'Asia', 'Armenia': 'Asia', 'Azerbaijan': 'Asia', 'Bahrain': 'Asia',
    'Bangladesh': 'Asia', 'Bhutan': 'Asia', 'Brunei': 'Asia', 'Cambodia': 'Asia',
    'China': 'Asia', 'Cyprus': 'Asia', 'Georgia': 'Asia', 'India': 'Asia',
    'Indonesia': 'Asia', 'Iran': 'Asia', 'Iraq': 'Asia', 'Israel': 'Asia',
    'Japan': 'Asia', 'Jordan': 'Asia', 'Kazakhstan': 'Asia', 'Kuwait': 'Asia',
    'Kyrgyzstan': 'Asia', 'Laos': 'Asia', 'Lebanon': 'Asia', 'Malaysia': 'Asia',
    'Maldives': 'Asia', 'Mongolia': 'Asia', 'Myanmar': 'Asia', 'Nepal': 'Asia',
    'North Korea': 'Asia', 'Oman': 'Asia', 'Pakistan': 'Asia', 'Palestine': 'Asia',
    'Philippines': 'Asia', 'Qatar': 'Asia', 'Saudi Arabia': 'Asia', 'Singapore': 'Asia',
    'South Korea': 'Asia', 'Sri Lanka': 'Asia', 'Syria': 'Asia', 'Taiwan': 'Asia',
    'Tajikistan': 'Asia', 'Thailand': 'Asia', 'Timor-Leste': 'Asia', 'Turkey': 'Asia',
    'Turkmenistan': 'Asia', 'United Arab Emirates': 'Asia', 'Uzbekistan': 'Asia',
    'Vietnam': 'Asia', 'Yemen': 'Asia',
    
    # Europe
    'Albania': 'Europe', 'Andorra': 'Europe', 'Austria': 'Europe', 'Belarus': 'Europe',
    'Belgium': 'Europe', 'Bosnia and Herzegovina': 'Europe', 'Bulgaria': 'Europe',
    'Croatia': 'Europe', 'Czech Republic': 'Europe', 'Czechia': 'Europe', 'Denmark': 'Europe',
    'Estonia': 'Europe', 'Finland': 'Europe', 'France': 'Europe', 'Germany': 'Europe',
    'Greece': 'Europe', 'Hungary': 'Europe', 'Iceland': 'Europe', 'Ireland': 'Europe',
    'Italy': 'Europe', 'Kosovo': 'Europe', 'Latvia': 'Europe', 'Liechtenstein': 'Europe',
    'Lithuania': 'Europe', 'Luxembourg': 'Europe', 'Malta': 'Europe', 'Moldova': 'Europe',
    'Monaco': 'Europe', 'Montenegro': 'Europe', 'Netherlands': 'Europe', 'North Macedonia': 'Europe',
    'Norway': 'Europe', 'Poland': 'Europe', 'Portugal': 'Europe', 'Romania': 'Europe',
    'Russia': 'Europe', 'San Marino': 'Europe', 'Serbia': 'Europe', 'Slovakia': 'Europe',
    'Slovenia': 'Europe', 'Spain': 'Europe', 'Sweden': 'Europe', 'Switzerland': 'Europe',
    'Ukraine': 'Europe', 'United Kingdom': 'Europe', 'Vatican City': 'Europe',
    
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
}

# Load data
print("Loading city data...")
temp_ctrl = pd.read_csv(f"{BASE_DIR}/city_max_temp_ctrl_sorted.csv")
temp_ssp = pd.read_csv(f"{BASE_DIR}/city_max_temp_ssp585_sorted.csv")
wind_ctrl = pd.read_csv(f"{BASE_DIR}/city_max_wind_ctrl_sorted.csv")
wind_ssp = pd.read_csv(f"{BASE_DIR}/city_max_wind_ssp585_sorted.csv")
precip_ctrl = pd.read_csv(f"{BASE_DIR}/city_max_precip_ctrl_sorted.csv")
precip_ssp = pd.read_csv(f"{BASE_DIR}/city_max_precip_ssp585_sorted.csv")

def merge_and_diff(ctrl, ssp, value_col, suffix):
    """Merge CTRL and SSP585 data, compute difference."""
    ctrl_sub = ctrl[['city_name', 'country', 'city_lat', 'city_lon', 'population', value_col]].copy()
    ctrl_sub.columns = ['city_name', 'country', 'lat', 'lon', 'population', f'{suffix}_ctrl']
    
    ssp_sub = ssp[['city_name', 'country', value_col]].copy()
    ssp_sub.columns = ['city_name', 'country', f'{suffix}_ssp']
    
    merged = ctrl_sub.merge(ssp_sub, on=['city_name', 'country'], how='inner')
    merged[f'{suffix}_diff'] = merged[f'{suffix}_ssp'] - merged[f'{suffix}_ctrl']
    
    # Add continent
    merged['continent'] = merged['country'].map(COUNTRY_TO_CONTINENT)
    return merged

temp_merged = merge_and_diff(temp_ctrl, temp_ssp, 'max_temp', 'temp')
wind_merged = merge_and_diff(wind_ctrl, wind_ssp, 'max_wind', 'wind')
precip_merged = merge_and_diff(precip_ctrl, precip_ssp, 'max_precip', 'precip')

# Check for unmapped countries
unmapped = temp_merged[temp_merged['continent'].isna()]['country'].unique()
if len(unmapped) > 0:
    print(f"Warning: {len(unmapped)} unmapped countries: {unmapped[:10]}")

print(f"Merged: temp={len(temp_merged)}, wind={len(wind_merged)}, precip={len(precip_merged)}")

# Divergent colormap: blue → light yellow → red
div_colors = ['#2166ac', '#67a9cf', '#d1e5f0', '#ffffbf', '#fddbc7', '#ef8a62', '#b2182b']
div_cmap = LinearSegmentedColormap.from_list('divergent', div_colors, N=256)

def create_continent_bar_plot(data, diff_col, title_base, ylabel, filename_base, ylim_min, ylim_max):
    """Create bar plots per continent with ALL cities, highlighting 19 largest."""
    
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania', 'Antarctica']
    
    for continent in continents:
        # Filter by continent - get ALL cities
        cont_data = data[data['continent'] == continent].copy()
        if len(cont_data) == 0:
            print(f"  No data for {continent}, skipping")
            continue
        
        # Get names of 19 largest cities by population for highlighting
        top_19_names = set(cont_data.nlargest(19, 'population')['city_name'].tolist())
        
        # Sort ALL cities by difference value for plotting
        data_sorted = cont_data.sort_values(diff_col).reset_index(drop=True)
        n = len(data_sorted)
        
        print(f"  {continent}: {n} cities, highlighting top 19")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 10))
        fig.subplots_adjust(right=0.88, top=0.88)
        
        # Symmetric log normalization
        vmax = max(abs(ylim_min), abs(ylim_max))
        norm = SymLogNorm(linthresh=0.1, vmin=-vmax, vmax=vmax)
        colors = div_cmap(norm(data_sorted[diff_col].values))
        
        # Plot ALL bars - highlight top 19 with black outline
        for i, (idx, row) in enumerate(data_sorted.iterrows()):
            is_top = row['city_name'] in top_19_names
            ax.bar(i, row[diff_col], color=colors[i], width=1.0,
                   edgecolor='black' if is_top else 'none',
                   linewidth=1.5 if is_top else 0,
                   zorder=2 if is_top else 1)
        
        # Zero line
        ax.axhline(y=0, color='black', linewidth=2, zorder=3)
        
        # Styling
        ax.set_xlim(-0.5, n + 0.5)
        ax.set_yscale('symlog', linthresh=0.1)
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title_base} - {continent}', fontweight='bold')
        
        # Colorbar
        sm = cm.ScalarMappable(cmap=div_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label(ylabel)
        cbar.ax.set_zorder(1)
        
        # Add labels for top 19 cities
        top_cities = data_sorted[data_sorted['city_name'].isin(top_19_names)].copy()
        texts = []
        for _, city in top_cities.iterrows():
            idx = data_sorted[data_sorted['city_name'] == city['city_name']].index[0]
            y_val = city[diff_col]
            txt = ax.text(idx, y_val, city['city_name'], fontsize=10, ha='center',
                         va='bottom' if y_val >= 0 else 'top', fontweight='bold',
                         zorder=10, bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                                              alpha=0.85, edgecolor='none'))
            texts.append(txt)
        
        # Adjust text positions
        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                       expand_points=(2.0, 2.0), expand_text=(1.5, 1.5),
                       force_text=(1.0, 2.0), iter_lim=500)
        
        # Grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        
        # Stats
        n_more = (data_sorted[diff_col] > 0).sum()
        n_less = (data_sorted[diff_col] < 0).sum()
        ax.text(0.02, 0.98, f'More extreme: {n_more} cities\nLess extreme: {n_less} cities',
                transform=ax.transAxes, fontsize=14, fontweight='bold', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        filename = f"{filename_base}_{continent.lower().replace(' ', '_')}.png"
        plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {filename}")

# Create plots for each variable
print("\nCreating temperature plots by continent...")
create_continent_bar_plot(
    temp_merged, 'temp_diff',
    'Change in Maximum Temperature (SSP585 − CTRL)',
    'Temperature Change (°C)',
    'city_extremes_temperature',
    ylim_min=-5, ylim_max=15
)

print("\nCreating wind plots by continent...")
create_continent_bar_plot(
    wind_merged, 'wind_diff',
    'Change in Maximum Wind Speed (SSP585 − CTRL)',
    'Wind Speed Change (km/h)',
    'city_extremes_wind',
    ylim_min=-50, ylim_max=50
)

print("\nCreating precipitation plots by continent...")
create_continent_bar_plot(
    precip_merged, 'precip_diff',
    'Change in Maximum Precipitation (SSP585 − CTRL)',
    'Precipitation Change (mm)',
    'city_extremes_precipitation',
    ylim_min=-100, ylim_max=200
)

print("\nDone!")
