#!/usr/bin/env python3
"""Add continent column to all city CSV files."""

import pandas as pd
import os

BASE_DIR = "/work/ab0246/a270092/software/GlobalLab_hackathon_2025"

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
    'Seychelles': 'Africa', 'Réunion': 'Africa', 'Mayotte': 'Africa',
    
    # Asia
    'Afghanistan': 'Asia', 'Armenia': 'Asia', 'Azerbaijan': 'Asia', 'Bahrain': 'Asia',
    'Bangladesh': 'Asia', 'Bhutan': 'Asia', 'Brunei': 'Asia', 'Cambodia': 'Asia',
    'China': 'Asia', 'Hong Kong S.A.R.': 'Asia', 'Macau S.A.R': 'Asia', 'Hong Kong': 'Asia',
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
    'Puerto Rico': 'North America', 'Greenland': 'North America',
    
    # South America
    'Argentina': 'South America', 'Bolivia': 'South America', 'Brazil': 'South America',
    'Chile': 'South America', 'Colombia': 'South America', 'Ecuador': 'South America',
    'Guyana': 'South America', 'Paraguay': 'South America', 'Peru': 'South America',
    'Suriname': 'South America', 'Uruguay': 'South America', 'Venezuela': 'South America',
    'French Guiana': 'South America',
    
    # Oceania
    'Australia': 'Oceania', 'Fiji': 'Oceania', 'Kiribati': 'Oceania',
    'Marshall Islands': 'Oceania', 'Micronesia': 'Oceania', 'Nauru': 'Oceania',
    'New Zealand': 'Oceania', 'Palau': 'Oceania', 'Papua New Guinea': 'Oceania',
    'Samoa': 'Oceania', 'Solomon Islands': 'Oceania', 'Tonga': 'Oceania',
    'Tuvalu': 'Oceania', 'Vanuatu': 'Oceania', 'New Caledonia': 'Oceania',
    'French Polynesia': 'Oceania', 'Guam': 'Oceania',
    
    # Antarctica
    'Antarctica': 'Antarctica',
    
    # Additional mappings
    'East Timor': 'Asia', 'Northern Cyprus': 'Europe', 'The Bahamas': 'North America',
    'Cayman Islands': 'North America', 'Bermuda': 'North America', 'Turks and Caicos Islands': 'North America',
    'Faroe Islands': 'Europe', 'Isle of Man': 'Europe', 'Jersey': 'Europe', 'Guernsey': 'Europe',
    'Svalbard and Jan Mayen Islands': 'Europe', 'Falkland Islands': 'South America',
    'South Georgia and the Islands': 'South America',
    'Kermadec Islands': 'Oceania', 'Cook Islands': 'Oceania', 'American Samoa': 'Oceania',
    'Niue': 'Oceania', 'Tokelau': 'Oceania', 'Chatham Islands': 'Oceania',
    'Northern Mariana Islands': 'Oceania', 'Federated States of Micronesia': 'Oceania',
    'French Southern Territories': 'Antarctica', 'Heard Island and McDonald Islands': 'Antarctica',
    'United States Virgin Islands': 'North America', 'Curacao': 'North America', 'Aruba': 'North America',
    'Necw Zealand': 'Oceania', 'Aland': 'Europe',
}

# List of CSV files to process
csv_files = [
    'city_max_precip_ctrl_all.csv',
    'city_max_precip_ctrl_sorted.csv',
    'city_max_precip_ssp585_all.csv',
    'city_max_precip_ssp585_sorted.csv',
    'city_max_temp_ctrl_all.csv',
    'city_max_temp_ctrl_sorted.csv',
    'city_max_temp_ssp585_all.csv',
    'city_max_temp_ssp585_sorted.csv',
    'city_max_wind_ctrl_all.csv',
    'city_max_wind_ctrl_sorted.csv',
    'city_max_wind_ssp585_all.csv',
    'city_max_wind_ssp585_sorted.csv',
    'city_precip_diff_sorted.csv',
    'city_temp_diff_sorted.csv',
    'city_temp_diff_top10_countries.csv',
    'city_wind_diff_sorted.csv',
]

print("Adding continent to CSV files...")

for filename in csv_files:
    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  {filename}: NOT FOUND, skipping")
        continue
    
    df = pd.read_csv(filepath)
    
    # Remove existing continent column to re-add with updated mappings
    if 'continent' in df.columns:
        df = df.drop(columns=['continent'])
    
    # Add continent column
    if 'country' in df.columns:
        df['continent'] = df['country'].map(COUNTRY_TO_CONTINENT)
        
        # Check for unmapped countries
        unmapped = df[df['continent'].isna()]['country'].unique()
        if len(unmapped) > 0:
            print(f"  {filename}: WARNING - {len(unmapped)} unmapped countries: {list(unmapped)[:5]}")
        
        # Save back
        df.to_csv(filepath, index=False)
        print(f"  {filename}: added continent ({df['continent'].notna().sum()}/{len(df)} mapped)")
    else:
        print(f"  {filename}: no 'country' column, skipping")

print("\nDone!")
