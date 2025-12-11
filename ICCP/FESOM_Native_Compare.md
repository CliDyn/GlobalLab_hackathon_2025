# FESOM Native Mesh Comparison Tool

Plot FESOM ocean variables on native triangular mesh (no interpolation) comparing different model resolutions side by side.

## Features

- Compares **TCO95** (CORE2, 245K triangles) with **TCO1279** (DART, 6.3M triangles)
- Plots on native unstructured mesh - no regridding or interpolation
- Date-based selection (not timestep index) for cross-model comparison
- Multiple predefined regions with automatic extent
- Dynamic colorbar range based on visible data
- Extensible for custom derived variables

## Usage

```bash
# Basic usage - Sea Surface Temperature
python fesom_native_compare.py -d 1950-07-15 -r korea -v temp

# Sea Surface Salinity
python fesom_native_compare.py -d 1950-04-11 -r baltic -v salt

# Different regions
python fesom_native_compare.py -d 1960-01-15 -r mediterranean -v temp
python fesom_native_compare.py -d 1950-07-15 -r arctic -v temp
python fesom_native_compare.py -d 1950-07-15 -r antarctic -v temp
```

## Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--date` | `-d` | Date to plot (YYYY-MM-DD) | 1950-04-11 |
| `--region` | `-r` | Region: global, korea, baltic, mediterranean, arctic, antarctic | global |
| `--variable` | `-v` | Variable: temp, salt | temp |
| `--output` | `-o` | Output filename | auto-generated |

## Available Regions

- **global** - Full global view
- **korea** - Korean Peninsula [120°E-135°E, 30°N-45°N]
- **baltic** - Baltic Sea [9°E-31°E, 53°N-66°N]
- **mediterranean** - Mediterranean Sea [6°W-37°E, 30°N-46°N]
- **arctic** - Arctic Ocean (North Polar Stereographic, >65°N)
- **antarctic** - Antarctic Ocean (South Polar Stereographic, <60°S)

## Adding Custom Variables

You can define derived variables in the `CUSTOM_VARIABLES` dictionary. Example for density anomaly:

```python
def compute_density_anomaly(data_dict):
    """Compute simple density anomaly from T and S."""
    T = data_dict['temp']
    S = data_dict['salt']
    alpha = 0.2  # thermal expansion
    beta = 0.8   # haline contraction
    return -alpha * (T - 10) + beta * (S - 35)

CUSTOM_VARIABLES['density'] = {
    'datasets': {'TCO95': 'ocean_daily', 'TCO1279': 'ocean_daily_temp'},
    'variables': ['temp1-31', 'salt1-31'],
    'compute': compute_density_anomaly,
    'units': 'kg/m³',
    'cmap': 'coolwarm',
    'long_name': 'Density Anomaly',
}
```

## Performance Limitations

⚠️ **Important**: This tool plots native triangular meshes which can be slow for large regions:

| Region | TCO95 triangles | TCO1279 triangles | Typical time |
|--------|-----------------|-------------------|--------------|
| Korea | ~1K | ~40K | Fast (<5s) |
| Baltic | ~2K | ~50K | Fast (<5s) |
| Mediterranean | ~3K | ~140K | Medium (~10s) |
| Arctic | ~28K | ~500K | Slow (~30s) |
| Antarctic | ~28K | ~580K | Slow (~30s) |
| Global | ~244K | ~6.3M | Very slow (minutes) |

**Tips for better performance:**
- Use regional views instead of global when possible
- The script automatically filters triangles to the visible region
- For global views, consider using regridded data instead

## Requirements

- Python 3.8+
- xarray, numpy, matplotlib, cartopy
- intake (for catalog access)
- tqdm (for progress bars)

## Data Access

Data is accessed via the GlobalLab catalog:
```
https://raw.githubusercontent.com/CliDyn/GlobalLab_hackathon_2025/refs/heads/main/catalog/main.yaml
```

Mesh files must be accessible on the HPC system.
