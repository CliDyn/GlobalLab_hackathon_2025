import intake
import numpy as np
import time as timer
import dask

def benchmark_max(data):
    """Benchmark finding max of data"""
    start = timer.time()
    result = data.max().compute()
    elapsed = timer.time() - start
    return elapsed, result.values

# Load catalog
gl_cat = intake.open_catalog("/work/ab0246/a270092/software/GlobalLab_hackathon_2025/catalog/main.yaml")

# Access TCo1279-DART-1950C 3-hourly native data
ds = gl_cat['ICCP']['TCo1279-DART-1950C']['atmos']['native']['atmos_3h'].to_dask()

print("=== Machine: Levante (256 threads, 501GB RAM) ===\n")
print("=== Data Info ===")
print("2t shape:", ds['2t'].shape)
n_timesteps, n_cells = ds['2t'].shape
print(f"Testing with 100 timesteps × {n_cells:,} cells")
print(f"Data size: ~{100 * n_cells * 4 / 1e9:.2f} GB\n")

# Subset to 100 timesteps
t2m_subset = ds['2t'].isel(time_counter=slice(0, 100))

results = []

# === Test 1: Threaded scheduler with 32 threads ===
print("=== Test 1: Threaded (32 threads) ===")
with dask.config.set(scheduler='threads', num_workers=32):
    elapsed, max_val = benchmark_max(t2m_subset)
    print(f"  Time: {elapsed:.2f}s, Max: {max_val:.2f} K")
    results.append(("32 threads", elapsed))

# === Test 3: Threaded scheduler with 64 threads ===
print("\n=== Test 3: Threaded (64 threads) ===")
with dask.config.set(scheduler='threads', num_workers=64):
    elapsed, max_val = benchmark_max(t2m_subset)
    print(f"  Time: {elapsed:.2f}s, Max: {max_val:.2f} K")
    results.append(("64 threads", elapsed))

# === Test 4: Threaded scheduler with 128 threads ===
print("\n=== Test 4: Threaded (128 threads) ===")
with dask.config.set(scheduler='threads', num_workers=128):
    elapsed, max_val = benchmark_max(t2m_subset)
    print(f"  Time: {elapsed:.2f}s, Max: {max_val:.2f} K")
    results.append(("128 threads", elapsed))

# === Summary ===
print("\n" + "="*50)
print("SUMMARY: 100 timesteps max computation")
print("="*50)
baseline = results[0][1]
for name, elapsed in results:
    speedup = baseline / elapsed
    print(f"{name:20s}: {elapsed:6.2f}s  (speedup: {speedup:.1f}x)")

# Extrapolate to year 1
print("\n=== Extrapolated Year 1 (2920 timesteps) ===")
for name, elapsed in results:
    year1_time = elapsed * 2920 / 100
    print(f"{name:20s}: {year1_time/60:.1f} min")
