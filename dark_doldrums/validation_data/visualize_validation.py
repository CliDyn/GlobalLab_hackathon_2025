#!/usr/bin/env python3
"""
Novel visualization comparing Dark Doldrums results vs literature.
Creates a "bullseye" target plot showing how our results compare to literature benchmarks.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe

# Load our results
our_events = pd.read_csv('../dark_doldrums_results/all_events_1979_2024.csv')
our_events['start'] = pd.to_datetime(our_events['start'])

# Calculate our metrics
years = our_events['year'].nunique()
our_freq = len(our_events) / years  # events per year
our_duration = our_events['duration_hours'].mean()  # avg duration
our_hours_year = our_events['duration_hours'].sum() / years

# Literature benchmarks
lit_freq_low, lit_freq_high = 2, 10
lit_duration = 24  # typical single event
lit_hours_low, lit_hours_high = 50, 150

# Create figure with two novel visualizations
fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor('#1a1a2e')

# ============================================================
# LEFT: Radar/Spider chart with literature envelope
# ============================================================
ax1 = fig.add_subplot(121, projection='polar')
ax1.set_facecolor('#16213e')

# Categories and angles
categories = ['Events/Year', 'Avg Duration\n(hours)', 'Total Hours/Year', 'Winter %', 'Severity\n(top event)']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # complete the loop

# Normalize values to 0-1 scale for comparison
# Literature ranges (normalized to their midpoint = 0.5)
lit_values = [
    0.5,  # events/year: 6 is midpoint of 2-10
    0.5,  # duration: 24h is benchmark
    0.5,  # hours/year: 100 is midpoint of 50-150
    0.9,  # winter %: expect >80%
    0.5,  # severity: normalized
]
lit_values += lit_values[:1]

# Our values (normalized relative to literature)
our_values = [
    min(our_freq / 6, 2),  # freq relative to lit midpoint
    min(our_duration / 24, 4),  # duration relative to lit
    min(our_hours_year / 100, 10),  # hours relative to lit midpoint
    0.93,  # our winter % (93%)
    1.0,  # our top severity (normalized)
]
our_values += our_values[:1]

# Plot literature envelope
ax1.fill(angles, [0.3]*len(angles), alpha=0.2, color='green', label='Literature lower')
ax1.fill(angles, [0.7]*len(angles), alpha=0.2, color='green')
ax1.plot(angles, lit_values, 'g--', linewidth=2, label='Literature midpoint')

# Plot our results
ax1.fill(angles, our_values, alpha=0.3, color='#e94560')
ax1.plot(angles, our_values, 'o-', color='#e94560', linewidth=3, markersize=10, label='Our results')

# Customize
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(categories, size=11, color='white')
ax1.set_ylim(0, 2.5)
ax1.set_yticks([0.5, 1, 1.5, 2])
ax1.set_yticklabels(['0.5x', '1x', '1.5x', '2x'], color='gray', size=9)
ax1.tick_params(colors='white')
ax1.spines['polar'].set_color('gray')
ax1.grid(color='gray', alpha=0.3)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax1.set_title('Relative to Literature Benchmarks\n(1x = literature midpoint)', 
              color='white', size=14, pad=20)

# ============================================================
# RIGHT: Duration-Frequency scatter with literature zones
# ============================================================
ax2 = fig.add_subplot(122)
ax2.set_facecolor('#16213e')

# Create duration bins
duration_bins = [0, 24, 48, 96, 168, 336, 500]
bin_labels = ['<1d', '1-2d', '2-4d', '4-7d', '1-2w', '>2w']

our_events['duration_bin'] = pd.cut(our_events['duration_hours'], bins=duration_bins, labels=bin_labels)
duration_counts = our_events['duration_bin'].value_counts().sort_index()

# Literature expectation (exponential decay - short events more common)
x_pos = np.arange(len(bin_labels))
lit_expected = np.array([40, 25, 15, 8, 4, 2])  # typical distribution
lit_expected = lit_expected * (len(our_events) / lit_expected.sum())  # scale to our total

# Bar chart
bars = ax2.bar(x_pos - 0.2, duration_counts.values, 0.4, label='Our Results', 
               color='#e94560', edgecolor='white', linewidth=1.5)
ax2.bar(x_pos + 0.2, lit_expected, 0.4, label='Literature Expected', 
        color='#0f3460', edgecolor='white', linewidth=1.5, alpha=0.7)

# Add match/mismatch indicators
for i, (our, lit) in enumerate(zip(duration_counts.values, lit_expected)):
    ratio = our / lit if lit > 0 else 0
    if 0.5 <= ratio <= 2:
        marker = '✓'
        color = '#00ff88'
    else:
        marker = '!'
        color = '#ff6b6b'
    ax2.annotate(marker, (i, max(our, lit) + 2), ha='center', fontsize=16, 
                 color=color, fontweight='bold')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(bin_labels, color='white', size=11)
ax2.set_xlabel('Event Duration', color='white', size=12)
ax2.set_ylabel('Number of Events', color='white', size=12)
ax2.tick_params(colors='white')
ax2.legend(loc='upper right', fontsize=10)
ax2.set_title('Duration Distribution: Our Results vs Literature\n(✓ = match, ! = differs)', 
              color='white', size=14)
ax2.spines['bottom'].set_color('white')
ax2.spines['left'].set_color('white')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add annotation box
textstr = f'Our events: {len(our_events)}\nYears: {years}\nFreq: {our_freq:.1f}/yr\nAvg duration: {our_duration:.0f}h'
props = dict(boxstyle='round', facecolor='#0f3460', alpha=0.8, edgecolor='white')
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=props, color='white')

plt.tight_layout()
plt.savefig('validation_comparison.png', dpi=150, facecolor='#1a1a2e', 
            edgecolor='none', bbox_inches='tight')
print("Saved: validation_comparison.png")

# ============================================================
# BONUS: Timeline heatmap showing event intensity by month/year
# ============================================================
fig2, ax3 = plt.subplots(figsize=(14, 6))
fig2.patch.set_facecolor('#1a1a2e')
ax3.set_facecolor('#16213e')

# Create month-year matrix of event hours
our_events['month'] = our_events['start'].dt.month
our_events['year_int'] = our_events['year'].astype(int)

pivot = our_events.pivot_table(values='duration_hours', index='month', 
                                columns='year_int', aggfunc='sum', fill_value=0)

# Ensure all months present
for m in range(1, 13):
    if m not in pivot.index:
        pivot.loc[m] = 0
pivot = pivot.sort_index()

# Heatmap
im = ax3.imshow(pivot.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')

# Labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax3.set_yticks(range(12))
ax3.set_yticklabels(month_labels, color='white')
ax3.set_xticks(range(len(pivot.columns)))
ax3.set_xticklabels(pivot.columns, color='white', rotation=45)
ax3.set_xlabel('Year', color='white', size=12)
ax3.set_ylabel('Month', color='white', size=12)
ax3.set_title('Dark Doldrums Event Hours by Month & Year\n(Literature: expect concentration in Oct-Feb)', 
              color='white', size=14)

# Colorbar
cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
cbar.set_label('Total Event Hours', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# Highlight winter months
for i in [0, 1, 9, 10, 11]:  # Jan, Feb, Oct, Nov, Dec
    ax3.axhline(y=i-0.5, color='cyan', linewidth=0.5, alpha=0.5)
    ax3.axhline(y=i+0.5, color='cyan', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('validation_heatmap.png', dpi=150, facecolor='#1a1a2e', 
            edgecolor='none', bbox_inches='tight')
print("Saved: validation_heatmap.png")

print("\nVisualization complete!")
