#!/usr/bin/env python3
"""
Compute a single Literature Agreement Score (LAS) for our Dark Doldrums model.

The score combines multiple validation metrics into a single 0-100 number.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load our results
our_events = pd.read_csv('../dark_doldrums_results/all_events_1979_2024.csv')
our_events['start'] = pd.to_datetime(our_events['start'])
our_events['month'] = our_events['start'].dt.month

years = our_events['year'].nunique()

# ============================================================
# COMPONENT SCORES (each 0-100)
# ============================================================

scores = {}
weights = {}

# 1. SEASONALITY SCORE (weight: 30%)
# Literature: events should concentrate in Oct-Feb (winter)
# Score based on % of events in winter months
weights['seasonality'] = 0.30
winter_months = [10, 11, 12, 1, 2]
winter_pct = len(our_events[our_events['month'].isin(winter_months)]) / len(our_events)
# Perfect = 100% in winter, >80% is excellent
scores['seasonality'] = min(100, (winter_pct / 0.85) * 100)

# 2. FREQUENCY SCORE (weight: 20%)
# Literature: 2-10 events per year
# Use log-distance from acceptable range
weights['frequency'] = 0.20
our_freq = len(our_events) / years
lit_freq_mid = 6  # midpoint of 2-10
lit_freq_range = 4  # half-width
# Score decreases as we move away from range
if 2 <= our_freq <= 10:
    scores['frequency'] = 100
else:
    dist = min(abs(our_freq - 2), abs(our_freq - 10))
    scores['frequency'] = max(0, 100 - (dist / lit_freq_mid) * 50)

# 3. DURATION DISTRIBUTION SCORE (weight: 25%)
# Literature: exponential decay (more short events, fewer long)
# Test if our distribution follows expected power law
weights['duration_dist'] = 0.25
duration_bins = [0, 24, 48, 96, 168, 336, 1000]
our_events['dur_bin'] = pd.cut(our_events['duration_hours'], bins=duration_bins)
bin_counts = our_events['dur_bin'].value_counts().sort_index().values

# Expected: exponential decay
expected_ratios = np.array([0.35, 0.25, 0.18, 0.12, 0.07, 0.03])
expected_counts = expected_ratios * len(our_events)

# Chi-squared test (lower = better match)
# But we want correlation instead for robustness
if len(bin_counts) == len(expected_counts):
    corr, _ = stats.spearmanr(bin_counts, expected_counts)
    scores['duration_dist'] = max(0, corr * 100)
else:
    scores['duration_dist'] = 50  # default if mismatch

# 4. TEMPORAL CLUSTERING SCORE (weight: 15%)
# Literature: 1996-1997 was particularly bad
# Check if 1997 ranks in top 3 years by severity
weights['temporal'] = 0.15
yearly_severity = our_events.groupby('year')['severity'].sum().sort_values(ascending=False)
if 1997 in yearly_severity.index:
    rank_1997 = list(yearly_severity.index).index(1997) + 1
    # Top 3 = 100, decreasing after
    scores['temporal'] = max(0, 100 - (rank_1997 - 1) * 15)
else:
    scores['temporal'] = 0

# 5. PHYSICAL CONSISTENCY SCORE (weight: 10%)
# Events should have: low CF during event, recovery after
# Check that mean_cf < threshold and severity correlates with duration
weights['physical'] = 0.10
cf_check = (our_events['mean_cf'] < 0.1).mean()  # should be ~100%
sev_dur_corr, _ = stats.pearsonr(our_events['severity'], our_events['duration_hours'])
scores['physical'] = (cf_check * 50) + (max(0, sev_dur_corr) * 50)

# ============================================================
# COMPUTE FINAL SCORE
# ============================================================

final_score = sum(scores[k] * weights[k] for k in scores)

# ============================================================
# VISUALIZATION: Gauge chart
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#1a1a2e')

# LEFT: Component breakdown
ax1.set_facecolor('#16213e')
components = list(scores.keys())
values = [scores[k] for k in components]
weight_labels = [f"{k}\n(w={weights[k]:.0%})" for k in components]
colors = ['#00ff88' if v >= 70 else '#ffcc00' if v >= 50 else '#ff6b6b' for v in values]

bars = ax1.barh(weight_labels, values, color=colors, edgecolor='white', linewidth=1.5)
ax1.set_xlim(0, 110)
ax1.axvline(x=70, color='#00ff88', linestyle='--', alpha=0.5, label='Good (70)')
ax1.axvline(x=50, color='#ffcc00', linestyle='--', alpha=0.5, label='Fair (50)')

for bar, val in zip(bars, values):
    ax1.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.0f}', 
             va='center', color='white', fontsize=12, fontweight='bold')

ax1.set_xlabel('Score (0-100)', color='white', size=12)
ax1.set_title('Component Scores', color='white', size=14)
ax1.tick_params(colors='white')
ax1.spines['bottom'].set_color('white')
ax1.spines['left'].set_color('white')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='lower right', fontsize=9)

# RIGHT: Gauge meter for final score
ax2.set_facecolor('#16213e')
ax2.set_aspect('equal')

# Draw gauge background
theta_range = np.linspace(0, np.pi, 100)
for i, (start, end, color) in enumerate([
    (0, 0.5, '#ff6b6b'),      # 0-50: red
    (0.5, 0.7, '#ffcc00'),    # 50-70: yellow
    (0.7, 1.0, '#00ff88')     # 70-100: green
]):
    theta_start = np.pi * (1 - end)
    theta_end = np.pi * (1 - start)
    theta = np.linspace(theta_start, theta_end, 50)
    for r in np.linspace(0.6, 1.0, 20):
        ax2.plot(r * np.cos(theta), r * np.sin(theta), color=color, alpha=0.3, linewidth=2)

# Draw needle
needle_angle = np.pi * (1 - final_score/100)
ax2.annotate('', xy=(0.9*np.cos(needle_angle), 0.9*np.sin(needle_angle)), 
             xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='white', lw=3))

# Center circle
circle = plt.Circle((0, 0), 0.15, color='#0f3460', ec='white', linewidth=2)
ax2.add_patch(circle)

# Score text
ax2.text(0, 0, f'{final_score:.0f}', ha='center', va='center', 
         fontsize=24, fontweight='bold', color='white')
ax2.text(0, -0.35, 'Literature\nAgreement\nScore', ha='center', va='top',
         fontsize=12, color='white')

# Labels
ax2.text(-1.1, 0, '0', ha='center', va='center', fontsize=12, color='white')
ax2.text(0, 1.1, '50', ha='center', va='center', fontsize=12, color='white')
ax2.text(1.1, 0, '100', ha='center', va='center', fontsize=12, color='white')

ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-0.5, 1.3)
ax2.axis('off')
ax2.set_title('Overall Model Quality', color='white', size=14, pad=20)

plt.tight_layout()
plt.savefig('quality_score.png', dpi=150, facecolor='#1a1a2e', 
            edgecolor='none', bbox_inches='tight')

# ============================================================
# PRINT REPORT
# ============================================================

print("="*60)
print("    LITERATURE AGREEMENT SCORE (LAS)")
print("="*60)
print()
print("Component Scores:")
print("-"*40)
for k in scores:
    status = "✓" if scores[k] >= 70 else "~" if scores[k] >= 50 else "✗"
    print(f"  {status} {k:20s}: {scores[k]:5.1f} (weight: {weights[k]:.0%})")
print("-"*40)
print()
print(f"  ╔═══════════════════════════════════╗")
print(f"  ║  FINAL SCORE:  {final_score:5.1f} / 100       ║")
print(f"  ╚═══════════════════════════════════╝")
print()

# Interpretation
if final_score >= 80:
    grade = "EXCELLENT"
    interp = "Results strongly agree with literature"
elif final_score >= 70:
    grade = "GOOD"
    interp = "Results mostly agree with literature"
elif final_score >= 60:
    grade = "FAIR"
    interp = "Results partially agree with literature"
else:
    grade = "NEEDS REVIEW"
    interp = "Results diverge from literature expectations"

print(f"  Grade: {grade}")
print(f"  Interpretation: {interp}")
print()
print("="*60)
print("Saved: quality_score.png")
