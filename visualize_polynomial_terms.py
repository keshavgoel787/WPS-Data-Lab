"""
Visualization of Polynomial Time Terms (time, time², time³)
Shows how each term contributes to the overall predicted trend
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

# Set style (this style ships with matplotlib; no seaborn package required)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('ggplot')

# ------------------------------------------------------------
# Fixed effects fitted live from the analytic dataset, so this
# visualization can never drift from the model it illustrates.
# Coefficients come from the time-only baseline (Model 1) on the
# same N=250 analytic sample saved by hierarchical_violations_model.py.
# ------------------------------------------------------------
_df = pd.read_csv('/Users/keshavgoel/Research/model_data_long.csv')
_df['state'] = pd.Categorical(_df['state'])
_baseline = MixedLM.from_formula(
    'violations ~ time + time2 + time3',
    data=_df, groups=_df['state'], re_formula='~time'
).fit(method='lbfgs')
_fe = _baseline.fe_params

intercept  = float(_fe['Intercept'])
beta_time  = float(_fe['time'])
beta_time2 = float(_fe['time2'])
beta_time3 = float(_fe['time3'])
print(f"Baseline coefficients (fitted, N={len(_df)}): "
      f"intercept={intercept:.4f}, time={beta_time:.4f}, "
      f"time2={beta_time2:.4f}, time3={beta_time3:.4f}")

# Create time values (centered on 2017)
years = np.array([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
time = years - 2017  # Centered time variable

# Calculate each component separately
component_intercept = np.full_like(time, intercept, dtype=float)
component_linear = beta_time * time
component_quadratic = beta_time2 * time**2
component_cubic = beta_time3 * time**3

# Calculate cumulative contributions
cumulative_intercept = component_intercept
cumulative_linear = cumulative_intercept + component_linear
cumulative_quadratic = cumulative_linear + component_quadratic
cumulative_total = cumulative_quadratic + component_cubic  # This is the final prediction

# Actual mean violations by year (computed from the same analytic sample)
_actual = _df.groupby('year')['violations'].mean()
actual_means = [float(_actual.get(y, np.nan)) for y in years]

# ============================================================
# FIGURE 1: Stacked Area showing cumulative contribution
# ============================================================
fig1, ax1 = plt.subplots(figsize=(12, 7))

# Plot as stacked areas from bottom up
ax1.fill_between(years, 0, component_intercept, alpha=0.3, color='#2E86AB', label=f'Intercept = {intercept:.2f}')
ax1.fill_between(years, component_intercept, cumulative_linear, alpha=0.3, color='#E94F37', label=f'+ time (β={beta_time:.2f})')
ax1.fill_between(years, cumulative_linear, cumulative_quadratic, alpha=0.3, color='#F6AE2D', label=f'+ time² (β={beta_time2:.2f})')
ax1.fill_between(years, cumulative_quadratic, cumulative_total, alpha=0.3, color='#86BA90', label=f'+ time³ (β={beta_time3:.2f})')

# Plot the final prediction line
ax1.plot(years, cumulative_total, 'k-', linewidth=3, marker='o', markersize=8, label='Total Prediction')

# Plot actual means
ax1.plot(years, actual_means, 's--', color='purple', linewidth=2, markersize=8, label='Actual Mean')

# Reference line at 2017
ax1.axvline(x=2017, color='gray', linestyle=':', alpha=0.7)
ax1.text(2017.1, ax1.get_ylim()[1]*0.95, 'Reference\n(time=0)', fontsize=9, color='gray')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Violations', fontsize=12)
ax1.set_title('How Polynomial Terms Build the Prediction\n(Cumulative Contribution of Each Term)', fontsize=14)
ax1.legend(loc='upper left', fontsize=10)
ax1.set_xticks(years)

plt.tight_layout()
plt.savefig('/Users/keshavgoel/Research/fig_polynomial_stacked.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_polynomial_stacked.png")

# ============================================================
# FIGURE 2: Individual component contributions
# ============================================================
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Linear term (time)
ax = axes[0, 0]
ax.bar(years, component_linear, color='#E94F37', alpha=0.7, edgecolor='white', linewidth=2)
ax.axhline(y=0, color='black', linewidth=1)
ax.axvline(x=2017, color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Contribution to Prediction')
ax.set_title(f'A. Linear Term: β₁ × time\n(β₁ = {beta_time:.2f})', fontweight='bold')
ax.set_xticks(years)
# Add value labels
for i, (y, v) in enumerate(zip(years, component_linear)):
    ax.text(y, v + (0.5 if v >= 0 else -1.5), f'{v:.1f}', ha='center', fontsize=9)

# Panel B: Quadratic term (time²)
ax = axes[0, 1]
ax.bar(years, component_quadratic, color='#F6AE2D', alpha=0.7, edgecolor='white', linewidth=2)
ax.axhline(y=0, color='black', linewidth=1)
ax.axvline(x=2017, color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Contribution to Prediction')
ax.set_title(f'B. Quadratic Term: β₂ × time²\n(β₂ = {beta_time2:.2f})', fontweight='bold')
ax.set_xticks(years)
for i, (y, v) in enumerate(zip(years, component_quadratic)):
    ax.text(y, v - 2, f'{v:.1f}', ha='center', fontsize=9)

# Panel C: Cubic term (time³)
ax = axes[1, 0]
colors_cubic = ['#86BA90' if v >= 0 else '#E94F37' for v in component_cubic]
ax.bar(years, component_cubic, color=colors_cubic, alpha=0.7, edgecolor='white', linewidth=2)
ax.axhline(y=0, color='black', linewidth=1)
ax.axvline(x=2017, color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Contribution to Prediction')
ax.set_title(f'C. Cubic Term: β₃ × time³\n(β₃ = {beta_time3:.2f})', fontweight='bold')
ax.set_xticks(years)
for i, (y, v) in enumerate(zip(years, component_cubic)):
    offset = 3 if v >= 0 else -5
    ax.text(y, v + offset, f'{v:.1f}', ha='center', fontsize=9)

# Panel D: All components combined
ax = axes[1, 1]
width = 0.2
x = np.arange(len(years))
ax.bar(x - 1.5*width, component_linear, width, label=f'time (β={beta_time:.2f})', color='#E94F37', alpha=0.7)
ax.bar(x - 0.5*width, component_quadratic, width, label=f'time² (β={beta_time2:.2f})', color='#F6AE2D', alpha=0.7)
ax.bar(x + 0.5*width, component_cubic, width, label=f'time³ (β={beta_time3:.2f})', color='#86BA90', alpha=0.7)
ax.bar(x + 1.5*width, cumulative_total - intercept, width, label='Sum (excl. intercept)', color='#2E86AB', alpha=0.7)
ax.axhline(y=0, color='black', linewidth=1)
ax.set_xlabel('Year')
ax.set_ylabel('Contribution to Prediction')
ax.set_title('D. All Components Side-by-Side\n(Excluding Intercept)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend(fontsize=9, loc='lower left')

plt.suptitle('Decomposition of Polynomial Time Terms', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/keshavgoel/Research/fig_polynomial_components.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_polynomial_components.png")

# ============================================================
# FIGURE 3: Smooth curves showing each term's shape
# ============================================================
fig3, ax3 = plt.subplots(figsize=(12, 7))

# Create smooth time values for plotting curves
time_smooth = np.linspace(-6, 2, 100)
years_smooth = time_smooth + 2017

# Calculate smooth curves
linear_smooth = beta_time * time_smooth
quadratic_smooth = beta_time2 * time_smooth**2
cubic_smooth = beta_time3 * time_smooth**3
total_smooth = intercept + linear_smooth + quadratic_smooth + cubic_smooth

# Plot each component
ax3.plot(years_smooth, linear_smooth, '-', color='#E94F37', linewidth=2.5,
         label=f'Linear: {beta_time:.2f} × time')
ax3.plot(years_smooth, quadratic_smooth, '-', color='#F6AE2D', linewidth=2.5,
         label=f'Quadratic: {beta_time2:.2f} × time²')
ax3.plot(years_smooth, cubic_smooth, '-', color='#86BA90', linewidth=2.5,
         label=f'Cubic: {beta_time3:.2f} × time³')

# Plot sum of polynomial terms (without intercept)
ax3.plot(years_smooth, linear_smooth + quadratic_smooth + cubic_smooth, '--',
         color='#2E86AB', linewidth=2.5, label='Sum of time terms')

# Reference lines
ax3.axhline(y=0, color='black', linewidth=1, alpha=0.5)
ax3.axvline(x=2017, color='gray', linestyle=':', alpha=0.7)

# Mark the actual data points
ax3.scatter(years, component_linear, color='#E94F37', s=80, zorder=5, edgecolor='white', linewidth=1.5)
ax3.scatter(years, component_quadratic, color='#F6AE2D', s=80, zorder=5, edgecolor='white', linewidth=1.5)
ax3.scatter(years, component_cubic, color='#86BA90', s=80, zorder=5, edgecolor='white', linewidth=1.5)

ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel('Contribution to Prediction (excl. intercept)', fontsize=12)
ax3.set_title('Shape of Each Polynomial Term\n(How each term changes across years)', fontsize=14)
ax3.legend(loc='lower left', fontsize=10)
ax3.set_xlim(2010.5, 2019.5)

# Add annotations explaining each term
ax3.annotate(f'Linear: Constant increase\nof {beta_time:.2f} per year',
             xy=(2019, linear_smooth[-1]), xytext=(2018, 8),
             fontsize=9, ha='center',
             arrowprops=dict(arrowstyle='->', color='#E94F37', lw=1.5))

ax3.annotate('Quadratic: Pulls down\nat extremes (U-shape)',
             xy=(2011, quadratic_smooth[0]), xytext=(2012.5, -25),
             fontsize=9, ha='center',
             arrowprops=dict(arrowstyle='->', color='#F6AE2D', lw=1.5))

ax3.annotate('Cubic: Asymmetry\n(positive before 2017,\nnegative after)',
             xy=(2011, cubic_smooth[0]), xytext=(2013, 30),
             fontsize=9, ha='center',
             arrowprops=dict(arrowstyle='->', color='#86BA90', lw=1.5))

plt.tight_layout()
plt.savefig('/Users/keshavgoel/Research/fig_polynomial_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_polynomial_curves.png")

# ============================================================
# FIGURE 4: Building up the prediction step by step
# ============================================================
fig4, ax4 = plt.subplots(figsize=(12, 7))

# Plot cumulative build-up
ax4.plot(years, cumulative_intercept, 'o-', color='#2E86AB', linewidth=2, markersize=8,
         label=f'Step 1: Intercept only ({intercept:.1f})')
ax4.plot(years, cumulative_linear, 's-', color='#E94F37', linewidth=2, markersize=8,
         label=f'Step 2: + Linear term')
ax4.plot(years, cumulative_quadratic, '^-', color='#F6AE2D', linewidth=2, markersize=8,
         label=f'Step 3: + Quadratic term')
ax4.plot(years, cumulative_total, 'D-', color='#86BA90', linewidth=3, markersize=10,
         label=f'Step 4: + Cubic term (FINAL)')

# Plot actual data
ax4.plot(years, actual_means, 'p--', color='purple', linewidth=2, markersize=10,
         label='Actual Mean Violations', alpha=0.7)

# Reference line
ax4.axvline(x=2017, color='gray', linestyle=':', alpha=0.7)

ax4.set_xlabel('Year', fontsize=12)
ax4.set_ylabel('Predicted Violations', fontsize=12)
ax4.set_title('Building the Prediction: Adding Each Polynomial Term\n(From Intercept to Full Model)', fontsize=14)
ax4.legend(loc='upper left', fontsize=10)
ax4.set_xticks(years)

# Add arrows showing the transformation
for i, y in enumerate(years):
    if i % 2 == 0:  # Only annotate some years to avoid clutter
        ax4.annotate('', xy=(y, cumulative_total[i]), xytext=(y, cumulative_intercept[i]),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=1))

plt.tight_layout()
plt.savefig('/Users/keshavgoel/Research/fig_polynomial_buildup.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_polynomial_buildup.png")

# ============================================================
# FIGURE 5: Summary table as visualization
# ============================================================
fig5, ax5 = plt.subplots(figsize=(14, 8))
ax5.axis('off')

# Create table data
table_data = [
    ['Year', 'time', 'time²', 'time³',
     f'β₁×time\n({beta_time:.2f}×t)',
     f'β₂×time²\n({beta_time2:.2f}×t²)',
     f'β₃×time³\n({beta_time3:.2f}×t³)',
     'Sum', 'Intercept\n+ Sum', 'Actual']
]

for i, y in enumerate(years):
    t = time[i]
    row = [
        str(y),
        str(t),
        str(t**2),
        str(t**3),
        f'{component_linear[i]:.2f}',
        f'{component_quadratic[i]:.2f}',
        f'{component_cubic[i]:.2f}',
        f'{component_linear[i] + component_quadratic[i] + component_cubic[i]:.2f}',
        f'{cumulative_total[i]:.2f}',
        f'{actual_means[i]:.2f}'
    ]
    table_data.append(row)

table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                  cellLoc='center', loc='center',
                  colColours=['#E8E8E8']*10)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Highlight the 2017 row (index 6 in the data, +1 for header)
for j in range(10):
    table[(7, j)].set_facecolor('#FFFACD')  # Light yellow for 2017

ax5.set_title('Polynomial Term Calculation Table\n(Yellow = 2017, the reference year where time=0)',
              fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/Users/keshavgoel/Research/fig_polynomial_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_polynomial_table.png")

# ============================================================
# Print summary
# ============================================================
print("\n" + "="*60)
print("POLYNOMIAL TERM VISUALIZATIONS COMPLETE")
print("="*60)
print("\nFiles created:")
print("  1. fig_polynomial_stacked.png   - Cumulative area chart")
print("  2. fig_polynomial_components.png - 4-panel component breakdown")
print("  3. fig_polynomial_curves.png    - Smooth curves showing term shapes")
print("  4. fig_polynomial_buildup.png   - Step-by-step prediction building")
print("  5. fig_polynomial_table.png     - Calculation table")

print("\n" + "="*60)
print("INTERPRETATION SUMMARY")
print("="*60)
print(f"""
The polynomial model: violations = {intercept:.2f} + {beta_time:.2f}×time + {beta_time2:.2f}×time² + {beta_time3:.2f}×time³

Each term's role:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. INTERCEPT ({intercept:.2f}):
   • Baseline prediction at time=0 (year 2017)
   • This is the "anchor" of the prediction

2. LINEAR TERM (β₁ = {beta_time:.2f}):
   • Adds {beta_time:.2f} violations for each year after 2017
   • Subtracts {abs(beta_time):.2f} violations for each year before 2017
   • Creates a straight upward slope
   • At 2011 (time=-6): contributes {beta_time * -6:.1f}
   • At 2019 (time=+2): contributes {beta_time * 2:.1f}

3. QUADRATIC TERM (β₂ = {beta_time2:.2f}):
   • Negative coefficient → INVERTED U-SHAPE
   • Pulls DOWN predictions at years far from 2017
   • Maximum effect at extremes (2011, 2019)
   • At 2011 (time²=36): contributes {beta_time2 * 36:.1f}
   • At 2017 (time²=0): contributes 0
   • Creates the "peak" around 2017

4. CUBIC TERM (β₃ = {beta_time3:.2f}):
   • Negative coefficient → ASYMMETRY
   • POSITIVE contribution before 2017 (time³ is negative × β₃ negative = positive)
   • NEGATIVE contribution after 2017 (time³ is positive × β₃ negative = negative)
   • At 2011 (time³=-216): contributes {beta_time3 * -216:.1f} (pushes UP early years)
   • At 2019 (time³=+8): contributes {beta_time3 * 8:.1f} (pushes DOWN later years)
   • Makes the decline after 2017 steeper than the rise before

COMBINED EFFECT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The three terms together create a curve that:
• Starts low in 2011-2014
• Rises sharply into 2016-2017
• Peaks around 2017-2018
• Declines after 2018
• The decline is steeper than the rise (asymmetry from cubic term)
""")
