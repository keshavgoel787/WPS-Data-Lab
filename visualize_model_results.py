"""
Visualizations for Hierarchical Mixed-Effects Model Results
Creates 6 visualizations to illustrate model findings
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

# Load the data
print("Loading data...")
df_model = pd.read_csv('/Users/keshavgoel/Research/model_data_long.csv')
re_df = pd.read_csv('/Users/keshavgoel/Research/state_random_effects.csv')
pred_df = pd.read_csv('/Users/keshavgoel/Research/predicted_trend.csv')

# ============================================================
# FIGURE 1: National Trend - Predicted vs Actual
# ============================================================
print("Creating Figure 1: National Trend...")
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Actual mean violations
ax1.plot(pred_df['year'], pred_df['actual_mean'], 'o-',
         color='#2E86AB', markersize=10, linewidth=2, label='Actual Mean')

# Predicted trend (fixed effects)
ax1.plot(pred_df['year'], pred_df['predicted_violations'], 's--',
         color='#E94F37', markersize=8, linewidth=2, label='Model Prediction')

# Add vertical line at 2017 (reference year)
ax1.axvline(x=2017, color='gray', linestyle=':', alpha=0.7, label='Reference Year (2017)')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Mean Violations', fontsize=12)
ax1.set_title('National Trend: EPA Violations (2011-2019)\nHierarchical Model Predictions vs Actual', fontsize=14)
ax1.legend(loc='best', fontsize=10)
ax1.set_xticks(range(2011, 2020))

plt.tight_layout()
plt.savefig('/Users/keshavgoel/Research/fig1_national_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig1_national_trend.png")

# ============================================================
# FIGURE 2: State Random Effects (Caterpillar Plot)
# ============================================================
print("Creating Figure 2: State Random Effects...")
fig2, ax2 = plt.subplots(figsize=(12, 14))

# Sort by random intercept
re_sorted = re_df.sort_values('random_intercept')

# Color based on positive/negative
colors = ['#E94F37' if x > 0 else '#2E86AB' for x in re_sorted['random_intercept']]

# Horizontal bar chart
bars = ax2.barh(range(len(re_sorted)), re_sorted['random_intercept'], color=colors, alpha=0.8)

# Add state names
ax2.set_yticks(range(len(re_sorted)))
ax2.set_yticklabels(re_sorted['state'], fontsize=9)

# Reference line at zero
ax2.axvline(x=0, color='black', linewidth=1)

ax2.set_xlabel('Random Intercept (Deviation from National Average)', fontsize=12)
ax2.set_title('State-Level Random Effects (BLUPs)\nDeviation from National Baseline Violations', fontsize=14)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#E94F37', alpha=0.8, label='Above Average'),
                   Patch(facecolor='#2E86AB', alpha=0.8, label='Below Average')]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/keshavgoel/Research/fig2_state_random_effects.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig2_state_random_effects.png")

# ============================================================
# FIGURE 3: State-Level Trajectories (Spaghetti Plot)
# ============================================================
print("Creating Figure 3: State Trajectories...")
fig3, ax3 = plt.subplots(figsize=(12, 8))

# Plot each state's trajectory
for state in df_model['state'].unique():
    state_data = df_model[df_model['state'] == state].sort_values('year')
    ax3.plot(state_data['year'], state_data['violations'],
             alpha=0.3, linewidth=1, color='gray')

# Overlay the national predicted trend (thicker line)
ax3.plot(pred_df['year'], pred_df['predicted_violations'],
         color='#E94F37', linewidth=3, label='National Trend (Model)')

# Also plot actual national mean
ax3.plot(pred_df['year'], pred_df['actual_mean'],
         color='#2E86AB', linewidth=3, linestyle='--', label='National Mean (Actual)')

ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel('Violations', fontsize=12)
ax3.set_title('State-Level Violation Trajectories (2011-2019)\nIndividual States (gray) vs National Trend', fontsize=14)
ax3.legend(loc='best', fontsize=10)
ax3.set_xticks(range(2011, 2020))

plt.tight_layout()
plt.savefig('/Users/keshavgoel/Research/fig3_state_trajectories.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig3_state_trajectories.png")

# ============================================================
# FIGURE 4: Random Intercept vs Random Slope Scatter
# ============================================================
print("Creating Figure 4: Random Effects Scatter...")
fig4, ax4 = plt.subplots(figsize=(10, 8))

# Check if random slopes exist (non-zero)
if re_df['random_slope_time'].abs().sum() > 0.001:
    scatter = ax4.scatter(re_df['random_intercept'], re_df['random_slope_time'],
                          s=100, alpha=0.7, c='#2E86AB', edgecolor='white', linewidth=1)

    # Add state labels for extreme values
    for idx, row in re_df.iterrows():
        if abs(row['random_intercept']) > re_df['random_intercept'].std() * 1.5 or \
           abs(row['random_slope_time']) > re_df['random_slope_time'].std() * 1.5:
            ax4.annotate(row['state'], (row['random_intercept'], row['random_slope_time']),
                        fontsize=8, ha='left', va='bottom')

    # Reference lines
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    ax4.set_xlabel('Random Intercept (Baseline Deviation)', fontsize=12)
    ax4.set_ylabel('Random Slope (Time Trend Deviation)', fontsize=12)
    ax4.set_title('State Random Effects: Baseline vs Time Trend\nEach point represents a state', fontsize=14)
else:
    # If no random slopes, show distribution of intercepts
    ax4.hist(re_df['random_intercept'], bins=15, color='#2E86AB', alpha=0.7, edgecolor='white')
    ax4.axvline(x=0, color='#E94F37', linestyle='--', linewidth=2, label='National Average')
    ax4.set_xlabel('Random Intercept (Baseline Deviation)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of State Random Intercepts', fontsize=14)
    ax4.legend()

plt.tight_layout()
plt.savefig('/Users/keshavgoel/Research/fig4_random_effects_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig4_random_effects_scatter.png")

# ============================================================
# FIGURE 5: Violations Heatmap (States x Years)
# ============================================================
print("Creating Figure 5: Violations Heatmap...")

# Pivot data for heatmap
heatmap_data = df_model.pivot_table(values='violations', index='state', columns='year', aggfunc='mean')

# Sort states by overall mean violations
heatmap_data = heatmap_data.loc[heatmap_data.mean(axis=1).sort_values(ascending=False).index]

fig5, ax5 = plt.subplots(figsize=(12, 16))

# Create heatmap (matplotlib imshow; NaNs shown as blank via masked array)
masked = np.ma.masked_invalid(heatmap_data.values)
cmap = plt.cm.YlOrRd.copy()
cmap.set_bad(color='#f0f0f0')
im = ax5.imshow(masked, cmap=cmap, aspect='auto')
ax5.set_xticks(range(len(heatmap_data.columns)))
ax5.set_xticklabels(heatmap_data.columns)
ax5.set_yticks(range(len(heatmap_data.index)))
ax5.set_yticklabels(heatmap_data.index, fontsize=9)
cbar = fig5.colorbar(im, ax=ax5, fraction=0.025, pad=0.02)
cbar.set_label('Violations')

ax5.set_xlabel('Year', fontsize=12)
ax5.set_ylabel('State', fontsize=12)
ax5.set_title('EPA Violations by State and Year (2011-2019)\nHeatmap sorted by average violations', fontsize=14)

plt.tight_layout()
plt.savefig('/Users/keshavgoel/Research/fig5_violations_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig5_violations_heatmap.png")

# ============================================================
# FIGURE 6: ICC Visualization (Variance Decomposition)
# ============================================================
print("Creating Figure 6: Variance Decomposition...")

# Variance components from the actual fitted model (time-only baseline,
# Model 1) on the same analytic sample, so the ICC here matches the model
# output rather than a crude group-means approximation.
df_fit = df_model.copy()
df_fit['state'] = pd.Categorical(df_fit['state'])
_m = MixedLM.from_formula(
    'violations ~ time + time2 + time3',
    data=df_fit, groups=df_fit['state'], re_formula='~time'
).fit(method='lbfgs')
between_state_var = float(_m.cov_re.iloc[0, 0])   # random intercept variance
within_state_var = float(_m.scale)                # residual variance
total_var = between_state_var + within_state_var
icc = between_state_var / total_var
print(f"  Variance components (fitted): between={between_state_var:.2f}, "
      f"within={within_state_var:.2f}, ICC={icc:.3f}")

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
sizes = [between_state_var, within_state_var]
labels = [f'Between-State\n({icc*100:.1f}%)', f'Within-State\n({(1-icc)*100:.1f}%)']
colors_pie = ['#2E86AB', '#A3CEF1']
explode = (0.02, 0)

ax6a.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
         autopct='', startangle=90, textprops={'fontsize': 11})
ax6a.set_title('Variance Decomposition\n(Intraclass Correlation)', fontsize=14)

# Bar chart showing variance components
var_components = pd.DataFrame({
    'Component': ['Between-State\nVariance', 'Within-State\n(Residual) Variance'],
    'Variance': [between_state_var, within_state_var]
})

bars = ax6b.bar(var_components['Component'], var_components['Variance'],
                color=['#2E86AB', '#A3CEF1'], edgecolor='white', linewidth=2)
ax6b.set_ylabel('Variance', fontsize=12)
ax6b.set_title(f'Variance Components\nICC = {icc:.3f}', fontsize=14)

# Add value labels on bars
for bar, val in zip(bars, var_components['Variance']):
    ax6b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(var_components['Variance'])*0.02,
              f'{val:.1f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('/Users/keshavgoel/Research/fig6_variance_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig6_variance_decomposition.png")

# ============================================================
# BONUS: Combined Summary Figure
# ============================================================
print("Creating Bonus: Combined Summary Figure...")

fig_summary, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: National Trend
ax = axes[0, 0]
ax.plot(pred_df['year'], pred_df['actual_mean'], 'o-', color='#2E86AB',
        markersize=8, linewidth=2, label='Actual Mean')
ax.plot(pred_df['year'], pred_df['predicted_violations'], 's--', color='#E94F37',
        markersize=6, linewidth=2, label='Model Prediction')
ax.axvline(x=2017, color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Mean Violations')
ax.set_title('A. National Trend: Predicted vs Actual', fontweight='bold')
ax.legend(fontsize=9)
ax.set_xticks(range(2011, 2020))

# Panel B: Top/Bottom States
ax = axes[0, 1]
top_5 = re_df.nlargest(5, 'random_intercept')
bottom_5 = re_df.nsmallest(5, 'random_intercept')
combined = pd.concat([top_5, bottom_5]).sort_values('random_intercept')
colors_bar = ['#E94F37' if x > 0 else '#2E86AB' for x in combined['random_intercept']]
ax.barh(range(len(combined)), combined['random_intercept'], color=colors_bar, alpha=0.8)
ax.set_yticks(range(len(combined)))
ax.set_yticklabels(combined['state'], fontsize=9)
ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlabel('Random Intercept')
ax.set_title('B. Top 5 and Bottom 5 States', fontweight='bold')

# Panel C: State Trajectories
ax = axes[1, 0]
for state in df_model['state'].unique():
    state_data = df_model[df_model['state'] == state].sort_values('year')
    ax.plot(state_data['year'], state_data['violations'], alpha=0.2, linewidth=0.8, color='gray')
ax.plot(pred_df['year'], pred_df['predicted_violations'], color='#E94F37', linewidth=3)
ax.set_xlabel('Year')
ax.set_ylabel('Violations')
ax.set_title('C. State Trajectories with National Trend', fontweight='bold')
ax.set_xticks(range(2011, 2020))

# Panel D: Year-over-Year Changes
ax = axes[1, 1]
yearly_stats = df_model.groupby('year')['violations'].agg(['mean', 'std'])
ax.errorbar(yearly_stats.index, yearly_stats['mean'], yerr=yearly_stats['std']/2,
            fmt='o-', color='#2E86AB', capsize=4, capthick=2, linewidth=2, markersize=8)
ax.fill_between(yearly_stats.index,
                yearly_stats['mean'] - yearly_stats['std']/2,
                yearly_stats['mean'] + yearly_stats['std']/2,
                alpha=0.2, color='#2E86AB')
ax.set_xlabel('Year')
ax.set_ylabel('Mean Violations')
ax.set_title('D. Violations by Year (with variability)', fontweight='bold')
ax.set_xticks(range(2011, 2020))

plt.suptitle('Hierarchical Mixed-Effects Model: EPA Violations Analysis (2011-2019)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/keshavgoel/Research/fig_summary_combined.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig_summary_combined.png")

print("\n" + "=" * 50)
print("All visualizations created successfully!")
print("=" * 50)
print("\nFiles saved:")
print("  1. fig1_national_trend.png")
print("  2. fig2_state_random_effects.png")
print("  3. fig3_state_trajectories.png")
print("  4. fig4_random_effects_scatter.png")
print("  5. fig5_violations_heatmap.png")
print("  6. fig6_variance_decomposition.png")
print("  7. fig_summary_combined.png (bonus: 4-panel summary)")
