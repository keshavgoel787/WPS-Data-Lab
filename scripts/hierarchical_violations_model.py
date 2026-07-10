"""
Hierarchical Mixed-Effects Model for Predicting Violations
State-Year Level Analysis (2011-2019)

This script:
1. Loads and reshapes EPA ECHO and WPS data
2. Loads, cleans, and inflation-adjusts agricultural spending data
3. Filters to 50 US states only
4. Constructs time variables centered on 2017
5. Fits a hierarchical mixed-effects model with:
   - Fixed effects: time, time², time³, (+ spending in Model 3)
   - Random effects: random intercept, random slope for time by state
"""

import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS
# ============================================================

# The 50 US states (full names, used for violations data)
US_STATES_50 = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
    'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
    'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
    'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
    'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
    'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
    'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
]

# State abbreviation → full name mapping (for spending data)
STATE_ABBREV_TO_NAME = {
    'AL': 'Alabama',      'AK': 'Alaska',        'AZ': 'Arizona',
    'AR': 'Arkansas',     'CA': 'California',    'CO': 'Colorado',
    'CT': 'Connecticut',  'DE': 'Delaware',      'FL': 'Florida',
    'GA': 'Georgia',      'HI': 'Hawaii',        'ID': 'Idaho',
    'IL': 'Illinois',     'IN': 'Indiana',       'IA': 'Iowa',
    'KS': 'Kansas',       'KY': 'Kentucky',      'LA': 'Louisiana',
    'ME': 'Maine',        'MD': 'Maryland',      'MA': 'Massachusetts',
    'MI': 'Michigan',     'MN': 'Minnesota',     'MS': 'Mississippi',
    'MO': 'Missouri',     'MT': 'Montana',       'NE': 'Nebraska',
    'NV': 'Nevada',       'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico',   'NY': 'New York',      'NC': 'North Carolina',
    'ND': 'North Dakota', 'OH': 'Ohio',          'OK': 'Oklahoma',
    'OR': 'Oregon',       'PA': 'Pennsylvania',  'RI': 'Rhode Island',
    'SC': 'South Carolina','SD': 'South Dakota', 'TN': 'Tennessee',
    'TX': 'Texas',        'UT': 'Utah',          'VT': 'Vermont',
    'VA': 'Virginia',     'WA': 'Washington',    'WV': 'West Virginia',
    'WI': 'Wisconsin',    'WY': 'Wyoming'
}

# CPI-U annual averages (BLS), used to convert nominal spending to 2017 dollars.
# Source: U.S. Bureau of Labor Statistics, CPI-U, All Urban Consumers.
# Base year = 2017 (245.120). Deflator = CPI_2017 / CPI_year.
CPI_U = {
    2011: 224.939,
    2012: 229.594,
    2013: 232.957,
    2014: 236.736,
    2015: 237.017,
    2016: 240.007,
    2017: 245.120,   # base year → deflator = 1.0
    2018: 251.107,
    2019: 255.657,
}

# Handle potential whitespace variants in state names
STATE_NAME_MAPPING = {
    'Massachusetts ': 'Massachusetts',
    'Oregon ': 'Oregon',
}

print("=" * 70)
print("HIERARCHICAL MIXED-EFFECTS MODEL FOR VIOLATIONS")
print("=" * 70)

# ============================================================
# [1] LOAD VIOLATIONS DATA
# ============================================================
print("\n[1] Loading violations data...")
echo_df = pd.read_csv('/Users/keshavgoel/Research/data/raw/establishments_data.csv', index_col=0)
wps_df  = pd.read_csv('/Users/keshavgoel/Research/data/raw/wps_data.csv')

print(f"  EPA ECHO data shape: {echo_df.shape}")
print(f"  WPS data shape:      {wps_df.shape}")

# Clean up state names
echo_df.index = echo_df.index.str.strip()
echo_df.index = echo_df.index.map(lambda x: STATE_NAME_MAPPING.get(x, x))
wps_df['state'] = wps_df['state'].str.strip().map(lambda x: STATE_NAME_MAPPING.get(x, x))

# ============================================================
# [2] RESHAPE VIOLATIONS TO LONG FORMAT
# ============================================================
print("\n[2] Reshaping violations data to long format...")

echo_long_list = []
for year in range(2011, 2020):
    col = f'violations-{year}'
    if col in echo_df.columns:
        echo_long_list.append(pd.DataFrame({
            'state':      echo_df.index,
            'year':       year,
            'violations': echo_df[col].values
        }))

echo_long = pd.concat(echo_long_list, ignore_index=True)

# ============================================================
# [3] FILTER TO 50 US STATES
# ============================================================
print("\n[3] Filtering to 50 US states only...")

echo_long_filtered = echo_long[echo_long['state'].isin(US_STATES_50)].copy()
print(f"  Observations after filtering: {len(echo_long_filtered)}")
print(f"  Unique states:                {echo_long_filtered['state'].nunique()}")

missing_states = set(US_STATES_50) - set(echo_long_filtered['state'].unique())
if missing_states:
    print(f"  States missing from violations data: {missing_states}")

# ============================================================
# [4] CONSTRUCT TIME VARIABLES (CENTERED ON 2017)
# ============================================================
print("\n[4] Constructing time variables centered on 2017...")

echo_long_filtered['time']  = echo_long_filtered['year'] - 2017
echo_long_filtered['time2'] = echo_long_filtered['time'] ** 2
echo_long_filtered['time3'] = echo_long_filtered['time'] ** 3

print("\n  Time variable mapping:")
for year in sorted(echo_long_filtered['year'].unique()):
    t = year - 2017
    print(f"    {year}: time = {t:+d}, time² = {t**2}, time³ = {t**3}")

# ============================================================
# [5] CLEAN VIOLATIONS DATA
# ============================================================
print("\n[5] Cleaning violations data...")

echo_long_filtered['violations'] = pd.to_numeric(
    echo_long_filtered['violations'], errors='coerce'
)
missing_v = echo_long_filtered['violations'].isna().sum()
print(f"  Missing violation values: {missing_v} (will be dropped)")

df_model = echo_long_filtered.dropna(subset=['violations']).copy()
df_model['state'] = pd.Categorical(df_model['state'])
print(f"  Final violations dataset: {len(df_model)} observations, {df_model['state'].nunique()} states")

# ============================================================
# [6] LOAD AND PROCESS SPENDING DATA
# ============================================================
print("\n[6] Loading and processing spending data...")

spending_raw = pd.read_csv('/Users/keshavgoel/Research/data/raw/spending_data_master.csv')
print(f"  Raw spending data shape: {spending_raw.shape}")
print(f"  Year range in file: {spending_raw['Year'].min()} – {spending_raw['Year'].max()}")

# --- 6a. Filter to study years 2011-2019 ---
spending = spending_raw[spending_raw['Year'].between(2011, 2019)].copy()
print(f"\n  After filtering to 2011–2019: {len(spending)} rows")

# --- 6b. Map state abbreviations to full names; keep only 50 states ---
spending['state'] = spending['State'].map(STATE_ABBREV_TO_NAME)
spending = spending[spending['state'].isin(US_STATES_50)].copy()
print(f"  After filtering to 50 states: {len(spending)} rows")
print(f"  States with spending data: {spending['state'].nunique()}")

states_no_spending = set(US_STATES_50) - set(spending['state'].unique())
if states_no_spending:
    print(f"  States with NO spending records: {states_no_spending}")

# --- 6c. Handle negative obligations (deobligations) ---
# Negative values represent money returned/clawed back; kept in the sum
# so the net figure is accurate.
neg_count = (spending['Total Obligation'] < 0).sum()
print(f"\n  Negative obligation rows (deobligations, kept in sum): {neg_count}")

# --- 6d. Aggregate to state-year level (sum of all grants) ---
spending_agg = (
    spending
    .groupby(['state', 'Year'], as_index=False)['Total Obligation']
    .sum()
    .rename(columns={'Year': 'year', 'Total Obligation': 'spending_nominal'})
)
print(f"\n  State-year spending observations: {len(spending_agg)}")

# --- 6e. Inflate to 2017 dollars using CPI-U ---
# Formula: spending_2017 = spending_nominal × (CPI_2017 / CPI_year)
CPI_2017 = CPI_U[2017]
spending_agg['cpi_deflator']  = spending_agg['year'].map(
    lambda y: CPI_2017 / CPI_U[y]
)
spending_agg['spending_2017'] = spending_agg['spending_nominal'] * spending_agg['cpi_deflator']

# Scale to millions of dollars for interpretable coefficients
spending_agg['spending_2017m'] = spending_agg['spending_2017'] / 1_000_000

print("\n  CPI-U deflators applied (base year 2017):")
for yr in sorted(CPI_U):
    if 2011 <= yr <= 2019:
        defl = CPI_2017 / CPI_U[yr]
        print(f"    {yr}: CPI = {CPI_U[yr]:.3f}, deflator = {defl:.4f}")

print("\n  Spending summary (2017 dollars, millions) by year:")
yearly_spend = spending_agg.groupby('year')['spending_2017m'].agg(['mean','sum','min','max','count'])
yearly_spend.columns = ['Mean ($M)', 'Total ($M)', 'Min ($M)', 'Max ($M)', 'N States']
print(yearly_spend.round(3).to_string())

# ============================================================
# [7] MERGE VIOLATIONS AND SPENDING
# ============================================================
print("\n[7] Merging violations and spending data...")

df_merged = df_model.merge(
    spending_agg[['state', 'year', 'spending_2017', 'spending_2017m']],
    on=['state', 'year'],
    how='left'
)

missing_spend = df_merged['spending_2017m'].isna().sum()
print(f"  Merged dataset shape: {df_merged.shape}")
print(f"  Observations missing spending data: {missing_spend}")

if missing_spend > 0:
    missing_spend_states = (
        df_merged[df_merged['spending_2017m'].isna()]
        .groupby('state')['year']
        .apply(list)
    )
    print("  States/years missing spending:")
    print(missing_spend_states.to_string())

# Drop rows where spending is missing so models 2 and 3 use the same sample
df_full = df_merged.dropna(subset=['spending_2017m']).copy()
df_full['state'] = pd.Categorical(df_full['state'])
print(f"\n  Final analytic dataset (all variables present): {len(df_full)} observations, "
      f"{df_full['state'].nunique()} states")

# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print("\nViolations summary:")
print(df_full['violations'].describe().round(2))

print("\nSpending summary (2017 $, millions):")
print(df_full['spending_2017m'].describe().round(3))

print("\nViolations by year:")
print(df_full.groupby('year')['violations']
      .agg(['mean','std','min','max','count'])
      .round(2))

print("\nSpending by year (2017 $M):")
print(df_full.groupby('year')['spending_2017m']
      .agg(['mean','std','min','max'])
      .round(3))

# ============================================================
# FIT MODELS
# ============================================================
print("\n" + "=" * 70)
print("FITTING HIERARCHICAL MIXED-EFFECTS MODELS")
print("=" * 70)

print("""
All models use:
  - Outcome:        violations
  - Grouping:       state (categorical, 50 levels)
  - Random effects: random intercept + random slope for time, by state
  - Estimation:     REML via L-BFGS-B

Model 1 (baseline):   violations ~ time + time² + time³
Model 2 (+ spending): violations ~ time + time² + time³ + spending_2017m
""")

# --- Model 1: Time only (baseline), on the matched sample ---
print("-" * 70)
print("MODEL 1: Baseline — Time Terms Only")
print("  violations ~ time + time² + time³ + (1 + time | state)")
print("-" * 70)

model1 = MixedLM.from_formula(
    'violations ~ time + time2 + time3',
    data=df_full,
    groups=df_full['state'],
    re_formula='~time'
)
result1 = model1.fit(method='lbfgs')
print(result1.summary())

# --- Model 2: Add inflation-adjusted spending ---
print("\n" + "-" * 70)
print("MODEL 2: With Spending — Time Terms + Inflation-Adjusted Spending")
print("  violations ~ time + time² + time³ + spending_2017m + (1 + time | state)")
print("-" * 70)

model2 = MixedLM.from_formula(
    'violations ~ time + time2 + time3 + spending_2017m',
    data=df_full,
    groups=df_full['state'],
    re_formula='~time'
)
result2 = model2.fit(method='lbfgs')
print(result2.summary())

# ============================================================
# DETAILED RESULTS INTERPRETATION
# ============================================================
print("\n" + "=" * 70)
print("DETAILED RESULTS — MODEL 2 (WITH SPENDING)")
print("=" * 70)

fe = result2.fe_params

print("\n--- FIXED EFFECTS ---")
print("(National-average effects, holding all else constant)\n")

print(f"  Intercept:        {fe['Intercept']:>10.4f}")
print(f"    → Expected violations at time=0 (2017) for a state with 0 spending")

print(f"\n  time:             {fe['time']:>10.4f}")
direction = 'increasing' if fe['time'] > 0 else 'decreasing'
print(f"    → Linear trend: violations {direction} by {abs(fe['time']):.2f} per year from 2017")

print(f"\n  time²:            {fe['time2']:>10.4f}")
shape = 'Inverted U-shape (peak near 2017)' if fe['time2'] < 0 else 'U-shape (trough near 2017)'
print(f"    → Quadratic curvature: {shape}")

print(f"\n  time³:            {fe['time3']:>10.4f}")
asym = 'Steeper decline post-2017' if fe['time3'] < 0 else 'Steeper rise post-2017'
print(f"    → Cubic asymmetry: {asym}")

print(f"\n  spending_2017m:   {fe['spending_2017m']:>10.4f}")
spend_dir = 'increases' if fe['spending_2017m'] > 0 else 'decreases'
print(f"    → Each additional $1M in 2017-adjusted agricultural spending")
print(f"      {spend_dir} violations by {abs(fe['spending_2017m']):.4f}, holding time constant")

print("\n--- RANDOM EFFECTS ---")
print("(Between-state variation)\n")
print("  Random Effects Covariance Matrix:")
print(result2.cov_re.round(4))

var_ri  = result2.cov_re.iloc[0, 0]
var_rs  = result2.cov_re.iloc[1, 1]
cov_ris = result2.cov_re.iloc[0, 1]
var_res = result2.scale
icc     = var_ri / (var_ri + var_res)

print(f"\n  Variance Components:")
print(f"    Random intercept variance (between-state baseline): {var_ri:.4f}")
print(f"    Random slope variance (between-state time trends):  {var_rs:.4f}")
print(f"    Intercept-slope covariance:                         {cov_ris:.4f}")
print(f"    Residual variance (within-state):                   {var_res:.4f}")
print(f"\n  ICC = {icc:.4f}  →  {icc*100:.1f}% of variance is between states")

# ============================================================
# MODEL COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(f"\n{'Metric':<30} {'Model 1 (no spending)':>22} {'Model 2 (+ spending)':>22}")
print("-" * 76)
print(f"{'Log-Likelihood':<30} {result1.llf:>22.2f} {result2.llf:>22.2f}")
print(f"{'Residual Variance':<30} {result1.scale:>22.4f} {result2.scale:>22.4f}")
print(f"{'Random Intercept Var':<30} {result1.cov_re.iloc[0,0]:>22.4f} {result2.cov_re.iloc[0,0]:>22.4f}")

lr_stat = 2 * (result2.llf - result1.llf)
print(f"\n  Likelihood Ratio Test: χ² = {lr_stat:.4f} (df = 1)")
if lr_stat > 3.841:
    print("  → Model 2 fits significantly better (p < .05)")
else:
    print("  → No significant improvement from adding spending (p > .05)")

# ============================================================
# STATE-SPECIFIC RANDOM EFFECTS (BLUPs)
# ============================================================
print("\n" + "=" * 70)
print("STATE-SPECIFIC RANDOM EFFECTS — MODEL 2 (BLUPs)")
print("=" * 70)

random_effects = result2.random_effects
re_df = pd.DataFrame({
    'state':              list(random_effects.keys()),
    'random_intercept':   [v['Group'] for v in random_effects.values()],
    'random_slope_time':  [v.get('time', 0) for v in random_effects.values()]
}).sort_values('random_intercept', ascending=False)

print("\n  States with HIGHEST baseline violations:")
print(re_df.head(10).to_string(index=False))

print("\n  States with LOWEST baseline violations:")
print(re_df.tail(10).to_string(index=False))

# ============================================================
# PREDICTED NATIONAL TREND (Model 2, at mean spending)
# ============================================================
print("\n" + "=" * 70)
print("PREDICTED NATIONAL TREND — MODEL 2")
print("=" * 70)

mean_spend = df_full['spending_2017m'].mean()
print(f"\n  Predictions evaluated at mean spending = ${mean_spend:.3f}M (2017 $)")

years_pred  = list(range(2011, 2020))
times_pred  = [y - 2017 for y in years_pred]

pred_df = pd.DataFrame({
    'year':  years_pred,
    'time':  times_pred,
    'time2': [t**2 for t in times_pred],
    'time3': [t**3 for t in times_pred],
})

pred_df['predicted_violations'] = (
    fe['Intercept']
    + fe['time']           * pred_df['time']
    + fe['time2']          * pred_df['time2']
    + fe['time3']          * pred_df['time3']
    + fe['spending_2017m'] * mean_spend
)

actual_means = df_full.groupby('year')['violations'].mean()
pred_df['actual_mean'] = pred_df['year'].map(actual_means)

print("\n  Year | time | Predicted | Actual Mean")
print("  " + "-" * 40)
for _, row in pred_df.iterrows():
    print(f"  {int(row['year'])} | {int(row['time']):+d}    | "
          f"{row['predicted_violations']:9.2f} | {row['actual_mean']:.2f}")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n[8] Saving results...")

df_full.to_csv('/Users/keshavgoel/Research/data/generated/model_data_long.csv', index=False)
print("  Saved: model_data_long.csv")

re_df.to_csv('/Users/keshavgoel/Research/data/generated/state_random_effects.csv', index=False)
print("  Saved: state_random_effects.csv")

pred_df.to_csv('/Users/keshavgoel/Research/data/generated/predicted_trend.csv', index=False)
print("  Saved: predicted_trend.csv")

spending_agg.to_csv('/Users/keshavgoel/Research/data/generated/spending_aggregated.csv', index=False)
print("  Saved: spending_aggregated.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
