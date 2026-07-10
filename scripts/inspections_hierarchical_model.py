"""
Hierarchical Mixed-Effects Model for Predicting WPS Inspections
State-Year Level Analysis (2011-2019)

DV: total WPS inspections per state-year (inspections-epa + inspections-state
    from establishments-data (2).csv)

Mirrors hierarchical_violations_model.py exactly; only the outcome variable
and output file names differ.
"""

import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS
# ============================================================

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

CPI_U = {
    2011: 224.939,
    2012: 229.594,
    2013: 232.957,
    2014: 236.736,
    2015: 237.017,
    2016: 240.007,
    2017: 245.120,
    2018: 251.107,
    2019: 255.657,
}

STATE_NAME_MAPPING = {
    'Massachusetts ': 'Massachusetts',
    'Oregon ': 'Oregon',
}

print("=" * 70)
print("HIERARCHICAL MIXED-EFFECTS MODEL FOR INSPECTIONS")
print("=" * 70)

# ============================================================
# [1] LOAD ECHO DATA
# ============================================================
print("\n[1] Loading ECHO data...")
echo_df = pd.read_csv('/Users/keshavgoel/Research/data/raw/establishments_data.csv', index_col=0)
echo_df.index = echo_df.index.str.strip()
echo_df.index = echo_df.index.map(lambda x: STATE_NAME_MAPPING.get(x, x))
print(f"  EPA ECHO data shape: {echo_df.shape}")

# ============================================================
# [2] RESHAPE INSPECTIONS TO LONG FORMAT
# ============================================================
print("\n[2] Reshaping inspections data to long format...")
print("  DV = inspections-epa-YYYY + inspections-state-YYYY (NaN treated as 0 if only one source missing)")

echo_long_list = []
for year in range(2011, 2020):
    col_epa   = f'inspections-epa-{year}'
    col_state = f'inspections-state-{year}'
    epa_vals   = pd.to_numeric(
        echo_df[col_epa]   if col_epa   in echo_df.columns else pd.Series(np.nan, index=echo_df.index),
        errors='coerce'
    )
    state_vals = pd.to_numeric(
        echo_df[col_state] if col_state in echo_df.columns else pd.Series(np.nan, index=echo_df.index),
        errors='coerce'
    )
    # Sum; fill_value=0 treats single-source NaN as 0;
    # pandas returns NaN only if both inputs are NaN.
    total = epa_vals.add(state_vals, fill_value=0)
    echo_long_list.append(pd.DataFrame({
        'state':       echo_df.index,
        'year':        year,
        'inspections': total.values
    }))

echo_long = pd.concat(echo_long_list, ignore_index=True)

# ============================================================
# [3] FILTER TO 50 US STATES
# ============================================================
print("\n[3] Filtering to 50 US states only...")
echo_long_filtered = echo_long[echo_long['state'].isin(US_STATES_50)].copy()
print(f"  Observations after filtering: {len(echo_long_filtered)}")
print(f"  Unique states:                {echo_long_filtered['state'].nunique()}")

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
# [5] CLEAN INSPECTIONS DATA
# ============================================================
print("\n[5] Cleaning inspections data...")
echo_long_filtered['inspections'] = pd.to_numeric(
    echo_long_filtered['inspections'], errors='coerce'
)
missing_v = echo_long_filtered['inspections'].isna().sum()
print(f"  Missing inspection values: {missing_v} (will be dropped)")

df_model = echo_long_filtered.dropna(subset=['inspections']).copy()
df_model['state'] = pd.Categorical(df_model['state'])
print(f"  Final inspections dataset: {len(df_model)} observations, {df_model['state'].nunique()} states")

# ============================================================
# [6] LOAD AND PROCESS SPENDING DATA
# ============================================================
print("\n[6] Loading and processing spending data...")

spending_raw = pd.read_csv('/Users/keshavgoel/Research/data/raw/spending_data_master.csv')
print(f"  Raw spending data shape: {spending_raw.shape}")
print(f"  Year range in file: {spending_raw['Year'].min()} – {spending_raw['Year'].max()}")

spending = spending_raw[spending_raw['Year'].between(2011, 2019)].copy()
print(f"\n  After filtering to 2011–2019: {len(spending)} rows")

spending['state'] = spending['State'].map(STATE_ABBREV_TO_NAME)
spending = spending[spending['state'].isin(US_STATES_50)].copy()
print(f"  After filtering to 50 states: {len(spending)} rows")

neg_count = (spending['Total Obligation'] < 0).sum()
print(f"\n  Negative obligation rows (deobligations, kept in sum): {neg_count}")

spending_agg = (
    spending
    .groupby(['state', 'Year'], as_index=False)['Total Obligation']
    .sum()
    .rename(columns={'Year': 'year', 'Total Obligation': 'spending_nominal'})
)
print(f"\n  State-year spending observations: {len(spending_agg)}")

CPI_2017 = CPI_U[2017]
spending_agg['cpi_deflator']  = spending_agg['year'].map(lambda y: CPI_2017 / CPI_U[y])
spending_agg['spending_2017'] = spending_agg['spending_nominal'] * spending_agg['cpi_deflator']
spending_agg['spending_2017m'] = spending_agg['spending_2017'] / 1_000_000

print("\n  Spending summary (2017 dollars, millions) by year:")
yearly_spend = spending_agg.groupby('year')['spending_2017m'].agg(['mean','sum','min','max','count'])
yearly_spend.columns = ['Mean ($M)', 'Total ($M)', 'Min ($M)', 'Max ($M)', 'N States']
print(yearly_spend.round(3).to_string())

# ============================================================
# [7] MERGE INSPECTIONS AND SPENDING
# ============================================================
print("\n[7] Merging inspections and spending data...")

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

print("\nInspections summary:")
print(df_full['inspections'].describe().round(2))

print("\nSpending summary (2017 $, millions):")
print(df_full['spending_2017m'].describe().round(3))

print("\nInspections by year:")
print(df_full.groupby('year')['inspections']
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
  - Outcome:        inspections (EPA + state, total per state-year)
  - Grouping:       state (categorical, 50 levels)
  - Random effects: random intercept + random slope for time, by state
  - Estimation:     REML via L-BFGS-B

Model 1 (baseline):   inspections ~ time + time² + time³
Model 2 (+ spending): inspections ~ time + time² + time³ + spending_2017m
""")

# --- Model 1: Time only (baseline), on the spending-matched sample ---
print("-" * 70)
print("MODEL 1: Baseline — Time Terms Only")
print("  inspections ~ time + time² + time³ + (1 + time | state)")
print("-" * 70)

model1 = MixedLM.from_formula(
    'inspections ~ time + time2 + time3',
    data=df_full,
    groups=df_full['state'],
    re_formula='~time'
)
result1 = model1.fit(method='lbfgs')
print(result1.summary())

# --- Model 2: Add inflation-adjusted spending ---
print("\n" + "-" * 70)
print("MODEL 2: With Spending — Time Terms + Inflation-Adjusted Spending")
print("  inspections ~ time + time² + time³ + spending_2017m + (1 + time | state)")
print("-" * 70)

model2 = MixedLM.from_formula(
    'inspections ~ time + time2 + time3 + spending_2017m',
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
print(f"    → Expected inspections at time=0 (2017) for a state with 0 spending")

print(f"\n  time:             {fe['time']:>10.4f}")
direction = 'increasing' if fe['time'] > 0 else 'decreasing'
print(f"    → Linear trend: inspections {direction} by {abs(fe['time']):.2f} per year from 2017")

print(f"\n  time²:            {fe['time2']:>10.4f}")
shape = 'Inverted U-shape (peak near 2017)' if fe['time2'] < 0 else 'U-shape (trough near 2017)'
print(f"    → Quadratic curvature: {shape}")

print(f"\n  time³:            {fe['time3']:>10.4f}")
asym = 'Steeper decline post-2017' if fe['time3'] < 0 else 'Steeper rise post-2017'
print(f"    → Cubic asymmetry: {asym}")

print(f"\n  spending_2017m:   {fe['spending_2017m']:>10.4f}")
spend_dir = 'increases' if fe['spending_2017m'] > 0 else 'decreases'
print(f"    → Each additional $1M in 2017-adjusted agricultural spending")
print(f"      {spend_dir} inspections by {abs(fe['spending_2017m']):.4f}, holding time constant")

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

print("\n  States with HIGHEST baseline inspections:")
print(re_df.head(10).to_string(index=False))

print("\n  States with LOWEST baseline inspections:")
print(re_df.tail(10).to_string(index=False))

# ============================================================
# PREDICTED NATIONAL TREND (Model 2, at mean spending)
# ============================================================
print("\n" + "=" * 70)
print("PREDICTED NATIONAL TREND — MODEL 2")
print("=" * 70)

mean_spend = df_full['spending_2017m'].mean()
print(f"\n  Predictions evaluated at mean spending = ${mean_spend:.3f}M (2017 $)")

years_pred = list(range(2011, 2020))
times_pred = [y - 2017 for y in years_pred]

pred_df = pd.DataFrame({
    'year':  years_pred,
    'time':  times_pred,
    'time2': [t**2 for t in times_pred],
    'time3': [t**3 for t in times_pred],
})

pred_df['predicted_inspections'] = (
    fe['Intercept']
    + fe['time']           * pred_df['time']
    + fe['time2']          * pred_df['time2']
    + fe['time3']          * pred_df['time3']
    + fe['spending_2017m'] * mean_spend
)

actual_means = df_full.groupby('year')['inspections'].mean()
pred_df['actual_mean'] = pred_df['year'].map(actual_means)

print("\n  Year | time | Predicted | Actual Mean")
print("  " + "-" * 40)
for _, row in pred_df.iterrows():
    print(f"  {int(row['year'])} | {int(row['time']):+d}    | "
          f"{row['predicted_inspections']:9.2f} | {row['actual_mean']:.2f}")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n[8] Saving results...")

df_full.to_csv('/Users/keshavgoel/Research/data/generated/inspections_model_data_long.csv', index=False)
print("  Saved: inspections_model_data_long.csv")

re_df.to_csv('/Users/keshavgoel/Research/data/generated/inspections_state_random_effects.csv', index=False)
print("  Saved: inspections_state_random_effects.csv")

pred_df.to_csv('/Users/keshavgoel/Research/data/generated/inspections_predicted_trend.csv', index=False)
print("  Saved: inspections_predicted_trend.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
