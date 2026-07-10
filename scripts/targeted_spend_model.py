"""
Targeted FIFRA-Aligned Spending Model — WPS Enforcement Analysis

Theory-driven specification retaining only the two legally protected
populations under FIFRA/WPS:

  SPEND_WORK_z  — spending per farmworker (OCC 45-2092)
  SPEND_APP_z   — spending per pesticide applicator (OCC 37-3012)
  SPEND_AREA_z  — spending per sq mile (structural geographic control)

Formula:
  violations ~ time + time2 + time3
             + SPEND_WORK_z
             + SPEND_APP_z  + SPEND_APP_z:time
             + SPEND_AREA_z + SPEND_AREA_z:time

Interactions retained for APP and AREA based on p < .20 in prior
individual stepwise models (.020 and .073 respectively).
SPEND_WORK interaction excluded to free degrees of freedom.
SPEND_FLC and SPEND_OP excluded — see targeted_spend_model_report.md.
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
    2011: 224.939, 2012: 229.594, 2013: 232.957, 2014: 236.736,
    2015: 237.017, 2016: 240.007, 2017: 245.120,
    2018: 251.107, 2019: 255.657,
}

STATE_NAME_MAPPING = {
    'Massachusetts ': 'Massachusetts',
    'Oregon ': 'Oregon',
}

print("=" * 70)
print("TARGETED FIFRA-ALIGNED SPENDING MODEL — WPS VIOLATIONS ANALYSIS")
print("=" * 70)

# ============================================================
# [1] LOAD BASE VIOLATIONS AND SPENDING DATA
# ============================================================
print("\n[1] Loading base violations and spending data...")

echo_df = pd.read_csv('/Users/keshavgoel/Research/data/raw/establishments_data.csv', index_col=0)
echo_df.index = echo_df.index.str.strip()
echo_df.index = echo_df.index.map(lambda x: STATE_NAME_MAPPING.get(x, x))

echo_long_list = []
for year in range(2011, 2020):
    col = f'violations-{year}'
    if col in echo_df.columns:
        echo_long_list.append(pd.DataFrame({
            'state': echo_df.index,
            'year': year,
            'violations': echo_df[col].values
        }))
echo_long = pd.concat(echo_long_list, ignore_index=True)
echo_long = echo_long[echo_long['state'].isin(US_STATES_50)].copy()
echo_long['violations'] = pd.to_numeric(echo_long['violations'], errors='coerce')
echo_long = echo_long.dropna(subset=['violations'])
echo_long['time']  = echo_long['year'] - 2017
echo_long['time2'] = echo_long['time'] ** 2
echo_long['time3'] = echo_long['time'] ** 3

spending_raw = pd.read_csv('/Users/keshavgoel/Research/data/raw/spending_data_master.csv')
spending = spending_raw[spending_raw['Year'].between(2011, 2019)].copy()
spending['state'] = spending['State'].map(STATE_ABBREV_TO_NAME)
spending = spending[spending['state'].isin(US_STATES_50)].copy()
spending_agg = (
    spending.groupby(['state', 'Year'], as_index=False)['Total Obligation']
    .sum().rename(columns={'Year': 'year', 'Total Obligation': 'spending_nominal'})
)
CPI_2017 = CPI_U[2017]
spending_agg['spending_2017'] = spending_agg.apply(
    lambda r: r['spending_nominal'] * CPI_2017 / CPI_U[r['year']], axis=1
)

df_model = (
    echo_long
    .merge(spending_agg[['state', 'year', 'spending_2017']], on=['state', 'year'], how='left')
    .dropna(subset=['spending_2017'])
    .copy()
)
df_model['state'] = pd.Categorical(df_model['state'])
print(f"  Base dataset: {len(df_model)} obs, {df_model['state'].nunique()} states")

# ============================================================
# [2] LOAD PRE-CONSTRUCTED SPEND VARIABLES
# ============================================================
print("\n[2] Loading pre-constructed spending variables (spend_bls_variables.csv)...")

spend_lv2 = pd.read_csv('/Users/keshavgoel/Research/data/generated/spend_bls_variables.csv')
df_full = df_model.merge(
    spend_lv2[['state', 'SPEND_WORK_z', 'SPEND_APP_z', 'SPEND_AREA_z']],
    on='state', how='left'
)
df_full['state'] = pd.Categorical(df_full['state'])

# Restrict to complete cases on all three predictors
pred_cols = ['SPEND_WORK_z', 'SPEND_APP_z', 'SPEND_AREA_z']
df_fit = df_full.dropna(subset=pred_cols).copy()
df_fit['state'] = pd.Categorical(df_fit['state'])

excluded = sorted(set(US_STATES_50) - set(df_fit['state'].unique()))
print(f"  Analytic sample: {len(df_fit)} obs, {df_fit['state'].nunique()} states")
print(f"  Excluded ({len(excluded)} states): {', '.join(excluded)}")

# ============================================================
# [3] BASELINE MODEL (RESTRICTED SAMPLE)
# ============================================================
print("\n" + "=" * 70)
print("BASELINE MODEL — TIME POLYNOMIAL ONLY (RESTRICTED SAMPLE)")
print("=" * 70)

m0 = MixedLM.from_formula(
    'violations ~ time + time2 + time3',
    data=df_fit, groups=df_fit['state'], re_formula='~time'
).fit(method='lbfgs')

baseline_var_ri = m0.cov_re.iloc[0, 0]
baseline_icc    = baseline_var_ri / (baseline_var_ri + m0.scale)

print(f"\n  Between-state variance (σ²_u0): {baseline_var_ri:.4f}")
print(f"  Within-state variance  (σ²_ε):  {m0.scale:.4f}")
print(f"  ICC: {baseline_icc:.4f}  ({baseline_icc*100:.1f}% between states)")
print(f"  Log-Likelihood: {m0.llf:.4f}")

# ============================================================
# [4] TARGETED MODEL
# ============================================================
print("\n" + "=" * 70)
print("TARGETED MODEL")
print("=" * 70)

formula = (
    'violations ~ time + time2 + time3'
    ' + SPEND_WORK_z'
    ' + SPEND_APP_z  + SPEND_APP_z:time'
    ' + SPEND_AREA_z + SPEND_AREA_z:time'
)
print(f"\n  Formula: {formula}")

m_target = MixedLM.from_formula(
    formula, data=df_fit, groups=df_fit['state'], re_formula='~time'
).fit(method='lbfgs')

fe  = m_target.fe_params
se  = m_target.bse
pv  = m_target.pvalues
ci  = m_target.conf_int()
var_ri  = m_target.cov_re.iloc[0, 0]
var_rs  = m_target.cov_re.iloc[1, 1]
var_res = m_target.scale
delta   = baseline_var_ri - var_ri
pct     = delta / baseline_var_ri * 100

print("\n  Fixed Effects:")
print(f"  {'Parameter':<30} {'β':>9} {'SE':>9} {'z':>7} {'p':>7} {'95% CI':>20}")
print("  " + "-" * 88)

param_order = [
    'Intercept', 'time', 'time2', 'time3',
    'SPEND_WORK_z',
    'SPEND_APP_z', 'SPEND_APP_z:time',
    'SPEND_AREA_z', 'SPEND_AREA_z:time',
]
for param in param_order:
    if param in fe:
        z_stat = fe[param] / se[param]
        lo, hi = ci.loc[param, 0], ci.loc[param, 1]
        sig = ('**' if pv[param] < .01 else
               ('*'  if pv[param] < .05 else
                ('+'  if pv[param] < .10 else ' ')))
        print(f"  {param:<30} {fe[param]:>9.4f} {se[param]:>9.4f} {z_stat:>7.3f} "
              f"{pv[param]:>7.3f} [{lo:>8.4f}, {hi:>8.4f}] {sig}")

print(f"\n  Random Effects:")
print(f"    Between-state variance (σ²_u0): {var_ri:.4f}")
print(f"    Random slope variance  (σ²_u1): {var_rs:.4f}")
print(f"    Within-state variance  (σ²_ε):  {var_res:.4f}")
print(f"    Covariance (u0, u1):            {m_target.cov_re.iloc[0,1]:.4f}")

print(f"\n  Model Fit:")
print(f"    Log-Likelihood:                  {m_target.llf:.4f}")
print(f"    Δσ²_u0 vs. baseline:            {delta:+.4f}")
print(f"    Between-state variance explained: {pct:.2f}%")
print(f"    Converged: {m_target.converged}")

# ============================================================
# [5] SIGNIFICANCE KEY
# ============================================================
print("\n  Significance: ** p<.01  * p<.05  + p<.10")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
