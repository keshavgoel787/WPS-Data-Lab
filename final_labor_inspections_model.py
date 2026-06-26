"""
FINAL MODEL — WPS Inspections, Labor/DOL Covariates

Curated final model implementing PI guidance (2026-06). Structural mirror of
final_labor_violations_model.py; two outcome-specific differences:

  1. Labor Intensity Index collinearity: only the 2017 wave is retained
     (lii_2012 / lii_2022 are near-collinear, r ≈ 0.977). Same as violations.

  2. TIME SPECIFICATION: inspections are modeled with LINEAR time only.
     The quadratic (time2) and cubic (time3) terms are dropped from the final
     model — the inspections series shows only a linear time trend (the
     curvilinear/cubic terms were non-significant), unlike the violations
     series whose 2016–2017 spike requires the cubic. The pseudo-R² baseline
     is likewise a linear-time baseline, re-estimated on the same analytic
     sample so the variance-reduction comparison is against a matched baseline.

Final covariate block (z-scored Level-2 predictors):
  lii_2017_z, dol_workers_cert_z, dol_n_cases_z, dol_demand_met_pct_z, pct_flc_z

×time interactions are screened one-at-a-time and included if p < .20. With a
linear-time model these interactions are linear (covariate × time) and are
therefore consistent with dropping the higher-order polynomial terms.
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

# Final-model time specification: inspections are LINEAR-time only.
TIME_TERMS  = ['time']
TIME_PREFIX = ' + '.join(TIME_TERMS)

# Final-model covariate block (lii_2012 dropped — collinear with lii_2017).
FINAL_COVARIATES = ['lii_2017', 'dol_workers_cert', 'dol_n_cases',
                    'dol_demand_met_pct', 'pct_flc']

print("=" * 70)
print("FINAL MODEL — WPS INSPECTIONS (LABOR/DOL COVARIATES, LINEAR TIME)")
print("=" * 70)

# ============================================================
# [1] BASE INSPECTIONS + SPENDING DATA
# ============================================================
print("\n[1] Loading base inspections and spending data...")

echo_df = pd.read_csv('/Users/keshavgoel/Research/establishments-data (2).csv', index_col=0)
echo_df.index = echo_df.index.str.strip()
echo_df.index = echo_df.index.map(lambda x: STATE_NAME_MAPPING.get(x, x))

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
    total = epa_vals.add(state_vals, fill_value=0)
    echo_long_list.append(pd.DataFrame({
        'state':       echo_df.index,
        'year':        year,
        'inspections': total.values
    }))

echo_long = pd.concat(echo_long_list, ignore_index=True)
echo_long = echo_long[echo_long['state'].isin(US_STATES_50)].copy()
echo_long['inspections'] = pd.to_numeric(echo_long['inspections'], errors='coerce')
echo_long = echo_long.dropna(subset=['inspections'])
echo_long['time']  = echo_long['year'] - 2017
echo_long['time2'] = echo_long['time'] ** 2
echo_long['time3'] = echo_long['time'] ** 3

spending_raw = pd.read_csv('/Users/keshavgoel/Research/spending_data_master(in) (1).csv')
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
spending_agg['spending_2017m'] = spending_agg['spending_2017'] / 1_000_000

df_base = echo_long.merge(
    spending_agg[['state', 'year', 'spending_2017m']],
    on=['state', 'year'], how='left'
).dropna(subset=['spending_2017m']).copy()
df_base['state'] = pd.Categorical(df_base['state'])
print(f"  Base dataset: {len(df_base)} obs, {df_base['state'].nunique()} states")

# ============================================================
# [2] CONSTRUCT FINAL LEVEL-2 COVARIATES (lii_2017 only)
# ============================================================
print("\n[2] Constructing final Level-2 covariates (lii_2017 only)...")

level2 = pd.DataFrame({'state': US_STATES_50})

# --- 2a. Labor Intensity Index — 2017 wave only ---
li_17 = pd.read_csv('/Users/keshavgoel/Research/labor_intensity_index_2017.csv')
li_17['state'] = li_17['state_name'].str.title()
level2 = level2.merge(li_17[['state', 'Labor_Intensity_Index']].rename(
    columns={'Labor_Intensity_Index': 'lii_2017'}), on='state', how='left')
print(f"    lii_2017: {level2['lii_2017'].notna().sum()} states "
      f"(2012/2022 waves excluded — collinear, r≈0.977)")

# --- 2b. DOL H-2A Annual Workers (2011–2019, excl. 2013/2014) ---
dol1 = pd.read_csv('/Users/keshavgoel/Research/dol_var1_workers_by_state_annual.csv')
dol_study = dol1[dol1['year'].between(2011, 2019) & (dol1['year'] != 2013)].copy()
dol_means = dol_study.groupby('state').agg(
    dol_workers_cert=('workers_certified', 'mean'),
    dol_n_cases=('n_cases', 'mean'),
    dol_demand_met_pct=('demand_met_pct', lambda x: x[x != np.inf].mean())
).reset_index()
dol_means['state_name'] = dol_means['state'].map(STATE_ABBREV_TO_NAME)
level2 = level2.merge(dol_means[['state_name', 'dol_workers_cert',
                                 'dol_n_cases', 'dol_demand_met_pct']].rename(
    columns={'state_name': 'state'}), on='state', how='left')

# --- 2c. DOL Employer Type (2020 proxy) ---
dol2 = pd.read_csv('/Users/keshavgoel/Research/dol_var2_employer_type_annual.csv')
d2020 = dol2[dol2['year'] == 2020].copy()
d2020['state_name'] = d2020['state'].map(STATE_ABBREV_TO_NAME)
d2020 = d2020.rename(columns={'pct_Farm Labor Contractor': 'pct_flc'})
level2 = level2.merge(
    d2020[['state_name', 'pct_flc']].rename(columns={'state_name': 'state'}),
    on='state', how='left'
)

# ============================================================
# [3] STANDARDIZE
# ============================================================
print("\n[3] Standardizing final covariates (z-score)...")
for col in FINAL_COVARIATES:
    mean_val = level2[col].mean()
    std_val  = level2[col].std()
    level2[col + '_z'] = (level2[col] - mean_val) / std_val
    print(f"  {col}: n={level2[col].notna().sum()}, mean={mean_val:.3f}, "
          f"SD={std_val:.3f}  → {col}_z")

z_cols = [c + '_z' for c in FINAL_COVARIATES]

# ============================================================
# [4] MERGE INTO MODEL DATASET
# ============================================================
print("\n[4] Merging into model dataset...")
df_full = df_base.merge(level2[['state'] + FINAL_COVARIATES + z_cols],
                        on='state', how='left')
df_full['state'] = pd.Categorical(df_full['state'])
print(f"  Full dataset: {len(df_full)} obs, {df_full['state'].nunique()} states")

# Final analytic sample = listwise-complete on ALL final covariates.
df_final = df_full.dropna(subset=z_cols).copy()
df_final['state'] = pd.Categorical(df_final['state'])
print(f"  Final analytic sample (listwise on {len(z_cols)} covariates): "
      f"{len(df_final)} obs, {df_final['state'].nunique()} states")

# ============================================================
# [5] MODEL FITTING HELPERS
# ============================================================

def fit_mixedlm(formula, data, pred_cols=None):
    df_fit = data.dropna(subset=[c for c in (pred_cols or []) if ':' not in c]).copy()
    df_fit['state'] = pd.Categorical(df_fit['state'])
    model = MixedLM.from_formula(formula, data=df_fit, groups=df_fit['state'],
                                 re_formula='~time')
    result = model.fit(method='lbfgs')
    return {
        'formula': formula, 'llf': result.llf,
        'var_ri': result.cov_re.iloc[0, 0], 'var_res': result.scale,
        'result': result, 'fe_params': result.fe_params,
        'pvalues': result.pvalues, 'bse': result.bse,
        'pred_cols': pred_cols or [],
        'n_obs': len(df_fit), 'n_states': df_fit['state'].nunique(),
    }


def print_params(m):
    fe, se, pv = m['fe_params'], m['bse'], m['pvalues']
    print(f"  {'Parameter':<35} {'β':>9} {'SE':>8} {'p':>7}")
    print("  " + "-" * 62)
    for param in ['Intercept'] + TIME_TERMS + m['pred_cols']:
        if param in fe:
            sig = '**' if pv[param] < .01 else ('*' if pv[param] < .05 else
                  ('+' if pv[param] < .10 else ' '))
            print(f"  {param:<35} {fe[param]:>9.4f} {se[param]:>8.4f} "
                  f"{pv[param]:>7.3f} {sig}")


# ============================================================
# [6] BASELINE — LINEAR-TIME (re-estimated on FINAL analytic sample)
# ============================================================
print("\n" + "=" * 70)
print("BASELINE — LINEAR TIME ONLY (final analytic sample)")
print("=" * 70)

m0 = fit_mixedlm(f'inspections ~ {TIME_PREFIX}', df_final, pred_cols=[])
baseline_var_ri = m0['var_ri']
icc = baseline_var_ri / (baseline_var_ri + m0['var_res'])
print_params(m0)
print(f"\n  Baseline σ²_u0 = {baseline_var_ri:.4f},  ICC = {icc*100:.1f}%")
print(f"  N = {m0['n_obs']} obs, {m0['n_states']} states")

# ============================================================
# [7] ×TIME INTERACTION SCREEN (p < .20)
# ============================================================
print("\n" + "=" * 70)
print("×TIME INTERACTION SCREEN (include in final if p < .20)")
print("=" * 70)
print(f"  {'Covariate':<24} {'×time β':>9} {'p':>8}  {'Decision'}")
print("  " + "-" * 56)

included_int = []
for zc in z_cols:
    mb = fit_mixedlm(f'inspections ~ {TIME_PREFIX} + {zc} + {zc}:time',
                     df_final, pred_cols=[zc, f'{zc}:time'])
    int_term = f'{zc}:time'
    pval = mb['pvalues'].get(int_term, float('nan'))
    beta = mb['fe_params'].get(int_term, float('nan'))
    keep = pval < 0.20
    if keep:
        included_int.append(zc)
    print(f"  {zc:<24} {beta:>9.4f} {pval:>8.3f}  "
          f"{'INCLUDE' if keep else 'exclude'}")

# ============================================================
# [8] FINAL MODEL
# ============================================================
print("\n" + "=" * 70)
print("FINAL MODEL — inspections ~ linear time + labor/DOL block")
print("=" * 70)

int_list = [f'{c}:time' for c in included_int]
final_formula = (f'inspections ~ {TIME_PREFIX} + ' + ' + '.join(z_cols) +
                 (' + ' + ' + '.join(int_list) if int_list else ''))
print(f"\nFormula: {final_formula}")
print(f"Interactions retained (p<.20): {included_int if included_int else 'none'}")

m_final = fit_mixedlm(final_formula, df_final, pred_cols=z_cols + int_list)
delta = baseline_var_ri - m_final['var_ri']
pct   = delta / baseline_var_ri * 100

print(f"\n  N = {m_final['n_obs']} obs, {m_final['n_states']} states")
print()
print_params(m_final)
print(f"\n  σ²_u0 = {m_final['var_ri']:.4f}  "
      f"[Δ={delta:+.4f}, pseudo-R² = {pct:.1f}% of between-state variance]")
print(f"  σ²_ε  = {m_final['var_res']:.4f}")
print(f"  Log-Likelihood: {m_final['llf']:.2f}")

print("\n" + "=" * 70)
print("FINAL INSPECTIONS MODEL COMPLETE")
print("=" * 70)
