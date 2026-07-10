"""
COMPREHENSIVE ("EVERY VARIABLE") MODEL — WPS Inspections

Per PI direction (2026-07): a single inspections model that enters EVERY
Level-2 covariate constructed anywhere in the project, plus the time-varying
STAG spending predictor. This is an intentional kitchen-sink specification —
many of these covariates share numerators/denominators and are collinear; it is
exploratory and complements the curated final_labor_inspections_model.py, it
does not replace it.

Outcome: total inspections (EPA + state), 2011–2019, DV mirror of the other
inspections scripts.

Time specification: LINEAR time only, matching the inspections convention set
by final_labor_inspections_model.py (the inspections series carries only a
linear trend; the cubic is a violations-only feature). The pseudo-R² baseline is
a linear-time baseline re-estimated on the same listwise-complete analytic sample.

Covariate roster (all z-scored Level-2 unless noted). n_cases is excluded and
dol_workers_cert is entered only as the h2a_per_farmworker ratio, per the same
2026-07 direction applied to the final models. LII uses the 2017 wave only.

  Spending / enforcement intensity
    spending_2017m          time-varying STAG obligations (2017 $M)   [Level-1]
    spending_per_estab      mean STAG $ / ECHO establishments (2014–2019 mean)
    SPEND_WORK              mean STAG $ / farmworkers (BLS 45-2092)
    SPEND_APP               mean STAG $ / pesticide applicators (BLS 37-3012)
    SPEND_FLC               mean STAG $ / frontline supervisors (BLS 45-1011)
    SPEND_OP                mean STAG $ / farming operations (Census 2017)
    SPEND_AREA              mean STAG $ / state land area (sq mi)
  Scale / exposure
    land_area_sqmi          state land area (sq mi)
    operations              farming operations (Census 2017)
    h2a_workers             H-2A workers (2017)
    workers_per_operation   H-2A workers / operations (2017)
  Labor structure
    lii_2017                Labor Intensity Index, 2017 wave
    h2a_per_farmworker      mean certified H-2A workers / farmworkers (BLS 45-2092)
    dol_demand_met_pct      certified / requested H-2A workers (2011–2019 mean)
    pct_flc                 % certifications via Farm Labor Contractor (2020 proxy)
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
    'MO': 'Missouri',     'MT': 'Montana',       'NE': 'Nebraska',
    'MI': 'Michigan',     'MN': 'Minnesota',     'MS': 'Mississippi',
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

STATE_LAND_AREA_SQMI = {
    'Alabama': 52420,      'Alaska': 663268,    'Arizona': 113990,
    'Arkansas': 53179,     'California': 163696, 'Colorado': 104094,
    'Connecticut': 5543,   'Delaware': 2489,    'Florida': 65758,
    'Georgia': 59425,      'Hawaii': 10932,     'Idaho': 83569,
    'Illinois': 57914,     'Indiana': 36420,    'Iowa': 56273,
    'Kansas': 82278,       'Kentucky': 40408,   'Louisiana': 52378,
    'Maine': 35380,        'Maryland': 12406,   'Massachusetts': 10554,
    'Michigan': 96714,     'Minnesota': 86936,  'Mississippi': 48432,
    'Missouri': 69707,     'Montana': 147040,   'Nebraska': 77358,
    'Nevada': 110572,      'New Hampshire': 9349,'New Jersey': 8723,
    'New Mexico': 121590,  'New York': 54555,   'North Carolina': 53819,
    'North Dakota': 70698, 'Ohio': 44826,       'Oklahoma': 69899,
    'Oregon': 98379,       'Pennsylvania': 46054,'Rhode Island': 1545,
    'South Carolina': 32020,'South Dakota': 77116,'Tennessee': 42144,
    'Texas': 268596,       'Utah': 84897,       'Vermont': 9616,
    'Virginia': 42775,     'Washington': 71298, 'West Virginia': 24230,
    'Wisconsin': 65496,    'Wyoming': 97813
}

# Inspections convention: LINEAR time only.
TIME_TERMS  = ['time']
TIME_PREFIX = ' + '.join(TIME_TERMS)

print("=" * 70)
print("COMPREHENSIVE INSPECTIONS MODEL — EVERY PROJECT COVARIATE (LINEAR TIME)")
print("=" * 70)

# ============================================================
# [1] BASE INSPECTIONS + SPENDING DATA
# ============================================================
print("\n[1] Loading base inspections and spending data...")

echo_df = pd.read_csv('/Users/keshavgoel/Research/data/raw/establishments_data.csv', index_col=0)
echo_df.index = echo_df.index.str.strip()
echo_df.index = echo_df.index.map(lambda x: STATE_NAME_MAPPING.get(x, x))

echo_long_list = []
for year in range(2011, 2020):
    col_epa   = f'inspections-epa-{year}'
    col_state = f'inspections-state-{year}'
    epa_vals = pd.to_numeric(
        echo_df[col_epa] if col_epa in echo_df.columns else pd.Series(np.nan, index=echo_df.index),
        errors='coerce')
    state_vals = pd.to_numeric(
        echo_df[col_state] if col_state in echo_df.columns else pd.Series(np.nan, index=echo_df.index),
        errors='coerce')
    total = epa_vals.add(state_vals, fill_value=0)
    echo_long_list.append(pd.DataFrame({
        'state': echo_df.index, 'year': year, 'inspections': total.values
    }))

echo_long = pd.concat(echo_long_list, ignore_index=True)
echo_long = echo_long[echo_long['state'].isin(US_STATES_50)].copy()
echo_long['inspections'] = pd.to_numeric(echo_long['inspections'], errors='coerce')
echo_long = echo_long.dropna(subset=['inspections'])
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
spending_agg['spending_2017m'] = spending_agg['spending_2017'] / 1_000_000

df_base = echo_long.merge(
    spending_agg[['state', 'year', 'spending_2017m']],
    on=['state', 'year'], how='left'
).dropna(subset=['spending_2017m']).copy()
df_base['state'] = pd.Categorical(df_base['state'])
print(f"  Base dataset: {len(df_base)} obs, {df_base['state'].nunique()} states")

# ============================================================
# [2] CONSTRUCT EVERY LEVEL-2 COVARIATE
# ============================================================
print("\n[2] Constructing every Level-2 covariate in the project...")

level2 = pd.DataFrame({'state': US_STATES_50})

# --- 2a. State mean inflation-adjusted spending (numerator for all SPEND_*) ---
mean_spending = spending_agg.groupby('state')['spending_2017'].mean().reset_index()
mean_spending.columns = ['state', 'mean_spending_2017']
level2 = level2.merge(mean_spending, on='state', how='left')

# --- 2b. Land area ---
level2['land_area_sqmi'] = level2['state'].map(STATE_LAND_AREA_SQMI)

# --- 2c. ECHO establishments (2014–2019 mean; CO/CT 2014 imputed from 2015) ---
est_cols = [f'establishments-{y}' for y in range(2014, 2020)]
est_df = echo_df[echo_df.index.isin(US_STATES_50)][est_cols].copy()
for c in est_cols:
    est_df[c] = pd.to_numeric(est_df[c], errors='coerce')
for st in ['Colorado', 'Connecticut']:
    if pd.isna(est_df.loc[st, 'establishments-2014']):
        est_df.loc[st, 'establishments-2014'] = est_df.loc[st, 'establishments-2015']
est_df['establishments_mean'] = est_df[est_cols].mean(axis=1)
level2['establishments_mean'] = level2['state'].map(est_df['establishments_mean'])
level2['spending_per_estab'] = level2['mean_spending_2017'] / level2['establishments_mean']

# --- 2d. Census of Agriculture 2017 (operations, H-2A workers) ---
h2a_2017 = pd.read_csv('/Users/keshavgoel/Research/data/raw/h2a_state_summary_2017.csv')
h2a_2017['state'] = h2a_2017['state_code'].map(STATE_ABBREV_TO_NAME)
h2a_2017 = h2a_2017[h2a_2017['state'].isin(US_STATES_50)].copy()
level2 = level2.merge(
    h2a_2017[['state', 'h2a_workers', 'operations', 'workers_per_operation']],
    on='state', how='left'
)

# --- 2e. BLS OEWS employment (2011 snapshot; 3 occupations) ---
bls = pd.read_csv('/Users/keshavgoel/Research/data/raw/bls_oews_panel.csv')
bls_wide = bls.pivot_table(index='area_title', columns='occ_code',
                           values='tot_emp', aggfunc='first').reset_index()
bls_wide.columns.name = None
bls_wide.rename(columns={
    'area_title': 'state',
    '37-3012': 'emp_pesticide_app',
    '45-1011': 'emp_flc_supervisor',
    '45-2092': 'emp_farmworker',
}, inplace=True)
level2 = level2.merge(
    bls_wide[['state', 'emp_farmworker', 'emp_pesticide_app', 'emp_flc_supervisor']],
    on='state', how='left'
)

# --- 2f. BLS-normalized spending variables ---
level2['SPEND_WORK'] = level2['mean_spending_2017'] / level2['emp_farmworker']
level2['SPEND_APP']  = level2['mean_spending_2017'] / level2['emp_pesticide_app']
level2['SPEND_FLC']  = level2['mean_spending_2017'] / level2['emp_flc_supervisor']
level2['SPEND_OP']   = level2['mean_spending_2017'] / level2['operations']
level2['SPEND_AREA'] = level2['mean_spending_2017'] / level2['land_area_sqmi']

# --- 2g. Labor Intensity Index (2017 wave only) ---
li_17 = pd.read_csv('/Users/keshavgoel/Research/data/raw/labor_intensity_index_2017.csv')
li_17['state'] = li_17['state_name'].str.title()
level2 = level2.merge(li_17[['state', 'Labor_Intensity_Index']].rename(
    columns={'Labor_Intensity_Index': 'lii_2017'}), on='state', how='left')

# --- 2h. DOL H-2A workers → demand met + h2a_per_farmworker ratio ---
#     n_cases excluded; dol_workers_cert used only as the ratio numerator.
dol1 = pd.read_csv('/Users/keshavgoel/Research/data/raw/dol_var1_workers_by_state_annual.csv')
dol_study = dol1[dol1['year'].between(2011, 2019) & (dol1['year'] != 2013)].copy()
dol_means = dol_study.groupby('state').agg(
    dol_workers_cert=('workers_certified', 'mean'),
    dol_demand_met_pct=('demand_met_pct', lambda x: x[x != np.inf].mean())
).reset_index()
dol_means['state_name'] = dol_means['state'].map(STATE_ABBREV_TO_NAME)
level2 = level2.merge(dol_means[['state_name', 'dol_workers_cert',
                                 'dol_demand_met_pct']].rename(
    columns={'state_name': 'state'}), on='state', how='left')
level2['h2a_per_farmworker'] = level2['dol_workers_cert'] / level2['emp_farmworker']

# --- 2i. DOL employer type (2020 proxy) → pct_flc ---
dol2 = pd.read_csv('/Users/keshavgoel/Research/data/raw/dol_var2_employer_type_annual.csv')
d2020 = dol2[dol2['year'] == 2020].copy()
d2020['state_name'] = d2020['state'].map(STATE_ABBREV_TO_NAME)
d2020 = d2020.rename(columns={'pct_Farm Labor Contractor': 'pct_flc'})
level2 = level2.merge(
    d2020[['state_name', 'pct_flc']].rename(columns={'state_name': 'state'}),
    on='state', how='left'
)

# ============================================================
# [3] STANDARDIZE ALL COVARIATES
# ============================================================
print("\n[3] Standardizing every covariate (z-score)...")

# Level-2 covariate roster (order = presentation order).
L2_COVARIATES = [
    'spending_per_estab',
    'SPEND_WORK', 'SPEND_APP', 'SPEND_FLC', 'SPEND_OP', 'SPEND_AREA',
    'land_area_sqmi', 'operations', 'h2a_workers', 'workers_per_operation',
    'lii_2017', 'h2a_per_farmworker', 'dol_demand_met_pct', 'pct_flc',
]
for col in L2_COVARIATES:
    mu, sd = level2[col].mean(), level2[col].std()
    level2[col + '_z'] = (level2[col] - mu) / sd
    print(f"  {col:<22} n={level2[col].notna().sum():>2}  mean={mu:>12.4f}  SD={sd:>12.4f}  → {col}_z")

z_l2 = [c + '_z' for c in L2_COVARIATES]

# ============================================================
# [4] MERGE + STANDARDIZE LEVEL-1 SPENDING
# ============================================================
print("\n[4] Merging Level-2 covariates and standardizing Level-1 spending...")
df_full = df_base.merge(level2[['state'] + L2_COVARIATES + z_l2], on='state', how='left')
mu_sp, sd_sp = df_full['spending_2017m'].mean(), df_full['spending_2017m'].std()
df_full['spending_2017m_z'] = (df_full['spending_2017m'] - mu_sp) / sd_sp
df_full['state'] = pd.Categorical(df_full['state'])

# Full predictor roster = Level-1 spending + all Level-2 covariates.
ALL_Z = ['spending_2017m_z'] + z_l2
print(f"  Full dataset: {len(df_full)} obs, {df_full['state'].nunique()} states")
print(f"  Total predictors entered: {len(ALL_Z)} "
      f"(1 Level-1 spending + {len(z_l2)} Level-2)")

# Listwise-complete analytic sample on ALL predictors.
df_final = df_full.dropna(subset=ALL_Z).copy()
df_final['state'] = pd.Categorical(df_final['state'])
print(f"  Analytic sample (listwise on all {len(ALL_Z)} predictors): "
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
    print(f"  {'Parameter':<28} {'β':>10} {'SE':>9} {'p':>7}")
    print("  " + "-" * 58)
    for param in ['Intercept'] + TIME_TERMS + m['pred_cols']:
        if param in fe:
            sig = '**' if pv[param] < .01 else ('*' if pv[param] < .05 else
                  ('+' if pv[param] < .10 else ' '))
            print(f"  {param:<28} {fe[param]:>10.4f} {se[param]:>9.4f} "
                  f"{pv[param]:>7.3f} {sig}")


# ============================================================
# [6] BASELINE — LINEAR TIME (final analytic sample)
# ============================================================
print("\n" + "=" * 70)
print("BASELINE — LINEAR TIME ONLY (analytic sample)")
print("=" * 70)
m0 = fit_mixedlm(f'inspections ~ {TIME_PREFIX}', df_final, pred_cols=[])
baseline_var_ri = m0['var_ri']
icc = baseline_var_ri / (baseline_var_ri + m0['var_res'])
print_params(m0)
print(f"\n  Baseline σ²_u0 = {baseline_var_ri:.4f},  ICC = {icc*100:.1f}%")
print(f"  N = {m0['n_obs']} obs, {m0['n_states']} states")

# ============================================================
# [7] FULL MAIN-EFFECTS MODEL — EVERY VARIABLE AT ONCE
# ============================================================
print("\n" + "=" * 70)
print("FULL MODEL — inspections ~ linear time + EVERY covariate (main effects)")
print("=" * 70)
full_formula = f'inspections ~ {TIME_PREFIX} + ' + ' + '.join(ALL_Z)
m_full = fit_mixedlm(full_formula, df_final, pred_cols=ALL_Z)
delta = baseline_var_ri - m_full['var_ri']
pct   = delta / baseline_var_ri * 100
print(f"\n  N = {m_full['n_obs']} obs, {m_full['n_states']} states")
print()
print_params(m_full)
print(f"\n  σ²_u0 = {m_full['var_ri']:.4f}  "
      f"[Δ={delta:+.4f}, pseudo-R² = {pct:.1f}% of between-state variance]")
print(f"  σ²_ε  = {m_full['var_res']:.4f}")
print(f"  Log-Likelihood: {m_full['llf']:.2f}")

# ============================================================
# [8] ×TIME INTERACTION SCREEN (one-at-a-time, p < .20)
# ============================================================
print("\n" + "=" * 70)
print("×TIME INTERACTION SCREEN (include in augmented model if p < .20)")
print("=" * 70)
print(f"  {'Covariate':<24} {'×time β':>10} {'p':>8}  {'Decision'}")
print("  " + "-" * 58)
included_int = []
for zc in ALL_Z:
    mb = fit_mixedlm(f'inspections ~ {TIME_PREFIX} + {zc} + {zc}:time',
                     df_final, pred_cols=[zc, f'{zc}:time'])
    pval = mb['pvalues'].get(f'{zc}:time', float('nan'))
    beta = mb['fe_params'].get(f'{zc}:time', float('nan'))
    keep = pval < 0.20
    if keep:
        included_int.append(zc)
    print(f"  {zc:<24} {beta:>10.4f} {pval:>8.3f}  "
          f"{'INCLUDE' if keep else 'exclude'}")

# ============================================================
# [9] AUGMENTED MODEL — main effects + retained ×time interactions
# ============================================================
print("\n" + "=" * 70)
print("AUGMENTED MODEL — every covariate + ×time interactions (p<.20)")
print("=" * 70)
int_list = [f'{c}:time' for c in included_int]
aug_formula = (f'inspections ~ {TIME_PREFIX} + ' + ' + '.join(ALL_Z) +
               (' + ' + ' + '.join(int_list) if int_list else ''))
print(f"\nInteractions retained (p<.20): {included_int if included_int else 'none'}")
m_aug = fit_mixedlm(aug_formula, df_final, pred_cols=ALL_Z + int_list)
delta_a = baseline_var_ri - m_aug['var_ri']
pct_a   = delta_a / baseline_var_ri * 100
print(f"\n  N = {m_aug['n_obs']} obs, {m_aug['n_states']} states")
print()
print_params(m_aug)
print(f"\n  σ²_u0 = {m_aug['var_ri']:.4f}  "
      f"[Δ={delta_a:+.4f}, pseudo-R² = {pct_a:.1f}% of between-state variance]")
print(f"  σ²_ε  = {m_aug['var_res']:.4f}")
print(f"  Log-Likelihood: {m_aug['llf']:.2f}")

print("\n" + "=" * 70)
print("COMPREHENSIVE INSPECTIONS MODEL COMPLETE")
print("=" * 70)
