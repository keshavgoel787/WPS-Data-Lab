"""
Labor Intensity & DOL H-2A Covariate Models — WPS Inspections Analysis

Zimmerman one-at-a-time stepwise for six new Level-2 state covariates
derived from five new datasets. Mirrors labor_covariates_violations_models.py;
only the outcome differs.

  lii_2017_z        — Labor Intensity Index, 2017 Census of Agriculture
  lii_2012_z        — Labor Intensity Index, 2012 Census of Agriculture
  dol_workers_cert_z — Mean annual H-2A certified workers, DOL 2011–2019
  dol_n_cases_z      — Mean annual H-2A employer cases, DOL 2011–2019
  dol_demand_met_z   — Mean % demand met, DOL 2011–2019 (excl. 2013, 2014)
  pct_flc_z          — % H-2A workers via Farm Labor Contractors (2020 proxy)
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
print("LABOR COVARIATE STEPWISE MODELS — WPS INSPECTIONS ANALYSIS")
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
# [2] CONSTRUCT NEW LEVEL-2 COVARIATES
# ============================================================
print("\n[2] Constructing new Level-2 covariates...")

level2 = pd.DataFrame({'state': US_STATES_50})

# --- 2a. Labor Intensity Index ---
print("\n  [2a] Labor Intensity Index (Census of Agriculture)...")
li_17 = pd.read_csv('/Users/keshavgoel/Research/data/raw/labor_intensity_index_2017.csv')
li_17['state'] = li_17['state_name'].str.title()
li_12 = pd.read_csv('/Users/keshavgoel/Research/data/raw/labor_intensity_index_2012.csv')
li_12['state'] = li_12['state_name'].str.title()

level2 = level2.merge(li_17[['state', 'Labor_Intensity_Index']].rename(
    columns={'Labor_Intensity_Index': 'lii_2017'}), on='state', how='left')
level2 = level2.merge(li_12[['state', 'Labor_Intensity_Index']].rename(
    columns={'Labor_Intensity_Index': 'lii_2012'}), on='state', how='left')

print(f"    lii_2017: {level2['lii_2017'].notna().sum()} states, "
      f"mean={level2['lii_2017'].mean():.3f}, SD={level2['lii_2017'].std():.3f}")
print(f"    lii_2012: {level2['lii_2012'].notna().sum()} states, "
      f"mean={level2['lii_2012'].mean():.3f}, SD={level2['lii_2012'].std():.3f}")
print(f"    Pearson r(lii_2012, lii_2017): {level2['lii_2012'].corr(level2['lii_2017']):.4f}")

# --- 2b. DOL H-2A Annual Workers ---
print("\n  [2b] DOL H-2A annual workers (2011–2019)...")
dol1 = pd.read_csv('/Users/keshavgoel/Research/data/raw/dol_var1_workers_by_state_annual.csv')
dol_study = dol1[dol1['year'].between(2011, 2019) & (dol1['year'] != 2013)].copy()
print(f"    Years used: {sorted(dol_study['year'].unique())} (2013 excluded: missing)")

dol_means = dol_study.groupby('state').agg(
    dol_workers_cert=('workers_certified', 'mean'),
    dol_n_cases=('n_cases', 'mean'),
    dol_demand_met_pct=('demand_met_pct', lambda x: x[x != np.inf].mean())
).reset_index()
dol_means['state_name'] = dol_means['state'].map(STATE_ABBREV_TO_NAME)
level2 = level2.merge(dol_means[['state_name', 'dol_workers_cert',
                                  'dol_n_cases', 'dol_demand_met_pct']].rename(
    columns={'state_name': 'state'}), on='state', how='left')

for v in ['dol_workers_cert', 'dol_n_cases', 'dol_demand_met_pct']:
    print(f"    {v}: {level2[v].notna().sum()} states, "
          f"mean={level2[v].mean():.1f}, SD={level2[v].std():.1f}")

# --- 2c. DOL Employer Type (2020 proxy) ---
print("\n  [2c] DOL employer type — pct Farm Labor Contractor (2020 proxy)...")
dol2 = pd.read_csv('/Users/keshavgoel/Research/data/raw/dol_var2_employer_type_annual.csv')
d2020 = dol2[dol2['year'] == 2020].copy()
d2020['state_name'] = d2020['state'].map(STATE_ABBREV_TO_NAME)
d2020 = d2020.rename(columns={'pct_Farm Labor Contractor': 'pct_flc'})
level2 = level2.merge(
    d2020[['state_name', 'pct_flc']].rename(columns={'state_name': 'state'}),
    on='state', how='left'
)
print(f"    pct_flc (2020): {level2['pct_flc'].notna().sum()} states, "
      f"mean={level2['pct_flc'].mean():.1f}%, SD={level2['pct_flc'].std():.1f}%")

# ============================================================
# [3] STANDARDIZE
# ============================================================
print("\n[3] Standardizing new covariates (z-score)...")

new_covariates = ['lii_2017', 'lii_2012', 'dol_workers_cert',
                  'dol_n_cases', 'dol_demand_met_pct', 'pct_flc']

for col in new_covariates:
    mean_val = level2[col].mean()
    std_val  = level2[col].std()
    level2[col + '_z'] = (level2[col] - mean_val) / std_val
    print(f"  {col}: mean={mean_val:.3f}, SD={std_val:.3f}  → {col}_z")

# ============================================================
# [4] MERGE INTO MODEL DATASET
# ============================================================
print("\n[4] Merging into model dataset...")
z_cols = [c + '_z' for c in new_covariates]
df_full = df_base.merge(level2[['state'] + new_covariates + z_cols], on='state', how='left')
df_full['state'] = pd.Categorical(df_full['state'])
print(f"  Final dataset: {len(df_full)} obs, {df_full['state'].nunique()} states")
for zc in z_cols:
    n_miss = df_full[zc].isna().sum()
    print(f"  {zc}: {n_miss} missing obs")

# ============================================================
# [5] MODEL FITTING HELPERS
# ============================================================

def fit_mixedlm(formula, data, label, pred_cols=None):
    df_fit = data.dropna(subset=[c for c in (pred_cols or []) if ':' not in c]).copy()
    df_fit['state'] = pd.Categorical(df_fit['state'])
    model = MixedLM.from_formula(formula, data=df_fit, groups=df_fit['state'], re_formula='~time')
    result = model.fit(method='lbfgs')
    return {
        'label': label, 'formula': formula,
        'llf': result.llf,
        'var_ri': result.cov_re.iloc[0, 0],
        'var_res': result.scale,
        'result': result,
        'fe_params': result.fe_params,
        'pvalues': result.pvalues,
        'bse': result.bse,
        'pred_cols': pred_cols or [],
        'n_obs': len(df_fit),
        'n_states': df_fit['state'].nunique(),
    }


def print_model(m, baseline_var_ri):
    delta = baseline_var_ri - m['var_ri']
    pct   = delta / baseline_var_ri * 100
    fe    = m['fe_params']
    se    = m['bse']
    pv    = m['pvalues']
    print(f"    Formula: {m['formula']}")
    print(f"    N = {m['n_obs']} obs, {m['n_states']} states")
    print(f"    {'Parameter':<35} {'β':>9} {'SE':>8} {'p':>7}")
    print("    " + "-" * 62)
    for param in ['Intercept', 'time', 'time2', 'time3'] + m['pred_cols']:
        if param in fe:
            sig = '**' if pv[param] < .01 else ('*' if pv[param] < .05 else
                  ('+' if pv[param] < .10 else ' '))
            print(f"    {param:<35} {fe[param]:>9.4f} {se[param]:>8.4f} {pv[param]:>7.3f} {sig}")
    print(f"    σ²_u0: {m['var_ri']:.4f}  [Δ={delta:+.4f}, {pct:.1f}%]  σ²_ε: {m['var_res']:.4f}  "
          f"LogLik: {m['llf']:.2f}")


# ============================================================
# [6] BASELINE MODEL
# ============================================================
print("\n" + "=" * 70)
print("MODEL 0 — BASELINE (TIME POLYNOMIAL ONLY)")
print("=" * 70)

m0 = fit_mixedlm('inspections ~ time + time2 + time3',
                 df_full, 'M0: Baseline', pred_cols=[])
print(m0['result'].summary())

baseline_var_ri = m0['var_ri']
icc = baseline_var_ri / (baseline_var_ri + m0['var_res'])
print(f"\nBaseline σ²_u0 = {baseline_var_ri:.4f},  ICC = {icc*100:.1f}%")

# ============================================================
# [7] STEPWISE MODELS
# ============================================================
print("\n" + "=" * 70)
print("STEPWISE MODELS — LABOR/DOL COVARIATES (DV = INSPECTIONS)")
print("=" * 70)

predictor_sequence = [
    ('lii_2017_z',           'Labor Intensity Index 2017 (Census of Agriculture)'),
    ('lii_2012_z',           'Labor Intensity Index 2012 (Census of Agriculture)'),
    ('dol_workers_cert_z',   'DOL H-2A certified workers mean (2011–2019 excl. 2013)'),
    ('dol_n_cases_z',        'DOL H-2A employer cases mean (2011–2019 excl. 2013)'),
    ('dol_demand_met_pct_z', 'DOL H-2A demand met % mean (excl. 2013, 2014)'),
    ('pct_flc_z',            'DOL % Farm Labor Contractor placement (2020 proxy)'),
]

all_models = [m0]
interaction_pvals = {}

for pred_col, pred_label in predictor_sequence:
    print(f"\n{'─'*70}")
    print(f"PREDICTOR: {pred_label}")
    print(f"{'─'*70}")

    ma = fit_mixedlm(
        f'inspections ~ time + time2 + time3 + {pred_col}',
        df_full, f'{pred_col} (main)', pred_cols=[pred_col]
    )
    print(f"\n  (a) Main effect only:")
    print_model(ma, baseline_var_ri)
    all_models.append(ma)

    mb = fit_mixedlm(
        f'inspections ~ time + time2 + time3 + {pred_col} + {pred_col}:time',
        df_full, f'{pred_col} (main + ×time)',
        pred_cols=[pred_col, f'{pred_col}:time']
    )
    print(f"\n  (b) Main + ×time interaction:")
    print_model(mb, baseline_var_ri)
    all_models.append(mb)

    int_term = f'{pred_col}:time'
    pval_int = mb['pvalues'].get(int_term, float('nan'))
    interaction_pvals[pred_col] = pval_int
    flag = '<.20 → INCLUDE in combined' if pval_int < 0.20 else '≥.20 → exclude from combined'
    print(f"\n  ×time p = {pval_int:.3f}  [{flag}]")

# ============================================================
# [8] SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY — BETWEEN-STATE VARIANCE DECOMPOSITION (DV = INSPECTIONS)")
print(f"Baseline σ²_u0 = {baseline_var_ri:.4f}  (ICC = {icc*100:.1f}%)")
print("=" * 70)

hdr = f"  {'Predictor':<40} {'N':>5} {'σ²_u0':>8} {'Δσ²_u0':>8} {'%Expl':>7} {'Main p':>8} {'×time p':>8}"
print(hdr)
print("  " + "-" * len(hdr))
for i, (pred_col, _) in enumerate(predictor_sequence):
    ma = all_models[1 + i * 2]
    mb = all_models[2 + i * 2]
    delta_a = baseline_var_ri - ma['var_ri']
    pct_a   = delta_a / baseline_var_ri * 100
    p_main  = ma['pvalues'].get(pred_col, float('nan'))
    p_int   = interaction_pvals[pred_col]
    print(f"  {pred_col:<40} {ma['n_obs']:>5} {ma['var_ri']:>8.4f} "
          f"{delta_a:>+8.4f} {pct_a:>6.1f}% {p_main:>8.3f} {p_int:>8.3f}")

# ============================================================
# [9] COMBINED MODEL
# ============================================================
print("\n" + "=" * 70)
print("COMBINED MODEL — ALL NEW COVARIATES SIMULTANEOUSLY")
print("=" * 70)

included_int = [p for p, pv in interaction_pvals.items() if pv < 0.20]
print(f"\nInteractions included (p<.20): {included_int if included_int else 'none'}")

z_list   = [c for c in z_cols]
int_list = [f'{c}:time' for c in included_int]
combined_formula = ('inspections ~ time + time2 + time3 + ' +
                    ' + '.join(z_list) +
                    (' + ' + ' + '.join(int_list) if int_list else ''))
print(f"Formula: {combined_formula}")

m_comb = fit_mixedlm(combined_formula, df_full, 'Combined (all new)',
                     pred_cols=z_list + int_list)
delta_c = baseline_var_ri - m_comb['var_ri']
pct_c   = delta_c / baseline_var_ri * 100

print(f"\n  N = {m_comb['n_obs']} obs, {m_comb['n_states']} states")
print(f"\n  {'Parameter':<35} {'β':>9} {'SE':>8} {'p':>7}")
print("  " + "-" * 62)
for param in ['Intercept', 'time', 'time2', 'time3'] + m_comb['pred_cols']:
    if param in m_comb['fe_params']:
        sig = '**' if m_comb['pvalues'][param] < .01 else (
              '*' if m_comb['pvalues'][param] < .05 else (
              '+' if m_comb['pvalues'][param] < .10 else ' '))
        print(f"  {param:<35} {m_comb['fe_params'][param]:>9.4f} "
              f"{m_comb['bse'][param]:>8.4f} {m_comb['pvalues'][param]:>7.3f} {sig}")

print(f"\n  σ²_u0 = {m_comb['var_ri']:.4f}  "
      f"[Δ={delta_c:+.4f}, {pct_c:.1f}% of baseline explained]")
print(f"  σ²_ε  = {m_comb['var_res']:.4f}")
print(f"  Log-Likelihood: {m_comb['llf']:.2f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
