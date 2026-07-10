"""
BLS-Normalized Spending Models — Zimmerman/HLM Approach
WPS Enforcement Analysis

Five new Level-2 spending variables, each = inflation-adjusted mean STAG funding
divided by a different state-level denominator:

  SPEND_WORK  — spending / farmworkers (BLS OCC 45-2092, 2011 snapshot)
  SPEND_APP   — spending / pesticide applicators (BLS OCC 37-3012, 2011 snapshot)
                [Note: user specified 47-3012 but that code is not in the BLS file;
                 37-3012 "Pesticide Handlers, Sprayers, and Applicators, Vegetation"
                 is the only pesticide applicator occupation present and is used here]
  SPEND_FLC   — spending / frontline supervisors (BLS OCC 45-1011, 2011 snapshot)
  SPEND_OP    — spending / farming operations (Census of Agriculture 2017;
                user's "observations_Z" = operations_z in this project's Level-2 data)
  SPEND_AREA  — spending / state land area in sq miles (US Census Bureau)

Analytic sequence (Zimmerman 2000):
  Step 1-2: For each variable, fit (a) main effect only, (b) main + ×time interaction
  Step 3:   Combined model with all five variables; ×time interactions included only
            for variables whose interaction p-value was <.20 in Step 2.
"""

import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS  (identical to stepwise_zimmerman_models.py)
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

print("=" * 70)
print("BLS-NORMALIZED SPENDING MODELS — WPS VIOLATIONS ANALYSIS")
print("=" * 70)

# ============================================================
# [1] BASE VIOLATIONS + SPENDING DATA
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
echo_long_filtered = echo_long[echo_long['state'].isin(US_STATES_50)].copy()
echo_long_filtered['violations'] = pd.to_numeric(echo_long_filtered['violations'], errors='coerce')
echo_long_filtered = echo_long_filtered.dropna(subset=['violations'])
echo_long_filtered['time']  = echo_long_filtered['year'] - 2017
echo_long_filtered['time2'] = echo_long_filtered['time'] ** 2
echo_long_filtered['time3'] = echo_long_filtered['time'] ** 3

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

df_model = echo_long_filtered.merge(
    spending_agg[['state', 'year', 'spending_2017', 'spending_2017m']],
    on=['state', 'year'], how='left'
).dropna(subset=['spending_2017m']).copy()
df_model['state'] = pd.Categorical(df_model['state'])
print(f"  Base dataset: {len(df_model)} obs, {df_model['state'].nunique()} states, "
      f"years {df_model['year'].min()}–{df_model['year'].max()}")

# ============================================================
# [2] BUILD LEVEL-2 SPENDING DENOMINATORS
# ============================================================
print("\n[2] Building Level-2 spending denominators...")

# --- 2a. State-level mean inflation-adjusted spending (2011-2019) ---
mean_spending = spending_agg.groupby('state')['spending_2017'].mean().reset_index()
mean_spending.columns = ['state', 'mean_spending_2017']

level2 = pd.DataFrame({'state': US_STATES_50})
level2 = level2.merge(mean_spending, on='state', how='left')
level2['land_area_sqmi'] = level2['state'].map(STATE_LAND_AREA_SQMI)

# --- 2b. Farming operations (Census of Agriculture 2017 via Yuri's data) ---
h2a_2017 = pd.read_csv('/Users/keshavgoel/Research/data/raw/h2a_state_summary_2017.csv')
h2a_2017['state'] = h2a_2017['state_code'].map(STATE_ABBREV_TO_NAME)
h2a_2017 = h2a_2017[h2a_2017['state'].isin(US_STATES_50)].copy()
level2 = level2.merge(h2a_2017[['state', 'operations']], on='state', how='left')

# --- 2c. BLS OEWS employment counts (2011 snapshot) ---
print("\n  Loading BLS OEWS employment data (2011)...")
print("  Note: OCC 47-3012 not found in BLS file; using 37-3012")
print("        (Pesticide Handlers, Sprayers, and Applicators, Vegetation)")

bls = pd.read_csv('/Users/keshavgoel/Downloads/bls_oews_panel.csv')
bls_wide = (
    bls.pivot_table(index='area_title', columns='occ_code', values='tot_emp', aggfunc='first')
    .reset_index()
)
bls_wide.columns.name = None
bls_wide.rename(columns={
    'area_title': 'state',
    '37-3012': 'emp_pesticide_app',
    '45-1011': 'emp_flc_supervisor',
    '45-2092': 'emp_farmworker',
}, inplace=True)

level2 = level2.merge(bls_wide[['state','emp_farmworker','emp_pesticide_app','emp_flc_supervisor']],
                      on='state', how='left')

# Report BLS coverage
for col, label in [('emp_farmworker','SPEND_WORK (45-2092)'),
                   ('emp_pesticide_app','SPEND_APP (37-3012)'),
                   ('emp_flc_supervisor','SPEND_FLC (45-1011)')]:
    missing = level2[level2[col].isna()]['state'].tolist()
    n = len(missing)
    if n:
        print(f"  {label}: {n} states missing BLS data → excluded from that model")
        print(f"    ({', '.join(missing)})")

# ============================================================
# [3] COMPUTE AND STANDARDIZE SPEND VARIABLES
# ============================================================
print("\n[3] Computing normalized spending variables...")

level2['SPEND_WORK'] = level2['mean_spending_2017'] / level2['emp_farmworker']
level2['SPEND_APP']  = level2['mean_spending_2017'] / level2['emp_pesticide_app']
level2['SPEND_FLC']  = level2['mean_spending_2017'] / level2['emp_flc_supervisor']
level2['SPEND_OP']   = level2['mean_spending_2017'] / level2['operations']
level2['SPEND_AREA'] = level2['mean_spending_2017'] / level2['land_area_sqmi']

spend_vars = ['SPEND_WORK', 'SPEND_APP', 'SPEND_FLC', 'SPEND_OP', 'SPEND_AREA']

print(f"\n  {'Variable':<12} {'N non-missing':>14} {'Mean':>12} {'SD':>12} {'Min':>12} {'Max':>12}")
print("  " + "-" * 74)
for v in spend_vars:
    s = level2[v].dropna()
    print(f"  {v:<12} {len(s):>14} {s.mean():>12.4f} {s.std():>12.4f} "
          f"{s.min():>12.4f} {s.max():>12.4f}")

print("\n  Standardizing (z-score) each variable...")
for v in spend_vars:
    col_z = v + '_z'
    mean_v = level2[v].mean()
    std_v  = level2[v].std()
    level2[col_z] = (level2[v] - mean_v) / std_v
    print(f"    {v}: mean={mean_v:.4f}, SD={std_v:.4f}  → {col_z}")

spend_z_vars = [v + '_z' for v in spend_vars]

# ============================================================
# [4] MERGE INTO MODEL DATASET
# ============================================================
print("\n[4] Merging into model dataset...")
df_full = df_model.merge(level2[['state'] + spend_vars + spend_z_vars], on='state', how='left')
df_full['state'] = pd.Categorical(df_full['state'])
print(f"  Final dataset: {len(df_full)} obs, {df_full['state'].nunique()} states")
for vz in spend_z_vars:
    n_miss = df_full[vz].isna().sum()
    n_states = df_full[df_full[vz].isna()]['state'].nunique()
    print(f"  {vz}: {n_miss} missing obs ({n_states} states excluded from that model)")

# ============================================================
# [5] MODEL FITTING HELPERS
# ============================================================

def fit_mixedlm(formula, data, label, pred_cols=None):
    """Fit MixedLM with random intercept + random slope for time."""
    df_fit = data.dropna(subset=[c for c in (pred_cols or []) if ':' not in c]).copy()
    df_fit['state'] = pd.Categorical(df_fit['state'])
    model = MixedLM.from_formula(formula, data=df_fit, groups=df_fit['state'], re_formula='~time')
    result = model.fit(method='lbfgs')
    return {
        'label':     label,
        'formula':   formula,
        'llf':       result.llf,
        'var_ri':    result.cov_re.iloc[0, 0],
        'var_rs':    result.cov_re.iloc[1, 1],
        'var_res':   result.scale,
        'result':    result,
        'fe_params': result.fe_params,
        'pvalues':   result.pvalues,
        'bse':       result.bse,
        'pred_cols': pred_cols or [],
        'n_obs':     len(df_fit),
        'n_states':  df_fit['state'].nunique(),
    }


def print_model(m, baseline_var_ri):
    delta = baseline_var_ri - m['var_ri']
    pct   = delta / baseline_var_ri * 100
    fe    = m['fe_params']
    se    = m['bse']
    pv    = m['pvalues']
    print(f"\n    Formula:  {m['formula']}")
    print(f"    N = {m['n_obs']} obs, {m['n_states']} states")
    print(f"    {'Parameter':<35} {'β':>9} {'SE':>8} {'p':>7}")
    print("    " + "-" * 62)
    for param in ['Intercept', 'time', 'time2', 'time3'] + m['pred_cols']:
        if param in fe:
            sig = '**' if pv[param] < .01 else ('*' if pv[param] < .05 else
                  ('+' if pv[param] < .10 else (' ' if pv[param] < .20 else ' ')))
            print(f"    {param:<35} {fe[param]:>9.4f} {se[param]:>8.4f} {pv[param]:>7.3f} {sig}")
    print(f"    σ²_u0 (between-state): {m['var_ri']:.4f}  "
          f"[Δ={delta:+.4f}, {pct:.1f}% of baseline variance explained]")
    print(f"    σ²_ε  (within-state):  {m['var_res']:.4f}")
    print(f"    Log-Likelihood: {m['llf']:.2f}")


# ============================================================
# [6] BASELINE MODEL (TIME ONLY)
# ============================================================
print("\n" + "=" * 70)
print("MODEL 0 — BASELINE (TIME POLYNOMIAL ONLY)")
print("=" * 70)

m0 = fit_mixedlm(
    'violations ~ time + time2 + time3',
    df_full, 'M0: Baseline (time only)', pred_cols=[]
)
print(m0['result'].summary())

baseline_var_ri = m0['var_ri']
icc = baseline_var_ri / (baseline_var_ri + m0['var_res'])
print(f"\nBaseline between-state variance (σ²_u0): {baseline_var_ri:.4f}")
print(f"Baseline within-state variance  (σ²_ε):  {m0['var_res']:.4f}")
print(f"Baseline ICC: {icc:.4f}  ({icc*100:.1f}% of variance is between states)")

# ============================================================
# [7] STEPWISE MODELS — ONE SPEND VARIABLE AT A TIME
# ============================================================
print("\n" + "=" * 70)
print("STEPWISE MODELS — BLS-NORMALIZED SPENDING VARIABLES")
print("=" * 70)
print("\nFor each variable: (a) main effect only, (b) main + ×time interaction")
print("  Significance flags: ** p<.01  * p<.05  + p<.10  (unmarked) p<.20 for interactions")

spend_labels = {
    'SPEND_WORK_z': 'SPEND_WORK (spending / farmworkers, OCC 45-2092)',
    'SPEND_APP_z':  'SPEND_APP  (spending / pest. applicators, OCC 37-3012)',
    'SPEND_FLC_z':  'SPEND_FLC  (spending / frontline supervisors, OCC 45-1011)',
    'SPEND_OP_z':   'SPEND_OP   (spending / farming operations, Census 2017)',
    'SPEND_AREA_z': 'SPEND_AREA (spending / land area sq mi)',
}

# Store interaction p-values to determine combined model structure
interaction_pvals = {}
all_models = [m0]

for vz, label in spend_labels.items():
    print(f"\n{'─'*70}")
    print(f"VARIABLE: {label}")
    print(f"{'─'*70}")

    # (a) Main effect only
    ma = fit_mixedlm(
        f'violations ~ time + time2 + time3 + {vz}',
        df_full, f'{vz} — main effect', pred_cols=[vz]
    )
    print(f"\n  (a) Main effect only:")
    print_model(ma, baseline_var_ri)
    all_models.append(ma)

    # (b) Main + ×time interaction
    mb = fit_mixedlm(
        f'violations ~ time + time2 + time3 + {vz} + {vz}:time',
        df_full, f'{vz} — main + ×time', pred_cols=[vz, f'{vz}:time']
    )
    print(f"\n  (b) Main effect + ×time interaction:")
    print_model(mb, baseline_var_ri)
    all_models.append(mb)

    # Record interaction p-value
    int_term = f'{vz}:time'
    pval_int = mb['pvalues'].get(int_term, float('nan'))
    interaction_pvals[vz] = pval_int
    flag = '<.20 → INCLUDE in combined model' if pval_int < 0.20 else '≥.20 → exclude from combined model'
    print(f"\n  ×time interaction p = {pval_int:.3f}  [{flag}]")

# ============================================================
# [8] SUMMARY VARIANCE TABLE
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY — BETWEEN-STATE VARIANCE DECOMPOSITION")
print(f"Baseline σ²_u0 (M0): {baseline_var_ri:.4f}  (ICC = {icc*100:.1f}%)")
print("=" * 70)

hdr = f"  {'Model':<50} {'N obs':>6} {'σ²_u0':>8} {'Δσ²_u0':>8} {'%Expl':>7} {'LogLik':>10}"
print(hdr)
print("  " + "-" * len(hdr))
for m in all_models:
    delta = baseline_var_ri - m['var_ri']
    pct   = delta / baseline_var_ri * 100
    lbl   = m['label'][:50]
    print(f"  {lbl:<50} {m['n_obs']:>6} {m['var_ri']:>8.4f} "
          f"{delta:>+8.4f} {pct:>6.1f}% {m['llf']:>10.2f}")

# ============================================================
# [9] COMBINED MODEL — ALL FIVE SPEND VARIABLES
# ============================================================
print("\n" + "=" * 70)
print("COMBINED MODEL — ALL FIVE SPENDING VARIABLES SIMULTANEOUSLY")
print("=" * 70)

# Determine which interactions to include (p < .20 in step 7b)
included_interactions = [vz for vz, p in interaction_pvals.items() if p < 0.20]
excluded_interactions = [vz for vz, p in interaction_pvals.items() if p >= 0.20]

print("\nInteraction term selection (p < .20 threshold):")
for vz in spend_z_vars:
    p = interaction_pvals[vz]
    decision = "INCLUDED" if p < 0.20 else "excluded"
    print(f"  {vz:<18}  ×time p = {p:.3f}  → {decision}")

# Build combined formula
base_terms = 'violations ~ time + time2 + time3'
main_terms  = ' + '.join(spend_z_vars)
int_terms   = ' + '.join([f'{vz}:time' for vz in included_interactions])

if int_terms:
    combined_formula = f'{base_terms} + {main_terms} + {int_terms}'
else:
    combined_formula = f'{base_terms} + {main_terms}'

print(f"\nCombined model formula:")
print(f"  {combined_formula}")

all_pred_cols = spend_z_vars + [f'{vz}:time' for vz in included_interactions]
m_combined = fit_mixedlm(combined_formula, df_full, 'Combined (all 5 SPEND)', pred_cols=all_pred_cols)

delta_c = baseline_var_ri - m_combined['var_ri']
pct_c   = delta_c / baseline_var_ri * 100

print(f"\nCombined model results:")
print(f"  N = {m_combined['n_obs']} obs, {m_combined['n_states']} states")
print(f"\n  {'Parameter':<35} {'β':>9} {'SE':>8} {'p':>7}")
print("  " + "-" * 62)
fe_c = m_combined['fe_params']
se_c = m_combined['bse']
pv_c = m_combined['pvalues']
for param in ['Intercept', 'time', 'time2', 'time3'] + all_pred_cols:
    if param in fe_c:
        sig = '**' if pv_c[param] < .01 else ('*' if pv_c[param] < .05 else
              ('+' if pv_c[param] < .10 else ' '))
        print(f"  {param:<35} {fe_c[param]:>9.4f} {se_c[param]:>8.4f} {pv_c[param]:>7.3f} {sig}")

print(f"\n  σ²_u0 (between-state): {m_combined['var_ri']:.4f}  "
      f"[Δ={delta_c:+.4f}, {pct_c:.1f}% of baseline explained]")
print(f"  σ²_ε  (within-state):  {m_combined['var_res']:.4f}")
print(f"  Log-Likelihood: {m_combined['llf']:.2f}")

# ============================================================
# [10] FINAL SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\nBaseline (M0) between-state variance: {baseline_var_ri:.4f}")
print()
print(f"  {'Variable':<12} {'Main β':>9} {'Main p':>7} {'×time β':>9} {'×time p':>8} "
      f"{'σ²_u0 (b)':>10} {'%Expl':>6} {'σ²_u0 (b+int)':>14} {'%Expl':>6}")
print("  " + "-" * 100)

for i, (vz, label) in enumerate(spend_labels.items()):
    ma = all_models[1 + i * 2]
    mb = all_models[2 + i * 2]
    b_main    = ma['fe_params'].get(vz, float('nan'))
    p_main    = ma['pvalues'].get(vz, float('nan'))
    b_int     = mb['fe_params'].get(f'{vz}:time', float('nan'))
    p_int     = mb['pvalues'].get(f'{vz}:time', float('nan'))
    pct_a     = (baseline_var_ri - ma['var_ri']) / baseline_var_ri * 100
    pct_b     = (baseline_var_ri - mb['var_ri']) / baseline_var_ri * 100
    short_name = vz.replace('_z', '')
    print(f"  {short_name:<12} {b_main:>9.4f} {p_main:>7.3f} {b_int:>9.4f} {p_int:>8.3f} "
          f"{ma['var_ri']:>10.4f} {pct_a:>5.1f}% {mb['var_ri']:>14.4f} {pct_b:>5.1f}%")

print(f"\n  Combined model:  σ²_u0 = {m_combined['var_ri']:.4f}  ({pct_c:.1f}% of baseline explained)")

int_in_str = ', '.join([v.replace('_z','') for v in included_interactions]) if included_interactions else 'none'
print(f"  Interactions included in combined model (p<.20): {int_in_str}")

# ============================================================
# [11] SAVE CONSTRUCTED VARIABLES TO CSV
# ============================================================
print("\n[11] Saving constructed spending variables...")

save_cols = ['state', 'mean_spending_2017', 'land_area_sqmi', 'operations',
             'emp_farmworker', 'emp_pesticide_app', 'emp_flc_supervisor',
             'SPEND_WORK', 'SPEND_APP', 'SPEND_FLC', 'SPEND_OP', 'SPEND_AREA',
             'SPEND_WORK_z', 'SPEND_APP_z', 'SPEND_FLC_z', 'SPEND_OP_z', 'SPEND_AREA_z']

level2[save_cols].to_csv('/Users/keshavgoel/Research/data/generated/spend_bls_variables.csv', index=False)
print("  Saved: spend_bls_variables.csv")
print(f"  Rows: {len(level2)}  |  Columns: {save_cols}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
