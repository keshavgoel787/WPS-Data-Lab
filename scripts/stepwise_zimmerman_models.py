"""
Stepwise Variable-by-Variable Models — Zimmerman/HLM Approach
WPS Enforcement Analysis

Approach (Zimmerman 2000 / Raudenbush & Bryk 2002):
  - Start with unconditional model (time polynomial only)
  - Add one predictor at a time as a state-level (Level-2) covariate
  - For each predictor: fit (a) main effect only, then (b) main + linear time interaction
  - Track between-state variance (random intercept) at each step
  - Report pseudo-R² = reduction in between-state variance from baseline

Covariates (all standardized to mean=0, SD=1 before entry):
  1. spending_per_estab       — inflation-adj. grant $ / mean ECHO WPS establishment count
                                (2014–2019 mean; CO and CT 2014 imputed from 2015)
                                [team-agreed operationalization: Stephane proposed, Joe and
                                 Kaitlyn agreed; matches enforcement dollars to regulated units]
  2. land_area_sqmi           — total state land area in sq miles (US Census Bureau)
                                [Joe's suggestion: larger states face greater travel burden
                                 for inspectors visiting the same number of establishments]
  3. farming_operations       — total farming operations per state (2017 Census of Agriculture,
                                from Yuri's h2a_state_summary data)
  4. h2a_workers              — total H-2A workers per state (2017 USCIS / DOL data,
                                from Yuri's h2a_state_summary)
  5. workers_per_operation    — H-2A workers / total operations (2017)

NOTE on dollars-per-acre operationalization:
  The originally requested starting variable (spending / harvested cropland acres from USDA
  NASS Quick Stats "Area Harvested") cannot be computed — that data is not currently in the
  project files. The team's agreed operationalization (spending_per_estab) is used as the
  primary spending variable instead. NASS data should be obtained to run the dollars-per-acre
  model as a robustness check.

NOTE on ECHO establishment coverage:
  ECHO WPS establishment counts available: 2014–2019 for 48 states; 2015–2019 for CO & CT.
  Years 2011–2013 are not available from ECHO; this script uses the 2015–2019 per-state
  mean as a time-invariant denominator. The USDA Census of Agriculture provides farm counts
  for 2012 and 2017 that could be used to fill 2011–2013; that substitution should be
  documented in the methods section if per-establishment is the final operationalization.
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

# Total state land area in square miles (US Census Bureau, 2020 decennial)
# Source: https://www.census.gov/geographies/reference-files/2010/geo/state-area.html
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
print("STEPWISE ZIMMERMAN MODELS — WPS VIOLATIONS ANALYSIS")
print("=" * 70)

# ============================================================
# [1] LOAD AND PREPARE BASE VIOLATIONS + SPENDING DATA
#     (same pipeline as hierarchical_violations_model.py)
# ============================================================
print("\n[1] Loading base violations and spending data...")

echo_df = pd.read_csv('/Users/keshavgoel/Research/data/raw/establishments_data.csv', index_col=0)
echo_df.index = echo_df.index.str.strip()
echo_df.index = echo_df.index.map(lambda x: STATE_NAME_MAPPING.get(x, x))

# Reshape violations to long
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

# Time variables centered on 2017
echo_long_filtered['time']  = echo_long_filtered['year'] - 2017
echo_long_filtered['time2'] = echo_long_filtered['time'] ** 2
echo_long_filtered['time3'] = echo_long_filtered['time'] ** 3

# Spending
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
)
df_model = df_model.dropna(subset=['spending_2017m']).copy()
df_model['state'] = pd.Categorical(df_model['state'])

print(f"  Base dataset: {len(df_model)} obs, {df_model['state'].nunique()} states, years {df_model['year'].min()}–{df_model['year'].max()}")

# ============================================================
# [2] ECHO ESTABLISHMENT COVERAGE AUDIT
# ============================================================
print("\n" + "=" * 70)
print("ECHO ESTABLISHMENT COVERAGE AUDIT")
print("=" * 70)

est_cols = [f'establishments-{y}' for y in range(2014, 2020)]
est_df = echo_df[echo_df.index.isin(US_STATES_50)][est_cols].copy()
for c in est_cols:
    est_df[c] = pd.to_numeric(est_df[c], errors='coerce')

print("\nAvailable ECHO establishment years: 2014–2019")
print("Study period years: 2011–2019")
print("Coverage gap: 2011, 2012, 2013 — no ECHO WPS establishment counts available")
print("\nMissing establishment values within 2014–2019:")
for c in est_cols:
    n_missing = est_df[c].isna().sum()
    missing_states = est_df[est_df[c].isna()].index.tolist()
    if n_missing > 0:
        print(f"  {c}: {n_missing} missing ({', '.join(missing_states)})")
    else:
        print(f"  {c}: complete (0 missing)")

print("\nRECOMMENDATION: For 2011–2013, substitute USDA Census of Agriculture farm counts")
print("  (2012 and 2017 Census values; interpolate for non-census years).")
print("  For CO and CT in 2014, impute from 2015 value.")
print("  This substitution must be documented in the methods section.")

# Impute CO and CT 2014 from 2015
for state in ['Colorado', 'Connecticut']:
    if pd.isna(est_df.loc[state, 'establishments-2014']):
        est_df.loc[state, 'establishments-2014'] = est_df.loc[state, 'establishments-2015']
        print(f"\n  Imputed {state} 2014 establishments from 2015 value: {est_df.loc[state,'establishments-2014']:.0f}")

# Compute per-state mean establishments (2014–2019, imputed)
est_df['establishments_mean'] = est_df[est_cols].mean(axis=1)
print("\nECHO establishments (2014–2019 mean) — top and bottom 5 states:")
print(est_df['establishments_mean'].sort_values(ascending=False).head(5).round(1))
print("...")
print(est_df['establishments_mean'].sort_values(ascending=False).tail(5).round(1))

# ============================================================
# [3] PREPARE LEVEL-2 STATE COVARIATES
# ============================================================
print("\n" + "=" * 70)
print("PREPARING LEVEL-2 STATE COVARIATES")
print("=" * 70)

# 3a. H2A / farming operations from Yuri's data (2017 Census of Agriculture)
print("\n[3a] Loading Yuri's H2A state summary (2017 Census)...")
h2a_2017 = pd.read_csv('/Users/keshavgoel/Research/data/raw/h2a_state_summary_2017.csv')
h2a_2017['state'] = h2a_2017['state_code'].map(STATE_ABBREV_TO_NAME)
h2a_2017 = h2a_2017[h2a_2017['state'].isin(US_STATES_50)].copy()
print(f"  States in H2A 2017 data: {h2a_2017['state'].nunique()}")
print(f"  Columns: {h2a_2017.columns.tolist()}")

# 3b. State-level mean spending (2011-2019)
mean_spending = spending_agg.groupby('state')['spending_2017'].mean().reset_index()
mean_spending.columns = ['state', 'mean_spending_2017']

# 3c. Build Level-2 covariate dataframe
level2 = pd.DataFrame({'state': US_STATES_50})

# Land area
level2['land_area_sqmi'] = level2['state'].map(STATE_LAND_AREA_SQMI)

# ECHO establishments (mean 2014-2019)
level2['establishments_mean'] = level2['state'].map(est_df['establishments_mean'])

# Mean spending
level2 = level2.merge(mean_spending, on='state', how='left')

# Spending per ECHO establishment
level2['spending_per_estab'] = level2['mean_spending_2017'] / level2['establishments_mean']

# H2A data
level2 = level2.merge(
    h2a_2017[['state', 'h2a_workers', 'operations', 'workers_per_operation']],
    on='state', how='left'
)

# Spending per farming operation (Census of Agriculture)
level2['spending_per_operation'] = level2['mean_spending_2017'] / level2['operations']

print("\nLevel-2 covariate summary:")
covariate_cols = ['land_area_sqmi', 'establishments_mean', 'spending_per_estab',
                  'operations', 'h2a_workers', 'workers_per_operation', 'spending_per_operation']
print(level2[covariate_cols].describe().round(3).to_string())

# ============================================================
# [4] STANDARDIZE LEVEL-2 COVARIATES
# ============================================================
print("\n[4] Standardizing Level-2 covariates (z-score)...")

std_covariates = {}
for col in covariate_cols:
    mean_val = level2[col].mean()
    std_val  = level2[col].std()
    std_col  = col + '_z'
    level2[std_col] = (level2[col] - mean_val) / std_val
    std_covariates[col] = std_col
    print(f"  {col}: mean={mean_val:.2f}, SD={std_val:.2f}  → {std_col}")

# ============================================================
# [5] MERGE LEVEL-2 INTO MODEL DATASET
# ============================================================
print("\n[5] Merging Level-2 covariates into model dataset...")

df_full = df_model.merge(level2, on='state', how='left')
df_full['state'] = pd.Categorical(df_full['state'])
print(f"  Final dataset: {len(df_full)} obs, {df_full['state'].nunique()} states")
for col in covariate_cols:
    z_col = col + '_z'
    n_missing = df_full[z_col].isna().sum()
    print(f"  {z_col}: {n_missing} missing values")

# ============================================================
# [6] MODEL FITTING HELPER
# ============================================================

def fit_mixedlm(formula, data, label, pred_cols=None):
    """Fit a MixedLM with random intercept + random slope for time."""
    model = MixedLM.from_formula(
        formula,
        data=data,
        groups=data['state'],
        re_formula='~time'
    )
    result = model.fit(method='lbfgs')
    var_ri  = result.cov_re.iloc[0, 0]
    var_rs  = result.cov_re.iloc[1, 1]
    var_res = result.scale
    return {
        'label':     label,
        'formula':   formula,
        'llf':       result.llf,
        'var_ri':    var_ri,
        'var_rs':    var_rs,
        'var_res':   var_res,
        'result':    result,
        'fe_params': result.fe_params,
        'pvalues':   result.pvalues,
        'bse':       result.bse,
        'pred_cols': pred_cols or [],
    }


def variance_table(models, baseline_var_ri):
    """Print variance decomposition table with fixed effect parameter estimates."""
    # -- Variance component block --
    hdr = f"{'Model':<45} {'σ²_u0':>8} {'σ²_ε':>8} {'Δσ²_u0':>8} {'%Expl':>7} {'LogLik':>10}"
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        delta    = baseline_var_ri - m['var_ri']
        pct_expl = (delta / baseline_var_ri * 100) if baseline_var_ri > 0 else 0
        print(f"  {m['label']:<43} {m['var_ri']:>8.4f} {m['var_res']:>8.4f} "
              f"{delta:>+8.4f} {pct_expl:>6.1f}% {m['llf']:>10.2f}")

    # -- Fixed effects block --
    print()
    print("Fixed Effect Parameter Estimates")
    print("-" * 90)
    fe_hdr = f"  {'Model':<38} {'Parameter':<35} {'β':>9} {'SE':>8} {'p':>7}"
    print(fe_hdr)
    print("  " + "-" * 88)
    for m in models:
        fe   = m['fe_params']
        se   = m['bse']
        pv   = m['pvalues']
        params_to_show = ['Intercept', 'time', 'time2', 'time3'] + m['pred_cols']
        first = True
        for param in params_to_show:
            if param in fe:
                label_col = m['label'] if first else ''
                print(f"  {label_col:<38} {param:<35} {fe[param]:>9.4f} {se[param]:>8.4f} {pv[param]:>7.3f}")
                first = False
        print()


# ============================================================
# [7] MODEL 0 — UNCONDITIONAL (TIME POLYNOMIAL ONLY)
# ============================================================
print("\n" + "=" * 70)
print("MODEL 0 — UNCONDITIONAL (TIME ONLY, BASELINE)")
print("=" * 70)

m0 = fit_mixedlm(
    'violations ~ time + time2 + time3',
    df_full, 'M0: Unconditional (time only)',
    pred_cols=[]
)
print(m0['result'].summary())

baseline_var_ri = m0['var_ri']
baseline_var_rs = m0['var_rs']
icc = baseline_var_ri / (baseline_var_ri + m0['var_res'])
print(f"\nBaseline between-state variance (σ²_u0): {baseline_var_ri:.4f}")
print(f"Baseline within-state variance  (σ²_ε):  {m0['var_res']:.4f}")
print(f"Baseline ICC: {icc:.4f}  ({icc*100:.1f}% of variance is between states)")

# ============================================================
# [8] STEPWISE MODELS — ONE PREDICTOR AT A TIME
# ============================================================
print("\n" + "=" * 70)
print("STEPWISE MODELS — ZIMMERMAN APPROACH")
print("Variable sequence per research team decision:")
print("  1. spending_per_estab      (spending / ECHO establishments mean 2014-2019)")
print("  2. land_area_sqmi          (state total area)")
print("  3. farming_operations      (total farming operations, Census 2017)")
print("  4. h2a_workers             (H-2A workers per state, 2017)")
print("  5. workers_per_operation   (H-2A workers per farming operation, 2017)")
print("NOTE: dollars-per-acre (NASS harvested acres) not yet computable — data not in files")
print("=" * 70)

all_models = [m0]

predictor_sequence = [
    ('spending_per_estab_z',      'spending per ECHO establishment (mean 2014–2019)'),
    ('land_area_sqmi_z',          'state land area (sq miles)'),
    ('operations_z',              'total farming operations (Census 2017)'),
    ('h2a_workers_z',             'H-2A workers per state (2017)'),
    ('workers_per_operation_z',   'H-2A workers per operation (2017)'),
]

for pred_col, pred_label in predictor_sequence:
    print(f"\n{'─'*70}")
    print(f"PREDICTOR: {pred_label}")
    print(f"{'─'*70}")

    # (a) Main effect only
    formula_a = f'violations ~ time + time2 + time3 + {pred_col}'
    label_a   = f'  + {pred_col[:30]} (main)'
    m_a = fit_mixedlm(formula_a, df_full, label_a, pred_cols=[pred_col])

    # (b) Main effect + linear time interaction
    formula_b = f'violations ~ time + time2 + time3 + {pred_col} + {pred_col}:time'
    label_b   = f'  + {pred_col[:30]} (main + ×time)'
    m_b = fit_mixedlm(formula_b, df_full, label_b,
                      pred_cols=[pred_col, f'{pred_col}:time'])

    all_models.extend([m_a, m_b])

    # Print fixed effects for these models
    for m, tag in [(m_a, '(a) main effect'), (m_b, '(b) main + ×time interaction')]:
        fe = m['result'].fe_params
        pvals = m['result'].pvalues
        delta = baseline_var_ri - m['var_ri']
        pct   = delta / baseline_var_ri * 100

        print(f"\n  Model {tag}:")
        print(f"    Formula:  {m['formula']}")
        print(f"    β_{pred_col}: {fe.get(pred_col, float('nan')):.4f}  "
              f"(p={pvals.get(pred_col, float('nan')):.3f})")
        if pred_col + ':time' in fe:
            print(f"    β_{pred_col}×time: {fe[pred_col+':time']:.4f}  "
                  f"(p={pvals.get(pred_col+':time', float('nan')):.3f})")
        print(f"    σ²_u0 (between-state): {m['var_ri']:.4f}  "
              f"[Δ={delta:+.4f}, {pct:.1f}% explained vs. M0]")
        print(f"    σ²_ε  (within-state):  {m['var_res']:.4f}")
        print(f"    Log-Likelihood: {m['llf']:.2f}")


# ============================================================
# [9] SUMMARY VARIANCE DECOMPOSITION TABLE
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY — BETWEEN-STATE VARIANCE DECOMPOSITION")
print(f"Baseline σ²_u0 (M0): {baseline_var_ri:.4f}")
print("=" * 70)
variance_table(all_models, baseline_var_ri)

# ============================================================
# [10] COMBINED MODEL — OPERATIONS + H-2A WORKERS (COLLINEARITY TEST)
# ============================================================
print("\n" + "=" * 70)
print("COMBINED MODEL — operations_z + h2a_workers_z (both with ×time)")
print("Tests whether the two strongest predictors explain independent")
print("variance or are effectively collinear.")
print("=" * 70)

# Pearson correlation between the two predictors
corr_ops_h2a = df_full[['operations_z', 'h2a_workers_z']].dropna().corr().iloc[0, 1]
print(f"\nPearson r (operations_z, h2a_workers_z): {corr_ops_h2a:.4f}")

combined_formula = (
    'violations ~ time + time2 + time3 '
    '+ operations_z + operations_z:time '
    '+ h2a_workers_z + h2a_workers_z:time'
)
m_combined = fit_mixedlm(
    combined_formula, df_full,
    'M6: operations + h2a_workers (both ×time)',
    pred_cols=['operations_z', 'operations_z:time', 'h2a_workers_z', 'h2a_workers_z:time']
)

fe_c  = m_combined['fe_params']
se_c  = m_combined['bse']
pv_c  = m_combined['pvalues']
delta_c = baseline_var_ri - m_combined['var_ri']
pct_c   = delta_c / baseline_var_ri * 100

print(f"\nCombined model fixed effects:")
print(f"  {'Parameter':<35} {'β':>9} {'SE':>8} {'p':>7}")
print("  " + "-" * 62)
for param in ['Intercept', 'time', 'time2', 'time3',
              'operations_z', 'operations_z:time',
              'h2a_workers_z', 'h2a_workers_z:time']:
    if param in fe_c:
        print(f"  {param:<35} {fe_c[param]:>9.4f} {se_c[param]:>8.4f} {pv_c[param]:>7.3f}")

print(f"\nVariance decomposition:")
print(f"  σ²_u0 (between-state): {m_combined['var_ri']:.4f}  "
      f"[Δ={delta_c:+.4f} vs. M0, {pct_c:.1f}% explained]")
print(f"  σ²_ε  (within-state):  {m_combined['var_res']:.4f}")
print(f"  Log-Likelihood: {m_combined['llf']:.2f}")

# Compare combined vs. individual best models
best_ops  = [m for m in all_models if 'operations_z' in m['formula'] and 'h2a' not in m['formula'] and ':time' in m['formula']]
best_h2a  = [m for m in all_models if 'h2a_workers_z' in m['formula'] and 'operations' not in m['formula'] and ':time' in m['formula']]
if best_ops:
    pct_ops = (baseline_var_ri - best_ops[0]['var_ri']) / baseline_var_ri * 100
    print(f"\n  M3b (operations only ×time):  {pct_ops:.1f}% between-state variance explained")
if best_h2a:
    pct_h2a = (baseline_var_ri - best_h2a[0]['var_ri']) / baseline_var_ri * 100
    print(f"  M4b (h2a_workers only ×time): {pct_h2a:.1f}% between-state variance explained")
print(f"  M6  (both ×time combined):    {pct_c:.1f}% between-state variance explained")
print()
pct_ops_val = (baseline_var_ri - best_ops[0]['var_ri']) / baseline_var_ri * 100 if best_ops else 0
pct_h2a_val = (baseline_var_ri - best_h2a[0]['var_ri']) / baseline_var_ri * 100 if best_h2a else 0
best_single  = max(pct_ops_val, pct_h2a_val)
increment    = pct_c - best_single
print(f"  Combined explains {pct_c:.1f}% vs. best single predictor ({best_single:.1f}%) — "
      f"increment = {increment:+.1f} pp")

all_models.append(m_combined)

# ============================================================
# [11] SAVE LEVEL-2 COVARIATE TABLE
# ============================================================
print("\n[11] Saving Level-2 covariate table...")
output_cols = ['state', 'land_area_sqmi', 'establishments_mean',
               'mean_spending_2017', 'spending_per_estab',
               'operations', 'h2a_workers', 'workers_per_operation',
               'spending_per_operation'] + [v for v in std_covariates.values()]
level2[output_cols].to_csv('/Users/keshavgoel/Research/data/generated/level2_covariates.csv', index=False)
print("  Saved: level2_covariates.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
