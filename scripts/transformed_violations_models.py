"""
EXPLORATORY — violations model suite re-fit under count transformations.

Fits the full violations model suite (time-only baseline, Zimmerman stepwise, BLS
spending, targeted FIFRA, labor/DOL stepwise, curated final, paper Table 3 build-up)
under raw / log / sqrt / Anscombe / Box-Cox transforms of the violation count.

AIC/loglik are NOT comparable across DV transforms — transforms are judged on
residual diagnostics (Shapiro-Wilk W, skew, kurtosis, heteroscedasticity) and on
the stability of substantive coefficients. Pseudo-R² (a variance ratio) is reported.

Outputs: data/generated/transform_comparison.csv, docs/transform_exploration.md,
figures/transform_qq_*.png. Manuscript artifacts are untouched.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

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
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska',
    'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina',
    'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
    'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}
STATE_NAME_MAPPING = {'Massachusetts ': 'Massachusetts', 'Oregon ': 'Oregon'}
RAW = '/Users/keshavgoel/Research/data/raw/'
GEN = '/Users/keshavgoel/Research/data/generated/'

echo = pd.read_csv(RAW + 'establishments_data.csv', index_col=0)
echo.index = echo.index.str.strip().map(lambda x: STATE_NAME_MAPPING.get(x, x))
rows = []
for year in range(2011, 2020):
    rows.append(pd.DataFrame({
        'state': echo.index, 'year': year,
        'violations': pd.to_numeric(echo[f'violations-{year}'], errors='coerce').values,
        'inspections': (pd.to_numeric(echo[f'inspections-epa-{year}'], errors='coerce').fillna(0).values
                        + pd.to_numeric(echo[f'inspections-state-{year}'], errors='coerce').fillna(0).values),
    }))
long = pd.concat(rows, ignore_index=True)
long = long[long['state'].isin(US_STATES_50)].copy()
long['time'] = long['year'] - 2017
long['time2'] = long['time'] ** 2
long['time3'] = long['time'] ** 3

# Level-2 covariates (z-scored; identical construction to paper_table_models.py)
level2 = pd.DataFrame({'state': US_STATES_50})
spend = pd.read_csv(GEN + 'spend_bls_variables.csv')
level2 = level2.merge(spend[['state', 'SPEND_APP_z', 'SPEND_WORK_z']], on='state', how='left')
# other SPEND_* z variants for the stepwise/targeted families, if present:
for c in ['SPEND_FLC_z', 'SPEND_OP_z', 'SPEND_AREA_z']:
    if c in spend.columns:
        level2 = level2.merge(spend[['state', c]], on='state', how='left')
li17 = pd.read_csv(RAW + 'labor_intensity_index_2017.csv')
li17['state'] = li17['state_name'].str.title()
level2 = level2.merge(li17[['state', 'Labor_Intensity_Index']].rename(
    columns={'Labor_Intensity_Index': 'lii_2017'}), on='state', how='left')
dol1 = pd.read_csv(RAW + 'dol_var1_workers_by_state_annual.csv')
dol_study = dol1[dol1['year'].between(2011, 2019) & (dol1['year'] != 2013)]
dol_means = dol_study.groupby('state').agg(
    dol_workers_cert=('workers_certified', 'mean'),
    dol_demand_met_pct=('demand_met_pct', lambda x: x[x != np.inf].mean())).reset_index()
dol_means['state'] = dol_means['state'].map(STATE_ABBREV_TO_NAME)
level2 = level2.merge(dol_means, on='state', how='left')
bls = pd.read_csv(RAW + 'bls_oews_panel.csv')
emp = (bls[bls['occ_code'] == '45-2092'].rename(
    columns={'area_title': 'state', 'tot_emp': 'emp_farmworker'})[['state', 'emp_farmworker']])
level2 = level2.merge(emp, on='state', how='left')
level2['h2a_per_farmworker'] = level2['dol_workers_cert'] / level2['emp_farmworker']
dol2 = pd.read_csv(RAW + 'dol_var2_employer_type_annual.csv')
d2020 = dol2[dol2['year'] == 2020].copy()
d2020['state'] = d2020['state'].map(STATE_ABBREV_TO_NAME)
d2020 = d2020.rename(columns={'pct_Farm Labor Contractor': 'pct_flc'})
level2 = level2.merge(d2020[['state', 'pct_flc']], on='state', how='left')
for col in ['lii_2017', 'h2a_per_farmworker', 'dol_demand_met_pct', 'pct_flc']:
    level2[col + '_z'] = (level2[col] - level2[col].mean()) / level2[col].std()

Z_SPEND = ['SPEND_APP_z', 'SPEND_WORK_z']
Z_LABOR = ['lii_2017_z']
Z_H2A = ['h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']
L2_KEEP = [c for c in level2.columns if c.endswith('_z') or c in
           ('lii_2017', 'h2a_per_farmworker', 'dol_demand_met_pct', 'pct_flc')]
df = long.merge(level2[['state'] + L2_KEEP], on='state', how='left')


def add_transform_columns(df):
    y = df['violations'].astype(float)
    df = df.copy()
    df['dv_raw'] = y
    df['dv_log'] = np.log(y + 1.0)
    df['dv_sqrt'] = np.sqrt(y)
    df['dv_anscombe'] = 2.0 * np.sqrt(y + 3.0 / 8.0)
    df['dv_ft'] = np.sqrt(y) + np.sqrt(y + 1.0)          # Freeman–Tukey companion
    # Box-Cox needs strictly positive input; estimate lambda once on (y+1).
    shifted = (y + 1.0).dropna()
    bc_vals, lam = stats.boxcox(shifted.values)
    df.loc[shifted.index, 'dv_boxcox'] = bc_vals
    return df, float(lam)

TRANSFORMS = {'raw': 'dv_raw', 'log': 'dv_log', 'sqrt': 'dv_sqrt',
              'anscombe': 'dv_anscombe', 'boxcox': 'dv_boxcox'}

# --- sanity-check guardrails ---
df, BOXCOX_LAMBDA = add_transform_columns(df)
assert df['state'].nunique() == 50, df['state'].nunique()
assert len(df) == 450, len(df)                              # 50 states x 9 years

# Baseline (cubic time) used for pseudo-R2 of the cubic-time families.
CUBIC_BASE = ['time', 'time2', 'time3']
# Paper Table 3 uses LINEAR time + inspections (matches current manuscript spec).
PAPER_BASE = ['inspections', 'time']

MODEL_SPECS = [
    # name, rhs, baseline_rhs
    ('baseline_cubic', CUBIC_BASE, CUBIC_BASE),
    # Zimmerman one-at-a-time (cubic time + one L2 covariate)
    ('zim_spend_app', CUBIC_BASE + ['SPEND_APP_z'], CUBIC_BASE),
    ('zim_spend_work', CUBIC_BASE + ['SPEND_WORK_z'], CUBIC_BASE),
    ('zim_lii', CUBIC_BASE + ['lii_2017_z'], CUBIC_BASE),
    ('zim_h2a', CUBIC_BASE + ['h2a_per_farmworker_z'], CUBIC_BASE),
    ('zim_demandmet', CUBIC_BASE + ['dol_demand_met_pct_z'], CUBIC_BASE),
    ('zim_flc', CUBIC_BASE + ['pct_flc_z'], CUBIC_BASE),
    # BLS spending combined (the two FIFRA populations)
    ('spend_combined', CUBIC_BASE + ['SPEND_APP_z', 'SPEND_WORK_z'], CUBIC_BASE),
    # Targeted FIFRA (SPEND_AREA_z only if present)
    ('targeted_fifra', CUBIC_BASE + ['SPEND_WORK_z', 'SPEND_APP_z'], CUBIC_BASE),
    # Curated final (cubic time + labor block)
    ('final_labor', CUBIC_BASE + ['lii_2017_z', 'h2a_per_farmworker_z',
                                   'dol_demand_met_pct_z', 'pct_flc_z'], CUBIC_BASE),
    # Paper Table 3 build-up (LINEAR time + inspections)
    ('paper_M1', PAPER_BASE, PAPER_BASE),
    ('paper_M2', PAPER_BASE + ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z'], PAPER_BASE),
    ('paper_M3', PAPER_BASE + ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
                               'h2a_per_farmworker_z', 'dol_demand_met_pct_z',
                               'pct_flc_z', 'pct_flc_z:time'], PAPER_BASE),
]
# Add SPEND_AREA_z to the targeted model only if it exists in the frame.
if 'SPEND_AREA_z' in df.columns:
    MODEL_SPECS = [(('targeted_fifra', CUBIC_BASE + ['SPEND_WORK_z', 'SPEND_APP_z',
                    'SPEND_AREA_z'], CUBIC_BASE) if n == 'targeted_fifra' else (n, r, b))
                   for (n, r, b) in MODEL_SPECS]


def _analytic(dv_col, rhs, data):
    need = {dv_col} | {t for term in rhs for t in term.split(':')}
    need = {c for c in need if c in data.columns}
    d = data.dropna(subset=list(need)).copy()
    d['state'] = pd.Categorical(d['state'])
    return d

def fit_spec(dv_col, rhs, data):
    d = _analytic(dv_col, rhs, data)
    formula = f"{dv_col} ~ " + " + ".join(rhs)
    res = MixedLM.from_formula(formula, data=d, groups=d['state'],
                               re_formula='~time').fit(method='lbfgs')
    return res, d

def pseudo_r2(res, d, dv_col, baseline_rhs):
    base, _ = fit_spec(dv_col, baseline_rhs, d)   # matched-N baseline on this sample
    s_model = float(res.cov_re.iloc[0, 0])
    s_base = float(base.cov_re.iloc[0, 0])
    return (s_base - s_model) / s_base * 100.0

def diagnostics(res, d):
    resid = np.asarray(res.resid)
    fitted = np.asarray(res.fittedvalues)
    # Shapiro on residuals (cap n at 5000; here always small)
    w, p = stats.shapiro(resid)
    skew = float(stats.skew(resid))
    kurt = float(stats.kurtosis(resid))          # excess kurtosis
    # Heteroscedasticity: OLS of resid^2 on fitted -> R^2 * n ~ chi2 (Breusch-Pagan-ish)
    X = np.column_stack([np.ones_like(fitted), fitted])
    beta, *_ = np.linalg.lstsq(X, resid ** 2, rcond=None)
    pred = X @ beta
    ss_tot = np.sum((resid ** 2 - (resid ** 2).mean()) ** 2)
    ss_res = np.sum((resid ** 2 - pred) ** 2)
    bp_r2 = 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return {'shapiro_w': float(w), 'shapiro_p': float(p),
            'resid_skew': skew, 'resid_kurtosis': kurt, 'bp_r2': float(bp_r2)}


def key_terms(rhs):
    """Non-time fixed effects worth reporting for stability."""
    drop = {'time', 'time2', 'time3', 'inspections'}
    return [t for t in rhs if t not in drop]

records = []
for name, rhs, base_rhs in MODEL_SPECS:
    for tkey, dv_col in TRANSFORMS.items():
        try:
            res, d = fit_spec(dv_col, rhs, df)
        except Exception as e:
            records.append({'model': name, 'transform': tkey, 'error': str(e)})
            continue
        diag = diagnostics(res, d)
        row = {'model': name, 'transform': tkey,
               'n_obs': len(d), 'n_states': d['state'].nunique(),
               'pseudo_r2_pct': pseudo_r2(res, d, dv_col, base_rhs), **diag}
        for t in key_terms(rhs):
            if t in res.fe_params.index:
                row[f'b__{t}'] = float(res.fe_params[t])
                row[f'p__{t}'] = float(res.pvalues[t])
        records.append(row)

out = pd.DataFrame(records)
out.to_csv(GEN + 'transform_comparison.csv', index=False)
print(f"Wrote {GEN}transform_comparison.csv  ({len(out)} rows)")
print(f"Box-Cox lambda = {BOXCOX_LAMBDA:.4f}")

# Headline diagnostics table to terminal
head = out[out['model'].isin(['baseline_cubic', 'final_labor', 'paper_M3'])]
cols = ['model', 'transform', 'shapiro_w', 'shapiro_p', 'resid_skew',
        'resid_kurtosis', 'bp_r2', 'pseudo_r2_pct']
print(head[cols].to_string(index=False))
