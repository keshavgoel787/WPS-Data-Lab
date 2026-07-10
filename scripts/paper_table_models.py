"""
PAPER TABLES 2 & 3 — combined hierarchical models for the WPS manuscript.

Fits the exact 3-model build-up shown in "WPS Table Sheels.docx" and writes every
coefficient / SE / p-value / variance component to
    data/generated/paper_table_params.json
which fill_wps_tables_docx.py then drops into the Word template.

Table 2 — DV = inspections (EPA + state), LINEAR time (PI direction, 2026-06):
    M1  inspections ~ time + time2 + time3                       (cubic trend, descriptive)
    M2  inspections ~ time + SPEND_APP_z + SPEND_WORK_z + lii_2017_z
    M3  M2 + h2a_per_farmworker_z + dol_demand_met_pct_z + pct_flc_z

Table 3 — DV = violations, CUBIC time (2016–2017 spike is asymmetric):
    M1  violations ~ inspections + time + time2 + time3
    M2  M1 + SPEND_APP_z(+:time) + SPEND_WORK_z(+:time) + lii_2017_z
    M3  M2 + h2a_per_farmworker_z + dol_demand_met_pct_z + pct_flc_z + pct_flc_z:time

"Inspections" enters Table 3 as a raw contemporaneous Level-1 count (PI direction).
The ×time interactions shown are pre-specified by the table, not screened.

Variance components (State-to-State σ² and Δ σ²) are reported for M2 and M3 only.
Δ σ² = % reduction in between-state intercept variance relative to a baseline
re-estimated on that column's own listwise-complete analytic sample:
    • inspections baseline = linear-time-only  (matches the linear final model)
    • violations  baseline = M1 spec (inspections + cubic time)
so every comparison is against a matched-N baseline (CLAUDE.md convention).

All Level-2 covariates are z-scored (project convention). LII uses the 2017 wave
only — lii_2012/lii_2017/lii_2022 are near-collinear (r≈0.977); see
docs/collinearity_notes.md.
"""

import json
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS (kept in sync with the other project scripts)
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

# ============================================================
# [1] BUILD LONG DV FRAMES (violations & inspections)
# ============================================================
echo = pd.read_csv(RAW + 'establishments_data.csv', index_col=0)
echo.index = echo.index.str.strip().map(lambda x: STATE_NAME_MAPPING.get(x, x))

rows = []
for year in range(2011, 2020):
    vcol = f'violations-{year}'
    iepa = f'inspections-epa-{year}'
    ist  = f'inspections-state-{year}'
    rows.append(pd.DataFrame({
        'state': echo.index,
        'year': year,
        'violations': pd.to_numeric(echo[vcol], errors='coerce').values,
        'inspections': (pd.to_numeric(echo[iepa], errors='coerce').fillna(0).values
                        + pd.to_numeric(echo[ist], errors='coerce').fillna(0).values),
    }))
long = pd.concat(rows, ignore_index=True)
long = long[long['state'].isin(US_STATES_50)].copy()
long['time']  = long['year'] - 2017
long['time2'] = long['time'] ** 2
long['time3'] = long['time'] ** 3

# ============================================================
# [2] LEVEL-2 COVARIATES  (z-scored; identical construction to the final models)
# ============================================================
level2 = pd.DataFrame({'state': US_STATES_50})

# Spending per applicator / per farmworker — pre-built, already z-scored.
spend = pd.read_csv(GEN + 'spend_bls_variables.csv')
level2 = level2.merge(spend[['state', 'SPEND_APP_z', 'SPEND_WORK_z']],
                      on='state', how='left')

# Labor Intensity Index — 2017 wave only (2012/2022 collinear, r≈0.977).
li17 = pd.read_csv(RAW + 'labor_intensity_index_2017.csv')
li17['state'] = li17['state_name'].str.title()
level2 = level2.merge(
    li17[['state', 'Labor_Intensity_Index']].rename(
        columns={'Labor_Intensity_Index': 'lii_2017'}), on='state', how='left')

# DOL H-2A workers / demand-met (2011–2019, excl. 2013 missing).
dol1 = pd.read_csv(RAW + 'dol_var1_workers_by_state_annual.csv')
dol_study = dol1[dol1['year'].between(2011, 2019) & (dol1['year'] != 2013)]
dol_means = dol_study.groupby('state').agg(
    dol_workers_cert=('workers_certified', 'mean'),
    dol_demand_met_pct=('demand_met_pct', lambda x: x[x != np.inf].mean())
).reset_index()
dol_means['state'] = dol_means['state'].map(STATE_ABBREV_TO_NAME)
level2 = level2.merge(dol_means, on='state', how='left')

# Farmworker employment (BLS OEWS OCC 45-2092, 2011) → H-2A per farmworker ratio.
bls = pd.read_csv(RAW + 'bls_oews_panel.csv')
emp = (bls[bls['occ_code'] == '45-2092']
       .rename(columns={'area_title': 'state', 'tot_emp': 'emp_farmworker'})
       [['state', 'emp_farmworker']])
level2 = level2.merge(emp, on='state', how='left')
level2['h2a_per_farmworker'] = level2['dol_workers_cert'] / level2['emp_farmworker']

# DOL employer type — % Farm Labor Contractor (2020 proxy).
dol2 = pd.read_csv(RAW + 'dol_var2_employer_type_annual.csv')
d2020 = dol2[dol2['year'] == 2020].copy()
d2020['state'] = d2020['state'].map(STATE_ABBREV_TO_NAME)
d2020 = d2020.rename(columns={'pct_Farm Labor Contractor': 'pct_flc'})
level2 = level2.merge(d2020[['state', 'pct_flc']], on='state', how='left')

# z-score the covariates constructed here (spending vars arrive pre-standardized).
for col in ['lii_2017', 'h2a_per_farmworker', 'dol_demand_met_pct', 'pct_flc']:
    level2[col + '_z'] = (level2[col] - level2[col].mean()) / level2[col].std()

Z_SPEND = ['SPEND_APP_z', 'SPEND_WORK_z']
Z_LABOR = ['lii_2017_z']
Z_H2A   = ['h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']
keep = Z_SPEND + Z_LABOR + Z_H2A
df = long.merge(level2[['state'] + keep], on='state', how='left')

# ============================================================
# [3] FITTING HELPERS
# ============================================================
def fit(dv, rhs, data):
    """Fit MixedLM (random intercept + random slope for time) on listwise-complete rows."""
    terms = [t.split(':')[0] for chunk in rhs for t in [chunk]] + [dv]
    need = {dv} | {t for term in rhs for t in term.split(':')}
    d = data.dropna(subset=[c for c in need if c in data.columns]).copy()
    d['state'] = pd.Categorical(d['state'])
    formula = f"{dv} ~ " + " + ".join(rhs)
    res = MixedLM.from_formula(formula, data=d, groups=d['state'],
                               re_formula='~time').fit(method='lbfgs')
    return res, d

def stars(p):
    return ('***' if p < .001 else '**' if p < .01 else
            '*' if p < .05 else '+' if p < .10 else '')

def extract(res, terms):
    out = {}
    for t in terms:
        if t in res.fe_params.index:
            out[t] = {'b': float(res.fe_params[t]),
                      'se': float(res.bse[t]),
                      'p': float(res.pvalues[t]),
                      'stars': stars(float(res.pvalues[t]))}
    return out

def sigma2(res):
    return float(res.cov_re.iloc[0, 0])

# ============================================================
# [4] TABLE 2 — INSPECTIONS  (linear time in M2/M3)
# ============================================================
insp = {}
# M1: cubic descriptive trend (full available sample)
r1, _ = fit('inspections', ['time', 'time2', 'time3'], df)
insp['M1'] = extract(r1, ['time', 'time2', 'time3'])

# M2: linear time + spending + labor
rhs2 = ['time'] + Z_SPEND + Z_LABOR
r2, d2 = fit('inspections', rhs2, df)
base2, _ = fit('inspections', ['time'], d2)              # matched linear baseline
insp['M2'] = extract(r2, rhs2)
insp['M2']['sigma2'] = sigma2(r2)
insp['M2']['delta_pct'] = (sigma2(base2) - sigma2(r2)) / sigma2(base2) * 100
insp['M2']['n_obs'], insp['M2']['n_states'] = len(d2), d2['state'].nunique()

# M3: + H-2A block
rhs3 = ['time'] + Z_SPEND + Z_LABOR + Z_H2A
r3, d3 = fit('inspections', rhs3, df)
base3, _ = fit('inspections', ['time'], d3)
insp['M3'] = extract(r3, rhs3)
insp['M3']['sigma2'] = sigma2(r3)
insp['M3']['delta_pct'] = (sigma2(base3) - sigma2(r3)) / sigma2(base3) * 100
insp['M3']['n_obs'], insp['M3']['n_states'] = len(d3), d3['state'].nunique()

# ============================================================
# [5] TABLE 3 — VIOLATIONS  (cubic time; inspections raw count predictor)
# ============================================================
viol = {}
base_terms = ['inspections', 'time', 'time2', 'time3']
# M1: inspections + cubic time
v1, _ = fit('violations', base_terms, df)
viol['M1'] = extract(v1, base_terms)

# M2: + spending(+:time) + labor
rhs2v = base_terms + ['SPEND_APP_z', 'SPEND_APP_z:time',
                      'SPEND_WORK_z', 'SPEND_WORK_z:time'] + Z_LABOR
v2, dv2 = fit('violations', rhs2v, df)
vbase2, _ = fit('violations', base_terms, dv2)           # matched M1-spec baseline
viol['M2'] = extract(v2, rhs2v)
viol['M2']['sigma2'] = sigma2(v2)
viol['M2']['delta_pct'] = (sigma2(vbase2) - sigma2(v2)) / sigma2(vbase2) * 100
viol['M2']['n_obs'], viol['M2']['n_states'] = len(dv2), dv2['state'].nunique()

# M3: + H-2A block + %FLC:time
rhs3v = rhs2v + Z_H2A + ['pct_flc_z:time']
v3, dv3 = fit('violations', rhs3v, df)
vbase3, _ = fit('violations', base_terms, dv3)
viol['M3'] = extract(v3, rhs3v)
viol['M3']['sigma2'] = sigma2(v3)
viol['M3']['delta_pct'] = (sigma2(vbase3) - sigma2(v3)) / sigma2(vbase3) * 100
viol['M3']['n_obs'], viol['M3']['n_states'] = len(dv3), dv3['state'].nunique()

# ============================================================
# [6] REPORT + DUMP
# ============================================================
def show(title, tab, order):
    models = ['M1', 'M2', 'M3']
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
    print(f"{'':32}" + "".join(f"{m:>13}" for m in models))
    for label, term in order:
        cells = []
        for m in models:
            cell = tab[m].get(term)
            cells.append(f"{cell['b']:.2f}{cell['stars']}" if cell else '')
        print(f"{label:32}" + "".join(f"{c:>13}" for c in cells))
    s2 = [f"{tab[m]['sigma2']:.2f}" if 'sigma2' in tab[m] else '' for m in models]
    dp = [f"{tab[m]['delta_pct']:.1f}%" if 'delta_pct' in tab[m] else '' for m in models]
    print(f"{'State-to-State sigma^2':32}" + "".join(f"{c:>13}" for c in s2))
    print(f"{'Delta State-to-State sigma^2':32}" + "".join(f"{c:>13}" for c in dp))
    for m in ['M2', 'M3']:
        print(f"  {m}: N={tab[m]['n_obs']} obs, {tab[m]['n_states']} states")

show("TABLE 2 — WPS INSPECTIONS", insp, [
    ('Time', 'time'), ('Time2', 'time2'), ('Time3', 'time3'),
    ('Spending/applicator', 'SPEND_APP_z'),
    ('Spending/farmworker', 'SPEND_WORK_z'),
    ('Labor Intensity', 'lii_2017_z'),
    ('H-2A farmworkers:BLS farmworkers', 'h2a_per_farmworker_z'),
    ('H-2A Authorized/H-2A Requested', 'dol_demand_met_pct_z'),
    ('%H-2A Authorized to FLC', 'pct_flc_z'),
])
show("TABLE 3 — WPS VIOLATIONS", viol, [
    ('Inspections', 'inspections'),
    ('Time', 'time'), ('Time2', 'time2'), ('Time3', 'time3'),
    ('Spending/applicator', 'SPEND_APP_z'),
    ('Spending/applicator*time', 'SPEND_APP_z:time'),
    ('Spending/farmworker', 'SPEND_WORK_z'),
    ('Spending/farmworker*time', 'SPEND_WORK_z:time'),
    ('Labor Intensity', 'lii_2017_z'),
    ('H-2A farmworkers:BLS farmworkers', 'h2a_per_farmworker_z'),
    ('H-2A Authorized/H-2A Requested', 'dol_demand_met_pct_z'),
    ('%H-2A Authorized to FLC', 'pct_flc_z'),
    ('%H-2A Authorized to FLC*time', 'pct_flc_z:time'),
])

with open(GEN + 'paper_table_params.json', 'w') as f:
    json.dump({'inspections': insp, 'violations': viol}, f, indent=2)
print(f"\nWrote {GEN}paper_table_params.json")
