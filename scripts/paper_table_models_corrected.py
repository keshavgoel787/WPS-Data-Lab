"""
PAPER TABLES 2 & 3 -- CORRECTED build (2026-07, post-meeting).

Supersedes paper_table_models.py with the four fixes from Joe's meeting:
  1. LOG DVs. Both outcomes modelled as log(count + 1) -- the transform-exploration
     recommendation Joe confirmed. `+1` because violations contain zeros.
     Inspections also enters Table 3 as a predictor on the SAME log(x+1) scale.
  2. MULTI-YEAR BLS denominators. SPEND_APP / SPEND_WORK now come from
     spend_bls_variables_multiyear.csv (mean 2011-2019 employment, not the 2011
     snapshot) -> recovers CO/NH/NM/UT/MN/NV. SPEND_WORK 50/50, SPEND_APP 47/50.
  3. $0-coded spending for MS/RI/WV (confirmed no pesticide-STAG award, like
     HI/TN/UT) -> no longer dropped for missing spending.
  4. YEAR-MATCHED H-2A ratio. h2a_per_farmworker is now a time-varying Level-1
     predictor (h2a_ratio_panel.csv): annual certified H-2A workers / annual BLS
     farmworkers, matched by year -- not a frozen 2011-denominator state mean.

Net effect on N: the 39-state substantive models become 47 (only AK/RI/VT drop,
and only in columns that use SPEND_APP -- BLS never publishes applicator
employment for those three). Δσ² baselines are re-estimated on each column's own
analytic sample (CLAUDE.md convention), now on the log scale.

Requires: rebuild_spend_bls_multiyear.py and build_h2a_ratio_panel.py to have run.
Output: data/generated/paper_table_params_corrected.json
"""

import json
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

US_STATES_50 = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
    'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
    'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
    'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
    'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
    'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
    'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
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
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'}
STATE_NAME_MAPPING = {'Massachusetts ': 'Massachusetts', 'Oregon ': 'Oregon'}
RAW = '/Users/keshavgoel/Research/data/raw/'
GEN = '/Users/keshavgoel/Research/data/generated/'

# ============================================================
# [1] LONG DV FRAME (+ log transforms)
# ============================================================
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
long['log_violations'] = np.log1p(long['violations'])
long['log_inspections'] = np.log1p(long['inspections'])

# ============================================================
# [2] COVARIATES
# ============================================================
level2 = pd.DataFrame({'state': US_STATES_50})

# spending (multi-year denominators, $0-coded MS/RI/WV) -- pre z-scored
spend = pd.read_csv(GEN + 'spend_bls_variables_multiyear.csv')
level2 = level2.merge(spend[['state', 'SPEND_APP_z', 'SPEND_WORK_z']], on='state', how='left')

# LII 2017 (2012/2022 collinear)
li17 = pd.read_csv(RAW + 'labor_intensity_index_2017.csv')
li17['state'] = li17['state_name'].str.title()
level2 = level2.merge(li17[['state', 'Labor_Intensity_Index']].rename(
    columns={'Labor_Intensity_Index': 'lii_2017'}), on='state', how='left')

# DOL demand-met % (Level-2 mean 2011-2019 excl 2013)
dol1 = pd.read_csv(RAW + 'dol_var1_workers_by_state_annual.csv')
dol_study = dol1[dol1['year'].between(2011, 2019) & (dol1['year'] != 2013)]
dm = dol_study.groupby('state')['demand_met_pct'].apply(lambda x: x[x != np.inf].mean()).reset_index()
dm['state'] = dm['state'].map(STATE_ABBREV_TO_NAME)
level2 = level2.merge(dm.rename(columns={'demand_met_pct': 'dol_demand_met_pct'}), on='state', how='left')

# DOL % Farm Labor Contractor (2020 proxy)
dol2 = pd.read_csv(RAW + 'dol_var2_employer_type_annual.csv')
d2020 = dol2[dol2['year'] == 2020].copy()
d2020['state'] = d2020['state'].map(STATE_ABBREV_TO_NAME)
d2020 = d2020.rename(columns={'pct_Farm Labor Contractor': 'pct_flc'})
level2 = level2.merge(d2020[['state', 'pct_flc']], on='state', how='left')

for col in ['lii_2017', 'dol_demand_met_pct', 'pct_flc']:
    level2[col + '_z'] = (level2[col] - level2[col].mean()) / level2[col].std()

Z_SPEND = ['SPEND_APP_z', 'SPEND_WORK_z']
Z_LABOR = ['lii_2017_z']
Z_H2A_L2 = ['dol_demand_met_pct_z', 'pct_flc_z']       # Level-2 members of H-2A block
keep = Z_SPEND + Z_LABOR + Z_H2A_L2
df = long.merge(level2[['state'] + keep], on='state', how='left')

# time-varying H-2A ratio (Level-1, merged on state+year)
h2a = pd.read_csv(GEN + 'h2a_ratio_panel.csv')
df = df.merge(h2a[['state', 'year', 'h2a_per_farmworker_z']], on=['state', 'year'], how='left')
Z_H2A = ['h2a_per_farmworker_z'] + Z_H2A_L2

# ============================================================
# [3] FIT HELPERS
# ============================================================
def fit(dv, rhs, data):
    need = {dv} | {t for term in rhs for t in term.split(':')}
    d = data.dropna(subset=[c for c in need if c in data.columns]).copy()
    d['state'] = pd.Categorical(d['state'])
    res = MixedLM.from_formula(f"{dv} ~ " + " + ".join(rhs), data=d,
                               groups=d['state'], re_formula='~time').fit(method='lbfgs')
    return res, d

def stars(p):
    return ('***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '+' if p < .10 else '')

def extract(res, terms):
    out = {}
    for t in terms:
        if t in res.fe_params.index:
            out[t] = {'b': float(res.fe_params[t]), 'se': float(res.bse[t]),
                      'p': float(res.pvalues[t]), 'stars': stars(float(res.pvalues[t]))}
    return out

def sig2(res):
    return float(res.cov_re.iloc[0, 0])

CUBIC = ['time', 'time2', 'time3']

def build(dv, base_terms, tag):
    tab = {}
    r1, _ = fit(dv, base_terms, df)
    tab['M1'] = extract(r1, base_terms)
    for m, extra in [('M2', Z_SPEND + Z_LABOR), ('M3', Z_SPEND + Z_LABOR + Z_H2A)]:
        rhs = base_terms + extra
        r, d = fit(dv, rhs, df)
        b, _ = fit(dv, base_terms, d)            # matched baseline on this sample
        tab[m] = extract(r, rhs)
        tab[m]['sigma2'] = sig2(r)
        tab[m]['delta_pct'] = (sig2(b) - sig2(r)) / sig2(b) * 100
        tab[m]['n_obs'], tab[m]['n_states'] = len(d), d['state'].nunique()
    return tab

insp = build('log_inspections', CUBIC, 'inspections')
viol = build('log_violations', ['log_inspections'] + CUBIC, 'violations')

# ============================================================
# [4] REPORT + DUMP
# ============================================================
def show(title, tab, order):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
    print(f"{'':32}" + "".join(f"{m:>13}" for m in ['M1', 'M2', 'M3']))
    for label, term in order:
        cells = [f"{tab[m][term]['b']:.3f}{tab[m][term]['stars']}" if tab[m].get(term) else ''
                 for m in ['M1', 'M2', 'M3']]
        print(f"{label:32}" + "".join(f"{c:>13}" for c in cells))
    print(f"{'State sigma^2':32}" + "".join(
        f"{tab[m].get('sigma2', ''):>13.3f}" if 'sigma2' in tab[m] else f"{'':>13}" for m in ['M1', 'M2', 'M3']))
    print(f"{'Delta sigma^2':32}" + "".join(
        f"{tab[m]['delta_pct']:>12.1f}%" if 'delta_pct' in tab[m] else f"{'':>13}" for m in ['M1', 'M2', 'M3']))
    for m in ['M2', 'M3']:
        print(f"  {m}: N={tab[m]['n_obs']} obs, {tab[m]['n_states']} states")

show("TABLE 2 -- WPS INSPECTIONS  [DV = log(inspections+1)]", insp, [
    ('Time', 'time'), ('Time^2', 'time2'), ('Time^3', 'time3'),
    ('Spending/applicator', 'SPEND_APP_z'), ('Spending/farmworker', 'SPEND_WORK_z'),
    ('Labor Intensity', 'lii_2017_z'),
    ('H-2A:farmworker (yr-matched)', 'h2a_per_farmworker_z'),
    ('H-2A demand-met %', 'dol_demand_met_pct_z'), ('% H-2A to FLC', 'pct_flc_z')])
show("TABLE 3 -- WPS VIOLATIONS  [DV = log(violations+1)]", viol, [
    ('log Inspections', 'log_inspections'),
    ('Time', 'time'), ('Time^2', 'time2'), ('Time^3', 'time3'),
    ('Spending/applicator', 'SPEND_APP_z'), ('Spending/farmworker', 'SPEND_WORK_z'),
    ('Labor Intensity', 'lii_2017_z'),
    ('H-2A:farmworker (yr-matched)', 'h2a_per_farmworker_z'),
    ('H-2A demand-met %', 'dol_demand_met_pct_z'), ('% H-2A to FLC', 'pct_flc_z')])

with open(GEN + 'paper_table_params_corrected.json', 'w') as f:
    json.dump({'inspections': insp, 'violations': viol}, f, indent=2)
print(f"\nWrote {GEN}paper_table_params_corrected.json")
