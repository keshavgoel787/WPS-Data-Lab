"""
PARAMETER WALKTHROUGH — variable-by-variable rebuild of manuscript Tables 2 & 3
with the previously-dropped TIME terms reintroduced.

Goal: make explicit *what each parameter estimate is doing* to the model — with a
particular focus on the time terms that the manuscript specification removed:
    • the cubic time polynomial (time2, time3), dropped per PI direction 2026-07
    • the spending x time interactions (SPEND_APP_z:time, SPEND_WORK_z:time),
      dropped per PI direction 2026-07
    • the surviving pct_flc_z:time interaction

Every model is fit variable-by-variable (cumulative add-one-in build-up). To make
the between-state variance story comparable across steps, every step of a given
table is fit on the SAME analytic sample — the listwise-complete cases of that
table's fullest model (fixed common sample). N therefore does NOT change as
covariates enter; only the specification changes.

Time is centered at 2017 (time = year - 2017), so time = 0 in 2017. This centering
is the key to reading the interactions: a covariate MAIN effect is its effect on
the DV *in 2017*, and the covariate:time interaction is how that effect changes per
additional year.

Estimation follows the project convention:
    MixedLM.from_formula(..., re_formula='~time').fit(method='lbfgs')   # REML
i.e. random intercept + random slope on (linear) time by state.

Outputs (exploratory):
    data/generated/parameter_walkthrough.json   — full coef/SE/p + variance comps per step
    (the companion write-up is docs/parameter_walkthrough.md)
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
# [1] BUILD LONG DV FRAMES (violations & inspections)  — identical to paper_table_models.py
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
# [2] LEVEL-2 COVARIATES  (z-scored; identical construction to paper_table_models.py)
# ============================================================
level2 = pd.DataFrame({'state': US_STATES_50})

spend = pd.read_csv(GEN + 'spend_bls_variables.csv')
level2 = level2.merge(spend[['state', 'SPEND_APP_z', 'SPEND_WORK_z']],
                      on='state', how='left')

li17 = pd.read_csv(RAW + 'labor_intensity_index_2017.csv')
li17['state'] = li17['state_name'].str.title()
level2 = level2.merge(
    li17[['state', 'Labor_Intensity_Index']].rename(
        columns={'Labor_Intensity_Index': 'lii_2017'}), on='state', how='left')

dol1 = pd.read_csv(RAW + 'dol_var1_workers_by_state_annual.csv')
dol_study = dol1[dol1['year'].between(2011, 2019) & (dol1['year'] != 2013)]
dol_means = dol_study.groupby('state').agg(
    dol_workers_cert=('workers_certified', 'mean'),
    dol_demand_met_pct=('demand_met_pct', lambda x: x[x != np.inf].mean())
).reset_index()
dol_means['state'] = dol_means['state'].map(STATE_ABBREV_TO_NAME)
level2 = level2.merge(dol_means, on='state', how='left')

bls = pd.read_csv(RAW + 'bls_oews_panel.csv')
emp = (bls[bls['occ_code'] == '45-2092']
       .rename(columns={'area_title': 'state', 'tot_emp': 'emp_farmworker'})
       [['state', 'emp_farmworker']])
level2 = level2.merge(emp, on='state', how='left')
level2['h2a_per_farmworker'] = level2['dol_workers_cert'] / level2['emp_farmworker']

dol2 = pd.read_csv(RAW + 'dol_var2_employer_type_annual.csv')
d2020 = dol2[dol2['year'] == 2020].copy()
d2020['state'] = d2020['state'].map(STATE_ABBREV_TO_NAME)
d2020 = d2020.rename(columns={'pct_Farm Labor Contractor': 'pct_flc'})
level2 = level2.merge(d2020[['state', 'pct_flc']], on='state', how='left')

for col in ['lii_2017', 'h2a_per_farmworker', 'dol_demand_met_pct', 'pct_flc']:
    level2[col + '_z'] = (level2[col] - level2[col].mean()) / level2[col].std()

Z_ALL = ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
         'h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']
df = long.merge(level2[['state'] + Z_ALL], on='state', how='left')

# ============================================================
# [3] FITTING HELPERS
# ============================================================
def fixed_sample(dv, all_terms, data):
    """Listwise-complete rows for the FULLEST spec — the common sample for every step."""
    need = {dv} | {t for term in all_terms for t in term.split(':')}
    need = {c for c in need if c in data.columns}
    d = data.dropna(subset=list(need)).copy()
    d['state'] = pd.Categorical(d['state'])
    return d

def fit(dv, rhs, data):
    formula = f"{dv} ~ " + " + ".join(rhs)
    return MixedLM.from_formula(formula, data=data, groups=data['state'],
                                re_formula='~time').fit(method='lbfgs')

def stars(p):
    return ('***' if p < .001 else '**' if p < .01 else
            '*' if p < .05 else '+' if p < .10 else '')

def variance_components(res):
    """σ²_u0 (intercept), σ²_u1 (time slope), σ_u01 (cov), σ²_e (residual), ICC@time0."""
    cov_re = res.cov_re
    s2_u0 = float(cov_re.iloc[0, 0])
    s2_u1 = float(cov_re.iloc[1, 1]) if cov_re.shape[0] > 1 else float('nan')
    s_u01 = float(cov_re.iloc[0, 1]) if cov_re.shape[0] > 1 else float('nan')
    s2_e  = float(res.scale)
    icc0  = s2_u0 / (s2_u0 + s2_e)
    return {'sigma2_u0': s2_u0, 'sigma2_u1': s2_u1, 'sigma_u01': s_u01,
            'sigma2_e': s2_e, 'icc_time0': icc0}

def extract_all(res):
    out = {}
    for t in res.fe_params.index:
        out[t] = {'b': float(res.fe_params[t]), 'se': float(res.bse[t]),
                  'p': float(res.pvalues[t]), 'stars': stars(float(res.pvalues[t]))}
    return out

def build(dv, steps, data, baseline_rhs):
    """Run a cumulative add-one-in build-up on a FIXED sample.

    steps: list of (label, [terms added at this step]).
    baseline_rhs: the spec whose σ²_u0 defines the pseudo-R² denominator
                  (fit on the same fixed sample).
    """
    all_terms = [t for _, terms in steps for t in terms]
    d = fixed_sample(dv, all_terms + baseline_rhs, data)
    base = fit(dv, baseline_rhs, d)
    base_s2 = float(base.cov_re.iloc[0, 0])

    rhs, results = [], []
    for label, terms in steps:
        rhs = rhs + terms
        res = fit(dv, rhs, d)
        vc = variance_components(res)
        results.append({
            'step': label,
            'added': terms,
            'formula': f"{dv} ~ " + " + ".join(rhs),
            'params': extract_all(res),
            'variance': vc,
            'pseudo_r2_vs_baseline': (base_s2 - vc['sigma2_u0']) / base_s2 * 100,
            'aic': float(res.aic), 'bic': float(res.bic),
            'loglike': float(res.llf),
        })
    return {'dv': dv, 'n_obs': int(len(d)), 'n_states': int(d['state'].nunique()),
            'baseline_rhs': baseline_rhs, 'baseline_sigma2_u0': base_s2,
            'steps': results}

# ============================================================
# [4] TABLE 2 — INSPECTIONS build-up (full time spec reintroduced)
# ============================================================
insp_steps = [
    ('0. linear time',            ['time']),
    ('1. + cubic time',           ['time2', 'time3']),
    ('2. + SPEND_APP_z',          ['SPEND_APP_z']),
    ('3. + SPEND_WORK_z',         ['SPEND_WORK_z']),
    ('4. + lii_2017_z',           ['lii_2017_z']),
    ('5. + h2a_per_farmworker_z', ['h2a_per_farmworker_z']),
    ('6. + dol_demand_met_pct_z', ['dol_demand_met_pct_z']),
    ('7. + pct_flc_z',            ['pct_flc_z']),
    ('8. + SPEND_APP_z:time',     ['SPEND_APP_z:time']),
    ('9. + SPEND_WORK_z:time',    ['SPEND_WORK_z:time']),
    ('10.+ pct_flc_z:time',       ['pct_flc_z:time']),
]
insp = build('inspections', insp_steps, df, baseline_rhs=['time'])

# ============================================================
# [5] TABLE 3 — VIOLATIONS build-up (full time spec reintroduced)
# ============================================================
viol_steps = [
    ('0. linear time',            ['time']),
    ('1. + cubic time',           ['time2', 'time3']),
    ('2. + inspections',          ['inspections']),
    ('3. + SPEND_APP_z',          ['SPEND_APP_z']),
    ('4. + SPEND_WORK_z',         ['SPEND_WORK_z']),
    ('5. + lii_2017_z',           ['lii_2017_z']),
    ('6. + h2a_per_farmworker_z', ['h2a_per_farmworker_z']),
    ('7. + dol_demand_met_pct_z', ['dol_demand_met_pct_z']),
    ('8. + pct_flc_z',            ['pct_flc_z']),
    ('9. + SPEND_APP_z:time',     ['SPEND_APP_z:time']),
    ('10.+ SPEND_WORK_z:time',    ['SPEND_WORK_z:time']),
    ('11.+ pct_flc_z:time',       ['pct_flc_z:time']),
]
# violations pseudo-R² baseline = inspections + linear time (matches paper convention)
viol = build('violations', viol_steps, df, baseline_rhs=['inspections', 'time'])

# ============================================================
# [6] REPORT
# ============================================================
def show(title, table):
    print("\n" + "=" * 90 + f"\n{title}  (N={table['n_obs']} obs, "
          f"{table['n_states']} states; fixed common sample)\n" + "=" * 90)
    print(f"baseline σ²_u0 ({' + '.join(table['baseline_rhs'])}) = "
          f"{table['baseline_sigma2_u0']:.2f}\n")
    for s in table['steps']:
        added = ", ".join(s['added'])
        v = s['variance']
        print(f"[{s['step']}]  add: {added}")
        # show only the newly-added terms' estimates plus intercept/time for context
        interesting = list(s['added']) + (['Intercept', 'time'] if s['step'].startswith('0') else [])
        for t in s['added']:
            if t in s['params']:
                c = s['params'][t]
                print(f"      {t:24} b={c['b']:+8.3f}  se={c['se']:6.3f}  "
                      f"p={c['p']:.3f} {c['stars']}")
        print(f"      σ²_u0={v['sigma2_u0']:7.2f}  σ²_u1={v['sigma2_u1']:6.3f}  "
              f"σ²_e={v['sigma2_e']:7.2f}  ICC0={v['icc_time0']:.3f}  "
              f"pseudoR²={s['pseudo_r2_vs_baseline']:+5.1f}%  AIC={s['aic']:.1f}")
    # full final-model coefficient block
    print("\n  --- FINAL (fullest) model coefficients ---")
    for t, c in table['steps'][-1]['params'].items():
        print(f"    {t:26} b={c['b']:+9.3f}  se={c['se']:7.3f}  p={c['p']:.4f} {c['stars']}")

show("TABLE 2 — WPS INSPECTIONS", insp)
show("TABLE 3 — WPS VIOLATIONS", viol)

with open(GEN + 'parameter_walkthrough.json', 'w') as f:
    json.dump({'inspections': insp, 'violations': viol}, f, indent=2)
print(f"\nWrote {GEN}parameter_walkthrough.json")
