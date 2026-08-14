"""
PAPER TABLES 2 & 3 -- EXTENDED to 2011-2021 (2026-08, Joe's request).

Joe asked whether the figures could run through 2021, or whether the models were
themselves constrained to 2011-2019. They were. The constraint was the DV source:
data/raw/establishments_data.csv is the ECHO State Pesticide Dashboard's
"establishments" view, which EPA publishes only through 2019 (re-verified against
the live dashboard 2026-08-13). Every covariate already ran past 2019.

This build swaps in the dashboard's WPS view, which does cover 2011-2021, and
widens every covariate window to match:
    DV        data/generated/wps_dv_panel_2011_2021.csv     (build_wps_dv_panel.py)
    SPEND_*   spend_bls_variables_multiyear_2021.csv        (rebuild_spend_bls_multiyear_2021.py)
    H-2A      h2a_ratio_panel_2011_2021.csv                 (build_h2a_ratio_panel.py 2021)
    BLS       bls_oews_panel_2011_2021.csv                  (harvest_bls_oews_panel.py 2021)

MODEL SPEC IS UNCHANGED from paper_table_models_corrected.py -- same M1/M2/M3
build-up, cubic time centred at 2017, no x time interactions, log(count+1) DVs,
random intercept + random slope on linear time, REML. Only the data window and the
DV source differ, so the tables stay structurally comparable to the corrected ones.

READ THIS BEFORE COMPARING TO THE 2011-2019 TABLES. The WPS series is a different
measure, not a longer one. It counts violations of the ten WPS provisions
(2,269 across the 50 states in 2017) where the ECHO establishments series counted
FIFRA-wide violations (967). State-level correlation between them is ~0.4-0.6 in
2011-2016 and ~0 from 2017 on. Coefficients here are NOT line-by-line comparable
with paper_table_params_corrected.json; this is a re-analysis on a WPS-specific
outcome, which is arguably the better-matched outcome for a WPS paper.

N: 49 states (Wyoming has no row in the WPS view at all), vs 47 in the corrected
tables. Columns using SPEND_APP drop AK/RI/VT, for which BLS never publishes
applicator employment.

COVID. 2020-2021 are pandemic years and WPS inspections fall ~15% in 2020 before
partly recovering. The tables are fit on the unchanged cubic spec as directed; a
COVID-indicator robustness variant is fit and reported at the end so the effect of
that choice is visible rather than buried.

Output: data/generated/paper_table_params_2021.json
"""

import json
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')

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
US_STATES_50 = sorted(STATE_ABBREV_TO_NAME.values())

RAW = '/Users/keshavgoel/Research/data/raw/'
GEN = '/Users/keshavgoel/Research/data/generated/'
END_YEAR = 2021

# ============================================================
# [1] LONG DV FRAME (WPS view, already log-transformed upstream)
# ============================================================
long = pd.read_csv(GEN + 'wps_dv_panel_2011_2021.csv')
long = long[long['state'].isin(US_STATES_50)].copy()
long['covid'] = (long['year'] >= 2020).astype(float)

# ============================================================
# [2] COVARIATES (identical construction, wider windows)
# ============================================================
level2 = pd.DataFrame({'state': US_STATES_50})

spend = pd.read_csv(GEN + 'spend_bls_variables_multiyear_2021.csv')
level2 = level2.merge(spend[['state', 'SPEND_APP_z', 'SPEND_WORK_z']], on='state', how='left')

li17 = pd.read_csv(RAW + 'labor_intensity_index_2017.csv')
li17['state'] = li17['state_name'].str.title()
level2 = level2.merge(li17[['state', 'Labor_Intensity_Index']].rename(
    columns={'Labor_Intensity_Index': 'lii_2017'}), on='state', how='left')

# DOL demand-met % (Level-2 mean 2011-2021 excl 2013, which DOL never published)
dol1 = pd.read_csv(RAW + 'dol_var1_workers_by_state_annual.csv')
dol_study = dol1[dol1['year'].between(2011, END_YEAR) & (dol1['year'] != 2013)]
dm = dol_study.groupby('state')['demand_met_pct'].apply(
    lambda x: x[x != np.inf].mean()).reset_index()
dm['state'] = dm['state'].map(STATE_ABBREV_TO_NAME)
level2 = level2.merge(dm.rename(columns={'demand_met_pct': 'dol_demand_met_pct'}),
                      on='state', how='left')

# DOL % Farm Labor Contractor -- dol_var2 starts in 2020, so the 2020 value is no
# longer a backward proxy for the study window; it now sits inside it.
dol2 = pd.read_csv(RAW + 'dol_var2_employer_type_annual.csv')
d2020 = dol2[dol2['year'] == 2020].copy()
d2020['state'] = d2020['state'].map(STATE_ABBREV_TO_NAME)
d2020 = d2020.rename(columns={'pct_Farm Labor Contractor': 'pct_flc'})
level2 = level2.merge(d2020[['state', 'pct_flc']], on='state', how='left')

for col in ['lii_2017', 'dol_demand_met_pct', 'pct_flc']:
    level2[col + '_z'] = (level2[col] - level2[col].mean()) / level2[col].std()

Z_SPEND = ['SPEND_APP_z', 'SPEND_WORK_z']
Z_LABOR = ['lii_2017_z']
Z_H2A_L2 = ['dol_demand_met_pct_z', 'pct_flc_z']
keep = Z_SPEND + Z_LABOR + Z_H2A_L2
df = long.merge(level2[['state'] + keep], on='state', how='left')

h2a = pd.read_csv(GEN + 'h2a_ratio_panel_2011_2021.csv')
df = df.merge(h2a[['state', 'year', 'h2a_per_farmworker_z']], on=['state', 'year'], how='left')
Z_H2A = ['h2a_per_farmworker_z'] + Z_H2A_L2

# ============================================================
# [3] FIT HELPERS (verbatim from paper_table_models_corrected.py)
# ============================================================
def fit(dv, rhs, data, re_formula='~time'):
    need = {dv} | {t for term in rhs for t in term.split(':')}
    d = data.dropna(subset=[c for c in need if c in data.columns]).copy()
    d['state'] = pd.Categorical(d['state'])
    kw = {} if re_formula is None else {'re_formula': re_formula}
    res = MixedLM.from_formula(f"{dv} ~ " + " + ".join(rhs), data=d,
                               groups=d['state'], **kw).fit(method='lbfgs')
    return res, d

def stars(p):
    return ('***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '+' if p < .10 else '')

def extract(res, terms, d, dv):
    sd_y = float(d[dv].std())
    out = {}
    for t in terms:
        if t in res.fe_params.index and t in d.columns:
            b = float(res.fe_params[t])
            sd_x = float(d[t].std())
            out[t] = {'b': b, 'se': float(res.bse[t]),
                      'p': float(res.pvalues[t]), 'stars': stars(float(res.pvalues[t])),
                      'beta': b * sd_x / sd_y if sd_y else float('nan')}
    return out

def re_components(res):
    return {'sigma2_u0': float(res.cov_re.iloc[0, 0]),
            'sigma2_u1': float(res.cov_re.iloc[1, 1]),
            'sigma_u01': float(res.cov_re.iloc[0, 1]),
            'sigma2_e':  float(res.scale)}

CUBIC = ['time', 'time2', 'time3']

def ri_decomp(dv, base_terms, rhs, d):
    r_ri, _ = fit(dv, rhs, d, re_formula=None)
    b_ri, _ = fit(dv, base_terms, d, re_formula=None)
    u0_f, u0_b = float(r_ri.cov_re.iloc[0, 0]), float(b_ri.cov_re.iloc[0, 0])
    return {'sigma2_u0_ri': u0_f, 'sigma2_u0_ri_baseline': u0_b,
            'sigma2_e_ri': float(r_ri.scale), 'sigma2_e_ri_baseline': float(b_ri.scale),
            'delta_pct_ri': (u0_b - u0_f) / u0_b * 100}


def build(dv, base_terms):
    tab = {}
    r1, d1 = fit(dv, base_terms, df)
    tab['M1'] = extract(r1, base_terms, d1, dv)
    comp1 = re_components(r1)
    tab['M1'].update(comp1)
    tab['M1'].update({k + '_baseline': v for k, v in comp1.items()})
    tab['M1']['sigma2'] = comp1['sigma2_u0']
    tab['M1']['delta_pct'] = 0.0
    ri1 = ri_decomp(dv, base_terms, base_terms, d1)
    ri1['delta_pct_ri'] = 0.0
    tab['M1'].update(ri1)
    tab['M1']['n_obs'], tab['M1']['n_states'] = len(d1), d1['state'].nunique()
    for m, extra in [('M2', Z_SPEND + Z_LABOR), ('M3', Z_SPEND + Z_LABOR + Z_H2A)]:
        rhs = base_terms + extra
        r, d = fit(dv, rhs, df)
        b, _ = fit(dv, base_terms, d)
        tab[m] = extract(r, rhs, d, dv)
        comp, comp_base = re_components(r), re_components(b)
        tab[m].update(comp)
        tab[m].update({k + '_baseline': v for k, v in comp_base.items()})
        tab[m]['sigma2'] = comp['sigma2_u0']
        tab[m]['delta_pct'] = (comp_base['sigma2_u0'] - comp['sigma2_u0']) / comp_base['sigma2_u0'] * 100
        tab[m].update(ri_decomp(dv, base_terms, rhs, d))
        tab[m]['n_obs'], tab[m]['n_states'] = len(d), d['state'].nunique()

    # Single Model-1 baseline (2026-08 team direction) -- every column starts from M1.
    ref_u0, ref_e = tab['M1']['sigma2_u0_ri'], tab['M1']['sigma2_e_ri']
    for m in ('M1', 'M2', 'M3'):
        tab[m]['sigma2_u0_ri_baseline_matched'] = tab[m]['sigma2_u0_ri_baseline']
        tab[m]['sigma2_e_ri_baseline_matched'] = tab[m]['sigma2_e_ri_baseline']
        tab[m]['delta_pct_ri_matched'] = tab[m]['delta_pct_ri']
        tab[m]['sigma2_u0_ri_baseline'] = ref_u0
        tab[m]['sigma2_e_ri_baseline'] = ref_e
        tab[m]['delta_pct_ri'] = (ref_u0 - tab[m]['sigma2_u0_ri']) / ref_u0 * 100
    return tab

insp = build('log_inspections', CUBIC)
viol = build('log_violations', ['log_inspections'] + CUBIC)

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
    print(f"{'State sigma^2_u0 (rand-int)':32}" + "".join(
        f"{tab[m].get('sigma2_u0_ri', ''):>13.3f}" if 'sigma2_u0_ri' in tab[m] else f"{'':>13}" for m in ['M1', 'M2', 'M3']))
    print(f"{'Delta sigma^2_u0 (vs Model 1)':32}" + "".join(
        f"{tab[m]['delta_pct_ri']:>12.1f}%" if 'delta_pct_ri' in tab[m] else f"{'':>13}" for m in ['M1', 'M2', 'M3']))
    for m in ['M1', 'M2', 'M3']:
        print(f"  {m}: N={tab[m]['n_obs']} obs, {tab[m]['n_states']} states")

ORDER_INSP = [('Time', 'time'), ('Time^2', 'time2'), ('Time^3', 'time3'),
              ('Spending/applicator', 'SPEND_APP_z'), ('Spending/farmworker', 'SPEND_WORK_z'),
              ('Labor Intensity', 'lii_2017_z'),
              ('H-2A:farmworker (yr-matched)', 'h2a_per_farmworker_z'),
              ('H-2A demand-met %', 'dol_demand_met_pct_z'), ('% H-2A to FLC', 'pct_flc_z')]
ORDER_VIOL = [('log Inspections', 'log_inspections')] + ORDER_INSP

show(f"TABLE 2 -- WPS INSPECTIONS 2011-{END_YEAR}  [DV = log(inspections+1)]", insp, ORDER_INSP)
show(f"TABLE 3 -- WPS VIOLATIONS 2011-{END_YEAR}  [DV = log(violations+1)]", viol, ORDER_VIOL)

with open(GEN + 'paper_table_params_2021.json', 'w') as f:
    json.dump({'inspections': insp, 'violations': viol}, f, indent=2)
print(f"\nWrote {GEN}paper_table_params_2021.json")

# ============================================================
# [5] COVID ROBUSTNESS (reported, not tabled)
# ============================================================
print("\n" + "=" * 72)
print("COVID ROBUSTNESS -- M3 with a 2020-21 indicator added")
print("=" * 72)
print("2020-21 are pandemic years; WPS inspections fall ~15% in 2020. If the cubic")
print("time trend is absorbing that shock, the time terms will move when a COVID")
print("dummy is added. Small movement => the 2011-2021 tables are safe as fit.\n")
for dv, base in [('log_inspections', CUBIC), ('log_violations', ['log_inspections'] + CUBIC)]:
    rhs = base + Z_SPEND + Z_LABOR + Z_H2A
    r_no, d_no = fit(dv, rhs, df)
    r_cv, _ = fit(dv, rhs + ['covid'], d_no)
    print(f"  {dv}:")
    for t in ['time', 'time2', 'time3'] + (['covid'] if True else []):
        if t == 'covid':
            b, p = float(r_cv.fe_params[t]), float(r_cv.pvalues[t])
            print(f"    {'covid (2020-21)':22} {'':>10}  ->{b:>9.4f}{stars(p):<4} (added term)")
        else:
            b0, b1 = float(r_no.fe_params[t]), float(r_cv.fe_params[t])
            print(f"    {t:22} {b0:>10.4f}  ->{b1:>9.4f}{stars(float(r_cv.pvalues[t])):<4} "
                  f"(shift {b1 - b0:+.4f})")
