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

# NOTE (2026-08-31): `warnings.filterwarnings('ignore')` used to sit here, as it
# did in paper_table_models_2021.py, where it hid a ConvergenceWarning whose
# non-converged estimates were then published. Warnings are now captured per fit
# in fit() and reported by fit_report(). Do not reinstate a blanket filter.

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
# Optimizers tried, in order. `lbfgs` stays FIRST and is kept whenever it
# converges cleanly, so every fit that was already sound reproduces bit for bit;
# only a fit lbfgs actually fails on falls through to the rest. Mirrors
# paper_table_models_2021.py -- keep the two in sync.
OPTIMIZERS = ['lbfgs', 'cg', 'powell', 'bfgs']

# Largest |gradient| still treated as converged. statsmodels can return a result
# whose `converged` flag is True while the gradient is far from zero, so the flag
# alone is not sufficient.
GRAD_TOL = 1e-2

# Not every warning means the fit is bad. "The MLE may be on the boundary of the
# parameter space" is a legitimate result (a variance component estimated at
# zero) and arrives with a tiny gradient; disqualifying it would churn fits that
# are already correct. Only warnings reporting an actual optimizer FAILURE
# disqualify a fit.
FAILURE_WARNING_MARKERS = ('failed', 'not converge', 'did not converge')

FIT_LOG = []  # one record per fit that needed a fallback; printed by fit_report()


def _is_failure(msgs):
    return any(any(m in str(w).lower() for m in FAILURE_WARNING_MARKERS)
               for w in msgs)


def _grad_norm(res):
    """Max |gradient| at the reported optimum, or nan if unavailable."""
    try:
        g = res.model.score(res.params_object, profile_fe=False)
        return float(np.max(np.abs(g)))
    except Exception:
        return float('nan')


def fit(dv, rhs, data, re_formula='~time'):
    """Fit the mixed model. re_formula='~time' = random intercept + random slope
    (paper spec, used for coefficients); re_formula=None = random intercept only
    (used for the between-state variance-explained statistic).

    Tries OPTIMIZERS in order and keeps the first that converges cleanly (no
    failure warning, converged flag set, small gradient). If none is clean, keeps
    the highest-log-likelihood candidate and records it, rather than silently
    returning whatever the first optimizer produced."""
    need = {dv} | {t for term in rhs for t in term.split(':')}
    d = data.dropna(subset=[c for c in need if c in data.columns]).copy()
    d['state'] = pd.Categorical(d['state'])
    kw = {} if re_formula is None else {'re_formula': re_formula}
    model = MixedLM.from_formula(f"{dv} ~ " + " + ".join(rhs), data=d,
                                 groups=d['state'], **kw)
    label = f"{dv} ~ {' + '.join(rhs)} | re={re_formula}"

    candidates = []
    for method in OPTIMIZERS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                res = model.fit(method=method)
            except Exception as exc:  # optimizer blew up; try the next one
                candidates.append((method, None, float('-inf'), float('nan'),
                                   [f'{type(exc).__name__}: {exc}']))
                continue
        msgs = sorted({str(w.message) for w in caught})
        gnorm = _grad_norm(res)
        clean = (not _is_failure(msgs)) and bool(getattr(res, 'converged', True)) \
            and np.isfinite(gnorm) and gnorm <= GRAD_TOL
        candidates.append((method, res, float(res.llf), gnorm, msgs))
        if clean:
            if method != OPTIMIZERS[0]:
                FIT_LOG.append({'fit': label, 'used': method,
                                'reason': f"{OPTIMIZERS[0]} not clean",
                                'llf': float(res.llf), 'grad': gnorm,
                                'first_llf': float(candidates[0][2]),
                                'first_grad': candidates[0][3],
                                'first_msgs': candidates[0][4]})
            return res, d

    usable = [c for c in candidates if c[1] is not None]
    if not usable:
        raise RuntimeError(f"every optimizer failed for {label}")
    method, res, llf, gnorm, msgs = max(usable, key=lambda c: c[2])
    FIT_LOG.append({'fit': label, 'used': method, 'reason': 'NO CLEAN FIT',
                    'llf': llf, 'grad': gnorm, 'first_llf': float(candidates[0][2]),
                    'first_grad': candidates[0][3], 'first_msgs': msgs})
    return res, d


def fit_report():
    """Print every fit that did not converge cleanly under the first optimizer."""
    print("\n" + "=" * 72)
    print("OPTIMIZER FALLBACKS")
    print("=" * 72)
    if not FIT_LOG:
        print(f"None -- every fit converged cleanly under '{OPTIMIZERS[0]}'.")
        return
    for r in FIT_LOG:
        print(f"\n  {r['fit']}")
        print(f"    {OPTIMIZERS[0]}: loglik {r['first_llf']:.3f}, "
              f"|grad| {r['first_grad']:.4g}")
        if r['first_msgs']:
            for m in r['first_msgs']:
                print(f"      warning: {m}")
        print(f"    used '{r['used']}': loglik {r['llf']:.3f}, "
              f"|grad| {r['grad']:.4g}  ({r['reason']})")

def stars(p):
    return ('***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '+' if p < .10 else '')

def extract(res, terms, d, dv):
    """Fixed-effect estimates + fully standardized beta on this analytic sample.

    beta = b * SD_sample(x) / SD_sample(y). Covariates are z-scored, but listwise
    deletion drifts their in-sample SD off 1, so recompute rather than assume it.
    """
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

def sig2(res):
    return float(res.cov_re.iloc[0, 0])

def re_components(res):
    """Full random-effects variance components (intercept + linear-time slope)."""
    return {'sigma2_u0': float(res.cov_re.iloc[0, 0]),   # between-state intercept var
            'sigma2_u1': float(res.cov_re.iloc[1, 1]),   # between-state time-slope var
            'sigma_u01': float(res.cov_re.iloc[0, 1]),   # intercept-slope covariance
            'sigma2_e':  float(res.scale)}               # within-state residual var

CUBIC = ['time', 'time2', 'time3']

def ri_decomp(dv, base_terms, rhs, d):
    """Random-INTERCEPT-only variance-explained on sample d. In a random-slope
    model sigma^2_u0 is the between-state variance AT time=0 (2017) and trades off
    with the slope variance/covariance, so its reduction is not bounded to [0,1]
    and can go negative; the conventional between-state variance-explained is read
    from random-intercept-only models. Coefficients still come from the random-slope
    fit -- this basis is used only for the variation block."""
    r_ri, _ = fit(dv, rhs, d, re_formula=None)
    b_ri, _ = fit(dv, base_terms, d, re_formula=None)
    u0_f, u0_b = float(r_ri.cov_re.iloc[0, 0]), float(b_ri.cov_re.iloc[0, 0])
    return {'sigma2_u0_ri': u0_f, 'sigma2_u0_ri_baseline': u0_b,
            'sigma2_e_ri': float(r_ri.scale), 'sigma2_e_ri_baseline': float(b_ri.scale),
            'delta_pct_ri': (u0_b - u0_f) / u0_b * 100}


def build(dv, base_terms, tag):
    tab = {}
    # M1: the base (time-only, + log_inspections for violations) model. It is its
    # own baseline, so original == became (delta 0).
    r1, d1 = fit(dv, base_terms, df)
    tab['M1'] = extract(r1, base_terms, d1, dv)
    comp1 = re_components(r1)
    tab['M1'].update(comp1)
    tab['M1'].update({k + '_baseline': v for k, v in comp1.items()})
    tab['M1']['sigma2'] = comp1['sigma2_u0']
    tab['M1']['delta_pct'] = 0.0
    ri1 = ri_decomp(dv, base_terms, base_terms, d1)          # baseline == model for M1
    ri1['delta_pct_ri'] = 0.0
    tab['M1'].update(ri1)
    tab['M1']['n_obs'], tab['M1']['n_states'] = len(d1), d1['state'].nunique()
    for m, extra in [('M2', Z_SPEND + Z_LABOR), ('M3', Z_SPEND + Z_LABOR + Z_H2A)]:
        rhs = base_terms + extra
        r, d = fit(dv, rhs, df)
        b, _ = fit(dv, base_terms, d)            # matched baseline on this sample
        tab[m] = extract(r, rhs, d, dv)
        comp, comp_base = re_components(r), re_components(b)
        tab[m].update(comp)
        tab[m].update({k + '_baseline': v for k, v in comp_base.items()})
        tab[m]['sigma2'] = comp['sigma2_u0']                 # kept for back-compat
        # random-SLOPE sigma^2_u0 reduction (kept for reference; can be negative)
        tab[m]['delta_pct'] = (comp_base['sigma2_u0'] - comp['sigma2_u0']) / comp_base['sigma2_u0'] * 100
        # random-INTERCEPT variance-explained (reported in the augmented table)
        tab[m].update(ri_decomp(dv, base_terms, rhs, d))
        tab[m]['n_obs'], tab[m]['n_states'] = len(d), d['state'].nunique()

    # Team direction (2026-08): report the between-state variance reduction against
    # a SINGLE Model-1 baseline (M1's random-intercept sigma^2 on its own sample), so
    # every column "starts from Model 1" rather than from a same-sample refit. The
    # rigorous same-sample values are retained under *_matched keys -- they avoid the
    # AK/RI/VT sample-drop confound that inflates the inspections column against M1.
    ref_u0, ref_e = tab['M1']['sigma2_u0_ri'], tab['M1']['sigma2_e_ri']
    for m in ('M1', 'M2', 'M3'):
        tab[m]['sigma2_u0_ri_baseline_matched'] = tab[m]['sigma2_u0_ri_baseline']
        tab[m]['sigma2_e_ri_baseline_matched'] = tab[m]['sigma2_e_ri_baseline']
        tab[m]['delta_pct_ri_matched'] = tab[m]['delta_pct_ri']
        tab[m]['sigma2_u0_ri_baseline'] = ref_u0
        tab[m]['sigma2_e_ri_baseline'] = ref_e
        tab[m]['delta_pct_ri'] = (ref_u0 - tab[m]['sigma2_u0_ri']) / ref_u0 * 100
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
    print(f"{'State sigma^2_u0 (rand-int)':32}" + "".join(
        f"{tab[m].get('sigma2_u0_ri', ''):>13.3f}" if 'sigma2_u0_ri' in tab[m] else f"{'':>13}" for m in ['M1', 'M2', 'M3']))
    print(f"{'Delta sigma^2_u0 (rand-int)':32}" + "".join(
        f"{tab[m]['delta_pct_ri']:>12.1f}%" if 'delta_pct_ri' in tab[m] else f"{'':>13}" for m in ['M1', 'M2', 'M3']))
    print(f"{'  [rand-slope sigma^2_u0 delta]':32}" + "".join(
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

fit_report()

with open(GEN + 'paper_table_params_corrected.json', 'w') as f:
    json.dump({'inspections': insp, 'violations': viol}, f, indent=2)
print(f"\nWrote {GEN}paper_table_params_corrected.json")
