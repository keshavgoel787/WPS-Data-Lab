"""
Validation gate for the violations-per-inspection ("ratio") models.

Not a unit-test suite -- an analysis artifact, matching the convention of
scripts/validate_count_models.py, scripts/validate_nb2_stepwise.py and
scripts/validate_jafari_crossed.py. Each check proves either that a step
reproduces something already known, or that an invariant the arm's conclusions
depend on actually holds.

The arm under test (scripts/jafari_ratio_models.R, scripts/report_jafari_ratio.py,
data/generated/jafari_ratio_results.json, docs/jafari_ratio_models.md) models
violations PER INSPECTION on the 2011-2021 WPS panel, crossed random intercepts
`(1 | state) + (1 | year)`, stepwise M1 -> M2 -> M3, three specifications:

    A  nbinom2, DV = violations, offset(log(inspections)), rows inspections > 0
    B  Spec A + a zero-inflation block mirroring the conditional predictors
    C  Gaussian LMM on log((violations + 1) / (inspections + 1)), all rows

TWO THINGS THIS FILE DELIBERATELY DOES THE HARD WAY.

1. Nothing central is read out of the JSON and compared to itself. The
   offset-drop derivation, the missing-violation rows, every analytic sample,
   both variance-reduction columns and the Spec B structural-zero probability
   are all RE-DERIVED from data/generated/count_model_panel_2021.csv (plus, for
   the ZI probability, the stored coefficients) and only then compared to what
   the arm reports.

2. The offset is NEW machinery in this project and nothing else gates it.
   validate_jafari_crossed.py [2] gates the crossed random-effects structure;
   no existing gate touches an `offset()` term. Section [9] therefore fits a
   Gaussian crossed model WITH the offset in both glmmTMB and
   statsmodels.MixedLM and requires agreement, and section [13] cross-checks the
   count-scale offset in a different package entirely.

Run: python3 scripts/validate_jafari_ratio.py            (exits 1 on any failure)
     python3 scripts/validate_jafari_ratio.py --skip-slow
         Skips [9], [12] and [13] -- the blocks that shell out to Rscript or
         refit in statsmodels. FOR ITERATION DURING DEVELOPMENT ONLY. A run with
         --skip-slow is NOT a passing run, and the SKIP lines say so.

ON READING THE OUTPUT: do not quote the total on its own. main() prints a
per-section breakdown and an explicit SKIP line for every bypassed block, so a
run in which a block silently vanished is visibly different from one in which it
ran.

No warnings are suppressed anywhere in this file. Everything is captured with
warnings.catch_warnings(record=True) and reported -- the project's signature
defect (a non-converged lbfgs fit that reached print) was hidden by exactly the
filterwarnings('ignore') idiom this file refuses to use.
"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

ROOT = '/Users/keshavgoel/Research/'
GEN = ROOT + 'data/generated/'
SCRIPTS = ROOT + 'scripts/'
DOCS = ROOT + 'docs/'

PANEL = GEN + 'count_model_panel_2021.csv'
RESULTS = GEN + 'jafari_ratio_results.json'
MEMO = DOCS + 'jafari_ratio_models.md'

FAILURES = []
SECTIONS = []
_CUR = None

SPECS = ('A', 'B', 'C')
MODELS = ('M1', 'M2', 'M3')
BASELINES = ('M1matched',)

CUBIC = ['time', 'time2', 'time3']
M2_ADD = ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z']
M3_ADD = M2_ADD + ['h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']
RHS = {'M1': CUBIC, 'M2': CUBIC + M2_ADD, 'M3': CUBIC + M3_ADD}
# The matched Model-1 baseline is the M1 FORMULA on the M2/M3 ROWS.
SAMPLE_RHS = {'M1': CUBIC, 'M2': CUBIC + M2_ADD, 'M3': CUBIC + M3_ADD,
              'M1matched': CUBIC + M3_ADD}
FORMULA_RHS = {'M1': CUBIC, 'M2': CUBIC + M2_ADD, 'M3': CUBIC + M3_ADD,
               'M1matched': CUBIC}

CROSSED_RE = '(1 | state) + (1 | year)'

# The committed panel this arm was fit on. A hash, not just a shape check: two
# CSVs can share 539 rows and 16 columns and still differ in every value, and
# every number below is derived from this file.
PANEL_SHA256 = '0ba612db5fc086f9318831402e37cb902fd3f8b36e2f2e94a9aff92dbac6ad2e'
PANEL_ROWS = 539
PANEL_STATES = 49
PANEL_YEARS = 11

# CLAUDE.md's recorded claim about what the offset costs, as a literal. The arm
# exists to measure this, so the validator must not learn it from the arm.
EXPECTED_OFFSET_DROPS = {
    ('Connecticut', 2018), ('Kansas', 2018), ('Massachusetts', 2018),
    ('Oregon', 2020), ('Oregon', 2021), ('Utah', 2017), ('West Virginia', 2012),
}
# Newly discovered by THIS arm and recorded nowhere else, so it gets an
# independent derivation of its own (section [3]).
EXPECTED_MISSING_VIOLATIONS = {
    ('Delaware', 2011), ('Indiana', 2011), ('Louisiana', 2011),
    ('North Dakota', 2011), ('Vermont', 2011), ('Virginia', 2011),
}
# No BLS pesticide-applicator series, hence no SPEND_APP_z, hence dropped from
# every M2/M3 sample.
EXPECTED_L2_DROPPED_STATES = {'Alaska', 'Rhode Island', 'Vermont'}

# A variance component is AT THE BOUNDARY when it is negligible relative to the
# residual variance of the SAME fit. There a relative tolerance is not a test:
# glmmTMB's 8.7e-10 against statsmodels' 2.6e-4 is a 100% relative "error" and
# also two packages agreeing the variance is zero. The floor is scale-relative
# (1e-4 x that fit's own sigma2_e) rather than a bare constant, because
# "negligible" is only defined against something. Idiom copied from
# validate_jafari_crossed.py, where an absolute floor was tried first and was
# wrong. sigma2_year is at or near the boundary in every fit in this arm.
VAR_ABS_FLOOR = 1e-6
VAR_REL_FLOOR = 1e-4

# statsmodels warnings that mean "this optimizer failed" and therefore disqualify
# a fit from being the reference. NOT included: "The MLE may be on the boundary
# of the parameter space" and "Random effects covariance is singular", which are
# LEGITIMATE results here -- sigma2_year really is at zero -- and disqualifying
# them would throw away the correct optimum. This distinction is the one
# paper_table_models_2021.py had to learn in 2026-08-31.
SM_FAILURE_MARKERS = ('failed to converge', 'did not converge',
                      'optimization failed')


# ------------------------------------------------------------------ plumbing


def section(tag, title):
    global _CUR
    _CUR = {'name': f'[{tag}] {title}', 'pass': 0, 'fail': 0, 'skip': 0}
    SECTIONS.append(_CUR)
    print(f"\n[{tag}] {title}")


def check(label, condition, detail=''):
    if condition:
        print(f"  PASS  {label}")
        if _CUR is not None:
            _CUR['pass'] += 1
    else:
        print(f"  FAIL  {label}" + (f" -- {detail}" if detail else ''))
        FAILURES.append(label)
        if _CUR is not None:
            _CUR['fail'] += 1


def skip(label, reason):
    print(f"  SKIP  {label} -- {reason}")
    if _CUR is not None:
        _CUR['skip'] += 1


def check_close(label, got, want, tol, kind='rel'):
    if got is None or want is None:
        check(f"{label} (both values present)", False, f"got {got!r}, want {want!r}")
        return
    delta = abs(got - want) if kind == 'abs' else abs(got - want) / max(abs(want), 1e-12)
    check(f"{label} ({kind} delta {delta:.2e} <= {tol:.0e})", delta <= tol,
          f"got {got!r}, want {want!r}")


def check_var_close(label, got, want, scale_ref=None, rel=2e-2):
    """Boundary-aware comparison of two variance components.

    Below max(VAR_ABS_FLOOR, VAR_REL_FLOOR * scale_ref) both numbers are read as
    "no variance at this level" and the comparison passes; above it the ordinary
    relative test applies, so a genuine disagreement (0.05 against 0.0 with
    sigma2_e ~0.8) still fails.
    """
    if got is None or want is None:
        check(f"{label} (both values present)", False, f"got {got!r}, want {want!r}")
        return
    floor = VAR_ABS_FLOOR
    if scale_ref is not None and scale_ref > 0:
        floor = max(floor, VAR_REL_FLOOR * scale_ref)
    if max(abs(got), abs(want)) < floor:
        check(f"{label} (both at the variance boundary, < {floor:.1e} "
              f"= 1e-4 x sigma2_e)", True)
        return
    check_close(label, got, want, rel, 'rel')


_PANEL_CACHE = {}


def panel():
    if 'p' not in _PANEL_CACHE:
        p = pd.read_csv(PANEL)
        p['log_ratio'] = np.log((p['violations'] + 1) / (p['inspections'] + 1))
        _PANEL_CACHE['p'] = p
    return _PANEL_CACHE['p']


def load_results():
    with open(RESULTS) as fh:
        return json.load(fh)


def analytic_rows(dv, sample_rhs, positive_denom):
    """Re-implementation of jafari_ratio_models.R::analytic_rows.

    Deliberately a re-implementation and not a call into anything the arm
    wrote: if the arm's row selection is wrong, reusing it would hide that.
    """
    p = panel()
    need = list(dict.fromkeys([dv] + sample_rhs + ['state', 'year', 'time']))
    d = p.dropna(subset=need)
    if positive_denom:
        d = d[d['inspections'] > 0]
    return d


def record_key(spec, model):
    return f'{spec}__{model}'


def expected_keys():
    """Derived from specs x (models + baselines), never hardcoded as 13."""
    keys = {record_key(s, m) for s in SPECS for m in MODELS + BASELINES}
    keys.add('__offset_derivation')
    return keys


def spec_dv(spec):
    return 'log_ratio' if spec == 'C' else 'violations'


def spec_offset(spec):
    return spec in ('A', 'B')


def sample_for(spec, model):
    return analytic_rows(spec_dv(spec), SAMPLE_RHS[model], spec_offset(spec))


# ============================================================
# [0] PRECONDITION -- the panel is the frozen one every number came from
# ============================================================
def validate_precondition():
    section('0', 'Precondition: count_model_panel_2021.csv is the frozen input')
    check('panel file exists', os.path.exists(PANEL), PANEL)
    if not os.path.exists(PANEL):
        return
    digest = hashlib.sha256(open(PANEL, 'rb').read()).hexdigest()
    check(f'panel sha256 == {PANEL_SHA256[:16]}...', digest == PANEL_SHA256,
          f'got {digest}')
    p = pd.read_csv(PANEL)
    check(f'panel rows == {PANEL_ROWS}', len(p) == PANEL_ROWS, f'got {len(p)}')
    check(f'panel states == {PANEL_STATES}', p['state'].nunique() == PANEL_STATES,
          f'got {p["state"].nunique()}')
    check(f'panel year levels == {PANEL_YEARS}', p['year'].nunique() == PANEL_YEARS,
          f'got {p["year"].nunique()}')
    check('panel covers 2011-2021',
          (int(p['year'].min()), int(p['year'].max())) == (2011, 2021),
          f'got {p["year"].min()}-{p["year"].max()}')
    for col in ['state', 'year', 'time', 'time2', 'time3', 'violations',
                'inspections', 'log_violations', 'log_inspections'] + M3_ADD:
        check(f'column present: {col}', col in p.columns)
    check('time == year - 2017 throughout', bool((p['time'] == p['year'] - 2017).all()))
    check('time2 == time^2', bool(np.allclose(p['time2'], p['time'] ** 2)))
    check('time3 == time^3', bool(np.allclose(p['time3'], p['time'] ** 3)))
    check('Wyoming absent (WPS view has no WY row)',
          'Wyoming' not in set(p['state']))
    # The arm's analytic_rows() runs complete.cases() over the DV and the
    # predictors but NOT over `inspections`, then filters `inspections > 0`. In
    # R a missing denominator would survive complete.cases and then index with
    # NA, injecting all-NA rows. That is inert today only because the column has
    # no missing values -- assert it, so the latent hazard cannot become live
    # silently.
    check('inspections has no missing values (the arm filters on it without a '
          'completeness check first)', int(p['inspections'].isna().sum()) == 0,
          f'{int(p["inspections"].isna().sum())} missing')
    check('inspections is non-negative', bool((p['inspections'] >= 0).all()))
    check('violations is non-negative where observed',
          bool((p['violations'].dropna() >= 0).all()))
    check('results JSON exists', os.path.exists(RESULTS))
    check('memo exists', os.path.exists(MEMO))


# ============================================================
# [1] RECORD STRUCTURE
# ============================================================
SCALAR_KEYS = ('n_obs', 'n_states', 'n_years', 'obs_zeros', 'aic', 'bic',
               'loglik', 'df', 'sigma2_state', 'sigma2_year')
META_KEYS = ('formula', 'zi_formula', 're_used', 'uses_offset', 'sample_rhs',
             'converged', 'pd_hess', 'conv_code', 'family_name', 'dispersion',
             'sigma2_e', 'spec', 'spec_label', 'model', 'dv', 'message',
             'zi_tier_reached', 'zi_tier_attempts', 'zi_degenerate',
             'zi_degenerate_reason', 'zi_loglik_criterion_applied',
             'zi_prob_mean', 'year_blups', 'cond', 'zi')
ZI_TIERS = ('mirror', 'covariates', 'reduced', 'intercept')


def _n_params(spec, model):
    """Expected df, from the specification alone.

    fixed effects (intercept + rhs) + 2 crossed variances + 1 dispersion
    (theta for nbinom2, sigma for Gaussian) + the ZI block for Spec B.
    """
    k = 1 + len(FORMULA_RHS[model]) + 2 + 1
    if spec == 'B':
        k += 1 + len(FORMULA_RHS[model])   # mirrored ZI block
    return k


def validate_records():
    section('1', 'Record structure (a loop over 13 records, not 13 findings)')
    r = load_results()
    want = expected_keys()
    check(f'record key set == specs x (models + baselines) + derivation '
          f'({len(want)} keys)', set(r) == want,
          f'missing {sorted(want - set(r))}; extra {sorted(set(r) - want)}')

    for spec in SPECS:
        for model in MODELS + BASELINES:
            k = record_key(spec, model)
            if k not in r:
                continue
            v = r[k]
            for key in SCALAR_KEYS + META_KEYS:
                check(f'{k}: key present: {key}', key in v)
            # --- types and finiteness ---
            for key in ('n_obs', 'n_states', 'n_years', 'obs_zeros', 'df'):
                val = v.get(key)
                check(f'{k}: {key} is a positive int',
                      isinstance(val, int) and val > 0, f'got {val!r}')
            for key in ('aic', 'bic', 'loglik', 'sigma2_state', 'sigma2_year',
                        'dispersion'):
                val = v.get(key)
                check(f'{k}: {key} is a finite float',
                      isinstance(val, float) and math.isfinite(val), f'got {val!r}')
            check(f'{k}: sigma2_state > 0', (v.get('sigma2_state') or 0) > 0)
            check(f'{k}: sigma2_year >= 0', (v.get('sigma2_year') if
                                             v.get('sigma2_year') is not None
                                             else -1) >= 0)
            # --- specification invariants ---
            check(f'{k}: crossed random intercepts, no random slope',
                  v.get('re_used') == CROSSED_RE and 'time | state' not in v['formula'],
                  v.get('formula'))
            check(f'{k}: spec/model tags self-consistent',
                  v.get('spec') == spec and v.get('model') == model,
                  f"{v.get('spec')}/{v.get('model')}")
            check(f'{k}: dv == {spec_dv(spec)}', v.get('dv') == spec_dv(spec))
            check(f'{k}: uses_offset == {spec_offset(spec)}',
                  bool(v.get('uses_offset')) is spec_offset(spec))
            check(f'{k}: offset(log(inspections)) in formula iff Spec A/B',
                  ('offset(log(inspections))' in v['formula']) is spec_offset(spec))
            # log_inspections is the DENOMINATOR in this arm and must never
            # appear as a predictor -- that is the whole point of the arm.
            check(f'{k}: log_inspections is NOT a conditional predictor',
                  'log_inspections' not in (v.get('cond') or {}))
            check(f'{k}: log_inspections is NOT a ZI predictor',
                  'log_inspections' not in (v.get('zi') or {}))
            check(f'{k}: formula RHS == the spec build-up',
                  all(t in v['formula'] for t in FORMULA_RHS[model]) and
                  not any(t in (v.get('cond') or {})
                          for t in set(M3_ADD) - set(FORMULA_RHS[model])),
                  v['formula'])
            check(f'{k}: sample_rhs == {model} sample definition',
                  v.get('sample_rhs') == SAMPLE_RHS[model],
                  f"got {v.get('sample_rhs')}")
            # --- family and dispersion semantics ---
            fam = 'gaussian' if spec == 'C' else 'nbinom2'
            check(f'{k}: family_name == {fam}', v.get('family_name') == fam,
                  f"got {v.get('family_name')}")
            if spec == 'C':
                check_close(f'{k}: sigma2_e == dispersion^2 (Gaussian)',
                            v.get('sigma2_e'), (v.get('dispersion') or 0) ** 2, 1e-9)
            else:
                # sigma(fit)^2 for nbinom2 is theta^2, not a residual variance.
                # Exporting it as sigma2_e is a documented defect of an earlier
                # arm; this arm must leave it null.
                check(f'{k}: sigma2_e is null (a count fit has no residual '
                      f'variance)', v.get('sigma2_e') is None,
                      f"got {v.get('sigma2_e')!r}")
            # --- coefficient blocks ---
            cond = v.get('cond') or {}
            want_terms = ['(Intercept)'] + FORMULA_RHS[model]
            check(f'{k}: conditional block == intercept + {len(FORMULA_RHS[model])} terms',
                  list(cond) == want_terms, f'got {list(cond)}')
            for term, c in cond.items():
                check(f'{k}: cond[{term}] has b/se/z/p',
                      set(c) == {'b', 'se', 'z', 'p'}, f'got {sorted(c)}')
                check(f'{k}: cond[{term}] all finite',
                      all(isinstance(c[x], float) and math.isfinite(c[x])
                          for x in ('b', 'se', 'z', 'p')), f'got {c}')
                check(f'{k}: cond[{term}] se > 0', c['se'] > 0)
                check(f'{k}: cond[{term}] p in [0, 1]', 0 <= c['p'] <= 1)
                check_close(f'{k}: cond[{term}] z == b/se', c['z'], c['b'] / c['se'],
                            1e-6)
            zi = v.get('zi') or {}
            if spec == 'B':
                check(f'{k}: ZI tier reached is one of {ZI_TIERS}',
                      v.get('zi_tier_reached') in ZI_TIERS,
                      f"got {v.get('zi_tier_reached')!r}")
                check(f'{k}: ZI block == mirror of the conditional block',
                      list(zi) == want_terms, f'got {list(zi)}')
                check(f'{k}: ZI terms are a SUBSET of conditional terms '
                      '(what keeps one analytic sample per model)',
                      set(zi) <= set(cond), f'extra: {set(zi) - set(cond)}')
                check(f'{k}: ZI formula has no random-effects term (so no ZI '
                      'variance exists to extract or to omit)',
                      '|' not in (v.get('zi_formula') or ''), v.get('zi_formula'))
                for term, c in zi.items():
                    check(f'{k}: zi[{term}] has b/se/z/p',
                          set(c) == {'b', 'se', 'z', 'p'}, f'got {sorted(c)}')
                    check(f'{k}: zi[{term}] all finite',
                          all(isinstance(c[x], float) and math.isfinite(c[x])
                              for x in ('b', 'se', 'z', 'p')), f'got {c}')
            else:
                check(f'{k}: no ZI block (Spec {spec} is not zero-inflated)', not zi)
                check(f'{k}: zi_formula == ~0', v.get('zi_formula') == '~0',
                      f"got {v.get('zi_formula')!r}")
                check(f'{k}: zi_tier_reached is null', v.get('zi_tier_reached') is None)
                check(f'{k}: no structural-zero probability is reported',
                      v.get('zi_prob_mean') is None)
            # --- internal arithmetic ---
            check_close(f'{k}: AIC == 2*df - 2*loglik', v['aic'],
                        2 * v['df'] - 2 * v['loglik'], 1e-9)
            check_close(f'{k}: BIC == log(n)*df - 2*loglik', v['bic'],
                        math.log(v['n_obs']) * v['df'] - 2 * v['loglik'], 1e-9)
            check(f'{k}: df == {_n_params(spec, model)} (fixed + 2 variances + '
                  f'dispersion{" + ZI mirror" if spec == "B" else ""})',
                  v['df'] == _n_params(spec, model), f"got {v['df']}")
            check(f'{k}: year_blups cover all {v["n_years"]} years',
                  len(v.get('year_blups') or {}) == v['n_years'])


# ============================================================
# [2] THE OFFSET-DROP DERIVATION, RE-DERIVED FROM THE PANEL
# ============================================================
def validate_offset_derivation():
    section('2', 'Offset-drop derivation re-derived from the panel (the arm\'s '
                 'central factual claim)')
    p = panel()
    zero_insp = p[p['inspections'] == 0]
    mine = {(row.state, int(row.year)) for row in zero_insp.itertuples()}

    check('exactly 7 panel rows have inspections == 0', len(zero_insp) == 7,
          f'got {len(zero_insp)}')
    check('the 7 zero-inspection state-years are the recorded ones',
          mine == EXPECTED_OFFSET_DROPS,
          f'missing {sorted(EXPECTED_OFFSET_DROPS - mine)}; '
          f'unexpected {sorted(mine - EXPECTED_OFFSET_DROPS)}')
    check('all 7 zero-inspection rows also have violations == 0',
          bool((zero_insp['violations'] == 0).all()),
          f"violations: {list(zero_insp['violations'])}")
    check('none of the 7 has a missing violation count',
          int(zero_insp['violations'].isna().sum()) == 0)

    # Only NOW is the stored record opened, and it is compared to the
    # independent derivation above rather than to itself.
    d = load_results()['__offset_derivation']
    check('stored derivation names the frozen panel',
          d['panel_csv'].endswith('count_model_panel_2021.csv'), d['panel_csv'])
    check(f'stored panel_rows == {PANEL_ROWS}', d['panel_rows'] == PANEL_ROWS,
          f"got {d['panel_rows']}")
    check(f'stored panel_states == {PANEL_STATES}',
          d['panel_states'] == PANEL_STATES, f"got {d['panel_states']}")
    check(f'stored panel_years == {PANEL_YEARS}',
          d['panel_years'] == PANEL_YEARS, f"got {d['panel_years']}")
    check('stored rows_zero_inspections == my count (7)',
          d['rows_zero_inspections'] == len(zero_insp),
          f"stored {d['rows_zero_inspections']}, derived {len(zero_insp)}")

    for model in MODELS:
        m = d['by_model'][model]
        keep_all = analytic_rows('violations', RHS[model], False)
        keep_pos = analytic_rows('violations', RHS[model], True)
        dropped = keep_all[keep_all['inspections'] <= 0]
        mine_m = {(row.state, int(row.year)) for row in dropped.itertuples()}
        tag = f'{model}'
        check(f'{tag}: stored n_all_rows == my derivation ({len(keep_all)})',
              m['n_all_rows'] == len(keep_all), f"stored {m['n_all_rows']}")
        check(f'{tag}: stored n_offset_rows == my derivation ({len(keep_pos)})',
              m['n_offset_rows'] == len(keep_pos), f"stored {m['n_offset_rows']}")
        check(f'{tag}: stored n_dropped == my derivation ({len(dropped)})',
              m['n_dropped'] == len(dropped), f"stored {m['n_dropped']}")
        check(f'{tag}: the dropped set is the recorded 7 state-years',
              mine_m == EXPECTED_OFFSET_DROPS,
              f'derived {sorted(mine_m)}')
        check(f'{tag}: stored dropped_rows list matches my derivation',
              {(x['state'], int(x['year'])) for x in m['dropped_rows']} == mine_m)
        check(f'{tag}: stored n_dropped_zero_violation == 7 and equals n_dropped',
              m['n_dropped_zero_violation'] == int((dropped['violations'] == 0).sum())
              == m['n_dropped'], f"stored {m['n_dropped_zero_violation']}")
        check(f'{tag}: stored n_dropped_positive_violation == 0',
              m['n_dropped_positive_violation']
              == int((dropped['violations'] > 0).sum()) == 0)
        check(f'{tag}: stored zeros_all_rows == my derivation '
              f'({int((keep_all["violations"] == 0).sum())})',
              m['zeros_all_rows'] == int((keep_all['violations'] == 0).sum()),
              f"stored {m['zeros_all_rows']}")
        check(f'{tag}: stored zeros_offset_rows == my derivation '
              f'({int((keep_pos["violations"] == 0).sum())})',
              m['zeros_offset_rows'] == int((keep_pos['violations'] == 0).sum()),
              f"stored {m['zeros_offset_rows']}")
        # THE substantive point: the offset removes only zeros, so every row it
        # deletes is one the zero-inflation machinery would otherwise model.
        check(f'{tag}: offset removes exactly n_dropped zeros '
              '(it deletes structural zeros and nothing else)',
              m['zeros_all_rows'] - m['zeros_offset_rows'] == m['n_dropped'],
              f"{m['zeros_all_rows']} -> {m['zeros_offset_rows']}, "
              f"dropped {m['n_dropped']}")


# ============================================================
# [3] THE SIX MISSING-VIOLATION ROWS
# ============================================================
def validate_missing_violations():
    section('3', 'The 6 missing-violation rows (new to this arm, recorded '
                 'nowhere else)')
    p = panel()
    miss = p[p['violations'].isna()]
    mine = {(row.state, int(row.year)) for row in miss.itertuples()}
    check('exactly 6 panel rows have a missing violation count', len(miss) == 6,
          f'got {len(miss)}')
    check('the 6 are DE/IN/LA/ND/VT/VA, all 2011', mine == EXPECTED_MISSING_VIOLATIONS,
          f'derived {sorted(mine)}')
    check('all 6 are from 2011', set(int(y) for _, y in mine) == {2011})
    check('all 6 have an OBSERVED inspection count (so it is the DV that is '
          'missing, not the row)', int(miss['inspections'].isna().sum()) == 0)
    check(f'{PANEL_ROWS} - 6 == 533 is the all-rows baseline, not {PANEL_ROWS}',
          PANEL_ROWS - len(miss) == 533)

    r = load_results()
    d = r['__offset_derivation']
    check('stored rows_missing_violations == my count (6)',
          d['rows_missing_violations'] == len(miss),
          f"stored {d['rows_missing_violations']}")
    # Dropped from EVERY spec, including Spec C: the log-ratio response is
    # log((violations + 1)/(inspections + 1)), which is NaN where violations is.
    check('log_ratio is missing on exactly those 6 rows',
          set(p.index[p['log_ratio'].isna()]) == set(miss.index))
    for spec in SPECS:
        for model in MODELS + BASELINES:
            v = r[record_key(spec, model)]
            samp = sample_for(spec, model)
            check(f'{record_key(spec, model)}: none of the 6 missing-violation '
                  'rows is in the analytic sample',
                  not (set(samp.index) & set(miss.index)))
            check(f'{record_key(spec, model)}: n_obs excludes them '
                  f'({v["n_obs"]} <= {PANEL_ROWS - 6})',
                  v['n_obs'] <= PANEL_ROWS - 6)
    check('Spec C Model 1 sample is 533, i.e. "all rows" means 533',
          r['C__M1']['n_obs'] == 533, f"got {r['C__M1']['n_obs']}")


# ============================================================
# [4] ANALYTIC SAMPLES
# ============================================================
EXPECTED_N = {
    ('A', 'M1'): (526, 49), ('A', 'M2'): (494, 46), ('A', 'M3'): (494, 46),
    ('B', 'M1'): (526, 49), ('B', 'M2'): (494, 46), ('B', 'M3'): (494, 46),
    ('C', 'M1'): (533, 49), ('C', 'M2'): (501, 46), ('C', 'M3'): (501, 46),
    ('A', 'M1matched'): (494, 46), ('B', 'M1matched'): (494, 46),
    ('C', 'M1matched'): (501, 46),
}


def validate_samples():
    section('4', 'Analytic samples: N and state sets implied by listwise '
                 'deletion + the offset')
    r = load_results()
    p = panel()
    all_states = set(p['state'])
    for spec in SPECS:
        for model in MODELS + BASELINES:
            k = record_key(spec, model)
            v = r[k]
            samp = sample_for(spec, model)
            want_n, want_s = EXPECTED_N[(spec, model)]
            states = set(samp['state'])
            check(f'{k}: N == {want_n} (derived from the panel)',
                  len(samp) == want_n == v['n_obs'],
                  f'derived {len(samp)}, stored {v["n_obs"]}, expected {want_n}')
            check(f'{k}: states == {want_s}',
                  len(states) == want_s == v['n_states'],
                  f'derived {len(states)}, stored {v["n_states"]}')
            check(f'{k}: n_years == 11 (no year lost to deletion)',
                  samp['year'].nunique() == 11 == v['n_years'],
                  f"derived {samp['year'].nunique()}, stored {v['n_years']}")
            if want_s == 46:
                check(f'{k}: the 3 missing states are exactly AK/RI/VT',
                      all_states - states == EXPECTED_L2_DROPPED_STATES,
                      f'got {sorted(all_states - states)}')
            else:
                check(f'{k}: no state dropped', all_states - states == set())
            # obs_zeros counts zero-VIOLATION rows in every spec, including
            # Spec C, whose response is the log ratio rather than the count.
            # Pinned here so the column's estimand cannot drift.
            check(f'{k}: obs_zeros counts zero-VIOLATION rows in this sample',
                  v['obs_zeros'] == int((samp['violations'] == 0).sum()),
                  f"stored {v['obs_zeros']}, derived "
                  f"{int((samp['violations'] == 0).sum())}")
            if spec_offset(spec):
                check(f'{k}: every row in the sample has inspections > 0 '
                      '(required for log(inspections) to exist)',
                      bool((samp['inspections'] > 0).all()))

    # The matched baseline is the reason the second Delta column means anything:
    # it must sit on the M2/M3 rows exactly, not merely have the same count.
    for spec in SPECS:
        base = sample_for(spec, 'M1matched')
        m2 = sample_for(spec, 'M2')
        m3 = sample_for(spec, 'M3')
        check(f'Spec {spec}: M1matched sample is row-identical to the M2 sample',
              set(base.index) == set(m2.index))
        check(f'Spec {spec}: M1matched sample is row-identical to the M3 sample',
              set(base.index) == set(m3.index))
        check(f'Spec {spec}: M1matched uses the M1 FORMULA '
              '(cubic time only, no level-2 covariates)',
              list(r[record_key(spec, "M1matched")]['cond'])
              == ['(Intercept)'] + CUBIC)
        check(f'Spec {spec}: M1matched sample is strictly smaller than M1\'s '
              '(that is what it exists to correct for)',
              len(base) < len(sample_for(spec, 'M1')))
        check(f'Spec {spec}: M2 and M3 share one sample '
              '(so their AICs are comparable within the spec)',
              set(m2.index) == set(m3.index))

    # Spec A and Spec B must share rows exactly -- the precondition for the only
    # AIC comparison the memo makes.
    for model in MODELS + BASELINES:
        a, b = sample_for('A', model), sample_for('B', model)
        check(f'{model}: Spec A and Spec B share one analytic sample',
              set(a.index) == set(b.index))
    # And Spec C keeps precisely the 7 rows the offset deletes.
    for model in MODELS:
        c = set(sample_for('C', model).index)
        a = set(sample_for('A', model).index)
        extra = c - a
        check(f'{model}: Spec C keeps exactly the 7 rows the offset deletes',
              len(extra) == 7 and
              {(p.loc[i, 'state'], int(p.loc[i, 'year'])) for i in extra}
              == EXPECTED_OFFSET_DROPS, f'{len(extra)} extra rows')


# ============================================================
# [5] VARIANCE REDUCTIONS -- RE-DERIVED, NON-CIRCULAR
# ============================================================
def _delta(base_s2, fit_s2):
    return 100.0 * (base_s2 - fit_s2) / base_s2


def validate_variance():
    section('5', 'Variance reductions re-derived from raw sigma2_state '
                 '(both bases, non-circular)')
    r = load_results()
    memo = open(MEMO).read()
    for spec in SPECS:
        s2_m1 = r[record_key(spec, 'M1')]['sigma2_state']
        s2_matched = r[record_key(spec, 'M1matched')]['sigma2_state']
        check(f'Spec {spec}: the two baselines are DIFFERENT fits '
              '(unmatched vs matched sigma2_state)',
              abs(s2_m1 - s2_matched) > 1e-6,
              f'M1 {s2_m1}, M1matched {s2_matched}')
        check(f'Spec {spec}: the matched baseline is fit on fewer rows',
              r[record_key(spec, 'M1matched')]['n_obs']
              < r[record_key(spec, 'M1')]['n_obs'])
        for model in ('M2', 'M3'):
            v = r[record_key(spec, model)]
            d_un = _delta(s2_m1, v['sigma2_state'])
            d_ma = _delta(s2_matched, v['sigma2_state'])
            tag = f'Spec {spec} {model}'
            # The memo is where these percentages are published; the JSON does
            # not store them. So the re-derivation is checked against print.
            check(f'{tag}: Delta vs M1 = {d_un:+.1f}% appears in the memo',
                  f'{d_un:+.1f}%' in memo, f'computed {d_un:+.4f}%')
            check(f'{tag}: Delta vs matched M1 = {d_ma:+.1f}% appears in the memo',
                  f'{d_ma:+.1f}%' in memo, f'computed {d_ma:+.4f}%')
            check(f'{tag}: the two Deltas are genuinely different numbers',
                  abs(d_un - d_ma) > 1e-6, f'{d_un} vs {d_ma}')
            # Direction: the matched baseline corrects the unmatched one, and
            # which way it corrects is a property of the data, not a choice.
            check(f'{tag}: matched Delta sits on the matched baseline '
                  '(sign of the correction follows sigma2_matched vs sigma2_M1)',
                  (d_ma < d_un) == (s2_matched < s2_m1),
                  f'{d_un:+.4f}% -> {d_ma:+.4f}%, baselines {s2_m1} / {s2_matched}')
            check(f'{tag}: reduction is in [-100, 100]%', -100 < d_un < 100)
        # A random-INTERCEPT-only model: sigma2_state already IS the published
        # delta_pct_ri quantity, which is why no 'ri' refit series exists here.
        check(f'Spec {spec}: no random slope, so no separate ri refit is needed',
              'time | state' not in r[record_key(spec, 'M1')]['formula'])
    check('the memo says the percentages are already on the random-intercept basis',
          'already IS the random-intercept quantity' in memo)


# ============================================================
# [6] AIC COMPARABILITY DISCIPLINE
# ============================================================
# Sentences that would only ever appear in an invalid cross-spec AIC claim.
MEMO_FORBIDDEN_AIC = [
    'Spec C fits better', 'Spec C fits best', 'Spec C has the lowest AIC',
    'lowest AIC of the three', 'best AIC of the three',
    'lower AIC than Spec A', 'lower AIC than Spec B',
    'C beats A', 'C beats B', 'Spec C beats',
    'AIC across all three specs', 'AIC across the three specs',
    'AIC ranks the three', 'compare the AICs of all three',
]
MEMO_REQUIRED_AIC = [
    '**AIC is comparable between Spec A and Spec B and nowhere else.**',
    'its AIC must never be compared to A or B',
    'This is the one AIC comparison in this memo that is strictly valid',
]


def validate_aic_discipline():
    section('6', 'AIC comparability: A vs B re-derived, C never compared')
    r = load_results()
    memo = open(MEMO).read()

    margins = []
    for model in MODELS:
        a, b = r[record_key('A', model)], r[record_key('B', model)]
        # The comparison is only valid because these hold; assert them here
        # rather than trusting the memo's prose.
        check(f'{model}: A and B share n_obs ({a["n_obs"]})',
              a['n_obs'] == b['n_obs'], f"{a['n_obs']} vs {b['n_obs']}")
        check(f'{model}: A and B share the conditional formula',
              a['formula'] == b['formula'])
        check(f'{model}: A and B share the family',
              a['family_name'] == b['family_name'] == 'nbinom2')
        check(f'{model}: the ZI block is the ONLY difference '
              f'(df {a["df"]} -> {b["df"]})',
              b['df'] - a['df'] == 1 + len(FORMULA_RHS[model]),
              f'df delta {b["df"] - a["df"]}')
        dl = b['aic'] - a['aic']
        margins.append(dl)
        check(f'{model}: Spec B beats Spec A on AIC (Delta {dl:+.2f})', dl < 0)
        check(f'{model}: the memo prints Delta AIC = {dl:+.2f}',
              f'**{dl:+.2f}**' in memo, f'computed {dl:+.4f}')
    lo, hi = min(abs(m) for m in margins), max(abs(m) for m in margins)
    check(f'the 3 margins span {lo:.2f}-{hi:.2f} AIC, i.e. the claimed 76-88 band',
          75 <= lo <= 90 and 75 <= hi <= 90, f'{lo:.2f}-{hi:.2f}')
    check(f'the memo states the range as {lo:.1f}--{hi:.1f}',
          f'{lo:.1f}--{hi:.1f} points' in memo,
          f'expected "{lo:.1f}--{hi:.1f} points"')

    for pat in MEMO_REQUIRED_AIC:
        check(f'memo states: {pat!r}', pat in memo)
    for pat in MEMO_FORBIDDEN_AIC:
        check(f'memo does NOT claim: {pat!r}', pat.lower() not in memo.lower())

    # Mechanical guard: no single line of the memo may put a Spec C AIC and a
    # Spec A/B AIC side by side. In the one table where all nine appear, each
    # row is a single spec, so this is satisfied by construction -- and the
    # check is what keeps it that way.
    ab_aic = {f"{r[record_key(s, m)]['aic']:.2f}" for s in ('A', 'B') for m in MODELS}
    c_aic = {f"{r[record_key('C', m)]['aic']:.2f}" for m in MODELS}
    bad = []
    for i, line in enumerate(memo.splitlines(), 1):
        if any(x in line for x in c_aic) and any(x in line for x in ab_aic):
            bad.append(i)
    check('no memo line places a Spec C AIC alongside a Spec A/B AIC',
          not bad, f'lines {bad}')
    # Spec C's own AICs are printed (they are legitimate WITHIN Spec C at M2/M3)
    # -- confirm the memo scopes even that.
    check('memo scopes the within-Spec-C comparison too '
          '(M1 sits on a larger sample)',
          'M1 sits on a larger sample' in memo)


# ============================================================
# [7] STRUCTURAL-ZERO PROBABILITY -- THE ESTIMAND THIS PROJECT GOT WRONG ONCE
# ============================================================
def validate_zi_probability():
    section('7', 'Spec B structural-zero probability: right estimand, '
                 're-derived from the panel')
    from scipy.special import expit
    r = load_results()
    memo = open(MEMO).read()

    for model in MODELS + BASELINES:
        k = record_key('B', model)
        v = r[k]
        zi = v['zi']
        samp = sample_for('B', model)
        stored = v.get('zi_prob_mean')

        # --- THE re-derivation. The ZI formula carries no random-effects term
        # (asserted in [1]), so predict(type="zprob") is exactly
        # plogis(x_i' beta_zi) row by row, and the sample mean of that is
        # computable here from the panel plus the stored coefficients. This is
        # an independent reconstruction of the estimand, not a round-trip.
        lin = np.full(len(samp), zi['(Intercept)']['b'], dtype=float)
        for term, c in zi.items():
            if term == '(Intercept)':
                continue
            lin = lin + c['b'] * samp[term].to_numpy(dtype=float)
        derived_mean = float(expit(lin).mean())
        check_close(f'{k}: reported probability IS the sample mean of the '
                    'per-observation zprob', stored, derived_mean, 1e-8)

        # --- and it is NOT either of the two wrong estimands. Both evaluate the
        # ZI linear predictor at x = 0, which describes no state in the data
        # once the ZI block carries covariates. A draft of the previous arm's
        # memo shipped the second of these.
        b0 = zi['(Intercept)']['b']
        plogis_b0 = float(expit(b0))
        check(f'{k}: reported value is NOT plogis(zi_intercept) '
              f'({plogis_b0:.6f})', abs(stored - plogis_b0) > 1e-6,
              f'reported {stored}, plogis(b0) {plogis_b0}')
        # With no ZI random effect the Gauss-Hermite marginal E[plogis(b0+u)]
        # degenerates to plogis(b0); assert the arm reports no ZI variance so
        # that quantity cannot be constructed and mistaken for the marginal.
        check(f'{k}: no ZI random-effect variance is reported (there is no ZI '
              'random effect to have one)',
              all(key not in v for key in
                  ('sigma2_zi_state', 'sigma2_zi_u0', 'sigma_zi_u0')))

        # --- THE bound that caught the earlier defect: structural zeros are a
        # SUBSET of all zeros, so the probability must sit strictly below the
        # observed zero rate.
        obs_rate = v['obs_zeros'] / v['n_obs']
        check(f'{k}: Pr(structural zero) {stored:.4f} < observed zero rate '
              f'{obs_rate:.4f}', stored < obs_rate,
              f'{stored} vs {obs_rate}')
        check(f'{k}: Pr(structural zero) is a probability', 0 < stored < 1)
        # And the same bound against the model's own simulated zero count, which
        # is where a factor-of-3 error would show up.
        exp_struct = stored * v['n_obs']
        slack = 3 * (v.get('exp_zeros_se') or 0)
        check(f'{k}: expected structural zeros ({exp_struct:.1f}) <= simulated '
              f'total zeros ({v["exp_zeros"]:.1f})',
              exp_struct <= v['exp_zeros'] + slack)

    check('memo names the estimand explicitly',
          "sample mean of each state-year's own" in memo
          and 'zero-inflation probability' in memo)
    check('memo explicitly rejects plogis(zi_intercept)',
          'not `plogis(zi_intercept)`' in memo)
    check('memo states the subset bound that makes the number checkable',
          'must sit below it, since structural zeros are a subset of all zeros'
          in memo)
    # A Gauss-Hermite marginal is the OTHER wrong answer; it must not be
    # reported anywhere in this arm.
    for pat in ('Gauss-Hermite', 'gauss_hermite', 'marginal probability'):
        check(f'memo does not report a {pat!r} structural-zero probability',
              pat.lower() not in memo.lower())


# ============================================================
# [8] CONVERGENCE
# ============================================================
def validate_convergence():
    section('8', 'Convergence: all 12 fits, re-derived from the records')
    r = load_results()
    memo = open(MEMO).read()
    n_fits = 0
    for spec in SPECS:
        for model in MODELS + BASELINES:
            k = record_key(spec, model)
            v = r[k]
            n_fits += 1
            check(f'{k}: converged', v.get('converged') is True,
                  v.get('message', ''))
            check(f'{k}: positive-definite Hessian', v.get('pd_hess') is True)
            check(f'{k}: convergence code 0', v.get('conv_code') == 0,
                  f"got {v.get('conv_code')}")
            check(f'{k}: finite SE on every conditional coefficient',
                  all(math.isfinite(c['se']) for c in v['cond'].values()))
            check(f'{k}: no fit-time warning was raised', not v.get('message'),
                  repr(v.get('message'))[:200])
            check(f'{k}: no simulation warning was raised',
                  not v.get('sim_message'), repr(v.get('sim_message'))[:200])
            check(f'{k}: not flagged zero-inflation-degenerate',
                  not v.get('zi_degenerate'), v.get('zi_degenerate_reason', ''))
            if spec == 'B':
                att = v.get('zi_tier_attempts') or []
                check(f'{k}: ZI ladder stopped at the first (richest) tier',
                      len(att) == 1 and att[0]['tier'] == 'mirror'
                      and att[0]['converged'] and not att[0]['zi_degenerate'],
                      f'attempts: {att}')
                check_close(f'{k}: the recorded attempt AIC is the fit\'s AIC',
                            att[0]['aic'] if att else None, v['aic'], 1e-12)
    check(f'{n_fits} fits in total (9 substantive + 3 matched baselines)',
          n_fits == 12, f'got {n_fits}')
    check('memo states all 12 converged',
          'All 9 substantive fits and all 3 matched Model-1 baselines '
          '(12 fits in total) converged' in memo)
    # No non-converged estimate may be presented as an estimate.
    for marker in ('DID NOT CONVERGE', 'NOT ALL FITS CONVERGED', ' ⚠'):
        check(f'memo carries no non-convergence marker {marker!r}',
              marker not in memo)


# ============================================================
# [9] GAUSSIAN ROUND-TRIP GATE -- the OFFSET has no existing gate
# ============================================================
# Embedded rather than committed as a script: this is validator-only scaffolding
# and the task scope is one new file. It is written to a temp dir per run.
GATE_R = r'''
suppressPackageStartupMessages({ library(glmmTMB); library(jsonlite) })
options(warn = 1)
args  <- commandArgs(trailingOnly = TRUE)
panel <- read.csv(args[1], stringsAsFactors = FALSE)
panel$log_ratio <- log((panel$violations + 1) / (panel$inspections + 1))

CUBIC  <- c("time", "time2", "time3")
M2_ADD <- c("SPEND_APP_z", "SPEND_WORK_z", "lii_2017_z")
M3_ADD <- c(M2_ADD, "h2a_per_farmworker_z", "dol_demand_met_pct_z", "pct_flc_z")
RHS <- list(M1 = CUBIC, M2 = c(CUBIC, M2_ADD), M3 = c(CUBIC, M3_ADD),
            M1matched = CUBIC)
SAMPLE <- list(M1 = CUBIC, M2 = c(CUBIC, M2_ADD), M3 = c(CUBIC, M3_ADD),
               M1matched = c(CUBIC, M3_ADD))

rows <- function(dv, sample_rhs, use_offset) {
  need <- unique(c(dv, sample_rhs, "state", "year", "time"))
  dd <- panel[stats::complete.cases(panel[, need, drop = FALSE]), , drop = FALSE]
  if (use_offset) dd <- dd[dd$inspections > 0, , drop = FALSE]
  dd$state <- factor(dd$state); dd$year <- factor(dd$year)
  dd
}

pack <- function(fit, dd, msgs = character(0), extra = list()) {
  cf <- summary(fit)$coefficients$cond
  c(list(n_obs = nrow(dd), n_states = length(unique(dd$state)),
         converged = isTRUE(fit$sdr$pdHess) && fit$fit$convergence == 0,
         loglik = as.numeric(logLik(fit)), aic = AIC(fit),
         message = if (length(msgs)) paste(unique(msgs), collapse = " | ") else "",
         cond = stats::setNames(as.list(cf[, 1]), rownames(cf)),
         se   = stats::setNames(as.list(cf[, 2]), rownames(cf)),
         sigma2_state = VarCorr(fit)$cond$state[1, 1],
         sigma2_year  = VarCorr(fit)$cond$year[1, 1]), extra)
}

# Every fit's warnings are CAPTURED and carried into the JSON, never muffled and
# never allowed to leak to stderr -- the same discipline jafari_ratio_models.R
# uses, and the reason the caller can assert that stderr holds nothing beyond
# the known TMB ABI notice. A refit that fails here is scaffolding failing, not
# the arm; it is reported, not hidden.
fit_capturing <- function(expr_fun) {
  msgs <- character(0)
  res <- tryCatch(
    withCallingHandlers(expr_fun(),
                        warning = function(w) {
                          msgs <<- c(msgs, conditionMessage(w))
                          invokeRestart("muffleWarning")
                        }),
    error = function(e) e)
  list(fit = res, msgs = msgs)
}

out <- list()

# --- (a) the OFFSET gate. Gaussian is a numerical gate, not a substantive
# model: what is being tested is that glmmTMB puts offset(log(inspections))
# into the linear predictor with coefficient exactly 1 on the same rows the
# arm uses. Under an identity link that is algebraically y - offset ~ X, which
# statsmodels can reproduce with no offset support of its own. ML, matching the
# arm's REML = FALSE.
for (m in c("M1", "M3")) {
  dd <- rows("log_violations", SAMPLE[[m]], TRUE)
  form <- stats::as.formula(sprintf(
    "log_violations ~ %s + offset(log(inspections)) + (1 | state) + (1 | year)",
    paste(RHS[[m]], collapse = " + ")))
  cap <- fit_capturing(function()
    glmmTMB(form, family = gaussian, data = dd, REML = FALSE))
  key <- paste0("gauss_offset__", m)
  if (inherits(cap$fit, "error")) {
    out[[key]] <- list(n_obs = nrow(dd), converged = FALSE,
                       error = conditionMessage(cap$fit))
  } else {
    out[[key]] <- pack(cap$fit, dd, cap$msgs, list(
      formula = deparse1(form), sigma2_e = sigma(cap$fit)^2))
  }
}

# --- (b) alternate-optimizer refits of all 12 published fits. glmmTMB's default
# is nlminb; this re-runs each fit under optim/BFGS from the same start. If any
# published fit is sitting below an alternative optimum the way three
# paper_table fits were in 2026-08-31, this is where it shows.
ctl <- glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS"))
for (spec in c("A", "B", "C")) {
  for (m in c("M1", "M2", "M3", "M1matched")) {
    dv  <- if (spec == "C") "log_ratio" else "violations"
    useoff <- spec != "C"
    dd <- rows(dv, SAMPLE[[m]], useoff)
    off <- if (useoff) " + offset(log(inspections))" else ""
    form <- stats::as.formula(sprintf("%s ~ %s%s + (1 | state) + (1 | year)",
                                      dv, paste(RHS[[m]], collapse = " + "), off))
    zi <- if (spec == "B")
      stats::as.formula(paste("~", paste(c("1", RHS[[m]]), collapse = " + ")))
      else ~0
    fam <- if (spec == "C") gaussian else nbinom2
    key <- paste0("altopt__", spec, "__", m)
    cap <- fit_capturing(function()
      glmmTMB(form, ziformula = zi, family = fam, data = dd,
              control = ctl, REML = FALSE))
    if (inherits(cap$fit, "error")) {
      out[[key]] <- list(n_obs = nrow(dd), converged = FALSE,
                         message = paste(unique(cap$msgs), collapse = " | "),
                         error = conditionMessage(cap$fit))
    } else {
      out[[key]] <- pack(cap$fit, dd, cap$msgs)
    }
  }
}

write_json(out, args[2], auto_unbox = TRUE, digits = 12, na = "null")
cat("wrote", args[2], "\n")
'''

_R_CACHE = {}

# The TMB ABI-mismatch warning (glmmTMB built against TMB 1.9.19, 1.9.23
# installed) is EXPECTED on every Rscript call in this project and is not
# suppressed anywhere. Anything ELSE on stderr is a genuinely new problem and
# must not be able to hide behind it.
TMB_KNOWN_MARKERS = ('check_dep_version', 'TMB package version',
                     'package version mismatch', 'Warning message:',
                     're-install glmmTMB', 'restore original', 'reinstalling')


def run_r_gate():
    """Run the embedded R helper once; both [9] and [12] read its output."""
    if 'out' in _R_CACHE:
        return _R_CACHE['out']
    with tempfile.TemporaryDirectory() as tmp:
        rpath = os.path.join(tmp, 'ratio_gate.R')
        jpath = os.path.join(tmp, 'ratio_gate.json')
        with open(rpath, 'w') as fh:
            fh.write(GATE_R)
        proc = subprocess.run(['Rscript', rpath, PANEL, jpath],
                              capture_output=True, text=True, cwd=ROOT)
        fits = None
        if proc.returncode == 0 and os.path.exists(jpath):
            with open(jpath) as fh:
                fits = json.load(fh)
        _R_CACHE['out'] = (proc, fits)
    return _R_CACHE['out']


def _statsmodels_crossed(d, dv, rhs, reml):
    """`dv ~ rhs + (1|state) + (1|year)` in statsmodels, live, multi-optimizer.

    statsmodels has no crossed-RE syntax; the documented recipe is one constant
    group carrying two variance components, each an indicator basis over a
    grouping factor -- algebraically the model glmmTMB fits from
    `(1 | state) + (1 | year)`.

    Several optimizers are tried and any fit raising a FAILURE warning (or
    reporting converged = False) is DISCARDED; the highest-log-likelihood
    survivor is returned. This is the live-vs-live idiom of
    validate_count_models.py [2c], and it is load-bearing here: statsmodels'
    default lbfgs/bfgs path genuinely fails on the Spec C M3 surface and stops
    7e-2 loglik short, while powell and nm both reach the optimum glmmTMB
    reports. Taking the default would have produced a false failure.
    """
    import statsmodels.formula.api as smf
    d = d.dropna(subset=[dv] + rhs + ['state', 'year']).copy()
    d['_grp'] = 1
    vcf = {'state': '0 + C(state)', 'year': '0 + C(year)'}
    md = smf.mixedlm(f"{dv} ~ {' + '.join(rhs)}", d, groups='_grp', vc_formula=vcf)
    best, tried = None, []
    for meth in ('lbfgs', 'bfgs', 'powell', 'nm', 'cg'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                res = md.fit(reml=reml, method=meth)
            except Exception as exc:
                tried.append(f'{meth}: raised {type(exc).__name__}')
                continue
            msgs = [str(w.message) for w in caught]
        failed = any(mk in m.lower() for m in msgs for mk in SM_FAILURE_MARKERS)
        if failed or not getattr(res, 'converged', True):
            tried.append(f'{meth}: DISCARDED ({"; ".join(msgs)[:70]})')
            continue
        tried.append(f'{meth}: llf={res.llf:.5f}')
        if best is None or res.llf > best[1].llf:
            best = (meth, res)
    if best is None:
        return None, tried, len(d)
    meth, res = best
    return (dict(res.fe_params), dict(zip(md.exog_vc.names, res.vcomp)),
            res.scale, res.llf, meth), tried, len(d)


def validate_gaussian_roundtrip(skip_slow):
    section('9', 'Gaussian round-trip gate: the OFFSET spec and Spec C, '
                 'glmmTMB vs statsmodels')
    if skip_slow:
        skip('offset + Spec C Gaussian round-trip',
             '--skip-slow passed; this run is NOT a passing run')
        return
    r = load_results()
    p = panel()

    # --- (a) the offset gate. No existing validator in this project gates an
    # offset specification; validate_jafari_crossed.py [2] gates only the
    # crossed structure.
    proc, fits = run_r_gate()
    if fits is None:
        check('R gate ran', False,
              f'exit {proc.returncode}; stderr tail: {proc.stderr.strip()[-500:]}')
        return
    check('R gate ran', True)
    unexpected = [ln for ln in proc.stderr.splitlines()
                  if ln.strip() and not any(m in ln for m in TMB_KNOWN_MARKERS)]
    check('R stderr contains nothing beyond the known TMB ABI-mismatch warning',
          not unexpected, f'unexpected: {unexpected!r}')

    poff = p[p['inspections'] > 0].copy()
    poff['_lv_minus_off'] = poff['log_violations'] - np.log(poff['inspections'])
    for model in ('M1', 'M3'):
        key = f'gauss_offset__{model}'
        g = fits.get(key)
        if g is None or not g.get('converged'):
            check(f'{key}: glmmTMB offset fit present and converged', False,
                  f'record: {g}')
            continue
        check(f'{key}: glmmTMB offset fit converged', True)
        rhs = RHS[model]
        # Identity link: E[y] = Xb + offset  <=>  y - offset = Xb. statsmodels
        # MixedLM has no offset argument, so the offset is moved to the response
        # -- which makes this a genuine independent implementation of it.
        got, tried, n = _statsmodels_crossed(poff, '_lv_minus_off', rhs, reml=False)
        if got is None:
            check(f'{key}: statsmodels reference fit available', False,
                  f'all optimizers discarded: {tried}')
            continue
        params, vcomp, scale, llf, meth = got
        print(f'    (statsmodels reference: {meth}; tried {tried})')
        check(f'{key}: same analytic N across packages ({n})', g['n_obs'] == n,
              f"glmmTMB {g['n_obs']}, statsmodels {n}")
        check(f'{key}: rows are the Spec A/B {model} sample',
              n == EXPECTED_N[('A', model)][0],
              f'{n} vs {EXPECTED_N[("A", model)][0]}')
        for term, b in g['cond'].items():
            sm_name = 'Intercept' if term == '(Intercept)' else term
            check_close(f'{key}: b[{term}]', b, params.get(sm_name), 1e-3)
        check_var_close(f'{key}: sigma2_state', g['sigma2_state'],
                        vcomp.get('state'), scale_ref=scale)
        check_var_close(f'{key}: sigma2_year', g['sigma2_year'],
                        vcomp.get('year'), scale_ref=scale)
        check_close(f'{key}: sigma2_e', g['sigma2_e'], scale, 2e-2)

        # --- the offset must not be silently ignored. Same rows, same model,
        # response NOT shifted: this must DISagree, or `offset()` was a no-op.
        got2, _, _ = _statsmodels_crossed(poff, 'log_violations', rhs, reml=False)
        if got2 is None:
            skip(f'{key}: offset-is-not-a-no-op probe',
                 'the unshifted statsmodels reference fit did not converge')
        else:
            p2 = got2[0]
            gap = abs(g['cond']['(Intercept)'] - p2['Intercept'])
            check(f'{key}: the offset is actually applied (intercept moves '
                  f'{gap:.3f} when the offset is removed)', gap > 0.5,
                  f'gap {gap}')

    # --- (b) Spec C is already Gaussian, so it round-trips DIRECTLY against the
    # published record -- no refit by this validator, no scaffolding in between.
    for model in MODELS + BASELINES:
        k = record_key('C', model)
        v = r[k]
        got, tried, n = _statsmodels_crossed(
            p[p[SAMPLE_RHS[model]].notna().all(axis=1)], 'log_ratio',
            FORMULA_RHS[model], reml=False)
        if got is None:
            check(f'{k}: statsmodels reference fit available', False,
                  f'all optimizers discarded: {tried}')
            continue
        params, vcomp, scale, llf, meth = got
        print(f'    (statsmodels reference: {meth}; tried {tried})')
        check(f'{k}: same analytic N across packages ({n})', v['n_obs'] == n,
              f"glmmTMB {v['n_obs']}, statsmodels {n}")
        # ML on both sides, same model: the log-likelihoods must agree, and
        # glmmTMB must not be BELOW the best statsmodels optimum.
        check_close(f'{k}: log-likelihood', v['loglik'], llf, 1e-5, 'abs')
        check(f'{k}: glmmTMB is not below the best statsmodels optimum',
              v['loglik'] >= llf - 1e-5, f"glmmTMB {v['loglik']}, statsmodels {llf}")
        for term, c in v['cond'].items():
            sm_name = 'Intercept' if term == '(Intercept)' else term
            check_close(f'{k}: b[{term}]', c['b'], params.get(sm_name), 1e-3)
        check_var_close(f'{k}: sigma2_state', v['sigma2_state'],
                        vcomp.get('state'), scale_ref=scale)
        check_var_close(f'{k}: sigma2_year', v['sigma2_year'],
                        vcomp.get('year'), scale_ref=scale)
        check_close(f'{k}: sigma2_e', v['sigma2_e'], scale, 2e-2)


# ============================================================
# [10] MEMO CONSISTENCY
# ============================================================
MEMO_REQUIRED = [
    'GENERATED by `scripts/report_jafari_ratio.py`. Never hand-edit',
    # The sign convention. It exists because the two blocks were read backwards
    # in a team meeting and a finding was reported inverted. It must not
    # silently disappear.
    'A **positive** ZI coefficient means **more '
    'structural zeros**, and therefore **FEWER events**',
    'A **positive** coefficient means more structural zeros, hence FEWER '
    'events',
    'the **opposite** substantive direction'.replace('**', ''),
    'never read the ZI block as if it were a second count model',
    # Scale and comparability caveats.
    'log LINK scale',
    'This arm is not comparable to `docs/count_models_zinb.md`',
    'No COVID indicator is fit',
    'behaves as a state-only random intercept in practice',
]
MEMO_FORBIDDEN = [
    'TODO', 'TBD', 'FIXME', ' nan ', '| nan', 'None |', '| None',
    'inf |', 'delta_pct_matched =',
]


def _split_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def _stars(p):
    return ('***' if p < .001 else '**' if p < .01 else '*' if p < .05
            else '+' if p < .10 else '')


def validate_memo():
    section('10', 'Memo consistency: every printed number round-trips to the JSON')
    memo = open(MEMO).read()
    r = load_results()

    for pat in MEMO_REQUIRED:
        check(f'memo states: {pat[:64]!r}', pat in memo)
    for pat in MEMO_FORBIDDEN:
        check(f'memo free of {pat!r}', pat not in memo)

    lines = memo.splitlines()
    # --- section 3: the side-by-side table, parsed and checked against the JSON
    seen3 = set()
    for line in lines:
        if not line.startswith('| ') or '|' not in line[2:]:
            continue
        cells = _split_row(line)
        if len(cells) != 10 or cells[0] not in SPECS or cells[1] not in MODELS:
            continue
        spec, model = cells[0], cells[1]
        seen3.add((spec, model))
        v = r[record_key(spec, model)]
        base = r[record_key(spec, 'M1')]['sigma2_state']
        matched = r[record_key(spec, 'M1matched')]['sigma2_state']
        tag = f'section 3 row {spec}/{model}'
        check(f'{tag}: N', cells[2] == str(v['n_obs']), f'printed {cells[2]}')
        check(f'{tag}: states', cells[3] == str(v['n_states']), f'printed {cells[3]}')
        check(f'{tag}: zeros (rate)',
              cells[4] == f"{v['obs_zeros']} ({v['obs_zeros'] / v['n_obs']:.3f})",
              f'printed {cells[4]}')
        check(f'{tag}: sigma2_state', cells[5] == f"{v['sigma2_state']:.4f}",
              f'printed {cells[5]}, JSON {v["sigma2_state"]}')
        want_d = '--' if model == 'M1' else f'{_delta(base, v["sigma2_state"]):+.1f}%'
        want_m = '--' if model == 'M1' else f'{_delta(matched, v["sigma2_state"]):+.1f}%'
        check(f'{tag}: Delta vs M1', cells[6] == want_d,
              f'printed {cells[6]}, computed {want_d}')
        check(f'{tag}: Delta vs matched M1', cells[7] == want_m,
              f'printed {cells[7]}, computed {want_m}')
        check(f'{tag}: AIC', cells[8] == f"{v['aic']:.2f}",
              f'printed {cells[8]}, JSON {v["aic"]}')
        check(f'{tag}: df', cells[9] == str(v['df']), f'printed {cells[9]}')
    check('section 3 prints all 9 substantive rows',
          seen3 == {(s, m) for s in SPECS for m in MODELS},
          f'found {sorted(seen3)}')
    # The matched baselines back the last Delta column but must not appear as
    # rows of their own -- they are not competitors. Tested on TABLE ROWS, not
    # on the whole text: the memo legitimately names the record key
    # `*__M1matched` in the prose that explains the column.
    baseline_rows = [ln for ln in lines
                     if ln.startswith('| ')
                     and len(_split_row(ln)) >= 2
                     and _split_row(ln)[0] in SPECS
                     and _split_row(ln)[1].startswith('M1matched')]
    check('section 3 does NOT print the matched baselines as rows',
          not baseline_rows, f'rows: {baseline_rows}')
    check('the memo does explain where the matched baselines come from',
          '`*__M1matched`' in memo)

    # --- section 4: the A-vs-B zero-inflation table
    seen4 = set()
    for line in lines:
        cells = _split_row(line) if line.startswith('| ') else []
        if len(cells) != 8 or cells[0] not in MODELS:
            continue
        model = cells[0]
        a, b = r[record_key('A', model)], r[record_key('B', model)]
        seen4.add(model)
        tag = f'section 4 row {model}'
        check(f'{tag}: Spec A AIC', cells[1] == f"{a['aic']:.2f}", f'printed {cells[1]}')
        check(f'{tag}: Spec B AIC', cells[2] == f"{b['aic']:.2f}", f'printed {cells[2]}')
        check(f'{tag}: Delta AIC', cells[3] == f"**{b['aic'] - a['aic']:+.2f}**",
              f'printed {cells[3]}')
        check(f'{tag}: ZI tier is the full mirror',
              cells[4] == 'full mirror (every conditional predictor)'
              and b['zi_tier_reached'] == 'mirror', f'printed {cells[4]}')
        check(f'{tag}: ZI parameter count', cells[5] == str(b['df'] - a['df']),
              f'printed {cells[5]}')
        check(f'{tag}: Pr(structural zero)',
              cells[6] == f"{b['zi_prob_mean']:.4f}", f'printed {cells[6]}')
        check(f'{tag}: observed zero rate',
              cells[7] == f"{b['obs_zeros'] / b['n_obs']:.4f}", f'printed {cells[7]}')
    check('section 4 prints all 3 models', seen4 == set(MODELS), f'found {seen4}')

    # --- section 5: the Model-3 coefficient blocks, per spec and per block
    spec_now, block_now = None, None
    seen5 = set()
    for line in lines:
        if line.startswith('### Spec '):
            spec_now = line.split()[2]
            block_now = None
            continue
        if 'Conditional block' in line:
            block_now = 'cond'
            continue
        if 'Zero-inflation block' in line:
            block_now = 'zi'
            continue
        if not line.startswith('| ') or spec_now is None or block_now is None:
            continue
        cells = _split_row(line)
        if len(cells) != 6 or cells[0] in ('Term', '---'):
            continue
        v = r[record_key(spec_now, 'M3')]
        # Map the printed label back to the JSON term name.
        labels = {'Intercept': '(Intercept)', 'time': 'time', 'time²': 'time2',
                  'time³': 'time3',
                  'STAG $ per applicator (z)': 'SPEND_APP_z',
                  'STAG $ per farmworker (z)': 'SPEND_WORK_z',
                  'Labor intensity index 2017 (z)': 'lii_2017_z',
                  'H-2A per farmworker (z)': 'h2a_per_farmworker_z',
                  'DOL demand met % (z)': 'dol_demand_met_pct_z',
                  '% farm labor contractor (z)': 'pct_flc_z'}
        term = labels.get(cells[0])
        if term is None:
            check(f'section 5 {spec_now}/{block_now}: row label recognised',
                  False, f'unknown label {cells[0]!r}')
            continue
        c = (v.get(block_now) or {}).get(term)
        if c is None:
            check(f'section 5 {spec_now}/{block_now}: {term} exists in the JSON',
                  False)
            continue
        seen5.add((spec_now, block_now, term))
        tag = f'section 5 {spec_now}/{block_now}/{term}'
        check(f'{tag}: b (+stars)', cells[1] == f"{c['b']:+.4f}{_stars(c['p'])}",
              f"printed {cells[1]}, JSON {c['b']:+.4f}{_stars(c['p'])}")
        check(f'{tag}: SE', cells[2] == f"({c['se']:.4f})", f'printed {cells[2]}')
        check(f'{tag}: z', cells[3] == f"{c['z']:.2f}", f'printed {cells[3]}')
        check(f'{tag}: p', cells[4] == f"{c['p']:.4f}", f'printed {cells[4]}')
        want_exp = (f"{math.exp(c['b']):.4g}" if block_now == 'zi'
                    else f"{math.exp(c['b']):.4f}")
        check(f'{tag}: exp(b)', cells[5] == want_exp, f'printed {cells[5]}')
    want5 = set()
    for spec in SPECS:
        v = r[record_key(spec, 'M3')]
        for blk in ('cond', 'zi'):
            for term in (v.get(blk) or {}):
                want5.add((spec, blk, term))
    check('section 5 prints every Model-3 coefficient in the JSON',
          seen5 == want5, f'missing {sorted(want5 - seen5)}; '
                          f'extra {sorted(seen5 - want5)}')

    # --- the narrative numbers that are not in a table
    d = r['__offset_derivation']
    check('memo prints the panel shape',
          f"The panel has {d['panel_rows']} rows, {d['panel_states']} states"
          in memo.replace('\n  ', ' ').replace('\n', ' '))
    for s, y in sorted(EXPECTED_OFFSET_DROPS):
        check(f'memo names the dropped row {s}-{y}',
              f'| {s} | {y} |' in memo)
    a1, m1 = r['A__M1'], r['A__M1matched']
    check("memo prints Spec A's matched baseline sigma2_state",
          f"{m1['sigma2_state']:.4f}" in memo)
    check("memo prints Spec A's Model-1 sigma2_state",
          f"{a1['sigma2_state']:.4f}" in memo)
    # sigma2_year: the "empty year dimension" claim, re-derived.
    ybound = [(s, m) for s in SPECS for m in MODELS
              if r[record_key(s, m)]['sigma2_year'] < 1e-4]
    worst = max(((s, m) for s in SPECS for m in MODELS),
                key=lambda sm: r[record_key(*sm)]['sigma2_year'])
    check(f'memo states sigma2_year is at the boundary in {len(ybound)} of 9 fits',
          f'in {len(ybound)} of 9 substantive fits' in memo
          or (len(ybound) == 9 and 'in all 9 substantive fits' in memo),
          f'{len(ybound)} boundary fits')
    wv = r[record_key(*worst)]['sigma2_year']
    check(f'memo prints the largest sigma2_year ({wv:.2e}, {worst[0]}/{worst[1]})',
          f'{wv:.2e}' in memo)


# ============================================================
# [11] FROZEN ARTIFACTS
# ============================================================
# Guarded in BOTH arms: these belong to arms that predate this branch.
FROZEN_HISTORY = [
    'scripts/count_models_zinb.R', 'scripts/report_count_models.py',
    'scripts/validate_count_models.py', 'scripts/build_count_model_panel.py',
    'scripts/build_count_model_memo_evidence.py',
    'data/generated/count_model_results.json',
    'data/generated/count_model_panel_2019.csv',
    'data/generated/count_model_panel_2021.csv',
    'docs/count_models_zinb.md',
    'scripts/nb2_stepwise_models.R', 'scripts/report_nb2_stepwise.py',
    'scripts/validate_nb2_stepwise.py', 'scripts/nb2_crosscheck_statsmodels.py',
    'data/generated/nb2_stepwise_results.json',
    'data/generated/nb2_stepwise_variance.csv',
    'data/generated/nb2_stepwise_coefficients.csv',
    'data/generated/nb2_stepwise_crosscheck.json',
    'docs/nb2_stepwise_models.md',
]
# Guarded in the WORKING TREE ONLY. The crossed arm was legitimately edited on
# THIS branch (63f678c, a presentation fix), so a merge-base..HEAD guard on it
# would fail on authorized work; what is still assertable is that nothing has
# touched it since.
FROZEN_WORKTREE_ONLY = [
    'scripts/jafari_crossed_models.R', 'scripts/report_jafari_crossed.py',
    'scripts/validate_jafari_crossed.py',
    'data/generated/jafari_crossed_results.json',
    'data/generated/jafari_crossed_gaussian.json',
    'data/generated/jafari_crossed_selection.csv',
    'data/generated/jafari_crossed_variance.csv',
    'data/generated/jafari_crossed_coefficients.csv',
    'data/generated/jafari_crossed_zi_strength.csv',
    'docs/jafari_crossed_models.md',
]

# NOT guarded by any git diff, deliberately. The stepwise arm is owned by a
# CONCURRENTLY RUNNING agent on this same branch, so `git diff HEAD` over its
# files answers "has anyone changed these?" -- which is not the question. The
# question is "did THIS arm change them", and a diff cannot attribute
# authorship: with the other agent mid-edit the guard fails on work that is
# authorized and not ours. (That is exactly what happened on the first run of
# this section, and the assertion, not the arm, was what was wrong.)
#
# The assertable version is STATIC: no script belonging to this arm may so much
# as name a stepwise path or CLAUDE.md. That cannot false-positive on another
# agent's work, and it fails if this arm ever grows a write to them.
CONCURRENT_OWNED_PREFIXES = ('scripts/jafari_stepwise', 'scripts/report_jafari_stepwise',
                             'scripts/validate_jafari_stepwise',
                             'data/generated/jafari_stepwise',
                             'docs/jafari_stepwise')
THIS_ARM_PRODUCERS = ['scripts/jafari_ratio_models.R',
                      'scripts/report_jafari_ratio.py']
THIS_VALIDATOR = 'scripts/validate_jafari_ratio.py'


def validate_frozen():
    section('11', 'Frozen artifacts: no other arm was modified')
    # --- (a) working tree vs HEAD. Catches uncommitted edits, including any
    # this run might have made.
    proc = subprocess.run(
        ['git', 'diff', 'HEAD', '--name-only', '--']
        + FROZEN_HISTORY + FROZEN_WORKTREE_ONLY,
        capture_output=True, text=True, cwd=ROOT)
    changed = [ln for ln in proc.stdout.split('\n') if ln.strip()]
    check('no count-model / NB2 / crossed artifact modified in the working tree',
          not changed, f'modified: {changed}')

    # --- the stepwise arm: static ownership guard, not a git diff. See the
    # comment on CONCURRENT_OWNED_PREFIXES for why a diff is the wrong test.
    # WRITES, not mentions. The arm's scripts legitimately NAME other arms'
    # memos (the ratio memo cross-references count_models_zinb.md,
    # nb2_stepwise_models.md and jafari_crossed_models.md, and the R script
    # cites CLAUDE.md's recorded claim about the 7 dropped rows) -- an earlier
    # version of this guard forbade the mention and failed on those legitimate
    # citations. What must be true is that neither producer writes anything
    # outside its own two artifacts.
    rep = ROOT + 'scripts/report_jafari_ratio.py'
    check('report_jafari_ratio.py exists', os.path.exists(rep))
    if os.path.exists(rep):
        rtext = open(rep).read()
        rwrites = set(re.findall(r"open\(\s*(.+?)\s*,\s*'w'\s*\)", rtext))
        check("report_jafari_ratio.py writes exactly one file, its own memo",
              rwrites == {"DOCS + 'jafari_ratio_models.md'"},
              f'write targets: {sorted(rwrites)}')
    rsrc = ROOT + 'scripts/jafari_ratio_models.R'
    check('jafari_ratio_models.R exists', os.path.exists(rsrc))
    if os.path.exists(rsrc):
        rrtext = open(rsrc).read()
        rrwrites = set(re.findall(
            r"(?:write_json|write\.csv|writeLines|saveRDS|png|pdf)\(\s*[^,)]*,\s*([A-Za-z_][\w.]*)",
            rrtext))
        check('jafari_ratio_models.R writes only to its <out_json> argument',
              rrwrites <= {'out_json'}, f'write targets: {sorted(rrwrites)}')
        check('jafari_ratio_models.R hardcodes no output path under data/ or docs/',
              not re.search(r"['\"][^'\"]*(?:data/generated|docs)/[^'\"]*['\"]",
                            rrtext.replace('data/generated/count_model_panel', 'X')),
              'a literal output path is present')
    # The validator itself: it is read-only over the project by construction,
    # and this pins that. Its ONLY write-mode open is the temp R scaffold, whose
    # target is built with os.path.join inside a TemporaryDirectory -- so it
    # cannot write a project file even by accident, whatever strings appear in
    # its comments.
    vtext = open(ROOT + THIS_VALIDATOR).read()
    writes = set(re.findall(r"open\(\s*([^,()]+?)\s*,\s*'w'\s*\)", vtext))
    check(f'{THIS_VALIDATOR}: the only write-mode open is the temp R scaffold',
          writes == {'rpath'}, f'write targets: {sorted(writes)}')
    check(f'{THIS_VALIDATOR}: that target lives in a TemporaryDirectory',
          "rpath = os.path.join(tmp, 'ratio_gate.R')" in vtext
          and 'with tempfile.TemporaryDirectory() as tmp:' in vtext)
    # And the empirical version of the same claim: validating must not
    # regenerate the thing being validated. If a run of this file rewrote the
    # arm's JSON or its memo, every "round-trips to the JSON" check above would
    # be comparing an artifact to itself.
    own = subprocess.run(
        ['git', 'diff', 'HEAD', '--name-only', '--',
         'data/generated/jafari_ratio_results.json', 'docs/jafari_ratio_models.md',
         'scripts/jafari_ratio_models.R', 'scripts/report_jafari_ratio.py'],
        capture_output=True, text=True, cwd=ROOT)
    own_changed = [ln for ln in own.stdout.split('\n') if ln.strip()]
    check('this arm\'s own artifacts are unchanged by validating them',
          not own_changed, f'modified: {own_changed}')
    # Informational, so the reader is not left thinking nothing moved: report
    # the stepwise arm's working-tree state and say plainly who owns it.
    proc_sw = subprocess.run(['git', 'diff', 'HEAD', '--name-only'],
                             capture_output=True, text=True, cwd=ROOT)
    sw = [ln for ln in proc_sw.stdout.split('\n')
          if ln.strip().startswith(CONCURRENT_OWNED_PREFIXES)]
    if sw:
        skip('stepwise-arm working-tree diff',
             f'{sw} differ from HEAD -- owned by the concurrently running '
             f'stepwise agent, not written by this arm (guarded statically above)')

    proc = subprocess.run(['git', 'diff', 'HEAD', '--name-only'],
                          capture_output=True, text=True, cwd=ROOT)
    touched = [ln for ln in proc.stdout.split('\n') if ln.strip()]
    bad = [f for f in touched if f.endswith('.docx') or 'paper_table' in f]
    check('no .docx and no paper_table artifact modified (working tree vs HEAD)',
          not bad, f'modified: {bad}')

    # --- (b) committed history. `git diff HEAD` is blind to anything already
    # committed on this branch, so diff against the merge-base with main.
    # KNOWN LIMITATION (same as validate_nb2_stepwise.py [8] and
    # validate_jafari_crossed.py [10]): on `main` the merge-base IS HEAD, the
    # range is empty and this arm passes vacuously. It verifies "this branch did
    # not touch them", which is the right question while a branch is in flight
    # and the wrong one afterwards.
    mb = subprocess.run(['git', 'merge-base', 'main', 'HEAD'],
                        capture_output=True, text=True, cwd=ROOT)
    merge_base = mb.stdout.strip()
    head = subprocess.run(['git', 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, cwd=ROOT).stdout.strip()
    ok = mb.returncode == 0 and bool(merge_base)
    check('git merge-base main HEAD resolves (required for the history guard)',
          ok, f'returncode {mb.returncode}; stderr {mb.stderr!r}')
    if not ok:
        check('no frozen artifact modified (merge-base..HEAD)', False,
              'merge-base could not be resolved -- guard did not run')
        return
    if merge_base == head:
        skip('committed-history arm',
             'merge-base == HEAD (on the default branch); this arm is vacuous here')
        return
    proc = subprocess.run(['git', 'diff', f'{merge_base}..HEAD', '--name-only',
                           '--'] + FROZEN_HISTORY,
                          capture_output=True, text=True, cwd=ROOT)
    hist = [ln for ln in proc.stdout.split('\n') if ln.strip()]
    check('no count-model or NB2 artifact modified in this branch\'s commits',
          not hist, f'modified since {merge_base[:8]}: {hist}')
    proc = subprocess.run(['git', 'diff', f'{merge_base}..HEAD', '--name-only'],
                          capture_output=True, text=True, cwd=ROOT)
    hist_all = [ln for ln in proc.stdout.split('\n') if ln.strip()]
    bad_hist = [f for f in hist_all if f.endswith('.docx') or 'paper_table' in f]
    check('no .docx and no paper_table artifact modified in this branch\'s commits',
          not bad_hist, f'modified since {merge_base[:8]}: {bad_hist}')


# ============================================================
# [12] OPTIMIZER ROBUSTNESS
# ============================================================
# This project's signature defect is a fit that reported success while sitting
# below the optimum (paper_table_models_2021.py, lbfgs, |grad| = 76.8, 8.1
# loglik short -- and the 2019 arm's inspections M1, 19.8 short). glmmTMB reports
# pd_hess and a convergence code but runs one optimizer, so "converged" here
# means "converged from nlminb's path". Refitting every published fit under
# optim/BFGS is the cheap check that no published number is a local optimum.
ALTOPT_LOGLIK_TOL = 0.05


def validate_optimizer_robustness(skip_slow):
    section('12', 'Optimizer robustness: no published fit sits below an '
                  'alternate-optimizer refit')
    if skip_slow:
        skip('alternate-optimizer refits of all 12 fits',
             '--skip-slow passed; this run is NOT a passing run')
        return
    proc, fits = run_r_gate()
    if fits is None:
        check('R gate ran', False, f'exit {proc.returncode}')
        return
    r = load_results()
    for spec in SPECS:
        for model in MODELS + BASELINES:
            k = record_key(spec, model)
            alt = fits.get(f'altopt__{spec}__{model}')
            v = r[k]
            if alt is None:
                check(f'{k}: alternate-optimizer refit present', False)
                continue
            if not alt.get('converged'):
                # A failed ALTERNATIVE carries no weight against the published
                # fit; recorded as a SKIP, never as evidence either way. The
                # captured R warnings are printed rather than swallowed. Spec C
                # is the only spec that lands here, and it is the one spec with
                # an INDEPENDENT optimum check already: [9] round-trips every
                # Spec C fit against a multi-optimizer statsmodels reference and
                # requires glmmTMB not to sit below it.
                extra = ' independently covered by [9]' if spec == 'C' else ''
                skip(f'{k}: alternate-optimizer comparison',
                     f"optim/BFGS refit did not converge "
                     f"({alt.get('error') or alt.get('message') or 'no pd Hessian'})"
                     f"{extra}")
                continue
            check(f'{k}: refit uses the same analytic sample ({v["n_obs"]})',
                  alt['n_obs'] == v['n_obs'], f"refit {alt['n_obs']}")
            gain = alt['loglik'] - v['loglik']
            check(f'{k}: optim/BFGS does not beat the published fit '
                  f'(delta loglik {gain:+.4f} <= {ALTOPT_LOGLIK_TOL})',
                  gain <= ALTOPT_LOGLIK_TOL,
                  f"published {v['loglik']}, optim/BFGS {alt['loglik']}")
            check_close(f'{k}: the two optimizers agree on the log-likelihood',
                        alt['loglik'], v['loglik'], ALTOPT_LOGLIK_TOL, 'abs')
            check_close(f'{k}: the two optimizers agree on sigma2_state',
                        alt['sigma2_state'], v['sigma2_state'], 5e-2, 'rel')


# ============================================================
# [13] CROSS-SOFTWARE CHECK OF THE OFFSET ON THE COUNT SCALE
# ============================================================
def validate_count_offset_crosscheck(skip_slow):
    section('13', 'Cross-software check: NB2 with offset in statsmodels '
                  '(state fixed effects)')
    if skip_slow:
        skip('statsmodels NB2-with-offset cross-check',
             '--skip-slow passed; this run is NOT a passing run')
        return
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import NegativeBinomialP

    r = load_results()
    p = panel()
    # SCOPE, stated up front. State dummies absorb every Level-2 (state-
    # invariant) covariate by construction, so SPEND_*, lii_2017_z and the H-2A
    # block are NOT identified here and are not checked -- the same limitation
    # nb2_crosscheck_statsmodels.py documents. YEAR dummies would additionally
    # absorb time/time2/time3 (they are functions of year alone), which would
    # leave nothing identified at all, so the year dimension is dropped from
    # the cross-check. That is defensible precisely because sigma2_year is at
    # the boundary in every fit of this arm. What IS checked is the Level-1
    # cubic time trend under the offset, in a different package, with a
    # different treatment of the state effects.
    print('    (Level-1 time terms only: state dummies absorb the Level-2 '
          'covariates; year dummies would absorb the cubic)')
    for model in ('M1', 'M3'):
        v = r[record_key('A', model)]
        rhs = RHS[model]
        d = p.dropna(subset=['violations'] + rhs + ['state', 'year'])
        d = d[d['inspections'] > 0]
        X = sm.add_constant(pd.concat([
            d[rhs].reset_index(drop=True),
            pd.get_dummies(d['state'], prefix='st',
                           drop_first=True).reset_index(drop=True),
        ], axis=1).astype(float))
        off = np.log(d['inspections'].to_numpy(dtype=float))
        y = d['violations'].to_numpy(dtype=float)
        msgs = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                # Poisson start values: from statsmodels' own defaults the M1
                # surface diverges (|b| ~ 1e4). This is a starting-value
                # problem in the cross-check, not a finding about the arm.
                start = np.append(
                    sm.GLM(y, X.values, family=sm.families.Poisson(),
                           offset=off).fit().params, 1.0)
                res = NegativeBinomialP(y, X.values, p=2, offset=off).fit(
                    start_params=start, disp=0, maxiter=2000)
                msgs = [str(w.message) for w in caught]
            conv = bool(res.mle_retvals.get('converged'))
        except Exception as exc:
            skip(f'A/{model}: NB2-with-offset cross-check',
                 f'statsmodels raised {type(exc).__name__}: {exc}')
            continue
        if not conv:
            skip(f'A/{model}: NB2-with-offset cross-check',
                 'statsmodels fixed-effects NB2 did not converge; carries no '
                 'interpretive weight either way')
            continue
        check(f'A/{model}: statsmodels NB2-with-offset converged', True)
        if msgs:
            print(f'    (warnings: {"; ".join(m[:70] for m in msgs)})')
        names = list(X.columns)
        for term in CUBIC:
            got = float(res.params[names.index(term)])
            want = v['cond'][term]['b']
            # Deliberately loose: fixed effects and random effects are different
            # estimands, and the two packages optimise different surfaces. This
            # detects a sign error, a misplaced offset or an order-of-magnitude
            # slip -- not a numerical discrepancy.
            check(f'A/{model}: b[{term}] same sign across packages',
                  np.sign(got) == np.sign(want), f'statsmodels {got}, glmmTMB {want}')
            check(f'A/{model}: b[{term}] within 0.05 of the glmmTMB estimate '
                  f'(|{got:.4f} - {want:.4f}| = {abs(got - want):.4f})',
                  abs(got - want) < 0.05,
                  f'statsmodels {got}, glmmTMB {want}')
        # The offset is only legal because the sample has no zero denominators.
        check(f'A/{model}: the cross-check sample has no zero inspections',
              bool((d['inspections'] > 0).all()))
        check(f'A/{model}: the cross-check sample is the Spec A {model} sample '
              f'({v["n_obs"]})', len(d) == v['n_obs'], f'got {len(d)}')


# ============================================================
def main():
    skip_slow = '--skip-slow' in sys.argv
    validate_precondition()
    validate_records()
    validate_offset_derivation()
    validate_missing_violations()
    validate_samples()
    validate_variance()
    validate_aic_discipline()
    validate_zi_probability()
    validate_convergence()
    validate_gaussian_roundtrip(skip_slow)
    validate_memo()
    validate_frozen()
    validate_optimizer_robustness(skip_slow)
    validate_count_offset_crosscheck(skip_slow)

    print('\n' + '=' * 82)
    print('PER-SECTION BREAKDOWN')
    print('The TOTAL below must never be quoted on its own: section [1] is a '
          'per-record loop\nover 13 records, not 13 independent findings, and a '
          'SKIP line marks every bypassed\ngated block. Read the breakdown.')
    print('=' * 82)
    print(f"{'SECTION':<64}{'PASS':>6}{'FAIL':>6}{'SKIP':>6}")
    for s in SECTIONS:
        print(f"{s['name']:<64}{s['pass']:>6}{s['fail']:>6}{s['skip']:>6}")
    tp = sum(s['pass'] for s in SECTIONS)
    tf = sum(s['fail'] for s in SECTIONS)
    ts = sum(s['skip'] for s in SECTIONS)
    print('-' * 82)
    print(f"{'TOTAL':<64}{tp:>6}{tf:>6}{ts:>6}")
    if skip_slow:
        print('\n--skip-slow was passed: sections [9], [12] and [13] did not '
              'run. This is NOT a passing run.')
    if FAILURES:
        print(f'\n{len(FAILURES)} FAILURE(S):')
        for f in FAILURES:
            print(f'  - {f}')
        sys.exit(1)
    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
