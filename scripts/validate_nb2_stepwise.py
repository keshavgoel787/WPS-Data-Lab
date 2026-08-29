"""
Validation gate for the NB2-only stepwise count models.

Not a unit-test suite -- an analysis artifact, matching the convention of
scripts/validate_count_models.py. Each check proves either that a step
reproduces something already known or that an invariant the spec depends on
holds. Run after any change to this arm.

Run: python3 scripts/validate_nb2_stepwise.py   (exits 1 on any failure)
     python3 scripts/validate_nb2_stepwise.py --skip-precondition
         Skips section [0] only. FOR ITERATION DURING DEVELOPMENT ONLY -- a run
         with [0] skipped is not a passing run, and the SKIP line says so.

ON READING THE OUTPUT: do not quote the total on its own. main() prints a
per-section breakdown and an explicit SKIP line for every bypassed block, so a
run in which a block silently vanished is visibly different from one in which
it ran.

Spec: docs/superpowers/specs/2026-08-28-nb2-stepwise-design.md
"""
import json
import subprocess
import sys

GEN = '/Users/keshavgoel/Research/data/generated/'
SCRIPTS = '/Users/keshavgoel/Research/scripts/'
ROOT = '/Users/keshavgoel/Research/'

FAILURES = []
SECTIONS = []
_CUR = None

CELLS = ('insp_2021', 'viol_cov_2021')
MODELS = ('M1', 'M2', 'M3')


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


def check_close(label, got, want, tol, kind='abs'):
    if got is None or want is None:
        check(f"{label} (both values present)", False, f"got {got!r}, want {want!r}")
        return
    delta = abs(got - want) if kind == 'abs' else abs(got - want) / abs(want)
    check(f"{label} ({kind} delta {delta:.2e} <= {tol:.0e})", delta <= tol,
          f"got {got!r}, want {want!r}")


def load_results():
    with open(GEN + 'nb2_stepwise_results.json') as fh:
        return json.load(fh)


# ============================================================
# [0] PRECONDITION -- the Gaussian round-trip gate
# ============================================================
def validate_precondition(skip_precondition):
    section('0', 'Precondition: the existing Gaussian round-trip gate passes')
    if skip_precondition:
        skip('validate_count_models.py [2a]/[2b]/[2c]',
             '--skip-precondition passed; this run is NOT a passing run')
        return
    proc = subprocess.run([sys.executable, SCRIPTS + 'validate_count_models.py'],
                          capture_output=True, text=True, cwd=ROOT)
    check('validate_count_models.py exits 0', proc.returncode == 0,
          f'exit {proc.returncode}; last stderr line: '
          f'{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "(none)"}')
    # The Gaussian sections specifically: family-agnostic, so their verdict
    # carries over to this arm unchanged (spec 5.1). Parsed out of the
    # per-section breakdown rather than trusting the exit code alone.
    for tag in ('[2a]', '[2b]', '[2c]'):
        rows = [ln for ln in proc.stdout.splitlines() if ln.startswith(tag)]
        # The breakdown table repeats the section name with PASS/FAIL/SKIP columns.
        tail = [ln for ln in rows if ln.rstrip()[-1:].isdigit()]
        ok = bool(tail) and int(tail[-1].split()[-2]) == 0 and int(tail[-1].split()[-3]) > 0
        check(f'Gaussian gate section {tag}: >0 PASS and 0 FAIL', ok,
              f'breakdown row: {tail[-1] if tail else "(not found)"}')


# ============================================================
# [1] PANEL IDENTITY
# ============================================================
def validate_panel():
    section('1', 'Panel identity (count_model_panel_2021.csv, unmodified)')
    import pandas as pd
    p = pd.read_csv(GEN + 'count_model_panel_2021.csv')
    check('N rows == 539', len(p) == 539, f'got {len(p)}')
    check('N states == 49', p['state'].nunique() == 49, f"got {p['state'].nunique()}")
    check('Wyoming absent (WPS view has no WY row)', 'Wyoming' not in set(p['state']))
    check('violations zeros == 93', int((p['violations'] == 0).sum()) == 93,
          f"got {int((p['violations'] == 0).sum())}")
    check('inspections zeros == 7', int((p['inspections'] == 0).sum()) == 7,
          f"got {int((p['inspections'] == 0).sum())}")
    check('time == year - 2017 throughout', bool((p['time'] == p['year'] - 2017).all()))
    for col in ('SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
                'h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z'):
        check(f'column present: {col}', col in p.columns)


# ============================================================
# [2] FIT INTEGRITY -- the six substantive fits
# ============================================================
def validate_fits():
    section('2', 'Fit integrity: six substantive nbinom2 fits at (1 + time | state)')
    res = load_results()
    for cell in CELLS:
        for model in MODELS:
            key = f'{cell}__{model}__nbinom2__rs'
            if key not in res:
                check(f'{key} present', False, 'missing from results JSON')
                continue
            r = res[key]
            check(f'{key} converged', bool(r['converged']), r.get('message', ''))
            if not r.get('converged'):
                # M5: a failed fit lands in the R error branch carrying only
                # 'converged'/'message' -- none of the fields the rest of this
                # block indexes directly. Report the failure and move on
                # instead of crashing the whole validator run on a KeyError.
                continue
            check(f'{key} positive-definite Hessian', bool(r['pd_hess']))
            check(f'{key} conv_code == 0', r['conv_code'] == 0, f"got {r['conv_code']}")
            check(f'{key} re_tier == rs (no fallback)', r['re_tier'] == 'rs',
                  f"got {r['re_tier']}")
            check(f'{key} re_used == (1 + time | state)',
                  r['re_used'] == '(1 + time | state)', f"got {r['re_used']}")
            check(f'{key} family_name == nbinom2', r['family_name'] == 'nbinom2')
            check(f'{key} dispersion (theta) finite and > 0',
                  r['dispersion'] is not None and r['dispersion'] > 0)
            check(f'{key} sigma2_u1 present (random slope was actually fit)',
                  r['sigma2_u1'] is not None and r['sigma2_u1'] > 0)
            check(f'{key} all SEs finite',
                  all(c['se'] is not None and c['se'] == c['se'] and c['se'] < float('inf')
                      for c in r['cond'].values()))
            check(f'{key} mu_fixed finite and > 0',
                  r['mu_fixed'] is not None and r['mu_fixed'] > 0)
    # Sample sizes: M1 keeps all 49 states; M2/M3 lose AK/RI/VT to the BLS
    # applicator series, which is why the spec reports two Delta baselines.
    for cell in CELLS:
        m1, m2, m3 = (res.get(f'{cell}__{m}__nbinom2__rs', {}) for m in MODELS)
        check(f'{cell}: M1 n_states == 49', m1.get('n_states') == 49,
              f"got {m1.get('n_states')}")
        check(f'{cell}: M2 n_states == 46', m2.get('n_states') == 46,
              f"got {m2.get('n_states')}")
        check(f'{cell}: M3 n_states == 46', m3.get('n_states') == 46,
              f"got {m3.get('n_states')}")
        check(f'{cell}: M2 and M3 share one analytic sample',
              m2.get('n_obs') == m3.get('n_obs'),
              f"M2 {m2.get('n_obs')} vs M3 {m3.get('n_obs')}")
    # Fixed-effect terms, exactly as the spec's build-up specifies.
    m2_add = ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z']
    m3_add = m2_add + ['h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']
    for cell in CELLS:
        base = ['time', 'time2', 'time3']
        if cell == 'viol_cov_2021':
            base = ['log_inspections'] + base
        for model, extra in (('M1', []), ('M2', m2_add), ('M3', m3_add)):
            r = res.get(f'{cell}__{model}__nbinom2__rs')
            if r is None or not r.get('converged'):
                continue
            want = {'(Intercept)'} | set(base) | set(extra)
            check(f'{cell} {model}: fixed-effect terms exactly as specified',
                  set(r['cond']) == want,
                  f"extra {set(r['cond']) - want}, missing {want - set(r['cond'])}")
            check(f'{cell} {model}: no interaction terms',
                  not any(':' in t for t in r['cond']))
    # M3: the sigma2_e-is-null invariant is a property of the FAMILY (a count
    # model has no residual variance), so it applies to every record this arm
    # writes -- not just the six __rs fits checked above. Extend it to all 16
    # (the __ri series, both __M1matched fits, and both __M3covid fits).
    for key, r in res.items():
        if not r.get('converged'):
            continue
        check(f'{key} sigma2_e is null (count family has no residual variance)',
              r.get('sigma2_e') is None, f"got {r.get('sigma2_e')!r}")


# ============================================================
# [3] OVERLAP REGRESSION against the frozen ZINB ladder
# ============================================================
def validate_overlap():
    section('3', 'Overlap regression: four fits already exist in the frozen ladder')
    res = load_results()
    with open(GEN + 'count_model_results.json') as fh:
        old = json.load(fh)
    # Both arms estimate by ML (count_models_zinb.R:62 defaults REML = FALSE)
    # on identical rows and formulas, so these should agree to optimizer noise,
    # not merely to a loose tolerance.
    for cell in CELLS:
        for model in MODELS:
            new_key = f'{cell}__{model}__nbinom2__rs'
            old_key = f'{cell}__{model}__nbinom2'
            if old_key not in old:
                skip(f'{new_key} vs frozen ladder',
                     'no counterpart: the ladder fits M2 only under the selected '
                     'family, and nbinom2 is neither cell\'s selected family')
                continue
            a, b = res.get(new_key), old[old_key]
            if a is None:
                check(f'{new_key} present', False, 'missing from results JSON')
                continue
            if not a.get('converged'):
                # M5: a failed fit has no 'n_obs'/'aic'/'cond' to compare.
                check(f'{new_key}: converged (required for the overlap check)',
                      False, a.get('message', ''))
                continue
            check(f'{new_key}: n_obs matches ladder', a['n_obs'] == b['n_obs'],
                  f"{a['n_obs']} vs {b['n_obs']}")
            check(f'{new_key}: n_states matches ladder',
                  a['n_states'] == b['n_states'], f"{a['n_states']} vs {b['n_states']}")
            check_close(f'{new_key}: AIC', a['aic'], b['aic'], 1e-6, 'rel')
            check_close(f'{new_key}: sigma2_u0', a['sigma2_u0'], b['sigma2_u0'],
                        1e-4, 'rel')
            check_close(f'{new_key}: dispersion (theta)', a['dispersion'],
                        b['dispersion'], 1e-4, 'rel')
            for term, co in b['cond'].items():
                check_close(f'{new_key}: b[{term}]', a['cond'][term]['b'],
                            co['b'], 1e-4, 'abs')


# ============================================================
# [4] RANDOM-INTERCEPT SERIES for the Delta sigma^2_u0 block
# ============================================================
def validate_ri_series():
    section('4', 'Random-intercept-only series and the matched Model-1 baseline')
    res = load_results()
    for cell in CELLS:
        for model in MODELS:
            key = f'{cell}__{model}__nbinom2__ri'
            r = res.get(key)
            if r is None:
                check(f'{key} present', False, 'missing from results JSON')
                continue
            check(f'{key} converged', bool(r['converged']), r.get('message', ''))
            if not r.get('converged'):
                continue
            check(f'{key} re_tier == ri', r['re_tier'] == 'ri', f"got {r['re_tier']}")
            check(f'{key} re_used == (1 | state)', r['re_used'] == '(1 | state)',
                  f"got {r['re_used']}")
            check(f'{key} has NO random slope', r['sigma2_u1'] is None,
                  f"got sigma2_u1 = {r['sigma2_u1']!r}")
            check(f'{key} has no intercept-slope covariance',
                  r['sigma_u01'] is None, f"got {r['sigma_u01']!r}")
            check(f'{key} sigma2_u0 finite and > 0',
                  r['sigma2_u0'] is not None and r['sigma2_u0'] > 0)
            # The RI refit must sit on EXACTLY the rows of its random-slope
            # twin. If it did not, the reduction would confound a variance
            # change with a sample change.
            rs = res.get(f'{cell}__{model}__nbinom2__rs', {})
            check(f'{key} shares its random-slope twin\'s sample',
                  r['n_obs'] == rs.get('n_obs') and r['n_states'] == rs.get('n_states'),
                  f"ri {r['n_obs']}/{r['n_states']} vs rs "
                  f"{rs.get('n_obs')}/{rs.get('n_states')}")
            # Same fixed effects as the twin -- only the RE structure differs.
            check(f'{key} fixed effects identical to its random-slope twin',
                  set(r['cond']) == set(rs.get('cond', {})),
                  f"ri {sorted(r['cond'])} vs rs {sorted(rs.get('cond', {}))}")
    # The matched baseline: Model 1's formula on Model 3's rows.
    for cell in CELLS:
        key = f'{cell}__M1matched__nbinom2__ri'
        r = res.get(key)
        if r is None:
            check(f'{key} present', False, 'missing from results JSON')
            continue
        m1 = res.get(f'{cell}__M1__nbinom2__ri', {})
        m3 = res.get(f'{cell}__M3__nbinom2__ri', {})
        check(f'{key} converged', bool(r['converged']), r.get('message', ''))
        if not r.get('converged'):
            continue
        check(f'{key} re_tier == ri', r['re_tier'] == 'ri')
        check(f'{key} records the sample it was matched to',
              r.get('sample_model') == 'M3', f"got {r.get('sample_model')!r}")
        check(f'{key} carries Model 1 fixed effects, not Model 3',
              set(r['cond']) == set(m1.get('cond', {})),
              f"got {sorted(r['cond'])}")
        check(f'{key} sits on the Model-3 sample',
              r['n_obs'] == m3.get('n_obs') and r['n_states'] == m3.get('n_states'),
              f"matched {r['n_obs']}/{r['n_states']} vs M3 "
              f"{m3.get('n_obs')}/{m3.get('n_states')}")
        check(f'{key} is a genuinely different sample from the M1 RI fit',
              r['n_obs'] != m1.get('n_obs'),
              'matched baseline equals the unmatched one -- the two Delta '
              'percentages would then be identical and one of them is wrong')


# ============================================================
# [5] COVID ROBUSTNESS under nbinom2
# ============================================================
def validate_covid():
    section('5', 'COVID robustness refit under nbinom2')
    res = load_results()
    import pandas as pd
    p = pd.read_csv(GEN + 'count_model_panel_2021.csv')
    # The indicator must be what it claims to be before any coefficient on it
    # is interpretable.
    covid_years = set(p.loc[p['covid'] == 1, 'year'])
    check('covid indicator marks exactly 2020 and 2021',
          covid_years == {2020, 2021}, f'got {sorted(covid_years)}')
    for cell in CELLS:
        key = f'{cell}__M3covid__nbinom2__rs'
        r = res.get(key)
        if r is None:
            check(f'{key} present', False, 'missing from results JSON')
            continue
        m3 = res.get(f'{cell}__M3__nbinom2__rs', {})
        check(f'{key} converged', bool(r['converged']), r.get('message', ''))
        if not r.get('converged'):
            continue
        check(f'{key} re_tier == rs', r['re_tier'] == 'rs')
        check(f'{key} carries a covid term', 'covid' in r['cond'])
        check(f'{key} is M3 + covid and nothing else',
              set(r['cond']) == set(m3.get('cond', {})) | {'covid'},
              f"got {sorted(r['cond'])}")
        check(f'{key} sits on the M3 sample',
              r['n_obs'] == m3.get('n_obs'),
              f"{r['n_obs']} vs M3 {m3.get('n_obs')}")
        check(f'{key} covid SE is finite',
              r['cond']['covid']['se'] == r['cond']['covid']['se']
              and r['cond']['covid']['se'] < float('inf'))
    # The whole point of the refit: it must be a DIFFERENT family from the
    # frozen ladder's COVID variants, or nothing has been re-derived.
    with open(GEN + 'count_model_results.json') as fh:
        old = json.load(fh)
    old_covid = [k for k in old if '__M3covid__' in k]
    # M2: `all(...)` over an empty set is vacuously True -- if the frozen
    # ladder ever had zero '__M3covid__' keys this check would pass having
    # verified nothing. Require the set to be non-empty too.
    check('frozen ladder COVID variants are not nbinom2 (so a refit was needed)',
          bool(old_covid) and all(not k.endswith('__nbinom2') for k in old_covid),
          f'found {old_covid}')


# ============================================================
# [6] CROSS-SOFTWARE CHECK (statsmodels NB2, state fixed effects)
# ============================================================
# Independently re-derived here (not imported from nb2_crosscheck_statsmodels.py)
# so the validator's notion of "which terms fixed effects can identify" can
# disagree with the cross-check script's own if either one has a bug. Five of
# the six Model-3-added covariates are Level-2 (state-level, time-invariant)
# by this project's own design and are collinear with a full set of state
# dummies by construction; they cannot be fit under state fixed effects at all.
_CC_WITHIN_STATE_SD_TOL = 1e-12
_CC_CUBIC = ['time', 'time2', 'time3']
_CC_M3_ADD = ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
              'h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']
_CC_CELL_BASE = {
    'insp_2021': _CC_CUBIC,
    'viol_cov_2021': ['log_inspections'] + _CC_CUBIC,
}
_CC_CELL_DV = {'insp_2021': 'inspections', 'viol_cov_2021': 'violations'}


def _identifiable_split(cell):
    """Return (time_varying, state_invariant) term sets for `cell`, computed
    directly from the panel on that cell's own M3 analytic sample."""
    import pandas as pd
    panel = pd.read_csv(GEN + 'count_model_panel_2021.csv')
    rhs = _CC_CELL_BASE[cell] + _CC_M3_ADD
    need = [_CC_CELL_DV[cell], 'state'] + rhs
    d = panel[need].dropna()
    within_sd = d.groupby('state')[rhs].std().max()
    varying = {t for t in rhs if within_sd[t] > _CC_WITHIN_STATE_SD_TOL}
    invariant = {t for t in rhs if within_sd[t] <= _CC_WITHIN_STATE_SD_TOL}
    return varying, invariant


def validate_crosscheck():
    section('6', 'Cross-software check: statsmodels NegativeBinomialP(p=2) + state FE')
    try:
        with open(GEN + 'nb2_stepwise_crosscheck.json') as fh:
            cc = json.load(fh)
    except FileNotFoundError:
        check('nb2_stepwise_crosscheck.json exists', False, 'not found')
        return
    for cell in CELLS:
        c = cc.get(cell)
        if c is None:
            check(f'{cell} present in cross-check', False)
            continue
        check(f'{cell}: statsmodels fit converged', bool(c['converged']))
        check(f'{cell}: same analytic sample as the glmmTMB M3 fit',
              c['n_obs'] == load_results()[f'{cell}__M3__nbinom2__rs']['n_obs'],
              f"crosscheck {c['n_obs']}")
        terms = c['terms']
        varying, invariant = _identifiable_split(cell)
        # Five of the six M3-added covariates are state-invariant and are
        # absorbed by a full set of state dummies by construction (verified:
        # the design matrix is rank-deficient by exactly the size of that
        # set). The cross-check must cover exactly the time-varying subset,
        # not "every M3 fixed effect except the intercept" -- that older
        # assertion asked for something mathematically impossible.
        check(f'{cell}: cross-check covers exactly the terms fixed effects '
              f'can identify (time-varying within at least one state)',
              set(terms) == varying,
              f'got {sorted(terms)}, expected {sorted(varying)}')
        check(f'{cell}: cross-check records exactly the state-invariant '
              f'terms as absorbed',
              set(c.get('terms_absorbed', [])) == invariant,
              f"got {sorted(c.get('terms_absorbed', []))}, expected {sorted(invariant)}")
        # The assertion. Sign agreement ONLY among terms both implementations
        # resolve away from zero -- a term neither can distinguish from zero has
        # no sign to agree about, and demanding one would be manufacturing a
        # check that passes by luck.
        contested = [t for t, v in terms.items()
                     if v['both_nonzero'] and not v['sign_agrees']]
        check(f'{cell}: all clearly-nonzero terms agree in sign across software',
              not contested, f'disagree: {contested}')
        n_tested = sum(1 for v in terms.values() if v['both_nonzero'])
        check(f'{cell}: at least one term was actually testable',
              n_tested > 0,
              'no term was distinguishable from zero in both fits, so this '
              'section proved nothing')
        print(f"    ({cell}: {n_tested}/{len(terms)} terms distinguishable from "
              f"zero in both fits; ratios reported in the memo, not asserted)")
        # A2: nb2_stepwise_crosscheck.json snapshots 'b_tmb' at ITS OWN run
        # time. If the R fits (and this reporter) were re-run without also
        # re-running the crosscheck script, the memo would print stale
        # glmmTMB coefficients next to fresh ones everywhere else, and
        # nothing above (only n_obs is compared) would catch it. Assert the
        # crosscheck's snapshot of each shared term's glmmTMB coefficient is
        # EXACTLY the fresh M3 fit's own coefficient.
        m3_fresh = load_results()[f'{cell}__M3__nbinom2__rs']
        for t, v in terms.items():
            check(f'{cell}: crosscheck b_tmb[{t}] matches the current M3 fit '
                  f'exactly (snapshot staleness guard)',
                  m3_fresh.get('converged') and v['b_tmb'] == m3_fresh['cond'][t]['b'],
                  f"crosscheck b_tmb={v['b_tmb']!r}, current b="
                  f"{m3_fresh.get('cond', {}).get(t, {}).get('b')!r}")


# ============================================================
# [7] VARIANCE ARITHMETIC
# ============================================================
def validate_variance_table():
    section('7', 'Variance components table: arithmetic and basis')
    import math
    import pandas as pd
    try:
        v = pd.read_csv(GEN + 'nb2_stepwise_variance.csv')
    except FileNotFoundError:
        check('nb2_stepwise_variance.csv exists', False, 'not found')
        return
    res = load_results()
    check('one row per cell x model', len(v) == 6, f'got {len(v)}')
    for _, row in v.iterrows():
        cell, model = row['cell'], row['model']
        rs = res[f'{cell}__{model}__nbinom2__rs']
        ri = res[f'{cell}__{model}__nbinom2__ri']
        tag = f'{cell} {model}'
        # The *_rs columns come from the random-slope fits...
        check_close(f'{tag}: sigma2_u0_rs from the rs fit',
                    row['sigma2_u0_rs'], rs['sigma2_u0'], 1e-9, 'rel')
        check_close(f'{tag}: sigma2_u1_rs from the rs fit',
                    row['sigma2_u1_rs'], rs['sigma2_u1'], 1e-9, 'rel')
        # ...and sigma2_u0_ri, theta and the ICC all come from the RI fit, so
        # the ICC and the Delta share one basis.
        check_close(f'{tag}: sigma2_u0_ri from the ri fit',
                    row['sigma2_u0_ri'], ri['sigma2_u0'], 1e-9, 'rel')
        check_close(f'{tag}: theta from the ri fit',
                    row['theta'], ri['dispersion'], 1e-9, 'rel')
        want_icc = ri['sigma2_u0'] / (ri['sigma2_u0'] + math.log(
            1 + 1 / ri['dispersion'] + 1 / ri['mu_fixed']))
        check_close(f'{tag}: icc_ri matches Nakagawa OLV formula',
                    row['icc_ri'], want_icc, 1e-9, 'rel')
        check(f'{tag}: icc_ri in (0, 1)', 0 < row['icc_ri'] < 1,
              f"got {row['icc_ri']}")
        base = res[f'{cell}__M1__nbinom2__ri']['sigma2_u0']
        matched = res[f'{cell}__M1matched__nbinom2__ri']['sigma2_u0']
        check_close(f'{tag}: delta_pct_ri arithmetic',
                    row['delta_pct_ri'],
                    100 * (base - ri['sigma2_u0']) / base, 1e-9, 'abs')
        check_close(f'{tag}: delta_pct_ri_matched arithmetic',
                    row['delta_pct_ri_matched'],
                    100 * (matched - ri['sigma2_u0']) / matched, 1e-9, 'abs')
        check(f'{tag}: n_obs/n_states agree with the rs fit',
              row['n_obs'] == rs['n_obs'] and row['n_states'] == rs['n_states'])
    for cell in CELLS:
        m1 = v[(v['cell'] == cell) & (v['model'] == 'M1')].iloc[0]
        check(f'{cell}: M1 delta_pct_ri == 0 (it is its own baseline)',
              abs(m1['delta_pct_ri']) < 1e-9, f"got {m1['delta_pct_ri']}")
    # No sigma2_e column anywhere -- a count family has no residual variance.
    check('no residual-variance column in the variance table',
          not any(c in v.columns for c in ('sigma2_e', 'sigma_e', 'residual_variance')),
          f'columns: {list(v.columns)}')


# ============================================================
# [8] FROZEN ARTIFACTS
# ============================================================
def validate_frozen():
    section('8', 'Frozen artifacts unchanged')
    frozen = [
        'scripts/count_models_zinb.R',
        'scripts/report_count_models.py',
        'scripts/validate_count_models.py',
        'scripts/build_count_model_panel.py',
        'data/generated/count_model_results.json',
        'data/generated/count_model_panel_2019.csv',
        'data/generated/count_model_panel_2021.csv',
        'docs/count_models_zinb.md',
    ]
    # --- (a) working-tree check: catches uncommitted edits. ---
    proc = subprocess.run(['git', 'diff', 'HEAD', '--name-only', '--'] + frozen,
                          capture_output=True, text=True, cwd=ROOT)
    changed = [ln for ln in proc.stdout.split('\n') if ln.strip()]
    check('no frozen count-model artifact modified (working tree vs HEAD)',
          not changed, f'modified: {changed}')
    proc = subprocess.run(['git', 'diff', 'HEAD', '--name-only'],
                          capture_output=True, text=True, cwd=ROOT)
    touched = [ln for ln in proc.stdout.split('\n') if ln.strip()]
    bad = [f for f in touched
           if f.endswith('.docx') or 'paper_table' in f]
    check('no .docx and no paper_table artifact modified (working tree vs HEAD)',
          not bad, f'modified: {bad}')
    # --- (b) committed-history check: `git diff HEAD` alone is blind to
    # anything already committed on this branch -- it only ever compares the
    # working tree to HEAD, so a frozen artifact edited and then committed
    # passes check (a) unconditionally, which is exactly the case this
    # section exists to catch (review Finding 2). Diff against the branch's
    # merge-base with main instead. The merge-base is computed fresh on every
    # run, never hardcoded, so it stays correct as the branch grows. If it
    # cannot be resolved, FAIL loudly rather than silently skip -- a guard
    # that quietly stops guarding is worse than no guard. ---
    mb = subprocess.run(['git', 'merge-base', 'main', 'HEAD'],
                        capture_output=True, text=True, cwd=ROOT)
    merge_base = mb.stdout.strip()
    mb_ok = mb.returncode == 0 and bool(merge_base)
    check('git merge-base main HEAD resolves (required for the '
          'committed-history guard)', mb_ok,
          f'returncode {mb.returncode}; stdout {mb.stdout!r}; '
          f'stderr {mb.stderr!r}')
    if not mb_ok:
        check('no frozen count-model artifact modified (merge-base vs HEAD)',
              False, 'merge-base could not be resolved -- guard did not run')
        check('no .docx and no paper_table artifact modified '
              '(merge-base vs HEAD)', False,
              'merge-base could not be resolved -- guard did not run')
        return
    proc = subprocess.run(['git', 'diff', f'{merge_base}..HEAD', '--name-only',
                          '--'] + frozen,
                          capture_output=True, text=True, cwd=ROOT)
    changed_hist = [ln for ln in proc.stdout.split('\n') if ln.strip()]
    check('no frozen count-model artifact modified (merge-base vs HEAD)',
          not changed_hist,
          f'modified since merge-base {merge_base}: {changed_hist}')
    proc = subprocess.run(['git', 'diff', f'{merge_base}..HEAD', '--name-only'],
                          capture_output=True, text=True, cwd=ROOT)
    touched_hist = [ln for ln in proc.stdout.split('\n') if ln.strip()]
    bad_hist = [f for f in touched_hist
                if f.endswith('.docx') or 'paper_table' in f]
    check('no .docx and no paper_table artifact modified (merge-base vs HEAD)',
          not bad_hist,
          f'modified since merge-base {merge_base}: {bad_hist}')


# ============================================================
# [9] MEMO
# ============================================================
MEMO_FORBIDDEN_PATTERNS = [
    # NB2 lost to the selected family at every rung. Any of these would read as
    # a fit-based win. Spec section 3.
    'nbinom2 was selected', 'NB2 was selected', 'NB2 is selected',
    'NB2 won', 'NB2 wins', 'nbinom2 wins',
    'best-fitting family', 'the preferred family',
    # A count family has no residual variance. NOTE: this also forbids the memo
    # from DENYING one in those exact words, which is why the memo writes the
    # denial hyphenated ('no residual-variance column'). That is deliberate --
    # do not "fix" the memo by removing the hyphen.
    'residual variance', 'sigma2_e', 'sigma^2_e',
    # The Delta comes from the RI series, never from the random-slope fits.
    'reduction in the random-slope',
]


def validate_memo():
    section('9', 'Memo')
    try:
        with open('/Users/keshavgoel/Research/docs/nb2_stepwise_models.md') as fh:
            memo = fh.read()
    except FileNotFoundError:
        check('docs/nb2_stepwise_models.md exists', False, 'not found')
        return
    check('memo marks itself generated',
          'generated' in memo.lower() and 'never hand-edited' in memo.lower())
    for pat in MEMO_FORBIDDEN_PATTERNS:
        check(f'memo does not contain: {pat!r}', pat.lower() not in memo.lower())
    # The caveats the spec requires by name.
    required = [
        ('AIC penalty is stated', 'editorial'),
        ('link-scale caveat present', 'link scale'),
        ('sample-change caveat present', 'AK, RI and VT'),
        ('ICC formula printed', 'Nakagawa'),
        ('COVID section present', 'COVID'),
        ('cross-check section present', 'state fixed effects'),
    ]
    # Case-SENSITIVE and exact-phrase. A case-folded 'AK' matched 'make',
    # 'take' and 'breaks', so that row passed no matter what the memo said.
    for label, needle in required:
        check(f'memo: {label}', needle in memo, f'{needle!r} not found')
    # Review Finding 1: the 'before COVID enters at all' bullet asserted a
    # fixed 'the two families already disagree' regardless of what the
    # p-values actually show, and for inspections neither `time` nor `time2`
    # flips significance status -- the memo was contradicting its own printed
    # numbers. Derive the expected verdict HERE, straight from the two JSONs,
    # independent of whatever wording the reporter produced, and check the
    # memo's per-cell bullet says the right thing.
    _covid_ladder_ref = {
        'insp_2021': ('insp_2021__M3__zinb', 'ZINB'),
        'viol_cov_2021': ('viol_cov_2021__M3__nbinom1', 'NB1'),
    }
    _outcome_name = {'insp_2021': 'inspections', 'viol_cov_2021': 'violations'}
    res = load_results()
    with open(GEN + 'count_model_results.json') as fh:
        ladder = json.load(fh)
    for cell in CELLS:
        old_key, old_fam = _covid_ladder_ref[cell]
        o_m3 = ladder[old_key]['cond']
        n_m3 = res[f'{cell}__M3__nbinom2__rs']['cond']
        flips = [t for t in ('time', 'time2')
                 if (o_m3[t]['p'] < .05) != (n_m3[t]['p'] < .05)]
        expect_disagree = bool(flips)
        outcome = _outcome_name[cell]
        marker = f'{outcome}, before COVID enters at all:'
        matches = [ln for ln in memo.splitlines() if marker in ln]
        if not matches:
            check(f'{cell}: pre-COVID time-trend bullet present', False,
                  f'marker {marker!r} not found in memo')
            continue
        line = matches[0]
        says_disagree = 'disagree' in line.lower()
        check(f'{cell}: pre-COVID time-trend bullet verdict matches the '
              f'JSONs (expected {"disagree" if expect_disagree else "agree"}, '
              f'flips={flips})',
              says_disagree == expect_disagree, f'line: {line!r}')
    # The ZI-NB tallies belong to the other memo; restating them here would
    # invite a reader to think this arm re-tested zero-inflation. It did not.
    check('memo does not restate the ZI-NB tallies',
          '17-0' not in memo and '17–0' not in memo and '17 of 17' not in memo)


# ============================================================
# [10] COEFFICIENT TABLE: round-trip and IRR arithmetic
# ============================================================
# M6: nb2_stepwise_coefficients.csv and the memo's coefficient tables had no
# validation at all, unlike the variance table's ~58 assertions. Every row
# must trace back to the results JSON exactly, and IRR must be exp(b).
def validate_coefficient_table():
    section('10', "Coefficient table: round-trip from the results JSON, IRR arithmetic")
    import math
    import pandas as pd
    try:
        ct = pd.read_csv(GEN + 'nb2_stepwise_coefficients.csv')
    except FileNotFoundError:
        check('nb2_stepwise_coefficients.csv exists', False, 'not found')
        return
    res = load_results()
    check('coefficient table is non-empty', len(ct) > 0, f'got {len(ct)} rows')
    for _, row in ct.iterrows():
        key = f"{row['cell']}__{row['model']}__nbinom2__rs"
        tag = f"{key}[{row['term']}]"
        r = res.get(key)
        if r is None or not r.get('converged') or row['term'] not in r.get('cond', {}):
            check(f'{tag}: source record present, converged and carries this term',
                  False, 'missing/not converged/term absent')
            continue
        c = r['cond'][row['term']]
        check_close(f'{tag}: b round-trips from the JSON', row['b'], c['b'],
                    1e-9, 'abs')
        check_close(f'{tag}: se round-trips from the JSON', row['se'], c['se'],
                    1e-9, 'abs')
        check_close(f'{tag}: p round-trips from the JSON', row['p'], c['p'],
                    1e-9, 'abs')
        check_close(f'{tag}: irr == exp(b)', row['irr'], math.exp(c['b']),
                    1e-9, 'rel')


def main():
    skip_pre = '--skip-precondition' in sys.argv
    validate_precondition(skip_pre)
    validate_panel()
    validate_fits()
    validate_overlap()
    validate_ri_series()
    validate_covid()
    validate_crosscheck()
    validate_variance_table()
    validate_frozen()
    validate_memo()
    validate_coefficient_table()

    print()
    print("=" * 78)
    print("CHECK BREAKDOWN BY SECTION (do not quote the total on its own: the "
          "per-fit")
    print("sections are record loops, not sets of independent findings)")
    print("=" * 78)
    print(f"{'section':62}{'PASS':>6}{'FAIL':>6}{'SKIP':>6}")
    for sec in SECTIONS:
        print(f"{sec['name'][:62]:62}{sec['pass']:>6}{sec['fail']:>6}{sec['skip']:>6}")
    tp = sum(x['pass'] for x in SECTIONS)
    tf = sum(x['fail'] for x in SECTIONS)
    tsk = sum(x['skip'] for x in SECTIONS)
    print(f"{'TOTAL (see the caveat above)':62}{tp:>6}{tf:>6}{tsk:>6}")
    if tsk:
        print(f"\n{tsk} gated block(s) were SKIPPED -- each is printed above "
              f"with its reason.")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
