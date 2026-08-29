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
            check(f'{key} positive-definite Hessian', bool(r['pd_hess']))
            check(f'{key} conv_code == 0', r['conv_code'] == 0, f"got {r['conv_code']}")
            check(f'{key} re_tier == rs (no fallback)', r['re_tier'] == 'rs',
                  f"got {r['re_tier']}")
            check(f'{key} re_used == (1 + time | state)',
                  r['re_used'] == '(1 + time | state)', f"got {r['re_used']}")
            check(f'{key} family_name == nbinom2', r['family_name'] == 'nbinom2')
            check(f'{key} sigma2_e is null (count family has no residual variance)',
                  r['sigma2_e'] is None, f"got {r['sigma2_e']!r}")
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
            if r is None:
                continue
            want = {'(Intercept)'} | set(base) | set(extra)
            check(f'{cell} {model}: fixed-effect terms exactly as specified',
                  set(r['cond']) == want,
                  f"extra {set(r['cond']) - want}, missing {want - set(r['cond'])}")
            check(f'{cell} {model}: no interaction terms',
                  not any(':' in t for t in r['cond']))


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


def main():
    skip_pre = '--skip-precondition' in sys.argv
    validate_precondition(skip_pre)
    validate_panel()
    validate_fits()
    validate_overlap()
    validate_ri_series()

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
