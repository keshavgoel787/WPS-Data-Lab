"""
Validation gate for the Jafari-aligned crossed random-effects count models.

Not a unit-test suite -- an analysis artifact, matching the convention of
scripts/validate_count_models.py and scripts/validate_nb2_stepwise.py. Each
check proves either that a step reproduces something already known or that an
invariant the spec depends on holds.

Run: python3 scripts/validate_jafari_crossed.py    (exits 1 on any failure)
     python3 scripts/validate_jafari_crossed.py --skip-precondition
         Skips section [0] only. FOR ITERATION DURING DEVELOPMENT ONLY -- a run
         with [0] skipped is not a passing run, and the SKIP line says so.

ON READING THE OUTPUT: do not quote the total on its own. main() prints a
per-section breakdown and an explicit SKIP line for every bypassed block.

Spec: docs/superpowers/specs/2026-09-10-jafari-crossed-re-design.md
"""
import json
import subprocess
import sys
import warnings

import pandas as pd

GEN = '/Users/keshavgoel/Research/data/generated/'
SCRIPTS = '/Users/keshavgoel/Research/scripts/'
ROOT = '/Users/keshavgoel/Research/'
GAUSS_JSON = GEN + 'jafari_crossed_gaussian.json'
MAIN_JSON = GEN + 'jafari_crossed_results.json'

FAILURES = []
SECTIONS = []
_CUR = None

CELLS = ('insp_2019', 'insp_2021', 'viol_2019', 'viol_2021')
CUBIC = ['time', 'time2', 'time3']
M2_ADD = ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z']
M3_ADD = M2_ADD + ['h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']

# (panel window, DV, base predictors) per cell.
CELL_SPEC = {
    'insp_2019': (2019, 'inspections', CUBIC),
    'insp_2021': (2021, 'inspections', CUBIC),
    'viol_2019': (2019, 'violations', ['log_inspections'] + CUBIC),
    'viol_2021': (2021, 'violations', ['log_inspections'] + CUBIC),
}

PANEL_FACTS = {
    2019: dict(rows=450, states=50, years=9, insp_zeros=15, viol_zeros=10),
    2021: dict(rows=539, states=49, years=11, insp_zeros=7, viol_zeros=93),
}

# A variance component is AT THE BOUNDARY when it is negligible relative to the
# residual variance of the same fit, and there a relative tolerance is not a
# meaningful test -- 5e-8 against 2.6e-6 is a 98% relative "error" and also two
# packages agreeing that the variance is zero.
#
# The floor is SCALE-RELATIVE, not a bare constant, because "negligible" is only
# defined against something: 1e-4 of that fit's own sigma2_e. An absolute floor
# was tried first and was wrong -- it passed insp_2019 (sigma2_year ~2e-10) and
# failed viol_2021 M3 (5e-8 vs 2.6e-6) despite both being the same phenomenon.
#
# This still catches a REAL disagreement: with sigma2_e ~0.7 the floor is ~7e-5,
# so a glmmTMB 0.05 against a statsmodels 0.0 goes to the relative branch and
# fails, as it should.
VAR_ABS_FLOOR = 1e-6
VAR_REL_FLOOR = 1e-4


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
    """Compare two variance components, boundary-aware.

    A relative tolerance is the right test for a variance that is genuinely
    estimated, and the WRONG test for one pinned at the boundary: glmmTMB
    reports 2.1e-10 where statsmodels reports 0.0 or 2.6e-6, and all of those
    mean the same thing -- no variance at this level.

    `scale_ref` is the fit's own sigma2_e, which is what makes "negligible"
    well-defined. Below max(VAR_ABS_FLOOR, VAR_REL_FLOOR * scale_ref) both
    values are treated as boundary; above it the relative test applies.
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


def load_gaussian():
    with open(GAUSS_JSON) as fh:
        return json.load(fh)


def load_results():
    with open(MAIN_JSON) as fh:
        return json.load(fh)


def panel(window):
    return pd.read_csv(GEN + f'count_model_panel_{window}.csv')


# ============================================================
# [0] PRECONDITION -- the existing R/Python bridge still works
# ============================================================
def validate_precondition(skip_precondition):
    section('0', 'Precondition: the existing Gaussian round-trip gate passes')
    if skip_precondition:
        skip('validate_count_models.py Gaussian sections',
             '--skip-precondition passed; this run is NOT a passing run')
        return
    proc = subprocess.run([sys.executable, SCRIPTS + 'validate_count_models.py'],
                          capture_output=True, text=True, cwd=ROOT)
    check('validate_count_models.py exits 0', proc.returncode == 0,
          f'exit {proc.returncode}; last stderr line: '
          f'{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "(none)"}')
    for tag in ('[2a]', '[2b]', '[2c]'):
        rows = [ln for ln in proc.stdout.splitlines() if ln.startswith(tag)]
        tail = [ln for ln in rows if ln.rstrip()[-1:].isdigit()]
        ok = bool(tail) and int(tail[-1].split()[-2]) == 0 and int(tail[-1].split()[-3]) > 0
        check(f'existing Gaussian gate section {tag}: >0 PASS and 0 FAIL', ok,
              f'breakdown row: {tail[-1] if tail else "(not found)"}')


# ============================================================
# [1] PANEL IDENTITY -- inputs are the frozen, unmodified panels
# ============================================================
def validate_panel():
    section('1', 'Panel identity (count_model_panel_{2019,2021}.csv, unmodified)')
    for window, facts in PANEL_FACTS.items():
        p = panel(window)
        check(f'{window}: N rows == {facts["rows"]}', len(p) == facts['rows'], f'got {len(p)}')
        check(f'{window}: N states == {facts["states"]}',
              p['state'].nunique() == facts['states'], f'got {p["state"].nunique()}')
        check(f'{window}: N year levels == {facts["years"]}',
              p['year'].nunique() == facts['years'], f'got {p["year"].nunique()}')
        check(f'{window}: inspections zeros == {facts["insp_zeros"]}',
              int((p['inspections'] == 0).sum()) == facts['insp_zeros'],
              f'got {int((p["inspections"] == 0).sum())}')
        check(f'{window}: violations zeros == {facts["viol_zeros"]}',
              int((p['violations'] == 0).sum()) == facts['viol_zeros'],
              f'got {int((p["violations"] == 0).sum())}')
        check(f'{window}: time == year - 2017 throughout',
              bool((p['time'] == p['year'] - 2017).all()))
        for col in M3_ADD + ['log_inspections', 'year', 'state']:
            check(f'{window}: column present: {col}', col in p.columns)
    check('2021 panel has no Wyoming row (WPS view)',
          'Wyoming' not in set(panel(2021)['state']))


# ============================================================
# [2] CROSSED-RE GAUSSIAN ROUND-TRIP -- glmmTMB vs statsmodels.MixedLM
# ============================================================
def _statsmodels_crossed(df, dv, rhs):
    """Fit `dv ~ rhs + (1|state) + (1|year)` in statsmodels, REML.

    statsmodels has no crossed-RE syntax. The documented recipe is a single
    constant group with two variance components, each an indicator basis over
    one grouping factor -- which is algebraically the same model glmmTMB fits
    from `(1 | state) + (1 | year)`.

    Returns (fe params dict, vcomp dict, scale, warning text).
    """
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[dv] + rhs + ['state', 'year']).copy()
    d['_grp'] = 1
    vcf = {'state': '0 + C(state)', 'year': '0 + C(year)'}
    md = smf.mixedlm(f"{dv} ~ {' + '.join(rhs)}", d, groups='_grp', vc_formula=vcf)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        mdf = md.fit(reml=True)
    msgs = '; '.join(str(w.message) for w in caught)
    # vcomp is positional; md.exog_vc.names carries the order. Zipping rather
    # than indexing [0]/[1] so a future statsmodels reordering cannot silently
    # swap sigma2_state and sigma2_year.
    return dict(mdf.fe_params), dict(zip(md.exog_vc.names, mdf.vcomp)), mdf.scale, msgs


def validate_crossed_roundtrip():
    section('2', 'Crossed-RE Gaussian round-trip: glmmTMB vs statsmodels.MixedLM')
    g = load_gaussian()
    for cell in CELLS:
        window, outcome, _ = CELL_SPEC[cell]
        dv = 'log_inspections' if outcome == 'inspections' else 'log_violations'
        gbase = CUBIC if outcome == 'inspections' else ['log_inspections'] + CUBIC
        for model in ('M1', 'M3'):
            rhs = gbase if model == 'M1' else gbase + M3_ADD
            key = f'{cell}__{model}__gaussian'
            rec = g.get(key)
            if rec is None or not rec.get('converged'):
                check(f'{key}: glmmTMB fit present and converged', False,
                      f'record: {rec.get("message") if rec else "missing"}')
                continue
            try:
                params, vcomp, scale, _ = _statsmodels_crossed(panel(window), dv, rhs)
            except Exception as exc:
                # Spec 10.2: the gate is NOT weakened silently. An unavailable
                # cross-check is a FAILURE so it cannot pass by accident.
                check(f'{key}: statsmodels crossed-RE fit available', False,
                      f'{type(exc).__name__}: {exc}')
                continue
            check(f'{key}: same analytic N across packages',
                  rec['n_obs'] == len(panel(window).dropna(
                      subset=[dv] + rhs + ['state', 'year'])),
                  f'glmmTMB {rec["n_obs"]}')
            for term, val in rec['cond'].items():
                sm_name = 'Intercept' if term == '(Intercept)' else term
                check_close(f'{key}: b[{term}]', val['b'], params.get(sm_name), 1e-3)
            check_var_close(f'{key}: sigma2_state', rec['sigma2_state'],
                            vcomp.get('state'), scale_ref=scale)
            check_var_close(f'{key}: sigma2_year', rec['sigma2_year'],
                            vcomp.get('year'), scale_ref=scale)
            check_close(f'{key}: sigma2_e', rec['sigma2_e'], scale, 2e-2)


def main():
    skip_pre = '--skip-precondition' in sys.argv
    validate_precondition(skip_pre)
    validate_panel()
    validate_crossed_roundtrip()

    print('\n' + '=' * 78)
    print('PER-SECTION BREAKDOWN (do not quote the total alone)')
    print('=' * 78)
    for s in SECTIONS:
        print(f"{s['name']:<60} {s['pass']:>5} {s['fail']:>5} {s['skip']:>5}")
    tp = sum(s['pass'] for s in SECTIONS)
    tf = sum(s['fail'] for s in SECTIONS)
    ts = sum(s['skip'] for s in SECTIONS)
    print(f"{'TOTAL':<60} {tp:>5} {tf:>5} {ts:>5}")
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f'  - {f}')
        sys.exit(1)
    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
