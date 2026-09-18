"""
Validation gate for the STEPWISE state-to-state variance arm
(scripts/jafari_stepwise_models.R + scripts/report_jafari_stepwise.py).

Not a unit-test suite -- an analysis artifact, matching the convention of
scripts/validate_count_models.py, scripts/validate_nb2_stepwise.py and
scripts/validate_jafari_crossed.py. Each check proves either that a step
reproduces something already known or that an invariant the arm's claims
depend on actually holds.

Nearest analogue: validate_nb2_stepwise.py. That arm is also a stepwise
variance-reduction arm with a matched Model-1 baseline, and the two share the
"re-derive the Delta rather than read it" requirement -- the reporter owns the
number, the validator must arrive at it by an independent route or the check
proves nothing.

Run: python3 scripts/validate_jafari_stepwise.py      (exits 1 on any failure)
     python3 scripts/validate_jafari_stepwise.py --skip-precondition
         Skips the UPSTREAM-VALIDATOR half of section [0] only (the panel
         identity half always runs). FOR ITERATION DURING DEVELOPMENT ONLY --
         a run with it skipped is not a passing run, and the SKIP line says so.
     python3 scripts/validate_jafari_stepwise.py --skip-gaussian
         Skips section [8], which shells out to Rscript. Same caveat.

ON READING THE OUTPUT: do not quote the total on its own. main() prints a
per-section breakdown and an explicit SKIP line for every bypassed block, so a
run in which a block silently vanished is visibly different from one in which
it ran.

NO warnings.filterwarnings('ignore') ANYWHERE in this file. That idiom is what
hid the non-converged lbfgs fit documented in CLAUDE.md; warnings are captured
with catch_warnings(record=True) and reported.
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import warnings

import pandas as pd

ROOT = '/Users/keshavgoel/Research/'
GEN = ROOT + 'data/generated/'
SCRIPTS = ROOT + 'scripts/'
DOCS = ROOT + 'docs/'

PANEL = GEN + 'count_model_panel_2021.csv'
RESULTS = GEN + 'jafari_stepwise_results.json'
CROSSED = GEN + 'jafari_crossed_results.json'
MEMO = DOCS + 'jafari_stepwise_variance.md'

FAILURES = []
SECTIONS = []
_CUR = None

# ---- the arm's design, restated here independently of the artifacts --------
CUBIC = ['time', 'time2', 'time3']
M2_ADD = ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z']
M3_ADD = M2_ADD + ['h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']
MODELS = ('M1', 'M2', 'M3', 'M1matched')
SUBSTANTIVE = ('M1', 'M2', 'M3')
CROSSED_RE = '(1 | state) + (1 | year)'

# (dv, base predictors, the crossed-arm cell this one anchors against or None)
CELL_SPEC = {
    'insp_2021':    ('inspections', CUBIC, 'insp_2021'),
    'viol_2021_wi': ('violations', ['log_inspections'] + CUBIC, 'viol_2021'),
    'viol_2021_ni': ('violations', CUBIC, None),
}
CELLS = tuple(CELL_SPEC)
# The cell whose family had no precedent and was raced at M3 in this script.
SELECTION_CELL = 'viol_2021_ni'
LADDER = ('poisson', 'nbinom1', 'nbinom2', 'zip',
          'zinb1', 'zinb2', 'zinb1_re', 'zinb2_re')
# Families that carry a zero-inflation component at all.
ZI_FAMILIES = ('zip', 'zinb1', 'zinb2', 'zinb1_re', 'zinb2_re')

# The three states BLS has never published a pesticide-applicator series for,
# which is why SPEND_APP_z is missing and M2/M3 fall to 46 states.
EXPECTED_DROPPED = {'Alaska', 'Rhode Island', 'Vermont'}

# The only fit in the artifact permitted to carry converged = FALSE. It is a
# losing candidate in the viol_2021_ni family race, never a reported estimate.
ALLOWED_NONCONVERGED = {f'{SELECTION_CELL}__M3__zinb1_re__selection'}

# Presentation maps, duplicated from report_jafari_stepwise.py ON PURPOSE. If
# the reporter's copy drifts, the memo checks below stop matching -- which is
# the point. Importing them would make section [9] circular.
CELL_LABEL = {
    'insp_2021': 'Inspections',
    'viol_2021_wi': 'Violations, WITH log(inspections)',
    'viol_2021_ni': 'Violations, WITHOUT inspections',
}
MODEL_LABEL = {
    'M1': 'M1 (cubic time only)',
    'M1matched': 'M1 refit on the M2/M3 sample (matched baseline)',
    'M2': 'M2 (+ spending, commodity mix)',
    'M3': 'M3 (+ H-2A block)',
}
FAMILY_LABEL = {
    'poisson': 'Poisson', 'nbinom1': 'NB1', 'nbinom2': 'NB2',
    'zip': 'ZIP', 'zinb1': 'ZINB1', 'zinb2': 'ZINB2',
    'zinb1_re': 'ZINB1 + ZI-RE', 'zinb2_re': 'ZINB2 + ZI-RE',
}
TERM_LABEL = {
    '(Intercept)': 'Intercept',
    'log_inspections': 'log(inspections)',
    'time': 'time (year - 2017)',
    'time2': 'time^2',
    'time3': 'time^3',
    'SPEND_APP_z': 'STAG $ per pesticide applicator (z)',
    'SPEND_WORK_z': 'STAG $ per farmworker (z)',
    'lii_2017_z': 'Labor intensity index 2017 (z)',
    'h2a_per_farmworker_z': 'H-2A per farmworker (z)',
    'dol_demand_met_pct_z': 'DOL demand met % (z)',
    'pct_flc_z': '% farm labor contractor (z)',
}

# Boundary-aware variance comparator, copied from validate_jafari_crossed.py.
# sigma^2_year is pinned at the optimizer boundary in most of these cells, and a
# relative tolerance between two near-zero numbers is not a test: glmmTMB's
# 1.0e-09 against statsmodels' 0.0 is a 100% relative "error" and also two
# packages agreeing there is no variance at that level. The floor is
# SCALE-RELATIVE (1e-4 x that fit's own sigma2_e) because "negligible" is only
# defined against something.
VAR_ABS_FLOOR = 1e-6
VAR_REL_FLOOR = 1e-4

# The benign TMB ABI version-mismatch warning glmmTMB emits on every Rscript
# call in this project. Anything in stderr that is NOT one of these is a leak.
BENIGN_R_STDERR = (
    'Warning message', 'Warning in check_dep_version', 'check_dep_version',
    'package version mismatch', 'glmmTMB was built with TMB',
    'Current TMB package version',
    'Please re-install glmmTMB from source',
)


# ============================================================
# harness
# ============================================================
def section(tag, title):
    global _CUR
    _CUR = {'name': f'[{tag}] {title}', 'pass': 0, 'fail': 0, 'skip': 0}
    SECTIONS.append(_CUR)
    print(f'\n[{tag}] {title}')


def check(label, condition, detail=''):
    if condition:
        print(f'  PASS  {label}')
        if _CUR is not None:
            _CUR['pass'] += 1
    else:
        print(f'  FAIL  {label}' + (f' -- {detail}' if detail else ''))
        FAILURES.append(label)
        if _CUR is not None:
            _CUR['fail'] += 1


def skip(label, reason):
    print(f'  SKIP  {label} -- {reason}')
    if _CUR is not None:
        _CUR['skip'] += 1


def check_close(label, got, want, tol, kind='rel'):
    if got is None or want is None:
        check(f'{label} (both values present)', False,
              f'got {got!r}, want {want!r}')
        return
    delta = (abs(got - want) if kind == 'abs'
             else abs(got - want) / max(abs(want), 1e-12))
    check(f'{label} ({kind} delta {delta:.2e} <= {tol:.0e})', delta <= tol,
          f'got {got!r}, want {want!r}')


def check_var_close(label, got, want, scale_ref=None, rel=2e-2):
    """Compare two variance components, boundary-aware (see VAR_*_FLOOR)."""
    if got is None or want is None:
        check(f'{label} (both values present)', False,
              f'got {got!r}, want {want!r}')
        return
    floor = VAR_ABS_FLOOR
    if scale_ref is not None and scale_ref > 0:
        floor = max(floor, VAR_REL_FLOOR * scale_ref)
    if max(abs(got), abs(want)) < floor:
        check(f'{label} (both at the variance boundary, < {floor:.1e} '
              f'= 1e-4 x sigma2_e)', True)
        return
    check_close(label, got, want, rel, 'rel')


_RES = None
_CRO = None
_PANEL = None
_MEMO = None


def results():
    global _RES
    if _RES is None:
        with open(RESULTS) as fh:
            _RES = json.load(fh)
    return _RES


def crossed():
    global _CRO
    if _CRO is None:
        with open(CROSSED) as fh:
            _CRO = json.load(fh)
    return _CRO


def panel():
    global _PANEL
    if _PANEL is None:
        _PANEL = pd.read_csv(PANEL)
    return _PANEL


def memo_text():
    global _MEMO
    if _MEMO is None:
        with open(MEMO) as fh:
            _MEMO = fh.read()
    return _MEMO


def fam_of(cell):
    return results()[f'{cell}__meta']['family_tag']


def series_of(cell):
    return list(results()[f'{cell}__meta']['series'])


def rec(cell, model, series):
    return results().get(f'{cell}__{model}__{fam_of(cell)}__{series}')


def rhs_for(cell, model):
    base = list(CELL_SPEC[cell][1])
    if model in ('M1', 'M1matched'):
        return base
    if model == 'M2':
        return base + M2_ADD
    if model == 'M3':
        return base + M3_ADD
    raise ValueError(model)


def analytic_sample(cell, model):
    """Re-derive a fit's analytic sample straight from the panel.

    Replicates fit_spec()'s rule exactly: listwise deletion over
    c(dv, sample_rhs, state, year, time), where sample_rhs is the M3 predictor
    set for M1matched and the model's own rhs otherwise. Derived here rather
    than read from the JSON -- reading n_obs back out of the artifact that
    produced it would prove nothing.
    """
    dv = CELL_SPEC[cell][0]
    sel = rhs_for(cell, 'M3') if model == 'M1matched' else rhs_for(cell, model)
    need, seen = [], set()
    for c in [dv] + list(sel) + ['state', 'year', 'time']:
        if c not in seen and c in panel().columns:
            need.append(c)
            seen.add(c)
    return panel().dropna(subset=need)


# ============================================================
# [0] PRECONDITION
# ============================================================
def validate_precondition(skip_upstream):
    section('0', 'Precondition: frozen panel + the upstream crossed arm')
    p = panel()
    check('panel: N rows == 539', len(p) == 539, f'got {len(p)}')
    check('panel: N states == 49', p['state'].nunique() == 49,
          f"got {p['state'].nunique()}")
    check('panel: N year levels == 11', p['year'].nunique() == 11,
          f"got {p['year'].nunique()}")
    check('panel: Wyoming absent (the WPS view has no WY row)',
          'Wyoming' not in set(p['state']))
    check('panel: time == year - 2017 throughout',
          bool((p['time'] == p['year'] - 2017).all()))
    check('panel: time2 == time^2', bool((p['time2'] == p['time'] ** 2).all()))
    check('panel: time3 == time^3', bool((p['time3'] == p['time'] ** 3).all()))
    check('panel: violations zeros == 93',
          int((p['violations'] == 0).sum()) == 93,
          f"got {int((p['violations'] == 0).sum())}")
    check('panel: inspections zeros == 7',
          int((p['inspections'] == 0).sum()) == 7,
          f"got {int((p['inspections'] == 0).sum())}")
    for col in (['state', 'year', 'time', 'violations', 'inspections',
                 'log_inspections'] + M3_ADD):
        check(f'panel: column present: {col}', col in p.columns)
    # Missingness is the whole reason this arm needs two Delta bases; pin it.
    check('panel: SPEND_APP_z missing for exactly 33 rows (3 states x 11 yr)',
          int(p['SPEND_APP_z'].isna().sum()) == 33,
          f"got {int(p['SPEND_APP_z'].isna().sum())}")
    check('panel: violations missing for exactly 6 rows',
          int(p['violations'].isna().sum()) == 6,
          f"got {int(p['violations'].isna().sum())}")
    check('panel: no other column has missing values',
          set(p.columns[p.isna().any()]) ==
          {'SPEND_APP_z', 'violations', 'log_violations'},
          f'got {sorted(p.columns[p.isna().any()])}')

    # Content identity against git, so "unmodified" is a fact rather than a
    # shape coincidence. No hash is hardcoded: the committed blob IS the
    # reference, recomputed every run.
    head = subprocess.run(['git', 'rev-parse',
                           'HEAD:data/generated/count_model_panel_2021.csv'],
                          capture_output=True, text=True, cwd=ROOT)
    cur = subprocess.run(['git', 'hash-object', PANEL],
                         capture_output=True, text=True, cwd=ROOT)
    check('panel: byte-identical to the committed blob',
          head.returncode == 0 and cur.returncode == 0
          and head.stdout.strip() == cur.stdout.strip()
          and bool(head.stdout.strip()),
          f'HEAD {head.stdout.strip()!r} vs working tree {cur.stdout.strip()!r}')

    # The upstream arm supplies this arm's correctness anchor ([2]); if its own
    # artifacts are broken, the anchor is anchored to nothing.
    if skip_upstream:
        skip('validate_jafari_crossed.py --skip-precondition',
             '--skip-precondition passed; this run is NOT a passing run')
        return
    proc = subprocess.run([sys.executable, SCRIPTS + 'validate_jafari_crossed.py',
                           '--skip-precondition'],
                          capture_output=True, text=True, cwd=ROOT)
    check('validate_jafari_crossed.py --skip-precondition exits 0',
          proc.returncode == 0,
          f'exit {proc.returncode}; last stdout line: '
          f'{proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "(none)"}')
    # Parse its per-section breakdown rather than trusting the exit code alone.
    for tag in ('[2]', '[3]', '[5]'):
        rows = [ln for ln in proc.stdout.splitlines() if ln.startswith(tag)]
        tail = [ln for ln in rows if ln.rstrip()[-1:].isdigit()]
        ok = (bool(tail) and int(tail[-1].split()[-2]) == 0
              and int(tail[-1].split()[-3]) > 0)
        check(f'upstream crossed-arm section {tag}: >0 PASS and 0 FAIL', ok,
              f'breakdown row: {tail[-1] if tail else "(not found)"}')


# ============================================================
# [1] RECORD STRUCTURE AND POPULATION
# ============================================================
FIT_REQUIRED = ('formula', 'zi_formula', 're_used', 'n_obs', 'n_states',
                'n_years', 'obs_zeros', 'converged', 'cell', 'model',
                'family_tag', 'series')
FIT_REQUIRED_CONVERGED = ('sigma2_state', 'sigma2_year', 'dispersion',
                          'family_name', 'loglik', 'df', 'aic', 'bic',
                          'cond', 'zi')


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def validate_records():
    section('1', 'Record structure and population (a record loop, not '
                 'independent findings)')
    r = results()

    # Population, derived from the design rather than compared against 37.
    expected = {'__run_meta'}
    for cell in CELLS:
        meta = r.get(f'{cell}__meta')
        if meta is None:
            check(f'{cell}__meta present', False)
            continue
        expected.add(f'{cell}__meta')
        # series is a design consequence of the family, not a free choice.
        want_series = (['mirror', 'zi_intercept'] if meta['family_is_zi']
                       else ['plain'])
        check(f'{cell}: series set follows from family_is_zi',
              list(meta['series']) == want_series,
              f"meta says {meta['series']}, family_is_zi={meta['family_is_zi']}")
        for ser in meta['series']:
            for m in MODELS:
                expected.add(f"{cell}__{m}__{meta['family_tag']}__{ser}")
        if CELL_SPEC[cell][2] is None:
            # No precedent -> this cell races the 8-family ladder at M3.
            expected.add(f'{cell}__selection_meta')
            for fam in LADDER:
                expected.add(f'{cell}__M3__{fam}__selection')
    check(f'record key set is exactly what the design implies '
          f'({len(expected)} keys)', set(r) == expected,
          f'extra {sorted(set(r) - expected)}, missing {sorted(expected - set(r))}')

    for key, v in r.items():
        if key == '__run_meta' or key.endswith('__meta') \
                or key.endswith('__selection_meta'):
            continue
        for f in FIT_REQUIRED:
            check(f'{key}: field present: {f}', f in v)
        check(f'{key}: re_used == {CROSSED_RE!r} (no random slope anywhere)',
              v.get('re_used') == CROSSED_RE, f"got {v.get('re_used')!r}")
        check(f'{key}: no random slope in the fitted formula',
              'time | state' not in (v.get('formula') or ''))
        check(f'{key}: crossed REs present in the fitted formula',
              CROSSED_RE in (v.get('formula') or ''))
        check(f'{key}: n_obs is a positive int',
              isinstance(v.get('n_obs'), int) and v['n_obs'] > 0)
        check(f'{key}: n_states is a positive int',
              isinstance(v.get('n_states'), int) and v['n_states'] > 0)
        check(f'{key}: n_years == 11', v.get('n_years') == 11,
              f"got {v.get('n_years')}")
        # A count family has no residual variance; nothing may export one.
        check(f'{key}: no sigma2_e field (count family has no residual '
              f'variance)', 'sigma2_e' not in v)
        if not v.get('converged'):
            check(f'{key}: non-converged fit carries a message',
                  bool(v.get('message')))
            continue
        for f in FIT_REQUIRED_CONVERGED:
            check(f'{key}: field present: {f}', f in v)
        check(f'{key}: sigma2_state finite and > 0',
              _finite(v.get('sigma2_state')) and v['sigma2_state'] > 0,
              f"got {v.get('sigma2_state')!r}")
        check(f'{key}: sigma2_year finite and >= 0',
              _finite(v.get('sigma2_year')) and v['sigma2_year'] >= 0,
              f"got {v.get('sigma2_year')!r}")
        check(f'{key}: dispersion finite and > 0',
              _finite(v.get('dispersion')) and v['dispersion'] > 0,
              f"got {v.get('dispersion')!r}")
        check(f'{key}: loglik finite', _finite(v.get('loglik')))
        check(f'{key}: df is a positive int',
              isinstance(v.get('df'), int) and v['df'] > 0)
        check(f'{key}: aic == -2*loglik + 2*df',
              _finite(v.get('aic'))
              and abs(v['aic'] - (-2 * v['loglik'] + 2 * v['df'])) < 1e-6,
              f"aic {v.get('aic')!r}, loglik {v.get('loglik')!r}, df {v.get('df')!r}")
        check(f'{key}: family_name in (nbinom1, nbinom2, poisson)',
              v.get('family_name') in ('nbinom1', 'nbinom2', 'poisson'),
              f"got {v.get('family_name')!r}")
        cond = v.get('cond') or {}
        check(f'{key}: conditional block non-empty and carries an intercept',
              bool(cond) and '(Intercept)' in cond)
        for t, c in cond.items():
            check(f'{key}: cond[{t}] has finite b/se/z/p',
                  all(_finite(c.get(f)) for f in ('b', 'se', 'z', 'p')),
                  f'got {c!r}')
            check(f'{key}: cond[{t}] p in [0, 1]', 0 <= c['p'] <= 1
                  if _finite(c.get('p')) else False)
        zi = v.get('zi') or {}
        if v['family_tag'] in ZI_FAMILIES:
            check(f'{key}: ZI block non-empty for a zero-inflated family',
                  bool(zi))
            for t, c in zi.items():
                check(f'{key}: zi[{t}] has finite b/se/z/p',
                      all(_finite(c.get(f)) for f in ('b', 'se', 'z', 'p')),
                      f'got {c!r}')
            # ZI predictors must be a SUBSET of conditional predictors -- that
            # is what keeps ONE analytic sample per (cell, model) and makes AIC
            # comparable across ZI tiers.
            extra = (set(zi) - {'(Intercept)'}) - (set(cond) - {'(Intercept)'})
            check(f'{key}: ZI terms are a subset of conditional terms',
                  not extra, f'extra: {sorted(extra)}')
            check(f'{key}: zi_prob_mean in [0, 1]',
                  _finite(v.get('zi_prob_mean')) and 0 <= v['zi_prob_mean'] <= 1,
                  f"got {v.get('zi_prob_mean')!r}")
            # Structural zeros are a SUBSET of all zeros: the model's own
            # expected structural-zero count can never exceed the number of
            # zeros it simulates. This is the consistency check that caught a
            # wrong ZI estimand in the crossed arm.
            if _finite(v.get('zi_prob_mean')) and _finite(v.get('exp_zeros')):
                slack = 3 * (v.get('exp_zeros_se') or 0.0)
                got = v['zi_prob_mean'] * v['n_obs']
                check(f'{key}: expected structural zeros <= simulated total '
                      f'zeros', got <= v['exp_zeros'] + slack,
                      f"{got:.2f} structural vs {v['exp_zeros']:.2f} total")
        else:
            check(f'{key}: plain family has an empty ZI block', not zi)
        # obs_zeros re-derived from the panel, not trusted.
        want_zeros = int((analytic_sample(v['cell'], v['model'])[
            CELL_SPEC[v['cell']][0]] == 0).sum())
        check(f'{key}: obs_zeros re-derived from the panel',
              v.get('obs_zeros') == want_zeros,
              f"got {v.get('obs_zeros')}, derived {want_zeros}")


# ============================================================
# [2] CORRECTNESS ANCHOR -- re-derived against the crossed arm
# ============================================================
def validate_anchor():
    section('2', 'Correctness anchor: the `mirror` series reproduces '
                 'jafari_crossed_results.json')
    r, c = results(), crossed()
    n_compared = 0
    for cell in CELLS:
        ck = CELL_SPEC[cell][2]
        meta = r[f'{cell}__meta']
        if ck is None:
            skip(f'{cell}: crossed-arm anchor',
                 'no precedent -- the crossed arm never fit this cell, which '
                 'is why the family was raced at M3 here (section [7])')
            # jsonlite writes an R NULL as `{}`, so "no anchor" arrives as an
            # empty object rather than as JSON null. Both are the same fact.
            check(f'{cell}: meta records that it has no crossed-arm anchor',
                  not meta.get('crossed_arm'),
                  f"got {meta.get('crossed_arm')!r}")
            continue
        cmeta = c.get(f'{ck}__meta')
        check(f'{cell}: crossed arm has a __meta for {ck}', cmeta is not None)
        if cmeta is None:
            continue
        # The family this arm inherited must BE the crossed arm's M3 winner --
        # the provenance claim in family_source, verified rather than believed.
        check(f'{cell}: inherited family == the crossed arm\'s M3 winner',
              meta['family_tag'] == cmeta['m3_winner'],
              f"inherited {meta['family_tag']}, crossed M3 winner "
              f"{cmeta['m3_winner']}")
        # The anchor copy stored in this arm's meta must match the live crossed
        # JSON, or the anchor table is checking against a stale snapshot.
        for model in SUBSTANTIVE:
            a = (meta.get('crossed_arm') or {}).get(f'anchor_{model}')
            check(f'{cell} {model}: meta carries a crossed-arm anchor',
                  a is not None)
            if a is None:
                continue
            src = c.get(a['key'])
            check(f'{cell} {model}: anchor key {a["key"]} exists in the '
                  f'crossed JSON', src is not None)
            if src is None:
                continue
            check_close(f'{cell} {model}: stored anchor sigma2_state is not '
                        f'stale', a['sigma2_state'], src['sigma2_state'],
                        0.0, 'abs')
            check(f'{cell} {model}: stored anchor n_obs is not stale',
                  a['n_obs'] == src['n_obs'])
            # THE anchor itself: this arm's mirror fit vs the crossed arm's.
            mine = rec(cell, model, 'mirror')
            check(f'{cell} {model}: mirror fit present and converged',
                  mine is not None and mine.get('converged'))
            if mine is None or not mine.get('converged'):
                continue
            n_compared += 1
            check(f'{cell} {model}: same conditional formula as the crossed arm',
                  mine['formula'] == src['formula'],
                  f"{mine['formula']!r} vs {src['formula']!r}")
            check(f'{cell} {model}: same ZI formula as the crossed arm',
                  mine['zi_formula'] == src['zi_formula'],
                  f"{mine['zi_formula']!r} vs {src['zi_formula']!r}")
            check(f'{cell} {model}: same analytic N as the crossed arm',
                  mine['n_obs'] == src['n_obs'],
                  f"{mine['n_obs']} vs {src['n_obs']}")
            check_close(f'{cell} {model}: sigma2_state reproduces the crossed '
                        f'arm', mine['sigma2_state'], src['sigma2_state'],
                        5e-4, 'abs')
            check_close(f'{cell} {model}: AIC reproduces the crossed arm',
                        mine['aic'], src['aic'], 1e-6, 'rel')
            check_close(f'{cell} {model}: logLik reproduces the crossed arm',
                        mine['loglik'], src['loglik'], 1e-6, 'rel')
            check(f'{cell} {model}: same ZI tier reached as the crossed arm',
                  mine.get('zi_tier_reached') == src.get('zi_tier_reached'),
                  f"{mine.get('zi_tier_reached')} vs {src.get('zi_tier_reached')}")
            for term, co in (src.get('cond') or {}).items():
                check_close(f'{cell} {model}: b[{term}] reproduces the crossed '
                            f'arm', (mine['cond'].get(term) or {}).get('b'),
                            co['b'], 1e-6, 'abs')
    check('at least six anchor values were actually compared',
          n_compared >= 6, f'compared {n_compared}')


# ============================================================
# [3] THE FAMILY IS GENUINELY HELD FIXED
# ============================================================
def validate_family_fixed():
    section('3', 'The family is held FIXED across M1/M2/M3 within each cell')
    r = results()
    for cell in CELLS:
        meta = r[f'{cell}__meta']
        fam = meta['family_tag']
        for ser in series_of(cell):
            names, ztypes = set(), set()
            for m in MODELS:
                v = rec(cell, m, ser)
                check(f'{cell} {ser} {m}: record present', v is not None)
                if v is None:
                    continue
                check(f'{cell} {ser} {m}: family_tag == {fam}',
                      v['family_tag'] == fam, f"got {v['family_tag']}")
                if v.get('converged'):
                    names.add(v.get('family_name'))
                # ZI structure: whether the ZI block carries a state random
                # intercept must never vary within a cell -- that is part of
                # "the family", and a change in it would move sigma2_state.
                ztypes.add('(1 | state)' in (v.get('zi_formula') or ''))
            check(f'{cell} {ser}: one and only one glmmTMB family across all '
                  f'four models', len(names) == 1, f'got {sorted(names)}')
            check(f'{cell} {ser}: ZI random-intercept presence constant across '
                  f'all four models', len(ztypes) == 1, f'got {sorted(ztypes)}')
            if ztypes:
                check(f'{cell} {ser}: ZI random intercept present iff the '
                      f'family is a *_re family',
                      next(iter(ztypes)) == bool(meta['family_has_zi_re']),
                      f"zi_formula says {next(iter(ztypes))}, "
                      f"family_has_zi_re={meta['family_has_zi_re']}")
        # The headline series must have a genuinely CONSTANT ZI block -- that
        # is the entire reason it, not `mirror`, is the reduction series.
        if 'zi_intercept' in series_of(cell):
            zforms = {rec(cell, m, 'zi_intercept')['zi_formula']
                      for m in MODELS if rec(cell, m, 'zi_intercept')}
            check(f'{cell}: headline (zi_intercept) ZI formula identical at '
                  f'every step', len(zforms) == 1, f'got {sorted(zforms)}')
            tiers = {rec(cell, m, 'zi_intercept').get('zi_tier_reached')
                     for m in MODELS if rec(cell, m, 'zi_intercept')}
            check(f'{cell}: headline series is intercept-only at every step',
                  tiers == {'intercept'}, f'got {sorted(tiers)}')
            zi_terms = set()
            for m in MODELS:
                zi_terms |= set((rec(cell, m, 'zi_intercept').get('zi') or {}))
            check(f'{cell}: headline ZI block carries the intercept and '
                  f'nothing else', zi_terms == {'(Intercept)'},
                  f'got {sorted(zi_terms)}')
        # The conditional build-up is strictly nested M1 -> M2 -> M3.
        for ser in series_of(cell):
            sets = {}
            for m in MODELS:
                v = rec(cell, m, ser)
                if v is None or not v.get('converged'):
                    continue
                sets[m] = set(v['cond'])
                want = {'(Intercept)'} | set(rhs_for(cell, m))
                check(f'{cell} {ser} {m}: conditional terms exactly as the '
                      f'build-up specifies', sets[m] == want,
                      f'extra {sorted(sets[m] - want)}, '
                      f'missing {sorted(want - sets[m])}')
                check(f'{cell} {ser} {m}: no interaction terms',
                      not any(':' in t for t in sets[m]))
            if {'M1', 'M2', 'M3'} <= set(sets):
                check(f'{cell} {ser}: M1 terms nested in M2 nested in M3',
                      sets['M1'] < sets['M2'] < sets['M3'])
            if 'M1matched' in sets and 'M1' in sets:
                check(f'{cell} {ser}: M1matched carries Model 1 fixed effects',
                      sets['M1matched'] == sets['M1'])


# ============================================================
# [4] ANALYTIC SAMPLES, RE-DERIVED FROM THE PANEL
# ============================================================
def validate_samples():
    section('4', 'Analytic samples re-derived from the panel by listwise '
                 'deletion')
    for cell in CELLS:
        state_sets = {}
        for ser in series_of(cell):
            for m in MODELS:
                v = rec(cell, m, ser)
                if v is None:
                    continue
                d = analytic_sample(cell, m)
                state_sets[m] = set(d['state'])
                check(f'{cell} {ser} {m}: n_obs matches listwise deletion on '
                      f'the panel', v['n_obs'] == len(d),
                      f"JSON {v['n_obs']}, derived {len(d)}")
                check(f'{cell} {ser} {m}: n_states matches listwise deletion',
                      v['n_states'] == d['state'].nunique(),
                      f"JSON {v['n_states']}, derived {d['state'].nunique()}")
        check(f'{cell}: M1 keeps 49 states', len(state_sets['M1']) == 49,
              f"got {len(state_sets['M1'])}")
        check(f'{cell}: M2 keeps 46 states', len(state_sets['M2']) == 46,
              f"got {len(state_sets['M2'])}")
        check(f'{cell}: M3 keeps 46 states', len(state_sets['M3']) == 46,
              f"got {len(state_sets['M3'])}")
        check(f'{cell}: M2 and M3 share one analytic sample',
              state_sets['M2'] == state_sets['M3']
              and len(analytic_sample(cell, 'M2'))
              == len(analytic_sample(cell, 'M3')))
        check(f'{cell}: the states M2/M3 drop are exactly AK/RI/VT',
              state_sets['M1'] - state_sets['M3'] == EXPECTED_DROPPED,
              f"got {sorted(state_sets['M1'] - state_sets['M3'])}")
        # The matched baseline is Model 1's FORMULA on Model 2/3's ROWS.
        check(f'{cell}: M1matched sits on exactly the M2/M3 state set',
              state_sets['M1matched'] == state_sets['M3'],
              f"got {sorted(state_sets['M1matched'] ^ state_sets['M3'])}")
        check(f'{cell}: M1matched sits on exactly the M2/M3 rows',
              len(analytic_sample(cell, 'M1matched'))
              == len(analytic_sample(cell, 'M3')))
        check(f'{cell}: M1matched is a genuinely different sample from M1',
              len(analytic_sample(cell, 'M1matched'))
              != len(analytic_sample(cell, 'M1')),
              'if these coincided the two Delta bases would be one computation')
        for ser in series_of(cell):
            v = rec(cell, 'M1matched', ser)
            if v is not None:
                check(f'{cell} {ser}: M1matched records sample_model == M3',
                      v.get('sample_model') == 'M3',
                      f"got {v.get('sample_model')!r}")
                check(f'{cell} {ser}: M1matched records its sample_rhs',
                      (v.get('sample_rhs') or '')
                      == ' + '.join(rhs_for(cell, 'M3')),
                      f"got {v.get('sample_rhs')!r}")


# ============================================================
# [5] VARIANCE REDUCTIONS RE-DERIVED, NON-CIRCULARLY
# ============================================================
def _delta(base, v):
    return 100.0 * (base - v) / base


def validate_deltas():
    section('5', 'Delta sigma^2_state re-derived from raw variances and '
                 'matched to the memo')
    tables = memo_variance_tables()
    for cell in CELLS:
        for ser in series_of(cell):
            m1 = rec(cell, 'M1', ser)
            mm = rec(cell, 'M1matched', ser)
            if not (m1 and mm and m1.get('converged') and mm.get('converged')):
                check(f'{cell} {ser}: both Model-1 baselines converged',
                      False, 'cannot re-derive a Delta without them')
                continue
            base_own, base_mat = m1['sigma2_state'], mm['sigma2_state']
            # The two bases must be DIFFERENT computations, or one of the two
            # reported columns is a duplicate wearing another name.
            check(f'{cell} {ser}: the unmatched and matched baselines are '
                  f'different numbers', abs(base_own - base_mat) > 1e-9,
                  f'both {base_own}')
            check(f'{cell} {ser}: the matched baseline is fit on fewer rows '
                  f'than the unmatched one', mm['n_obs'] < m1['n_obs'],
                  f"{mm['n_obs']} vs {m1['n_obs']}")
            check(f'{cell} {ser}: the two baselines share Model 1\'s formula',
                  mm['formula'] == m1['formula'])
            for m in MODELS:
                v = rec(cell, m, ser)
                if v is None or not v.get('converged'):
                    continue
                want_own = _delta(base_own, v['sigma2_state'])
                want_mat = (None if m == 'M1'
                            else _delta(base_mat, v['sigma2_state']))
                row = tables[ser].get((cell, m))
                check(f'{cell} {ser} {m}: memo row present', row is not None)
                if row is None:
                    continue
                check(f'{cell} {ser} {m}: memo Delta % vs M1 == re-derived '
                      f'{want_own:+.1f}', row['d_own'] == f'{want_own:+.1f}',
                      f"memo {row['d_own']!r}")
                if want_mat is None:
                    # M1 sits on a different sample from the matched baseline;
                    # a "matched" reduction for it would compare two samples.
                    check(f'{cell} {ser} M1: memo withholds the matched Delta',
                          row['d_mat'] == '--', f"memo {row['d_mat']!r}")
                else:
                    check(f'{cell} {ser} {m}: memo Delta % vs matched M1 == '
                          f're-derived {want_mat:+.1f}',
                          row['d_mat'] == f'{want_mat:+.1f}',
                          f"memo {row['d_mat']!r}")
            # Self-consistency: a model's own baseline gives exactly 0.
            check(f'{cell} {ser}: M1 is its own unmatched baseline (Delta 0)',
                  abs(_delta(base_own, m1['sigma2_state'])) < 1e-12)
            check(f'{cell} {ser}: M1matched is its own matched baseline '
                  f'(Delta 0)', abs(_delta(base_mat, mm['sigma2_state'])) < 1e-12)
            # And the two bases must actually disagree about at least one
            # model, otherwise reporting both proves nothing.
            m3 = rec(cell, 'M3', ser)
            if m3 and m3.get('converged'):
                check(f'{cell} {ser}: the two Delta bases give materially '
                      f'different answers at M3',
                      abs(_delta(base_own, m3['sigma2_state'])
                          - _delta(base_mat, m3['sigma2_state'])) > 0.05,
                      'the sample-change correction is doing nothing')


# ============================================================
# [6] CONVERGENCE
# ============================================================
def validate_convergence():
    section('6', 'Convergence: every fit in every REPORTED series converged')
    r = results()
    bad = {k for k, v in r.items()
           if isinstance(v, dict) and 'converged' in v and not v['converged']}
    check('the set of non-converged records is exactly the one permitted '
          '(a family-SELECTION candidate, never a reported estimate)',
          bad == ALLOWED_NONCONVERGED,
          f'unexpected: {sorted(bad - ALLOWED_NONCONVERGED)}; '
          f'expected-but-converged: {sorted(ALLOWED_NONCONVERGED - bad)}')
    for cell in CELLS:
        for ser in series_of(cell):
            for m in MODELS:
                v = rec(cell, m, ser)
                check(f'{cell} {ser} {m}: converged',
                      v is not None and v.get('converged'),
                      (v or {}).get('message', 'missing'))
                if v is None or not v.get('converged'):
                    continue
                check(f'{cell} {ser} {m}: positive-definite Hessian',
                      bool(v.get('pd_hess')))
                check(f'{cell} {ser} {m}: conv_code == 0',
                      v.get('conv_code') == 0, f"got {v.get('conv_code')}")
                check(f'{cell} {ser} {m}: not ZI-boundary-degenerate',
                      not v.get('zi_degenerate'),
                      v.get('zi_degenerate_reason', ''))
                check(f'{cell} {ser} {m}: every standard error is finite',
                      all(_finite(c.get('se'))
                          for c in (v.get('cond') or {}).values()))
    # No estimate from a non-converged fit may appear in the memo. Checked on
    # the values that are distinctive enough for the test to mean something:
    # the variance components, which are not round numbers.
    t = memo_text()
    for key in sorted(bad):
        v = r[key]
        check(f'memo mentions {key} exactly once (in the failed-candidate '
              f'bullet)', t.count(key) == 1, f'found {t.count(key)} times')
        for field in ('sigma2_state', 'sigma2_zi_state', 'dispersion'):
            val = v.get(field)
            if not _finite(val) or abs(val) < 0.01:
                continue
            for spec in ('.4f', '.3f'):
                s = format(val, spec)
                check(f'memo does not print {key}.{field} = {s}',
                      s not in t, 'a non-converged estimate reached the memo')


# ============================================================
# [7] FAMILY SELECTION RE-DERIVED
# ============================================================
def validate_selection():
    section('7', f'Family selection for `{SELECTION_CELL}` re-derived from the '
                 f'candidate fits')
    r = results()
    meta = r.get(f'{SELECTION_CELL}__selection_meta')
    check('selection_meta present', meta is not None)
    if meta is None:
        return
    check('selection happened at Model 3', meta['model_selected_at'] == 'M3',
          f"got {meta['model_selected_at']!r}")
    cands = {c['family_tag']: c for c in meta['candidates']}
    check('all eight ladder families were raced',
          set(cands) == set(LADDER), f'got {sorted(cands)}')

    # Re-derive eligibility and the winner from the RECORDS, not from the
    # candidate table, so a corrupted table cannot certify itself.
    elig = []
    for fam in LADDER:
        v = r.get(f'{SELECTION_CELL}__M3__{fam}__selection')
        check(f'{fam}: candidate record present', v is not None)
        if v is None:
            continue
        row = cands.get(fam)
        check(f'{fam}: candidate table row round-trips from the record',
              row is not None
              and row['converged'] == bool(v.get('converged'))
              and row['n_obs'] == v['n_obs']
              and row['zi_degenerate'] == bool(v.get('zi_degenerate'))
              and (row['aic'] is None) == (v.get('aic') is None)
              and (row['aic'] is None
                   or abs(row['aic'] - v['aic']) < 1e-9),
              f'table {row!r}')
        ok = (bool(v.get('converged')) and _finite(v.get('aic'))
              and not v.get('collapsed_to') and not v.get('zi_degenerate'))
        check(f'{fam}: eligible_for_selection re-derived',
              bool(v.get('eligible_for_selection')) == ok,
              f"record says {v.get('eligible_for_selection')}, derived {ok}")
        if ok:
            elig.append((fam, v['aic']))
    check('n_eligible re-derived', meta['n_eligible'] == len(elig),
          f"meta {meta['n_eligible']}, derived {len(elig)}")
    check('at least two families were eligible (a race, not a walkover)',
          len(elig) >= 2, f'{len(elig)} eligible')
    want = min(elig, key=lambda kv: kv[1])[0] if elig else None
    check(f'winner re-derived by minimum AIC == {want}',
          meta['selected_family'] == want,
          f"meta says {meta['selected_family']}")
    check('the winner is the family actually used in the reported series',
          fam_of(SELECTION_CELL) == want,
          f'reported series use {fam_of(SELECTION_CELL)}, winner {want}')
    check('the winner is the family in every reported record key',
          all(rec(SELECTION_CELL, m, s)['family_tag'] == want
              for s in series_of(SELECTION_CELL) for m in MODELS))
    # The winner's margin, so a tie that got broken by dict order is visible.
    rest = sorted(a for f, a in elig if f != want)
    if rest and want is not None:
        margin = rest[0] - dict(elig)[want]
        check(f'the winner beats the runner-up by a non-trivial AIC margin '
              f'({margin:.2f})', margin > 2.0, f'margin {margin}')
    # The two inherited cells must NOT carry a selection race.
    for cell in CELLS:
        if cell == SELECTION_CELL:
            continue
        check(f'{cell}: no family race (it inherits from the crossed arm)',
              f'{cell}__selection_meta' not in r)


# ============================================================
# [8] GAUSSIAN CROSSED ROUND-TRIP -- glmmTMB vs statsmodels.MixedLM
# ============================================================
# validate_jafari_crossed.py [2] gates the crossed `(1 | state) + (1 | year)`
# structure, but only for ITS cells' formulas -- it never sees this arm's
# `viol_2021_ni` specification (violations WITHOUT log_inspections), never sees
# Model 2 at all, and never sees the M1matched construction where the fitted
# formula and the sample-defining variable set deliberately differ. Those are
# exactly the places a sample- or formula-assembly bug in this arm would live,
# so the bridge is re-gated here on this arm's own twelve specifications.
R_GAUSSIAN = r'''
suppressPackageStartupMessages({library(glmmTMB); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
d0 <- read.csv(args[1], stringsAsFactors = FALSE)
CUBIC  <- c("time", "time2", "time3")
M2_ADD <- c("SPEND_APP_z", "SPEND_WORK_z", "lii_2017_z")
M3_ADD <- c(M2_ADD, "h2a_per_farmworker_z", "dol_demand_met_pct_z", "pct_flc_z")
CELLS <- list(
  insp_2021    = list(dv = "log_inspections", base = CUBIC),
  viol_2021_wi = list(dv = "log_violations",  base = c("log_inspections", CUBIC)),
  viol_2021_ni = list(dv = "log_violations",  base = CUBIC))
res <- list()
for (cn in names(CELLS)) {
  cl <- CELLS[[cn]]
  for (m in c("M1", "M2", "M3", "M1matched")) {
    rhs <- switch(m, M1 = cl$base, M1matched = cl$base,
                  M2 = c(cl$base, M2_ADD), M3 = c(cl$base, M3_ADD))
    sel <- if (m == "M1matched") c(cl$base, M3_ADD) else rhs
    need <- unique(c(cl$dv, sel, "state", "year", "time"))
    need <- need[need %in% names(d0)]
    dd <- d0[stats::complete.cases(d0[, need, drop = FALSE]), , drop = FALSE]
    dd$state <- factor(dd$state); dd$year <- factor(dd$year)
    f <- stats::as.formula(sprintf("%s ~ %s + (1 | state) + (1 | year)",
                                   cl$dv, paste(rhs, collapse = " + ")))
    fit <- glmmTMB(f, family = gaussian, data = dd, REML = TRUE)
    co <- summary(fit)$coefficients$cond
    res[[sprintf("%s__%s", cn, m)]] <- list(
      cell = cn, model = m, dv = cl$dv, rhs = rhs,
      formula = deparse1(f), n_obs = nrow(dd),
      n_states = length(unique(dd$state)),
      converged = isTRUE(fit$sdr$pdHess) && fit$fit$convergence == 0,
      cond = stats::setNames(lapply(seq_len(nrow(co)),
               function(i) list(b = co[i, 1], se = co[i, 2])), rownames(co)),
      sigma2_state = VarCorr(fit)$cond$state[1, 1],
      sigma2_year  = VarCorr(fit)$cond$year[1, 1],
      sigma2_e     = sigma(fit)^2)
  }
}
write_json(res, args[2], auto_unbox = TRUE, digits = 12, na = "null")
'''

GAUSS_DV = {'insp_2021': 'log_inspections',
            'viol_2021_wi': 'log_violations',
            'viol_2021_ni': 'log_violations'}


SM_OPTIMIZERS = (None, 'powell')


def _statsmodels_crossed(d, dv, rhs):
    """Fit `dv ~ rhs + (1|state) + (1|year)` in statsmodels, REML.

    statsmodels has no crossed-RE syntax; the documented recipe is one constant
    group carrying two variance components, each an indicator basis over one
    grouping factor -- algebraically the model glmmTMB fits from
    `(1 | state) + (1 | year)`. Same idiom as validate_jafari_crossed.py [2].

    LIVE-VS-LIVE OPTIMIZER SELECTION, as validate_count_models.py [2c] does for
    the random-slope bridge. statsmodels is fit under several optimizers and
    glmmTMB is compared against the HIGHEST-log-likelihood survivor, not
    against whatever the default returned.

    This is not a loosened test, it is a stricter one, and it was added for a
    concrete reason. The default `lbfgs` stops early along the near-flat
    sigma^2_year direction in `viol_2021_wi` M2 and M1matched: it returns
    1.7232e-04 where glmmTMB returns 1.6608e-04 (3.8% apart) while reporting
    the SAME REML log-likelihood to eight decimals. `powell` reaches a
    marginally better log-likelihood at 1.6611e-04 -- 2e-04 relative from
    glmmTMB. The packages were never disagreeing about the model; one
    optimizer was quitting early on a flat ridge. Comparing against the best
    optimum tests the model instead of the stopping rule.
    """
    import statsmodels.formula.api as smf
    dd = d.copy()
    dd['_grp'] = 1
    vcf = {'state': '0 + C(state)', 'year': '0 + C(year)'}
    md = smf.mixedlm(f"{dv} ~ {' + '.join(rhs)}", dd, groups='_grp',
                     vc_formula=vcf)
    best, msgs = None, []
    for opt in SM_OPTIMIZERS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                mdf = (md.fit(reml=True) if opt is None
                       else md.fit(reml=True, method=opt, maxiter=10000))
            except Exception as exc:          # noqa: BLE001 -- reported, not hidden
                msgs.append(f'[{opt or "default"}] {type(exc).__name__}: {exc}')
                continue
        msgs += [f'[{opt or "default"}] {w.message}' for w in caught]
        if not all(map(math.isfinite, (mdf.llf,) + tuple(mdf.vcomp))):
            msgs.append(f'[{opt or "default"}] non-finite fit discarded')
            continue
        if best is None or mdf.llf > best[1].llf:
            best = (opt or 'default', mdf)
    if best is None:
        raise RuntimeError('no statsmodels optimizer produced a usable fit')
    opt, mdf = best
    # vcomp is positional; zip against exog_vc.names so a future statsmodels
    # reordering cannot silently swap sigma2_state and sigma2_year.
    return (dict(mdf.fe_params), dict(zip(md.exog_vc.names, mdf.vcomp)),
            mdf.scale, msgs, opt)


def validate_gaussian_roundtrip(skip_gaussian):
    section('8', "Gaussian crossed round-trip on THIS arm's formulas "
                 '(glmmTMB vs statsmodels.MixedLM)')
    if skip_gaussian:
        skip('Gaussian crossed round-trip',
             '--skip-gaussian passed; this run is NOT a passing run')
        return
    tmp = tempfile.mkdtemp(prefix='jafari_stepwise_gauss_')
    rfile, jfile = os.path.join(tmp, 'g.R'), os.path.join(tmp, 'g.json')
    with open(rfile, 'w') as fh:
        fh.write(R_GAUSSIAN)
    proc = subprocess.run(['Rscript', rfile, PANEL, jfile],
                          capture_output=True, text=True, cwd=ROOT)
    check('Rscript exits 0', proc.returncode == 0,
          f'exit {proc.returncode}; stderr: {proc.stderr[-800:]}')
    # A benign TMB ABI-mismatch warning is expected on every Rscript call in
    # this project; anything else on stderr is a leak and must be seen.
    leak = [ln for ln in proc.stderr.splitlines()
            if ln.strip() and not any(p in ln for p in BENIGN_R_STDERR)]
    check('R stderr carries nothing but the benign TMB ABI warning',
          not leak, f'unexpected stderr: {leak}')
    if proc.returncode != 0 or not os.path.exists(jfile):
        check('Gaussian round-trip artifact produced', False,
              'the gate did NOT run -- treat nothing downstream as verified')
        return
    with open(jfile) as fh:
        g = json.load(fh)
    for cell in CELLS:
        dv = GAUSS_DV[cell]
        for m in MODELS:
            key = f'{cell}__{m}'
            rg = g.get(key)
            check(f'{key}: glmmTMB Gaussian fit present and converged',
                  rg is not None and rg.get('converged'))
            if rg is None or not rg.get('converged'):
                continue
            rhs = rhs_for(cell, m)
            d = analytic_sample(cell, m)
            # The Gaussian fit must sit on the SAME rows as the count fit --
            # that is what makes this a gate on the sample logic and not only
            # on the linear algebra.
            check(f'{key}: same analytic N as the count fit',
                  rg['n_obs'] == rec(cell, m, series_of(cell)[0])['n_obs']
                  == len(d),
                  f"gaussian {rg['n_obs']}, count "
                  f"{rec(cell, m, series_of(cell)[0])['n_obs']}, derived {len(d)}")
            try:
                params, vcomp, scale, msgs, opt = _statsmodels_crossed(
                    d, dv, rhs)
            except Exception as exc:
                # An unavailable cross-check is a FAILURE, never a silent pass.
                check(f'{key}: statsmodels crossed-RE fit available', False,
                      f'{type(exc).__name__}: {exc}')
                continue
            print(f'    ({key}: statsmodels optimizer kept = {opt}; '
                  f'{len(msgs)} warning(s)'
                  + (f'; first: {msgs[0][:90]}' if msgs else '') + ')')
            for term, val in rg['cond'].items():
                sm_name = 'Intercept' if term == '(Intercept)' else term
                check_close(f'{key}: b[{term}]', val['b'],
                            params.get(sm_name), 1e-3)
            check_var_close(f'{key}: sigma2_state', rg['sigma2_state'],
                            vcomp.get('state'), scale_ref=scale)
            check_var_close(f'{key}: sigma2_year', rg['sigma2_year'],
                            vcomp.get('year'), scale_ref=scale)
            check_close(f'{key}: sigma2_e', rg['sigma2_e'], scale, 2e-2)


# ============================================================
# [9] MEMO CONSISTENCY
# ============================================================
def _md_tables(text):
    """Return [(heading, [row-cell-lists])] for every markdown table."""
    out, heading, cur = [], '', None
    for line in text.splitlines():
        if line.startswith('#'):
            heading = line.strip()
            continue
        if line.startswith('|'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if all(set(c) <= set('-: ') and c for c in cells):
                continue          # separator row
            if cur is None:
                cur = (heading, [])
                out.append(cur)
            cur[1].append(cells)
        else:
            cur = None
    return out


def _section_body(text, prefix):
    """The text of the top-level section whose heading starts with `prefix`,
    up to the next top-level heading."""
    out, inside = [], False
    for line in text.splitlines():
        if line.startswith('## '):
            inside = line.startswith(prefix)
            if inside:
                continue
        if inside:
            out.append(line)
    return '\n'.join(out)


_LABEL_TO_CELL = {v: k for k, v in CELL_LABEL.items()}
_LABEL_TO_MODEL = {v: k for k, v in MODEL_LABEL.items()}
_LABEL_TO_TERM = {v: k for k, v in TERM_LABEL.items()}


def memo_variance_tables():
    """The section-2 (zi_intercept) and section-3 (mirror) variance tables,
    keyed (cell, model)."""
    out = {'zi_intercept': {}, 'mirror': {}}
    for heading, rows in _md_tables(memo_text()):
        if heading.startswith('## 2.'):
            ser = 'zi_intercept'
        elif heading.startswith('## 3.') and len(rows[0]) == 9 \
                and rows[0][0] == 'Cell':
            ser = 'mirror'
        else:
            continue
        for r in rows[1:]:
            if len(r) != 9:
                continue
            cell = _LABEL_TO_CELL.get(r[0])
            model = _LABEL_TO_MODEL.get(r[1].replace(
                ' **(DID NOT CONVERGE)**', ''))
            if cell is None or model is None:
                continue
            out[ser][(cell, model)] = {
                'n_obs': r[2], 'n_states': r[3], 's2': r[4],
                'd_own': r[5], 'd_mat': r[6], 'aic': r[7], 'd_aic': r[8]}
    return out


MEMO_REQUIRED = [
    # The generated banner -- the memo must never be hand-edited.
    '**GENERATED by `scripts/report_jafari_stepwise.py`. Never hand-edit -- '
    're-run it.**',
    # THE sign convention. This sentence exists because the two blocks were
    # read backwards in a team meeting and a finding was reported with its sign
    # inverted (see CLAUDE.md, 2026-09-17). It must not silently disappear.
    'A positive `b` means MORE structural zeros, i.e. FEWER events.',
    'MORE structural zeros**, i.e. **FEWER** events',
    'the log-odds of being a STRUCTURAL ZERO',
    'the COUNT model',
    # Caveats the arm's honesty depends on.
    'LOG LINK scale',
    'Alaska, Rhode Island and Vermont',
    'behaves as a state-only random intercept',
    'No COVID indicator',
    'SAMPLE MEAN',
    'the family was raced at',
    # The mirror series' ZI-tier fallback, without which section 3's Deltas
    # look like a clean series and are not.
    'the full mirror was not estimable at that step',
    # A count family has no residual variance, and the memo must actively DENY
    # that the dispersion parameter is one. This is a required DENIAL, not a
    # forbidden phrase: forbidding the words 'residual variance' outright would
    # forbid the denial as well (the same trap validate_nb2_stepwise.py [9]
    # documents, where the memo has to hyphenate to get past its own guard).
    '**not** a residual variance',
]

MEMO_FORBIDDEN = [
    'TODO', 'TBD', 'FIXME',
    # The Delta is a between-state variance reduction on the LOG LINK scale and
    # is never the manuscript's Gaussian sigma2_u0_ri.
    'same as the published',
]

# Values that must never appear as a TABLE CELL. Checked structurally rather
# than as a substring of the whole memo, because 'NA/NaN function evaluation'
# is a legitimate verbatim quote of an R optimizer message in the
# failed-candidate bullet, and a naive substring guard fires on it.
BAD_CELL_VALUES = {'nan', 'NaN', 'None', 'none', 'NA', 'null', 'inf', '-inf',
                   'Inf', '-Inf'}


def validate_memo():
    section('9', 'Memo consistency: every printed number traces to the JSON')
    t = memo_text()
    for pat in MEMO_REQUIRED:
        check(f'memo states: {pat[:70]!r}', pat in t, 'not found')
    for pat in MEMO_FORBIDDEN:
        check(f'memo free of {pat!r}', pat not in t)
    bad_cells = sorted({c for _, rows in _md_tables(t) for row in rows
                        for c in row if c in BAD_CELL_VALUES})
    check('no table cell in the memo is a missing/placeholder value',
          not bad_cells, f'found cells: {bad_cells}')

    r = results()
    tables = memo_variance_tables()
    # --- (a) the two variance tables ---------------------------------------
    for cell in CELLS:
        for ser in series_of(cell):
            base_aic = rec(cell, 'M1matched', ser)
            for m in MODELS:
                v = rec(cell, m, ser)
                row = tables[ser].get((cell, m))
                check(f'{cell} {ser} {m}: present in the memo variance table',
                      row is not None)
                if row is None or v is None or not v.get('converged'):
                    continue
                check(f'{cell} {ser} {m}: memo N', row['n_obs'] == str(v['n_obs']),
                      f"memo {row['n_obs']}, JSON {v['n_obs']}")
                check(f'{cell} {ser} {m}: memo states',
                      row['n_states'] == str(v['n_states']))
                check(f'{cell} {ser} {m}: memo sigma^2_state',
                      row['s2'] == f"{v['sigma2_state']:.4f}",
                      f"memo {row['s2']}, JSON {v['sigma2_state']:.4f}")
                check(f'{cell} {ser} {m}: memo AIC',
                      row['aic'] == f"{v['aic']:.2f}",
                      f"memo {row['aic']}, JSON {v['aic']:.2f}")
                # AIC vs the matched baseline, re-derived. Withheld for M1,
                # which sits on a different number of rows.
                if m == 'M1':
                    check(f'{cell} {ser} M1: memo withholds the AIC delta '
                          f'(different sample)', row['d_aic'] == '--',
                          f"memo {row['d_aic']!r}")
                elif base_aic and base_aic.get('converged'):
                    want = v['aic'] - base_aic['aic']
                    check(f'{cell} {ser} {m}: memo AIC vs matched M1 '
                          f're-derived', row['d_aic'] == f'{want:+.2f}',
                          f"memo {row['d_aic']}, derived {want:+.2f}")

    # --- (b) the correctness-anchor table ----------------------------------
    anchor_rows = [rows for h, rows in _md_tables(t)
                   if h.startswith('### Correctness anchor')]
    check('memo carries the correctness-anchor table', len(anchor_rows) == 1)
    if anchor_rows:
        verdicts = [row[6] for row in anchor_rows[0][1:]]
        check('every anchor row reads MATCH', set(verdicts) == {'MATCH'},
              f'got {verdicts}')
        check('the anchor table covers all six comparable values',
              len(verdicts) == 6, f'got {len(verdicts)}')
        check('memo reports 6 of 6 anchor values reproduced',
              '**6 of 6 anchor values reproduced**' in t)

    # --- (c) the sigma^2_year table in 5.3 ---------------------------------
    yr = [rows for h, rows in _md_tables(t)
          if h.startswith('### 5.3')]
    check('memo carries the sigma^2_year table', len(yr) == 1)
    if yr:
        for row in yr[0][1:]:
            cell = _LABEL_TO_CELL.get(row[0])
            if cell is None:
                continue
            v = rec(cell, 'M3', 'zi_intercept')
            check(f'{cell}: memo 5.3 sigma^2_state',
                  row[1] == f"{v['sigma2_state']:.4f}")
            check(f'{cell}: memo 5.3 sigma^2_year',
                  row[2] == f"{v['sigma2_year']:.3e}")
            want = 100 * v['sigma2_year'] / (v['sigma2_state'] + v['sigma2_year'])
            check(f'{cell}: memo 5.3 year share re-derived',
                  row[3] == f'{want:.3f}%', f'memo {row[3]}, derived {want:.3f}%')

    # --- (d) section 6: Model-3 coefficient blocks -------------------------
    label_for = {'zi_intercept': 'headline, ZI block = intercept only',
                 'mirror': 'Jafari mirror, ZI block = full predictor list'}
    n_coef = 0
    for cell in CELLS:
        for ser in series_of(cell):
            head = f'### {CELL_LABEL[cell]} -- Model 3 ({label_for[ser]})'
            check(f'{cell} {ser}: memo has a Model-3 coefficient section',
                  head in t)
            blocks = [rows for h, rows in _md_tables(t) if h == head]
            check(f'{cell} {ser}: that section carries two coefficient blocks '
                  f'(conditional + ZI)', len(blocks) == 2,
                  f'found {len(blocks)}')
            v = rec(cell, 'M3', ser)
            if len(blocks) != 2 or v is None or not v.get('converged'):
                continue
            # The prose line above the tables.
            mm = re.search(
                re.escape(head) + r'.*?N = (\d+) \((\d+) states, (\d+) years\), '
                r'observed zeros = (\d+), AIC = ([-\d.]+), '
                r'logLik = ([-\d.]+) on (\d+) df', t, re.S)
            check(f'{cell} {ser}: memo prints the fit header', mm is not None)
            if mm:
                check(f'{cell} {ser}: header N/states/zeros/AIC/logLik/df '
                      f'round-trip from the JSON',
                      (int(mm.group(1)), int(mm.group(2)), int(mm.group(4)),
                       mm.group(5), mm.group(6), int(mm.group(7)))
                      == (v['n_obs'], v['n_states'], v['obs_zeros'],
                          f"{v['aic']:.2f}", f"{v['loglik']:.2f}", v['df']),
                      f'memo {mm.groups()}')
            for block, rows in zip(('cond', 'zi'), blocks):
                co = v.get(block) or {}
                printed = set()
                for row in rows[1:]:
                    term = _LABEL_TO_TERM.get(row[0])
                    check(f'{cell} {ser} {block}: memo term {row[0]!r} is a '
                          f'known term', term is not None)
                    if term is None:
                        continue
                    printed.add(term)
                    c = co.get(term)
                    check(f'{cell} {ser} {block}[{term}]: present in the JSON',
                          c is not None)
                    if c is None:
                        continue
                    n_coef += 1
                    check(f'{cell} {ser} {block}[{term}]: b/SE/z/p/exp(b) all '
                          f'round-trip from the JSON',
                          (row[1], row[2], row[3], row[4], row[6])
                          == (f"{c['b']:.4f}", f"{c['se']:.4f}",
                              f"{c['z']:.2f}", f"{c['p']:.4f}",
                              f'{math.exp(c["b"]):.4f}'),
                          f'memo {row[1:5]}/{row[6]} vs JSON '
                          f"{c['b']:.4f}/{c['se']:.4f}/{c['z']:.2f}/"
                          f"{c['p']:.4f}/{math.exp(c['b']):.4f}")
                    want_stars = ('***' if c['p'] < .001 else
                                  '**' if c['p'] < .01 else
                                  '*' if c['p'] < .05 else '')
                    check(f'{cell} {ser} {block}[{term}]: significance stars',
                          row[5] == want_stars,
                          f'memo {row[5]!r}, derived {want_stars!r}')
                check(f'{cell} {ser} {block}: memo prints every term in the '
                      f'block and no others', printed == set(co),
                      f'memo-only {sorted(printed - set(co))}, '
                      f'json-only {sorted(set(co) - printed)}')
    # Expected coefficient count DERIVED from the six Model-3 records, so the
    # guard tracks the artifact instead of resting on a round number I picked.
    want_coef = sum(len(rec(c, 'M3', s).get('cond') or {})
                    + len(rec(c, 'M3', s).get('zi') or {})
                    for c in CELLS for s in series_of(c)
                    if rec(c, 'M3', s) and rec(c, 'M3', s).get('converged'))
    check(f'every Model-3 coefficient in the JSON was round-tripped against '
          f'the memo ({want_coef} expected)', n_coef == want_coef,
          f'{n_coef} checked, {want_coef} expected')

    # --- (e) section 4: the with/without-inspections comparison ------------
    # This is the section that answers Joe's second request, and its whole
    # reading rests on the two violations cells differing in ONE term. If the
    # two cells do not in fact differ in one term only, the memo must say so --
    # otherwise the "log(inspections) absorbs ~45% of the between-state
    # variance" headline attributes to that term whatever else differs.
    wi, ni = 'viol_2021_wi', 'viol_2021_ni'
    sec4 = _section_body(t, '## 4.')
    check('memo has a section 4', bool(sec4))
    same_family = fam_of(wi) == fam_of(ni)
    zi_wi = rec(wi, 'M3', 'zi_intercept')['zi_formula']
    zi_ni = rec(ni, 'M3', 'zi_intercept')['zi_formula']
    same_zi = zi_wi == zi_ni
    if same_family and same_zi:
        check('the two violations cells differ in exactly one term, so the '
              'memo may say so', 'differ in exactly one term' in sec4)
    else:
        # They do NOT. The comparison changes the family / ZI structure as
        # well as the log(inspections) term, so the "absorbed share" figure is
        # not attributable to that term alone and the memo must not imply it
        # is. Reported as a defect rather than waived.
        check('memo does NOT claim the two violations cells "differ in '
              'exactly one term" when in fact they do not',
              'differ in exactly one term' not in sec4,
              f'{wi} is {fam_of(wi)} (ziformula {zi_wi}) and {ni} is '
              f'{fam_of(ni)} (ziformula {zi_ni}) -- the zero-inflation '
              f'structure differs too, so the absorbed-variance headline is '
              f'confounded with it')
        check('section 4 discloses that the two violations cells also differ '
              'in family / zero-inflation structure',
              FAMILY_LABEL[fam_of(wi)] in sec4
              and FAMILY_LABEL[fam_of(ni)] in sec4
              and ('also differ' in sec4 or 'differ in **two**' in sec4
                   or 'zero-inflation structure' in sec4),
              'no disclosure found in section 4 of the memo')
    shares = [rows for h, rows in _md_tables(t) if h.startswith('## 4.')]
    check('memo carries the with/without comparison tables', len(shares) >= 2)
    if shares:
        for row in shares[0][1:]:
            m = _LABEL_TO_MODEL.get(row[0])
            if m is None:
                continue
            a = rec(wi, m, 'zi_intercept')
            b = rec(ni, m, 'zi_intercept')
            if not (a and b and a['converged'] and b['converged']):
                continue
            check(f'section 4 {m}: memo sigma^2_state WITH',
                  row[1] == f"{a['sigma2_state']:.4f}")
            check(f'section 4 {m}: memo sigma^2_state WITHOUT',
                  row[2] == f"{b['sigma2_state']:.4f}")
            want = 100 * (b['sigma2_state'] - a['sigma2_state']) / b['sigma2_state']
            check(f'section 4 {m}: memo absorbed share re-derived',
                  row[4] == f'{want:.1f}%', f'memo {row[4]}, derived {want:.1f}%')

    # --- (f) the family table in section 1 ---------------------------------
    for cell in CELLS:
        meta = r[f'{cell}__meta']
        check(f'{cell}: memo names the family held fixed '
              f'({FAMILY_LABEL[meta["family_tag"]]})',
              f"| `{cell}` | {meta['outcome']} | "
              f"**{FAMILY_LABEL[meta['family_tag']]}** |" in t)


# ============================================================
# [10] FROZEN ARTIFACTS
# ============================================================
# Every other arm's files. The ratio arm (jafari_ratio_*) is owned by another
# worker and is listed here only so this arm can prove it did not touch it.
FROZEN = [
    'scripts/count_models_zinb.R', 'scripts/report_count_models.py',
    'scripts/validate_count_models.py',
    'scripts/build_count_model_memo_evidence.py',
    'scripts/build_count_model_panel.py',
    'docs/count_models_zinb.md',
    'data/generated/count_model_results.json',
    'data/generated/count_model_panel_2019.csv',
    'data/generated/count_model_panel_2021.csv',
    'scripts/nb2_stepwise_models.R', 'scripts/report_nb2_stepwise.py',
    'scripts/validate_nb2_stepwise.py', 'scripts/nb2_crosscheck_statsmodels.py',
    'docs/nb2_stepwise_models.md', 'data/generated/nb2_stepwise_results.json',
    'scripts/jafari_crossed_models.R', 'scripts/report_jafari_crossed.py',
    'scripts/validate_jafari_crossed.py', 'docs/jafari_crossed_models.md',
    'data/generated/jafari_crossed_results.json',
    'data/generated/jafari_crossed_gaussian.json',
]
FROZEN_PREFIXES = ('scripts/jafari_ratio_', 'data/generated/jafari_ratio_',
                   'docs/jafari_ratio_', 'scripts/paper_table_',
                   'data/generated/paper_table_')


def validate_frozen():
    section('10', 'Frozen artifacts untouched by this arm')
    # SCOPE, stated honestly: this verifies "this BRANCH did not touch the
    # frozen artifacts", not "they have never changed". On `main` the merge-base
    # IS HEAD, so the committed-history arm is vacuous and only the working-tree
    # arm bites -- the same documented limitation CLAUDE.md records for
    # validate_nb2_stepwise.py [8] and validate_jafari_crossed.py [10]. This arm
    # is developed on a feature branch, where the question it asks is the right
    # one.
    def flagged(paths):
        bad = [p for p in paths if p in FROZEN]
        bad += [p for p in paths
                if p.startswith(FROZEN_PREFIXES) or p.endswith('.docx')]
        return sorted(set(bad))

    dirty = subprocess.run(['git', 'status', '--porcelain'],
                           capture_output=True, text=True, cwd=ROOT).stdout
    dirty_paths = {ln[3:].strip() for ln in dirty.split('\n') if ln.strip()}
    for f in FROZEN:
        check(f'unmodified in the working tree: {f}', f not in dirty_paths)
    check('no ratio-arm / paper-table / .docx artifact touched in the working '
          'tree', not flagged(dirty_paths), f'touched: {flagged(dirty_paths)}')

    mb = subprocess.run(['git', 'merge-base', 'main', 'HEAD'],
                        capture_output=True, text=True, cwd=ROOT)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, cwd=ROOT)
    base, tip = mb.stdout.strip(), head.stdout.strip()
    check('git merge-base main HEAD resolves (required for the '
          'committed-history guard)', mb.returncode == 0 and bool(base),
          f'returncode {mb.returncode}; stderr {mb.stderr!r}')
    if not base:
        check('committed-history guard ran', False,
              'merge-base could not be resolved -- the guard did NOT run')
        return
    if base == tip:
        skip('committed-history arm',
             'merge-base == HEAD (on the default branch); this arm is vacuous '
             'here -- only the working-tree arm above bites')
        return
    # SCOPE, second refinement. Diffing the whole merge-base..HEAD range is
    # the right question only when the branch carries ONE arm. This branch
    # (`jafari-stepwise-and-ratio-arms`) carries three: the stepwise arm, a
    # concurrently-developed violations-per-inspection arm owned by another
    # worker, and a presentation fix to the crossed memo. A whole-range diff
    # therefore reports other people's legitimate commits as freeze
    # violations, which is a false alarm, not a finding.
    #
    # The promise this guard exists to enforce is the one CLAUDE.md states for
    # the NB2 arm: "the freeze bound the arm's OWN commits and it kept that
    # promise". So: every commit that touches a jafari_stepwise_* path must
    # touch nothing frozen. That is checkable, and it does not pass vacuously
    # -- if this arm's commit had also edited the crossed arm, it would fire.
    revs = subprocess.run(['git', 'rev-list', f'{base}..HEAD'],
                          capture_output=True, text=True,
                          cwd=ROOT).stdout.split()
    mine = []
    for rev in revs:
        files = subprocess.run(
            ['git', 'show', '--pretty=format:', '--name-only', rev],
            capture_output=True, text=True, cwd=ROOT).stdout.split()
        if any('jafari_stepwise' in f for f in files):
            mine.append((rev, files))
    check("this arm has at least one commit in this branch's history (the "
          'guard is not vacuous)', bool(mine),
          f'no commit in {base[:8]}..HEAD touches a jafari_stepwise_* path')
    for rev, files in mine:
        for f in FROZEN:
            check(f'commit {rev[:8]} (a jafari_stepwise commit) leaves {f} '
                  f'unmodified', f not in files)
        check(f'commit {rev[:8]} touches no ratio-arm / paper-table / .docx '
              f'artifact', not flagged(files), f'touched: {flagged(files)}')
    # And, reported rather than asserted: what the REST of the branch touched,
    # so a reader can see this guard's scope instead of inferring it.
    others = [r for r in revs if r not in {m[0] for m in mine}]
    if others:
        print(f'    (scope note: {len(others)} commit(s) in '
              f'{base[:8]}..HEAD belong to other arms on this shared branch '
              f'and are outside this guard by design)')


# ============================================================
def main():
    skip_pre = '--skip-precondition' in sys.argv
    skip_g = '--skip-gaussian' in sys.argv
    validate_precondition(skip_pre)
    validate_records()
    validate_anchor()
    validate_family_fixed()
    validate_samples()
    validate_deltas()
    validate_convergence()
    validate_selection()
    validate_gaussian_roundtrip(skip_g)
    validate_memo()
    validate_frozen()

    print('\n' + '=' * 82)
    print('PER-SECTION BREAKDOWN')
    print('DO NOT QUOTE THE TOTAL ALONE: sections [1], [4] and [9] are record '
          'loops over')
    print('the artifact, not sets of independent findings, and they dominate '
          'the count.')
    print('=' * 82)
    print(f"{'section':<66}{'PASS':>5}{'FAIL':>5}{'SKIP':>5}")
    for s in SECTIONS:
        print(f"{s['name'][:66]:<66}{s['pass']:>5}{s['fail']:>5}{s['skip']:>5}")
    tp = sum(s['pass'] for s in SECTIONS)
    tf = sum(s['fail'] for s in SECTIONS)
    ts = sum(s['skip'] for s in SECTIONS)
    print(f"{'TOTAL (see the caveat above)':<66}{tp:>5}{tf:>5}{ts:>5}")
    if ts:
        print(f'\n{ts} gated block(s) were SKIPPED -- each is printed above '
              f'with its reason.')
    if FAILURES:
        print(f'\n{len(FAILURES)} CHECK(S) FAILED:')
        for f in FAILURES:
            print(f'  - {f}')
        sys.exit(1)
    print('\nALL CHECKS PASSED')


if __name__ == '__main__':
    main()
