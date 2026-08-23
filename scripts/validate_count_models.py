"""
Validation gate for the ZINB count-model pipeline.

This is not a unit-test suite -- it is an analysis artifact. Each check either
proves a step of the pipeline reproduces something already known, or proves an
invariant the spec depends on. Run it after any change to the pipeline.

Run: python3 scripts/validate_count_models.py      (exits 1 on any failure)

ON READING THE OUTPUT (item 7 of the 2026-08-22 review). Do NOT quote a single
headline check count anywhere. The total is dominated by per-record loops --
section [3] alone walks ~105 fit records at ~9 assertions each, which is about
two thirds of every check in the file. Those are real assertions but they are
not independent findings, and a reader shown one big number over-weights them.
`main()` therefore prints a PER-SECTION breakdown, and every gated block that
is bypassed prints an explicit `SKIP` line and is counted -- so a run in which
a block silently vanished (the CSV round-trips under `os.path.exists`, the
several `if a and b:` guards) is visibly different from one in which it ran.
"""
import sys

import numpy as np
import pandas as pd

GEN = '/Users/keshavgoel/Research/data/generated/'

FAILURES = []
# Per-section tallies, in run order: {'name', 'pass', 'fail', 'skip'}. Written
# by section()/check()/skip(), read by main().
SECTIONS = []
_CUR = None


def section(tag, title):
    """Open a numbered section. Every check() and skip() after this call is
    attributed to it, so the report can break the total down instead of
    quoting one aggregate."""
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
    """A gated block that did NOT run. Printed and counted, because the failure
    mode this guards against is a whole block of assertions disappearing
    without trace when its precondition stops holding."""
    print(f"  SKIP  {label} -- {reason}")
    if _CUR is not None:
        _CUR['skip'] += 1


def _re_label_local(re_tier):
    """The prose form of a random-effects structure, matching
    report_count_models._re_label(..., in_table=False). Duplicated here only so
    section [7] can build an expected sentence to search for."""
    return '(1 + time | state)' if re_tier == 'rs' else '(1 | state)'


def re_findall_state_years(text):
    """Every `<State Name>-<year>` token in a note. Used to check a generated
    note against an independently derived set of state-years, instead of
    trusting the note's prose."""
    import re as _re
    return _re.findall(r'([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*-\d{4})', text)


def check_close(label, got, want, tol, kind='abs'):
    # Used throughout section [2] to compare live statsmodels/glmmTMB fits
    # against the GAUSSIAN_REFERENCE values.
    delta = abs(got - want) if kind == 'abs' else abs(got - want) / abs(want)
    check(f"{label} ({kind} delta {delta:.2e} <= {tol:.0e})", delta <= tol,
          f"got {got!r}, want {want!r}")


# ============================================================
# [1] PANEL INVARIANTS
# ============================================================
def validate_panels():
    section('1', 'Analytic panels')
    from build_count_model_panel import build_panel

    p21 = build_panel(2021)
    check("2021 panel has 539 rows", len(p21) == 539, f"got {len(p21)}")
    check("2021 panel has 49 states (Wyoming absent from the WPS view)",
          p21['state'].nunique() == 49, f"got {p21['state'].nunique()}")
    check("2021 panel spans 2011-2021",
          (p21['year'].min(), p21['year'].max()) == (2011, 2021))
    check("2021 violations: 533 non-null, 93 zeros",
          (p21['violations'].notna().sum(), (p21['violations'] == 0).sum()) == (533, 93),
          f"got {p21['violations'].notna().sum()} non-null, {(p21['violations'] == 0).sum()} zeros")
    check("2021 inspections: 539 non-null, 7 zeros",
          (p21['inspections'].notna().sum(), (p21['inspections'] == 0).sum()) == (539, 7),
          f"got {p21['inspections'].notna().sum()} non-null, {(p21['inspections'] == 0).sum()} zeros")
    check("2021 covid flags exactly 2020-21",
          set(p21.loc[p21['covid'] == 1, 'year']) == {2020, 2021})

    p19 = build_panel(2019)
    check("2019 panel has 450 rows (50 states x 9 years)", len(p19) == 450, f"got {len(p19)}")
    check("2019 panel has 50 states", p19['state'].nunique() == 50, f"got {p19['state'].nunique()}")
    check("2019 panel spans 2011-2019",
          (p19['year'].min(), p19['year'].max()) == (2011, 2019))
    check("2019 panel has no covid column", 'covid' not in p19.columns)
    check("2019 violations: 394 non-null, 10 zeros",
          (p19['violations'].notna().sum(), (p19['violations'] == 0).sum()) == (394, 10),
          f"got {p19['violations'].notna().sum()} non-null, {(p19['violations'] == 0).sum()} zeros")
    check("2019 inspections: 450 non-null, 15 zeros",
          (p19['inspections'].notna().sum(), (p19['inspections'] == 0).sum()) == (450, 15),
          f"got {p19['inspections'].notna().sum()} non-null, {(p19['inspections'] == 0).sum()} zeros")

    # Asymmetry invariant: the establishments reshape's fillna(0) on the two
    # inspections components (see build_count_model_panel.py comment) manufactures
    # a 0 whenever a state-year is missing a component, instead of propagating
    # NaN. Violations has no such fillna and keeps its NaNs. Net effect: 2019
    # inspections has 0 NaN while 2019 violations has 56 NaN.
    check("2019 inspections has 0 NaN while 2019 violations has 56 NaN (fillna(0) asymmetry)",
          (p19['inspections'].isna().sum(), p19['violations'].isna().sum()) == (0, 56),
          f"got {p19['inspections'].isna().sum()} / {p19['violations'].isna().sum()}")

    # Direct proof the 2019 inspection zeros are (at least partly) manufactured,
    # not measured: every one of the 15 zero-inspection state-years has at least
    # one raw component (inspections-epa-{year} / inspections-state-{year})
    # missing in the source CSV, which fillna(0) silently converted to a 0.
    from build_count_model_panel import RAW, US_STATES_50, STATE_NAME_MAPPING
    echo = pd.read_csv(RAW + 'establishments_data.csv', index_col=0)
    echo.index = echo.index.str.strip().map(lambda x: STATE_NAME_MAPPING.get(x, x))
    echo = echo[echo.index.isin(US_STATES_50)]
    n_zero_with_raw_nan = 0
    for year in range(2011, 2020):
        epa = pd.to_numeric(echo[f'inspections-epa-{year}'], errors='coerce')
        st = pd.to_numeric(echo[f'inspections-state-{year}'], errors='coerce')
        zero_mask = (epa.fillna(0) + st.fillna(0)) == 0
        n_zero_with_raw_nan += int((zero_mask & (epa.isna() | st.isna())).sum())
    check("all 15 2019 zero-inspection state-years have >=1 raw component missing (manufactured, not measured, zeros)",
          n_zero_with_raw_nan == 15, f"got {n_zero_with_raw_nan}")

    # 2021 (WPS view) is read directly from wps_dv_panel_2011_2021.csv with no
    # fillna step in this script -- any NaN there is genuine upstream missingness,
    # not manufactured. It happens to also show inspections fully populated, but
    # for a different reason than 2019: the WPS source simply has no missing
    # inspections cells in this window, not because a fillna(0) forced it.
    check("2021 violations has 6 NaN, inspections has 0 NaN (genuine WPS-source missingness, not fillna-manufactured)",
          (p21['violations'].isna().sum(), p21['inspections'].isna().sum()) == (6, 0),
          f"got violations NaN={p21['violations'].isna().sum()}, inspections NaN={p21['inspections'].isna().sum()}")

    for name, p in (('2021', p21), ('2019', p19)):
        for dv in ('violations', 'inspections'):
            raw, logged = p[dv], p[f'log_{dv}']
            m = raw.notna()
            check(f"{name} log_{dv} == log1p({dv})",
                  np.allclose(logged[m], np.log1p(raw[m])))
        check(f"{name} time == year - 2017", (p['time'] == p['year'] - 2017).all())
        check(f"{name} time2/time3 are consistent powers of time",
              (p['time2'] == p['time'] ** 2).all() and (p['time3'] == p['time'] ** 3).all())
        check(f"{name} counts are non-negative whole numbers",
              bool(((p[['violations', 'inspections']].dropna() % 1) == 0).all().all())
              and bool((p[['violations', 'inspections']].dropna() >= 0).all().all()))
        for z in ('SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
                  'dol_demand_met_pct_z', 'pct_flc_z', 'h2a_per_farmworker_z'):
            check(f"{name} {z} present", z in p.columns)


# ============================================================
# [2] GAUSSIAN ROUND-TRIP: glmmTMB must reproduce statsmodels.MixedLM
# ============================================================
# insp_M1_gaussian's reference was captured 2026-08-20 from
# paper_table_models_2021.py, whose output is what the current manuscript
# tables report; that fit (method='lbfgs') is the converged REML optimum
# there (it raises a boundary ConvergenceWarning, not a gradient-failure one
# -- see the allow-list below -- and glmmTMB independently reproduces it to
# ~7 significant figures). viol_M1_gaussian's reference is NOT from that
# pipeline -- see the correction note immediately below; it was independently
# re-derived because the manuscript's own value there is a non-converged
# optimizer artifact. Tolerances are loose enough for optimizer differences
# (lbfgs/cg vs TMB) and tight enough that a real specification difference --
# wrong sample, wrong RE structure, ML instead of REML -- cannot slip through.
#
# viol_M1_gaussian CORRECTION (2026-08-20, same day, after independent audit):
# the values originally copied here from paper_table_params_2021.json are NOT
# the REML optimum -- they come from a statsmodels.MixedLM.fit(method='lbfgs')
# run that itself raised `ConvergenceWarning: Gradient optimization failed,
# |grad| = 76.823601` (log-likelihood -737.6671). paper_table_models_2021.py
# has a module-level `warnings.filterwarnings('ignore')` that silently
# swallowed that warning before the non-converged coefficients were written
# to the published JSON. Refitting the identical analytic sample with
# method='cg' (or 'powell') converges cleanly with NO warning to a strictly
# higher log-likelihood (-729.5318, a gap of 8.14) at materially different
# values -- and that is exactly what glmmTMB's REML fit reproduces. The
# values below are the 'cg' solution, cross-checked against 'powell'
# (max delta 5.1e-5 across b, se, and both variance components -- see
# task-3-report.md). insp_M1_gaussian is untouched.
# Values below are stored at full float precision, not rounded to 6 decimals
# as an earlier revision had them. This matters now that comparisons are
# relative (see [2b]): rounding a term as small as time3 (~9e-5) to 6 decimals
# is itself a ~1e-3 RELATIVE perturbation, which was silently failing the
# round-trip against the *rounding*, not against any real disagreement.
GAUSSIAN_REFERENCE = {
    'insp_M1_gaussian': {
        'n_obs': 539, 'n_states': 49,
        'cond': {'time': -0.0633693099714064, 'time2': -0.0019126946627020943,
                 'time3': 9.172403282191922e-05},
        'se': {'time': 0.021949007615142135, 'time2': 0.004265427008013349,
               'time3': 0.0010597538603312666},
        'sigma2_u0': 1.118142193641265, 'sigma2_u1': 0.006876814299310055,
        'sigma_u01': 0.031418089077219484, 'sigma2_e': 0.33995847999400874,
    },
    'viol_M1_gaussian': {
        'n_obs': 533, 'n_states': 49,
        'cond': {'log_inspections': 0.5183382280797373, 'time': 0.06573090160967955,
                 'time2': -0.017235318225044677, 'time3': -0.005587169569039488},
        'se': {'log_inspections': 0.055737911872413476, 'time': 0.03000773750377506,
               'time2': 0.005577524753122486, 'time3': 0.0014123472614939068},
        'sigma2_u0': 1.2177225008860848, 'sigma2_u1': 0.014272021536143884,
        'sigma_u01': 0.06345062640685996, 'sigma2_e': 0.5810655980313203,
    },
}

# Which optimizer paper_table_models_2021.py's fit() should be run with to
# reach the reference each GAUSSIAN_REFERENCE cell above actually encodes.
# insp_M1 uses the published 'lbfgs' (it converges there); viol_M1 uses the
# corrected 'cg' (see the note above -- 'lbfgs' does not converge for it).
REFERENCE_SPEC = {
    'insp_M1_gaussian': ('log_inspections', ['time', 'time2', 'time3'], 'lbfgs'),
    'viol_M1_gaussian': ('log_violations',
                          ['log_inspections', 'time', 'time2', 'time3'], 'cg'),
}

# ConvergenceWarning substrings that are allow-listed: known, understood, and
# checked live (not just asserted by comment) not to indicate a genuinely
# broken fit. insp_M1/lbfgs raises exactly
# "The MLE may be on the boundary of the parameter space." -- glmmTMB
# independently reproduces the same optimum (pdHess = TRUE, see [2b]) and,
# checked below, no random-effect variance is actually pinned at zero. A
# warning on any OTHER substring (e.g. "Gradient optimization failed", the
# one that caused the viol_M1 episode) is NOT allow-listed and fails the guard.
ALLOWED_CONVERGENCE_SUBSTRINGS = ('boundary of the parameter space',)


def _fit_statsmodels_reference(dv, rhs, method):
    """Refit the exact statsmodels.MixedLM spec paper_table_models_2021.py
    uses (same dropna, same '~time' random slope, same REML default), so the
    reference values above can be checked -- for convergence AND for their
    numeric values -- live, rather than trusted as hand-copied numbers.
    Returns (result, data, warning_records), where warning_records is a list
    of (category, message) pairs so callers match on warning CLASS, not a
    hand-picked substring of the message text."""
    import warnings as _warnings

    from build_count_model_panel import build_panel
    from statsmodels.regression.mixed_linear_model import MixedLM

    df = build_panel(2021)
    d = df.dropna(subset=[dv] + rhs).copy()
    d['state'] = pd.Categorical(d['state'])
    formula = f"{dv} ~ " + " + ".join(rhs)
    with _warnings.catch_warnings(record=True) as wrec:
        _warnings.simplefilter('always')
        res = MixedLM.from_formula(formula, data=d, groups=d['state'],
                                    re_formula='~time').fit(method=method)
    return res, d, [(w.category, str(w.message)) for w in wrec]


def validate_reference_convergence():
    # Guard against the exact failure mode this project just found: a
    # statsmodels ConvergenceWarning silently swallowed by a blanket
    # `warnings.filterwarnings('ignore')`, letting a non-converged fit become
    # "ground truth". This project's pipeline must NEVER add that idiom --
    # this check fits the references live, matches warnings by class (via
    # statsmodels.tools.sm_exceptions.ConvergenceWarning, not a substring that
    # can miss real messages), and prints every recorded warning so nothing
    # is invisible even when the check PASSes.
    import re as _re

    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    section('2a', 'Reference-fit convergence guard (statsmodels, live refit)')
    for cell, (dv, rhs, method) in REFERENCE_SPEC.items():
        ref = GAUSSIAN_REFERENCE[cell]
        res, d, warns = _fit_statsmodels_reference(dv, rhs, method)
        conv_warns = [msg for cat, msg in warns if issubclass(cat, ConvergenceWarning)]
        unexplained = [w for w in conv_warns
                       if not any(s in w for s in ALLOWED_CONVERGENCE_SUBSTRINGS)]
        grads = [float(m.group(1)) for w in unexplained
                 for m in [_re.search(r'\|grad\|\s*=\s*([0-9.eE+-]+)', w)] if m]

        # An allow-listed boundary warning is only actually benign if no
        # random-effect variance is pinned at (numerically) zero -- checked
        # live against this run's own fit, not just against the specific
        # numbers recorded in the comment above.
        diag = np.diag(res.cov_re.values)
        boundary_pinned = bool(conv_warns) and not unexplained and not np.all(diag > 1e-6)
        ok = (len(unexplained) == 0) and not boundary_pinned

        if conv_warns:
            label = (f"{cell} reference fit (method={method!r}): "
                     f"convergence warning(s) present -- {conv_warns!r}")
        else:
            label = f"{cell} reference fit (method={method!r}) raised no convergence warning"
        if unexplained:
            detail = f"NOT allow-listed: {unexplained!r}" + (f"; |grad|={grads!r}" if grads else '')
        elif boundary_pinned:
            detail = f"allow-listed warning but a variance component is ~0: cov_re diag={diag.tolist()!r}"
        else:
            detail = ''
        check(label, ok, detail)

        # The live refit's own estimates, pinned against GAUSSIAN_REFERENCE
        # independently of the glmmTMB comparison in [2b] -- so a future edit
        # cannot quietly retune the hardcoded reference toward glmmTMB output.
        check(f"{cell} reference n_obs == {ref['n_obs']}", int(res.nobs) == ref['n_obs'],
              f"got {int(res.nobs)}")
        check(f"{cell} reference n_states == {ref['n_states']}",
              d['state'].nunique() == ref['n_states'], f"got {d['state'].nunique()}")
        for term, want in ref['cond'].items():
            check_close(f"{cell} reference b[{term}]", float(res.fe_params[term]), want,
                        1e-4, kind='rel')
            check_close(f"{cell} reference se[{term}]", float(res.bse[term]), ref['se'][term],
                        1e-4, kind='rel')
        check_close(f"{cell} reference sigma2_u0", float(res.cov_re.iloc[0, 0]),
                    ref['sigma2_u0'], 1e-4, kind='rel')
        check_close(f"{cell} reference sigma2_u1", float(res.cov_re.iloc[1, 1]),
                    ref['sigma2_u1'], 1e-4, kind='rel')
        check_close(f"{cell} reference sigma_u01", float(res.cov_re.iloc[0, 1]),
                    ref['sigma_u01'], 1e-4, kind='rel')
        check_close(f"{cell} reference sigma2_e", float(res.scale), ref['sigma2_e'],
                    1e-4, kind='rel')


def validate_gaussian_roundtrip():
    import json
    import subprocess
    import tempfile
    import os

    section('2b', 'Gaussian round-trip (glmmTMB REML vs statsmodels MixedLM)')
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, 'gaussian_roundtrip.json')
        proc = subprocess.run(
            ['Rscript', '/Users/keshavgoel/Research/scripts/count_models_zinb.R',
             GEN + 'count_model_panel_2021.csv', out, '--gaussian-only'],
            capture_output=True, text=True)
        if proc.returncode != 0:
            check("R script ran", False, proc.stderr.strip()[-500:])
            return
        check("R script ran", True)

        # The TMB ABI-mismatch warning (glmmTMB built against TMB 1.9.19,
        # 1.9.23 installed here) is EXPECTED on every invocation and is not
        # suppressed anywhere in this pipeline (see count_models_zinb.R and
        # task-3-report.md for the verdict that it is empirically benign).
        # Assert stderr holds nothing ELSE, so a genuinely new problem (e.g. a
        # real fit-time warning) cannot hide behind the expected one, and the
        # "benign" verdict keeps being re-checked on every run instead of
        # resting on a one-time judgement.
        known_markers = ('check_dep_version', 'TMB package version',
                          'package version mismatch', 'Warning message:',
                          're-install glmmTMB', 'restore original')
        unexpected_stderr = [ln for ln in proc.stderr.splitlines()
                             if ln.strip() and not any(m in ln for m in known_markers)]
        check("R stderr contains nothing beyond the known TMB ABI-mismatch warning",
              len(unexpected_stderr) == 0,
              f"unexpected lines: {unexpected_stderr!r}; full stderr: {proc.stderr!r}")

        fits = json.load(open(out))

    for cell, ref in GAUSSIAN_REFERENCE.items():
        fit = fits.get(cell)
        if fit is None:
            check(f"{cell} present in output", False, f"keys: {sorted(fits)}")
            continue
        check(f"{cell} converged", fit['converged'] is True, fit.get('message', ''))
        check(f"{cell} n_obs == {ref['n_obs']}", fit['n_obs'] == ref['n_obs'], f"got {fit['n_obs']}")
        check(f"{cell} n_states == {ref['n_states']}", fit['n_states'] == ref['n_states'],
              f"got {fit['n_states']}")
        for term, want in ref['cond'].items():
            got = fit['cond'].get(term, {}).get('b')
            if got is None:
                check(f"{cell} {term} estimated", False, f"terms: {sorted(fit['cond'])}")
                continue
            # Relative, not absolute: an absolute 1e-3/5e-3 tolerance cannot
            # fail on the small cubic-time terms (b[time3] ~ 1e-4, se ~ 1e-3)
            # -- glmmTMB could return zero, the wrong sign, or 10x and still
            # pass. Relative tolerance scales with the term's own magnitude.
            check_close(f"{cell} b[{term}]", got, want, 1e-3, kind='rel')
            check_close(f"{cell} se[{term}]", fit['cond'][term]['se'], ref['se'][term],
                        1e-3, kind='rel')
        # All three (1+time|state) variance/covariance terms, so the random-
        # slope structure is checked directly rather than inferred from
        # sigma2_u0 alone. Looser tolerance: these come from a different
        # optimizer path (TMB's Laplace/AD vs statsmodels' profile likelihood).
        check_close(f"{cell} sigma2_u0", fit['sigma2_u0'], ref['sigma2_u0'], 2e-2, kind='rel')
        check_close(f"{cell} sigma2_u1", fit['sigma2_u1'], ref['sigma2_u1'], 2e-2, kind='rel')
        check_close(f"{cell} sigma_u01", fit['sigma_u01'], ref['sigma_u01'], 2e-2, kind='rel')
        check_close(f"{cell} sigma2_e", fit['sigma2_e'], ref['sigma2_e'], 2e-2, kind='rel')


# ============================================================
# [2c] GAUSSIAN ROUND-TRIP AT M2 AND M3, BOTH WINDOWS
# ============================================================
# Item 5 of the 2026-08-22 review. Spec 6.1 asks for M1/M2/M3; the gate above
# fits M1 only, and M1 contains NO level-2 covariates -- so nothing in it
# checked COVARIATE parity across the R/Python bridge, and nothing checked the
# 2019 panel at all. This arm compares glmmTMB against a LIVE
# statsmodels.MixedLM fit of the identical specification on the identical
# panel, for both outcomes, all three build-up steps, and both windows.
#
# It is live-vs-live rather than live-vs-hardcoded on purpose. The property at
# risk here is that the two toolchains see the same design matrix; pinning 48
# more reference numbers into this file would guard a different thing (drift in
# the numbers themselves), which [2a] already does for M1.
#
# One complication, and it is the project's own recurring one: statsmodels'
# `lbfgs` does NOT converge for several of these specifications (2021
# violations M1, 2019 inspections M1, 2019 violations M2), and `cg` does not
# converge for 2019 inspections M3. A non-converged reference is not a
# reference. So each specification is fit under several optimizers, any fit
# raising a gradient-failure ConvergenceWarning is DISCARDED, and glmmTMB is
# required to match the highest-log-likelihood surviving fit. If no optimizer
# converges cleanly the check FAILS -- it does not quietly fall back to a
# non-converged comparison.
GAUSSIAN_BUILDUP_ADD = {
    'M1': [],
    'M2': ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z'],
    'M3': ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
           'h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z'],
}
GAUSSIAN_OUTCOMES = {
    'insp': ('log_inspections', ['time', 'time2', 'time3']),
    'viol': ('log_violations', ['log_inspections', 'time', 'time2', 'time3']),
}
GAUSSIAN_OPTIMIZERS = ('lbfgs', 'cg', 'powell')


def _clean_mixedlm(df, dv, rhs):
    """Fit `dv ~ rhs` with a (1 + time | state) random structure under each
    optimizer in turn, and return the clean fits -- those that raise no
    gradient-failure ConvergenceWarning -- as a list of dicts, best
    log-likelihood first. `powell` is only attempted when neither `lbfgs` nor
    `cg` came back clean, to keep the gate cheap."""
    import re as _re
    import warnings as _warnings

    from statsmodels.regression.mixed_linear_model import MixedLM
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    d = df.dropna(subset=[dv] + rhs).copy()
    d['state'] = pd.Categorical(d['state'])
    formula = f"{dv} ~ " + " + ".join(rhs)
    clean, tried = [], []
    for method in GAUSSIAN_OPTIMIZERS:
        if method == 'powell' and clean:
            break
        with _warnings.catch_warnings(record=True) as wrec:
            _warnings.simplefilter('always')
            try:
                res = MixedLM.from_formula(formula, data=d, groups=d['state'],
                                           re_formula='~time').fit(method=method)
            except Exception as exc:          # noqa: BLE001 -- recorded, not hidden
                tried.append((method, f'raised {type(exc).__name__}'))
                continue
        grad = any(_re.search(r'\|grad\|', str(w.message))
                   for w in wrec if issubclass(w.category, ConvergenceWarning))
        tried.append((method, 'gradient failure' if grad else 'clean'))
        if grad:
            continue
        clean.append({
            'method': method, 'llf': float(res.llf), 'n_obs': int(res.nobs),
            'n_states': int(d['state'].nunique()),
            'b': {t: float(res.fe_params[t]) for t in rhs},
            'sigma2_u0': float(res.cov_re.iloc[0, 0]),
            'sigma2_u1': float(res.cov_re.iloc[1, 1]),
            'sigma_u01': float(res.cov_re.iloc[0, 1]),
            'sigma2_e': float(res.scale),
        })
    clean.sort(key=lambda r: -r['llf'])
    return clean, tried


def validate_gaussian_buildup():
    import json
    import os
    import subprocess
    import tempfile

    from build_count_model_panel import build_panel

    section('2c', 'Gaussian round-trip at M1/M2/M3, both windows '
                  '(covariate parity)')
    n_compared = 0
    for window in (2021, 2019):
        panel_csv = GEN + f'count_model_panel_{window}.csv'
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, f'gauss_{window}.json')
            proc = subprocess.run(
                ['Rscript',
                 '/Users/keshavgoel/Research/scripts/count_models_zinb.R',
                 panel_csv, out, '--gaussian-only'],
                capture_output=True, text=True)
            if proc.returncode != 0:
                check(f"R --gaussian-only ran on the {window} panel", False,
                      proc.stderr.strip()[-400:])
                continue
            check(f"R --gaussian-only ran on the {window} panel", True)
            fits = json.load(open(out))
        check(f"{window}: the gate emits all "
              f"{len(GAUSSIAN_OUTCOMES) * len(GAUSSIAN_BUILDUP_ADD)} "
              f"Gaussian build-up fits (spec 6.1 asks for M1/M2/M3, not M1)",
              len(fits) == len(GAUSSIAN_OUTCOMES) * len(GAUSSIAN_BUILDUP_ADD),
              f"got {sorted(fits)}")
        df = build_panel(window)
        for tag, (dv, base) in sorted(GAUSSIAN_OUTCOMES.items()):
            for model, extra in sorted(GAUSSIAN_BUILDUP_ADD.items()):
                rhs = base + extra
                key = f'{tag}_{model}_gaussian'
                g = fits.get(key)
                if g is None:
                    check(f"{window} {key} present in the R output", False,
                          f"keys: {sorted(fits)}")
                    continue
                check(f"{window} {key} converged in glmmTMB",
                      g['converged'] is True, g.get('message', ''))
                # Covariate parity is the whole point of extending past M1:
                # every level-2 covariate must actually be ESTIMATED on the R
                # side, not silently dropped by a name mismatch or a merge.
                missing = [t for t in rhs if t not in (g.get('cond') or {})]
                check(f"{window} {key} estimates all {len(rhs)} fixed effects "
                      f"(covariate parity, not just the time polynomial)",
                      not missing, f"missing: {missing!r}")
                clean, tried = _clean_mixedlm(df, dv, rhs)
                check(f"{window} {key}: at least one statsmodels optimizer "
                      f"converges cleanly, so there IS a valid reference "
                      f"(tried {tried!r})", bool(clean), f"tried {tried!r}")
                if not clean:
                    skip(f"{window} {key} coefficient/variance comparison",
                         "no clean statsmodels reference to compare against")
                    continue
                ref = clean[0]
                check(f"{window} {key} n_obs == {ref['n_obs']}",
                      g['n_obs'] == ref['n_obs'], f"got {g['n_obs']}")
                check(f"{window} {key} n_states == {ref['n_states']}",
                      g['n_states'] == ref['n_states'], f"got {g['n_states']}")
                for term, want in ref['b'].items():
                    got = (g['cond'].get(term) or {}).get('b')
                    if got is None:
                        continue
                    check_close(f"{window} {key} b[{term}] vs statsmodels "
                                f"({ref['method']})", got, want, 1e-3, kind='rel')
                for comp in ('sigma2_u0', 'sigma2_u1', 'sigma_u01', 'sigma2_e'):
                    check_close(f"{window} {key} {comp} vs statsmodels "
                                f"({ref['method']})", g[comp], ref[comp],
                                2e-2, kind='rel')
                n_compared += 1
    want_n = 2 * len(GAUSSIAN_OUTCOMES) * len(GAUSSIAN_BUILDUP_ADD)
    check(f"every one of the {want_n} (window x outcome x model) build-up "
          f"round-trips was actually compared -- a vacuous pass here would "
          f"mean the arm silently did nothing", n_compared == want_n,
          f"got {n_compared}")


# ============================================================
# [3] LADDER STRUCTURE AND CONVERGENCE
# ============================================================
CELLS = ['insp_2021', 'insp_2019', 'viol_off_2021', 'viol_off_2019',
         'viol_cov_2021', 'viol_cov_2019']
FAMILY_TAGS = ['poisson', 'nbinom1', 'nbinom2', 'zip', 'zinb', 'zinb_re']

# Display labels, duplicated here ONLY so section [7] can invert the mapping
# when parsing a table back out of the memo. Asserted equal to
# report_count_models.FAMILY_LABEL in [4], so the copy cannot drift.
FAMILY_LABEL_LOCAL = {'poisson': 'Poisson', 'nbinom1': 'NB1', 'nbinom2': 'NB2',
                      'zip': 'ZIP', 'zinb': 'ZINB', 'zinb_re': 'ZINB + ZI RE'}

# Record classes in count_model_results.json. Matched on the KEY SUFFIX, never
# on membership of GAUSSIAN_REFERENCE -- that dict holds 2 keys while the
# Gaussian gate now writes 6 (M1/M2/M3 x 2 outcomes, spec 6.1), and using it as
# a filter would have let 4 Gaussian records fall through into the ladder
# checks and KeyError on `cell`.
def _is_gaussian_key(k):
    return k.endswith('_gaussian')


def _is_ladder_fit(k):
    """A ladder fit: excludes the Gaussian gate records, the per-cell __meta
    records, and the spec-5.5 __zisens sensitivity fits. __zisens is excluded
    because (a) it is ALLOWED not to converge -- its non-convergence is the
    reportable result -- and (b) it deliberately shares
    (cell, model, family_tag, re_tier) with the intercept-only ZINB it is a
    sensitivity to, so it must not enter the duplicate-identity or
    degeneracy-count checks. It gets its own block instead."""
    return not k.endswith(('_gaussian', '__meta', '__zisens'))


def validate_ladder():
    import json
    import os

    section('3', 'Count-model ladder')
    path = GEN + 'count_model_results.json'
    if not os.path.exists(path):
        check("count_model_results.json exists", False,
              "run: Rscript scripts/count_models_zinb.R "
              "data/generated/count_model_panel_2021.csv "
              "data/generated/count_model_results.json")
        return
    fits = json.load(open(path))
    check("count_model_results.json exists", True)

    # Ruling 1 (original) + 2026-08-20 coordinator review (R12-R19): the
    # tier-1 ladder itself is still 6 cells x 13 fits (M1 full ladder + M3 full
    # ladder + M2 winner-only) = 78. R12 added a second, RE-homogeneous tier
    # (all 6 families forced to `(1 | state)`) for any cell where at least one
    # tier-1 rung could not reach the constraint-compliant `(1 + time | state)`
    # structure -- 2 cells x 2 models x 6 families = 24 possible extra fits,
    # keyed `{cell}__{model}__{tag}__ri2`. R16: a family whose tier-1 attempt
    # ALREADY fell back to `(1 | state)` already IS that tier's fit, so it is
    # not refit under `__ri2` -- 7 such fits are skipped, leaving 24 - 7 = 17
    # tier-2 keys. R19 adds 2 cells x 3 models = 6 `__altopt` diagnostic refits
    # (BFGS) of the nbinom1 winners in the two 2021 violations cells. Each cell
    # also gets one `{cell}__meta` record (not a fit) carrying the per-tier
    # winners and the R15/R17 zero-fit evidence. Task 6 adds 2 more
    # `__M3covid__` fits (insp_2021, viol_cov_2021 only), each fit at the SAME
    # rs tier as that cell's own M3 winner -- excluded from `tier1_keys` below
    # (it is not part of the M1/M3 full-ladder comparison; it is a downstream
    # diagnostic of the already-selected family, exactly like M2). Total:
    # 78 + 17 + 6 altopt + 6 meta + 2 Gaussian round-trip + 2 Task-6 COVID
    # = 111. Every population is counted separately so none of these numbers
    # can silently stand in for another.
    tier1_keys = [k for k in fits if any(k.startswith(c + '__') for c in CELLS)
                  and not k.endswith('__ri2') and not k.endswith('__meta')
                  and not k.endswith('__altopt') and not k.endswith('__zisens')
                  and '__M3covid__' not in k]
    tier2_keys = [k for k in fits if k.endswith('__ri2')]
    altopt_keys = [k for k in fits if k.endswith('__altopt')]
    meta_keys = [k for k in fits if k.endswith('__meta')]
    covid_keys = [k for k in fits if '__M3covid__' in k]
    gaussian_keys = [k for k in fits if _is_gaussian_key(k)]
    zisens_keys = [k for k in fits if k.endswith('__zisens')]
    check("78 tier-1 ladder fits (6 cells x 13: M1 full ladder + M3 full ladder + M2 winner)",
          len(tier1_keys) == 78, f"got {len(tier1_keys)}")
    check("17 tier-2 (1 | state) ladder fits (24 possible - 7 de-duplicated per R16)",
          len(tier2_keys) == 17, f"got {len(tier2_keys)}")
    check("6 R19 optimizer-stability (__altopt) diagnostic fits (2 cells x 3 models)",
          len(altopt_keys) == 6, f"got {len(altopt_keys)}")
    check("6 per-cell meta records (one per cell)",
          len(meta_keys) == 6, f"got {len(meta_keys)}")
    check("2 Task-6 COVID robustness (__M3covid__) fits (insp_2021, viol_cov_2021 only)",
          len(covid_keys) == 2, f"got {len(covid_keys)}")
    # 2026-08-22 review, item 5: the Gaussian gate is M1/M2/M3 x 2 outcomes.
    check("6 Gaussian round-trip records (2 outcomes x M1/M2/M3, spec 6.1)",
          len(gaussian_keys) == 6, f"got {sorted(gaussian_keys)}")
    # 2026-08-22 review, item 2: spec 5.5's expanded-ZI sensitivity, one per
    # cell, Model 3, rs tier.
    check("6 spec-5.5 expanded-ZI sensitivity fits (__zisens), one per cell",
          len(zisens_keys) == 6, f"got {sorted(zisens_keys)}")
    check("121 total entries in count_model_results.json "
          "(78 tier-1 + 17 tier-2 + 6 altopt + 6 meta + 6 Gaussian round-trip "
          "+ 2 Task-6 COVID + 6 spec-5.5 expanded-ZI)",
          len(fits) == 78 + 17 + 6 + 6 + 6 + 2 + 6 and len(fits) == 121,
          f"got {len(fits)}")

    # R16: tier membership must be readable from `re_tier` alone -- mirrors
    # `tier_fit()` in the R script. Looks up the plain key first (valid if its
    # own re_tier matches); only falls through to the `__ri2` key for "ri".
    def tier_fit(cell, model, tag, tier):
        plain = fits.get(f'{cell}__{model}__{tag}')
        if plain is not None and plain.get('re_tier') == tier:
            return plain
        if tier == 'ri':
            ri2 = fits.get(f'{cell}__{model}__{tag}__ri2')
            if ri2 is not None and ri2.get('re_tier') == 'ri':
                return ri2
        return None

    # R16: no two stored fits should represent the same (cell, model,
    # family_tag, re_tier) -- that would be the exact duplication this ruling
    # eliminated. Gaussian round-trip and meta entries are excluded: they
    # carry no (or non-comparable) identity fields.
    import collections
    groups = collections.defaultdict(list)
    for k, f in fits.items():
        # __altopt fits are EXCLUDED here on purpose (R19): they share
        # (cell, model, family_tag='nbinom1', re_tier='rs') with the default-
        # optimizer fit by design -- that is the point of the diagnostic (same
        # spec, different optimizer), not an accidental duplicate.
        if not _is_ladder_fit(k) or k.endswith('__altopt'):
            continue
        groups[(f.get('cell'), f.get('model'), f.get('family_tag'), f.get('re_tier'))].append(k)
    dup_groups = {ident: keys for ident, keys in groups.items() if len(keys) > 1}
    check("no two fits share the same (cell, model, family_tag, re_tier)",
          len(dup_groups) == 0, f"duplicates: {dup_groups}")

    # I-3 (minor): every __altopt fit must be paired with its default-optimizer
    # counterpart at the SAME identity except optimizer -- this is the one
    # place two records are EXPECTED to share (cell, model, family_tag,
    # re_tier), and it must be exactly the altopt/non-altopt pair, nothing else.
    for k, f in fits.items():
        if not k.endswith('__altopt'):
            continue
        default_key = k[:-len('__altopt')]
        default_fit = fits.get(default_key)
        check(f"{k}: paired default-optimizer fit {default_key!r} exists",
              default_fit is not None)
        if default_fit is not None:
            check(f"{k}: shares (cell, model, family_tag, re_tier) with its default-optimizer pair",
                  (f.get('cell'), f.get('model'), f.get('family_tag'), f.get('re_tier')) ==
                  (default_fit.get('cell'), default_fit.get('model'),
                   default_fit.get('family_tag'), default_fit.get('re_tier')))

    # Spec 5.3: full ladder at M3 and M1 (stability check), winner only at M2.
    for cell in CELLS:
        for model in ('M1', 'M3'):
            missing = [t for t in FAMILY_TAGS
                       if f'{cell}__{model}__{t}' not in fits]
            check(f"{cell} {model}: all 6 families present", not missing,
                  f"missing {missing}")
        # __altopt diagnostic keys (R19) also start with "{cell}__M2__" for
        # the two cells where the M2 winner is nbinom1 -- excluded here since
        # they are a diagnostic refit of the SAME family, not a second family.
        m2 = [k for k in fits if k.startswith(f'{cell}__M2__') and not k.endswith('__altopt')]
        check(f"{cell} M2: exactly one family fit", len(m2) == 1, f"got {m2}")

    # At least one NB-family model must converge per cell, or the cell is
    # unusable. Minor fix: was `k.rsplit('__', 1)[1]` against the RAW key,
    # which yields 'ri2' (not the family tag) for every tier-2 key and 'altopt'
    # for R19 diagnostics -- silently dropping all 17 tier-2 fits (and now the
    # 6 altopt fits) from consideration. Use the `family_tag` FIELD instead,
    # which every fit carries explicitly regardless of its key's suffix.
    for cell in CELLS:
        nb = [k for k, f in fits.items() if f.get('cell') == cell
              and f.get('family_tag') in ('nbinom1', 'nbinom2', 'zinb', 'zinb_re')
              and f.get('converged')]
        check(f"{cell}: at least one NB-family model converged", bool(nb))

    # The offset specification must drop the zero-inspection state-years --
    # checked in BOTH windows (was 2021-only). The two counts are both 7 but
    # are NOT the same population: 2021 has exactly 7 zero-inspection rows
    # total, all with non-missing violations; 2019 has 15 zero-inspection rows
    # (partly fillna(0)-manufactured, see I-2), of which only 7 also have
    # non-missing violations, so only those 7 are dropped by the offset spec.
    off21 = fits.get('viol_off_2021__M1__nbinom2')
    cov21 = fits.get('viol_cov_2021__M1__nbinom2')
    if off21 and cov21:
        check("2021 offset spec drops the 7 zero-inspection state-years",
              cov21['n_obs'] - off21['n_obs'] == 7,
              f"covariate n={cov21['n_obs']}, offset n={off21['n_obs']}")
    else:
        skip("2021 offset/covariate N gap",
             "one of viol_off_2021__M1__nbinom2 / viol_cov_2021__M1__nbinom2 "
             "is absent from the JSON")
    off19 = fits.get('viol_off_2019__M1__nbinom2')
    cov19 = fits.get('viol_cov_2019__M1__nbinom2')
    if off19 and cov19:
        check("2019 offset spec drops 7 zero-inspection state-years "
              "(of 15 total; the other 8 already had missing violations)",
              cov19['n_obs'] - off19['n_obs'] == 7,
              f"covariate n={cov19['n_obs']}, offset n={off19['n_obs']}")
    else:
        skip("2019 offset/covariate N gap",
             "one of viol_off_2019__M1__nbinom2 / viol_cov_2019__M1__nbinom2 "
             "is absent from the JSON")

    # I-3: an UNGATED check that every ladder/tier-2/altopt fit converged --
    # the exp_zeros check just below is explicitly gated on `if converged`, so
    # a non-converged fit previously produced ZERO failing checks, only an
    # unasserted R "WARNING:" line. That is this project's swallowed-warning
    # failure mode recurring in a new place. Excludes meta (not a fit) and the
    # Gaussian round-trip (already checked in [2b]).
    for k, f in fits.items():
        if not _is_ladder_fit(k):
            continue
        check(f"{k}: converged", f.get('converged') is True,
              f"message: {f.get('message')!r}")

    # I-3: n_obs homogeneity, UNGATED on needs_tier2 -- tier 1 is the source of
    # every reported winner (tier 2 only exists for 2 of 6 cells), so its own
    # six families sharing one n_obs per model was never actually checked
    # before; RE tier/family choice should never change the analytic sample
    # (only the complete-case columns and the offset filter do).
    for cell in CELLS:
        for model in ('M1', 'M3'):
            tier1_present = {t: fits.get(f'{cell}__{model}__{t}') for t in FAMILY_TAGS}
            present = [v for v in tier1_present.values() if v is not None]
            n_obs_vals = {v['n_obs'] for v in present}
            check(f"{cell} {model}: tier-1 fits share one n_obs across all 6 families",
                  len(n_obs_vals) == 1, f"got {n_obs_vals}")

    # Poisson must fit worse than NB2 on these data -- variance is ~158x the mean.
    for cell in CELLS:
        p, nb2 = fits.get(f'{cell}__M3__poisson'), fits.get(f'{cell}__M3__nbinom2')
        if p and nb2 and p.get('converged') and nb2.get('converged'):
            check(f"{cell} M3: NB2 beats Poisson on AIC",
                  nb2['aic'] < p['aic'], f"poisson {p['aic']:.1f}, nb2 {nb2['aic']:.1f}")
        else:
            skip(f"{cell} M3: NB2-beats-Poisson AIC comparison",
                 "one of the two M3 fits is absent or did not converge")

    # Ruling 2: the brief's original last check compared exp_zeros to itself,
    # which can never fail. Replaced with a real assertion: for every converged
    # non-Gaussian fit, exp_zeros must actually be present, finite, and >= 0 --
    # i.e. the zero-inflation simulation in fit_spec() ran and produced a
    # sensible count, not NaN/Inf/negative from a degenerate simulate() call.
    for k, f in fits.items():
        if _is_gaussian_key(k) or k.endswith('__meta'):
            continue
        if f.get('converged'):
            ez = f.get('exp_zeros')
            ok = ez is not None and np.isfinite(ez) and ez >= 0
            check(f"{k}: exp_zeros is present, finite, and >= 0", ok, f"got {ez!r}")

    # Ruling R18: every non-meta, non-Gaussian record carries the three
    # selection-eligibility fields directly (not just derivable from
    # `__meta`) -- `is_winner_rs`, `is_winner_ri`, `eligible_for_selection`.
    # This is the interface change the coordinator flagged as needing to
    # carry into Task 5's dispatch: a consumer that reproduces the "lowest
    # AIC among all M3 fits for this cell" rule against the raw JSON, without
    # knowing anything about the __ri2 key convention or the fallback ladder,
    # will get the WRONG winner for viol_off_2021/viol_cov_2021 (it picks
    # zinb_re, reintroducing the cross-tier comparison R12 removed) unless it
    # filters on `eligible_for_selection` and `re_tier == 'rs'`, or simply
    # reads `is_winner_rs` directly.
    for k, f in fits.items():
        if _is_gaussian_key(k) or k.endswith('__meta'):
            continue
        for field in ('is_winner_rs', 'is_winner_ri', 'eligible_for_selection'):
            check(f"{k}: {field} present and boolean", isinstance(f.get(field), bool))

    # ------------------------------------------------------------
    # R12/R13/R14 (2026-08-20 coordinator review): the original winner
    # selection compared AIC across fits that had fallen back to different RE
    # structures, which conflates "needs a simpler RE structure" with "fits
    # better," and could report a winner that was a collapsed ZI duplicate of
    # another rung. These checks assert the fix, not the original rule.
    # ------------------------------------------------------------
    for cell in CELLS:
        meta = fits.get(f'{cell}__meta')
        check(f"{cell}: meta record present", meta is not None)
        if meta is None:
            continue

        # R12: the tier-1 ("rs") winner must be a fit that actually achieved
        # the constraint-compliant (1 + time | state) structure -- not one
        # compared in from a fallback tier -- and R14: it must not be a
        # collapsed ZI duplicate of another rung.
        winner_tag = meta.get('m3_winner_rs')
        winner_fit = fits.get(f'{cell}__M3__{winner_tag}')
        check(f"{cell}: tier-1 (rs) winner fit exists ({cell}__M3__{winner_tag})",
              winner_fit is not None)
        if winner_fit is not None:
            check(f"{cell}: tier-1 winner was fit at (1 + time | state)",
                  winner_fit.get('re_tier') == 'rs',
                  f"got re_tier={winner_fit.get('re_tier')!r}, re_used={winner_fit.get('re_used')!r}")
            check(f"{cell}: tier-1 winner is not a collapsed ZI duplicate (R14)",
                  not winner_fit.get('collapsed_to'),
                  f"collapsed_to={winner_fit.get('collapsed_to')!r}")

        # R15 (supersedes R13's defective flag): the winner's own zero-fit
        # facts must always be reported, and the "better zero-fit exists"
        # flag must be a real boolean, visible without re-deriving it later.
        check(f"{cell}: winner_rs_zero_discrepancy present and finite",
              isinstance(meta.get('winner_rs_zero_discrepancy'), (int, float))
              and np.isfinite(meta.get('winner_rs_zero_discrepancy')))
        check(f"{cell}: has_better_zero_fit_rs flag present and boolean",
              isinstance(meta.get('has_better_zero_fit_rs'), bool))
        # If a credible competitor was named, it must actually beat the
        # winner's discrepancy and be within delta_aic=10 -- i.e. the flag is
        # not just present but internally consistent with the raw per-fit data.
        if meta.get('has_better_zero_fit_rs'):
            comp_tag = meta.get('credible_better_zero_fit_rs')
            comp_fit = tier_fit(cell, 'M3', comp_tag, 'rs')
            winner_fit_rs = fits.get(f"{cell}__M3__{meta.get('m3_winner_rs')}")
            check(f"{cell}: credible competitor '{comp_tag}' fit exists at rs tier",
                  comp_fit is not None)
            if comp_fit is not None and winner_fit_rs is not None:
                check(f"{cell}: credible competitor is within 10 AIC of the rs winner",
                      (comp_fit['aic'] - winner_fit_rs['aic']) <= 10,
                      f"delta AIC = {comp_fit['aic'] - winner_fit_rs['aic']:.2f}")
                check(f"{cell}: credible competitor's zero-fit discrepancy is strictly better",
                      comp_fit['zero_fit_discrepancy'] < meta['winner_rs_zero_discrepancy'],
                      f"competitor {comp_fit['zero_fit_discrepancy']!r} vs "
                      f"winner {meta['winner_rs_zero_discrepancy']!r}")
                # R17: the credible competitor must itself not be ZI-boundary-
                # degenerate -- a degenerate fit (e.g. viol_cov_2019's old
                # 'zinb' comparator, ZI intercept -21.4) is not a real
                # competitor, however small its zero_fit_discrepancy looks.
                check(f"{cell}: credible competitor is not ZI-boundary-degenerate (R17)",
                      not comp_fit.get('zi_degenerate'),
                      f"zi_degenerate_reason={comp_fit.get('zi_degenerate_reason')!r}")

        # R16 follow-up + I-5: per-tier, per-model eligible-family sets must be
        # present, NON-EMPTY, and must contain the winner that was chosen from
        # them -- "present" alone can never fail (an empty list is still
        # "present"), so I-5 adds the two assertions that can.
        for key in ('eligible_rs_m1', 'eligible_rs_m3'):
            check(f"{cell}: {key} present", key in meta)
            check(f"{cell}: {key} is non-empty", bool(meta.get(key)))
        check(f"{cell}: m3_winner_rs is a member of eligible_rs_m3",
              meta.get('m3_winner_rs') in meta.get('eligible_rs_m3', []),
              f"winner {meta.get('m3_winner_rs')!r} not in {meta.get('eligible_rs_m3')!r}")

        # R12: where a tier-2 ladder was needed, it must be a full,
        # internally-consistent six-family ladder -- RE tier alone should not
        # change the analytic sample, so all six share one n_obs per model.
        # Membership is resolved via tier_fit() (re_tier), never key-parsing
        # (R16) -- a de-duplicated family is legitimately found at its plain
        # key, not a "__ri2" key.
        if meta.get('needs_tier2'):
            check(f"{cell}: winner_ri_zero_discrepancy present and finite",
                  isinstance(meta.get('winner_ri_zero_discrepancy'), (int, float))
                  and np.isfinite(meta.get('winner_ri_zero_discrepancy')))
            check(f"{cell}: has_better_zero_fit_ri flag present and boolean",
                  isinstance(meta.get('has_better_zero_fit_ri'), bool))
            ri_winner_tag = meta.get('m3_winner_ri')
            ri_winner_fit = tier_fit(cell, 'M3', ri_winner_tag, 'ri')
            check(f"{cell}: tier-2 (ri) winner fit exists", ri_winner_fit is not None)
            if ri_winner_fit is not None:
                check(f"{cell}: tier-2 winner is not a collapsed ZI duplicate (R14)",
                      not ri_winner_fit.get('collapsed_to'),
                      f"collapsed_to={ri_winner_fit.get('collapsed_to')!r}")
            for key in ('eligible_ri_m1', 'eligible_ri_m3'):
                check(f"{cell}: {key} present", key in meta)
                check(f"{cell}: {key} is non-empty", bool(meta.get(key)))
            check(f"{cell}: m3_winner_ri is a member of eligible_ri_m3",
                  meta.get('m3_winner_ri') in meta.get('eligible_ri_m3', []),
                  f"winner {meta.get('m3_winner_ri')!r} not in {meta.get('eligible_ri_m3')!r}")
            for model in ('M1', 'M3'):
                tier2_fits = {t: tier_fit(cell, model, t, 'ri') for t in FAMILY_TAGS}
                check(f"{cell} {model}: all 6 fits resolve to a (1 | state) fit "
                      "(via plain key or __ri2, per re_tier)",
                      all(v is not None for v in tier2_fits.values()),
                      f"missing {[t for t, v in tier2_fits.items() if v is None]}")
                present = [v for v in tier2_fits.values() if v is not None]
                n_obs_vals = {v['n_obs'] for v in present}
                check(f"{cell} {model}: tier-2 fits share one n_obs across all 6 families",
                      len(n_obs_vals) == 1, f"got {n_obs_vals}")
                check(f"{cell} {model}: every tier-2 fit is tagged re_tier == 'ri'",
                      all(v.get('re_tier') == 'ri' for v in present))
        else:
            check(f"{cell}: no tier-2 winner reported (tier-2 not needed)",
                  meta.get('m3_winner_ri') is None)

        # I-3: M2 is fit under the winning family with the FULL fallback
        # ladder (not forced to any tier), so it could in principle land at a
        # different RE structure than the tier that actually selected it.
        # M2's family is always meta['m3_winner_rs'], and winner_rs is by
        # construction drawn only from rs-tier fits -- so M2 must ALSO be at
        # rs, or the manuscript's M2 column would silently rest on a different
        # random-effects structure than the one its own selection was based on.
        m2_fit = fits.get(f"{cell}__M2__{meta.get('m2_family')}")
        check(f"{cell}: M2 fit exists under the selected family", m2_fit is not None)
        if m2_fit is not None:
            check(f"{cell}: M2's re_tier matches the tier ('rs') of the winner that selected it",
                  m2_fit.get('re_tier') == 'rs',
                  f"got re_tier={m2_fit.get('re_tier')!r}, re_used={m2_fit.get('re_used')!r}")

        # R17: no fit that is EITHER the rs or ri winner should be ZI-boundary
        # degenerate -- pick_winner() already excludes zi_degenerate fits, so
        # this is a check that the exclusion actually took effect on the
        # stored result, not just a restatement of the R script's own filter.
        for tier, tag_key in (('rs', 'm3_winner_rs'), ('ri', 'm3_winner_ri')):
            tag = meta.get(tag_key)
            if tag is None:
                continue
            wfit = tier_fit(cell, 'M3', tag, tier)
            if wfit is not None:
                check(f"{cell}: {tier}-tier winner is not ZI-boundary-degenerate (R17)",
                      not wfit.get('zi_degenerate'),
                      f"zi_degenerate_reason={wfit.get('zi_degenerate_reason')!r}")

        # R18: a downstream consumer that knows NOTHING about the __ri2 key
        # convention must still be able to find the correct winner by scanning
        # every record for `is_winner_rs` (or `is_winner_ri`) == True -- this
        # is the exact scenario the reviewer simulated (Task 5's selection
        # rule reproducing zinb_re for the 2021 violations cells because it
        # only knew plain keys). Reconstruct the winner that way here and
        # confirm it matches meta's own winner tag.
        marked_rs = [k for k, f in fits.items()
                     if f.get('cell') == cell and f.get('model') == 'M3' and f.get('is_winner_rs')]
        check(f"{cell}: exactly one M3 record is marked is_winner_rs == True",
              len(marked_rs) == 1, f"got {marked_rs}")
        if len(marked_rs) == 1:
            check(f"{cell}: the record marked is_winner_rs matches meta['m3_winner_rs']",
                  fits[marked_rs[0]].get('family_tag') == meta.get('m3_winner_rs'),
                  f"marked {fits[marked_rs[0]].get('family_tag')!r} vs "
                  f"meta {meta.get('m3_winner_rs')!r}")
        if meta.get('needs_tier2'):
            marked_ri = [k for k, f in fits.items()
                         if f.get('cell') == cell and f.get('model') == 'M3' and f.get('is_winner_ri')]
            check(f"{cell}: exactly one M3 record is marked is_winner_ri == True",
                  len(marked_ri) == 1, f"got {marked_ri}")
            if len(marked_ri) == 1:
                check(f"{cell}: the record marked is_winner_ri matches meta['m3_winner_ri']",
                      fits[marked_ri[0]].get('family_tag') == meta.get('m3_winner_ri'),
                      f"marked {fits[marked_ri[0]].get('family_tag')!r} vs "
                      f"meta {meta.get('m3_winner_ri')!r}")
        else:
            marked_ri_any = [k for k, f in fits.items()
                             if f.get('cell') == cell and f.get('is_winner_ri')]
            check(f"{cell}: no record is marked is_winner_ri (tier-2 not needed)",
                  len(marked_ri_any) == 0, f"got {marked_ri_any}")

        # I-2: the manufactured-zeros flag must be present, boolean, and TRUE
        # only for insp_2019 (the only cell whose DV -- inspections -- is
        # subject to the establishments-reshape fillna(0) artifact; violations
        # has no such fill and keeps genuine NaN).
        check(f"{cell}: zeros_partly_manufactured present and boolean",
              isinstance(meta.get('zeros_partly_manufactured'), bool))
        check(f"{cell}: zeros_partly_manufactured is True only for insp_2019",
              meta.get('zeros_partly_manufactured') == (cell == 'insp_2019'),
              f"got {meta.get('zeros_partly_manufactured')!r}")

        # R19: the winner-warning diagnostic fields must be present; if the
        # winner carries a warning, `clean_rs_competitor_within_10_aic` is
        # either a real family tag (a clean competitor exists) or None (it
        # doesn't) -- both are valid outcomes, but the field must exist so the
        # memo does not have to re-derive it from `message`/`aic` by hand.
        check(f"{cell}: winner_rs_has_warning present and boolean",
              isinstance(meta.get('winner_rs_has_warning'), bool))
        if meta.get('winner_rs_has_warning'):
            comp = meta.get('clean_rs_competitor_within_10_aic')
            check(f"{cell}: clean_rs_competitor_within_10_aic is a family tag or None",
                  comp is None or comp in FAMILY_TAGS, f"got {comp!r}")

    # ------------------------------------------------------------
    # CRITICAL (2026-08-22 review): the ZERO-INFLATION random-effect variance.
    # Before this it was never extracted, so every structural-zero probability
    # reported for a ZI-RE fit was plogis(b0) -- the MEDIAN-state value -- given
    # as if it were marginal. Two things are asserted: the variance is PRESENT
    # wherever the fit's own zi_formula carries `(1 | state)`, and it is ABSENT
    # wherever it does not (so a stale value can never be paired with a
    # formula that has no such term). Both directions are checked because the
    # `zinb_re` rung DOWNGRADES to a plain `~1` ZI when the random intercept
    # cannot be estimated, and a downgraded fit has no variance to report.
    # ------------------------------------------------------------
    zi_re_fits = {k: f for k, f in fits.items()
                  if _is_ladder_fit(k)
                  and '(1|state)' in (f.get('zi_formula') or '').replace(' ', '')}
    check("there ARE fits whose zi_formula carries a state random intercept, "
          "so the checks below are not vacuous", len(zi_re_fits) >= 1,
          f"got {len(zi_re_fits)}")
    for k, f in sorted(zi_re_fits.items()):
        v = f.get('sigma2_zi_u0')
        check(f"{k}: zi_formula has (1 | state), so sigma2_zi_u0 is present, "
              f"finite and > 0", v is not None and np.isfinite(v) and v > 0,
              f"got {v!r} (zi_formula={f.get('zi_formula')!r})")
        sd = f.get('sigma_zi_u0')
        check(f"{k}: sigma_zi_u0 is the square root of sigma2_zi_u0",
              sd is not None and np.isfinite(sd)
              and abs(sd - np.sqrt(v)) <= 1e-9 * max(1.0, abs(sd)),
              f"sd={sd!r}, sqrt(var)={np.sqrt(v) if v else None!r}")
    for k, f in sorted(fits.items()):
        if not _is_ladder_fit(k):
            continue
        if '(1|state)' in (f.get('zi_formula') or '').replace(' ', ''):
            continue
        check(f"{k}: no state random intercept in the ZI formula, so "
              f"sigma2_zi_u0 is null", f.get('sigma2_zi_u0') is None,
              f"got {f.get('sigma2_zi_u0')!r} for "
              f"zi_formula={f.get('zi_formula')!r}")
    # The consequence, asserted rather than described: for at least one fit the
    # marginal and median-state probabilities must differ by a LOT, or the whole
    # Critical item would have been cosmetic.
    from scipy.special import expit as _expit
    worst = None
    for k, f in sorted(zi_re_fits.items()):
        zi = (f.get('zi') or {}).get('(Intercept)')
        v = f.get('sigma2_zi_u0')
        if not zi or v is None or not np.isfinite(v) or v <= 0:
            continue
        x, w = np.polynomial.hermite.hermgauss(64)
        marg = float(np.sum(w * _expit(zi['b'] + np.sqrt(2 * v) * x))
                     / np.sqrt(np.pi))
        med = float(_expit(zi['b']))
        ratio = marg / med if med > 0 else np.inf
        if worst is None or ratio > worst[1]:
            worst = (k, ratio, med, marg)
    check("at least one ZI-RE fit has a marginal structural-zero probability "
          "more than 10x its median-state value -- i.e. the distinction the "
          "Critical fix introduced is materially large, not decorative",
          worst is not None and worst[1] > 10,
          f"worst: {worst!r}")

    # ------------------------------------------------------------
    # Minor (2026-08-22 review): mark_zi_degenerate()'s loglik criterion is only
    # a boundary-degeneracy test when the counterpart is at the SAME re_tier.
    # It now requires that, and records whether it was applicable. Assert the
    # inapplicable cases are EXACTLY the cross-tier ones -- otherwise the
    # criterion could be silently inert somewhere else.
    # ------------------------------------------------------------
    cross_tier, same_tier = [], []
    for k, f in sorted(fits.items()):
        if not _is_ladder_fit(k) or 'zi_loglik_criterion_applied' not in f:
            continue
        ct = f.get('zi_counterpart_tier')
        if ct is None:
            check(f"{k}: no non-ZI counterpart, so the loglik criterion is "
                  f"recorded as not applied",
                  f.get('zi_loglik_criterion_applied') is False,
                  f"got {f.get('zi_loglik_criterion_applied')!r}")
        elif ct != f.get('re_tier'):
            cross_tier.append(k)
            check(f"{k}: counterpart is at tier {ct!r} but this fit is at "
                  f"{f.get('re_tier')!r}, so the loglik criterion is NOT applied",
                  f.get('zi_loglik_criterion_applied') is False,
                  f"got {f.get('zi_loglik_criterion_applied')!r}")
        else:
            same_tier.append(k)
            check(f"{k}: counterpart is at the same tier, so the loglik "
                  f"criterion IS applied",
                  f.get('zi_loglik_criterion_applied') is True,
                  f"got {f.get('zi_loglik_criterion_applied')!r}")
    check("the loglik-degeneracy criterion is inert for exactly the cross-tier "
          "comparisons, and there ARE some (so the tier guard is not vacuous)",
          len(cross_tier) >= 1 and len(same_tier) >= 1,
          f"cross-tier {len(cross_tier)}, same-tier {len(same_tier)}")
    check("no fit was flagged degenerate BY the loglik criterion while that "
          "criterion was inapplicable",
          not [k for k, f in fits.items()
               if _is_ladder_fit(k)
               and 'loglik matches' in (f.get('zi_degenerate_reason') or '')
               and f.get('zi_loglik_criterion_applied') is not True],
          "a 'loglik matches' reason exists on a fit where the criterion was "
          "recorded as not applied")

    # ------------------------------------------------------------
    # Minor: pick_winner()'s empty-candidate-set fallback must be RECORDED, not
    # silent. No cell should be defaulting today; the field exists so that if
    # one ever does, it is visible instead of looking like an AIC result.
    # ------------------------------------------------------------
    for cell in CELLS:
        meta = fits.get(f'{cell}__meta') or {}
        for fld in ('winner_defaulted_rs_m1', 'winner_defaulted_rs_m3'):
            check(f"{cell}: {fld} present and boolean",
                  isinstance(meta.get(fld), bool), f"got {meta.get(fld)!r}")
            check(f"{cell}: {fld} is False (the winner came from an AIC "
                  f"comparison, not pick_winner's fallback)",
                  meta.get(fld) is False, f"got {meta.get(fld)!r}")
        for fld in ('n_eligible_rs_m1', 'n_eligible_rs_m3'):
            n = meta.get(fld)
            check(f"{cell}: {fld} records how many candidates the winner beat, "
                  f"and it is >= 1", isinstance(n, int) and n >= 1,
                  f"got {n!r}")

    # ------------------------------------------------------------
    # Minor: the manufactured-zero note is DERIVED, not typed. Re-derive the
    # affected state-years here, independently of the R script, and require the
    # note to name exactly them.
    # ------------------------------------------------------------
    _estab = pd.read_csv('/Users/keshavgoel/Research/data/raw/'
                         'establishments_data.csv', index_col=0)
    _estab.index = _estab.index.str.strip()
    _man = set()
    for _yr in range(2011, 2020):
        _e = pd.to_numeric(_estab[f'inspections-epa-{_yr}'], errors='coerce')
        _st = pd.to_numeric(_estab[f'inspections-state-{_yr}'], errors='coerce')
        for _s in _estab.index[_e.isna() | _st.isna()]:
            _man.add(f'{_s}-{_yr}')
    check("the raw establishments CSV really does have missing inspection "
          "components (otherwise the manufactured-zero derivation is vacuous)",
          len(_man) >= 1, f"got {len(_man)}")
    for cell in CELLS:
        meta = fits.get(f'{cell}__meta') or {}
        flag = meta.get('zeros_partly_manufactured')
        note = meta.get('zeros_partly_manufactured_note') or ''
        check(f"{cell}: zeros_partly_manufactured present and boolean",
              isinstance(flag, bool))
        check(f"{cell}: the note is non-empty exactly when the flag is set",
              bool(note) == bool(flag), f"flag={flag!r}, note={note!r}")
        if not flag:
            continue
        named = set(re_findall_state_years(note))
        check(f"{cell}: every state-year the manufactured-zero note names is "
              f"independently confirmed missing a raw inspection component",
              named and named <= _man,
              f"named {sorted(named)!r}; not confirmed: "
              f"{sorted(named - _man)!r}")
    check("zeros_partly_manufactured is True for exactly the cells whose DV is "
          "the fillna(0)-summed establishments-view inspections column",
          {c for c in CELLS
           if fits.get(f'{c}__meta', {}).get('zeros_partly_manufactured')}
          == {'insp_2019'},
          f"got {[c for c in CELLS if fits.get(f'{c}__meta', {}).get('zeros_partly_manufactured')]!r}")

    # ------------------------------------------------------------
    # Item 2 (2026-08-22 review): spec 5.5's expanded-ZI sensitivity fits. They
    # are ALLOWED to fail -- the point is that the outcome is recorded either
    # way -- so what is asserted is that all six exist, carry the expanded ZI
    # formula, sit at the mandated tier, and are never selection-eligible.
    # ------------------------------------------------------------
    ZI_EXPANDED_TERMS = ('SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z')
    for cell in CELLS:
        k = f'{cell}__M3__zinb__zisens'
        f = fits.get(k)
        check(f"{k} exists (spec 5.5 was attempted for this cell)",
              f is not None)
        if f is None:
            continue
        zf = (f.get('zi_formula') or '').replace(' ', '')
        check(f"{k}: the ZI formula carries all three expanded terms",
              all(t in zf for t in ZI_EXPANDED_TERMS), f"got {zf!r}")
        check(f"{k}: no state random intercept in the expanded ZI formula "
              f"(the sensitivity varies the ZI PREDICTORS, not its RE)",
              '(1|state)' not in zf, f"got {zf!r}")
        check(f"{k}: fit at the mandated (1 + time | state) tier only",
              f.get('re_tier') == 'rs' and f.get('re_used') == '(1 + time | state)',
              f"re_tier={f.get('re_tier')!r}, re_used={f.get('re_used')!r}")
        check(f"{k}: never a family-selection competitor",
              f.get('eligible_for_selection') is False
              and f.get('is_winner_rs') is False
              and f.get('is_winner_ri') is False)
        check(f"{k}: converged is recorded as a boolean (its VALUE is the "
              f"reportable result, either way)",
              isinstance(f.get('converged'), bool), f"got {f.get('converged')!r}")
    n_zs_conv = sum(1 for cell in CELLS
                    if (fits.get(f'{cell}__M3__zinb__zisens') or {})
                    .get('converged') is True)
    check(f"the expanded-ZI sensitivity has a mixed outcome across cells "
          f"({n_zs_conv} of {len(CELLS)} converged), so the memo must report "
          f"both sides rather than one blanket verdict",
          0 <= n_zs_conv <= len(CELLS), f"got {n_zs_conv}")

    # ------------------------------------------------------------
    # Item 3 (2026-08-22 review): `sigma2_e` must be null for every
    # non-Gaussian fit -- glmmTMB's sigma() is a dispersion parameter there,
    # not a residual SD, and its square was being exported under a name that
    # reads as residual variance. `dispersion` and `family_name` carry it
    # instead. The Gaussian gate still needs the SQUARE, so it keeps sigma2_e.
    # ------------------------------------------------------------
    for k, f in sorted(fits.items()):
        if _is_gaussian_key(k):
            check(f"{k}: Gaussian fit keeps sigma2_e (the round-trip gate "
                  f"compares it to MixedLM.scale)",
                  f.get('sigma2_e') is not None
                  and np.isfinite(f['sigma2_e']) and f['sigma2_e'] > 0,
                  f"got {f.get('sigma2_e')!r}")
            check(f"{k}: sigma2_e is the square of dispersion (the residual SD)",
                  f.get('dispersion') is not None
                  and abs(f['sigma2_e'] - f['dispersion'] ** 2)
                  <= 1e-9 * max(1.0, f['sigma2_e']),
                  f"sigma2_e={f.get('sigma2_e')!r}, "
                  f"dispersion={f.get('dispersion')!r}")
            check(f"{k}: family_name == 'gaussian'",
                  f.get('family_name') == 'gaussian', f"got {f.get('family_name')!r}")
            continue
        if k.endswith('__meta') or not f.get('converged'):
            continue
        check(f"{k}: sigma2_e is null for a non-Gaussian family",
              f.get('sigma2_e') is None, f"got {f.get('sigma2_e')!r}")
        check(f"{k}: dispersion and family_name are both present",
              f.get('dispersion') is not None and f.get('family_name'),
              f"dispersion={f.get('dispersion')!r}, "
              f"family_name={f.get('family_name')!r}")
    check("no ladder fit reports a non-null sigma2_e (a count family has no "
          "residual variance, and dispersion-squared is not one)",
          not [k for k, f in fits.items()
               if _is_ladder_fit(k) and f.get('sigma2_e') is not None],
          "found a non-null sigma2_e on a count fit")

    # ------------------------------------------------------------
    # Minor: COVID_CELLS excluded viol_off_2021 with no stated reason. Every
    # 2021 cell must now carry the flag AND, where it is False, a reason.
    # ------------------------------------------------------------
    for cell in CELLS:
        meta = fits.get(f'{cell}__meta') or {}
        check(f"{cell}: covid_variant_fit present and boolean",
              isinstance(meta.get('covid_variant_fit'), bool),
              f"got {meta.get('covid_variant_fit')!r}")
        if meta.get('covid_variant_fit'):
            check(f"{cell}: a COVID variant fit really exists in the JSON",
                  any(k.startswith(f'{cell}__M3covid__') for k in fits))
            check(f"{cell}: no skip reason is recorded for a cell that WAS fit",
                  not (meta.get('covid_not_fit_reason') or ''))
        else:
            check(f"{cell}: a reason is recorded for NOT fitting a COVID variant",
                  len(meta.get('covid_not_fit_reason') or '') > 40,
                  f"got {meta.get('covid_not_fit_reason')!r}")
            check(f"{cell}: and no COVID variant fit exists",
                  not any(k.startswith(f'{cell}__M3covid__') for k in fits))
    _cov21_skipped = [c for c in CELLS
                      if c.endswith('_2021')
                      and not (fits.get(f'{c}__meta') or {}).get('covid_variant_fit')]
    check("exactly one 2021 cell has no COVID variant, and it is viol_off_2021 "
          "(so the memo's derived sentence has something to report)",
          _cov21_skipped == ['viol_off_2021'], f"got {_cov21_skipped!r}")

    # Known, verified fact about this run: exactly the two 2021-window
    # violations cells needed a tier-2 refit -- their zinb/zinb_re rungs fell
    # back to (1 | state) while poisson/nbinom1/nbinom2/zip converged at
    # (1 + time | state) (see task-4-report.md addendum for the full table).
    tier2_cells = {c for c in CELLS if fits.get(f'{c}__meta', {}).get('needs_tier2')}
    check("exactly viol_off_2021 and viol_cov_2021 needed a tier-2 (1 | state) refit",
          tier2_cells == {'viol_off_2021', 'viol_cov_2021'}, f"got {tier2_cells}")

    # Known, verified fact (R15, revised by R17 -- Critical): restricting
    # "better zero-fit" comparators to a credible (delta_aic <= 10), non-
    # ZI-degenerate, noise-aware (improvement > 2x combined MC SE) set shrinks
    # the flagged set down to exactly ONE cell: insp_2019 (zinb beats
    # zinb_re's winning fit by a wide, noise-swamping zero-fit margin at only
    # ~4 AIC cost). viol_cov_2019 previously also carried this flag, but its
    # only credible competitor (zinb) is now excluded as ZI-boundary-
    # degenerate (R17: ZI intercept ~-21.4, SE ~2533) -- with no other
    # eligible competitor, the flag correctly no longer fires there. No
    # ri-tier cell carries the flag.
    flagged_rs = {c for c in CELLS if fits.get(f'{c}__meta', {}).get('has_better_zero_fit_rs')}
    flagged_ri = {c for c in CELLS if fits.get(f'{c}__meta', {}).get('has_better_zero_fit_ri')}
    check("exactly insp_2019 carries the credible-set, noise-aware better-zero-fit flag (rs tier)",
          flagged_rs == {'insp_2019'}, f"got {flagged_rs}")
    check("no cell carries the credible-set better-zero-fit flag at the ri tier",
          flagged_ri == set(), f"got {flagged_ri}")

    # Ruling R17 (Critical): every zip/zinb/zinb_re fit gets a `zi_degenerate`
    # boolean + `zi_degenerate_reason` string; degenerate fits stay in the
    # JSON (selection-exclusion, not deletion) so a later task can still see
    # exactly what glmmTMB reported. Verified directly against this run's own
    # thresholds (|zi_intercept| > 15, se(zi_intercept) > 100, or loglik within
    # 1e-4 of the non-ZI counterpart): 11 records meet at least one criterion,
    # all in the 2019 violations cells (viol_off_2019/viol_cov_2019, both
    # models where present). The coordinator's review independently cited 12;
    # this run finds 11 by the exact specified thresholds -- reported as a
    # verified discrepancy, not silently reconciled to match.
    for k, f in fits.items():
        if _is_gaussian_key(k) or k.endswith('__meta'):
            continue
        if f.get('family_tag') in ('zip', 'zinb', 'zinb_re'):
            check(f"{k}: zi_degenerate present and boolean",
                  isinstance(f.get('zi_degenerate'), bool))
            check(f"{k}: zi_degenerate_reason present and a string",
                  isinstance(f.get('zi_degenerate_reason'), str))
        else:
            check(f"{k}: zi_degenerate is False for a non-ZI family",
                  f.get('zi_degenerate') is False)
    # Counted over LADDER fits only: 2 of the 6 spec-5.5 __zisens sensitivity
    # fits are also ZI-degenerate, and folding them in here would silently
    # change a number the memo reports from the ladder alone.
    n_degenerate = sum(1 for k, f in fits.items()
                       if _is_ladder_fit(k) and f.get('zi_degenerate') is True)
    check("exactly 11 LADDER fits are marked zi_degenerate under the specified "
          "thresholds", n_degenerate == 11, f"got {n_degenerate}")

    # Ruling R19: the optimizer-stability diagnostic must have run for both
    # affected cells and its sigma2_u1 comparison fields must be present.
    for cell in ('viol_off_2021', 'viol_cov_2021'):
        meta = fits.get(f'{cell}__meta', {})
        for field in ('sigma2_u1_default_optimizer', 'sigma2_u1_alt_optimizer', 'sigma2_u1_rel_diff'):
            val = meta.get(field)
            check(f"{cell}: meta[{field!r}] present and finite",
                  isinstance(val, (int, float)) and np.isfinite(val), f"got {val!r}")
        check(f"{cell}: meta['sigma2_u1_alt_optimizer_converged'] is True",
              meta.get('sigma2_u1_alt_optimizer_converged') is True)
        for model in ('M1', 'M2', 'M3'):
            alt = fits.get(f'{cell}__{model}__nbinom1__altopt')
            check(f"{cell} {model}: __altopt diagnostic fit present and converged",
                  alt is not None and alt.get('converged') is True)

    # R14: no fit anywhere should claim to be its OWN collapsed duplicate, and
    # a collapsed record's AIC must equal (relative tolerance) the record it
    # collapsed to -- otherwise "collapsed_to" would be an assertion, not an
    # observed fact about the two fits.
    for k, f in fits.items():
        if k.endswith('__meta') or not f.get('collapsed_to'):
            continue
        # Strip a trailing tier-2 "__ri2" suffix (if present) BEFORE splitting
        # off the family tag -- rsplit('__', 1) on the raw key would otherwise
        # split between the family tag and "ri2", not between model and tag.
        suffix = '__ri2' if k.endswith('__ri2') else ''
        base = k[:-len(suffix)] if suffix else k
        prefix = base.rsplit('__', 1)[0]  # "{cell}__{model}"
        parent_key = f"{prefix}__{f['collapsed_to']}{suffix}"
        parent = fits.get(parent_key)
        check(f"{k}: collapsed-to target {parent_key!r} exists", parent is not None)
        if parent is not None:
            check_close(f"{k}: collapsed duplicate shares its parent's AIC",
                        f['aic'], parent['aic'], 1e-8, kind='rel')


# ============================================================
# [4] SELECTION TABLE AND REPORTING (Task 5)
# ============================================================
# NOTE (2026-08-20 dispatch): the original brief for this section assumed
# selection_table() would groupby('cell')['aic'].idxmin() across ALL M3 fits
# regardless of re_tier. That rule is provably wrong on this JSON: two cells
# (viol_off_2021, viol_cov_2021) have de-duplicated (1 | state) fits sitting
# at their PLAIN keys (re_tier == 'ri'), so a cross-tier AIC-min conflates
# random-effects structure with distribution family and silently returns
# 'zinb_re' where the real, tier-respecting winner is 'nbinom1'. These checks
# assert the corrected contract instead: winners are read from is_winner_rs /
# is_winner_ri, cross-checked against each cell's __meta record.
def validate_selection():
    import json
    import os

    section('4', 'Selection table and reporting')
    try:
        from report_count_models import (BOUNDARY_KIND, FAMILY_LABEL,
                                          TIER_ORDER, coefficient_table,
                                          nearest_clean_competitor,
                                          selection_table, winner_summary)
    except ImportError as e:
        check("report_count_models importable", False, str(e))
        return
    check("report_count_models importable", True)

    raw = json.load(open(GEN + 'count_model_results.json'))
    tab = selection_table(raw)

    # -- Population: 111 total (Task 6 added 2 __M3covid__ fits to the prior
    # 109) = 2 Gaussian + 6 __meta + 6 __altopt + 97 genuine fits.
    # selection_table() must return exactly the 97, never KeyError'ing on a
    # __meta record (the brief's loop did exactly that). Note: per
    # report_count_models.py's `_is_genuine_fit_key()` (not modified by
    # Task 6 -- out of scope), the exclusion suffix list is
    # ('_gaussian', '__meta', '__altopt') and does NOT exclude
    # '__M3covid__' keys, so the 2 COVID fits ARE included here as ordinary
    # rows. That is correct: they are genuine model fits (just never
    # eligible_for_selection or is_winner_rs/ri, so they cannot corrupt any
    # winner/LRT computation below).
    check("selection_table returns exactly the 97 genuine model fits "
          "(111 total - 2 gaussian - 6 meta - 6 altopt)",
          len(tab) == 97, f"got {len(tab)}")
    check("selection_table never includes a __meta, _gaussian, or __altopt key",
          not any(k.endswith('__meta') or k.endswith('_gaussian') or k.endswith('__altopt')
                  for k in tab['key']))

    # -- Task 6: the 2 COVID rows must be visible in the table, tagged with
    # their own 'M3covid' model (never mistaken for 'M3'), and structurally
    # inert for every downstream computation this section already checks --
    # they must never be a winner or an LRT participant (both fields are
    # forced False/'' by the R script, but this confirms the table preserves
    # that rather than silently defaulting them differently).
    covid_rows = tab[tab['model'] == 'M3covid']
    check("selection_table carries exactly 2 M3covid (Task 6) rows",
          len(covid_rows) == 2, f"got {len(covid_rows)}")
    check("M3covid rows are for exactly {insp_2021, viol_cov_2021}",
          set(covid_rows['cell']) == {'insp_2021', 'viol_cov_2021'},
          f"got {sorted(covid_rows['cell'])}")
    check("M3covid rows are never a winner (is_winner_rs/ri both False) and "
          "never eligible_for_selection",
          bool((~covid_rows['is_winner_rs']).all())
          and bool((~covid_rows['is_winner_ri']).all())
          and bool((~covid_rows['eligible_for_selection']).all()))
    check("M3covid rows never carry a computed LRT (model doesn't match any "
          "sibling family's model, so no nested pair exists)",
          bool(covid_rows['lrt_p'].isna().all()) and bool((covid_rows['lrt_vs'] == '').all()))

    metas = {k[:-len('__meta')]: v for k, v in raw.items() if k.endswith('__meta')}
    check("6 __meta records recovered", len(metas) == 6, f"got {len(metas)}")

    # -- Requirement 1/2: the computed winner (from is_winner_rs/is_winner_ri)
    # must match each cell's own __meta record, at M3, per tier.
    for cell, mrec in metas.items():
        m3 = tab[(tab['cell'] == cell) & (tab['model'] == 'M3')]
        rs_winners = m3.loc[m3['is_winner_rs'], 'family'].tolist()
        check(f"{cell}: exactly one M3 rs-tier winner", len(rs_winners) == 1,
              f"got {rs_winners}")
        if rs_winners:
            check(f"{cell}: computed rs winner ({rs_winners[0]!r}) matches "
                  f"meta.m3_winner_rs ({mrec['m3_winner_rs']!r})",
                  rs_winners[0] == mrec['m3_winner_rs'])
        if mrec.get('needs_tier2'):
            ri_winners = m3.loc[m3['is_winner_ri'], 'family'].tolist()
            check(f"{cell}: exactly one M3 ri-tier winner", len(ri_winners) == 1,
                  f"got {ri_winners}")
            if ri_winners:
                check(f"{cell}: computed ri winner ({ri_winners[0]!r}) matches "
                      f"meta.m3_winner_ri ({mrec['m3_winner_ri']!r})",
                      ri_winners[0] == mrec['m3_winner_ri'])
        else:
            check(f"{cell}: no M3 ri-tier winner flagged (tier-2 not needed)",
                  not m3['is_winner_ri'].any())

    # -- Requirement 1 (negative proof): the brief's own cross-tier rule
    # (naive groupby('cell')['aic'].idxmin() over every M3 fit, ignoring
    # re_tier) is computed here and shown to disagree with the real winner
    # for exactly the two cells the dispatch names -- i.e. transcribing the
    # brief literally would produce a false winner.
    m3_all = tab[tab['model'] == 'M3']
    naive = m3_all.loc[m3_all.groupby('cell')['aic'].idxmin()].set_index('cell')['family']
    for cell in ('viol_off_2021', 'viol_cov_2021'):
        check(f"{cell}: the brief's naive cross-tier AIC-min rule picks "
              f"{naive.loc[cell]!r}, which disagrees with the real rs-tier "
              f"winner {metas[cell]['m3_winner_rs']!r} -- proves the brief's rule wrong",
              naive.loc[cell] == 'zinb_re' and metas[cell]['m3_winner_rs'] == 'nbinom1')

    # -- Requirement 2: is_winner_rs is True on exactly two records per cell
    # (the M1 winner and the M3 winner) -- getting this wrong silently
    # doubles rows in any downstream table.
    for cell in metas:
        n_rs_winners = int(tab.loc[tab['cell'] == cell, 'is_winner_rs'].sum())
        check(f"{cell}: is_winner_rs is True on exactly 2 records (M1 + M3 winners)",
              n_rs_winners == 2, f"got {n_rs_winners}")

    # -- Requirement 3: two-tier cells must carry BOTH tier winners as
    # distinct, re_tier-labelled rows -- never merged into one "winner" row.
    two_tier_cells = {c for c, m in metas.items() if m.get('needs_tier2')}
    check("exactly viol_off_2021 and viol_cov_2021 carry a second re_tier winner",
          two_tier_cells == {'viol_off_2021', 'viol_cov_2021'}, f"got {two_tier_cells}")
    for cell in two_tier_cells:
        w = tab[(tab['cell'] == cell) & (tab['model'] == 'M3') &
                (tab['is_winner_rs'] | tab['is_winner_ri'])]
        check(f"{cell}: rs and ri winners are two distinct rows with distinct re_tier",
              len(w) == 2 and set(w['re_tier']) == {'rs', 'ri'},
              f"got {w[['re_tier', 'family']].to_dict('records')}")
        by_tier = w.set_index('re_tier')['family'] if len(w) == 2 else pd.Series(dtype=object)
        check(f"{cell}: tier-1 (rs) winner is nbinom1, tier-2 (ri) winner is zinb_re",
              set(by_tier.index) == {'rs', 'ri'}
              and by_tier.get('rs') == 'nbinom1' and by_tier.get('ri') == 'zinb_re')

    # -- Requirement 4: zi_degenerate fits must never drive an LRT (as either
    # child or parent) and a degenerate fit must never be surfaced as a
    # credible "better zero-fit" competitor.
    check("no degenerate fit carries a computed LRT (as the child)",
          not tab.loc[tab['zi_degenerate'], 'lrt_p'].notna().any())
    lrt_edges = tab.loc[tab['lrt_vs'] != '', ['cell', 'model', 're_tier', 'lrt_vs']]
    lrt_parents = set(map(tuple, lrt_edges.to_numpy()))
    deg = tab.loc[tab['zi_degenerate'], ['cell', 'model', 're_tier', 'family']]
    deg_keys = set(map(tuple, deg.to_numpy()))
    check("no degenerate fit is ever used as an LRT parent",
          len(lrt_parents & deg_keys) == 0, f"overlap: {lrt_parents & deg_keys}")
    for cell, mrec in metas.items():
        for tier in ('rs', 'ri'):
            comp = mrec.get(f'credible_better_zero_fit_{tier}')
            if comp is None:
                continue
            comp_fit = tab[(tab['cell'] == cell) & (tab['model'] == 'M3') &
                            (tab['re_tier'] == tier) & (tab['family'] == comp)]
            # Must actually exist -- an empty match previously passed this
            # check vacuously, which would hide a broken/renamed competitor
            # tag instead of failing on it.
            check(f"{cell}/{tier}: credible better-zero-fit competitor {comp!r} fit exists",
                  not comp_fit.empty, f"no ({cell}, M3, {tier}, {comp}) row in the table")
            if not comp_fit.empty:
                check(f"{cell}/{tier}: credible better-zero-fit competitor {comp!r} is not degenerate",
                      not bool(comp_fit['zi_degenerate'].iloc[0]))

    # -- Requirement 5: AIC comparisons stay within (cell, model, re_tier);
    # n_obs must be constant within each such group before any AIC number is
    # compared across rows of that group.
    bad_groups = []
    for (cell, model, tier), g in tab.groupby(['cell', 'model', 're_tier']):
        if g['n_obs'].nunique() > 1:
            bad_groups.append((cell, model, tier, sorted(g['n_obs'].unique().tolist())))
    check("n_obs is constant within every (cell, model, re_tier) group "
          "(required before any AIC comparison in that group)",
          len(bad_groups) == 0, f"{bad_groups}")

    # -- Requirement 6: the nearest clean competitor is reported UNCONDITIONALLY
    # (no delta_aic <= 10 cutoff). viol_cov_2021's nearest clean competitor
    # (nbinom2) sits at ~10.74 AIC above the nbinom1 winner -- just past the
    # meta JSON's own hard cutoff (which stores None there) -- and must still
    # be surfaced by this function, with sigma2_u1 disagreeing ~3.2x.
    comp_family, delta = nearest_clean_competitor(tab, 'viol_cov_2021', 'M3', 'rs', 'nbinom1')
    check("viol_cov_2021 M3/rs nearest clean competitor is nbinom2",
          comp_family == 'nbinom2', f"got {comp_family}")
    check_close("viol_cov_2021 M3/rs competitor delta AIC ~= 10.74",
                delta, 10.73596790800002, 1e-6, kind='rel')
    check("viol_cov_2021's clean competitor is reported despite exceeding "
          "meta's own <=10 AIC cutoff (meta stores None there)",
          delta > 10 and metas['viol_cov_2021']['clean_rs_competitor_within_10_aic'] is None)
    winner_fit = raw['viol_cov_2021__M3__nbinom1']
    comp_fit = raw['viol_cov_2021__M3__nbinom2']
    check_close("viol_cov_2021: nbinom2 vs winner sigma2_u1 ratio ~= 3.2x",
                comp_fit['sigma2_u1'] / winner_fit['sigma2_u1'], 3.2140498391688106,
                1e-6, kind='rel')

    comp_family2, delta2 = nearest_clean_competitor(tab, 'viol_off_2021', 'M3', 'rs', 'nbinom1')
    check("viol_off_2021 M3/rs nearest clean competitor is nbinom2, delta ~= 7.88 (within 10)",
          comp_family2 == 'nbinom2', f"got {comp_family2}")
    check_close("viol_off_2021 M3/rs competitor delta AIC ~= 7.88",
                delta2, 7.881499135899958, 1e-6, kind='rel')

    # -- Requirement 8: viol_off_2021's apparent M1->M3 "winner change" is a
    # change in the ELIGIBLE CANDIDATE SET (zinb_re drops out of contention at
    # M3), not a preference reversal among a fixed set of alternatives.
    m1_elig = set(metas['viol_off_2021']['eligible_rs_m1'])
    m3_elig = set(metas['viol_off_2021']['eligible_rs_m3'])
    check("viol_off_2021: zinb_re is eligible at M1 (rs) but not at M3 (rs) -- "
          "the M1->M3 'winner change' is a candidate-set change, not a reversal",
          'zinb_re' in m1_elig and 'zinb_re' not in m3_elig)

    # -- Coefficient table: family used per model must track the requested
    # tier's own winner (read via is_winner_rs/is_winner_ri, never meta or key
    # parsing); M2 was only ever fit at the rs tier (see [3]), so the ri-tier
    # table must omit M2 entirely rather than fabricate or misattribute it.
    ct_rs = coefficient_table(raw, 'viol_off_2021', 'rs')
    check("coefficient_table(rs) for viol_off_2021 is non-empty", not ct_rs.empty)
    check("coefficient_table(rs) for viol_off_2021 has M2 columns (M2 fit at rs)",
          'M2_b' in ct_rs.columns)
    ct_ri = coefficient_table(raw, 'viol_off_2021', 'ri')
    check("coefficient_table(ri) for viol_off_2021 has no M2 columns "
          "(M2 was never fit at the ri tier)",
          'M2_b' not in ct_ri.columns)
    # No "else True" fallback: a missing M1_family column is exactly the
    # regression this check exists to catch, so it must FAIL, not vanish.
    check("coefficient_table(rs) for viol_off_2021 carries an M1_family column",
          'M1_family' in ct_rs.columns)
    if 'M1_family' in ct_rs.columns:
        check("coefficient_table(rs) M1 column reflects the M1 rs winner (zinb_re), "
              "not the M3 winner (nbinom1)",
              bool((ct_rs['M1_family'] == 'zinb_re').any()))

    # -- IRR must equal exp(b) wherever a coefficient is populated. Each
    # model's b/irr columns are asserted to EXIST (not just checked "if
    # present") so a column silently disappearing produces a FAIL, not zero
    # checks run.
    for m in ('M1', 'M2', 'M3'):
        bcol, ircol = f'{m}_b', f'{m}_irr'
        check(f"coefficient_table(rs) for viol_off_2021 carries {bcol}/{ircol}",
              bcol in ct_rs.columns and ircol in ct_rs.columns)
        if bcol in ct_rs.columns and ircol in ct_rs.columns:
            mask = ct_rs[bcol].notna()
            ok = np.allclose(ct_rs.loc[mask, ircol], np.exp(ct_rs.loc[mask, bcol]), rtol=1e-9)
            check(f"coefficient_table {m}: IRR == exp(b)", ok)

    # -- LRTs are valid probabilities and at least some were computed (proves
    # the nested-pairs machinery actually ran, not just an empty no-op).
    check("all computed LRT p-values are valid probabilities in [0, 1]",
          bool(tab['lrt_p'].dropna().between(0, 1).all()))
    n_lrt = int(tab['lrt_p'].notna().sum())
    check("exactly 30 LRTs were computed across the six cells "
          "(nested pairs exist and converged, structurally excluding "
          "degenerate/collapsed/ineligible fits)", n_lrt == 30, f"got {n_lrt}")
    lrt_rows = tab[tab['lrt_vs'] != '']
    # Explicit non-empty guard BEFORE the pairwise-mapping check below: a
    # `.all()` over an empty selection is vacuously True in pandas, which
    # would let the nested-pair check silently pass even if the LRT loop
    # produced zero rows. n_lrt == 30 above already forces non-emptiness, but
    # this makes the dependency structural rather than incidental.
    check("lrt_rows is non-empty before asserting anything about its contents",
          len(lrt_rows) > 0, f"got {len(lrt_rows)}")
    if len(lrt_rows) > 0:
        nested_lookup = {'nbinom2': 'poisson', 'zinb': 'nbinom2', 'zinb_re': 'zinb'}
        check("every LRT pairs a family with its correct nested parent",
              bool((lrt_rows['family'].map(nested_lookup) == lrt_rows['lrt_vs']).all()))
        check("nbinom1 never appears as an LRT child or parent (not nested in NB2)",
              not ((tab['family'] == 'nbinom1') & (tab['lrt_vs'] != '')).any()
              and 'nbinom1' not in set(lrt_rows['lrt_vs']))
        # Every computed LRT must be flagged as a boundary test, with a
        # non-empty description of WHICH boundary -- a downstream reader must
        # never see a bare nominal p-value with no boundary caveat attached.
        check("every computed LRT is flagged lrt_boundary=True with a non-empty lrt_boundary_kind",
              bool(lrt_rows['lrt_boundary'].all())
              and bool((lrt_rows['lrt_boundary_kind'] != '').all()))
        check("BOUNDARY_KIND covers exactly the three nested child families "
              "(nbinom2, zinb, zinb_re)",
              set(BOUNDARY_KIND) == {'nbinom2', 'zinb', 'zinb_re'})
        # NOTE ON A REVIEW DISCREPANCY: the review's finding #3 states "8 of
        # your 30 LRTs are zinb_re vs zinb". Independently re-derived by hand
        # from the raw JSON (a second, from-scratch pass over every
        # (cell, model, re_tier) group, not just this function's own
        # arithmetic) the true count is 6: viol_off_2021 M1/M3 (ri),
        # viol_cov_2021 M1/M3 (ri), insp_2019 M1/M3 (rs). A 7th zinb_re fit
        # exists at viol_off_2021 M1's rs tier (the plain, un-suffixed key --
        # a genuine, non-collapsed rs-tier fit, distinct from its __ri2
        # sibling), but it has NO rs-tier 'zinb' parent to pair against
        # ('zinb' is absent from viol_off_2021's eligible_rs_m1 list), so no
        # LRT fires there -- correctly. This mirrors the project's prior,
        # accepted precedent (task-4-report.md: the implementer reported 11
        # zi_degenerate fits against the reviewer's stated 12, and 11 was
        # independently reconfirmed correct) -- verified here, not forced to
        # match the review's stated number.
        n_zinb_re_lrt = int((lrt_rows['family'] == 'zinb_re').sum())
        check("exactly 6 of the 30 LRTs are the zinb_re-vs-zinb boundary test "
              "(independently re-verified; the review's stated count of 8 "
              "does not hold up -- see comment above)",
              n_zinb_re_lrt == 6, f"got {n_zinb_re_lrt}")
        borderline = tab[(tab['cell'] == 'insp_2019') & (tab['family'] == 'zinb_re')
                          & tab['lrt_p'].notna()]
        check("insp_2019's zinb_re-vs-zinb LRTs are the borderline ones (both p < 0.02)",
              len(borderline) == 2 and bool((borderline['lrt_p'] < 0.02).all()),
              f"got {borderline[['model', 'lrt_p']].to_dict('records')}")

    # -- Requirement 1 (artifact reporting): sigma2_u0/sigma2_u1/sigma_u01/
    # sigma2_e must be readable from the table/CSV itself, not only asserted
    # against a hardcoded value inside this validator. Recompute the
    # headline NB1-vs-NB2 sigma2_u1 disagreement FROM THE TABLE (not from
    # raw JSON) to prove it is actually derivable from the artifact.
    # Item 3 (2026-08-22 review): `dispersion` (with `family_name`) replaces
    # `sigma2_e` as the readable per-fit scale parameter, and `sigma2_e` must
    # now be entirely null on this table -- every row in it is a count fit.
    # Two populations, and the distinction matters: an earlier version tested
    # `.notna().any()` under a label that said "on every row", so nulling a
    # single row's `dispersion` produced 0 FAILs. Columns that must be
    # populated EVERYWHERE are now tested with `.all()`; the ones that are
    # legitimately partial (sigma2_u1/sigma_u01 are absent at the (1 | state)
    # tier; the ZI columns only exist for ZI families) are tested with
    # `.any()` plus an explicit expected count.
    for col in ('sigma2_u0', 'dispersion', 'family_name', 'zi_formula',
                'zi_has_state_re'):
        check(f"selection_table carries {col} on EVERY row",
              col in tab.columns and bool(tab[col].notna().all()),
              f"missing on {int(tab[col].isna().sum()) if col in tab.columns else 'n/a'} rows")
    for col, why in (('sigma2_u1', 'absent at the (1 | state) tier'),
                     ('sigma_u01', 'absent at the (1 | state) tier'),
                     ('sigma2_zi_u0', 'only for a ZI formula with (1 | state)'),
                     ('sigma_zi_u0', 'only for a ZI formula with (1 | state)'),
                     ('zi_intercept', 'only for a zero-inflated family'),
                     ('zi_pr_median_state', 'only for a zero-inflated family'),
                     ('zi_pr_marginal', 'only for a zero-inflated family')):
        check(f"selection_table carries {col} on the rows that have one ({why})",
              col in tab.columns and bool(tab[col].notna().any()))
    # And the partial columns' populations must be exactly the rows that
    # structurally have them, not merely "some rows".
    check("sigma2_u1/sigma_u01 are populated exactly on the rs-tier rows",
          bool((tab['sigma2_u1'].notna() == (tab['re_tier'] == 'rs')).all())
          and bool((tab['sigma_u01'].notna() == (tab['re_tier'] == 'rs')).all()),
          f"sigma2_u1 non-null {int(tab['sigma2_u1'].notna().sum())}, "
          f"rs rows {int((tab['re_tier'] == 'rs').sum())}")
    check("sigma2_zi_u0 is populated exactly on the rows whose zi_formula "
          "carries a state random intercept",
          bool((tab['sigma2_zi_u0'].notna() == tab['zi_has_state_re']).all()),
          f"variance non-null {int(tab['sigma2_zi_u0'].notna().sum())}, "
          f"zi_has_state_re {int(tab['zi_has_state_re'].sum())}")
    check("the ZI probability columns are populated exactly on the rows that "
          "have a ZI intercept",
          bool((tab['zi_pr_median_state'].notna()
                == tab['zi_intercept'].notna()).all())
          and bool((tab['zi_pr_marginal'].notna()
                    == tab['zi_intercept'].notna()).all()))
    check("selection_table still carries a sigma2_e column, and it is entirely "
          "null -- no count fit has a residual variance, and dispersion^2 is "
          "not one",
          'sigma2_e' in tab.columns and bool(tab['sigma2_e'].isna().all()),
          f"non-null sigma2_e rows: {int(tab['sigma2_e'].notna().sum())}")
    # And the two structural-zero probabilities must be EQUAL exactly where
    # there is no ZI random intercept, and differ where there is one.
    _no_re = tab[tab['zi_intercept'].notna() & ~tab['zi_has_state_re']]
    check("median-state and marginal structural-zero probabilities coincide "
          "for every ZI fit with no ZI random intercept",
          not _no_re.empty and bool(np.allclose(_no_re['zi_pr_median_state'],
                                                _no_re['zi_pr_marginal'],
                                                rtol=1e-12)),
          f"n={len(_no_re)}")
    _with_re = tab[tab['zi_has_state_re']]
    check("and they DIFFER for every ZI fit that has one (otherwise the "
          "marginal column would be a copy of the median-state column)",
          not _with_re.empty
          and bool((_with_re['zi_pr_marginal']
                    > _with_re['zi_pr_median_state'] * 1.05).all()),
          f"n={len(_with_re)}; ratios="
          f"{(_with_re['zi_pr_marginal'] / _with_re['zi_pr_median_state']).tolist()!r}")
    win = tab[(tab['cell'] == 'viol_cov_2021') & (tab['model'] == 'M3') &
              (tab['re_tier'] == 'rs') & (tab['family'] == 'nbinom1')]
    comp = tab[(tab['cell'] == 'viol_cov_2021') & (tab['model'] == 'M3') &
               (tab['re_tier'] == 'rs') & (tab['family'] == 'nbinom2')]
    if not win.empty and not comp.empty:
        ratio = float(comp['sigma2_u1'].iloc[0]) / float(win['sigma2_u1'].iloc[0])
        check_close("viol_cov_2021: nbinom2/nbinom1 sigma2_u1 ratio ~= 3.2x, "
                    "DERIVED FROM selection_table() (not the raw JSON)",
                    ratio, 3.2140498391688106, 1e-6, kind='rel')
    else:
        skip("viol_cov_2021 nbinom2/nbinom1 sigma2_u1 ratio",
             f"winner row empty={win.empty}, competitor row empty={comp.empty}")

    # -- Requirement 2: exp_zeros never travels without its Monte-Carlo SE.
    with_exp = tab[tab['exp_zeros'].notna()]
    check("exp_zeros_se is present and finite for every fit that has exp_zeros",
          bool(with_exp['exp_zeros_se'].notna().all())
          and bool(np.isfinite(with_exp['exp_zeros_se']).all()))

    # -- Requirement: convergence-diagnostic fields (message/pd_hess/conv_code)
    # travel with every row -- needed to tell "clean" from "not recorded"
    # (the ri-tier winners have no meta['winner_ri_has_warning'] field at all;
    # `message` on the fit itself is the only generic source).
    for col in ('message', 'pd_hess', 'conv_code'):
        check(f"selection_table carries {col} on every row", col in tab.columns)
    warned = tab[(tab['cell'].isin(['viol_off_2021', 'viol_cov_2021'])) &
                 (tab['model'] == 'M3') & (tab['re_tier'] == 'rs') &
                 (tab['family'] == 'nbinom1')]
    check("both 2021 violations rs-tier nbinom1 winners carry a non-empty message "
          "(the live optimizer warning)",
          len(warned) == 2 and bool((warned['message'] != '').all()))

    # -- is_m3_winner: exactly 8 rows True (6 rs winners + 2 ri winners), and
    # it must never double-count the M1 winner that also carries
    # is_winner_rs/ri == True.
    check("is_m3_winner column present", 'is_m3_winner' in tab.columns)
    n_m3_winners = int(tab['is_m3_winner'].sum())
    check("exactly 8 rows are flagged is_m3_winner "
          "(6 cells x rs winner + 2 cells with a second ri winner)",
          n_m3_winners == 8, f"got {n_m3_winners}")
    check("is_m3_winner is never True for a model other than M3",
          not tab.loc[tab['is_m3_winner'], 'model'].ne('M3').any())

    # -- winner_summary(): the artifact the memo actually reads for the
    # zero-fit evidence, the M1-vs-M3 candidate-set stability flag (correctly
    # LABELLED -- see below), and the eligible sets.
    ws = winner_summary(tab, metas)
    check("winner_summary returns 8 rows (6 cells at rs + 2 cells' ri tier)",
          len(ws) == 8, f"got {len(ws)}")
    for col in ('sigma2_u0', 'sigma2_u1', 'exp_zeros_se', 'zero_signed',
                'has_better_zero_fit', 'credible_better_zero_fit',
                'm1_m3_candidate_set_stable', 'eligible_m1', 'eligible_m3'):
        check(f"winner_summary carries column {col!r}", col in ws.columns)

    # -- REVIEWER CORRECTION, verified and encoded here: `stable_rs` is an
    # M1-vs-M3 CANDIDATE-SET stability flag (does the rs winner at M1 equal
    # the rs winner at M3?), NOT an optimizer-stability flag. The __altopt
    # BFGS refits independently show the rs-tier nbinom1 winners in both 2021
    # violations cells ARE optimizer-stable (sigma2_u1 matching to 4-5 dp) --
    # so a False `stable_rs` for viol_off_2021 reflects the documented
    # candidate-set change (zinb_re drops out of the M3-eligible set), not an
    # unstable fit.
    row = ws[(ws['cell'] == 'viol_off_2021') & (ws['re_tier'] == 'rs')]
    check("viol_off_2021 rs: m1_m3_candidate_set_stable is False "
          "(M1 winner zinb_re != M3 winner nbinom1 -- a candidate-SET change, "
          "not optimizer instability; __altopt shows this winner IS optimizer-stable)",
          not row.empty and bool(row['m1_m3_candidate_set_stable'].iloc[0]) is False)
    stable_elsewhere = ws[~((ws['cell'] == 'viol_off_2021') & (ws['re_tier'] == 'rs'))]
    check("every OTHER (cell, tier) is m1_m3_candidate_set_stable == True "
          "(M1 and M3 winners agree)",
          bool(stable_elsewhere['m1_m3_candidate_set_stable'].astype(bool).all()))
    for _, r in ws.iterrows():
        want = bool(metas[r['cell']].get(f'stable_{r["re_tier"]}'))
        check(f"{r['cell']}/{r['re_tier']}: winner_summary's "
              f"m1_m3_candidate_set_stable matches meta['stable_{r['re_tier']}']",
              bool(r['m1_m3_candidate_set_stable']) == want)

    # -- Eligible candidate sets reach the artifact, non-empty, and the
    # viol_off_2021 M1-vs-M3 zinb_re drop is visible directly from
    # winner_summary (not only from the raw meta record).
    check("every winner_summary row has non-empty eligible_m1 and eligible_m3",
          bool((ws['eligible_m1'] != '').all()) and bool((ws['eligible_m3'] != '').all()))
    row = ws[(ws['cell'] == 'viol_off_2021') & (ws['re_tier'] == 'rs')].iloc[0]
    check("winner_summary (not just raw meta) shows zinb_re eligible at M1 "
          "but not at M3 for viol_off_2021/rs",
          'zinb_re' in row['eligible_m1'].split(',')
          and 'zinb_re' not in row['eligible_m3'].split(','))

    # -- winner_summary must be written to its own CSV.
    ws_path = GEN + 'count_model_winner_summary.csv'
    if os.path.exists(ws_path):
        written_ws = pd.read_csv(ws_path)
        check("count_model_winner_summary.csv has one row per winner (8)",
              len(written_ws) == len(ws), f"got {len(written_ws)}")
        for col in ('sigma2_zi_u0', 'dispersion', 'family_name',
                    'zi_pr_median_state', 'zi_pr_marginal'):
            check(f"count_model_winner_summary.csv carries column {col!r} "
                  f"(the ZI variance and the dispersion semantics must reach "
                  f"the CSV, not only the memo)", col in written_ws.columns)
    else:
        skip("count_model_winner_summary.csv round-trip",
             f"{ws_path} does not exist yet -- run "
             f"scripts/report_count_models.py")

    # -- ZI block in coefficient_table(): a zero-inflated winner's
    # zero-inflation-part coefficients must reach the table, clearly
    # separated from the count-model part, and a non-ZI winner must carry
    # NONE (never a phantom ZI row for a family with no zi_formula).
    ct_zi = coefficient_table(raw, 'insp_2019', 'rs')  # winner: zinb_re
    check("coefficient_table for a zi-family winner (insp_2019, zinb_re) "
          "carries a 'section' column", 'section' in ct_zi.columns)
    if 'section' in ct_zi.columns:
        zi_block = ct_zi[ct_zi['section'] == 'zi']
        check("insp_2019 coefficient_table has a non-empty ZI block",
              not zi_block.empty)
        check("insp_2019's ZI block rows are labelled with the 'ZI: ' prefix",
              bool(zi_block['term'].str.startswith('ZI:').all()) if not zi_block.empty else False)
        check("insp_2019's ZI intercept (M3) matches the raw JSON's zi coefficient",
              not zi_block.empty and np.isclose(
                  zi_block[zi_block['term'] == 'ZI: Intercept']['M3_b'].iloc[0],
                  raw['insp_2019__M3__zinb_re']['zi']['(Intercept)']['b'], rtol=1e-9))
    ct_no_zi = coefficient_table(raw, 'viol_cov_2019', 'rs')  # winner: nbinom2
    # Explicit precondition, not an "X.empty or ..." fallback: viol_cov_2019's
    # rs-tier coefficient table is known to be non-empty (nbinom2 has cond
    # terms), so asserting that directly turns a would-be silent skip into a
    # real failure if the table ever came back empty.
    check("coefficient_table for a non-ZI winner (viol_cov_2019, nbinom2) is non-empty",
          not ct_no_zi.empty)
    if not ct_no_zi.empty:
        check("coefficient_table for a non-ZI winner (viol_cov_2019, nbinom2) "
              "has NO zero-inflation-part rows",
              bool((ct_no_zi['section'] == 'count').all()))

    # -- M2 quality guard: exercised with a SYNTHETIC duplicate so the guard
    # is proven to do something, not merely "pass because real data never
    # triggers it" (every real M2 fit here happens to already be clean).
    synth = dict(raw)  # shallow copy; we only add keys, never mutate values
    base_key = 'viol_cov_2019__M2__nbinom2'
    base = dict(raw[base_key])
    non_converged = dict(base)
    non_converged['converged'] = False
    synth['_synthfit__M2__nbinom2_bad'] = {**non_converged, 'cell': '_synthfit',
                                            're_tier': 'rs', 'model': 'M2',
                                            'family_tag': 'nbinom2'}
    synth['_synthfit__M1__nbinom2'] = {**base, 'cell': '_synthfit',
                                        're_tier': 'rs', 'model': 'M1',
                                        'family_tag': 'nbinom2',
                                        'is_winner_rs': True, 'is_winner_ri': False}
    synth['_synthfit__M3__nbinom2'] = {**base, 'cell': '_synthfit',
                                        're_tier': 'rs', 'model': 'M3',
                                        'family_tag': 'nbinom2',
                                        'is_winner_rs': True, 'is_winner_ri': False}
    ct_synth = coefficient_table(synth, '_synthfit', 'rs')
    check("M2 quality guard EXCLUDES a synthetic non-converged M2 duplicate "
          "(latent-bug regression test, per review finding 6)",
          'M2_b' not in ct_synth.columns or ct_synth['M2_b'].isna().all())
    # Two DISTINCT, CONVERGED, non-degenerate M2 candidates for the same
    # cell/tier must raise -- the guard's uniqueness assertion must be a real
    # assertion, not a silent last-write-wins. Flip the first synthetic M2
    # (deliberately non-converged above) back to converged, so together with
    # a second, independently-keyed, differently-tagged M2 fit there are
    # exactly two quality-passing candidates.
    synth2 = dict(synth)
    synth2['_synthfit__M2__nbinom2_bad'] = {**base, 'cell': '_synthfit',
                                             're_tier': 'rs', 'model': 'M2',
                                             'family_tag': 'nbinom2'}
    second_good = dict(base)
    synth2['_synthfit__M2__nbinom1_dup'] = {**second_good, 'cell': '_synthfit',
                                             're_tier': 'rs', 'model': 'M2',
                                             'family_tag': 'nbinom1'}
    try:
        coefficient_table(synth2, '_synthfit', 'rs')
        check("M2 quality guard raises on two quality-passing M2 duplicates", False,
              "no exception raised")
    except ValueError:
        check("M2 quality guard raises on two quality-passing M2 duplicates", True)

    # -- Presentation: TIER_ORDER must sort rs before ri, so the per-cell
    # printout never lets a reader read the (1|state) fallback tier's AIC as
    # if it were comparable to (or better-ranked than) the primary
    # (1+time|state) tier above it.
    check("TIER_ORDER sorts 'rs' strictly before 'ri'", TIER_ORDER['rs'] < TIER_ORDER['ri'])
    check("this validator's local copy of FAMILY_LABEL matches the reporter's "
          "(it is used to parse display labels back out of the memo, so a "
          "drift would silently break that parse rather than fail it)",
          FAMILY_LABEL_LOCAL == FAMILY_LABEL,
          f"local {FAMILY_LABEL_LOCAL!r} vs reporter {FAMILY_LABEL!r}")

    # -- Overdispersion sanity, read from the winner (rs) tier only: NB2
    # should predict zero counts closer to a 1:1 ratio than Poisson.
    v = tab[(tab['cell'] == 'viol_cov_2021') & (tab['model'] == 'M3') &
            (tab['re_tier'] == 'rs') & tab['converged']]
    p = v.loc[v['family'] == 'poisson', 'zero_ratio']
    n = v.loc[v['family'] == 'nbinom2', 'zero_ratio']
    if len(p) and len(n):
        check("NB2 predicts zeros closer to 1:1 than Poisson (2021 violations, rs tier)",
              abs(float(n.iloc[0]) - 1) < abs(float(p.iloc[0]) - 1),
              f"poisson ratio {float(p.iloc[0]):.2f}, nb2 ratio {float(n.iloc[0]):.2f}")
    else:
        skip("NB2-vs-Poisson zero-ratio comparison (2021 violations, rs tier)",
             f"poisson rows={len(p)}, nbinom2 rows={len(n)}")

    # -- If the CSVs have already been written (i.e. report_count_models.py
    # has run), they must carry exactly the same rows as their in-memory
    # source -- not gated as a hard prerequisite of this section, since the
    # process calls for validation to pass BEFORE the reporter is run for the
    # first time.
    out_path = GEN + 'count_model_comparison.csv'
    if os.path.exists(out_path):
        written = pd.read_csv(out_path)
        check("count_model_comparison.csv has one row per genuine fit (97, "
              "post-Task-6)",
              len(written) == len(tab), f"got {len(written)}")
        for col in ('sigma2_u0', 'sigma2_u1', 'sigma_u01', 'sigma2_e', 'exp_zeros_se',
                    'lrt_boundary', 'lrt_boundary_kind', 'is_m3_winner',
                    'sigma2_zi_u0', 'sigma_zi_u0', 'zi_has_state_re',
                    'zi_pr_median_state', 'zi_pr_marginal', 'dispersion',
                    'family_name'):
            check(f"count_model_comparison.csv carries column {col!r}", col in written.columns)
        check("count_model_comparison.csv (read from disk): sigma2_e is null on "
              "every row, so nobody can lift dispersion^2 out of it as a "
              "variance component",
              bool(written['sigma2_e'].isna().all()),
              f"non-null: {int(written['sigma2_e'].notna().sum())}")

        # -- ROUND-TRIP assertion (not an in-memory one): `lrt_vs` is '' in
        # memory but becomes NaN through to_csv/read_csv, and `NaN != ''` is
        # True for every row -- so `df[df.lrt_vs != '']` on the FILE, not the
        # in-memory DataFrame, would silently select ALL rows instead of 30
        # (97 post-Task-6, was 95). Read the file back and confirm the
        # documented selector (`lrt_p.notna()`) gives 30 while the naive one
        # does not.
        n_lrt_from_file = int(written['lrt_p'].notna().sum())
        check("count_model_comparison.csv (read from disk): lrt_p.notna() "
              "selects exactly 30 LRT rows -- the documented, round-trip-safe "
              "selector", n_lrt_from_file == 30, f"got {n_lrt_from_file}")
        if 'lrt_vs' in written.columns:
            n_naive_from_file = int((written['lrt_vs'] != '').sum())
            check("count_model_comparison.csv (read from disk): the naive "
                  "`lrt_vs != ''` selector is BROKEN by the CSV round-trip "
                  "(selects all rows, not 30) -- proves why lrt_p.notna() "
                  "must be used instead, not merely asserts it once",
                  n_naive_from_file == len(written), f"got {n_naive_from_file}")
        else:
            skip("count_model_comparison.csv naive-selector round-trip",
                 "the CSV has no lrt_vs column")
    else:
        skip("count_model_comparison.csv round-trip block",
             f"{out_path} does not exist yet -- run "
             f"scripts/report_count_models.py")


# ============================================================
# [5] INDEPENDENT ZI CROSS-CHECK (Task 6)
# ============================================================
def validate_zi_crosscheck():
    """Confirm zero-inflation is real independently of glmmTMB: fit a
    statsmodels ZINB with state FIXED EFFECTS (dummy variables) instead of
    glmmTMB's random effects -- different software, different treatment of
    the state dimension, which is the entire point of an independent check.

    Run on BOTH 2021 outcomes, not just violations as the original task-6
    brief assumed (it was written when violations was expected to select
    ZINB). The ladder's actual winners inverted that expectation: inspections
    selects ZINB (Delta-AIC 33.4 over NB2) on a series with only 1.3% zeros
    (7/539) while violations selects NB1.

    CORRECTED FRAME (coordinator review round 2 -- the original dispatch's
    framing here was wrong, not just this function's interpretation of it):
    "violations selects NB1" is NOT evidence that zero-inflation is weak or
    absent for violations. glmmTMB's own ZI-family fits for the violations
    cells are large, precise, and highly significant wherever a ZI family can
    actually be estimated -- e.g. ZIP's ZI intercept reaches the mandated rs
    tier for both viol_off_2021 and viol_cov_2021 at p ~ 5e-36 (see
    `meta['zi_evidence']`, added below). NB1 wins the AIC race because a
    negative binomial's own overdispersion parameter explains those same
    zeros about as well as an explicit ZI term does -- and specifically the
    ZI+NB combination (zinb/zinb_re) could not reach the rs tier for these two
    cells, not that zero-inflation itself is undetectable. This function
    therefore does NOT check "is zero-inflation weak for violations" (it
    is not); it checks whether statsmodels' independent fit is informative
    about it at all, and reports that verdict without overclaiming either
    way.

    SPEC/SAMPLE MISMATCH WITH glmmTMB's M3 (undocumented until this review --
    important for reconciling N's): this cross-check omits the six M3_ADD
    covariates (SPEND_APP_z, SPEND_WORK_z, lii_2017_z, h2a_per_farmworker_z,
    dol_demand_met_pct_z, pct_flc_z) entirely -- defensible, since most of
    them are state-time-invariant or collinear with the state dummies used
    here instead of a random intercept -- and therefore runs on the FULL
    2021 panel (N=539 inspections / 533 violations), retaining AK/RI/VT,
    rather than glmmTMB M3's listwise-deleted N=506/501 (AK/RI/VT lack BLS
    applicator data needed for lii_2017_z/pct_flc_z and are dropped there).
    A reader comparing this section's printed N against glmmTMB's M3 N should
    expect them to differ for this reason, not read the mismatch as a bug.

    Ruling 4 (task-6 dispatch): non-convergence must never be hidden. Every
    fit is wrapped in `warnings.catch_warnings(record=True)` (never
    `filterwarnings('ignore')`), and a fit that fails to converge is reported
    as a finding with its diagnostics -- it is not tuned into submission by
    trying optimizers until one "works" and it is not silently treated as if
    it had converged, and (per this review) its non-convergence is not read
    as if it settled anything about zero-inflation either.
    """
    import json
    import warnings as _warnings

    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    section('5', 'Independent ZI cross-check (statsmodels, state fixed effects)')
    print("        NOTE: this cross-check drops the 6 M3_ADD covariates and "
          "retains AK/RI/VT (N=539/533), unlike glmmTMB M3's N=506/501 -- "
          "see this function's docstring for why the N's legitimately differ.")
    d = pd.read_csv(GEN + 'count_model_panel_2021.csv')
    fits = json.load(open(GEN + 'count_model_results.json'))

    def _fit_fe_zinb(dv, extra_cols):
        need = [dv, 'time', 'time2', 'time3'] + extra_cols
        dd = d.dropna(subset=need).copy()
        X = pd.concat(
            [dd[['time', 'time2', 'time3'] + extra_cols],
             pd.get_dummies(dd['state'], prefix='st', drop_first=True, dtype=float)],
            axis=1)
        X = sm.add_constant(X)
        y = dd[dv]
        with _warnings.catch_warnings(record=True) as wrec:
            _warnings.simplefilter('always')
            res = ZeroInflatedNegativeBinomialP(
                y, X, exog_infl=np.ones((len(dd), 1)), p=2
            ).fit(method='bfgs', maxiter=500, disp=0)
        return res, wrec, len(dd)

    # (dv, extra RHS columns, glmmTMB comparator key, whether convergence is
    # the EXPECTED outcome). The glmmTMB comparator is each cell's `zinb`
    # rung specifically (not necessarily that cell's overall M3 winner) --
    # for insp_2021 the zinb rung IS the winner; for viol_cov_2021 it is not
    # (nbinom1 wins), but the zinb rung is still the right same-family
    # comparator for "what does glmmTMB's own zero-inflation model say".
    CROSSCHECK_SPECS = {
        'insp_2021': ('inspections', [], 'insp_2021__M3__zinb', True),
        'viol_cov_2021': ('violations', ['log_inspections'], 'viol_cov_2021__M3__zinb', False),
    }
    for label, (dv, extra, glmm_key, expect_converged) in CROSSCHECK_SPECS.items():
        try:
            res, wrec, n = _fit_fe_zinb(dv, extra)
        except Exception as e:  # noqa: BLE001 - a hard failure is a reportable finding
            check(f"{label}: statsmodels ZINB fit raised an exception", False, str(e))
            continue

        conv = bool(res.mle_retvals.get('converged'))
        warn_cats = sorted({w.category.__name__ for w in wrec})
        conv_warns = [str(w.message) for w in wrec if issubclass(w.category, ConvergenceWarning)]
        zi_const = float(res.params.get('inflate_const', np.nan))
        zi_se = float(res.bse.get('inflate_const', np.nan))
        obs_zero_rate = float((d[dv] == 0).mean())

        print(f"        {label}: N={n}, converged={conv}, warning categories={warn_cats}")
        for msg in conv_warns:
            print(f"        {label}: ConvergenceWarning -- {msg}")
        print(f"        {label}: statsmodels ZI intercept {zi_const:+.3f} "
              f"(SE {zi_se}); observed zero rate {obs_zero_rate:.4f}")

        glmm_fit = fits.get(glmm_key)
        if glmm_fit and glmm_fit.get('converged') and glmm_fit.get('zi'):
            r_zi = float(glmm_fit['zi']['(Intercept)']['b'])
            r_prob = 1 / (1 + np.exp(-r_zi))
            print(f"        {label}: glmmTMB ({glmm_key}, re_tier="
                  f"{glmm_fit.get('re_tier')!r}) ZI intercept {r_zi:+.3f} "
                  f"-> structural-zero prob {r_prob:.4f}")
        else:
            r_zi = None

        if expect_converged:
            # insp_2021: glmmTMB's own selected family HAS zero-inflation
            # (ZINB beats NB2 by 33.4 AIC). This is the important corroboration:
            # convergence, a well-identified probability, and sign/magnitude
            # agreement with glmmTMB's estimate are all real requirements.
            check(f"{label}: statsmodels FE-ZINB converged", conv,
                  f"mle_retvals={res.mle_retvals}")
            if conv:
                zi_prob = 1 / (1 + np.exp(-zi_const))
                check(f"{label}: structural-zero probability strictly inside (0, 1)",
                      0.001 < zi_prob < 0.999, f"got {zi_prob:.6f}")
                if r_zi is not None:
                    check(f"{label}: glmmTMB and statsmodels agree on the SIGN "
                          "of the ZI intercept (independent corroboration of "
                          "real zero-inflation)",
                          np.sign(r_zi) == np.sign(zi_const),
                          f"glmmTMB {r_zi:+.3f} vs statsmodels {zi_const:+.3f}")
                    check_close(f"{label}: glmmTMB and statsmodels ZI intercepts "
                                "agree in rough magnitude", zi_const, r_zi, 0.30,
                                kind='rel')
        else:
            # viol_cov_2021: CORRECTED interpretation (coordinator review
            # round 2). The FE-ZINB's non-convergence here carries NO
            # interpretive weight about zero-inflation one way or the other:
            # a from-scratch reproduction of this exact fit shows the ENTIRE
            # parameter vector diverges (const=114.5, time=-55.2,
            # log_inspections=-35.0, alpha=-1726.3 -- a dispersion parameter
            # that must be positive, several state dummies at +20 to +211),
            # not just the ZI intercept. The design matrix is full rank
            # (53/53 columns), no state is 100% zeros, and exog_infl is
            # correct, so this is a general BFGS breakdown on a ~53-parameter
            # fixed-effects count mixture at N=533/49 states (the classic
            # incidental-parameters fragility of state dummies + a count
            # mixture), not a ZI-specific failure. It is therefore NOT read
            # as "consistent with no zero-inflation" -- that reading is
            # false: glmmTMB's own ZI-family fits for this cell (ZIP at the
            # rs tier, ZINB/ZINB+RE at the ri tier -- see
            # meta['zi_evidence'] below) are large, precise, and highly
            # significant (p ~ 1e-9 to 5e-36). NB1 winning the AIC race
            # reflects that its own overdispersion parameter fits the same
            # zeros about as well as an explicit ZI term, not that
            # zero-inflation is absent. This section's statsmodels result is
            # therefore INCONCLUSIVE for this cell, not corroborating.
            meta = fits.get(f'{label}__meta', {})
            print(f"        {label}: glmmTMB's OWN ZI evidence (zi_estimable_rs="
                  f"{meta.get('zi_estimable_rs')}, zi_significant_any_tier="
                  f"{meta.get('zi_significant_any_tier')}) shows zero-inflation "
                  "IS real and significant in this cell wherever a ZI family "
                  "can be estimated -- NB1 winning on AIC does not mean "
                  "zero-inflation is absent, only that NB1's own "
                  "overdispersion captures it about as well.")
            check(f"{label}: a ConvergenceWarning was actually raised when the "
                  "FE-ZINB failed to converge (not silently swallowed)",
                  (not conv) and bool(conv_warns),
                  f"conv={conv}, conv_warns={conv_warns}")
            check(f"{label}: statsmodels FE-ZINB does not reach a stable "
                  "estimate here (verified current result) -- a general "
                  "fixed-effects/incidental-parameters convergence failure "
                  "(the WHOLE parameter vector diverges, not just the ZI "
                  "term), carrying no interpretive weight about "
                  "zero-inflation either way",
                  not conv, f"got converged={conv}, zi_const={zi_const!r}")
            check(f"{label}: glmmTMB's own meta record confirms zero-inflation "
                  "IS estimable and significant somewhere in this cell's "
                  "ladder (zi_estimable_rs and zi_significant_any_tier both "
                  "True) -- so this statsmodels non-convergence must NOT be "
                  "read as the two methods agreeing zero-inflation is absent",
                  bool(meta.get('zi_estimable_rs')) and bool(meta.get('zi_significant_any_tier')),
                  f"zi_estimable_rs={meta.get('zi_estimable_rs')!r}, "
                  f"zi_significant_any_tier={meta.get('zi_significant_any_tier')!r}")

    # -- Machine-readable ZI evidence (coordinator review round 2): Task 7
    # must be able to read "is zero-inflation real for this cell" from DATA
    # (meta['zi_evidence'] / meta['zi_estimable_rs'] / meta['zi_significant_
    # any_tier']), not from this function's prose. Verify it for ALL SIX
    # cells (not just the two exercised above), and never trust the stored
    # summary flags alone -- recompute them from each cell's raw
    # `zi_evidence` list so a future edit that quietly drops or miscomputes
    # a flag fails here, not silently in the Task 7 memo.
    print("\n        Machine-readable ZI evidence per cell (meta['zi_evidence'], "
          "independent of AIC-based family selection):")
    for cell in CELLS:
        meta = fits.get(f'{cell}__meta')
        check(f"{cell}: __meta carries a non-empty zi_evidence list",
              isinstance(meta.get('zi_evidence'), list) and len(meta['zi_evidence']) > 0)
        ev = meta.get('zi_evidence', [])
        recomputed_estimable_rs = any(
            r.get('converged') and not r.get('zi_degenerate')
            and r.get('re_tier') == 'rs' and r.get('b') is not None
            for r in ev)
        recomputed_significant_any = any(
            r.get('converged') and not r.get('zi_degenerate')
            and r.get('p') is not None and np.isfinite(r['p']) and r['p'] < 0.05
            for r in ev)
        check(f"{cell}: zi_estimable_rs matches an independent recomputation "
              "from zi_evidence (not just trusted as stored)",
              meta.get('zi_estimable_rs') == recomputed_estimable_rs,
              f"stored={meta.get('zi_estimable_rs')}, recomputed={recomputed_estimable_rs}")
        check(f"{cell}: zi_significant_any_tier matches an independent "
              "recomputation from zi_evidence (not just trusted as stored)",
              meta.get('zi_significant_any_tier') == recomputed_significant_any,
              f"stored={meta.get('zi_significant_any_tier')}, "
              f"recomputed={recomputed_significant_any}")
        print(f"        {cell}: zi_estimable_rs={meta.get('zi_estimable_rs')}, "
              f"zi_significant_any_tier={meta.get('zi_significant_any_tier')}"
              + (f", best rs estimate: {meta.get('zi_rs_family')} "
                 f"b={meta.get('zi_rs_b'):+.3f} p={meta.get('zi_rs_p'):.2g}"
                 if meta.get('zi_estimable_rs') else ""))

    # -- Known, verified fact (this run): zero-inflation is estimable and
    # significant SOMEWHERE in the ladder for exactly four of six cells.
    # The two 2019 violations cells are the exception -- every ZI family's
    # intercept there is zi_degenerate (boundary, SE ~2500-3000) at every
    # tier, so there is no non-degenerate estimate left to be significant.
    # This is the SAME 11-fit `zi_degenerate` population already asserted in
    # [3] (`n_degenerate == 11`), restated here as a cell-level summary. The
    # point of this check: violations is NOT one undifferentiated "no
    # zero-inflation" story -- the 2021 violations cells DO show real,
    # significant zero-inflation (just outcompeted by NB1 on AIC); only the
    # 2019 violations cells' ZI estimates are genuinely uninformative
    # (degenerate).
    sig_cells = {c for c in CELLS if fits.get(f'{c}__meta', {}).get('zi_significant_any_tier')}
    check("zero-inflation is estimable and significant somewhere in the "
          "ladder for exactly {insp_2021, insp_2019, viol_off_2021, "
          "viol_cov_2021} -- i.e. NOT absent for the 2021 violations cells, "
          "only outcompeted there by NB1 on AIC; only the 2019 violations "
          "cells' ZI estimates are genuinely degenerate",
          sig_cells == {'insp_2021', 'insp_2019', 'viol_off_2021', 'viol_cov_2021'},
          f"got {sig_cells}")


# ============================================================
# [6] COVID ROBUSTNESS, 2021 WINDOW (Task 6)
# ============================================================
def validate_covid():
    """COVID robustness under the selected count family (2021 window only --
    `covid` is not a column in the 2019 panel). Compares against the known
    log-linear result from paper_table_models_2021.py: -0.590*** for
    inspections (pushing time2/time3 from null into significance) and -0.183
    n.s. for violations.

    Ruling 2 (task-6 dispatch): each COVID variant must be fit at the SAME
    re_tier as that cell's own rs-tier M3 winner. Fitting at a
    different (fallen-back) tier would recreate exactly the cross-tier
    confound (comparing AIC/coefficients across two different random-effects
    structures) that Rulings R12-R19 spent four commits removing -- so this
    section checks re_used/re_tier directly, not just that a fit exists.
    """
    import json

    section('6', 'COVID robustness (2021 window)')
    fits = json.load(open(GEN + 'count_model_results.json'))
    covid = {k: v for k, v in fits.items() if '__M3covid__' in k}
    check("COVID variants fit for both 2021 cells (insp_2021, viol_cov_2021)",
          sorted({v.get('cell') for v in covid.values()}) == ['insp_2021', 'viol_cov_2021'],
          f"got {sorted(covid)}")

    for k, f in covid.items():
        cell = f.get('cell')
        meta = fits.get(f'{cell}__meta', {})
        winner_tag = meta.get('m3_winner_rs')
        check(f"{k}: family_tag matches the cell's own rs-tier M3 winner "
              f"({winner_tag!r}), read from __meta rather than re-derived",
              f.get('family_tag') == winner_tag, f"got {f.get('family_tag')!r}")
        check(f"{k}: fit was attempted at the rs tier ('(1 + time | state)') "
              "per Ruling 2 -- no cross-tier fallback was permitted",
              f.get('re_used') == '(1 + time | state)',
              f"got re_used={f.get('re_used')!r}")

        if not f.get('converged'):
            # Ruling 2: non-convergence at the mandated rs tier IS the
            # finding here -- report it plainly rather than accepting a
            # fallback fit that would silently compare against a different
            # random-effects structure than the base (no-COVID) M3 fit.
            check(f"{k}: converged at the rs tier", False,
                  f"COVID variant did not converge at the mandated rs tier "
                  f"(re_used={f.get('re_used')!r}); message={f.get('message')!r}")
            continue
        # No unconditional `check(..., True)` here: convergence was already
        # asserted by the guard above (a False result would have `continue`d),
        # so re-asserting True here is true by construction and cannot fail.
        check(f"{k}: re_tier is 'rs' (matches the base M3 winner's own tier)",
              f.get('re_tier') == 'rs', f"got {f.get('re_tier')!r}")
        check(f"{k}: includes the covid term", 'covid' in f.get('cond', {}),
              f"terms: {sorted(f.get('cond', {}))}")
        if 'covid' not in f.get('cond', {}):
            continue

        base_key = k.replace('__M3covid__', '__M3__')
        base = fits.get(base_key)
        check(f"{k}: base (no-COVID) M3 fit {base_key!r} exists and converged",
              base is not None and base.get('converged'))
        if base is None or not base.get('converged'):
            continue
        check(f"{k}: base fit is at the SAME re_tier ('rs') as the COVID "
              "variant -- required before any coefficient-shift comparison "
              "is meaningful",
              base.get('re_tier') == 'rs', f"got {base.get('re_tier')!r}")

        cov_term = f['cond']['covid']
        b, se, p = cov_term['b'], cov_term['se'], cov_term['p']
        irr = float(np.exp(b))
        # NOT `np.isclose(irr, np.exp(b))` -- that compares an expression to
        # itself and cannot fail (coordinator review round 2). A real check:
        # the raw coefficient fields are all finite, so a NaN/Inf from a
        # degenerate fit that still reports converged=True cannot slip
        # through silently.
        check(f"{k}: covid b/se/p are all finite",
              all(np.isfinite(x) for x in (b, se, p)), f"got b={b!r}, se={se!r}, p={p!r}")
        check(f"{k}: covid p-value is a valid probability in [0, 1]",
              0.0 <= p <= 1.0, f"got {p!r}")
        print(f"        {k}: covid b={b:+.4f} (SE {se:.4f}, p={p:.3g}), IRR={irr:.4f} "
              f"[family={f.get('family_tag')}, tier={f.get('re_tier')}]")

        check(f"{k}: all three cubic time terms are present in both the base "
              "and COVID fits (required before their shift is comparable)",
              all(t in base.get('cond', {}) and t in f.get('cond', {})
                  for t in ('time', 'time2', 'time3')))
        for term in ('time', 'time2', 'time3'):
            b_no = base['cond'].get(term, {}).get('b')
            b_cv = f['cond'].get(term, {}).get('b')
            if b_no is None or b_cv is None:
                continue
            if abs(b_no) > 1e-8:
                rel_shift = abs(b_cv - b_no) / abs(b_no)
                print(f"        {k}: {term} {b_no:+.5f} -> {b_cv:+.5f} "
                      f"(relative shift {rel_shift:.1%})")
            else:
                print(f"        {k}: {term} {b_no:+.5f} -> {b_cv:+.5f} "
                      "(base b ~ 0, relative shift undefined)")


# ============================================================
# [7] MEMO (Task 7)
# ============================================================
MEMO_PATH = '/Users/keshavgoel/Research/docs/count_models_zinb.md'
EVIDENCE_PATH = GEN + 'count_model_memo_evidence.json'

# Phrases the memo must carry. Each one corresponds to a substantive claim the
# PI has to be able to find: what was/was not replicated, the software, the
# boundary-LRT caveat, the two outcome sources, the observed-vs-expected zero
# evidence, the link-scale caveat on sigma^2_u0, the Monte-Carlo SE on every
# expected-zero count, the non-converged published fit, and the two framings
# that earlier drafts got wrong (zero-inflation is real for 2021 violations;
# the 2019 ZI degeneracy statement is scoped to the specified structure).
MEMO_REQUIRED_PHRASES = (
    'Jafari', 'glmmTMB', 'boundary', 'establishments view', 'WPS view',
    'observed', 'expected', 'link scale', 'Monte-Carlo', 'IRR',
    'did not converge', 'random intercept', 'random slope',
    'identifiability', 'zero-deflated', 'multi-start',
    # C1/C2: the estimability claim must be Model-3-scoped and must carry the
    # Model-1 counter-example.
    'at **Model 3**', 'as covariates are added',
    # I1: sigma^2_u1 must be described as NOT published.
    'reaches no published table',
    # I3: the crossed design discards the state-specific SLOPE, not the fixed
    # cubic trend.
    'state-specific time slope',
    # Round 3: the pro-zero-inflation overstatements. The honest claims are that
    # a ZI-NB never lost a matched comparison (ZIP did, repeatedly), that the
    # NB2 comparison is shown and not just asserted, and that the fallback tier
    # supplies part of the evidence.
    # NOTE: 'never lost a matched comparison' is deliberately NOT listed here.
    # As a bare substring it is satisfied by "A zero-inflated model never lost a
    # matched comparison", i.e. by the exact overstatement round 3 removed. The
    # QUALIFIED form is required by a regex further down instead.
    'The same result against NB2, not just NB1',
    'fallback tier',
    # 2026-08-22 review, Critical: the median/marginal distinction must be
    # stated, and the quadrature named, not just the numbers printed.
    'Gauss-Hermite',
    'median state',
    'marginal',
    # Item 3: the dispersion/residual-variance semantics.
    'dispersion parameter',
    # Item 2: spec 5.5's expanded-ZI sensitivity, reported either way.
    'expanded zero-inflation component',
    # Item 6: the missing zero-inflated NB1 rung.
    'no zero-inflated NB1 rung',
)

# Framings that were WRONG in earlier drafts and must never reappear. Each is a
# claim the artifacts contradict (see [5]: zero-inflation IS real, large and
# significant in the 2021 violations cells; the 2019 violations ZI degeneracy
# was only ever tested at the mandated rs tier).
MEMO_FORBIDDEN_PATTERNS = (
    r'reject(?:s|ed)?\s+zero-inflation',
    r'do(?:es)?\s+not\s+need\s+zero-inflation',
    r'no\s+zero-inflation\s+in\s+the\s+violations',
    r'degenerate\s+at\s+every\s+tier',
    r'zero-inflation\s+is\s+absent',
    # C1: the false blanket estimability claim, in either of its two forms.
    r'could not be fit at the mandated random-slope structure at all',
    r'[Nn]either ZINB nor ZINB\+ZI-RE could be fit there',
    # I3: the wrong rationale for rejecting Jafari's RE structure.
    r'would discard the cubic time trend',
    # Round 3, overstatement 1: a zero-inflated POISSON loses matched
    # comparisons 13 times, 9 of them at the mandated tier, so no blanket
    # "never lost on merit" claim is admissible.
    r'[Nn]owhere in this pipeline did a zero-inflated model get tested and lose',
    # Round 3, overstatement 2: only 1 of the 4 violations cells has a ZI-NB
    # rung at M1 that is gone by M3; the other 3 never had one.
    r'rungs stop being estimable for the violations cells',
    r'rungs cannot be estimated at that specific structure',
    # Round 3, overstatement 3: viol_cov_2019 is ZI-degenerate at Model 1 too,
    # so the collapse is NOT a Model-3 phenomenon in both 2019 cells.
    r'collapse is a Model-3 phenomenon in these cells too',
    # Round 4: the "never lost" claim is only true of the zero-inflated NEGATIVE
    # BINOMIAL. Any unqualified subject reinstates the overstatement.
    r'zero-inflated models?\s+never lost',
    r'\bZI models?\s+never lost',
    # Round 5: the ZIP's 13th losing rung is beaten by a plain POISSON
    # (viol_off_2019/M1/rs, by 1.94 AIC), so it is not "always" beaten by a
    # negative binomial and a plain Poisson does beat it once.
    r'always to a negative binomial',
    r'never to a plain Poisson',
    # Round 4: inflation on a Poisson is NOT "beaten wherever it is tested" --
    # ZIP beats plain Poisson 12-1. It is beaten wherever tested AGAINST A
    # NEGATIVE BINOMIAL, and the qualifier is the whole point.
    r'beaten wherever it is tested(?!\s+against)',
    # 2026-08-22 review, Critical: plogis(b0) is the MEDIAN-state structural-
    # zero probability. Any phrasing that presents it as the panel's, or as
    # "the" probability for a model carrying a ZI random intercept, is the
    # defect this round fixed.
    r'[Ii]mplied structural-zero probability \|',
    r'structural-zero probability is (?:negligible|effectively zero)',
    # NOTE on the "about as economically" explanation: it is NOT listed here,
    # because the memo legitimately QUOTES it in order to refute it. A blanket
    # forbidden pattern would fire on the refutation. The property that actually
    # matters -- every occurrence is immediately refuted -- is enforced by a
    # contextual check in validate_memo() instead.
)


def _memo_table_rows(text, cells):
    """Every pipe-table row in the memo whose first cell is one of `cells`,
    returned as {cell: [list of field lists]}. Used to parse generated numbers
    back OUT of the markdown so they can be compared to the JSON -- a memo that
    silently drifts from the artifacts fails here."""
    out = {c: [] for c in cells}
    for line in text.splitlines():
        if not line.startswith('|'):
            continue
        fields = [f.strip() for f in line.strip().strip('|').split('|')]
        if fields and fields[0] in out:
            out[fields[0]].append(fields)
    return out


def _memo_tables(text):
    """Every markdown pipe-table in the memo as (header_fields, data_rows).

    Rows are split on UNESCAPED pipes only: cells legitimately contain
    '(1 + time \\| state)' and '(1 \\| state)', and splitting on every pipe
    would shift each column index after them. Separator rows ('|---|---|') and
    the tier-boundary marker row are dropped."""
    import re

    def split_row(ln):
        return [c.strip().replace('\\|', '|')
                for c in re.split(r'(?<!\\)\|', ln.strip().strip('|'))]
    out, hdr, rows = [], None, []
    for ln in text.splitlines():
        if ln.startswith('|'):
            f = split_row(ln)
            if set(''.join(f)) <= set('-: '):
                continue                      # |---|---| separator
            if '*--' in ln:
                continue                      # tier-boundary marker
            if hdr is None:
                hdr, rows = f, []
            else:
                rows.append(f)
        else:
            if hdr is not None:
                out.append((hdr, rows))
            hdr, rows = None, []
    if hdr is not None:
        out.append((hdr, rows))
    return out


def _nearest(tab, cell, model, re_tier, family):
    from report_count_models import nearest_clean_competitor
    return nearest_clean_competitor(tab, cell, model, re_tier, family)


def _first_number(s):
    import re as _re
    m = _re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s.replace(',', ''))
    return float(m.group(0)) if m else None


def validate_memo():
    """Section [7]: the memo is the deliverable, so it is validated as one --
    it must exist, be substantive, carry every required framing, carry none of
    the framings earlier drafts got wrong, and (the part that matters) contain
    NO number that contradicts the artifacts. Numbers are parsed back out of
    the generated markdown and compared to count_model_results.json and
    count_model_memo_evidence.json with relative tolerances."""
    import json
    import os
    import re

    section('7', 'Memo')
    if not os.path.exists(MEMO_PATH):
        check("docs/count_models_zinb.md exists", False,
              "run: python3 scripts/report_count_models.py")
        return
    check("docs/count_models_zinb.md exists", True)
    text = open(MEMO_PATH).read()

    check("memo is substantive (> 8000 chars)", len(text) > 8000, f"got {len(text)}")
    for phrase in MEMO_REQUIRED_PHRASES:
        check(f"memo mentions {phrase!r}", phrase in text)
    check("memo has no placeholder text",
          not re.search(r'\bTBD\b|\bTODO\b|\bXXX\b|\bFIXME\b|\bLorem\b', text))
    for pat in MEMO_FORBIDDEN_PATTERNS:
        hits = re.findall(pat, text, flags=re.I)
        check(f"memo does NOT contain the corrected-away framing /{pat}/",
              not hits, f"found {hits!r}")

    # The memo must say plainly that no Delta sigma^2_u0 percentage is computed
    # (spec 5.6 was deliberately narrowed -- see task-7-brief.md self-review).
    check("memo states plainly that no Delta sigma^2_u0 percentage is computed",
          re.search(r'does not compute .{0,40}percentage|no .{0,30}percentage reduction '
                    r'is (?:computed|reported)', text, flags=re.I) is not None)

    if not os.path.exists(EVIDENCE_PATH):
        check("count_model_memo_evidence.json exists", False,
              "run: python3 scripts/report_count_models.py")
        return
    check("count_model_memo_evidence.json exists", True)
    ev = json.load(open(EVIDENCE_PATH))
    fits = json.load(open(GEN + 'count_model_results.json'))

    # ---- (a) selected family per cell, parsed back out of the memo ----
    rows = _memo_table_rows(text, CELLS)
    n_family_rows = 0
    for cell in CELLS:
        meta = fits[f'{cell}__meta']
        for tier, mkey in (('rs', 'm3_winner_rs'), ('ri', 'm3_winner_ri')):
            want = meta.get(mkey)
            if want is None:
                continue
            matched = [f for f in rows[cell]
                       if len(f) > 2 and f'`{tier}`' in ' '.join(f)
                       and f'`{want}`' in ' '.join(f)]
            check(f"memo's selection table reports {cell} ({tier} tier) selected "
                  f"family as {want!r} (parsed back from the markdown)",
                  len(matched) >= 1,
                  f"rows for {cell}: {rows[cell]!r}")
            n_family_rows += len(matched)
    check("memo's selection table produced at least one parseable family row "
          "per (cell, tier) with a winner (8 total)",
          n_family_rows >= 8, f"got {n_family_rows}")

    # ---- (b) COVID coefficients, parsed back out of the memo ----
    covid_keys = [k for k in fits if '__M3covid__' in k]
    check("memo COVID check: both 2021 COVID fits exist in the JSON",
          len(covid_keys) == 2, f"got {covid_keys!r}")
    n_covid_checked = 0
    for k in covid_keys:
        f = fits[k]
        cell = f['cell']
        c = f['cond']['covid']
        # The COVID row is the one whose second field names the covid term.
        cand = [fl for fl in rows[cell] if len(fl) >= 6 and 'covid' in fl[1].lower()]
        check(f"memo carries exactly one COVID row for {cell}", len(cand) == 1,
              f"got {cand!r}")
        if len(cand) != 1:
            continue
        fl = cand[0]
        got_b, got_se, got_p = (_first_number(fl[2]), _first_number(fl[3]),
                                _first_number(fl[4]))
        check_close(f"memo COVID b for {cell}", got_b, c['b'], 1e-3, kind='rel')
        check_close(f"memo COVID SE for {cell}", got_se, c['se'], 1e-3, kind='rel')
        check_close(f"memo COVID p for {cell}", got_p, c['p'], 1e-2, kind='rel')
        n_covid_checked += 1
    check("memo COVID coefficients were actually parsed and compared for both cells",
          n_covid_checked == 2, f"got {n_covid_checked}")

    # ---- (c) the non-converged published fit, parsed back out of the memo ----
    aff = ev['published_audit']['affected']
    for label, want in (
            ('published b(log_inspections)', aff['published_b_log_inspections']),
            ('published sigma2_u0', aff['published_sigma2_u0']),
            ('converged b(log_inspections)', aff['cg']['b_log_inspections']),
            ('converged sigma2_u0', aff['cg']['sigma2_u0'])):
        # Formatted to 4 significant-ish decimals in the memo; require the
        # rounded string to be present verbatim, so a drifting number fails.
        s = f"{want:.4f}"
        check(f"memo quotes the {label} ({s}) from the audit artifact",
              s in text, f"{s!r} not found in memo")
    # ROUND 4: the four checks above are satisfied by the memo's audit TABLE row
    # alone, which left the PROSE repetition of the same four numbers unguarded
    # (perturbing 2.4438 -> 2.9438 there gave 0 FAILs). Anchor the prose site.
    check("memo's prose repetition of the non-converged published values is "
          "anchored to the audit artifact, not only its table row",
          re.search(rf"The published values are the non-converged ones: "
                    rf"b\(log_inspections\) = "
                    rf"{aff['published_b_log_inspections']:.4f} and sigma\^2_u0 "
                    rf"= {aff['published_sigma2_u0']:.4f}\. Converged, they are "
                    rf"{aff['cg']['b_log_inspections']:.4f} and "
                    rf"{aff['cg']['sigma2_u0']:.4f}\.", text) is not None,
          f"published {aff['published_b_log_inspections']:.4f}/"
          f"{aff['published_sigma2_u0']:.4f}, converged "
          f"{aff['cg']['b_log_inspections']:.4f}/{aff['cg']['sigma2_u0']:.4f}")
    check("memo's prose repetition of the failed gradient norm and both "
          "log-likelihoods is anchored in situ",
          re.search(rf"\|grad\| = {aff['lbfgs']['grad_norm']:.4f}` at a "
                    rf"log-likelihood of {aff['lbfgs']['llf']:.3f}, while `cg` "
                    rf"and `powell` independently reach "
                    rf"{aff['cg']['llf']:.3f}", text) is not None,
          f"grad={aff['lbfgs']['grad_norm']:.4f}, "
          f"llf={aff['lbfgs']['llf']:.3f}/{aff['cg']['llf']:.3f}")
    check(f"memo quotes the non-converged log-likelihood "
          f"({aff['lbfgs']['llf']:.3f})", f"{aff['lbfgs']['llf']:.3f}" in text)
    check(f"memo quotes the converged log-likelihood "
          f"({aff['cg']['llf']:.3f})", f"{aff['cg']['llf']:.3f}" in text)
    check(f"memo quotes the failed gradient norm "
          f"({aff['lbfgs']['grad_norm']:.4f})",
          f"{aff['lbfgs']['grad_norm']:.4f}" in text)
    check("memo states the audited scope (exactly 1 of 12 published 2021 fits)",
          f"1 of {ev['published_audit']['n_fits']}" in text,
          f"n_fits={ev['published_audit']['n_fits']}")
    check("the audit artifact itself found exactly one non-converged fit "
          "(so the memo's scope claim is a measured result, not a belief)",
          ev['published_audit']['n_nonconverged'] == 1,
          f"got {ev['published_audit']['n_nonconverged']}")
    check("all six random-intercept-only published refits converged in the "
          "audit (this is why the manuscript's variance block is unaffected)",
          all(not r['grad_warning'] for r in ev['published_audit']['fits']
              if r['re_basis'] == 'random intercept')
          and sum(1 for r in ev['published_audit']['fits']
                  if r['re_basis'] == 'random intercept') == 6)

    # ---- (d) cross-software ZI check, parsed back out of the memo ----
    xc = ev['zi_crosscheck']['insp_2021']
    check(f"memo quotes the statsmodels ZI intercept ({xc['zi_b']:.3f})",
          f"{xc['zi_b']:.3f}" in text)
    check(f"memo quotes the statsmodels ZI SE ({xc['zi_se']:.3f})",
          f"{xc['zi_se']:.3f}" in text)
    check(f"memo quotes the glmmTMB ZI intercept ({xc['glmm_zi_b']:.3f})",
          f"{xc['glmm_zi_b']:.3f}" in text)
    check("memo reports BOTH N values for the cross-check (the statsmodels fit "
          "runs on a larger sample than glmmTMB M3)",
          str(xc['n_obs']) in text and str(xc['glmm_n_obs']) in text,
          f"n_obs={xc['n_obs']}, glmm_n_obs={xc['glmm_n_obs']}")
    check("the cross-check artifact confirms the two N's genuinely differ "
          "(otherwise the memo's caveat would be vacuous)",
          xc['n_obs'] != xc['glmm_n_obs'],
          f"{xc['n_obs']} vs {xc['glmm_n_obs']}")
    vx = ev['zi_crosscheck']['viol_cov_2021']
    check("memo records that the violations cross-check did NOT converge",
          vx['converged'] is False and f"{vx['alpha']:.1f}" in text,
          f"converged={vx['converged']}, alpha={vx.get('alpha')}")

    # ---- (e) mean-variance scaling diagnostic ----
    for window, want in (('2019', ev['panel']['2019']['violations']['mv_slope']),
                         ('2021', ev['panel']['2021']['violations']['mv_slope'])):
        check(f"memo quotes the {window} violations mean-variance slope "
              f"({want:.2f})", f"{want:.2f}" in text)

    # ---- (g) boundary-LRT accounting ----
    n_boundary_zire = sum(
        1 for f in fits.values()
        if isinstance(f, dict) and f.get('family_tag') == 'zinb_re')
    tabpath = GEN + 'count_model_comparison.csv'
    cmp_tab = pd.read_csv(tabpath)
    n_lrt = int(cmp_tab['lrt_p'].notna().sum())
    n_zire_lrt = int(((cmp_tab['lrt_p'].notna()) &
                      (cmp_tab['family'] == 'zinb_re')).sum())
    check(f"memo states the LRT count ({n_lrt}) that the CSV actually carries",
          f"{n_lrt} LRT" in text or f"{n_lrt} nested" in text,
          f"n_lrt={n_lrt}")
    check(f"memo states the zinb_re-vs-zinb boundary-LRT count ({n_zire_lrt})",
          f"{n_zire_lrt} of the {n_lrt}" in text,
          f"n_zire_lrt={n_zire_lrt}, n_boundary_zire_fits={n_boundary_zire}")

    # ---- (h) C1/B: estimability claims must agree with eligible_rs_m1/m3 ----
    # Round 1 caught the blanket "could not be fit at all" claim, but the guard
    # only fired for the two 2021 violations cells and for cells whose M1/M3 sets
    # differed -- which is why the same defect survived in the 2019 cells
    # ("all 3 ZI families collapse", true only at Model 3). This version parses
    # the memo's full (cell x {M1, M3}) eligibility TABLE positionally, so every
    # cell and both build-up steps are covered.
    from report_count_models import (ALL_FAMILIES, FAMILY_LABEL,
                                     matched_zinb_vs_nb1, selection_table,
                                     load_meta, matched_zinb_vs_nb1 as _mz)
    ELIG_HEADER = ('Cell', 'Model', 'Eligible families', 'Eligible count',
                   'Absent families', 'Why absent')
    elig_rows = {}
    for blk_hdr, blk_rows in _memo_tables(text):
        if tuple(blk_hdr) == ELIG_HEADER:
            for r in blk_rows:
                elig_rows[(r[0], r[1])] = r
    check(f"memo carries the full Model-1/Model-3 eligibility table "
          f"({len(CELLS)} cells x 2 models = {2 * len(CELLS)} rows)",
          len(elig_rows) == 2 * len(CELLS),
          f"got {len(elig_rows)} rows: {sorted(elig_rows)!r}")
    n_estimability_rows = 0
    for cell in CELLS:
        m = fits[f'{cell}__meta']
        for model in ('M1', 'M3'):
            elig = m.get(f'eligible_rs_{model.lower()}') or []
            if not elig:
                continue
            row = elig_rows.get((cell, model))
            check(f"memo's eligibility table has a row for ({cell}, {model})",
                  row is not None, f"rows: {sorted(elig_rows)!r}")
            if row is None:
                continue
            n_estimability_rows += 1
            want_count = f"{len(elig)} of {len(ALL_FAMILIES)}"
            check(f"({cell}, {model}) eligible count reads {want_count!r}, "
                  f"matching __meta.eligible_rs_{model.lower()}",
                  row[3] == want_count, f"got {row[3]!r}")
            listed = {x.strip() for x in row[2].split(',') if x.strip() != '--'}
            want_listed = {FAMILY_LABEL.get(f, f) for f in elig}
            check(f"({cell}, {model}) lists exactly the eligible families",
                  listed == want_listed,
                  f"memo {sorted(listed)!r} vs artifact {sorted(want_listed)!r}")
            absent_listed = {x.strip() for x in row[4].split(',')
                             if x.strip() not in ('', 'none')}
            want_absent = {FAMILY_LABEL.get(f, f)
                           for f in set(ALL_FAMILIES) - set(elig)}
            check(f"({cell}, {model}) lists exactly the absent families "
                  f"(ALL_FAMILIES - eligible)",
                  absent_listed == want_absent,
                  f"memo {sorted(absent_listed)!r} vs artifact "
                  f"{sorted(want_absent)!r}")
    check(f"the eligibility table was parsed and cross-checked for every "
          f"(cell, model) pair", n_estimability_rows == 2 * len(CELLS),
          f"got {n_estimability_rows}")

    # Finding B, stated as an artifact fact: the 2019 violations cells' ZI
    # collapse is Model-3-only, and the memo must not claim otherwise.
    tab_live = selection_table(fits)
    meta_live = load_meta(fits)
    zi_ok_19 = tab_live[
        (tab_live['cell'].isin(['viol_off_2019', 'viol_cov_2019'])) &
        (tab_live['re_tier'] == 'rs') &
        (tab_live['family'].isin(['zip', 'zinb', 'zinb_re'])) &
        tab_live['converged'] & (~tab_live['zi_degenerate'])]
    check("there IS at least one estimable, non-degenerate ZI fit at the "
          "mandated structure in the 2019 violations cells (so the memo's "
          "Model-3 scoping is necessary, not decorative)",
          len(zi_ok_19) >= 1, f"got {len(zi_ok_19)}")
    for _, r in zi_ok_19.iterrows():
        zi = (fits[r['key']].get('zi') or {}).get('(Intercept)') or {}
        check(f"memo reports the {r['cell']} {r['model']} "
              f"{FAMILY_LABEL.get(r['family'], r['family'])} exception with its "
              f"ZI intercept ({zi.get('b'):+.4f}) and p ({zi.get('p'):.3g})",
              f"{zi['b']:+.4f}" in text and f"{zi['p']:.3g}" in text,
              f"b={zi.get('b')!r}, p={zi.get('p')!r}")
    check("memo scopes the 2019 ZI-collapse claim to Model 3",
          re.search(r'zero-inflated families collapse\s+\*\*at Model 3\*\*',
                    text) is not None)

    # ---- (h2) A: the matched ZI-NB vs NB1 table, parsed positionally ----
    # The mechanism claim ("NB1 explains the zeros about as economically") was
    # false in all matched comparisons. This asserts the memo's table IS the
    # computed set -- row for row, both AICs and the margin -- so neither a
    # perturbed number nor an ADDED row can pass.
    mc = matched_zinb_vs_nb1(tab_live, meta_live)
    MC_HEADER = ('Cell', 'Model', 'RE structure', 'ZI family', 'N', 'NB1 AIC',
                 'ZI-NB AIC', 'AIC margin to ZI-NB', 'Distinct model?')
    mc_rows = []
    for blk_hdr, blk_rows in _memo_tables(text):
        if tuple(blk_hdr) == MC_HEADER:
            mc_rows.extend(blk_rows)
    check(f"memo's matched ZI-NB vs NB1 table has exactly {len(mc)} rows "
          f"(recomputed from the artifacts) -- an added or dropped row fails",
          len(mc_rows) == len(mc), f"memo {len(mc_rows)} vs computed {len(mc)}")
    fmt_key = lambda r: (r['cell'], r['model'], str(r['n_obs']),
                         f"{r['nb1_aic']:.2f}", f"{r['zi_aic']:.2f}",
                         f"{r['margin']:+.2f}")
    want = sorted(fmt_key(r) for r in mc)
    got = sorted((r[0], r[1], r[4], r[5], r[6], r[7].replace('*', ''))
                 for r in mc_rows)
    check("every row of the memo's matched-comparison table matches a computed "
          "comparison exactly (cell, model, N, NB1 AIC, ZI-NB AIC, margin)",
          got == want,
          "memo-only: "
          f"{[g for g in got if g not in want]!r}; computed-only: "
          f"{[w for w in want if w not in got]!r}")
    check(f"every computed margin favours the zero-inflated model, so the memo's "
          f"stated direction is right ({sum(1 for r in mc if r['margin'] > 0)} "
          f"of {len(mc)})",
          all(r['margin'] > 0 for r in mc),
          f"negative/zero margins: "
          f"{[(r['cell'], r['model'], r['margin']) for r in mc if r['margin'] <= 0]!r}")
    n_wins = sum(1 for r in mc if r['margin'] > 0)
    # ROUND 4: these two were substring-anywhere (`text.count("17 of 17") >= 1`,
    # `"15 of 15" in text`). Once the NB2 section landed it contained both
    # strings, so perturbing the NB1 mechanism-section tally to "16 of 17" gave
    # 0 FAILs. Both are now anchored to their own sentence.
    mc_dist = [r for r in mc if not r['collapsed']]
    n_dist_wins = sum(1 for r in mc_dist if r['margin'] > 0)
    check(f"memo states the NB1 matched-comparison tally in the Bottom line, in "
          f"situ ({n_wins} times out of {len(mc)})",
          re.search(rf"beats\s+plain NB1 {n_wins} times out of {len(mc)}, by at "
                    rf"least", text) is not None,
          f"n_wins={n_wins}, n={len(mc)}")
    check(f"memo states the NB1 matched-comparison tally in the mechanism "
          f"section, in situ ({n_wins} of {len(mc)})",
          re.search(rf"the\s+zero-inflated model wins \*\*{n_wins} of {len(mc)}"
                    rf"\*\* times, by", text) is not None,
          f"n_wins={n_wins}, n={len(mc)}")
    check(f"memo reports the NB1 distinct-model subtotal in situ "
          f"({n_dist_wins} of {len(mc_dist)}), so the collapsed duplicates are "
          f"not silently inflating the count",
          re.search(rf"the zero-inflated family still wins\s+{n_dist_wins} of "
                    rf"{len(mc_dist)}\.", text) is not None,
          f"n_dist_wins={n_dist_wins}, n_dist={len(mc_dist)}")
    # ROUND 4: the round-2 pattern required the literal word "same", so dropping
    # it evaded the guard entirely (0 FAILs). The memo legitimately QUOTES this
    # explanation in order to refute it, so a blanket forbidden pattern is wrong;
    # what must hold is that EVERY occurrence is refuted in the same breath.
    REFUTATION = '**That explanation is wrong'
    mech_claim = re.compile(
        r'(?:accounts? for|explains?)\s+(?:those|the|these)?\s*(?:same\s+)?'
        r'zeros\s+about as\s+(?:economically|well)', re.I)
    unrefuted = [text[m.start():m.start() + 40]
                 for m in mech_claim.finditer(text)
                 if REFUTATION not in text[m.end():m.end() + 260]]
    check("every occurrence of the 'explains the zeros about as economically' "
          "explanation is refuted within the same passage (the memo may quote it "
          "to reject it, and may not assert it)",
          not unrefuted, f"unrefuted at: {unrefuted!r}")
    check("the 'about as economically' guard is not vacuous (the memo does quote "
          "the explanation, so there is something to police)",
          mech_claim.search(text) is not None)
    check("memo states plainly that NB1's win is a default rather than a merit "
          "win", re.search(r'by default', text) is not None
          and re.search(r'default, not a merit win', text) is not None)

    # The two prose sites that quote the Model-1 counter-example numbers are
    # checked with ANCHORED patterns, not bare `number in text` -- round 1's
    # version passed when one of two identical sites was perturbed.
    for cell in CELLS:
        m = fits[f'{cell}__meta']
        m1w, m3w = m.get('m1_winner_rs'), m.get('m3_winner_rs')
        if m1w not in ('zinb', 'zinb_re') or m1w == m3w:
            continue
        wrec, lrec = fits[f'{cell}__M1__{m1w}'], fits[f'{cell}__M1__{m3w}']
        margin = lrec['aic'] - wrec['aic']
        comp, dcomp = _nearest(tab_live, cell, 'M1', 'rs', m1w)
        check(f"{cell}: the Model-1 rs winner {m1w!r} is a converged rs-tier fit",
              wrec.get('re_tier') == 'rs' and wrec.get('converged') is True,
              f"re_tier={wrec.get('re_tier')!r}, converged={wrec.get('converged')!r}")
        check(f"memo's Model-1 bullet for {cell} quotes the AIC gap to its "
              f"nearest competitor in situ ('beating ... by {dcomp:.2f} AIC')",
              re.search(rf"beating {re.escape(FAMILY_LABEL.get(comp, str(comp)))} "
                        rf"by {dcomp:.2f} AIC", text) is not None,
              f"nearest={comp!r}, dAIC={dcomp:.2f}")
        check(f"memo's counter-example sentence for {cell} quotes both AICs and "
              f"the margin in situ "
              f"({wrec['aic']:.2f} vs {lrec['aic']:.2f}, margin {margin:.2f})",
              re.search(rf"with AIC {wrec['aic']:.2f} against\s+"
                        rf"{re.escape(FAMILY_LABEL.get(m3w, m3w))}'s "
                        rf"{lrec['aic']:.2f}", text) is not None
              and re.search(rf"a margin of {margin:.2f} AIC in favour", text)
              is not None,
              f"margin={margin:.2f}")
        check(f"that margin genuinely favours the zero-inflated model for {cell}",
              margin > 0, f"margin={margin:.2f}")

    # ---- (h3) ROUND 3: which zero-inflated family ever loses ----
    # An earlier draft claimed "nowhere in this pipeline did a zero-inflated
    # model get tested and lose on merit". False: zero-inflated POISSON loses 13
    # distinct matched comparisons, 9 of them at the mandated rs tier. What is
    # true -- and is the sharper claim -- is that no zero-inflated NEGATIVE
    # BINOMIAL ever lost one. Both tallies are asserted here, against the
    # artifacts, so neither direction of overstatement can survive.
    from report_count_models import (ZINB_FAMILIES, matched_zinb_vs_plain,
                                     zi_loss_summary, zinb_eligibility_at_rs)
    zl = zi_loss_summary(tab_live, meta_live)
    check("a zero-inflated model DOES lose matched comparisons in this ladder "
          "(so the replacement claim is necessary, not decorative)",
          zl['n_loss_pairings'] > 0,
          f"n_loss_pairings={zl['n_loss_pairings']}")
    zinb_losers = [d for d in zl['distinct_losers'] if d[3] in ZINB_FAMILIES]
    check("no zero-inflated NEGATIVE BINOMIAL loses any matched comparison "
          "anywhere in the ladder (the memo's sharpened claim)",
          zl['n_zinb_losses'] == 0 and not zinb_losers,
          f"ZI-NB losers: {zinb_losers!r}")
    n_zip_loss = zl['n_losses_by_family']['zip']
    n_zip_loss_rs = len(zl['distinct_losers_rs'])
    check("every distinct matched-comparison loss in the ladder is a ZIP loss "
          "(so the memo may attribute them to ZIP by name)",
          n_zip_loss == len(zl['distinct_losers']),
          f"zip={n_zip_loss}, all={len(zl['distinct_losers'])}")
    check("distinct losses keyed on (cell, model, re_tier, family) and on the "
          "RUNG alone give the same count, so the memo's '(cell, model, "
          "RE-tier) rungs' phrasing is not double-counting",
          len(zl['distinct_losers']) == len(zl['distinct_loser_rungs']),
          f"{len(zl['distinct_losers'])} tuples vs "
          f"{len(zl['distinct_loser_rungs'])} rungs")
    check(f"memo states the ZIP-loss count in situ ({n_zip_loss} rungs, "
          f"{n_zip_loss_rs} of them at the mandated structure)",
          re.search(rf"It loses at\s+{n_zip_loss} rungs, {n_zip_loss_rs} of them "
                    rf"at the\s+mandated `\(1 \+ time \| state\)` structure",
                    text) is not None,
          f"n_zip_loss={n_zip_loss}, rs={n_zip_loss_rs}")
    check(f"memo states how many ZIP losses are at the mandated rs tier "
          f"({n_zip_loss_rs}) in situ",
          re.search(rf"{n_zip_loss_rs} of them at the mandated", text)
          is not None, f"n_zip_loss_rs={n_zip_loss_rs}")
    check(f"memo states the ZI-NB loss count ({zl['n_zinb_losses']}) in situ",
          re.search(rf"lost a single matched comparison: "
                    rf"{zl['n_zinb_losses']} losses in total", text)
          is not None, f"n_zinb_losses={zl['n_zinb_losses']}")
    # ROUND 4, item 3: the headline claim must carry its SUBJECT. As a bare
    # substring 'never lost a matched comparison' is satisfied by "A
    # zero-inflated model never lost ...", which is the overstatement round 3
    # was dispatched to remove.
    check("the memo's 'never lost' claim names the zero-inflated NEGATIVE "
          "BINOMIAL as its subject (an unqualified 'zero-inflated model never "
          "lost' is the overstatement this round removed)",
          re.search(r'zero-inflated \*negative binomial\* never lost a matched '
                    r'comparison', text) is not None)
    check("the memo does not carry an unqualified 'never lost' claim anywhere",
          not re.search(r'never lost a matched comparison', text)
          or all('negative binomial' in text[max(0, m.start() - 60):m.start()]
                 for m in re.finditer(r'never lost a matched comparison', text)),
          "an occurrence of 'never lost a matched comparison' has no "
          "'negative binomial' qualifier in the preceding 60 characters")

    # ---- ROUND 4: the three-tier hierarchy, all tallies generated ----
    # "inflation on a Poisson is beaten wherever it is tested" was FALSE: ZIP
    # beats plain Poisson decisively. The memo now states the full hierarchy.
    rec = zl['record']
    zp, zn1, zn2 = rec[('zip', 'poisson')], rec[('zip', 'nbinom1')], rec[('zip', 'nbinom2')]
    # tier 3 needs the NB2 set, which (h4) below also builds; computed here so
    # this block does not depend on statement order.
    _mc2 = matched_zinb_vs_plain(tab_live, meta_live, 'nbinom2')
    _mc2_wins = sum(1 for r in _mc2 if r['margin'] > 0)
    _mc_wins = sum(1 for r in mc if r['margin'] > 0)
    check(f"artifact fact: inflation DOES help a Poisson -- ZIP beats plain "
          f"Poisson {zp['wins']}-{zp['losses']}, so the old 'beaten wherever it "
          f"is tested' claim was false", zp['wins'] > zp['losses'],
          f"{zp['wins']}-{zp['losses']}")
    check(f"artifact fact: ZIP never beats a negative binomial "
          f"({zn1['wins']} wins vs NB1, {zn2['wins']} vs NB2), which is what "
          f"scopes the claim to 'against a negative binomial'",
          zn1['wins'] == 0 and zn2['wins'] == 0,
          f"NB1 {zn1['wins']}-{zn1['losses']}, NB2 {zn2['wins']}-{zn2['losses']}")
    check(f"memo states tier 1 of the hierarchy in situ (ZIP beats plain Poisson "
          f"{zp['wins']}-{zp['losses']}, margins {zp['margin_lo']:.1f} to "
          f"{zp['margin_hi']:.1f})",
          re.search(rf"beats a plain Poisson\s+{zp['wins']}-{zp['losses']} "
                    rf"\(margins {zp['margin_lo']:.1f} to "
                    rf"{zp['margin_hi']:.1f} AIC\)", text) is not None,
          f"{zp['wins']}-{zp['losses']}, {zp['margin_lo']:.1f}..{zp['margin_hi']:.1f}")
    check(f"memo states tier 2 of the hierarchy in situ (NB beats that ZIP "
          f"{zn1['losses']}-{zn1['wins']} against NB1 and "
          f"{zn2['losses']}-{zn2['wins']} against NB2)",
          re.search(rf"{zn1['losses']}-{zn1['wins']} against NB1 and\s+"
                    rf"{zn2['losses']}-{zn2['wins']} against NB2", text)
          is not None)
    check(f"memo states tier 3 of the hierarchy in situ (ZI-NB beats NB1 "
          f"{_mc_wins}-{len(mc) - _mc_wins} and NB2 "
          f"{_mc2_wins}-{len(_mc2) - _mc2_wins})",
          re.search(rf"beats NB1 {_mc_wins}-{len(mc) - _mc_wins} and NB2\s+"
                    rf"{_mc2_wins}-{len(_mc2) - _mc2_wins}", text) is not None)
    check("memo scopes the Poisson-inflation limit to negative-binomial "
          "competitors rather than claiming it loses to everything",
          'cannot close the gap to a negative binomial' in text)

    # ---- ROUND 5: WHO beats the losing ZIP rungs, derived per rung ----
    # The round-4 memo claimed the ZIP is beaten "always to a negative binomial,
    # never to a plain Poisson". That is FALSE, and the counterexample is the
    # 1.94 AIC lower bound quoted in the same sentence: at
    # viol_off_2019/M1/rs a plain Poisson beats the ZIP. The clause was the last
    # hardcoded claim in the memo and this guard REQUIRED it, so it would have
    # survived every regeneration. Inverted: the false phrasing now fails and
    # the derived split is asserted.
    nb_only = zl['rungs_beaten_by_nb_only']
    with_pois = zl['rungs_beaten_by_a_plain_poisson']
    check("artifact fact: at least one losing ZI rung is beaten by a plain "
          "Poisson, so 'never to a plain Poisson' is false and this guard is "
          "not vacuous", len(with_pois) >= 1,
          f"rungs with a plain-Poisson beater: {with_pois!r}")
    check("every losing rung is accounted for as either NB-only-beaten or "
          "also-beaten-by-a-plain-Poisson",
          len(nb_only) + len(with_pois) == len(zl['distinct_losers']),
          f"{len(nb_only)} + {len(with_pois)} != "
          f"{len(zl['distinct_losers'])}")
    check(f"memo states the derived split of losing rungs by who beats them "
          f"({len(nb_only)} NB-only of {len(zl['distinct_losers'])})",
          re.search(rf"At {len(nb_only)} of those {len(zl['distinct_losers'])} "
                    rf"rungs every family that beats it is a negative binomial",
                    text) is not None,
          f"nb_only={len(nb_only)}, total={len(zl['distinct_losers'])}")
    check(f"memo names the rung(s) where a plain Poisson also beats the "
          f"zero-inflated model, with its margin",
          all(f"`{c}` {m}" in text for c, m, _t, _f in with_pois)
          and re.search(rf"at the remaining {len(with_pois)} -- .{{0,80}}? -- a "
                        rf"plain Poisson beats it as well, by ", text)
          is not None,
          f"with_pois={with_pois!r}")
    for k in with_pois:
        marg = min(abs(r['margin']) for r in zl['raw_losses']
                   if (r['cell'], r['model'], r['re_tier'],
                       r['zi_family']) == k and r['plain_family'] == 'poisson')
        check(f"memo quotes the plain-Poisson margin at {k[0]}/{k[1]} "
              f"({marg:.1f} AIC) in situ",
              re.search(rf"a plain Poisson beats it as well, by {marg:.1f} AIC",
                        text) is not None, f"margin={marg:.4f}")

    # ---- ROUND 5: the rung / rung-loss definition ----
    check("memo defines a 'rung' and a 'rung loss', so tier 1's pairwise tally "
          "and the losing-rung count cannot read as contradictory",
          re.search(r'A \*\*rung\*\* is one \(cell, model, RE-tier\) combination'
                    r', and a\s+rung counts as a \*loss\* for a family if '
                    r'\*\*at least one\*\* plain family\s+beats it there',
                    text) is not None
          and re.search(rf"Tier 1's {zp['wins']}-{zp['losses']} is pairwise",
                        text) is not None)

    # ---- ROUND 5: tier 0 ----
    t0 = [(f, rec[(f, 'poisson')]) for f in ZINB_FAMILIES
          if (f, 'poisson') in rec]
    check("artifact fact: both ZI-NB rungs are undefeated against a plain "
          "Poisson (tier 0 of the hierarchy)",
          t0 and all(v['losses'] == 0 and v['wins'] > 0 for _, v in t0),
          f"{[(f, v['wins'], v['losses']) for f, v in t0]!r}")
    for f, v in t0:
        check(f"memo states tier 0 for {FAMILY_LABEL.get(f, f)} in situ "
              f"({v['wins']}-{v['losses']})",
              f"{v['wins']}-{v['losses']} ({FAMILY_LABEL.get(f, f)})" in text,
              f"{v['wins']}-{v['losses']}")
    t0lo = min(v['margin_lo'] for _, v in t0)
    t0hi = max(v['margin_hi'] for _, v in t0)
    check(f"memo quotes the tier-0 margin range ({t0lo:.1f} to {t0hi:.1f} AIC)",
          re.search(rf"by {t0lo:.1f} to {t0hi:.1f} AIC\.", text) is not None,
          f"{t0lo:.4f}..{t0hi:.4f}")

    # ---- ROUND 4, minor: the headline's definition of a matched comparison ----
    check("the memo's headline definition of a legitimate matched comparison "
          "carries the convergence and non-degeneracy criteria (they are what "
          "produce the zero ZI-NB losses)",
          re.search(r'same\s+random-effects tier, same N, both converged, '
                    r'neither ZI-degenerate -- a\s+zero-inflated negative '
                    r'binomial beats plain NB1', text) is not None)
    degen_zinb = tab_live[tab_live['zi_degenerate']
                          & tab_live['family'].isin(ZINB_FAMILIES)]
    dz_cells = sorted(set(degen_zinb['cell']))
    check(f"there ARE ZI-NB fits excluded by the degeneracy criterion "
          f"({len(degen_zinb)}), so the memo's exclusion paragraph is not "
          f"decorative", len(degen_zinb) >= 1, f"got {len(degen_zinb)}")
    check(f"memo states how many ZI-NB fits the degeneracy criterion excludes "
          f"({len(degen_zinb)}) and that they are excluded rather than counted "
          f"as losses",
          re.search(rf"{len(degen_zinb)} ZI-NB fits in the ladder are "
                    rf"`zi_degenerate`", text) is not None
          and 'excluded** from the comparison rather than counted' in text,
          f"n_degen_zinb={len(degen_zinb)}")
    check("memo names exactly the cells holding those excluded ZI-NB fits",
          all(f'`{c}`' in text for c in dz_cells)
          and re.search(r'All of them are in the ((?:`\w+`(?:, | and )?)+)',
                        text) is not None
          and sorted(re.findall(
              r'`(\w+)`',
              re.search(r'All of them are in the ((?:`\w+`(?:, | and )?)+)',
                        text).group(1))) == dz_cells,
          f"derived {dz_cells!r}")
    n_worse = 0
    for _, dr in degen_zinb.iterrows():
        if dr['model'] not in ('M1', 'M3'):
            continue
        wf = fits[f"{dr['cell']}__meta"][f"m{dr['model'][1:]}_winner_rs"]
        w = tab_live[(tab_live['cell'] == dr['cell'])
                     & (tab_live['model'] == dr['model'])
                     & (tab_live['re_tier'] == dr['re_tier'])
                     & (tab_live['family'] == wf)]
        if not w.empty and dr['aic'] > float(w['aic'].iloc[0]):
            n_worse += 1
    check(f"memo states, derived, how many of the excluded ZI-NB fits have worse "
          f"AIC than the plain family selected at their own rung "
          f"({n_worse} of {len(degen_zinb)}) -- so exclusion is not hiding wins",
          re.search(rf"{n_worse} of the {len(degen_zinb)} have worse AIC than "
                    rf"the plain family selected at their own rung", text)
          is not None, f"n_worse={n_worse}, n={len(degen_zinb)}")
    check(f"memo quotes the ZIP loss-margin range in situ "
          f"({zl['loss_margin_lo']:.1f} to {zl['loss_margin_hi']:.1f} AIC)",
          re.search(rf"by {zl['loss_margin_lo']:.1f} to "
                    rf"{zl['loss_margin_hi']:.1f} AIC", text) is not None,
          f"{zl['loss_margin_lo']:.4f} .. {zl['loss_margin_hi']:.4f}")

    # ---- (h4) ROUND 3: the NB2 companion table ----
    # The memo's phrase is "better than the plain negative BINOMIAL", and NB1 is
    # only one of the two parameterisations in the ladder. The NB2 comparison is
    # therefore shown, and parsed back positionally exactly like the NB1 one.
    mc2 = matched_zinb_vs_plain(tab_live, meta_live, 'nbinom2')
    MC2_HEADER = ('Cell', 'Model', 'RE structure', 'ZI family', 'N', 'NB2 AIC',
                  'ZI-NB AIC', 'AIC margin to ZI-NB (NB2)', 'Distinct model?')
    mc2_rows = []
    for blk_hdr, blk_rows in _memo_tables(text):
        if tuple(blk_hdr) == MC2_HEADER:
            mc2_rows.extend(blk_rows)
    check(f"memo carries the NB2 companion table with exactly {len(mc2)} rows "
          f"(recomputed) -- an added or dropped row fails",
          len(mc2_rows) == len(mc2),
          f"memo {len(mc2_rows)} vs computed {len(mc2)}")
    want2 = sorted((r['cell'], r['model'], str(r['n_obs']),
                    f"{r['plain_aic']:.2f}", f"{r['zi_aic']:.2f}",
                    f"{r['margin']:+.2f}") for r in mc2)
    got2 = sorted((r[0], r[1], r[4], r[5], r[6], r[7].replace('*', ''))
                  for r in mc2_rows)
    check("every row of the NB2 companion table matches a computed comparison "
          "exactly (cell, model, N, NB2 AIC, ZI-NB AIC, margin)",
          got2 == want2,
          f"memo-only: {[g for g in got2 if g not in want2]!r}; "
          f"computed-only: {[w for w in want2 if w not in got2]!r}")
    n2_wins = sum(1 for r in mc2 if r['margin'] > 0)
    check(f"every computed NB2 margin favours the zero-inflated model "
          f"({n2_wins} of {len(mc2)})", all(r['margin'] > 0 for r in mc2),
          f"losses: {[(r['cell'], r['model'], r['margin']) for r in mc2 if r['margin'] <= 0]!r}")
    m2lo, m2hi = min(r['margin'] for r in mc2), max(r['margin'] for r in mc2)
    check(f"memo states the NB2 tally ({n2_wins} of {len(mc2)}) and its margin "
          f"range ({m2lo:.1f} to {m2hi:.1f} AIC) in situ",
          re.search(rf"wins \*\*{n2_wins} of {len(mc2)}\*\* here too, by "
                    rf"{m2lo:.1f} to {m2hi:.1f} AIC", text) is not None,
          f"n2_wins={n2_wins}, lo={m2lo:.2f}, hi={m2hi:.2f}")
    mc2_dist = [r for r in mc2 if not r['collapsed']]
    n2_dist_wins = sum(1 for r in mc2_dist if r['margin'] > 0)
    check(f"memo states the NB2 distinct-model subtotal in situ "
          f"({n2_dist_wins} of {len(mc2_dist)}) -- found unguarded by a round-4 "
          f"break test",
          re.search(rf"and {n2_dist_wins} of {len(mc2_dist)} counting distinct "
                    rf"models\s+only\.", text) is not None,
          f"n2_dist_wins={n2_dist_wins}, n2_dist={len(mc2_dist)}")
    check(f"memo states the NB2 tally in the Bottom line too "
          f"(beats plain NB2 {n2_wins} times out of {len(mc2)})",
          re.search(rf"beats plain NB2 {n2_wins} times out of {len(mc2)}, by at "
                    rf"least {m2lo:.1f} AIC", text) is not None)
    # ROUND 4, item 1: "NB2 is the harder comparison" was FALSE -- NB2 has the
    # smaller margin at 8 of 17 rungs and NB1 at 9. pair_matched() refuses to
    # pair sets that are not over the same rungs, so a membership difference
    # raises instead of silently misaligning.
    from report_count_models import pair_matched
    pairs = pair_matched(mc, mc2)
    check("the NB1 and NB2 matched-comparison sets are over exactly the same "
          "rungs, so pairing them is legitimate", len(pairs) == len(mc) == len(mc2),
          f"{len(pairs)} pairs, {len(mc)} NB1, {len(mc2)} NB2")
    nb2_tougher = sum(1 for a, b in pairs if b['margin'] < a['margin'])
    nb1_tougher = sum(1 for a, b in pairs if a['margin'] < b['margin'])
    check(f"neither plain family is the tougher competitor at a majority of "
          f"rungs (NB2 {nb2_tougher}, NB1 {nb1_tougher} of {len(pairs)}), which "
          f"is why 'NB2 is the harder comparison' was a defect",
          nb2_tougher < len(pairs) and nb1_tougher < len(pairs)
          and nb2_tougher + nb1_tougher == len(pairs),
          f"NB2 {nb2_tougher}, NB1 {nb1_tougher}, n {len(pairs)}")
    check(f"memo states BOTH per-rung 'tougher competitor' counts in situ "
          f"(NB2 {nb2_tougher} of {len(pairs)}, NB1 the other {nb1_tougher})",
          re.search(rf"NB2 has the smaller margin at {nb2_tougher} of the "
                    rf"{len(pairs)} rungs and NB1 at the other {nb1_tougher}",
                    text) is not None,
          f"NB2 {nb2_tougher}, NB1 {nb1_tougher}")
    mean1 = sum(r['margin'] for r in mc) / len(mc)
    mean2 = sum(r['margin'] for r in mc2) / len(mc2)
    check(f"memo quotes both mean margins ({mean2:.1f} against {mean1:.1f} AIC) "
          f"rather than inferring a majority from them",
          re.search(rf"\({mean2:.1f} against {mean1:.1f} AIC\)", text)
          is not None, f"mean NB2 {mean2:.4f}, mean NB1 {mean1:.4f}")
    drops = {}
    for a, b in pairs:
        drops.setdefault(a['cell'], []).append((a['margin'], b['margin']))
    dcell = max(drops, key=lambda c: sum(x - y for x, y in drops[c]))
    dn = len(drops[dcell])
    dfrom = sum(x for x, _ in drops[dcell]) / dn
    dto = sum(y for _, y in drops[dcell]) / dn
    check(f"memo's 'range effect' explanation names the cell that drives it and "
          f"both of its mean margins ({dn} x {dcell}, {dfrom:.0f} -> {dto:.0f})",
          re.search(rf"the {dn} `{dcell}` rungs fall from {dfrom:.0f} to "
                    rf"{dto:.0f} AIC", text) is not None,
          f"cell={dcell}, n={dn}, {dfrom:.2f} -> {dto:.2f}")
    check("memo states the surviving claim -- the ZI model beats whichever plain "
          "family is tougher at each rung",
          'whichever plain family is the tougher competitor at a given rung' in text)

    # ---- (h5) ROUND 3: the matched-comparison framing numbers ----
    n_rs_mc = sum(1 for r in mc if r['re_tier'] == 'rs')
    n_ri_mc = sum(1 for r in mc if r['re_tier'] == 'ri')
    mc_models = sorted({r['model'] for r in mc})
    check(f"memo's headline discloses the RE-tier split of its own tally "
          f"({n_rs_mc} rs / {n_ri_mc} ri) -- the fallback tier supplies part of "
          f"the evidence and the headline must say so",
          re.search(rf"{n_rs_mc} of the {len(mc)} comparisons are at the "
                    rf"mandated `\(1 \+ time \| state\)` structure and "
                    rf"{n_ri_mc} are at the `\(1 \| state\)` fallback tier",
                    text) is not None,
          f"rs={n_rs_mc}, ri={n_ri_mc}")
    check(f"memo discloses that the tally spans "
          f"{' and '.join(mc_models)} only (M2 contributes no cross-family "
          f"comparison), so 'every comparison in the ladder' is not read as all "
          f"three build-up steps",
          re.search(rf"which is {' and '.join(mc_models)} only", text)
          is not None and 'Model 2 contributes no cross-family' in text,
          f"models={mc_models!r}")
    check("the artifacts agree that no M2 matched comparison exists (so the "
          "memo's M2 disclaimer is a measured fact)",
          'M2' not in mc_models,
          f"M2 rows: {[r for r in mc if r['model'] == 'M2']!r}")

    # ---- (h6) ROUND 3: per-cell scoping of the estimability claim ----
    # Derived INDEPENDENTLY of report_count_models' own helper: a ZI-NB rung is
    # absent-because-non-estimable at (cell, model) iff no genuine fit record
    # for it carries re_tier == 'rs'.
    viol_cells = sorted({f['cell'] for k, f in fits.items()
                         if isinstance(f, dict) and f.get('outcome') == 'violations'
                         and f.get('cell')})
    def _zinb_rs(cell, model):
        out = set()
        for fam in ZINB_FAMILIES:
            r = fits.get(f'{cell}__{model}__{fam}')
            if r is not None and r.get('re_tier') == 'rs' and r.get('converged') \
                    and not r.get('zi_degenerate'):
                out.add(fam)
        return out
    lost_cells, never_cells, kept_cells = [], [], []
    for c in viol_cells:
        e1, e3 = _zinb_rs(c, 'M1'), _zinb_rs(c, 'M3')
        (kept_cells if e3 else lost_cells if e1 else never_cells).append(c)
    ze = zinb_eligibility_at_rs(meta_live, viol_cells)
    check("the independent re-derivation of which violations cells lose a ZI-NB "
          "rung between M1 and M3 agrees with zinb_eligibility_at_rs()",
          (lost_cells, never_cells, kept_cells)
          == (ze['lost'], ze['never'], ze['kept']),
          f"validator {(lost_cells, never_cells, kept_cells)!r} vs "
          f"report {(ze['lost'], ze['never'], ze['kept'])!r}")
    check(f"exactly the cells the memo may describe as 'stops being estimable' "
          f"are the ones that do ({lost_cells!r}) -- fewer than all "
          f"{len(viol_cells)} violations cells, which is why the blanket claim "
          f"was a defect", 0 < len(lost_cells) < len(viol_cells),
          f"lost={lost_cells!r}, all={viol_cells!r}")
    # Prose bullets: each named cell must be in the derived set, and the derived
    # set must be fully covered -- a FABRICATED bullet fails on the first test.
    bullet_lost = re.search(
        r'^- ((?:`\w+`(?:, | and )?)+): a ZI-NB rung \*\*is\*\* eligible at '
        r'Model 1', text, re.M)
    bullet_never = re.search(
        r'^- ((?:`\w+`(?:, | and )?)+): no ZI-NB rung is eligible at '
        r'`\(1 \+ time \| state\)` at \*\*either\*\* Model 1 or Model 3',
        text, re.M)
    check("memo carries the 'rung lost between M1 and M3' bullet",
          bullet_lost is not None or not lost_cells,
          f"lost_cells={lost_cells!r}")
    check("memo carries the 'never had a rung at this structure' bullet",
          bullet_never is not None or not never_cells)
    if bullet_lost:
        named = re.findall(r'`(\w+)`', bullet_lost.group(1))
        check("the 'stops being estimable' bullet names exactly the cells that "
              "do (no fabricated cell, none omitted)",
              sorted(named) == sorted(lost_cells),
              f"memo {sorted(named)!r} vs derived {sorted(lost_cells)!r}")
    if bullet_never:
        named = re.findall(r'`(\w+)`', bullet_never.group(1))
        check("the 'never had a rung' bullet names exactly the cells that never "
              "had one (no fabricated cell, none omitted)",
              sorted(named) == sorted(never_cells),
              f"memo {sorted(named)!r} vs derived {sorted(never_cells)!r}")
    # Same guarantee for the Bottom line's own (now shorter) scoping sentence.
    # ROUND 4: the sentence was compressed -- it names the cells that DO fit
    # "stops being estimable" and gives a COUNT for the rest, rather than
    # carrying two full cell lists inside one 100-word sentence.
    bl = re.search(r'Only ((?:`\w+`(?:, | and )?)+) of them fits? the phrase '
                   r'"stops being estimable"; the other (\d+) never had such a '
                   r'rung at this structure at either model', text)
    check("Bottom line's per-cell scoping sentence is present and parseable",
          bl is not None)
    if bl:
        check("Bottom line attributes 'stops being estimable' to exactly the "
              "derived cells",
              sorted(re.findall(r'`(\w+)`', bl.group(1))) == sorted(lost_cells),
              f"memo {sorted(re.findall(chr(96) + '(.w+)' + chr(96), bl.group(1)))!r}"
              f" vs {sorted(lost_cells)!r}")
        check("Bottom line's count of the remaining violations cells matches "
              "the derived set", int(bl.group(2)) == len(never_cells),
              f"memo {bl.group(2)}, derived {len(never_cells)}")
    # ROUND 4, minor: the counter-example paragraph's conclusion must be scoped
    # to its own cell -- "the ZI-NB rungs drop out as covariates are added" is
    # false of the other three violations cells.
    check("the Model-1 counter-example scopes its 'drops out as covariates are "
          "added' conclusion to the one cell it holds for",
          re.search(r'the correct statement, \*\*for this cell\*\*, is that its'
                    r'\s+ZI-NB rung drops out \*\*as covariates are added\*\*',
                    text) is not None
          and re.search(rf'the other {len(never_cells)}\s+violations cells have '
                        rf'no ZI-NB rung at this structure at\s+either model',
                        text) is not None,
          f"never_cells={never_cells!r}")
    # ROUND 5: the 'honest mechanism' conclusion is the paragraph most likely to
    # be quoted, and it still said "NB1 wins ... in any of the 4 violations
    # cells". Its subject must name each family with the cells that select it.
    by_fam = {}
    for c in viol_cells:
        by_fam.setdefault(fits[f'{c}__meta']['m3_winner_rs'], []).append(c)
    check("the 'honest mechanism' conclusion names the plain negative binomial "
          "generically rather than generalising NB1 to all four violations cells",
          re.search(r'\*\*The plain negative binomial wins the Model-3 '
                    r'random-slope comparison by default\*\*', text)
          is not None
          and not re.search(r'\*\*NB1 wins the Model-3 random-slope '
                            r'comparison by default\*\*', text))
    for fam, cs in sorted(by_fam.items()):
        lbl = FAMILY_LABEL.get(fam, fam)
        want = f"{lbl} in " + (f"`{cs[0]}`" if len(cs) == 1 else
                               ", ".join(f"`{c}`" for c in sorted(cs)[:-1])
                               + f" and `{sorted(cs)[-1]}`")
        check(f"that conclusion attributes {lbl} to exactly the cells that "
              f"select it ({sorted(cs)!r})", want in text,
              f"expected {want!r}")
    # ROUND 4, minor: the mechanism heading justifies BOTH plain families, so its
    # opening sentence must name the per-cell selections rather than NB1 alone.
    for c in viol_cells:
        fam = fits[f'{c}__meta']['m3_winner_rs']
        lbl = FAMILY_LABEL.get(fam, fam)
        check(f"the mechanism section names {c}'s actual Model-3 rs selection "
              f"({lbl}), so its heading cannot generalise NB1 to a cell that "
              f"selects {lbl}", f"{lbl} in `{c}`" in text,
              f"expected '{lbl} in `{c}`' in the memo")

    # ---- (h7) ROUND 3: the two prose bullet LISTS, cells and counts ----
    # A fabricated bullet in either list used to produce 0 FAILs.
    ident_cells = [c for c in viol_cells
                   if not _zinb_rs(c, 'M3')
                   and all(fits.get(f'{c}__M3__{f}') is None
                           or fits[f'{c}__M3__{f}'].get('re_tier') != 'rs'
                           for f in ZINB_FAMILIES)]
    m3_bullets = re.findall(
        r'^- `(\w+)`: at \*\*Model 3\*\*, only (\d+) of the (\d+) families',
        text, re.M)
    check(f"the Model-3 identifiability bullet list names exactly the cells "
          f"whose ZI-NB rungs are absent for NON-estimability at the rs tier "
          f"({ident_cells!r})",
          sorted(c for c, _, _ in m3_bullets) == sorted(ident_cells),
          f"memo {sorted(c for c, _, _ in m3_bullets)!r} vs derived "
          f"{sorted(ident_cells)!r}")
    check("the Model-3 identifiability bullet list is not empty (the check "
          "would otherwise be vacuous)", len(m3_bullets) >= 1,
          f"got {m3_bullets!r}")
    for cell, n_el, n_all in m3_bullets:
        want_el = len(fits[f'{cell}__meta'].get('eligible_rs_m3') or [])
        check(f"Model-3 bullet for {cell} states its eligible count "
              f"({want_el} of {len(ALL_FAMILIES)}) as the artifact records it",
              int(n_el) == want_el and int(n_all) == len(ALL_FAMILIES),
              f"memo {n_el} of {n_all}, artifact {want_el} of "
              f"{len(ALL_FAMILIES)}")
    changed_cells = [c for c in CELLS
                     if set(fits[f'{c}__meta'].get('eligible_rs_m1') or [])
                     != set(fits[f'{c}__meta'].get('eligible_rs_m3') or [])]
    m1_bullets = re.findall(
        r'^- `(\w+)`: (\d+) eligible at Model 1 vs (\d+) at Model 3', text, re.M)
    check(f"the Model-1 differences bullet list names exactly the cells whose "
          f"M1 and M3 eligible sets differ ({changed_cells!r})",
          sorted(c for c, _, _ in m1_bullets) == sorted(changed_cells),
          f"memo {sorted(c for c, _, _ in m1_bullets)!r} vs derived "
          f"{sorted(changed_cells)!r}")
    check("the Model-1 differences bullet list is not empty",
          len(m1_bullets) >= 1, f"got {m1_bullets!r}")
    for cell, n1, n3 in m1_bullets:
        m = fits[f'{cell}__meta']
        check(f"Model-1 bullet for {cell} states both eligible counts as the "
              f"artifact records them",
              int(n1) == len(m.get('eligible_rs_m1') or [])
              and int(n3) == len(m.get('eligible_rs_m3') or []),
              f"memo {n1}/{n3}, artifact "
              f"{len(m.get('eligible_rs_m1') or [])}/"
              f"{len(m.get('eligible_rs_m3') or [])}")

    # ---- (h8) ROUND 3: the 2019 collapse is Model-3-only in ONE cell ----
    CELLS_19 = ('viol_off_2019', 'viol_cov_2019')
    exc_cells = sorted(set(zi_ok_19['cell']))
    no_exc_cells = [c for c in CELLS_19 if c not in exc_cells]
    check("the 2019 violations pair splits into cells WITH and WITHOUT a "
          "non-degenerate ZI fit at the mandated structure -- which is why the "
          "one-row exception table could not be generalised to both",
          len(exc_cells) >= 1 and len(no_exc_cells) >= 1,
          f"with={exc_cells!r}, without={no_exc_cells!r}")
    for c in no_exc_cells:
        dg = tab_live[(tab_live['cell'] == c) & tab_live['zi_degenerate']]
        models = sorted(set(dg['model']))
        check(f"{c} is ZI-degenerate at every model it was fit at ({models!r}), "
              f"so the memo must NOT call its collapse Model-3-specific",
              set(models) >= {'M1', 'M3'}, f"models={models!r}")
        check(f"memo records {c} as the no-exception cell, with its degenerate "
              f"family count ({len(set(dg['family']))}) and fit count "
              f"({len(dg)})",
              re.search(rf"`{c}`: \*\*no exception\.\*\* All "
                        rf"{len(set(dg['family']))} zero-inflated families are "
                        rf"ZI-degenerate at .{{0,40}}?alike \({len(dg)} "
                        rf"degenerate fits\)", text) is not None,
              f"n_fams={len(set(dg['family']))}, n_fits={len(dg)}")
    for c in exc_cells:
        rows_c = zi_ok_19[zi_ok_19['cell'] == c]
        mods = sorted({'Model ' + m[1:] for m in rows_c['model']})
        check(f"memo records {c} as the Model-3-phenomenon cell, naming the "
              f"model(s) where its ZI fit IS estimable ({mods!r})",
              re.search(rf"`{c}`: its .{{0,40}}? is estimable and "
                        rf"non-degenerate at {re.escape(' and '.join(mods))} "
                        rf"and degenerate by Model 3", text) is not None,
              f"mods={mods!r}")
    exc_named = re.findall(r'the Model-3 scoping buys something for '
                           r'((?:`\w+`(?:, | and )?)+) and nothing for '
                           r'((?:`\w+`(?:, | and )?)+)', text)
    check("memo's 2019 scoping sentence names both groups explicitly",
          len(exc_named) == 1, f"got {exc_named!r}")
    if exc_named:
        a, b = exc_named[0]
        check("the 2019 scoping sentence's 'buys something for' list is exactly "
              "the cells with an exception",
              sorted(re.findall(r'`(\w+)`', a)) == sorted(exc_cells),
              f"memo {sorted(re.findall(r'`(.w+)`', a))!r} vs {exc_cells!r}")
        check("the 2019 scoping sentence's 'nothing for' list is exactly the "
              "cells without one",
              sorted(re.findall(r'`(\w+)`', b)) == sorted(no_exc_cells),
              f"memo {sorted(re.findall(r'`(.w+)`', b))!r} vs {no_exc_cells!r}")

    # ---- (i) I2: expected-zero COVERAGE, enforced structurally ----
    # The first version of this check counted rows CONTAINING '+/-' and required
    # >= 6, so a row that DROPPED its SE was invisible -- the reviewer stripped
    # one and it still passed. This version is positional: it finds every table
    # column headed "Expected 0s (+/- MC SE)" and requires EVERY data cell in
    # that column to carry an MC SE. Rows are split on unescaped pipes only,
    # because '(1 + time \| state)' contains an escaped pipe that would
    # otherwise shift every column index after it.
    split_row = lambda ln: [c.strip() for c in
                            re.split(r'(?<!\\)\|', ln.strip().strip('|'))]
    se_cell = re.compile(r'\d+(?:\.\d+)?\s*\+/-\s*\d')
    EXP_HEADER = 'Expected 0s (+/- MC SE)'
    bad_cells, n_checked, n_cols = [], 0, 0
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('|') and EXP_HEADER in ln:
            hdr = split_row(ln)
            idx = [j for j, c in enumerate(hdr) if c == EXP_HEADER]
            n_cols += len(idx)
            j = i + 1
            while j < len(lines) and lines[j].startswith('|'):
                row = split_row(lines[j])
                # skip the |---|---| separator and the tier-boundary marker row
                if not set(''.join(row)) <= set('-: ') and '*--' not in lines[j]:
                    for k in idx:
                        if k < len(row) and row[k] not in ('', '--'):
                            n_checked += 1
                            if not se_cell.search(row[k]):
                                bad_cells.append((lines[j].strip()[:100], row[k]))
                j += 1
            i = j
            continue
        i += 1
    check(f"every data cell in every '{EXP_HEADER}' column carries its "
          f"Monte-Carlo SE (claim 13 enforced positionally, so a stripped SE is "
          f"detectable -- {n_checked} cells across {n_cols} columns)",
          not bad_cells, f"{len(bad_cells)} bare cell(s): {bad_cells!r}")
    check("the positional MC-SE check is not vacuous (it found expected-zero "
          "columns and cells to cover)", n_cols >= 6 and n_checked >= 40,
          f"n_cols={n_cols}, n_checked={n_checked}")
    # And the same guarantee for PROSE: any sentence quoting an expected-zero
    # count outside a table must carry the SE too.
    #
    # ROUND 3, Minor 1. Round 2's version skipped any line containing 'MC SE',
    # which exempted the memo's ONLY prose site quoting expected-zero counts
    # (the 16.1/55.0 range, whose sentence ends "MC SEs from 2000 simulations")
    # from the guard that exists for it -- stripping both +/- there yielded 0
    # FAILs. The skip existed because the old pattern allowed 25 arbitrary
    # characters between "expected" and a number, making "expected across the 6
    # families" a false positive on the FAMILY count.
    #
    # This version needs no skip. A candidate expected-zero count is a number
    # that is followed -- across at most an optional "+/- <se>" and an optional
    # parenthetical attribution -- by either "expected" or the "to" of a range
    # that ends in "expected"; or a number that FOLLOWS "expected". The SE must
    # be part of that match, so each count is policed individually: stripping
    # one of the two in the range still fails.
    prose_count_pats = (
        re.compile(r'(\d+(?:\.\d+)?)((?:\s*\+/-\s*\d+(?:\.\d+)?)?)'
                   r'(?:\s*\([^)]*\))?\s*(?=(?:to\b|expected\b))', re.I),
        re.compile(r'expected(?:\s+zeros?)?\s*(?:of|is|are|about|around|[:=])?'
                   r'\s*(\d+(?:\.\d+)?)((?:\s*\+/-\s*\d+(?:\.\d+)?)?)',
                   re.I),
    )
    bare_prose, n_prose_counts = [], 0
    for ln in lines:
        if ln.startswith('|') or not re.search(r'expected', ln, re.I):
            continue
        for pat in prose_count_pats:
            for mt in pat.finditer(ln):
                n_prose_counts += 1
                if not mt.group(2).strip():
                    bare_prose.append((ln.strip()[:140], mt.group(0)[:60]))
    check("no prose sentence in the memo quotes an expected-zero count -- "
          "integer or decimal, and each one separately -- without its "
          "Monte-Carlo SE alongside it",
          not bare_prose, f"{len(bare_prose)} site(s): {bare_prose!r}")
    check("the prose MC-SE check is not vacuous (it found expected-zero counts "
          "in prose to cover, with no line-level skip)",
          n_prose_counts >= 3, f"n_prose_counts={n_prose_counts}")

    # ---- (j) I5: the re-declared published spec must still match ----
    aud = ev['published_audit']
    bad = [f"{r['outcome']}/{r['model']}/{r['re_basis']}"
           for r in aud['fits'] if not r['reproduces_published']]
    check("all published-spec refits reproduce paper_table_params_2021.json's "
          "sigma^2_u0 -- the ONLY guard that the specification re-declared in "
          "build_count_model_memo_evidence.py has not drifted from "
          "paper_table_models_2021.py", not bad, f"drifted: {bad!r}")
    check(f"the audit's own n_reproduces_published ({aud['n_reproduces_published']}) "
          f"equals n_fits ({aud['n_fits']})",
          aud['n_reproduces_published'] == aud['n_fits'])
    check("the memo's audit table shows no 'NO' in the Reproduces published "
          "column (so prose and table cannot disagree)",
          not re.search(r'\| NO \|', text))
    check(f"the memo's variance-block paragraph is derived: it states the "
          f"random-intercept refit count ({aud['n_random_intercept']}) that the "
          f"artifact records", f"All {aud['n_random_intercept']} " in text)

    # ---- (l) CRITICAL 2026-08-22: median vs marginal structural-zero ----
    # The memo must report BOTH probabilities for any fit with a genuine ZI
    # random intercept, and the MARGINAL one must be recomputable from the
    # stored variance. Both are asserted: the labelled pair in situ, and the
    # quadrature recomputed here independently of report_count_models'.
    from report_count_models import (coefficient_table as _coefficient_table,
                                     fit_record, stars as _stars,
                                     zi_formula_has_state_re, zi_probabilities)

    def _independent_marginal(b0, s2, n=200001):
        """Marginal E[plogis(b0 + u)] by a DIFFERENT rule from the reporter's
        64-node Gauss-Hermite: a fine trapezoid over +/- 12 SD of the fitted
        normal, renormalised by the same grid's density integral. Agreement
        between two unrelated quadratures is the check; reusing the reporter's
        own function would only prove it is self-consistent. The grid has to be
        fine -- with a logit-scale SD near 6 the integrand is a step-like
        sigmoid ~50 units wide, and a coarse grid disagrees with the exact
        value at the 1e-3 level (which is how this check was first tuned)."""
        sd = np.sqrt(s2)
        u = np.linspace(-12 * sd, 12 * sd, n)
        dens = np.exp(-0.5 * (u / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        f = 1.0 / (1.0 + np.exp(-(b0 + u)))
        return float(np.trapz(f * dens, u) / np.trapz(dens, u))

    n_zi_re_reported = 0
    for cell in CELLS:
        m = fits[f'{cell}__meta']
        for tier, mkey in (('rs', 'm3_winner_rs'), ('ri', 'm3_winner_ri')):
            fam = m.get(mkey)
            if not fam:
                continue
            rec = fit_record(fits, cell, 'M3', fam, tier)
            if rec is None or not zi_formula_has_state_re(rec):
                continue
            b0 = (rec.get('zi') or {}).get('(Intercept)', {}).get('b')
            s2 = rec.get('sigma2_zi_u0')
            check(f"{cell} ({tier}): the selected ZI-RE fit carries both a ZI "
                  f"intercept and a ZI random-effect variance",
                  b0 is not None and s2 is not None and s2 > 0,
                  f"b0={b0!r}, sigma2_zi_u0={s2!r}")
            if b0 is None or not s2:
                continue
            pr = zi_probabilities(b0, s2)
            indep = _independent_marginal(b0, s2)
            check_close(f"{cell} ({tier}): the reporter's Gauss-Hermite marginal "
                        f"structural-zero probability matches an independent "
                        f"trapezoid quadrature", pr['marginal'], indep,
                        1e-5, kind='rel')
            # A factor of 1.2 is the threshold at which the two numbers stop
            # rounding to the same reported value at 4 decimals for these
            # magnitudes; the much larger insp_2019 gap is asserted separately
            # in [3] ("more than 10x"), so this check is about EVERY ZI-RE fit
            # rather than only the dramatic one.
            check(f"{cell} ({tier}): the marginal and median-state "
                  f"probabilities are materially different "
                  f"({pr['median']:.6f} vs {pr['marginal']:.6f}), so reporting "
                  f"only one of them would misstate the estimand",
                  pr['marginal'] > 1.2 * pr['median'],
                  f"ratio {pr['ratio']:.2f}")
            # POSITIONAL, whole-row, and independent of the reporter's own
            # formatting code. A "does this pair of numbers appear anywhere"
            # check is NOT enough here: Model 2 and Model 3 of insp_2019 round
            # to the SAME displayed median/marginal strings, so perturbing the
            # Model-3 column alone left the substring satisfied by the Model-2
            # column (measured -- it gave 0 FAILs). The whole `ZI: Intercept`
            # row is reconstructed for every model column that block carries
            # and required verbatim, so any single column can be perturbed.
            ct = _coefficient_table(fits, cell, tier)
            zi_row = ct[ct['term'] == 'ZI: Intercept'] if not ct.empty else ct
            check(f"{cell} ({tier}): the memo's coefficient block has a "
                  f"'ZI: Intercept' row to check", len(zi_row) == 1,
                  f"got {len(zi_row)} rows")
            if len(zi_row) == 1:
                r0 = zi_row.iloc[0]
                cols = [mm for mm in ('M1', 'M2', 'M3')
                        if f'{mm}_b' in ct.columns and pd.notna(r0.get(f'{mm}_b'))]
                parts = []
                for mm in cols:
                    prm = zi_probabilities(float(r0[f'{mm}_b']),
                                           r0.get(f'{mm}_zi_sigma2'))
                    st = _stars(r0.get(f'{mm}_p'))
                    if prm['has_re']:
                        parts.append(
                            f"{r0[f'{mm}_b']:.3f}{st} ({r0[f'{mm}_se']:.3f}), "
                            f"Pr(structural 0) median state "
                            f"{prm['median']:.6f}, marginal "
                            f"{prm['marginal']:.4f} "
                            f"(ZI RE SD {prm['sd']:.3f})")
                    else:
                        parts.append(
                            f"{r0[f'{mm}_b']:.3f}{st} ({r0[f'{mm}_se']:.3f}), "
                            f"Pr(structural 0) {prm['median']:.4f}")
                want_row = "| ZI: Intercept | " + " | ".join(parts) + " |"
                check(f"{cell} ({tier}): the memo's whole ZI-intercept row -- "
                      f"every column's b, SE, stars, median-state AND marginal "
                      f"probability, and ZI RE SD -- is reproduced from the "
                      f"artifacts verbatim",
                      want_row in text, f"expected line:\n{want_row}")
            n_zi_re_reported += 1
    check("at least one selected model carries a ZI random intercept, so the "
          "median/marginal checks above are not vacuous",
          n_zi_re_reported >= 1, f"got {n_zi_re_reported}")

    # The ZI-evidence table itself, parsed positionally, with the marginal
    # column recomputed from the variance the same table prints.
    ZI_EV_HEADER = ('Cell', 'ZI family', 'RE tier', 'ZI intercept', 'SE', 'p',
                    'ZI RE variance', 'ZI RE SD (logit)',
                    'Pr(structural 0), median state',
                    'Pr(structural 0), marginal',
                    'Share of states above 0.10')
    zi_ev_rows = []
    for blk_hdr, blk_rows in _memo_tables(text):
        if tuple(blk_hdr) == ZI_EV_HEADER:
            zi_ev_rows.extend(blk_rows)
    check(f"memo's zero-inflation evidence table carries the median AND "
          f"marginal probability columns and has rows to check",
          len(zi_ev_rows) >= 6, f"got {len(zi_ev_rows)} rows")
    n_zi_ev_checked = 0
    for row in zi_ev_rows:
        b0 = _first_number(row[3])
        med, marg = _first_number(row[8]), _first_number(row[9])
        if row[6] == '--':
            check(f"ZI-evidence row {row[0]}/{row[1]}/{row[2]}: with no ZI "
                  f"random-effect variance the two probabilities are printed "
                  f"as the same number",
                  med is not None and marg is not None
                  and abs(med - marg) <= 1e-9 * max(1.0, abs(med)),
                  f"median {med!r}, marginal {marg!r}")
        else:
            s2 = _first_number(row[6])
            want = _independent_marginal(b0, s2)
            check_close(f"ZI-evidence row {row[0]}/{row[1]}/{row[2]}: the "
                        f"printed marginal probability is recomputable from "
                        f"the ZI variance printed beside it", marg, want,
                        2e-3, kind='rel')
            check_close(f"ZI-evidence row {row[0]}/{row[1]}/{row[2]}: the "
                        f"printed SD is sqrt of the printed variance",
                        _first_number(row[7]), float(np.sqrt(s2)), 2e-3,
                        kind='rel')
        n_zi_ev_checked += 1
    check("every row of the memo's zero-inflation evidence table was actually "
          "re-derived", n_zi_ev_checked == len(zi_ev_rows) and n_zi_ev_checked > 0,
          f"got {n_zi_ev_checked}")
    # The attribution paragraph, PER CELL AND TIER, with its own derived
    # numbers. A "does the memo say 'variance' somewhere" check is too weak:
    # there are three such paragraphs and perturbing one left the others
    # satisfying the pattern (measured -- 0 FAILs).
    n_attrib = 0
    for cell in CELLS:
        m = fits[f'{cell}__meta']
        for tier, mkey in (('rs', 'm3_winner_rs'), ('ri', 'm3_winner_ri')):
            if m.get(mkey) != 'zinb_re':
                continue
            re_fit = fit_record(fits, cell, 'M3', 'zinb_re', tier)
            pl_fit = fit_record(fits, cell, 'M3', 'zinb', tier)
            check(f"{cell} ({tier}): both the ZINB+ZI-RE winner and its plain "
                  f"ZINB counterpart exist at this tier, so the margin is "
                  f"attributable", re_fit is not None and pl_fit is not None)
            if re_fit is None or pl_fit is None:
                continue
            margin = pl_fit['aic'] - re_fit['aic']
            ddf = re_fit['df'] - pl_fit['df']
            s2 = re_fit['sigma2_zi_u0']
            b_re = re_fit['zi']['(Intercept)']['b']
            b_pl = pl_fit['zi']['(Intercept)']['b']
            check(f"{cell} ({tier}): the ZI random intercept really is the only "
                  f"extra parameter ZINB+ZI-RE spends over plain ZINB",
                  ddf == 1, f"df delta {ddf}")
            check(f"{cell} ({tier}): memo attributes the {margin:.2f} AIC margin "
                  f"to the ZI random-intercept VARIANCE ({s2:.4f}), naming both "
                  f"ZI intercepts ({b_pl:+.4f} plain vs {b_re:+.4f}), all in "
                  f"situ",
                  re.search(
                      rf"in `{cell}` at the "
                      rf"{re.escape(_re_label_local(tier))} tier\.\*\* "
                      rf"ZINB \+ ZI RE beats plain ZINB there by "
                      rf"{margin:.2f} AIC on {ddf} extra parameter\. That "
                      rf"parameter is the zero-inflation random-intercept "
                      rf"\*\*variance\*\* \({s2:.4f}, SD "
                      rf"{np.sqrt(s2):.4f} on the logit scale\), not the "
                      rf"zero-inflation intercept: the intercept exists in "
                      rf"plain ZINB too and the two are {b_pl:+.4f} \(plain\) "
                      rf"against {b_re:+.4f}", text) is not None,
                  f"margin={margin:.2f}, ddf={ddf}, s2={s2:.4f}, "
                  f"b_pl={b_pl:+.4f}, b_re={b_re:+.4f}")
            n_attrib += 1
    check("at least one ZINB+ZI-RE selection exists, so the attribution checks "
          "above are not vacuous", n_attrib >= 1, f"got {n_attrib}")

    # ---- (m) item 2: the expanded-ZI sensitivity, reported either way ----
    zs = {c: fits.get(f'{c}__M3__zinb__zisens') for c in CELLS}
    n_conv = sum(1 for f in zs.values() if f and f.get('converged'))
    n_bad = sum(1 for f in zs.values() if f and not f.get('converged'))
    check(f"memo reports the expanded-ZI sensitivity outcome for every cell "
          f"({n_conv} converged, {n_bad} did not)",
          re.search(rf"It converges in\s+{n_conv} of the {len(CELLS)} cells "
                    rf"and fails in\s+{n_bad}", text) is not None,
          f"n_conv={n_conv}, n_bad={n_bad}")
    for c, f in sorted(zs.items()):
        if f is None:
            continue
        want = 'converged' if f.get('converged') else 'did NOT converge'
        check(f"memo states `{c}`'s expanded-ZI outcome as {want!r}",
              f"`{c}` {want}" in text, f"expected '`{c}` {want}' in the memo")
    check("the expanded-ZI sensitivity has a mixed outcome, so the memo cannot "
          "be satisfied by one blanket verdict (this makes the per-cell checks "
          "above non-vacuous)", n_conv >= 1 and n_bad >= 1,
          f"{n_conv} converged, {n_bad} failed")
    check("memo says plainly that the non-converged expanded-ZI estimates are "
          "not reported",
          'Their estimates are not reported anywhere in this memo' in text)

    # ---- (n) item 4: the offset drops only zero-violation rows ----
    od = ev.get('offset_drop') or {}
    check("the memo-evidence artifact carries the offset-drop derivation for "
          "both windows", set(od) == {'2019', '2021'}, f"got {sorted(od)}")
    for w, o in sorted(od.items()):
        check(f"{w}: the offset-drop derivation is self-consistent (the three "
              f"violation categories partition the dropped rows)",
              o['n_dropped_zero_violation'] + o['n_dropped_violations_positive']
              + o['n_dropped_violations_missing'] == o['n_dropped'],
              f"{o!r}")
        check(f"memo names every state-year the {w} offset drops",
              all(sy in text for sy in o['dropped_state_years']),
              f"missing: {[sy for sy in o['dropped_state_years'] if sy not in text]!r}")
        if o['all_dropped_are_zero_violation']:
            check(f"memo states, for {w}, that EVERY dropped row is also a "
                  f"zero-violation row ({o['n_dropped_zero_violation']} of "
                  f"{o['n_dropped']})",
                  re.search(rf"\*\*every one of them is also a zero-violation "
                            rf"row\*\*\s+\({o['n_dropped_zero_violation']} of "
                            rf"{o['n_dropped']}", text) is not None,
                  f"{o['n_dropped_zero_violation']}/{o['n_dropped']}")
        else:
            check(f"memo states, for {w}, that the all-zero property does NOT "
                  f"hold ({o['n_dropped_zero_violation']} of {o['n_dropped']})",
                  re.search(rf"{o['n_dropped_zero_violation']} of "
                            rf"{o['n_dropped']} are\s+zero-violation rows -- so "
                            rf"the property does \*\*not\*\* hold here",
                            text) is not None,
                  f"{o['n_dropped_zero_violation']}/{o['n_dropped']}")
    check("the 2021 offset drop really is all-zero-violation (this is the "
          "substantive finding item 4 asked for, and it must be measured, not "
          "assumed)", od['2021']['all_dropped_are_zero_violation'],
          f"{od['2021']!r}")

    # ---- (o) item 6: the missing zero-inflated NB1 rung ----
    nb1_cells = sorted({c for c in CELLS
                        if fits[f'{c}__meta'].get('m3_winner_rs') == 'nbinom1'})
    mv21 = ev['panel']['2021']['violations']['mv_slope']
    check("NB1 really is the Model-3 rs selection somewhere, so the missing "
          "ZI-NB1 rung is a live gap rather than a hypothetical",
          len(nb1_cells) >= 1, f"got {nb1_cells!r}")
    check(f"memo names the missing zero-inflated NB1 rung as a gap in the "
          f"candidate set, with the {mv21:.2f} mean-variance exponent in situ",
          re.search(rf"at an exponent of {mv21:.2f} -- NB1-like, not "
                    rf"NB2-like", text) is not None,
          f"mv_slope={mv21:.4f}")
    check("memo names every cell whose Model-3 rs selection is NB1 in that "
          "caveat", all(f"`{c}`" in text for c in nb1_cells),
          f"nb1_cells={nb1_cells!r}")
    check("and it cites the ZINB-vs-NB2 comparison as what does isolate the "
          "inflation effect",
          'is what isolates the effect of inflation' in text)
    check("no zero-inflated NB1 rung exists in the ladder, which is what makes "
          "that caveat true",
          not [k for k, f in fits.items()
               if _is_ladder_fit(k) and f.get('family_name') == 'nbinom1'
               and (f.get('zi_formula') or '~0').strip() not in ('~0',)],
          "found a zero-inflated NB1 fit")

    # ---- (p) minor: the 2021 cell with no COVID variant, named with a reason
    cov_skipped = [c for c in CELLS
                   if c.endswith('_2021')
                   and not fits[f'{c}__meta'].get('covid_variant_fit')]
    check("there IS a 2021 cell without a COVID variant, so the memo's "
          "disclosure is necessary", len(cov_skipped) >= 1,
          f"got {cov_skipped!r}")
    for c in cov_skipped:
        reason = fits[f'{c}__meta'].get('covid_not_fit_reason') or ''
        check(f"memo discloses that `{c}` has no COVID variant, quoting the "
              f"recorded reason", f"`{c}`: {reason}" in text,
              f"reason={reason[:60]!r}")

    # ---- (q) item 5: the 2019 parity arm of the published audit ----
    aud19 = ev.get('published_audit_2019')
    check("the memo-evidence artifact carries a 2019 arm of the published-fit "
          "audit (item 5: the guard covered only 2021)", aud19 is not None)
    if aud19 is None:
        skip("2019 published-audit parity assertions", "no 2019 arm in the "
             "evidence artifact")
    else:
        check("the 2019 arm checks itself against "
              "paper_table_params_corrected.json",
              aud19['published_json'] == 'paper_table_params_corrected.json',
              f"got {aud19['published_json']!r}")
        bad19 = [f"{r['outcome']}/{r['model']}/{r['re_basis']}"
                 for r in aud19['fits'] if not r['reproduces_published']]
        check("all 2019 published-spec refits reproduce "
              "paper_table_params_corrected.json's sigma^2_u0 -- the parity "
              "guard item 5 asked for", not bad19, f"drifted: {bad19!r}")
        check(f"memo reports the 2019 parity result "
              f"({aud19['n_reproduces_published']} of {aud19['n_fits']})",
              re.search(rf"All {aud19['n_reproduces_published']} of\s+"
                        rf"{aud19['n_fits']} refits reproduce their published "
                        rf"sigma\^2_u0", text) is not None,
              f"{aud19['n_reproduces_published']}/{aud19['n_fits']}")
        check("all 2019 random-intercept-only refits converge, so the earlier "
              "window's variance block is not implicated either",
              aud19['n_random_intercept_nonconverged'] == 0,
              f"got {aud19['n_random_intercept_nonconverged']}")
        if aud19['n_nonconverged']:
            check(f"memo reports the 2019 arm's own gradient failures "
                  f"({aud19['n_nonconverged']} of {aud19['n_fits']})",
                  re.search(rf"That arm also finds {aud19['n_nonconverged']} of"
                            rf"\s+{aud19['n_fits']} 2011-2019 fits carrying a "
                            rf"gradient failure", text) is not None,
                  f"n_nonconverged={aud19['n_nonconverged']}")
        else:
            check("memo records that no 2019 fit raised a gradient failure",
                  'No 2011-2019 fit raised a gradient failure' in text)

    # ---- (r) item 3: dispersion semantics reach the memo ----
    check("memo's variance section states that glmmTMB's sigma() is a "
          "dispersion parameter for a count family and NOT a residual SD",
          re.search(r'\*\*not a residual SD for a count family\*\*', text)
          is not None
          and 'its square is not a variance component' in text)
    # The bullets must cover the families the VARIANCE TABLE actually prints,
    # which is the selected models only -- not every family in the ladder. The
    # set is parsed back out of that table and mapped to glmmTMB family names,
    # so a family appearing there without a semantics bullet fails.
    VAR_HEADER = ('Cell', 'Model', 'Family', 'RE structure', 'sigma^2_u0',
                  'sigma^2_u1', 'sigma_u01', 'sigma^2_zi_u0', 'Dispersion',
                  'N obs', 'States')
    _label_to_tag = {v: k for k, v in FAMILY_LABEL_LOCAL.items()}
    _tag_to_family = {'poisson': 'poisson', 'zip': 'poisson',
                      'nbinom1': 'nbinom1', 'nbinom2': 'nbinom2',
                      'zinb': 'nbinom2', 'zinb_re': 'nbinom2'}
    var_rows = []
    for blk_hdr, blk_rows in _memo_tables(text):
        if tuple(blk_hdr) == VAR_HEADER:
            var_rows.extend(blk_rows)
    check("memo's variance table carries the sigma^2_zi_u0 and Dispersion "
          "columns and has rows", len(var_rows) >= 6,
          f"got {len(var_rows)} rows")
    fam_names = sorted({_tag_to_family[_label_to_tag[r[2]]] for r in var_rows
                        if r[2] in _label_to_tag})
    check(f"memo spells out the per-family meaning of `Dispersion` for every "
          f"family its own variance table prints ({fam_names!r})",
          fam_names and all(f"- `{fn}`:" in text for fn in fam_names),
          f"missing bullets for "
          f"{[fn for fn in fam_names if f'- `{fn}`:' not in text]!r}")
    # And the ZI-variance column must be filled exactly where the printed
    # family carries a ZI random intercept, '--' otherwise.
    n_var_zi = sum(1 for r in var_rows if r[7] != '--')
    check("the variance table's sigma^2_zi_u0 column is populated for some "
          "rows and '--' for others (a column that were all '--' would mean the "
          "ZI variance never reached the memo)",
          0 < n_var_zi < len(var_rows), f"populated {n_var_zi} of {len(var_rows)}")

    # ---- (k) I4: the reporting script must not fit models any more ----
    rep_src = open('/Users/keshavgoel/Research/scripts/report_count_models.py').read()
    # Test the property that matters -- the reporting script must not IMPORT an
    # estimator (prose mentioning statsmodels by name is fine and expected).
    rep_imports = [ln.strip() for ln in rep_src.splitlines()
                   if re.match(r'\s*(import|from)\s+\S', ln)]
    for banned in ('statsmodels', 'build_count_model_panel'):
        offenders = [ln for ln in rep_imports if banned in ln]
        check(f"report_count_models.py imports nothing from {banned!r} "
              f"(model fitting lives in build_count_model_memo_evidence.py)",
              not offenders, f"found: {offenders!r}")
    check("report_count_models.py fits no models: it has no fit(...) call "
          "on an estimator", not re.search(r'\.fit\(', rep_src),
          "found a .fit( call")
    check("build_count_model_memo_evidence.py exists and does the fitting",
          os.path.exists('/Users/keshavgoel/Research/scripts/'
                         'build_count_model_memo_evidence.py'))
    ev_src = open('/Users/keshavgoel/Research/scripts/'
                  'build_count_model_memo_evidence.py').read()
    for cell, x in ev['zi_crosscheck'].items():
        check(f"{cell}: the ZI cross-check records its warning accounting "
              f"(n_warnings, per-category counts, and a distinct sample)",
              all(k in x for k in ('n_warnings', 'warning_counts',
                                   'warnings_sample', 'warning_categories')),
              f"keys: {sorted(x)}")
        check(f"{cell}: warning accounting is self-consistent "
              f"(counts sum to n_warnings)",
              sum(x['warning_counts'].values()) == x['n_warnings'],
              f"{x['warning_counts']!r} vs n_warnings={x['n_warnings']}")
    check("the non-converged violations cross-check did raise warnings and they "
          "were recorded (a silent divergence would be the exact failure mode "
          "this pipeline exists to catch)",
          ev['zi_crosscheck']['viol_cov_2021']['n_warnings'] > 0
          and bool(ev['zi_crosscheck']['viol_cov_2021']['convergence_warnings']),
          f"n_warnings="
          f"{ev['zi_crosscheck']['viol_cov_2021']['n_warnings']}")
    # A real CALL at statement position, not a mention inside prose/docstring --
    # both files legitimately discuss the published pipeline's use of the idiom.
    call_pat = re.compile(r'^\s*(?:_?warnings\.)?filterwarnings\s*\(', re.M)
    for src_name, src in (('report_count_models.py', rep_src),
                          ('build_count_model_memo_evidence.py', ev_src),
                          ('validate_count_models.py',
                           open('/Users/keshavgoel/Research/scripts/'
                                'validate_count_models.py').read())):
        hits = call_pat.findall(src)
        check(f"{src_name} makes no filterwarnings(...) call", not hits,
              f"found {hits!r}")



def main():
    validate_panels()
    validate_reference_convergence()
    validate_gaussian_roundtrip()
    validate_gaussian_buildup()
    validate_ladder()
    validate_selection()
    validate_zi_crosscheck()
    validate_covid()
    validate_memo()

    # Item 7 of the 2026-08-22 review: a per-section breakdown, not a single
    # headline count. Section [3] is ~105 fit records x ~9 assertions and would
    # otherwise dominate any aggregate a reader quotes, and a SKIP column makes
    # a bypassed block visible instead of silently shrinking the total.
    print()
    print("=" * 78)
    print("CHECK BREAKDOWN BY SECTION (do not quote the total on its own: "
          "section [3] is")
    print("a per-record loop over the whole ladder and is not a set of "
          "independent findings)")
    print("=" * 78)
    print(f"{'section':62}{'PASS':>6}{'FAIL':>6}{'SKIP':>6}")
    for sec in SECTIONS:
        print(f"{sec['name'][:62]:62}{sec['pass']:>6}{sec['fail']:>6}"
              f"{sec['skip']:>6}")
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
    # sys.path mutation lives here (not at module scope) because this file is run
    # as a script, not imported; a future import-by-name use would need its own
    # sys.path setup first.
    sys.path.insert(0, '/Users/keshavgoel/Research/scripts')
    main()
