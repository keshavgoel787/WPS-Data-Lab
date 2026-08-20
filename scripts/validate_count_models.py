"""
Validation gate for the ZINB count-model pipeline.

This is not a unit-test suite -- it is an analysis artifact. Each check either
proves a step of the pipeline reproduces something already known, or proves an
invariant the spec depends on. Run it after any change to the pipeline.

Run: python3 scripts/validate_count_models.py      (exits 1 on any failure)
"""
import sys

import numpy as np
import pandas as pd

GEN = '/Users/keshavgoel/Research/data/generated/'

FAILURES = []


def check(label, condition, detail=''):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f" -- {detail}" if detail else ''))
        FAILURES.append(label)


def check_close(label, got, want, tol, kind='abs'):
    # Unused by this task's checks -- scaffold for later sections (e.g. comparing
    # fitted ZINB parameters/statistics against known tolerances).
    delta = abs(got - want) if kind == 'abs' else abs(got - want) / abs(want)
    check(f"{label} ({kind} delta {delta:.2e} <= {tol:.0e})", delta <= tol,
          f"got {got!r}, want {want!r}")


# ============================================================
# [1] PANEL INVARIANTS
# ============================================================
def validate_panels():
    print("\n[1] Analytic panels")
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
# Reference values captured 2026-08-20 from paper_table_models_2021.py, whose
# output is what the current manuscript tables report. Tolerances are loose
# enough for optimizer differences (lbfgs vs TMB) and tight enough that a real
# specification difference -- wrong sample, wrong RE structure, ML instead of
# REML -- cannot slip through.
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
# task-3-report.md). insp_M1_gaussian is untouched: its own 'lbfgs' fit
# raises no such warning and is the genuine optimum.
GAUSSIAN_REFERENCE = {
    'insp_M1_gaussian': {
        'n_obs': 539, 'n_states': 49,
        'cond': {'time': -0.063369, 'time2': -0.001913, 'time3': 0.000092},
        'se': {'time': 0.021949, 'time2': 0.004265, 'time3': 0.001060},
        'sigma2_u0': 1.118142, 'sigma2_e': 0.339958,
    },
    'viol_M1_gaussian': {
        'n_obs': 533, 'n_states': 49,
        'cond': {'log_inspections': 0.518338, 'time': 0.065731,
                 'time2': -0.017235, 'time3': -0.005587},
        'se': {'log_inspections': 0.055738, 'time': 0.030008,
               'time2': 0.005578, 'time3': 0.001412},
        'sigma2_u0': 1.217723, 'sigma2_e': 0.581066,
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


def _fit_statsmodels_reference(dv, rhs, method):
    """Refit the exact statsmodels.MixedLM spec paper_table_models_2021.py
    uses (same dropna, same '~time' random slope, same REML default), so the
    reference values above can be checked for convergence live rather than
    trusted as hand-copied numbers. Returns (result, warning_messages)."""
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
    return res, [str(w.message) for w in wrec]


def validate_reference_convergence():
    # Guard against the exact failure mode this project just found: a
    # statsmodels ConvergenceWarning silently swallowed by a blanket
    # `warnings.filterwarnings('ignore')`, letting a non-converged fit become
    # "ground truth". This project's pipeline must NEVER add that idiom --
    # this check fits the references live and asserts none is hiding.
    import re as _re

    print("\n[2a] Reference-fit convergence guard (statsmodels, live refit)")
    for cell, (dv, rhs, method) in REFERENCE_SPEC.items():
        res, warns = _fit_statsmodels_reference(dv, rhs, method)
        conv_warns = [w for w in warns if 'onverg' in w or 'grad' in w.lower()]
        check(f"{cell} reference fit (method={method!r}) raised no convergence warning",
              len(conv_warns) == 0, f"warnings: {conv_warns}")
        grads = [float(m.group(1)) for w in conv_warns
                 for m in [_re.search(r'\|grad\|\s*=\s*([0-9.eE+-]+)', w)] if m]
        if grads:
            check(f"{cell} reference fit (method={method!r}) |grad| < 1.0",
                  max(grads) < 1.0, f"|grad| values seen: {grads}")


def validate_gaussian_roundtrip():
    import json
    import subprocess
    import tempfile
    import os

    print("\n[2b] Gaussian round-trip (glmmTMB REML vs statsmodels MixedLM)")
    out = os.path.join(tempfile.mkdtemp(), 'gaussian_roundtrip.json')
    proc = subprocess.run(
        ['Rscript', '/Users/keshavgoel/Research/scripts/count_models_zinb.R',
         GEN + 'count_model_panel_2021.csv', out, '--gaussian-only'],
        capture_output=True, text=True)
    if proc.returncode != 0:
        check("R script ran", False, proc.stderr.strip()[-500:])
        return
    check("R script ran", True)

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
            check_close(f"{cell} b[{term}]", got, want, 1e-3)
            check_close(f"{cell} se[{term}]", fit['cond'][term]['se'], ref['se'][term], 5e-3)
        # Variance components come from a different optimizer path, so relative.
        check_close(f"{cell} sigma2_u0", fit['sigma2_u0'], ref['sigma2_u0'], 2e-2, kind='rel')
        check_close(f"{cell} sigma2_e", fit['sigma2_e'], ref['sigma2_e'], 2e-2, kind='rel')


def main():
    validate_panels()
    validate_reference_convergence()
    validate_gaussian_roundtrip()
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
