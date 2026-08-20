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
    # Used throughout section [2] to compare live statsmodels/glmmTMB fits
    # against the GAUSSIAN_REFERENCE values.
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

    print("\n[2a] Reference-fit convergence guard (statsmodels, live refit)")
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

    print("\n[2b] Gaussian round-trip (glmmTMB REML vs statsmodels MixedLM)")
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
# [3] LADDER STRUCTURE AND CONVERGENCE
# ============================================================
CELLS = ['insp_2021', 'insp_2019', 'viol_off_2021', 'viol_off_2019',
         'viol_cov_2021', 'viol_cov_2019']
FAMILY_TAGS = ['poisson', 'nbinom1', 'nbinom2', 'zip', 'zinb', 'zinb_re']


def validate_ladder():
    import json
    import os

    print("\n[3] Count-model ladder")
    path = GEN + 'count_model_results.json'
    if not os.path.exists(path):
        check("count_model_results.json exists", False,
              "run: Rscript scripts/count_models_zinb.R "
              "data/generated/count_model_panel_2021.csv "
              "data/generated/count_model_results.json")
        return
    fits = json.load(open(path))
    check("count_model_results.json exists", True)

    # Ruling 1: the ladder itself is 6 cells x 13 fits (M1 full ladder + M3 full
    # ladder + M2 winner-only) = 78. The same write_json call also carries the
    # 2 Gaussian round-trip entries (insp_M1_gaussian, viol_M1_gaussian) written
    # earlier in the script, so the file on disk holds 80 entries total. Both
    # counts are checked explicitly against their own population so neither
    # number silently drifts into standing for the other.
    ladder_keys = [k for k in fits if any(k.startswith(c + '__') for c in CELLS)]
    check("80 total entries in count_model_results.json (78 ladder + 2 Gaussian round-trip)",
          len(fits) == 80, f"got {len(fits)}")
    check("78 ladder fits (6 cells x 13: M1 full ladder + M3 full ladder + M2 winner)",
          len(ladder_keys) == 78, f"got {len(ladder_keys)}")

    # Spec 5.3: full ladder at M3 and M1 (stability check), winner only at M2.
    for cell in CELLS:
        for model in ('M1', 'M3'):
            missing = [t for t in FAMILY_TAGS
                       if f'{cell}__{model}__{t}' not in fits]
            check(f"{cell} {model}: all 6 families present", not missing,
                  f"missing {missing}")
        m2 = [k for k in fits if k.startswith(f'{cell}__M2__')]
        check(f"{cell} M2: exactly one family fit", len(m2) == 1, f"got {m2}")

    # At least one NB-family model must converge per cell, or the cell is unusable.
    for cell in CELLS:
        nb = [k for k in fits if k.startswith(cell + '__')
              and k.rsplit('__', 1)[1] in ('nbinom1', 'nbinom2', 'zinb', 'zinb_re')
              and fits[k].get('converged')]
        check(f"{cell}: at least one NB-family model converged", bool(nb))

    # The offset specification must drop the 7 zero-inspection state-years.
    off = fits.get('viol_off_2021__M1__nbinom2')
    cov = fits.get('viol_cov_2021__M1__nbinom2')
    if off and cov:
        check("2021 offset spec drops the 7 zero-inspection state-years",
              cov['n_obs'] - off['n_obs'] == 7,
              f"covariate n={cov['n_obs']}, offset n={off['n_obs']}")

    # Poisson must fit worse than NB2 on these data -- variance is ~158x the mean.
    for cell in CELLS:
        p, nb2 = fits.get(f'{cell}__M3__poisson'), fits.get(f'{cell}__M3__nbinom2')
        if p and nb2 and p.get('converged') and nb2.get('converged'):
            check(f"{cell} M3: NB2 beats Poisson on AIC",
                  nb2['aic'] < p['aic'], f"poisson {p['aic']:.1f}, nb2 {nb2['aic']:.1f}")

    # Ruling 2: the brief's original last check compared exp_zeros to itself,
    # which can never fail. Replaced with a real assertion: for every converged
    # non-Gaussian fit, exp_zeros must actually be present, finite, and >= 0 --
    # i.e. the zero-inflation simulation in fit_spec() ran and produced a
    # sensible count, not NaN/Inf/negative from a degenerate simulate() call.
    for k, f in fits.items():
        if k in GAUSSIAN_REFERENCE:
            continue
        if f.get('converged'):
            ez = f.get('exp_zeros')
            ok = ez is not None and np.isfinite(ez) and ez >= 0
            check(f"{k}: exp_zeros is present, finite, and >= 0", ok, f"got {ez!r}")


def main():
    validate_panels()
    validate_reference_convergence()
    validate_gaussian_roundtrip()
    validate_ladder()
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
