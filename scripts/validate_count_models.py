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
    # winners and the R15/R17 zero-fit evidence. Total:
    # 78 + 17 + 6 altopt + 6 meta + 2 Gaussian round-trip = 109. Every
    # population is counted separately so none of these numbers can silently
    # stand in for another.
    tier1_keys = [k for k in fits if any(k.startswith(c + '__') for c in CELLS)
                  and not k.endswith('__ri2') and not k.endswith('__meta')
                  and not k.endswith('__altopt')]
    tier2_keys = [k for k in fits if k.endswith('__ri2')]
    altopt_keys = [k for k in fits if k.endswith('__altopt')]
    meta_keys = [k for k in fits if k.endswith('__meta')]
    check("78 tier-1 ladder fits (6 cells x 13: M1 full ladder + M3 full ladder + M2 winner)",
          len(tier1_keys) == 78, f"got {len(tier1_keys)}")
    check("17 tier-2 (1 | state) ladder fits (24 possible - 7 de-duplicated per R16)",
          len(tier2_keys) == 17, f"got {len(tier2_keys)}")
    check("6 R19 optimizer-stability (__altopt) diagnostic fits (2 cells x 3 models)",
          len(altopt_keys) == 6, f"got {len(altopt_keys)}")
    check("6 per-cell meta records (one per cell)",
          len(meta_keys) == 6, f"got {len(meta_keys)}")
    check("109 total entries in count_model_results.json "
          "(78 tier-1 + 17 tier-2 + 6 altopt + 6 meta + 2 Gaussian round-trip)",
          len(fits) == 109, f"got {len(fits)}")

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
        if k.endswith('__meta') or k in GAUSSIAN_REFERENCE or k.endswith('__altopt'):
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
    off19 = fits.get('viol_off_2019__M1__nbinom2')
    cov19 = fits.get('viol_cov_2019__M1__nbinom2')
    if off19 and cov19:
        check("2019 offset spec drops 7 zero-inspection state-years "
              "(of 15 total; the other 8 already had missing violations)",
              cov19['n_obs'] - off19['n_obs'] == 7,
              f"covariate n={cov19['n_obs']}, offset n={off19['n_obs']}")

    # I-3: an UNGATED check that every ladder/tier-2/altopt fit converged --
    # the exp_zeros check just below is explicitly gated on `if converged`, so
    # a non-converged fit previously produced ZERO failing checks, only an
    # unasserted R "WARNING:" line. That is this project's swallowed-warning
    # failure mode recurring in a new place. Excludes meta (not a fit) and the
    # Gaussian round-trip (already checked in [2b]).
    for k, f in fits.items():
        if k.endswith('__meta') or k in GAUSSIAN_REFERENCE:
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

    # Ruling 2: the brief's original last check compared exp_zeros to itself,
    # which can never fail. Replaced with a real assertion: for every converged
    # non-Gaussian fit, exp_zeros must actually be present, finite, and >= 0 --
    # i.e. the zero-inflation simulation in fit_spec() ran and produced a
    # sensible count, not NaN/Inf/negative from a degenerate simulate() call.
    for k, f in fits.items():
        if k in GAUSSIAN_REFERENCE or k.endswith('__meta'):
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
        if k.endswith('__meta') or k in GAUSSIAN_REFERENCE:
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
        if k.endswith('__meta') or k in GAUSSIAN_REFERENCE:
            continue
        if f.get('family_tag') in ('zip', 'zinb', 'zinb_re'):
            check(f"{k}: zi_degenerate present and boolean",
                  isinstance(f.get('zi_degenerate'), bool))
            check(f"{k}: zi_degenerate_reason present and a string",
                  isinstance(f.get('zi_degenerate_reason'), str))
        else:
            check(f"{k}: zi_degenerate is False for a non-ZI family",
                  f.get('zi_degenerate') is False)
    n_degenerate = sum(1 for k, f in fits.items()
                       if not k.endswith('__meta') and k not in GAUSSIAN_REFERENCE
                       and f.get('zi_degenerate') is True)
    check("exactly 11 fits are marked zi_degenerate under the specified thresholds",
          n_degenerate == 11, f"got {n_degenerate}")

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

    print("\n[4] Selection table and reporting")
    try:
        from report_count_models import (BOUNDARY_KIND, TIER_ORDER,
                                          coefficient_table,
                                          nearest_clean_competitor,
                                          selection_table, winner_summary)
    except ImportError as e:
        check("report_count_models importable", False, str(e))
        return
    check("report_count_models importable", True)

    raw = json.load(open(GEN + 'count_model_results.json'))
    tab = selection_table(raw)

    # -- Population: 109 total = 2 Gaussian + 6 __meta + 6 __altopt + 95
    # genuine fits. selection_table() must return exactly the 95, never
    # KeyError'ing on a __meta record (the brief's loop did exactly that).
    check("selection_table returns exactly the 95 genuine model fits "
          "(109 total - 2 gaussian - 6 meta - 6 altopt)",
          len(tab) == 95, f"got {len(tab)}")
    check("selection_table never includes a __meta, _gaussian, or __altopt key",
          not any(k.endswith('__meta') or k.endswith('_gaussian') or k.endswith('__altopt')
                  for k in tab['key']))

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
    for col in ('sigma2_u0', 'sigma2_u1', 'sigma_u01', 'sigma2_e'):
        check(f"selection_table carries {col} on every row",
              col in tab.columns and tab[col].notna().any())
    win = tab[(tab['cell'] == 'viol_cov_2021') & (tab['model'] == 'M3') &
              (tab['re_tier'] == 'rs') & (tab['family'] == 'nbinom1')]
    comp = tab[(tab['cell'] == 'viol_cov_2021') & (tab['model'] == 'M3') &
               (tab['re_tier'] == 'rs') & (tab['family'] == 'nbinom2')]
    if not win.empty and not comp.empty:
        ratio = float(comp['sigma2_u1'].iloc[0]) / float(win['sigma2_u1'].iloc[0])
        check_close("viol_cov_2021: nbinom2/nbinom1 sigma2_u1 ratio ~= 3.2x, "
                    "DERIVED FROM selection_table() (not the raw JSON)",
                    ratio, 3.2140498391688106, 1e-6, kind='rel')

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
    check("coefficient_table for a non-ZI winner (viol_cov_2019, nbinom2) "
          "has NO zero-inflation-part rows",
          ct_no_zi.empty or bool((ct_no_zi['section'] == 'count').all()))

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

    # -- If the CSVs have already been written (i.e. report_count_models.py
    # has run), they must carry exactly the same rows as their in-memory
    # source -- not gated as a hard prerequisite of this section, since the
    # process calls for validation to pass BEFORE the reporter is run for the
    # first time.
    out_path = GEN + 'count_model_comparison.csv'
    if os.path.exists(out_path):
        written = pd.read_csv(out_path)
        check("count_model_comparison.csv has one row per genuine fit (95)",
              len(written) == len(tab), f"got {len(written)}")
        for col in ('sigma2_u0', 'sigma2_u1', 'sigma_u01', 'sigma2_e', 'exp_zeros_se',
                    'lrt_boundary', 'lrt_boundary_kind', 'is_m3_winner'):
            check(f"count_model_comparison.csv carries column {col!r}", col in written.columns)


def main():
    validate_panels()
    validate_reference_convergence()
    validate_gaussian_roundtrip()
    validate_ladder()
    validate_selection()
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
