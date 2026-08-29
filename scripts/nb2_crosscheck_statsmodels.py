"""
Independent cross-software check for the NB2 stepwise arm.

WHAT THIS IS. statsmodels has no multilevel negative binomial, so this fits the
same Model-3 fixed effects with STATE FIXED EFFECTS (dummies) in place of the
random intercept and slope, using a different implementation in a different
language, and compares the shared coefficients against glmmTMB's.

WHAT THIS IS NOT. It is not a numerical equivalence test. Fixed effects are not
random effects; a random-effects estimator shrinks state deviations toward zero
and a dummy-variable estimator does not, so the coefficients differ by more than
optimizer noise and are expected to. The assertion in validate_nb2_stepwise [6]
is therefore SIGN AGREEMENT among terms both implementations resolve away from
zero. Ratios are reported for the reader; they are not thresholded. Tightening a
tolerance until it passes would prove nothing.

This replaces the ZI cross-check in validate_count_models.py [5], which has no
counterpart here: an NB2-only arm has no zero-inflation component to check.

FIX ROUND (2026-08-29): five of the six Model-3-added covariates
(SPEND_APP_z, SPEND_WORK_z, lii_2017_z, dol_demand_met_pct_z, pct_flc_z) are
Level-2 (state-level, time-invariant) by this project's own design -- one
value per state, constant across all years. A full set of state fixed effects
absorbs any state-invariant regressor BY CONSTRUCTION: the covariate column
is then an exact linear combination of the state-dummy columns, so the design
matrix is rank-deficient and statsmodels cannot invert the Hessian to produce
ANY standard error (verified: 0/56 finite SEs in both cells under the
original all-covariates design). Fitting those five terms was mathematically
impossible, not a solver failure. This version derives, programmatically and
per cell, which terms fixed effects can actually identify (max within-state
SD > WITHIN_STATE_SD_TOL on that cell's own analytic sample) and fits ONLY
those; the excluded terms are recorded by name in the JSON output
('terms_absorbed', 'absorbed_reason'), not silently dropped.

Removing the absorbed columns leaves a full-rank, well-conditioned design in
both cells (verified: rank == n columns; condition numbers ~1e3-4e3), but it
exposed a second, unrelated issue: statsmodels' all-zeros default start,
against 45 unscaled 0/1 state dummies and a heavy-tailed count DV
(max(violations) = 908), sends BFGS into an overflow region for
viol_cov_2021 that did not occur when the (now-removed) collinear columns
were still present. A generic, answer-agnostic start_params
(log of the sample mean for the intercept, 0 for every slope, alpha=1) fixes
it; verified to reproduce, bit-for-bit in log-likelihood, the same solution
the all-zeros default already found for insp_2021 (-2189.7314 either way), so
it is a robustness fix for the optimizer, not a change to the optimum.

Spec: docs/superpowers/specs/2026-08-28-nb2-stepwise-design.md section 5.2
Run:  python3 scripts/nb2_crosscheck_statsmodels.py
"""
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import NegativeBinomialP

GEN = '/Users/keshavgoel/Research/data/generated/'

# A term is "state-invariant" (absorbed by full state fixed effects) if its
# maximum within-state standard deviation on the cell's own analytic sample is
# at or below this tolerance. Named constant so it is not a buried magic
# number; 1e-12 comfortably separates exact-zero (true state-level constants,
# which measure as 0.0 here) from any genuine floating-point time variation.
WITHIN_STATE_SD_TOL = 1e-12

ABSORBED_REASON = ('state-invariant on this analytic sample (max within-state '
                    'SD <= tolerance) -- collinear with the state dummies by '
                    'construction, so a full state-fixed-effects design cannot '
                    'identify it')

CUBIC = ['time', 'time2', 'time3']
M3_ADD = ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
          'h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']

CELLS = {
    'insp_2021': {'dv': 'inspections', 'base': CUBIC},
    'viol_cov_2021': {'dv': 'violations', 'base': ['log_inspections'] + CUBIC},
}


def split_identifiable(d, rhs):
    """Split candidate terms into (kept, absorbed) using each term's maximum
    within-state standard deviation on the analytic sample `d`. Terms with
    max within-state SD > WITHIN_STATE_SD_TOL vary within at least one state
    and are identifiable under state fixed effects; terms at or below it are
    constant within every state and are collinear with the state dummies."""
    within_sd = d.groupby('state')[rhs].std().max()
    kept = [t for t in rhs if within_sd[t] > WITHIN_STATE_SD_TOL]
    absorbed = [t for t in rhs if within_sd[t] <= WITHIN_STATE_SD_TOL]
    return sorted(kept, key=rhs.index), sorted(absorbed)


def fit_one(panel, dv, rhs):
    """NB2 with state dummies, fit only over the subset of `rhs` that varies
    within at least one state (the rest are dropped from the design entirely
    -- they are what breaks the Hessian). Returns (params, bse, n_obs,
    n_states, converged, warning messages, kept terms, absorbed terms)."""
    need = [dv, 'state'] + rhs
    d = panel[need].dropna().copy()

    kept, absorbed = split_identifiable(d, rhs)

    X = d[kept].astype(float)
    dummies = pd.get_dummies(d['state'], prefix='st', drop_first=True).astype(float)
    X = pd.concat([X, dummies], axis=1)
    X.insert(0, 'const', 1.0)
    y = d[dv].astype(float)

    # Dropping the absorbed (state-invariant) columns leaves a full-rank,
    # well-conditioned design (verified: rank == n columns in both cells,
    # condition number ~4e3), but statsmodels' all-zeros default start,
    # combined with 45 unscaled 0/1 state dummies against a heavy-tailed
    # count DV (max(violations) = 908), sends BFGS into an overflow region
    # for viol_cov_2021 -- confirmed reproducible, and NOT present when the
    # (now-dropped) collinear columns were still in the design. A generic,
    # answer-agnostic start (log of the sample mean for the intercept, zero
    # for every slope, alpha=1) resolves it: verified to reproduce the exact
    # same log-likelihood/converged solution insp_2021 already reached under
    # the all-zeros default (-2189.7314 either way), so it is not moving the
    # optimum, only helping BFGS reach it reliably in both cells.
    start = np.zeros(X.shape[1] + 1)
    start[0] = np.log(y.mean())
    start[-1] = 1.0

    # Warnings are captured and reported, never filtered -- a
    # filterwarnings('ignore') is what hid a non-converged published fit in this
    # project once already.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = NegativeBinomialP(y.values, X.values, p=2)
        fit = model.fit(method='bfgs', maxiter=5000, disp=0, start_params=start)
    msgs = sorted({str(w.message) for w in caught})

    params = pd.Series(fit.params[:len(X.columns)], index=X.columns)
    bse = pd.Series(fit.bse[:len(X.columns)], index=X.columns)
    return (params, bse, len(d), d['state'].nunique(),
            bool(fit.mle_retvals['converged']), msgs, kept, absorbed)


def main():
    panel = pd.read_csv(GEN + 'count_model_panel_2021.csv')
    with open(GEN + 'nb2_stepwise_results.json') as fh:
        res = json.load(fh)

    out = {}
    for cell, cfg in CELLS.items():
        rhs = cfg['base'] + M3_ADD
        (params, bse, n_obs, n_states, converged, msgs,
         kept, absorbed) = fit_one(panel, cfg['dv'], rhs)
        tmb = res[f'{cell}__M3__nbinom2__rs']['cond']

        terms = {}
        for t in kept:
            b_sm, se_sm = float(params[t]), float(bse[t])
            b_tmb, se_tmb = float(tmb[t]['b']), float(tmb[t]['se'])
            # "Distinguishable from zero" == |b| > 2*SE in BOTH fits. A term
            # neither implementation resolves has no sign to agree about.
            both_nonzero = abs(b_sm) > 2 * se_sm and abs(b_tmb) > 2 * se_tmb
            terms[t] = {
                'b_sm': b_sm, 'se_sm': se_sm,
                'b_tmb': b_tmb, 'se_tmb': se_tmb,
                'ratio': b_sm / b_tmb if b_tmb != 0 else float('nan'),
                'sign_agrees': bool(np.sign(b_sm) == np.sign(b_tmb)),
                'both_nonzero': bool(both_nonzero),
            }

        out[cell] = {'n_obs': n_obs, 'n_states': n_states,
                     'converged': converged, 'warnings': msgs, 'terms': terms,
                     'terms_absorbed': absorbed,
                     'absorbed_reason': ABSORBED_REASON}

        print(f"\n{cell}  (statsmodels NB2 + {n_states - 1} state dummies, "
              f"N={n_obs}, converged={converged})")
        if msgs:
            print("  warnings raised during the fit (captured, not filtered):")
            for m in msgs:
                print(f"    - {m}")
        if absorbed:
            print(f"  excluded (state-invariant, collinear with state dummies): "
                  f"{absorbed}")
            print(f"    reason: {ABSORBED_REASON}")
        print(f"  {'term':24}{'b (statsmodels)':>18}{'b (glmmTMB)':>16}"
              f"{'ratio':>10}{'both != 0':>11}{'sign':>7}")
        for t, v in terms.items():
            print(f"  {t:24}{v['b_sm']:>18.4f}{v['b_tmb']:>16.4f}"
                  f"{v['ratio']:>10.3f}{str(v['both_nonzero']):>11}"
                  f"{('ok' if v['sign_agrees'] else 'DIFFERS'):>7}")

    with open(GEN + 'nb2_stepwise_crosscheck.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f"\nWrote {GEN}nb2_stepwise_crosscheck.json")
    print("\nNOTE: ratios are reported, not thresholded. State fixed effects are "
          "not\nstate random effects -- the estimates differ by more than "
          "optimizer noise by\nconstruction. Only sign agreement among "
          "clearly-nonzero terms is asserted.\nState-invariant terms are "
          "excluded from the design entirely (see terms_absorbed);\nthey are "
          "collinear with the state dummies and cannot be identified this way.")


if __name__ == '__main__':
    main()
