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



# ============================================================
# [3] PER-RECORD STRUCTURE
# ============================================================
FAMILIES = ('poisson', 'nbinom1', 'nbinom2', 'zip',
            'zinb1', 'zinb2', 'zinb1_re', 'zinb2_re')
ZI_FAMILIES = ('zip', 'zinb1', 'zinb2', 'zinb1_re', 'zinb2_re')
COUNTERPART = {'zip': 'poisson', 'zinb1': 'nbinom1', 'zinb2': 'nbinom2',
               'zinb1_re': 'nbinom1', 'zinb2_re': 'nbinom2'}


def validate_records():
    section('3', 'Per-record structure (a loop over the ladder, not independent findings)')
    r = load_results()
    expected = 4 * (8 + 8 + 1 + 1) + 8
    check(f'record count == {expected}', len(r) == expected, f'got {len(r)}')
    for k, v in r.items():
        if k.endswith('__meta'):
            continue
        # THE central constraint of this arm.
        check(f'{k}: no random slope', 'time | state' not in v['formula'])
        check(f'{k}: crossed REs present',
              '(1 | state) + (1 | year)' in v['formula'])
        if v.get('family_tag') != 'gaussian':
            check(f'{k}: sigma2_e is null (count fit)', v.get('sigma2_e') is None)
        if v.get('converged'):
            check(f'{k}: sigma2_state finite and positive',
                  v.get('sigma2_state') is not None and v['sigma2_state'] > 0)
            check(f'{k}: sigma2_year present', v.get('sigma2_year') is not None)
        fam = v.get('family_tag')
        if fam in ZI_FAMILIES:
            check(f'{k}: zi_tier_reached recorded', v.get('zi_tier_reached') is not None)
            for att in (v.get('zi_tier_attempts') or []):
                if att.get('skipped'):
                    check(f'{k}: skipped tier {att["tier"]} carries a reason',
                          bool(att.get('skip_reason')))
        elif fam in ('poisson', 'nbinom1', 'nbinom2'):
            check(f'{k}: plain family has an empty ZI block', not v.get('zi'))


# ============================================================
# [4] SAMPLE INVARIANCE ACROSS ZI TIERS
#     The precondition that makes cross-tier AIC valid (spec 5.3).
# ============================================================
def validate_sample_invariance():
    section('4', 'One analytic sample per (cell, model), across every ZI tier')
    r = load_results()
    for cell in CELLS:
        for model in ('M1', 'M3'):
            ns = {f: r[f'{cell}__{model}__{f}'].get('n_obs') for f in FAMILIES}
            check(f'{cell} {model}: all 8 families share one n_obs',
                  len(set(ns.values())) == 1, f'got {ns}')
            # And the ZI predictor set is a subset of the conditional one, which
            # is WHY the samples coincide -- asserted structurally, not observed.
            for f in ZI_FAMILIES:
                v = r[f'{cell}__{model}__{f}']
                cond_terms = set(v.get('cond') or {}) - {'(Intercept)'}
                zi_terms = set(v.get('zi') or {}) - {'(Intercept)'}
                check(f'{cell} {model} {f}: ZI terms subset of conditional terms',
                      zi_terms <= cond_terms, f'extra: {zi_terms - cond_terms}')


# ============================================================
# [5] SELECTION RE-DERIVED INDEPENDENTLY
# ============================================================
def validate_selection():
    section('5', 'Winners re-derived from the records, not read from __meta')
    r = load_results()
    for cell in CELLS:
        meta = r[f'{cell}__meta']
        for model, meta_key in (('M1', 'm1_winner'), ('M3', 'm3_winner')):
            elig = [(f, r[f'{cell}__{model}__{f}']) for f in FAMILIES]
            elig = [(f, v) for f, v in elig
                    if v.get('converged') and v.get('aic') is not None
                    and not v.get('collapsed_to') and not v.get('zi_degenerate')]
            if not elig:
                check(f'{cell} {model}: defaulted winner is flagged',
                      meta[f'winner_defaulted_{model.lower()}'] is True)
                continue
            want = min(elig, key=lambda kv: kv[1]['aic'])[0]
            check(f'{cell} {model}: winner == {want}', meta[meta_key] == want,
                  f'meta says {meta[meta_key]}')
            check(f'{cell} {model}: n_eligible matches',
                  meta[f'n_eligible_{model.lower()}'] == len(elig),
                  f'meta {meta[f"n_eligible_{model.lower()}"]} vs {len(elig)}')
            check(f'{cell} {model}: is_winner set on exactly one record',
                  sum(bool(r[f'{cell}__{model}__{f}'].get('is_winner'))
                      for f in FAMILIES) == 1)
        check(f'{cell}: a defaulted M3 winner is never reported as a selection result',
              not meta['winner_defaulted_m3'])
        check(f'{cell}: M2 fit exists under the M3 winner',
              f'{cell}__M2__{meta["m3_winner"]}' in r)


# ============================================================
# [6] DEGENERACY AND COLLAPSE GUARDS
# ============================================================
def validate_guards():
    section('6', 'ZI degeneracy and collapse guards fired where applicable')
    r = load_results()
    for cell in CELLS:
        for model in ('M1', 'M3'):
            for f in ZI_FAMILIES:
                v = r[f'{cell}__{model}__{f}']
                cp = r[f'{cell}__{model}__{COUNTERPART[f]}']
                want = bool(v.get('converged') and cp.get('converged')
                            and v.get('loglik') is not None
                            and cp.get('loglik') is not None)
                check(f'{cell} {model} {f}: loglik criterion applicability recorded',
                      bool(v.get('zi_loglik_criterion_applied')) == want,
                      f'recorded {v.get("zi_loglik_criterion_applied")}, expected {want}')
                if v.get('zi_degenerate'):
                    check(f'{cell} {model} {f}: degenerate fit carries a reason',
                          bool(v.get('zi_degenerate_reason')))
                    check(f'{cell} {model} {f}: degenerate fit is not eligible',
                          not v.get('eligible_for_selection'))
                if v.get('collapsed_to'):
                    check(f'{cell} {model} {f}: collapsed fit is not eligible',
                          not v.get('eligible_for_selection'))
                    check(f'{cell} {model} {f}: collapse target is its plain twin',
                          v['collapsed_to'] == f.replace('_re', ''))


# ============================================================
# [7] STRUCTURAL-ZERO PROBABILITY -- right estimand, re-derived
# ============================================================
def validate_zi_probability():
    section('7', 'Structural-zero probabilities: right estimand, re-derived')
    from scipy.special import expit
    sys.path.insert(0, SCRIPTS)
    from report_jafari_crossed import zi_marginal_prob

    r = load_results()
    var = pd.read_csv(GEN + 'jafari_crossed_variance.csv')
    ran = 0
    for _, row in var.iterrows():
        if pd.isna(row['pr_structural_zero_sample_mean']):
            continue
        ran += 1
        cell = row['cell']
        v = r[f'{cell}__M3__{row["family"]}']
        tier = v.get('zi_tier_reached')

        check_close(f'{cell}: sample-mean ZI probability round-trips from the JSON',
                    row['pr_structural_zero_sample_mean'], v.get('zi_prob_mean'), 1e-12)

        # THE consistency check that caught the wrong estimand: structural zeros
        # are a SUBSET of all zeros, so expected structural zeros can never
        # exceed the model's own simulated total zero count. The GH marginal
        # failed this by a factor of ~3 for viol_2021 (0.46 x 501 = 233 against
        # 78 simulated zeros); the sample mean must not.
        exp_struct = row['pr_structural_zero_sample_mean'] * row['n_obs']
        slack = 3 * (row['exp_zeros_se'] or 0)
        check(f'{cell}: expected structural zeros <= simulated total zeros',
              exp_struct <= row['exp_zeros'] + slack,
              f'{exp_struct:.2f} structural vs {row["exp_zeros"]:.2f} total zeros')

        if tier == 'intercept':
            b0 = v['zi']['(Intercept)']['b']
            s2 = v.get('sigma2_zi_state')
            check_close(f'{cell}: median-state probability == plogis(b0)',
                        row['pr_structural_zero_median_state'], float(expit(b0)), 1e-9)
            if s2 is not None and s2 > 0:
                check_close(f'{cell}: marginal probability re-derived',
                            row['pr_structural_zero_marginal'],
                            zi_marginal_prob(b0, s2), 1e-9)
                check_close(f'{cell}: GH stable under node doubling',
                            zi_marginal_prob(b0, s2, 240),
                            zi_marginal_prob(b0, s2, 480), 1e-4)
        else:
            # The guard against the defect itself: an intercept-evaluated
            # probability must NOT be reported when the ZI block carries
            # covariates, because then x = 0 describes no state in the sample.
            check(f'{cell} (tier={tier}): median-state probability withheld',
                  pd.isna(row['pr_structural_zero_median_state']))
            check(f'{cell} (tier={tier}): GH marginal withheld',
                  pd.isna(row['pr_structural_zero_marginal']))
    if ran == 0:
        skip('structural-zero probability re-derivation',
             'no selected model has a zero-inflation component')


# ============================================================
# [8] INDEPENDENT ZI CROSS-CHECK -- different package, fixed effects
# ============================================================
def validate_zi_crosscheck():
    section('8', 'Cross-software ZI check (statsmodels, state+year fixed effects)')
    import numpy as np
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    import statsmodels.api as sm

    r = load_results()
    ran = 0
    for cell in CELLS:
        meta = r[f'{cell}__meta']
        fam = meta['m3_winner']
        if fam not in ('zinb1', 'zinb2', 'zinb1_re', 'zinb2_re'):
            skip(f'{cell}: ZI cross-check', f'selected family {fam} has no ZI component')
            continue
        ran += 1
        window, outcome, base = CELL_SPEC[cell]
        v = r[f'{cell}__M3__{fam}']
        rhs = base + M3_ADD
        d = panel(window).dropna(subset=[outcome] + rhs + ['state', 'year']).copy()
        X = sm.add_constant(pd.concat([
            d[rhs].reset_index(drop=True),
            pd.get_dummies(d['state'], prefix='st', drop_first=True).reset_index(drop=True),
            pd.get_dummies(d['year'], prefix='yr', drop_first=True).reset_index(drop=True),
        ], axis=1).astype(float))
        zi_vars = [t for t in (v.get('zi') or {}) if t != '(Intercept)']
        Z = (sm.add_constant(d[zi_vars].reset_index(drop=True).astype(float))
             if zi_vars else np.ones((len(d), 1)))
        p_par = 1 if fam.startswith('zinb1') else 2
        res = None
        conv = False
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                res = ZeroInflatedNegativeBinomialP(
                    d[outcome].values, X.values, exog_infl=np.asarray(Z, float),
                    p=p_par).fit(disp=0, maxiter=500)
            conv = bool(res.mle_retvals.get('converged'))
        except Exception as exc:
            print(f'    (cross-check raised {type(exc).__name__}: {exc})')
        if not conv:
            # A non-converged cross-check carries NO interpretive weight in
            # either direction. The previous arm's equivalent diverged for
            # violations on a full-rank ~53-parameter design -- the known
            # incidental-parameters fragility of a fixed-effects count mixture.
            # Recorded as a SKIP, never as evidence against the glmmTMB fit.
            skip(f'{cell}: ZI intercept agreement',
                 'statsmodels fixed-effects ZINB did not converge; carries no '
                 'interpretive weight either way')
            continue
        sm_b0 = float(res.params[0])
        tmb_b0 = v['zi']['(Intercept)']['b']
        # Deliberately loose: different package, and fixed effects vs. random
        # effects are different estimands. This detects a sign or
        # order-of-magnitude error, not a numerical discrepancy.
        check(f'{cell}: ZI intercept same sign across packages',
              np.sign(sm_b0) == np.sign(tmb_b0), f'statsmodels {sm_b0}, glmmTMB {tmb_b0}')
        check(f'{cell}: ZI intercept within 2.0 on the logit scale',
              abs(sm_b0 - tmb_b0) < 2.0, f'statsmodels {sm_b0}, glmmTMB {tmb_b0}')
    if ran == 0:
        skip('ZI cross-check', 'no cell selected a ZI-NB family at Model 3')


# ============================================================
# [9] MEMO CONSISTENCY AND FORBIDDEN CLAIMS
# ============================================================
MEMO = '/Users/keshavgoel/Research/docs/jafari_crossed_models.md'

# Claims this design KNOWS to be false. Each string is chosen so it can appear
# ONLY in a false claim -- a bare phrase like "random slope" is useless as a
# guard here, because the memo legitimately says there is NOT one.
MEMO_FORBIDDEN = [
    'delta_pct',                      # no variance-reduction % is computed in this arm
    'Δσ',                   # nor written as a delta-sigma
    'plogis(b0) is the marginal',
    'this arm fits a random slope',
    'directly comparable to the previous arm',
]

MEMO_REQUIRED = [
    'Not comparable to `docs/count_models_zinb.md`',
    'log link',
    'sigma^2_year rests on',
    '*Conditional Model*',
    '*Zero-inflated Model*',
    'not** comparable to the COVID coefficients',
    'sample mean',
]

FAMILY_LABEL = {'poisson': 'Poisson', 'nbinom1': 'NB1', 'nbinom2': 'NB2',
                'zip': 'ZIP', 'zinb1': 'ZI-NB1', 'zinb2': 'ZI-NB2',
                'zinb1_re': 'ZI-NB1 + ZI-RE', 'zinb2_re': 'ZI-NB2 + ZI-RE'}


def validate_memo():
    section('9', 'Memo consistency and forbidden claims')
    t = open(MEMO).read()
    for pat in MEMO_FORBIDDEN:
        check(f'memo does NOT claim: {pat!r}', pat not in t)
    for pat in MEMO_REQUIRED:
        check(f'memo states: {pat!r}', pat in t)
    for bad in ('TODO', 'TBD', 'FIXME', ' nan ', 'None |', '| nan'):
        check(f'memo free of {bad!r}', bad not in t)

    r = load_results()
    for cell in CELLS:
        fam = r[f'{cell}__meta']['m3_winner']
        check(f'{cell}: memo names the selected family {FAMILY_LABEL[fam]!r}',
              f'**Selected family:** {FAMILY_LABEL[fam]}' in t)

    sel = pd.read_csv(GEN + 'jafari_crossed_selection.csv')
    bad = 0
    for _, row in sel.iterrows():
        v = r[f'{row.cell}__{row.model}__{row.family}']
        if row.converged and abs((v['aic'] or 0) - row.aic) > 1e-9:
            bad += 1
    check('every selection-CSV AIC round-trips from the JSON', bad == 0, f'{bad} mismatches')

    # The memo's headline tally must be the one the CSV supports.
    tal = pd.read_csv(GEN + 'jafari_crossed_zi_strength.csv')
    own = tal[tal.is_own_counterpart]
    claim = f'**{int(own.zi_wins.sum())} of {len(own)}** matched'
    check(f'memo own-counterpart tally matches the CSV ({claim})', claim in t)


# ============================================================
# [10] FREEZE GUARD
# ============================================================
FROZEN = [
    'scripts/count_models_zinb.R', 'scripts/report_count_models.py',
    'scripts/validate_count_models.py', 'scripts/build_count_model_memo_evidence.py',
    'docs/count_models_zinb.md', 'docs/results_count_models_zinb.md',
    'data/generated/count_model_results.json',
    'scripts/nb2_stepwise_models.R', 'scripts/report_nb2_stepwise.py',
    'scripts/validate_nb2_stepwise.py', 'scripts/nb2_crosscheck_statsmodels.py',
    'docs/nb2_stepwise_models.md', 'data/generated/nb2_stepwise_results.json',
    'data/generated/count_model_panel_2019.csv',
    'data/generated/count_model_panel_2021.csv',
]


def validate_freeze():
    section('10', 'Frozen artifacts untouched by this branch')
    # SCOPE, stated honestly: this verifies "this BRANCH did not touch the
    # frozen artifacts", not "they have never changed". On `main` the merge-base
    # IS HEAD, so the history arm is vacuous and only the working-tree arm
    # bites -- the same limitation recorded in CLAUDE.md for
    # validate_nb2_stepwise.py [8]. This arm is developed on a feature branch,
    # where the question it answers is the right one.
    base = subprocess.run(['git', 'merge-base', 'main', 'HEAD'],
                          capture_output=True, text=True, cwd=ROOT).stdout.strip()
    head = subprocess.run(['git', 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, cwd=ROOT).stdout.strip()
    if base == head:
        skip('committed-history arm',
             'merge-base == HEAD (on the default branch); this arm is vacuous here')
    else:
        changed = subprocess.run(['git', 'diff', '--name-only', f'{base}..HEAD'],
                                 capture_output=True, text=True, cwd=ROOT).stdout.split()
        for f in FROZEN:
            check(f"unmodified in this branch's commits: {f}", f not in changed)
    dirty = subprocess.run(['git', 'status', '--porcelain'],
                           capture_output=True, text=True, cwd=ROOT).stdout.split('\n')
    dirty = {ln[3:].strip() for ln in dirty if ln.strip()}
    for f in FROZEN:
        check(f'unmodified in the working tree: {f}', f not in dirty)


def main():
    skip_pre = '--skip-precondition' in sys.argv
    validate_precondition(skip_pre)
    validate_panel()
    validate_crossed_roundtrip()
    validate_records()
    validate_sample_invariance()
    validate_selection()
    validate_guards()
    validate_zi_probability()
    validate_zi_crosscheck()
    validate_memo()
    validate_freeze()

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
