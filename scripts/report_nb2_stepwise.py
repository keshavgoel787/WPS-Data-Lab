"""
Reads the NB2 stepwise artifacts and produces the tables and the memo.

PURE ARTIFACT READER. It fits nothing. Every number it prints traces to one of
three sources: data/generated/nb2_stepwise_results.json (the six substantive
fits, the RI series, the M1matched baseline, the COVID refits),
nb2_stepwise_crosscheck.json (the statsmodels cross-check), or -- for the AIC
penalty comparison and the COVID-family comparison ONLY -- the frozen
data/generated/count_model_results.json (the six-family ladder), opened
read-only.

This is the ONLY script in the arm that REPORTS delta_pct_ri,
delta_pct_ri_matched and icc_ri, computed here as:

  delta_pct_ri         = 100 * (s2_u0_ri[M1]        - s2_u0_ri[m]) / s2_u0_ri[M1]
  delta_pct_ri_matched = 100 * (s2_u0_ri[M1matched] - s2_u0_ri[m]) / s2_u0_ri[M1matched]
  icc_ri               = s2 / (s2 + log(1 + 1/theta + 1/mu)),  all from the RI fit

validate_nb2_stepwise.py [7] independently RE-DERIVES all three straight from
the results JSON rather than importing these functions -- that is what makes
[7] a non-circular cross-check, and it must stay that way.

Spec: docs/superpowers/specs/2026-08-28-nb2-stepwise-design.md
Run:  python3 scripts/report_nb2_stepwise.py
"""
import json
import math

import pandas as pd

GEN = '/Users/keshavgoel/Research/data/generated/'
DOCS = '/Users/keshavgoel/Research/docs/'

CELLS = {'insp_2021': 'inspections', 'viol_cov_2021': 'violations'}
MODELS = ('M1', 'M2', 'M3')

TERM_ORDER = ['(Intercept)', 'log_inspections', 'time', 'time2', 'time3',
              'SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
              'h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z',
              'covid']


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


# The frozen ladder's COVID variants, for the family comparison in the memo.
# They sit under DIFFERENT families -- that is the whole point of re-deriving
# them under NB2 -- so the key carries the family it was fit under.
# Display labels only (cosmetic) for the ladder's own family_name strings --
# the family that wins each rung is still looked up from the ladder, not
# hardcoded; this dict only controls how its name is printed.
FAMILY_DISPLAY = {'poisson': 'Poisson', 'nbinom1': 'NB1', 'nbinom2': 'NB2',
                   'zip': 'ZIP', 'zinb': 'ZINB', 'zinb_re': 'ZINB+ZI-RE'}

COVID_LADDER_REF = {
    'insp_2021': ('insp_2021__M3covid__zinb', 'ZINB'),
    'viol_cov_2021': ('viol_cov_2021__M3covid__nbinom1', 'NB1'),
}


def load():
    with open(GEN + 'nb2_stepwise_results.json') as fh:
        res = json.load(fh)
    with open(GEN + 'nb2_stepwise_crosscheck.json') as fh:
        cc = json.load(fh)
    # READ-ONLY. The ladder artifact is frozen; it is opened here only to quote
    # its COVID variants' coefficients beside ours.
    with open(GEN + 'count_model_results.json') as fh:
        ladder = json.load(fh)
    return res, cc, ladder


def icc_nb2(s2_u0, theta, mu):
    """Nakagawa's observation-level variance for a log-link NB2. A
    LINK-SCALE approximation -- not comparable to the project's Gaussian-LMM
    ICC on log(count + 1)."""
    return s2_u0 / (s2_u0 + math.log(1 + 1 / theta + 1 / mu))


# --------------------------------------------------------------------------
# Fix for review B2: the memo used to hardcode the AIC penalty NB2 pays
# against the ladder's selected family ("about 33 ... about 11 ... at both
# Model 1 and Model 3"). That text was wrong (M1's violations penalty is
# +7.06, not +11) and never covered M2 at all. Derive every penalty straight
# from the already-loaded `ladder` dict instead of typing any of it in.
# --------------------------------------------------------------------------
def _plain_ladder_rungs(ladder, cell, model):
    """The frozen ladder's own fits at this exact (cell, model): keys of the
    form '{cell}__{model}__{family}' (exactly three '__'-separated parts).
    Excludes '__altopt'/'__zisens' variants and other model tags
    (M3covid/M1matched), which all split into more than three parts."""
    out = {}
    for key, r in ladder.items():
        parts = key.split('__')
        if len(parts) == 3 and parts[0] == cell and parts[1] == model:
            out[parts[2]] = r
    return out


def best_ladder_rung(ladder, cell, model):
    """The minimum-AIC converged, non-degenerate, rs-tier ladder family at
    this rung -- the comparator the spec's own AIC-penalty table (section 3)
    uses. Returns (family_name, aic)."""
    candidates = {
        fam: r for fam, r in _plain_ladder_rungs(ladder, cell, model).items()
        if r.get('re_tier') == 'rs' and r.get('converged')
        and not r.get('zi_degenerate', False)
    }
    fam, r = min(candidates.items(), key=lambda kv: kv[1]['aic'])
    return fam, r['aic']


def aic_penalty_table(res, ladder):
    """NB2's AIC penalty against the best available ladder family, at every
    (cell, model) rung NB2 was fit -- M1, M2 and M3, not just M1/M3.

    M5 (residual finding, fix wave 2026-08-29): a record that failed to
    converge lands in the R error branch and carries only
    'converged'/'message' -- no 'aic'. Direct indexing used to raise
    KeyError here, and because this function runs at the TOP of
    write_memo(), that crash happened after main() had already written both
    CSVs, leaving the memo stale relative to them. Gate on 'converged' and
    emit a None-valued row instead, so all three artifacts stay in sync
    (either all written fresh, or the whole run aborts before any of them
    do -- never a partial write)."""
    rows = []
    for cell, outcome in CELLS.items():
        for model in MODELS:
            r = res[f'{cell}__{model}__nbinom2__rs']
            if not r.get('converged'):
                rows.append({'cell': cell, 'outcome': outcome, 'model': model,
                             'nb2_aic': None, 'best_aic': None,
                             'best_family': None, 'penalty': None})
                continue
            nb2_aic = r['aic']
            fam, best_aic = best_ladder_rung(ladder, cell, model)
            rows.append({'cell': cell, 'outcome': outcome, 'model': model,
                         'nb2_aic': nb2_aic, 'best_aic': best_aic,
                         'best_family': fam, 'penalty': nb2_aic - best_aic})
    return pd.DataFrame(rows)


def nb1_margin_table(res, ladder):
    """NB2's AIC margin over a plain NB1 fit at the SAME rung, wherever the
    frozen ladder actually has one. Positive margin = NB2 beats NB1.

    Coverage is NOT symmetric across cells, and must not be described as if
    it were (residual-review Finding 2: an earlier draft claimed "the ladder
    fits M2 only under each cell's selected family, so neither cell has an
    M2 nbinom1 entry" -- true for inspections, false for violations, whose
    selected family IS nbinom1, so the ladder fits one at every model
    including M2). Which (cell, model) rows this table actually returns is
    read back out of `ladder` here, at call time, by callers that need to
    describe the coverage in prose -- nothing above should be restated as a
    fixed M1/M3 claim.

    M5: a non-converged NB2 record has no 'aic' either; gated the same way
    as aic_penalty_table for the same reason (write_memo() calls this before
    the memo is assembled, after the CSVs are already on disk)."""
    rows = []
    for cell, outcome in CELLS.items():
        for model in MODELS:
            key = f'{cell}__{model}__nbinom1'
            if key not in ladder:
                continue
            r = res[f'{cell}__{model}__nbinom2__rs']
            if not r.get('converged'):
                rows.append({'cell': cell, 'outcome': outcome, 'model': model,
                             'nb1_aic': ladder[key]['aic'], 'nb2_aic': None,
                             'margin': None})
                continue
            nb1_aic = ladder[key]['aic']
            nb2_aic = r['aic']
            rows.append({'cell': cell, 'outcome': outcome, 'model': model,
                         'nb1_aic': nb1_aic, 'nb2_aic': nb2_aic,
                         'margin': nb1_aic - nb2_aic})
    return pd.DataFrame(rows)


def _fmt(v, spec='.2f'):
    """Safe formatter for the memo's derived tables/prose: None (a
    non-converged rung, M5) prints as 'NC' instead of raising on `:spec`."""
    return 'NC' if v is None or v != v else format(v, spec)


def _sig(p):
    return p < .05


def pre_covid_time_trend_bullet(outcome, old_fam, o_m3, n_m3):
    """The 'before COVID enters at all' bullet. Fix for review Finding 1: the
    old text asserted 'the two families already disagree' unconditionally,
    but for inspections neither `time` nor `time2` actually flips
    significance status (both p<.05 on `time`, both n.s. on `time2`) -- the
    families agree there. Compute the verdict from the p-values themselves,
    per term, and phrase the sentence accordingly, following the same
    derive-don't-assert pattern used for the sample-effect direction/reading
    a few sections up in this file. Never claim a disagreement the numbers
    don't show, and never suppress a genuine one (violations' `time` really
    does flip)."""
    terms = ('time', 'time2')
    flipped = [t for t in terms if _sig(o_m3[t]['p']) != _sig(n_m3[t]['p'])]
    detail = ' and '.join(
        f'`{t}` p = {o_m3[t]["p"]:.3g} under {old_fam} against '
        f'{n_m3[t]["p"]:.3g} under NB2'
        for t in terms)
    if flipped:
        flip_list = ' and '.join(f'`{t}`' for t in flipped)
        verb = 'disagree'
        tail = (f'Significance status (p < .05) flips on {flip_list}. That '
                'gap is a property of the family choice, not of the '
                'pandemic.')
    else:
        verb = 'agree'
        statuses = ', '.join(
            f'`{t}` {"significant" if _sig(o_m3[t]["p"]) else "non-significant"} '
            'under both families'
            for t in terms)
        tail = f'Both families agree: {statuses}.'
    return (f'- **{outcome}, before COVID enters at all:** the two families '
            f'{verb} on Model 3\'s own time trend -- {detail}. {tail}')


def variance_table(res):
    """M5: a fit that failed to converge lands in the R error branch carrying
    only 'converged'/'message' -- no 'sigma2_u0', 'cond', etc. Direct
    indexing on those keys would raise KeyError and abort report generation
    before validate_nb2_stepwise.py ever runs (which is exactly the crash the
    spec calls a defect, not just in the validator). Every field access below
    is therefore gated on 'converged' first; a non-converged row is emitted
    with None in the fields that fit could not have produced, rather than
    raising."""
    rows = []
    for cell, outcome in CELLS.items():
        base_r = res[f'{cell}__M1__nbinom2__ri']
        matched_r = res[f'{cell}__M1matched__nbinom2__ri']
        base = base_r['sigma2_u0'] if base_r.get('converged') else None
        matched = matched_r['sigma2_u0'] if matched_r.get('converged') else None
        for model in MODELS:
            rs = res[f'{cell}__{model}__nbinom2__rs']
            ri = res[f'{cell}__{model}__nbinom2__ri']
            rs_ok, ri_ok = rs.get('converged'), ri.get('converged')
            sigma2_u0_ri = ri['sigma2_u0'] if ri_ok else None
            theta = ri['dispersion'] if ri_ok else None
            mu_fixed = ri['mu_fixed'] if ri_ok else None
            rows.append({
                'cell': cell, 'outcome': outcome, 'model': model,
                'sigma2_u0_rs': rs['sigma2_u0'] if rs_ok else None,
                'sigma2_u1_rs': rs['sigma2_u1'] if rs_ok else None,
                'sigma_u01_rs': rs['sigma_u01'] if rs_ok else None,
                'aic_rs': rs['aic'] if rs_ok else None,
                'theta': theta,
                'sigma2_u0_ri': sigma2_u0_ri,
                'icc_ri': (icc_nb2(sigma2_u0_ri, theta, mu_fixed)
                           if ri_ok else None),
                'mu_fixed': mu_fixed,
                'delta_pct_ri': (100 * (base - sigma2_u0_ri) / base
                                 if ri_ok and base is not None else None),
                'delta_pct_ri_matched': (
                    100 * (matched - sigma2_u0_ri) / matched
                    if ri_ok and matched is not None else None),
                'n_obs': rs.get('n_obs'), 'n_states': rs.get('n_states'),
            })
    return pd.DataFrame(rows)


def coefficient_table(res):
    rows = []
    for cell, outcome in CELLS.items():
        for model in list(MODELS) + ['M3covid']:
            r = res.get(f'{cell}__{model}__nbinom2__rs')
            # M5: skip records that exist but never converged -- they carry
            # no 'cond' block to iterate.
            if r is None or not r.get('converged'):
                continue
            for term, c in r['cond'].items():
                rows.append({'cell': cell, 'outcome': outcome, 'model': model,
                             'term': term, 'b': c['b'], 'se': c['se'],
                             'z': c['z'], 'p': c['p'], 'irr': math.exp(c['b'])})
    df = pd.DataFrame(rows)
    df['_ord'] = df['term'].apply(
        lambda t: TERM_ORDER.index(t) if t in TERM_ORDER else len(TERM_ORDER))
    return df.sort_values(['cell', 'model', '_ord']).drop(columns='_ord')


def md_table(df, cols, fmt):
    head = '| ' + ' | '.join(cols) + ' |'
    rule = '|' + '|'.join(['---'] * len(cols)) + '|'
    body = []
    for _, r in df.iterrows():
        body.append('| ' + ' | '.join(fmt(r, c) for c in cols) + ' |')
    return '\n'.join([head, rule] + body)


def write_memo(res, cc, ladder, vt, ct):
    L = []
    A = L.append
    A('# NB2-Only Stepwise Count Models, 2011-2021')
    A('')
    A('**This file is generated by `scripts/report_nb2_stepwise.py` and is '
      'never hand-edited.** Re-run that script instead. Every number traces to '
      'one of three sources: `data/generated/nb2_stepwise_results.json` (the '
      'six substantive fits, the RI series, the M1matched baseline, the COVID '
      'refits), `nb2_stepwise_crosscheck.json` (the statsmodels cross-check), '
      'or -- for the AIC-penalty comparison and the COVID-family comparison '
      'only -- the frozen `count_model_results.json` (the six-family ladder), '
      'opened read-only.')
    A('')
    A('Spec: `docs/superpowers/specs/2026-08-28-nb2-stepwise-design.md`.')
    A('')
    A('## What this is, and what it is not')
    A('')
    A('Six models: inspections M1-M3 and violations M1-M3 on the 2011-2021 WPS '
      'panel, all under `nbinom2`, all at `(1 + time | state)`. Model 1 is '
      'cubic time only; Model 2 adds commodity mix and spending '
      '(`lii_2017_z`, `SPEND_APP_z`, `SPEND_WORK_z`); Model 3 adds the H-2A '
      'block (`h2a_per_farmworker_z`, `dol_demand_met_pct_z`, `pct_flc_z`). '
      'The violations models carry `log_inspections` as a Level-1 covariate, '
      'matching the published Table 3.')
    A('')
    apt = aic_penalty_table(res, ladder)
    nb1t = nb1_margin_table(res, ladder)
    insp_pen = apt[apt['cell'] == 'insp_2021']['penalty']
    viol_pen = apt[apt['cell'] == 'viol_cov_2021']['penalty']
    A('**Holding the family at NB2 is an editorial decision, not a fit-based '
      'one.** At this random-effects structure NB2 loses to the ladder\'s '
      'selected family at every rung it was fit -- Model 1, Model 2 and Model '
      '3, for both outcomes:')
    A('')
    A('| Cell | Model | NB2 AIC | Best available AIC | Family | NB2 penalty |')
    A('|---|---|---|---|---|---|')
    for _, r in apt.iterrows():
        A(f"| {r['outcome']} | {r['model']} | {_fmt(r['nb2_aic'])} | "
          f"{_fmt(r['best_aic'])} | "
          f"{FAMILY_DISPLAY.get(r['best_family'], r['best_family']) or 'NC'} | "
          f"{_fmt(r['penalty'], '+.2f')} |")
    A('')
    A(f'That is {insp_pen.min():.2f}-{insp_pen.max():.2f} AIC against ZINB for '
      f'inspections and {viol_pen.min():.2f}-{viol_pen.max():.2f} AIC against '
      'NB1 for violations -- NB2 never wins a rung. What fixing the family '
      'buys is a table whose three columns are the same model class, so the '
      'between-state variance actually forms a reduction sequence instead of '
      'being three different models\' parameters.')
    A('')
    # Finding 2 (residual-review fix wave 2026-08-29): the old paragraph
    # asserted, for BOTH outcomes at once, that the ladder's NB1 coverage is
    # "M1 and M3 only... neither cell has an M2 NB1 entry". That is true for
    # inspections (ZINB, not NB1, is its selected family, so the ladder never
    # fits a plain NB1 for insp_2021's M2) but false for violations, whose
    # selected family IS nbinom1 -- the ladder fits one at every model there,
    # M2 included (viol_cov_2021__M2__nbinom1 exists, converged, AIC
    # 3630.469), and the very next clause of the old sentence quoted that
    # M2 value without noticing the contradiction. Derive each outcome's
    # actual model coverage from `nb1t` (built straight from the ladder JSON)
    # instead of asserting it, so this cannot rot the same way twice.
    insp_present = [m for m in MODELS
                    if m in set(nb1t.loc[nb1t['cell'] == 'insp_2021', 'model'])]
    viol_present = [m for m in MODELS
                    if m in set(nb1t.loc[nb1t['cell'] == 'viol_cov_2021', 'model'])]
    insp_missing = [m for m in MODELS if m not in insp_present]
    viol_missing = [m for m in MODELS if m not in viol_present]
    nb1_bits = ', '.join(
        f"{_fmt(r['margin'], '+.2f')} AIC at {r['model']}"
        for _, r in nb1t[nb1t['cell'] == 'insp_2021'].iterrows())
    viol_bits = ', '.join(
        f"{_fmt(r['penalty'], '+.2f')} AIC at {r['model']}"
        for _, r in apt[apt['cell'] == 'viol_cov_2021'].iterrows())
    insp_gap = (f'inspections only has an NB1 entry at {"/".join(insp_present)} '
                f'(no {"/".join(insp_missing)}), because ZINB, not NB1, is its '
                'selected family there'
                if insp_missing else
                'the ladder fits an NB1 at every model for inspections too')
    viol_gap = (f'violations only has an NB1 entry at {"/".join(viol_present)} '
                f'(no {"/".join(viol_missing)})'
                if viol_missing else
                'unlike inspections, the ladder fits an NB1 at every model for '
                'violations too, because NB1 is precisely violations\' own '
                'selected family')
    A(f'NB2 is nonetheless the least-bad single choice across both outcomes: '
      f'wherever the frozen ladder has its own plain-NB1 fit at the same rung, '
      f'NB2 beats it for inspections by {nb1_bits} ({insp_gap}), while '
      f'costing NB2 {viol_bits} against NB1 for violations at every model, '
      f'{"/".join(viol_present)} ({viol_gap} -- so that M2 cost is the same '
      'number as its penalty-table row above, not a gap in coverage).')
    A('')
    A('The family-selection evidence, including the zero-inflation comparisons, '
      'lives in `docs/count_models_zinb.md` and is untouched by this arm. '
      'Nothing here re-tests zero-inflation.')
    A('')
    A('## Variance components')
    A('')
    A('**Two bases in one table, labelled per column.** The `*_rs` columns come '
      'from the random-slope fits the coefficients come from. Everything else '
      '-- theta, ICC, and both reductions -- comes from a parallel '
      '`(1 | state)` refit series on the same rows. That split is deliberate: '
      'under a random slope, sigma^2_u0 is the between-state variance *at the '
      '2017 centering year* and trades off against sigma^2_u1, so its '
      'reduction is not bounded to [0, 1] and can go negative. That is not '
      'hypothetical -- it produced a -12.5% figure in the log-LMM violations '
      'table.')
    A('')
    A('All of these are on the log link scale. Their magnitudes are not '
      'comparable to the sigma^2_u0 values in published Tables 2 and 3, though '
      'a percentage reduction is.')
    A('')
    A('There is no residual-variance column. A count family has none; theta is '
      'the NB2 size parameter in `Var = mu + mu^2/theta`, and its square is not '
      'a variance component.')
    A('')
    A('ICC uses Nakagawa\'s observation-level variance for a log-link NB2, '
      '`sigma^2_u0 / (sigma^2_u0 + ln(1 + 1/theta + 1/mu))`, with mu the '
      'exponentiated mean of the fixed-effects linear predictor. It is a '
      'link-scale approximation and is **not** the project\'s headline '
      '"ICC ~77%", which comes from a Gaussian LMM on `log(count + 1)`.')
    A('')
    # A3: the reported ICC is computed entirely from the RI fit (sigma2_u0,
    # theta and mu all from the (1 | state) refit), but the coefficients two
    # sections below come from the random-slope (rs) fit, whose implied ICC
    # differs because its own sigma2_u0/theta/mu differ. Derive both numbers
    # for Model 1 of each cell rather than asserting a generic caveat.
    icc_lines = []
    for cell, outcome in CELLS.items():
        rs = res[f'{cell}__M1__nbinom2__rs']
        ri = res[f'{cell}__M1__nbinom2__ri']
        if rs.get('converged') and ri.get('converged'):
            icc_rs = icc_nb2(rs['sigma2_u0'], rs['dispersion'], rs['mu_fixed'])
            icc_ri = icc_nb2(ri['sigma2_u0'], ri['dispersion'], ri['mu_fixed'])
            icc_lines.append(f'{outcome} M1 is {icc_ri:.3f} on the RI basis '
                              f'versus {icc_rs:.3f} on the RS basis')
    A('**The reported ICC is on the random-intercept basis, not the '
      'random-slope basis the coefficients come from.** ' +
      '; '.join(icc_lines) + ' -- the two are not the same quantity, because '
      'sigma^2_u0, theta and mu all differ between the RI and RS fits. Read '
      'the ICC column as describing the RI series alongside it, not the '
      'random-slope models whose coefficients follow in the next section.')
    A('')
    for cell, outcome in CELLS.items():
        sub = vt[vt['cell'] == cell]
        A(f'### {outcome} ({cell})')
        A('')
        A('| Model | sigma^2_u0 (rs) | sigma^2_u1 (rs) | sigma_u01 (rs) | '
          'AIC (rs) | theta (ri) | sigma^2_u0 (ri) | ICC (ri) | '
          'Delta sigma^2_u0 % | Delta % matched | N obs | States |')
        A('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in sub.iterrows():
            def fmt(v, spec):
                return 'NC' if v is None or v != v else format(v, spec)
            A(f"| {r['model']} | {fmt(r['sigma2_u0_rs'], '.4f')} | "
              f"{fmt(r['sigma2_u1_rs'], '.5f')} | "
              f"{fmt(r['sigma_u01_rs'], '.5f')} | {fmt(r['aic_rs'], '.2f')} | "
              f"{fmt(r['theta'], '.4f')} | {fmt(r['sigma2_u0_ri'], '.4f')} | "
              f"{fmt(r['icc_ri'], '.3f')} | {fmt(r['delta_pct_ri'], '+.1f')} | "
              f"{fmt(r['delta_pct_ri_matched'], '+.1f')} | "
              f"{fmt(r['n_obs'], '.0f')} | {fmt(r['n_states'], '.0f')} |")
        A('')
    # A1: the Delta sigma^2_u0 % column can rise from M2 to M3 even when the
    # added H-2A block does not improve the fit -- derive the AIC verdict and
    # the added terms' significance directly, rather than leaving the reader
    # to infer it from the variance table alone.
    A('**Did the H-2A block (Model 3) improve on Model 2 by AIC?** Derived '
      'directly, not asserted:')
    A('')
    addon_terms = ['h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']
    for cell, outcome in CELLS.items():
        m2 = res[f'{cell}__M2__nbinom2__rs']
        m3 = res[f'{cell}__M3__nbinom2__rs']
        if not (m2.get('converged') and m3.get('converged')):
            A(f'- **{outcome}:** cannot be derived -- Model 2 or Model 3 did '
              'not converge.')
            continue
        d_aic = m3['aic'] - m2['aic']
        verdict = ('improved on' if d_aic < 0 else
                   'did not improve on' if d_aic > 0 else 'exactly matched')
        ps = {t: m3['cond'][t]['p'] for t in addon_terms}
        any_sig = any(p < .05 for p in ps.values())
        p_bits = ', '.join(f'`{t}` p = {p:.3g}' for t, p in ps.items())
        A(f'- **{outcome}:** Model 3 {verdict} Model 2 by AIC '
          f'({m3["aic"]:.2f} vs {m2["aic"]:.2f}, delta = {d_aic:+.2f} for '
          f'three added parameters); the added H-2A-block coefficients are '
          f'{p_bits} -- {"none" if not any_sig else "at least one"} '
          'significant at p<.05.')
    A('')
    A('**Why two reduction columns.** Model 1 keeps all 49 states; Models 2 and '
      '3 keep 46, because AK, RI and VT have no BLS pesticide-applicator series '
      'and drop by listwise deletion once `SPEND_APP_z` enters. `Delta '
      'sigma^2_u0 %` is measured against a single Model-1 baseline on Model 1\'s '
      'own sample, so every column starts from Model 1. `Delta % matched` refits '
      'the Model-1 baseline on the Model-2/3 sample, so the difference between '
      'the two columns is exactly what those three states contributed. Neither '
      'number alone tells the truth; read both.')
    A('')
    # A4: on the M1 row `delta_pct_ri` is 0 by construction (M1 is its own
    # baseline), but `delta_pct_ri_matched` is NOT -- it compares the SAME
    # Model-1 formula fit on two different samples (49 vs 46 states), so
    # whatever value appears there is entirely the sample effect described
    # above, not anything a covariate did (M1 has no covariates at all).
    m1_match_bits = []
    for cell, outcome in CELLS.items():
        m1_row = vt[(vt['cell'] == cell) & (vt['model'] == 'M1')].iloc[0]
        v = m1_row['delta_pct_ri_matched']
        if v == v and v is not None:
            m1_match_bits.append(f'{outcome} {v:+.1f}%')
    A('**On the Model-1 row specifically, `Delta % matched` is not zero even '
      'though Model 1 has no covariates to explain anything** (' +
      '; '.join(m1_match_bits) + '). That is not model improvement -- Model 1 '
      'has only one formula, fit twice, once on 49 states and once on 46. The '
      'entire M1-row value is exactly the sample effect described in the '
      'paragraph above, and nothing else.')
    A('')
    # The direction of the sample effect is NOT the same for both outcomes, so
    # this paragraph is derived per cell rather than asserted. Writing the
    # inspections direction as if it were general would mis-describe the
    # violations column: there the Model-1 basis UNDERstates the covariates.
    A('**And the two columns differ in opposite directions by outcome**, which '
      'is why the generic warning is not written here:')
    A('')
    # B3 fix: the closing sentence used to assert, unconditionally, that
    # "violations are unaffected by [AK/RI/VT's] loss" -- but the very
    # d_base/d_match values computed in THIS loop show the opposite for
    # violations (+19.7% vs +21.5%). That claim belongs to the published
    # log-linear tables, where the violations column never dropped those
    # states at all; it does not describe this arm. Cache both cells'
    # figures here so the closing paragraph can be derived, not asserted.
    sample_effect = {}
    for cell, outcome in CELLS.items():
        base_r = res[f'{cell}__M1__nbinom2__ri']
        matched_r = res[f'{cell}__M1matched__nbinom2__ri']
        m3_r = res[f'{cell}__M3__nbinom2__ri']
        if not (base_r.get('converged') and matched_r.get('converged')
                and m3_r.get('converged')):
            A(f'- **{outcome}:** cannot be derived -- the M1, M1matched or M3 '
              'random-intercept refit did not converge.')
            continue
        base, matched, m3 = (base_r['sigma2_u0'], matched_r['sigma2_u0'],
                              m3_r['sigma2_u0'])
        d_base = 100 * (base - m3) / base
        d_match = 100 * (matched - m3) / matched
        direction = ('lowers' if matched < base else 'raises')
        reading = ('overstates' if d_base > d_match else 'understates')
        sample_effect[cell] = {
            'd_base': d_base, 'd_match': d_match, 'affected': d_base != d_match}
        A(f'- **{outcome}:** dropping AK, RI and VT {direction} the Model-1 '
          f'between-state variance ({base:.4f} on 49 states -> {matched:.4f} on '
          f'46), so the Model-1 basis {reading} what the covariates do: '
          f'Model 3 reduces sigma^2_u0 by {d_base:+.1f}% against Model 1 but '
          f'{d_match:+.1f}% against the matched baseline.')
    A('')
    viol = sample_effect.get('viol_cov_2021')
    if viol is None:
        viol_sentence = ('Whether this arm\'s violations column is affected by '
                          'the same states could not be derived here (see the '
                          'bullet above).')
    elif viol['affected']:
        viol_sentence = (
            'In those published tables the violations column never dropped '
            'AK/RI/VT at all, so it was unaffected by their loss -- that does '
            'NOT carry over to this arm. Here violations does drop those three '
            'states once `SPEND_APP_z` enters, and the reduction changes just '
            f"as inspections' does: {viol['d_base']:+.1f}% against Model 1 "
            f"versus {viol['d_match']:+.1f}% against the matched baseline (the "
            'bullet above).')
    else:
        viol_sentence = ('In this arm the violations column happens to show '
                          'the same reduction against both baselines, so the '
                          'published-tables claim of no effect holds here too.')
    A('The inspections direction reproduces, under a different model class, an '
      'asymmetry this project already documented for the published '
      f'log-linear tables: AK/RI/VT carry much of the between-state '
      f'inspection variance. {viol_sentence}')
    A('')
    A('## Coefficients')
    A('')
    A('`b (SE)` with significance stars, and IRR = exp(b). Stars: '
      '* p<.05, ** p<.01, *** p<.001.')
    A('')
    for cell, outcome in CELLS.items():
        A(f'### {outcome} ({cell})')
        A('')
        sub = ct[(ct['cell'] == cell) & (ct['model'].isin(MODELS))]
        terms = list(dict.fromkeys(sub['term']))
        A('| Term | ' + ' | '.join(f'{m} b (SE)' + ' | ' + f'{m} IRR'
                                   for m in MODELS) + ' |')
        A('|' + '|'.join(['---'] * (1 + 2 * len(MODELS))) + '|')
        for t in terms:
            cells_out = []
            for m in MODELS:
                row = sub[(sub['model'] == m) & (sub['term'] == t)]
                if len(row):
                    r = row.iloc[0]
                    cells_out += [f"{r['b']:.3f} ({r['se']:.3f}){stars(r['p'])}",
                                  f"{r['irr']:.3f}"]
                else:
                    cells_out += ['', '']
            A(f'| {t} | ' + ' | '.join(cells_out) + ' |')
        A('')
    A('## COVID robustness')
    A('')
    A('2020 and 2021 are pandemic years. Each cell\'s Model 3 is refit with a '
      '2020-21 indicator, under NB2 at the same random-effects structure. This '
      'had to be re-derived rather than carried over: the six-family ladder\'s '
      'COVID variants sit under ZINB (inspections) and NB1 (violations).')
    A('')
    A('| Outcome | covid b (SE) | p | IRR | time p, M3 -> M3+covid | '
      'time2 p, M3 -> M3+covid | time3 p, M3 -> M3+covid |')
    A('|---|---|---|---|---|---|---|')
    for cell, outcome in CELLS.items():
        m3_r = res[f'{cell}__M3__nbinom2__rs']
        cv_r = res[f'{cell}__M3covid__nbinom2__rs']
        # M5: guard against a non-converged M3 or M3covid record (R error
        # branch, no 'cond') the same way variance_table()/coefficient_table()
        # already do, instead of indexing 'cond' directly.
        if not (m3_r.get('converged') and cv_r.get('converged')):
            A(f'| {outcome} | NC | NC | NC | NC | NC | NC |')
            continue
        m3, cv = m3_r['cond'], cv_r['cond']
        c = cv['covid']
        A(f"| {outcome} | {c['b']:.3f} ({c['se']:.3f}){stars(c['p'])} | "
          f"{c['p']:.3g} | {math.exp(c['b']):.3f} | " +
          ' | '.join(f"{m3[t]['p']:.3g} -> {cv[t]['p']:.3g}"
                     for t in ('time', 'time2', 'time3')) + ' |')
    A('')
    # Ruling R6. The frozen ladder's COVID variants sit under DIFFERENT
    # families (ZINB for inspections, NB1 for violations), and the comparison
    # splits three ways. Reported as three distinct facts, each derived from
    # the two JSONs rather than asserted, because CLAUDE.md currently records
    # the NB1 coefficient as though it were general.
    A('**How this compares to the ladder\'s own COVID variants, which sit '
      'under different families.** Three separate things are true and they '
      'must not be merged:')
    A('')
    for cell, outcome in CELLS.items():
        old_key, old_fam = COVID_LADDER_REF[cell]
        o_cv = ladder[old_key]['cond']
        o_m3 = ladder[old_key.replace('M3covid', 'M3')]['cond']
        n_cv_r = res[f'{cell}__M3covid__nbinom2__rs']
        n_m3_r = res[f'{cell}__M3__nbinom2__rs']
        # M5: same guard as the table above -- a non-converged M3 or
        # M3covid record has no 'cond' to index.
        if not (n_cv_r.get('converged') and n_m3_r.get('converged')):
            A(f'- **{outcome}:** cannot be derived -- the NB2 Model 3 or '
              'Model 3+COVID refit did not converge.')
            continue
        n_cv, n_m3 = n_cv_r['cond'], n_m3_r['cond']
        A(f'- **{outcome}, the indicator itself:** {old_fam} gives '
          f'b = {o_cv["covid"]["b"]:+.4f} (p = {o_cv["covid"]["p"]:.3g}); NB2 '
          f'gives b = {n_cv["covid"]["b"]:+.4f} '
          f'(p = {n_cv["covid"]["p"]:.3g}).')
        A(f'- **{outcome}, the time trend under the indicator:** `time2` moves '
          f'{o_m3["time2"]["p"]:.3g} -> {o_cv["time2"]["p"]:.3g} under '
          f'{old_fam}, and {n_m3["time2"]["p"]:.3g} -> '
          f'{n_cv["time2"]["p"]:.3g} under NB2.')
        A(pre_covid_time_trend_bullet(outcome, old_fam, o_m3, n_m3))
    A('')
    A('The practical consequence for the violations column: the COVID '
      'coefficient recorded elsewhere in this project is an NB1 estimate and '
      'does not carry over to NB2, while the qualitative caution it supports '
      '-- that `time2` stops being significant once the indicator enters -- '
      'does carry over. Cite the number with its family attached.')
    A('')
    A('## Cross-software check')
    A('')
    A('`statsmodels` has no multilevel negative binomial, so the independent '
      'check fits the same Model-3 fixed effects with **state fixed effects** '
      '(dummies) in place of the random intercept and slope, in a different '
      'language and a different implementation. This is not a numerical '
      'equivalence test and the estimates are not expected to match: a '
      'random-effects estimator shrinks state deviations toward zero and a '
      'dummy-variable estimator does not. What is asserted is sign agreement '
      'among terms both implementations resolve away from zero (|b| > 2 SE in '
      'both). The ratios below are reported for judgement, not thresholded.')
    A('')
    A('**What this check can and cannot cover.** State fixed effects absorb any '
      'regressor that is constant within a state, so the Level-2 covariates are '
      'perfectly collinear with the state dummies and their coefficients are '
      'not identified at all under this estimator -- the design matrix is '
      'rank-deficient by exactly the number of them. They are therefore dropped '
      'from the comparison, by name, below. This is a property of fixed-effects '
      'estimation, not a shortcoming of either implementation, and it means the '
      'cross-check validates the **Level-1 (time-varying) terms only**. The '
      'Level-2 covariates -- the spending, commodity-mix and H-2A-share '
      'variables that Models 2 and 3 are built from -- are NOT independently '
      'confirmed by this check, and no reader should take them as such.')
    A('')
    A('The zero-inflation cross-check in the six-family pipeline has no '
      'counterpart here -- an NB2-only arm has no zero-inflation component.')
    A('')
    for cell, outcome in CELLS.items():
        c = cc[cell]
        A(f'### {outcome} ({cell}) -- N = {c["n_obs"]}, '
          f'{c["n_states"] - 1} state dummies, converged = {c["converged"]}')
        A('')
        absorbed = c.get('terms_absorbed', [])
        if absorbed:
            A(f'Absorbed by the state dummies and excluded from the comparison '
              f'({len(absorbed)}): ' +
              ', '.join(f'`{t}`' for t in absorbed) + '.')
            A('')
        A('| Term | b (statsmodels FE) | b (glmmTMB RE) | ratio | '
          'both distinguishable from 0 | sign |')
        A('|---|---|---|---|---|---|')
        for t, v in c['terms'].items():
            A(f"| {t} | {v['b_sm']:.4f} | {v['b_tmb']:.4f} | "
              f"{v['ratio']:.3f} | {'yes' if v['both_nonzero'] else 'no'} | "
              f"{'agrees' if v['sign_agrees'] else 'DIFFERS'} |")
        A('')
        if c['warnings']:
            A('Warnings raised during the `statsmodels` fit (captured, not '
              'filtered):')
            A('')
            for m in c['warnings']:
                A(f'- {m}')
            A('')
    A('## Caveats')
    A('')
    A('- NB2 is an editorial choice and loses on AIC to the selected family at '
      'every rung. See the first section.')
    A('- Every variance is on the log link scale.')
    A('- Model 1 has 49 states; Models 2 and 3 have 46. Read both reduction '
      'columns.')
    A('- The ICC is a link-scale approximation, not the project\'s Gaussian-LMM '
      'ICC.')
    A('- This arm covers the 2011-2021 WPS-view window only. It does not fit '
      'the 2011-2019 establishments window or the violations-as-offset '
      'specification.')
    A('- No manuscript `.docx` was regenerated.')
    A('')
    with open(DOCS + 'nb2_stepwise_models.md', 'w') as fh:
        fh.write('\n'.join(L) + '\n')


def main():
    res, cc, ladder = load()
    vt = variance_table(res)
    ct = coefficient_table(res)
    vt.to_csv(GEN + 'nb2_stepwise_variance.csv', index=False)
    ct.to_csv(GEN + 'nb2_stepwise_coefficients.csv', index=False)
    print(vt.to_string(index=False))
    print()
    print(ct.to_string(index=False))
    write_memo(res, cc, ladder, vt, ct)
    print(f"\nWrote {GEN}nb2_stepwise_variance.csv, "
          f"{GEN}nb2_stepwise_coefficients.csv, {DOCS}nb2_stepwise_models.md")


if __name__ == '__main__':
    main()
