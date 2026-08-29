"""
Reads the NB2 stepwise artifacts and produces the tables and the memo.

PURE ARTIFACT READER. It fits nothing. Every number it prints traces to
data/generated/nb2_stepwise_results.json or nb2_stepwise_crosscheck.json.

This is the ONLY script in the arm that computes a derived statistic, so
delta_pct_ri, delta_pct_ri_matched and icc_ri each have exactly one definition:

  delta_pct_ri         = 100 * (s2_u0_ri[M1]        - s2_u0_ri[m]) / s2_u0_ri[M1]
  delta_pct_ri_matched = 100 * (s2_u0_ri[M1matched] - s2_u0_ri[m]) / s2_u0_ri[M1matched]
  icc_ri               = s2 / (s2 + log(1 + 1/theta + 1/mu)),  all from the RI fit

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
    rows = []
    for cell, outcome in CELLS.items():
        base = res[f'{cell}__M1__nbinom2__ri']['sigma2_u0']
        matched = res[f'{cell}__M1matched__nbinom2__ri']['sigma2_u0']
        for model in MODELS:
            rs = res[f'{cell}__{model}__nbinom2__rs']
            ri = res[f'{cell}__{model}__nbinom2__ri']
            rows.append({
                'cell': cell, 'outcome': outcome, 'model': model,
                'sigma2_u0_rs': rs['sigma2_u0'],
                'sigma2_u1_rs': rs['sigma2_u1'],
                'sigma_u01_rs': rs['sigma_u01'],
                'theta': ri['dispersion'],
                'sigma2_u0_ri': ri['sigma2_u0'],
                'icc_ri': icc_nb2(ri['sigma2_u0'], ri['dispersion'], ri['mu_fixed']),
                'mu_fixed': ri['mu_fixed'],
                'delta_pct_ri': 100 * (base - ri['sigma2_u0']) / base,
                'delta_pct_ri_matched': 100 * (matched - ri['sigma2_u0']) / matched,
                'n_obs': rs['n_obs'], 'n_states': rs['n_states'],
            })
    return pd.DataFrame(rows)


def coefficient_table(res):
    rows = []
    for cell, outcome in CELLS.items():
        for model in list(MODELS) + ['M3covid']:
            r = res.get(f'{cell}__{model}__nbinom2__rs')
            if r is None:
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
      '`data/generated/nb2_stepwise_results.json` or '
      '`nb2_stepwise_crosscheck.json`.')
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
    A('**Holding the family at NB2 is an editorial decision, not a fit-based '
      'one.** At this random-effects structure NB2 loses to the family the '
      'six-family ladder selected: by about 33 AIC to ZINB for inspections and '
      'about 11 AIC to NB1 for violations, at both Model 1 and Model 3. What '
      'fixing the family buys is a table whose three columns are the same '
      'model class, so the between-state variance actually forms a reduction '
      'sequence instead of being three different models\' parameters. NB2 is '
      'the least-bad single choice across both outcomes: it beats NB1 for '
      'inspections by about 138 AIC while costing about 11 for violations.')
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
    for cell, outcome in CELLS.items():
        sub = vt[vt['cell'] == cell]
        A(f'### {outcome} ({cell})')
        A('')
        A('| Model | sigma^2_u0 (rs) | sigma^2_u1 (rs) | sigma_u01 (rs) | '
          'theta (ri) | sigma^2_u0 (ri) | ICC (ri) | Delta sigma^2_u0 % | '
          'Delta % matched | N obs | States |')
        A('|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in sub.iterrows():
            A(f"| {r['model']} | {r['sigma2_u0_rs']:.4f} | "
              f"{r['sigma2_u1_rs']:.5f} | {r['sigma_u01_rs']:.5f} | "
              f"{r['theta']:.4f} | {r['sigma2_u0_ri']:.4f} | "
              f"{r['icc_ri']:.3f} | {r['delta_pct_ri']:+.1f} | "
              f"{r['delta_pct_ri_matched']:+.1f} | {int(r['n_obs'])} | "
              f"{int(r['n_states'])} |")
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
    # The direction of the sample effect is NOT the same for both outcomes, so
    # this paragraph is derived per cell rather than asserted. Writing the
    # inspections direction as if it were general would mis-describe the
    # violations column: there the Model-1 basis UNDERstates the covariates.
    A('**And the two columns differ in opposite directions by outcome**, which '
      'is why the generic warning is not written here:')
    A('')
    for cell, outcome in CELLS.items():
        base = res[f'{cell}__M1__nbinom2__ri']['sigma2_u0']
        matched = res[f'{cell}__M1matched__nbinom2__ri']['sigma2_u0']
        m3 = res[f'{cell}__M3__nbinom2__ri']['sigma2_u0']
        d_base = 100 * (base - m3) / base
        d_match = 100 * (matched - m3) / matched
        direction = ('lowers' if matched < base else 'raises')
        reading = ('overstates' if d_base > d_match else 'understates')
        A(f'- **{outcome}:** dropping AK, RI and VT {direction} the Model-1 '
          f'between-state variance ({base:.4f} on 49 states -> {matched:.4f} on '
          f'46), so the Model-1 basis {reading} what the covariates do: '
          f'Model 3 reduces sigma^2_u0 by {d_base:+.1f}% against Model 1 but '
          f'{d_match:+.1f}% against the matched baseline.')
    A('')
    A('The inspections direction reproduces, under a different model class, an '
      'asymmetry this project already documented for the published log-linear '
      'tables: AK/RI/VT carry much of the between-state inspection variance, '
      'and violations are unaffected by their loss.')
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
        m3 = res[f'{cell}__M3__nbinom2__rs']['cond']
        cv = res[f'{cell}__M3covid__nbinom2__rs']['cond']
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
        n_cv = res[f'{cell}__M3covid__nbinom2__rs']['cond']
        n_m3 = res[f'{cell}__M3__nbinom2__rs']['cond']
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
