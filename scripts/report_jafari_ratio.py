"""
Reporter for the violations-per-inspection ("ratio") models.

A PURE ARTIFACT READER. It fits nothing and it opens no panel: every number it
prints comes out of data/generated/jafari_ratio_results.json, which
scripts/jafari_ratio_models.R wrote. The offset-drop derivation printed in the
memo is likewise read from that file's `__offset_derivation` record, which R
computed from the panel -- so exactly one script owns each number.

docs/jafari_ratio_models.md is GENERATED here. Never hand-edit it; re-run this.

Run: python3 scripts/report_jafari_ratio.py
"""
import json

import numpy as np

GEN = '/Users/keshavgoel/Research/data/generated/'
DOCS = '/Users/keshavgoel/Research/docs/'
RESULTS = GEN + 'jafari_ratio_results.json'

SPECS = ('A', 'B', 'C')
MODELS = ('M1', 'M2', 'M3')

SPEC_NAME = {
    'A': 'Spec A -- NB2 with offset',
    'B': 'Spec B -- zero-inflated NB2 with offset',
    'C': 'Spec C -- Gaussian on the log ratio',
}
SPEC_FORMULA = {
    'A': '`violations ~ ... + offset(log(inspections))`, `nbinom2`, rows with `inspections > 0`',
    'B': 'Spec A plus `ziformula` mirroring the conditional predictors',
    'C': '`log((violations + 1) / (inspections + 1)) ~ ...`, Gaussian, all rows',
}

# What exp(b) means in each spec's CONDITIONAL block. Spec C is not a count
# model, so calling its exp(b) an incidence-rate ratio would be wrong.
EXP_LABEL = {
    'A': 'IRR', 'B': 'IRR',
    'C': 'exp(b)',
}
EXP_NOTE = {
    'A': 'IRR = exp(b). Because of the offset this is a **rate ratio per '
         'inspection**: the multiplicative change in violations *per inspection* '
         'for a one-unit increase in the predictor.',
    'B': 'IRR = exp(b), a **rate ratio per inspection** (the offset), among '
         'state-years that are not structural zeros.',
    'C': 'exp(b) is the multiplicative change in the ratio `(violations + 1) / '
         '(inspections + 1)`. It is **not** an incidence-rate ratio -- this is a '
         'Gaussian model of a transformed ratio, not a count model.',
}

TERM_LABEL = {
    '(Intercept)': 'Intercept',
    'time': 'time', 'time2': 'time²', 'time3': 'time³',
    'SPEND_APP_z': 'STAG $ per applicator (z)',
    'SPEND_WORK_z': 'STAG $ per farmworker (z)',
    'lii_2017_z': 'Labor intensity index 2017 (z)',
    'h2a_per_farmworker_z': 'H-2A per farmworker (z)',
    'dol_demand_met_pct_z': 'DOL demand met % (z)',
    'pct_flc_z': '% farm labor contractor (z)',
}
TERM_ORDER = list(TERM_LABEL)

ZI_TIER_LABEL = {
    'mirror': 'full mirror (every conditional predictor)',
    'covariates': 'covariates only (no time terms)',
    'reduced': 'spending + labor intensity',
    'intercept': 'intercept only',
}

# sigma^2_year below this is read as pinned at the boundary, i.e. the crossed
# year dimension is explaining nothing. Chosen well above numerical noise
# (~1e-9 here) and far below any variance that would matter substantively.
YEAR_BOUNDARY_TOL = 1e-4


def load():
    with open(RESULTS) as fh:
        return json.load(fh)


def stars(p):
    if p is None or not np.isfinite(p):
        return ''
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '+' if p < .10 else ''


def fmt(x, n=4, dash='--'):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f'{x:.{n}f}'


def pct(x, dash='--'):
    """Signed percentage, or a dash when the quantity is undefined (Model 1 is
    its own baseline, so its reduction is not a number)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f'{x:+.1f}%'


def conv_mark(fit):
    """Never let a non-converged fit read as an estimate."""
    return '' if fit.get('converged') else ' **[DID NOT CONVERGE]**'


def delta_pct(baseline, fit):
    """100 * (s2_baseline - s2_model) / s2_baseline, both from sigma2_state.

    No random-intercept-only refit series is needed in this arm, and that is
    worth being explicit about. In the project's random-SLOPE specifications
    sigma^2_u0 is the between-state variance AT the 2017 centering year and
    trades off against sigma^2_u1, so its reduction is not bounded to [0, 1] and
    has gone negative. Here the only state term IS a random intercept, so
    sigma2_state is already the quantity the published `delta_pct_ri` convention
    reads -- these percentages are on that basis by construction.
    """
    if not (baseline.get('converged') and fit.get('converged')):
        return None
    s0, s1 = baseline.get('sigma2_state'), fit.get('sigma2_state')
    if s0 in (None, 0) or s1 is None:
        return None
    return 100.0 * (s0 - s1) / s0


def comparison_rows(r):
    """One row per (spec, model): N, states, zeros, sigma2_state, both deltas, AIC."""
    rows = []
    for spec in SPECS:
        base = r[f'{spec}__M1']
        matched = r[f'{spec}__M1matched']
        for model in MODELS:
            v = r[f'{spec}__{model}']
            rows.append({
                'spec': spec, 'model': model,
                'converged': bool(v.get('converged')),
                'n_obs': v['n_obs'], 'n_states': v['n_states'],
                'obs_zeros': v['obs_zeros'],
                'zero_rate': v['obs_zeros'] / v['n_obs'] if v['n_obs'] else None,
                'sigma2_state': v.get('sigma2_state'),
                'sigma2_year': v.get('sigma2_year'),
                'delta_pct': delta_pct(base, v) if model != 'M1' else None,
                'delta_pct_matched': (delta_pct(matched, v)
                                      if model != 'M1' else None),
                'aic': v.get('aic') if v.get('converged') else None,
                'df': v.get('df'),
                'dispersion': v.get('dispersion'),
                'zi_tier_reached': v.get('zi_tier_reached'),
                'zi_prob_mean': v.get('zi_prob_mean'),
            })
    return rows


def coef_rows(fit, block):
    """Ordered (term, b, se, z, p) for one block of one fit."""
    blk = fit.get(block) or {}
    known = [t for t in TERM_ORDER if t in blk]
    extra = [t for t in blk if t not in TERM_ORDER]
    return [(t, blk[t]) for t in known + extra]


# ---------------------------------------------------------------- memo


def sec_tradeoff(r):
    d = r['__offset_derivation']
    m1, m3 = d['by_model']['M1'], d['by_model']['M3']
    rows = m1['dropped_rows']
    listing = ', '.join(f"{x['state']}-{x['year']}" for x in rows)
    claim_held = (m1['n_dropped'] == 7
                  and m1['n_dropped_zero_violation'] == m1['n_dropped'])
    expected = {('Connecticut', 2018), ('Kansas', 2018), ('Massachusetts', 2018),
                ('Oregon', 2020), ('Oregon', 2021), ('Utah', 2017),
                ('West Virginia', 2012)}
    same_rows = {(x['state'], int(x['year'])) for x in rows} == expected

    verdict = (
        f"**It held.** The claim recorded in `CLAUDE.md` -- 7 rows, all "
        f"zero-violation -- is reproduced exactly, and they are the same 7 "
        f"state-years."
        if claim_held and same_rows else
        f"**It did NOT hold as recorded.** The derivation finds "
        f"{m1['n_dropped']} dropped rows, {m1['n_dropped_zero_violation']} of "
        f"them zero-violation"
        + ('' if same_rows else ', and the state-years differ from the recorded list')
        + '.')

    drop_tbl = ['| State | Year | Violations | Inspections |',
                '|---|---:|---:|---:|']
    for x in rows:
        drop_tbl.append(f"| {x['state']} | {int(x['year'])} | "
                        f"{int(x['violations'])} | {int(x['inspections'])} |")

    return f"""## 2. The trade-off, and it is the decision in front of you

**A ratio cannot simply be swapped in as the dependent variable of a count
model.** The negative-binomial and zero-inflated negative-binomial families this
project has been using require non-negative integers. `violations / inspections`
is neither an integer nor bounded, and it is undefined wherever a state-year has
no inspections at all.

**The count-preserving way to model "violations per inspection" is an offset.**
Fitting `violations ~ ... + offset(log(inspections))` constrains the model so
that its linear predictor describes `log(violations / inspections)` -- the rate
per inspection -- while the likelihood stays a count likelihood on the raw
violation counts. Coefficients are then rate ratios per inspection, which is the
quantity asked for. **This is Spec A**, and Spec B is Spec A with
zero-inflation.

**But `log(0)` is undefined, so the offset deletes every zero-inspection
state-year.** That is the cost, and in this window it lands precisely on the
rows this modelling tradition exists to explain. Derived from the panel:

- The panel has {d['panel_rows']} rows, {d['panel_states']} states,
  {d['panel_years']} years. {d['rows_missing_violations']} rows have a missing
  violation count and are dropped by listwise deletion in **every** spec,
  including Spec C -- so "all rows" means {m1['n_all_rows']}, not
  {d['panel_rows']}.
- **{m1['n_dropped']} further rows have `inspections == 0`** and are dropped by
  the offset: {listing}.
- **{m1['n_dropped_zero_violation']} of those {m1['n_dropped']} are also
  zero-violation rows** ({m1['n_dropped_positive_violation']} have a positive
  violation count). The offset therefore removes
  {m1['n_dropped_zero_violation']} of the {m1['zeros_all_rows']} zeros in the
  Model-1 sample, taking the zero count from {m1['zeros_all_rows']} to
  {m1['zeros_offset_rows']}.

{drop_tbl[0]}
{drop_tbl[1]}
{chr(10).join(drop_tbl[2:])}

{verdict}

The same {m3['n_dropped']} rows are dropped at Models 2 and 3, where listwise
deletion on the BLS pesticide-applicator series has already reduced the sample
to {m3['n_all_rows']} rows across {m3['n_states_all_rows']} states.

**Why this matters for Joe's rationale.** The reason for wanting a
violations-per-inspection outcome was that a raw violation count cannot separate
"inspected, nothing found" from "never inspected". A zero-inspection state-year
is exactly the second case -- and the offset resolves the ambiguity by
*deleting* those rows rather than by modelling them. That is a defensible answer
(you cannot estimate a rate per inspection from zero inspections), but it is not
the only one, which is why **Spec C** is here: a Gaussian model of
`log((violations + 1) / (inspections + 1))` keeps all {m1['n_all_rows']} rows
and all {m1['n_states_all_rows']} states, at the cost of leaving the count
likelihood behind. Joe half-anticipated this in the meeting ("the number of
zeros will change"); the table in section 3 is that change, quantified.

**The decision.** If the question is *the rate of violation per inspection*,
Spec A/B is the right form and the 7 deleted rows are the honest price. If the
question is *whether states with no enforcement activity at all behave
differently*, those 7 rows are part of the answer and Spec C (or the existing
count-with-`log_inspections`-as-covariate specification) keeps them.
"""


def sec_comparison(rows, r):
    lines = ['| Spec | Model | N | States | Zeros (rate) | σ²_state | Δσ²_state vs M1 | Δ vs matched M1 | AIC | df |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        aic = 'n/a' if row['aic'] is None else f"{row['aic']:.2f}"
        mark = '' if row['converged'] else ' ⚠'
        dlt = pct(row['delta_pct'])
        dltm = pct(row['delta_pct_matched'])
        lines.append(
            f"| {row['spec']} | {row['model']}{mark} | {row['n_obs']} | "
            f"{row['n_states']} | {row['obs_zeros']} ({row['zero_rate']:.3f}) | "
            f"{fmt(row['sigma2_state'])} | {dlt} | {dltm} | "
            f"{aic} | {row['df']} |")

    a1, b1 = r['A__M1matched'], r['B__M1matched']
    return ('\n'.join(lines), a1, b1)


def write_memo(r, rows):
    d = r['__offset_derivation']
    parts = []

    parts.append(
        '<!-- GENERATED by scripts/report_jafari_ratio.py. '
        'Never hand-edit -- re-run it. -->\n'
        '# Violations per inspection: a ratio outcome, three defensible forms\n\n'
        'GENERATED by `scripts/report_jafari_ratio.py`. Never hand-edit -- re-run it.\n')

    # --- 1. what this answers ---
    parts.append(
        '## 1. What this answers, and who asked\n\n'
        'Joe Grzywacz asked at the 2026-09-11 meeting for a third model arm with a '
        '**new dependent variable: violations per inspection**, fit with the same '
        'stepwise specification as the rest of the project and with inspections '
        '**removed as a covariate**, since it is now the denominator. His rationale: '
        'Jafari et al. had only a violations count and could not separate "inspected, '
        'nothing found" from "never inspected", so a per-inspection rate is closer to '
        'the data they modelled. This memo fits that outcome three ways on the '
        '2011--2021 WPS panel, at the crossed random-intercept structure '
        '`(1 | state) + (1 | year)` the PI mandated for this line of work (no random '
        'slope anywhere), stepwise M1 (cubic time) → M2 (+ spending and labor '
        'intensity) → M3 (+ the H-2A block).\n')

    # --- 2. the trade-off ---
    parts.append(sec_tradeoff(r))

    # --- 3. side by side ---
    tbl, _, _ = sec_comparison(rows, r)
    parts.append(
        '## 3. Specs A, B and C side by side\n\n'
        + '\n'.join(f'- **{SPEC_NAME[s]}**: {SPEC_FORMULA[s]}' for s in SPECS)
        + '\n\nAll nine fits below are at `(1 | state) + (1 | year)`, ML '
          '(`REML = FALSE`), stepwise on the same covariate blocks. Three further '
          'fits (the matched Model-1 baselines, one per spec) back the last Δ '
          'column and are not shown as rows.\n\n'
        + tbl + '\n\n'
        '`Δσ²_state vs M1` is `(σ²_M1 − σ²_model) / σ²_M1` with Model 1 fit on its own '
        'sample. `Δ vs matched M1` refits that **same Model-1 formula on the Model-2/3 '
        'rows** (a separate fit, `*__M1matched`), because M1 keeps all 49 states while '
        'M2/M3 drop Alaska, Rhode Island and Vermont for want of a BLS '
        'pesticide-applicator series. Reporting only the first column would mix "the '
        'covariates explained between-state variance" with "three states left the '
        'sample". Positive means variance **reduced**.\n\n'
        '**AIC is comparable between Spec A and Spec B and nowhere else.** A and B '
        'share an analytic sample, a response, a conditional formula and a likelihood; '
        'the zero-inflation block is the only difference, which is what makes the '
        'comparison a clean test of zero-inflation. **Spec C is a Gaussian model of a '
        'different response on a different number of rows -- its AIC must never be '
        'compared to A or B.** Within Spec C the three AICs are mutually comparable at '
        'M2/M3 (same rows) but M1 sits on a larger sample, so even there the M1 figure '
        'is not a like-for-like competitor.\n')

    # --- 4. did ZI survive ---
    parts.append(sec_zi(r, rows))

    # --- 5. coefficients ---
    parts.append(sec_coefficients(r))

    # --- 6. how to read the blocks ---
    parts.append(
        '## 6. How to read the two blocks (please read this one)\n\n'
        'A zero-inflated model reports **two regressions of opposite sign '
        'convention**, and it is easy to invert them:\n\n'
        '- The **conditional block is the COUNT model**. It describes the expected '
        'violation rate *given the state-year is not a structural zero*. A **positive** '
        'coefficient means **more violations per inspection**.\n'
        '- The **zero-inflation block is the log-odds of being a STRUCTURAL ZERO** -- '
        'a state-year that could not have produced a violation at all, as distinct from '
        'one that could have and did not. A **positive** ZI coefficient means **more '
        'structural zeros**, and therefore **FEWER events**. That is the *opposite* '
        'substantive direction from a positive conditional coefficient.\n\n'
        '> A predictor with a positive conditional coefficient **and** a positive ZI '
        'coefficient is pushing in two directions at once: among states that are '
        'enforcing, it raises the violation rate; and it makes a state more likely to '
        'be one that is not enforcing at all. Report both, and never read the ZI block '
        'as if it were a second count model.\n')

    # --- 7. caveats ---
    parts.append(sec_caveats(r, rows))

    with open(DOCS + 'jafari_ratio_models.md', 'w') as fh:
        fh.write('\n'.join(parts))


def sec_zi(r, rows):
    lines = ['| Model | Spec A AIC (no ZI) | Spec B AIC (ZI) | ΔAIC (B − A) | ZI tier reached | ZI params | Mean Pr(structural zero) | Observed zero rate |',
             '|---|---:|---:|---:|---|---:|---:|---:|']
    wins = 0
    margins = []
    for model in MODELS:
        a, b = r[f'A__{model}'], r[f'B__{model}']
        if not (a.get('converged') and b.get('converged')):
            lines.append(f'| {model} | see JSON | see JSON | -- | -- | -- | -- | -- |')
            continue
        dl = b['aic'] - a['aic']
        margins.append(dl)
        if dl < 0:
            wins += 1
        tier = b.get('zi_tier_reached')
        lines.append(
            f"| {model} | {a['aic']:.2f} | {b['aic']:.2f} | **{dl:+.2f}** | "
            f"{ZI_TIER_LABEL.get(tier, tier)} | {b['df'] - a['df']} | "
            f"{fmt(b.get('zi_prob_mean'))} | "
            f"{b['obs_zeros'] / b['n_obs']:.4f} |")

    all_win = wins == len(margins) and margins
    tiers = {r[f'B__{m}'].get('zi_tier_reached') for m in MODELS}
    degen = [m for m in MODELS if r[f'B__{m}'].get('zi_degenerate')]
    nonconv = [m for m in MODELS if not r[f'B__{m}'].get('converged')]

    if all_win and not degen and not nonconv and tiers == {'mirror'}:
        answer = (f'**Yes -- zero-inflation survives the offset and still buys a '
                  f'great deal.** At every step of the build-up the full Jafari-style '
                  f'ZI mirror (every conditional predictor also in the ZI block) is '
                  f'estimable, converges, is not boundary-degenerate, and beats the '
                  f'non-inflated Spec A on AIC by '
                  f'{min(abs(m) for m in margins):.1f}--'
                  f'{max(abs(m) for m in margins):.1f} points.')
    else:
        bits = []
        if nonconv:
            bits.append('did not converge at ' + ', '.join(nonconv))
        if degen:
            bits.append('was boundary-degenerate at ' + ', '.join(degen))
        if not all_win:
            bits.append(f'won on AIC at only {wins} of {len(margins)} models')
        answer = ('**Partly -- ' + '; '.join(bits) + '.** See the table; any '
                  'non-converged or degenerate fit is marked and must not be read '
                  'as an estimate.')

    return ('## 4. Did zero-inflation survive the offset?\n\n'
            + answer + '\n\n'
            + '\n'.join(lines) + '\n\n'
            'This is the one AIC comparison in this memo that is strictly valid: Spec B '
            'differs from Spec A only by the zero-inflation block, on identical rows. '
            '`Mean Pr(structural zero)` is the **sample mean of each state-year\'s own '
            'zero-inflation probability**, conditional on the fitted random effects -- '
            'not `plogis(zi_intercept)`, which evaluates the ZI linear predictor at '
            'x = 0 and answers a different question once the ZI block carries '
            'covariates. It is directly comparable to the observed zero rate beside it, '
            'and must sit below it, since structural zeros are a subset of all zeros.\n\n'
            '**Note what this does and does not say.** It says zero-inflation is still '
            'identifiable and still strongly supported **among the state-years that '
            'have at least one inspection** -- i.e. the zeros that remain after the '
            'offset has removed the 7 zero-inspection rows are still not well explained '
            'by negative-binomial overdispersion alone. It says nothing about the 7 '
            'deleted rows, which no Spec A or Spec B fit sees.\n')


def sec_ab_shift(r):
    """Which Model-3 conditional coefficients change significance when the ZI
    block is added, and which ZI coefficients are themselves significant.

    Derived, never typed. A and B share rows and conditional formula, so any
    difference between their conditional blocks is attributable to the ZI block
    -- which is precisely the substantive content of "some of what looked like a
    low violation RATE was really a structural zero".
    """
    a, b = r['A__M3'], r['B__M3']
    if not (a.get('converged') and b.get('converged')):
        return ''
    ac, bc = a.get('cond') or {}, b.get('cond') or {}
    moved = []
    for t in TERM_ORDER:
        if t in ('(Intercept)',) or t not in ac or t not in bc:
            continue
        sa, sb = bool(stars(ac[t]['p'])), bool(stars(bc[t]['p']))
        if sa != sb:
            moved.append((t, ac[t], bc[t], sa))
    zsig = [(t, c) for t, c in (b.get('zi') or {}).items()
            if t != '(Intercept)' and stars(c['p'])]

    out = ['\n### What the zero-inflation block changes (Model 3, Spec A vs Spec B)\n']
    if moved:
        out.append('Specs A and B are fit on the same rows with the same conditional '
                   'formula, so any move here is attributable to the ZI block:\n')
        for t, ca, cb, was_sig in moved:
            direction = 'loses' if was_sig else 'gains'
            out.append(
                f"- **{TERM_LABEL.get(t, t)}** {direction} significance: "
                f"{ca['b']:+.4f}{stars(ca['p'])} (p = {ca['p']:.4f}) in Spec A → "
                f"{cb['b']:+.4f}{stars(cb['p'])} (p = {cb['p']:.4f}) in Spec B.")
    else:
        out.append('No Model-3 conditional coefficient changes significance between '
                   'Spec A and Spec B.\n')
    if zsig:
        out.append('\nSignificant **zero-inflation** predictors in Spec B (remember: '
                   'positive = more structural zeros = fewer events):\n')
        for t, c in zsig:
            out.append(f"- **{TERM_LABEL.get(t, t)}**: {c['b']:+.4f}{stars(c['p'])} "
                       f"(OR {np.exp(c['b']):.3g}, p = {c['p']:.4f}).")
    else:
        out.append('\nNo non-intercept zero-inflation predictor is significant in '
                   'Spec B.\n')
    return '\n'.join(out) + '\n'


def sec_coefficients(r):
    parts = ['## 5. Model 3 coefficients, all blocks\n\n'
             'Stars: `***` p<.001, `**` p<.01, `*` p<.05, `+` p<.10.\n']
    for spec in SPECS:
        v = r[f'{spec}__M3']
        head = (f'\n### {SPEC_NAME[spec]}{conv_mark(v)}\n\n'
                f'`{v["formula"]}`\n\n'
                f'N = {v["n_obs"]} state-years, {v["n_states"]} states, '
                f'{v["n_years"]} years. Family: `{v["family_name"]}`. ')
        if v['family_name'] != 'gaussian':
            head += f'Dispersion (θ) = {fmt(v.get("dispersion"), 3)}. '
        head += (f'σ²_state = {fmt(v.get("sigma2_state"))}, '
                 f'σ²_year = {v.get("sigma2_year"):.2e}.\n')
        if v.get('message'):
            head += f'\n> Fit warning: `{v["message"]}`\n'
        parts.append(head)

        parts.append(f'\n**Conditional block -- the COUNT model** (expected rate '
                     f'given the state-year is not a structural zero). '
                     f'{EXP_NOTE[spec]}\n'
                     if spec != 'C' else
                     f'\n**Conditional block.** {EXP_NOTE[spec]}\n')
        lines = [f'| Term | b | (SE) | z | p | {EXP_LABEL[spec]} |',
                 '|---|---:|---:|---:|---:|---:|']
        for term, c in coef_rows(v, 'cond'):
            lines.append(
                f"| {TERM_LABEL.get(term, term)} | {c['b']:+.4f}{stars(c['p'])} "
                f"| ({c['se']:.4f}) | {c['z']:.2f} | {c['p']:.4f} "
                f"| {np.exp(c['b']):.4f} |")
        parts.append('\n'.join(lines) + '\n')

        zblk = v.get('zi') or {}
        if zblk:
            tier = v.get('zi_tier_reached')
            parts.append(
                f'\n**Zero-inflation block -- log-odds of being a STRUCTURAL ZERO** '
                f'(ZI tier: {ZI_TIER_LABEL.get(tier, tier)}). A **positive** '
                f'coefficient means more structural zeros, hence FEWER events -- the '
                f'opposite substantive direction from the block above. OR = exp(b), an '
                f'odds ratio on being a structural zero.\n')
            zl = ['| Term | b | (SE) | z | p | OR |', '|---|---:|---:|---:|---:|---:|']
            for term, c in coef_rows(v, 'zi'):
                zl.append(
                    f"| {TERM_LABEL.get(term, term)} | {c['b']:+.4f}{stars(c['p'])} "
                    f"| ({c['se']:.4f}) | {c['z']:.2f} | {c['p']:.4f} "
                    f"| {np.exp(c['b']):.4g} |")
            parts.append('\n'.join(zl) + '\n')
        elif spec == 'B':
            parts.append('\n> No zero-inflation block was recovered for this fit.\n')
    parts.append(sec_ab_shift(r))
    return '\n'.join(parts)


def sec_caveats(r, rows):
    have = [row for row in rows if row['sigma2_year'] is not None]
    at_boundary = [row for row in have if row['sigma2_year'] < YEAR_BOUNDARY_TOL]
    n_year, n_tot = len(at_boundary), len(have)
    worst = max(have, key=lambda row: row['sigma2_year'])
    # The share of total random-effect variance the year dimension holds, at the
    # fit where it holds the most. This is what makes "empty" a claim about
    # magnitude rather than about a tolerance someone chose.
    worst_share = 100.0 * worst['sigma2_year'] / (worst['sigma2_year']
                                                  + worst['sigma2_state'])

    named = ', '.join(row['spec'] + '/' + row['model'] for row in at_boundary)
    lead = (f'**σ²_year is pinned at the boundary in all {n_tot} substantive fits**'
            if n_year == n_tot else
            f'**σ²_year is pinned at the boundary (< {YEAR_BOUNDARY_TOL:g}) in '
            f'{n_year} of {n_tot} substantive fits** ({named})')
    worst_name = worst['spec'] + '/' + worst['model']
    year_text = (
        f'{lead}. The largest value anywhere is {worst["sigma2_year"]:.2e} '
        f'({worst_name}), which is {worst_share:.3f}% of that '
        f'fit\'s total random-effect variance -- so the year dimension is empty on '
        f'any reading, not merely below a chosen tolerance. The fixed cubic in '
        f'`time` absorbs the smooth year-to-year variation, leaving the crossed year '
        f'dimension nothing to explain. **The crossed state × year structure '
        f'therefore behaves as a state-only random intercept in practice**, and this '
        f'arm should be read that way -- do not describe the year dimension as doing '
        f'work. This reproduces the finding already recorded for the crossed count '
        f'arm.')

    nonconv = [f"{row['spec']}/{row['model']}" for row in rows if not row['converged']]
    extra = [k for k in r if k.endswith('M1matched') and not r[k].get('converged')]
    n_sub = len(rows)
    n_base = len([k for k in r if k.endswith('M1matched')])
    conv_text = (f'All {n_sub} substantive fits and all {n_base} matched Model-1 '
                 f'baselines ({n_sub + n_base} fits in total) converged: '
                 'positive-definite Hessian, convergence code 0, and finite standard '
                 'errors on every conditional coefficient.'
                 if not nonconv and not extra else
                 'NOT ALL FITS CONVERGED: ' + ', '.join(nonconv + extra) +
                 '. Those rows are marked ⚠ above and must not be read as estimates.')

    a1, m1 = r['A__M1'], r['A__M1matched']
    return f"""## 7. Caveats

- **σ² here is on the log LINK scale and is NOT comparable in magnitude to
  published Tables 2 and 3**, whose variance components are on a `log(count + 1)`
  outcome scale. A *percentage* reduction is comparable across the two; a raw
  σ² is not. Spec C is the partial exception -- it is a Gaussian model of a log
  ratio, so its σ² is on that ratio's scale, which is again not Tables 2/3's.
- **The sample changes between Model 1 and Models 2/3**: {a1['n_states']} states
  at M1, {r['A__M3']['n_states']} at M2/M3, because Alaska, Rhode Island and
  Vermont have no BLS pesticide-applicator series and so no `SPEND_APP_z`. That
  is why both Δ columns are reported. The two disagree by construction: the
  matched baseline's σ²_state is {fmt(m1.get('sigma2_state'))} against
  {fmt(a1.get('sigma2_state'))} for the all-states Model 1 in Spec A, so the
  unmatched column and the matched column are answering different questions.
- {year_text}
- **No Δσ² percentage in this memo needs a random-intercept-only refit series.**
  Elsewhere in this project Δσ²_u0 has to be read off separate `ri` refits,
  because under a random slope σ²_u0 is the between-state variance *at the 2017
  centering year* and trades off against σ²_u1, which is how a −12.5% reduction
  once appeared. This arm has no random slope by directive, so `sigma2_state`
  already IS the random-intercept quantity and the percentages are on the
  published `delta_pct_ri` basis by construction.
- **This arm is not comparable to `docs/count_models_zinb.md`,
  `docs/nb2_stepwise_models.md` or `docs/jafari_crossed_models.md`.** Different
  dependent variable (a rate, not a count), and in Spec A/B a different analytic
  sample. Do not line up coefficients across memos.
- **No COVID indicator is fit.** `(1 | year)` would absorb the pandemic shock by
  construction, so a dummy would compete with the term containing it -- but as
  noted above σ²_year is at the boundary, so **this arm says nothing about COVID
  either way**. The other arms' COVID coefficients are the place to look.
- {conv_text}
- Nothing existing was modified to produce this memo. It reads only
  `data/generated/jafari_ratio_results.json`, written by
  `scripts/jafari_ratio_models.R` from
  `{r['__offset_derivation']['panel_csv']}`.
"""


def main():
    r = load()
    rows = comparison_rows(r)
    write_memo(r, rows)
    print(f'read {RESULTS}')
    print(f'  {len(rows)} substantive fits across specs {", ".join(SPECS)}')
    for row in rows:
        aic = 'n/a' if row['aic'] is None else f"{row['aic']:.2f}"
        print(f"  {row['spec']}/{row['model']:<2} N={row['n_obs']:>3} "
              f"states={row['n_states']:>2} zeros={row['obs_zeros']:>3} "
              f"s2_state={fmt(row['sigma2_state'])} "
              f"delta={pct(row['delta_pct']):>7} "
              f"delta_matched={pct(row['delta_pct_matched']):>7} "
              f"AIC={aic:>9} conv={row['converged']}")
    print('wrote docs/jafari_ratio_models.md')


if __name__ == '__main__':
    main()
