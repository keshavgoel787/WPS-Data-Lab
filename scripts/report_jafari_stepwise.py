"""
Reads data/generated/jafari_stepwise_results.json and writes
docs/jafari_stepwise_variance.md.

PURE ARTIFACT READER. It fits nothing. Every number in the memo traces to that
one JSON, which is produced by scripts/jafari_stepwise_models.R.

This is the ONLY script in this arm that REPORTS a derived quantity. The three
it owns are:

  delta_pct         = 100 * (s2_state[M1]        - s2_state[m]) / s2_state[M1]
  delta_pct_matched = 100 * (s2_state[M1matched] - s2_state[m]) / s2_state[M1matched]
  IRR / OR          = exp(b)
  absorbed share    = 100 * (s2_state[without] - s2_state[with]) / s2_state[without]

The absorbed share (section 4) is reported on THREE bases: the two violations
cells as each is actually selected (which is confounded whenever their families
differ), and once for each family with BOTH cells held at it. Which bases exist
is derived from the artifact at run time -- nothing here hardcodes which family
a cell carries -- and if a matched-family companion fit is missing or
non-converged the memo falls back to the upper-bound framing and says which
direction is unavailable.

`delta_pct` measures every model against Model 1 fit on Model 1's OWN 49-state
sample; `delta_pct_matched` against Model 1 refit on the Model 2/3 46-state
sample. Both are reported because neither alone tells the truth: M2/M3 drop
AK/RI/VT (no BLS pesticide-applicator series), so an unmatched reduction mixes
"the covariates explained variance" with "three states left the sample".

Run: python3 scripts/report_jafari_stepwise.py
"""
import json
import math

GEN = '/Users/keshavgoel/Research/data/generated/'
DOCS = '/Users/keshavgoel/Research/docs/'
RESULTS = GEN + 'jafari_stepwise_results.json'
MEMO = DOCS + 'jafari_stepwise_variance.md'

MODELS = ('M1', 'M1matched', 'M2', 'M3')
SUBSTANTIVE = ('M1', 'M2', 'M3')

MODEL_LABEL = {
    'M1': 'M1 (cubic time only)',
    'M1matched': 'M1 refit on the M2/M3 sample (matched baseline)',
    'M2': 'M2 (+ spending, commodity mix)',
    'M3': 'M3 (+ H-2A block)',
}

CELL_LABEL = {
    'insp_2021': 'Inspections',
    'viol_2021_wi': 'Violations, WITH log(inspections)',
    'viol_2021_ni': 'Violations, WITHOUT inspections',
}

FAMILY_LABEL = {
    'poisson': 'Poisson', 'nbinom1': 'NB1', 'nbinom2': 'NB2',
    'zip': 'ZIP', 'zinb1': 'ZINB1', 'zinb2': 'ZINB2',
    'zinb1_re': 'ZINB1 + ZI-RE', 'zinb2_re': 'ZINB2 + ZI-RE',
}

TERM_ORDER = ['(Intercept)', 'log_inspections', 'time', 'time2', 'time3',
              'SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
              'h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']

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


def stars(p):
    if p is None or p != p:
        return ''
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


def fmt(v, spec='.4f', na='--'):
    if v is None or (isinstance(v, float) and v != v):
        return na
    return format(v, spec)


def load():
    with open(RESULTS) as fh:
        return json.load(fh)


def cells_of(res):
    """Cell order is taken from the JSON's own `__meta` records, in insertion
    order, so the memo can never disagree with what was actually fit."""
    return [k[:-len('__meta')] for k in res
            if k.endswith('__meta') and not k.startswith('__')]


def key_for(res, cell, model, series):
    fam = res[f'{cell}__meta']['family_tag']
    return f'{cell}__{model}__{fam}__{series}'


def get(res, cell, model, series):
    return res.get(key_for(res, cell, model, series))


def variance_rows(res, cell, series):
    """One row per model, with both delta bases.

    A non-converged fit lands in the R error branch carrying only
    `converged`/`message` -- no `sigma2_state`. Every field access is therefore
    gated on `converged` first and a None is emitted rather than raising, so a
    broken fit is REPORTED as broken instead of crashing the memo."""
    def s2(model):
        r = get(res, cell, model, series)
        if r is None or not r.get('converged'):
            return None
        return r.get('sigma2_state')

    base_own = s2('M1')
    base_matched = s2('M1matched')
    rows = []
    for model in MODELS:
        r = get(res, cell, model, series)
        ok = r is not None and r.get('converged')
        v = s2(model)
        d_own = (100 * (base_own - v) / base_own
                 if ok and base_own not in (None, 0) else None)
        d_mat = (100 * (base_matched - v) / base_matched
                 if ok and base_matched not in (None, 0) else None)
        # M1 sits on a DIFFERENT (larger) sample than the matched baseline, so
        # a "matched" reduction for it would compare two different samples and
        # mean nothing. Withheld rather than printed.
        if model == 'M1':
            d_mat = None
        rows.append({
            'cell': cell, 'model': model, 'series': series,
            'converged': bool(ok),
            'n_obs': r.get('n_obs') if r else None,
            'n_states': r.get('n_states') if r else None,
            'obs_zeros': r.get('obs_zeros') if r else None,
            'sigma2_state': v,
            'sigma2_year': r.get('sigma2_year') if ok else None,
            'sigma2_zi_state': r.get('sigma2_zi_state') if ok else None,
            'aic': r.get('aic') if ok else None,
            'zi_tier': r.get('zi_tier_reached') if ok else None,
            'zi_prob_mean': r.get('zi_prob_mean') if ok else None,
            'delta_pct': d_own,
            'delta_pct_matched': d_mat,
            'message': (r.get('message') or '') if r else '',
        })
    return rows


def variance_table_md(res, cells, series):
    L = ['| Cell | Model | N | States | sigma^2_state | Delta % vs M1 | '
         'Delta % vs matched M1 | AIC | AIC vs matched M1 |',
         '|---|---|---|---|---|---|---|---|---|']
    for cell in cells:
        if series not in res[f'{cell}__meta']['series']:
            continue
        rows = variance_rows(res, cell, series)
        base_aic = next((r['aic'] for r in rows if r['model'] == 'M1matched'
                         and r['converged']), None)
        for row in rows:
            note = '' if row['converged'] else ' **(DID NOT CONVERGE)**'
            # AIC is comparable only WITHIN one sample. M1 sits on the 49-state
            # sample; M1matched/M2/M3 all sit on the 46-state one. Comparing the
            # first against the others would be comparing likelihoods computed
            # over different rows, so it is withheld rather than printed.
            d_aic = (row['aic'] - base_aic
                     if row['converged'] and base_aic is not None
                     and row['model'] != 'M1' else None)
            L.append(
                f"| {CELL_LABEL.get(row['cell'], row['cell'])} | "
                f"{MODEL_LABEL[row['model']]}{note} | "
                f"{row['n_obs']} | {row['n_states']} | "
                f"{fmt(row['sigma2_state'])} | "
                f"{fmt(row['delta_pct'], '+.1f')} | "
                f"{fmt(row['delta_pct_matched'], '+.1f')} | "
                f"{fmt(row['aic'], '.2f')} | "
                f"{fmt(d_aic, '+.2f')} |")
    return '\n'.join(L)


def coef_block_md(r, block):
    """One coefficient block. `block` is 'cond' or 'zi'; the exponentiated
    column is labelled IRR for the conditional block and OR for the ZI block,
    because they are not the same quantity and must never share a header."""
    co = r.get(block) or {}
    if not co:
        return '_(no terms in this block)_'
    expcol = 'IRR = exp(b)' if block == 'cond' else 'OR = exp(b)'
    L = [f'| Term | b | SE | z | p | | {expcol} |',
         '|---|---|---|---|---|---|---|']
    terms = sorted(co, key=lambda t: TERM_ORDER.index(t)
                   if t in TERM_ORDER else len(TERM_ORDER))
    for t in terms:
        c = co[t]
        b, se, z, p = c['b'], c['se'], c['z'], c['p']
        ex = math.exp(b) if b is not None and abs(b) < 700 else None
        L.append(f"| {TERM_LABEL.get(t, t)} | {fmt(b)} | {fmt(se)} | "
                 f"{fmt(z, '.2f')} | {fmt(p, '.4f')} | {stars(p)} | "
                 f"{fmt(ex, '.4f')} |")
    return '\n'.join(L)


def selection_table_md(res, cell):
    meta = res.get(f'{cell}__selection_meta')
    if meta is None:
        return None
    L = ['| Family | ZI tier reached | AIC | logLik | df | Converged | '
         'ZI-degenerate | Collapsed onto | Eligible |',
         '|---|---|---|---|---|---|---|---|---|']
    cands = sorted(meta['candidates'],
                   key=lambda c: (c['aic'] is None, c['aic']
                                  if c['aic'] is not None else 0))
    for c in cands:
        L.append(
            f"| {FAMILY_LABEL.get(c['family_tag'], c['family_tag'])} | "
            f"{c['zi_tier_reached'] or '--'} | {fmt(c['aic'], '.2f')} | "
            f"{fmt(c['loglik'], '.2f')} | {c['df'] if c['df'] else '--'} | "
            f"{'yes' if c['converged'] else 'NO'} | "
            f"{'yes' if c['zi_degenerate'] else 'no'} | "
            f"{c['collapsed_to'] or '--'} | "
            f"{'yes' if c['eligible'] else 'no'} |")
    return '\n'.join(L)


def nonconverged(res):
    """Every record in the artifact carrying converged = FALSE, with its role.

    `series == 'selection'` records are family-selection CANDIDATES for
    `viol_2021_ni` -- losing candidates in a race, never quoted as estimates.
    `is_companion` records are matched-family companions: diagnostic fits for
    section 4's bracket, not reported models, so a failure there costs the
    bracket rather than a reported row. Anything else is a substantive fit in a
    reported series and would invalidate the row it belongs to, which is why
    the three are distinguished here rather than lumped into one count."""
    out = []
    for key, r in res.items():
        if not isinstance(r, dict) or 'converged' not in r:
            continue
        if not r.get('converged'):
            role = ('companion' if r.get('is_companion')
                    else (r.get('series') or 'unknown'))
            # Repeated identical optimizer messages ("NA/NaN function
            # evaluation" x13) carry no extra information; collapse them.
            parts = [p.strip() for p in (r.get('message') or '').split('|')]
            seen, uniq = set(), []
            for p in parts:
                if p and p not in seen:
                    seen.add(p)
                    uniq.append(p)
            out.append({'key': key, 'role': role,
                        'message': ' | '.join(uniq)})
    return out


COMPANION_SUFFIX = '__companion'


def rec_at_family(res, cell, model, fam, series):
    """The fit for (cell, model, series) estimated under family `fam`.

    When `fam` is the cell's OWN selected family this is the reported record.
    Otherwise it is a matched-family COMPANION record -- a fit that exists only
    so the with/without-inspections comparison can hold the family constant,
    and which is never a reported model.

    Returns None when the fit is absent or did not converge, so that a caller
    can fall back to the confounded comparison instead of printing a hole."""
    own = res[f'{cell}__meta']['family_tag']
    key = f'{cell}__{model}__{fam}__{series}'
    if fam != own:
        key += COMPANION_SUFFIX
    r = res.get(key)
    return r if (r is not None and r.get('converged')) else None


def comparison_bases(res, wi, ni):
    """The (label, family used for WI, family used for NI) bases to report.

    Derived from the artifact, never typed. The first is always the cells as
    actually selected -- which may be confounded. Each additional basis holds
    ONE family fixed across both cells; there is one per distinct family in
    play, so both directions appear and neither is privileged. If the two cells
    already share a family the list collapses to a single (unconfounded) basis.
    """
    fam_wi = res[f'{wi}__meta']['family_tag']
    fam_ni = res[f'{ni}__meta']['family_tag']
    out, seen = [], set()

    def add(label, a, b, matched):
        if (a, b) in seen:
            return
        seen.add((a, b))
        out.append({'label': label, 'fam_wi': a, 'fam_ni': b,
                    'matched': matched})

    add('As selected (each cell keeps its own family)', fam_wi, fam_ni,
        fam_wi == fam_ni)
    for fam in (fam_wi, fam_ni):
        add(f'Both at {FAMILY_LABEL.get(fam, fam)}', fam, fam, True)
    return out


def bracket_rows(res, wi, ni, model, series):
    """One row per available basis at `model`: sigma^2_state either side and
    the share of between-state variance the `log(inspections)` term accounts
    for. A basis whose fit is missing or non-converged is reported as
    unavailable rather than dropped."""
    rows = []
    for basis in comparison_bases(res, wi, ni):
        a = rec_at_family(res, wi, model, basis['fam_wi'], series)
        b = rec_at_family(res, ni, model, basis['fam_ni'], series)
        row = dict(basis, model=model, wi=a, ni=b, available=(a is not None
                                                              and b is not None))
        if row['available']:
            row['share'] = (100 * (b['sigma2_state'] - a['sigma2_state'])
                            / b['sigma2_state'])
        else:
            row['share'] = None
            row['unavailable_reason'] = (
                f"missing or non-converged fit: "
                f"{wi}@{basis['fam_wi']}={'ok' if a else 'UNAVAILABLE'}, "
                f"{ni}@{basis['fam_ni']}={'ok' if b else 'UNAVAILABLE'}")
        rows.append(row)
    return rows


def headline_series(res, cell):
    """The series the memo headlines for this cell: `zi_intercept` when the
    family is zero-inflated (ZI block held constant across all three steps, so
    the only thing that changes between M1, M2 and M3 is the conditional
    predictor set), `plain` when it is not zero-inflated at all."""
    ser = res[f'{cell}__meta']['series']
    return 'zi_intercept' if 'zi_intercept' in ser else ser[0]


def write_memo(res):
    cells = cells_of(res)
    run = res['__run_meta']
    L = []
    A = L.append

    A('# Stepwise State-to-State Variance, Jafari-Aligned Crossed Random '
      'Intercepts (2011-2021 WPS view)')
    A('')
    A('**GENERATED by `scripts/report_jafari_stepwise.py`. Never hand-edit -- '
      're-run it.**')
    A('')
    A('Every number below is read from `data/generated/jafari_stepwise_results'
      '.json`, produced by `scripts/jafari_stepwise_models.R`. This reporter '
      'fits nothing.')
    A('')

    # ---------------------------------------------------------------- 1
    A('## 1. What this answers, and who asked for it')
    A('')
    A('Joe Grzywacz, in the meeting of **2026-09-11**, named one thing as '
      'blocking submission: the Jafari-aligned crossed random-effects models '
      'enter every covariate at once and report **no variance reduction at '
      'all**, so there is nothing to say about how much of the state-to-state '
      'differences the covariates actually explain. This arm supplies that: a '
      '**stepwise Model 1 -> Model 2 -> Model 3 build-up** with the '
      'between-state variance `sigma^2_state` at every step.')
    A('')
    A('It also answers his second request. Jafari et al. had no inspections '
      'variable; ours carries `log(inspections)` as a Level-1 covariate in the '
      'violations models. So the violations series is fit **twice** -- once '
      'with that term and once without -- and section 4 compares them '
      'directly, with **matched-family companion fits** so that the '
      'comparison is not confounded with the distributional family each cell '
      'happens to carry. Whether those companions are usable is stated in '
      'section 4 rather than assumed here.')
    A('')
    A(f'**Window: {run["window"]} only.** Per PI direction 2026-09-11 the '
      '2011-2019 establishments-view window is out of scope here. Panel: '
      f'`{run["panel"]}`, {run["n_panel_rows"]} rows, '
      f'{run["n_panel_states"]} states.')
    A('')
    A(f'**Random effects: `{run["re_structure"]}` -- crossed random '
      'INTERCEPTS only, no random slope anywhere** (standing PI '
      'directive). Estimated with glmmTMB '
      f'{run["glmmTMB_version"]} under {run["R_version"]}.')
    A('')
    A('### The models')
    A('')
    A('| Step | Adds |')
    A('|---|---|')
    A('| Model 1 | cubic time (`time`, `time2`, `time3`), centered at 2017 '
      '(plus `log(inspections)` in the "with inspections" violations cell) |')
    A('| Model 2 | `SPEND_APP_z`, `SPEND_WORK_z`, `lii_2017_z` |')
    A('| Model 3 | `h2a_per_farmworker_z`, `dol_demand_met_pct_z`, '
      '`pct_flc_z` |')
    A('')

    # families
    A('### The family is HELD FIXED within each cell')
    A('')
    A('This is the whole point of the arm. If the distributional family were '
      'allowed to change between steps, a change in `sigma^2_state` would '
      'reflect the family swap and not the covariates, and the reduction '
      'sequence would be meaningless.')
    A('')
    A('| Cell | Outcome | Family held fixed | Where that family came from |')
    A('|---|---|---|---|')
    for cell in cells:
        m = res[f'{cell}__meta']
        A(f"| `{cell}` | {m['outcome']} | "
          f"**{FAMILY_LABEL.get(m['family_tag'], m['family_tag'])}** | "
          f"{m['family_source']} |")
    A('')

    for cell in cells:
        sm = res.get(f'{cell}__selection_meta')
        if sm is None:
            continue
        A(f'#### Family selection for `{cell}`')
        A('')
        A(f"No precedent existed for this cell, so the family was raced at "
          f"**{sm['model_selected_at']}** across the same eight-family ladder "
          f"the crossed arm uses, then held fixed for M1 and M2. "
          f"{sm['n_eligible']} of 8 candidates were eligible (converged, not "
          f"ZI-boundary-degenerate, not collapsed onto a twin). Winner: "
          f"**{FAMILY_LABEL.get(sm['selected_family'], sm['selected_family'])}"
          f"**.")
        A('')
        A(selection_table_md(res, cell))
        A('')

    # ---------------------------------------------------------------- 2
    A('## 2. Headline: state-to-state variance at each step')
    A('')
    A('**Zero-inflation block held CONSTANT (`ziformula = ~1`) across all '
      'three steps.** This is the clean variance-reduction series: the only '
      'thing that changes between Model 1, Model 2 and Model 3 is the '
      'conditional predictor set, so a change in `sigma^2_state` is '
      'attributable to the covariates and nothing else. Section 3 reports the '
      'Jafari-fidelity `mirror` series, where the ZI block grows as covariates '
      'enter and therefore does **not** isolate them.')
    A('')
    A(variance_table_md(res, cells, 'zi_intercept'))
    A('')
    A('**How to read the two Delta columns.** `Delta % vs M1` uses Model 1 fit '
      'on its own 49-state sample as the baseline. `Delta % vs matched M1` '
      'refits that same Model-1 formula on the 46-state Model-2/3 sample '
      'instead. Positive = the covariates reduced between-state variance. See '
      'the caveat in section 5.2 -- the two columns can point in opposite '
      'directions, and in this arm they do.')
    A('')
    A('**The AIC column.** `AIC vs matched M1` is each model\'s AIC minus the '
      'matched Model-1 baseline\'s. It is shown only for the rows fit on the '
      '46-state sample, because a likelihood computed over 539 rows cannot be '
      'compared with one computed over 506. **Negative = the covariates '
      'improve fit; positive = they do not pay for their parameters.** A '
      'covariate block can reduce `sigma^2_state` and still lose on AIC -- '
      'those are different questions, and both belong in the write-up.')
    A('')

    # Per-cell plain-English reduction summary, derived.
    A('### In words')
    A('')
    for cell in cells:
        ser = headline_series(res, cell)
        rows = {r['model']: r for r in variance_rows(res, cell, ser)}
        m1, m2, m3 = rows['M1'], rows['M2'], rows['M3']
        mm = rows['M1matched']
        if not (m1['converged'] and m3['converged']):
            A(f"- **{CELL_LABEL.get(cell, cell)}**: not reportable -- at least "
              f"one fit in the series did not converge.")
            continue
        direction = ('lowers' if mm['sigma2_state'] < m1['sigma2_state']
                     else 'raises')
        effect = ('overstates' if mm['sigma2_state'] < m1['sigma2_state']
                  else 'understates')
        A(f"- **{CELL_LABEL.get(cell, cell)}** "
          f"({FAMILY_LABEL.get(res[f'{cell}__meta']['family_tag'], '')}): "
          f"`sigma^2_state` goes {fmt(m1['sigma2_state'], '.3f')} (M1) -> "
          f"{fmt(m2['sigma2_state'], '.3f')} (M2) -> "
          f"{fmt(m3['sigma2_state'], '.3f')} (M3). Against Model 1 that is "
          f"{fmt(m2['delta_pct'], '+.1f')}% and "
          f"{fmt(m3['delta_pct'], '+.1f')}%; against the matched Model-1 "
          f"baseline, {fmt(m2['delta_pct_matched'], '+.1f')}% and "
          f"{fmt(m3['delta_pct_matched'], '+.1f')}%. Dropping AK/RI/VT "
          f"{direction} the Model-1 baseline "
          f"({fmt(m1['sigma2_state'], '.3f')} -> "
          f"{fmt(mm['sigma2_state'], '.3f')}), so the unmatched column "
          f"{effect} what the covariates do.")
    A('')

    # ---------------------------------------------------------------- 3
    A('## 3. The Jafari-fidelity `mirror` series (secondary)')
    A('')
    A('Here the zero-inflation formula **mirrors the conditional predictor '
      'list at every step**, which is what Jafari et al. do and what the '
      'existing crossed arm does. It is reported for comparability, but it is '
      '**not** the variance-reduction series: as covariates enter, they enter '
      '*both* blocks, so a change in `sigma^2_state` confounds the conditional '
      'covariates with the ZI block. Read section 2 for the reduction and this '
      'table for fidelity.')
    A('')
    A(variance_table_md(res, cells, 'mirror'))
    A('')
    A('Where the ZI tier column in the JSON reads something other than '
      '`mirror`, the full mirror was not estimable at that step and the ladder '
      'fell back to the richest estimable subset (`covariates` -> `reduced` -> '
      '`intercept`); at those steps the `mirror` and `zi_intercept` series can '
      'coincide exactly.')
    A('')

    # correctness anchor
    anchor_lines = []
    for cell in cells:
        ca = res[f'{cell}__meta'].get('crossed_arm')
        if not ca:
            continue
        for model in SUBSTANTIVE:
            a = ca.get(f'anchor_{model}')
            if not a:
                continue
            mine = get(res, cell, model, 'mirror')
            if mine is None or not mine.get('converged'):
                continue
            diff = abs(mine['sigma2_state'] - a['sigma2_state'])
            anchor_lines.append(
                f"| `{cell}` | {model} | {fmt(a['sigma2_state'], '.4f')} | "
                f"{fmt(mine['sigma2_state'], '.4f')} | {diff:.2e} | "
                f"{a['n_obs']} / {mine['n_obs']} | "
                f"{'MATCH' if diff < 5e-4 else '**MISMATCH**'} |")
    n_match = sum(1 for ln in anchor_lines if 'MISMATCH' not in ln)
    if anchor_lines:
        A('### Correctness anchor: does `mirror` reproduce the existing '
          'crossed arm?')
        A('')
        A('The two cells that inherit their family from '
          '`jafari_crossed_results.json` should reproduce that arm\'s fits '
          'exactly in the `mirror` series, because the specification is '
          'identical. This is the check that would catch a spec mismatch in '
          'this script.')
        A('')
        A('| Cell | Model | Crossed arm sigma^2_state | This arm | '
          'abs. difference | N (crossed / here) | Verdict |')
        A('|---|---|---|---|---|---|---|')
        L.extend(anchor_lines)
        A('')
        A(f'**{n_match} of {len(anchor_lines)} anchor values reproduced** to '
          'within 5e-4.'
          + ('' if n_match == len(anchor_lines) else
             ' **At least one did NOT -- treat every number in this memo as '
             'suspect until that is explained.**'))
        A('')

    # ---------------------------------------------------------------- 4
    A('## 4. With vs without inspections (Joe\'s second request)')
    A('')
    wi, ni = 'viol_2021_wi', 'viol_2021_ni'
    if f'{wi}__meta' in res and f'{ni}__meta' in res:
        swi, sni = headline_series(res, wi), headline_series(res, ni)
        rwi = {r['model']: r for r in variance_rows(res, wi, swi)}
        rni = {r['model']: r for r in variance_rows(res, ni, sni)}
        # What actually differs between the two cells is DERIVED, not
        # asserted. An earlier draft of this paragraph said the two series
        # "differ in exactly one term"; they do not, because the family is
        # selected per cell and the two cells did not land on the same one.
        # Claiming one difference while there are two would hand the whole of
        # the absorbed-variance figure below to `log(inspections)`.
        fam_wi = res[f'{wi}__meta']['family_tag']
        fam_ni = res[f'{ni}__meta']['family_tag']
        zi_wi = (get(res, wi, 'M3', swi) or {}).get('zi_formula')
        zi_ni = (get(res, ni, 'M3', sni) or {}).get('zi_formula')
        A('Both series model the same outcome (`violations`) on the same '
          'window, over the same rows, with the same conditional covariate '
          'build-up and the same `(1 | state) + (1 | year)` random effects.')
        A('')
        confounded = (fam_wi != fam_ni) or (zi_wi != zi_ni)
        # The matched-family companion fits, keyed by model. Each row is one
        # BASIS: the two cells as selected, or both forced onto one family.
        bracket = {m: bracket_rows(res, wi, ni, m, 'zi_intercept')
                   for m in MODELS}

        def matched_rows(model):
            return [r for r in bracket[model] if r['matched'] and r['available']]

        def selected_row(model):
            """The as-selected basis -- always the first one built."""
            return bracket[model][0]

        # The bracket is reportable only if BOTH matched directions are
        # available at every substantive step. Anything less and the memo
        # falls back to the upper-bound framing rather than printing a hole.
        n_matched_expected = len({fam_wi, fam_ni})
        have_bracket = all(len(matched_rows(m)) == n_matched_expected
                           for m in MODELS)

        if not confounded:
            A('They differ in exactly one term: `log(inspections)` as a '
              'Level-1 covariate. Jafari et al. had no such variable.')
            A('')
        else:
            A('**As each cell is reported they differ in two things, not one, '
              'and the second is easy to miss.**')
            A('')
            A('1. `log(inspections)` enters the conditional block of '
              f'`{wi}` and not `{ni}`. That is the comparison Joe asked for; '
              'Jafari et al. had no such variable.')
            A(f'2. **The family is not the same.** `{wi}` is fit as '
              f'**{FAMILY_LABEL.get(fam_wi, fam_wi)}** '
              f'(ziformula `{zi_wi}`) and `{ni}` as '
              f'**{FAMILY_LABEL.get(fam_ni, fam_ni)}** '
              f'(ziformula `{zi_ni}`). Each cell\'s family was chosen for that '
              'cell (section 1), and the two choices did not coincide.')
            A('')
            zire_wi = '(1 | state)' in (zi_wi or '')
            zire_ni = '(1 | state)' in (zi_ni or '')
            if zire_wi != zire_ni:
                has, hasnt = ((wi, ni) if zire_wi else (ni, wi))
                s2zi = (get(res, has, 'M3', headline_series(res, has))
                        or {}).get('sigma2_zi_state')
                A(f'That second difference matters in a specific direction. '
                  f'`{has}` carries a state random intercept in its '
                  f'**zero-inflation** block '
                  + (f'(variance {fmt(s2zi, ".3f")} at Model 3) '
                     if s2zi is not None else '')
                  + f'and `{hasnt}` does not. Between-state heterogeneity in '
                    f'the zero process therefore has somewhere to go in '
                    f'`{has}` and nowhere to go in `{hasnt}`, where it must '
                    f'load onto the conditional `sigma^2_state` instead. So '
                    f'the as-selected gap is **not** attributable to '
                    f'`log(inspections)` alone: part of it is the '
                    f'zero-inflation random effect.')
                A('')
            if have_bracket:
                A('**That confound is now removed.** Each violations cell is '
                  'also fit under the *other* cell\'s family -- a '
                  '**matched-family companion** fit -- so the comparison can '
                  'be made with family held constant. Both directions are '
                  'fit, never one: each direction forces a cell onto a family '
                  'that loses for that cell, so neither is privileged, and '
                  'together they **bracket** the answer. The companions are '
                  'tagged `is_companion` in the JSON and are **not** reported '
                  'models -- every series in sections 2, 3 and 6 still uses '
                  'its own cell\'s selected family, unchanged.')
                A('')
            else:
                A('**The matched-family companions are not available**, so '
                  'the share below remains an **upper bound** on what '
                  '`log(inspections)` does rather than an estimate of it. '
                  'Unavailable bases:')
                for m in MODELS:
                    for r in bracket[m]:
                        if r['matched'] and not r['available']:
                            A(f"  - {MODEL_LABEL[m]}, {r['label']}: "
                              f"{r['unavailable_reason']}")
                A('')

        # ---- 4.1 the as-selected comparison ------------------------------
        A('### 4.1 The comparison as each cell is reported')
        A('')
        A('Each cell under its own selected family'
          + (' -- **confounded**, because those families differ (see above); '
             'read the share as an upper bound.'
             if confounded else '.'))
        A('')
        A('| Model | sigma^2_state WITH inspections | sigma^2_state WITHOUT | '
          'Difference | Share of between-state variance absorbed by '
          'log(inspections) |')
        A('|---|---|---|---|---|')
        for model in MODELS:
            a, b = rwi.get(model), rni.get(model)
            if not (a and b and a['converged'] and b['converged']):
                continue
            share = (100 * (b['sigma2_state'] - a['sigma2_state'])
                     / b['sigma2_state'])
            A(f"| {MODEL_LABEL[model]} | {fmt(a['sigma2_state'], '.4f')} | "
              f"{fmt(b['sigma2_state'], '.4f')} | "
              f"{fmt(a['sigma2_state'] - b['sigma2_state'], '+.4f')} | "
              f"{fmt(share, '.1f')}% |")
        A('')

        # ---- 4.2 the bracket ----------------------------------------------
        if confounded and have_bracket:
            A('### 4.2 Matched-family companions: the bracket')
            A('')
            A('Every row below is the same two cells over the same rows, with '
              'the same conditional build-up; the only thing that changes '
              'down the table is which family both cells are held at. The '
              'first row of each block repeats section 4.1 so the confounded '
              'figure and the two matched ones can be read together.')
            A('')
            A('| Model | Basis | Family, WITH | Family, WITHOUT | '
              'sigma^2_state WITH | sigma^2_state WITHOUT | '
              'Share absorbed by log(inspections) |')
            A('|---|---|---|---|---|---|---|')
            for model in MODELS:
                for r in bracket[model]:
                    if not r['available']:
                        continue
                    tag = '' if r['matched'] else ' (confounded)'
                    A(f"| {MODEL_LABEL[model]} | {r['label']}{tag} | "
                      f"{FAMILY_LABEL.get(r['fam_wi'], r['fam_wi'])} | "
                      f"{FAMILY_LABEL.get(r['fam_ni'], r['fam_ni'])} | "
                      f"{fmt(r['wi']['sigma2_state'], '.4f')} | "
                      f"{fmt(r['ni']['sigma2_state'], '.4f')} | "
                      f"{fmt(r['share'], '.1f')}% |")
            A('')
            A('Both cells sit on identical analytic samples at every step '
              '(N = '
              + ', '.join(f"{m} {bracket[m][0]['wi']['n_obs']}"
                          for m in MODELS)
              + '), so nothing in this table is a sample effect.')
            A('')

        # ---- 4.3 the plain-language answer --------------------------------
        A('### 4.3 What `log(inspections)` actually does')
        A('')
        m1a, m1b = rwi['M1'], rni['M1']
        if not (m1a['converged'] and m1b['converged']):
            A('Not reportable -- at least one Model-1 fit did not converge.')
            A('')
        elif not confounded:
            share1 = (100 * (m1b['sigma2_state'] - m1a['sigma2_state'])
                      / m1b['sigma2_state'])
            A(f'At Model 1 the between-state variance in violations is '
              f'{fmt(m1b["sigma2_state"], ".3f")} without `log(inspections)` '
              f'and {fmt(m1a["sigma2_state"], ".3f")} with it: the term '
              f'accounts for **{share1:.0f}%** of the state-to-state variance '
              f'in violation counts. The two cells share a family, so this is '
              f'an estimate, not a bound.')
            A('')
        elif have_bracket:
            lo1 = min(r['share'] for r in matched_rows('M1'))
            hi1 = max(r['share'] for r in matched_rows('M1'))
            lo3 = min(r['share'] for r in matched_rows('M3'))
            hi3 = max(r['share'] for r in matched_rows('M3'))
            sel1 = selected_row('M1')['share']
            sel3 = selected_row('M3')['share']
            ratio3 = [r['ni']['sigma2_state'] / r['wi']['sigma2_state']
                      for r in matched_rows('M3')]
            A('**Once the family is held constant, `log(inspections)` '
              f'accounts for {lo1:.1f}-{hi1:.1f}% of the state-to-state '
              'variance in violation counts at Model 1, and '
              f'{lo3:.1f}-{hi3:.1f}% at Model 3.** Each range is a bracket, '
              'not a confidence interval: its two ends are the two ways of '
              'matching the family (both cells at '
              f'{FAMILY_LABEL.get(fam_wi, fam_wi)}, and both at '
              f'{FAMILY_LABEL.get(fam_ni, fam_ni)}), and neither end is '
              'privileged, because each forces one cell onto a family that '
              'loses for it.')
            A('')
            A('In plain terms: **states differ in recorded violations in '
              'large part because they differ in how much they inspect, and '
              'that stays true after the zero-inflation structure is held '
              'constant.** At Model 3 the between-state variance is '
              f'{min(ratio3):.1f}-{max(ratio3):.1f} times larger without the '
              'term than with it, and the state-level covariates do not '
              'close that gap. The variable is doing real work, and the '
              'finding is not an artifact of the two cells having been fit '
              'under different families.')
            A('')
            direction = ('above' if sel1 > hi1 and sel3 > hi3
                         else 'below' if sel1 < lo1 and sel3 < lo3
                         else 'outside' if (sel1 > hi1 or sel1 < lo1)
                         and (sel3 > hi3 or sel3 < lo3) else 'inside')
            A(f'The as-selected figures in section 4.1 -- {sel1:.1f}% at '
              f'Model 1 and {sel3:.1f}% at Model 3 -- sit **{direction}** the '
              'bracket'
              + ('. They were an upper bound, as the note above predicted, '
                 'and the confound was inflating them by '
                 f'{sel1 - hi1:.1f} points at Model 1 and '
                 f'{sel3 - hi3:.1f} at Model 3.'
                 if direction == 'above' else
                 ', which the two-differences note above did not predict; '
                 'treat that as a finding, not a rounding artifact.'))
            A('')
            A('Do **not** quote the as-selected figure on its own as "'
              f'log(inspections) absorbs {sel1:.0f}% of the between-state '
              'variance". Quote the bracket.')
            A('')
        else:
            share1 = (100 * (m1b['sigma2_state'] - m1a['sigma2_state'])
                      / m1b['sigma2_state'])
            A(f'At Model 1 the between-state variance in violations is '
              f'{fmt(m1b["sigma2_state"], ".3f")} without it and '
              f'{fmt(m1a["sigma2_state"], ".3f")} with it -- a gap of about '
              f'**{share1:.0f}%** of the state-to-state variance in violation '
              f'counts, before any covariate is entered. States differ in '
              f'violations in large part because they differ in how much they '
              f'inspect. But the two cells do not share a zero-inflation '
              f'structure and the matched-family companions are unavailable, '
              f'so **{share1:.0f}% is an upper bound on what that single term '
              f'does**, not an estimate of it. Do not quote it as '
              f'"log(inspections) absorbs {share1:.0f}% of the between-state '
              f'variance".')
            A('')

        # ---- 4.4 which family actually fits -------------------------------
        if confounded and have_bracket:
            A('### 4.4 Which family fits better, once the ZI block is held at '
              'the intercept')
            A('')
            A('The companions make one further comparison possible that the '
              'arm could not make before. Within a cell, the own-family fit '
              'and its companion sit on the **same rows with the same '
              'conditional formula and an intercept-only ZI block**; the only '
              'difference is the ZI state random intercept. Their AICs are '
              'therefore directly comparable.')
            A('')
            A('| Cell | Model | AIC under its own family | AIC under the '
              'companion family | Better fit here |')
            A('|---|---|---|---|---|')
            reversals = []
            for cell, own, other in ((wi, fam_wi, fam_ni), (ni, fam_ni, fam_wi)):
                for model in MODELS:
                    a = rec_at_family(res, cell, model, own, 'zi_intercept')
                    b = rec_at_family(res, cell, model, other, 'zi_intercept')
                    if a is None or b is None:
                        continue
                    better = (FAMILY_LABEL.get(own, own) if a['aic'] < b['aic']
                              else FAMILY_LABEL.get(other, other))
                    if a['aic'] > b['aic']:
                        reversals.append((cell, model, b['aic'] - a['aic']))
                    A(f"| {CELL_LABEL.get(cell, cell)} | "
                      f"{MODEL_LABEL[model]} | "
                      f"{FAMILY_LABEL.get(own, own)} "
                      f"{fmt(a['aic'], '.2f')} | "
                      f"{FAMILY_LABEL.get(other, other)} "
                      f"{fmt(b['aic'], '.2f')} | "
                      f"**{better}** ({fmt(b['aic'] - a['aic'], '+.2f')} AIC "
                      f"for the companion) |")
            A('')
            if reversals:
                cells_rev = sorted({c for c, _, _ in reversals})
                A('**Worth flagging.** In '
                  + ', '.join(f'`{c}`' for c in cells_rev)
                  + ' the companion family fits *better* than the family the '
                    'cell actually carries, at every step ('
                  + ', '.join(f'{m} {d:+.2f}'
                              for c, m, d in reversals
                              if c == cells_rev[0])
                  + ' AIC). That is not a contradiction of section 1: the '
                    'family race walks the ZI **fidelity** ladder first and '
                    'compares families at whatever ZI tier each one reached, '
                    'so the two candidates were not compared at the same ZI '
                    'specification.')
                sm = res.get(f'{cells_rev[0]}__selection_meta')
                if sm:
                    tiers = {c['family_tag']: c['zi_tier_reached']
                             for c in sm['candidates']}
                    aics = {c['family_tag']: c['aic'] for c in sm['candidates']}
                    pair = [f for f in (fam_wi, fam_ni) if f in tiers]
                    A('')
                    A('| Family | ZI tier reached in the race | AIC in the '
                      'race | AIC at Model 3 with the ZI block held at the '
                      'intercept |')
                    A('|---|---|---|---|')
                    for f in pair:
                        r3 = rec_at_family(res, cells_rev[0], 'M3', f,
                                           'zi_intercept')
                        A(f"| {FAMILY_LABEL.get(f, f)} | "
                          f"{tiers.get(f) or '--'} | "
                          f"{fmt(aics.get(f), '.2f')} | "
                          f"{fmt(r3['aic'] if r3 else None, '.2f')} |")
                    A('')
                    A('The race is the arm\'s stated rule and section 1\'s '
                      'selection stands. But the headline series holds the ZI '
                      'block at the intercept, and at *that* specification the '
                      'ranking reverses -- which is a reason to read the '
                      'bracket rather than either end of it as the answer, and '
                      'a reason not to describe either family as "the '
                      'best-fitting one" without saying at which ZI tier.')
            else:
                A('In both cells the family the cell actually carries also '
                  'fits better than its companion at this ZI specification, '
                  'so the selection in section 1 and the intercept-only '
                  'headline series agree.')
            A('')

        # ---- 4.5 the within-cell reduction, side by side -------------------
        A('### 4.5 Variance reduction within each cell, side by side')
        A('')
        A('| Model | Delta % vs M1, WITH | Delta % vs M1, WITHOUT | '
          'Delta % vs matched M1, WITH | Delta % vs matched M1, WITHOUT |')
        A('|---|---|---|---|---|')
        for model in ('M2', 'M3'):
            a, b = rwi.get(model), rni.get(model)
            if not (a and b):
                continue
            A(f"| {MODEL_LABEL[model]} | {fmt(a['delta_pct'], '+.1f')} | "
              f"{fmt(b['delta_pct'], '+.1f')} | "
              f"{fmt(a['delta_pct_matched'], '+.1f')} | "
              f"{fmt(b['delta_pct_matched'], '+.1f')} |")
        A('')
        # Do the covariate conclusions change? Derived from M3 coefficients.
        A('### 4.6 Do the covariate conclusions change?')
        A('')
        A('**Do the covariate conclusions change?** Model 3 conditional-block '
          'coefficients, side by side'
          + (' (each cell under its OWN family, so the same caveat as 4.1 '
             'applies: the two columns also differ in zero-inflation '
             'structure, not only in `log(inspections)`; 4.2 is where that '
             'is held constant)'
             if confounded else '') + ':')
        A('')
        a3 = get(res, wi, 'M3', swi)
        b3 = get(res, ni, 'M3', sni)
        if a3 and b3 and a3.get('converged') and b3.get('converged'):
            A('| Term | b (with insp.) | p | | b (without insp.) | p | | '
              'Same sign? | Same significance at p<.05? |')
            A('|---|---|---|---|---|---|---|---|---|')
            shared = [t for t in TERM_ORDER
                      if t in (a3.get('cond') or {}) and t in (b3.get('cond') or {})
                      and t != '(Intercept)']
            flips = []
            for t in shared:
                ca, cb = a3['cond'][t], b3['cond'][t]
                same_sign = (ca['b'] > 0) == (cb['b'] > 0)
                same_sig = (ca['p'] < .05) == (cb['p'] < .05)
                if not same_sig:
                    flips.append(t)
                A(f"| {TERM_LABEL.get(t, t)} | {fmt(ca['b'])} | "
                  f"{fmt(ca['p'], '.4f')} | {stars(ca['p'])} | "
                  f"{fmt(cb['b'])} | {fmt(cb['p'], '.4f')} | "
                  f"{stars(cb['p'])} | {'yes' if same_sign else '**NO**'} | "
                  f"{'yes' if same_sig else '**NO**'} |")
            A('')
            if flips:
                A('Significance status at p < .05 **changes** for '
                  + ', '.join(f'`{t}`' for t in flips)
                  + ' when `log(inspections)` is removed. That term is '
                    'therefore not inert: dropping it changes what the model '
                    'says about the covariates, not only how much variance is '
                    'left to explain.')
            else:
                A('No covariate changes significance status at p < .05 when '
                  '`log(inspections)` is removed. The substantive covariate '
                  'conclusions are the same either way; what changes is how '
                  'much between-state variance remains for them to explain.')
            A('')

    # ---------------------------------------------------------------- 5
    A('## 5. Caveats -- read these before quoting any number above')
    A('')
    A('### 5.1 These variances are on the LOG LINK scale')
    A('')
    A('`sigma^2_state` here is the between-state variance of the log-link '
      'random intercept in a count model. The published manuscript Tables 2 '
      'and 3 report `sigma2_u0_ri` from a Gaussian linear mixed model on '
      '`log(count + 1)`. **The magnitudes are not comparable and must never be '
      'placed in the same column.** A *percentage* reduction is roughly '
      'comparable; an absolute variance is not. The same applies to this '
      'project\'s headline "ICC ~77%", which is a Gaussian-LMM quantity on '
      '`log(count + 1)`.')
    A('')
    A('### 5.2 The sample changes between M1 and M2/M3 -- hence two Delta '
      'columns')
    A('')
    A('Model 1 keeps all 49 states in the WPS view. Models 2 and 3 require '
      '`SPEND_APP_z`, which does not exist for **Alaska, Rhode Island and '
      'Vermont** (BLS has never published a pesticide-applicator series for '
      'them), so those three states are dropped by listwise deletion and N '
      'falls to 46 states. A naive M1 -> M3 reduction therefore mixes "the '
      'covariates explained variance" with "three states left the sample".')
    A('')
    A('`Delta % vs matched M1` exists to separate the two: it is the **same '
      'Model-1 formula refit on the Model-2/3 rows**. Both columns are '
      'reported because neither alone is honest. Per-cell, the direction of '
      'the sample effect is:')
    A('')
    A('| Cell | sigma^2_state, M1 on 49 states | M1 refit on 46 states | '
      'Effect of dropping AK/RI/VT |')
    A('|---|---|---|---|')
    for cell in cells:
        ser = headline_series(res, cell)
        rows = {r['model']: r for r in variance_rows(res, cell, ser)}
        m1, mm = rows['M1'], rows['M1matched']
        if not (m1['converged'] and mm['converged']):
            continue
        d = mm['sigma2_state'] - m1['sigma2_state']
        word = ('LOWERS the baseline, so the unmatched Delta overstates the '
                'covariates' if d < 0 else
                'RAISES the baseline, so the unmatched Delta understates the '
                'covariates')
        A(f"| {CELL_LABEL.get(cell, cell)} | {fmt(m1['sigma2_state'], '.4f')} "
          f"| {fmt(mm['sigma2_state'], '.4f')} | {fmt(d, '+.4f')} -- {word} |")
    A('')

    A('### 5.3 The `year` random effect is essentially empty -- this is '
      '"crossed" in name more than in practice')
    A('')
    A('The random-effects structure is crossed state x year, but the fixed '
      'cubic in `time` already absorbs the smooth year-to-year movement, '
      'leaving the `(1 | year)` term almost nothing to explain. **If you are '
      'writing "crossed random effects" in the methods, say this too.** '
      'Per-cell, at Model 3 in the headline series:')
    A('')
    A('| Cell | sigma^2_state | sigma^2_year | year share of (state + year) |')
    A('|---|---|---|---|')
    boundary, small = [], []
    for cell in cells:
        ser = headline_series(res, cell)
        r = get(res, cell, 'M3', ser)
        if r is None or not r.get('converged'):
            continue
        s2s, s2y = r['sigma2_state'], r['sigma2_year']
        share = 100 * s2y / (s2s + s2y)
        A(f"| {CELL_LABEL.get(cell, cell)} | {fmt(s2s, '.4f')} | "
          f"{s2y:.3e} | {share:.3f}% |")
        (boundary if s2y < 1e-6 else small).append((cell, s2y, share))
    A('')
    if boundary:
        A('In ' + ('all ' if not small else '')
          + f'{len(boundary)} of {len(boundary) + len(small)} cells '
          + '`sigma^2_year` is pinned at the optimizer boundary ('
          + ', '.join(f'`{c}` = {v:.1e}' for c, v, _ in boundary)
          + '), i.e. numerically zero.')
    if small:
        A('In '
          + ', '.join(f'`{c}`' for c, _, _ in small)
          + ' it is not exactly zero but is still negligible ('
          + ', '.join(f'{v:.2e}, {s:.3f}% of the state + year total'
                      for c, v, s in small)
          + ').')
    A('')
    A('**Practical reading: the crossed state x year structure behaves as a '
      'state-only random intercept in these data.** Do not describe the year '
      'dimension as doing explanatory work. This is the cubic-versus-'
      '`(1 | year)` overlap the crossed-arm design spec flagged, '
      'materialising.')
    A('')

    A('### 5.4 Other things not to over-read')
    A('')
    A('- **No COVID indicator.** `(1 | year)` absorbs the pandemic shock by '
      'construction, so a 2020/21 dummy would compete with the term that '
      'already contains it. Since `sigma^2_year` is ~0 (5.3), this arm says '
      'nothing at all about COVID. Do not compare it to the other arms\' COVID '
      'coefficients.')
    A('- **These are not the published Tables 2/3 models.** Different family, '
      'different link, different random-effects structure. Coefficients are '
      'not line-by-line comparable to the manuscript tables.')
    A('- **Structural-zero probabilities** reported below are the SAMPLE MEAN '
      'of the per-observation zero-inflation probability, not '
      '`plogis(zi intercept)`.')
    nc = nonconverged(res)
    reported = [x for x in nc
                if x['role'] not in ('selection', 'companion')]
    cand = [x for x in nc if x['role'] == 'selection']
    comp = [x for x in nc if x['role'] == 'companion']
    if not nc:
        A('- **Every fit in this artifact converged.**')
    else:
        if reported:
            A(f'- **{len(reported)} fit(s) in a REPORTED series did not '
              'converge.** Those rows are marked "DID NOT CONVERGE" in the '
              'tables above and no estimate is taken from them:')
            for x in reported:
                A(f"  - `{x['key']}`: {x['message']}")
        else:
            A('- **Every fit in every reported series converged.** No number '
              'in sections 2, 3, 4 or 6 comes from a failed optimisation.')
        if cand:
            A(f'- {len(cand)} family-SELECTION candidate(s) failed to '
              'converge. These are losing entries in the `viol_2021_ni` '
              'family race (section 1) and were excluded from selection for '
              'exactly that reason; none is a reported model:')
            for x in cand:
                A(f"  - `{x['key']}`: {x['message']}")
        if comp:
            A(f'- {len(comp)} matched-family COMPANION fit(s) did not '
              'converge. These are section 4\'s off-family diagnostics, not '
              'reported models; where one is missing, section 4 marks that '
              'direction of the bracket unavailable and falls back to the '
              'upper-bound framing:')
            for x in comp:
                A(f"  - `{x['key']}`: {x['message']}")
        else:
            A('- **Every matched-family companion fit converged**, so '
              'section 4 reports both directions of the bracket.')
    A('')

    # ---------------------------------------------------------------- 6
    A('## 6. Full Model 3 coefficients, both blocks')
    A('')
    A('Reported from the headline (`zi_intercept`) series, so the '
      'zero-inflation block is the same intercept-only block used at every '
      'step. The `mirror` series\' Model-3 fits, where the ZI block carries '
      'the full predictor list, follow each cell.')
    A('')
    for cell in cells:
        meta = res[f'{cell}__meta']
        # Headline series FIRST, then the rest, so the ordering on the page
        # matches what the paragraph above promises.
        head = headline_series(res, cell)
        ordered = [head] + [s for s in meta['series'] if s != head]
        for ser in ordered:
            r = get(res, cell, 'M3', ser)
            if r is None:
                continue
            label = ('headline, ZI block = intercept only'
                     if ser == 'zi_intercept' else
                     'Jafari mirror, ZI block = full predictor list'
                     if ser == 'mirror' else ser)
            A(f'### {CELL_LABEL.get(cell, cell)} -- Model 3 ({label})')
            A('')
            if not r.get('converged'):
                A('**THIS FIT DID NOT CONVERGE. No estimate is reported.** '
                  f"Message: {r.get('message', '')}")
                A('')
                continue
            A(f"Family **{FAMILY_LABEL.get(meta['family_tag'], meta['family_tag'])}**, "
              f"`{r['formula']}`, ziformula `{r['zi_formula']}`; "
              f"N = {r['n_obs']} ({r['n_states']} states, {r['n_years']} years), "
              f"observed zeros = {r['obs_zeros']}, AIC = {fmt(r['aic'], '.2f')}, "
              f"logLik = {fmt(r['loglik'], '.2f')} on {r['df']} df.")
            A('')
            A(f"Variance components: `sigma^2_state` = "
              f"{fmt(r['sigma2_state'])}, `sigma^2_year` = "
              f"{r['sigma2_year']:.3e}"
              + (f", ZI `sigma^2_state` = {fmt(r['sigma2_zi_state'])}"
                 if r.get('sigma2_zi_state') is not None else '')
              + f". Dispersion parameter = {fmt(r['dispersion'], '.4f')} "
                f"(this is theta for NB2, **not** a residual variance).")
            A('')
            if r.get('zi_prob_mean') is not None:
                A(f"Sample-mean structural-zero probability = "
                  f"{fmt(r['zi_prob_mean'], '.4f')}; observed zero rate = "
                  f"{r['obs_zeros'] / r['n_obs']:.4f}; simulated zeros = "
                  f"{fmt(r.get('exp_zeros'), '.1f')} of {r['n_obs']}.")
                A('')
            A('**Conditional block -- the COUNT model.** These describe the '
              'expected count *given the state-year is not a structural '
              'zero*. A positive `b` means MORE events.')
            A('')
            A(coef_block_md(r, 'cond'))
            A('')
            A('**Zero-inflation block -- the log-odds of being a STRUCTURAL '
              'ZERO.** A positive `b` means MORE structural zeros, i.e. FEWER '
              'events. This is the opposite direction from the block above.')
            A('')
            A(coef_block_md(r, 'zi'))
            A('')

    # ---------------------------------------------------------------- 7
    A('## 7. How to read the two blocks (please read this before '
      'interpreting section 6)')
    A('')
    A('A zero-inflated count model fits **two equations at once**, and their '
      'coefficients point in **opposite directions**. This is the single '
      'easiest thing to get backwards.')
    A('')
    A('| | Conditional (count) block | Zero-inflation block |')
    A('|---|---|---|')
    A('| What it models | the expected count, given the observation is not a '
      'structural zero | the log-odds that the observation is a **structural '
      'zero** |')
    A('| Link | log | logit |')
    A('| Exponentiated coefficient | **IRR** -- multiplicative change in the '
      'expected count per one-unit increase in the predictor (= 1 SD for the '
      '`_z` covariates, which are z-scored; `time` and `log(inspections)` are '
      'NOT) | **OR** -- multiplicative change in the odds of being a '
      'structural zero, same units |')
    A('| A **positive** coefficient means | **MORE** events | **MORE '
      'structural zeros**, i.e. **FEWER** events |')
    A('| A **negative** coefficient means | **FEWER** events | **FEWER** '
      'structural zeros, i.e. **MORE** events |')
    A('')
    A('So a predictor with a **positive** conditional coefficient and a '
      '**positive** zero-inflation coefficient is pulling in two directions: '
      'it raises counts among the states that report at all, while also making '
      'a state more likely to be one that structurally reports nothing. The '
      'two blocks are not two estimates of the same thing and they should '
      'never be averaged, summed or read as agreeing/disagreeing.')
    A('')
    A('Worked reading of one row: a zero-inflation coefficient of `b = 0.50` '
      'is `OR = exp(0.50) = 1.65` -- a one-unit increase in that predictor '
      '(one SD, if it is one of the `_z` covariates) multiplies the odds of a '
      'state-year being a **structural zero** by 1.65, which is a statement '
      'about **fewer** recorded events, not more.')
    A('')

    with open(MEMO, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    return MEMO


def main():
    res = load()
    path = write_memo(res)
    cells = cells_of(res)
    print(f'wrote {path}')
    print()
    print('Headline series (zi_intercept) -- sigma^2_state and Delta %:')
    for cell in cells:
        ser = headline_series(res, cell)
        for row in variance_rows(res, cell, ser):
            print(f"  {cell:14s} {row['model']:10s} N={row['n_obs']} "
                  f"states={row['n_states']} "
                  f"s2_state={fmt(row['sigma2_state'])} "
                  f"delta={fmt(row['delta_pct'], '+.1f')} "
                  f"delta_matched={fmt(row['delta_pct_matched'], '+.1f')} "
                  f"conv={row['converged']}")
    nc = nonconverged(res)
    print()
    print(f'non-converged fits: {len(nc)}')
    for x in nc:
        print(f"  {x['key']}  (role={x['role']})")


if __name__ == '__main__':
    main()
