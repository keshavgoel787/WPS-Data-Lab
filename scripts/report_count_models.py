"""
Reads data/generated/count_model_results.json and produces the selection
evidence and coefficient tables for the ZINB count models (Task 5 of the
2026-08-20 multilevel-ZINB spec).

CONTRACT NOTE (supersedes the original task-5-brief.md, which was written
before Task 4 evolved the JSON across four commits and six controller
rulings -- see task-5-report.md for the full list of places the brief's code
would silently produce a wrong table if transcribed):

  - 109 total records = 2 Gaussian round-trip validation artifacts
    (*_gaussian, no cell/model/family_tag -- excluded) + 6 __meta records
    (per-cell selection results, NOT fits -- excluded from selection_table,
    read separately via load_meta) + 6 __altopt optimizer-stability
    diagnostics (excluded from selection) + 95 genuine model fits.
  - The winner is READ from `is_winner_rs` / `is_winner_ri` on each fit, never
    recomputed by a cross-tier `groupby('cell')['aic'].idxmin()` -- that rule
    conflates random-effects structure with distribution family and returns
    'zinb_re' for viol_off_2021/viol_cov_2021 where the real, tier-respecting
    winner is 'nbinom1' (see validate_count_models.py [4] for the proof).
  - `is_winner_rs` is True on TWO records per cell (M1 winner + M3 winner) --
    always filter model == 'M3' for the headline family (`is_m3_winner`).
  - Two cells (viol_off_2021, viol_cov_2021) carry a SECOND (1 | state)
    winner because no ZI-NB rung could converge at (1 + time | state). Both
    tiers are reported, labelled by re_tier, and never merged into one row.
    They are also NOT AIC-comparable to each other (different RE structure) --
    the per-cell printout sorts rs before ri and prints an explicit warning
    between the two blocks so a reader cannot read across the boundary.
  - zi_degenerate fits (11 of them; ZI intercepts numerically at the boundary,
    loglik identical to their non-ZI counterpart) are kept in the table but
    excluded from every LRT (structurally, via `eligible_for_selection` and
    `collapsed_to`, not merely because their df happens to coincide) and from
    every "better zero-fit" comparison.
  - AIC/LRT comparisons are only ever drawn within one (cell, model, re_tier)
    group, after checking n_obs is constant there.
  - The nearest clean competitor is reported unconditionally (no delta_aic
    <= 10 cutoff) -- viol_cov_2021's nearest clean competitor sits at ~10.74
    AIC, just past that cutoff, and is exactly the decision-relevant case.
  - `sigma2_u0`, `sigma2_u1`, `sigma_u01`, `sigma2_e` are carried into every
    row of `selection_table()` (and therefore into the CSV and the per-cell
    printout) -- e.g. the NB1/NB2 disagreement on sigma2_u1 in the 2021
    violations cells is directly readable from the artifact, not only
    assertable inside the validator.
  - `exp_zeros_se` (the Monte-Carlo SE of the simulated expected-zero count,
    2000 draws) travels with every zero count printed or written -- an
    expected-zero count is never reported bare.
  - `winner_summary()` is written to its own CSV
    (count_model_winner_summary.csv) carrying the zero-fit signed
    discrepancy, `has_better_zero_fit` / `credible_better_zero_fit`, the
    M1-vs-M3 eligible candidate sets, and `m1_m3_candidate_set_stable`
    (== meta's `stable_{tier}` -- this is whether the M1 winner and the M3
    winner AGREE, i.e. candidate-set stability, NOT optimizer stability; the
    `__altopt` records independently confirm the rs-tier nbinom1 winners ARE
    optimizer-stable in both 2021 violations cells, sigma2_u1 matching to
    3-4 decimals under BFGS).
  - Zero-inflated winners (zip/zinb/zinb_re) get their zero-inflation-part
    coefficients (`f['zi']`) in a separate, clearly labelled "ZI:" block in
    `coefficient_table()` -- e.g. insp_2019's winner is zinb_re and its ZI
    intercept (-9.423, SE 2.603, p=2.9e-4) is exactly the parameter behind
    its whole 4-AIC margin over plain ZINB; presenting that model as if it
    were an ordinary NB would hide that.

Selection protocol (spec 5.4): AIC/BIC across the six families available at
each tier, LRTs on the nested pairs ONLY -- Poisson subset NB2 (dispersion ->
infinity boundary), NB2 subset ZINB (zero-inflation probability -> 0
boundary), and ZINB subset ZINB+ZI-RE (zero-inflation random-effect variance
-> 0 boundary, 8 of the 30 LRTs here, including the two borderline p-values
in the whole set: insp_2019 M1/M3 at p=0.0138/0.0141). NB1 is not nested in
NB2 and gets AIC/BIC only, no LRT. ALL THREE pairs are boundary tests: the
true null-distribution reference is a 1/2 chi-sq_0 + 1/2 chi-sq_1 mixture, not
a plain chi-sq_1, so the nominal p-value from chi-sq_1 is roughly 2x too
large -- CONSERVATIVE, full stop (there is no sense in which it is
anti-conservative). `lrt_boundary`/`lrt_boundary_kind` columns mark every
computed LRT so a reader is never handed a bare nominal p. No Vuong test: it
is routinely misapplied to non-nested ZI comparisons.

DIAGNOSTIC (from task review, belongs in the Task 7 memo): the apparent NB1
(2021 violations) vs NB2 (2019 violations) reversal is coherent, not a bug.
Regressing log(within-state variance) on log(within-state mean) gives slope
~2.01 for 2019 violations (pure NB2 quadratic mean-variance scaling) versus
~1.46 for 2021 (closer to NB1's linear phi*mu scaling), with zero fractions
2.2% vs 17.3% -- different outcome windows genuinely have different
mean-variance scaling, not an inconsistency in the selection procedure. This
number is attributed to the independent reviewer's calculation, not
re-derived here.

Run: python3 scripts/report_count_models.py
Output: data/generated/count_model_comparison.csv,
        data/generated/count_model_winner_summary.csv
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

GEN = '/Users/keshavgoel/Research/data/generated/'
DOCS = '/Users/keshavgoel/Research/docs/'

FAMILY_LABEL = {'poisson': 'Poisson', 'nbinom1': 'NB1', 'nbinom2': 'NB2',
                'zip': 'ZIP', 'zinb': 'ZINB', 'zinb_re': 'ZINB + ZI RE'}

# Nested pairs only. NB1 is not nested in NB2, so it gets AIC/BIC and no LRT.
# Keyed by the CHILD family; each is a genuine boundary test (see docstring).
NESTED = {'nbinom2': 'poisson', 'zinb': 'nbinom2', 'zinb_re': 'zinb'}
BOUNDARY_KIND = {
    'nbinom2': 'dispersion -> infinity (Poisson is NB2 at phi->inf)',
    'zinb': 'zero-inflation probability -> 0',
    'zinb_re': 'zero-inflation random-effect variance -> 0',
}

TERM_LABELS = [
    ('(Intercept)', 'Intercept'),
    ('log_inspections', 'log Inspections'),
    ('time', 'Time'), ('time2', 'Time^2'), ('time3', 'Time^3'),
    ('SPEND_APP_z', 'Spending/applicator'), ('SPEND_WORK_z', 'Spending/farmworker'),
    ('lii_2017_z', 'Labor Intensity'),
    ('h2a_per_farmworker_z', 'H-2A:farmworker (yr-matched)'),
    ('dol_demand_met_pct_z', 'H-2A demand-met %'), ('pct_flc_z', '% H-2A to FLC'),
]

# Row-ordering priority for the per-cell printout: the rs tier (the primary,
# mandated (1 + time | state) structure) always prints before the ri
# fallback tier, and the two must never be read as one AIC-ranked list.
TIER_ORDER = {'rs': 0, 'ri': 1}

# Key-suffix exclusions applied everywhere a "genuine fit" is required.
_EXCLUDE_SUFFIXES = ('_gaussian', '__meta', '__altopt')


def _is_genuine_fit_key(key):
    return not any(key.endswith(s) for s in _EXCLUDE_SUFFIXES)


def load_results(path=GEN + 'count_model_results.json'):
    """Raw JSON, completely unfiltered (109 records). Callers that iterate
    this as if every value were a fit will KeyError on the 6 __meta records
    (they carry no 'cell'/'model'/'family_tag') -- use selection_table() for
    genuine fits and load_meta() for the per-cell selection results."""
    return json.load(open(path))


def load_meta(raw):
    """Per-cell __meta records, keyed by cell (suffix stripped)."""
    return {k[:-len('__meta')]: v for k, v in raw.items() if k.endswith('__meta')}


def selection_table(raw):
    """One row per genuine model fit (95 rows out of 109 total records):
    excludes the 2 Gaussian round-trip records, the 6 __meta records (not
    fits), and the 6 __altopt diagnostic refits. Carries the full random-
    effects variance block (sigma2_u0/sigma2_u1/sigma_u01/sigma2_e) and the
    convergence-diagnostic fields (message/pd_hess/conv_code) on every row,
    and adds nested-pair boundary LRTs, computed only within
    (cell, model, re_tier) between converged, non-degenerate, non-collapsed,
    selection-eligible fits.
    """
    rows = []
    for key, f in raw.items():
        if not _is_genuine_fit_key(key):
            continue
        obs, exp = f.get('obs_zeros'), f.get('exp_zeros')
        rows.append({
            'key': key,
            'cell': f['cell'], 'outcome': f['outcome'], 'window': f['window'],
            'exposure': f['exposure'], 'model': f['model'],
            'family': f['family_tag'], 're_tier': f['re_tier'],
            'converged': bool(f.get('converged')),
            'message': f.get('message', '') or '',
            'pd_hess': f.get('pd_hess'), 'conv_code': f.get('conv_code'),
            'n_obs': f['n_obs'], 'n_states': f['n_states'],
            'aic': f.get('aic'), 'bic': f.get('bic'),
            'loglik': f.get('loglik'), 'df': f.get('df'),
            'sigma2_u0': f.get('sigma2_u0'), 'sigma2_u1': f.get('sigma2_u1'),
            'sigma_u01': f.get('sigma_u01'), 'sigma2_e': f.get('sigma2_e'),
            'obs_zeros': obs, 'exp_zeros': exp, 'exp_zeros_se': f.get('exp_zeros_se'),
            'zero_ratio': (exp / obs) if (obs not in (None, 0) and exp is not None) else np.nan,
            'zero_fit_discrepancy': f.get('zero_fit_discrepancy'),
            'zi_degenerate': bool(f.get('zi_degenerate', False)),
            'zi_degenerate_reason': f.get('zi_degenerate_reason', ''),
            'collapsed_to': f.get('collapsed_to') or '',
            'eligible_for_selection': bool(f.get('eligible_for_selection', False)),
            'is_winner_rs': bool(f.get('is_winner_rs', False)),
            'is_winner_ri': bool(f.get('is_winner_ri', False)),
        })
    tab = pd.DataFrame(rows)
    tab['winner'] = tab['is_winner_rs'] | tab['is_winner_ri']
    # The headline, single-row-per-cell-per-tier family: True only at M3,
    # where the manuscript's family choice actually applies (is_winner_rs/ri
    # are also True at M1 -- see the module docstring).
    tab['is_m3_winner'] = (tab['model'] == 'M3') & tab['winner']

    # LRTs on nested pairs, within the same (cell, model, re_tier). Lookup is
    # keyed on the FIELDS (cell, model, family, re_tier), never on parsing the
    # JSON key -- the __ri2 suffix convention is an R-script storage detail,
    # not part of the identity of a fit (see validate_ladder()'s tier_fit()).
    lookup = {(r['cell'], r['model'], r['family'], r['re_tier']): i
              for i, r in tab.iterrows()}
    tab['lrt_vs'] = ''
    tab['lrt_boundary'] = False
    tab['lrt_boundary_kind'] = ''
    for c in ('lrt_chisq', 'lrt_df', 'lrt_p'):
        tab[c] = np.nan
    for i, r in tab.iterrows():
        parent_family = NESTED.get(r['family'])
        if parent_family is None:
            continue
        # Structural exclusions -- not merely "happens to be caught by df<=0
        # or an equal loglik": a degenerate, collapsed, or selection-
        # ineligible fit is never a valid LRT participant on either side.
        if (r['zi_degenerate'] or not r['converged']
                or r['collapsed_to'] or not r['eligible_for_selection']):
            continue
        pidx = lookup.get((r['cell'], r['model'], parent_family, r['re_tier']))
        if pidx is None:
            continue
        prow = tab.loc[pidx]
        if (not prow['converged'] or prow['zi_degenerate']
                or prow['collapsed_to'] or not prow['eligible_for_selection']):
            continue
        if r['n_obs'] != prow['n_obs']:
            continue  # AIC/LRT comparisons require the same analytic sample
        chisq = 2 * (r['loglik'] - prow['loglik'])
        ddf = r['df'] - prow['df']
        if ddf <= 0 or not np.isfinite(chisq):
            continue
        tab.loc[i, 'lrt_vs'] = parent_family
        tab.loc[i, 'lrt_chisq'] = chisq
        tab.loc[i, 'lrt_df'] = ddf
        tab.loc[i, 'lrt_p'] = float(stats.chi2.sf(max(chisq, 0), ddf))
        tab.loc[i, 'lrt_boundary'] = True
        tab.loc[i, 'lrt_boundary_kind'] = BOUNDARY_KIND.get(r['family'], '')
    return tab


def nearest_clean_competitor(tab, cell, model, re_tier, winner_family):
    """Among converged, non-degenerate, non-collapsed, selection-eligible
    fits at (cell, model, re_tier) other than the winner, return
    (family, delta_aic) for the smallest AIC gap above the winner.
    Reported UNCONDITIONALLY -- no delta_aic <= 10 cutoff is applied here
    (that cutoff is the meta JSON's own choice and it misses
    viol_cov_2021's most decision-relevant competitor at ~10.74)."""
    win = tab[(tab['cell'] == cell) & (tab['model'] == model) &
              (tab['re_tier'] == re_tier) & (tab['family'] == winner_family)]
    if win.empty or pd.isna(win['aic'].iloc[0]):
        return None, np.nan
    win_aic = float(win['aic'].iloc[0])
    g = tab[(tab['cell'] == cell) & (tab['model'] == model) &
            (tab['re_tier'] == re_tier) & tab['converged'] &
            (~tab['zi_degenerate']) & (tab['collapsed_to'] == '') &
            tab['eligible_for_selection'] &
            (tab['family'] != winner_family)].dropna(subset=['aic']).copy()
    if g.empty:
        return None, np.nan
    g['delta'] = g['aic'] - win_aic
    row = g.loc[g['delta'].idxmin()]
    return row['family'], float(row['delta'])


def winner_summary(tab, meta):
    """One row per (cell, tier) that actually has a winner: 6 rows at the rs
    tier (one per cell) plus 2 more at the ri tier for the two cells that
    needed a second RE structure -- 8 rows total, never merged across tiers.
    Carries the full variance block, the zero-fit signed discrepancy with its
    Monte-Carlo SE, the credible-competitor zero-fit flag, the M1-vs-M3
    eligible candidate sets, and `m1_m3_candidate_set_stable` (whether the M1
    and M3 winners at this tier AGREE -- a candidate-set-stability statement,
    NOT an optimizer-stability statement; see the module docstring).
    """
    rows = []
    for cell, m in meta.items():
        for tier, flag in (('rs', 'is_winner_rs'), ('ri', 'is_winner_ri')):
            w = tab[(tab['cell'] == cell) & (tab['model'] == 'M3') & tab[flag]]
            if w.empty:
                continue
            w = w.iloc[0]
            comp_family, delta_aic = nearest_clean_competitor(
                tab, cell, 'M3', tier, w['family'])
            rows.append({
                'cell': cell, 're_tier': tier, 'winner': w['family'],
                'aic': w['aic'], 'converged': w['converged'],
                'message': w['message'], 'pd_hess': w['pd_hess'],
                'conv_code': w['conv_code'],
                'has_warning': bool(w['message']),
                'sigma2_u0': w['sigma2_u0'], 'sigma2_u1': w['sigma2_u1'],
                'sigma_u01': w['sigma_u01'], 'sigma2_e': w['sigma2_e'],
                'obs_zeros': w['obs_zeros'], 'exp_zeros': w['exp_zeros'],
                'exp_zeros_se': w['exp_zeros_se'], 'zero_ratio': w['zero_ratio'],
                'zero_signed': m.get(f'winner_{tier}_zero_signed'),
                'zero_discrepancy_rel': m.get(f'winner_{tier}_zero_discrepancy'),
                'nearest_clean_competitor': comp_family,
                'delta_aic_vs_competitor': delta_aic,
                'zeros_partly_manufactured': m.get('zeros_partly_manufactured'),
                'has_better_zero_fit': m.get(f'has_better_zero_fit_{tier}'),
                'credible_better_zero_fit': m.get(f'credible_better_zero_fit_{tier}'),
                'credible_better_zero_fit_discrepancy':
                    m.get(f'credible_better_zero_fit_{tier}_discrepancy'),
                'm1_m3_candidate_set_stable': m.get(f'stable_{tier}'),
                'eligible_m1': ','.join(m.get(f'eligible_{tier}_m1') or []),
                'eligible_m3': ','.join(m.get(f'eligible_{tier}_m3') or []),
            })
    return pd.DataFrame(rows)


def coefficient_table(raw, cell, re_tier):
    """Coefficient table for one cell at one RE tier. For M1/M3 this picks
    the fit marked is_winner_rs (re_tier='rs') or is_winner_ri (re_tier='ri')
    directly off the fit records -- never off __meta or a key-parsing
    heuristic.

    M2 quality guard: M2 is only ever fit at the rs tier (validate_ladder()
    checks this), so re_tier='ri' correctly omits M2. At the rs tier, M2 is
    only included if it is `converged` and not `zi_degenerate` -- a non-
    converged or ZI-boundary-degenerate M2 must never reach a coefficient
    table with significance stars attached. NOTE: `eligible_for_selection` is
    NOT used as an M2 gate -- every M2 fit in this JSON carries
    eligible_for_selection == False by construction (M2 is fit only under
    the already-selected family, so it never entered the cross-family
    selection pool that field marks); gating on it would silently drop M2
    from every cell. A defensive assertion raises if more than one
    quality-passing M2 fit is found for a cell (should never happen; the
    ladder produces exactly one M2 family per cell).

    Because the winning family can differ between M1 and M3 within the same
    tier (viol_off_2021's rs tier: M1 winner is zinb_re, M3 winner is
    nbinom1 -- a candidate-set change, not a preference reversal, see
    task-5-report.md), each populated column also carries its own
    `{model}_family` so a reader is never shown two different families'
    coefficients side by side without knowing it.

    Zero-inflated winners (zip/zinb/zinb_re) get a second block of rows,
    labelled "ZI: <term>", built from `f['zi']` -- the zero-inflation part's
    own intercept/coefficients, SEs, and p-values. This is never merged into
    the main (count-model) term rows.
    """
    winner_flag = 'is_winner_rs' if re_tier == 'rs' else 'is_winner_ri'
    picked = {}
    for key, f in raw.items():
        if not _is_genuine_fit_key(key):
            continue
        if f.get('cell') != cell or f.get('re_tier') != re_tier:
            continue
        if f.get('model') in ('M1', 'M3') and f.get(winner_flag):
            picked[f['model']] = (f['family_tag'], f)

    if re_tier == 'rs':
        m2_candidates = [
            f for key, f in raw.items()
            if _is_genuine_fit_key(key) and f.get('cell') == cell
            and f.get('re_tier') == 'rs' and f.get('model') == 'M2'
            and f.get('converged') and not f.get('zi_degenerate', False)
        ]
        if len(m2_candidates) > 1:
            raise ValueError(
                f"{cell}: {len(m2_candidates)} quality-passing M2 fits found "
                f"at the rs tier, expected at most 1: "
                f"{[c['family_tag'] for c in m2_candidates]}")
        if m2_candidates:
            picked['M2'] = (m2_candidates[0]['family_tag'], m2_candidates[0])

    rows = []
    for term, label in TERM_LABELS:
        row = {'term': label, 'section': 'count'}
        present = False
        for m in ('M1', 'M2', 'M3'):
            if m not in picked:
                continue
            fam, f = picked[m]
            row[f'{m}_family'] = fam
            c = f.get('cond', {}).get(term)
            if c is None:
                continue
            present = True
            row[f'{m}_b'] = c['b']
            row[f'{m}_se'] = c['se']
            row[f'{m}_p'] = c['p']
            row[f'{m}_irr'] = float(np.exp(c['b']))
        if present:
            rows.append(row)

    # Zero-inflation part, only for zip/zinb/zinb_re winners. Term set is
    # gathered dynamically from whatever `zi` dicts the picked models carry
    # (in this project always just '(Intercept)'), never assumed.
    term_label_map = dict(TERM_LABELS)
    zi_terms, seen = [], set()
    for m, (fam, f) in picked.items():
        for term in (f.get('zi') or {}):
            if term not in seen:
                seen.add(term)
                zi_terms.append(term)
    for term in zi_terms:
        row = {'term': 'ZI: ' + term_label_map.get(term, term), 'section': 'zi'}
        present = False
        for m in ('M1', 'M2', 'M3'):
            if m not in picked:
                continue
            fam, f = picked[m]
            row[f'{m}_family'] = fam
            c = (f.get('zi') or {}).get(term)
            if c is None:
                continue
            present = True
            row[f'{m}_b'] = c['b']
            row[f'{m}_se'] = c['se']
            row[f'{m}_p'] = c['p']
            row[f'{m}_irr'] = float(np.exp(c['b']))
        if present:
            rows.append(row)

    return pd.DataFrame(rows)


def stars(p):
    if p is None or not np.isfinite(p):
        return ''
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '+' if p < .10 else ''


def num(x, fmt='.3f', width=0, blank=''):
    """Format-or-blank helper. Used throughout main() instead of the brief's
    fragile chained conditional inside an f-string (which mixed a ternary
    with adjacent f-string literals in a way that silently produced the
    wrong string for non-finite values)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        s = blank
    else:
        s = format(x, fmt)
    return s.rjust(width) if width else s


def fmt_p(p, digits=3):
    """A p-value is never exactly 0 -- an underflowed chi2.sf() returns 0.0
    in float, which is a representation limit, not a claim of impossibility.
    Printed (never stored -- the CSV keeps the raw float) as '<1e-300'."""
    if p is None or not np.isfinite(p):
        return ''
    if p == 0.0:
        return '<1e-300'
    return f'{p:.{digits}g}'


def _row_join(cells):
    """Join preformatted, already-padded column strings with a guaranteed
    single-space separator -- fixes the column-overflow garbling where a
    right-justified field whose content exceeds its width (e.g. a long LRT p
    string with stars) ran directly into the next field with no gap."""
    return ' '.join(cells)


def main():
    raw = load_results()
    meta = load_meta(raw)
    tab = selection_table(raw)
    tab.to_csv(GEN + 'count_model_comparison.csv', index=False)
    print(f"Wrote {GEN}count_model_comparison.csv "
          f"({len(tab)} genuine fits; excludes 2 Gaussian round-trip + "
          f"6 __meta + 6 __altopt records from the 109 total)")

    ws = winner_summary(tab, meta)
    ws.to_csv(GEN + 'count_model_winner_summary.csv', index=False)
    print(f"Wrote {GEN}count_model_winner_summary.csv ({len(ws)} winner rows: "
          f"6 cells x rs tier + 2 cells with a second ri-tier winner)\n")

    print("=" * 100)
    print("WINNERS BY CELL AND RE TIER (rs and ri tiers reported separately, never conflated)")
    print("=" * 100)
    for _, r in ws.iterrows():
        print(f"{r['cell']:16} tier={r['re_tier']:2}  "
              f"winner={FAMILY_LABEL.get(r['winner'], r['winner']):14}"
              f"AIC={num(r['aic'], '.1f', 9)}  "
              f"nearest clean competitor={str(r['nearest_clean_competitor']):8} "
              f"(dAIC={num(r['delta_aic_vs_competitor'], '.2f', 6)})  "
              f"warning={r['has_warning']}  manufactured_zeros={r['zeros_partly_manufactured']}")
        print(f"{'':16}         "
              f"sigma2_u0={num(r['sigma2_u0'], '.4f', 8)}  sigma2_u1={num(r['sigma2_u1'], '.5f', 9)}  "
              f"obs0={num(r['obs_zeros'], '.0f', 4)}  exp0={num(r['exp_zeros'], '.1f', 7)} "
              f"+/-{num(r['exp_zeros_se'], '.2f', 5)} (MC SE)  "
              f"zero_signed={num(r['zero_signed'], '+.1f', 8)}  "
              f"better_zero_fit_exists={r['has_better_zero_fit']}"
              + (f" (competitor={r['credible_better_zero_fit']})"
                 if r['has_better_zero_fit'] else ""))
        print(f"{'':16}         "
              f"m1_m3_candidate_set_stable={r['m1_m3_candidate_set_stable']}  "
              f"eligible@M1=[{r['eligible_m1']}]  eligible@M3=[{r['eligible_m3']}]")
    print()

    for cell in sorted(meta):
        g = tab[(tab['cell'] == cell) & (tab['model'] == 'M3')].copy()
        g['_tier_rank'] = g['re_tier'].map(TIER_ORDER)
        g = g.sort_values(['_tier_rank', 'aic'], na_position='last')
        r0 = g.iloc[0]
        print("=" * 100)
        print(f"{cell} -- {r0['outcome']} {r0['window']}, exposure={r0['exposure']}, "
              f"M3, all RE tiers")
        print("=" * 100)
        tiers_present = list(dict.fromkeys(g['re_tier']))  # order-preserving
        header = _row_join([
            f"{'tier':4}", f"{'family':13}", f"{'conv':>4}", f"{'N':>4}",
            f"{'AIC':>9}", f"{'BIC':>9}", f"{'s2_u0':>7}", f"{'s2_u1':>8}",
            f"{'obs0':>5}", f"{'exp0':>7}", f"{'0-ratio':>7}",
            f"{'LRT vs':>8}", f"{'LRT p':>10}", f"{'degen':>6}", 'winner/collapsed',
        ])
        print(header)
        last_tier = None
        for _, r in g.iterrows():
            if last_tier is not None and r['re_tier'] != last_tier:
                print("  " + "-" * 96)
                print("  ^^ TIER BOUNDARY -- rows above and below use different random-effects "
                      "structures (rs = (1+time|state), ri = (1|state)).")
                print("  AIC IS NOT COMPARABLE ACROSS THIS LINE. Each tier's winner is chosen "
                      "only against its own tier.")
                print("  " + "-" * 96)
            last_tier = r['re_tier']
            lrt_p_str = fmt_p(r['lrt_p']) + stars(r['lrt_p']) if pd.notna(r['lrt_p']) else ''
            mark = []
            if r['is_winner_rs']:
                mark.append('rs winner')
            if r['is_winner_ri']:
                mark.append('ri winner')
            if r['collapsed_to']:
                mark.append(f"collapsed_to={r['collapsed_to']}")
            row_str = _row_join([
                f"{r['re_tier']:4}", f"{FAMILY_LABEL.get(r['family'], r['family']):13}",
                f"{'yes' if r['converged'] else 'NO':>4}", num(r['n_obs'], '.0f', 4),
                num(r['aic'], '.1f', 9), num(r['bic'], '.1f', 9),
                num(r['sigma2_u0'], '.3f', 7), num(r['sigma2_u1'], '.4f', 8),
                num(r['obs_zeros'], '.0f', 5), num(r['exp_zeros'], '.1f', 7),
                num(r['zero_ratio'], '.2f', 7),
                f"{r['lrt_vs']:>8}", f"{lrt_p_str:>10}",
                f"{'YES' if r['zi_degenerate'] else '':>6}",
                ('<-- ' + ', '.join(mark)) if mark else '',
            ])
            print(row_str)
        print()

    for cell in sorted(meta):
        for tier in ('rs', 'ri'):
            ct = coefficient_table(raw, cell, tier)
            if ct.empty:
                continue
            fam_cols = {m: ct[f'{m}_family'].iloc[0] for m in ('M1', 'M2', 'M3')
                        if f'{m}_family' in ct.columns}
            print("=" * 100)
            header = ", ".join(f"{m}={FAMILY_LABEL.get(f, f)}" for m, f in fam_cols.items())
            print(f"COEFFICIENTS -- {cell} ({tier}-tier)   [{header}]   b(SE), IRR=exp(b)")
            print("=" * 100)
            cols = [m for m in ('M1', 'M2', 'M3') if m in fam_cols]
            print(f"{'term':30}" + "".join(f"{m + ' b(SE)':>20}{m + ' IRR':>10}" for m in cols))
            last_section = None
            for _, r in ct.iterrows():
                if r.get('section') != last_section:
                    label = ('-- Count-model part --' if r.get('section') == 'count'
                             else '-- Zero-inflation part --')
                    print(label)
                    last_section = r.get('section')
                cells_out = []
                for m in cols:
                    b, se, p, irr = r.get(f'{m}_b'), r.get(f'{m}_se'), r.get(f'{m}_p'), r.get(f'{m}_irr')
                    if pd.notna(b):
                        cells_out.append((f"{b:.3f}({se:.3f}){stars(p)}", f"{irr:.3f}"))
                    else:
                        cells_out.append(('', ''))
                print(f"{r['term']:30}" +
                      "".join(f"{bse:>20}{irr:>10}" for bse, irr in cells_out))
            print()

    print("=" * 100)
    print("NOTE: all three nested-pair LRTs are BOUNDARY tests --")
    print("  Poisson < NB2  : dispersion -> infinity")
    print("  NB2 < ZINB     : zero-inflation probability -> 0")
    print("  ZINB < ZINB+RE : zero-inflation random-effect variance -> 0")
    print("The true null-reference distribution for a boundary test is a 1/2 chi-sq_0 +")
    print("1/2 chi-sq_1 mixture, not chi-sq_1 -- so every nominal p above is CONSERVATIVE")
    print("(roughly 2x too large), full stop. NB1 is not nested in NB2: AIC/BIC only, no LRT.")
    print("zi_degenerate fits (ZI intercept numerically at the boundary; 11 of them, all in the")
    print("2019 violations cells) are shown but structurally excluded from every LRT (both as")
    print("child and parent) and from every best-zero-fit comparison.")
    print("=" * 100)


if __name__ == '__main__':
    main()
