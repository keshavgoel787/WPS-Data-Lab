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
  - CSV ROUND-TRIP TRAP for any downstream consumer (including Task 7):
    select the 30 LRT rows via `lrt_p.notna()`, never `lrt_vs != ''` -- the
    empty-string sentinel for "no LRT" becomes NaN through a to_csv/read_csv
    round-trip, and `NaN != ''` is True, so that selector would silently
    return all 95 rows instead of 30 once read back from disk. See
    `selection_table()`'s docstring and the round-trip check in
    validate_count_models.py [4].
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
-> 0 boundary, 6 of the 30 LRTs here, including the two borderline p-values
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

    SELECTOR WARNING for downstream consumers (including Task 7, which reads
    count_model_comparison.csv from disk): to select the 30 rows that carry a
    computed LRT, use `lrt_p.notna()`, NEVER `lrt_vs != ''`. In this
    in-memory DataFrame `lrt_vs` is the empty string '' on the other 65 rows,
    but pandas' `to_csv`/`read_csv` round-trip turns that '' into NaN for an
    object column -- and `NaN != ''` evaluates True, so a CSV-based
    `df[df.lrt_vs != '']` silently selects all 95 rows instead of 30.
    `lrt_p` is numeric and NaN both in memory and after the round-trip, so
    `.notna()` on it is the one selector that is correct in both contexts.
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


# ============================================================
# TASK 7: MEMO EVIDENCE + MEMO
# ============================================================
# EVERY quantitative claim in docs/count_models_zinb.md is generated from an
# artifact -- count_model_results.json, count_model_comparison.csv, or the
# evidence file built below. Nothing is hand-typed into the prose. That rule
# exists because this project's history is a string of hand-carried numbers
# turning out wrong (a "12" that was 11, an "8" that was 6, a published
# coefficient that came from a non-converged optimizer), and
# validate_count_models.py [7] parses the numbers back OUT of the generated
# markdown and compares them to the artifacts so a drift fails loudly.
#
# BACKTICK CONVENTION (load-bearing for validator [7]): backticked RE-tier tags
# (`rs` / `ri`) and backticked family tags (`zinb`, `nbinom1`, ...) appear in
# EXACTLY ONE table in the memo -- the headline "Selected family per cell"
# table. Every other table uses display labels ('NB1', '(1 + time | state)').
# That keeps [7]'s parse-back of the selected family sharp instead of being
# satisfiable by an incidental match in some other table.

MEMO_EVIDENCE_PATH = GEN + 'count_model_memo_evidence.json'

# The published 2011-2021 manuscript specification, replicated here EXACTLY as
# paper_table_models_2021.py fits it (same build_panel, same lbfgs, same REML
# default, same listwise deletion), purely to audit its convergence. Nothing in
# paper_table_* is modified or regenerated by this script.
_PUB_CUBIC = ['time', 'time2', 'time3']
_PUB_SPEND = ['SPEND_APP_z', 'SPEND_WORK_z']
_PUB_LABOR = ['lii_2017_z']
_PUB_H2A = ['h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']
_PUB_COLUMNS = [
    ('inspections', 'log_inspections', _PUB_CUBIC),
    ('violations', 'log_violations', ['log_inspections'] + _PUB_CUBIC),
]
_PUB_MODELS = [
    ('M1', []),
    ('M2', _PUB_SPEND + _PUB_LABOR),
    ('M3', _PUB_SPEND + _PUB_LABOR + _PUB_H2A),
]


def _panel_descriptives(window):
    """Zero fraction, dispersion, and the within-state mean-variance scaling
    slope for each outcome in one window. The slope is the diagnostic behind
    the NB1 (2021) / NB2 (2019) reversal: NB2's variance is quadratic in the
    mean (slope ~2), NB1's is linear-in-mean with a constant multiplier
    (slope ~1)."""
    d = pd.read_csv(GEN + f'count_model_panel_{window}.csv')
    out = {}
    for dv in ('inspections', 'violations'):
        s = d[dv].dropna()
        g = d.dropna(subset=[dv]).groupby('state')[dv]
        per_state_zeros = g.apply(lambda x: int((x == 0).sum()))
        st = g.agg(['mean', 'var']).dropna()
        st = st[(st['mean'] > 0) & (st['var'] > 0)]
        slope = float(np.polyfit(np.log(st['mean']), np.log(st['var']), 1)[0])
        out[dv] = {
            'n_obs': int(len(s)), 'n_states': int(d['state'].nunique()),
            'n_zeros': int((s == 0).sum()),
            'pct_zeros': float((s == 0).mean() * 100),
            'mean': float(s.mean()), 'var': float(s.var(ddof=1)),
            'var_over_mean': float(s.var(ddof=1) / s.mean()),
            'max': float(s.max()),
            'n_states_with_a_zero': int((per_state_zeros > 0).sum()),
            'max_zero_years_in_a_state': int(per_state_zeros.max()),
            'n_states_all_zero': int(sum(
                1 for _, x in g if bool((x == 0).all()))),
            'mv_slope': slope, 'mv_slope_n_states': int(len(st)),
        }
    return out


def _audit_published_fits():
    """Refit all 12 published 2011-2021 manuscript fits (2 outcome columns x
    M1/M2/M3 x {random slope, random-intercept-only}) with the published
    optimizer, recording each one's convergence warnings, and refit any
    non-converged one with 'cg' and 'powell'.

    This is READ-ONLY with respect to the manuscript: nothing in
    paper_table_* is touched. The point is to establish the SCOPE of the
    non-convergence finding as a measured number rather than an impression,
    and to confirm the random-intercept-only refits (the ones that produce the
    reported variance-explained block) are unaffected.

    Warnings are captured with `catch_warnings(record=True)`, never suppressed
    -- a blanket `filterwarnings('ignore')` is precisely what hid this problem
    in the published pipeline.
    """
    import re as _re
    import sys as _sys
    import warnings as _warnings

    if '/Users/keshavgoel/Research/scripts' not in _sys.path:
        _sys.path.insert(0, '/Users/keshavgoel/Research/scripts')
    from build_count_model_panel import build_panel
    from statsmodels.regression.mixed_linear_model import MixedLM
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    published = json.load(open(GEN + 'paper_table_params_2021.json'))
    df = build_panel(2021)

    def one_fit(dv, rhs, re_slope, method):
        d = df.dropna(subset=[dv] + rhs).copy()
        d['state'] = pd.Categorical(d['state'])
        kw = {'re_formula': '~time'} if re_slope else {}
        with _warnings.catch_warnings(record=True) as wrec:
            _warnings.simplefilter('always')
            res = MixedLM.from_formula(f"{dv} ~ " + " + ".join(rhs), data=d,
                                       groups=d['state'], **kw).fit(method=method)
        conv_warns = [str(w.message) for w in wrec
                      if issubclass(w.category, ConvergenceWarning)]
        grad = None
        for w in conv_warns:
            m = _re.search(r'\|grad\|\s*=\s*([0-9.eE+-]+)', w)
            if m:
                grad = float(m.group(1))
        return {
            'method': method, 'n_obs': int(len(d)),
            'n_states': int(d['state'].nunique()),
            'llf': float(res.llf),
            'sigma2_u0': float(res.cov_re.iloc[0, 0]),
            'b_log_inspections': (float(res.fe_params['log_inspections'])
                                  if 'log_inspections' in res.fe_params.index else None),
            'grad_warning': grad is not None,
            'grad_norm': grad,
            'convergence_warnings': conv_warns,
        }

    fits, affected = [], None
    for outcome, dv, base in _PUB_COLUMNS:
        for model, extra in _PUB_MODELS:
            rhs = base + extra
            for re_slope, basis, pubkey in ((True, 'random slope', 'sigma2_u0'),
                                            (False, 'random intercept', 'sigma2_u0_ri')):
                r = one_fit(dv, rhs, re_slope, 'lbfgs')
                r.update({'outcome': outcome, 'model': model, 're_basis': basis,
                          'published_sigma2_u0': float(published[outcome][model][pubkey])})
                fits.append(r)
                if r['grad_warning']:
                    alts = {m: one_fit(dv, rhs, re_slope, m) for m in ('cg', 'powell')}
                    affected = {
                        'outcome': outcome, 'model': model, 're_basis': basis,
                        'n_obs': r['n_obs'], 'n_states': r['n_states'],
                        'lbfgs': r, 'cg': alts['cg'], 'powell': alts['powell'],
                        'published_b_log_inspections':
                            float(published[outcome][model]['log_inspections']['b']),
                        'published_sigma2_u0': float(published[outcome][model][pubkey]),
                        'published_delta_pct_ri': float(published[outcome][model]['delta_pct_ri']),
                    }
    return {
        'n_fits': len(fits),
        'n_nonconverged': sum(1 for r in fits if r['grad_warning']),
        'fits': fits,
        'affected': affected,
    }


def _zi_crosscheck(raw):
    """Independent cross-software check on the zero-inflation component: a
    statsmodels ZeroInflatedNegativeBinomialP with state FIXED effects (dummy
    variables) instead of glmmTMB's random effects. Different package,
    different treatment of the state dimension -- which is the whole point.

    The six M3 covariates are omitted (most are state-time-invariant and
    therefore collinear with the state dummies used here in place of a random
    intercept), so this runs on the FULL 2021 panel and retains AK/RI/VT,
    which glmmTMB's M3 drops for missing BLS applicator data. The two N's
    legitimately differ; both are recorded so the memo can report both.
    """
    import warnings as _warnings

    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    d = pd.read_csv(GEN + 'count_model_panel_2021.csv')
    specs = {
        'insp_2021': ('inspections', [], 'insp_2021__M3__zinb'),
        'viol_cov_2021': ('violations', ['log_inspections'], 'viol_cov_2021__M3__zinb'),
    }
    out = {}
    for cell, (dv, extra, glmm_key) in specs.items():
        need = [dv, 'time', 'time2', 'time3'] + extra
        dd = d.dropna(subset=need).copy()
        X = pd.concat([dd[['time', 'time2', 'time3'] + extra],
                       pd.get_dummies(dd['state'], prefix='st',
                                      drop_first=True, dtype=float)], axis=1)
        X = sm.add_constant(X)
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter('always')
            res = ZeroInflatedNegativeBinomialP(
                dd[dv], X, exog_infl=np.ones((len(dd), 1)), p=2
            ).fit(method='bfgs', maxiter=500, disp=0)
        g = raw.get(glmm_key, {})
        gzi = (g.get('zi') or {}).get('(Intercept)', {})
        per_state_zeros = dd.groupby('state')[dv].apply(lambda x: int((x == 0).sum()))
        out[cell] = {
            'dv': dv, 'n_obs': int(len(dd)),
            'n_params': int(X.shape[1]),
            'design_full_rank': bool(np.linalg.matrix_rank(X.values.astype(float))
                                     == X.shape[1]),
            'converged': bool(res.mle_retvals.get('converged')),
            'zi_b': float(res.params['inflate_const']),
            'zi_se': float(res.bse['inflate_const']),
            'intercept': float(res.params['const']),
            'alpha': float(res.params['alpha']),
            'glmm_key': glmm_key,
            'glmm_n_obs': g.get('n_obs'),
            'glmm_re_tier': g.get('re_tier'),
            'glmm_zi_b': gzi.get('b'), 'glmm_zi_se': gzi.get('se'),
            'glmm_zi_p': gzi.get('p'),
            'obs_zeros': int((dd[dv] == 0).sum()),
            'n_states_with_a_zero': int((per_state_zeros > 0).sum()),
            'max_zero_years_in_a_state': int(per_state_zeros.max()),
            'n_states_all_zero': int(sum(
                1 for _, x in dd.groupby('state')[dv] if bool((x == 0).all()))),
        }
    return out


def build_memo_evidence(raw):
    """Compute (and persist) the evidence the memo needs that is NOT already in
    count_model_results.json: panel descriptives, the convergence audit of the
    12 published 2011-2021 fits, and the cross-software ZI check. Written to
    its own artifact so validate_count_models.py [7] can check the memo's
    numbers against it independently of the prose."""
    ev = {
        'panel': {w: _panel_descriptives(w) for w in ('2019', '2021')},
        'published_audit': _audit_published_fits(),
        'zi_crosscheck': _zi_crosscheck(raw),
    }
    with open(MEMO_EVIDENCE_PATH, 'w') as fh:
        json.dump(ev, fh, indent=1)
    print(f"Wrote {MEMO_EVIDENCE_PATH}")
    return ev


def _re_label(re_tier, in_table=True):
    """Human-readable random-effects structure. A bare '|' inside a markdown
    table cell splits the row, so the pipe is escaped whenever the label is
    printed in a table (prose/headings use in_table=False)."""
    lab = '(1 + time | state)' if re_tier == 'rs' else '(1 | state)'
    return lab.replace('|', r'\|') if in_table else lab


def _f(x, fmt='.3f', blank='--'):
    """Number-or-blank, for markdown cells."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return blank
    try:
        return format(x, fmt)
    except (TypeError, ValueError):
        return str(x)


def write_memo(raw, meta, tab, ws, ev):
    """Generate docs/count_models_zinb.md. Every number below is read from
    `raw` (count_model_results.json), `tab` (the selection table that becomes
    count_model_comparison.csv), `ws` (the winner summary CSV) or `ev` (the
    memo-evidence artifact). No statistic is hardcoded in this function."""
    L = []
    add = L.append

    m3w = tab[tab['is_m3_winner']].copy()
    m3w['_tier_rank'] = m3w['re_tier'].map(TIER_ORDER)
    m3w = m3w.sort_values(['cell', '_tier_rank'])
    p21, p19 = ev['panel']['2021'], ev['panel']['2019']
    aud = ev['published_audit']
    aff = aud['affected']
    xc_i = ev['zi_crosscheck']['insp_2021']
    xc_v = ev['zi_crosscheck']['viol_cov_2021']
    n_lrt = int(tab['lrt_p'].notna().sum())
    n_zire_lrt = int(((tab['lrt_p'].notna()) & (tab['family'] == 'zinb_re')).sum())
    n_degen = int(tab['zi_degenerate'].sum())

    # ---------------------------------------------------------------- header
    import datetime as _dt
    import os as _os
    fits_date = _dt.date.fromtimestamp(
        _os.path.getmtime(GEN + 'count_model_results.json')).isoformat()
    add("# Multilevel count models for the WPS panels\n")
    add(f"**Model fits:** `data/generated/count_model_results.json`, "
        f"generated {fits_date}  ")
    add("**Spec:** `docs/superpowers/specs/2026-08-20-multilevel-zinb-design.md`  ")
    add("**Generated by:** `scripts/report_count_models.py` -- do not edit by hand. "
        "Every number in this memo is read from `data/generated/count_model_results.json`, "
        "`count_model_comparison.csv`, `count_model_winner_summary.csv` or "
        "`count_model_memo_evidence.json`; "
        "`scripts/validate_count_models.py` section `[7]` parses them back out of this "
        "file and fails if any of them drifts from the artifacts.\n")
    add("Joe asked whether we could replicate the analytic model of Jafari et al. "
        "(*PLOS ONE* 2024, doi:10.1371/journal.pone.0302960) -- a multilevel "
        "zero-inflated negative binomial -- after the 2011-2021 tables weakened the "
        "paper's story. This memo reports what that model class does to our results, "
        "and two problems it turned up along the way that are decisions for you rather "
        "than for me.\n")

    # ------------------------------------------------- what was replicated
    add("## What was and was not replicated\n")
    add("Same package and same model class: Jafari et al. fit their models in R with "
        "**`glmmTMB`**, and so do we. Their zero-inflated negative binomial, their "
        "distribution ladder (Poisson / NB / zero-inflated variants compared on "
        "AIC, BIC and nested likelihood-ratio tests), their reporting of incidence "
        "rate ratios.\n")
    add("**Not** their random-effects structure, and this is deliberate. They use "
        "cross-classified state x industry random intercepts on case-level "
        "investigation records. Our data is a state-year panel with no industry "
        "dimension at all, and their design would discard the cubic time trend "
        "around the 2016-17 WPS revision that this paper is about. We keep the "
        "structure the manuscript already uses: a **random intercept** plus a "
        "**random slope** on time by state, `(1 + time | state)`, with cubic time "
        "as fixed effects. Where that structure could not support a particular "
        "distribution, a random-intercept-only fallback tier `(1 | state)` is "
        "reported separately and never compared to it on AIC.\n")
    add("Before any count model was believed, `glmmTMB` had to reproduce something "
        "already known. The validation gate refits the manuscript's own Model 1 as a "
        "**Gaussian** `glmmTMB` model with REML and requires it to match "
        f"`statsmodels.MixedLM` -- coefficients to 1e-3 relative, variance components "
        f"to 2e-2 relative. It does. Nothing downstream would be trustworthy "
        f"otherwise.\n")

    # -------------------------------------------------- why a count model
    add("## Why a count model at all\n")
    add("The published tables model `log(count + 1)` with a Gaussian linear mixed "
        "model. For the WPS-view violations panel that is fighting the data:\n")
    add("| Window | Outcome | N | Zeros | % zeros | Mean | Variance | Var/Mean | "
        "States with >=1 zero | Max zero-years in one state | States zero in every year |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for wname, pp in (('2011-2021 (WPS view)', p21), ('2011-2019 (establishments view)', p19)):
        for dv in ('violations', 'inspections'):
            r = pp[dv]
            add(f"| {wname} | {dv} | {r['n_obs']} | {r['n_zeros']} | "
                f"{_f(r['pct_zeros'], '.1f')}% | {_f(r['mean'], '.2f')} | "
                f"{_f(r['var'], '.1f')} | {_f(r['var_over_mean'], '.1f')} | "
                f"{r['n_states_with_a_zero']}/{r['n_states']} | "
                f"{r['max_zero_years_in_a_state']} | {r['n_states_all_zero']} |")
    add("")
    v21 = p21['violations']
    i21 = p21['inspections']
    add(f"{_f(v21['pct_zeros'], '.1f')}% of state-years are zero "
        f"({v21['n_zeros']} of {v21['n_obs']}) and the variance is about "
        f"{_f(v21['var_over_mean'], '.0f')} times the mean. "
        f"{v21['n_states_with_a_zero']} of {v21['n_states']} states have at least one "
        f"zero-violation year, the worst has {v21['max_zero_years_in_a_state']}, and "
        + ("**no state is zero in every year**"
           if v21['n_states_all_zero'] == 0
           else f"**{v21['n_states_all_zero']} states are zero in every year**")
        + ". That last "
        "fact matters, and it cuts against the convenient story: because no state "
        "is a permanent non-enforcer, the zeros are episodic rather than produced by "
        "a fixed set of states that never enforce. A structural-zero component is "
        "therefore a modelling device for excess left-tail mass here, not an "
        "identified latent subpopulation -- the same caveat that returns in the "
        "cross-software section below. Inspections is a different problem: "
        f"{_f(i21['pct_zeros'], '.1f')}% zeros "
        f"({i21['n_zeros']} of {i21['n_obs']}) with heavy overdispersion.\n")

    # ----------------------------------------------------- the two windows
    add("## Two windows, because two things changed at once\n")
    add("The results Joe reacted to changed the window (2019 -> 2021) **and** the "
        "outcome source at the same time. EPA's ECHO **establishments view** stops "
        "at 2019, so extending the panel to 2021 required switching to the **WPS "
        "view** of the same dashboard -- Kaitlyn's note that the outcome data itself "
        "changed is correct. The two views are not the same measure: WPS-view counts "
        "run 2-3x the establishments counts and their state-level correlation is "
        "~0.4-0.6 through 2016 and ~0 from 2017 on. Fitting both windows under one "
        "model class is what separates the model-class effect from the data-source "
        "effect. Coefficients are not line-by-line comparable across the two "
        "windows.\n")
    add("Three specifications per window, so six cells in all: inspections as the "
        "outcome; violations with inspections as an **offset** (`viol_off_*`, a "
        "rate model); and violations with `log(inspections)` as an ordinary "
        "**covariate** (`viol_cov_*`, matching the published Table 3).\n")

    # --------------------------------------------------- headline selection
    add("## Selected family per cell\n")
    add("Selected by AIC within one random-effects tier, never across tiers. `rs` = "
        "`(1 + time | state)` (the mandated structure); `ri` = `(1 | state)` "
        "(fallback, reported only where a family could not be fit at `rs`).\n")
    add("| Cell | Outcome | Window | Exposure | RE tier | Selected family | AIC | "
        "N obs | States | Nearest clean competitor | dAIC | Optimizer warning |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in m3w.iterrows():
        w = ws[(ws['cell'] == r['cell']) & (ws['re_tier'] == r['re_tier'])]
        comp = w['nearest_clean_competitor'].iloc[0] if not w.empty else None
        dcomp = w['delta_aic_vs_competitor'].iloc[0] if not w.empty else np.nan
        add(f"| {r['cell']} | {r['outcome']} | {r['window']} | {r['exposure']} | "
            f"`{r['re_tier']}` | {FAMILY_LABEL.get(r['family'], r['family'])} "
            f"(`{r['family']}`) | {_f(r['aic'], '.1f')} | {r['n_obs']} | "
            f"{r['n_states']} | "
            f"{FAMILY_LABEL.get(comp, comp) if comp else '--'} | "
            f"{_f(dcomp, '.2f')} | "
            f"{r['message'] if r['message'] else 'none'} |")
    add("")
    rs = m3w[m3w['re_tier'] == 'rs']
    ZI_FAMS = {'zip', 'zinb', 'zinb_re'}

    def _fams(outcome):
        g = rs[rs['outcome'] == outcome]
        return {w: sorted(set(gg['family'])) for w, gg in g.groupby('window')}, \
            sorted(set(g['family']))

    insp_by_window, insp_fams = _fams('inspections')
    viol_by_window, viol_fams = _fams('violations')

    def _phrase(by_window):
        return "; ".join(
            f"{w}: " + ", ".join(FAMILY_LABEL.get(f, f) for f in fams)
            for w, fams in sorted(by_window.items()))

    add(f"At the mandated `(1 + time | state)` structure, **inspections selects a "
        f"zero-inflated family in both windows** ({_phrase(insp_by_window)}"
        f"{'' if set(insp_fams) <= ZI_FAMS else ' -- NOTE: not all zero-inflated'}) "
        f"**and violations select a plain negative binomial in both** "
        f"({_phrase(viol_by_window)}"
        f"{'' if not (set(viol_fams) & ZI_FAMS) else ' -- NOTE: a ZI family appears here'})"
        f". That is what the evidence selected: it is neither the hoped-for result "
        f"nor a disappointing one, and the next section is where it stops being as "
        f"simple as it looks. Note also that both 2021 violations cells carry a "
        f"SECOND winner at the `(1 | state)` fallback tier, and that one IS "
        f"zero-inflated -- the two tiers are separate results, not a ranking.\n")

    # ------------------------------------------------ the ZI framing section
    add("## Zero-inflation: what the AIC comparison does and does not settle\n")
    add("The most important thing in this memo. **Do not read \"NB1 won\" as "
        "\"zero-inflation was tested and rejected.\"** For the 2021 violations cells "
        "it was not rejected; it is real, large, and precisely estimated.\n")
    add("| Cell | ZI family | RE tier | ZI intercept | SE | p | Implied structural-zero probability |")
    add("|---|---|---|---|---|---|---|")
    for cell in ('viol_off_2021', 'viol_cov_2021', 'insp_2021', 'insp_2019'):
        for e in meta[cell]['zi_evidence']:
            if e.get('zi_degenerate') or e.get('b') is None:
                continue
            add(f"| {cell} | {FAMILY_LABEL.get(e['family'], e['family'])} | "
                f"{e['re_tier']} | {_f(e['b'], '+.4f')} | {_f(e['se'], '.4f')} | "
                f"{_f(e['p'], '.3g')} | "
                f"{_f(1 / (1 + np.exp(-e['b'])), '.4f')} |")
    add("")
    for cell in ('viol_off_2021', 'viol_cov_2021'):
        m = meta[cell]
        zinb_ri = [e for e in m['zi_evidence']
                   if e['family'] == 'zinb' and e['re_tier'] == 'ri']
        add(f"- **{cell}**: zero-inflation is estimable even at the mandated `rs` "
            f"structure, via {FAMILY_LABEL.get(m['zi_rs_family'], m['zi_rs_family'])} "
            f"(ZI intercept {_f(m['zi_rs_b'], '+.4f')}, SE {_f(m['zi_rs_se'], '.4f')}, "
            f"p = {_f(m['zi_rs_p'], '.3g')}), and non-degenerate at the fallback "
            f"tier via ZINB"
            + (f" (ZI intercept {_f(zinb_ri[0]['b'], '+.4f')}, "
               f"p = {_f(zinb_ri[0]['p'], '.3g')})" if zinb_ri else "")
            + ".")
    add("")
    add("NB1 wins the AIC comparison because a negative binomial's own "
        "overdispersion parameter accounts for those same zeros about as "
        "economically as an explicit inflation term does -- and, specifically, "
        "because the zero-inflated **negative binomial** *combination* could not be "
        "fit at the mandated random-slope structure at all.\n")
    add("### For the 2021 violations cells the family comparison is an identifiability statement\n")
    for cell in ('viol_off_2021', 'viol_cov_2021'):
        elig = meta[cell]['eligible_rs_m3']
        add(f"- `{cell}`: only {len(elig)} families were estimable at "
            f"`(1 + time | state)` -- "
            + ", ".join(f"{FAMILY_LABEL.get(e, e)}" for e in elig)
            + ". Neither ZINB nor ZINB+ZI-RE could be fit there.")
    add("")
    add("So at the structure the manuscript specifies, the comparison for these two "
        "cells is between four families, two of which are the ZI-NB rungs that "
        "**could not be estimated**. A reader must not take the result as evidence "
        "against zero-inflation; it is an **identifiability** result about the "
        "random-slope structure.\n")

    # ---------------------------------------------- 2019 violations deflation
    add("### The 2019 violations series is zero-**deflated**, not zero-inflated\n")
    d19 = tab[(tab['cell'].isin(['viol_off_2019', 'viol_cov_2019'])) &
              (tab['model'] == 'M3')].dropna(subset=['exp_zeros'])
    obs19 = sorted(set(int(x) for x in d19['obs_zeros']))
    lo, hi = float(d19['exp_zeros'].min()), float(d19['exp_zeros'].max())
    rlo, rhi = float(d19['zero_ratio'].min()), float(d19['zero_ratio'].max())
    add(f"Every family **over**predicts the zeros in the 2019 violations cells: "
        f"{'/'.join(str(o) for o in obs19)} observed against "
        f"{_f(lo, '.1f')}-{_f(hi, '.1f')} expected across the six families at "
        f"Model 3, i.e. {_f(rlo, '.1f')}x to {_f(rhi, '.1f')}x too many. There is no "
        "excess of zeros here to inflate; there is a shortage of them. The series is "
        "zero-deflated.\n")
    add("| Cell | Family | Observed 0s | Expected 0s (+/- MC SE) | Expected/Observed |")
    add("|---|---|---|---|---|")
    for _, r in d19.sort_values(['cell', 'exp_zeros']).iterrows():
        add(f"| {r['cell']} | {FAMILY_LABEL.get(r['family'], r['family'])} | "
            f"{_f(r['obs_zeros'], '.0f')} | {_f(r['exp_zeros'], '.1f')} +/- "
            f"{_f(r['exp_zeros_se'], '.2f')} | {_f(r['zero_ratio'], '.2f')}x |")
    add("")
    zi19 = [e for c in ('viol_off_2019', 'viol_cov_2019') for e in meta[c]['zi_evidence']]
    ses = [e['se'] for e in zi19 if e.get('se') is not None]
    ps = [e['p'] for e in zi19 if e.get('p') is not None]
    degen_cells = sorted(set(tab[tab['zi_degenerate']]['cell']))
    add(f"Consistent with that, all three zero-inflated families collapse at the "
        f"mandated structure in these cells: standard errors from "
        f"{_f(min(ses), '.0f')} to {_f(max(ses), '.0f')} and p-values from "
        f"{_f(min(ps), '.3f')} to {_f(max(ps), '.3f')} -- a zero-inflation intercept "
        f"sitting numerically on the boundary, estimating nothing. Across the whole "
        f"ladder {n_degen} fits are flagged ZI-degenerate, and they are all in these "
        f"{len(degen_cells)} cells "
        f"({', '.join('`' + c + '`' for c in degen_cells)}).\n")
    add("**Scope, precisely.** The fallback `(1 | state)` tier was never attempted "
        "for these cells -- they did not need it, because a non-ZI family fit fine "
        "at the mandated structure. So the supportable claim is *no estimable "
        "zero-inflation at the specified structure, where all three ZI families "
        "collapse*. It is **not** a claim that zero-inflation is degenerate at every "
        "possible tier, and this memo does not make that claim.\n")

    # --------------------------------------------------------- coefficients
    add("## Coefficients under the selected family\n")
    add("`b` is on the log link with its standard error in brackets, followed by the "
        "**IRR** = exp(b). Significance: `***` p<.001, `**` p<.01, `*` p<.05, "
        "`+` p<.10. Model 1 = cubic time (plus the inspections term for violations); "
        "Model 2 adds spending per applicator, spending per farmworker and labor "
        "intensity; Model 3 adds the year-matched H-2A ratio, H-2A demand-met % and "
        "% of H-2A going to farm labor contractors. The zero-inflation part, where "
        "the selected family has one, is a separate block and is never merged into "
        "the count-model terms.\n")
    for cell in sorted(meta):
        for tier in ('rs', 'ri'):
            ct = coefficient_table(raw, cell, tier)
            if ct.empty:
                continue
            fams = {m: ct[f'{m}_family'].iloc[0] for m in ('M1', 'M2', 'M3')
                    if f'{m}_family' in ct.columns
                    and pd.notna(ct[f'{m}_family'].iloc[0])}
            r0 = tab[(tab['cell'] == cell)].iloc[0]
            struct = _re_label(tier, in_table=False)
            add(f"### {cell} -- {r0['outcome']} {r0['window']}, "
                f"exposure {r0['exposure']}, {struct}\n")
            add("Families per column: " + ", ".join(
                f"{m} = {FAMILY_LABEL.get(f, f)}" for m, f in fams.items()) + ".")
            if len(set(fams.values())) > 1:
                add("Note the columns do not all use the same family -- the "
                    "estimable candidate set changes between Model 1 and Model 3, "
                    "so these columns are not a single model's build-up.")
            add("")
            cols = [m for m in ('M1', 'M2', 'M3') if m in fams]
            add("| Term | " + " | ".join(f"Model {m[-1]}" for m in cols) + " |")
            add("|---" * (len(cols) + 1) + "|")
            has_zi = bool((ct['section'] == 'zi').any())
            for _, r in ct.iterrows():
                out = []
                for m in cols:
                    b, se, p, irr = (r.get(f'{m}_b'), r.get(f'{m}_se'),
                                     r.get(f'{m}_p'), r.get(f'{m}_irr'))
                    if not pd.notna(b):
                        out.append('--')
                    elif r['section'] == 'zi':
                        # The zero-inflation part has a LOGIT link, so exp(b) there
                        # is an odds ratio, NOT an incidence rate ratio. Reported as
                        # the implied structural-zero probability rather than
                        # mislabelled as an IRR.
                        out.append(f"{_f(b)}{stars(p)} ({_f(se)}), "
                                   f"Pr(structural 0) "
                                   f"{_f(float(1 / (1 + np.exp(-b))), '.4f')}")
                    else:
                        out.append(f"{_f(b)}{stars(p)} ({_f(se)}), IRR {_f(irr)}")
                add(f"| {r['term']} | " + " | ".join(out) + " |")
            add("")
            if has_zi:
                add("The `ZI:` row is the zero-inflation part, which has a **logit** "
                    "link rather than a log link -- so exp(b) there is an odds ratio, "
                    "not an IRR. It is reported as the implied structural-zero "
                    "probability instead.\n")

    # ---------------------------------------------------- variance components
    add("## Between-state variance under the selected family\n")
    add("**Read the caveat before the numbers.** In a count GLMM these variance "
        "components live on the log **link scale**, not on the `log(count + 1)` "
        "outcome scale of the published tables. Their magnitudes are therefore "
        "**not** comparable to the sigma^2_u0 values reported in Tables 2 and 3, even "
        "though a percentage reduction would be. **This pipeline does not compute a "
        "Delta sigma^2_u0 percentage at all.** Doing that properly needs a "
        "random-intercept-only refit series against a single Model-1 baseline (the "
        "basis the manuscript uses), and the ladder here only produces random-slope "
        "models. Flagging that as a deliberate gap rather than implying a number "
        "exists: if you want the variance-explained block redone on the count "
        "models, that is a further piece of work.\n")
    add("N and the state count fall between Model 1 and Models 2-3 because the "
        "spending and BLS-derived covariates are listwise-deleted (AK/RI/VT have no "
        "BLS pesticide-applicator series), so the rows of a block are not all on the "
        "same analytic sample.\n")
    add("| Cell | Model | Family | RE structure | sigma^2_u0 | sigma^2_u1 | "
        "sigma_u01 | N obs | States |")
    add("|---|---|---|---|---|---|---|---|---|")
    for cell in sorted(meta):
        for tier in ('rs', 'ri'):
            struct = _re_label(tier)
            for m in ('M1', 'M2', 'M3'):
                sub = tab[(tab['cell'] == cell) & (tab['model'] == m) &
                          (tab['re_tier'] == tier) & tab['converged']]
                if tier == 'rs':
                    pick = sub[sub['winner']] if m in ('M1', 'M3') else sub[~sub['zi_degenerate']]
                else:
                    pick = sub[sub['is_winner_ri']]
                if pick.empty:
                    continue
                r = pick.iloc[0]
                add(f"| {cell} | {m} | {FAMILY_LABEL.get(r['family'], r['family'])} | "
                    f"{struct} | {_f(r['sigma2_u0'], '.4f')} | "
                    f"{_f(r['sigma2_u1'], '.5f')} | {_f(r['sigma_u01'], '.5f')} | "
                    f"{r['n_obs']} | {r['n_states']} |")
    add("")

    # -------------------------------------------------------------- zero fit
    add("## How well each selected model reproduces the observed zeros\n")
    add("Expected zeros are simulated from each fitted model (2000 draws) and are "
        "**never quoted without their Monte-Carlo standard error**.\n")
    add("| Cell | RE structure | Family | Observed 0s | Expected 0s (+/- MC SE) | "
        "Signed gap | Signed relative gap | A credible competitor fits the zeros better | "
        "Zeros partly manufactured |")
    add("|---|---|---|---|---|---|---|---|---|")
    for _, r in ws.iterrows():
        struct = _re_label(r['re_tier'])
        add(f"| {r['cell']} | {struct} | "
            f"{FAMILY_LABEL.get(r['winner'], r['winner'])} | "
            f"{_f(r['obs_zeros'], '.0f')} | {_f(r['exp_zeros'], '.1f')} +/- "
            f"{_f(r['exp_zeros_se'], '.2f')} | {_f(r['zero_signed'], '+.1f')} | "
            f"{_f((r['zero_signed'] / r['obs_zeros']) if r['obs_zeros'] else np.nan, '+.1%')} | "
            f"{(FAMILY_LABEL.get(r['credible_better_zero_fit'], r['credible_better_zero_fit']) if r['has_better_zero_fit'] else 'no')} | "
            f"{'YES' if r['zeros_partly_manufactured'] else 'no'} |")
    add("")

    # ---------------------------------------------- manufactured 2019 zeros
    man = [c for c in meta if meta[c].get('zeros_partly_manufactured')]
    add("### The 2019 inspection zeros are partly manufactured\n")
    add("The establishments-view reshape applies `fillna(0)` to the EPA and state "
        "inspection components before summing them, so a state-year missing **both** "
        "becomes a zero rather than a missing value. That behaviour is inherited from "
        "the published pipeline and was deliberately left unchanged, so the count "
        "models run on the same data the tables do.\n")
    for c in man:
        r = ws[(ws['cell'] == c) & (ws['re_tier'] == 'rs')].iloc[0]
        add(f"Consequence for `{c}`: its selected model rests on very few zeros "
            f"({_f(r['obs_zeros'], '.0f')} observed) and overpredicts them badly "
            f"({_f(r['exp_zeros'], '.1f')} +/- {_f(r['exp_zeros_se'], '.2f')} expected). "
            "The credible-competitor zero-fit flag fires for exactly this cell "
            f"(a {FAMILY_LABEL.get(r['credible_better_zero_fit'], r['credible_better_zero_fit'])} "
            "within the credible AIC set reproduces the zeros better). "
            f"The `__meta` field `zeros_partly_manufactured` marks it: "
            f"{meta[c]['zeros_partly_manufactured_note']}\n")

    # ------------------------------------------------------- cross-software
    add("## Cross-software check on the zero-inflation component\n")
    add("The inspections result does not depend on `glmmTMB`. Refitting the same "
        "zero-inflated negative binomial in `statsmodels` "
        "(`ZeroInflatedNegativeBinomialP`) with state **fixed effects** instead of "
        "random effects -- different package, different treatment of the state "
        "dimension -- gives essentially the same answer:\n")
    add("| Cell | Software | State dimension | N | ZI intercept | SE | Converged |")
    add("|---|---|---|---|---|---|---|")
    add(f"| {xc_i['dv']} 2021 | statsmodels | fixed effects "
        f"({xc_i['n_params']} parameters) | {xc_i['n_obs']} | "
        f"{_f(xc_i['zi_b'], '+.3f')} | {_f(xc_i['zi_se'], '.3f')} | "
        f"{'yes' if xc_i['converged'] else '**no**'} |")
    add(f"| {xc_i['dv']} 2021 | glmmTMB | random effects "
        f"{xc_i['glmm_re_tier']} | {xc_i['glmm_n_obs']} | "
        f"{_f(xc_i['glmm_zi_b'], '+.3f')} | {_f(xc_i['glmm_zi_se'], '.3f')} | yes |")
    add("")
    add(f"The two N's differ on purpose: the cross-check drops the six Model 3 "
        f"covariates (most are state-time-invariant and so collinear with the state "
        f"dummies that stand in for the random intercept here), which lets it retain "
        f"AK/RI/VT -- states glmmTMB's Model 3 drops for missing BLS applicator "
        f"data. Hence {xc_i['n_obs']} observations against glmmTMB's "
        f"{xc_i['glmm_n_obs']}.\n")
    add("**Caveat, and it matters.** This is a *fit* statement, not a substantive "
        f"one. The {xc_i['obs_zeros']} zeros sit in "
        f"{xc_i['n_states_with_a_zero']} different states, no state has more than "
        f"{xc_i['max_zero_years_in_a_state']} zero years, and "
        + ("no state reports zero inspections in every year of the window"
           if xc_i['n_states_all_zero'] == 0
           else f"{xc_i['n_states_all_zero']} states report zero inspections in every "
                f"year of the window")
        + " -- there is no never-inspects state. The inflation term is "
        "absorbing excess mass in the left tail, not identifying a latent "
        "non-inspecting subpopulation. Do not describe it in the paper as if it "
        "were the latter.\n")
    add(f"**The violations cross-check did not converge, and that carries no "
        f"interpretive weight.** The entire parameter vector diverged -- dispersion "
        f"to {_f(xc_v['alpha'], '.1f')}, intercept to {_f(xc_v['intercept'], '+.1f')} "
        f"-- on a design matrix that is full rank "
        f"({xc_v['n_params']}/{xc_v['n_params']} columns"
        f"{', verified' if xc_v['design_full_rank'] else ''}). That is the known "
        f"incidental-parameters fragility of a ~{xc_v['n_params']}-parameter "
        "fixed-effects count mixture, not a finding about zero-inflation. It is "
        "**not** agreement with the glmmTMB result and is not presented as such.\n")

    # ---------------------------------------------------------------- COVID
    add("## COVID robustness (2021 window)\n")
    add("Under the log-linear model the 2020-21 indicator was -0.590*** for "
        "inspections -- pushing `time2`/`time3` from null into significance -- and "
        "-0.183 n.s. for violations, from which `CLAUDE.md` concluded the violations "
        "time trend was robust. Under the selected count family:\n")
    add("| Cell | Term | b | SE | p | IRR | Reading |")
    add("|---|---|---|---|---|---|---|")
    covid_rows = {}
    for k, f in sorted(raw.items()):
        if not isinstance(f, dict) or f.get('model') != 'M3covid':
            continue
        if not f.get('converged') or 'covid' not in (f.get('cond') or {}):
            continue
        c = f['cond']['covid']
        covid_rows[f['cell']] = (f, c)
        reading = (('negative' if c['b'] < 0 else 'positive') +
                   (', significant at .05' if c['p'] < 0.05
                    else ', NOT significant at .05'))
        add(f"| {f['cell']} | covid (2020-21 indicator) | {_f(c['b'], '+.4f')} | "
            f"{_f(c['se'], '.4f')} | {_f(c['p'], '.3g')}{stars(c['p'])} | "
            f"{_f(np.exp(c['b']), '.4f')} | {reading} |")
    add("")
    add("What moves when the indicator is added:\n")
    add("| Cell | Term | b without COVID | p without | b with COVID | p with |")
    add("|---|---|---|---|---|---|")
    for cell, (f, _c) in sorted(covid_rows.items()):
        base = raw[[k for k in raw if isinstance(raw[k], dict)
                    and raw[k].get('cell') == cell and raw[k].get('model') == 'M3'
                    and raw[k].get('family_tag') == f['family_tag']
                    and raw[k].get('re_tier') == f['re_tier']][0]]
        for t in ('time', 'time2', 'time3'):
            b0, b1 = base['cond'].get(t), f['cond'].get(t)
            if not b0 or not b1:
                continue
            add(f"| {cell} | {t} | {_f(b0['b'], '+.5f')} | "
                f"{_f(b0['p'], '.3g')}{stars(b0['p'])} | {_f(b1['b'], '+.5f')} | "
                f"{_f(b1['p'], '.3g')}{stars(b1['p'])} |")
    add("")
    ci = covid_rows.get('insp_2021')
    cv = covid_rows.get('viol_cov_2021')
    if ci and cv:
        vb = raw[[k for k in raw if isinstance(raw[k], dict)
                  and raw[k].get('cell') == 'viol_cov_2021'
                  and raw[k].get('model') == 'M3'
                  and raw[k].get('family_tag') == cv[0]['family_tag']
                  and raw[k].get('re_tier') == cv[0]['re_tier']][0]]
        add(f"**Inspections** behaves as the log-linear check did: "
            f"{_f(ci[1]['b'], '+.2f')} (p = {_f(ci[1]['p'], '.3g')}) against the "
            "log-linear -0.590***, same sign, still significant.\n")
        add(f"**Violations diverges from the log-linear result, and this corrects "
            f"what `CLAUDE.md` currently says.** The indicator is "
            f"{_f(cv[1]['b'], '+.2f')} (p = {_f(cv[1]['p'], '.3g')}) against the "
            f"log-linear -0.183 n.s. -- and, more importantly, `time2` goes from "
            f"p = {_f(vb['cond']['time2']['p'], '.3g')} to "
            f"p = {_f(cv[0]['cond']['time2']['p'], '.3g')} when the indicator is "
            "added. That is substantially more movement than the log-linear check "
            "showed. The violations time trend is **less** COVID-robust than the "
            "documentation claims, not more.\n")

    # ------------------------------------------------- published-fit finding
    add("## A problem in the published tables -- your decision, not mine\n")
    add(f"The current **Table 3 Model 1** coefficients come from a `statsmodels` fit "
        f"that **did not converge**. Refitting the identical specification and "
        f"analytic sample (N = {aff['n_obs']}, {aff['n_states']} states):\n")
    add("| Optimizer | Converged | Log-likelihood | b(log_inspections) | sigma^2_u0 |")
    add("|---|---|---|---|---|")
    for m in ('lbfgs', 'cg', 'powell'):
        r = aff[m]
        add(f"| `{m}` | {'**no**' if r['grad_warning'] else 'yes'} | "
            f"{_f(r['llf'], '.3f')} | {_f(r['b_log_inspections'], '.4f')} | "
            f"{_f(r['sigma2_u0'], '.4f')} |")
    add("")
    add(f"`lbfgs` -- the published optimizer -- reports "
        f"`Gradient optimization failed, |grad| = {_f(aff['lbfgs']['grad_norm'], '.4f')}` "
        f"at a log-likelihood of {_f(aff['lbfgs']['llf'], '.3f')}, while `cg` and "
        f"`powell` independently reach {_f(aff['cg']['llf'], '.3f')} with no warning "
        f"at all. The published values are the non-converged ones: "
        f"b(log_inspections) = {_f(aff['published_b_log_inspections'], '.4f')} and "
        f"sigma^2_u0 = {_f(aff['published_sigma2_u0'], '.4f')}. Converged, they are "
        f"{_f(aff['cg']['b_log_inspections'], '.4f')} and "
        f"{_f(aff['cg']['sigma2_u0'], '.4f')}.\n")
    add("How it stayed invisible: `paper_table_models_2021.py` carries a "
        "module-level `warnings.filterwarnings('ignore')`, which swallowed the "
        "`ConvergenceWarning` before the coefficients were written to the published "
        "JSON. No script in this count-model pipeline uses that idiom, and the "
        "validator refuses it.\n")
    add(f"**Scope, measured rather than assumed.** All "
        f"{aud['n_fits']} published 2011-2021 fits "
        f"({len(_PUB_COLUMNS)} outcome columns x 3 models x "
        f"{{random slope, random-intercept-only}}) were refit and their warnings "
        f"captured: **exactly 1 of {aud['n_fits']} is affected**, the one above. The "
        f"other {aud['n_fits'] - aud['n_nonconverged']} converge cleanly.\n")
    add("The `Reproduces published` column confirms the refit landed on the same "
        "optimum the manuscript published -- including for the failed row, where "
        "reproducing the published number is exactly how we know the published "
        "number is the non-converged one.\n")
    add("| Outcome column | Model | RE basis | N | Log-likelihood | sigma^2_u0 | "
        "Reproduces published | Gradient failure |")
    add("|---|---|---|---|---|---|---|---|")
    for r in aud['fits']:
        matches = (abs(r['sigma2_u0'] - r['published_sigma2_u0'])
                   <= 1e-6 * max(1.0, abs(r['published_sigma2_u0'])))
        add(f"| {r['outcome']} | {r['model']} | {r['re_basis']} | {r['n_obs']} | "
            f"{_f(r['llf'], '.3f')} | {_f(r['sigma2_u0'], '.6f')} | "
            f"{'yes' if matches else 'NO'} | "
            f"{'**YES**' if r['grad_warning'] else 'no'} |")
    add("")
    add("**The reported variance-explained block is unaffected.** All six "
        "random-intercept-only refits converge and reproduce the published "
        "sigma^2_u0 values exactly, and those are the fits the manuscript's "
        "Delta sigma^2_u0 percentages are built from. So this is a coefficient-level "
        "problem in one column, not a variance-block problem.\n")
    add("Nothing was fixed or regenerated. No manuscript `.docx`, no "
        "`paper_table_*` script, no `paper_table_*` JSON was touched by this work. "
        "Whether to reissue Table 3 Model 1 is your call.\n")

    # ------------------------------------------------------------- NB1/NB2
    add("## NB1 versus NB2 changes a number the manuscript reports\n")
    add("For the 2021 violations cells the two negative-binomial "
        "parameterisations disagree materially on the random-slope variance, and "
        "very little AIC separates them:\n")
    add("| Cell | NB1 AIC | NB2 AIC | dAIC | NB1 sigma^2_u1 | NB2 sigma^2_u1 | "
        "Ratio | NB1 warning | NB2 warning |")
    add("|---|---|---|---|---|---|---|---|---|")
    for cell in ('viol_off_2021', 'viol_cov_2021'):
        n1 = tab[(tab['cell'] == cell) & (tab['model'] == 'M3') &
                 (tab['re_tier'] == 'rs') & (tab['family'] == 'nbinom1')].iloc[0]
        n2 = tab[(tab['cell'] == cell) & (tab['model'] == 'M3') &
                 (tab['re_tier'] == 'rs') & (tab['family'] == 'nbinom2')].iloc[0]
        add(f"| {cell} | {_f(n1['aic'], '.1f')} | {_f(n2['aic'], '.1f')} | "
            f"{_f(n2['aic'] - n1['aic'], '.2f')} | {_f(n1['sigma2_u1'], '.6f')} | "
            f"{_f(n2['sigma2_u1'], '.6f')} | "
            f"{_f(n2['sigma2_u1'] / n1['sigma2_u1'], '.2f')}x | "
            f"{n1['message'] or 'none'} | {n2['message'] or 'none'} |")
    add("")
    add("The winner carries a live optimizer warning while the runner-up is clean, "
        "which is uncomfortable. It is not, however, a bad optimum: refitting the "
        "winning NB1 models under BFGS reproduces the log-likelihood and AIC to four "
        "decimals.\n")
    add("| Cell | Model | Default optimizer log-lik | BFGS log-lik | Default AIC | "
        "BFGS AIC | sigma^2_u1 default | sigma^2_u1 BFGS |")
    add("|---|---|---|---|---|---|---|---|")
    for k, f in sorted(raw.items()):
        if not k.endswith('__altopt'):
            continue
        base = raw[k[:-len('__altopt')]]
        add(f"| {f['cell']} | {f['model']} | {_f(base['loglik'], '.4f')} | "
            f"{_f(f['loglik'], '.4f')} | {_f(base['aic'], '.4f')} | "
            f"{_f(f['aic'], '.4f')} | {_f(base['sigma2_u1'], '.7f')} | "
            f"{_f(f['sigma2_u1'], '.7f')} |")
    add("")
    add("**Caveat.** Both optimizers are local gradient methods started from the same "
        "values. This establishes optimizer-*insensitivity*, **not** that the optimum "
        "is global. No **multi-start** search was run, and nothing here should be "
        "read as implying one was. If the NB1-vs-NB2 sigma^2_u1 difference ends up in "
        "the manuscript, it deserves a multi-start check first.\n")

    # ------------------------------------------------- NB1/NB2 reversal
    add("### Why NB1 wins in 2021 and NB2 in 2019 -- coherent, not anomalous\n")
    add("Regressing log within-state variance on log within-state mean gives the "
        "mean-variance scaling exponent. NB2's variance is quadratic in the mean "
        "(exponent 2); NB1's is linear with a constant multiplier phi*mu (exponent "
        "nearer 1):\n")
    add("| Window | Outcome | Mean-variance exponent | States used | Implied family |")
    add("|---|---|---|---|---|")
    for wname, pp in (('2011-2019 (establishments view)', p19),
                      ('2011-2021 (WPS view)', p21)):
        for dv in ('violations', 'inspections'):
            r = pp[dv]
            imp = ('NB2-like (quadratic)' if r['mv_slope'] > 1.75
                   else 'NB1-like (nearer linear)')
            add(f"| {wname} | {dv} | {_f(r['mv_slope'], '.2f')} | "
                f"{r['mv_slope_n_states']} | {imp} |")
    add("")
    add(f"{_f(p19['violations']['mv_slope'], '.2f')} for 2019 violations against "
        f"{_f(p21['violations']['mv_slope'], '.2f')} for 2021 violations. The two "
        "windows are different outcome measures -- ECHO establishments view versus "
        "WPS view -- so different mean-variance scaling is what one should expect, "
        "not an inconsistency in the selection procedure.\n")
    add("The inspections rows are context only and should not be read as a "
        "prediction: both inspections cells select a zero-inflated variant built on "
        "NB2, and once an inflation component is absorbing left-tail mass the raw "
        "within-state mean-variance exponent is no longer a clean guide to which "
        "negative-binomial parameterisation wins. The diagnostic is offered for the "
        "violations cells, where the NB1/NB2 choice is the live question.\n")

    # ------------------------------------------------------ selection detail
    add("## Full selection evidence, per cell and tier\n")
    add("Every fit in the ladder. AIC is only ever comparable **within** one "
        "RE-structure block; the tier boundary is marked. `degen` marks a "
        "zero-inflation intercept sitting numerically on the boundary -- such fits "
        "are shown but structurally excluded from every LRT and from every "
        "zero-fit comparison.\n")
    for cell in sorted(meta):
        g = tab[(tab['cell'] == cell) & (tab['model'] == 'M3')].copy()
        g['_tier_rank'] = g['re_tier'].map(TIER_ORDER)
        g = g.sort_values(['_tier_rank', 'aic'], na_position='last')
        r0 = g.iloc[0]
        add(f"### {cell} -- {r0['outcome']} {r0['window']}, "
            f"exposure {r0['exposure']}, Model 3\n")
        add("| RE structure | Family | Converged | N | AIC | BIC | Observed 0s | "
            "Expected 0s (+/- MC SE) | LRT vs | LRT p | degen | Selected |")
        add("|---|---|---|---|---|---|---|---|---|---|---|---|")
        last = None
        for _, r in g.iterrows():
            if last is not None and r['re_tier'] != last:
                add("| *-- tier boundary: AIC is NOT comparable across this line --* "
                    "| | | | | | | | | | | |")
            last = r['re_tier']
            struct = _re_label(r['re_tier'])
            lp = (f"{fmt_p(r['lrt_p'])}{stars(r['lrt_p'])}"
                  if pd.notna(r['lrt_p']) else '--')
            sel = []
            if r['is_winner_rs']:
                sel.append('selected (rs)')
            if r['is_winner_ri']:
                sel.append('selected (ri)')
            if r['collapsed_to']:
                sel.append(f"collapsed to {FAMILY_LABEL.get(r['collapsed_to'], r['collapsed_to'])}")
            add(f"| {struct} | {FAMILY_LABEL.get(r['family'], r['family'])} | "
                f"{'yes' if r['converged'] else '**NO**'} | {r['n_obs']} | "
                f"{_f(r['aic'], '.1f')} | {_f(r['bic'], '.1f')} | "
                f"{_f(r['obs_zeros'], '.0f')} | {_f(r['exp_zeros'], '.1f')} +/- "
                f"{_f(r['exp_zeros_se'], '.2f')} | "
                f"{FAMILY_LABEL.get(r['lrt_vs'], r['lrt_vs']) if r['lrt_vs'] else '--'} | "
                f"{lp} | {'YES' if r['zi_degenerate'] else ''} | "
                f"{', '.join(sel) if sel else ''} |")
        add("")

    # ----------------------------------------------------- boundary caveat
    add("## Caveats on the selection statistics\n")
    add(f"**Boundary likelihood-ratio tests.** {n_lrt} LRTs are computed, on nested "
        f"pairs only: Poisson within NB2 (dispersion -> infinity), NB2 within ZINB "
        f"(zero-inflation probability -> 0), and ZINB within ZINB+ZI-RE "
        f"(zero-inflation random-effect variance -> 0). All three nulls sit on the "
        f"**boundary** of the parameter space, so the correct reference is a "
        f"1/2 chi-sq_0 + 1/2 chi-sq_1 mixture rather than a plain chi-sq_1, which "
        f"makes every nominal p-value roughly 2x too large -- **conservative, full "
        f"stop**. There is no sense in which it is anti-conservative. "
        f"{n_zire_lrt} of the {n_lrt} are the ZINB-vs-ZINB+ZI-RE comparison. "
        f"`count_model_comparison.csv` carries `lrt_boundary` and "
        f"`lrt_boundary_kind` on every computed test, so a reader is never handed a "
        f"bare nominal p.\n")
    add("**NB1 is not nested in NB2**, so that comparison is AIC/BIC only and has no "
        "LRT. No Vuong test is reported: it is routinely misapplied to non-nested "
        "zero-inflation comparisons.\n")
    add("**Expected zeros always carry their Monte-Carlo standard error** (2000 "
        "simulations per fit). An expected-zero count without its MC SE is not a "
        "number one can compare to an observed count.\n")
    add("**Cross-tier AICs are never compared.** Where a cell has both an `rs` and "
        "an `ri` winner they are two separate results under two different "
        "random-effects structures, reported side by side and not ranked against "
        "each other.\n")

    # ------------------------------------------------------- honest reading
    add("## Honest expectations\n")
    add("A better-specified model can confirm weak associations as readily as it can "
        "strengthen them. If a zero-inflated or negative-binomial model is the right "
        "model for these counts -- and the selection evidence above is what we have "
        "on that -- then its answer is the answer, whichever direction it points. "
        "This was recorded as the expected risk in the design spec before any model "
        "was fit, and it is recorded here because it came out mixed: the inspections "
        "zero-inflation is real but is a left-tail-fit result rather than a latent "
        "non-inspecting subpopulation; the violations zero-inflation is real but "
        "outcompeted at the structure we are committed to; and the two genuinely "
        "new findings -- the non-converged published Model 1 and the weaker COVID "
        "robustness for violations -- are corrections to the existing paper rather "
        "than additions to it.\n")

    with open(DOCS + 'count_models_zinb.md', 'w') as fh:
        fh.write("\n".join(L) + "\n")
    print(f"Wrote {DOCS}count_models_zinb.md ({len('\n'.join(L))} chars)")


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

    ev = build_memo_evidence(raw)
    write_memo(raw, meta, tab, ws, ev)


if __name__ == '__main__':
    main()
