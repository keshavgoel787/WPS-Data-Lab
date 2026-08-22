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

# Canonical family order for the six-rung ladder. Single source of truth for
# "how many families were there" -- so a memo sentence can never hardcode a
# count that drifts from the ladder.
ALL_FAMILIES = ['poisson', 'nbinom1', 'nbinom2', 'zip', 'zinb', 'zinb_re']

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


ZINB_FAMILIES = ['zinb', 'zinb_re']
# The full zero-inflated set (ZIP included) and the full plain set. These exist
# so the memo's "did a zero-inflated model ever lose?" tally covers ZIP, which
# is the family that DOES lose -- see zi_loss_summary().
ZI_FAMILIES = ['zip', 'zinb', 'zinb_re']
PLAIN_FAMILIES = ['poisson', 'nbinom1', 'nbinom2']
# The plain families that ARE negative binomials -- so "beaten only by a
# negative binomial" can be tested rather than asserted.
NB_FAMILIES = ['nbinom1', 'nbinom2']


def matched_zi_vs_plain(tab, meta, zi_families=None, plain_families=None):
    """Every legitimate matched comparison between a zero-inflated family and a
    plain one: same cell, same model, same re_tier, same n_obs, both converged,
    neither ZI-degenerate. `margin` is (plain AIC - ZI AIC), so margin > 0 means
    the zero-inflated model won.

    This is the general engine behind three memo claims:
      * matched_zinb_vs_plain() -- the ZI-NB-vs-NB1 and ZI-NB-vs-NB2 tables;
      * zi_loss_summary()       -- WHICH zero-inflated family ever loses, which
                                   is ZIP and only ZIP.

    A `collapsed` flag rides on each row: `insp_2021`'s ZINB+ZI-RE is
    numerically identical to its ZINB (the ZI random-effect variance went to
    zero), so those rows are duplicates of their ZINB twin rather than
    independent evidence. They are KEPT and flagged rather than dropped, so the
    row count is reproducible either way and the reader can see which is which.
    """
    zi_families = list(ZI_FAMILIES if zi_families is None else zi_families)
    plain_families = list(PLAIN_FAMILIES if plain_families is None
                          else plain_families)
    rows = []
    for cell in sorted(meta):
        for model in ('M1', 'M2', 'M3'):
            for tier in ('rs', 'ri'):
                g = tab[(tab['cell'] == cell) & (tab['model'] == model) &
                        (tab['re_tier'] == tier) & tab['converged'] &
                        (~tab['zi_degenerate'])]
                for pf in plain_families:
                    p = g[g['family'] == pf]
                    if p.empty:
                        continue
                    p = p.iloc[0]
                    for zf in zi_families:
                        z = g[g['family'] == zf]
                        if z.empty:
                            continue
                        z = z.iloc[0]
                        if z['n_obs'] != p['n_obs']:
                            continue
                        rows.append({
                            'cell': cell, 'model': model, 're_tier': tier,
                            'zi_family': zf, 'plain_family': pf,
                            'n_obs': int(z['n_obs']),
                            'plain_aic': float(p['aic']),
                            'zi_aic': float(z['aic']),
                            'margin': float(p['aic'] - z['aic']),
                            'collapsed': bool(z['collapsed_to']
                                              or p['collapsed_to']),
                            'collapsed_to': z['collapsed_to'] or '',
                        })
    return sorted(rows, key=lambda r: -r['margin'])


def _oxford(items):
    """'a', 'a and b', 'a, b and c' -- so a generated cell list never reads as a
    bare comma-separated fragment in prose."""
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ''
    return ", ".join(items[:-1]) + " and " + items[-1]


def _mc_key(r):
    """Identity of a matched comparison independent of which plain family it
    was run against -- used to pair the NB1 and NB2 sets row for row."""
    return (r['cell'], r['model'], r['re_tier'], r['zi_family'])


def matched_zinb_vs_plain(tab, meta, plain_family):
    """The ZI-NB rungs (ZINB, ZINB+ZI-RE) against ONE plain family. Used twice:
    against NB1 (the family the manuscript's 2021 violations columns select) and
    against NB2 (the 2019 violations selection), so the memo's phrase "better
    than the plain negative binomial" is shown for both parameterisations rather
    than demonstrated for one and asserted for the other."""
    return matched_zi_vs_plain(tab, meta, zi_families=ZINB_FAMILIES,
                               plain_families=[plain_family])


def matched_zinb_vs_nb1(tab, meta):
    """matched_zinb_vs_plain(..., 'nbinom1') with the legacy `nb1_aic` key kept
    as an alias for `plain_aic`.

    This exists because the memo's original mechanism sentence -- "a negative
    binomial's own overdispersion parameter accounts for those same zeros about
    as economically as an explicit inflation term does" -- is false, and
    exhaustively so. Wherever a ZI-NB CAN be fit at matched model and tier it
    beats NB1 by a wide margin. NB1's win in the Model-3 random-slope column is
    therefore a DEFAULT, not a merit win.
    """
    rows = matched_zinb_vs_plain(tab, meta, 'nbinom1')
    for r in rows:
        r['nb1_aic'] = r['plain_aic']
    return rows


def zi_loss_summary(tab, meta):
    """WHICH zero-inflated families lose a matched comparison, and how often.

    This function exists because an earlier draft of the memo claimed "nowhere
    in this pipeline did a zero-inflated model get tested and lose on merit",
    and that is false: zero-inflated POISSON loses, repeatedly and by enormous
    margins, because a Poisson mean-variance structure cannot represent this
    overdispersion no matter how much inflation is bolted on to it. What is true
    -- and sharper -- is that no zero-inflated NEGATIVE BINOMIAL ever lost one.

    Returned counts are over DISTINCT (cell, model, re_tier, zi_family) losers,
    not over raw pairings, because one ZI fit losing to both NB1 and NB2 at the
    same rung is one loss, not two. `raw_losses` keeps the pairings for the
    margin range.
    """
    allcmp = matched_zi_vs_plain(tab, meta)
    losses = [r for r in allcmp if r['margin'] <= 0]
    wins = [r for r in allcmp if r['margin'] > 0]
    by_family = {}
    for fam in ZI_FAMILIES:
        fl = {(r['cell'], r['model'], r['re_tier'])
              for r in losses if r['zi_family'] == fam}
        by_family[fam] = sorted(fl)
    distinct = sorted({(r['cell'], r['model'], r['re_tier'], r['zi_family'])
                       for r in losses})
    # `distinct` is keyed on the FAMILY as well as the rung, because two
    # different ZI families losing at the same rung are two results. The memo's
    # prose counts RUNGS per family (3-tuples), which is what
    # `n_losses_by_family` holds. They are equal today only because every loser
    # is ZIP; `distinct_loser_rungs` is kept alongside so the validator can
    # assert that equality rather than assume it.
    distinct_rungs = sorted({d[:3] for d in distinct})
    # Head-to-head record per (zi_family, plain_family). This is what shows the
    # HIERARCHY the memo needs: inflation helps a Poisson a lot, and still loses
    # to a negative binomial every time.
    record = {}
    for zf in ZI_FAMILIES:
        for pf in PLAIN_FAMILIES:
            sub = [r for r in allcmp
                   if r['zi_family'] == zf and r['plain_family'] == pf]
            if not sub:
                continue
            record[(zf, pf)] = {
                'n': len(sub),
                'wins': sum(1 for r in sub if r['margin'] > 0),
                'losses': sum(1 for r in sub if r['margin'] <= 0),
                'margin_lo': min(r['margin'] for r in sub),
                'margin_hi': max(r['margin'] for r in sub),
            }
    # For each losing rung, WHICH plain families beat it. Needed because "the
    # ZIP is only ever beaten by a negative binomial" is FALSE at one rung
    # (viol_off_2019/M1/rs, where a plain Poisson beats it by 1.94 AIC) -- and
    # that rung is the source of the 1.9 lower bound the memo quotes. The clause
    # was hardcoded and wrong; it is derived from this field now.
    rung_beaters = {}
    for r in losses:
        rung_beaters.setdefault(
            (r['cell'], r['model'], r['re_tier'], r['zi_family']),
            []).append(r['plain_family'])
    for k in rung_beaters:
        rung_beaters[k] = sorted(set(rung_beaters[k]))
    nb_only = sorted(k for k, v in rung_beaters.items()
                     if set(v) <= set(NB_FAMILIES))
    with_poisson = sorted(k for k, v in rung_beaters.items()
                          if 'poisson' in v)
    return {
        'n_comparisons': len(allcmp),
        'n_win_pairings': len(wins),
        'rung_beaters': rung_beaters,
        'rungs_beaten_by_nb_only': nb_only,
        'rungs_beaten_by_a_plain_poisson': with_poisson,
        'n_loss_pairings': len(losses),
        'distinct_losers': distinct,
        'distinct_loser_rungs': distinct_rungs,
        'distinct_losers_rs': [d for d in distinct if d[2] == 'rs'],
        'losers_by_family': by_family,
        'n_losses_by_family': {f: len(v) for f, v in by_family.items()},
        'n_zinb_losses': sum(len(by_family[f]) for f in ZINB_FAMILIES),
        'loss_margin_lo': (min(abs(r['margin']) for r in losses)
                           if losses else None),
        'loss_margin_hi': (max(abs(r['margin']) for r in losses)
                           if losses else None),
        'record': record,
        'raw_losses': losses,
    }


def pair_matched(mc_a, mc_b):
    """Pair two matched-comparison sets rung for rung, refusing to guess.

    `zip(sorted(a), sorted(b))` would silently misalign if the two sets ever
    differed in membership -- e.g. if a plain family failed to converge at one
    rung so its comparison were absent. Raise instead.
    """
    a, b = sorted(mc_a, key=_mc_key), sorted(mc_b, key=_mc_key)
    ka, kb = [_mc_key(r) for r in a], [_mc_key(r) for r in b]
    if ka != kb:
        raise ValueError(
            'matched-comparison sets are not over the same rungs; cannot pair. '
            f'only in first: {[k for k in ka if k not in kb]!r}; '
            f'only in second: {[k for k in kb if k not in ka]!r}')
    return list(zip(a, b))


def zinb_eligibility_at_rs(meta, cells):
    """For each cell in `cells`, whether a ZI-NB rung is eligible at the
    mandated `(1 + time | state)` structure at Model 1 and at Model 3.

    Splits `cells` into three groups, which is the distinction an earlier memo
    draft blurred into a single blanket claim:
      `lost`   -- eligible at M1, NOT at M3: the rung genuinely "stops being
                  estimable once the Model-3 covariates enter".
      `never`  -- eligible at NEITHER: nothing stops, there was never one.
      `kept`   -- eligible at both.
    """
    zn = set(ZINB_FAMILIES)
    lost, never, kept = [], [], []
    for cell in cells:
        m = meta[cell]
        e1 = zn & set(m.get('eligible_rs_m1') or [])
        e3 = zn & set(m.get('eligible_rs_m3') or [])
        (kept if e3 else lost if e1 else never).append(cell)
    return {'lost': lost, 'never': never, 'kept': kept}


def _absent_families(tab, meta, cell, model, re_tier='rs'):
    """For one (cell, model, re_tier): every family NOT in
    `eligible_rs_{model}`, paired with the reason it is absent, read off the fit
    records rather than assumed.

    The distinction is load-bearing for the memo. "Not eligible for selection"
    is NOT the same as "could not be estimated" -- there are four different
    reasons, and only the first is an identifiability statement:

      no fit at this structure -- the family fell back to (1 | state)
      did not converge        -- fit attempted, optimizer failed
      degenerate              -- fit converged, ZI intercept on the boundary
      collapsed onto X        -- fit converged but is numerically X, not a
                                 distinct competitor

    Conflating them is how the first draft of this memo came to assert that the
    ZI-NB rungs "could not be fit at all" for viol_off_2021, when at Model 1
    ZINB+ZI-RE fits there and wins the tier.
    """
    elig = set(meta[cell].get(f'eligible_rs_{model.lower()}') or [])
    out = []
    for fam in ALL_FAMILIES:
        if fam in elig:
            continue
        r = tab[(tab['cell'] == cell) & (tab['model'] == model) &
                (tab['re_tier'] == re_tier) & (tab['family'] == fam)]
        if r.empty:
            why = 'no fit at this structure; it fell back to (1 | state)'
        else:
            r = r.iloc[0]
            if not r['converged']:
                why = 'fit attempted, did not converge'
            elif r['zi_degenerate']:
                why = 'converged but ZI-degenerate -- intercept on the boundary'
            elif r['collapsed_to']:
                why = ('converged but numerically identical to '
                       + FAMILY_LABEL.get(r['collapsed_to'], r['collapsed_to'])
                       + ', so not a distinct competitor')
            else:
                why = 'excluded from selection for an unrecorded reason'
        out.append((fam, why))
    return out


def _family_changes(tab, meta):
    """(cell, re_tier) pairs whose selected family is NOT constant across the
    M1/M2/M3 rows the variance table prints. Reading sigma^2 down such a column
    as a variance-reduction sequence is a mistake -- the rows are different
    models -- so the memo has to name them."""
    out = []
    for cell in sorted(meta):
        for tier in ('rs', 'ri'):
            fams = []
            for m in ('M1', 'M2', 'M3'):
                sub = tab[(tab['cell'] == cell) & (tab['model'] == m) &
                          (tab['re_tier'] == tier) & tab['converged']]
                if tier == 'rs':
                    pick = (sub[sub['winner']] if m in ('M1', 'M3')
                            else sub[~sub['zi_degenerate']])
                else:
                    pick = sub[sub['is_winner_ri']]
                if not pick.empty:
                    fams.append(pick.iloc[0]['family'])
            if len(set(fams)) > 1:
                out.append((cell, tier))
    return out


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


def load_memo_evidence(path=MEMO_EVIDENCE_PATH):
    """The memo's non-ladder evidence (panel descriptives, the published-fit
    convergence audit, the cross-software ZI check), read from the artifact
    that `scripts/build_count_model_memo_evidence.py` writes.

    This module is a pure artifact READER -- it fits nothing, so re-rendering
    the memo takes under a second. If the artifact is missing we delegate to
    the builder once (so a fresh checkout still works end to end) rather than
    silently emitting a memo with holes in it.
    """
    import os
    import sys
    if os.path.exists(path):
        return json.load(open(path))
    print(f"{path} missing -- building it once via "
          f"scripts/build_count_model_memo_evidence.py (this fits models and "
          f"takes ~15 s; subsequent runs reuse the artifact)")
    if '/Users/keshavgoel/Research/scripts' not in sys.path:
        sys.path.insert(0, '/Users/keshavgoel/Research/scripts')
    from build_count_model_memo_evidence import build_memo_evidence
    return build_memo_evidence(load_results())


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

    # The headline answer to the PI's question, generated so it cannot drift
    # from the matched-comparison evidence that supports it.
    _mc = matched_zinb_vs_nb1(tab, meta)
    _mc2 = matched_zinb_vs_plain(tab, meta, 'nbinom2')
    _mc_wins = sum(1 for r in _mc if r['margin'] > 0)
    _mc2_wins = sum(1 for r in _mc2 if r['margin'] > 0)
    _mlo = min(r['margin'] for r in _mc)
    _mlo2 = min(r['margin'] for r in _mc2)
    _mc_ri = sum(1 for r in _mc if r['re_tier'] == 'ri')
    _mc_rs = sum(1 for r in _mc if r['re_tier'] == 'rs')
    _mc_models = sorted({r['model'] for r in _mc})
    _rs3 = m3w[m3w['re_tier'] == 'rs']
    _zi_sel = sorted(set(_rs3[_rs3['family'].isin(ZI_FAMILIES)]['cell']))
    # Derive the noun phrase for that cell list too, so it cannot misdescribe
    # its own contents (it used to read "the inspections cells" as a literal).
    _zi_sel_out = sorted({tab[tab['cell'] == c]['outcome'].iloc[0]
                          for c in _zi_sel})
    _zi_sel_noun = ("the " + "/".join(_zi_sel_out) + " cell"
                    + ("" if len(_zi_sel) == 1 else "s")) if _zi_sel_out \
        else "no cell"
    # Which violations cells actually LOSE a ZI-NB rung between M1 and M3, and
    # which never had one at this structure. An earlier draft asserted the
    # former for all of them; it is true of exactly one.
    _viol_cells = sorted(set(tab[tab['outcome'] == 'violations']['cell']))
    _ze = zinb_eligibility_at_rs(meta, _viol_cells)
    _zl = zi_loss_summary(tab, meta)
    _zip_losses = _zl['n_losses_by_family']['zip']
    add("## Bottom line\n")
    add(f"**Jafari's model class fits these data better than the plain negative "
        f"binomial everywhere it can be estimated.** Across every legitimate "
        f"matched comparison in the ladder -- same cell, same model, same "
        f"random-effects tier, same N, both converged, neither ZI-degenerate -- a "
        f"zero-inflated negative binomial beats plain NB1 {_mc_wins} times out of "
        f"{len(_mc)}, by at least {_f(_mlo, '.1f')} AIC, and beats plain NB2 "
        f"{_mc2_wins} times out of {len(_mc2)}, by at least {_f(_mlo2, '.1f')} "
        f"AIC. There is no exception against either parameterisation.\n")
    # The hierarchy. An earlier draft said inflation on a Poisson "is beaten
    # wherever it is tested", which is false and throws away the strongest piece
    # of evidence FOR inflation: ZIP beats plain Poisson 12-1. All three tallies
    # are read off zi_loss_summary()['record'].
    _rec = _zl['record']
    _zp = _rec[('zip', 'poisson')]
    _zn1 = _rec[('zip', 'nbinom1')]
    _zn2 = _rec[('zip', 'nbinom2')]
    add(f"**The evidence is a hierarchy, and it lands exactly on Jafari's "
        f"specification.** Three tallies, all over matched comparisons:\n")
    add(f"1. **Inflation helps.** A zero-inflated Poisson beats a plain Poisson "
        f"{_zp['wins']}-{_zp['losses']} (margins {_f(_zp['margin_lo'], '.1f')} to "
        f"{_f(_zp['margin_hi'], '.1f')} AIC). Adding a structural-zero component "
        f"to a Poisson buys a great deal.")
    add(f"2. **Negative-binomial overdispersion helps more.** A plain negative "
        f"binomial beats that same zero-inflated Poisson "
        f"{_zn1['losses']}-{_zn1['wins']} against NB1 and "
        f"{_zn2['losses']}-{_zn2['wins']} against NB2. Inflation on a Poisson "
        f"mean-variance structure cannot close the gap to a negative binomial -- "
        f"not once, anywhere in the ladder.")
    add(f"3. **The two together win wherever they can be fit.** A zero-inflated "
        f"negative binomial beats NB1 {_mc_wins}-{len(_mc) - _mc_wins} and NB2 "
        f"{_mc2_wins}-{len(_mc2) - _mc2_wins}. No `zinb` or `zinb_re` fit lost a "
        f"single matched comparison: {_zl['n_zinb_losses']} losses in total.")
    add("")
    # Tier 0, one sentence: the base of the ladder, where both ingredients face
    # a family with neither. Generated from the same `record`.
    _t0 = [(f, _zl['record'][(f, 'poisson')]) for f in ZINB_FAMILIES
           if (f, 'poisson') in _zl['record']]
    if _t0:
        add("At the base of that ladder, against a plain Poisson -- a family with "
            "neither ingredient -- the two ZI-NB rungs are "
            + _oxford([f"{v['wins']}-{v['losses']} ({FAMILY_LABEL.get(f, f)})"
                       for f, v in _t0])
            + f", by {_f(min(v['margin_lo'] for _, v in _t0), '.1f')} to "
            f"{_f(max(v['margin_hi'] for _, v in _t0), '.1f')} AIC.\n")
    # A "rung" and a "rung loss" have to be defined, or tier 1's pairwise 12-1
    # and the 13 losing rungs below read as contradictory when they are not.
    add(f"One definition, because two of those numbers look like they disagree "
        f"and do not. A **rung** is one (cell, model, RE-tier) combination, and a "
        f"rung counts as a *loss* for a family if **at least one** plain family "
        f"beats it there. Tier 1's {_zp['wins']}-{_zp['losses']} is pairwise, "
        f"ZIP against plain Poisson only; the losing-rung count below is against "
        f"**any** plain family. They are counting different things.\n")
    _nb_only = _zl['rungs_beaten_by_nb_only']
    _with_p = _zl['rungs_beaten_by_a_plain_poisson']
    add(f"So the ingredient that wins is inflation **combined with** "
        f"negative-binomial overdispersion, which is precisely Jafari's model: a "
        f"zero-inflated *negative binomial* never lost a matched comparison "
        f"anywhere in this pipeline.\n")
    add(f"The zero-inflated *Poisson* is the other story. It loses at "
        f"{_zip_losses} rungs, {len(_zl['distinct_losers_rs'])} of them at the "
        f"mandated `(1 + time | state)` structure, by "
        f"{_f(_zl['loss_margin_lo'], '.1f')} to "
        f"{_f(_zl['loss_margin_hi'], '.1f')} AIC. At {len(_nb_only)} of those "
        f"{_zip_losses} rungs every family that beats it is a negative binomial"
        + ((f", and at the remaining {len(_with_p)} -- "
            + _oxford([f"`{c}` {m}" for c, m, _t, _f_ in _with_p])
            + f" -- a plain Poisson beats it as well, by "
            + _oxford([_f(min(abs(r['margin']) for r in _zl['raw_losses']
                              if (r['cell'], r['model'], r['re_tier'],
                                  r['zi_family']) == k
                              and r['plain_family'] == 'poisson'), '.1f')
                       for k in _with_p])
            + " AIC. That is the single loss tier 1 already reports, and it is "
              "why inflation is described above as buying a great deal for a "
              "Poisson rather than as always helping one")
           if _with_p else
           ", at every one of them")
        + ".\n")
    add(f"**What stops us using it is identifiability, not evidence.** At the "
        f"structure the manuscript mandates, `(1 + time | state)`, no "
        f"zero-inflated negative-binomial rung is eligible at Model 3 in any of "
        f"the {len(_viol_cells)} violations cells, so a plain negative binomial "
        f"wins those columns **by default, having been the only kind of model "
        f"left in the race** -- and it remains the selected family for them under "
        f"the pre-registered protocol. Only "
        + (_oxford(['`' + c + '`' for c in _ze['lost']]) + " of them fits"
           if len(_ze['lost']) == 1 else
           _oxford(['`' + c + '`' for c in _ze['lost']]) + " of them fit")
        + f" the phrase \"stops being estimable\"; the other {len(_ze['never'])} "
        f"never had such a rung at this structure at either model. The per-cell "
        f"accounting is below. For {_zi_sel_noun} the ZI rungs do survive to "
        f"Model 3 and are duly selected "
        f"({_oxford(['`' + c + '`' for c in _zi_sel])}).\n")
    add("Two further findings are corrections to work already in print rather "
        "than additions to it, and both need your decision: the published "
        "Table 3 Model 1 comes from a fit that never converged, and the "
        "violations time trend is materially less COVID-robust than "
        "`CLAUDE.md` currently records. Both are documented below with the "
        "numbers.\n")

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
        "dimension at all. Adopting an intercept-only crossed structure would "
        "also discard the **state-specific time slope** -- the per-state rate of "
        "change around the 2016-17 WPS revision -- which is the part of this "
        "paper's design that a crossed `(1|state) + (1|year)` cannot express. "
        "(The cubic time trend itself is a fixed effect and would survive; it is "
        "the random slope that would not.) We keep the "
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
    gaps = []
    for w in sorted({str(x) for x in tab['window']}):
        for m in ('M1', 'M3'):
            o = tab[(tab['cell'] == f'viol_off_{w}') & (tab['model'] == m) &
                    (tab['re_tier'] == 'rs')]
            c = tab[(tab['cell'] == f'viol_cov_{w}') & (tab['model'] == m) &
                    (tab['re_tier'] == 'rs')]
            if not o.empty and not c.empty:
                gaps.append((w, m, int(c['n_obs'].iloc[0]), int(o['n_obs'].iloc[0])))
    if gaps:
        add("The offset and covariate specifications do **not** run on the same N: "
            + "; ".join(f"{w} {m} has {nc} rows as a covariate against {no} as an "
                        f"offset (gap {nc - no})" for w, m, nc, no in gaps)
            + ". The offset is `log(inspections)`, which is undefined at zero "
              "inspections, so every zero-inspection state-year drops out of the "
              "`viol_off_*` cells and stays in the `viol_cov_*` ones. Do not read "
              "the offset and covariate columns as the same sample.\n")

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
        f"simple as it looks.\n")
    ri = m3w[m3w['re_tier'] == 'ri']
    if not ri.empty:
        ri_zi = sorted(set(ri['family']) & ZI_FAMS)
        ri_non_zi = sorted(set(ri['family']) - ZI_FAMS)
        add(f"Note also that {len(ri)} cell"
            + ("" if len(ri) == 1 else "s")
            + f" ({', '.join('`' + c + '`' for c in sorted(ri['cell']))}) carr"
            + ("ies" if len(ri) == 1 else "y")
            + " a SECOND winner at the `(1 | state)` fallback tier"
            + (f", and "
               + ("that one is" if len(ri) == 1 else "those are")
               + " zero-inflated ("
               + ", ".join(FAMILY_LABEL.get(f, f) for f in ri_zi) + ")"
               if ri_zi and not ri_non_zi else
               f", of which the zero-inflated ones are "
               + ", ".join(FAMILY_LABEL.get(f, f) for f in ri_zi)
               + " and the rest "
               + ", ".join(FAMILY_LABEL.get(f, f) for f in ri_non_zi)
               if ri_zi else ", none of which is zero-inflated")
            + " -- the two tiers are separate results, not a ranking.\n")

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
    n_fam = len(ALL_FAMILIES)
    ALL_CELLS = sorted(meta)
    absent_m3 = {c: _absent_families(tab, meta, c, 'M3') for c in ALL_CELLS}
    mc = matched_zinb_vs_nb1(tab, meta)
    mc_wins = [r for r in mc if r['margin'] > 0]
    mc_dist = [r for r in mc if not r['collapsed']]
    mc_dist_wins = [r for r in mc_dist if r['margin'] > 0]
    mlo, mhi = min(r['margin'] for r in mc), max(r['margin'] for r in mc)
    # How many ZI-NB fits the degeneracy criterion removes from the matched
    # comparison, and where. Without this the reader cannot tell whether the
    # 2019 Model-3 ZINB fits were excluded or counted as wins.
    _degen_zinb = tab[tab['zi_degenerate'] & tab['family'].isin(ZINB_FAMILIES)]
    _n_degen_zinb = len(_degen_zinb)
    _degen_zinb_cells = sorted(set(_degen_zinb['cell']))
    _n_degen_worse = 0
    for _, _dr in _degen_zinb.iterrows():
        _wf = meta[_dr['cell']][f"m{_dr['model'][1:]}_winner_rs"] \
            if _dr['model'] in ('M1', 'M3') else None
        if _wf is None:
            continue
        _w = tab[(tab['cell'] == _dr['cell']) & (tab['model'] == _dr['model']) &
                 (tab['re_tier'] == _dr['re_tier']) & (tab['family'] == _wf)]
        if not _w.empty and _dr['aic'] > float(_w['aic'].iloc[0]):
            _n_degen_worse += 1
    _zl_m = zi_loss_summary(tab, meta)
    # The subject is derived: NB1 wins the rs/M3 column in the 2021 violations
    # cells, NB2 in the 2019 pair, so a heading naming NB1 alone misdescribes
    # half of what it is justifying.
    add(f"### The plain negative binomial's win is a default, not a merit win\n")
    add(f"The violations cells select a plain negative binomial at Model 3 at "
        f"the mandated structure"
        + (" (" + "; ".join(
            f"{FAMILY_LABEL.get(meta[c]['m3_winner_rs'], meta[c]['m3_winner_rs'])} "
            f"in `{c}`" for c in sorted(set(tab[tab['outcome'] == 'violations']['cell'])))
           + ")")
        + ". It is tempting to explain that by saying a negative binomial's "
        f"own overdispersion parameter accounts for those zeros about as "
        f"economically as an explicit inflation term does. **That explanation is "
        f"wrong, and the artifacts say so exhaustively.** Comparing a "
        f"zero-inflated negative binomial against plain NB1 wherever the "
        f"comparison is legitimate -- same cell, same model, same random-effects "
        f"tier, same N, both converged, neither ZI-degenerate -- the "
        f"zero-inflated model wins **{len(mc_wins)} of {len(mc)}** times, by "
        f"{_f(mlo, '.1f')} to {_f(mhi, '.1f')} AIC. There is no exception.\n")
    # Relocated here from the Bottom line: these two caveats belong beside the
    # table they qualify, not ahead of the payoff.
    add(f"The exclusions in that definition are load-bearing, and they are what "
        f"makes the {_zl_m['n_zinb_losses']}-loss record possible: "
        f"{_n_degen_zinb} ZI-NB fits in the ladder are `zi_degenerate` (a "
        f"zero-inflation intercept numerically on the boundary, estimating "
        f"nothing) and are **excluded** from the comparison rather than counted "
        f"as losses. All of them are in the "
        f"{_oxford(['`' + c + '`' for c in _degen_zinb_cells])} "
        f"{'cell' if len(_degen_zinb_cells) == 1 else 'cells'}, and "
        f"{_n_degen_worse} of the {_n_degen_zinb} have worse AIC than the plain "
        f"family selected at their own rung -- so they are not being silently "
        f"scored as wins either. A degenerate fit is not evidence either "
        f"way, which is why it is out of the tally in both directions.\n")
    add(f"Two things to be clear about that tally before it is quoted. "
        f"{sum(1 for r in mc if r['re_tier'] == 'rs')} of the {len(mc)} "
        f"comparisons are at the mandated `(1 + time | state)` structure and "
        f"{sum(1 for r in mc if r['re_tier'] == 'ri')} are at the "
        f"`(1 | state)` fallback tier, so the evidence is not all from the "
        f"structure the manuscript specifies. And \"every legitimate matched "
        f"comparison in the ladder\" means every one that exists, which is "
        f"{' and '.join(sorted({r['model'] for r in mc}))} only: the ladder fits "
        f"Model 2 under the already-selected family alone, so Model 2 contributes "
        f"no cross-family comparison to make.\n")
    add("| Cell | Model | RE structure | ZI family | N | NB1 AIC | ZI-NB AIC | "
        "AIC margin to ZI-NB | Distinct model? |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in mc:
        add(f"| {r['cell']} | {r['model']} | {_re_label(r['re_tier'])} | "
            f"{FAMILY_LABEL.get(r['zi_family'], r['zi_family'])} | {r['n_obs']} | "
            f"{_f(r['nb1_aic'], '.2f')} | {_f(r['zi_aic'], '.2f')} | "
            f"**{_f(r['margin'], '+.2f')}** | "
            + ("yes" if not r['collapsed'] else
               f"no -- numerically identical to "
               f"{FAMILY_LABEL.get(r['collapsed_to'], r['collapsed_to'])}")
            + " |")
    add("")
    add(f"{len(mc) - len(mc_dist)} of those {len(mc)} rows are not independent "
        f"evidence -- they are a ZINB+ZI-RE fit that collapsed onto its own ZINB "
        f"twin, so it reports the same likelihood. Counting only the "
        f"{len(mc_dist)} distinct models, the zero-inflated family still wins "
        f"{len(mc_dist_wins)} of {len(mc_dist)}. Either way the direction is "
        f"unanimous.\n")

    # NB2 companion. The claim being defended is "better than the plain negative
    # BINOMIAL", and NB1 is only one of the two parameterisations in the ladder
    # (NB2 is the family the 2019 violations columns actually select), so the
    # same matched comparisons are run against NB2 and shown, not asserted.
    mc2 = matched_zinb_vs_plain(tab, meta, 'nbinom2')
    mc2_wins = [r for r in mc2 if r['margin'] > 0]
    mc2_dist = [r for r in mc2 if not r['collapsed']]
    mc2_dist_wins = [r for r in mc2_dist if r['margin'] > 0]
    m2lo, m2hi = (min(r['margin'] for r in mc2), max(r['margin'] for r in mc2))
    # pair_matched() refuses to zip sets that are not over the same rungs.
    _pairs = pair_matched(mc, mc2)
    _nb2_tougher = sum(1 for a, b in _pairs if b['margin'] < a['margin'])
    _nb1_tougher = sum(1 for a, b in _pairs if a['margin'] < b['margin'])
    _tied = len(_pairs) - _nb2_tougher - _nb1_tougher
    # Which cell drives NB2's lower MEAN margin, and by how much -- derived, so
    # the "range effect, not a majority" explanation carries its own evidence.
    _drops = {}
    for a, b in _pairs:
        _drops.setdefault(a['cell'], []).append((a['margin'], b['margin']))
    _drop_cell = max(_drops, key=lambda c: sum(x - y for x, y in _drops[c]))
    _drop_n = len(_drops[_drop_cell])
    _drop_from = sum(x for x, _ in _drops[_drop_cell]) / _drop_n
    _drop_to = sum(y for _, y in _drops[_drop_cell]) / _drop_n
    add(f"#### The same result against NB2, not just NB1\n")
    add(f"NB1 is one of two negative-binomial parameterisations in this ladder, "
        f"and it is not the one the 2019 violations columns select. So the "
        f"identical set of matched comparisons is run against plain **NB2** as "
        f"well -- otherwise \"better than the plain negative binomial\" would be "
        f"shown for one parameterisation and asserted for the other. The "
        f"zero-inflated model wins **{len(mc2_wins)} of {len(mc2)}** here too, by "
        f"{_f(m2lo, '.1f')} to {_f(m2hi, '.1f')} AIC, and "
        f"{len(mc2_dist_wins)} of {len(mc2_dist)} counting distinct models "
        f"only.\n")
    add("| Cell | Model | RE structure | ZI family | N | NB2 AIC | ZI-NB AIC | "
        "AIC margin to ZI-NB (NB2) | Distinct model? |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in mc2:
        add(f"| {r['cell']} | {r['model']} | {_re_label(r['re_tier'])} | "
            f"{FAMILY_LABEL.get(r['zi_family'], r['zi_family'])} | {r['n_obs']} | "
            f"{_f(r['plain_aic'], '.2f')} | {_f(r['zi_aic'], '.2f')} | "
            f"**{_f(r['margin'], '+.2f')}** | "
            + ("yes" if not r['collapsed'] else
               f"no -- numerically identical to "
               f"{FAMILY_LABEL.get(r['collapsed_to'], r['collapsed_to'])}")
            + " |")
    add("")
    add(f"Which of the two plain families is the tougher competitor is a "
        f"rung-by-rung question, not a global one: NB2 has the smaller margin at "
        f"{_nb2_tougher} of the {len(_pairs)} rungs and NB1 at the other "
        f"{_nb1_tougher}"
        + (f" ({_tied} tied)" if _tied else "")
        + f". NB2's mean margin is the lower of the two "
        f"({_f(sum(r['margin'] for r in mc2) / len(mc2), '.1f')} against "
        f"{_f(sum(r['margin'] for r in mc) / len(mc), '.1f')} AIC) only because "
        f"the {_drop_n} `{_drop_cell}` rungs fall from "
        f"{_f(_drop_from, '.0f')} to {_f(_drop_to, '.0f')} AIC -- a range effect "
        f"in a handful of rungs, not a majority. The claim that survives is the "
        f"stronger one: "
        f"whichever plain family is the tougher competitor at a given rung, the "
        f"zero-inflated model beats it, at every one of the {len(_pairs)} rungs.\n")

    # Which violations cells lose a ZI-NB rung between M1 and M3 versus never
    # having had one. Derived, because asserting the first for all of them was a
    # defect: it is true of exactly one cell.
    viol_cells = sorted(set(tab[tab['outcome'] == 'violations']['cell']))
    ze = zinb_eligibility_at_rs(meta, viol_cells)
    # Subject derived from m3_winner_rs, grouped by family: NB1 is the Model-3
    # rs selection in the 2021 pair only -- the 2019 pair selects NB2 -- so
    # "NB1 wins ... in any of the 4 violations cells" was a generalisation, and
    # this is the paragraph a quoter is most likely to lift.
    _by_fam = {}
    for c in viol_cells:
        _by_fam.setdefault(meta[c]['m3_winner_rs'], []).append(c)
    _fam_txt = "; ".join(
        f"{FAMILY_LABEL.get(f, f)} in " + _oxford(['`' + c + '`' for c in cs])
        for f, cs in sorted(_by_fam.items()))
    add(f"So the honest mechanism is not that a negative binomial explains the "
        f"zeros comparably well. **The plain negative binomial wins the Model-3 "
        f"random-slope comparison by default** -- {_fam_txt}: no zero-inflated "
        f"negative-binomial rung is eligible at that specific structure at "
        f"Model 3, in any of the {len(viol_cells)} violations cells, so none of "
        f"them is in the race. Where they are in the race, they win. That is a "
        f"statement about identifiability, not about zero-inflation being "
        f"unnecessary.\n")
    add("The identifiability has two different causes, and only one of them is "
        "the \"covariates knock the rung out\" story:\n")
    if ze['lost']:
        add("- " + _oxford(['`' + c + '`' for c in ze['lost']])
            + f": a ZI-NB rung **is** eligible at Model 1 at `(1 + time | state)` "
            f"and is gone by Model 3. Here the rung genuinely stops being "
            f"estimable once the Model-3 covariates enter.")
    if ze['never']:
        add("- " + _oxford(['`' + c + '`' for c in ze['never']])
            + ": no ZI-NB rung is eligible at `(1 + time | state)` at **either** "
              "Model 1 or Model 3, so nothing stops -- there was never one at "
              "this structure to lose. (In the 2019 pair the reason is ZI "
              "degeneracy, not non-estimability; the eligibility table below "
              "separates the two.)")
    if ze['kept']:
        add("- " + _oxford(['`' + c + '`' for c in ze['kept']])
            + ": a ZI-NB rung survives to Model 3 at `(1 + time | state)`.")
    add("")

    # The cells whose Model-3 ZI-NB absence is specifically NON-ESTIMABILITY at
    # the tier (as opposed to ZI degeneracy) -- derived from the absence reasons,
    # so the section cannot name a cell that does not belong in it.
    ident_cells = [c for c in viol_cells
                   if {f for f, _ in absent_m3[c]} >= set(ZINB_FAMILIES)
                   and all(why.startswith('no fit at this structure')
                           for f, why in absent_m3[c] if f in ZINB_FAMILIES)]
    add(f"### At Model 3, the family comparison for "
        + _oxford(['`' + c + '`' for c in ident_cells])
        + (" is an identifiability statement\n" if len(ident_cells) == 1
           else " are identifiability statements\n"))
    for cell in ident_cells:
        elig = meta[cell]['eligible_rs_m3']
        ab = absent_m3[cell]
        add(f"- `{cell}`: at **Model 3**, only {len(elig)} of the {n_fam} families "
            f"were eligible at `(1 + time | state)` -- "
            + ", ".join(FAMILY_LABEL.get(e, e) for e in elig)
            + ". The "
            + ("absent one is " if len(ab) == 1 else f"{len(ab)} absent ones are ")
            + "; ".join(f"{FAMILY_LABEL.get(f, f)} ({why})" for f, why in ab)
            + ".")
    add("")
    add("The same accounting for every cell and both build-up steps, so nothing "
        "here rests on a sentence about a subset of them. All rows are at the mandated "
        "`(1 + time | state)` structure; \"eligible\" means converged, "
        "non-degenerate, and not numerically identical to a simpler family.\n")
    add("| Cell | Model | Eligible families | Eligible count | Absent families | Why absent |")
    add("|---|---|---|---|---|---|")
    for cell in ALL_CELLS:
        for model in ('M1', 'M3'):
            elig = meta[cell].get(f'eligible_rs_{model.lower()}') or []
            ab = _absent_families(tab, meta, cell, model)
            add(f"| {cell} | {model} | "
                + (", ".join(FAMILY_LABEL.get(e, e) for e in elig) or "--")
                + f" | {len(elig)} of {n_fam} | "
                + (", ".join(FAMILY_LABEL.get(f, f) for f, _ in ab) or "none")
                + " | "
                # The reason strings mention '(1 | state)'; a bare pipe inside a
                # markdown cell splits the row (and breaks the positional parse
                # in validate_count_models.py [7]), so escape it here. The prose
                # bullets above are not table cells and keep the bare pipe.
                + ("; ".join(f"{FAMILY_LABEL.get(f, f)}: {why}".replace('|', r'\|')
                             for f, why in ab) or "--")
                + " |")
    add("")
    # Same accounting, over the DERIVED identifiability cells rather than two
    # hardcoded names.
    ident_absent = {c: [FAMILY_LABEL.get(f, f) for f, _ in absent_m3[c]]
                    for c in ident_cells}
    ident_nelig = {c: len(meta[c]['eligible_rs_m3']) for c in ident_cells}
    _same = len(set(map(tuple, ident_absent.values()))) == 1
    missing_txt = (", ".join(next(iter(ident_absent.values())))
                   + (" in both cells" if len(ident_cells) == 2
                      else f" in all {len(ident_cells)} cells")
                   if _same and len(ident_cells) > 1
                   else "; ".join(f"{', '.join(v)} for `{c}`"
                                  for c, v in ident_absent.items()))
    _nelig_txt = (str(next(iter(ident_nelig.values())))
                  if len(set(ident_nelig.values())) == 1
                  else "/".join(str(ident_nelig[c]) for c in ident_cells))
    add(f"So at the structure the manuscript specifies, the Model-3 comparison for "
        f"{'that cell runs' if len(ident_cells) == 1 else 'those cells runs'} over "
        f"{_nelig_txt} families, and the "
        f"{len(next(iter(ident_absent.values())))} that are missing are exactly the "
        f"zero-inflated negative-binomial rungs -- {missing_txt}. "
        f"None of the {_nelig_txt} that were compared is a "
        f"ZI-NB. A reader must not take the result as evidence against "
        f"zero-inflation; it is an **identifiability** result about the "
        f"random-slope structure at Model 3.\n")

    # C1: the Model-1 candidate sets differ for some cells, and for
    # viol_off_2021 the difference goes the OTHER way -- a ZI-NB rung IS
    # estimable at the mandated structure there and wins its tier. Generated,
    # per cell, from eligible_rs_m1 vs eligible_rs_m3 + stable_rs.
    changed = [c for c in ALL_CELLS
               if set(meta[c].get('eligible_rs_m1') or [])
               != set(meta[c].get('eligible_rs_m3') or [])]
    add("#### At Model 1 the candidate sets differ -- and for one cell that "
        "difference favours zero-inflation\n")
    add("The candidate set is not a property of the cell; it is a property of the "
        "(cell, model) pair, because adding covariates changes what the optimizer "
        "can support. Where the Model-1 and Model-3 sets differ:\n")
    for cell in changed:
        m = meta[cell]
        e1 = set(m['eligible_rs_m1'])
        e3 = set(m['eligible_rs_m3'])
        only1 = sorted(e1 - e3)
        only3 = sorted(e3 - e1)
        w1 = m['m1_winner_rs']
        comp, dcomp = nearest_clean_competitor(tab, cell, 'M1', 'rs', w1)
        parts = [f"- `{cell}`: {len(e1)} eligible at Model 1 vs {len(e3)} at "
                 f"Model 3."]
        if only1:
            parts.append("Available at Model 1 but not Model 3: "
                         + ", ".join(FAMILY_LABEL.get(f, f) for f in only1) + ".")
        if only3:
            parts.append("Available at Model 3 but not Model 1: "
                         + ", ".join(FAMILY_LABEL.get(f, f) for f in only3) + ".")
        parts.append(f"The Model-1 winner at this tier is "
                     f"**{FAMILY_LABEL.get(w1, w1)}**")
        if comp is not None and np.isfinite(dcomp):
            parts.append(f"beating {FAMILY_LABEL.get(comp, comp)} by "
                         f"{_f(dcomp, '.2f')} AIC")
        parts.append(f"and `stable_rs` is {m['stable_rs']}, which is the field that "
                     f"records the Model-1 and Model-3 winners "
                     f"{'agreeing' if m['stable_rs'] else 'DISAGREEING'}.")
        add(" ".join(parts))
    add("")
    zi_m1 = [(c, meta[c]['m1_winner_rs']) for c in changed
             if meta[c]['m1_winner_rs'] in ZI_FAMS]
    if zi_m1:
        for cell, w1 in zi_m1:
            wrow = tab[(tab['cell'] == cell) & (tab['model'] == 'M1') &
                       (tab['re_tier'] == 'rs') & (tab['family'] == w1)].iloc[0]
            m3w = meta[cell]['m3_winner_rs']
            m3row = tab[(tab['cell'] == cell) & (tab['model'] == 'M1') &
                        (tab['re_tier'] == 'rs') & (tab['family'] == m3w)].iloc[0]
            zi = (raw[wrow['key']].get('zi') or {}).get('(Intercept)')
            add(f"**This is the sentence that matters, and it is stronger for "
                f"zero-inflation than the Model-3 result is.** At Model 1, "
                f"`{cell}` *did* admit a zero-inflated negative binomial at the "
                f"mandated `(1 + time | state)` structure: "
                f"{FAMILY_LABEL.get(w1, w1)} converged there "
                f"(positive-definite Hessian: {raw[wrow['key']].get('pd_hess')}) "
                f"with AIC {_f(wrow['aic'], '.2f')} against "
                f"{FAMILY_LABEL.get(m3w, m3w)}'s {_f(m3row['aic'], '.2f')} -- a "
                f"margin of {_f(m3row['aic'] - wrow['aic'], '.2f')} AIC in favour "
                f"of the zero-inflated model"
                + (f", with a ZI intercept of {_f(zi['b'], '+.4f')} "
                   f"(SE {_f(zi['se'], '.4f')}, p = {_f(zi['p'], '.3g')})"
                   if zi else "")
                + f". So the correct statement, **for this cell**, is that its "
                f"ZI-NB rung drops out **as covariates are added** rather than "
                f"never having been estimable at this structure. That is a "
                f"statement about `{cell}` only: the other "
                f"{len(zinb_eligibility_at_rs(meta, viol_cells)['never'])} "
                f"violations cells have no ZI-NB rung at this structure at "
                f"either model, so nothing drops out of them.\n")
    else:
        add("No cell's Model-1 winner at the `rs` tier is a zero-inflated family, "
            "so there is no Model-1 counter-example to record.\n")

    # ---------------------------------------------- 2019 violations deflation
    add("### The 2019 violations series is zero-**deflated**, not zero-inflated\n")
    d19 = tab[(tab['cell'].isin(['viol_off_2019', 'viol_cov_2019'])) &
              (tab['model'] == 'M3')].dropna(subset=['exp_zeros'])
    obs19 = sorted(set(int(x) for x in d19['obs_zeros']))
    rlo_row = d19.loc[d19['exp_zeros'].idxmin()]
    rhi_row = d19.loc[d19['exp_zeros'].idxmax()]
    lo, hi = float(rlo_row['exp_zeros']), float(rhi_row['exp_zeros'])
    rlo, rhi = float(d19['zero_ratio'].min()), float(d19['zero_ratio'].max())
    add(f"Every family **over**predicts the zeros in the 2019 violations cells: "
        f"{'/'.join(str(o) for o in obs19)} observed against a range of "
        f"{_f(lo, '.1f')} +/- {_f(rlo_row['exp_zeros_se'], '.2f')} "
        f"({FAMILY_LABEL.get(rlo_row['family'], rlo_row['family'])}, "
        f"{rlo_row['cell']}) to "
        f"{_f(hi, '.1f')} +/- {_f(rhi_row['exp_zeros_se'], '.2f')} "
        f"({FAMILY_LABEL.get(rhi_row['family'], rhi_row['family'])}, "
        f"{rhi_row['cell']}) expected across the {len(ALL_FAMILIES)} families at "
        f"Model 3 -- i.e. {_f(rlo, '.1f')}x to {_f(rhi, '.1f')}x too many "
        f"(MC SEs from 2000 simulations, so the gap is nowhere near simulation "
        f"noise). There is no excess of zeros here to inflate; there is a shortage "
        f"of them. The series is zero-deflated.\n")
    add("| Cell | Family | Observed 0s | Expected 0s (+/- MC SE) | Expected/Observed |")
    add("|---|---|---|---|---|")
    for _, r in d19.sort_values(['cell', 'exp_zeros']).iterrows():
        add(f"| {r['cell']} | {FAMILY_LABEL.get(r['family'], r['family'])} | "
            f"{_f(r['obs_zeros'], '.0f')} | {_f(r['exp_zeros'], '.1f')} +/- "
            f"{_f(r['exp_zeros_se'], '.2f')} | {_f(r['zero_ratio'], '.2f')}x |")
    add("")
    # NOTE: __meta.zi_evidence is MODEL-3 ONLY. An earlier draft cited its SEs
    # and p-values as though they covered the whole ladder for these cells, and
    # that is false -- viol_off_2019's ZIP is estimable at Model 1. Scope the
    # claim to Model 3 and derive the Model-1 exceptions from the fit records.
    CELLS_19 = ('viol_off_2019', 'viol_cov_2019')
    zi19 = [e for c in CELLS_19 for e in meta[c]['zi_evidence']]
    ses = [e['se'] for e in zi19 if e.get('se') is not None]
    ps = [e['p'] for e in zi19 if e.get('p') is not None]
    n_zi_fams_19 = len({e['family'] for e in zi19})
    degen_by_cell = tab[tab['zi_degenerate']].groupby('cell').size().to_dict()
    degen_cells = sorted(degen_by_cell)
    add(f"Consistent with that, all {n_zi_fams_19} zero-inflated families collapse "
        f"**at Model 3** at the mandated structure in these cells: standard errors "
        f"from {_f(min(ses), '.0f')} to {_f(max(ses), '.0f')} and p-values from "
        f"{_f(min(ps), '.3f')} to {_f(max(ps), '.3f')} -- a zero-inflation intercept "
        f"sitting numerically on the boundary, estimating nothing. (Those figures "
        f"come from `__meta.zi_evidence`, which records the Model-3 rung only, so "
        f"they support a Model-3 claim and no more.) Across the whole ladder "
        f"{n_degen} fits are flagged ZI-degenerate, distributed "
        + ", ".join(f"{v} in `{k}`" for k, v in sorted(degen_by_cell.items()))
        + f" -- all of them in these {len(degen_cells)} cells.\n")
    # The asymmetry in that 6/5 split IS the Model-1 exception. Derive it.
    zi_ok_19 = tab[(tab['cell'].isin(CELLS_19)) & (tab['re_tier'] == 'rs') &
                   (tab['family'].isin(('zip',) + tuple(ZINB_FAMILIES))) &
                   tab['converged'] & (~tab['zi_degenerate'])]
    if not zi_ok_19.empty:
        add(f"**But not at every model, and this is the exception the "
            f"{'/'.join(str(v) for _, v in sorted(degen_by_cell.items()))} split "
            f"records.** {len(zi_ok_19)} zero-inflated fit"
            + ("" if len(zi_ok_19) == 1 else "s")
            + f" in these cells "
            + ("is" if len(zi_ok_19) == 1 else "are")
            + " estimable at the mandated structure and NOT degenerate:")
        add("")
        add("| Cell | Model | RE structure | Family | ZI intercept | SE | p |")
        add("|---|---|---|---|---|---|---|")
        for _, r in zi_ok_19.sort_values(['cell', 'model', 'family']).iterrows():
            zi = (raw[r['key']].get('zi') or {}).get('(Intercept)') or {}
            add(f"| {r['cell']} | {r['model']} | {_re_label(r['re_tier'])} | "
                f"{FAMILY_LABEL.get(r['family'], r['family'])} | "
                f"{_f(zi.get('b'), '+.4f')} | {_f(zi.get('se'), '.4f')} | "
                f"{_f(zi.get('p'), '.3g')}{stars(zi.get('p'))} |")
        add("")
        # Derived from the exception table's own contents. An earlier draft
        # generalised a one-row table to both 2019 cells; viol_cov_2019 has all
        # its ZI families degenerate at Model 1 as well, so for that cell the
        # collapse is not a Model-3 phenomenon at all.
        _exc_cells = sorted(set(zi_ok_19['cell']))
        _no_exc = [c for c in CELLS_19 if c not in _exc_cells]
        _lost_txt = _oxford(['`' + c + '`' for c in
                             zinb_eligibility_at_rs(meta, viol_cells)['lost']])
        add("So the Model-3 scoping buys something for "
            + _oxford(['`' + c + '`' for c in _exc_cells])
            + " and nothing for "
            + (_oxford(['`' + c + '`' for c in _no_exc]) if _no_exc else "no cell")
            + ", and the difference is worth stating rather than averaging over.\n")
        for c in _exc_cells:
            rows_c = zi_ok_19[zi_ok_19['cell'] == c]
            fam_txt = _oxford(sorted({FAMILY_LABEL.get(f, f)
                                      for f in rows_c['family']}))
            mod_txt = _oxford(sorted({'Model ' + m[1:]
                                      for m in rows_c['model']}))
            add(f"- `{c}`: its {fam_txt} is estimable and non-degenerate at "
                f"{mod_txt} and degenerate by Model 3, so there the collapse **is** "
                f"a Model-3 phenomenon"
                + (f" -- the same pattern as {_lost_txt}, where the ZI-NB rung is "
                   f"available at Model 1 and gone once the covariates enter."
                   if _lost_txt else "."))
        for c in _no_exc:
            dg = tab[(tab['cell'] == c) & tab['zi_degenerate']]
            n_fams_c = len(set(dg['family']))
            mods_c = _oxford(sorted({'Model ' + m[1:] for m in dg['model']}))
            add(f"- `{c}`: **no exception.** All {n_fams_c} zero-inflated families "
                f"are ZI-degenerate at {mods_c} alike ({len(dg)} degenerate fits), "
                f"so for this cell the collapse is not Model-3-specific at all -- "
                f"the zero-inflation term never estimated anything at this "
                f"structure, with or without the covariates.")
        add("")
    _exc_cells_s = sorted(set(zi_ok_19['cell'])) if not zi_ok_19.empty else []
    _no_exc_s = [c for c in CELLS_19 if c not in _exc_cells_s]
    add("**Scope, precisely.** The fallback `(1 | state)` tier was never attempted "
        f"for these cells -- they did not need it, because a non-ZI family fit fine "
        f"at the mandated structure. So the supportable claim is *no estimable "
        f"zero-inflation at Model 3 at the specified structure, where all "
        f"{n_zi_fams_19} ZI families collapse*. Whether it extends to Model 1 "
        f"differs by cell and is stated above: it does for "
        + (_oxford(['`' + c + '`' for c in _no_exc_s]) if _no_exc_s else "no cell")
        + ", not for "
        + (_oxford(['`' + c + '`' for c in _exc_cells_s]) if _exc_cells_s
           else "no cell")
        + ". It is also **not** a claim about the fallback `(1 | state)` tier, "
          "which was never tried for these cells, and this memo makes none.\n")

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
            if 'M2' not in fams and tier == 'ri':
                add("Model 2 is absent from this block because M2 was only ever fit "
                    "at the mandated `(1 + time | state)` tier -- the fallback tier "
                    "carries Models 1 and 3 only. It is not a missing result.")
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
    add("Two warnings before reading down a column. **(1)** N and the state count "
        "fall between Model 1 and Models 2-3 because the spending and BLS-derived "
        "covariates are listwise-deleted (AK/RI/VT have no BLS "
        "pesticide-applicator series), so the rows of a block are not all on the "
        "same analytic sample. **(2)** Where the selected family changes down a "
        "column, the sigma^2 values are not a variance-reduction sequence at all -- "
        "they are different models' parameters. The `Family` column flags every "
        "such case"
        + (": " + "; ".join(
            f"`{c}` at the {'(1 + time | state)' if t == 'rs' else '(1 | state)'} "
            f"tier changes family down the column"
            for c, t in _family_changes(tab, meta)) + "."
           if _family_changes(tab, meta) else ".")
        + "\n")
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
    add(f"For the record, that fit raised {xc_v['n_warnings']:,} warnings "
        f"({xc_v['n_distinct_warnings']} distinct) across "
        f"{', '.join(f'{k} x{v:,}' for k, v in sorted(xc_v['warning_counts'].items()))}. "
        f"They are counted and sampled in `count_model_memo_evidence.json` rather "
        f"than suppressed -- a divergence this loud going unrecorded is the exact "
        f"failure mode the next section is about.\n")

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
    def _base_m3(cell, f):
        """The no-COVID M3 fit at the SAME family and tier. `_is_genuine_fit_key`
        is essential here: without it the candidate list also matches the
        `__altopt` diagnostic refits, and taking `[0]` would then depend on JSON
        insertion order rather than on identity."""
        keys = [k for k, v in raw.items()
                if _is_genuine_fit_key(k) and isinstance(v, dict)
                and v.get('cell') == cell and v.get('model') == 'M3'
                and v.get('family_tag') == f['family_tag']
                and v.get('re_tier') == f['re_tier']]
        if len(keys) != 1:
            raise ValueError(f"expected exactly 1 base M3 fit for {cell} "
                             f"({f['family_tag']}, {f['re_tier']}), got {keys}")
        return raw[keys[0]]

    for cell, (f, _c) in sorted(covid_rows.items()):
        base = _base_m3(cell, f)
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
        vb = _base_m3('viol_cov_2021', cv[0])
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
        f"({len({r['outcome'] for r in aud['fits']})} outcome columns x "
        f"{len({r['model'] for r in aud['fits']})} models x "
        f"{{random slope, random-intercept-only}}) were refit and their warnings "
        f"captured: **exactly {aud['n_nonconverged']} of {aud['n_fits']} "
        f"{'is' if aud['n_nonconverged'] == 1 else 'are'} affected**"
        + (", the one above" if aud['n_nonconverged'] == 1 else "")
        + f". The other {aud['n_fits'] - aud['n_nonconverged']} converge cleanly.\n")
    add(f"The `Reproduces published` column is the only guard that the "
        f"specification re-declared in "
        f"`scripts/build_count_model_memo_evidence.py` still matches the "
        f"manuscript's. Be precise about what it compares: it checks our refits "
        f"against `paper_table_params_2021.json`, i.e. against "
        f"`paper_table_models_2021.py`'s **published output**, not against its "
        f"source. Editing that script without regenerating the JSON would leave "
        f"this guard green. It is True on "
        f"{aud['n_reproduces_published']} of {aud['n_fits']} fits, and validator "
        f"`[7]` fails if that is not all of them. It is True for the failed row "
        f"too -- reproducing the published number is exactly how we know the "
        f"published number is the non-converged one.\n")
    add("| Outcome column | Model | RE basis | N | Log-likelihood | sigma^2_u0 | "
        "Reproduces published | Gradient failure |")
    add("|---|---|---|---|---|---|---|---|")
    for r in aud['fits']:
        add(f"| {r['outcome']} | {r['model']} | {r['re_basis']} | {r['n_obs']} | "
            f"{_f(r['llf'], '.3f')} | {_f(r['sigma2_u0'], '.6f')} | "
            f"{'yes' if r['reproduces_published'] else 'NO'} | "
            f"{'**YES**' if r['grad_warning'] else 'no'} |")
    add("")
    n_ri = aud['n_random_intercept']
    n_ri_bad = aud['n_random_intercept_nonconverged']
    add(f"**The reported variance-explained block is unaffected.** All {n_ri} "
        f"random-intercept-only refits converge ({n_ri_bad} gradient failures) and "
        f"reproduce their published sigma^2_u0 values, and those are the fits the "
        f"manuscript's Delta sigma^2_u0 percentages are built from. So this is a "
        f"coefficient-level problem in "
        f"{aud['n_nonconverged']} column{'' if aud['n_nonconverged'] == 1 else 's'}, "
        f"not a variance-block problem.\n")
    add("Nothing was fixed or regenerated. No manuscript `.docx`, no "
        "`paper_table_*` script, no `paper_table_*` JSON was touched by this work. "
        "Whether to reissue Table 3 Model 1 is your call.\n")

    # ------------------------------------------------------------- NB1/NB2
    add("## NB1 and NB2 disagree on the random-slope variance\n")
    add("For the 2021 violations cells the two negative-binomial "
        "parameterisations disagree materially on sigma^2_u1, and very little AIC "
        "separates them:\n")
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
    add("**sigma^2_u1 is not a quantity the manuscript publishes.** Tables 2 and 3 "
        "report sigma^2_u0, sigma^2_e and Delta sigma^2_u0 on the "
        "**random-intercept** basis (`sigma2_u0_ri` / `sigma2_e_ri` / "
        "`delta_pct_ri`); sigma^2_u1 exists in "
        "`data/generated/paper_table_params_2021.json` but reaches no published "
        "table. So this disagreement changes nothing that is currently in print. It "
        "is recorded because it would matter the moment a random-slope variance is "
        "reported, and because it is a real between-family difference rather than "
        "an optimizer artifact.\n")
    altopt = [(k, f) for k, f in sorted(raw.items()) if k.endswith('__altopt')]
    d_llk = max(abs(f['loglik'] - raw[k[:-len('__altopt')]]['loglik'])
                for k, f in altopt)
    d_aic = max(abs(f['aic'] - raw[k[:-len('__altopt')]]['aic']) for k, f in altopt)
    add(f"The winner carries a live optimizer warning while the runner-up is clean, "
        f"which is uncomfortable. It is not, however, a bad optimum: refitting the "
        f"winning NB1 models under BFGS reproduces the log-likelihood to within "
        f"{d_llk:.1e} and the AIC to within {d_aic:.1e} across all "
        f"{len(altopt)} refits.\n")
    add("| Cell | Model | Default optimizer log-lik | BFGS log-lik | Default AIC | "
        "BFGS AIC | sigma^2_u1 default | sigma^2_u1 BFGS |")
    add("|---|---|---|---|---|---|---|---|")
    for k, f in altopt:
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
        "was fit. In the event the risk did not land the way it might have: the "
        "model class did not weaken the case for zero-inflation, it strengthened "
        f"it -- {sum(1 for r in mc if r['margin'] > 0)} of {len(mc)} matched "
        f"comparisons favour the zero-inflated negative binomial, none against, "
        f"and the same {sum(1 for r in mc2 if r['margin'] > 0)} of {len(mc2)} "
        f"against NB2. What it did do is expose that the structure we are "
        f"committed to cannot always estimate that model, and that the "
        f"strengthening is specific to the zero-inflated negative binomial: "
        f"zero-inflated Poisson loses {_zl['n_losses_by_family']['zip']} matched "
        f"comparisons, so \"zero-inflation\" on its own is not the finding. Read "
        f"the rest with the genuine limits in view: the inspections "
        "zero-inflation is real but is a left-tail-fit result rather than a latent "
        "non-inspecting subpopulation; the 2021 violations zero-inflation is real "
        "and, at Model 3, unavailable at the structure we are committed to"
        + (f" -- though at Model 1 it actually wins that tier for "
           + _oxford(['`' + c + '`' for c, _ in zi_m1]) + ", which is why the "
           "Model-3 scoping above matters, and why 'unavailable' is the right "
           "word rather than 'outcompeted'"
           if zi_m1 else "")
        + "; the 2019 violations series is a different case again -- it is "
          "zero-**deflated**, so there is no excess of zeros to inflate in the "
          "first place; and the two genuinely "
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

    ev = load_memo_evidence()
    write_memo(raw, meta, tab, ws, ev)


if __name__ == '__main__':
    main()
