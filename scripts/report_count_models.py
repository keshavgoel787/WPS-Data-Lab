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
    always filter model == 'M3' for the headline family.
  - Two cells (viol_off_2021, viol_cov_2021) carry a SECOND (1 | state)
    winner because no ZI-NB rung could converge at (1 + time | state). Both
    tiers are reported, labelled by re_tier, and never merged into one row.
  - zi_degenerate fits (11 of them; ZI intercepts numerically at the boundary,
    loglik identical to their non-ZI counterpart) are kept in the table but
    excluded from every LRT and from every "better zero-fit" comparison.
  - AIC/LRT comparisons are only ever drawn within one (cell, model, re_tier)
    group, after checking n_obs is constant there.
  - The nearest clean competitor is reported unconditionally (no delta_aic
    <= 10 cutoff) -- viol_cov_2021's nearest clean competitor sits at ~10.74
    AIC, just past that cutoff, and is exactly the decision-relevant case.

Selection protocol (spec 5.4): AIC/BIC across the six families available at
each tier, LRTs on the nested pairs ONLY (Poisson subset NB2, NB2 subset
ZINB, ZINB subset ZINB+ZI-RE) -- NB1 is not nested in NB2 and gets AIC/BIC
only, no LRT. Both LRTs are BOUNDARY tests: dispersion -> infinity and
zero-inflation probability -> 0 sit on the edge of the parameter space, so the
nominal chi-square p-values are conservative (anti-conservative in the usual
direction -- true significance may be understated less than the nominal p
implies, but the classical asymptotics do not strictly apply). No Vuong test:
it is routinely misapplied to non-nested ZI comparisons.

Run: python3 scripts/report_count_models.py
Output: data/generated/count_model_comparison.csv
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
NESTED = {'nbinom2': 'poisson', 'zinb': 'nbinom2', 'zinb_re': 'zinb'}

TERM_LABELS = [
    ('(Intercept)', 'Intercept'),
    ('log_inspections', 'log Inspections'),
    ('time', 'Time'), ('time2', 'Time^2'), ('time3', 'Time^3'),
    ('SPEND_APP_z', 'Spending/applicator'), ('SPEND_WORK_z', 'Spending/farmworker'),
    ('lii_2017_z', 'Labor Intensity'),
    ('h2a_per_farmworker_z', 'H-2A:farmworker (yr-matched)'),
    ('dol_demand_met_pct_z', 'H-2A demand-met %'), ('pct_flc_z', '% H-2A to FLC'),
]

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
    fits), and the 6 __altopt diagnostic refits. Adds nested-pair LRTs,
    computed only within (cell, model, re_tier) and only between converged,
    non-degenerate fits.
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
            'n_obs': f['n_obs'], 'n_states': f['n_states'],
            'aic': f.get('aic'), 'bic': f.get('bic'),
            'loglik': f.get('loglik'), 'df': f.get('df'),
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

    # LRTs on nested pairs, within the same (cell, model, re_tier). Lookup is
    # keyed on the FIELDS (cell, model, family, re_tier), never on parsing the
    # JSON key -- the __ri2 suffix convention is an R-script storage detail,
    # not part of the identity of a fit (see validate_ladder()'s tier_fit()).
    lookup = {(r['cell'], r['model'], r['family'], r['re_tier']): i
              for i, r in tab.iterrows()}
    tab['lrt_vs'] = ''
    for c in ('lrt_chisq', 'lrt_df', 'lrt_p'):
        tab[c] = np.nan
    for i, r in tab.iterrows():
        parent_family = NESTED.get(r['family'])
        if parent_family is None or r['zi_degenerate'] or not r['converged']:
            continue
        pidx = lookup.get((r['cell'], r['model'], parent_family, r['re_tier']))
        if pidx is None:
            continue
        prow = tab.loc[pidx]
        if not prow['converged'] or prow['zi_degenerate']:
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
    return tab


def nearest_clean_competitor(tab, cell, model, re_tier, winner_family):
    """Among converged, non-degenerate, non-collapsed fits at
    (cell, model, re_tier) other than the winner, return
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
                'has_warning': m.get(f'winner_{tier}_has_warning'),
                'obs_zeros': w['obs_zeros'], 'exp_zeros': w['exp_zeros'],
                'exp_zeros_se': w['exp_zeros_se'],
                'zero_signed': m.get(f'winner_{tier}_zero_signed'),
                'zero_discrepancy_rel': m.get(f'winner_{tier}_zero_discrepancy'),
                'nearest_clean_competitor': comp_family,
                'delta_aic_vs_competitor': delta_aic,
                'zeros_partly_manufactured': m.get('zeros_partly_manufactured'),
                'has_better_zero_fit': m.get(f'has_better_zero_fit_{tier}'),
                'credible_better_zero_fit': m.get(f'credible_better_zero_fit_{tier}'),
            })
    return pd.DataFrame(rows)


def coefficient_table(raw, cell, re_tier):
    """Coefficient table for one cell at one RE tier. For M1/M3 this picks
    the fit marked is_winner_rs (re_tier='rs') or is_winner_ri (re_tier='ri')
    directly off the fit records -- never off __meta or a key-parsing
    heuristic. M2 is only ever fit at the rs tier (validate_ladder() checks
    this): requesting re_tier='ri' correctly omits M2 rather than fabricating
    or misattributing it.

    Because the winning family can differ between M1 and M3 within the same
    tier (viol_off_2021's rs tier: M1 winner is zinb_re, M3 winner is
    nbinom1 -- a candidate-set change, not a preference reversal, see
    task-5-report.md), each populated column also carries its own
    `{model}_family` so a reader is never shown two different families'
    coefficients side by side without knowing it.
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
        elif f.get('model') == 'M2' and re_tier == 'rs':
            picked['M2'] = (f['family_tag'], f)

    rows = []
    for term, label in TERM_LABELS:
        row = {'term': label}
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


def main():
    raw = load_results()
    meta = load_meta(raw)
    tab = selection_table(raw)
    tab.to_csv(GEN + 'count_model_comparison.csv', index=False)
    print(f"Wrote {GEN}count_model_comparison.csv "
          f"({len(tab)} genuine fits; excludes 2 Gaussian round-trip + "
          f"6 __meta + 6 __altopt records from the 109 total)\n")

    ws = winner_summary(tab, meta)
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
    print()

    for cell in sorted(meta):
        g = tab[(tab['cell'] == cell) & (tab['model'] == 'M3')].sort_values(
            ['re_tier', 'aic'], na_position='last')
        r0 = g.iloc[0]
        print("=" * 100)
        print(f"{cell} -- {r0['outcome']} {r0['window']}, exposure={r0['exposure']}, "
              f"M3, all RE tiers")
        print("=" * 100)
        print(f"{'tier':5}{'family':14}{'conv':>5}{'N':>5}{'AIC':>10}{'BIC':>10}"
              f"{'obs0':>6}{'exp0':>8}{'0-ratio':>8}{'LRT vs':>9}{'LRT p':>10}{'degen':>7}  winner")
        for _, r in g.iterrows():
            lrt_p = num(r['lrt_p'], '.3g') + stars(r['lrt_p']) if pd.notna(r['lrt_p']) else ''
            mark = ''
            if r['is_winner_rs']:
                mark += ' <-- rs winner'
            if r['is_winner_ri']:
                mark += ' <-- ri winner'
            print(f"{r['re_tier']:5}{FAMILY_LABEL.get(r['family'], r['family']):14}"
                  f"{'yes' if r['converged'] else 'NO':>5}{num(r['n_obs'], '.0f', 5)}"
                  f"{num(r['aic'], '.1f', 10)}{num(r['bic'], '.1f', 10)}"
                  f"{num(r['obs_zeros'], '.0f', 6)}{num(r['exp_zeros'], '.1f', 8)}"
                  f"{num(r['zero_ratio'], '.2f', 8)}{r['lrt_vs']:>9}{lrt_p:>10}"
                  f"{'YES' if r['zi_degenerate'] else '':>7}{mark}")
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
            for _, r in ct.iterrows():
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
    print("NOTE: both LRTs (Poisson<NB2, NB2<ZINB) are BOUNDARY tests -- dispersion->inf and")
    print("zero-inflation probability->0 sit on the edge of the parameter space, so nominal")
    print("chi-square p-values are conservative. NB1 is not nested in NB2: AIC/BIC only, no LRT.")
    print("zi_degenerate fits (ZI intercept numerically at the boundary) are shown but excluded")
    print("from every LRT and from every best-zero-fit comparison.")
    print("=" * 100)


if __name__ == '__main__':
    main()
