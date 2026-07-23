"""
Build the YEAR-MATCHED, time-varying H-2A-to-farmworker ratio (Joe's correction).

The old h2a_per_farmworker was a single Level-2 number: mean 2011-2019 certified
H-2A workers divided by the BLS *2011* farmworker snapshot -- numerator and
denominator on different time bases, and the denominator frozen at one suppressed
year. This rebuilds it as a proper Level-1 panel, matched by year:

    h2a_per_farmworker[state, year]
        = H-2A workers certified[state, year]  (DOL, dol_var1_workers_by_state_annual)
        / BLS farmworkers 45-2092[state, year]  (OEWS, bls_oews_panel_2011_2019)

Gap handling (both series are smooth, monotone-ish over the window):
  - DOL has no 2013 at all -> linear-interpolate certified workers within state.
  - BLS suppresses a few state-years -> interpolate + ffill/bfill within state.
Result is a complete 50-state x 9-year panel with no listwise holes. The ratio is
z-scored across all state-years (pooled) for entry as h2a_per_farmworker_z.

Output: data/generated/h2a_ratio_panel.csv
    columns: state, year, h2a_workers_cert, emp_farmworker,
             h2a_per_farmworker, h2a_per_farmworker_z
"""

import pandas as pd
import numpy as np

RAW = '/Users/keshavgoel/Research/data/raw/'
GEN = '/Users/keshavgoel/Research/data/generated/'
YEARS = list(range(2011, 2020))

STATE_ABBREV_TO_NAME = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska',
    'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina',
    'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
    'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'}
US_STATES_50 = sorted(STATE_ABBREV_TO_NAME.values())

grid = pd.MultiIndex.from_product([US_STATES_50, YEARS],
                                  names=['state', 'year']).to_frame(index=False)

# ---- numerator: DOL certified H-2A workers, year-matched --------------------
dol = pd.read_csv(RAW + 'dol_var1_workers_by_state_annual.csv')
dol['state'] = dol['state'].map(STATE_ABBREV_TO_NAME)
dol = dol[dol['state'].isin(US_STATES_50) & dol['year'].between(2011, 2019)]
num = grid.merge(dol[['state', 'year', 'workers_certified']], on=['state', 'year'], how='left')

# ---- denominator: BLS farmworkers 45-2092, year-matched ---------------------
bls = pd.read_csv(RAW + 'bls_oews_panel_2011_2019.csv')
fw = bls[bls['occ_code'] == '45-2092'][['state', 'year', 'tot_emp']].rename(
    columns={'tot_emp': 'emp_farmworker'})
den = num.merge(fw, on=['state', 'year'], how='left').sort_values(['state', 'year'])


# ---- interpolate within-state gaps (2013 DOL; sporadic BLS suppression) -----
def fill(g, col):
    g[col] = g[col].interpolate(method='linear', limit_direction='both')
    return g


panel = den.copy()
panel = panel.groupby('state', group_keys=False).apply(
    lambda g: fill(fill(g, 'workers_certified'), 'emp_farmworker'))
panel = panel.rename(columns={'workers_certified': 'h2a_workers_cert'})

panel['h2a_per_farmworker'] = panel['h2a_workers_cert'] / panel['emp_farmworker']
m, s = panel['h2a_per_farmworker'].mean(), panel['h2a_per_farmworker'].std()
panel['h2a_per_farmworker_z'] = (panel['h2a_per_farmworker'] - m) / s

panel = panel[['state', 'year', 'h2a_workers_cert', 'emp_farmworker',
               'h2a_per_farmworker', 'h2a_per_farmworker_z']]
panel.to_csv(GEN + 'h2a_ratio_panel.csv', index=False)

print(f"Wrote {GEN}h2a_ratio_panel.csv: {len(panel)} state-years, "
      f"{panel['state'].nunique()} states, {panel['year'].nunique()} years")
print(f"Complete (no NaN ratio): {panel['h2a_per_farmworker'].notna().all()}")
print("\nRatio by year (mean across states):")
print(panel.groupby('year')['h2a_per_farmworker'].mean().round(3).to_string())
print("\nTop 5 state-mean ratios:")
print(panel.groupby('state')['h2a_per_farmworker'].mean().nlargest(5).round(3).to_string())
