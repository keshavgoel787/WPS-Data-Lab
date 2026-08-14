"""
Rebuild the Level-2 SPEND_* variables over the EXTENDED 2011-2021 window.

Companion to rebuild_spend_bls_multiyear.py, which does the same thing for the
original 2011-2019 ECHO window. Both numerator and denominator are widened so the
spending variable spans the same years as the WPS DV panel:

  numerator   mean inflation-adjusted (2017 $) STAG obligation per state,
              2011-2021, summed from data/raw/spending_data_master.csv
              (the master is hand-curated to pesticide-relevant awards -- sum it as
              shipped; do NOT re-derive from all 7 CFDAs, which inflates ~43x)
  denominator mean of every NON-suppressed year 2011-2021 from
              data/raw/bls_oews_panel_2011_2021.csv

CPI-U is extended with the 2020 and 2021 BLS annual averages (258.811, 270.970);
base year stays 2017, unchanged from the rest of the project.

Carried over unchanged from spend_bls_variables.csv: land_area_sqmi, operations
(Census of Agriculture 2017 -- time-invariant, so the wider window doesn't affect
them). MS/RI/WV stay $0-coded, confirmed $0 in CFDA 66.700 rather than missing.

Output: data/generated/spend_bls_variables_multiyear_2021.csv
"""

import pandas as pd
import numpy as np

RAW = '/Users/keshavgoel/Research/data/raw/'
GEN = '/Users/keshavgoel/Research/data/generated/'

YEAR_MIN, YEAR_MAX = 2011, 2021
ZERO_SPEND_STATES = ['Mississippi', 'Rhode Island', 'West Virginia']

CPI_U = {
    2011: 224.939, 2012: 229.594, 2013: 232.957, 2014: 236.736,
    2015: 237.017, 2016: 240.007, 2017: 245.120,
    2018: 251.107, 2019: 255.657, 2020: 258.811, 2021: 270.970,
}

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

# ---- numerator: mean 2017-dollar obligation per state, 2011-2021 ---------------
sp = pd.read_csv(RAW + 'spending_data_master.csv')
sp = sp[sp['Year'].between(YEAR_MIN, YEAR_MAX)].copy()
sp['state'] = sp['State'].map(STATE_ABBREV_TO_NAME)
sp = sp[sp['state'].isin(US_STATES_50)]

agg = (sp.groupby(['state', 'Year'], as_index=False)['Total Obligation'].sum()
         .rename(columns={'Year': 'year', 'Total Obligation': 'spending_nominal'}))
agg['spending_2017'] = agg['spending_nominal'] * CPI_U[2017] / agg['year'].map(CPI_U)
mean_spend = (agg.groupby('state')['spending_2017'].mean()
                 .rename('mean_spending_2017').reset_index())

# ---- start from the existing file for land_area / operations -------------------
base = pd.read_csv(GEN + 'spend_bls_variables.csv')
base = base.drop(columns=['mean_spending_2017', 'emp_farmworker',
                          'emp_pesticide_app', 'emp_flc_supervisor'])
base = base.merge(mean_spend, on='state', how='left')
base.loc[base['state'].isin(ZERO_SPEND_STATES), 'mean_spending_2017'] = 0.0

# ---- denominators: mean non-suppressed employment 2011-2021 --------------------
bls = pd.read_csv(RAW + 'bls_oews_panel_2011_2021.csv')
emp = (bls.dropna(subset=['tot_emp'])
          .groupby(['state', 'occ_code'])['tot_emp'].mean().unstack('occ_code')
          .rename(columns={'37-3012': 'emp_pesticide_app',
                           '45-1011': 'emp_flc_supervisor',
                           '45-2092': 'emp_farmworker'}))
base = base.merge(emp[['emp_farmworker', 'emp_pesticide_app', 'emp_flc_supervisor']],
                  left_on='state', right_index=True, how='left')

# ---- recompute SPEND_* and z-scores --------------------------------------------
base['SPEND_WORK'] = base['mean_spending_2017'] / base['emp_farmworker']
base['SPEND_APP'] = base['mean_spending_2017'] / base['emp_pesticide_app']
base['SPEND_FLC'] = base['mean_spending_2017'] / base['emp_flc_supervisor']
base['SPEND_OP'] = base['mean_spending_2017'] / base['operations']
base['SPEND_AREA'] = base['mean_spending_2017'] / base['land_area_sqmi']
for v in ['SPEND_WORK', 'SPEND_APP', 'SPEND_FLC', 'SPEND_OP', 'SPEND_AREA']:
    base[v + '_z'] = (base[v] - base[v].mean()) / base[v].std()

out = GEN + 'spend_bls_variables_multiyear_2021.csv'
base.to_csv(out, index=False)

# ---- report --------------------------------------------------------------------
print(f"Wrote {out}  (window {YEAR_MIN}-{YEAR_MAX})")
for v in ['SPEND_WORK', 'SPEND_APP', 'SPEND_FLC']:
    miss = sorted(base.loc[base[v].isna(), 'state'])
    print(f"  {v}: {base[v].notna().sum()}/50 states | missing: {miss or 'NONE'}")
print("\nSpending=$0 states:", sorted(base.loc[base['mean_spending_2017'] == 0, 'state']))

old = pd.read_csv(GEN + 'spend_bls_variables_multiyear.csv')[
    ['state', 'SPEND_WORK', 'SPEND_APP']].rename(
    columns={'SPEND_WORK': 'SPEND_WORK_2019', 'SPEND_APP': 'SPEND_APP_2019'})
chk = base.merge(old, on='state')
for v in ['SPEND_WORK', 'SPEND_APP']:
    c = chk.dropna(subset=[v, f'{v}_2019'])
    c = c[c[f'{v}_2019'] > 0]
    r = np.corrcoef(c[v], c[f'{v}_2019'])[0, 1]
    print(f"Sanity {v}: 2021-window vs 2019-window corr={r:.3f}, "
          f"median ratio={(c[v] / c[f'{v}_2019']).median():.3f}")
