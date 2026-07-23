"""
Rebuild spend_bls_variables.csv with MULTI-YEAR BLS employment denominators and
$0-coded spending for the states that genuinely receive no pesticide-STAG money.

Two corrections (2026-07, post-meeting):
  1. Denominators. The old file divided mean 2011-2019 STAG spending by the BLS
     *2011* employment snapshot, in which small occupational cells are suppressed
     (37-3012 blank for CO/NH/NM/UT/VT, 45-2092 blank for MN/NV, AK absent). We now
     use the mean of every NON-suppressed year 2011-2019 from the full OEWS harvest
     (data/raw/bls_oews_panel_2011_2019.csv). This matches the mean-spending
     numerator's window and recovers CO, NH, NM, UT, MN, NV. Only AK, RI, VT remain
     without an applicator (37-3012) figure -- BLS never publishes one, so SPEND_APP
     stays missing there (accepted N=47 for SPEND_APP models; PI direction).
  2. Missing-spending states. RI/MS/WV had no in-window rows in the master and were
     dropped. USASpending confirms they receive $0 in the pesticide-STAG program
     (66.700) 2011-2019 -- same as HI/TN/UT, which the master already codes $0.
     So we code MS/RI/WV spending = $0 too, keeping them in the models.

Everything else (spending numerator for the 44 funded states, land_area, operations)
is carried over unchanged from the existing spend_bls_variables.csv. SPEND_* and
their z-scores are recomputed. Output: spend_bls_variables_multiyear.csv (the
original file is left untouched).
"""

import pandas as pd
import numpy as np

RAW = '/Users/keshavgoel/Research/data/raw/'
GEN = '/Users/keshavgoel/Research/data/generated/'

ZERO_SPEND_STATES = ['Mississippi', 'Rhode Island', 'West Virginia']  # confirmed $0 in 66.700

# ---- start from the existing variable file (keeps numerator, land, operations) --
base = pd.read_csv(GEN + 'spend_bls_variables.csv')

# 1. code the 3 confirmed-$0 states (join HI/TN/UT, already 0)
base.loc[base['state'].isin(ZERO_SPEND_STATES), 'mean_spending_2017'] = 0.0

# 2. multi-year mean employment per occupation (mean of non-suppressed years) -----
bls = pd.read_csv(RAW + 'bls_oews_panel_2011_2019.csv')
emp = (bls.dropna(subset=['tot_emp'])
          .groupby(['state', 'occ_code'])['tot_emp'].mean().unstack('occ_code'))
emp = emp.rename(columns={'37-3012': 'emp_pesticide_app',
                          '45-1011': 'emp_flc_supervisor',
                          '45-2092': 'emp_farmworker'})
base = base.drop(columns=['emp_farmworker', 'emp_pesticide_app', 'emp_flc_supervisor'])
base = base.merge(emp[['emp_farmworker', 'emp_pesticide_app', 'emp_flc_supervisor']],
                  left_on='state', right_index=True, how='left')

# 3. recompute SPEND_* and z-scores ----------------------------------------------
base['SPEND_WORK'] = base['mean_spending_2017'] / base['emp_farmworker']
base['SPEND_APP'] = base['mean_spending_2017'] / base['emp_pesticide_app']
base['SPEND_FLC'] = base['mean_spending_2017'] / base['emp_flc_supervisor']
base['SPEND_OP'] = base['mean_spending_2017'] / base['operations']
base['SPEND_AREA'] = base['mean_spending_2017'] / base['land_area_sqmi']
for v in ['SPEND_WORK', 'SPEND_APP', 'SPEND_FLC', 'SPEND_OP', 'SPEND_AREA']:
    base[v + '_z'] = (base[v] - base[v].mean()) / base[v].std()

out = GEN + 'spend_bls_variables_multiyear.csv'
base.to_csv(out, index=False)

# ---- coverage report -----------------------------------------------------------
print(f"Wrote {out}")
for v in ['SPEND_WORK', 'SPEND_APP', 'SPEND_FLC']:
    miss = sorted(base.loc[base[v].isna(), 'state'])
    print(f"  {v}: {base[v].notna().sum()}/50 states | missing: {miss or 'NONE'}")
print("\nSpending=$0 states:", sorted(base.loc[base['mean_spending_2017'] == 0, 'state']))
# sanity: how much did multi-year denom move SPEND_WORK vs the old 2011-based file?
old = pd.read_csv(GEN + 'spend_bls_variables.csv')[['state', 'SPEND_WORK']].rename(
    columns={'SPEND_WORK': 'SPEND_WORK_old'})
chk = base.merge(old, on='state').dropna(subset=['SPEND_WORK', 'SPEND_WORK_old'])
chk = chk[chk['SPEND_WORK_old'] > 0]
r = np.corrcoef(chk['SPEND_WORK'], chk['SPEND_WORK_old'])[0, 1]
print(f"\nSanity: SPEND_WORK new-vs-old (funded states) corr={r:.3f}, "
      f"median ratio={ (chk['SPEND_WORK']/chk['SPEND_WORK_old']).median():.2f}")
