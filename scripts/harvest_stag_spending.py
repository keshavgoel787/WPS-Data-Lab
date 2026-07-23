"""
Re-harvest EPA STAG / pesticide grant obligations for ALL 50 states, 2011-2019.

Why: the shipped spending_data_master.csv is incomplete inside the study window.
Rhode Island is absent entirely; Mississippi has only 2008 rows and West Virginia
only a 2022 row -- so all three collapse to missing spending and get dropped
listwise. Joe's point: every state that reports to ECHO should have STAG funding,
so the gap is a harvest artifact, not a true zero. This pulls a clean, complete,
uniformly-sourced 50-state panel from USASpending.gov.

Scope reproduces the exact CFDA set already present in the master (summed, no
filter), so the spending *definition* is unchanged -- only completeness improves:
    66.700  Consolidated Pesticide Enforcement Cooperative Agreements (FIFRA STAG
            Compliance Monitoring) -- the core WPS-relevant program
    66.714  Pesticide Environmental Stewardship Program (PESP) Grants
    66.720  PRIA 5: Farm Worker & Health Care Provider Training
    66.605  Performance Partnership Grants
    66.716  Research, Development, Monitoring, Public Education, Outreach, Training
    66.717  Source Reduction Assistance
    66.509  Science To Achieve Results (STAR) Research

Award types 02-05 (block/formula/project grants + cooperative agreements).
Aggregation basis: transaction obligations by recipient-location state, grouped
by CALENDAR year of action_date (matches the calendar-year ECHO violation DV).

Outputs:
    data/generated/spending_reharvest_state_year.csv   (state, year, spending_nominal)
    data/generated/spending_reharvest_validation.csv   (old vs new state totals)
The raw master is left untouched; the validation file lets us confirm the new
harvest reproduces the old magnitudes on the 47 overlapping states before we
adopt it.
"""

import json
import subprocess
import pandas as pd
import numpy as np

GEN = '/Users/keshavgoel/Research/data/generated/'
RAW = '/Users/keshavgoel/Research/data/raw/'

CFDA = ['66.700', '66.714', '66.720', '66.605', '66.716', '66.717', '66.509']
AWARD_TYPES = ['02', '03', '04', '05']
YEARS = range(2011, 2020)

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
US_STATES_50 = set(STATE_ABBREV_TO_NAME.values())


def api_geo(start, end):
    """spending_by_geography: state -> summed transaction obligations in [start,end]."""
    payload = {
        "scope": "recipient_location", "geo_layer": "state",
        "filters": {
            "time_period": [{"start_date": start, "end_date": end}],
            "award_type_codes": AWARD_TYPES,
            "program_numbers": CFDA,
        },
    }
    out = subprocess.run(
        ['curl', '-sS', '--max-time', '90', '-X', 'POST',
         'https://api.usaspending.gov/api/v2/search/spending_by_geography/',
         '-H', 'Content-Type: application/json', '-d', json.dumps(payload)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)['results']


# ---- per-calendar-year state panel -----------------------------------------
rows = []
for y in YEARS:
    for r in api_geo(f'{y}-01-01', f'{y}-12-31'):
        st = STATE_ABBREV_TO_NAME.get(r['shape_code'])
        if st in US_STATES_50 and r.get('aggregated_amount') is not None:
            rows.append({'state': st, 'year': y,
                         'spending_nominal': float(r['aggregated_amount'])})
panel = pd.DataFrame(rows)
# fill absent state-years with 0 (no obligations that year), keep true panel
full = pd.MultiIndex.from_product([sorted(US_STATES_50), list(YEARS)],
                                  names=['state', 'year']).to_frame(index=False)
panel = full.merge(panel, on=['state', 'year'], how='left').fillna({'spending_nominal': 0.0})
panel.to_csv(GEN + 'spending_reharvest_state_year.csv', index=False)
print(f"Wrote spending_reharvest_state_year.csv: {len(panel)} state-years, "
      f"{panel.state.nunique()} states")

# ---- validation vs old master (47 overlap states) --------------------------
old = pd.read_csv(RAW + 'spending_data_master.csv')
old = old[old['Year'].between(2011, 2019)].copy()
old['state'] = old['State'].map(STATE_ABBREV_TO_NAME)
old_tot = old.groupby('state')['Total Obligation'].sum().rename('old_total')
new_tot = panel.groupby('state')['spending_nominal'].sum().rename('new_total')
comp = pd.concat([old_tot, new_tot], axis=1).reindex(sorted(US_STATES_50)).fillna(0)
comp['abs_diff'] = comp['new_total'] - comp['old_total']
comp['ratio'] = np.where(comp['old_total'] > 0, comp['new_total'] / comp['old_total'], np.nan)
comp.to_csv(GEN + 'spending_reharvest_validation.csv')

overlap = comp[comp['old_total'] > 0]
r = np.corrcoef(overlap['old_total'], overlap['new_total'])[0, 1]
print(f"\nValidation on {len(overlap)} overlap states: corr(old,new)={r:.3f}, "
      f"median new/old ratio={overlap['ratio'].median():.2f}")
print("Newly recovered (old total = 0):")
for st, row in comp[comp['old_total'] == 0].iterrows():
    print(f"  {st:16} new_total=${row['new_total']:,.0f}")
