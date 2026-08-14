"""
Build the WPS-specific outcome panel, 2011-2021 (extends the study period to 2021).

WHY THIS EXISTS
The models through 2026-07 took both DVs from data/raw/establishments_data.csv --
the ECHO State Pesticide Dashboard's "establishments" view, which EPA publishes only
through 2019. That is the sole reason the study period stopped at 2019; every
covariate (STAG spending, DOL H-2A, BLS OEWS) already ran past it. Verified against
the live dashboard 2026-08-13: the establishments payload still ends at
`penalties-epa-2019`, so there is no 2020/2021 to harvest there.

The dashboard's OTHER view -- the WPS view, in data/raw/wps_data.csv -- does run
2011-2021, and is WPS-specific (broken out by the ten WPS provisions) rather than
FIFRA-wide. This script turns it into the same state-year DV frame the old ECHO
extract provided, so the whole pipeline can be re-fit on 2011-2021.

DV CONSTRUCTION
    wps_inspections[s,y] = insp-{state,tribe,epa}-y
    wps_violations[s,y]  = sum over the 10 provision categories of
                           {retaliation, information, emergency, decon, mix, ppe,
                            entry, notice, central, safety}-{state,tribe,epa}-y
Summing the three lead agencies reproduces the dashboard's National bars exactly
(checked: 2011 insp 3,760 / viol 1,431; 2017 viol 2,296; 2021 viol 1,492).

Row structure (verified, no double counting): the 50-state rows carry only
`-state-` and occasional `-epa-` counts; their `-tribe-` columns are empty. Tribal
lands are SEPARATE rows (Navajo Nation etc.) carrying only `-tribe-` counts, and
are excluded here along with territories -- tribal pesticide programs are
separately administered and have no state-level covariates. Including the
always-empty `-tribe-` columns of state rows is therefore a no-op, kept only so the
definition matches the dashboard's.

WYOMING has no row in the WPS view at all (it is not a WPS grantee in this table),
so the panel is 49 states, not 50. For reference the current ECHO-based paper
models run on 47 states, so coverage is slightly better, not worse.

NOT COMPARABLE TO THE OLD DV. This is a different measure, not a longer one:
counts run 2-3x the ECHO series (2,296 vs 967 violations across the 50 states in
2017) and state-level correlation between the two is ~0.4-0.6 in 2011-2016 and
~0 from 2017 on. Coefficients fit on this panel cannot be compared line-by-line
with the 2011-2019 ECHO tables.

Output: data/generated/wps_dv_panel_2011_2021.csv
    columns: state, year, time, time2, time3, wps_inspections, wps_violations,
             log_inspections, log_violations
"""

import numpy as np
import pandas as pd

RAW = '/Users/keshavgoel/Research/data/raw/'
GEN = '/Users/keshavgoel/Research/data/generated/'

YEARS = list(range(2011, 2022))
AGENTS = ['state', 'tribe', 'epa']
VIOL_CATS = ['viol_retaliation', 'viol_information', 'viol_emergency', 'viol_decon',
             'viol_mix', 'viol_ppe', 'viol_entry', 'viol_notice', 'viol_central',
             'viol_safety']

US_STATES_50 = sorted([
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
    'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
    'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
    'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
    'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
    'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
    'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'])

wps = pd.read_csv(RAW + 'wps_data.csv')
wps['state'] = wps['state'].str.strip()
wps = wps[wps['state'].isin(US_STATES_50)].set_index('state')

missing_states = sorted(set(US_STATES_50) - set(wps.index))


def total(stems, year):
    """Sum stems x lead agencies for one year. All-NaN across the agency columns
    stays NaN (state genuinely absent that year); a partial report sums what exists."""
    cols = [f'{s}-{a}-{year}' for s in stems for a in AGENTS]
    block = wps[cols].apply(pd.to_numeric, errors='coerce')
    return block.sum(axis=1).where(block.notna().any(axis=1))


rows = []
for year in YEARS:
    rows.append(pd.DataFrame({
        'state': wps.index,
        'year': year,
        'wps_inspections': total(['insp'], year).values,
        'wps_violations': total(VIOL_CATS, year).values,
    }))
panel = pd.concat(rows, ignore_index=True).sort_values(['state', 'year'])

panel['time'] = panel['year'] - 2017
panel['time2'] = panel['time'] ** 2
panel['time3'] = panel['time'] ** 3
panel['log_inspections'] = np.log1p(panel['wps_inspections'])
panel['log_violations'] = np.log1p(panel['wps_violations'])

panel = panel[['state', 'year', 'time', 'time2', 'time3',
               'wps_inspections', 'wps_violations',
               'log_inspections', 'log_violations']]
panel.to_csv(GEN + 'wps_dv_panel_2011_2021.csv', index=False)

# ---- report ----------------------------------------------------------------
print(f"Wrote {GEN}wps_dv_panel_2011_2021.csv")
print(f"  {len(panel)} state-years | {panel['state'].nunique()} states "
      f"| {panel['year'].min()}-{panel['year'].max()}")
print(f"  states absent from the WPS view: {missing_states or 'NONE'}")

print("\nNational totals by year (should match the dashboard's bars):")
print(f"  {'year':>6}{'inspections':>14}{'violations':>13}{'states':>9}")
for y in YEARS:
    s = panel[panel['year'] == y]
    print(f"  {y:>6}{s['wps_inspections'].sum():>14.0f}{s['wps_violations'].sum():>13.0f}"
          f"{s['wps_violations'].notna().sum():>9}")

print("\nListwise completeness (state-years with a non-null DV):")
for c in ['wps_inspections', 'wps_violations']:
    print(f"  {c}: {panel[c].notna().sum()}/{len(panel)}")
