"""
Analytic panels for the ZINB count models (spec: 2026-08-20-multilevel-zinb-design.md).

Two windows, because the results Joe reacted to changed TWO things at once --
the window (2019 -> 2021) and the outcome source (ECHO establishments view ->
WPS view). Refitting both under one model class is what separates the
model-class effect from the data-source effect.

    end_year=2021 -> WPS view,            2011-2021, 49 states (Wyoming absent)
    end_year=2019 -> establishments view, 2011-2019, 50 states

Unlike the paper-table scripts this keeps the RAW counts -- that is the whole
point of a count model. The log(count+1) columns ride along so the same file
supports the Gaussian round-trip validation.

Covariate construction is IDENTICAL to paper_table_models_{2021,corrected}.py.
Any drift would make the count-vs-log comparison meaningless.

Run: python3 scripts/build_count_model_panel.py
Output: data/generated/count_model_panel_{2019,2021}.csv
"""

import numpy as np
import pandas as pd

RAW = '/Users/keshavgoel/Research/data/raw/'
GEN = '/Users/keshavgoel/Research/data/generated/'

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
STATE_NAME_MAPPING = {'Massachusetts ': 'Massachusetts', 'Oregon ': 'Oregon'}

Z_COVARS = ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
            'dol_demand_met_pct_z', 'pct_flc_z', 'h2a_per_farmworker_z']


def _dv_frame_2021():
    """WPS view of the ECHO State Pesticide Dashboard, 2011-2021."""
    long = pd.read_csv(GEN + 'wps_dv_panel_2011_2021.csv')
    long = long[long['state'].isin(US_STATES_50)].copy()
    return long.rename(columns={'wps_violations': 'violations',
                                'wps_inspections': 'inspections'})


def _dv_frame_2019():
    """Establishments view of the same dashboard, which EPA publishes only to 2019."""
    echo = pd.read_csv(RAW + 'establishments_data.csv', index_col=0)
    echo.index = echo.index.str.strip().map(lambda x: STATE_NAME_MAPPING.get(x, x))
    rows = []
    for year in range(2011, 2020):
        rows.append(pd.DataFrame({
            'state': echo.index, 'year': year,
            'violations': pd.to_numeric(echo[f'violations-{year}'], errors='coerce').values,
            'inspections': (pd.to_numeric(echo[f'inspections-epa-{year}'], errors='coerce').fillna(0).values
                            + pd.to_numeric(echo[f'inspections-state-{year}'], errors='coerce').fillna(0).values),
        }))
    long = pd.concat(rows, ignore_index=True)
    return long[long['state'].isin(US_STATES_50)].copy()


def _level2(end_year, spend_file):
    """State-level (Level-2) covariates, z-scored. Mirrors the paper-table scripts."""
    level2 = pd.DataFrame({'state': US_STATES_50})

    spend = pd.read_csv(GEN + spend_file)
    level2 = level2.merge(spend[['state', 'SPEND_APP_z', 'SPEND_WORK_z']], on='state', how='left')

    li17 = pd.read_csv(RAW + 'labor_intensity_index_2017.csv')
    li17['state'] = li17['state_name'].str.title()
    level2 = level2.merge(li17[['state', 'Labor_Intensity_Index']].rename(
        columns={'Labor_Intensity_Index': 'lii_2017'}), on='state', how='left')

    # DOL demand-met %: study-window mean, excluding 2013 (DOL never published it).
    dol1 = pd.read_csv(RAW + 'dol_var1_workers_by_state_annual.csv')
    dol_study = dol1[dol1['year'].between(2011, end_year) & (dol1['year'] != 2013)]
    dm = dol_study.groupby('state')['demand_met_pct'].apply(
        lambda x: x[x != np.inf].mean()).reset_index()
    dm['state'] = dm['state'].map(STATE_ABBREV_TO_NAME)
    level2 = level2.merge(dm.rename(columns={'demand_met_pct': 'dol_demand_met_pct'}),
                          on='state', how='left')

    # DOL % Farm Labor Contractor: dol_var2 starts in 2020, so 2020 is the only
    # available value -- a backward proxy for the 2019 window, in-window for 2021.
    dol2 = pd.read_csv(RAW + 'dol_var2_employer_type_annual.csv')
    d2020 = dol2[dol2['year'] == 2020].copy()
    d2020['state'] = d2020['state'].map(STATE_ABBREV_TO_NAME)
    d2020 = d2020.rename(columns={'pct_Farm Labor Contractor': 'pct_flc'})
    level2 = level2.merge(d2020[['state', 'pct_flc']], on='state', how='left')

    for col in ['lii_2017', 'dol_demand_met_pct', 'pct_flc']:
        level2[col + '_z'] = (level2[col] - level2[col].mean()) / level2[col].std()

    return level2[['state', 'SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
                   'dol_demand_met_pct_z', 'pct_flc_z']]


def build_panel(end_year):
    """Assemble the analytic panel for one window. end_year is 2019 or 2021."""
    if end_year == 2021:
        long = _dv_frame_2021()
        spend_file, h2a_file = 'spend_bls_variables_multiyear_2021.csv', 'h2a_ratio_panel_2011_2021.csv'
    elif end_year == 2019:
        long = _dv_frame_2019()
        spend_file, h2a_file = 'spend_bls_variables_multiyear.csv', 'h2a_ratio_panel.csv'
    else:
        raise ValueError(f"end_year must be 2019 or 2021, got {end_year!r}")

    long['time'] = long['year'] - 2017
    long['time2'] = long['time'] ** 2
    long['time3'] = long['time'] ** 3
    long['log_violations'] = np.log1p(long['violations'])
    long['log_inspections'] = np.log1p(long['inspections'])

    cols = ['state', 'year', 'time', 'time2', 'time3',
            'violations', 'inspections', 'log_violations', 'log_inspections']
    if end_year == 2021:
        long['covid'] = (long['year'] >= 2020).astype(float)
        cols.append('covid')

    df = long[cols].merge(_level2(end_year, spend_file), on='state', how='left')

    h2a = pd.read_csv(GEN + h2a_file)
    df = df.merge(h2a[['state', 'year', 'h2a_per_farmworker_z']],
                  on=['state', 'year'], how='left')

    return df.sort_values(['state', 'year']).reset_index(drop=True)


if __name__ == '__main__':
    for end_year in (2019, 2021):
        panel = build_panel(end_year)
        out = GEN + f'count_model_panel_{end_year}.csv'
        panel.to_csv(out, index=False)
        n_z = (panel['violations'] == 0).sum()
        print(f"{out}\n  {len(panel)} rows, {panel['state'].nunique()} states, "
              f"{panel['year'].min()}-{panel['year'].max()}, "
              f"{n_z} zero-violation state-years ({n_z / len(panel):.1%})")
        missing = [c for c in Z_COVARS if panel[c].isna().any()]
        for c in missing:
            drops = sorted(panel.loc[panel[c].isna(), 'state'].unique())
            print(f"  {c}: missing for {len(drops)} state(s) -> {', '.join(drops)}")
