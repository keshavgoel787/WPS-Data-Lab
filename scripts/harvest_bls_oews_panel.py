"""
Harvest BLS OEWS state-level employment for 2011-2019 (all years, not just 2011).

Fixes the data gap that forced the 50->39 state drop: the old bls_oews_panel.csv
held ONLY the 2011 snapshot, in which BLS suppresses small occupational cells
(37-3012 blank for CO/NH/NM/UT/VT, 45-2092 blank for MN/NV, AK absent entirely).
Pulling all nine years lets a cell suppressed in one year be recovered from another.

Source: BLS OEWS "State" special-request files, one zip per year:
    https://www.bls.gov/oes/special-requests/oesm{YY}st.zip
  2011-2013 ship an .xls (UPPERCASE columns, separate STATE name column);
  2014-2019 ship an .xlsx nested in a folder (lowercase columns, state in area_title).
Downloads are cached under data/raw/bls_oews_cache/ (skipped if already present).

Occupations kept (same three the project uses):
    37-3012  Pesticide Handlers, Sprayers, and Applicators   (SPEND_APP denominator)
    45-1011  First-Line Supervisors of Farming/Forestry       (SPEND_FLC denominator)
    45-2092  Farmworkers and Laborers, Crop/Nursery/Greenhouse(SPEND_WORK denominator,
                                                               H-2A ratio denominator)

Output: data/raw/bls_oews_panel_2011_2019.csv  (long: one row per state-year-occ)
    columns: year, state, occ_code, occ_title, tot_emp, h_mean, a_mean
Suppressed / non-released cells (BLS markers '*', '**', '#') become NaN tot_emp.
The original single-year bls_oews_panel.csv is left untouched.
"""

import os
import subprocess
import zipfile
import pandas as pd
import numpy as np

RAW = '/Users/keshavgoel/Research/data/raw/'
CACHE = os.path.join(RAW, 'bls_oews_cache')
os.makedirs(CACHE, exist_ok=True)

US_STATES_50 = {
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
    'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
    'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
    'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
    'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
    'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
    'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'}

TARGET_OCC = {'37-3012', '45-1011', '45-2092'}
YEARS = range(2011, 2020)


def download(year):
    yy = f'{year % 100:02d}'
    dest = os.path.join(CACHE, f'oesm{yy}st.zip')
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        return dest
    url = f'https://www.bls.gov/oes/special-requests/oesm{yy}st.zip'
    # urllib hits an SSL cert-chain error behind the proxy here; curl works.
    subprocess.run(['curl', '-sS', '--max-time', '120',
                    '-A', 'research goel.ke@northeastern.edu',
                    '-o', dest, url], check=True)
    return dest


def read_year(year):
    zf = zipfile.ZipFile(download(year))
    member = next(m for m in zf.namelist()
                  if 'state_M' in m and m.lower().endswith(('.xls', '.xlsx')))
    with zf.open(member) as f:
        df = pd.read_excel(f)
    df.columns = [c.strip().lower() for c in df.columns]

    state_col = 'state' if 'state' in df.columns else 'area_title'
    occ = df['occ_code'].astype(str).str.strip()
    keep = df[occ.isin(TARGET_OCC)].copy()
    keep['occ_code'] = occ[occ.isin(TARGET_OCC)]

    def num(s):
        return pd.to_numeric(
            s.astype(str).str.replace(',', '', regex=False)
             .replace({'**': np.nan, '*': np.nan, '#': np.nan, '': np.nan}),
            errors='coerce')

    out = pd.DataFrame({
        'year': year,
        'state': keep[state_col].astype(str).str.strip(),
        'occ_code': keep['occ_code'].values,
        'occ_title': keep['occ_title'].astype(str).str.strip().values,
        'tot_emp': num(keep['tot_emp']).values,
        'h_mean': num(keep['h_mean']).values,
        'a_mean': num(keep['a_mean']).values,
    })
    return out[out['state'].isin(US_STATES_50)]


frames = [read_year(y) for y in YEARS]
panel = pd.concat(frames, ignore_index=True).sort_values(
    ['occ_code', 'state', 'year']).reset_index(drop=True)

out_path = os.path.join(RAW, 'bls_oews_panel_2011_2019.csv')
panel.to_csv(out_path, index=False)

# ---- coverage report -------------------------------------------------------
print(f"Wrote {out_path}: {len(panel)} rows, years {panel.year.min()}-{panel.year.max()}")
for occ in sorted(TARGET_OCC):
    sub = panel[panel.occ_code == occ]
    have = sub.dropna(subset=['tot_emp']).groupby('state').size()
    states_any = have.index.nunique()
    never = sorted(US_STATES_50 - set(have.index))
    print(f"\nOCC {occ}: {states_any}/50 states have >=1 non-suppressed year")
    if never:
        print(f"  NEVER reported (all 9 yrs suppressed/absent): {never}")
