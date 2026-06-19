"""
Diagnostic: crosstab of inspections vs. violations by state-year.

Question: do any state-years record violations without any inspection?
A violation logged with zero inspections would indicate a data-integrity
problem, since under WPS a violation can only be cited via an inspection.

DV definitions mirror the modeling scripts:
  inspections = inspections-epa-YYYY + inspections-state-YYYY
  violations  = violations-YYYY
NaN handling matches inspections_hierarchical_model.py: a single-source NaN
is treated as 0; only an all-NaN sum stays NaN.

Output: inspection_violation_crosstab_diagnostic.md
"""

import numpy as np
import pandas as pd

US_STATES_50 = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
    'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
    'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
    'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
    'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
    'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
    'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
]

STATE_NAME_MAPPING = {
    'Massachusetts ': 'Massachusetts',
    'Oregon ': 'Oregon',
}

# ------------------------------------------------------------
# Load and reshape ECHO data to long format (state, year)
# ------------------------------------------------------------
echo_df = pd.read_csv('/Users/keshavgoel/Research/establishments-data (2).csv', index_col=0)
echo_df.index = echo_df.index.str.strip()
echo_df.index = echo_df.index.map(lambda x: STATE_NAME_MAPPING.get(x, x))

rows = []
for year in range(2011, 2020):
    col_epa   = f'inspections-epa-{year}'
    col_state = f'inspections-state-{year}'
    col_viol  = f'violations-{year}'

    epa   = pd.to_numeric(echo_df.get(col_epa,   pd.Series(np.nan, index=echo_df.index)), errors='coerce')
    state = pd.to_numeric(echo_df.get(col_state, pd.Series(np.nan, index=echo_df.index)), errors='coerce')
    viol  = pd.to_numeric(echo_df.get(col_viol,  pd.Series(np.nan, index=echo_df.index)), errors='coerce')

    insp_total = epa.add(state, fill_value=0)  # single-source NaN -> 0; both NaN -> NaN

    rows.append(pd.DataFrame({
        'state': echo_df.index,
        'year': year,
        'insp_epa': epa.values,
        'insp_state': state.values,
        'inspections': insp_total.values,
        'violations': viol.values,
    }))

long = pd.concat(rows, ignore_index=True)
long = long[long['state'].isin(US_STATES_50)].copy()

# ------------------------------------------------------------
# Build categorical buckets for the crosstab
# ------------------------------------------------------------
def insp_bucket(x):
    if pd.isna(x):
        return 'inspections=NaN'
    return 'inspections=0' if x == 0 else 'inspections>0'

def viol_bucket(x):
    if pd.isna(x):
        return 'violations=NaN'
    return 'violations=0' if x == 0 else 'violations>0'

long['insp_cat'] = long['inspections'].apply(insp_bucket)
long['viol_cat'] = long['violations'].apply(viol_bucket)

crosstab = pd.crosstab(long['insp_cat'], long['viol_cat'], margins=True, margins_name='Total')

# ------------------------------------------------------------
# The diagnostic flag: violations > 0 but inspections == 0 (or NaN)
# ------------------------------------------------------------
viol_pos = long['violations'].fillna(0) > 0
insp_zero = (long['inspections'].fillna(0) == 0)
flagged = long[viol_pos & insp_zero].copy()
flagged = flagged.sort_values(['year', 'state'])

# Per-year summary
per_year = (
    long.assign(
        viol_pos=viol_pos,
        insp_zero_total=(long['inspections'].fillna(0) == 0),
        flag=(viol_pos & insp_zero),
    )
    .groupby('year')
    .agg(
        n_stateyears=('state', 'count'),
        n_viol_pos=('viol_pos', 'sum'),
        n_insp_zero=('insp_zero_total', 'sum'),
        n_flagged=('flag', 'sum'),
    )
    .reset_index()
)

# ------------------------------------------------------------
# Write markdown report
# ------------------------------------------------------------
def df_to_md(df, index=True):
    return df.to_markdown(index=index)

n_total = len(long)
n_flagged = len(flagged)

lines = []
lines.append("# Diagnostic: Inspections × Violations Crosstab by State-Year\n")
lines.append("**Question:** Do any state-years record violations without an inspection? "
             "Under WPS a violation is cited through an inspection, so a "
             "`violations > 0` with `inspections == 0` row would signal a data-integrity issue.\n")
lines.append("**Source:** `establishments-data (2).csv`, 50 U.S. states, 2011–2019.\n")
lines.append("**DV definitions:** `inspections = inspections-epa + inspections-state` "
             "(single-source NaN treated as 0); `violations = violations-YYYY`.\n")
lines.append(f"**Total state-year observations:** {n_total} (50 states × 9 years = 450; "
             f"`violations` columns exist for all years).\n")

lines.append("\n## 1. Crosstab (counts of state-years)\n")
lines.append(df_to_md(crosstab))

lines.append("\n\n## 2. Headline result\n")
if n_flagged == 0:
    lines.append("✅ **No state-year has `violations > 0` while `inspections == 0`.** "
                 "Every recorded violation co-occurs with at least one inspection. "
                 "The data passes this integrity check.\n")
else:
    lines.append(f"⚠️ **{n_flagged} state-year(s) record `violations > 0` with `inspections == 0`.** "
                 "These are listed below and warrant inspection of the raw source.\n")

lines.append("\n## 3. Per-year summary\n")
lines.append(df_to_md(per_year, index=False))
lines.append("\n\n- `n_viol_pos` = state-years with at least one violation\n"
             "- `n_insp_zero` = state-years with zero total inspections\n"
             "- `n_flagged` = state-years with violations but zero inspections\n")

lines.append("\n## 4. Flagged state-years (violations > 0 AND inspections == 0)\n")
if n_flagged == 0:
    lines.append("_None._\n")
else:
    show = flagged[['state', 'year', 'insp_epa', 'insp_state', 'inspections', 'violations']]
    lines.append(df_to_md(show, index=False))

# Also report all-NaN inspection cells with any violation activity, for completeness
nan_insp = long[long['inspections'].isna()]
lines.append("\n\n## 5. Note on missing inspection values\n")
lines.append(f"State-years where **both** inspection sources are NaN (sum = NaN): {len(nan_insp)}.\n")
if len(nan_insp) > 0:
    nan_with_viol = nan_insp[nan_insp['violations'].fillna(0) > 0]
    lines.append(f"Of those, state-years that also report violations > 0: {len(nan_with_viol)}.\n")
    if len(nan_with_viol) > 0:
        lines.append(df_to_md(
            nan_with_viol[['state', 'year', 'insp_epa', 'insp_state', 'violations']], index=False))

report = "\n".join(lines) + "\n"
with open('/Users/keshavgoel/Research/inspection_violation_crosstab_diagnostic.md', 'w') as f:
    f.write(report)

# Console echo
print(crosstab)
print(f"\nFlagged (violations>0 & inspections==0): {n_flagged}")
print(f"Both-source NaN inspection cells: {len(nan_insp)}")
print("\nWrote inspection_violation_crosstab_diagnostic.md")
