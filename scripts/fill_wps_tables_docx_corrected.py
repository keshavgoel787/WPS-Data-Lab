"""
Populate "WPS Table Sheels.docx" with the CORRECTED (2026-07) model results.

Reads data/generated/paper_table_params_corrected.json (paper_table_models_corrected.py)
and fills each coefficient's b / SE by model→column position (M1=cols 1-2, M2=3-4,
M3=5-6). Differences from fill_wps_tables_docx.py:
  - both DVs are log(count+1); the violations "Inspections" predictor row maps to the
    log_inspections term and is relabelled accordingly;
  - coefficients print to 3 decimals (log-scale effects are small);
  - the three ×time interaction rows carry no term in the corrected spec and are dropped;
  - the closing note documents the log transform, the multi-year BLS denominators,
    the $0-coded states, the year-matched H-2A ratio, and the N=47 analytic sample.

Output: docs/WPS_Table_Sheels_filled_corrected.docx  (source in ~/Downloads untouched).
"""
import json
from docx import Document

SRC = '/Users/keshavgoel/Downloads/WPS Table Sheels.docx'
OUT = '/Users/keshavgoel/Research/docs/WPS_Table_Sheels_filled_corrected.docx'
JSON = '/Users/keshavgoel/Research/data/generated/paper_table_params_corrected.json'

with open(JSON) as f:
    P = json.load(f)

MODELS = ['M1', 'M2', 'M3']
PLACEHOLDERS = {'X.XX', 'XX.XX', 'XX.X%'}
COEF_COLS = {'M1': (1, 2), 'M2': (3, 4), 'M3': (5, 6)}

LABEL_TO_TERM = {
    'Inspections': 'log_inspections',            # DV is log(x+1); predictor is too
    'Time': 'time', 'Time2': 'time2', 'Time3': 'time3',
    'Spending/applicator': 'SPEND_APP_z',
    'Spending/farmworker': 'SPEND_WORK_z',
    'Labor Intensity': 'lii_2017_z',
    'H-2A farmworkers:BLS farmworkers': 'h2a_per_farmworker_z',
    'H-2A Authorized/H-2A Requested': 'dol_demand_met_pct_z',
    '%H-2A Authorized to FLC': 'pct_flc_z',
    # ×time interaction rows carry no term in the corrected spec -> dropped
    'Spending/applicator*time': 'SPEND_APP_z:time',
    'Spending/farmworker*time': 'SPEND_WORK_z:time',
    '%H-2A Authorized to FLC*time': 'pct_flc_z:time',
}


def term_present(tab, term):
    return any(tab[m].get(term) is not None for m in MODELS)


def fill_coef_row(row, tab, term):
    for m in MODELS:
        cell = tab[m].get(term)
        if cell is None:
            continue
        bcol, secol = COEF_COLS[m]
        row.cells[bcol].text = f"{cell['b']:.3f}{cell['stars']}"
        row.cells[secol].text = f"{cell['se']:.3f}"


def is_placeholder(text):
    return text.strip() in PLACEHOLDERS


def fill_row(row, values):
    targets = [c for c in row.cells if is_placeholder(c.text)]
    if len(targets) != len(values):
        raise SystemExit(f"  MISMATCH in row '{row.cells[0].text.strip()}': "
                         f"{len(targets)} placeholders but {len(values)} values")
    for cell, val in zip(targets, values):
        cell.text = val


def remove_row(row):
    row._element.getparent().remove(row._element)


def relabel(row, text):
    """Rename the row's label cell (keep formatting of the first run)."""
    cell = row.cells[0]
    if cell.paragraphs and cell.paragraphs[0].runs:
        cell.paragraphs[0].runs[0].text = text
        for extra in cell.paragraphs[0].runs[1:]:
            extra.text = ''
    else:
        cell.text = text


def fill_table(table, tab):
    for row in list(table.rows):
        label = row.cells[0].text.strip()
        if not label:
            continue
        # variance rows use the random-INTERCEPT basis (delta_pct_ri / sigma2_u0_ri):
        # in the random-slope spec sigma^2_u0 is centered at 2017 and its reduction is
        # not bounded to [0,1] (Violations M2 read -12.5%); the random-intercept value
        # is the conventional between-state variance-explained. Template placeholders
        # exist only for M2/M3 (the M1 variance column is left blank).
        if label.startswith('Δ'):
            fill_row(row, [f"{tab[m]['delta_pct_ri']:.1f}%" for m in ('M2', 'M3')])
        elif 'State-to-State' in label:
            fill_row(row, [f"{tab[m]['sigma2_u0_ri']:.3f}" for m in ('M2', 'M3')])
        elif label in LABEL_TO_TERM:
            term = LABEL_TO_TERM[label]
            if term_present(tab, term):
                if label == 'Inspections':
                    relabel(row, 'log(Inspections+1)')
                fill_coef_row(row, tab, term)
            else:
                remove_row(row)


doc = Document(SRC)

# Study period: data run 2011-2019, not 2021.
for p in doc.paragraphs:
    if '2011 to 2021' in p.text:
        for run in p.runs:
            run.text = run.text.replace('2011 to 2021', '2011 to 2019')

fill_table(doc.tables[0], P['inspections'])       # DV = log(inspections+1)
fill_table(doc.tables[1], P['violations'])        # DV = log(violations+1)

doc.add_paragraph()
note = doc.add_paragraph()
note.add_run('Note on model specification (2026-07 revision). ').bold = True
note.add_run(
    'Both outcomes are modelled as log(count + 1); the violations model includes '
    'log(inspections + 1) as a contemporaneous predictor on the same scale. '
    'Spending/applicator and Spending/farmworker use mean 2011-2019 STAG obligations '
    'over mean 2011-2019 BLS employment in the respective occupation; six states '
    '(Hawaii, Mississippi, Rhode Island, Tennessee, Utah, West Virginia) received no '
    'obligations under the pesticide-enforcement program (CFDA 66.700) in the study '
    'window and are coded $0. The H-2A-to-farmworker ratio is matched by year '
    '(annual certified H-2A workers over annual BLS farmworkers). The substantive '
    'models are estimated on 47 states: Alaska, Rhode Island, and Vermont are omitted '
    'from Spending/applicator columns because BLS never publishes pesticide-applicator '
    '(SOC 37-3012) employment for them. Standard errors in parentheses; '
    '+ p<.10, * p<.05, ** p<.01, *** p<.001. State-to-State σ² and Δ State-to-State σ² '
    'are read from random-intercept-only refits (baseline and model, same sample): Δ is '
    'the percent of between-state intercept variance explained. Coefficients come from the '
    'random-slope specification; the random-intercept basis is used for the variance-'
    'explained statistic because in the random-slope model σ² is centered at 2017 and its '
    'reduction is not bounded to [0,1].')

doc.save(OUT)
print(f"Saved filled tables to {OUT}")
