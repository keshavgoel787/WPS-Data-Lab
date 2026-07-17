"""
Populate "WPS Table Sheels.docx" with the fitted coefficients.

Reads data/generated/paper_table_params.json (written by paper_table_models.py) and
replaces every X.XX / XX.XX / XX.X% placeholder in the two tables with the real
value, in reading order (M1_b, M1_se, M2_b, M2_se, M3_b, M3_se — only the columns
where a term is present, matching the template's placeholder layout).

Output: docs/WPS_Table_Sheels_filled.docx  (the source in ~/Downloads is untouched).
"""
import json
from docx import Document

SRC = '/Users/keshavgoel/Downloads/WPS Table Sheels.docx'
OUT = '/Users/keshavgoel/Research/docs/WPS_Table_Sheels_filled.docx'
JSON = '/Users/keshavgoel/Research/data/generated/paper_table_params.json'

with open(JSON) as f:
    P = json.load(f)

MODELS = ['M1', 'M2', 'M3']
PLACEHOLDERS = {'X.XX', 'XX.XX', 'XX.X%'}

# label (stripped) -> JSON term key
LABEL_TO_TERM = {
    'Inspections': 'inspections',
    'Time': 'time', 'Time2': 'time2', 'Time3': 'time3',
    'Spending/applicator': 'SPEND_APP_z',
    'Spending/applicator*time': 'SPEND_APP_z:time',
    'Spending/farmworker': 'SPEND_WORK_z',
    'Spending/farmworker*time': 'SPEND_WORK_z:time',
    'Labor Intensity': 'lii_2017_z',
    'H-2A farmworkers:BLS farmworkers': 'h2a_per_farmworker_z',
    'H-2A Authorized/H-2A Requested': 'dol_demand_met_pct_z',
    '%H-2A Authorized to FLC': 'pct_flc_z',
    '%H-2A Authorized to FLC*time': 'pct_flc_z:time',
}


def coef_values(tab, term):
    """Ordered fill list for a coefficient row: b then SE for each model present."""
    vals = []
    for m in MODELS:
        cell = tab[m].get(term)
        if cell is not None:
            vals.append(f"{cell['b']:.2f}{cell['stars']}")
            vals.append(f"{cell['se']:.2f}")
    return vals


def is_placeholder(text):
    t = text.strip()
    return t in PLACEHOLDERS or t == 'X.XX'


def fill_row(row, values):
    """Fill the row's placeholder cells left-to-right with `values`."""
    targets = [c for c in row.cells if is_placeholder(c.text)]
    if len(targets) != len(values):
        raise SystemExit(
            f"  MISMATCH in row '{row.cells[0].text.strip()}': "
            f"{len(targets)} placeholder cells but {len(values)} values")
    for cell, val in zip(targets, values):
        cell.text = val


def remove_row(row):
    """Delete a table row (used for terms no longer in the model)."""
    row._element.getparent().remove(row._element)


def fill_table(table, tab):
    for row in list(table.rows):
        label = row.cells[0].text.strip()
        if not label:
            continue
        if label.startswith('Δ'):                       # Δ State-to-State σ²
            fill_row(row, [f"{tab[m]['delta_pct']:.1f}%"
                           for m in MODELS if 'delta_pct' in tab[m]])
        elif 'State-to-State' in label:                       # State-to-State σ²
            fill_row(row, [f"{tab[m]['sigma2']:.2f}"
                           for m in MODELS if 'sigma2' in tab[m]])
        elif label in LABEL_TO_TERM:
            values = coef_values(tab, LABEL_TO_TERM[label])
            if not values:      # term dropped from model (e.g. spending×time) → drop row
                remove_row(row)
            else:
                fill_row(row, values)


doc = Document(SRC)

# Correct the study period in the titles (data runs 2011–2019, not 2021).
for p in doc.paragraphs:
    if '2011 to 2021' in p.text:
        for run in p.runs:
            run.text = run.text.replace('2011 to 2021', '2011 to 2019')

# doc.tables[0] = inspections (16 rows); doc.tables[1] = violations (20 rows)
fill_table(doc.tables[0], P['inspections'])
fill_table(doc.tables[1], P['violations'])

# Append a collinearity note at the end (full discussion in docs/collinearity_notes.md).
doc.add_paragraph()
note = doc.add_paragraph()
note.add_run('Note on collinearity. ').bold = True
note.add_run(
    'The Labor Intensity Index is entered for the 2017 Census wave only; the '
    '2012, 2017, and 2022 waves are near-collinear (r ≈ 0.977) and produce '
    'unstable, sign-flipping estimates when entered jointly. Spending/applicator '
    'and Spending/farmworker share a common numerator (state STAG obligations) '
    'and are moderately correlated; both are retained as the two FIFRA-protected '
    'populations, but their standard errors should be read with that overlap in '
    'mind. The H-2A block variables are expressed as ratios (H-2A relative to the '
    'BLS farmworker base; authorized relative to requested) to avoid the scale '
    'collinearity of raw counts. See docs/collinearity_notes.md for detail.')

doc.save(OUT)
print(f"Saved filled tables to {OUT}")
