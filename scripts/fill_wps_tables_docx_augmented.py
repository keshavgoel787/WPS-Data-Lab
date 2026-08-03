"""
AUGMENTED WPS Tables 2 & 3 (2026-07).

Opens the original Word shell (~/Downloads/WPS Table Sheels.docx) so the augmented
tables INHERIT the shell's "Table 2." / "Table 3." captions and significance notes
(Helvetica 11, "inspections"/"violations" bold-italic), correcting the study period
2021 -> 2019. It then replaces each placeholder table IN PLACE with a freshly built
augmented table (kept fresh -- not placeholder-edited -- because the extra beta
columns and the 4-row variance block change the column grid) that ADDS, on top of
the corrected b/(SE) shell:

  1. a separate standardized-beta column after each model's b / (SE);
  2. a "State-to-State Variation" block reporting the full random-effects set
     (sigma^2_u0 intercept, sigma^2_u1 time-slope, sigma_u01 covariance,
     sigma^2_e residual) as ORIGINAL (time-only baseline, same sample) -> FITTED,
     plus the Delta sigma^2_u0 percent.

The committed docs/WPS_Table_Sheels_filled_corrected.docx is left untouched.

Requires paper_table_models_corrected.py to have run.
Output: docs/WPS_Table_Sheels_augmented.docx
"""
import json
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

SRC = '/Users/keshavgoel/Downloads/WPS Table Sheels.docx'
JSON = '/Users/keshavgoel/Research/data/generated/paper_table_params_corrected.json'
OUT = '/Users/keshavgoel/Research/docs/WPS_Table_Sheels_augmented.docx'
MODELS = ['M1', 'M2', 'M3']
FONT = 'Helvetica'          # match the shell's caption/body typeface

with open(JSON) as f:
    P = json.load(f)

# (row label, json term). Order mirrors the corrected docx.
INSPECTIONS_ROWS = [
    ('Time', 'time'), ('Time²', 'time2'), ('Time³', 'time3'),
    (None, None),
    ('Spending/applicator', 'SPEND_APP_z'),
    ('Spending/farmworker', 'SPEND_WORK_z'),
    ('Labor Intensity', 'lii_2017_z'),
    (None, None),
    ('H-2A farmworkers:BLS farmworkers', 'h2a_per_farmworker_z'),
    ('H-2A Authorized/H-2A Requested', 'dol_demand_met_pct_z'),
    ('%H-2A Authorized to FLC', 'pct_flc_z'),
]
VIOLATIONS_ROWS = [
    ('log(Inspections+1)', 'log_inspections'),
    ('Time', 'time'), ('Time²', 'time2'), ('Time³', 'time3'),
    (None, None),
    ('Spending/applicator', 'SPEND_APP_z'),
    ('Spending/farmworker', 'SPEND_WORK_z'),
    ('Labor Intensity', 'lii_2017_z'),
    (None, None),
    ('H-2A farmworkers:BLS farmworkers', 'h2a_per_farmworker_z'),
    ('H-2A Authorized/H-2A Requested', 'dol_demand_met_pct_z'),
    ('%H-2A Authorized to FLC', 'pct_flc_z'),
]
# variance block: (label, base_key, fit_key) -- random-INTERCEPT basis, so the
# between-state variance-explained is the conventional [0,1] pseudo-R^2 (the
# random-slope sigma^2_u0 is centered at 2017 and can rise when predictors enter).
VAR_ROWS = [
    ('  σ²_u0  (between-state intercept)', 'sigma2_u0_ri_baseline', 'sigma2_u0_ri'),
    ('  σ²_e   (within-state residual)', 'sigma2_e_ri_baseline', 'sigma2_e_ri'),
]


def set_cell(cell, text, bold=False, align='center', size=9):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if align == 'center' else WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = FONT           # inherit the shell's typeface


def coef_cells(tab, term):
    """Return [b1,se1,beta1, b2,se2,beta2, b3,se3,beta3] strings ('' if absent)."""
    out = []
    for m in MODELS:
        c = tab[m].get(term)
        if c is None or not isinstance(c, dict) or 'b' not in c:
            out += ['', '', '']
        else:
            out += [f"{c['b']:.3f}{c['stars']}", f"({c['se']:.3f})", f"{c['beta']:.3f}"]
    return out


def make_table(doc, tab, rows):
    """Build the augmented table (appended at doc end) and return the Table.

    No caption/heading paragraphs are added -- the caller relocates this table
    directly under the shell's own "Table N." caption so that styling is inherited.
    """
    ncol = 1 + 3 * 3  # label + (b,SE,beta) x 3 models
    t = doc.add_table(rows=0, cols=ncol)
    t.style = 'Table Grid'

    # group header: Model 1 / Model 2 / Model 3
    hr = t.add_row().cells
    set_cell(hr[0], '')
    for i, m in enumerate(['Model 1', 'Model 2', 'Model 3']):
        base = 1 + i * 3
        merged = hr[base].merge(hr[base + 1]).merge(hr[base + 2])
        set_cell(merged, m, bold=True)
    # sub header: b (SE) beta
    sr = t.add_row().cells
    set_cell(sr[0], 'Predictor', bold=True, align='left')
    for i in range(3):
        set_cell(sr[1 + i * 3], 'b', bold=True)
        set_cell(sr[2 + i * 3], '(SE)', bold=True)
        set_cell(sr[3 + i * 3], 'β', bold=True)

    # coefficient rows
    for label, term in rows:
        r = t.add_row().cells
        if label is None:                      # spacer
            continue
        set_cell(r[0], label, align='left')
        for j, val in enumerate(coef_cells(tab, term)):
            set_cell(r[1 + j], val)

    # variance block header
    vh = t.add_row().cells
    merged = vh[0]
    for c in vh[1:]:
        merged = merged.merge(c)
    set_cell(merged, 'State-to-State Variation — random-intercept basis  (Model 1 baseline → model)',
             bold=True, align='left')

    # variance component rows: one merged cell per model = "baseline -> fitted"
    for label, bkey, fkey in VAR_ROWS:
        r = t.add_row().cells
        set_cell(r[0], label, align='left')
        for i, m in enumerate(MODELS):
            base = 1 + i * 3
            cell = r[base].merge(r[base + 1]).merge(r[base + 2])
            b, fdat = tab[m].get(bkey), tab[m].get(fkey)
            set_cell(cell, f"{b:.3f} → {fdat:.3f}" if b is not None and fdat is not None else '')
    # delta sigma2_u0 % (random-intercept basis: between-state variance explained)
    r = t.add_row().cells
    set_cell(r[0], '  Δ σ²_u0 (between-state variance explained)', align='left')
    for i, m in enumerate(MODELS):
        base = 1 + i * 3
        cell = r[base].merge(r[base + 1]).merge(r[base + 2])
        set_cell(cell, f"{tab[m]['delta_pct_ri']:.1f}%" if 'delta_pct_ri' in tab[m] else '')

    # N row
    r = t.add_row().cells
    set_cell(r[0], '  N (state-years / states)', align='left')
    for i, m in enumerate(MODELS):
        base = 1 + i * 3
        cell = r[base].merge(r[base + 1]).merge(r[base + 2])
        no, ns = tab[m].get('n_obs'), tab[m].get('n_states')
        set_cell(cell, f"{no} / {ns}" if no is not None else '')
    return t


def replace_table(doc, old_table, tab, rows):
    """Swap a shell placeholder table for a freshly built augmented one, in place."""
    new = make_table(doc, tab, rows)                 # appended at doc end
    old_table._tbl.addprevious(new._tbl)             # move into the old slot
    old_table._tbl.getparent().remove(old_table._tbl)


# Open the shell so captions ("Table 2." / "Table 3.") and significance notes,
# with their Helvetica styling, are inherited.
doc = Document(SRC)

# Study period: the data run 2011-2019, not 2021.
for p in doc.paragraphs:
    if '2011 to 2021' in p.text:
        for run in p.runs:
            run.text = run.text.replace('2011 to 2021', '2011 to 2019')

# Capture the two placeholder tables BEFORE we append new ones.
shell_insp, shell_viol = doc.tables[0], doc.tables[1]
replace_table(doc, shell_insp, P['inspections'], INSPECTIONS_ROWS)
replace_table(doc, shell_viol, P['violations'], VIOLATIONS_ROWS)

note = doc.add_paragraph()
r0 = note.add_run('Note. ')
r0.bold = True
r1 = note.add_run(
    'Each model reports the unstandardized coefficient b, its standard error (SE), '
    'and the fully standardized coefficient β = b·SD(x)/SD(y), with SDs taken on that '
    "column's own analytic sample. β for the time polynomials and log(Inspections+1) is "
    'reported for completeness but is of limited interpretive value. Coefficients (b, SE, β) '
    'come from the paper spec — a random intercept plus a random linear-time slope by state. '
    'The State-to-State Variation block is read from random-INTERCEPT-only models. Every '
    'column is referenced to the same Model 1 baseline (Model 1’s σ²): the σ²_u0 and σ²_e '
    'rows show that Model 1 value → the value under the fitted model, and Δ σ²_u0 is the '
    'percent of Model 1 between-state intercept variance explained. The random-intercept '
    'basis is used because in the random-slope model σ²_u0 is the between-state variance at '
    'the centering year (2017) and trades off against the slope variance/covariance, so its '
    'reduction is not bounded to [0,1]. Note: in the inspections table, Models 2–3 omit '
    'Alaska, Rhode Island, and Vermont (no BLS applicator data), and those three states carry '
    'much of the between-state inspection variance, so part of that column’s reduction against '
    'Model 1 reflects the narrower sample as well as the covariates; the violations table is '
    'unaffected. Both outcomes are log(count+1); the violations '
    'model includes log(inspections+1) as a contemporaneous predictor. Spending variables use '
    'mean 2011–2019 STAG obligations over mean 2011–2019 BLS employment; MS/RI/WV (like '
    'HI/TN/UT) received no CFDA 66.700 obligations and are coded $0. AK/RI/VT drop from '
    'Spending/applicator columns (BLS never publishes SOC 37-3012 for them), giving N=47 states. '
    '+ p<.10, * p<.05, ** p<.01, *** p<.001.')
for r in (r0, r1):
    r.font.name = FONT
    r.font.size = Pt(9)

doc.save(OUT)
print(f"Saved augmented tables to {OUT}")
