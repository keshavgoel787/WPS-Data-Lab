"""
AUGMENTED WPS Tables 2 & 3 (2026-07).

Builds fresh Word tables from paper_table_params_corrected.json that ADD, on top
of the corrected b/(SE) shells:

  1. a separate standardized-beta column after each model's b / (SE);
  2. a "State-to-State Variation" block reporting the full random-effects set
     (sigma^2_u0 intercept, sigma^2_u1 time-slope, sigma_u01 covariance,
     sigma^2_e residual) as ORIGINAL (time-only baseline, same sample) -> FITTED,
     plus the Delta sigma^2_u0 percent.

Fresh tables (not the fixed placeholder shell) because the extra beta columns and
the 4-row variance block change the column grid beyond safe placeholder edits.
The committed docs/WPS_Table_Sheels_filled_corrected.docx is left untouched.

Requires paper_table_models_corrected.py to have run.
Output: docs/WPS_Table_Sheels_augmented.docx
"""
import json
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

JSON = '/Users/keshavgoel/Research/data/generated/paper_table_params_corrected.json'
OUT = '/Users/keshavgoel/Research/docs/WPS_Table_Sheels_augmented.docx'
MODELS = ['M1', 'M2', 'M3']

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
# variance block: (label, base_key, fit_key)
VAR_ROWS = [
    ('  σ²_u0  (between-state intercept)', 'sigma2_u0_baseline', 'sigma2_u0'),
    ('  σ²_u1  (between-state time slope)', 'sigma2_u1_baseline', 'sigma2_u1'),
    ('  σ_u01  (intercept–slope covariance)', 'sigma_u01_baseline', 'sigma_u01'),
    ('  σ²_e   (within-state residual)', 'sigma2_e_baseline', 'sigma2_e'),
]


def set_cell(cell, text, bold=False, align='center', size=9):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if align == 'center' else WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)


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


def build_table(doc, title, subtitle, tab, rows):
    doc.add_paragraph().add_run(title).bold = True
    doc.add_paragraph(subtitle)
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
    set_cell(merged, 'State-to-State Variation  (original → became)', bold=True, align='left')

    # variance component rows: one merged cell per model = "baseline -> fitted"
    for label, bkey, fkey in VAR_ROWS:
        r = t.add_row().cells
        set_cell(r[0], label, align='left')
        for i, m in enumerate(MODELS):
            base = 1 + i * 3
            cell = r[base].merge(r[base + 1]).merge(r[base + 2])
            b, fdat = tab[m].get(bkey), tab[m].get(fkey)
            set_cell(cell, f"{b:.3f} → {fdat:.3f}" if b is not None and fdat is not None else '')
    # delta sigma2_u0 %
    r = t.add_row().cells
    set_cell(r[0], '  Δ σ²_u0 (reduction vs baseline)', align='left')
    for i, m in enumerate(MODELS):
        base = 1 + i * 3
        cell = r[base].merge(r[base + 1]).merge(r[base + 2])
        set_cell(cell, f"{tab[m]['delta_pct']:.1f}%" if 'delta_pct' in tab[m] else '')

    # N row
    r = t.add_row().cells
    set_cell(r[0], '  N (state-years / states)', align='left')
    for i, m in enumerate(MODELS):
        base = 1 + i * 3
        cell = r[base].merge(r[base + 1]).merge(r[base + 2])
        no, ns = tab[m].get('n_obs'), tab[m].get('n_states')
        set_cell(cell, f"{no} / {ns}" if no is not None else '')
    doc.add_paragraph()


doc = Document()
doc.add_paragraph().add_run(
    'WPS Enforcement Multilevel Models, 2011–2019 (augmented)').bold = True

build_table(doc, 'Table 2. WPS Inspections',
            'DV = log(inspections + 1). Random intercept + random linear-time slope by state (REML).',
            P['inspections'], INSPECTIONS_ROWS)
build_table(doc, 'Table 3. WPS Violations',
            'DV = log(violations + 1). Random intercept + random linear-time slope by state (REML).',
            P['violations'], VIOLATIONS_ROWS)

note = doc.add_paragraph()
note.add_run('Note. ').bold = True
note.add_run(
    'Each model reports the unstandardized coefficient b, its standard error (SE), '
    'and the fully standardized coefficient β = b·SD(x)/SD(y), with SDs taken on that '
    "column's own analytic sample. β for the time polynomials and log(Inspections+1) is "
    'reported for completeness but is of limited interpretive value. The State-to-State '
    'Variation block shows each random-effects component as the value under a time-only '
    'baseline (re-estimated on the same sample) → the value under the fitted model; '
    'Δ σ²_u0 is the percent reduction in between-state intercept variance. A negative '
    'Δ (e.g., Violations Model 2) means the added Level-2 predictors raised between-state '
    'intercept variance on that sample — a genuine suppression/reallocation effect given the '
    'correlated random slope, not a sign error. Both outcomes are log(count+1); the violations '
    'model includes log(inspections+1) as a contemporaneous predictor. Spending variables use '
    'mean 2011–2019 STAG obligations over mean 2011–2019 BLS employment; MS/RI/WV (like '
    'HI/TN/UT) received no CFDA 66.700 obligations and are coded $0. AK/RI/VT drop from '
    'Spending/applicator columns (BLS never publishes SOC 37-3012 for them), giving N=47 states. '
    '+ p<.10, * p<.05, ** p<.01, *** p<.001.')

doc.save(OUT)
print(f"Saved augmented tables to {OUT}")
