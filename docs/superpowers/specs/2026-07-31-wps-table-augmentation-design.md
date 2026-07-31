# WPS Table Augmentation + Spaghetti Plots — Design Spec

**Date:** 2026-07-31
**Author:** Keshav Goel
**Baseline:** commit `6253156` — the corrected (2026-07) table shells
(`docs/WPS_Table_Sheels_filled_corrected.docx`, produced by
`paper_table_models_corrected.py` → `paper_table_params_corrected.json` →
`fill_wps_tables_docx_corrected.py`). Both DVs are `log(count+1)`, N=47.

## Goal

Five additions to the corrected inspections (Table 2) and violations (Table 3)
outputs, all driven off the existing corrected pipeline:

1. State-to-state variation parameter estimates at the bottom of both tables,
   shown as a **before/after comparison** (original time-only baseline → fitted).
2. Spaghetti plot for **inspections** — 15 randomly selected states.
3. Matching spaghetti plot for **violations** — same procedure, independent draw.
4. Verify the sign on the Model-2 violations variance estimate (is −12.5% right?).
5. Standardized β coefficients added to each column of both tables.

## Decisions (confirmed with PI)

- **Three time periods (spaghetti eligibility):** Pre = 2011–2015, Spike =
  2016–2017, Post = 2018–2019.
- **Spaghetti state selection:** re-sampled **independently** per outcome
  (inspections and violations get separate random draws). Fixed seed for
  reproducibility.
- **Variation block:** full random-effects set — σ²_u0 (intercept), σ²_u1
  (time slope), σ_u01 (intercept–slope covariance), σ²_e (residual) — for
  M1/M2/M3, each shown as **original (time-only baseline, same sample) → fitted**.
- **Standardized β:** **separate β column** after each model's (b, SE).
- **Plot scale:** raw counts (descriptive; the 2016–17 spike reads clearly).

## Component A — extend `paper_table_models_corrected.py`

Add to the per-column dict written to `paper_table_params_corrected.json`,
without disturbing existing fields:

1. **Full RE components** from `res.cov_re` / `res.scale`, for **M1, M2, M3**:
   - `sigma2_u0 = cov_re.iloc[0,0]`
   - `sigma2_u1 = cov_re.iloc[1,1]`
   - `sigma_u01 = cov_re.iloc[0,1]`
   - `sigma2_e  = res.scale`
2. **Baseline RE components** (the time-only model re-estimated on the same
   analytic sample — already fit as `b` inside `build()`): `sigma2_u0_baseline`,
   `sigma2_u1_baseline`, `sigma_u01_baseline`, `sigma2_e_baseline`. Enables the
   original→became comparison. For M1 the baseline equals the model itself.
3. **Standardized β** per fixed effect: `beta = b · SD_sample(x) / SD_sample(y)`,
   with SDs on that column's own listwise-deleted analytic sample (z-scored vars
   drift from SD=1 after deletion, so recompute rather than assume 1). Stored as
   `beta` alongside each term's `b`/`se`/`p`/`stars`. Computed for all fixed
   effects incl. time polynomials and `log_inspections` (std-β on time terms is
   of limited interpretive value; flagged in the docx note).

`sig2()` helper stays; add a `re_components(res)` helper. Re-run the script to
regenerate the JSON. The existing `delta_pct` field is retained.

## Component B — `fill_wps_tables_docx_augmented.py`

New script (does not overwrite the committed corrected docx). Builds **fresh
tables** rather than mutating the fixed placeholder grid, because separate β
columns and a 4-row RE block change the column structure beyond what safe
placeholder substitution allows. Per table:

- Header: for each model M1/M2/M3, three sub-columns — `b`, `(SE)`, `β`.
- Coefficient rows in the same order as the corrected docx.
- Bottom **State-to-State Variation** block: one row per RE component
  (σ²_u0, σ²_u1, σ_u01, σ²_e), each cell showing `baseline → fitted`, plus the
  existing Δσ²_u0 % row.
- Closing note: carries over the log-scale / multi-year BLS / $0-coded states /
  year-matched H-2A / N=47 note, and adds the std-β definition + the time-term
  caveat.

**Output:** `docs/WPS_Table_Sheels_augmented.docx`.

## Component C — `scripts/spaghetti_plots.py`

- Build the state×year panel from `establishments_data.csv` (same DV construction
  as the model script: violations; inspections = EPA + state).
- **Eligibility:** state has ≥1 **nonzero** DV value in each of Pre/Spike/Post.
  (The raw panel is otherwise complete for all 50 states, so a plain
  "has a row" filter would select everyone; the nonzero reading is the
  substantive one.)
- Randomly draw 15 eligible states per outcome, **independent draws**, fixed
  seed.
- One line per state, year (2011–2019) on x, raw count on y. dataviz-skill
  styling: muted lines + endpoint labels (15 lines exceeds the categorical
  palette limit), light single-theme PNG suitable for the manuscript.
- **Outputs:** `figures/fig_spaghetti_inspections.png`,
  `figures/fig_spaghetti_violations.png`.

## Component D — verify M2 violations variance sign

Independently recompute σ²_u0 for the M2 violations baseline (time-only + log
inspections, on the M2 sample) vs the fitted M2 model. Confirm:
- σ²_u0 itself is positive (variance can't be negative);
- the **−12.5% Δσ²_u0** is a genuine estimate — adding SPEND+LII *raised*
  between-state intercept variance (plausible via the correlated random slope /
  a suppression effect), not a sign bug in the fill/computation.

Report the finding. Only change code if it is an actual sign error.

## Run order

```
paper_table_models_corrected.py      # regenerates augmented JSON
fill_wps_tables_docx_augmented.py    # builds docs/WPS_Table_Sheels_augmented.docx
spaghetti_plots.py                   # standalone; two PNGs
```

Requires (already present in `data/generated/`):
`spend_bls_variables_multiyear.csv`, `h2a_ratio_panel.csv`.

## Out of scope

- No change to the committed corrected docx or the earlier cubic/paper tables.
- No new modeling specification (same M1/M2/M3 build-up, cubic time, no ×time
  interactions).
- No re-harvest of BLS/spending/H-2A data.
