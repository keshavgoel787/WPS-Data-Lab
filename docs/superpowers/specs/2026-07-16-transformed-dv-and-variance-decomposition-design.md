# Design — Transformed-DV Violations Models & Variance-Component Decomposition

**Date:** 2026-07-16
**Status:** Approved (design), pending spec review
**Scope:** Two exploratory analysis modules for the WPS project. Neither touches the
manuscript pipeline (`paper_table_models.py`, `fill_wps_tables_docx.py`, or the
filled `.docx` outputs). Both reuse existing data-build conventions and constants.

---

## Motivation

1. **Transformed DV (Part A).** WPS violation counts are right-skewed with a sharp
   2016–2017 spike and many small values. A Gaussian `MixedLM` on the raw count may
   violate normality/homoscedasticity assumptions. We want to see how the full
   violations model suite behaves under standard count transformations, and which
   transformation best satisfies the LMM assumptions without changing substantive
   conclusions.
2. **Variance decomposition (Part B).** The manuscript reports a single pseudo-R²
   (proportional reduction in between-state *intercept* variance). That hides the
   rest of the random-effects structure — the random slope, the intercept–slope
   covariance, the Level-1 residual — and it produced a puzzling *negative* Δσ² in
   the inspections table. We want a full component breakdown, a time-varying
   variance partition, and a diagnosis of the negative Δσ².

---

## Part A — Transformed-DV Violations Models

### A.1 Transformations

Each transform is fit and documented **separately**, with `raw` as the reference.

| Key | Transform | Rationale / note |
|---|---|---|
| `raw` | `y` | reference (current spec) |
| `log` | `log(y + 1)` | zeros-safe; coefficients read as ≈ proportional change |
| `sqrt` | `sqrt(y)` | variance-stabilizer for Poisson-like counts; defined at 0 |
| `anscombe` | `2 * sqrt(y + 3/8)` | small-count refinement; **Freeman–Tukey** `sqrt(y) + sqrt(y+1)` reported alongside for comparison |
| `boxcox` | Box-Cox on `(y + 1)`, λ estimated by MLE | report the fitted λ; Box-Cox requires strictly positive input, hence the `+1` shift. **Yeo–Johnson** noted in the write-up as the zeros-native alternative. |

Implementation notes:
- Box-Cox λ is estimated **once** on the full `(violations+1)` series via
  `scipy.stats.boxcox`, then that fixed λ is applied to build the DV column. The
  fitted λ is reported.
- The transform is applied to the DV column only; all predictors and the random-
  effects structure are unchanged.

### A.2 Model suite

One driver script defines every violations model specification once (as a list of
`(name, rhs, re_formula)` specs) and fits each spec under every transform, so the
transforms are strictly comparable. Specs mirror the existing violations scripts:

- **Time-only baseline** — cubic time, random intercept + random slope
  (`re_formula='~time'`). Mirrors `hierarchical_violations_model.py` M2.
- **Zimmerman stepwise** — each Level-2 covariate entered one-at-a-time
  (`spending_per_estab`, `land_area`, `farming_operations`, `h2a_workers`,
  `workers_per_operation`). Mirrors `stepwise_zimmerman_models.py`.
- **BLS spending** — the 5 `SPEND_*` variables one-at-a-time + the combined model.
  Mirrors `spending_bls_models.py`.
- **Targeted FIFRA** — `SPEND_WORK + SPEND_APP + SPEND_AREA + 2 interactions`.
  Mirrors `targeted_spend_model.py`.
- **Labor/DOL stepwise** — the 6 labor/DOL covariates one-at-a-time. Mirrors
  `labor_covariates_violations_models.py`.
- **Curated final** — cubic time + labor block (`lii_2017_z`,
  `h2a_per_farmworker_z`, `dol_demand_met_pct_z`, `pct_flc_z`) + interactions
  screened at p<.20. Mirrors `final_labor_violations_model.py`.
- **Paper Table 3 build-up** — M1/M2/M3 (inspections + time + spending + labor +
  H-2A). Mirrors the violations half of `paper_table_models.py`. Uses the current
  linear-time spec for the paper models; the time-only and final specs keep cubic
  time per their source scripts.

All models use `MixedLM.from_formula(..., re_formula='~time').fit(method='lbfgs')`,
REML, per project convention. Level-2 covariates are z-scored before entry.

### A.3 Comparison metrics — the AIC caveat

**Log-likelihood and AIC are not comparable across different DV transformations**
without a Jacobian (change-of-variables) correction, so we do **not** rank
transforms by raw AIC/loglik. Instead, per model spec, we report:

1. **Residual diagnostics** (the primary basis for judging the transform):
   - Shapiro–Wilk W and p on the (marginal, conditional) residuals
   - residual skewness and excess kurtosis
   - a heteroscedasticity check (residual-vs-fitted; Breusch–Pagan-style statistic)
   - QQ plot (saved to `figures/`)
2. **Substantive stability**: sign and significance of the key coefficients across
   transforms — does the story hold?
3. **Pseudo-R²** — the proportional reduction in between-state intercept variance
   relative to a matched-N baseline. Being a variance *ratio* it is comparable
   across transforms and is reported per transform.

### A.4 Scope / volume note

The stepwise "one-at-a-time" families produce many small models. In the terminal
output and the CSV these are reported compactly (one row per model). The write-up
narrative focuses on the models that carry the manuscript conclusions (time-only
baseline, curated final, paper Table 3 build-up); the stepwise results are
tabulated but not individually discussed.

### A.5 Outputs

- `scripts/transformed_violations_models.py` — the driver.
- Terminal: per-transform summary table.
- `data/generated/transform_comparison.csv` — one row per (transform × model) with
  coefficients, p-values, pseudo-R², and diagnostic statistics.
- `docs/transform_exploration.md` — write-up, interpretation, and a recommendation
  on which transform (if any) to prefer for a robustness appendix.
- `figures/transform_qq_<model>_<transform>.png` (and residual-vs-fitted) for the
  headline models.

---

## Part B — Variance-Component Decomposition

### B.1 What is extracted

For the paper-table **violations** and **inspections** models (M1/M2/M3), extract
the full random-effects structure from each fitted `MixedLM`:

- σ²_u0 — between-state intercept variance (`cov_re.iloc[0,0]`)
- σ²_u1 — random-slope (time) variance (`cov_re.iloc[1,1]`)
- σ_u01 — intercept–slope covariance (`cov_re.iloc[0,1]`), and corr(u0,u1)
- σ²_e — Level-1 (within-state) residual variance (`res.scale`)

### B.2 Derived quantities

- **ICC** at time = 0: σ²_u0 / (σ²_u0 + σ²_e).
- **Time-varying VPC**, because a random slope makes the between-state share depend
  on time:

  VPC(t) = ( σ²_u0 + 2t·σ_u01 + t²·σ²_u1 ) / ( σ²_u0 + 2t·σ_u01 + t²·σ²_u1 + σ²_e )

  evaluated across the study window (t = year − 2017, 2011–2019) and plotted.
- **Per-block component shifts**: how each covariate block (spending, labor, H-2A)
  changes *each* variance component — not only σ²_u0. This is what the single
  pseudo-R² hides and what explains the negative inspections Δσ².

### B.3 Negative-Δσ² diagnosis

The inspections table showed Δσ² of −0.2% / −1.3% (adding covariates *raised*
between-state intercept variance). In a mixed model σ²_u0 is not bounded to fall
when fixed effects are added — it can rise when a covariate re-partitions variance
between the intercept and slope components or between levels. The report will show
the full component movement for the inspections M2/M3 to make this concrete
(e.g., whether σ²_u1 or σ²_e absorbed the change), rather than leaving the negative
value unexplained.

### B.4 Outputs

- `scripts/variance_decomposition.py` — the driver.
- Terminal: component tables for both DVs.
- `docs/variance_decomposition.md` — the decomposition math, the component tables,
  the VPC(t) discussion, and the negative-Δσ² diagnosis.
- `figures/vpc_<dv>.png` (VPC across years) and a component bar chart per DV.

---

## Shared conventions (both parts)

- Constants (`US_STATES_50`, `STATE_ABBREV_TO_NAME`, `STATE_NAME_MAPPING`) match
  the existing scripts. Inflation adjustment is not re-done here — spending
  variables arrive pre-standardized from `data/generated/spend_bls_variables.csv`.
- Data build (long DV frame + z-scored Level-2 covariates) mirrors
  `paper_table_models.py`; a small shared build is duplicated in each script rather
  than imported, matching the project's current no-shared-module convention.
- `time = year − 2017`. Baselines for pseudo-R² / Δσ² are re-estimated on each
  model's own listwise-complete analytic sample (matched N), per CLAUDE.md.
- Everything is exploratory: terminal + `data/generated/` + `docs/` + `figures/`
  only. No manuscript artifact is modified.

## Non-goals

- No switch to a true count GLMM (Poisson/NB mixed) — the request is to *transform*
  the DV and re-fit the existing Gaussian `MixedLM` suite.
- No changes to the manuscript tables, the filled `.docx`, or `paper_table_models.py`.
- No new raw data.
