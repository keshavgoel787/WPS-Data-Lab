# Final WPS Models — Labor/DOL Covariates

Curated final hierarchical models for the WPS enforcement study, implementing PI
direction (June 2026). These supersede the all-covariate combined models in the
exploratory `labor_covariates_*` scripts, which are retained as the stepwise
record.

| Script | DV | Time spec |
|---|---|---|
| `final_labor_violations_model.py` | WPS violations (state-year) | Cubic (`time + time2 + time3`) |
| `final_labor_inspections_model.py` | WPS inspections, EPA + state (state-year) | **Linear** (`time` only) |

## Running

```bash
# Run from /Users/keshavgoel/Research/ — all paths are absolute and hardcoded.
python3 final_labor_violations_model.py
python3 final_labor_inspections_model.py
```

No build step, no arguments, terminal output only. Requires standard scientific
Python (`pandas`, `numpy`, `statsmodels`). The scripts are self-contained — they
re-derive everything from the raw inputs and depend on no generated intermediate
files.

## Inputs

| File | Provides |
|---|---|
| `establishments-data (2).csv` | EPA ECHO violations and EPA/state inspection counts (wide, one column per year) |
| `spending_data_master(in) (1).csv` | Nominal STAG grant obligations by state abbreviation and year |
| `labor_intensity_index_2017.csv` | Labor Intensity Index, 2017 Census of Agriculture |
| `dol_var1_workers_by_state_annual.csv` | DOL H-2A annual certified workers / cases / demand-met % |
| `dol_var2_employer_type_annual.csv` | DOL H-2A employer-type breakdown (2020 used as proxy) |

## The two PI directives these scripts implement

**1. Labor Intensity Index → 2017 wave only.**
The 2012 / 2017 / 2022 LII waves are near-collinear (r ≈ 0.977) and flip parameter
signs when entered together. The final models include only **`lii_2017`**, matching
the time reference year (`time = year − 2017`). The 2012-vs-2017 comparison stays
documented in `labor_covariates_violations_models.py` / `..._inspections_models.py`.

**2. Inspections → linear time only.**
The inspections series carries only a linear time trend, so `time2` and `time3` are
dropped from the final inspections model. Violations keep the cubic because the
2016–2017 violation spike is asymmetric and requires it. This is the one explicit
exception to the project's otherwise-strict "never drop time terms" rule (documented
in `CLAUDE.md`).

## Model specification

Both scripts fit a two-level hierarchical model (state-years nested in states) via
`statsmodels.MixedLM`, REML, `lbfgs`:

- **Random effects:** random intercept + random slope for linear time by state
  (`re_formula='~time'`).
- **Final covariate block** (z-scored Level-2 predictors, mean 0 / SD 1):
  `lii_2017_z`, `dol_workers_cert_z`, `dol_n_cases_z`, `dol_demand_met_pct_z`,
  `pct_flc_z`.
- **×time interaction screen:** each covariate's `×time` interaction is fit
  one-at-a-time and retained in the final model if `p < .20`. (For inspections,
  `×time` is linear and consistent with the linear-time spec.)
- **Pseudo-R²:** reduction in between-state intercept variance (`σ²_u0`) relative to
  a time-only baseline, `(σ²_u0_baseline − σ²_u0_final) / σ²_u0_baseline`. The
  baseline is **re-estimated on the final listwise-complete analytic sample** and
  with the **same time spec** as the final model (cubic for violations, linear for
  inspections), so the comparison is against a matched baseline.

## What each run prints

1. Data load and analytic-sample sizes.
2. Covariate standardization (mean / SD / n per covariate).
3. Baseline model (time-only) with σ²_u0 and ICC.
4. The ×time interaction screen with per-covariate decisions.
5. The final model: fixed effects with β / SE / p, σ²_u0, pseudo-R², σ²_ε, and
   log-likelihood.

## Results summary (current data)

| Model | N (obs / states) | Pseudo-R² | Notes |
|---|---|---|---|
| Violations (cubic) | 250 / 47 | 43.2% | `pct_flc`, `dol_workers_cert`, `dol_n_cases` significant; `lii_2017` n.s. (retained as control); 3 ×time interactions retained |
| Inspections (linear) | 282 / 47 | 17.0% | `dol_workers_cert` significant; no ×time interaction survives; baseline linear time itself weak (p ≈ .20), confirming linear-only spec |

## Notes and caveats

- **`dol_workers_cert` vs `dol_n_cases`** are moderately correlated (r = 0.67) — not
  the LII-level collinearity, so both are retained. Revisit only if the PI requests a
  parallel single-variable cleanup.
- **`pct_flc`** uses 2020 DOL employer-type data as a structural proxy (earliest year
  available); it does not cover the 2011–2019 study period.
- **DOL H-2A means** exclude 2013 (missing from source) and exclude 2014 from
  `demand_met_pct` (requested = 0 → infinite).
- **Ridge / penalized models** (PI aside) are not implemented here. Open question to
  settle first: whether a penalized model must share one covariate set across both
  DV models.
