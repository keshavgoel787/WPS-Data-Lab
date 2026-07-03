# Final Models Update — July 2026

This note documents the PI-directed revisions to the two curated final models and
the new comprehensive ("every variable") inspections model.

Scripts affected:

| Script | Change |
|---|---|
| `final_labor_violations_model.py` | Covariate block revised (see below); cubic time retained |
| `final_labor_inspections_model.py` | Same covariate block; linear time retained |
| `comprehensive_inspections_model.py` | **New** — inspections model entering every project covariate |

All models keep the established estimation convention:
`MixedLM.from_formula(..., re_formula='~time').fit(method='lbfgs')` (random
intercept + random slope for linear time, REML), Level-2 covariates z-scored,
pseudo-R² = reduction in between-state variance vs. a baseline re-estimated on
the same listwise-complete analytic sample.

---

## 1. Covariate block changes (both final models)

**Before:** `lii_2017_z, dol_workers_cert_z, dol_n_cases_z, dol_demand_met_pct_z, pct_flc_z`

**After:** `lii_2017_z, h2a_per_farmworker_z, dol_demand_met_pct_z, pct_flc_z`

| Change | Detail |
|---|---|
| **New ratio** | `h2a_per_farmworker` = mean certified H-2A workers (numerator) ÷ farmworker employment. Denominator is **BLS OEWS OCC 45-2092** ("Farmworkers and Laborers, Crop, Nursery, and Greenhouse", 2011 snapshot). Expresses H-2A certified visas relative to the agricultural workforce rather than as a raw count. |
| **Replaced** | `dol_workers_cert` (raw certified H-2A count) is no longer entered on its own — it survives only as the numerator of the ratio above. |
| **Dropped** | `dol_n_cases` removed from the block. |
| **Kept** | `dol_demand_met_pct` (certified ÷ requested H-2A workers, 2011–2019 mean, excl. 2013; `inf` states excluded from the mean) and `pct_flc` (% certifications via Farm Labor Contractor, 2020 proxy). |
| **LII** | Labor Intensity Index uses the **2017 wave only**; 2012 and 2022 waves are not read (near-collinear, r ≈ 0.977). |

**Sample effect:** BLS suppresses OCC 45-2092 for ~6 states, so the new ratio is
non-missing for 44 states. Listwise deletion on the block therefore yields **44
states** (down from 50), re-baselined accordingly.

---

## 2. Final Violations Model (cubic time)

`violations ~ time + time2 + time3 + [block]`, N = **235 obs, 44 states**.

**×time interaction screen (p < .20):** only `pct_flc_z` retained (p = .013).

Final fixed effects:

| Parameter | β | SE | p |
|---|---:|---:|---:|
| Intercept | 17.807 | 3.216 | 0.000 ** |
| time | 3.048 | 0.804 | 0.000 ** |
| time² | −0.683 | 0.355 | 0.054 + |
| time³ | −0.156 | 0.054 | 0.004 ** |
| lii_2017_z | −0.454 | 1.739 | 0.794 |
| h2a_per_farmworker_z | −0.923 | 1.067 | 0.387 |
| dol_demand_met_pct_z | 1.959 | 1.096 | 0.074 + |
| pct_flc_z | 10.392 | 3.347 | 0.002 ** |
| pct_flc_z:time | 1.432 | 0.585 | 0.014 * |

σ²_u0 = 295.99 (**pseudo-R² = 15.2%** of between-state variance vs. re-baselined
time-only model); σ²_ε = 97.91; LL = −906.86.

**Read:** `pct_flc` remains the dominant labor predictor of violations (strong
positive main effect and a positive ×time interaction — states with more
Farm-Labor-Contractor certifications diverge upward over time). `dol_demand_met_pct`
is marginal (+). The new H-2A/farmworker ratio and LII are non-significant.

---

## 3. Final Inspections Model (linear time)

`inspections ~ time + [block]`, N = **261 obs, 44 states**. Baseline σ²_u0 =
623.02, ICC = 83.0%.

**×time interaction screen (p < .20):** none retained.

Final fixed effects:

| Parameter | β | SE | p |
|---|---:|---:|---:|
| Intercept | 27.343 | 4.225 | 0.000 ** |
| time | −0.482 | 0.565 | 0.393 |
| lii_2017_z | −2.052 | 6.655 | 0.758 |
| h2a_per_farmworker_z | 1.068 | 4.288 | 0.803 |
| dol_demand_met_pct_z | 3.163 | 4.903 | 0.519 |
| pct_flc_z | 4.414 | 4.759 | 0.354 |

σ²_u0 = 622.36 (**pseudo-R² = 0.1%**); σ²_ε = 129.19; LL = −1081.60.

**Read:** consistent with prior inspections results — the labor/DOL block
explains essentially no between-state variance in inspections. Inspection counts
are not tracked by these labor-structure covariates the way violations are.

---

## 4. Comprehensive Inspections Model (new — "every variable")

`comprehensive_inspections_model.py` enters **every** covariate constructed
anywhere in the project into a single inspections model. This is an intentional
kitchen-sink specification — many predictors share numerators/denominators and
are collinear — so it is **exploratory** and complements, not replaces, the
curated final inspections model. Time is linear (inspections convention).

**Roster (15 predictors):** `spending_2017m` (Level-1, time-varying) +
`spending_per_estab`, `SPEND_WORK`, `SPEND_APP`, `SPEND_FLC`, `SPEND_OP`,
`SPEND_AREA`, `land_area_sqmi`, `operations`, `h2a_workers`,
`workers_per_operation`, `lii_2017`, `h2a_per_farmworker`, `dol_demand_met_pct`,
`pct_flc` (all z-scored Level-2). `dol_n_cases` is excluded and `dol_workers_cert`
enters only via the ratio, matching the final-model direction.

Analytic sample = listwise-complete on all 15 → **224 obs, 38 states** (BLS
suppression is the binding constraint). Baseline (linear time) σ²_u0 = 637.72,
ICC = 84.7%.

**Full main-effects model:** σ²_u0 = 695.43, **pseudo-R² = −9.0%** — the negative
value reflects random-effects instability under 15 collinear Level-2 predictors on
38 states, a known kitchen-sink artifact. The only significant main effect is
`pct_flc_z` (β = −11.28, p = .023).

**×time interaction screen (p < .20 retained):** `SPEND_FLC_z` (p = .160),
`h2a_workers_z` (p = .030), `workers_per_operation_z` (p = .075).

**Augmented model** (main effects + 3 retained interactions): σ²_u0 = 470.65,
**pseudo-R² = 26.2%**; LL = −870.06. `pct_flc_z` remains the only significant main
effect (β = −10.85, p = .019); the retained interactions are individually
non-significant once entered jointly, so the variance reduction is driven by the
joint interaction block rather than any single robust term.

**Caveat:** with 16 collinear predictors on 38 states, coefficients are unstable
and should not be interpreted individually. The value of this model is descriptive
coverage, not inference. The curated `final_labor_inspections_model.py` remains
the model of record for inspections.

---

## 5. Data lineage for the new variable

```
dol_var1_workers_by_state_annual.csv  → workers_certified (2011–2019 mean, excl. 2013)  [numerator]
bls_oews_panel.csv (OCC 45-2092)      → tot_emp farmworkers (2011 snapshot)              [denominator]
                                        ─────────────────────────────────────────────
                                        h2a_per_farmworker  (z-scored before entry)
```

No new raw data files were added. `bls_oews_panel.csv` (already in the repo) is
now also read by the two final models, not just the BLS spending scripts.
