# Results — `comprehensive_inspections_model.py`

**Outcome:** total inspections (EPA + state) per state-year, 2011–2019.
**Time specification:** linear time only (`time`), inspections convention.
**Estimator:** `MixedLM`, random intercept + random slope for time, REML (`lbfgs`).
**Design:** intentional **kitchen-sink / "every variable"** specification —
every covariate constructed anywhere in the project is entered into one
inspections model. Many predictors share numerators/denominators and are
collinear, so this is **exploratory** and complements (does not replace) the
curated `final_labor_inspections_model.py`.

---

## Predictor roster (15 total)

**Level-1 (time-varying):** `spending_2017m` (inflation-adjusted STAG obligations).

**Level-2 (time-invariant, z-scored):**

| Group | Variables |
|---|---|
| Spending / enforcement intensity | `spending_per_estab`, `SPEND_WORK`, `SPEND_APP`, `SPEND_FLC`, `SPEND_OP`, `SPEND_AREA` |
| Scale / exposure | `land_area_sqmi`, `operations`, `h2a_workers`, `workers_per_operation` |
| Labor structure | `lii_2017`, `h2a_per_farmworker`, `dol_demand_met_pct`, `pct_flc` |

`dol_n_cases` is excluded and `dol_workers_cert` enters only as the
`h2a_per_farmworker` ratio, matching the final-model direction (2026-07).

### Covariate coverage (states non-missing before listwise deletion)

| Covariate | n | Covariate | n |
|---|---:|---|---:|
| spending_per_estab | 47 | operations | 50 |
| SPEND_WORK | 44 | h2a_workers | 50 |
| SPEND_APP | 41 | workers_per_operation | 50 |
| SPEND_FLC | 45 | lii_2017 | 50 |
| SPEND_OP | 47 | h2a_per_farmworker | 47 |
| SPEND_AREA | 47 | dol_demand_met_pct | 50 |
| land_area_sqmi | 50 | pct_flc | 50 |

**`SPEND_APP` (BLS OCC 37-3012, n=41) is the binding constraint.** Listwise
deletion on all 15 predictors →  **224 obs, 38 states**.

---

## Baseline (linear time, analytic sample)

| Parameter | β | SE | p |
|---|---:|---:|---:|
| Intercept | 31.094 | 4.665 | 0.000 ** |
| time | −0.361 | 0.805 | 0.654 |

Baseline σ²_u0 = **637.72**, ICC = **84.7%**.

---

## Full main-effects model (every covariate at once)

Selected fixed effects (all 15 entered; full table in script output):

| Parameter | β | SE | p |
|---|---:|---:|---:|
| pct_flc_z | **−11.279** | 4.966 | **0.023** * |
| lii_2017_z | −7.291 | 6.658 | 0.273 |
| h2a_workers_z | 22.188 | 24.454 | 0.364 |
| SPEND_WORK_z | −9.462 | 8.427 | 0.261 |
| spending_per_estab_z | −21.481 | 53.235 | 0.687 |
| SPEND_OP_z | 8.044 | 60.503 | 0.894 |
| … (all others) | | | n.s. |

- σ²_u0 = 695.43 → **pseudo-R² = −9.0%** (see note below)
- σ²_ε = 127.61
- Log-Likelihood = −874.42

> **Negative pseudo-R² is a collinearity artifact.** With 14 collinear Level-2
> predictors on 38 states, the random-intercept variance is estimated *higher*
> than in the parsimonious baseline — a symptom of an over-parameterized
> mixed model, not evidence that covariates increase between-state variance.
> The huge standard errors (e.g. SPEND_OP SE ≈ 60) confirm the instability.
> Only `pct_flc_z` is significant.

---

## ×time interaction screen (retain if p < .20)

| Covariate | ×time β | p | Decision |
|---|---:|---:|---|
| **SPEND_FLC_z** | −0.871 | 0.160 | **INCLUDE** |
| **h2a_workers_z** | 1.171 | 0.030 | **INCLUDE** |
| **workers_per_operation_z** | 1.152 | 0.075 | **INCLUDE** |
| SPEND_AREA_z | −0.732 | 0.216 | exclude |
| land_area_sqmi_z | 1.203 | 0.235 | exclude |
| (all others) | | ≥ .29 | exclude |

---

## Augmented model (main effects + 3 retained ×time interactions)

| Parameter | β | SE | p |
|---|---:|---:|---:|
| pct_flc_z | **−10.847** | 4.632 | **0.019** * |
| h2a_workers_z | 24.676 | 23.769 | 0.299 |
| SPEND_FLC_z:time | −0.545 | 0.536 | 0.309 |
| h2a_workers_z:time | 1.481 | 1.638 | 0.366 |
| workers_per_operation_z:time | −0.519 | 1.760 | 0.768 |
| … (all others) | | | n.s. |

- σ²_u0 = 470.65 → **pseudo-R² = 26.2%** of between-state variance
- σ²_ε = 134.04
- Log-Likelihood = −870.06

*Sig: ** p<.01, * p<.05, + p<.10*

---

## Interpretation

- **Only `pct_flc` is a stable, significant predictor** across both the full and
  augmented models (β ≈ −11, p ≈ .02). Interestingly the sign is **negative** for
  inspections — the opposite of its strong *positive* effect on violations — but
  given the collinearity here this single coefficient should be read cautiously.
- **The 26.2% variance reduction in the augmented model is driven by the joint
  ×time interaction block, not by any individual term** — all three retained
  interactions are non-significant once entered together, and the main-effects
  model alone has negative pseudo-R². This is classic over-fitting on 38 states.
- **Coefficients should not be interpreted individually.** Standard errors are
  enormous (SPEND_OP, spending_per_estab, land area all have SEs in the tens),
  a direct consequence of entering multiple spending ratios that share the same
  numerator plus overlapping scale variables.

**Bottom line:** this model is for descriptive coverage — it confirms that once
*everything* is thrown in, `pct_flc` is the only labor variable that pushes on
inspections at all, and even the apparent 26% variance reduction is an unstable
interaction-block effect. The curated `final_labor_inspections_model.py`
(pseudo-R² ≈ 0%, all terms clean and n.s.) remains the model of record for
inspections; this kitchen-sink run does not overturn that conclusion.
