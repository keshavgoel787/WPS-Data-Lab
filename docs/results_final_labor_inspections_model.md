# Results — `final_labor_inspections_model.py`

**Outcome:** total inspections (EPA + state) per state-year, 2011–2019.
**Time specification:** **linear time only** (`time`), `time = year − 2017` — PI
direction; the inspections series carries only a linear trend (the cubic is a
violations-only feature).
**Estimator:** `MixedLM`, random intercept + random slope for time, REML (`lbfgs`).
**Run date:** 2026-07 revision. Structural mirror of the violations final model.

---

## Sample construction

| Step | Obs | States |
|---|---:|---:|
| Base (inspections × spending) | 282 | 47 |
| After merging Level-2 block | 282 | 47 |
| **Final analytic sample** (listwise on 4 covariates) | **261** | **44** |

Same covariate block and same 44-state listwise sample as the violations model
(the DV differs, so obs counts differ: 261 vs. 235). The binding constraint is
again the BLS OCC 45-2092 denominator in `h2a_per_farmworker` (47 non-missing).

**Covariate block (z-scored):** `lii_2017_z`, `h2a_per_farmworker_z`,
`dol_demand_met_pct_z`, `pct_flc_z` (identical standardization to the violations
model).

---

## Baseline (linear time, re-estimated on the analytic sample)

| Parameter | β | SE | p |
|---|---:|---:|---:|
| Intercept | 27.786 | 4.138 | 0.000 ** |
| time | −0.525 | 0.557 | 0.346 |

Baseline σ²_u0 = **623.02**, ICC = **83.0%**. Note even the linear time slope is
non-significant (p = .35) — inspections have no strong secular trend, unlike the
violations spike.

---

## ×time interaction screen (retain if p < .20)

| Covariate | ×time β | p | Decision |
|---|---:|---:|---|
| lii_2017_z | 0.605 | 0.439 | exclude |
| h2a_per_farmworker_z | −0.475 | 0.358 | exclude |
| dol_demand_met_pct_z | −0.048 | 0.924 | exclude |
| pct_flc_z | 0.576 | 0.307 | exclude |

**None** retained — no covariate has a time-varying effect on inspections.

---

## Final model

`inspections ~ time + lii_2017_z + h2a_per_farmworker_z + dol_demand_met_pct_z + pct_flc_z`

| Parameter | β | SE | p |
|---|---:|---:|---:|
| Intercept | 27.343 | 4.225 | 0.000 ** |
| time | −0.482 | 0.565 | 0.393 |
| lii_2017_z | −2.052 | 6.655 | 0.758 |
| h2a_per_farmworker_z | 1.068 | 4.288 | 0.803 |
| dol_demand_met_pct_z | 3.163 | 4.903 | 0.519 |
| pct_flc_z | 4.414 | 4.759 | 0.354 |

- σ²_u0 = 622.36 → **pseudo-R² = 0.1%** of between-state variance vs. baseline
- σ²_ε = 129.19
- Log-Likelihood = −1081.60

*Sig: ** p<.01, * p<.05, + p<.10*

---

## Interpretation

- **The labor/DOL block explains essentially nothing** in inspections
  (pseudo-R² = 0.1%). Every covariate is non-significant, and no ×time
  interaction survives the p < .20 screen.
- This is the sharp contrast with the violations model: the exact same four
  covariates explain 15.2% of between-state variance in **violations** (driven by
  `pct_flc`) but ≈0% in **inspections**. `pct_flc` in particular flips from a
  strong predictor of violations to a null one for inspections.
- **Substantive read:** inspection *counts* are an administrative/resource
  process not tracked by state labor structure, whereas violation *counts* are.
  Inspections are also flatter over time (baseline time slope n.s.), which is why
  the model is restricted to linear time.

**Bottom line:** the revised block does not predict inspections. The result is a
useful negative control — it shows the violations findings are not an artifact of
the covariate block being mechanically correlated with any enforcement count.
