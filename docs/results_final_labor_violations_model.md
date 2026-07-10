# Results — `final_labor_violations_model.py`

**Outcome:** total WPS violations per state-year, 2011–2019.
**Time specification:** cubic polynomial (`time + time2 + time3`), `time = year − 2017`.
**Estimator:** `MixedLM`, random intercept + random slope for time, REML (`lbfgs`).
**Run date:** 2026-07 revision.

---

## Sample construction

| Step | Obs | States |
|---|---:|---:|
| Base (violations × spending) | 250 | 47 |
| After merging Level-2 block | 250 | 47 |
| **Final analytic sample** (listwise on 4 covariates) | **235** | **44** |

The binding constraint is `h2a_per_farmworker`: its BLS OCC 45-2092 denominator
is suppressed for 3 states (47 non-missing), and combined with the violations DV
coverage the listwise sample lands at 44 states.

**Covariate block (z-scored):** `lii_2017_z`, `h2a_per_farmworker_z`,
`dol_demand_met_pct_z`, `pct_flc_z`.

| Covariate | n | mean | SD |
|---|---:|---:|---:|
| lii_2017 | 50 | 0.000 | 1.292 |
| h2a_per_farmworker | 47 | 3.282 | 3.651 |
| dol_demand_met_pct | 50 | 94.881 | 3.252 |
| pct_flc | 50 | 25.918 | 23.367 |

---

## Baseline (time-only, re-estimated on the analytic sample)

| Parameter | β | SE | p |
|---|---:|---:|---:|
| Intercept | 18.084 | 3.460 | 0.000 ** |
| time | 3.061 | 0.829 | 0.000 ** |
| time² | −0.658 | 0.352 | 0.061 + |
| time³ | −0.152 | 0.053 | 0.004 ** |

Baseline σ²_u0 = **349.07**, ICC = **78.6%** (≈4/5 of variance is between states —
the HLM is justified).

---

## ×time interaction screen (retain if p < .20)

| Covariate | ×time β | p | Decision |
|---|---:|---:|---|
| lii_2017_z | 0.115 | 0.899 | exclude |
| h2a_per_farmworker_z | 0.419 | 0.458 | exclude |
| dol_demand_met_pct_z | −0.019 | 0.974 | exclude |
| **pct_flc_z** | **1.441** | **0.013** | **INCLUDE** |

Only the Farm-Labor-Contractor share has a time-varying effect.

---

## Final model

`violations ~ time + time2 + time3 + lii_2017_z + h2a_per_farmworker_z + dol_demand_met_pct_z + pct_flc_z + pct_flc_z:time`

| Parameter | β | SE | p |
|---|---:|---:|---:|
| Intercept | 17.807 | 3.216 | 0.000 ** |
| time | 3.048 | 0.804 | 0.000 ** |
| time² | −0.683 | 0.355 | 0.054 + |
| time³ | −0.156 | 0.054 | 0.004 ** |
| lii_2017_z | −0.454 | 1.739 | 0.794 |
| h2a_per_farmworker_z | −0.923 | 1.067 | 0.387 |
| dol_demand_met_pct_z | 1.959 | 1.096 | 0.074 + |
| **pct_flc_z** | **10.392** | 3.347 | **0.002** ** |
| **pct_flc_z:time** | **1.432** | 0.585 | **0.014** * |

- σ²_u0 = 295.99 → **pseudo-R² = 15.2%** of between-state variance vs. baseline
- σ²_ε = 97.91
- Log-Likelihood = −906.86

*Sig: ** p<.01, * p<.05, + p<.10*

---

## Interpretation

- **`pct_flc` is the dominant labor predictor.** A 1-SD increase in the Farm-Labor-
  Contractor certification share is associated with ≈10.4 more violations at 2017,
  and the positive `pct_flc_z:time` interaction (β = 1.43) means high-FLC states
  diverge further upward over the study window. This is the whole story of the
  block's explanatory power.
- **`dol_demand_met_pct` is marginal (+, p = .074):** states whose H-2A demand is
  more fully met trend toward slightly more violations, but the effect is weak.
- **The new `h2a_per_farmworker` ratio and `lii_2017` are non-significant.**
  Normalizing certified H-2A visas by the farmworker workforce does not, on its
  own, predict violation levels once FLC share is in the model.
- The cubic time terms remain jointly important (linear and cubic both p < .01),
  consistent with the asymmetric 2016–2017 spike that motivated the polynomial.

**Bottom line:** the revised block explains 15.2% of between-state variance in
violations, essentially all of it carried by `pct_flc`.
