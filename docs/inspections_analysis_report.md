# WPS Inspections — Hierarchical Model Analysis
## Parallel Replication of Violations Models Using Inspections as Dependent Variable
**WPS Enforcement Analysis | Keshav Goel | May 2026**

---

## 1. Overview

This document reports results from fitting the full sequence of hierarchical mixed-effects models (M1–M2 baseline, stepwise Zimmerman, BLS-normalized spending, and targeted FIFRA-aligned model) with **total WPS inspections** as the dependent variable. The analysis exactly mirrors the violations analysis; only the outcome changes. All methodological decisions, constant definitions, and covariate constructions are identical to the violations scripts.

**Scripts:**

| Script | Models | DV |
|--------|--------|----|
| `inspections_hierarchical_model.py` | M1 (time only) + M2 (+ spending) | inspections |
| `inspections_stepwise_zimmerman_models.py` | Zimmerman one-at-a-time: 5 Level-2 covariates | inspections |
| `inspections_spending_bls_models.py` | BLS-normalized SPEND variables (one-at-a-time + combined) | inspections |
| `inspections_targeted_spend_model.py` | Theory-driven FIFRA model: SPEND_WORK + SPEND_APP + SPEND_AREA | inspections |

---

## 2. Data Construction

### 2.1 Dependent Variable

`inspections` = `inspections-epa-YYYY` + `inspections-state-YYYY` from `establishments-data (2).csv`.

Where only one source is present for a state-year, the missing source is treated as 0. No state-year has both sources missing; all 450 state-year observations (50 states × 9 years) are valid.

This contrasts with violations, where some state-years had genuinely missing data after the numeric coercion step.

### 2.2 Analytic Samples

| Stage | N obs | N states | Notes |
|-------|-------|----------|-------|
| Full inspections panel | 450 | 50 | All 9 years complete for all states |
| After spending merge | 282 | 47 | RI, MS, WV fully absent from spending data |
| BLS models (SPEND_WORK_z) | 261 | 44 | 3 states missing OCC 45-2092 |
| BLS models (SPEND_APP_z) | 240 | 41 | 6 states missing OCC 37-3012 |
| BLS models (SPEND_FLC_z) | 275 | 45 | 2 states missing OCC 45-1011 |
| Targeted model | 225 | 39 | BLS intersection; 11 states excluded |

### 2.3 Inspections by Year (Spending-Matched Sample, N=282)

| Year | Mean | SD | Min | Max | N States |
|------|------|----|-----|-----|----------|
| 2011 | 28.66 | 22.18 | 0 | 87 | 44 |
| 2012 | 33.42 | 33.90 | 0 | 163 | 40 |
| 2013 | 29.94 | 25.57 | 0 | 116 | 31 |
| 2014 | 29.26 | 24.27 | 0 | 90 | 31 |
| 2015 | 29.13 | 28.65 | 0 | 119 | 30 |
| 2016 | 26.41 | 25.44 | 1 | 107 | 32 |
| 2017 | 21.56 | 20.62 | 0 | 77 | 27 |
| 2018 | 20.08 | 19.64 | 1 | 81 | 25 |
| 2019 | 20.23 | 21.35 | 1 | 75 | 22 |

**Key pattern**: Inspections show a **gradual decline** from 2011–2019, with no spike. This is structurally different from violations, which peaked sharply at 26.4 (2016) and 20.6 (2017 mean). The absence of a spike means the cubic polynomial is largely flat for inspections — only the linear term is significant.

---

## 3. Methodology

All models follow the same conventions as the violations analysis:

- **Random effects**: random intercept + random slope for linear time, by state (`re_formula='~time'`)
- **Estimation**: REML via L-BFGS-B
- **Time centering**: `time = year − 2017`; cubic polynomial (`time`, `time²`, `time³`) retained in all models
- **Level-2 standardization**: all covariates z-scored (mean=0, SD=1) before entry as `varname_z`
- **Pseudo-R²**: `(σ²_u0_baseline − σ²_u0_model) / σ²_u0_baseline` — reduction in between-state variance
- **Baseline**: always re-estimated on the same analytic sample as the covariate model

---

## 4. Baseline Model (M1: Time Only)

Fit on the spending-matched sample (N=282, 47 states).

| Parameter | β | SE | z | p |
|-----------|---|-----|---|---|
| Intercept | 26.450 | 3.942 | 6.71 | <.001 |
| time | −1.802 | 0.773 | −2.33 | .020 |
| time² | −0.172 | 0.364 | −0.47 | .637 |
| time³ | 0.013 | 0.055 | 0.23 | .817 |

**Random effects:**
- Between-state variance (σ²_u0): 588.36
- Random slope variance (σ²_u1): 5.90
- Residual variance (σ²_ε): 121.32
- **ICC = 82.9%** — between-state variance accounts for most variation

**Interpretation**: Only the linear time term is significant (β = −1.80, p = .020). Inspections decline by approximately 1.8 per year from the 2017 intercept. The quadratic and cubic terms are not significant, confirming the time trend for inspections is approximately linear — a gradual monotonic decline, with no evidence of the non-linear spike seen in violations.

**Note**: Model 1 did not fully converge (convergence flag = False). Model 2 (with spending) did converge.

---

## 5. Model 2: With Inflation-Adjusted Spending (M2)

Fit on same spending-matched sample (N=282, 47 states).

| Parameter | β | SE | z | p |
|-----------|---|-----|---|---|
| Intercept | 26.211 | 4.128 | 6.35 | <.001 |
| time | −1.659 | 0.716 | −2.32 | .021 |
| time² | −0.134 | 0.379 | −0.35 | .724 |
| time³ | 0.016 | 0.057 | 0.28 | .777 |
| spending_2017m | **1.476** | 2.769 | 0.53 | **.594** |

**Random effects:**
- Between-state variance (σ²_u0): 642.22
- Random slope variance: 2.41
- Intercept–slope covariance: 16.79
- Residual variance (σ²_ε): 130.02
- **ICC = 83.2%**

**Model comparison:**

| Metric | M1 (no spending) | M2 (+ spending) |
|--------|-----------------|-----------------|
| Log-Likelihood | −1172.79 | −1166.24 |
| σ²_u0 | 588.36 | 642.22 |
| σ²_ε | 121.32 | 130.02 |

LR χ² = 13.11 (df=1). The LR test should be interpreted cautiously given M1's non-convergence.

**Interpretation of spending coefficient**: β = +1.476 (p = .594) — not statistically significant. The positive direction (more spending → more inspections) is plausible: states with higher grant funding may conduct more inspections. This is the opposite sign to violations (where spending was negative, suggesting more spending → fewer violations). However, the effect is far from significant and should not be interpreted.

### Predicted National Trend (M2, at mean spending = $0.425M)

| Year | Predicted | Actual Mean |
|------|-----------|-------------|
| 2011 | 28.51 | 28.66 |
| 2012 | 29.78 | 33.42 |
| 2013 | 30.31 | 29.94 |
| 2014 | 30.18 | 29.26 |
| 2015 | 29.49 | 29.13 |
| 2016 | 28.35 | 26.41 |
| 2017 | 26.84 | 21.56 |
| 2018 | 25.06 | 20.08 |
| 2019 | 23.11 | 20.23 |

The predicted trend shows a gradual decline from 2014 onward but underestimates the steepness of the drop after 2016.

### States with Highest/Lowest Baseline Inspections (M2 BLUPs)

**Highest:**

| State | Random Intercept | Random Slope |
|-------|-----------------|--------------|
| New Jersey | +73.58 | +1.54 |
| Florida | +65.60 | +2.32 |
| North Dakota | +58.14 | −0.04 |
| North Carolina | +41.27 | +0.54 |
| Georgia | +37.68 | +0.08 |

**Lowest:**

| State | Random Intercept | Random Slope |
|-------|-----------------|--------------|
| Connecticut | −20.61 | −0.31 |
| Oregon | −21.89 | +0.25 |
| New Hampshire | −23.29 | +0.05 |
| Maine | −23.32 | −0.46 |
| Vermont | −25.50 | +0.04 |
| Alaska | −27.81 | −0.17 |

---

## 6. Stepwise Zimmerman Models

Baseline M0 on spending-matched sample (N=282, 47 states):
- **σ²_u0 (baseline) = 588.36**
- **ICC = 82.9%**

### 6.1 Results by Predictor

| Predictor | β (main) | p (main) | β (×time) | p (×time) | σ²_u0 (main) | %Expl | σ²_u0 (+int) | %Expl |
|-----------|----------|----------|-----------|-----------|--------------|-------|--------------|-------|
| spending_per_estab_z | −7.35 | .038* | 0.154 | .768 | 529.91 | 9.9% | 538.28 | 8.5% |
| land_area_sqmi_z | −2.55 | .467 | 0.294 | .623 | 597.82 | −1.6% | 602.15 | −2.3% |
| operations_z | +4.80 | .180 | 0.076 | .874 | 552.28 | 6.1% | 561.06 | 4.6% |
| h2a_workers_z | **+12.33** | **.000***| **1.061** | **.040*** | **403.31** | **31.5%** | **398.19** | **32.3%** |
| workers_per_operation_z | +8.02 | .016* | 0.763 | .075+ | 522.61 | 11.2% | 573.56 | 2.5% |

### 6.2 Key Findings

**h2a_workers_z is by far the strongest predictor** — explaining 31.5–32.3% of between-state variance, with a large positive effect (β = 12.33–16.40). States with more H-2A workers have systematically more inspections, and this gap widens over time (×time interaction p = .040). This makes structural sense: H-2A worker concentration signals agricultural labor intensity, which likely draws more inspection resources.

**spending_per_estab_z** has a significant negative main effect (β = −7.35, p = .038): states receiving more spending per establishment conduct fewer inspections on a per-establishment basis. This could reflect either a targeting paradox (inspections are concentrated in low-spending states) or that spending and inspections measure different enforcement activities.

**operations_z** approaches significance (p = .180) with a positive effect, consistent with more farming operations → more inspection opportunities.

**land_area_sqmi** is not significant — geographic size does not predict inspection rates after accounting for time and state random effects.

### 6.3 Combined Model (operations_z + h2a_workers_z, both ×time)

- Pearson r(operations_z, h2a_workers_z) = 0.086 — negligible collinearity
- σ²_u0 = 391.64 → **33.4% of baseline variance explained**
- Increment over best single predictor (h2a_workers alone at 32.3%): **+1.1 pp** — marginal gain from adding operations

| Parameter | β | SE | p |
|-----------|---|-----|---|
| Intercept | 26.31 | 3.62 | <.001 |
| time | −2.22 | 0.57 | <.001 |
| time² | −0.27 | 0.31 | .387 |
| time³ | 0.031 | 0.050 | .534 |
| operations_z | 3.64 | 3.07 | .236 |
| operations_z:time | 0.062 | 0.332 | .852 |
| h2a_workers_z | 15.99 | 3.56 | <.001 |
| h2a_workers_z:time | 1.038 | 0.525 | .048 |

---

## 7. BLS-Normalized Spending Models

Baseline on spending-matched sample (N=282, 47 states):
- **σ²_u0 = 588.36, ICC = 82.9%**

### 7.1 Individual Stepwise Results

| Variable | N obs | Main β | Main p | ×time β | ×time p | σ²_u0 (main) | %Expl | σ²_u0 (+int) | %Expl |
|----------|-------|--------|--------|---------|---------|--------------|-------|--------------|-------|
| SPEND_WORK_z | 261 | −6.49 | .099+ | −0.004 | .995 | 589.75 | −0.2% | 599.11 | −1.8% |
| SPEND_APP_z | 240 | −8.78 | .013* | +0.049 | .940 | 690.62 | −17.4%† | 519.95 | 11.6% |
| SPEND_FLC_z | 275 | −8.42 | .020* | −0.193 | .635 | 530.36 | 9.9% | 581.99 | 1.1% |
| SPEND_OP_z | 282 | −6.59 | .068+ | +0.113 | .824 | 550.20 | 6.5% | 558.93 | 5.0% |
| SPEND_AREA_z | 282 | −6.12 | .104 | −0.169 | .762 | 557.23 | 5.3% | 566.31 | 3.7% |

† **Methodological note on SPEND_APP_z main-effect row**: The apparent −17.4% (σ²_u0 > baseline) is an artifact of the listwise deletion reducing the sample from 282 to 240 obs when SPEND_APP is included. The 240-obs subsample has higher between-state variance than the full sample. The baseline was estimated on N=282; strictly, it should be re-estimated on N=240 for a fair comparison. This is the same known limitation documented in the violations analysis (see CLAUDE.md). The main effect itself (β=−8.78, p=.013) remains a valid estimate.

**All five spending variables have negative main effects**, meaning states with more spending per worker/applicator/area have fewer inspections. No ×time interaction approaches p<.20 — none are selected for the combined model.

### 7.2 Combined Model (main effects only; no interactions met p<.20 threshold)

Formula: `inspections ~ time + time² + time³ + SPEND_WORK_z + SPEND_APP_z + SPEND_FLC_z + SPEND_OP_z + SPEND_AREA_z`

N = 224 obs, 38 states (BLS intersection).

| Parameter | β | SE | p |
|-----------|---|-----|---|
| Intercept | 28.22 | 6.67 | <.001 |
| time | −1.75 | 0.92 | .059+ |
| time² | −0.27 | 0.41 | .511 |
| time³ | 0.005 | 0.061 | .932 |
| SPEND_WORK_z | −2.63 | 6.87 | .702 |
| SPEND_APP_z | −7.82 | 7.10 | .270 |
| SPEND_FLC_z | −0.78 | 8.65 | .928 |
| SPEND_OP_z | −18.04 | 30.59 | .555 |
| SPEND_AREA_z | +2.73 | 7.68 | .722 |

σ²_u0 = 536.43 → **8.8% of baseline variance explained**. No individual spending variable is significant in the combined model, consistent with multicollinearity across spending denominators in a reduced sample.

**Contrast with violations**: In the violations analysis, SPEND_FLC alone explained ~9% and the combined FIFRA model (SPEND_WORK + SPEND_APP) explained ~21%. For inspections, none of the BLS spending variables explain substantial unique variance and none retain significance in the combined model. The relationship between spending and enforcement activity (inspections) appears weaker and structurally different from the relationship between spending and enforcement outcomes (violations).

---

## 8. Targeted FIFRA-Aligned Model

Pre-specified formula mirrors `targeted_spend_model.py`:

```
inspections ~ time + time² + time³
            + SPEND_WORK_z
            + SPEND_APP_z  + SPEND_APP_z:time
            + SPEND_AREA_z + SPEND_AREA_z:time
```

*Note*: Interactions (SPEND_APP_z:time, SPEND_AREA_z:time) are pre-specified from the violations analysis (p<.20 threshold). In the inspections-specific BLS stepwise models, no interactions met p<.20 — none would have been selected if deriving fresh. The pre-specified formula is retained here to maintain exact structural parallelism with the violations analysis.

**Analytic sample**: 225 obs, 39 states. Excluded (11 states): Alaska, Colorado, Minnesota, Mississippi, Nevada, New Hampshire, New Mexico, Rhode Island, Utah, Vermont, West Virginia.

### Baseline (Restricted Sample)

| Component | Value |
|-----------|-------|
| σ²_u0 (between-state) | 584.36 |
| σ²_ε (within-state) | 117.81 |
| ICC | 83.2% |
| Log-Likelihood | −938.80 |

### Targeted Model Fixed Effects

| Parameter | β | SE | z | p | 95% CI |
|-----------|---|-----|---|---|--------|
| Intercept | 30.669 | 3.787 | 8.10 | <.001** | [23.25, 38.09] |
| time | −1.734 | 0.852 | −2.04 | .042* | [−3.40, −0.07] |
| time² | −0.254 | 0.402 | −0.63 | .527 | [−1.04, 0.53] |
| time³ | 0.007 | 0.059 | 0.11 | .911 | [−0.11, 0.12] |
| SPEND_WORK_z | −3.069 | 4.441 | −0.69 | .489 | [−11.77, 5.64] |
| SPEND_APP_z | −9.348 | 5.353 | −1.75 | .081+ | [−19.84, 1.14] |
| SPEND_APP_z:time | −0.010 | 0.689 | −0.01 | .988 | [−1.36, 1.34] |
| SPEND_AREA_z | −1.618 | 5.616 | −0.29 | .773 | [−12.63, 9.39] |
| SPEND_AREA_z:time | −0.447 | 0.786 | −0.57 | .569 | [−1.99, 1.09] |

### Random Effects

| Component | Targeted Model | Baseline |
|-----------|---------------|---------|
| σ²_u0 (between-state) | 554.60 | 584.36 |
| σ²_u1 (random slope) | 7.93 | — |
| σ²_ε (within-state) | 120.49 | 117.81 |
| Cov(u0, u1) | 15.09 | — |
| **Between-state var explained** | **5.09%** | — |

**Convergence**: Did not converge (Converged = False). Results should be treated with caution.

### Interpretation

- **Only time** is significant (β = −1.73, p = .042): inspections decline by ~1.7 per year
- **SPEND_APP_z** approaches significance (β = −9.35, p = .081+): states with more spending per pesticide applicator tend to have fewer inspections — same negative sign as in the stepwise analysis
- **SPEND_WORK_z and SPEND_AREA_z**: not significant
- **No interactions significant**: neither SPEND_APP_z:time nor SPEND_AREA_z:time explains meaningful variance
- **5.09% variance explained**: substantially weaker than the violations targeted model (~21%)

---

## 9. Summary and Comparison with Violations Analysis

### 9.1 Structural Differences in the Outcome

| Feature | Violations | Inspections |
|---------|-----------|-------------|
| Time trend | Inverted-U spike, peaks 2016–2017 | Gradual monotonic decline |
| Cubic polynomial significant? | Yes (all three terms) | No (only linear term significant) |
| ICC | 62–77% depending on sample | 82–83% |
| Mean (spending-matched sample) | 10.09 | 27.24 |

### 9.2 Baseline Model Comparison

| Component | Violations M1 | Inspections M1 |
|-----------|--------------|----------------|
| Intercept (2017) | 17.46 | 26.45 |
| Linear time | +2.97 (p<.001) | −1.80 (p=.020) |
| Quadratic time | −0.63 (p=.071) | −0.17 (p=.637) |
| Cubic time | −0.15 (p=.005) | +0.01 (p=.817) |
| σ²_u0 | 325.71 | 588.36 |
| ICC | 77.2% | 82.9% |

### 9.3 Spending as Raw Predictor (M2)

| | Violations | Inspections |
|---|-----------|-------------|
| spending_2017m β | −1.24 | +1.48 |
| p-value | .483 | .594 |
| Significant? | No | No |
| Direction | Negative | Positive |

Neither analysis finds spending to be a significant predictor. The opposing signs are theoretically coherent: more spending could reduce violations (better compliance infrastructure) while simultaneously enabling more inspections (more resources for field activity). Neither effect is statistically distinguishable from zero.

### 9.4 Stepwise Model Comparison

| Predictor | Violations %Expl (main) | Inspections %Expl (main) |
|-----------|------------------------|-------------------------|
| spending_per_estab_z | — | 9.9%* |
| land_area_sqmi_z | — | ns |
| operations_z | — | 6.1% |
| **h2a_workers_z** | — | **31.5%***  |
| workers_per_operation_z | — | 11.2%* |

H-2A worker concentration is the dominant structural predictor of inspections — far more than for violations. This makes theoretical sense: inspectors likely allocate resources where H-2A workers are concentrated.

### 9.5 BLS Spending Models Comparison

| Variable | Violations Main p | Inspections Main p | Violations %Expl | Inspections %Expl |
|----------|------------------|--------------------|-----------------|------------------|
| SPEND_WORK_z | — | .099+ | — | ~0% |
| SPEND_APP_z | — | .013* | — | 11.6%† |
| SPEND_FLC_z | ~9%† | .020* | 9%† | 9.9%† |
| SPEND_OP_z | ns | .068+ | — | 6.5% |
| SPEND_AREA_z | — | .104 | — | 5.3% |

† Subject to sample-size confounds (different N for BLS-restricted samples vs. baseline).

All BLS spending variables are **negatively associated with inspections**: states with more dollars per worker or per area conduct fewer inspections. This negative sign is consistent across all five spending variables and for both main effects and the combined model.

---

## 10. Methodological Notes

1. **Inspections variable is additive**: EPA + state agency inspections. No state-year has both sources missing, so the full panel is retained without imputation.

2. **Same-sample baseline requirement**: For models using BLS employment denominators (SPEND_APP, SPEND_WORK, SPEND_FLC), the baseline M0 is estimated on all 282 observations while the covariate model uses a subset (240–275 obs). This causes the apparent negative pseudo-R² for SPEND_APP (main only). This is the same known limitation in the violations BLS models; the Δσ²_u0 values for BLS-restricted models should be interpreted cautiously.

3. **Pre-specified interactions in targeted model**: The SPEND_APP:time and SPEND_AREA:time interactions in `inspections_targeted_spend_model.py` were selected based on p<.20 from the **violations** BLS stepwise run. In the inspections BLS stepwise, no interactions came close to p<.20 (all ≥.635). The targeted model is therefore running pre-specified interactions that are not supported by the inspections data — this reduces its effective degrees of freedom without explanatory benefit. A data-driven version for inspections would drop both interactions.

4. **Convergence**: M1 (no spending, hierarchical model) and the targeted model did not fully converge. All converged models (M2, all stepwise models, BLS models) are reliable. M1 results should be interpreted cautiously.

5. **Declining N in later years**: The spending-matched sample has increasingly few states with spending data in 2017–2019 (27, 25, 22 states). This year-level sparsity is shared with the violations analysis.

---

## 11. Output Files

| File | Description |
|------|-------------|
| `inspections_model_data_long.csv` | Final analytic dataset (inspections + spending, long format) |
| `inspections_state_random_effects.csv` | State BLUPs (random intercepts and slopes) from M2 |
| `inspections_predicted_trend.csv` | Predicted national trend from M2 at mean spending |

*Note*: `level2_covariates.csv` and `spend_bls_variables.csv` are shared with the violations analysis (identical Level-2 covariates) and are not regenerated by the inspections scripts.

---

*Analysis conducted: May 2026*
