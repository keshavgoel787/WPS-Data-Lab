# Stepwise Variable-by-Variable Models — WPS Enforcement Analysis
**Zimmerman/HLM Approach | Keshav Goel | May 2026**

---

## Overview

This document reports results from the variable-by-variable model-building sequence following the Zimmerman (2000) / Raudenbush & Bryk (2002) HLM approach. Each predictor is entered one at a time into the unconditional model, and the reduction in between-state variance is tracked at each step. A combined model testing the two strongest predictors jointly is included to assess collinearity.

**Sample:** 250 observations, 47 states, 2011–2019  
**Outcome:** WPS violations per state-year  
**Random effects:** Random intercept + random slope for time, by state (REML)  
**Time variable:** Centered on 2017 (time = year − 2017)

---

## Team Analytical Decisions (from email thread)

The following decisions were reached before running these models and are reflected in the variable sequence below.

**Spending operationalization: dollars per WPS-regulated agricultural establishment**

Stephane's rationale, agreed to by Joe and Kaitlyn:
- The spending variable needs to answer: *does the state have sufficient resources to cover its inspection workload?*
- Inspection workload is defined by the number of **establishments an inspector must visit**, not by the number of workers employed there
- Per-establishment funding directly matches the numerator (dollars allocated to enforcement) to the denominator (regulated units that enforcement must cover)
- Aim 3 already includes agricultural workforce size as a standalone predictor; using dollars per 1,000 workers would put the spending denominator and a separate predictor on the same underlying measure — this creates structural variance overlap, requiring VIF management and risking coefficient suppression
- Per-establishment spending does not share this structural overlap

**Land mass as a covariate** (Joe's addition, agreed by Kaitlyn):
- Visiting 100 establishments within 100 square miles is substantially easier than 100 establishments spread across 10,000 square miles
- State land area (sq miles) included as a fixed covariate

**Covariates from Yuri's data:**
- Total farming operations by state (Census of Agriculture) and H-2A workers per operation are available for the variable-by-variable sequence; added one predictor at a time per the Zimmerman approach

---

## ECHO Establishment Coverage Audit

Stephane requested confirmation that EPA ECHO WPS establishment counts are available and consistent across all 50 states for **2015–2022** before committing to per-establishment as the spending denominator. Findings below.

**Current ECHO data on hand covers 2014–2019 only.** Years 2020–2022 are not yet in the project files — this needs to be pulled before per-establishment spending can be used for any analysis extending beyond 2019.

| Year | Coverage | Notes |
|------|----------|-------|
| 2011 | **Not available** | Outside ECHO data on hand |
| 2012 | **Not available** | Outside ECHO data on hand |
| 2013 | **Not available** | Outside ECHO data on hand |
| 2014 | 48/50 states | CO and CT missing |
| 2015 | 50/50 states — complete | |
| 2016 | 50/50 states — complete | |
| 2017 | 50/50 states — complete | |
| 2018 | 50/50 states — complete | |
| 2019 | 50/50 states — complete | |
| 2020 | **Not yet pulled** | Needed if study period extends to 2022 |
| 2021 | **Not yet pulled** | Needed if study period extends to 2022 |
| 2022 | **Not yet pulled** | Needed if study period extends to 2022 |

**Action items:**
- Pull ECHO WPS establishment counts for 2020–2022 (Stephane's specific ask) — required before per-establishment spending can be used for any analysis period extending to 2022
- For **2011–2013**: substitute USDA Census of Agriculture farm counts (2012 and 2017 census values; interpolate non-census years). Must be documented in the methods section.
- For **CO and CT in 2014**: imputed from their 2015 values for this analysis (CO = 130, CT = 47)

---

## Model 0 — Unconditional Baseline

**Formula:** `violations ~ time + time² + time³ + (1 + time | state)`

#### Fixed Effects

| Parameter | β | SE | z | p |
|-----------|--:|---:|--:|--:|
| Intercept | 17.4569 | 3.2025 | 5.451 | < .001 |
| time | 2.9648 | 0.7754 | 3.824 | < .001 |
| time² | −0.6260 | 0.3464 | −1.807 | .071 |
| time³ | −0.1453 | 0.0520 | −2.794 | .005 |

#### Random Effects & Fit

| Component | Value |
|-----------|------:|
| σ²_u0 — between-state (Group Var) | 325.71 |
| σ²_u0×time Cov | 51.84 |
| σ²_u1 — time slope Var | 8.52 |
| σ²_ε — within-state (Scale) | 95.99 |
| **ICC** | **0.772** |
| Log-Likelihood | −974.99 |

**77.2% of total variance is between states**, confirming strong state-level clustering and justifying the HLM structure.

---

## Covariate Preparation Notes

All Level-2 predictors are **z-score standardized** (mean = 0, SD = 1) before entry so coefficients are directly comparable across predictors.

| Covariate | Source | Mean (raw) | SD (raw) |
|-----------|--------|-----------|---------|
| `spending_per_estab` | Mean 2017$ spending / ECHO establishment mean (2015–2019) | $5,209 | $13,949 |
| `land_area_sqmi` | US Census Bureau (total sq miles) | 75,891 | 97,066 |
| `operations` | Total farming operations, Census 2017 (from Yuri) | 40,844 | 39,314 |
| `h2a_workers` | H-2A workers per state, 2017 (from Yuri) | 3,340 | 6,562 |
| `workers_per_operation` | H-2A workers / total operations, 2017 (from Yuri) | 0.098 | 0.158 |

> **Note on dollars-per-acre:** The originally requested starting variable (spending / harvested cropland acres from USDA NASS Quick Stats) cannot be computed — that data is not currently in the project files. `spending_per_estab` is used as the primary spending variable per the team's agreed operationalization.

---

## Stepwise Model Results

For each predictor, two models are fit:
- **(a) Main effect only** — predictor entered as a fixed effect
- **(b) Main effect + linear time interaction** — tests whether the predictor moderates the linear time trend

All predictors are z-score standardized; coefficients represent the change in violations per 1 SD increase in the predictor. The key tracking metric is **% between-state variance explained** relative to M0 (σ²_u0 = 325.71).

---

### 1. Spending per ECHO Establishment

#### (a) Main effect only
**Formula:** `violations ~ time + time² + time³ + spending_per_estab_z + (1 + time | state)`

| Parameter | β | SE | p |
|-----------|--:|---:|--:|
| Intercept | 17.6270 | 3.1190 | < .001 |
| time | 2.9639 | 0.7459 | < .001 |
| time² | −0.6327 | 0.3471 | .068 |
| time³ | −0.1437 | 0.0523 | .006 |
| **spending_per_estab_z** | **−1.9799** | **1.6307** | **.225** |

| σ²_u0 | 311.30 | Δ vs. M0 | +14.41 **(4.4% explained)** |
|--------|--------|----------|---------------------------|
| σ²_ε | 97.56 | LogLik | −972.33 |

#### (b) Main effect + linear time interaction
**Formula:** `violations ~ time + time² + time³ + spending_per_estab_z + spending_per_estab_z×time + (1 + time | state)`

| Parameter | β | SE | p |
|-----------|--:|---:|--:|
| Intercept | 17.8496 | 3.1244 | < .001 |
| time | 3.0075 | 0.7471 | < .001 |
| time² | −0.6391 | 0.3477 | .066 |
| time³ | −0.1467 | 0.0524 | .005 |
| **spending_per_estab_z** | **−4.7848** | **3.0872** | **.121** |
| **spending_per_estab_z × time** | **−0.7645** | **0.7267** | **.293** |

| σ²_u0 | 311.98 | Δ vs. M0 | +13.72 **(4.2% explained)** |
|--------|--------|----------|---------------------------|
| σ²_ε | 97.46 | LogLik | −971.10 |

Neither coefficient is statistically significant. The negative direction is consistent with the expected direction — states with more funding per establishment have fewer violations — but the effect is not reliably estimated in this sample.

---

### 2. State Land Area (sq miles)

#### (a) Main effect only
**Formula:** `violations ~ time + time² + time³ + land_area_sqmi_z + (1 + time | state)`

| Parameter | β | SE | p |
|-----------|--:|---:|--:|
| Intercept | 17.5575 | 3.2051 | < .001 |
| time | 2.9455 | 0.7755 | < .001 |
| time² | −0.6283 | 0.3470 | .070 |
| time³ | −0.1441 | 0.0522 | .006 |
| **land_area_sqmi_z** | **−0.7970** | **1.4447** | **.581** |

| σ²_u0 | 325.21 | Δ vs. M0 | +0.49 **(0.2% explained)** |
|--------|--------|----------|--------------------------|
| σ²_ε | 96.33 | LogLik | −973.79 |

#### (b) Main effect + linear time interaction
**Formula:** `violations ~ time + time² + time³ + land_area_sqmi_z + land_area_sqmi_z×time + (1 + time | state)`

| Parameter | β | SE | p |
|-----------|--:|---:|--:|
| Intercept | 17.4368 | 3.2458 | < .001 |
| time | 2.9307 | 0.7881 | < .001 |
| time² | −0.6167 | 0.3478 | .076 |
| time³ | −0.1423 | 0.0524 | .007 |
| **land_area_sqmi_z** | **0.0210** | **3.0226** | **.994** |
| **land_area_sqmi_z × time** | **0.1884** | **0.6286** | **.764** |

| σ²_u0 | 328.68 | Δ vs. M0 | −2.97 **(−0.9% — variance increases)** |
|--------|--------|----------|----------------------------------------|
| σ²_ε | 96.36 | LogLik | −973.35 |

Land area explains essentially none of the between-state variance and is not significant in either model form. Joe's travel-burden hypothesis is theoretically sound but not supported by this data.

---

### 3. Total Farming Operations (Census 2017)

#### (a) Main effect only
**Formula:** `violations ~ time + time² + time³ + operations_z + (1 + time | state)`

| Parameter | β | SE | p |
|-----------|--:|---:|--:|
| Intercept | 17.3142 | 3.1197 | < .001 |
| time | 2.9343 | 0.8200 | < .001 |
| time² | −0.6297 | 0.3568 | .078 |
| time³ | −0.1452 | 0.0532 | .006 |
| **operations_z** | **0.9838** | **1.9218** | **.609** |

| σ²_u0 | 271.46 | Δ vs. M0 | +54.25 **(16.7% explained)** |
|--------|--------|----------|------------------------------|
| σ²_ε | 101.60 | LogLik | −973.39 |

#### (b) Main effect + linear time interaction
**Formula:** `violations ~ time + time² + time³ + operations_z + operations_z×time + (1 + time | state)`

| Parameter | β | SE | p |
|-----------|--:|---:|--:|
| Intercept | 17.1166 | 2.9132 | < .001 |
| time | 2.8799 | 0.7465 | < .001 |
| time² | −0.6227 | 0.3504 | .076 |
| time³ | −0.1440 | 0.0524 | .006 |
| **operations_z** | **7.9188** | **2.6206** | **.003** |
| **operations_z × time** | **1.3045** | **0.4442** | **.003** |

| σ²_u0 | 264.67 | Δ vs. M0 | +61.04 **(18.7% explained)** |
|--------|--------|----------|------------------------------|
| σ²_ε | 98.85 | LogLik | −969.25 |

The largest single reduction in between-state variance. Both the main effect and time interaction are highly significant (p = .003) in model (b) — states with more farming operations have higher baseline violations and a steeper upward trend over time.

---

### 4. H-2A Workers per State (2017)

#### (a) Main effect only
**Formula:** `violations ~ time + time² + time³ + h2a_workers_z + (1 + time | state)`

| Parameter | β | SE | p |
|-----------|--:|---:|--:|
| Intercept | 16.7964 | 3.0268 | < .001 |
| time | 2.8575 | 0.7819 | < .001 |
| time² | −0.6365 | 0.3451 | .065 |
| time³ | −0.1470 | 0.0517 | .004 |
| **h2a_workers_z** | **3.1907** | **0.9117** | **< .001** |

| σ²_u0 | 271.39 | Δ vs. M0 | +54.32 **(16.7% explained)** |
|--------|--------|----------|------------------------------|
| σ²_ε | 96.61 | LogLik | −967.61 |

#### (b) Main effect + linear time interaction
**Formula:** `violations ~ time + time² + time³ + h2a_workers_z + h2a_workers_z×time + (1 + time | state)`

| Parameter | β | SE | p |
|-----------|--:|---:|--:|
| Intercept | 16.9438 | 2.9971 | < .001 |
| time | 2.9863 | 0.7739 | < .001 |
| time² | −0.6447 | 0.3435 | .061 |
| time³ | −0.1521 | 0.0515 | .003 |
| **h2a_workers_z** | **9.9413** | **3.0930** | **.001** |
| **h2a_workers_z × time** | **1.3876** | **0.6104** | **.023** |

| σ²_u0 | 266.10 | Δ vs. M0 | +59.61 **(18.3% explained)** |
|--------|--------|----------|------------------------------|
| σ²_ε | 95.09 | LogLik | −964.63 |

Nearly identical variance absorption as total farming operations. Both the main effect and time interaction are significant. H-2A workers and total operations were suspected to be collinear — the combined model below tests this directly.

---

### 5. H-2A Workers per Operation (2017)

#### (a) Main effect only
**Formula:** `violations ~ time + time² + time³ + workers_per_operation_z + (1 + time | state)`

| Parameter | β | SE | p |
|-----------|--:|---:|--:|
| Intercept | 17.0260 | 3.2645 | < .001 |
| time | 2.8939 | 0.8083 | < .001 |
| time² | −0.6330 | 0.3412 | .064 |
| time³ | −0.1467 | 0.0512 | .004 |
| **workers_per_operation_z** | **2.3152** | **1.1611** | **.046** |

| σ²_u0 | 329.78 | Δ vs. M0 | −4.07 **(−1.3% — variance increases)** |
|--------|--------|----------|----------------------------------------|
| σ²_ε | 92.32 | LogLik | −974.06 |

#### (b) Main effect + linear time interaction
**Formula:** `violations ~ time + time² + time³ + workers_per_operation_z + workers_per_operation_z×time + (1 + time | state)`

| Parameter | β | SE | p |
|-----------|--:|---:|--:|
| Intercept | 16.9319 | 3.2617 | < .001 |
| time | 2.9477 | 0.7999 | < .001 |
| time² | −0.6151 | 0.3444 | .074 |
| time³ | −0.1462 | 0.0516 | .005 |
| **workers_per_operation_z** | **5.0423** | **3.1872** | **.114** |
| **workers_per_operation_z × time** | **0.5398** | **0.6122** | **.378** |

| σ²_u0 | 327.43 | Δ vs. M0 | −1.73 **(−0.5% — variance increases)** |
|--------|--------|----------|----------------------------------------|
| σ²_ε | 95.00 | LogLik | −969.44 |

Although the main effect reaches nominal significance in model (a) (p = .046), between-state variance actually increases relative to M0 in both model forms. This suggests instability — possibly driven by collinearity with `h2a_workers` or `operations`, or the ratio measure masking underlying scale differences across states.

---

## Summary Variance Decomposition Table (with Fixed Effect Estimates)

Baseline σ²_u0 (M0) = **325.71**

### Variance Components

| Model | Predictor | σ²_u0 | σ²_ε | Δσ²_u0 | % Explained | LogLik |
|-------|-----------|------:|-----:|-------:|------------:|-------:|
| M0 | Time only (baseline) | 325.71 | 95.99 | — | — | −974.99 |
| M1a | + spending/estab | 311.30 | 97.56 | +14.41 | 4.4% | −972.33 |
| M1b | + spending/estab × time | 311.98 | 97.46 | +13.72 | 4.2% | −971.10 |
| M2a | + land area | 325.21 | 96.33 | +0.49 | 0.2% | −973.79 |
| M2b | + land area × time | 328.68 | 96.36 | −2.97 | −0.9% | −973.35 |
| M3a | + farming operations | 271.46 | 101.60 | +54.25 | **16.7%** | −973.39 |
| M3b | + farming operations × time | 264.67 | 98.85 | +61.04 | **18.7%** | −969.25 |
| M4a | + H-2A workers | 271.39 | 96.61 | +54.32 | **16.7%** | −967.61 |
| M4b | + H-2A workers × time | 266.10 | 95.09 | +59.61 | **18.3%** | −964.63 |
| M5a | + workers/operation | 329.78 | 92.32 | −4.07 | −1.3% | −974.06 |
| M5b | + workers/operation × time | 327.43 | 95.00 | −1.73 | −0.5% | −969.44 |
| **M6** | **+ operations + H-2A (both × time)** | **229.14** | **95.31** | **+96.57** | **29.6%** | **−960.32** |

### Fixed Effect Estimates — All Models

| Model | Parameter | β | SE | p |
|-------|-----------|--:|---:|--:|
| **M0 (baseline)** | Intercept | 17.4569 | 3.2025 | < .001 |
| | time | 2.9648 | 0.7754 | < .001 |
| | time² | −0.6260 | 0.3464 | .071 |
| | time³ | −0.1453 | 0.0520 | .005 |
| **M1a** | Intercept | 17.6270 | 3.1190 | < .001 |
| | time | 2.9639 | 0.7459 | < .001 |
| | time² | −0.6327 | 0.3471 | .068 |
| | time³ | −0.1437 | 0.0523 | .006 |
| | spending_per_estab_z | −1.9799 | 1.6307 | .225 |
| **M1b** | Intercept | 17.8496 | 3.1244 | < .001 |
| | time | 3.0075 | 0.7471 | < .001 |
| | time² | −0.6391 | 0.3477 | .066 |
| | time³ | −0.1467 | 0.0524 | .005 |
| | spending_per_estab_z | −4.7848 | 3.0872 | .121 |
| | spending_per_estab_z × time | −0.7645 | 0.7267 | .293 |
| **M2a** | Intercept | 17.5575 | 3.2051 | < .001 |
| | time | 2.9455 | 0.7755 | < .001 |
| | time² | −0.6283 | 0.3470 | .070 |
| | time³ | −0.1441 | 0.0522 | .006 |
| | land_area_sqmi_z | −0.7970 | 1.4447 | .581 |
| **M2b** | Intercept | 17.4368 | 3.2458 | < .001 |
| | time | 2.9307 | 0.7881 | < .001 |
| | time² | −0.6167 | 0.3478 | .076 |
| | time³ | −0.1423 | 0.0524 | .007 |
| | land_area_sqmi_z | 0.0210 | 3.0226 | .994 |
| | land_area_sqmi_z × time | 0.1884 | 0.6286 | .764 |
| **M3a** | Intercept | 17.3142 | 3.1197 | < .001 |
| | time | 2.9343 | 0.8200 | < .001 |
| | time² | −0.6297 | 0.3568 | .078 |
| | time³ | −0.1452 | 0.0532 | .006 |
| | operations_z | 0.9838 | 1.9218 | .609 |
| **M3b** | Intercept | 17.1166 | 2.9132 | < .001 |
| | time | 2.8799 | 0.7465 | < .001 |
| | time² | −0.6227 | 0.3504 | .076 |
| | time³ | −0.1440 | 0.0524 | .006 |
| | operations_z | 7.9188 | 2.6206 | .003 |
| | operations_z × time | 1.3045 | 0.4442 | .003 |
| **M4a** | Intercept | 16.7964 | 3.0268 | < .001 |
| | time | 2.8575 | 0.7819 | < .001 |
| | time² | −0.6365 | 0.3451 | .065 |
| | time³ | −0.1470 | 0.0517 | .004 |
| | h2a_workers_z | 3.1907 | 0.9117 | < .001 |
| **M4b** | Intercept | 16.9438 | 2.9971 | < .001 |
| | time | 2.9863 | 0.7739 | < .001 |
| | time² | −0.6447 | 0.3435 | .061 |
| | time³ | −0.1521 | 0.0515 | .003 |
| | h2a_workers_z | 9.9413 | 3.0930 | .001 |
| | h2a_workers_z × time | 1.3876 | 0.6104 | .023 |
| **M5a** | Intercept | 17.0260 | 3.2645 | < .001 |
| | time | 2.8939 | 0.8083 | < .001 |
| | time² | −0.6330 | 0.3412 | .064 |
| | time³ | −0.1467 | 0.0512 | .004 |
| | workers_per_operation_z | 2.3152 | 1.1611 | .046 |
| **M5b** | Intercept | 16.9319 | 3.2617 | < .001 |
| | time | 2.9477 | 0.7999 | < .001 |
| | time² | −0.6151 | 0.3444 | .074 |
| | time³ | −0.1462 | 0.0516 | .005 |
| | workers_per_operation_z | 5.0423 | 3.1872 | .114 |
| | workers_per_operation_z × time | 0.5398 | 0.6122 | .378 |

---

## Combined Model — Operations + H-2A Workers (Collinearity Test)

**Formula:** `violations ~ time + time² + time³ + operations_z + operations_z×time + h2a_workers_z + h2a_workers_z×time + (1 + time | state)`

### Collinearity Check

**Pearson r (operations_z, h2a_workers_z) = 0.056** — near-zero correlation. The two predictors are essentially orthogonal in this sample and are not collinear.

### Fixed Effect Estimates

| Parameter | β | SE | p |
|-----------|--:|---:|--:|
| Intercept | 16.6827 | 2.8217 | < .001 |
| time | 2.9215 | 0.7568 | < .001 |
| time² | −0.6514 | 0.3440 | .058 |
| time³ | −0.1525 | 0.0515 | .003 |
| **operations_z** | **6.8620** | **2.5191** | **.006** |
| **operations_z × time** | **1.1424** | **0.4675** | **.015** |
| **h2a_workers_z** | **9.2657** | **2.9639** | **.002** |
| **h2a_workers_z × time** | **1.2624** | **0.5920** | **.033** |

### Variance Decomposition

| Component | Value |
|-----------|------:|
| σ²_u0 (between-state) | 229.14 |
| σ²_ε (within-state) | 95.31 |
| **Δ vs. M0** | **+96.57** |
| **% between-state variance explained** | **29.6%** |
| Log-Likelihood | −960.32 |

### Comparison Against Single-Predictor Best Models

| Model | % Between-State Variance Explained |
|-------|-----------------------------------:|
| M3b — operations only (+×time) | 18.7% |
| M4b — H-2A workers only (+×time) | 18.3% |
| **M6 — both together (+×time)** | **29.6%** |
| **Increment over best single predictor** | **+10.9 pp** |

### Interpretation

The combined model confirms that farming operations and H-2A workers capture **independent variance** in state-level violations. Despite absorbing nearly identical amounts of between-state variance individually (~18–19% each), the two predictors are almost uncorrelated (r = 0.056). When entered together, they explain 29.6% of between-state variance — a meaningful 10.9 percentage-point increment over the best single predictor alone.

Both predictors remain statistically significant in the joint model:
- **operations_z**: β = 6.86, p = .006 (main effect); β = 1.14, p = .015 (×time interaction)
- **h2a_workers_z**: β = 9.27, p = .002 (main effect); β = 1.26, p = .033 (×time interaction)

This suggests that total farming scale (operations) and agricultural labor demand (H-2A workers) capture distinct dimensions of state agricultural activity — both of which independently predict WPS violation rates and their change over time.

---

## Key Takeaways

**1. Spending variables explain little between-state variance (~2–4%) and are not statistically significant.** This is worth discussing with the team — it may reflect that raw or normalized spending levels are not the primary driver of state-level baseline violations, or that the spending measure lacks sufficient variation once normalized.

**2. Total farming operations and H-2A workers are the strongest single predictors**, each absorbing ~17–19% of between-state variance when interacted with time. Both are significant in their respective models.

**3. Farming operations and H-2A workers are NOT collinear (r = 0.056).** Including both in the combined model yields additive benefit — 29.6% of between-state variance explained, versus ~18% for either alone. This rules out coefficient suppression concerns and supports including both predictors in the final model.

**4. Land area does not empirically reduce between-state variance.** While the travel-burden hypothesis is plausible, the data do not support it as a meaningful covariate in this model.

**5. Workers per operation adds noise rather than signal.** The ratio measure may be less stable than its constituent parts; the team should decide whether to retain it.

---

## Next Steps

- [ ] **Pull ECHO WPS establishment counts for 2020–2022** (Stephane's specific ask) — required before per-establishment spending can be used for any analysis period extending to 2022
- [ ] Obtain USDA Census of Agriculture farm counts (2012 and 2017) to fill the 2011–2013 ECHO establishment gap; document substitution in methods
- [ ] Obtain USDA NASS Quick Stats harvested cropland acres by state to compute the priority dollars-per-acre operationalization
- [ ] **Build toward the full Aim 3 model**: operations + H-2A workers (both × time) is the current best specification. Next step is adding the spending predictor jointly and testing whether it contributes independent variance beyond the agricultural scale predictors.
- [ ] Discuss with team: spending per establishment explains only ~4% of between-state variance and is not significant; revisit operationalization or examine whether spending is driving violations in this period

---

*Script: `stepwise_zimmerman_models.py` | Level-2 covariates: `level2_covariates.csv`*
