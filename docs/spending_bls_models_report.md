# BLS-Normalized Spending Models
## WPS Enforcement Analysis — Stepwise HLM Results

---

## 1. Research Objective

This analysis examines whether state-level EPA STAG (State and Tribal Assistance Grant) funding predicts Worker Protection Standard (WPS) violation rates across U.S. states (2011–2019), after accounting for the size of the population or geographic area the funding is meant to serve.

Rather than treating raw spending as the predictor, five normalized spending variables are constructed by dividing each state's mean inflation-adjusted STAG funding by a different denominator that reflects the scale of the regulated workforce or geography. Each normalized variable is then entered as a Level-2 (state-level, time-invariant) covariate in a hierarchical linear model (HLM) predicting annual WPS violations.

---

## 2. Data Sources

| Source | Content | Years |
|---|---|---|
| EPA ECHO | WPS violations per state per year | 2011–2019 |
| EPA STAG spending records | Nominal grant obligations by state and year | 2011–2019 |
| BLS Occupational Employment and Wage Statistics (OEWS) | State-level employment counts by occupation code | 2011 (snapshot) |
| USDA Census of Agriculture (via Yuri's `h2a_state_summary_2017.csv`) | Total farming operations per state | 2017 |
| U.S. Census Bureau | State land area in square miles | 2020 decennial |
| BLS CPI-U | Annual price index for inflation adjustment | 2011–2019 |

**BLS OEWS file used:** `bls_oews_panel.csv`

---

## 3. Variable Construction

### 3.1 Spending Numerator

All five variables share the same numerator: **mean inflation-adjusted STAG spending per state**, computed as:

1. Annual nominal STAG obligations are summed to the state-year level.
2. Each year's nominal dollars are converted to constant 2017 dollars using CPI-U:

$$\text{spending}_{2017} = \text{spending}_{\text{nominal}} \times \frac{\text{CPI}_{2017}}{\text{CPI}_{\text{year}}}$$

3. The inflation-adjusted values are averaged across all years (2011–2019) for each state, yielding one time-invariant state-level spending figure.

### 3.2 Denominators and Resulting Variables

Each normalized spending variable divides the mean inflation-adjusted spending by a state-level count or area measure. All five are then **z-scored** (mean = 0, SD = 1) before entry into models.

| Variable | Formula | Denominator Source | OCC / Data |
|---|---|---|---|
| **SPEND\_WORK** | mean spending / # farmworkers | BLS OEWS 2011 | OCC 45-2092 — Farmworkers and Laborers, Crop, Nursery, and Greenhouse |
| **SPEND\_APP** | mean spending / # pesticide applicators | BLS OEWS 2011 | OCC 37-3012 — Pesticide Handlers, Sprayers, and Applicators, Vegetation |
| **SPEND\_FLC** | mean spending / # frontline supervisors | BLS OEWS 2011 | OCC 45-1011 — First-Line Supervisors of Farming, Fishing, and Forestry Workers |
| **SPEND\_OP** | mean spending / # farming operations | Census of Agriculture 2017 | Total farming operations per state |
| **SPEND\_AREA** | mean spending / land area (sq mi) | U.S. Census Bureau | State total land area |

> **Note on OCC 47-3012:** The original request specified OCC code 47-3012 for pesticide applicators. This code is not present in the BLS OEWS file. The only pesticide applicator occupation in the file is **37-3012** ("Pesticide Handlers, Sprayers, and Applicators, Vegetation"), which was used in its place. This should be confirmed against the intended BLS classification before finalizing.

> **Note on SPEND\_OP:** The farming operations denominator corresponds to the `operations_z` variable in the project's Level-2 data file (`level2_covariates.csv`), drawn from Yuri's 2017 Census of Agriculture summary. The user's label "observations\_Z" refers to this variable.

### 3.3 Descriptive Statistics of Raw (Pre-Standardization) Spend Variables

| Variable | N States | Mean | SD | Min | Max |
|---|---|---|---|---|---|
| SPEND\_WORK | 44 | 485.61 | 668.54 | 0.00 | 3,037.98 |
| SPEND\_APP | 41 | 978.60 | 1,105.40 | 0.00 | 4,623.64 |
| SPEND\_FLC | 45 | 1,319.45 | 1,440.01 | 0.00 | 6,077.09 |
| SPEND\_OP | 47 | 19.56 | 47.97 | 0.00 | 326.29 |
| SPEND\_AREA | 47 | 7.42 | 9.69 | 0.00 | 41.37 |

*Units for SPEND\_WORK, SPEND\_APP, SPEND\_FLC are dollars per worker/supervisor. SPEND\_OP is dollars per farming operation. SPEND\_AREA is dollars per square mile.*

### 3.4 Missing Data by Variable

Some states are absent from the 2011 BLS OEWS file for specific occupations. These states are excluded listwise from models using that variable.

| Variable | States with Missing BLS Data |
|---|---|
| SPEND\_WORK | Alaska, Minnesota, Nevada |
| SPEND\_APP | Alaska, Colorado, New Hampshire, New Mexico, Rhode Island, Utah, Vermont |
| SPEND\_FLC | Alaska, Rhode Island, Wyoming |
| SPEND\_OP | None (uses Census of Agriculture data) |
| SPEND\_AREA | None (uses Census Bureau land area) |

---

## 4. Analytic Approach

### 4.1 Model Framework

All models are **hierarchical linear models (HLM)** with:
- **Level 1:** Annual WPS violations nested within states
- **Level 2:** Time-invariant state-level covariates (the five normalized spending variables)
- **Random effects:** Random intercept + random slope for linear time, by state
- **Estimation:** REML via L-BFGS-B optimizer (statsmodels `MixedLM`)

### 4.2 Time Specification

Time is centered on 2017 (`time = year − 2017`, range −6 to +2). A cubic polynomial is retained in all models:

$$\text{violations}_{it} = \beta_0 + \beta_1\text{time} + \beta_2\text{time}^2 + \beta_3\text{time}^3 + u_{0i} + u_{1i}\text{time} + \varepsilon_{it}$$

### 4.3 Zimmerman Stepwise Sequence

Following Zimmerman (2000) and Raudenbush & Bryk (2002), each Level-2 predictor is entered one at a time to cleanly attribute between-state variance reduction:

**Step 1 (Model a):** Add the standardized spending variable as a main effect only.

$$\text{violations}_{it} = \beta_0 + \beta_1\text{time} + \beta_2\text{time}^2 + \beta_3\text{time}^3 + \gamma_{01}X_i + u_{0i} + u_{1i}\text{time} + \varepsilon_{it}$$

**Step 2 (Model b):** Add the cross-level interaction of the spending variable with linear time.

$$\text{violations}_{it} = \ldots + \gamma_{01}X_i + \gamma_{11}X_i \times \text{time} + u_{0i} + u_{1i}\text{time} + \varepsilon_{it}$$

**Step 3 (Combined model):** All five spending variables entered simultaneously. A spending variable's interaction with time is included in the combined model **only if its interaction p-value was < .20 in Step 2**.

### 4.4 Key Metric: Pseudo-R² (Between-State)

$$\text{Pseudo-}R^2 = \frac{\sigma^2_{u0,\text{baseline}} - \sigma^2_{u0,\text{model}}}{\sigma^2_{u0,\text{baseline}}}$$

This captures the proportion of between-state variance explained by each predictor relative to the baseline (time-only) model.

---

## 5. Baseline Model

**Model 0: Unconditional (time polynomial only)**

| Parameter | β | SE | p |
|---|---|---|---|
| Intercept | 17.457 | 3.203 | < .001 |
| time | 2.965 | 0.775 | < .001 |
| time² | −0.626 | 0.346 | .071 |
| time³ | −0.145 | 0.052 | .005 |

| Variance Component | Estimate |
|---|---|
| Between-state (σ²\_u0) | **325.71** |
| Within-state (σ²\_ε) | 95.99 |
| ICC | **77.2%** |
| N | 250 obs, 47 states |

The ICC of 77.2% confirms that the large majority of variance in WPS violations is between states rather than within states over time — justifying the HLM approach and the focus on state-level predictors.

---

## 6. Stepwise Model Results

*Significance: \*\* p < .01 · \* p < .05 · + p < .10 · (blank) p < .20*

---

### 6.1 SPEND\_WORK — Spending per Farmworker (OCC 45-2092)

*N = 235 obs, 44 states (Alaska, Minnesota, Nevada excluded)*

#### Model a: Main effect only

| Parameter | β | SE | p | |
|---|---|---|---|---|
| Intercept | 18.199 | 3.311 | < .001 | \*\* |
| time | 3.018 | 0.802 | < .001 | \*\* |
| time² | −0.668 | 0.354 | .059 | + |
| time³ | −0.151 | 0.054 | .005 | \*\* |
| SPEND\_WORK\_z | **−2.713** | 1.114 | **.015** | \* |

- Between-state variance (σ²\_u0): **314.99** → **3.3% explained** vs. baseline

#### Model b: Main effect + ×time interaction

| Parameter | β | SE | p | |
|---|---|---|---|---|
| Intercept | 18.962 | 3.338 | < .001 | \*\* |
| time | 3.274 | 0.818 | < .001 | \*\* |
| time² | −0.641 | 0.351 | .068 | + |
| time³ | −0.150 | 0.053 | .005 | \*\* |
| SPEND\_WORK\_z | **−8.441** | 3.227 | **.009** | \*\* |
| SPEND\_WORK\_z × time | **−1.124** | 0.595 | **.059** | + |

- Between-state variance (σ²\_u0): **315.52** → **3.1% explained** vs. baseline
- **Interaction p = .059 → INCLUDED in combined model**

**Interpretation:** States with higher spending per farmworker had significantly fewer violations. The negative interaction indicates this protective effect grew slightly stronger over time (i.e., the violation gap between high- and low-spending states widened across years).

---

### 6.2 SPEND\_APP — Spending per Pesticide Applicator (OCC 37-3012)

*N = 220 obs, 41 states (Alaska, CO, NH, NM, RI, UT, VT excluded)*

#### Model a: Main effect only

| Parameter | β | SE | p | |
|---|---|---|---|---|
| Intercept | 20.031 | 3.448 | < .001 | \*\* |
| time | 3.008 | 0.915 | .001 | \*\* |
| time² | −0.831 | 0.404 | .040 | \* |
| time³ | −0.171 | 0.059 | .004 | \*\* |
| SPEND\_APP\_z | **−2.960** | 1.593 | **.063** | + |

- Between-state variance (σ²\_u0): **305.25** → **6.3% explained** vs. baseline

#### Model b: Main effect + ×time interaction

| Parameter | β | SE | p | |
|---|---|---|---|---|
| Intercept | 21.600 | 3.368 | < .001 | \*\* |
| time | 3.463 | 0.874 | < .001 | \*\* |
| time² | −0.818 | 0.399 | .040 | \* |
| time³ | −0.175 | 0.059 | .003 | \*\* |
| SPEND\_APP\_z | **−9.359** | 3.071 | **.002** | \*\* |
| SPEND\_APP\_z × time | **−1.309** | 0.561 | **.020** | \* |

- Between-state variance (σ²\_u0): **302.91** → **7.0% explained** vs. baseline
- **Interaction p = .020 → INCLUDED in combined model**

**Interpretation:** States spending more per pesticide applicator had fewer violations. The significant interaction indicates the spending effect strengthened over time — high-spending states diverged increasingly from low-spending states across the study period.

---

### 6.3 SPEND\_FLC — Spending per Frontline Supervisor (OCC 45-1011)

*N = 246 obs, 45 states (Alaska, Rhode Island, Wyoming excluded)*

#### Model a: Main effect only

| Parameter | β | SE | p | |
|---|---|---|---|---|
| Intercept | 18.335 | 3.190 | < .001 | \*\* |
| time | 2.904 | 0.799 | < .001 | \*\* |
| time² | −0.665 | 0.350 | .058 | + |
| time³ | −0.145 | 0.053 | .006 | \*\* |
| SPEND\_FLC\_z | **−2.904** | 0.977 | **.003** | \*\* |

- Between-state variance (σ²\_u0): **297.74** → **8.6% explained** vs. baseline

#### Model b: Main effect + ×time interaction

| Parameter | β | SE | p | |
|---|---|---|---|---|
| Intercept | 20.055 | 3.259 | < .001 | \*\* |
| time | 3.435 | 0.821 | < .001 | \*\* |
| time² | −0.634 | 0.347 | .068 | + |
| time³ | −0.147 | 0.052 | .005 | \*\* |
| SPEND\_FLC\_z | **−9.229** | 3.015 | **.002** | \*\* |
| SPEND\_FLC\_z × time | **−1.260** | 0.556 | **.023** | \* |

- Between-state variance (σ²\_u0): **296.42** → **9.0% explained** vs. baseline
- **Interaction p = .023 → INCLUDED in combined model**

**Interpretation:** SPEND\_FLC is the strongest individual predictor of the five. States spending more per frontline supervisor had significantly fewer violations, and this effect grew over time. This suggests that targeting funds relative to supervisory capacity may be especially consequential.

---

### 6.4 SPEND\_OP — Spending per Farming Operation (Census of Agriculture 2017)

*N = 250 obs, 47 states (no missing)*

#### Model a: Main effect only

| Parameter | β | SE | p | |
|---|---|---|---|---|
| Intercept | 17.586 | 3.134 | < .001 | \*\* |
| time | 2.964 | 0.743 | < .001 | \*\* |
| time² | −0.627 | 0.347 | .070 | + |
| time³ | −0.144 | 0.052 | .006 | \*\* |
| SPEND\_OP\_z | −1.577 | 1.578 | .318 | |

- Between-state variance (σ²\_u0): **316.98** → **2.7% explained** vs. baseline

#### Model b: Main effect + ×time interaction

| Parameter | β | SE | p | |
|---|---|---|---|---|
| Intercept | 17.782 | 3.164 | < .001 | \*\* |
| time | 2.995 | 0.763 | < .001 | \*\* |
| time² | −0.641 | 0.349 | .066 | + |
| time³ | −0.147 | 0.053 | .005 | \*\* |
| SPEND\_OP\_z | −4.623 | 3.147 | .142 | |
| SPEND\_OP\_z × time | −0.813 | 0.734 | .268 | |

- Between-state variance (σ²\_u0): **318.19** → **2.3% explained** vs. baseline
- **Interaction p = .268 → EXCLUDED from combined model**

**Interpretation:** Spending per farming operation explains very little between-state variance and is not statistically significant. This replicates the prior finding with the `spending_per_operation` variable from the Census of Agriculture, confirming that normalizing by number of agricultural operations does not meaningfully predict violation rates.

---

### 6.5 SPEND\_AREA — Spending per Square Mile of Land Area

*N = 250 obs, 47 states (no missing)*

#### Model a: Main effect only

| Parameter | β | SE | p | |
|---|---|---|---|---|
| Intercept | 17.477 | 2.939 | < .001 | \*\* |
| time | 2.928 | 0.606 | < .001 | \*\* |
| time² | −0.637 | 0.340 | .061 | + |
| time³ | −0.146 | 0.052 | .005 | \*\* |
| SPEND\_AREA\_z | −1.001 | — | — | |

- Between-state variance (σ²\_u0): **319.46** → **1.9% explained** vs. baseline
- *Note: SE for SPEND\_AREA\_z did not converge in this model (NaN). The point estimate is reported; interpret with caution. The interaction model below converged normally.*

#### Model b: Main effect + ×time interaction

| Parameter | β | SE | p | |
|---|---|---|---|---|
| Intercept | 18.002 | 3.188 | < .001 | \*\* |
| time | 3.159 | 0.778 | < .001 | \*\* |
| time² | −0.605 | 0.345 | .080 | + |
| time³ | −0.145 | 0.052 | .005 | \*\* |
| SPEND\_AREA\_z | **−6.653** | 3.350 | **.047** | \* |
| SPEND\_AREA\_z × time | **−1.078** | 0.601 | **.073** | + |

- Between-state variance (σ²\_u0): **321.15** → **1.4% explained** vs. baseline
- **Interaction p = .073 → INCLUDED in combined model**

**Interpretation:** Spending per square mile is the weakest individual predictor in terms of variance explained. The main effect convergence issue in Model a suggests instability when the predictor explains very little, though the interaction model is more stable. The negative interaction indicates states spending more relative to land area had modestly fewer violations, particularly in later years.

---

## 7. Summary: Between-State Variance Decomposition

*Baseline σ²\_u0 = 325.71 (ICC = 77.2%)*

| Model | N obs | σ²\_u0 | Δσ²\_u0 | % Baseline Explained | Log-Lik |
|---|---|---|---|---|---|
| M0: Baseline (time only) | 250 | 325.71 | — | — | −974.99 |
| SPEND\_WORK: main | 235 | 314.99 | +10.72 | 3.3% | −914.21 |
| SPEND\_WORK: main + ×time | 235 | 315.52 | +10.19 | 3.1% | −912.06 |
| SPEND\_APP: main | 220 | 305.25 | +20.46 | 6.3% | −863.14 |
| SPEND\_APP: main + ×time | 220 | 302.91 | +22.80 | 7.0% | −860.35 |
| SPEND\_FLC: main | 246 | 297.74 | +27.97 | **8.6%** | −955.05 |
| SPEND\_FLC: main + ×time | 246 | 296.42 | +29.28 | **9.0%** | −952.07 |
| SPEND\_OP: main | 250 | 316.98 | +8.72 | 2.7% | −972.70 |
| SPEND\_OP: main + ×time | 250 | 318.19 | +7.52 | 2.3% | −971.52 |
| SPEND\_AREA: main | 250 | 319.46 | +6.24 | 1.9% | −973.09 |
| SPEND\_AREA: main + ×time | 250 | 321.15 | +4.55 | 1.4% | −971.36 |

---

## 8. Combined Model — All Five Spending Variables Simultaneously

### 8.1 Interaction Selection (p < .20 threshold)

| Variable | ×time p-value | Decision |
|---|---|---|
| SPEND\_WORK\_z | .059 | **Included** |
| SPEND\_APP\_z | .020 | **Included** |
| SPEND\_FLC\_z | .023 | **Included** |
| SPEND\_OP\_z | .268 | Excluded |
| SPEND\_AREA\_z | .073 | **Included** |

### 8.2 Combined Model Formula

```
violations ~ time + time² + time³
           + SPEND_WORK_z + SPEND_APP_z + SPEND_FLC_z + SPEND_OP_z + SPEND_AREA_z
           + SPEND_WORK_z×time + SPEND_APP_z×time + SPEND_FLC_z×time + SPEND_AREA_z×time
```

*N = 207 obs, 38 states (listwise deletion of states missing any BLS variable)*

### 8.3 Combined Model Results

| Parameter | β | SE | p | |
|---|---|---|---|---|
| Intercept | 24.459 | 4.060 | < .001 | \*\* |
| time | 3.991 | 0.953 | < .001 | \*\* |
| time² | −0.815 | 0.404 | .044 | \* |
| time³ | −0.181 | 0.060 | .003 | \*\* |
| SPEND\_WORK\_z | −8.533 | 6.377 | .181 | |
| SPEND\_APP\_z | **−11.443** | 4.984 | **.022** | \* |
| SPEND\_FLC\_z | 2.141 | 7.964 | .788 | |
| SPEND\_OP\_z | 6.969 | 10.792 | .518 | |
| SPEND\_AREA\_z | −3.153 | 5.553 | .570 | |
| SPEND\_WORK\_z × time | −1.106 | 1.171 | .345 | |
| SPEND\_APP\_z × time | **−1.507** | 0.886 | **.089** | + |
| SPEND\_FLC\_z × time | 0.315 | 1.455 | .828 | |
| SPEND\_AREA\_z × time | −0.480 | 0.948 | .613 | |

| Variance Component | Estimate |
|---|---|
| Between-state (σ²\_u0) | 303.77 |
| Within-state (σ²\_ε) | 108.19 |
| Baseline variance explained | **6.7%** |
| Log-Likelihood | −795.28 |

---

## 9. Interpretation and Key Findings

### 9.1 Individual Variable Performance

**SPEND\_FLC (spending per frontline supervisor)** is the strongest individual predictor:
- Main effect: β = −2.90, p = .003; explains **8.6%** of between-state variance
- With interaction: β = −9.23, p = .002; explains **9.0%** of between-state variance
- Frontline supervisors are the direct supervisory layer responsible for WPS compliance training and oversight. Funding adequacy relative to this workforce may be the most proximal mechanism linking STAG grants to enforcement outcomes.

**SPEND\_APP (spending per pesticide applicator)** is the second-strongest:
- Main effect: β = −2.96, p = .063; explains **6.3%** of between-state variance
- With interaction: β = −9.36, p = .002; explains **7.0%** of between-state variance
- The significant interaction (p = .020) indicates the gap in violations between high- and low-spending states widened across the study period.

**SPEND\_WORK (spending per farmworker)** explains less variance (3.1–3.3%) but both main and interaction effects are statistically significant:
- Main: p = .015\*; Interaction: p = .059+

**SPEND\_OP (spending per farming operation)** is non-significant in both models (p = .318 and p = .268). This is consistent with the prior analysis using the same denominator (`spending_per_operation`), which also showed minimal explanatory power (~2%).

**SPEND\_AREA (spending per square mile)** explains the least variance (1.4–1.9%). The main-effect-only model had a convergence issue (NaN SE), though the interaction model converged. This supports the prior finding that geographic travel burden, as captured by land area normalization, does not strongly predict enforcement outcomes.

### 9.2 Combined Model Findings

When all five spending variables are entered simultaneously (207 obs, 38 states):

- **Multicollinearity** among the three worker-count-based variables (SPEND\_WORK, SPEND\_APP, SPEND\_FLC) suppresses individual estimates. All three become non-significant in each other's presence.
- **SPEND\_APP** is the only variable that retains statistical significance as a main effect (β = −11.44, p = .022), and its interaction with time remains marginally significant (β = −1.51, p = .089).
- **SPEND\_OP** and **SPEND\_AREA** are non-significant in the combined model, consistent with their weak individual-model results.
- The combined model explains **6.7% of baseline between-state variance** — less than SPEND\_FLC alone (9.0%), reflecting the loss of statistical power from listwise deletion (207 vs. 246 obs) and collinearity among the three workforce-normalized variables.

### 9.3 Recommendation

Given the collinearity among workforce-normalized spending variables and the sample size loss in the combined model, a **parsimonious approach** would favor **SPEND\_FLC** as the primary spending operationalization for subsequent analysis. It:
- Explains the most between-state variance individually (9.0%)
- Has the strongest and most consistent significance across both model steps
- Is conceptually proximal to the WPS enforcement mechanism (supervisory compliance capacity)
- Has the smallest listwise deletion loss among the BLS-based variables (2 states excluded vs. 6–7 for SPEND\_APP)

If a workforce-agnostic operationalization is preferred, **SPEND\_OP** (spending per farming operation) offers complete coverage (47 states, no missing) and replicates prior analysis — though it is consistently non-significant.

---

## 10. Technical Notes

- All models estimated with REML via L-BFGS-B in Python `statsmodels.MixedLM`
- Time is centered at 2017 (time = year − 2017); cubic polynomial retained across all models
- STAG spending inflated to 2017 dollars using annual BLS CPI-U values
- BLS OEWS employment counts are from 2011 (the only year available in the panel file) and treated as time-invariant Level-2 denominators — consistent with how Census of Agriculture 2017 data is used for farming operations
- Z-scoring uses the sample mean and SD of states with non-missing values for each variable separately
- The combined model uses listwise deletion: only states with valid values on all five BLS-based spending variables are included (N = 207 obs, 38 states)

---

*Script: `spending_bls_models.py` | Report generated: 2026-05-07*
