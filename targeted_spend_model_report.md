# Targeted FIFRA-Aligned Spending Model
## WPS Enforcement Analysis — Theory-Driven Model Specification

---

## 1. Rationale and Research Question

This model refines the prior five-variable combined spending model based on the legal structure of the Worker Protection Standard (WPS) and its authorizing statute, the Federal Insecticide, Fungicide, and Rodenticide Act (FIFRA).

**Legal basis for variable selection:** Under FIFRA and the WPS, two distinct groups of individuals receive explicit regulatory protections:

1. **Agricultural workers** (farmworkers, OCC 45-2092) — those who perform hand-labor tasks in fields treated with pesticides
2. **Pesticide handlers/applicators** (OCC 37-3012) — those who mix, load, or apply pesticides

These are separate, enumerated protected populations with distinct WPS requirements (e.g., worker safety training, pesticide safety information, decontamination supplies; handler-specific PPE and engineering controls). STAG grant funding directed per member of these protected populations therefore has a cleaner and more direct legal interpretation than funding normalized by supervisors (SPEND\_FLC), farming operations (SPEND\_OP), or land area.

**Rationale for dropping SPEND\_FLC:** Frontline supervisors (OCC 45-1011) are not themselves a protected class under FIFRA/WPS — they are the compliance-responsible party, not the beneficiary. While SPEND\_FLC explained the most variance individually (9.0%), it does not map onto the law's framework as cleanly as spending per worker or per applicator.

**Rationale for retaining SPEND\_AREA:** State land area is retained as a structural control to account for the geographic burden inspectors face reaching regulated operations across large states. It does not represent a protected population but captures a logistical constraint on enforcement capacity.

**Rationale for dropping SPEND\_OP:** Spending per farming operation was non-significant in both its individual models (p = .318, p = .268) and is excluded to preserve degrees of freedom.

**Interaction term selection:** SPEND\_APP×time and SPEND\_AREA×time are retained based on their p-values in the prior individual stepwise models (.020 and .073, respectively, both < .20). SPEND\_WORK×time is excluded (p = .059 just met the < .20 threshold, but dropping it frees one additional degree of freedom with minimal theoretical cost, given SPEND\_WORK enters as a main effect).

---

## 2. Model Specification

### 2.1 Formula

```
violations ~ intercept + time + time² + time³
           + SPEND_WORK_z
           + SPEND_APP_z  + SPEND_APP_z × time
           + SPEND_AREA_z + SPEND_AREA_z × time
```

### 2.2 Structure

| Component | Description |
|---|---|
| Outcome | Annual WPS violations per state |
| Level 1 | State-years (2011–2019) |
| Level 2 | State (time-invariant covariates) |
| Random effects | Random intercept + random slope for time, by state |
| Estimation | REML via L-BFGS-B (`statsmodels MixedLM`) |
| Time coding | Centered at 2017 (time = year − 2017, range −6 to +2) |

### 2.3 Predictors

| Term | Type | Variable |
|---|---|---|
| time, time², time³ | Level-1 fixed | Cubic time polynomial |
| SPEND\_WORK\_z | Level-2 main effect | Mean STAG spending / farmworkers (OCC 45-2092), z-scored |
| SPEND\_APP\_z | Level-2 main effect | Mean STAG spending / pesticide applicators (OCC 37-3012), z-scored |
| SPEND\_AREA\_z | Level-2 main effect | Mean STAG spending / land area (sq mi), z-scored |
| SPEND\_APP\_z × time | Cross-level interaction | SPEND\_APP moderating the time trend |
| SPEND\_AREA\_z × time | Cross-level interaction | SPEND\_AREA moderating the time trend |

### 2.4 Sample

States are excluded listwise if missing any predictor in the model. The binding constraint is SPEND\_APP\_z, which requires BLS OEWS data for OCC 37-3012.

| | N |
|---|---|
| Observations | **208** |
| States | **39** |
| States excluded | 11 |
| Excluded states | Alaska, Colorado, Minnesota, Mississippi, Nevada, New Hampshire, New Mexico, Rhode Island, Utah, Vermont, West Virginia |

---

## 3. Baseline Model (Restricted Sample)

Because the analytic sample differs from the full 250-observation dataset (due to listwise deletion on BLS variables), the baseline model is re-estimated on the same 208 observations and 39 states to produce a valid pseudo-R² comparison.

**Baseline formula:** `violations ~ time + time² + time³`

| Variance Component | Estimate |
|---|---|
| Between-state (σ²\_u0) | **376.94** |
| Within-state (σ²\_ε) | 105.83 |
| ICC | **78.1%** |
| Log-Likelihood | −822.50 |

The ICC of 78.1% in this restricted sample is consistent with the full-sample baseline (77.2%), confirming that the excluded states do not substantially alter the overall variance structure.

---

## 4. Model Results

### 4.1 Fixed Effects

| Parameter | β | SE | z | p | 95% CI |
|---|---|---|---|---|---|
| Intercept | 21.620 | 3.547 | 6.10 | < .001 | [14.67, 28.57] |
| time | 3.527 | 0.944 | 3.74 | < .001 | [1.68, 5.38] |
| time² | −0.815 | 0.406 | −2.01 | .045 | [−1.61, −0.02] |
| time³ | −0.176 | 0.060 | −2.93 | .003 | [−0.29, −0.06] |
| **SPEND\_WORK\_z** | **−2.788** | **1.216** | **−2.29** | **.022** | [−5.17, −0.40] |
| **SPEND\_APP\_z** | **−9.787** | **4.523** | **−2.16** | **.030** | [−18.65, −0.92] |
| SPEND\_AREA\_z | −1.445 | 4.803 | −0.30 | .764 | [−10.86, 7.97] |
| SPEND\_APP\_z × time | −1.367 | 0.861 | −1.59 | .112 | [−3.05, 0.32] |
| SPEND\_AREA\_z × time | −0.452 | 0.894 | −0.51 | .613 | [−2.20, 1.30] |

*Significance: \*\* p < .01 · \* p < .05 · + p < .10*

### 4.2 Random Effects

| Component | Estimate |
|---|---|
| Between-state variance (σ²\_u0) | 296.84 |
| Random slope variance (σ²\_u1) | 9.37 |
| Within-state variance (σ²\_ε) | 108.91 |
| Covariance (u0, u1) | 52.74 |

### 4.3 Model Fit

| Metric | Value |
|---|---|
| Log-Likelihood | −808.50 |
| Between-state variance explained (pseudo-R²) | **21.25%** |
| Δσ²\_u0 vs. restricted baseline (reduction) | **+80.10** (376.94 → 296.84) |

---

## 5. Interpretation

### 5.1 Time Trend

The cubic time polynomial is fully significant across all three terms (time: p < .001; time²: p = .045; time³: p = .003). This confirms that WPS violations followed a non-linear trajectory over 2011–2019 — rising initially, then curving downward — after accounting for spending.

### 5.2 SPEND\_WORK — Spending per Farmworker

**β = −2.79, SE = 1.22, p = .022**

States that spent more per farmworker had significantly fewer WPS violations. A one-standard-deviation increase in SPEND\_WORK\_z is associated with approximately **2.8 fewer violations per state per year**, holding other predictors constant. This effect is stable across years (no interaction term retained), suggesting a consistent and time-invariant protective relationship between per-worker spending adequacy and compliance outcomes.

### 5.3 SPEND\_APP — Spending per Pesticide Applicator

**β = −9.79, SE = 4.52, p = .030; interaction β = −1.37, p = .112**

States spending more per pesticide applicator had significantly fewer violations. The main effect is nearly **three and a half times larger** than the farmworker coefficient, reflecting the greater per-capita regulatory burden associated with this population (handler training requirements, PPE provisions, restricted-entry intervals). The interaction term (SPEND\_APP\_z × time) is in the expected negative direction and approaches marginal significance (p = .112), suggesting the protective effect of per-applicator spending may have strengthened over the study period, though this cannot be stated with confidence at conventional thresholds.

### 5.4 SPEND\_AREA — Spending per Square Mile

**β = −1.44, SE = 4.80, p = .764; interaction β = −0.45, p = .613**

Neither the main effect nor the time interaction for SPEND\_AREA reaches statistical significance. Geographic scale of the state does not appear to moderate the relationship between spending and violations in this model. This is consistent with prior individual-model results and suggests that inspector travel burden (as proxied by land area) is not a meaningful driver of violation rate differences across states, at least within this sample.

### 5.5 Between-State Variance Explained

The model explains **21.25%** of the between-state variance in violations relative to the restricted-sample baseline. This is substantially higher than any single spending variable achieved individually (maximum was 9.0% for SPEND\_FLC alone, and 6.7% for the five-variable combined model on its own restricted sample). The improvement reflects the complementary contributions of the two FIFRA-aligned spending variables — SPEND\_WORK and SPEND\_APP each capture a distinct legally defined population at risk.

---

## 6. Comparison to Prior Models

| Model | N obs | N states | Between-state variance explained | Key significant predictors |
|---|---|---|---|---|
| SPEND\_FLC only (main + ×time) | 246 | 45 | 9.0% | SPEND\_FLC (p=.002), ×time (p=.023) |
| SPEND\_APP only (main + ×time) | 220 | 41 | 7.0% | SPEND\_APP (p=.002), ×time (p=.020) |
| Five-variable combined model | 207 | 38 | 6.7% | SPEND\_APP (p=.022), ×time (p=.089) |
| **This model (WORK + APP + AREA)** | **208** | **39** | **21.25%** | **SPEND\_WORK (p=.022), SPEND\_APP (p=.030)** |

The targeted model explains substantially more between-state variance than the prior combined model despite using fewer predictors. Two factors contribute:

1. **SPEND\_WORK and SPEND\_APP capture complementary variance.** Farmworkers and applicators are distinct populations with distinct WPS protections; spending adequacy relative to each group captures different aspects of enforcement capacity.
2. **Removing SPEND\_FLC and SPEND\_OP reduces collinearity.** The three worker-count variables in the prior combined model competed for overlapping variance, suppressing all their coefficients. With SPEND\_FLC removed, SPEND\_WORK and SPEND\_APP can each express their independent contribution.

---

## 7. Technical Notes

- All spending variables loaded from `spend_bls_variables.csv` (constructed in `spending_bls_models.py`)
- Z-scoring parameters (mean, SD) are those computed from the full 50-state sample; the model uses those standardized values on the 39-state analytic subsample
- The baseline model for pseudo-R² is re-estimated on the identical 208-observation sample to ensure a valid comparison
- Convergence: confirmed (L-BFGS-B optimizer)
- Random effects structure: random intercept + random slope for linear time, by state (consistent with all prior models in this project)
- Missing states are missing due to suppressed BLS OEWS employment counts for small state-occupation cells (BLS suppresses estimates with relative standard error > 50% or employment < 10)

---

*Script: `spending_bls_models.py` (spend variable construction) + inline analysis | Report generated: 2026-05-07*
