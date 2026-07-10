# Adding Agricultural Spending to the Hierarchical Model

## Overview

This document describes the addition of federal agricultural spending data to the existing hierarchical mixed-effects model for EPA violations. It covers where the spending data comes from, how it was cleaned and inflation-adjusted, how it was integrated into the model, and what the results show.

---

## 1. Why Add Spending?

The baseline model captures *when* violations occur (time trend) and *where* they occur (state variation), but it does not capture any state-level resources that might explain the variation. Agricultural spending — federal grants allocated to state agriculture departments for pesticide regulation, worker protection, and compliance — is a plausible predictor of violations: states that receive more funding may have better enforcement infrastructure, or conversely, higher funding may reflect a higher pre-existing violation burden.

Adding spending allows us to ask: **after accounting for time trends and state-level heterogeneity, does federal agricultural spending predict the number of violations?**

---

## 2. Data Source

**File**: `spending_data_master(in) (1).csv`

| Attribute | Detail |
|-----------|--------|
| **Source** | Federal grant obligation records |
| **CFDA Numbers** | 66.7 (Pesticide Programs), 66.714 (IPM Grants), 66.716 (Pesticide Risk Reduction) |
| **Raw rows** | 1,420 |
| **Year range in file** | 2008–2024 |
| **Unit** | Individual grant/award per recipient per year |

Each row represents a single grant obligation to a recipient (state agency, university, tribal organization, etc.), identified by a state code and year.

---

## 3. Spending Data Processing

### Step 1 — Filter to Study Years (2011–2019)

The spending file covers 2008–2024, but the violations model covers only 2011–2019. Rows outside this range were dropped.

```
Raw rows:              1,420
After year filter:       700 rows (2011–2019)
```

### Step 2 — Map State Abbreviations to Full Names

The spending data uses 2-letter state abbreviations (e.g., `CA`, `TX`). The violations data uses full state names (e.g., `California`, `Texas`). A complete abbreviation-to-name lookup was applied so the two datasets could be joined.

### Step 3 — Filter to 50 US States

Rows were kept only if the state code mapped to one of the 50 US states. No rows were dropped at this step — all state codes in the 2011–2019 file belonged to US states.

```
After state filter:    700 rows, 47 unique states
```

Note: **Rhode Island, Mississippi, and West Virginia** had zero spending records in the file across all years and are entirely absent from the spending dataset.

### Step 4 — Handle Negative Obligations

Six rows in the full file had negative `Total Obligation` values. These represent **deobligations** — cases where previously awarded funds were returned or clawed back. Rather than dropping them, they were retained so that the aggregated state-year total reflects **net** spending. Three of these negative rows fell within 2011–2019.

### Step 5 — Aggregate to State-Year Level

Each state-year can have multiple rows (multiple grants). All `Total Obligation` values were summed within each state-year to produce one spending figure per state per year.

```
State-year observations after aggregation: 282
(out of a possible 47 states × 9 years = 423)
```

Many state-years are missing because not every state received a grant in every year.

### Step 6 — Inflation Adjustment to 2017 Dollars

Because nominal dollar amounts from different years are not directly comparable (a dollar in 2011 is worth more than a dollar in 2019), all spending figures were converted to **constant 2017 dollars** using the Consumer Price Index for All Urban Consumers (**CPI-U**) annual averages published by the U.S. Bureau of Labor Statistics.

**Formula:**

```
spending_2017 = spending_nominal × (CPI_2017 / CPI_year)
```

Where 2017 is the base year (deflator = 1.0), earlier years have deflators > 1 (inflating the nominal amount upward), and later years have deflators < 1 (deflating downward).

**CPI-U Deflators Applied:**

| Year | CPI-U | Deflator | Interpretation |
|------|-------|----------|----------------|
| 2011 | 224.939 | **1.0897** | 2011 dollars inflated up by 8.97% to reach 2017 value |
| 2012 | 229.594 | **1.0676** | 2012 dollars inflated up by 6.76% |
| 2013 | 232.957 | **1.0522** | 2013 dollars inflated up by 5.22% |
| 2014 | 236.736 | **1.0354** | 2014 dollars inflated up by 3.54% |
| 2015 | 237.017 | **1.0342** | 2015 dollars inflated up by 3.42% |
| 2016 | 240.007 | **1.0213** | 2016 dollars inflated up by 2.13% |
| **2017** | **245.120** | **1.0000** | Base year — no adjustment |
| 2018 | 251.107 | **0.9762** | 2018 dollars deflated down by 2.38% |
| 2019 | 255.657 | **0.9588** | 2019 dollars deflated down by 4.12% |

*Source: U.S. Bureau of Labor Statistics, CPI-U, All Urban Consumers, annual averages.*

### Step 7 — Scale to Millions of Dollars

After adjustment, spending figures were divided by 1,000,000 to express them in **millions of 2017 dollars** (`spending_2017m`). This makes the regression coefficient interpretable as the change in violations associated with each additional $1 million in spending, rather than per dollar.

---

## 4. Spending Descriptive Statistics

### By Year (2017 $M, states with data only)

| Year | Mean ($M) | Total ($M) | Min ($M) | Max ($M) | N States |
|------|-----------|------------|----------|----------|----------|
| 2011 | 0.453 | 18.37 | 0.00 | 2.79 | 44 |
| 2012 | 0.497 | 18.25 | 0.00 | 2.98 | 40 |
| 2013 | 0.336 | 10.06 | 0.00 | 0.97 | 31 |
| 2014 | 0.425 | 12.41 | 0.00 | 1.51 | 31 |
| 2015 | 0.427 | 12.81 | 0.00 | 2.05 | 30 |
| 2016 | 0.422 | 12.88 | 0.00 | 1.59 | 32 |
| 2017 | 0.496 | 12.45 | 0.00 | 1.50 | 27 |
| 2018 | 0.451 | 11.07 | 0.00 | 1.15 | 25 |
| 2019 | 0.502 | 11.50 | 0.00 | 1.42 | 22 |

**Notable pattern**: The number of states receiving spending declines sharply after 2012, suggesting a reduction in grant activity over the study period.

### Overall (2017 $M, analytic sample)

| Statistic | Value |
|-----------|-------|
| Mean | $0.446M |
| Std Dev | $0.473M |
| Min | $0.000M |
| Max | $2.975M |
| Median | $0.333M |

---

## 5. Missing Data After Merging

After merging spending onto the violations dataset (394 obs), **144 observations were missing spending** — either because the state had no grants in that year, or the state had no records at all (Rhode Island, Mississippi, West Virginia).

**States entirely absent from spending data:**
- Rhode Island (all years missing)
- Mississippi (all years missing)
- West Virginia (all years missing)

**States with partial missing (selected examples):**

| State | Years Missing Spending |
|-------|----------------------|
| Iowa | 2013–2019 |
| Missouri | 2013–2019 |
| Tennessee | 2012–2019 |
| North Carolina | 2017–2019 |
| Pennsylvania | 2017–2019 |
| Connecticut | 2012–2015, 2017–2019 |

These missing observations were dropped via listwise deletion, producing a final analytic sample of **250 observations across 47 states**.

> **Important implication**: The baseline model (time terms only) was also re-estimated on this reduced 250-observation sample so that Model 1 and Model 2 are directly comparable on the same data. Results from this baseline differ slightly from the original 394-observation estimates.

---

## 6. Updated Model Specification

Two models were fit on the matched analytic sample (N = 250, 47 states):

### Model 1 — Baseline (Time Only, Matched Sample)

```
violations ~ time + time² + time³ + (1 + time | state)
```

This replicates the original hierarchical model on the spending-matched subsample, providing a fair comparison baseline.

### Model 2 — With Spending

```
violations ~ time + time² + time³ + spending_2017m + (1 + time | state)
```

`spending_2017m` is the state's total federal agricultural grant obligation in that year, expressed in millions of 2017 dollars.

Both models use:
- **Grouping variable**: state (47 categories, categorical)
- **Random effects**: random intercept + random slope for time, by state
- **Estimation**: REML via L-BFGS-B

---

## 7. Results

### Fixed Effects

| Parameter | Model 1 (No Spending) | Model 2 (+ Spending) |
|-----------|----------------------|----------------------|
| **Intercept** | 17.46 (SE=3.20, p<.001) | 17.82 (SE=3.24, p<.001) |
| **time** | 2.97 (SE=0.78, p<.001) | 2.92 (SE=0.79, p<.001) |
| **time²** | -0.63 (SE=0.35, p=.071) | -0.63 (SE=0.35, p=.068) |
| **time³** | -0.15 (SE=0.05, p=.005) | -0.15 (SE=0.05, p=.005) |
| **spending_2017m** | — | **-1.24 (SE=1.76, p=.483)** |

### Variance Components

| Component | Model 1 | Model 2 |
|-----------|---------|---------|
| Random intercept variance | 325.71 | 325.86 |
| Random slope variance | 8.52 | 8.49 |
| Intercept–slope covariance | 51.84 | 51.75 |
| Residual variance | 95.99 | 96.17 |
| **ICC** | **77.2%** | **77.2%** |

### Model Comparison

| Metric | Model 1 | Model 2 |
|--------|---------|---------|
| Log-Likelihood | -974.99 | -973.32 |
| Δ Log-Likelihood | — | +1.67 |
| LR χ² (df=1) | — | 3.34 |
| p-value | — | .068 |

---

## 8. Interpretation of Results

### Spending Coefficient (β = -1.24, p = .483)

The spending coefficient is **negative** in direction and **not statistically significant**.

- **Direction (negative)**: States that received more federal agricultural funding in a given year had *slightly fewer* violations, on average, after accounting for time trends and state-level random effects. This is in the expected direction — more resources could improve compliance infrastructure and reduce violations.

- **Magnitude**: Each additional $1 million in 2017-adjusted spending is associated with approximately 1.24 fewer violations, holding the time trend constant. Given a mean violation count of about 10 in the analytic sample, this would represent a ~12% reduction per $1M — a substantively meaningful effect if real.

- **Significance (p = .483)**: The effect is not statistically significant. We cannot rule out that the observed association is due to chance. The confidence interval for the spending coefficient spans from -4.69 to +2.22 — wide enough to be consistent with both a meaningful negative effect and a small positive one.

### Why Spending May Not Be Significant

Several factors may contribute to the null result:

1. **Missing data**: 37% of violation observations (144/394) were dropped because spending data was missing. This substantially reduces power and may introduce selection bias if the states/years with spending data are not representative.

2. **Sparse coverage in later years**: The number of states with spending records declines sharply after 2012. This means spending is least observed precisely in the years closest to the 2017 violation spike — years where the relationship may be most informative.

3. **Aggregation mismatch**: Spending is summed across all grant types (pesticide enforcement, IPM, worker protection), some of which may not directly affect violation counts. More targeted spending variables might show stronger effects.

4. **Lagged effects**: Spending in year *t* may reduce violations in year *t+1* or *t+2*, not the same year. A contemporaneous relationship may understate the true effect.

### Time Terms Remain Robust

The time trend coefficients (time, time², time³) are nearly unchanged between Model 1 and Model 2, confirming that the polynomial time structure is stable and not sensitive to the addition of spending. The non-linear inverted U-shape centered on 2017 holds regardless of whether spending is in the model.

### ICC Remains High

The ICC stays at **77.2%** after adding spending, indicating that spending does not explain away the between-state variance. Most of the state-level heterogeneity in violations is not accounted for by federal spending alone.

---

## 9. State-Specific Random Effects (Model 2 BLUPs)

### Highest Baseline Violations

| State | Random Intercept | Random Slope |
|-------|-----------------|--------------|
| North Carolina | +80.26 | +13.05 |
| Illinois | +42.01 | +6.63 |
| California | +25.43 | +4.02 |
| Texas | +23.49 | +3.86 |
| Florida | +21.93 | +2.97 |

### Lowest Baseline Violations

| State | Random Intercept | Random Slope |
|-------|-----------------|--------------|
| Maine | -13.96 | -2.21 |
| Vermont | -14.07 | -2.23 |
| New Hampshire | -13.22 | -2.06 |
| New Mexico | -13.15 | -1.99 |
| Wyoming | -12.70 | -2.03 |

---

## 10. Sample Comparison: Full vs. Spending-Matched

The inclusion of spending data changes the analytic sample significantly. For transparency:

| Attribute | Original Model | With Spending |
|-----------|---------------|---------------|
| Observations | 394 | 250 |
| States | 50 | 47 |
| Mean violations | 11.62 | 10.09 |
| ICC | 62.4% | 77.2% |
| Intercept (2017 baseline) | 16.21 | 17.46–17.82 |

The reduction in sample size and the shift in ICC suggest the subsample of states with spending data differs from the full sample — states without spending records (Rhode Island, Mississippi, West Virginia, and many states in later years) may systematically differ in violation rates and enforcement patterns. This limits the generalizability of the spending model.

---

## 11. Summary

Federal agricultural spending was added to the hierarchical mixed-effects model as a fixed effect after being inflation-adjusted to 2017 dollars using CPI-U annual averages (BLS). Spending was aggregated from grant-level records to the state-year level and expressed in millions of dollars for interpretability.

**Key findings:**
- The spending coefficient is **negative** (-1.24 per $1M), suggesting more spending is associated with fewer violations — a theoretically plausible direction
- The effect is **not statistically significant** (p = .483), likely due to substantial missing data in the spending file that reduces the sample from 394 to 250 observations
- The **time trend is unchanged** by adding spending, confirming the robustness of the polynomial time structure
- The **ICC remains at 77.2%**, indicating that spending does not explain state-level heterogeneity in violations

The spending variable should be interpreted cautiously in its current form. Future work could explore lagged spending effects, disaggregated spending by grant type, or imputation strategies for the missing state-years.

---

## 12. Files

| File | Description |
|------|-------------|
| `spending_data_master(in) (1).csv` | Raw spending data (grant-level, 2008–2024) |
| `spending_aggregated.csv` | Aggregated and inflation-adjusted spending (state-year, 2017 $M) |
| `hierarchical_violations_model.py` | Updated model script including spending pipeline |
| `model_data_long.csv` | Final analytic dataset (violations + spending merged) |

---

*CPI-U Source: U.S. Bureau of Labor Statistics, Consumer Price Index for All Urban Consumers*
*Spending Source: Federal grant obligation records (CFDA 66.7, 66.714, 66.716)*
*Analysis conducted: February 2026*
