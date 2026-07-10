# Labor Intensity & DOL H-2A Covariate Models
## WPS Enforcement Analysis — New Data Sources and Stepwise Results
**Keshav Goel | May 2026**

---

## 1. New Datasets

Five new files were added to the project. This section documents each source, its structure, and how variables are constructed from it.

### 1.1 Labor Intensity Index (3 files)

| File | Census Year | Shape | Key Column |
|------|-------------|-------|------------|
| `labor_intensity_index_2012.csv` | 2012 Census of Agriculture | 50 × 2 | `Labor_Intensity_Index` |
| `labor_intensity_index_2017.csv` | 2017 Census of Agriculture | 50 × 2 | `Labor_Intensity_Index` |
| `labor_intensity_index_2022.csv` | 2022 Census of Agriculture | 50 × 2 | `Labor_Intensity_Index` |

**What it measures**: A composite index of agricultural labor intensity per state, constructed from Census of Agriculture data. Higher values indicate states where agriculture is more labor-intensive relative to output or land.

**State name format**: UPPERCASE (e.g., `ALABAMA`) — converted to title case via `.str.title()` for merging.

**Distribution (2017)**: mean = 0.000, SD = 1.292. The index is approximately mean-zero but not unit-variance — re-standardized to true z-score before model entry.

**Cross-year correlations**:
- r(2012, 2017) = **0.977** — near-identical across census waves
- r(2017, 2022) = **0.965** — stable structural property of states

The 2022 file is outside the study period (2011–2019) and is not used in models. LII_2012 and LII_2017 are both tested in the stepwise sequence; their near-identical results confirm the index is stable.

### 1.2 DOL H-2A Workers by State, Annual (`dol_var1_workers_by_state_annual.csv`)

**Source**: U.S. Department of Labor H-2A temporary agricultural worker program records.

**Shape**: 850 rows × 7 columns (50 states × 17 years, 2008–2025).

**Columns**:

| Column | Description |
|--------|-------------|
| `state` | 2-letter state abbreviation |
| `year` | Program year |
| `workers_requested` | H-2A workers requested by employers |
| `workers_certified` | H-2A workers certified by DOL |
| `n_cases` | Number of H-2A employer applications (cases) |
| `workers_gap` | workers_certified − workers_requested |
| `demand_met_pct` | 100 × workers_certified / workers_requested |

**Coverage in study period (2011–2019)**:
- **2013 is entirely absent** from the source file — all 50 states missing for that year
- **2014**: `workers_requested = 0` for all 50 states → `demand_met_pct = inf` for all rows (data recording issue — requests not captured that year)

**Variable construction** (state-level means):
- `dol_workers_cert`: mean of `workers_certified` across 2011, 2012, 2014–2019 (2013 excluded; 2014 included since certifications are valid even without request data)
- `dol_n_cases`: mean of `n_cases` across same years
- `dol_demand_met_pct`: mean of `demand_met_pct` excluding 2013 (missing) **and** 2014 (inf values) → effectively mean of 2011, 2012, 2015–2019 (7 years)

All 50 states have non-missing values for all three constructed variables.

**Descriptive Statistics (state-level means)**:

| Variable | Mean | SD | Min | Max |
|----------|------|-----|-----|-----|
| dol_workers_cert | 3,612.6 | 5,828.1 | 22.0 | 25,733.3 |
| dol_n_cases | 225.0 | 300.2 | 1.0 | 1,282.6 |
| dol_demand_met_pct | 94.9% | 3.3% | 81.4% | 100.0% |

**Correlation with existing h2a_workers (Yuri, 2017 Census)**:
- r(dol_workers_cert, h2a_workers_2017) = **0.833** — strongly related but distinct. DOL captures annual program certifications 2011–2019; Yuri's data is a single 2017 Census snapshot. The DOL measure better reflects the average annual labor demand across the study period.

### 1.3 DOL H-2A Workers by Employer Type, Annual (`dol_var2_employer_type_annual.csv`)

**Source**: U.S. Department of Labor H-2A program records, disaggregated by employer type.

**Shape**: 300 rows × 9 columns (50 states × 6 years, **2020–2025 only**).

**⚠️ Critical limitation**: This file covers **2020–2025 only**. It does not overlap with the study period (2011–2019). The 2020 values are used as a **structural proxy** for each state's employer composition under the assumption that Farm Labor Contractor share is a relatively stable state characteristic.

**Columns**:

| Column | Description |
|--------|-------------|
| `Association / Joint Employer` | Certified workers via association/joint arrangements |
| `Farm Labor Contractor` | Certified workers via FLC intermediaries |
| `Owner / Operator (Direct)` | Certified workers placed directly by farm operators |
| `pct_Farm Labor Contractor` | % of certified workers via FLC |
| `pct_Owner / Operator (Direct)` | % via direct employer |
| `pct_Association / Joint Employer` | % via associations |

**Variable constructed**: `pct_flc` — percentage of H-2A workers placed through Farm Labor Contractors (2020).

**Why only pct_flc**: pct_flc and pct_direct have r = **−0.989** — essentially perfectly negatively correlated. Only one can enter models. pct_flc (mean = 25.9%, SD = 23.4%) has higher between-state variance and is the theoretically more interesting measure: FLC intermediaries may reduce direct employer accountability and affect WPS compliance differently than direct-hire arrangements.

---

## 2. Model Variables

Six new Level-2 (state-level, time-invariant) covariates are tested:

| Variable | Source | N states | Mean | SD |
|----------|---------|----------|------|-----|
| `lii_2017_z` | LII 2017 Census | 50 | 0.000 | 1.000 |
| `lii_2012_z` | LII 2012 Census | 50 | 0.000 | 1.000 |
| `dol_workers_cert_z` | DOL H-2A, mean 2011–2019 | 50 | 0.000 | 1.000 |
| `dol_n_cases_z` | DOL H-2A, mean 2011–2019 | 50 | 0.000 | 1.000 |
| `dol_demand_met_pct_z` | DOL H-2A, mean 2011–2019 | 50 | 0.000 | 1.000 |
| `pct_flc_z` | DOL employer type, 2020 | 50 | 0.000 | 1.000 |

All are merged onto the spending-matched base dataset (N=250 obs, 47 states for violations; N=282 obs, 47 states for inspections). No listwise deletion occurs — all 50 states have complete data on all six new variables, so base sample sizes are unchanged.

All variables are z-scored (mean=0, SD=1) before model entry, following the same convention as existing Level-2 covariates. β coefficients are therefore interpretable as the change in outcome associated with a 1 SD increase in the predictor.

---

## 3. Methodology

Identical to the existing Zimmerman stepwise approach:

- **Model structure**: `outcome ~ time + time² + time³ + predictor_z (+ predictor_z:time)` with random intercept + random slope for time by state, estimated via REML/L-BFGS-B
- **Sequence**: For each predictor, fit (a) main effect only, then (b) main + linear time interaction
- **Combined model**: All six predictors simultaneously; ×time interactions included only for variables with p < .20 in step (b)
- **Pseudo-R²**: Reduction in between-state variance (σ²_u0) relative to time-only baseline; baseline re-estimated on same analytic sample

Scripts: `labor_covariates_violations_models.py`, `labor_covariates_inspections_models.py`

---

## 4. Results — Violations DV

**Baseline M0** (N=250 obs, 47 states): σ²_u0 = 325.71, ICC = 77.2%

### 4.1 Stepwise Summary

| Predictor | N | σ²_u0 (main) | %Expl | Main β | Main p | ×time β | ×time p |
|-----------|---|--------------|-------|--------|--------|---------|---------|
| lii_2017_z | 250 | 330.98 | −1.6% | −0.386 | .766 | −0.254 | .669 |
| lii_2012_z | 250 | 330.89 | −1.6% | −0.103 | .934 | −0.257 | .662 |
| dol_workers_cert_z | 250 | 256.79 | **21.2%** | **2.793** | **.036*** | **2.158** | **<.001***|
| dol_n_cases_z | 250 | 285.97 | 12.2% | — | (conv.) | **2.126** | **<.001***|
| dol_demand_met_pct_z | 250 | 331.46 | −1.8% | 0.130 | .913 | 0.020 | .970 |
| pct_flc_z | 250 | 277.74 | **14.7%** | 2.293 | .177 | **1.451** | **.007***|

*conv. = convergence issue in main-only model (SE not estimable); ×time model converges normally.*

### 4.2 Key Individual Findings

**`dol_workers_cert_z`** is the strongest single new predictor of violations:
- Main effect: β = 2.793 (p = .036) — states with more annual H-2A certifications have more violations, holding time constant; explains 21.2% of between-state variance
- With interaction: β = 13.19 (p < .001), ×time = 2.16 (p < .001) — both the level and the time-trajectory of violations scales with H-2A volume; 25.8% of variance explained
- Interpretation: states with chronically high H-2A certification volumes not only have higher violations but their violations increased more steeply around the 2016–2017 spike

**`pct_flc_z`** (Farm Labor Contractor share):
- Main effect alone: β = 2.293 (p = .177, borderline) — 14.7% variance explained
- With interaction: β = 9.816 (p = .001), ×time = 1.451 (p = .007) — strong positive effect with significant time interaction; 16.5% explained
- Interpretation: states where more H-2A workers are placed through FLC intermediaries have substantially higher violations, and this gap grew over time. Consistent with reduced direct-employer accountability under FLC arrangements.

**`dol_n_cases_z`** (employer case volume):
- Significant in the ×time model only (p < .001 for both main and interaction); ×time = 2.126 (p < .001); 16.7% explained
- dol_n_cases and dol_workers_cert are conceptually related (more employers → more workers), so both capturing the same underlying H-2A intensity dynamic is expected

**`lii_2017_z` and `lii_2012_z`**: Neither significant in any specification (p > .60). Labor intensity as measured by the Census index does not predict violations rates after accounting for time trends and state random effects.

**`dol_demand_met_pct_z`**: Not significant (p > .90 for both main and interaction). Whether states are able to fulfill their H-2A labor demand does not predict violation rates.

### 4.3 Combined Model (N=250, interactions for dol_workers_cert, dol_n_cases, pct_flc)

| Parameter | β | SE | p |
|-----------|---|-----|---|
| Intercept | 16.459 | 2.601 | <.001** |
| time | 2.761 | 0.713 | <.001** |
| time² | −0.659 | 0.340 | .052+ |
| time³ | −0.150 | 0.051 | .003** |
| lii_2017_z | −2.529 | 6.239 | .685 |
| lii_2012_z | 2.630 | 6.124 | .668 |
| dol_workers_cert_z | **6.933** | 3.297 | **.035*** |
| dol_n_cases_z | **6.531** | 3.208 | **.042*** |
| dol_demand_met_pct_z | 1.263 | 1.185 | .287 |
| pct_flc_z | **9.330** | 2.851 | **.001***|
| dol_workers_cert_z:time | 0.801 | 0.615 | .193 |
| dol_n_cases_z:time | **1.676** | 0.607 | **.006***|
| pct_flc_z:time | **1.430** | 0.515 | **.005***|

σ²_u0 = 187.05 → **42.6% of baseline between-state variance explained**

The combined model jointly explains 42.6% of the between-state variance in violations — substantially more than any individual variable. Three predictors survive in the combined model: `dol_workers_cert_z` (volume of H-2A certifications), `dol_n_cases_z` (employer case volume), and `pct_flc_z` (FLC share). The LII variables are not significant once DOL variables are included.

---

## 5. Results — Inspections DV

**Baseline M0** (N=282 obs, 47 states): σ²_u0 = 588.36, ICC = 82.9%

### 5.1 Stepwise Summary

| Predictor | N | σ²_u0 (main) | %Expl | Main β | Main p | ×time β | ×time p |
|-----------|---|--------------|-------|--------|--------|---------|---------|
| lii_2017_z | 282 | 582.19 | 1.0% | −3.934 | .276 | 0.310 | .505 |
| lii_2012_z | 282 | 570.17 | 3.1% | −4.602 | .213 | 0.304 | .536 |
| dol_workers_cert_z | 282 | 474.78 | **19.3%** | **11.524** | **<.001***| 0.389 | .486 |
| dol_n_cases_z | 282 | 519.16 | **11.8%** | **7.411** | **.033*** | −0.087 | .872 |
| dol_demand_met_pct_z | 282 | 597.08 | −1.5% | 1.325 | .722 | −0.099 | .837 |
| pct_flc_z | 282 | 557.92 | 5.2% | 3.828 | .317 | 0.428 | .404 |

### 5.2 Key Individual Findings

**`dol_workers_cert_z`** is again the strongest single predictor:
- β = 11.524 (p < .001) — explains 19.3% of between-state variance
- No interaction (×time p = .486): the H-2A volume effect on inspections is constant over time
- Interpretation: states with more H-2A certifications have more inspections throughout the period — inspectors allocate effort proportional to H-2A labor concentration, and this allocation pattern is stable across years

**`dol_n_cases_z`**: β = 7.411 (p = .033), 11.8% explained; no interaction. Corroborates the workers_cert finding.

**`lii_2012_z`**: β = −4.602 (p = .213) — approaches but does not reach significance. The negative sign (more labor-intensive states → fewer inspections) could reflect that inspectors underserve the most labor-intensive states, but the effect is not reliable.

**`pct_flc_z`**: Not significant for inspections (p = .317, no interaction). FLC share predicts violations but not inspection rates — states with high FLC use do not receive disproportionately more or fewer inspections, despite having more violations.

**`dol_demand_met_pct_z`**: Not significant for inspections (p = .722).

### 5.3 Combined Model (N=282, main effects only — no interactions met p<.20)

| Parameter | β | SE | p |
|-----------|---|-----|---|
| Intercept | 25.616 | 3.666 | <.001** |
| time | −1.670 | 0.707 | .018* |
| time² | −0.142 | 0.380 | .708 |
| time³ | 0.013 | 0.057 | .826 |
| lii_2017_z | 12.008 | 17.241 | .486 |
| lii_2012_z | −17.048 | 17.618 | .333 |
| dol_workers_cert_z | **11.541** | 4.516 | **.011*** |
| dol_n_cases_z | −1.130 | 4.714 | .810 |
| dol_demand_met_pct_z | 0.467 | 3.925 | .905 |
| pct_flc_z | −0.112 | 4.049 | .978 |

σ²_u0 = 489.01 → **16.9% of baseline between-state variance explained**

Only `dol_workers_cert_z` remains significant in the combined model (p = .011). The LII variables are near-perfectly collinear with each other (r = 0.977) and produce inflated SEs with opposite signs — this is a sign of multicollinearity, not a real reversal. `dol_n_cases_z` loses significance once `dol_workers_cert_z` is included (the two share variance).

---

## 6. Cross-DV Comparison

| Variable | Violations %Expl | Violations Main p | Inspections %Expl | Inspections Main p |
|----------|-----------------|------------------|--------------------|-------------------|
| lii_2017_z | −1.6% | .766 | 1.0% | .276 |
| lii_2012_z | −1.6% | .934 | 3.1% | .213 |
| dol_workers_cert_z | **21.2%** | **.036*** | **19.3%** | **<.001***|
| dol_n_cases_z | 12.2% | (conv.) | **11.8%** | **.033*** |
| dol_demand_met_pct_z | −1.8% | .913 | −1.5% | .722 |
| pct_flc_z | **14.7%** | .177 | 5.2% | .317 |

### Key Contrasts

1. **`dol_workers_cert_z` is the dominant new predictor for both outcomes** (~19–21% of between-state variance explained), confirming that H-2A certification volume is a fundamental structural characteristic that shapes both how many inspections states receive and how many violations they produce.

2. **`pct_flc_z` matters for violations but not inspections.** States with higher Farm Labor Contractor share have more violations (β = 9.83, p = .001 with interaction) — consistent with FLC arrangements reducing direct accountability — but are not inspected more frequently (β = 3.83, p = .317). This gap is a potential policy-relevant finding: the mechanism through which FLC use increases violations is not more inspections, suggesting inspectors may not be targeting high-FLC states appropriately.

3. **`lii_2017_z` (Labor Intensity Index) is not significant for either outcome.** The Census-based composite index adds no explanatory power once DOL certification volumes are in the model. DOL annual program data appears to be a more precise and empirically informative measure of agricultural labor demand than the Census index.

4. **`dol_demand_met_pct_z` is not significant for either outcome.** Whether or not states successfully certify their requested workers does not predict enforcement activity or outcomes.

5. **Time interactions**: For violations, significant ×time interactions for `dol_workers_cert_z` and `pct_flc_z` indicate the relationship between H-2A volume/FLC share and violations grew over time — these structural characteristics of states became more consequential during the 2016–2017 spike period. For inspections, no such interactions were found, suggesting that the way inspectors allocate effort across states (proportional to H-2A volume) was constant throughout 2011–2019.

---

## 7. Methodological Notes

1. **DOL var2 temporal gap**: `pct_flc_z` uses 2020 data as a proxy for 2011–2019 employer composition. This assumes FLC share is a stable structural characteristic of state agricultural labor markets. Given that FLC share in 2020 (mean = 25.9%) likely reflects long-term market structure, this assumption is reasonable but should be flagged in any manuscript.

2. **DOL 2013 gap**: The entire year 2013 is absent from the H-2A program file. State-level means for `dol_workers_cert` and `dol_n_cases` are therefore averages of 8 years (2011, 2012, 2014–2019) rather than 9. This reduces precision but does not introduce bias.

3. **DOL 2014 demand_met_pct**: All 50 states show `workers_requested = 0` in 2014, making `demand_met_pct = inf`. This appears to be a data recording change (requests not logged that year). The 2014 year is excluded from `dol_demand_met_pct_mean` only; certifications and cases from 2014 are valid and included in other means.

4. **Collinearity between LII waves**: LII_2012 and LII_2017 have r = 0.977. Including both in a combined model inflates SEs for both (opposite signs, large SEs in combined model). For any final specification, use only one (recommend LII_2017 as it aligns with the census year matching the time center).

5. **Collinearity between dol_workers_cert and dol_n_cases**: In the combined inspections model, `dol_n_cases_z` drops to p = .810 once `dol_workers_cert_z` is included. The two variables share substantial variance (states with more H-2A workers also have more employer cases). `dol_workers_cert_z` is the stronger predictor and should be preferred in parsimonious specifications.

---

## 8. Output Files

| File | Description |
|------|-------------|
| `labor_covariates_violations_models.py` | Stepwise models with new variables, DV = violations |
| `labor_covariates_inspections_models.py` | Stepwise models with new variables, DV = inspections |
| `labor_covariates_report.md` | This document |

---

*Analysis conducted: May 2026*
