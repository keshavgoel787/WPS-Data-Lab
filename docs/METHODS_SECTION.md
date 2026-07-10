# Methods Section: Hierarchical Mixed-Effects Analysis of EPA Violations

## 1. Data Sources

This study utilized two primary data sources:

1. **EPA ECHO (Enforcement and Compliance History Online) Establishments Data**: Contains facility-level compliance and enforcement information aggregated to the state level, including annual violation counts for environmental regulations.

2. **WPS (Worker Protection Standard) Data**: Contains state-level data on agricultural worker protection compliance.

**Study Period**: 2011–2019 (9 years)

**Geographic Scope**: Analysis was restricted to the 50 United States. Territories (e.g., Puerto Rico, Guam), tribal nations, and regional aggregates were excluded to ensure a consistent and comparable set of jurisdictions.

---

## 2. Unit of Analysis

The unit of analysis was the **state-year** (N = 394 observations after listwise deletion of missing data). Each observation represents one state in one year, creating a panel dataset with repeated measurements nested within states.

| Metric | Value |
|--------|-------|
| Total potential observations | 450 (50 states × 9 years) |
| Observations with missing data | 56 |
| Final analytic sample | 394 |
| Number of states | 50 |
| Years per state | 4–9 (mean = 7.9) |

---

## 3. Variables

### 3.1 Outcome Variable

**Violations** (`violations`): The count of EPA violations recorded for each state in each year. This variable was extracted from the EPA ECHO dataset and converted to numeric format.

| Statistic | Value |
|-----------|-------|
| Mean | 11.62 |
| Standard Deviation | 16.26 |
| Minimum | 0 |
| Maximum | 172 |
| Median | 6 |
| Interquartile Range | 2–15 |

### 3.2 Time Variables

To model the non-linear temporal pattern in violations, we constructed a set of polynomial time variables centered on 2017—the year with the highest mean violations (20.57).

#### 3.2.1 Linear Time (`time`)

$$\text{time} = \text{year} - 2017$$

This centering ensures that:
- 2017 corresponds to time = 0
- Years after 2017 have positive values (2018 = 1, 2019 = 2)
- Years before 2017 have negative values (2016 = -1, 2015 = -2, ..., 2011 = -6)

**Rationale**: Centering on 2017 makes the intercept interpretable as the expected violations at the peak year and reduces multicollinearity among polynomial terms.

#### 3.2.2 Quadratic Time (`time2`)

$$\text{time2} = \text{time}^2$$

**Rationale**: The quadratic term captures curvilinear patterns—specifically, whether violations follow a U-shape (positive coefficient) or inverted U-shape (negative coefficient) over time.

#### 3.2.3 Cubic Time (`time3`)

$$\text{time3} = \text{time}^3$$

**Rationale**: The cubic term captures asymmetry in the time trend—whether the pattern before the reference year differs from the pattern after. This is essential for modeling data where the rise and fall are not mirror images.

#### Time Variable Summary Table

| Year | time | time² | time³ |
|------|------|-------|-------|
| 2011 | -6 | 36 | -216 |
| 2012 | -5 | 25 | -125 |
| 2013 | -4 | 16 | -64 |
| 2014 | -3 | 9 | -27 |
| 2015 | -2 | 4 | -8 |
| 2016 | -1 | 1 | -1 |
| 2017 | 0 | 0 | 0 |
| 2018 | 1 | 1 | 1 |
| 2019 | 2 | 4 | 8 |

### 3.3 Grouping Variable

**State** (`state`): A categorical variable representing the 50 U.S. states. States were coded as a factor variable (not a continuous numeric variable) to serve as the grouping/clustering variable in the hierarchical model. Each state was represented by its full name (e.g., "California", "Texas").

### 3.4 Variables Excluded

**Penalties and Actions**: Same-year enforcement variables (penalties imposed, enforcement actions taken) were deliberately excluded from the model. Including these variables would introduce endogeneity, as enforcement actions in year *t* are responses to violations in year *t*, creating a simultaneity problem that would bias estimates and obscure the causal structure.

---

## 4. Data Preparation

### 4.1 Data Reshaping

The original EPA ECHO data was in wide format (one row per state, separate columns for each year's violations). We reshaped this to long format (one row per state-year) to facilitate panel data analysis.

```
Wide format: State | violations-2011 | violations-2012 | ... | violations-2019
Long format: State | Year | Violations
```

### 4.2 Filtering

Data were filtered to include only the 50 U.S. states using an explicit list of state names. Observations for territories, tribal nations, and regional aggregates were removed.

### 4.3 Missing Data

Observations with missing violation counts (n = 56) were excluded via listwise deletion, resulting in a final analytic sample of 394 state-year observations.

### 4.4 Variable Transformations

| Variable | Original Form | Transformation | Final Form |
|----------|---------------|----------------|------------|
| State | String (state name) | Converted to categorical | Factor with 50 levels |
| Year | Integer (2011-2019) | Centered on 2017 | `time = year - 2017` |
| Violations | Mixed (some non-numeric) | Coerced to numeric | Continuous count |

---

## 5. Analytic Approach

### 5.1 Rationale for Hierarchical Modeling

The data exhibit a nested structure: state-year observations (Level 1) are clustered within states (Level 2). Standard ordinary least squares (OLS) regression assumes independence of observations, which is violated when multiple observations come from the same state. Ignoring this clustering would:

1. Underestimate standard errors
2. Inflate Type I error rates
3. Fail to account for state-specific heterogeneity

A hierarchical (multilevel) mixed-effects model addresses these issues by:
- Partitioning variance into between-state and within-state components
- Allowing state-specific intercepts and slopes
- Producing valid standard errors and confidence intervals

### 5.2 Model Specification

We estimated two nested models using restricted maximum likelihood (REML):

#### Model 1: Random Intercept Model

$$\text{violations}_{it} = \gamma_{00} + \gamma_{10}\text{time}_{it} + \gamma_{20}\text{time}_{it}^2 + \gamma_{30}\text{time}_{it}^3 + u_{0i} + e_{it}$$

Where:
- $\gamma_{00}$ = fixed intercept (national average at time = 0)
- $\gamma_{10}, \gamma_{20}, \gamma_{30}$ = fixed effects for time, time², time³
- $u_{0i}$ = random intercept for state *i* ~ N(0, $\tau_{00}$)
- $e_{it}$ = residual error ~ N(0, $\sigma^2$)

#### Model 2: Random Intercept and Random Slope Model (Preferred)

$$\text{violations}_{it} = \gamma_{00} + \gamma_{10}\text{time}_{it} + \gamma_{20}\text{time}_{it}^2 + \gamma_{30}\text{time}_{it}^3 + u_{0i} + u_{1i}\text{time}_{it} + e_{it}$$

Where:
- $u_{0i}$ = random intercept for state *i*
- $u_{1i}$ = random slope for time for state *i*
- Random effects are assumed to follow a multivariate normal distribution:

$$\begin{pmatrix} u_{0i} \\ u_{1i} \end{pmatrix} \sim N\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} \tau_{00} & \tau_{01} \\ \tau_{01} & \tau_{11} \end{pmatrix}\right)$$

### 5.3 Model Notation (Alternative Form)

Using mixed-model notation:

**Model 1**: `violations ~ time + time2 + time3 + (1 | state)`

**Model 2**: `violations ~ time + time2 + time3 + (1 + time | state)`

### 5.4 Estimation

Models were estimated using the `MixedLM` function from Python's `statsmodels` library with:
- **Estimation method**: Restricted Maximum Likelihood (REML)
- **Optimizer**: L-BFGS-B algorithm

### 5.5 Model Comparison

Models were compared using:
- Log-likelihood values
- Likelihood ratio test (LRT) for nested models
- Akaike Information Criterion (AIC)
- Bayesian Information Criterion (BIC)

### 5.6 Variance Decomposition

The intraclass correlation coefficient (ICC) was calculated to quantify the proportion of total variance attributable to between-state differences:

$$\text{ICC} = \frac{\tau_{00}}{\tau_{00} + \sigma^2}$$

Where:
- $\tau_{00}$ = between-state variance (random intercept variance)
- $\sigma^2$ = within-state (residual) variance

---

## 6. Results

### 6.1 Descriptive Statistics

#### Violations by Year

| Year | Mean | SD | Min | Max | N |
|------|------|-----|-----|-----|---|
| 2011 | 6.66 | 6.15 | 1 | 24 | 44 |
| 2012 | 8.46 | 7.18 | 1 | 30 | 41 |
| 2013 | 7.00 | 6.04 | 1 | 22 | 36 |
| 2014 | 6.83 | 5.85 | 1 | 22 | 36 |
| 2015 | 5.96 | 6.86 | 0 | 26 | 49 |
| 2016 | 14.19 | 26.90 | 1 | 172 | 47 |
| 2017 | 20.57 | 23.59 | 1 | 103 | 47 |
| 2018 | 16.67 | 18.67 | 1 | 68 | 46 |
| 2019 | 15.54 | 14.83 | 1 | 54 | 48 |

Violations exhibited a non-linear pattern over time, with low and stable counts from 2011–2015 (mean range: 5.96–8.46), a sharp increase in 2016–2017 (peak mean: 20.57 in 2017), followed by a gradual decline in 2018–2019.

### 6.2 Model Results

#### 6.2.1 Fixed Effects (Model 2)

| Parameter | Estimate | SE | z | p-value | 95% CI |
|-----------|----------|-----|---|---------|--------|
| Intercept | 16.21 | 2.17 | 7.47 | <.001 | [11.95, 20.46] |
| time | 2.19 | 0.49 | 4.45 | <.001 | [1.23, 3.16] |
| time² | -0.95 | 0.27 | -3.52 | <.001 | [-1.49, -0.42] |
| time³ | -0.18 | 0.04 | -4.21 | <.001 | [-0.27, -0.10] |

**Interpretation**:
- **Intercept (16.21)**: At the reference year (2017, time = 0), the expected number of violations for an average state was 16.21.
- **time (2.19)**: The linear component indicates violations were increasing at a rate of 2.19 per year at the reference point.
- **time² (-0.95)**: The negative quadratic coefficient indicates an inverted U-shaped curve, with violations peaking around the center of the time series.
- **time³ (-0.18)**: The negative cubic coefficient indicates asymmetry—the post-2017 decline was steeper than the pre-2017 rise.

#### 6.2.2 Random Effects Variance Components (Model 2)

| Component | Variance | Interpretation |
|-----------|----------|----------------|
| Random Intercept ($\tau_{00}$) | 188.72 | Between-state variance in baseline violations |
| Random Slope ($\tau_{11}$) | 3.25 | Between-state variance in time trends |
| Covariance ($\tau_{01}$) | 24.78 | Covariance between intercepts and slopes |
| Residual ($\sigma^2$) | 113.62 | Within-state (unexplained) variance |

**Interpretation**:
- Substantial between-state variance exists in baseline violation rates (188.72)
- States also differ in their temporal trajectories (slope variance = 3.25)
- The positive covariance (24.78) indicates that states with higher baseline violations tend to have steeper increasing trends

#### 6.2.3 Intraclass Correlation Coefficient

$$\text{ICC} = \frac{188.72}{188.72 + 113.62} = 0.624$$

**Interpretation**: 62.4% of the total variance in violations is attributable to between-state differences. This high ICC strongly justifies the use of hierarchical modeling, as ignoring the state-level clustering would substantially bias inference.

### 6.3 Model Comparison

| Metric | Model 1 (RI) | Model 2 (RI + RS) |
|--------|--------------|-------------------|
| Log-Likelihood | -1581.04 | -1549.71 |
| Parameters | 6 | 8 |
| Residual Variance | 140.01 | 113.62 |

**Likelihood Ratio Test**:
$$\chi^2 = 2 \times (-1549.71 - (-1581.04)) = 62.65, \quad df = 2, \quad p < .001$$

The likelihood ratio test indicates that Model 2 (with random slopes) fits significantly better than Model 1, confirming that states differ not only in their baseline violation rates but also in their temporal trajectories.

### 6.4 State-Specific Effects (BLUPs)

Best Linear Unbiased Predictors (BLUPs) were estimated for each state's random intercept and random slope.

#### States with Highest Baseline Violations

| State | Random Intercept | Random Slope |
|-------|------------------|--------------|
| North Carolina | +48.63 | +6.38 |
| Pennsylvania | +26.48 | +3.48 |
| California | +25.09 | +3.29 |
| Illinois | +24.46 | +3.21 |
| Texas | +22.34 | +2.93 |

#### States with Lowest Baseline Violations

| State | Random Intercept | Random Slope |
|-------|------------------|--------------|
| Wyoming | -11.03 | -1.45 |
| Kentucky | -11.51 | -1.51 |
| Maine | -11.56 | -1.52 |
| Vermont | -11.60 | -1.52 |

### 6.5 Predicted National Trend

Using fixed effects only, the model-predicted national trend was:

| Year | Predicted | Observed Mean | Residual |
|------|-----------|---------------|----------|
| 2011 | 8.28 | 6.66 | -1.62 |
| 2012 | 4.30 | 8.46 | +4.16 |
| 2013 | 3.90 | 7.00 | +3.10 |
| 2014 | 5.99 | 6.83 | +0.84 |
| 2015 | 9.47 | 5.96 | -3.51 |
| 2016 | 13.24 | 14.19 | +0.95 |
| 2017 | 16.21 | 20.57 | +4.36 |
| 2018 | 17.26 | 16.67 | -0.59 |
| 2019 | 15.32 | 15.54 | +0.22 |

---

## 7. Summary

A hierarchical mixed-effects model was used to analyze EPA violations across 50 U.S. states from 2011 to 2019. The model included:

- **Fixed effects**: Linear, quadratic, and cubic time terms (centered on 2017)
- **Random effects**: State-specific intercepts and slopes for time

Key findings:
1. Violations followed an asymmetric inverted U-shaped pattern, peaking in 2017
2. 62.4% of variance was between states (ICC = 0.624)
3. States differed significantly in both baseline violation rates and temporal trajectories
4. North Carolina exhibited the highest violations; Vermont the lowest

The hierarchical approach was essential given the nested data structure and high intraclass correlation.

---

## 8. Software

All analyses were conducted in Python 3.x using:
- `pandas` (v1.x) for data manipulation
- `numpy` (v1.x) for numerical operations
- `statsmodels` (v0.13+) for mixed-effects model estimation
- `matplotlib` and `seaborn` for visualization

---

## 9. Limitations

1. **Missing data**: 56 observations (12.4%) were excluded due to missing violation counts; results assume data are missing at random.

2. **Model convergence**: The random slope model did not fully converge; estimates should be interpreted with caution.

3. **Distributional assumptions**: Violations are count data; a generalized linear mixed model (e.g., Poisson or negative binomial) may be more appropriate than the linear model used.

4. **Outliers**: North Carolina exhibited substantially higher violations than other states, potentially influencing model estimates.

5. **Temporal scope**: The 9-year window may not capture longer-term trends or cyclical patterns in regulatory enforcement.
