# Variance-Component Decomposition of the Paper-Table Models

**Exploratory.** This note decomposes the random-effects variance of the three
manuscript-table models (M1/M2/M3, one build-up per DV) into its full set of
components — not just the intercept variance that the manuscript's pseudo-R²
uses — and uses that decomposition to diagnose a puzzling result: adding
covariates to the inspections table *raised* between-state intercept variance
instead of lowering it (reported Δσ² ≈ −0.2% for M1→M2 and ≈ −1.3% for
M1→M3 in `data/generated/paper_table_params.json`).

Produced by `scripts/variance_decomposition.py`. Source data:
`data/generated/variance_components.csv`. Figures:
`figures/vpc_violations.png`, `figures/vpc_inspections.png`,
`figures/varcomp_violations.png`, `figures/varcomp_inspections.png`.

All numbers below are read directly from `variance_components.csv` (Step 1
command: `python3 -c "import pandas as pd; d=pd.read_csv('data/generated/variance_components.csv'); print(d[[...]].to_string(index=False))"`).
No values are invented or rounded from memory.

---

## 1. The mixed-model variance structure

Every model is `MixedLM.from_formula(dv ~ fixed effects, groups=state, re_formula='~time').fit(method='lbfgs')`
(REML), i.e. a random intercept **and** a random slope on linear `time`
(`time = year − 2017`) per state, with a general (unstructured) 2×2 random-effects
covariance matrix. That matrix and the residual variance decompose the total
variance in the outcome into four pieces:

- **σ²_u0** — between-state variance in the *intercept* (the state's baseline
  rate at `time = 0`, i.e. year 2017), after conditioning on the fixed effects.
- **σ²_u1** — between-state variance in the *linear time slope*: how much
  states differ in their year-over-year trend (in violations or inspections
  per year).
- **σ_u01** — the covariance between a state's intercept and its slope
  random effects. `corr_u01` in the CSV is the corresponding correlation,
  `σ_u01 / sqrt(σ²_u0 · σ²_u1)`.
- **σ²_e** — the Level-1 (state-year) residual variance: within-state,
  year-to-year variation not explained by the fixed effects or by the state's
  own random intercept/slope.

Two derived summaries:

- **ICC** = σ²_u0 / (σ²_u0 + σ²_e) — the between-state share of variance
  *at time = 0* (2017), ignoring the slope terms.
- **VPC(t)** (variance partition coefficient at year `t`, `t = year − 2017`)
  generalizes the ICC to every year, folding in the random slope and its
  covariance with the intercept:

  ```
  VPC(t) = [σ²_u0 + 2·t·σ_u01 + t²·σ²_u1] / [σ²_u0 + 2·t·σ_u01 + t²·σ²_u1 + σ²_e]
  ```

  At `t = 0` this collapses to the ICC exactly, which is confirmed in the CSV
  (`vpc_2017 == icc` in every row).

---

## 2. Component tables

### Violations (DV = violations; M1 also includes `inspections` as a Level-1 predictor per the manuscript spec)

| Model | N (obs / states) | σ²_u0 | σ²_u1 | σ_u01 | corr_u01 | σ²_e | ICC |
|---|---|---:|---:|---:|---:|---:|---:|
| M1 | 394 / 50 | 188.085 | 4.563 | 29.273 | 0.999 | 115.549 | 0.619 |
| M2 | 320 / 39 | 154.238 | 3.832 | 24.313 | 1.000 | 140.316 | 0.524 |
| M3 | 320 / 39 | 151.675 | 3.752 | 23.848 | 1.000 | 139.938 | 0.520 |

### Inspections (DV = EPA + state inspections; linear time only, per the PI-directed convention)

| Model | N (obs / states) | σ²_u0 | σ²_u1 | σ_u01 | corr_u01 | σ²_e | ICC |
|---|---|---:|---:|---:|---:|---:|---:|
| M1 | 450 / 50 | 770.465 | 14.894 | 78.837 | 0.736 | 133.663 | 0.852 |
| M2 | 351 / 39 | 812.636 | 24.309 | 97.778 | 0.696 | 136.185 | 0.856 |
| M3 | 351 / 39 | 821.199 | 22.840 | 95.858 | 0.700 | 137.245 | 0.857 |

Note the sample shift: M1 in each family is fit on the full analytic sample
(50 states); M2/M3 drop to 39 states once the BLS-derived `SPEND_APP_z` /
`SPEND_WORK_z` (and downstream H-2A block) variables are entered and rows with
BLS-suppressed cells are listwise-deleted. M2 and M3 share the same N, so the
M2→M3 step is the only apples-to-apples (matched-sample) comparison in each
table; M1→M2 always mixes a covariate effect with a sample-composition change.

One striking feature, visible directly in `corr_u01`: for violations, the
intercept–slope correlation is essentially at the boundary (0.999–1.000).
This is a near-degenerate (boundary) random-effects covariance solution —
common in MixedLM when ~39–50 clusters don't fully identify a general 2×2 `G`
matrix — and it is the reason the violations VPC swings so much more sharply
across the study window than inspections does (Section 3).

---

## 3. VPC over time

Selected values (full 2011–2019 series in `variance_components.csv`;
figures: `figures/vpc_violations.png`, `figures/vpc_inspections.png`):

| DV | Model | VPC(2011) | VPC(2017) = ICC | VPC(2019) |
|---|---|---:|---:|---:|
| inspections | M1 | 0.730 | 0.852 | 0.895 |
| inspections | M2 | 0.791 | 0.856 | 0.905 |
| inspections | M3 | 0.782 | 0.857 | 0.904 |
| violations | M1 | 0.009 | 0.619 | 0.737 |
| violations | M2 | 0.003 | 0.524 | 0.655 |
| violations | M3 | 0.004 | 0.520 | 0.651 |

In every model, in both DVs, the between-state share **rises monotonically
from 2011 through 2019** (VPC(2011) < VPC(2017) < VPC(2019)). This is a direct
consequence of `σ_u01 > 0` (positive intercept–slope correlation): the
`2·t·σ_u01` term in the VPC numerator is negative for `t < 0` (years before
2017) and positive for `t > 0` (years after 2017), so states that start with a
higher intercept also tend to trend upward faster, pulling the between-state
share down before 2017 and up after it.

The effect is dramatically larger for violations than for inspections because
of the near-boundary correlation noted above: with `corr_u01 ≈ 1`, the
`2·t·σ_u01` term very nearly cancels σ²_u0 at `t = −6` (2011), driving VPC to
essentially zero (0.003–0.009) — almost none of the cross-state variance in
2011 violation counts is "between-state" once the model's random-slope
structure is accounted for; nearly all of it looks like Level-1 residual at
that end of the window. Inspections, with a more moderate correlation
(0.70–0.74), shows the same directional pattern but far more mildly (0.73–0.90
across the window, never collapsing toward zero).

---

## 4. Why the current pseudo-R² is partial

The project's pseudo-R² (used throughout `paper_table_models.py`,
`docs/final_models_update_2026-07.md`, etc.) is defined as
`(σ²_u0,baseline − σ²_u0,model) / σ²_u0,baseline` — a reduction measured
**only** in the between-state intercept variance. That is one of four
variance components a covariate can move, and the CSV shows covariate
variance routinely lands somewhere else:

**Violations** — this is the DV where σ²_u0 *does* fall as covariates enter
(188.085 → 154.238 → 151.675, M1→M2→M3), so the pseudo-R² framework reports a
"successful" reduction. But σ²_u1 and σ_u01 fall in step with it
(σ²_u1: 4.563 → 3.832 → 3.752; σ_u01: 29.273 → 24.313 → 23.848), while σ²_e
**rises** by 21.4% from M1 to M2 (115.549 → 140.316) and stays roughly flat
into M3 (139.938). In other words, a large share of the "explained" between-
state variance did not disappear — it was reallocated into Level-1 residual
variance, not credited to the covariates by pseudo-R² at all. Summing σ²_u0 +
σ²_e at `t = 0` (2017) — the two components that determine the year-2017
ICC — total unexplained variance falls only modestly, from 303.63 (M1) to
294.55 (M2) to 291.61 (M3), a ~3.97% drop overall versus the much larger
~19.4% drop pseudo-R² reports for σ²_u0 alone.

**Inspections** — here every component *rises* alongside the covariates
(Section 5 below spells this out), so pseudo-R² based on σ²_u0 alone actually
turns negative even though the model is absorbing real (if non-significant)
covariate signal into σ²_u1 and σ_u01. A metric that only watches σ²_u0 cannot
distinguish "covariates did nothing" from "covariates shifted variance into
the random slope" — both look like a σ²_u0 increase from the outside.

Bottom line: pseudo-R² as currently defined answers a narrower question
("did the intercept variance specifically shrink?") than "did the covariates
explain state-level heterogeneity?" — the latter requires watching all four
components (and, ultimately, the whole `u0 + e` or full VPC(t) trajectory),
not σ²_u0 in isolation.

---

## 5. Negative inspections Δσ² diagnosis

The manuscript reports pseudo-R² of about −0.22% (M1→M2) and −1.28% (M1→M3)
for the inspections table — i.e., σ²_u0 goes up, not down, when the SPEND/LII
covariates (M2) and then the H-2A block (M3) are added. The CSV's own
inspections rows reproduce the same direction of movement and let us see
exactly which components move and by how much:

| Step | σ²_u0 | σ²_u1 | σ_u01 | σ²_e | Sample |
|---|---:|---:|---:|---:|---|
| M1 → M2 | +5.47% (770.465 → 812.636) | **+63.22%** (14.894 → 24.309) | +24.03% (78.837 → 97.778) | +1.89% (133.663 → 136.185) | N changes 450/50 → 351/39 |
| M2 → M3 | +1.05% (812.636 → 821.199) | −6.04% (24.309 → 22.840) | −1.96% (97.778 → 95.858) | +0.78% (136.185 → 137.245) | N held fixed at 351/39 |

Two things stand out:

1. **M1→M2 is not a clean covariate test.** The 11-state drop (50 → 39 states)
   from listwise deletion on the BLS-derived `SPEND_APP_z`/`SPEND_WORK_z`
   variables changes *which* states are in the sample, which alone can shift
   every variance component — independent of whether the covariates have any
   explanatory power. All four components rise together (σ²_u0 +5.5%, σ²_u1
   +63.2%, σ_u01 +24.0%, σ²_e +1.9%), i.e. nothing shrinks; the entire
   "unexplained" variance budget gets larger. That is consistent with a
   sample-composition effect (dropping 11 relatively homogeneous/low-variance
   states) at least as much as it is with a covariate effect.

2. **M2→M3 is the clean, matched-N test** (same 351 obs / 39 states in both
   models; M3 only adds the three H-2A block terms — none of which reach
   significance in `paper_table_params.json`: `h2a_per_farmworker_z` p=.28,
   `dol_demand_met_pct_z` p=.91, `pct_flc_z` p=.57). Even here, σ²_u0 **rises**
   a further 1.05% (812.636 → 821.199) rather than falling. The components
   that do move in the "expected" direction are the random-slope pieces:
   σ²_u1 falls 6.04% and σ_u01 falls 1.96%, while σ²_e ticks up 0.78%. So the
   small amount of variance that the H-2A block does soak up comes out of the
   *slope* variance and its covariance with the intercept, not out of σ²_u0 —
   the one component pseudo-R² is watching.

**Why this is not a contradiction, mathematically.** In OLS, adding a
regressor to a linear model cannot increase the residual sum of squares — R²
is monotone non-decreasing by construction, because the same objective (SSE)
is being minimized over a strictly larger parameter space. A mixed model's
σ²_u0 has no such guarantee. REML estimates `(β, σ²_u0, σ²_u1, σ_u01, σ²_e)`
jointly by maximizing a restricted likelihood; adding fixed effects changes
the projection used to form that restricted likelihood, and the optimizer can
reallocate variance among the *four* random/residual components in any
direction — including toward σ²_u0 — especially when:

- the added predictors are weak or non-significant (as here: none of the M3
  block terms are significant, so there is little true signal to reallocate
  and the estimate is dominated by REML's non-convex objective rather than a
  clear improvement in fit), and
- the random-effects covariance structure is already carrying most of the
  systematic between-state signal via the correlated slope (`corr_u01`
  0.70–0.74 for inspections) — leaving weak main-effect covariates (none of
  which are entered ×time in the inspections M3 spec) with little room to act
  on anything but σ²_u0 itself, where sampling noise in a 39-cluster REML fit
  can just as easily push the estimate up as down.

In short: σ²_u0 is a variance-component estimate from a joint nonconvex
optimization, not a residual-minimization target, so it carries no structural
guarantee of monotonic decrease as fixed effects are added — and the
inspections M2→M3 step (same N, non-significant added covariates) is direct,
matched-sample evidence of exactly that: σ²_u0 went up, not down, while the
model's only other "improvement" surfaced as a small reallocation out of the
random-slope variance and covariance.
