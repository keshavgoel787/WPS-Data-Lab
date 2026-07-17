# Collinearity Notes — Tables 2 & 3

Why several candidate predictors were collapsed, dropped, or re-expressed before
entering the manuscript models (`scripts/paper_table_models.py`). All correlations
below are computed on the 50-state Level-2 frame.

## 1. Labor Intensity Index — one wave only

The Census of Agriculture Labor Intensity Index is available for 2012, 2017, and
2022. The in-study waves are almost perfectly correlated:

| pair | r |
|---|---|
| LII 2012 vs LII 2017 | **0.977** (n = 50) |

Entering more than one wave is effectively entering the same variable twice: the
estimates become unstable and flip sign, and the standard errors inflate. The
tables therefore retain **LII 2017 only**, matching the time-centering reference
year (`time = year − 2017`). The 2022 wave falls outside the 2011–2019 study
period and is not modeled. (This mirrors the PI direction already applied in
`final_labor_violations_model.py` / `final_labor_inspections_model.py`.)

## 2. Spending denominators — two FIFRA populations only

All `SPEND_*` variables share one numerator (mean inflation-adjusted STAG
obligations per state) divided by different denominators, so they are correlated
by construction. The full set:

| | WORK | APP | FLC | OP | AREA |
|---|---|---|---|---|---|
| SPEND_WORK (per farmworker)   | 1.00 | 0.35 | 0.71 | 0.50 | 0.56 |
| SPEND_APP (per applicator)    | 0.35 | 1.00 | 0.47 | 0.75 | 0.41 |
| SPEND_FLC (per FL supervisor) | 0.71 | 0.47 | 1.00 | 0.57 | 0.52 |
| SPEND_OP (per operation)      | 0.50 | 0.75 | 0.57 | 1.00 | 0.14 |
| SPEND_AREA (per sq mile)      | 0.56 | 0.41 | 0.52 | 0.14 | 1.00 |

The manuscript tables enter only **Spending/applicator (SPEND_APP)** and
**Spending/farmworker (SPEND_WORK)** — the two populations legally protected
under FIFRA/WPS. Critically, these two correlate only **r = 0.35**, so they can
sit in the same model without destabilizing each other. The heavier pairwise
overlaps (SPEND_WORK–SPEND_FLC = 0.71, SPEND_APP–SPEND_OP = 0.75) all involve the
*excluded* denominators, which is a second reason SPEND_FLC and SPEND_OP are left
out. Their shared numerator still means the two retained coefficients should be
read as a joint spending block rather than as fully independent effects.

**No ×time interactions (PI direction, 2026-07).** Earlier drafts of Table 3 carried
`SPEND_APP × time`, `SPEND_WORK × time`, and `%H-2A Authorized to FLC × time`
interactions. All three have been removed; the violations table now enters every
covariate as a main effect only. Table 2 (inspections) never carried any ×time
interactions.

## 3. Time trend — cubic in both tables

Both tables carry a **cubic** time polynomial (`time`, `time2`, `time3`;
`time = year − 2017`) in every column, per PI direction (2026-07). This restored
the quadratic/cubic terms — which had briefly been dropped in favor of a linear
trend — and reverted the earlier PI decision (2026-06) that had set the inspections
table to linear time. The cubic captures the asymmetric 2016–2017 violation spike;
the inspections table now uses the same cubic form for consistency. Each column's
Δσ² baseline is the matched cubic-time spec re-estimated on that column's sample
(inspections: `time + time2 + time3`; violations: `inspections + time + time2 + time3`).

## 4. H-2A block — ratios, not raw counts

The H-2A predictors are expressed as ratios to remove the state-size scaling that
makes raw counts collinear with every other size-driven variable (farmworker
employment, farming operations, land area):

- **H-2A farmworkers : BLS farmworkers** — mean certified H-2A workers divided by
  farmworker employment (BLS OEWS OCC 45-2092, 2011). Replaces the raw certified
  count `dol_workers_cert`, which scaled with state agricultural size.
- **H-2A Authorized / H-2A Requested** (`dol_demand_met_pct`) — an intensity rate
  bounded independently of size.
- **% H-2A Authorized to FLC** (`pct_flc`) — a composition share.

`dol_n_cases` was dropped entirely: it is a near-duplicate of the certified-worker
count and adds no independent information.

## 5. Sample sizes

BLS suppresses small employment cells, so the occupation denominators
(OCC 37-3012 pesticide applicators; OCC 45-2092 farmworkers) are missing for
~11 states. Models M2 and M3 are therefore fit on **39 states** (351 inspection
obs / 320 violation obs) by listwise deletion. Each column's Δσ² is computed
against a baseline re-estimated on that same analytic sample, so the
variance-reduction figures are matched-N comparisons (never against the full-N
M1 baseline).

## 6. Reading the variance components

- **Violations (Table 3):** the spending + labor block reduces between-state
  intercept variance by ~28% (M2 27.8%, M3 28.1%) relative to the
  inspections + cubic-time baseline — a substantial, meaningful reduction.
- **Inspections (Table 2):** Δσ² is negligible (M2 +0.1%, M3 −0.9%). In a mixed
  model the between-state variance is not bounded to fall when predictors are
  added; values this close to zero mean these Level-2 covariates explain
  essentially **no** between-state variation in inspection counts beyond the cubic
  time trend. This is reported as-is rather than floored at zero.
