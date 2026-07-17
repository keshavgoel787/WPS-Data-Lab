# Transformed-DV Exploration: Violations Models Under 5 Dependent-Variable Transforms

**Status:** Exploratory. Does not modify or supersede the manuscript pipeline
(`paper_table_models.py`, `fill_wps_tables_docx.py`, or any `.docx` output).

**Source data:** `data/generated/transform_comparison.csv` (65 rows = 13 violations
model specs × 5 DV transforms), produced by `scripts/transformed_violations_models.py`.

**Figures:** `figures/transform_qq_baseline_cubic.png`, `figures/transform_qq_final_labor.png`,
`figures/transform_qq_paper_M3.png`.

---

## 1. Purpose & method

WPS violation counts are right-skewed with a sharp 2016–2017 spike and many
small/zero values (mean 20.57 in 2017 vs. ~7 in 2011–2015). The manuscript models
fit `MixedLM` (Gaussian family) directly on the raw count, which can violate the
normality and homoscedasticity assumptions the model relies on for valid standard
errors. This exploration refits the full violations model suite (13 specs,
mirroring every existing violations script) under 5 DV transforms to see which
transform, if any, best satisfies those assumptions without changing the
substantive story.

**The 5 transforms** (raw is the untransformed reference):

| Key | Transform | Note |
|---|---|---|
| `raw` | `y` | reference; current manuscript spec |
| `log` | `log(y + 1)` | zeros-safe; the `+1` shift is standard for count data with zeros |
| `sqrt` | `sqrt(y)` | classic variance-stabilizer for Poisson-like counts |
| `anscombe` | `2·sqrt(y + 3/8)` | small-count refinement of the square-root transform |
| `boxcox` | Box–Cox on `(y + 1)`, λ estimated by MLE | requires strictly positive input, hence the `+1` shift |

All transforms are applied to the DV column only; predictors and the random-effects
structure (`re_formula='~time'`, random intercept + random slope, REML,
`method='lbfgs'`) are unchanged, per project convention.

**Fitted Box–Cox λ = −0.1102.** This is close to λ = 0 (the log transform), so the
Box-Cox and log results are expected to — and do — track each other closely
throughout this exploration.

**AIC / log-likelihood are NOT comparable across DV transforms.** A change of
variables (e.g., `y → log(y+1)`) changes the scale on which the likelihood is
evaluated; comparing loglik/AIC across transforms without a Jacobian correction is
invalid and was not attempted. Transforms are instead judged on:

1. **Residual diagnostics** — Shapiro–Wilk W/p, residual skew and excess kurtosis,
   and a Breusch–Pagan-style R² (residual² regressed on fitted values, a
   heteroscedasticity check where lower values indicate more homoscedastic
   residuals).
2. **Coefficient stability** — whether the sign and significance of the
   substantively important covariates hold across transforms.
3. **Pseudo-R²** — the proportional reduction in between-state intercept variance
   (`cov_re.iloc[0,0]`) relative to a matched-N, same-transform baseline. This is a
   variance *ratio*, not a likelihood-based quantity, so unlike AIC it *is*
   comparable across transforms and is reported per transform below.

---

## 2. Diagnostics by transform

Three headline models are reported in full: the time-only baseline (mirrors
`hierarchical_violations_model.py` M2, N=394/50 states), the curated final labor
model (mirrors `final_labor_violations_model.py`, N=375/47 states), and the
manuscript Table 3 build-up's fullest model, paper_M3 (mirrors
`paper_table_models.py`, N=320/39 states). The remaining 10 stepwise/targeted specs
are in the CSV but are not discussed individually here, per the exploration's
scope note. Raw is listed first in each table as the reference row.

### 2.1 Baseline cubic (time-only, N=394, 50 states)

| Transform | Shapiro W | Shapiro p | Skew | Excess kurtosis | BP-R² | Pseudo-R² |
|---|---|---|---|---|---|---|
| raw | 0.7193 | 3.34e-25 | 4.4475 | 47.5024 | 0.1150 | 0.0% |
| log | 0.9942 | 0.1425 | 0.0399 | 0.4081 | 0.0003 | 0.0% |
| sqrt | 0.9448 | 6.03e-11 | 1.0659 | 5.6965 | 0.0977 | 0.0% |
| anscombe | 0.9370 | 7.25e-12 | 1.1720 | 6.4330 | 0.1090 | 0.0% |
| boxcox | 0.9931 | 0.0684 | −0.1068 | 0.4894 | 0.0061 | 0.0% |

Pseudo-R² is trivially 0.0% for every transform because the baseline has no
covariates to compare against itself. The raw-scale residuals are extremely
non-normal (W=0.72, massive excess kurtosis of 47.5 driven by the 2016–2017 spike
outliers) and show the largest BP-R² of the five options (0.115, alongside
anscombe's 0.109) — i.e., the most residual heteroscedasticity. Both `log` and
`boxcox` bring Shapiro's W above 0.99 with p > 0.05 (non-rejection of normality)
and cut BP-R² by roughly two orders of magnitude (0.0003 and 0.0061 vs. 0.115 on
raw). `sqrt` and `anscombe` are intermediate: they reduce skew/kurtosis
substantially relative to raw but Shapiro's W (0.94–0.95) still rejects normality
at p ≈ 1e-10, and BP-R² stays close to the raw level.

### 2.2 Curated final labor model (cubic time + labor/DOL block, N=375, 47 states)

| Transform | Shapiro W | Shapiro p | Skew | Excess kurtosis | BP-R² | Pseudo-R² |
|---|---|---|---|---|---|---|
| raw | 0.7194 | 1.28e-24 | 4.4226 | 46.9625 | 0.1133 | 14.16% |
| log | 0.9972 | 0.7793 | −0.0491 | 0.1947 | 0.0029 | 57.99% |
| sqrt | 0.9497 | 5.43e-10 | 1.0335 | 5.4767 | 0.1072 | 53.68% |
| anscombe | 0.9437 | 9.66e-11 | 1.1316 | 6.1276 | 0.1173 | 53.22% |
| boxcox | 0.9944 | 0.1814 | −0.1850 | 0.2364 | 0.0022 | 58.24% |

The diagnostic pattern mirrors the baseline: `log` and `boxcox` are the only two
transforms with Shapiro p > 0.05 and near-zero BP-R² (0.0029 and 0.0022 vs. 0.1133
on raw). Notably, the pseudo-R² is far from transform-invariant here: the labor/DOL
covariate block explains only 14.2% of between-state intercept variance on the raw
scale but 53–58% under every transform, with `boxcox` (58.24%) and `log` (57.99%)
the highest. This is consistent with the raw-scale model's between-state variance
being dominated by a few extreme high-violation state-years that the covariates
don't predict; compressing those outliers (log/Box-Cox most aggressively) lets the
same covariates explain a much larger share of what's left.

### 2.3 Paper Table 3, model M3 (cubic time + spend + labor + H-2A + raw inspections, N=320, 39 states)

| Transform | Shapiro W | Shapiro p | Skew | Excess kurtosis | BP-R² | Pseudo-R² |
|---|---|---|---|---|---|---|
| raw | 0.7299 | 1.89e-22 | 4.1218 | 37.8284 | 0.1111 | 28.37% |
| log | 0.9960 | 0.5918 | 0.0095 | 0.1590 | 0.0004 | 40.83% |
| sqrt | 0.9546 | 2.17e-08 | 0.9950 | 3.9269 | 0.1071 | 33.74% |
| anscombe | 0.9501 | 5.99e-09 | 1.0622 | 4.2949 | 0.1154 | 32.78% |
| boxcox | 0.9960 | 0.5869 | −0.1168 | 0.3200 | 0.0070 | 41.96% |

Same ranking again: `log` and `boxcox` both pass the Shapiro normality check
(p ≈ 0.59) with BP-R² near zero (0.0004 / 0.0070), while `sqrt`/`anscombe` remain
non-normal (p < 1e-7) with BP-R² essentially unchanged from raw (~0.11). Pseudo-R²
rises from 28.4% (raw) to 33–42% under every transform, again highest under
`boxcox` (41.96%) and `log` (40.83%).

**Across all three headline models the ranking is consistent:** `boxcox` ≈ `log`
(both pass Shapiro at p > 0.05, both cut BP-R² by 1–2 orders of magnitude) >
`sqrt` ≈ `anscombe` (partial improvement, Shapiro still rejects) > `raw` (fails
badly on every metric). This is unsurprising given λ = −0.1102 sits very close to
the log transform's λ = 0.

---

## 3. Coefficient stability

Pulled directly from `data/generated/transform_comparison.csv` for the two models
that carry substantive spending/labor conclusions.

### 3.1 `paper_M3` (SPEND_APP_z, SPEND_WORK_z, pct_flc_z, pct_flc_z:time)

| Transform | b(SPEND_APP_z) | p | b(SPEND_WORK_z) | p | b(pct_flc_z) | p | b(pct_flc_z:time) | p |
|---|---|---|---|---|---|---|---|---|
| raw | −0.3977 | 0.849 | −2.1510 | 0.225 | 5.2936 | 0.0249 | 0.6434 | 0.111 |
| log | −0.1187 | 0.352 | −0.2934 | 0.0123 | 0.3424 | 0.0079 | 0.0168 | 0.289 |
| sqrt | −0.1245 | 0.537 | −0.3853 | 0.0350 | 0.6764 | 0.0056 | 0.0570 | 0.101 |
| anscombe | −0.2385 | 0.541 | −0.7149 | 0.0429 | 1.3208 | 0.0057 | 0.1145 | 0.096 |
| boxcox | −0.0946 | 0.356 | −0.2461 | 0.0090 | 0.2611 | 0.0095 | 0.0102 | 0.402 |

- **SPEND_APP_z**: negative sign and statistically non-significant (p ranges
  0.35–0.85) in every transform. Not a robustness concern — the null result is
  the same story on every scale.
- **SPEND_WORK_z**: negative sign in all five transforms; non-significant on raw
  (p=0.225) but significant at p<.05 under log, sqrt, anscombe, and boxcox
  (p = 0.012, 0.035, 0.043, 0.009 respectively). The raw scale is the outlier here
  — every transformed scale agrees the effect is negative and significant, raw
  alone fails to detect it (consistent with raw's poor residual behavior masking a
  real signal).
- **pct_flc_z**: positive sign and significant at p<.05 in every transform,
  including raw (p=0.0249) — the most robust of the three coefficients. Magnitude
  scales with the DV's units (much larger on the raw count scale, ~0.26–1.3 on the
  transformed scales) but sign/significance never change.
- **pct_flc_z:time interaction**: never significant at conventional thresholds on
  any transform (p ranges 0.096–0.402); this null is itself stable across
  transforms — no scale manufactures a spurious time interaction.

### 3.2 `final_labor` (SPEND_APP_z / SPEND_WORK_z not in this spec; pct_flc_z and its interaction)

The curated final model (per PI direction) does not include `SPEND_APP_z` or
`SPEND_WORK_z` — those columns are blank (NaN) for `final_labor` in the CSV, which
is expected: the final labor model's covariate block is `lii_2017_z,
h2a_per_farmworker_z, dol_demand_met_pct_z, pct_flc_z` only, with no ×time
interaction retained (the interaction column is also all-NaN for this spec, i.e.
it did not survive the p<.20 screen). `pct_flc_z` itself:

| Transform | b(pct_flc_z) | p |
|---|---|---|
| raw | 2.7286 | 0.0512 |
| log | 0.4812 | 0.0001 |
| sqrt | 0.5952 | 0.0022 |
| anscombe | 1.1131 | 0.0027 |
| boxcox | 0.4012 | 0.0001 |

Positive sign in all five transforms. Significance strengthens under every
transform relative to raw: raw is borderline (p=0.0512, just above the
conventional .05 threshold) while all four transforms push it comfortably below
.01 (p ≤ 0.0027). Again, no transform reverses the story — it only sharpens it.
`lii_2017_z`, `h2a_per_farmworker_z`, and `dol_demand_met_pct_z` (not tabulated
above) are non-significant (p > 0.08) across all five transforms in this model,
consistent with the raw-scale final model's reported results.

---

## 4. Recommendation

**`log(y + 1)` is the recommended transform for a manuscript robustness appendix,**
with Box–Cox reported alongside as corroborating evidence (its fitted λ = −0.1102
is close enough to 0 that it behaves almost identically to log on every metric
above, and log is easier to explain/interpret — coefficients read approximately as
proportional changes in violations — than an arbitrary λ = −0.11 power transform).

Rationale:
- `log` and `boxcox` are the only two transforms that pass the Shapiro–Wilk
  normality check (p > 0.05) and cut the Breusch–Pagan-style heteroscedasticity
  statistic by roughly 1–2 orders of magnitude relative to raw, in all three
  headline models.
- Neither transform changes the substantive story: `pct_flc_z` remains positive
  and significant everywhere (it is actually *more* significant under log/Box-Cox
  than on raw); `SPEND_APP_z` remains null everywhere; `SPEND_WORK_z` goes from
  marginal/non-significant on raw to significant under every transform (log
  included) — if anything, log strengthens rather than overturns the manuscript's
  conclusions.
- `sqrt`/`anscombe` only partially fix the raw scale's non-normality
  (Shapiro still rejects, p < 1e-7) and leave heteroscedasticity essentially
  unchanged from raw, so they are not recommended as the primary robustness
  transform, though they can be mentioned as intermediate checks.
- This warrants inclusion as a **robustness appendix**, not a replacement for the
  raw-scale manuscript models: the raw scale is what's directly interpretable as
  "violation counts," and the substantive conclusions do not depend on the
  transform, which is itself the appendix's main point. Given the AIC-non-
  comparability caveat (Section 1), the appendix should present this exactly as
  done here — residual diagnostics and coefficient stability, not a model-selection
  ranking by likelihood.

---

## 5. Figures

- `figures/transform_qq_baseline_cubic.png` — QQ plots of residuals across the 5
  transforms for the time-only baseline model. Raw residuals show the heavy
  right-tail departure driven by the 2016–2017 spike; log/Box-Cox track the
  reference line closely.
- `figures/transform_qq_final_labor.png` — same panel for the curated final labor
  model (cubic time + `lii_2017_z, h2a_per_farmworker_z, dol_demand_met_pct_z,
  pct_flc_z`).
- `figures/transform_qq_paper_M3.png` — same panel for the manuscript Table 3 M3
  build-up (spending + labor + H-2A + raw inspections as a Level-1 predictor).

All three figures show the same pattern described in Section 2: raw is the most
visibly non-normal, log and Box-Cox are the closest to the 45° reference line, and
sqrt/anscombe fall in between.
