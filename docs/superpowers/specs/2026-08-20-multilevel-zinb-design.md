# Multilevel Zero-Inflated Negative Binomial Models — Design

**Date:** 2026-08-20
**Requested by:** Joe Grzywacz (email, 2026-08), following the 2011–2021 table update
**Status:** approved design, not yet implemented

---

## 1. Why

Joe's read of the 2011–2021 tables is that "these newer results shift our results and
discussion in a way that renders our 'story' less compelling," and he asked whether we
can replicate the analytic model of Jafari and colleagues — specifically a **multilevel
zero-inflated negative binomial (ZINB) model**.

Reference: Jafari et al., "Enhancing detection of labor violations in the agricultural
sector: A multilevel generalized linear regression model of H-2A violation counts,"
*PLOS ONE* 2024, doi:10.1371/journal.pone.0302960.

There is an independent statistical case for the change, separate from Joe's concern
about the narrative. Our current specification models `log(count + 1)` with a Gaussian
LMM. That is a defensible transform-based approach and it is what the current tables
report, but for the WPS-view violations panel it is fighting the data:

| DV (WPS view, 2011–2021) | N | zeros | mean | variance |
|---|---|---|---|---|
| `wps_violations` | 533 | 93 (**17.3%**) | 29.70 | 4,681.7 |
| `wps_inspections` | 539 | 7 (1.3%) | 68.32 | 15,491.7 |

Variance is ~158× the mean for violations — severe overdispersion — and 17.3% of
state-years are zero. 31 of 49 states have at least one zero-violation year (maximum 7
of 11); **no state is zero in all years**. That pattern is exactly the structural-zero /
sampling-zero mixture a ZINB is built for, and it is substantively interpretable: a
state with no functioning WPS enforcement program generates structural zeros, while an
enforcing state with a quiet year generates a sampling zero.

Inspections is a different problem. 1.3% zeros with massive overdispersion is a plain
negative-binomial situation; a zero-inflation component there will likely have a
near-zero, poorly identified logit intercept. The expected honest outcome is **ZINB for
violations, NB for inspections**, but that is an empirical question the fits will settle,
not an assumption.

### What "replicate Jafari" can and cannot mean

Jafari et al. fit their model in **R 4.2.2 using `glmmTMB`** on 11,976 case-level DOL
investigation records, with **cross-classified state × industry random intercepts in both
the conditional and the zero-inflation components**.

Our design is a 49-state × 11-year panel of 539 state-years with no industry dimension.
The *estimator* ports over cleanly. The *design* does not. This spec therefore ports
Jafari's model class onto our existing random-effects structure rather than copying their
formula, and says so explicitly wherever the two diverge. Anything else would be
misrepresenting the replication.

---

## 2. Decisions taken

Settled with Keshav before this spec was written:

1. **Role of the models.** Fit the full count-model ladder *alongside* the existing
   log-LMM and let AIC/BIC/LRT choose the primary specification, the way Jafari did.
   Do not swap specs on faith. The log-LMM remains a reported robustness check either way.
2. **Random-effects structure.** Keep our specification: random intercept + random slope
   on linear time by state, `(1 + time | state)`, with cubic time as fixed effects. Only
   the distribution and link change, so the new tables stay comparable to the current
   ones. The zero-inflation component gets a state random intercept if it converges,
   otherwise a fixed intercept. *Not* Jafari's crossed `(1|state) + (1|year)`, which would
   discard the cubic time trend the 2016–17 WPS-revision story rests on.
3. **Exposure.** For violations, an offset of `log(inspections)` is the primary
   specification — it makes the DV a violation *rate per inspection*, which is the cleaner
   substantive claim and the natural count-model idiom. `log(inspections + 1)` as a
   right-hand-side covariate (the current table's form) runs as a check so the choice is
   visibly not driving results.
4. **Windows.** Both. 2011–2021 WPS view *and* 2011–2019 establishments view.

Decision 4 deserves its rationale recorded, because it is the one that answers Joe
directly. The results he is reacting to changed **two things at once**: the window
(2019 → 2021) *and* the outcome source (ECHO establishments view → WPS view). Refitting
both windows under the same model class is the only way to say which of those two changes
weakened the associations. This is also the conflation Kaitlyn flagged.

---

## 3. Toolchain

**R + `glmmTMB` for the fits; Python for everything around them.**

`glmmTMB` is the package Jafari used, which matters when a reviewer asks whether we
replicated their model or approximated it. It supports ZI random effects, offsets,
AIC/BIC, and LRTs natively. Verified 2026-08-20 on this machine: R 4.5.2,
`aarch64-apple-darwin20`, and CRAN serves prebuilt arm64 binaries for `glmmTMB` 1.1.14
and `TMB` 1.9.23 — no compilation step.

Rejected alternatives:

- **Pure Python.** `statsmodels` has no multilevel ZINB. `pymc` means Bayesian inference
  with no comparable AIC/BIC/LRT and a different inferential frame to explain in the
  methods section. Hand-rolled maximum likelihood with adaptive Gauss-Hermite quadrature
  means a 3-dimensional integral per state (random intercept, random slope, ZI random
  intercept) — writing a worse `glmmTMB` that would still need validating against the
  real one.
- **R for the whole pipeline.** Would orphan the existing Python covariate construction,
  which is the part that must not drift.

R entering the pipeline is not unprecedented: the repo already carries Stata `.do` ports
of three core scripts.

The Python cross-check from the rejected option is retained as a *validation step*, not a
second implementation — see §6.

---

## 4. Data bridge

`scripts/build_count_model_panel.py` assembles both analytic frames and writes:

- `data/generated/count_model_panel_2021.csv` — WPS view, 2011–2021, 49 states
- `data/generated/count_model_panel_2019.csv` — establishments view, 2011–2019, 50 states

Both carry **raw unlogged counts** (that is the entire point) alongside the existing
`log_*` columns, so the same file also supports the Gaussian validation fit in §6.

Covariate construction is **imported from the existing scripts, not retyped.** The
2021 frame reuses the merge logic in `paper_table_models_2021.py` (spending from
`spend_bls_variables_multiyear_2021.csv`, LII 2017, DOL demand-met %, `pct_flc`,
year-matched `h2a_per_farmworker_z` from `h2a_ratio_panel_2011_2021.csv`); the 2019 frame
reuses `paper_table_models_corrected.py` (the non-`_2021` inputs and the
`establishments_data.csv` wide→long reshape, `inspections` = EPA + state). Any drift
between the count models and the published tables would make the comparison worthless, so
the shared code gets factored into one place rather than duplicated a third time.

Columns written: `state`, `year`, `time`, `time2`, `time3`, `violations`, `inspections`,
`log_violations`, `log_inspections`, `covid` (2021 frame only), and the six z-scored
covariates.

---

## 5. Models

### 5.1 Fixed-effect build-up

Unchanged from the current Tables 2 and 3, so columns stay aligned:

| Model | Terms |
|---|---|
| M1 | `time + time2 + time3` (+ exposure, violations only) |
| M2 | M1 + `SPEND_APP_z + SPEND_WORK_z + lii_2017_z` |
| M3 | M2 + `h2a_per_farmworker_z + dol_demand_met_pct_z + pct_flc_z` |

### 5.2 Distribution ladder

Fit per outcome × window × exposure variant, all on `(1 + time | state)` and identical
fixed effects:

| # | Family | `ziformula` |
|---|---|---|
| 1 | `poisson` | — |
| 2 | `nbinom1` | — |
| 3 | `nbinom2` | — |
| 4 | `poisson` (ZIP) | `~ 1` |
| 5 | `nbinom2` (ZINB) | `~ 1` |
| 6 | `nbinom2` (ZINB) | `~ 1 + (1 \| state)` |

### 5.3 Where the ladder is run

The ladder is **not** run at every build-up step — that would multiply into ~108 fits and
allow different families to win at different steps, with no rule for resolving the
conflict. Instead, per cell (outcome × window × exposure variant):

1. Run all six families on the **M3** specification. M3 is the fullest model and the one a
   distribution has to survive with every covariate present.
2. Re-run all six on **M1** purely as a stability check — if the winner flips between M1
   and M3, that is reported, not hidden.
3. Fit **M2** under the winning family only.

Cells: inspections × 2 windows, violations × 2 windows × 2 exposure variants = 6 cells.

### 5.4 Selection protocol

- AIC and BIC across all six.
- Likelihood-ratio tests on the nested pairs only: Poisson ⊂ NB2, NB2 ⊂ ZINB. **Both are
  boundary tests** (dispersion → ∞; zero-inflation probability → 0), so the nominal
  p-values are conservative. This is stated as a caveat — Jafari's paper glosses it.
- Observed vs. expected zero counts per model, and a dispersion statistic. AIC alone is a
  weak argument for zero-inflation; the zero-count comparison is the direct evidence.
- No Vuong test. It is routinely misapplied to non-nested ZI comparisons and would invite
  a referee objection.

### 5.5 Zero-inflation predictors

Jafari put their full predictor set in the ZI component. With 49 states and 93 zeros that
will be unstable here. **Primary specification is intercept-only ZI.** An expanded ZI
component — `SPEND_APP_z + SPEND_WORK_z + lii_2017_z`, the variables that plausibly
generate a *structural* "this state does not enforce" zero — is fit as sensitivity and
reported **only if it converges cleanly**.

### 5.6 Reporting

- Coefficients as `b (SE)` **and** IRR = `exp(b)`, since count models are read as rate
  ratios. Jafari report log-scale estimates and interpret `e^b` in text.
- The existing **State-to-State Variation** block carries over: σ²_u0 is still the
  between-state variance, now on the log *link* scale rather than the log(count+1)
  outcome scale, so its magnitude is not comparable to the current tables' values even
  though the Δσ²_u0 *percentage* is. Following the convention already in force (CLAUDE.md,
  2026-07/08), Δσ²_u0 is read from **random-intercept-only refits** — `(1 | state)`, ZI
  and family held at the winning specification — against a **single Model-1 baseline**, so
  every column starts from Model 1. Both the matched-sample and Model-1-baseline values
  are stored, as in `paper_table_params_2021.json`.
- Convergence code and any singular / non-finite SEs reported per model.

### 5.7 COVID robustness (2021 window only)

`paper_table_models_2021.py` fits and reports a 2020–21 indicator because the pandemic
depresses WPS inspections ~15% in 2020, and under the log-LMM the indicator was
−0.590\*\*\* for inspections (pushing `time2`/`time3` into significance) but −0.183 n.s.
for violations. That check is repeated here on the winning family for the 2021 window, so
the two model classes can be compared on the same question. It is reported, not tabled.

---

## 6. Validation

Nothing downstream is trustworthy until the bridge is proven, so validation comes first.

1. **Gaussian round-trip (the critical test).** Fit a Gaussian `glmmTMB` on
   `log_violations` and `log_inspections` with the identical `(1 + time | state)`
   structure and M1/M2/M3 fixed effects, and confirm coefficients match
   `statsmodels.MixedLM` from `paper_table_models_2021.py` to ~3 decimal places. A match
   proves the CSV bridge, the analytic sample, the RE structure, and the R script in one
   shot. A mismatch means stop and fix before fitting anything else.
   *Note:* `glmmTMB` defaults to ML, `MixedLM` to REML — use `REML = TRUE` for this
   comparison.
2. **Independent ZI cross-check.** `statsmodels.ZeroInflatedNegativeBinomialP` with state
   dummies instead of random effects, confirming the ZI intercept's sign and rough
   magnitude. Different software, different RE handling — if both say the zero-inflation
   component is real, it is not an artifact of the R specification.
3. **Convergence gate.** Any model that does not converge cleanly is marked as such, never
   quietly tabled.

### Fallback ladder (convergence risk is real)

533 observations and 49 states may not support a random slope *plus* a ZI random
intercept. Documented degradation order, applied per model and recorded in the output:

1. `(1 + time | state)` + `ziformula = ~1 + (1|state)`
2. `(1 + time | state)` + `ziformula = ~1`
3. `(1 | state)` + `ziformula = ~1`
4. Report failure to converge; do not report the estimates.

---

## 7. Artifacts

| Path | Role |
|---|---|
| `scripts/build_count_model_panel.py` | builds both analytic panels |
| `scripts/count_models_zinb.R` | `glmmTMB` ladder, both outcomes × windows × exposure variants |
| `scripts/report_count_models.py` | reads results, prints tables, writes the memo |
| `data/generated/count_model_panel_{2019,2021}.csv` | analytic frames |
| `data/generated/count_model_results.json` | all fits: coefficients, IRRs, variance components, fit statistics, convergence codes |
| `data/generated/count_model_comparison.csv` | the AIC/BIC/LRT/zero-count selection table |
| `docs/count_models_zinb.md` | memo for Joe: comparison table, recommendation, caveats |

**Out of scope:** regenerating the manuscript `.docx`. That waits until we know which
specification wins. The existing 2011–2019 and 2011–2021 tables and figures are left
untouched.

---

## 8. Risks and honest expectations

- **Convergence.** Mitigated by the §6 fallback ladder.
- **The offset drops rows.** 7 state-years have zero inspections, where `log(inspections)`
  is undefined. Those rows leave the offset specification; the covariate specification
  keeps them. Both N values are reported.
- **The establishments window probably will not support ZINB.** ~2.8% zeros there. If it
  lands on plain NB, that is a finding about the two outcome measures, not a failure.
- **This may not rescue the story.** A better-specified model can equally well confirm
  that the associations are weak. If ZINB is the right model for a 17%-zero overdispersed
  count, its answer is the answer. Recording this here so it is not a surprise later.
- **Two things changed at once in the results Joe is reacting to.** The dual-window design
  is what separates the model-class effect from the data-source effect. If the story
  weakened because the WPS view is a genuinely different measure rather than because the
  model was misspecified, the dual-window fits will show that, and the discussion section
  should say so.

---

## 9. Related context

- The establishments view vs. WPS view distinction, and why extending to 2021 required
  switching outcome sources, is documented at length in `CLAUDE.md` (2026-08 section).
  Kaitlyn's reading of the table note is correct: the outcome data itself changed, not
  only the years.
- Existing transform exploration that led to the `log(count+1)` choice:
  `docs/transform_exploration.md`.
- Variance-component conventions and the Model-1 baseline direction:
  `docs/variance_decomposition.md` and `CLAUDE.md`.
