# NB2-Only Stepwise Count Models (2011–2021) — Design

**Date:** 2026-08-28
**Requested by:** team checklist following the count-model memo
**Status:** approved design, not yet implemented

---

## 1. Why

The count-model work (`docs/superpowers/specs/2026-08-20-multilevel-zinb-design.md`,
delivered as `docs/count_models_zinb.md`) answered a *model-selection* question: does
Jafari's model class fit these data better than a plain negative binomial? It does, and
the memo documents that at length. But it left the project without a **reportable table**,
because the selection ladder chose a different family per cell and the resulting
"variance components" block is not a variance-reduction sequence — the family changes
down some columns, so consecutive rows are different models' parameters rather than
successive stages of one model.

The team's direction is to settle on **one family, NB2, and refit stepwise**: Model 1 time
only, Model 2 adding commodity mix and spending, Model 3 adding the H-2A block, for both
outcomes. Holding the family fixed is what makes the three columns comparable and makes a
Δσ²_u0 block meaningful again.

Three checklist items drive this spec:

1. Refit all final models using NB2 only, stepwise, for inspections and violations.
2. Update the variance components table to the new stepwise structure (inspections
   Models 1–3, violations Models 1–3).
3. Confirm whether the cross-software check and the COVID robustness check need re-running.

Item 3 is answered in §5, not deferred.

### The covariate blocks already exist

No new variables are constructed. `scripts/count_models_zinb.R:198-199` already defines
exactly the blocks the checklist describes:

```r
M2_ADD <- c("SPEND_APP_z", "SPEND_WORK_z", "lii_2017_z")
M3_ADD <- c(M2_ADD, "h2a_per_farmworker_z", "dol_demand_met_pct_z", "pct_flc_z")
```

There is no separate commodity-mix variable anywhere in `data/raw/`. `lii_2017` — the
Census of Agriculture Labor Intensity Index — is this project's commodity-mix proxy, and
it already sits in the M2 block alongside the two spending measures. "Model 2: +
commodity mix/spending" and "Model 3: + H-2A variables" therefore map 1:1 onto the
existing `M2_ADD` / `M3_ADD` definitions, and this spec reuses them rather than
redefining them.

---

## 2. Decisions taken

Settled with Keshav before this spec was written:

1. **Scope: the 2021 window only, two cells.** `insp_2021` (inspections, no exposure) and
   `viol_cov_2021` (violations with `log_inspections` as a Level-1 covariate — the spec
   the published Table 3 uses). The 2011–2019 establishments arm and the
   violations-as-offset variant are **not** refit. Six substantive models.
2. **Family: `nbinom2` only.** No ladder, no zero-inflation, no selection step.
3. **RE structure: `(1 + time | state)`, no fallback tier.** If an NB2 random-slope fit
   fails, that is a finding to report, not something to silently downgrade — see §4.3.
4. **Variance components with a Δσ²_u0 percentage**, read from a parallel
   random-intercept-only NB2 refit series (§4.4). The existing count-model pipeline
   deliberately reports no Δ% at all; this spec closes that gap for the NB2 arm.
5. **A new standalone arm.** The six-family ladder, its results JSON, and its memo stay
   frozen and byte-identical, so the Jafari family comparison remains on the record as its
   own result. No `.docx` is touched.

---

## 3. NB2-only is a deliberate override of the AIC selection — say so

**This is the single most important caveat in this arm, and it must appear in the memo,
not only here.**

NB2 is not the AIC-selected family for either 2021 cell. Measured from
`data/generated/count_model_results.json`, at the mandated `(1 + time | state)` tier:

| Cell | Model | NB2 AIC | Best available AIC | Family | NB2 penalty |
|---|---|---|---|---|---|
| `insp_2021` | M1 | 4748.35 | 4715.61 | ZINB | **+32.74** |
| `insp_2021` | M3 | 4565.97 | 4532.61 | ZINB | **+33.36** |
| `viol_cov_2021` | M1 | 3840.37 | 3833.32 | NB1 | **+7.05** |
| `viol_cov_2021` | M3 | 3640.89 | 3630.16 | NB1 | **+10.74** |

NB2 loses in all four. Choosing it anyway is a legitimate editorial decision — one
family across six columns buys interpretability, a coherent variance-reduction sequence,
and a table a reader can follow — but it is a decision, not a result, and the memo must
not imply NB2 won anything. Two claims in particular must survive intact:

- The ZI-NB-over-plain-NB tallies in `docs/count_models_zinb.md` (17–0 against NB1, 17–0
  against NB2) are **unaffected** by this arm. Nothing here re-tests zero-inflation.
- For `insp_2021` specifically, the ~33 AIC gap to ZINB is substantial. The memo states
  the cost of the simplification rather than burying it.

NB2 is nonetheless the defensible single choice: it beats NB1 decisively for inspections
(4565.97 vs 4704.30 at M3, a 138.3 AIC margin) while costing only ~10.7 for violations, so
it is the family that is least bad across both outcomes. That reasoning goes in the memo
too.

---

## 4. Models

### 4.1 Panel and cells

`data/generated/count_model_panel_2021.csv` — raw counts, 2011–2021 WPS view, 49 states
(Wyoming has no row in the WPS view). Built by `scripts/build_count_model_panel.py`, which
this spec does not modify.

| Cell | DV (raw count) | Base terms |
|---|---|---|
| `insp_2021` | `inspections` | `time + time2 + time3` |
| `viol_cov_2021` | `violations` | `log_inspections + time + time2 + time3` |

`time = year − 2017`, matching every other model in the project.

### 4.2 Fixed-effect build-up

| Model | Right-hand side |
|---|---|
| M1 | base |
| M2 | base + `SPEND_APP_z + SPEND_WORK_z + lii_2017_z` |
| M3 | M2 + `h2a_per_farmworker_z + dol_demand_met_pct_z + pct_flc_z` |

No `× time` interactions, matching the published tables.

### 4.3 Estimation

`glmmTMB(family = nbinom2, ziformula = ~0, REML = FALSE)` with `(1 + time | state)`.

ML, not REML: AIC and LRT comparisons across nested fixed-effect specifications require
it, and the M1→M2→M3 build-up is exactly such a comparison. REML is used only in the
Gaussian gate (§5.1), where the comparison target `statsmodels.MixedLM` is REML.

Four of the six substantive fits already exist and converge cleanly at this tier
(`insp_2021` M1/M3, `viol_cov_2021` M1/M3). The two **M2 fits are new** — the existing
ladder fits M2 only under each cell's selected family, and NB2 is neither cell's selected
family. Convergence at M2 is expected but not assumed.

**No fallback tier.** The ZINB ladder degrades to `(1 | state)` when a fit fails, which is
correct when the goal is a family comparison. Here the random slope is the object of
interest — it is the per-state rate of change around the WPS revision — and a table whose
rows sit at different RE structures is the exact defect this arm exists to fix. A
non-converging fit is reported as non-converging and its estimates are withheld.

### 4.4 Variance components and Δσ²_u0

Coefficients come from the random-slope fits in §4.3. **The Δσ²_u0 percentage does not.**

In the random-slope spec, σ²_u0 is the between-state variance *at the 2017 centering
year*; it trades off against σ²_u1 and σ_u01, so its reduction is not bounded to [0,1] and
can go negative. That is not hypothetical here — it is what produced the −12.5% artifact
in the log-LMM violations table (CLAUDE.md, 2026-07). Following the convention already in
force for the published tables, Δσ²_u0 is read from a **parallel `(1 | state)` NB2 refit
series**, family and fixed effects otherwise identical, fit on each model's own rows.

Two percentages are stored per column, mirroring `delta_pct_ri` / `*_matched` in
`paper_table_params_2021.json`:

- **`delta_pct_ri`** (the reported value): against a single Model-1 RI baseline fit on
  M1's own sample, so every column "starts from Model 1".
- **`delta_pct_ri_matched`**: against a Model-1 RI baseline refit on the M2/M3 sample.

Both are needed because **the sample changes down the column**: M1 has N=539 / 49 states,
M2 and M3 have N=506 / 46, since AK, RI and VT have no BLS pesticide-applicator series and
are listwise-deleted once `SPEND_APP_z` enters. Part of any Model-1-baseline reduction
therefore reflects the narrower sample rather than the covariates. The memo states this
per table, as the published tables already do.

Reported table columns, per cell × model:

| Column | Meaning |
|---|---|
| σ²_u0 | random-intercept variance, log link scale, at time = 0 (2017) |
| σ²_u1 | random-slope variance |
| σ_u01 | intercept–slope covariance |
| θ | NB2 size parameter, `Var = μ + μ²/θ` |
| ICC | see below |
| Δσ²_u0 % | from the RI series, Model-1 baseline |
| Δσ²_u0 %, matched | from the RI series, same-sample baseline |
| N obs, N states | analytic sample |

**`σ²_e` is null throughout.** A count family has no residual variance; θ is a dispersion
parameter and its square is not a variance component. A previous revision of the ZINB
pipeline exported dispersion-squared under the name `sigma2_e` and that invited exactly
the wrong reading — this arm does not repeat it.

**ICC** uses Nakagawa's observation-level variance for an NB2 with log link:

```
ICC = σ²_u0 / (σ²_u0 + ln(1 + 1/θ + 1/μ))
```

with μ the fitted marginal mean. The formula is printed in the memo beside the number.
This is a link-scale approximation and is **not** comparable to the project's headline
"ICC ~77%", which comes from a Gaussian LMM on `log(count+1)`; the memo says so.

Every variance figure in this arm is on the log **link** scale. Magnitudes are not
comparable to σ²_u0 in published Tables 2 and 3, though a percentage reduction is.

---

## 5. Cross-software and COVID checks — the answer to checklist item 3

### 5.1 Gaussian round-trip: run it, do not redesign it

`validate_count_models.py` sections `[2a]`/`[2b]`/`[2c]` fit the manuscript's own M1/M2/M3
as **Gaussian** `glmmTMB` models and require them to match `statsmodels.MixedLM`
(coefficients 1e-3 relative, variance components 2e-2). That is a test of the CSV bridge,
the analytic sample, the RE structure, and the R/Python boundary. It is **family-agnostic**
— it never touches `nbinom2` — so its verdict carries over to this arm unchanged.

**Verdict: must be run, must not be modified.** The NB2 validator invokes the existing
gate as a precondition and refuses to report a number if it fails. No new Gaussian work.

### 5.2 The ZI cross-check has no NB2 counterpart — replace it

Section `[5]` cross-checks the zero-inflation component against
`statsmodels.ZeroInflatedNegativeBinomialP` with state dummies. An NB2-only arm has no ZI
component, so this check has nothing to test. It is not "re-run"; it is **retired for this
arm** and replaced by the analogous NB2 check:

> Fit `statsmodels.NegativeBinomialP(p=2)` with state fixed effects (dummies) and the same
> fixed-effect specification, and confirm the conditional-model coefficients agree with
> `glmmTMB`'s NB2 in sign and rough magnitude.

Same design as `[5]`, different estimand. Fixed effects are not random effects and the
estimates will not match to tolerance — this is a sign-and-magnitude sanity check across
two independent implementations, and the tolerance is stated loosely and honestly rather
than tightened until it passes. This is the one genuinely new check in this arm.

### 5.3 COVID robustness: must be re-run

**Yes, unambiguously.** Verified in `count_model_results.json`: the two existing COVID
variants are `insp_2021__M3covid__zinb` and `viol_cov_2021__M3covid__nbinom1`. Neither is
NB2, so neither transfers.

This matters beyond bookkeeping. CLAUDE.md records a load-bearing finding for violations —
COVID indicator −0.279 (p ≈ 0.096), with `time2` moving from p ≈ 2.0e-5 to p ≈ 0.26 once
the indicator enters — and that is an **NB1** result. Under NB2 it must be re-derived. If
the sensitivity moves, the CLAUDE.md note is corrected to match; if it holds, that is
worth stating too.

Both cells get M3 + `covid` refit under NB2 at the same `(1 + time | state)` tier. Note
that `viol_off_2021` was excluded from `COVID_CELLS` in the ZINB pipeline for a recorded
reason (the offset drops every zero-inspection state-year); that exclusion is moot here,
since this arm does not fit the offset spec at all.

---

## 6. Validation

A new `scripts/validate_nb2_stepwise.py`, following the conventions of
`validate_count_models.py`: per-section breakdown, an explicit `SKIP` line for any bypassed
gated block, and no headline count quoted as if it were independent findings.

1. **Precondition.** The existing Gaussian gate (§5.1) passes.
2. **Panel identity.** The NB2 arm reads `count_model_panel_2021.csv` unmodified; assert
   its shape, state count, and DV zero counts against the ZINB pipeline's own assertions.
3. **Fit integrity.** Every reported fit converged, has a positive-definite Hessian, finite
   SEs, and `re_tier == "rs"`. Any failure is surfaced, never dropped.
4. **Overlap with existing fits.** The four NB2 fits that already exist in
   `count_model_results.json` (`insp_2021` M1/M3, `viol_cov_2021` M1/M3) must reproduce to
   tolerance. This is a free regression test on the whole R path and it should not be
   skipped just because the fits are re-run in a new script. The comparison is valid
   because the ladder's `fit_spec` also defaults to `REML = FALSE`
   (`count_models_zinb.R:62`), so both arms estimate by ML on identical rows and formulas —
   these should agree to optimizer noise, not merely to a loose tolerance.
5. **Variance arithmetic.** Δ percentages recompute from the stored σ² values; the RI
   series is confirmed to be `(1 | state)` and to share each model's rows.
6. **Frozen artifacts.** `count_models_zinb.R`, `count_model_results.json`,
   `docs/count_models_zinb.md`, every `paper_table_*` artifact and every `.docx` are
   unchanged (content hash compared against `git` HEAD).
7. **Memo.** Generated-file check plus forbidden-pattern guards (§7).

Inherited from the ZINB pipeline and non-negotiable: **no script may use
`warnings.filterwarnings('ignore')`** — that idiom is what hid the non-converged published
Table 3 Model 1. Warnings are captured with `catch_warnings(record=True)` and reported. The
R script sets `options(warn = 1)`.

---

## 7. Artifacts

| Path | Role |
|---|---|
| `scripts/nb2_stepwise_models.R` | the six NB2 fits, the RI refit series, the COVID variants |
| `scripts/report_nb2_stepwise.py` | pure artifact reader; prints tables, writes the memo |
| `scripts/validate_nb2_stepwise.py` | §6 gate |
| `data/generated/nb2_stepwise_results.json` | all fits: coefficients, IRRs, variance components, convergence codes |
| `data/generated/nb2_stepwise_variance.csv` | the variance components table |
| `data/generated/nb2_stepwise_coefficients.csv` | coefficient table, `b (SE)` and IRR |
| `docs/nb2_stepwise_models.md` | memo — **generated, never hand-edited** |

Run order:

```bash
Rscript scripts/r_env_check.R
python3 scripts/build_count_model_panel.py          # if the panel is absent
Rscript scripts/nb2_stepwise_models.R \
        data/generated/count_model_panel_2021.csv \
        data/generated/nb2_stepwise_results.json
python3 scripts/report_nb2_stepwise.py
python3 scripts/validate_nb2_stepwise.py            # must pass before quoting any estimate
```

**Forbidden patterns in the memo**, enforced by the validator, all of them errors this
project has actually made before:

- any claim that NB2 was selected, won, or was preferred on fit grounds
- `sigma2_e` or "residual variance" applied to a count fit
- a Δσ²_u0 percentage attributed to the random-slope fits rather than the RI series
- an ICC compared directly to the log-LMM's ~77% without the link-scale caveat
- any restatement of the ZI-NB tallies, which belong to the other memo

**Out of scope:** regenerating any manuscript `.docx`; the 2011–2019 window; the
violations-as-offset spec; any change to the six-family ladder or its memo.

---

## 8. Risks and honest expectations

- **The M2 fits may not converge.** They are the only genuinely unknown fits. If either
  fails at `(1 + time | state)`, §4.3 says report the failure — and that failure would be
  a real result about what a 46-state panel supports, which the fallback tier in the ZINB
  pipeline was designed to hide for a different and legitimate purpose.
- **The AIC penalty in §3 will be visible.** A reader who has seen the ZINB memo will ask
  why the reported family is not the selected one. The answer is in §3 and belongs in the
  memo's first section, not a footnote.
- **The Model-1-baseline Δσ²_u0 is partly a sample effect** for both cells, since M1 has
  three more states than M2/M3. The matched value is reported alongside for exactly this
  reason; neither number alone tells the truth.
- **This does not resolve the story question.** Fixing the family makes the tables
  readable and comparable. It does not make an association stronger. If the H-2A block
  adds little between-state variance explained in Model 3, that is the finding.

---

## 9. Related context

- The family selection this arm overrides: `docs/count_models_zinb.md` and
  `docs/superpowers/specs/2026-08-20-multilevel-zinb-design.md`.
- Δσ²_u0 conventions, the Model-1 baseline direction, and the random-slope centering
  artifact: `CLAUDE.md` (2026-07 and 2026-08 sections) and `docs/variance_decomposition.md`.
- The establishments-view vs WPS-view distinction, and why the 2021 window uses a
  different outcome source: `CLAUDE.md` (2026-08 section).
- The non-converged published Table 3 Model 1, still unresolved and untouched by this arm:
  `docs/count_models_zinb.md`.
