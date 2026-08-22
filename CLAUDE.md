# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Research analysis studying whether state EPA STAG grant funding predicts Worker Protection Standard (WPS) violation rates across 50 U.S. states (2011–2019). All analysis is in Python using `statsmodels.MixedLM`. There is no build system, test suite, or package to install beyond standard scientific Python.

## Repository Layout

```
Research/
  CLAUDE.md               ← this file (stays at root)
  data/
    raw/                  ← original input datasets (never regenerated)
    generated/            ← intermediate CSVs produced by scripts
  scripts/                ← all analysis + visualization Python
  figures/                ← generated PNG figures
  docs/                   ← methods write-ups, model reports, result tables
```

## Running Scripts

```bash
# Violations (DV = total violations per state-year)
python3 scripts/hierarchical_violations_model.py   # Original M1/M2 models + visualizations
python3 scripts/stepwise_zimmerman_models.py       # Stepwise Level-2 covariate models
python3 scripts/spending_bls_models.py             # BLS-normalized SPEND_* variables + stepwise models
python3 scripts/targeted_spend_model.py            # FIFRA-aligned model (SPEND_WORK + SPEND_APP + SPEND_AREA)
python3 scripts/visualize_model_results.py         # Figures for hierarchical model
python3 scripts/visualize_polynomial_terms.py      # Figures for polynomial time terms

# Inspections (DV = EPA + state inspections per state-year; exact structural mirror)
python3 scripts/inspections_hierarchical_model.py
python3 scripts/inspections_stepwise_zimmerman_models.py
python3 scripts/inspections_spending_bls_models.py    # requires data/generated/spend_bls_variables.csv
python3 scripts/inspections_targeted_spend_model.py   # requires data/generated/spend_bls_variables.csv

# Labor intensity & DOL H-2A covariate models (new datasets)
python3 scripts/labor_covariates_violations_models.py
python3 scripts/labor_covariates_inspections_models.py

# Final curated models (PI-directed: lii_2017 only; inspections linear-time)
python3 scripts/final_labor_violations_model.py
python3 scripts/final_labor_inspections_model.py

# Comprehensive "every variable" inspections model (kitchen-sink, exploratory)
python3 scripts/comprehensive_inspections_model.py

# Manuscript Tables 2 & 3 (combined spending + labor + H-2A build-up)
python3 scripts/paper_table_models.py       # fits 6 models → data/generated/paper_table_params.json
python3 scripts/fill_wps_tables_docx.py     # fills the Word template → docs/WPS_Table_Sheels_filled_cubic.docx

# Exploratory (2026-07): transformed-DV robustness + variance decomposition
python3 scripts/transformed_violations_models.py   # violations suite across log/sqrt/Anscombe/Box-Cox
python3 scripts/variance_decomposition.py          # full random-effects component breakdown + VPC
python3 scripts/parameter_walkthrough_models.py    # variable-by-variable rebuild of Tables 2&3 with dropped time terms put back

# Post-meeting corrections (2026-07): full-BLS harvest, year-matched H-2A ratio, log DVs
# Run in this order (each depends on the previous):
python3 scripts/harvest_bls_oews_panel.py     # BLS OEWS 2011-2019 (all years) → data/raw/bls_oews_panel_2011_2019.csv
python3 scripts/harvest_stag_spending.py       # USASpending re-harvest + validation (diagnostic; see notes below)
python3 scripts/rebuild_spend_bls_multiyear.py # multi-year SPEND denominators + $0 for MS/RI/WV → spend_bls_variables_multiyear.csv
python3 scripts/build_h2a_ratio_panel.py       # year-matched time-varying H-2A ÷ farmworker ratio → h2a_ratio_panel.csv
python3 scripts/paper_table_models_corrected.py# CORRECTED Tables 2&3: log(y+1) DVs, N=47 → paper_table_params_corrected.json
```

**2026-07 post-meeting corrections (supersede the earlier paper tables).** After Joe's
meeting the models were rebuilt to fix the 50→39 state drop and adopt log DVs:
- **Why states dropped**: BLS OEWS suppression in the 2011-only snapshot (8 states) +
  a spending-window gap (MS/RI/WV had no in-window rows). See
  `memory/project_wps_missing_states.md`.
- **BLS**: `harvest_bls_oews_panel.py` pulls all years 2011-2019; farmworker
  employment (45-2092) is now 50/50, pesticide applicators (37-3012) 47/50 (AK/RI/VT
  never published by BLS — accepted, PI direction).
- **Spending**: the master is hand-curated to pesticide-relevant awards, dominated by
  CFDA 66.700 (Consolidated Pesticide Enforcement Cooperative Agreements). Summing all
  7 CFDAs inflates spending ~43× (66.605 PPG is $3.95B) — do NOT do that. MS/RI/WV
  genuinely receive $0 in 66.700 (like HI/TN/UT already coded), so they are $0-coded,
  not dropped. `harvest_stag_spending.py` is the diagnostic that established this.
- **H-2A ratio**: now year-matched and time-varying (annual certified H-2A ÷ annual
  BLS farmworkers), a Level-1 predictor — replaces the frozen-2011-denominator state mean.
- **DVs**: both modelled as `log(count+1)`; inspections enters the violations table on
  the same `log(x+1)` scale. Δσ² baselines re-estimated per column on the log scale.
- **Result**: substantive models N = 47 (was 39).

**2026-07 augmentation (on top of the corrected tables).** Adds three things to the
corrected pipeline without touching its models; see
`docs/superpowers/specs/2026-07-31-wps-table-augmentation-design.md`:
- `paper_table_models_corrected.py` now also emits, per column, the **full random-effects
  set** (`sigma2_u0`, `sigma2_u1`, `sigma_u01`, `sigma2_e`) for M1/M2/M3 *and* their
  time-only-baseline counterparts (`*_baseline`), plus a fully standardized `beta`
  (`b·SD(x)/SD(y)`, SDs on each column's own analytic sample) on every fixed effect.
- `fill_wps_tables_docx_augmented.py` **opens the original shell** (`~/Downloads/WPS Table
  Sheels.docx`) so the augmented tables inherit its `Table 2.`/`Table 3.` captions and
  significance notes (Helvetica 11, "inspections"/"violations" bold-italic; study period
  corrected 2021→2019), then swaps each placeholder table **in place** for a freshly built
  one carrying a separate **β column** after each model's b/(SE) and a bottom **State-to-State
  Variation** block (σ²_u0 and σ²_e as *baseline → fitted*, plus Δσ²_u0 %) on the
  **random-intercept basis**. Output: `docs/WPS_Table_Sheels_augmented.docx` (the committed
  `WPS_Table_Sheels_filled_corrected.docx` is left untouched). Run
  `paper_table_models_corrected.py` first.
- **Variance-explained basis (2026-07)**: the augmented table's Δσ²_u0 is read from
  **random-intercept-only** refits (baseline + model, same sample) — `delta_pct_ri` in the
  JSON — because in the random-SLOPE paper spec σ²_u0 is the between-state variance *at the
  centering year 2017* and trades off against σ²_u1/σ_u01, so its reduction is not bounded to
  [0,1] and can go negative. Coefficients still come from the random-slope model. The JSON
  keeps both: `delta_pct` (random-slope σ²_u0, can be negative) and `delta_pct_ri`
  (random-intercept, the reported value). **Both** docx report the random-intercept values:
  `fill_wps_tables_docx_corrected.py` and `fill_wps_tables_docx_augmented.py` read
  `sigma2_u0_ri` / `delta_pct_ri` (M2/M3 columns; the M1 variance column is blank in the
  corrected shell).
- **Model-1 baseline (2026-08 team direction)**: `delta_pct_ri` / `sigma2_u0_ri_baseline`
  now reference a **single Model-1 baseline** (M1's random-intercept σ² on its own sample) so
  every column "starts from Model 1" — Inspections M2/M3 = 40.3%/37.5%, Violations = 18.4%/37.8%.
  The rigorous same-sample refit values are retained under `*_matched` keys (Inspections
  10.6%/6.4%). **Caveat**: inspections M2/M3 drop AK/RI/VT (no BLS applicator data), which carry
  much of the between-state inspection variance, so part of that column's reduction against
  Model 1 reflects the narrower sample, not just the covariates; violations is unaffected. Both
  docx notes state this.
- `spaghetti_plots.py` draws per-outcome trajectory plots: eligibility = ≥1 nonzero DV in
  EACH of Pre (2011-15) / revised WPS (2016-17) / Post (2018-19); 15 states drawn at random per
  outcome (independent seeds); **distinct per-state colors (tab20) with matching bold endpoint
  labels** + thick black 15-state mean overlay; **no figure title** (Joe drafts his own); the
  shaded gray band is labeled "revised WPS". Raw-count and log(count+1) variants →
  `figures/fig_spaghetti_{inspections,violations}{,_log}.png`.
- **Verified (2026-07)**: the −12.5% Δσ²_u0 in Violations Model 2 was a *genuine* estimate,
  not a sign error — but it is an artifact of the random-slope centering (σ²_u0 measured at
  2017 rises when predictors enter). Diagnostic: with a random slope the M2 reduction is
  −12.5%; random-intercept-only it is +19.2%, and the M2 predictors correlate −0.36/−0.40
  with state baseline intercepts. The augmented table now reports the random-intercept
  value (+19.2%). See the `delta_pct_ri` note above.

**2026-08 extension to 2011–2021 (Joe's request).** Joe asked whether the figures could
run through 2021 or whether the models were themselves stuck at 2011–2019. They were, and
the constraint was the DV source. **This is the single most important data fact in the
project:**

- `data/raw/establishments_data.csv` is the **"establishments" view** of EPA's ECHO State
  Pesticide Dashboard, and **EPA publishes it only through 2019**. Re-verified against the
  live dashboard 2026-08-13: last column is `penalties-epa-2019`. There is no 2020/2021 to
  harvest. Every covariate (STAG spending → 2024, DOL H-2A → 2025, BLS OEWS → present) already
  ran past 2019, so the DV was the sole binding constraint.
- `data/raw/wps_data.csv` is the **"WPS" view** of the same dashboard and **does run
  2011–2021**. Both local files are byte-identical to what the dashboard currently serves.
- Provenance (previously unrecorded): both files come from that dashboard's download button.
  The page inlines both CSVs as strings on `drupalSettings.echoDashboards` (`estData` /
  `wpsData`); there is no data API. One fetch of any `?view=` returns both, all states.
- **The two views are NOT the same measure.** WPS counts run 2–3× the establishments counts
  (2,296 vs 967 violations across the 50 states in 2017) and state-level correlation is
  ~0.4–0.6 in 2011–2016 and ~0 from 2017 on. The 2011–2021 tables are a **re-analysis on a
  WPS-specific outcome**, not an extension of the 2011–2019 ones, and coefficients are not
  line-by-line comparable. (It is arguably the better-matched outcome for a WPS paper.)

Extending to 2021 therefore means switching the DV to the WPS view. The 2011–2019 pipeline is
left fully intact; every extended artifact is written alongside it under a `_2021` name, and
four scripts now take an optional end-year argument (default reproduces the 2019 outputs
byte-for-byte — verified). Run in this order:

```bash
python3 scripts/build_wps_dv_panel.py                  # WPS view → wps_dv_panel_2011_2021.csv (49 states; WY absent)
python3 scripts/harvest_bls_oews_panel.py 2021         # → data/raw/bls_oews_panel_2011_2021.csv
python3 scripts/rebuild_spend_bls_multiyear_2021.py    # → spend_bls_variables_multiyear_2021.csv (CPI-U extended)
python3 scripts/build_h2a_ratio_panel.py 2021          # → h2a_ratio_panel_2011_2021.csv
python3 scripts/paper_table_models_2021.py             # → paper_table_params_2021.json  (+ COVID robustness)
python3 scripts/spaghetti_plots.py 2021                # → figures/fig_spaghetti_*_2021.png
python3 scripts/fill_wps_tables_docx_augmented.py 2021 # → docs/WPS_Table_Sheels_augmented_2021.docx
```

Model spec is **unchanged** (M1/M2/M3 build-up, cubic time centered 2017, no ×time
interactions, log(count+1) DVs, random intercept + random slope, REML) — only the window and
DV source differ. N = 49 states (M1) / 46 (M2–M3, AK/RI/VT lack BLS applicator data);
Wyoming has no row in the WPS view at all.

- **COVID caveat (important, and it differs by outcome).** 2020–21 are pandemic years; WPS
  inspections fall ~15% in 2020. `paper_table_models_2021.py` fits and prints a
  COVID-indicator robustness check. **Inspections:** the dummy is −0.590*** and adding it
  pushes `time2`/`time3` from null into significance — so the inspections time trend over
  2011–2021 partly reflects enforcement capacity, not compliance, and the cubic is absorbing
  the shock. **Violations:** in the log-linear check the dummy is −0.183 n.s. and the time
  terms barely move, so that table looks robust as fit. Both docx notes state this.
  **CORRECTED 2026-08 by the count models — the "barely move" part does not survive.** Under
  the selected count family (`viol_cov_2021`, NB1) the COVID indicator is −0.279 (p ≈ 0.096)
  and `time2` goes from p ≈ 2.0e-5 to p ≈ 0.26 when the indicator is added. The violations
  time trend is *less* COVID-robust than this sentence originally implied; the log-linear
  check understated the sensitivity. See `docs/count_models_zinb.md`.
- Spending over the wider window tracks the old one closely (SPEND_WORK r=0.963,
  SPEND_APP r=0.997, median ratio ≈1.00), so the window change does not distort the
  spending measure.
- The H-2A ratio rises sharply at the end of the window (state mean 2.68 in 2011 → 4.04 in
  2020 → 5.30 in 2021): certified H-2A grew while BLS farmworker employment fell.
- `figures/fig_spaghetti_*_2021.png` extend Post to 2018–2021 and add a dotted divider with a
  "COVID-19" band label. The 2011–2019 figures are untouched.
- The 2011–2019 `docs/WPS_Table_Sheels_augmented.docx` was regenerated with **one** added
  sentence in its Note recording that its outcome comes from the establishments view, which
  EPA publishes only through 2019. No numbers changed.

**2026-08 multilevel count models (Jafari replication).** Joe asked whether we could
replicate Jafari et al. (*PLOS ONE* 2024, doi:10.1371/journal.pone.0302960) — a multilevel
zero-inflated negative binomial — after the 2011–2021 tables weakened the story. Design spec:
`docs/superpowers/specs/2026-08-20-multilevel-zinb-design.md`. Deliverable memo:
`docs/count_models_zinb.md` (**generated**, never hand-edited). Run in this order:

```bash
Rscript scripts/r_env_check.R                   # gate: glmmTMB installed and able to fit a ZINB
python3 scripts/build_count_model_panel.py      # → count_model_panel_{2019,2021}.csv (RAW counts)
Rscript scripts/count_models_zinb.R \
        data/generated/count_model_panel_2021.csv \
        data/generated/count_model_results.json # 6 families x 6 cells + COVID + diagnostics
python3 scripts/build_count_model_memo_evidence.py  # ~15 s: 12 MixedLM refits of the
                                                #   published 2021 spec + 2 statsmodels
                                                #   FE-ZINB fits
                                                # → count_model_memo_evidence.json
python3 scripts/report_count_models.py          # pure artifact reader (<1 s)
                                                # → count_model_comparison.csv,
                                                #   count_model_winner_summary.csv,
                                                #   docs/count_models_zinb.md
python3 scripts/validate_count_models.py        # sections [1]–[7]; ALL must pass before
                                                # trusting any estimate from this pipeline
```

- **R enters the pipeline here, and only here.** `statsmodels` has no multilevel ZINB;
  `glmmTMB` is the package Jafari used. First use of R in this project. Python still builds
  every panel and does all reporting. Expect a benign TMB ABI-mismatch warning on every
  `Rscript` call — the validator asserts stderr contains *nothing else*.
- **The random-effects structure is OURS, not Jafari's.** We keep `(1 + time | state)` with
  cubic time. Their crossed state × industry design has no analogue here (no industry
  dimension), and an intercept-only crossed structure would discard the **state-specific
  time slope** — the per-state rate of change around the WPS revision. (The cubic trend
  itself is a *fixed* effect and would survive either way; it is the random slope that
  would not. An earlier draft of this note got that rationale wrong.) Where a family cannot be fit at
  `(1 + time | state)`, a `(1 | state)` fallback tier is reported **separately**; AIC is never
  compared across tiers.
- **Gaussian round-trip gate.** `validate_count_models.py [2a]/[2b]` refits the manuscript's
  own Model 1 as a Gaussian `glmmTMB` (REML) and requires it to match `statsmodels.MixedLM`
  (coefficients 1e-3 rel., variance components 2e-2 rel.). Nothing downstream is trustworthy
  if that fails. `[5]` independently cross-checks the ZI component against
  `statsmodels.ZeroInflatedNegativeBinomialP` with state dummies.
- **σ²_u0 here is on the log LINK scale**, not the `log(count+1)` outcome scale of the
  published tables — magnitudes are **not** comparable to Tables 2/3 even though a percentage
  reduction would be. This pipeline computes **no Δσ²_u0 percentage** at all: that needs a
  random-intercept-only refit series against a single Model-1 baseline, and the ladder only
  produces random-slope models. Deliberate gap, stated as such in the memo.
- **Both windows are fit** (2011–2021 WPS view and 2011–2019 establishments view), because the
  results that weakened the story changed the window AND the outcome source at once; the dual
  fit is what separates those two effects. Three specs per window (inspections; violations
  with inspections as offset; violations with log(inspections) as covariate) = 6 cells.
- **Headline answer**: Jafari's model class fits these data **better** than the plain negative
  binomial everywhere it can be estimated. Across every legitimate matched comparison in the
  ladder (same cell, model, `re_tier` and N; both converged; neither ZI-degenerate) a
  zero-inflated negative binomial beats plain NB1 in **17 of 17** comparisons, by 34.7–195.0
  AIC, with no exception (15 of 15 counting only distinct models — 2 rows are a ZINB+ZI-RE
  that collapsed onto its ZINB twin). The binding constraint is **identifiability at the
  random-slope structure**, not evidence against zero-inflation. Where a plain NB wins a
  column it wins **by default**, the ZI-NB rungs having been unavailable at that structure —
  never on merit. Do not write that a negative binomial's own overdispersion explains the
  zeros "about as well"; that is false in every matched comparison. The same 17 comparisons
  run against plain **NB2** also give 17 of 17, by 24.1–83.2 AIC, so the claim covers both
  negative-binomial parameterisations, not just NB1.
- **But it is the ZI-NB specifically, not zero-inflation generally, and do NOT write that a
  zero-inflated model never lost on merit anywhere.** The evidence is a three-tier
  **hierarchy**, and every tally is generated in `docs/count_models_zinb.md`:
  (0) against a plain Poisson the two ZI-NB rungs are **8–0** (ZINB) and **9–0** (ZINB+ZI-RE),
  by 390.2–3519.6 AIC; (1) **inflation helps** — ZIP beats plain Poisson **12–1** (margins
  −1.9 to +625.7 AIC); (2) **NB overdispersion helps more** — a plain NB beats that same ZIP
  **13–0** against NB1 and **13–0** against NB2; (3) **the two together win wherever they can
  be fit** — ZI-NB beats NB1 **17–0** and NB2 **17–0**. That is precisely the case for
  Jafari's specification.
  A zero-inflated **Poisson** loses at **13** rungs (a *rung* is one (cell, model, `re_tier`);
  a rung is a loss if **at least one** plain family beats it there — which is why the pairwise
  12–1 in tier 1 and these 13 losing rungs are consistent, not contradictory) — **9 of them at
  the mandated `rs` tier** — by 1.9–2967.5 AIC. At **12** of the 13 every family beating it is
  a negative binomial; at the **13th** (`viol_off_2019` M1 `rs`) a **plain Poisson beats it
  too, by 1.94 AIC**, and that rung IS the 1.9 lower bound and the single loss in tier 1. Do
  NOT write "always by a negative binomial, never by a plain Poisson" — that was a round-4
  defect. Also do NOT write that inflation on a Poisson "is beaten wherever it is tested"
  without the "against a negative binomial" qualifier; unqualified it is false. No `zinb` or
  `zinb_re` ever lost a matched comparison (**0** losses); always name the *negative binomial*
  as the subject of that claim. Also note that **8 of the 17**
  ZI-NB-vs-NB1 comparisons are at the `(1 | state)` fallback tier, that **M2 contributes
  none** (the ladder fits M2 under the already-selected family only) so "every comparison in
  the ladder" means M1 and M3, and that **8 ZI-NB fits are excluded as `zi_degenerate`** (all
  in the 2019 violations cells; 8 of 8 have worse AIC than their rung's selected plain
  family, so exclusion hides no wins).
- **Neither plain family is the "harder comparison".** NB2 has the smaller ZI-NB margin at
  **8** of the 17 rungs and NB1 at the other **9**. NB2's mean margin is lower (47.4 vs 81.1)
  only because the 4 `insp_2021` rungs fall from ~183 to ~33 AIC — a range effect, not a
  majority. The claim that holds is that the ZI-NB beats whichever plain family is tougher at
  each rung, at all 17.
- **Selected families**: inspections selects a zero-inflated family in both windows (2019
  ZINB+ZI-RE, 2021 ZINB); violations select a plain negative binomial in both (2019 NB2, 2021
  NB1) at the mandated tier. **This is NOT evidence against zero-inflation for violations** —
  ZI is real, large and precisely estimated in the 2021 violations cells (ZIP ZI intercept
  ≈ −2.2 to −2.3, p ≈ 5e-36 at the `rs` tier). **At Model 3** only four of the six families
  were eligible at `(1 + time | state)` for those cells and neither ZI-NB rung reached that
  tier, so the family comparison there is an *identifiability* result, not a test of
  zero-inflation. **Scope this to Model 3**: at **Model 1**, `viol_off_2021` does admit
  ZINB+ZI-RE at `(1 + time | state)` and it wins that tier by 53.48 AIC (which is what
  `__meta.stable_rs = False` records) — for **that cell** the ZI-NB rung drops out *as
  covariates are added*. **Scope this to that one cell too**: `viol_off_2021` is the ONLY
  violations cell with a ZI-NB rung at `rs`/M1 that is absent by M3. `viol_cov_2021`,
  `viol_off_2019` and `viol_cov_2019` have **no** ZI-NB rung at `rs` at *either* model, so
  nothing "stops" for them — do not write that the ZI-NB rungs stop being estimable "for the
  violations cells".
  The 2019 violations series is zero-**deflated** (every family overpredicts its zeros).
  **The 2019 violations ZI collapse is Model-3-only in `viol_off_2019` and in NO other cell**:
  `__meta.zi_evidence` records the Model-3 rung only, and `viol_off_2019__M1__zip` is
  `re_tier='rs'`, converged and non-degenerate (ZI intercept −6.1675, SE 3.246, p = 0.0574) —
  same pattern as `viol_off_2021`. But `viol_cov_2019` has all **3** ZI families
  `zi_degenerate` at **both** M1 and M3 (that is exactly what its 6 degenerate fits are), so
  for that cell the collapse is not Model-3-specific at all. Any claim of the form "all ZI
  families collapse" must be scoped to Model 3; any claim that the collapse is Model-3-only
  must be scoped to `viol_off_2019`.
- **σ²_u1 is not a published quantity.** Tables 2/3 report `sigma2_u0_ri`, `sigma2_e_ri` and
  `delta_pct_ri`; σ²_u1 exists in `paper_table_params_2021.json` but reaches no published
  table. The NB1-vs-NB2 σ²_u1 disagreement (≈3.2–3.8×) therefore changes nothing currently
  in print.
- **NEW FINDING requiring a PI decision: the published Table 3 Model 1 never converged.**
  `paper_table_models_2021.py`'s `lbfgs` fit of `log_violations` M1 (random slope) reports
  `|grad| = 76.8236` at loglik −737.667; `cg` and `powell` both reach −729.532 cleanly. The
  published b(log_inspections) = 0.4954 and σ²_u0 = 2.4438 are the non-converged values
  (converged: 0.5183 / 1.2177). The module-level `warnings.filterwarnings('ignore')` in that
  script swallowed the `ConvergenceWarning`. Scope, measured: **exactly 1 of 12** published
  2021 fits is affected; the other 11 converge; and the reported variance-explained block is
  **unaffected** because all six random-intercept-only refits converge and reproduce the
  published values. **Not fixed here** — see `docs/count_models_zinb.md`.
- **Manuscript `.docx` regeneration was deliberately NOT part of this work.** No
  `docs/WPS_Table_Sheels_*.docx`, no `paper_table_*` script and no `paper_table_*` JSON was
  modified or regenerated by the count-model pipeline.
- **No script in this pipeline may use `warnings.filterwarnings('ignore')`** — that idiom is
  what hid the finding above. Warnings are captured with `catch_warnings(record=True)` and
  reported.

Scripts must be run from `/Users/keshavgoel/Research/` — all file paths are absolute and hardcoded to the `data/`, `figures/`, and `docs/` subdirectories.

Stata ports of three core scripts also live in `scripts/` (`hierarchical_violations_model.do`, `visualize_model_results.do`, `visualize_polynomial_terms.do`), merged from the `stata` branch. They read/write the same `data/` and `figures/` paths as their Python counterparts.

## Data Pipeline and File Dependencies

```
Raw inputs (data/raw/)
  establishments_data.csv         ← EPA ECHO violations + establishment counts (wide: one col per year)
  spending_data_master.csv        ← Nominal STAG grant obligations by state abbreviation + year
  h2a_state_summary_2017.csv      ← Yuri's data: farming operations + H-2A workers (Census 2017)
  bls_oews_panel.csv              ← BLS OEWS employment by OCC code (2011 snapshot, 3 occupations)
  wps_data.csv                    ← WPS facilities/workers (used in hierarchical_violations_model.py)

Intermediate outputs (data/generated/ — created by scripts, consumed by later scripts)
  model_data_long.csv             ← Generated by hierarchical_violations_model.py
  predicted_trend.csv             ← Generated by hierarchical_violations_model.py; required by visualize_model_results.py
  spending_aggregated.csv         ← Generated by hierarchical_violations_model.py (state-year spending, 2017 $M)
  state_random_effects.csv        ← Generated by hierarchical_violations_model.py
  level2_covariates.csv           ← Generated by stepwise_zimmerman_models.py
  spend_bls_variables.csv         ← Generated by spending_bls_models.py; required by targeted_spend_model.py
```

Script dependencies:
- `hierarchical_violations_model.py` must run before `visualize_model_results.py` (needs `predicted_trend.csv`)
- `spending_bls_models.py` must run before `targeted_spend_model.py`
- `spending_bls_models.py` must also run before `inspections_spending_bls_models.py` and `inspections_targeted_spend_model.py` (both read `spend_bls_variables.csv`)

Additional raw data files (data/raw/, used by `labor_covariates_*.py`):
  labor_intensity_index_2012.csv  ← Census of Agriculture LII, 2012 (50 states × 2 cols)
  labor_intensity_index_2017.csv  ← Census of Agriculture LII, 2017
  dol_var1_workers_by_state_annual.csv ← DOL H-2A annual workers/cases 2008–2025 (2013 missing; 2014 requests=0)
  dol_var2_employer_type_annual.csv    ← DOL H-2A employer type breakdown 2020–2025 only (2020 used as proxy)

## Modeling Conventions (Apply to Every Script)

**Always use these fixed constants** — they are duplicated across scripts and must stay in sync:
- `US_STATES_50`: canonical 50-state filter (excludes territories, tribes, regions)
- `STATE_ABBREV_TO_NAME`: spending data uses 2-letter abbreviations; ECHO data uses full names
- `CPI_U`: BLS CPI-U annual averages 2011–2019, base year 2017 for inflation adjustment
- `STATE_NAME_MAPPING = {'Massachusetts ': 'Massachusetts', 'Oregon ': 'Oregon'}`: known trailing-space issues in ECHO index

**Time variable**: always `time = year − 2017`. Cubic polynomial (`time`, `time2`, `time3`) is retained in all violations and all *exploratory* models. Never drop terms — **except** in the curated final **inspections** models (`final_labor_inspections_model.py`), where per PI direction (2026-06) the inspections series carries only a linear time trend, so `time2`/`time3` are dropped and the pseudo-R² baseline is re-estimated as linear-time on the same analytic sample. Violations final models keep the cubic. **The manuscript paper tables (`paper_table_models.py`) use cubic time in all columns of both the inspections and violations tables (per PI direction 2026-07, which restored `time2`/`time3` and reverted the earlier linear-time inspections spec), and carry NO ×time interactions (the spending×time and pct_flc_z×time interactions were all dropped). Each column's Δσ² baseline is the cubic-time spec re-estimated on its own analytic sample.**

**Model estimation**: always `MixedLM.from_formula(..., re_formula='~time').fit(method='lbfgs')` — random intercept + random slope for linear time by state, REML.

**Level-2 covariates**: z-score (mean=0, SD=1) before entry. Convention is `varname_z` for standardized version.

**Pseudo-R²**: `(σ²_u0_baseline − σ²_u0_model) / σ²_u0_baseline` — reduction in between-state variance (`cov_re.iloc[0,0]`) relative to the time-only baseline.

**Baseline model must be re-estimated on the same analytic sample** when listwise deletion changes N (e.g., BLS variables have state-level missingness). Do not compare pseudo-R² against a baseline fit on a different N.

## Analysis Sequence and What Each Script Does

| Script | Models | Key output |
|---|---|---|
| `hierarchical_violations_model.py` | M1 (random intercept), M2 (+ random slope) — time polynomial only, full N=394 | Figures, `model_data_long.csv`, `state_random_effects.csv` |
| `stepwise_zimmerman_models.py` | Zimmerman one-at-a-time: spending_per_estab, land_area, farming_operations, h2a_workers, workers_per_operation | `level2_covariates.csv` |
| `spending_bls_models.py` | Zimmerman one-at-a-time for 5 BLS-normalized SPEND variables; combined model | `spend_bls_variables.csv` |
| `targeted_spend_model.py` | Single theory-driven model: SPEND_WORK + SPEND_APP + SPEND_AREA + two interactions | Terminal output only |
| `inspections_hierarchical_model.py` | Same as violations M1/M2 but DV = total inspections (EPA + state) | `inspections_model_data_long.csv`, `inspections_state_random_effects.csv`, `inspections_predicted_trend.csv` |
| `inspections_stepwise_zimmerman_models.py` | Same Zimmerman sequence with inspections DV | Terminal output only |
| `inspections_spending_bls_models.py` | Same BLS spending stepwise with inspections DV; reads existing `spend_bls_variables.csv` | Terminal output only |
| `inspections_targeted_spend_model.py` | Same FIFRA targeted model with inspections DV | Terminal output only |
| `labor_covariates_violations_models.py` | Zimmerman stepwise for 6 new labor/DOL covariates, DV = violations | Terminal output only |
| `labor_covariates_inspections_models.py` | Same, DV = inspections | Terminal output only |
| `final_labor_violations_model.py` | Curated final model: cubic time + labor/DOL block; ×time interactions screened at p<.20 | Terminal output only |
| `final_labor_inspections_model.py` | Curated final model: **linear** time + labor/DOL block; linear-time baseline | Terminal output only |
| `comprehensive_inspections_model.py` | Kitchen-sink inspections model entering **every** project covariate (15 predictors) + ×time screen; linear time; exploratory | Terminal output only |
| `paper_table_models.py` | Manuscript Tables 2 & 3: 3-model build-up per DV combining SPEND_APP/SPEND_WORK + LII 2017 + H-2A block; **both tables = cubic time** in all columns (time2/time3 restored per PI direction 2026-07), violations table adds the raw inspections count as a Level-1 predictor. **No ×time interactions** — the two spending×time and the pct_flc_z×time interactions were all dropped (2026-07). | `data/generated/paper_table_params.json` |
| `fill_wps_tables_docx.py` | Reads `paper_table_params.json` and fills the Word template (`~/Downloads/WPS Table Sheels.docx`); corrects the study period to 2011–2019, fills coefficient rows by model→column position (so cubic time2/time3 land in inspections M2/M3, which the template left blank), drops rows for terms no longer in any model (the ×time interactions), and appends a collinearity note. Requires `python-docx`. | `docs/WPS_Table_Sheels_filled_cubic.docx` |
| `transformed_violations_models.py` | Full violations suite re-fit under raw/log/sqrt/Anscombe/Box-Cox; residual diagnostics + pseudo-R² (AIC not cross-comparable). Exploratory. | `data/generated/transform_comparison.csv`, `docs/transform_exploration.md`, `figures/transform_qq_*.png` |
| `variance_decomposition.py` | Full variance-component decomposition (σ²_u0/σ²_u1/σ_u01/σ²_e, ICC, time-varying VPC) of the paper-table models; diagnoses the negative inspections Δσ². Exploratory. | `data/generated/variance_components.csv`, `docs/variance_decomposition.md`, `figures/vpc_*.png` |
| `parameter_walkthrough_models.py` | Variable-by-variable (cumulative add-one-in) rebuild of Tables 2 & 3 on a **fixed common sample**, with all dropped time terms reintroduced (cubic `time2`/`time3` + `SPEND_APP_z:time`/`SPEND_WORK_z:time`/`pct_flc_z:time`). Reports every coefficient + all four variance components per step to show what each parameter does — esp. the time interactions (centering re-interpretation, σ²_u1 vs σ²_u0, collinearity). Exploratory; does not change the paper. | `data/generated/parameter_walkthrough.json`, `docs/parameter_walkthrough.md` |

Manuscript-table dependency: `paper_table_models.py` must run before `fill_wps_tables_docx.py`, and both require `spending_bls_models.py` (for `data/generated/spend_bls_variables.csv`) to have run first. Collinearity rationale for the table specification is in `docs/collinearity_notes.md`.

**Final-model covariate block (2026-07 revision, both final models):** `lii_2017_z, h2a_per_farmworker_z, dol_demand_met_pct_z, pct_flc_z`. `h2a_per_farmworker` = mean certified H-2A workers ÷ BLS OCC 45-2092 farmworkers (2011). Replaces the raw `dol_workers_cert`; `dol_n_cases` dropped. BLS suppression → N=44 states. See `docs/final_models_update_2026-07.md`.

## BLS OEWS Data Notes

`bls_oews_panel.csv` contains only **2011 data** for 3 OCC codes:
- `37-3012` — Pesticide Handlers, Sprayers, and Applicators (used as SPEND_APP denominator)
- `45-1011` — First-Line Supervisors of Farming/Forestry Workers (SPEND_FLC)
- `45-2092` — Farmworkers and Laborers, Crop/Nursery/Greenhouse (SPEND_WORK)

OCC `47-3012` does **not exist** in this file. `37-3012` is the correct pesticide applicator code. BLS suppresses small cells, so 6–7 states are missing for some occupations — these are excluded listwise from models using those variables.

## Key Research Context

- **Outcome**: WPS violations spike dramatically in 2016–2017 (mean 20.57 in 2017 vs. ~7 in 2011–2015), then decline. The cubic polynomial captures this asymmetric pattern.
- **ICC ~77%**: The large majority of variance is between states, justifying the HLM approach.
- **Spending variables**: All use the same numerator — mean inflation-adjusted STAG spending 2011–2019 per state — divided by different denominators. SPEND_FLC explains the most individual variance (~9%) but SPEND_WORK + SPEND_APP together (the two FIFRA-protected populations) explain ~21% in the targeted model.
- **SPEND_OP** (spending per farming operation) is consistently non-significant across all models.
