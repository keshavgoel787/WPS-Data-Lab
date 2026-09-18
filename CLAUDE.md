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
  check understated the sensitivity. See `docs/count_models_zinb.md`. **This −0.279/p≈0.096
  figure is specific to NB1** — it is the selected family for that cell, not a family-free
  fact. The 2026-08-28 NB2-only stepwise arm (see below) refits the identical Model 3 + COVID
  specification under `nbinom2` and gets a materially different, non-significant coefficient:
  b = −0.188 (p = 0.446). Both numbers are correct for the fit they come from; neither
  supersedes the other, because the two arms hold different families fixed on purpose. Cite
  whichever family the surrounding analysis is actually using. See
  `docs/nb2_stepwise_models.md`.
  **Also newly documented: the two families disagree on the violations time trend even
  *before* COVID enters.** At Model 3 with no COVID indicator, `time` is p = 0.0129 under NB1
  versus p = 0.1597 under NB2 — a family-driven disagreement on the plain (non-pandemic) time
  trend, not something the pandemic years introduce. See `docs/nb2_stepwise_models.md`.
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

`build_count_model_panel.py` consumes the **2021-extension** pipeline's outputs as well as
the 2019 ones, so on a fresh checkout those must exist first or it raises `FileNotFoundError`.
The full prerequisite set is: `wps_dv_panel_2011_2021.csv`, `spend_bls_variables_multiyear.csv`
and `spend_bls_variables_multiyear_2021.csv`, `h2a_ratio_panel.csv` and
`h2a_ratio_panel_2011_2021.csv`, plus `data/raw/establishments_data.csv`. Run:

```bash
# --- prerequisites (from the 2026-08 extension block above) ---
python3 scripts/build_wps_dv_panel.py           # → wps_dv_panel_2011_2021.csv
python3 scripts/rebuild_spend_bls_multiyear.py  # → spend_bls_variables_multiyear.csv      (2019 arm)
python3 scripts/rebuild_spend_bls_multiyear_2021.py  # → spend_bls_variables_multiyear_2021.csv
python3 scripts/build_h2a_ratio_panel.py        # → h2a_ratio_panel.csv                    (2019 arm)
python3 scripts/build_h2a_ratio_panel.py 2021   # → h2a_ratio_panel_2011_2021.csv

# --- the count-model pipeline itself ---
Rscript scripts/r_env_check.R                   # gate: glmmTMB installed and able to fit a ZINB
python3 scripts/build_count_model_panel.py      # → count_model_panel_{2019,2021}.csv (RAW counts)
Rscript scripts/count_models_zinb.R \
        data/generated/count_model_panel_2021.csv \
        data/generated/count_model_results.json # 6 families x 6 cells + COVID + diagnostics
                                                # NOTE: the <panel_csv> argument drives ONLY
                                                # the Gaussian round-trip gate; build_cells()
                                                # opens BOTH panels by absolute path.
python3 scripts/build_count_model_memo_evidence.py  # ~30 s: 24 MixedLM refits (the published
                                                #   2021 AND 2019 specs) + 2 statsmodels
                                                #   FE-ZINB fits + the offset-drop derivation
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
  if that fails. `[2c]` extends it to **M1/M2/M3 for both windows** (spec 6.1) — M1 has no
  level-2 covariates, so before that arm existed nothing checked **covariate** parity across
  the R/Python bridge. `[2c]` is live-vs-live: statsmodels is fit under several optimizers,
  any fit raising a gradient failure is discarded, and glmmTMB must match the
  highest-log-likelihood survivor. (`lbfgs` genuinely fails for 2021 violations M1, 2019
  inspections M1 and 2019 violations M2; `cg` fails for 2019 inspections M3. glmmTMB reaches
  the good optimum in all of them.) `[5]` independently cross-checks the ZI component against
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
  zero-inflated model never lost on merit anywhere.** The evidence is a **four-tier**
  hierarchy — numbered (0) through (3) below, so "three-tier" is wrong — and every tally is
  generated in `docs/count_models_zinb.md`:
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
  NOT write "always **to** a negative binomial, never **to** a plain Poisson" — that was a
  round-4 defect. Quote the preposition exactly as written here: the enforced guards in
  `validate_count_models.py` (`MEMO_FORBIDDEN_PATTERNS`) match `always to a negative binomial`
  and `never to a plain Poisson`, and an earlier revision of this file wrote "by" instead,
  so a writer copying CLAUDE.md's own phrasing back into the memo would have evaded both. Also do NOT write that inflation on a Poisson "is beaten wherever it is tested"
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
  script swallowed the `ConvergenceWarning`. **FIXED 2026-08-31 — and the original scope note
  here was an undercount.** See the 2026-08-31 section at the end of this file: removing the
  warning filter exposed **three** distinct non-converged fits, not one (the `log_inspections`
  cubic-time baseline refit fails too, at `|grad| = 174.6`, 7.6 loglik below the optimum).
  What was right in the original note is the conclusion about the published variance block:
  **zero** `*_ri` values move, so the variance-explained rows in both tables are genuinely
  unaffected. What moves in print is the **violations Model 1 column only** — 9 cells.
- **Manuscript `.docx` regeneration was deliberately NOT part of this work.** No
  `docs/WPS_Table_Sheels_*.docx`, no `paper_table_*` script and no `paper_table_*` JSON was
  modified or regenerated by the count-model pipeline.
- **No script in this pipeline may use `warnings.filterwarnings('ignore')`** — that idiom is
  what hid the finding above. Warnings are captured with `catch_warnings(record=True)` and
  reported. `count_models_zinb.R` sets `options(warn = 1)` so anything that escapes its own
  per-fit handlers prints with its call instead of as an anonymous "There were N warnings"
  line (today: 11 × `sqrt(diag(vcovs)): NaNs produced`, all from fit attempts that were then
  discarded by the fallback ladder).

**2026-08-22 whole-branch review, fix wave (one pass, no second wave).** Applied on top of the
above. Headline tallies were re-verified unchanged after the ladder re-run: ZI-NB over NB1
17–0 (34.67–194.95), over NB2 17–0 (24.08–83.20), ZIP over Poisson 12–1, NB over ZIP 13–0/13–0,
15/15 distinct, 11 degenerate split 6/5, 30 LRTs, same six selected families.

- **CRITICAL, fixed: the zero-inflation random-effect variance was never extracted, so every
  reported "structural-zero probability" for a ZI-RE model was the wrong estimand.**
  `count_models_zinb.R` read only `VarCorr(fit)$cond$state`; `VarCorr(fit)$zi$state` was
  never read and no artifact carried it, so the reported figure was `plogis(b0)` — the
  probability at a **median state** (u = 0) — presented as marginal. For
  `insp_2019__M3__zinb_re` (the cell the memo headlines) the ZI random-intercept SD is
  **6.04** on the logit scale: median-state 0.000081 against a **marginal of 0.0675**, a factor
  of ~835, with **11.6%** of the fitted state distribution above a structural-zero probability
  of 0.10. A PI reading "0.0001" would conclude the inspections zero-inflation was negligible.
  Now: `sigma2_zi_u0`/`sigma_zi_u0` are in the JSON and both CSVs, both probabilities are
  reported side by side and labelled everywhere (ZI-evidence table, every coefficient block,
  the variance block), and the marginal is `E[plogis(b0 + u)]` by **Gauss-Hermite quadrature**
  (`GH_NODES = 240`; 64 nodes — the first value — was wrong in the 4th significant figure, and
  `zi_marginal_prob` now raises if halving the node count moves the answer by >1e-4 rel).
  `viol_cov_2021`'s selected `ri`-tier ZINB+ZI-RE: median-state 0.0547, marginal **0.0965**.
- **Also corrected: what buys `zinb_re` its margin.** The margin over plain ZINB is bought by
  the ZI random-intercept **variance**, not the ZI intercept — plain `zinb` has an intercept
  too. Never attribute it to the intercept.
- **`sigma2_e` is now null for every count fit.** `sigma(fit)^2` is θ² for `nbinom2` and the
  squared dispersion multiplier for `nbinom1`, and it was being exported in both CSVs under a
  name that reads as residual variance (43.87 for `insp_2021`). `dispersion` (with
  `family_name`) carries the parameter itself; the memo's variance section documents the
  per-family semantics. The Gaussian gate still uses the square, so `sigma2_e` survives on
  the `*_gaussian` records only.
- **Spec §5.5's expanded-ZI sensitivity is now implemented, and it is unstable.** One extra
  rung per cell (`{cell}__M3__zinb__zisens`, ZI formula
  `~1 + SPEND_APP_z + SPEND_WORK_z + lii_2017_z`, M3 and the `rs` tier only, never a selection
  competitor). It converges in **3 of 6** cells (`insp_2019`, `insp_2021`, `viol_off_2019`)
  and fails with a non-positive-definite Hessian in the other 3. Intercept-only ZI stands as
  the primary specification. Non-converged estimates are not reported anywhere.
- **The offset removes the structural zeros, and in the 2021 window it removes ONLY them.**
  All **7** rows the `log(inspections)` offset drops in 2021 — CT-2018, KS-2018, MA-2018,
  OR-2020, OR-2021, UT-2017, WV-2012 — are also zero-violation rows (7.5% of that window's 93).
  In 2019 the property does **not** hold (15 dropped; 1 zero, 6 positive, 8 missing). Derived
  per window in `build_count_model_memo_evidence.py::_offset_drop`, never asserted.
- **There is no zero-inflated NB1 rung**, while NB1 is the selected family for both 2021
  violations cells and their mean-variance exponent is **1.46** (NB1-like). Spec §5.2 fixed
  the six families; the family was deliberately NOT added (that would reopen selection across
  the whole ladder). The memo carries the caveat. The comparison that *does* isolate
  inflation is ZINB vs plain **NB2** (17–0), which holds the parameterisation fixed.
- **The published-fit audit now has a 2019 arm** against `paper_table_params_corrected.json`
  (item 5): all 12 refits reproduce their published σ²_u0, and that arm independently finds
  **2 of 12** 2011–2019 fits carrying a gradient failure under the published `lbfgs`
  (inspections M1 and violations M2, both random-slope). All 6 of its random-intercept-only
  refits converge, so as in 2021 the variance-explained block is not implicated. Recorded,
  not fixed.
- **`mark_zi_degenerate`'s loglik criterion now requires tier equality.** It never asserted
  it, so the criterion was silently inert for the **7** cross-tier comparisons (all healthy —
  nothing was mis-flagged). `zi_loglik_criterion_applied` / `zi_counterpart_tier` record
  applicability and the validator asserts the inert cases are exactly the cross-tier ones.
- **`zeros_partly_manufactured` and its note are derived**, from the raw establishments CSV
  plus the cell's own M1/M3 analytic samples, rather than `identical(cell_name, "insp_2019")`
  and four typed state-years. The derivation reproduces them exactly (4 of 4: Minnesota-2019,
  Montana-2011, Montana-2015, Utah-2013; Alaska's 8 and Vermont's 3 lost to listwise deletion).
- **`COVID_CELLS` excludes `viol_off_2021` for a stated reason**, recorded per cell as
  `__meta.covid_variant_fit` / `covid_not_fit_reason` and surfaced in the memo: the log-LMM
  check spec §5.7 compares against exists only for the two published columns, and the offset
  spec drops every zero-inspection state-year.
- **`pick_winner`'s empty-candidate-set fallback is recorded** (`__meta.winner_defaulted_*`,
  `n_eligible_rs_*`) instead of silently returning `nbinom2`. No cell defaults today.
- **Do not quote a single headline check count from `validate_count_models.py`.** It prints a
  **per-section breakdown** and an explicit `SKIP` line for every bypassed gated block.
  Section `[3]` alone is ~1,450 assertions — a per-record loop over the ladder, not
  independent findings. Total today: 2,277 pass / 0 fail / 0 skip, of which [3] is 1,452.
- Record population in `count_model_results.json` is now **121** = 78 tier-1 + 17 tier-2 +
  6 `__altopt` + 6 `__meta` + **6 Gaussian** (M1/M2/M3 × 2 outcomes, was 2) + 2 COVID +
  **6 `__zisens`**. `selection_table()` still returns **97** genuine fits: `__zisens` is
  excluded by key suffix, so no headline tally moved.

**2026-08-28 NB2-only stepwise count models (standalone arm, ladder untouched).** The
six-family ladder above answers a family-selection question; this arm answers a different
one — holding the family fixed at `nbinom2` across all three build-up steps so the two
outcomes' Model 1 → Model 2 → Model 3 sequences are directly comparable as one model class,
which the ladder's per-cell family selection does not give you. It reuses the frozen 2021 WPS
panel and covariate blocks; it fits nothing new about zero-inflation and does not touch
`docs/count_models_zinb.md` or any ladder artifact. Run in this order:

```bash
Rscript scripts/nb2_stepwise_models.R \
        data/generated/count_model_panel_2021.csv \
        data/generated/nb2_stepwise_results.json  # 6 substantive M1/M2/M3 fits + RI series +
                                                #   M1matched baseline + COVID refit, all
                                                #   nbinom2, (1 + time | state), no fallback
                                                # → data/generated/nb2_stepwise_results.json
python3 scripts/nb2_crosscheck_statsmodels.py  # statsmodels NegativeBinomialP(p=2) + state
                                                #   fixed effects, Level-1 terms only
                                                # → data/generated/nb2_stepwise_crosscheck.json
python3 scripts/report_nb2_stepwise.py         # pure artifact reader; the only script that
                                                #   REPORTS delta_pct_ri, delta_pct_ri_matched,
                                                #   icc_ri -- validate_nb2_stepwise.py [7]
                                                #   independently RE-DERIVES all three as a
                                                #   cross-check, which is what makes [7] non-
                                                #   circular, and must stay that way
                                                # → data/generated/nb2_stepwise_variance.csv,
                                                #   nb2_stepwise_coefficients.csv,
                                                #   docs/nb2_stepwise_models.md
python3 scripts/validate_nb2_stepwise.py       # sections [0]-[10]; [0] reruns the existing
                                                #   Gaussian gate as a precondition
```

- **Six models, two cells, 2011–2021 WPS window only**: `insp_2021` and `viol_cov_2021`
  (violations with `log_inspections` as a Level-1 covariate, matching Table 3), each M1
  (cubic time only) → M2 (+ `SPEND_APP_z`, `SPEND_WORK_z`, `lii_2017_z`) → M3 (+
  `h2a_per_farmworker_z`, `dol_demand_met_pct_z`, `pct_flc_z`), all at `(1 + time | state)`
  with no fallback tier. This does not extend to the 2011–2019 establishments window or to
  the violations-as-offset specification.
- **NB2 is an editorial choice, not a fit-based one — it is never the winning family.** At
  this random-effects structure it loses to the ladder's selected family at every rung it was
  fit — M1, M2 *and* M3, not just M1/M3: 32.74–33.58 AIC to ZINB for inspections, 7.06–13.73
  AIC to NB1 for violations (all six penalties are in `docs/nb2_stepwise_models.md`'s first
  table, derived from `count_model_results.json`, not typed in). **Do not write "~11 AIC at
  both M1 and M3"** — a review-round-1 defect: M1's violations penalty is 7.06, not 11, and
  M2's is 13.73. What fixing the family buys is a single model class across all three
  build-up steps, so the between-state variance actually forms one reduction sequence rather
  than three unrelated models' parameters. NB2 is the least-bad single choice across both
  outcomes — wherever the ladder has its own plain-NB1 fit at the same rung, NB2 beats it for
  inspections by 162.22 AIC at M1 and 138.33 AIC at M3 (inspections has **no** M2 NB1 entry,
  because ZINB, not NB1, is its selected family there), while costing NB2 7.06/13.73/10.74 AIC
  (M1/M2/M3) against NB1 for violations — a compromise, not a win. **Do not write "neither
  cell has an M2 NB1 entry"** — a residual-review-round defect: that is true for inspections
  only. Violations' selected family IS nbinom1, so the ladder fits one at every model there,
  M2 included (`viol_cov_2021__M2__nbinom1` exists, converged, AIC 3630.469), which is exactly
  why the M2 figure above exists at all. Scope any "M1 and M3 only" NB1-coverage claim to
  inspections; for violations it is M1, M2 *and* M3.
- **Two reduction bases, both reported, because the sample changes between M1 and M2/M3.**
  M1 keeps all 49 states; M2/M3 drop AK/RI/VT (no BLS pesticide-applicator series) to 46.
  `delta_pct_ri` measures every model against a single Model-1 baseline on Model 1's own
  49-state sample; `delta_pct_ri_matched` refits that same Model-1 baseline on the
  Model-2/3 46-state sample instead. Both are on a random-intercept-only (`ri`) refit series,
  never the random-slope (`rs`) fits the coefficients come from — under a random slope,
  σ²_u0 is the between-state variance *at the 2017 centering year* and trades off against
  σ²_u1, so it is not bounded to [0, 1] and can go negative (as documented above for the
  log-LMM violations table). The two bases disagree in **opposite directions by outcome**:
  dropping AK/RI/VT *lowers* the inspections Model-1 baseline (so the unmatched column
  *overstates* what the covariates do — inspections M3 is +16.3% against Model 1 but only
  +5.1% against the matched baseline) and *raises* the violations one (so the unmatched
  column *understates* it — violations M3 is +19.7% against Model 1 but +21.5% matched).
- **`icc_ri`** is Nakagawa's observation-level (distribution-specific) variance for a
  log-link NB2, `σ²_u0 / (σ²_u0 + ln(1 + 1/θ + 1/μ))`, all four quantities from the same `ri`
  fit. It is a link-scale approximation and is **not** comparable to this project's headline
  "ICC ~77%", which is a Gaussian LMM quantity on `log(count + 1)`.
- **The cross-check is retired-and-replaced, not extended.** The six-family ladder's
  zero-inflation cross-check has no counterpart here — an NB2-only arm has no ZI component.
  Instead, `nb2_crosscheck_statsmodels.py` fits the same Model-3 fixed effects with state
  dummies in `statsmodels`, which cannot identify the five Level-2 (state-invariant)
  covariates at all (collinear with the dummies by construction) — the check therefore
  covers the Level-1 (time-varying) terms only, and the Level-2 spending/commodity-mix/H-2A
  coefficients are **not** independently confirmed by it.
- **COVID robustness was re-derived under `nbinom2`, not carried over** — the ladder's own
  COVID variants sit under ZINB (inspections) and NB1 (violations), different families. See
  the corrected COVID note above for the violations-specific numbers this produced.
- Full memo: `docs/nb2_stepwise_models.md` (generated, never hand-edited — re-run
  `report_nb2_stepwise.py` instead). Validator: `scripts/validate_nb2_stepwise.py`, sections
  `[0]`-`[10]`; do not quote a single total, read the per-section breakdown (the same
  convention as `validate_count_models.py`).

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

**2026-08-31 convergence fix in `paper_table_models_2021.py` (manuscript tables regenerated).**
The non-converged published fit flagged above is fixed, and fixing it revealed the original
scope note was wrong. `warnings.filterwarnings('ignore')` is removed from that script — it is
what hid the failure — and `fit()` now tries `OPTIMIZERS = [lbfgs, cg, powell, bfgs]` in order,
keeping the FIRST that converges cleanly. `lbfgs` stays first, so every already-sound fit
reproduces bit for bit and only genuinely broken fits move. `fit_report()` prints every
fallback before the JSON is written.

- **A fit is "clean" iff** no *failure* warning was raised, `converged` is set, and
  `|grad| <= GRAD_TOL = 1e-2`. The failure test matters: `"The MLE may be on the boundary of
  the parameter space"` is a LEGITIMATE result (a variance component at zero) that arrives
  with `|grad| ~ 5e-05`, and an earlier draft that disqualified any warning churned four
  already-correct fits onto other optimizers. Only `FAILURE_WARNING_MARKERS`
  (`failed`, `not converge`, `did not converge`) disqualify. The `converged` flag alone is
  also insufficient — statsmodels can set it while the gradient is far from zero.
- **Three distinct fits were broken, not one:**
  | fit | lbfgs | used | recovered |
  |---|---|---|---|
  | `log_inspections ~ cubic` (baseline refit) | loglik −575.313, \|grad\| 174.6 | `bfgs` | −567.725, +7.59 |
  | `log_violations ~ log_insp + cubic` (M1) | loglik −737.667, \|grad\| 76.8 | `cg` | −729.532, +8.14 |
  | `log_violations ~ log_insp + cubic` (baseline refit) | loglik −696.030, \|grad\| 80.7 | `cg` | −687.533, +8.50 |
  (5 fallback events; the two baselines are refit more than once.)
- **What moved in the JSON: 44 numeric values. What moved in print: 9 cells.** Every changed
  key is either a random-SLOPE quantity or a violations-M1 coefficient. **Zero `*_ri` values
  changed**, so the published State-to-State Variation block — which reads `sigma2_u0_ri` /
  `delta_pct_ri` — is untouched, exactly as the earlier note claimed.
- **The 9 changed `.docx` cells are all Table 3 (violations) Model 1:**
  b(log inspections) 0.495→0.518, SE 0.052→0.056, β 0.394→0.412; `time` **0.062+ → 0.066\***
  (a significance-star change, p 0.0682 → 0.0285), SE 0.034→0.030, β 0.128→0.134; `time2`
  SE 0.005→0.006; `time3` −0.005→−0.006, β −0.257→−0.264. Table 2 (inspections) is unchanged
  in print.
- **The unpublished `delta_pct` (random-slope basis) was badly wrong and is now corrected** —
  inspections M2/M3 go +41.97%/+40.82% → −7.15%/−9.27%. This never reached a table (the
  published basis is `delta_pct_ri`), but anything quoting `delta_pct` from the old JSON is
  wrong. Note the corrected values are negative, which is the documented random-slope
  centering artifact, not a new defect.
- Regenerated: `data/generated/paper_table_params_2021.json` and
  `docs/WPS_Table_Sheels_augmented_2021.docx` (via `fill_wps_tables_docx_augmented.py 2021`).
  The 2019-window artifacts are untouched. **The 2019 arm has a known analogous problem**
  (`validate_count_models.py` item 5 finds 2 of 12 published 2011–2019 fits carrying a
  gradient failure under `lbfgs`) and has NOT been given this treatment — that is the obvious
  next piece of work.
- **Known limitation of `validate_nb2_stepwise.py` section [8], found 2026-08-31.** The
  committed-history arm diffs `git merge-base main HEAD`..HEAD. On a feature branch that is
  the branch point and the check is real; **on `main` the merge-base IS HEAD**, so the range
  is empty and that arm passes vacuously. Only the working-tree arm still bites there. The
  guard therefore verifies "this branch did not touch the frozen artifacts", not "they have
  never changed" — which is the right question while a branch is in flight and the wrong one
  afterwards. The 2026-08-31 convergence fix legitimately modified `paper_table_params_2021.json`
  and `WPS_Table_Sheels_augmented_2021.docx`; that is authorized work, not a freeze violation,
  because the freeze bound the NB2 arm's own commits and it kept that promise.

**2026-08-31 same convergence fix applied to the 2019 arm (`paper_table_models_corrected.py`).**
Identical treatment to the 2021 arm above: `warnings.filterwarnings('ignore')` removed, `fit()`
given the same `OPTIMIZERS` ladder, `GRAD_TOL` and `FAILURE_WARNING_MARKERS`, and a
`fit_report()` before the JSON write. **Keep the two scripts' `fit()` in sync.**

- **Exactly 2 fits were broken, independently confirming what `validate_count_models.py`
  item 5 predicted** (inspections M1 and violations M2, both random-slope):
  | fit | lbfgs | used | recovered |
  |---|---|---|---|
  | `log_inspections ~ cubic` (M1) | loglik −419.372, \|grad\| 69.2 | `cg` | −399.553, **+19.82** |
  | `log_violations ~ log_insp + cubic + SPEND_APP_z + SPEND_WORK_z + lii_2017_z` (M2) | loglik −393.616, \|grad\| 21.9 | `cg` | −392.626, +0.99 |
  The inspections M1 error is **19.8 log-likelihood units** — far larger than anything in the
  2021 window, where the worst was 8.5.
- **48 JSON values changed; 0 published `*_ri` values changed**, so as in 2021 the
  State-to-State Variation block is unaffected. But the printed coefficient changes here are
  **more consequential than 2021's**, and two of them change what the paper can say:
  - **Inspections M1 `time`: `-0.069*` → `-0.069***`.** The coefficient is identical; the SE
    nearly halves (0.0339 → 0.0203), moving p from 0.041 to 0.0007. The non-converged fit was
    *understating* the precision of the inspections time trend.
  - **Violations M2 `SPEND_APP_z`: −0.159 (n.s.) → −0.172+ (p < .10).** A spending predictor
    moves from null to marginally significant. SE 0.1079 → 0.1019.
  - **Violations M2 `lii_2017_z` SIGN FLIP: +0.0388 → −0.0052.** Non-significant either way,
    so nothing is claimed on it, but any prose describing its direction is now wrong.
  - Violations M2 `SPEND_WORK_z` −0.315* → −0.329*, `log_inspections` 0.172** → 0.178**.
- Regenerated: `paper_table_params_corrected.json`, `docs/WPS_Table_Sheels_augmented.docx`
  (15 cells) and `docs/WPS_Table_Sheels_filled_corrected.docx` (9 cells). Both 2019 docx now
  differ from the versions circulated on 2026-08-14.
- **Both windows are now fixed.** `paper_table_models.py` (the ORIGINAL pre-correction 2019
  script) still carries `filterwarnings('ignore')` and has NOT been treated — it feeds
  `WPS_Table_Sheels_filled_cubic.docx`, which is superseded and not in use.

**2026-09-10 Jafari-aligned crossed random-effects models (standalone arm).** Three PI
directives: random intercepts only (**no random slope anywhere**); crossed random effects of
**state and year**, substituting time for Jafari's industry dimension; and a best-fitting
family selected **independently per outcome** — the PI explicitly retracted the NB2 arm's
symmetry rationale ("I was wrong; let's not substitute model symmetry for precision"). Plus a
methodological correction: the zero-inflation component now carries the **same predictors as
the conditional component** wherever estimable, and **both blocks are reported** in Jafari
Table 5 format. Design spec:
`docs/superpowers/specs/2026-09-10-jafari-crossed-re-design.md`. Plan:
`docs/superpowers/plans/2026-09-10-jafari-crossed-re.md`. Memo:
`docs/jafari_crossed_models.md` (**generated**, never hand-edited).

```bash
Rscript scripts/r_env_check.R
Rscript scripts/jafari_crossed_models.R data/generated/jafari_crossed_gaussian.json --gaussian-only
Rscript scripts/jafari_crossed_models.R data/generated/jafari_crossed_results.json
python3 scripts/report_jafari_crossed.py
python3 scripts/validate_jafari_crossed.py   # sections [0]-[10]
```

- **`glmmTMB` already fit the two components simultaneously.** The PI's diagnosis that Jafari
  fits the structural-zero and conditional models jointly while ours did not was right about
  the *output* and wrong about the *estimation*: `ziformula` is estimated in the same
  likelihood, and always was. The two real gaps were that the ladder set `ziformula = ~1`
  where Jafari mirrors the full predictor list, and that the reporter never printed a ZI
  block. Both are closed here; **no new estimation machinery was needed.**
- **Result: a zero-inflated family wins 3 of 4 cells, and the FULL Jafari ZI mirror is
  estimable in all three** — `insp_2019` ZI-NB2 (AIC 3236.50), `insp_2021` ZI-NB2 (4585.73),
  `viol_2021` **ZI-NB2 + ZI-RE** (3592.05), each with all 10 conditional predictors also in
  the ZI block. The spec predicted only `viol_2021` could carry the mirror; that was too
  pessimistic. Dropping the random slope for crossed random *intercepts* frees enough
  parameters for a 10-parameter ZI block. **`viol_2019` is the exception**: plain NB2, with
  all ZI rungs sitting at exactly plain-AIC + 2.00 — one parameter buying zero likelihood —
  consistent with that cell being zero-**deflated** in the six-family ladder too. 10 fits are
  `zi_degenerate`; 6 fail to converge (all `*_re` rungs).
- **Winner stability:** `insp_2019` is the one cell where the M1 and M3 winners differ
  (ZI-NB2 + ZI-RE → ZI-NB2). The other three are stable.
- **ZI strength, 25 of 27** against its own negative-binomial counterpart (the comparison that
  isolates inflation while holding the parameterisation fixed). The **2 losses are both
  ZI-NB1 + ZI-RE vs NB1** (`insp_2019` M3, −0.65 AIC; `insp_2021` M1, −7.35). Do **not** write
  that a zero-inflated family never lost here — that was true of the previous arm's matched
  set, and is false of this one. ZI-NB2 specifically is 6–0 against NB2 and 6–0 against NB1.
  This arm fits **ZI-NB1 rungs**, closing the previous ladder's documented blind spot, so an
  NB1 win is now tested against its own zero-inflated counterpart rather than winning by
  default.
- **KEY METHODOLOGICAL FINDING: the year random effect is empty in 3 of the 4 count models.**
  σ²_year = 2.8e-10 (`insp_2019`), 3.7e-07 (`insp_2021`), 2.9e-09 (`viol_2021`) — all pinned
  at the boundary — against 3.07e-02 for `viol_2019` alone. The 2020/2021 year BLUPs are
  ~1e-05 or smaller. The fixed cubic absorbs the smooth year variation, leaving the crossed
  year dimension nothing to explain. **The crossed state × time structure therefore behaves
  as a state-only random intercept in practice**, and any reading of this arm should say so
  rather than implying the year dimension is doing work. This is the cubic/`(1|year)` overlap
  the spec flagged in §4.3, materialising.
- **No COVID indicator.** `(1|year)` absorbs the pandemic shock by construction; a dummy would
  compete with the term containing it. Year BLUPs are reported instead — and per the finding
  above they are ~0, so this arm says nothing about COVID. **Not** comparable to the other
  arms' COVID coefficients (−0.590*** log-LMM inspections; −0.279 NB1 / −0.188 NB2 violations).
- **Structural-zero probabilities are the SAMPLE MEAN of the per-observation ZI probability**
  (`zi_prob_mean`, from `predict(type="zprob")`), never `plogis(b0)` and never the
  Gauss-Hermite marginal `E[plogis(b0+u)]`. Both of the latter evaluate the ZI linear
  predictor at x = 0, which is the whole story ONLY for an intercept-only ZI block; with
  `log_inspections` (sample mean ~3.5, never z-scored) in a mirrored ZI formula, x = 0
  describes no state in the data. A draft of this memo reported **0.4639** for `viol_2021`
  while the same fitted model simulates 78 zeros in 501 rows. Correct values: 0.0094 /
  0.0116 / 0.1193, each just below its observed zero rate (0.0095 / 0.0138 / 0.1717), as
  must hold since structural zeros are a subset of all zeros. The intercept-based quantities
  are withheld at any non-`intercept` tier, and `validate_jafari_crossed.py [7]` enforces
  both the withholding and the bound that caught the defect.
- **ZI predictors are always a SUBSET of conditional predictors.** This keeps one analytic
  sample per (cell, model), which is what makes AIC comparable across ZI tiers — asserted in
  `[4]`, not assumed. Consequently at M1 the `covariates`/`reduced` tiers are skipped (empty)
  and at M2 `reduced` duplicates `covariates`; every skip carries a recorded reason.
  **Within a family the ZI ladder is a FIDELITY rule, not an AIC rule** (richest estimable
  mirror); AIC is used only ACROSS families.
- **Four cells, not six.** The violations-as-offset variant is dropped: the `log(inspections)`
  offset deletes every zero-inspection state-year, and in 2011–2021 all 7 rows it drops are
  also zero-violation rows — it throws away the structural zeros the arm exists to model.
- **Not comparable to `docs/count_models_zinb.md`.** Different random-effects structure, so
  the fixed effects condition on a different variance partition. σ² here is on the **log
  link** scale; **no Δσ² percentage is computed in this arm at all** (that needs a
  random-intercept-only refit series against a single Model-1 baseline, which this ladder
  does not produce).
- **The crossed-RE R/Python bridge is gated independently.** The previous arm's Gaussian gate
  validated `(1 + time | state)` and says nothing about a crossed structure.
  `validate_jafari_crossed.py [2]` fits the same Gaussian crossed model in
  `statsmodels.MixedLM` via `vc_formula` and matches `glmmTMB` to ~7 significant figures.
  Its variance comparator is **boundary-aware with a scale-relative floor** (1e-4 × σ²_e),
  because σ²_year is at the boundary in 3 of 8 Gaussian fits and a relative tolerance between
  two near-zero numbers is not a test.
- **Nothing existing was modified.** Section `[10]` guards the frozen artifacts; its history
  arm is vacuous on `main` (same known limitation as `validate_nb2_stepwise.py [8]`), so this
  arm was developed on the `jafari-crossed-re` branch where the question it asks is the right
  one.

**2026-09-17 post-meeting arms (PI meeting 2026-09-11).** Joe walked through
`docs/jafari_crossed_models.md` in the 2026-09-11 meeting and assigned three pieces of work,
plus one scope direction. All three are implemented as **standalone arms**; the crossed arm
(`jafari_crossed_models.R`) is untouched except for a presentation fix to its reporter. Run
order for the two new arms (each is independent of the other):

```bash
# Stepwise state-to-state variance (Joe's Thing 1 + Thing 2)
Rscript scripts/jafari_stepwise_models.R      # → data/generated/jafari_stepwise_results.json
python3 scripts/report_jafari_stepwise.py     # → docs/jafari_stepwise_variance.md
python3 scripts/validate_jafari_stepwise.py

# Violations per inspection (Joe's Thing 3)
Rscript scripts/jafari_ratio_models.R         # → data/generated/jafari_ratio_results.json
python3 scripts/report_jafari_ratio.py        # → docs/jafari_ratio_models.md
python3 scripts/validate_jafari_ratio.py
```

Both consume `data/generated/count_model_panel_2021.csv` (WPS view, 2011–2021, RAW counts) and
nothing else, so the 2021-extension pipeline must have been run first.

**Scope direction: WPS view only.** At 18:44 in the meeting, after reading the ECHO dashboard
definitions aloud, Joe settled a question that had been open: the **establishments** view
covers pesticide *producers, sellers and distributors*, not WPS-protected agricultural
operations, so reporting focuses on the **WPS view**. Both new arms are WPS-only (2011–2021).
The 2011–2019 establishments cells are **kept** in `docs/jafari_crossed_models.md` as the
record of what was fit — do not delete them; `validate_jafari_crossed.py` checks them and the
memo flags them as the record rather than the reported results.

**2026-09-17 stepwise state-to-state variance arm.** The crossed arm entered every covariate
simultaneously and computes no variance reduction at all, which Joe named as the single thing
blocking submission ("the only thing that we're missing at the moment is state to state
variation"). This arm supplies it. Three cells: `insp_2021`; `viol_2021_wi` (violations WITH
`log_inspections`); `viol_2021_ni` (violations WITHOUT it — Joe's Thing 2). M1 (base) → M2
(+`SPEND_APP_z`, `SPEND_WORK_z`, `lii_2017_z`) → M3 (+`h2a_per_farmworker_z`,
`dol_demand_met_pct_z`, `pct_flc_z`), crossed random intercepts `(1|state) + (1|year)`, no
random slope.

- **The family is held FIXED across M1/M2/M3 within a cell** (`zinb2` / `zinb2_re` / `zinb2`).
  This is the arm's central methodological commitment: the crossed ladder re-selects a family
  per model, and under that policy a σ²_state change between steps reflects the family, not
  the covariates. Same reasoning that motivated the NB2-only arm. `viol_2021_ni` had no
  precedent and was raced at M3 across the same 8-family ladder — **ZINB2 wins** (AIC 3727.78;
  next is ZINB2+ZI-RE 3742.09, then NB2 3781.74).
- **Two ZI series, and the headline is `zi_intercept`, not `mirror`.** Under a mirrored ZI
  block the zero-inflation predictors grow as the conditional block grows, so a σ²_state
  reduction confounds the two. `zi_intercept` holds `ziformula = ~1` constant across all
  steps and is the reduction series. `mirror` is reported beside it and is **explicitly not**
  the reduction series.
- **Correctness anchor**: the `mirror` series reproduces `jafari_crossed_results.json` on all
  six comparable values to **0.00e+00** (insp M1/M2/M3 = 1.0992/0.9300/0.9162;
  viol M1/M2/M3 = 0.8358/0.8302/0.7507). If a future change breaks that, the arm has drifted.
- **Both reduction bases are reported and they disagree in OPPOSITE directions by outcome** —
  the same trap the NB2 arm documented. M1 keeps 49 states; M2/M3 drop AK/RI/VT (no BLS
  applicator series) to 46, so an `M1matched` baseline is refit on the M2/M3 sample.
  Inspections M3 is **+16.6% unmatched but only +5.0% matched** (dropping AK/RI/VT *lowers*
  that baseline, so the unmatched column **overstates** the covariates); both violations cells
  go the other way — `viol_2021_wi` M3 is +10.4% unmatched, **+15.5% matched**, and
  `viol_2021_ni` M2 is **negative** (−1.0%) unmatched but +5.7% matched. **Never quote one
  column alone** — it misreports every cell, and in different directions.
- **The violations cells differ in TWO ways, not one — so 45% is an UPPER BOUND, not an
  attribution.** M1 σ²_state is 1.5329 without `log(inspections)` and 0.8411 with it, a gap of
  about 45%. It is tempting, and **wrong**, to write that `log(inspections)` absorbs 45% of the
  between-state variance by itself. The two cells also carry **different families**:
  `viol_2021_wi` is **ZINB2 + ZI-RE** (`ziformula ~1 + (1 | state)`, ZI random-intercept
  variance **1.741** at M3) because it inherits the crossed arm's M3 winner, while
  `viol_2021_ni` is **ZINB2** (`ziformula ~1`) because it raced its own ladder and ZINB2 beat
  ZINB2+ZI-RE by 14.31 AIC. Per-cell family selection is correct by design, but it means
  between-state heterogeneity in the zero process has somewhere to go in `wi` and **nowhere to
  go in `ni`**, where it must load onto the conditional σ²_state. **Do NOT write
  "log(inspections) absorbs 45%."** The memo derives this comparison at run time from
  `family_tag` + `zi_formula` and prints the clean one-term sentence only if the two cells ever
  do share a family.
- **RESOLVED 2026-09-17 — the matched-family companions are fit and the memo reports a
  BRACKET.** Eight companion fits were added to the headline `zi_intercept` series:
  `viol_2021_ni` under ZINB2 + ZI-RE and `viol_2021_wi` under plain ZINB2, each at
  M1/M2/M3/M1matched. **Both directions were fit deliberately** — each forces one cell onto a
  family that loses for it, so neither is privileged and together they bracket the answer.
  All 8 converged. Absorbed share = (σ²_without − σ²_with)/σ²_without:

  | Model | as selected (confounded) | both at ZINB2+ZI-RE | both at plain ZINB2 |
  |---|---|---|---|
  | M1 | 45.1% | **39.5%** | **40.1%** |
  | M3 | 49.8% | **45.2%** | **47.8%** |

  The confounded figure sits **above both matched ends at every step**, so the upper-bound
  framing was right and the confound was inflating it by 5.1 points at M1 and 2.1 at M3. The
  range is a **bracket, not a confidence interval**. Report it as: holding the family constant,
  `log(inspections)` accounts for **39.5–40.1% of the between-state variance at Model 1 and
  45.2–47.8% at Model 3**. **The answer to Thing 2 is yes, the inspections term is needed.**
- **The companions are NOT reported models.** Each cell's primary series keeps its own selected
  family; companions carry `is_companion: true`, `family_matched_to`, `companion_of` and a
  `__companion` key suffix, and the validator excludes them from the family-held-fixed check
  `[3]` and the selection check `[7]` **by an explicit rule**, asserting each is flagged and
  off-family rather than skipping them by accident.
- **The `viol_2021_ni` family ranking REVERSES at the ZI tier the headline series uses — worth
  a PI note.** The race picked plain ZINB2 (AIC 3727.78) over ZINB2+ZI-RE (3742.09), but those
  were compared at *different ZI tiers*: ZINB2 reached the full `mirror` (10 ZI params),
  ZINB2+ZI-RE only `reduced`. **With the ZI block held at the intercept — which is what the
  headline series does — ZINB2+ZI-RE beats plain ZINB2 by 15.03–24.99 AIC at every step.** This
  does not overturn the selection (the race follows the arm's stated fidelity-then-AIC rule),
  but it means the 39.5% / 45.2% end of the bracket is the **better-fitting** end at the
  specification actually in use. Never call either family "best-fitting" without naming the ZI
  tier. Memo §4.4, derived at run time.
- **The two violations cells sit on byte-identical state-year sets** (533 at M1, 501 at M2/M3)
  despite `log_inspections` appearing in only one rhs — that column has no missingness beyond
  `violations`. Now asserted per state-year, so the bracket cannot silently become a sample
  effect.
- **What is NOT confounded**: removing `log(inspections)` **flips one significance verdict** —
  `time` goes from b = 0.122, p < .0001 to b = 0.056, p = 0.069 at M3. No spending or labour
  covariate flips. The variable is not inert.
- **The covariates LOSE on AIC everywhere** (+3.4 to +6.5 against the matched M1 baseline)
  even in the cells where they cut σ²_state. Variance reduction and model selection are
  different questions; the memo carries an `AIC vs matched M1` column and says so without
  editorialising. Do not report the variance reduction without this.
- **σ²_year is NOT uniformly at the boundary here**, contrary to the crossed arm's pattern:
  `insp_2021` 9.8e-10 and `viol_2021_wi` 5.0e-09 are pinned at zero, but `viol_2021_ni` is
  **1.2e-03** — small (0.081% of state+year variance) but not boundary. The conclusion still
  holds that the crossed state × year structure behaves as a **state-only random intercept**,
  but the claim must be derived per cell, not asserted for all of them.
- One fit failed to converge: `viol_2021_ni__M3__zinb1_re__selection`, a losing
  family-selection candidate. **Every fit in every reported series converged.** The memo
  separates "failed in a reported series" from "failed as a selection candidate" rather than
  giving a single count.
- **Validator**: `scripts/validate_jafari_stepwise.py`, sections `[0]`–`[10]`, **4,205 pass /
  0 fail / 1 skip** (the skip is reasoned — `viol_2021_ni` has no crossed-arm anchor, which is
  *why* its family was raced). Do not quote the total alone; `[1]`, `[4]` and `[9]` are record
  loops. `[0]` runs `validate_jafari_crossed.py --skip-precondition` as an upstream gate. `[8]`
  is a Gaussian round-trip on **this arm's own 12 specifications** — `viol_2021_ni`, Model 2
  and the `M1matched` sample construction are all invisible to `validate_jafari_crossed.py
  [2]`, and those are exactly where a sample-assembly bug would live. It uses the project's
  live-vs-live multi-optimizer idiom (`powell` wins 3 of 12; statsmodels' default `lbfgs`
  quits early on a near-flat σ²_year ridge at the *same* REML loglik to 8 decimals — not a
  package disagreement). The validator is **mutation-tested**: 8 deliberate corruptions
  (falsified anchor, family swapped at M2, matched baseline replaced by the unmatched one,
  perturbed coefficient) each produce the expected FAIL.
- **`[10]`'s history arm is scoped per commit, not per branch.** This branch carries three
  arms, so a whole `merge-base..HEAD` diff flags the other two as freeze violations. It
  asserts the promise CLAUDE.md actually states: every commit touching a `jafari_stepwise_*`
  path must touch nothing frozen. Vacuous on `main` (same known limitation as
  `validate_nb2_stepwise.py [8]`).

**2026-09-17 violations-per-inspection arm.** Joe's Thing 3: a rate outcome, stepwise, with
inspections removed as a covariate because it is now the denominator. His rationale (33:06)
is that Jafari et al. had only a violations count and could not separate "no violations" from
"never inspected", so a rate sits closer to their data.

- **A ratio cannot be swapped in as a count DV** — `glmmTMB`'s NB/ZINB families need
  non-negative integers. The count-preserving form is an **offset**, `log(inspections)`.
- **This deliberately re-opens something the crossed arm closed, and the two notes do not
  contradict each other.** The crossed arm's "Four cells, not six" bullet above drops the
  offset variant because the offset deletes every zero-inspection state-year and therefore
  the structural zeros that arm exists to model. That reasoning stands **for that arm**. Here
  the offset is the point of the exercise, and the empirical question — does zero-inflation
  survive it — is answered rather than assumed. It does (below). Cite whichever arm the
  surrounding analysis is actually using.
- **Three specs, so the trade-off is visible instead of decided silently**: **A** `nbinom2` +
  `offset(log(inspections))`, `inspections > 0`; **B** `zinb2` with the same offset, ZI
  mirroring the conditional block; **C** Gaussian LMM on `log((violations+1)/(inspections+1))`,
  all available rows. **Never AIC-compare C to A or B** — different likelihood, different
  outcome scale. A vs B is valid (same rows, same conditional formula, ZI block the only
  difference).
- **Zero-inflation survives the offset decisively**: the full Jafari-style mirror is
  estimable, converged and non-degenerate at all three steps, and B beats A by **75.88 /
  88.29 / 80.13 AIC** at M1/M2/M3. So the offset does not destroy the zeros. Mean
  Pr(structural zero) 0.0892/0.0810/0.0869 against observed zero rates 0.1635/0.1599/0.1599 —
  below the observed rate in every model, as must hold.
- **The offset costs 7 of 533 rows** — CT-2018, KS-2018, MA-2018, OR-2020, OR-2021, UT-2017,
  WV-2012, all 7 zero-violation, none positive — **derived from the panel, not asserted**
  (`__offset_derivation`). Spec C keeps them and reproduces A's substantive picture, which is
  the evidence that the dropped rows are not driving conclusions.
- **NEW, recorded nowhere else: 6 rows have a MISSING violation count** (DE/IN/LA/ND/VT/VA,
  all 2011) and drop from **every** spec including C. So the all-rows baseline is **533, not
  539** — do not write that Spec C keeps all 539 rows.
- **Spec B is the recommendation, and it reinterprets the spending result.** Adding the ZI
  block moves both significant Model-3 conditional predictors to null (`SPEND_WORK_z`
  −0.394\* → −0.226 n.s.; `h2a_per_farmworker_z` −0.150\* → −0.013 n.s.) while `SPEND_WORK_z`
  re-emerges in the **ZI block at +0.714\*\* (OR 2.04)**. Under the sign convention below that
  says STAG spending per farmworker predicts a state being a **structural zero**, not a lower
  violation rate among states that are enforcing. That is a different claim from the one
  sketched in the meeting (30:25) and from what Spec A or C supports on their own.
- σ²_year is at the boundary in 8 of 9 fits (largest anywhere 1.0e-03), so as elsewhere the
  crossed structure behaves as a state-only random intercept. No COVID indicator is fit; this
  arm says nothing about COVID.
- **Validator**: `scripts/validate_jafari_ratio.py`, sections `[0]`–`[13]`, **2,109 pass / 0
  fail / 5 skip**. Do not quote the total alone; `[1]` is a loop over 13 records. `--skip-slow`
  bypasses `[9]`/`[12]`/`[13]` and says so. `[9]` is **the first offset gate in this project** —
  nothing previously validated an offset specification across the R/Python bridge — and a
  deliberate probe confirms the offset is *applied* (the intercept moves ~3.5 when removed).
  `[7]` reconstructs the structural-zero probability from the panel plus the stored
  coefficients rather than round-tripping it (legitimate here because the ZI formula carries
  no random term, so `predict(type="zprob")` is exactly `plogis(x'β_zi)` row by row); it agrees
  to <1e-8 and is demonstrably **not** `plogis(b0)` (0.0869 vs 0.1143 at M3). Mutation-tested,
  including setting `zi_prob_mean` to the historical wrong estimand — caught.
- **A near-miss worth remembering**: statsmodels' default optimizer genuinely fails on the
  Spec C M3 surface, stopping 0.07 loglik short and reporting σ²_state 0.8522 against
  glmmTMB's 0.8159. A naive gate would have reported that 4.3% gap as an arm defect; `powell`
  and `nm` both reach the same optimum as glmmTMB to ~6 significant figures. The
  discard-failures/take-highest-loglik idiom is **load-bearing here, not decorative**.
- **Latent hazard, currently inert**: `analytic_rows()` in `jafari_ratio_models.R` runs
  `complete.cases()` over the DV and predictors but **not** over `inspections`, then filters
  `inspections > 0`. An `NA` denominator would survive `complete.cases` and then index with
  `NA`, injecting all-NA rows. Inert only because that column has no missing values today;
  `[0]` asserts that precondition so it cannot go live silently. Not fixed — there is no
  defect in the current output.
- Two memo-wording imprecisions are pinned by the validator rather than fixed: the ZI
  probability is described as "conditional on the fitted random effects" when this arm has no
  ZI random effects (vacuous, but a reader could infer one exists), and the "Zeros (rate)"
  column counts zero-**violation** rows in every spec including Spec C, whose response is the
  log ratio. Both are deliberate in the R; `[4]` pins the estimand so it cannot drift.

**2026-09-17 the two model blocks must never be presented unlabelled again.** In the meeting
the PI read `docs/jafari_crossed_models.md` aloud and made two inverted readings, then said he
would draft the results and discussion from them. Both traced to the memo's presentation, and
both are now structurally prevented:

- **He swapped the blocks** — "is there anything predicting the number of zeros? So that's the
  conditional model" (29:56) is backwards. Because of it he reported the `insp_2021` H-2A
  finding as predicting *more zeros*, when `h2a_per_farmworker_z` is **+0.0874, p = 0.028, in
  the CONDITIONAL (count) block** (more inspections) and −4.16 **n.s.** in the ZI block.
- **He read raw coefficients as risk ratios.** The memo printed log/logit coefficients;
  `exp(b)` existed only in the JSON and only for the conditional block. He read ZI
  `SPEND_APP_z = 1.3042` as "a 30% increase in the incidence of having an inspection" (26:00).
  It is a logit coefficient: **OR = 3.68**, ~3.7× the odds of being a structural zero, i.e.
  **FEWER** inspections. Same inversion applies to ZI `dol_demand_met_pct_z` = 1.6606
  (OR 5.26) and to the violations ZI `SPEND_WORK_z` = +0.7595 (OR 2.14).
- **THE SIGN RULE, which every memo in this project must now carry**: the conditional block is
  the **COUNT** model (expected count given the state-year is NOT a structural zero); the ZI
  block is the **log-odds of BEING a structural zero**. A **positive ZI coefficient means MORE
  structural zeros and therefore FEWER events** — the opposite direction from a positive
  conditional coefficient. An IRR and an OR are not interchangeable and neither may be quoted
  as the other.
- `report_jafari_crossed.py` now prints an `exp(b)` column labelled **IRR** in the conditional
  block and **OR** in the ZI block (the OR was not computed anywhere before), gives both blocks
  self-describing headers, and carries a "How to read these two blocks" section whose
  worked-example numbers are **read from the JSON** so they cannot drift from the tables below
  them. `exp(b)` is an em dash on intercept rows — exponentiating an intercept describes a
  state-year with every predictor at zero, including `log Inspections = 0`, which no state
  occupies. **No estimate, SE or p-value changed**; this was presentation only.
- `validate_jafari_crossed.py`'s `MEMO_REQUIRED` asserted the literal strings
  `*Conditional Model*` / `*Zero-inflated Model*` — the exact labels that were misread — so
  they are retargeted to the new headers, plus four new guards. Sections `[0]`–`[10]`:
  **774 pass / 0 fail / 3 skip** (the 3 skips pre-existing and unchanged).
