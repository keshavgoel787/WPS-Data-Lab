# Jafari-aligned crossed random-effects count models — design

**Date:** 2026-09-10
**Status:** design, approved for planning
**Supersedes:** nothing. This is a standalone arm alongside
`docs/superpowers/specs/2026-08-20-multilevel-zinb-design.md` (the six-family ladder) and
`docs/superpowers/specs/2026-08-28-nb2-stepwise-design.md` (the NB2-only stepwise arm).
Neither is modified.

---

## 1. Purpose

Three PI directives, received together on 2026-09-10, none of which the existing arms satisfy:

1. **Random intercepts only.** Jafari et al. (*PLOS ONE* 2024,
   doi:10.1371/journal.pone.0302960) fit random intercepts, not random slopes. Our ladder
   mandates `(1 + time | state)`. Drop the random slope.
2. **Crossed random effects.** Jafari's design is crossed `(1|state) + (1|industry)`. This
   project has no industry dimension. Substitute **time** for industry, giving
   `(1 | state) + (1 | year)`.
3. **Per-outcome family selection, no symmetry constraint.** The PI explicitly retracted the
   symmetry rationale behind the NB2-only arm: *"I was wrong; let's not substitute model
   symmetry for precision."* Inspections and violations each get their own best-fitting
   family, chosen on AIC.

And one methodological correction, from the PI's reading of Jafari's Table 5:

4. **The zero-inflation component carries the same predictors as the conditional component,
   and both blocks are reported.**

### 1.1 What directive 4 does and does not mean

The PI's phrasing was that Jafari "fit the structural zero model and the conditional model
simultaneously, and I do not see those distinctions in our output or tables." The diagnosis
of the *output* is correct; the diagnosis of the *estimation* is not, and the distinction
matters for scoping this work.

`glmmTMB` already fits the conditional and zero-inflation components jointly, in a single
likelihood. That is how the existing ladder's `zinb`/`zinb_re` rungs were estimated, and it
already matches Jafari. **No new estimation machinery is required.** The two genuine gaps,
both of which are ours to close, are:

- the existing ladder sets `ziformula = ~1` — a bare intercept — where Jafari mirrors the
  full conditional predictor list; and
- `report_count_models.py` prints the conditional block plus that single ZI intercept, never
  a separate labelled ZI block.

This spec closes both. It does not change how the likelihood is formed.

---

## 2. Diff against the existing six-family ladder

| Dimension | Existing arm (`count_models_zinb.R`) | This arm |
|---|---|---|
| Conditional RE | `(1 + time \| state)`, with `(1 \| state)` fallback tier | `(1 \| state) + (1 \| year)`, **no fallback** |
| RE tier machinery | `rs`/`ri` tiers, `__ri2` keys, no cross-tier AIC | none — one structure |
| Families | 6; all ZI rungs NB2-based | 8; NB1 and NB2 each get a ZI counterpart |
| ZI formula | `~1`, or `~1 + (1\|state)` | tiered ladder, richest estimable mirror |
| ZI reporting | conditional block + one ZI intercept | two labelled blocks, Jafari Table 5 format |
| Cells | 6 (incl. violations-as-offset) | 4 (offset variant dropped, §3.2) |
| Selection | per cell, per tier | per cell, single structure |
| COVID check | indicator under selected family | replaced by year random effects (§7.3) |

Coefficients from this arm are **not** line-by-line comparable to the existing memo: the
random-effects structure differs, so the fixed effects are conditional on a different
partition of the variance. Any comparison between the two arms must say which it is quoting.

---

## 3. Data and cells

### 3.1 Inputs

Both panels already exist and already carry every column needed, including `year`. Nothing
upstream is rebuilt. `jafari_crossed_models.R` opens **both** panels by absolute path, as
`count_models_zinb.R::build_cells()` does — the dual-window design is not optional — so its
single command-line argument is the output JSON, not a panel.

- `data/generated/count_model_panel_2019.csv` — 450 rows, 50 states, years 2011–2019
  (ECHO **establishments** view)
- `data/generated/count_model_panel_2021.csv` — 539 rows, 49 states, years 2011–2021
  (ECHO **WPS** view; Wyoming has no row in this view)

The two windows are **different measures, not a longer one** — WPS-view counts run 2–3× the
establishments-view counts and correlate ≈0 from 2017 on. Both are fit precisely so a family
difference can be attributed to window-vs-data-source rather than to model class.

### 3.2 Cells

Four cells. The violations-as-offset variant present in the existing ladder is **dropped**,
for a substantive reason and not only economy: the `log(inspections)` offset is undefined at
zero and therefore deletes every zero-inspection state-year. In the 2011–2021 window all 7
rows it drops are *also* zero-violation rows. An offset specification throws away the
structural zeros that this entire arm exists to model.

| Cell | DV | Panel | Conditional base predictors |
|---|---|---|---|
| `insp_2019` | `inspections` | 2019 | `time + time2 + time3` |
| `insp_2021` | `inspections` | 2021 | `time + time2 + time3` |
| `viol_2019` | `violations` | 2019 | `log_inspections + time + time2 + time3` |
| `viol_2021` | `violations` | 2021 | `log_inspections + time + time2 + time3` |

The violations cells enter inspections as `log_inspections` (= `log(inspections + 1)`),
matching published Table 3's form.

### 3.3 Zeros, and what they permit

| Cell | N | Zeros | ZI block realistically identifiable? |
|---|---:|---:|---|
| `insp_2019` | 450 | 15 (3.3%) | intercept, possibly a 3-predictor block |
| `insp_2021` | 539 | 7 (1.3%) | intercept only |
| `viol_2019` | 394 | 10 (2.5%) | intercept only |
| `viol_2021` | 533 | **93 (17.4%)** | plausibly the full Jafari mirror |

**This is the arm's central constraint and it must be stated in the memo, not buried.** A
mirrored ZI block for Model 3 carries 10 logistic parameters. Only `viol_2021` has enough
zeros to identify them. The existing ladder's 3-predictor ZI sensitivity already failed in 3
of 6 cells at a simpler ZI specification. Expect at most one cell to reach the full mirror.
§5 defines how that is handled without silently substituting a different model.

Note also that the 2019 inspection zeros are partly **manufactured**: the establishments
reshape applies `fillna(0)` to the EPA and state inspection components before summing, so a
state-year missing both becomes a zero. This is already derived in the existing pipeline and
the derivation is reused, not re-typed.

---

## 4. Model specification

### 4.1 Conditional random effects — fixed, mandated, no fallback

```
<dv> ~ <fixed predictors> + (1 | state) + (1 | factor(year))
```

for every family, cell and model in this arm. A family that fails to converge at this
structure is recorded as non-converged and **excluded from selection**. It is never
downgraded to a simpler structure. The existing arm needed `rs`/`ri` tier machinery because
`(1 | state)` was a documented convergence fallback from a mandated `(1 + time | state)`;
here there is a single structure, so that machinery is deleted rather than ported.

### 4.2 Fixed effects: the build-up

Unchanged from the existing arms, so the covariate blocks stay comparable.

- **M1** = base predictors (§3.2) only
- **M2** = M1 + `SPEND_APP_z + SPEND_WORK_z + lii_2017_z`
- **M3** = M2 + `h2a_per_farmworker_z + dol_demand_met_pct_z + pct_flc_z`

Family selection races at **M3** — the model a family must survive with every covariate
present. M1 is refit across the full ladder as a stability check, and gets **its own recorded
winner** (`m1_winner`) so that "the winner changed as covariates entered" is a readable fact
rather than an inference; only the M3 winner is the cell's reported family. M2 is fit under
the M3 winner only, and is never a selection competitor.

Each model walks the ZI tier ladder (§5) **independently**, because the mirror is defined
against that model's own conditional predictor set: M2's mirror is not M3's. A model may
therefore reach a different `zi_tier_reached` than its neighbours in the build-up, and that
is recorded per fit.

M1 keeps all states; M2/M3 drop AK/RI/VT, which have no BLS pesticide-applicator series
(`SPEND_APP_z`). That sample change is a property of the covariate block, not of this arm.

### 4.3 The fixed cubic and the year random effect coexist — with a caveat

Per PI direction, `time + time2 + time3` remain **fixed effects** alongside `(1 | year)`.
`time = year − 2017` exactly, so the cubic is a smooth function of the same grouping factor
the year random intercept is built on. The model is estimable — the cubic is the smooth
component and the year random effect the shrunk residual departure from it — but two
consequences must be reported rather than discovered later:

1. **The cubic may lose significance.** The year random effect can absorb the 2016–17 WPS
   revision spike that the cubic exists to capture. If it does, that is a finding about how
   much of the trend is a smooth trend versus a sequence of year-specific shocks, and the
   memo states it as such.
2. **σ²_year is estimated from 9 (2019) or 11 (2021) levels.** That is at or below the
   conventional minimum for a variance component. σ²_year and its standard error are
   reported with this caveat attached, and no substantive claim rests on σ²_year alone.

### 4.4 Family ladder — eight families

| Tag | Conditional family | ZI component |
|---|---|---|
| `poisson` | Poisson | none |
| `nbinom1` | NB1 | none |
| `nbinom2` | NB2 | none |
| `zip` | Poisson | fixed effects (§5) |
| `zinb1` | NB1 | fixed effects |
| `zinb2` | NB2 | fixed effects |
| `zinb1_re` | NB1 | fixed effects + `(1 \| state)` |
| `zinb2_re` | NB2 | fixed effects + `(1 \| state)` |

Two deliberate changes from the existing six:

- **A zero-inflated NB1 now exists.** The old ladder's ZI rungs were all NB2-based, so when
  plain NB1 won both 2021 violations cells it beat a candidate set lacking its own ZI
  counterpart — a documented blind spot that was left open only because closing it would
  have reopened selection. Selection is being reopened here, so it is closed.
- **ZI random intercepts are on `state` only, not `year`.** A year-level ZI random intercept
  would be identified off as few as 7 zeros spread across 11 years. The state dimension is
  where the existing arm found a genuinely large ZI variance (SD 6.04 on the logit scale).

---

## 5. The ZI tier ladder

### 5.1 Tiers

For a ZI family at a given (cell, model), the ZI formula is tried in this order and the
**first that converges cleanly and is non-degenerate** is kept. This is a fidelity rule, not
an AIC rule: within a family we want the richest mirror that is estimable.

| Tier | ZI formula | Rationale |
|---|---|---|
| `mirror` | every fixed predictor in that model's conditional part | Jafari Table 5 |
| `covariates` | the level-2 covariate block only (no time terms, no `log_inspections`) | drops the terms most collinear with `(1\|year)` |
| `reduced` | `SPEND_APP_z + SPEND_WORK_z + lii_2017_z` | matches the existing arm's §5.5 sensitivity |
| `intercept` | `~ 1` | the existing arm's primary ZI specification |

### 5.2 ZI predictors are always a subset of conditional predictors

A ZI tier may never introduce a predictor absent from the conditional model. Two reasons:
it preserves Jafari's mirroring principle, and — decisively — it guarantees that every tier
within a (cell, model) shares **one analytic sample**, because listwise deletion is computed
over the conditional predictor set and the ZI set adds nothing to it. If a `reduced` ZI
block were attached to an M1 whose conditional part has no covariates, `SPEND_APP_z` would
silently drop AK/RI/VT and change N.

Consequences, which are recorded per fit rather than assumed:

- At **M1** the conditional part has no covariates, so `covariates` and `reduced` are empty
  or out-of-subset and are **skipped**; the ladder is `mirror → intercept`.
- At **M2** `covariates` and `reduced` are the same set, so `reduced` is skipped as a
  duplicate.
- Every skip is recorded with a reason (`zi_tier_skipped_reason`), never inferred by a
  reader from a gap in the output.

### 5.3 AIC across ZI tiers is valid, and is used

An earlier draft of this design, and one option card shown to the PI, asserted "no cross-tier
AIC comparison." **That was wrong and is not carried into this spec.** It was reflex from the
old `rs`/`ri` rule, where the tiers were incomparable because they were different
random-effects structures fit as a fallback. ZI tiers are not that. They are nested models on
identical data with an identical conditional part and an identical likelihood, differing only
in the ZI parameterisation, so AIC is comparable across them.

The two rules therefore differ by scope:

- **Within a family** — the tier ladder picks the richest estimable ZI block (§5.1), not the
  lowest-AIC one.
- **Across families** — AIC picks the cell winner, with each competitor entering at whatever
  tier it reached. Every competitor's `zi_tier_reached` is reported alongside its AIC, so a
  reader can see whether the winner won carrying a rich or a bare ZI block.

The sample-invariance property that makes this valid is **asserted by the validator** (§8,
section [4]), not assumed.

---

## 6. Selection and guards

### 6.1 Selection rule

Per cell, at M3: lowest AIC among fits that are converged, non-degenerate (§6.2) and
non-collapsed (§6.3). Winners are chosen **independently per cell** — the two outcomes may
select different families, and per the PI's retraction of the symmetry rationale that is an
acceptable and expected result, not a defect to be smoothed over.

`pick_winner`'s empty-candidate-set fallback is retained and **recorded**
(`winner_defaulted`, `n_eligible`): a defaulted winner is not a selection result and must
never be reported as one.

### 6.2 ZI-boundary degeneracy

Ported unchanged in substance from the existing arm's Ruling R17. A ZI fit is degenerate if
any of:

- `|ZI intercept| > 15`
- `SE(ZI intercept) > 100`
- its log-likelihood matches its non-ZI counterpart (Poisson for `zip`; NB1 for `zinb1*`;
  NB2 for `zinb2*`) within `1e-4`

The existing arm had to guard the log-likelihood criterion with a tier-equality test, because
it was silently inert across tiers. **That guard is unnecessary here and is removed**, since
there is only one RE structure — but the *applicability* of the criterion is still recorded
per fit so the validator can assert it fired wherever a counterpart existed.

A degenerate fit stays in the JSON with its reason. It is excluded from selection, not
deleted.

### 6.3 Collapse

If a `*_re` family's ZI random intercept degenerates such that the fit becomes
formula-for-formula identical to its non-RE twin at the same cell, model and ZI tier, it is
marked `collapsed_to` and treated as a duplicate rather than a distinct competitor.

### 6.4 Structural-zero probability

Wherever a ZI-RE family is reported, the structural-zero probability is reported
**marginally**, as `E[plogis(b0 + u)]` by Gauss-Hermite quadrature, **and** as the
median-state `plogis(b0)`, both explicitly labelled. This is non-negotiable: in the existing
arm the two differed by a factor of ~835 for the headline inspections cell, and reporting
`plogis(b0)` alone as if marginal would have led a reader to conclude the zero-inflation was
negligible. Node count follows the existing implementation (`GH_NODES = 240`, with a
halving-stability assertion at 1e-4 relative).

---

## 7. Reporting

Output: `docs/jafari_crossed_models.md`, generated by `scripts/report_jafari_crossed.py`.
**Generated, never hand-edited** — the same convention as `docs/count_models_zinb.md` and
`docs/nb2_stepwise_models.md`.

### 7.1 The Jafari Table 5 deliverable

One two-block table per cell, for the selected family at M3, in Jafari's layout:

```
Factor                          Estimate      Pr(>|z|)
--------------------------------------------------------
Conditional Model
  Intercept                       ...           ...
  log Inspections                 ...           ...        (violations cells only)
  time / time² / time³            ...           ...
  SPEND_APP_z ... pct_flc_z       ...           ...
Zero-inflated Model
  Intercept                       ...           ...
  <same rows, or "—" where the ZI tier reached excluded that term>
```

Rows the ZI tier did not include are rendered `—` with the tier named in the caption, so the
table never implies a term was estimated at zero when it was simply absent.

Significance follows this project's existing convention (`***` p<.001, `**` p<.01, `*` p<.05,
`+` p<.10), which differs in rendering from Jafari's own legend; the memo states this so the
two tables are not misread against each other.

### 7.2 Supporting tables

- **Family selection** per cell: every fit's AIC, ΔAIC to the winner, `zi_tier_reached`,
  convergence and degeneracy flags.
- **Zero-inflation strength** — the directive-2 deliverable. Pairwise matched comparisons
  (same cell, model, ZI tier; both converged; neither degenerate) of each ZI family against
  its plain counterpart, with a win/loss tally and AIC margin range. This is the "test the
  strength of each zero-inflated specification" the PI asked to repeat, now including the
  previously missing ZI-NB1 rungs.
- **Variance components**: σ²_state, σ²_year, σ²_zi_state, dispersion (with `family_name`, so
  the NB1-vs-NB2 semantics are readable), observed vs expected zeros.
- **M1 → M2 → M3 build-up** under the selected family per cell.

`sigma2_e` is **null for every count fit**, as in the existing arm: `sigma()` is θ for NB2
and a dispersion multiplier for NB1, and exporting its square under a name that reads as
residual variance was a documented defect.

No Δσ² percentage is computed. That requires a random-intercept-only refit series against a
single Model-1 baseline; σ² here is on the **log link** scale and is not comparable in
magnitude to the published tables' `log(count+1)` scale. This is a deliberate gap, stated as
such.

### 7.3 COVID: replaced, not ported

The existing arm fits a 2020–21 COVID indicator. **Under this design that check is close to
meaningless and is not ported.** `(1 | year)` absorbs the pandemic shock by construction, so
a COVID dummy competes directly with the term that already contains it and is near
unidentifiable.

It is replaced by the year random effects themselves: σ²_year plus the 2020 and 2021 year
BLUPs, which show the pandemic drop directly. The memo states why the substitution was made,
and explicitly warns that the resulting numbers are **not** comparable to the existing arm's
COVID coefficients (the ladder's −0.590*** inspections figure, or the family-dependent
violations figures of −0.279 under NB1 and −0.188 under NB2).

---

## 8. Validation

`scripts/validate_jafari_crossed.py`, sectioned `[0]`–`[10]`. Following established project
convention: **do not quote a single headline check count** — the validator prints a
per-section breakdown and an explicit `SKIP` line for every bypassed gated block, and some
sections are per-record loops rather than independent findings.

| § | Check |
|---|---|
| `[0]` | **Precondition.** Re-run the existing Gaussian R↔Python gate (`count_models_zinb.R --gaussian-only` vs `statsmodels.MixedLM`). Nothing here is trustworthy if the bridge itself is broken. Same pattern as `validate_nb2_stepwise.py [0]`. |
| `[1]` | Panel invariants: cell definitions, N, state counts, zero counts, year levels; `year` present and factor-able. |
| `[2]` | **Crossed-RE Gaussian round-trip.** Gaussian `(1\|state) + (1\|year)` in `glmmTMB` vs `statsmodels.MixedLM` via `vc_formula`, M1/M3, both outcomes, both windows. See §10 for the risk here. |
| `[3]` | Per-record structural checks over the ladder JSON. |
| `[4]` | **Sample invariance across ZI tiers** within each (cell, model) — the precondition that makes §5.3's cross-tier AIC comparison valid. |
| `[5]` | Independent re-derivation of every winner from the JSON, not read from `__meta`. |
| `[6]` | Degeneracy and collapse guard applicability: every ZI fit with a counterpart had the log-likelihood criterion applied; every skipped ZI tier carries a reason. |
| `[7]` | Re-derivation of marginal structural-zero probabilities, including the GH node-count stability assertion. |
| `[8]` | Independent ZI cross-check: `statsmodels.ZeroInflatedNegativeBinomialP` with state **and year** fixed effects and the mirrored ZI predictors. Reported honestly — the existing arm's equivalent diverged for violations, and a non-converged cross-check carries no interpretive weight in either direction. |
| `[9]` | Memo consistency: every number in `docs/jafari_crossed_models.md` traceable to the JSON, plus a forbidden-pattern guard on claims this design knows to be false (§9). |
| `[10]` | Freeze guard (§9). |

`r_env_check.R` remains the gate for `glmmTMB` availability. Expect the benign TMB
ABI-mismatch warning on every `Rscript` call; the validator asserts stderr contains nothing
else. No script in this arm may use `warnings.filterwarnings('ignore')` — that idiom is what
hid three non-converged published fits.

---

## 9. Blast radius and freeze

**Nothing existing is modified or regenerated.** Specifically frozen:

- `scripts/count_models_zinb.R`, `report_count_models.py`, `validate_count_models.py`,
  `build_count_model_memo_evidence.py`, `docs/count_models_zinb.md`,
  `docs/results_count_models_zinb.md`, `data/generated/count_model_results.json`
- the entire NB2 arm (`nb2_stepwise_*`, `docs/nb2_stepwise_models.md`)
- every `paper_table_*` script and JSON, and every `docs/WPS_Table_Sheels_*.docx`
- both `count_model_panel_*.csv` (read-only inputs)

Section `[10]` enforces this over the working tree and this branch's own commits. Its scope
is stated honestly and with a known limitation: per the 2026-08-31 note in `CLAUDE.md`, a
`git merge-base main HEAD`..`HEAD` history arm is **vacuous on `main`**, where the merge-base
is `HEAD`. The guard therefore verifies "this branch did not touch the frozen artifacts," not
"they have never changed." This arm is developed on a feature branch, where that is the right
question.

**Forbidden-pattern guard** — claims this design knows to be false, asserted absent from the
generated memo:

- that this arm's coefficients are comparable to the existing ladder's (different RE
  structure)
- that σ²_u0 here is comparable in magnitude to the published tables' (log link vs
  `log(count+1)`)
- that a `plogis(b0)` figure is a marginal structural-zero probability
- that a defaulted winner is a selection result
- that a COVID coefficient from this arm is comparable to the existing arm's

---

## 10. Known risks and limitations

1. **The full Jafari ZI mirror will probably only be estimable in `viol_2021`** (§3.3). If it
   is estimable nowhere, that is itself the reportable result and the memo says so plainly
   rather than presenting a lower tier as if it were the mirror.
2. **§8 `[2]` is the one piece with real implementation risk.** `statsmodels.MixedLM` fits
   crossed random effects via `vc_formula` with a constant group, which is workable but slow
   and awkward. If it proves intractable the gate is **not** quietly weakened: the validator
   reports the check as unavailable with its reason, `[0]`'s bridge precondition still runs,
   and the limitation goes in the memo.
3. **σ²_year rests on 9–11 levels** (§4.3). Reported with that caveat; no substantive claim
   depends on it alone.
4. **The cubic and `(1|year)` overlap** (§4.3). A loss of cubic significance is a finding to
   report, not a defect to tune away.
5. **Convergence risk is higher than the existing arm's**, because a mirrored ZI block adds
   up to 10 logistic parameters. The tier ladder is the designed response; non-convergence at
   any tier is recorded, never masked.
6. **`insp_2019`'s zeros are partly `fillna(0)`-manufactured.** The existing derivation is
   reused rather than re-typed, and the flag is carried into this arm's memo.

---

## 11. Deliverables and run order

New files only:

```
scripts/jafari_crossed_models.R        # the ladder
scripts/report_jafari_crossed.py       # pure artifact reader
scripts/validate_jafari_crossed.py     # sections [0]-[10]
docs/jafari_crossed_models.md          # GENERATED
data/generated/jafari_crossed_results.json
data/generated/jafari_crossed_selection.csv
data/generated/jafari_crossed_coefficients.csv
```

Plus a `CLAUDE.md` section recording the arm, its directives, and the caveats in §10.

Run order (prerequisite panels already exist):

```bash
Rscript scripts/r_env_check.R
Rscript scripts/jafari_crossed_models.R \
        data/generated/jafari_crossed_results.json
python3 scripts/report_jafari_crossed.py
python3 scripts/validate_jafari_crossed.py
```

---

## 12. Success criteria

1. Four cells fit at `(1 | state) + (1 | year)` with no random slope anywhere.
2. Eight families raced per cell at M3; a winner named for **each outcome in each window**,
   independently, with its AIC margin and the `zi_tier_reached` of every competitor.
3. A zero-inflation strength tally covering the previously missing ZI-NB1 rungs.
4. A Jafari Table 5–format two-block table per cell, with the ZI tier named.
5. Every section of `validate_jafari_crossed.py` passing, with its per-section breakdown.
6. No frozen artifact modified.
