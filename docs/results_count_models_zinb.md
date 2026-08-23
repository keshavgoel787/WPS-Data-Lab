# Results — `count_models_zinb.R`

**Outcomes:** WPS violations and inspections per state-year, as **raw counts** (no `log(x+1)`).
**Windows:** 2011–2021 (ECHO *WPS* view) and 2011–2019 (ECHO *establishments* view).
**Time specification:** cubic polynomial (`time + time2 + time3`), `time = year − 2017`.
**Estimator:** `glmmTMB`, random intercept + random slope for time by state, ML.
**Families compared:** Poisson, NB1, NB2, ZIP, ZINB, ZINB + ZI random intercept.
**Purpose:** replicate the model class of Jafari et al. (*PLOS ONE* 2024) — a multilevel
zero-inflated negative binomial — as a count-model alternative to the `MixedLM`
`log(count+1)` tables. Interpretation and caveats: `docs/count_models_zinb.md`.

---

## Sample construction

Six analytic cells: two outcomes × two windows, with violations fit twice — once with
inspections as a `log` **offset** (violations *per inspection*) and once with
`log(inspections+1)` as a **covariate** (the published tables' form).

| Cell | Outcome | Window | Exposure | M1 | M2 / M3 |
|---|---|---|---:|---:|---:|
| `insp_2019` | inspections | 2011–2019 | — | 450 / 50 st | 423 / 47 st |
| `insp_2021` | inspections | 2011–2021 | — | 539 / 49 st | 506 / 46 st |
| `viol_off_2019` | violations | 2011–2019 | offset | 387 / 50 st | 375 / 47 st |
| `viol_off_2021` | violations | 2011–2021 | offset | 526 / 49 st | 494 / 46 st |
| `viol_cov_2019` | violations | 2011–2019 | covariate | 394 / 50 st | 378 / 47 st |
| `viol_cov_2021` | violations | 2011–2021 | covariate | 533 / 49 st | 501 / 46 st |

The M1 → M2 drop is `SPEND_APP` (BLS OCC 37-3012 is never published for AK/RI/VT). The
offset cells sit below their covariate twins because `log(inspections)` is undefined at
zero: 7 state-years drop in 2021 — **all 7 are also zero-violation rows** — and 15 in
2019, of which 1 is zero-violation, 6 positive, 8 missing.

**Covariate block (z-scored, identical to `paper_table_models_2021.py`):** `SPEND_APP_z`,
`SPEND_WORK_z`, `lii_2017_z`, `h2a_per_farmworker_z`, `dol_demand_met_pct_z`, `pct_flc_z`.

---

## Why a count model

| Window | Outcome | N | Zeros | Mean | Variance | Var/Mean |
|---|---|---:|---:|---:|---:|---:|
| 2011–2021 | violations | 533 | 93 (17.4%) | 29.70 | 4681.7 | 157.7 |
| 2011–2021 | inspections | 539 | 7 (1.3%) | 68.32 | 15491.7 | 226.7 |
| 2011–2019 | violations | 394 | 10 (2.5%) | 11.62 | 264.5 | 22.8 |
| 2011–2019 | inspections | 450 | 15 (3.3%) | 27.11 | 761.5 | 28.1 |

Variance runs 23–227× the mean everywhere, so a Poisson is not viable in any cell. The
2011–2021 violations series adds 17.4% zeros. The 2019 inspection zeros are partly
**manufactured**: the establishments reshape applies `fillna(0)` to both the EPA and state
inspection components before summing, so a state-year missing both becomes a zero.

---

## Family selection

Selected by lowest AIC among converged, non-degenerate fits **within** a random-effects
tier. `rs` = `(1 + time | state)`, the mandated structure; `ri` = `(1 | state)`, reported
separately where a family could not reach `rs`. **AIC is never compared across tiers.**

| Cell | Tier | Selected | AIC | Nearest clean competitor | ΔAIC |
|---|---|---|---:|---|---:|
| `insp_2019` | rs | ZINB + ZI RE | 3160.05 | ZINB | 4.03 |
| `insp_2021` | rs | ZINB | 4532.61 | NB2 | 33.36 |
| `viol_off_2019` | rs | NB2 | 2415.97 | NB1 | 73.48 |
| `viol_off_2021` | rs | NB1 | 3651.08 | NB2 | 7.88 |
| `viol_off_2021` | ri | ZINB + ZI RE | 3632.61 | ZINB | 20.14 |
| `viol_cov_2019` | rs | NB2 | 2328.64 | NB1 | 78.69 |
| `viol_cov_2021` | rs | NB1 | 3630.16 | NB2 | 10.74 |
| `viol_cov_2021` | ri | ZINB + ZI RE | 3624.02 | ZINB | 18.95 |

Inspections selects a zero-inflated family in **both** windows. Violations select a plain
negative binomial in both — but **by default, not on merit**: at Model 3 neither ZI-NB
rung is estimable at `(1 + time | state)` in any violations cell. Wherever a matched
comparison is possible (same cell, model, tier, N; both converged; neither degenerate) a
zero-inflated NB beats NB1 **17–0** (34.7–195.0 AIC) and NB2 **17–0** (24.1–83.2 AIC).

---

## Model 3 coefficients — inspections

`b` on the log link; IRR = exp(b). Significance: `***` p<.001, `**` p<.01, `*` p<.05, `+` p<.10.

| Parameter | 2019 (ZINB + ZI RE) | | 2021 (ZINB) | |
|---|---:|---|---:|---|
| | b (SE) | p | b (SE) | p |
| Intercept | 2.9282 (0.1219) | *** | 3.6518 (0.1539) | *** |
| time | −0.0826 (0.0193) | *** | −0.0556 (0.0181) | ** |
| time² | −0.0143 (0.0088) | 0.106 | −0.0045 (0.0034) | 0.184 |
| time³ | −0.0009 (0.0014) | 0.495 | −0.0001 (0.0008) | 0.862 |
| SPEND_APP_z | −0.3186 (0.1191) | ** | 0.0775 (0.1683) | 0.645 |
| SPEND_WORK_z | −0.1288 (0.1468) | 0.380 | 0.0806 (0.1765) | 0.648 |
| lii_2017_z | −0.2780 (0.1664) | + | 0.0789 (0.2271) | 0.728 |
| h2a_per_farmworker_z | 0.0135 (0.0493) | 0.784 | 0.0586 (0.0487) | 0.230 |
| dol_demand_met_pct_z | 0.0031 (0.1343) | 0.981 | 0.1352 (0.2092) | 0.518 |
| pct_flc_z | −0.0018 (0.1285) | 0.989 | −0.0052 (0.1800) | 0.977 |
| **ZI intercept** (logit) | −9.4231 (2.6032) | *** | −4.4057 (0.4276) | *** |

`SPEND_APP_z` is the only significant covariate in either inspections column, negative, and
only in 2011–2019. No labour or H-2A covariate reaches p<.10 in the 2021 column.

---

## Model 3 coefficients — violations (covariate specification)

Directly comparable in form to published Table 3, which also enters inspections as
`log(inspections+1)`.

| Parameter | 2019 (NB2) | | 2021 (NB1) | |
|---|---:|---|---:|---|
| | b (SE) | p | b (SE) | p |
| Intercept | 1.5609 (0.2334) | *** | −0.0314 (0.2818) | 0.911 |
| log Inspections | 0.2352 (0.0690) | *** | 0.7401 (0.0618) | *** |
| time | 0.2328 (0.0284) | *** | 0.0660 (0.0265) | * |
| time² | −0.0579 (0.0159) | *** | −0.0201 (0.0047) | *** |
| time³ | −0.0133 (0.0026) | *** | −0.0057 (0.0011) | *** |
| SPEND_APP_z | −0.1469 (0.1163) | 0.206 | 0.1959 (0.1389) | 0.158 |
| SPEND_WORK_z | −0.3748 (0.1460) | * | −0.4669 (0.1703) | ** |
| lii_2017_z | −0.0406 (0.1591) | 0.799 | 0.0358 (0.1910) | 0.852 |
| h2a_per_farmworker_z | −0.0327 (0.0808) | 0.686 | −0.0217 (0.0774) | 0.779 |
| dol_demand_met_pct_z | 0.1446 (0.1302) | 0.267 | −0.0943 (0.1793) | 0.599 |
| pct_flc_z | 0.3979 (0.1170) | *** | 0.2570 (0.1549) | + |

**`SPEND_WORK_z` is negative and significant in both windows** (IRR 0.687 and 0.627): more
STAG spending per farmworker, fewer violations. `pct_flc_z` is positive in both, significant
in 2019 only. The offset specification agrees on sign and significance for `SPEND_WORK_z`
(2019 −0.2663, p=0.108; 2021 −0.5406, p=0.003) and `pct_flc_z` (2019 0.3996, p=0.003;
2021 0.2508, p=0.147).

---

## Variance components (selected family, log link)

σ²_u0 here is on the **log link** scale, not the `log(count+1)` outcome scale of the
published tables — magnitudes are not comparable to Tables 2/3. This pipeline computes
**no** Δσ²_u0 percentage; that needs a random-intercept-only refit series the ladder does
not produce.

| Cell | Tier | σ²_u0 | σ²_u1 | Dispersion | σ²_zi_u0 | Pr(structural 0), marginal |
|---|---|---:|---:|---:|---:|---:|
| `insp_2019` | rs | 0.6035 | 0.0077 | 14.230 | 36.485 | 0.0675 |
| `insp_2021` | rs | 0.9708 | 0.0052 | 6.623 | — | 0.0121 |
| `viol_off_2019` | rs | 0.6636 | 0.0102 | 3.138 | — | — |
| `viol_off_2021` | rs | 0.8960 | 0.0068 | 10.357 | — | — |
| `viol_off_2021` | ri | 0.7697 | — | 2.567 | 2.433 | 0.0943 |
| `viol_cov_2019` | rs | 0.4419 | 0.0027 | 4.162 | — | — |
| `viol_cov_2021` | rs | 0.8499 | 0.0075 | 10.863 | — | — |
| `viol_cov_2021` | ri | 0.7538 | — | 2.657 | 1.741 | 0.0965 |

For a ZI random intercept the structural-zero probability must be read **marginally**, not
as `plogis(b0)`: `insp_2019`'s ZI RE SD is 6.04 on the logit scale, so the median-state
value is 0.0001 while the marginal is **0.0675**, with 11.6% of states above 0.10.

`dispersion` is θ for NB2 and the multiplier for NB1 — not a residual variance.

---

## Observed vs expected zeros

Expected counts are means over 2000 simulations from each fitted model; `± ` is the
Monte-Carlo SE.

| Cell | Tier | Observed | Expected | Ratio |
|---|---|---:|---:|---:|
| `insp_2019` | rs | 4 | 29.89 ± 0.29 | 7.47 |
| `insp_2021` | rs | 7 | 7.47 ± 0.06 | 1.07 |
| `viol_off_2019` | rs | 7 | 35.45 ± 0.15 | 5.06 |
| `viol_off_2021` | rs | 79 | 99.18 ± 0.31 | 1.26 |
| `viol_off_2021` | ri | 79 | 74.30 ± 0.28 | 0.94 |
| `viol_cov_2019` | rs | 7 | 21.12 ± 0.12 | 3.02 |
| `viol_cov_2021` | rs | 86 | 89.41 ± 0.31 | 1.04 |
| `viol_cov_2021` | ri | 86 | 73.14 ± 0.26 | 0.85 |

The 2021 cells reproduce their zero counts closely. **Both 2019 violations cells are
zero-deflated** — every family overpredicts, by 2.3–7.9× across the ladder — which is why
all three zero-inflated families are degenerate there at Model 3. `insp_2019` is the one
cell where a credible competitor (ZINB, ΔAIC 4.03) fits zeros better (29.89 vs 5.28), and
its 4 remaining zeros are exactly the `fillna(0)` cases.

---

## Cross-software check on the zero-inflation component

`statsmodels.ZeroInflatedNegativeBinomialP` with **state fixed effects** instead of random
effects, as an independent confirmation.

| Cell | statsmodels ZI intercept | `glmmTMB` ZI intercept | Converged |
|---|---:|---:|---|
| `insp_2021` | −4.4578 (0.4248) | −4.4057 (0.4276) | yes |
| `viol_cov_2021` | — | −2.2815 (0.2110) | **no** |

Different package, different treatment of the state dimension, essentially the same answer
for inspections. The violations check did not converge — the entire parameter vector
diverged on a full-rank design, the known incidental-parameters fragility of a ~53-parameter
fixed-effects count mixture — so it carries **no** interpretive weight either way.

---

## COVID robustness (2011–2021)

| Cell | Family | COVID b (SE) | p | IRR |
|---|---|---:|---:|---:|
| `insp_2021` | ZINB | −0.4160 (0.1328) | 0.0017 | 0.660 |
| `viol_cov_2021` | NB1 | −0.2786 (0.1676) | 0.0964 | 0.757 |

Inspections keeps the log-linear result's sign and significance (log-linear: −0.590***).
**Violations do not.** `CLAUDE.md` previously recorded that the violations time terms
"barely move"; under NB1 the indicator is −0.279 (p≈0.096) and `time2` goes from
p≈2.0e-5 to **p≈0.26** once it is added. That table is less COVID-robust than the
log-linear check implied.

---

## Notes

- **The two windows are different measures, not a longer one.** WPS-view counts run 2–3×
  the establishments-view counts and correlate ≈0 from 2017 on. The count models find this
  independently: log within-state variance on log within-state mean gives an exponent of
  2.01 for 2019 violations (NB2's quadratic form) against 1.46 for 2021 (NB1's linear form),
  which is why the selected family differs by window.
- **No ZI-NB1 rung was fit.** The six families are fixed by spec and every ZI rung is
  NB2-based, so plain NB1 won the two 2021 violations cells against a candidate set lacking
  its own zero-inflated counterpart. The ZINB-vs-NB2 comparison (17–0) is what isolates the
  inflation effect cleanly.
- **Spec §5.5 expanded-ZI sensitivity** (spending + labour intensity in the ZI part)
  converges in 3 of 6 cells (`insp_2019`, `insp_2021`, `viol_off_2019`) and returns a
  non-positive-definite Hessian in the other three. Never entered selection.
- **Boundary LRTs.** 6 of 30 likelihood-ratio tests are ZINB + ZI RE vs ZINB, where the
  null sits on the parameter-space boundary; the reference is a ½χ²₀ + ½χ²₁ mixture, so the
  nominal p is roughly 2× too large — conservative.
- **A separate finding, not a count-model result:** three fits behind the published tables
  never converged (2011–2021 violations M1; 2011–2019 inspections M1 and violations M2),
  hidden by a module-level `warnings.filterwarnings('ignore')`. One materially affects a
  printed coefficient. Both variance-explained blocks are unaffected. See
  `docs/count_models_zinb.md`; nothing in the manuscript pipeline was changed.

**Reproduce:** `Rscript scripts/r_env_check.R` → `python3 scripts/build_count_model_panel.py`
→ `Rscript scripts/count_models_zinb.R data/generated/count_model_panel_2021.csv data/generated/count_model_results.json`
→ `python3 scripts/build_count_model_memo_evidence.py` → `python3 scripts/report_count_models.py`
→ `python3 scripts/validate_count_models.py` (2277 checks must pass).
