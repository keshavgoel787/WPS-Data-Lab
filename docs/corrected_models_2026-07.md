# Corrected WPS Models — 2026-07 Post-Meeting Revision

This memo documents the model rebuild that followed Joe's meeting summary. It
covers (1) why the substantive models were dropping from 50 to 39 states, (2) the
four data/spec corrections applied, and (3) the corrected Table 2 and Table 3
results. It supersedes the earlier `paper_table_models.py` output.

Study period is **2011–2019** (the ECHO violation/inspection outcomes do not
extend to 2021).

---

## 1. Why the models were losing states (50 → 39)

The listwise drop had **two independent causes**, verified against the raw data:

**Cause A — BLS suppression (8 states: AK, CO, MN, NV, NH, NM, UT, VT).**
`SPEND_APP` and `SPEND_WORK` divide STAG dollars by BLS employment counts for
pesticide applicators (SOC 37-3012) and farmworkers (SOC 45-2092). We held only
the **2011** BLS snapshot, in which BLS suppresses small occupational cells:
37-3012 was blank for CO/NH/NM/UT/VT, 45-2092 for MN/NV, and Alaska was absent
entirely.

**Cause B — spending-window gap (3 states: MS, RI, WV).** In the spending master,
Rhode Island had no rows at all, Mississippi only 2008 rows, and West Virginia
only a 2022 row — none inside 2011–2019 — so all three collapsed to missing
spending and were dropped.

8 + 3 = the 11 states lost.

---

## 2. Corrections applied

### 2.1 Full BLS OEWS harvest, 2011–2019
`harvest_bls_oews_panel.py` pulls the BLS OEWS state files for every year
(`bls.gov/oes/special-requests/oesm{YY}st.zip`) and rebuilds the panel. A cell
suppressed in one year is usually reported in another, so coverage jumps:

| Occupation (denominator) | 2011 only | All 9 years |
|---|---|---|
| Farmworkers 45-2092 (SPEND_WORK, H-2A ratio) | missing MN, NV, AK | **50 / 50** |
| Pesticide applicators 37-3012 (SPEND_APP) | missing CO, NH, NM, UT, VT, AK | 47 / 50 |
| FLC supervisors 45-1011 | — | 48 / 50 |

AK, RI, VT never receive a published 37-3012 figure in any year — a genuine BLS
disclosure limit, not a harvest gap — so `SPEND_APP` columns are estimated on 47
states (PI-approved).

### 2.2 Spending re-harvest — what it revealed
`harvest_stag_spending.py` pulls EPA assistance obligations from USASpending and
validates them against the master. Two findings shaped the decision:

- The master is **hand-curated to pesticide-relevant awards**, dominated by CFDA
  **66.700 — Consolidated Pesticide Enforcement Cooperative Agreements** (the
  FIFRA STAG Compliance Monitoring program). Naïvely summing all seven CFDA codes
  present in the master inflates spending **~43×**, because it sweeps in giant
  general programs the original harvest excluded (Performance Partnership Grants,
  CFDA 66.605, is $3.95 **billion**). 66.700 alone matches the master within 5%.
- Under 66.700, **RI, MS, and WV genuinely received $0** in 2011–2019 — identical
  under both recipient-location and place-of-performance scopes — as did HI, TN,
  and UT, which the master already codes $0. These six states most likely bundle
  pesticide-enforcement funding into their Performance Partnership Grant rather
  than taking the standalone cooperative agreement.

**Decision:** keep the 66.700-scope spending definition and **code MS/RI/WV as
$0** (joining HI/TN/UT), so they remain in the models instead of being dropped.
`rebuild_spend_bls_multiyear.py` produces the corrected `spend_bls_variables_multiyear.csv`,
using **mean 2011–2019 employment** as the denominator (matching the mean-spending
numerator's window). The rebuilt `SPEND_WORK` correlates r = 0.91 with the old
2011-based version, so the variable's cross-state structure is preserved.

### 2.3 Year-matched H-2A ratio
`build_h2a_ratio_panel.py` rebuilds the H-2A intensity measure as a **time-varying
Level-1 predictor**: annual certified H-2A workers ÷ annual BLS farmworkers,
matched by year (previously a single state mean over a frozen 2011 denominator).
The result is a complete 50×9 panel (2013 DOL gap and sporadic BLS suppression
filled by within-state linear interpolation). The mean ratio rises from 2.68
(2011) to 3.47 (2019) as H-2A certifications outpace the farmworker base.

### 2.4 Log-transformed outcomes
Both DVs are modelled as **log(count + 1)** — the transform-exploration
recommendation Joe confirmed (`+1` because violations contain zeros). Inspections
enters the violations model as a predictor on the same `log(x+1)` scale. Δ σ²
baselines are re-estimated on the log scale, on each column's own analytic sample.

**Net effect: the substantive models go from 39 to 47 states.**

---

## 3. Corrected results

Coefficients are fixed effects with standard errors in parentheses.
`+ p<.10, * p<.05, ** p<.01, *** p<.001`. Random intercept + random slope for time
by state (REML). Δ σ² = reduction in between-state intercept variance vs. a
time-only baseline re-estimated on the same sample.

### Table 2 — WPS Inspections · DV = log(inspections + 1)

| Term | M1 | M2 | M3 |
|---|---|---|---|
| Time | −0.069\* (0.034) | −0.076\*\*\* (0.021) | −0.078\*\*\* (0.022) |
| Time² | −0.018+ (0.010) | −0.018 (0.011) | −0.018 (0.011) |
| Time³ | −0.002 (0.002) | −0.001 (0.002) | −0.001 (0.002) |
| Spending / applicator | | −0.276\* (0.114) | −0.259\* (0.128) |
| Spending / farmworker | | −0.113 (0.157) | −0.119 (0.161) |
| Labor Intensity (2017) | | −0.204 (0.174) | −0.192 (0.181) |
| H-2A : farmworker (yr-matched) | | | 0.024 (0.059) |
| H-2A demand-met % | | | 0.028 (0.146) |
| % H-2A to FLC | | | 0.058 (0.139) |
| State-to-State σ² | | 0.612 | 0.630 |
| Δ State-to-State σ² | | 7.3% | 4.6% |
| N | | 423 obs / 47 states | 423 obs / 47 states |

### Table 3 — WPS Violations · DV = log(violations + 1)

| Term | M1 | M2 | M3 |
|---|---|---|---|
| log(Inspections + 1) | 0.222\*\*\* (0.058) | 0.172\*\* (0.061) | 0.189\*\* (0.060) |
| Time | 0.210\*\*\* (0.024) | 0.206\*\*\* (0.024) | 0.209\*\*\* (0.025) |
| Time² | −0.021 (0.014) | −0.025+ (0.014) | −0.025+ (0.014) |
| Time³ | −0.007\*\*\* (0.002) | −0.008\*\*\* (0.002) | −0.008\*\*\* (0.002) |
| Spending / applicator | | −0.159 (0.108) | −0.072 (0.101) |
| Spending / farmworker | | −0.315\* (0.145) | −0.319\* (0.126) |
| Labor Intensity (2017) | | 0.039 (0.163) | 0.017 (0.141) |
| H-2A : farmworker (yr-matched) | | | −0.032 (0.068) |
| H-2A demand-met % | | | 0.134 (0.115) |
| % H-2A to FLC | | | 0.358\*\*\* (0.106) |
| State-to-State σ² | | 0.715 | 0.402 |
| Δ State-to-State σ² | | −12.5% | 36.8% |
| N | | 378 obs / 47 states | 378 obs / 47 states |

(Violations N = 378 rather than 423 because a handful of state-year cells have no
reported violation count in ECHO; inspections are zero-filled and so keep all 423.)

---

## 4. What the results say

- **Inspections trend downward** over the window (Time < 0, significant) — consistent
  with the declining purchasing power of enforcement funding.
- **Violations trend upward** with a significant cubic shape (the 2016–2017 spike),
  and **time is the dominant violations predictor**.
- **Inspections positively predict violations** (log-log ≈ elasticity: a 1% rise in
  inspections is associated with ~0.17–0.22% more violations found) — more looking,
  more found.
- **Spending / applicator is the significant spending predictor for inspections;
  Spending / farmworker is the significant one for violations.** Both are negative:
  higher per-capita STAG spending is associated with fewer inspections/violations.
- **% H-2A to FLC** is a strong positive predictor of violations in M3.
- The **labor intensity index and the year-matched H-2A ratio do not significantly
  predict** either outcome.

**Caveat — violations M2 Δσ² is negative (−12.5%).** Adding the spending/labor block
raises the between-state *intercept* variance because variance shifts into the random
*slope*; this is the random-effects artifact documented in
`docs/variance_decomposition.md`, not a sign the covariates hurt fit. M3 is +36.8%.

---

## 5. Open questions for the team

1. **Study window** — confirm 2011–2019 (the meeting summary said 2021; the outcome
   data ends in 2019).
2. **The six $0-funding states** — the assumption that every ECHO-reporting state
   receives dedicated STAG pesticide funding does not hold; six states appear to
   bundle it into Performance Partnership Grants. Confirm the $0 coding, or decide
   whether to attribute a pesticide share of PPG (not cleanly separable — not
   recommended).
3. **LII methods table** (Yueran) and the **Jabari (2024)** comparison remain open.

---

## 6. Reproduction

Run from `/Users/keshavgoel/Research/`, in order:

```bash
python3 scripts/harvest_bls_oews_panel.py        # → data/raw/bls_oews_panel_2011_2019.csv
python3 scripts/harvest_stag_spending.py          # USASpending diagnostic + validation
python3 scripts/rebuild_spend_bls_multiyear.py    # → data/generated/spend_bls_variables_multiyear.csv
python3 scripts/build_h2a_ratio_panel.py          # → data/generated/h2a_ratio_panel.csv
python3 scripts/paper_table_models_corrected.py   # → data/generated/paper_table_params_corrected.json
python3 scripts/fill_wps_tables_docx_corrected.py # → docs/WPS_Table_Sheels_filled_corrected.docx
```
