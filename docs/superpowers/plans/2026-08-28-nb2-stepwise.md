# NB2-Only Stepwise Count Models — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refit the 2011–2021 inspections and violations count models under `nbinom2` only, as a clean M1/M2/M3 stepwise, and report a variance-components table with a meaningful Δσ²_u0 percentage.

**Architecture:** A new standalone arm, parallel to the existing six-family ZINB ladder, which stays frozen. One R script does every `glmmTMB` fit and writes a flat JSON; one Python script does an independent `statsmodels` cross-check; one Python script reads both artifacts and writes the CSVs and the memo; one Python script is the validation gate. No script computes a statistic that another script also computes.

**Tech Stack:** R 4.5.2 + `glmmTMB` 1.1.14 + `jsonlite` (fits); Python 3 + `statsmodels` 0.14.6 + `pandas` + `numpy` (cross-check, reporting, validation).

**Spec:** `docs/superpowers/specs/2026-08-28-nb2-stepwise-design.md` — read it before Task 1. The plan argues from the spec; where they appear to disagree, the spec wins and the plan is wrong.

## Global Constraints

Copied verbatim from the spec and from CLAUDE.md. Every task's requirements implicitly include this section.

- **Working directory is `/Users/keshavgoel/Research/`.** All paths in this project are absolute and hardcoded. Do not add path-discovery logic.
- **No script may use `warnings.filterwarnings('ignore')`.** That idiom is what hid a non-converged published fit. Capture with `catch_warnings(record=True)` and report. The R script sets `options(warn = 1)`.
- **Family is `nbinom2` only.** No ladder, no zero-inflation, no family selection anywhere in this arm.
- **RE structure for every substantive fit is `(1 + time | state)`.** No fallback tier. A non-converging fit is reported as non-converging and its estimates are withheld.
- **Estimation is ML (`REML = FALSE`)**, matching `count_models_zinb.R:62`, because the M1→M2→M3 build-up is a nested fixed-effect comparison.
- **`time = year − 2017`.** Cubic time (`time`, `time2`, `time3`) in every model. No `× time` interactions.
- **Covariate blocks, reused not redefined:**
  - `M2_ADD = c("SPEND_APP_z", "SPEND_WORK_z", "lii_2017_z")`
  - `M3_ADD = c(M2_ADD, "h2a_per_farmworker_z", "dol_demand_met_pct_z", "pct_flc_z")`
- **`sigma2_e` is null for every count fit.** A count family has no residual variance. `dispersion` carries θ, and θ² is not a variance component.
- **All variances are on the log link scale** and are not comparable in magnitude to published Tables 2/3.
- **Frozen artifacts — must not be modified by any task:** `scripts/count_models_zinb.R`, `scripts/report_count_models.py`, `scripts/validate_count_models.py`, `scripts/build_count_model_panel.py`, `data/generated/count_model_results.json`, `data/generated/count_model_panel_*.csv`, `docs/count_models_zinb.md`, every `paper_table_*` script and JSON, every `docs/*.docx`.
- **The memo `docs/nb2_stepwise_models.md` is generated, never hand-edited.**
- **NB2 is a deliberate override of the AIC selection, not a fit-based win.** The memo must say so. Never write that NB2 was selected, won, or was preferred on fit grounds.

## Testing Convention For This Repo

**This repo has no pytest suite and is not getting one.** CLAUDE.md is explicit: "There is no build system, test suite, or package to install beyond standard scientific Python." The established equivalent is a validator script of `section()` / `check()` / `skip()` assertions — `scripts/validate_count_models.py` — run as a gate after any pipeline change.

The red/green cycle in this plan is therefore:

1. Add the check to `scripts/validate_nb2_stepwise.py`.
2. Run `python3 scripts/validate_nb2_stepwise.py` and **watch it FAIL** (missing artifact, or wrong value).
3. Implement.
4. Re-run and watch it PASS.
5. Commit.

Do not skip step 2. A check that has never been observed failing has not been shown to test anything.

## File Structure

| Path | Responsibility | Created in |
|---|---|---|
| `scripts/nb2_stepwise_models.R` | every `glmmTMB` fit (16 total); writes the flat results JSON. Fits only — computes no derived statistic. | Task 1 |
| `scripts/nb2_crosscheck_statsmodels.py` | independent `statsmodels` NB2 + state-fixed-effects fit; writes its own JSON. | Task 4 |
| `scripts/report_nb2_stepwise.py` | pure artifact reader. Δσ²_u0 %, ICC, CSVs, memo. The only place derived statistics are computed. | Task 5 |
| `scripts/validate_nb2_stepwise.py` | the gate. Grows one section per task. | Task 1, extended by 2–5 |
| `data/generated/nb2_stepwise_results.json` | all 16 fits | Task 1 |
| `data/generated/nb2_stepwise_crosscheck.json` | cross-check estimates | Task 4 |
| `data/generated/nb2_stepwise_variance.csv` | the variance components table | Task 5 |
| `data/generated/nb2_stepwise_coefficients.csv` | coefficient table, `b (SE)` and IRR | Task 5 |
| `docs/nb2_stepwise_models.md` | the memo | Task 5 |

### The 16 fits, by JSON key

Key format: `{cell}__{model}__nbinom2__{tier}`.

| Key | Purpose | Task |
|---|---|---|
| `insp_2021__M1__nbinom2__rs` | substantive | 1 |
| `insp_2021__M2__nbinom2__rs` | substantive (**new** — no counterpart exists) | 1 |
| `insp_2021__M3__nbinom2__rs` | substantive | 1 |
| `viol_cov_2021__M1__nbinom2__rs` | substantive | 1 |
| `viol_cov_2021__M2__nbinom2__rs` | substantive (**new**) | 1 |
| `viol_cov_2021__M3__nbinom2__rs` | substantive | 1 |
| `insp_2021__M1__nbinom2__ri` | Δσ²_u0 series | 2 |
| `insp_2021__M2__nbinom2__ri` | Δσ²_u0 series | 2 |
| `insp_2021__M3__nbinom2__ri` | Δσ²_u0 series | 2 |
| `viol_cov_2021__M1__nbinom2__ri` | Δσ²_u0 series | 2 |
| `viol_cov_2021__M2__nbinom2__ri` | Δσ²_u0 series | 2 |
| `viol_cov_2021__M3__nbinom2__ri` | Δσ²_u0 series | 2 |
| `insp_2021__M1matched__nbinom2__ri` | M1 baseline refit on the M2/M3 sample | 2 |
| `viol_cov_2021__M1matched__nbinom2__ri` | M1 baseline refit on the M2/M3 sample | 2 |
| `insp_2021__M3covid__nbinom2__rs` | COVID robustness | 3 |
| `viol_cov_2021__M3covid__nbinom2__rs` | COVID robustness | 3 |

---

## Task 1: R fitting engine and the six substantive fits

**Files:**
- Create: `scripts/nb2_stepwise_models.R`
- Create: `scripts/validate_nb2_stepwise.py`

**Interfaces:**
- Consumes: `data/generated/count_model_panel_2021.csv` (existing, unmodified — columns `state, year, time, time2, time3, violations, inspections, log_violations, log_inspections, covid, SPEND_APP_z, SPEND_WORK_z, lii_2017_z, dol_demand_met_pct_z, pct_flc_z, h2a_per_farmworker_z`).
- Produces: `data/generated/nb2_stepwise_results.json` — a JSON object keyed as in the table above. Every value is a record with these fields, which Tasks 2–5 rely on by name:
  - `cell` (str), `model` (str), `outcome` (str), `re_tier` (`"rs"` | `"ri"`), `re_used` (str), `formula` (str)
  - `n_obs` (int), `n_states` (int), `obs_zeros` (int)
  - `converged` (bool), `message` (str), `pd_hess` (bool), `conv_code` (int)
  - `aic`, `bic`, `loglik` (float), `df` (int)
  - `cond` — object mapping term name → `{b, se, z, p}`
  - `sigma2_u0`, `sigma2_u1`, `sigma_u01` (float or null), `dispersion` (float, = θ), `sigma2_e` (always null), `family_name` (always `"nbinom2"`)
  - `mu_fixed` (float) — `exp(mean(fixed-effects linear predictor))`, the μ used for ICC in Task 5
- Produces: `scripts/validate_nb2_stepwise.py` with module-level helpers `section(tag, title)`, `check(label, condition, detail='')`, `skip(label, reason)`, `check_close(label, got, want, tol, kind='abs')` and a `main()` — Tasks 2–5 add sections to this file.

- [ ] **Step 1: Write the failing validator**

Create `scripts/validate_nb2_stepwise.py`. The harness helpers are deliberately copied from `validate_count_models.py:34-80` rather than imported, because that file is frozen and importing it would run its module-level code.

```python
"""
Validation gate for the NB2-only stepwise count models.

Not a unit-test suite -- an analysis artifact, matching the convention of
scripts/validate_count_models.py. Each check proves either that a step
reproduces something already known or that an invariant the spec depends on
holds. Run after any change to this arm.

Run: python3 scripts/validate_nb2_stepwise.py   (exits 1 on any failure)
     python3 scripts/validate_nb2_stepwise.py --skip-precondition
         Skips section [0] only. FOR ITERATION DURING DEVELOPMENT ONLY -- a run
         with [0] skipped is not a passing run, and the SKIP line says so.

ON READING THE OUTPUT: do not quote the total on its own. main() prints a
per-section breakdown and an explicit SKIP line for every bypassed block, so a
run in which a block silently vanished is visibly different from one in which
it ran.

Spec: docs/superpowers/specs/2026-08-28-nb2-stepwise-design.md
"""
import json
import subprocess
import sys

GEN = '/Users/keshavgoel/Research/data/generated/'
SCRIPTS = '/Users/keshavgoel/Research/scripts/'
ROOT = '/Users/keshavgoel/Research/'

FAILURES = []
SECTIONS = []
_CUR = None

CELLS = ('insp_2021', 'viol_cov_2021')
MODELS = ('M1', 'M2', 'M3')


def section(tag, title):
    global _CUR
    _CUR = {'name': f'[{tag}] {title}', 'pass': 0, 'fail': 0, 'skip': 0}
    SECTIONS.append(_CUR)
    print(f"\n[{tag}] {title}")


def check(label, condition, detail=''):
    if condition:
        print(f"  PASS  {label}")
        if _CUR is not None:
            _CUR['pass'] += 1
    else:
        print(f"  FAIL  {label}" + (f" -- {detail}" if detail else ''))
        FAILURES.append(label)
        if _CUR is not None:
            _CUR['fail'] += 1


def skip(label, reason):
    print(f"  SKIP  {label} -- {reason}")
    if _CUR is not None:
        _CUR['skip'] += 1


def check_close(label, got, want, tol, kind='abs'):
    if got is None or want is None:
        check(f"{label} (both values present)", False, f"got {got!r}, want {want!r}")
        return
    delta = abs(got - want) if kind == 'abs' else abs(got - want) / abs(want)
    check(f"{label} ({kind} delta {delta:.2e} <= {tol:.0e})", delta <= tol,
          f"got {got!r}, want {want!r}")


def load_results():
    with open(GEN + 'nb2_stepwise_results.json') as fh:
        return json.load(fh)


# ============================================================
# [0] PRECONDITION -- the Gaussian round-trip gate
# ============================================================
def validate_precondition(skip_precondition):
    section('0', 'Precondition: the existing Gaussian round-trip gate passes')
    if skip_precondition:
        skip('validate_count_models.py [2a]/[2b]/[2c]',
             '--skip-precondition passed; this run is NOT a passing run')
        return
    proc = subprocess.run([sys.executable, SCRIPTS + 'validate_count_models.py'],
                          capture_output=True, text=True, cwd=ROOT)
    check('validate_count_models.py exits 0', proc.returncode == 0,
          f'exit {proc.returncode}; last stderr line: '
          f'{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "(none)"}')
    # The Gaussian sections specifically: family-agnostic, so their verdict
    # carries over to this arm unchanged (spec 5.1). Parsed out of the
    # per-section breakdown rather than trusting the exit code alone.
    for tag in ('[2a]', '[2b]', '[2c]'):
        rows = [ln for ln in proc.stdout.splitlines() if ln.startswith(tag)]
        # The breakdown table repeats the section name with PASS/FAIL/SKIP columns.
        tail = [ln for ln in rows if ln.rstrip()[-1:].isdigit()]
        ok = bool(tail) and int(tail[-1].split()[-2]) == 0 and int(tail[-1].split()[-3]) > 0
        check(f'Gaussian gate section {tag}: >0 PASS and 0 FAIL', ok,
              f'breakdown row: {tail[-1] if tail else "(not found)"}')


# ============================================================
# [1] PANEL IDENTITY
# ============================================================
def validate_panel():
    section('1', 'Panel identity (count_model_panel_2021.csv, unmodified)')
    import pandas as pd
    p = pd.read_csv(GEN + 'count_model_panel_2021.csv')
    check('N rows == 539', len(p) == 539, f'got {len(p)}')
    check('N states == 49', p['state'].nunique() == 49, f"got {p['state'].nunique()}")
    check('Wyoming absent (WPS view has no WY row)', 'Wyoming' not in set(p['state']))
    check('violations zeros == 93', int((p['violations'] == 0).sum()) == 93,
          f"got {int((p['violations'] == 0).sum())}")
    check('inspections zeros == 7', int((p['inspections'] == 0).sum()) == 7,
          f"got {int((p['inspections'] == 0).sum())}")
    check('time == year - 2017 throughout', bool((p['time'] == p['year'] - 2017).all()))
    for col in ('SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
                'h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z'):
        check(f'column present: {col}', col in p.columns)


# ============================================================
# [2] FIT INTEGRITY -- the six substantive fits
# ============================================================
def validate_fits():
    section('2', 'Fit integrity: six substantive nbinom2 fits at (1 + time | state)')
    res = load_results()
    for cell in CELLS:
        for model in MODELS:
            key = f'{cell}__{model}__nbinom2__rs'
            if key not in res:
                check(f'{key} present', False, 'missing from results JSON')
                continue
            r = res[key]
            check(f'{key} converged', bool(r['converged']), r.get('message', ''))
            check(f'{key} positive-definite Hessian', bool(r['pd_hess']))
            check(f'{key} conv_code == 0', r['conv_code'] == 0, f"got {r['conv_code']}")
            check(f'{key} re_tier == rs (no fallback)', r['re_tier'] == 'rs',
                  f"got {r['re_tier']}")
            check(f'{key} re_used == (1 + time | state)',
                  r['re_used'] == '(1 + time | state)', f"got {r['re_used']}")
            check(f'{key} family_name == nbinom2', r['family_name'] == 'nbinom2')
            check(f'{key} sigma2_e is null (count family has no residual variance)',
                  r['sigma2_e'] is None, f"got {r['sigma2_e']!r}")
            check(f'{key} dispersion (theta) finite and > 0',
                  r['dispersion'] is not None and r['dispersion'] > 0)
            check(f'{key} sigma2_u1 present (random slope was actually fit)',
                  r['sigma2_u1'] is not None and r['sigma2_u1'] > 0)
            check(f'{key} all SEs finite',
                  all(c['se'] is not None and c['se'] == c['se'] and c['se'] < float('inf')
                      for c in r['cond'].values()))
            check(f'{key} mu_fixed finite and > 0',
                  r['mu_fixed'] is not None and r['mu_fixed'] > 0)
    # Sample sizes: M1 keeps all 49 states; M2/M3 lose AK/RI/VT to the BLS
    # applicator series, which is why the spec reports two Delta baselines.
    for cell in CELLS:
        m1, m2, m3 = (res.get(f'{cell}__{m}__nbinom2__rs', {}) for m in MODELS)
        check(f'{cell}: M1 n_states == 49', m1.get('n_states') == 49,
              f"got {m1.get('n_states')}")
        check(f'{cell}: M2 n_states == 46', m2.get('n_states') == 46,
              f"got {m2.get('n_states')}")
        check(f'{cell}: M3 n_states == 46', m3.get('n_states') == 46,
              f"got {m3.get('n_states')}")
        check(f'{cell}: M2 and M3 share one analytic sample',
              m2.get('n_obs') == m3.get('n_obs'),
              f"M2 {m2.get('n_obs')} vs M3 {m3.get('n_obs')}")
    # Fixed-effect terms, exactly as the spec's build-up specifies.
    m2_add = ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z']
    m3_add = m2_add + ['h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']
    for cell in CELLS:
        base = ['time', 'time2', 'time3']
        if cell == 'viol_cov_2021':
            base = ['log_inspections'] + base
        for model, extra in (('M1', []), ('M2', m2_add), ('M3', m3_add)):
            r = res.get(f'{cell}__{model}__nbinom2__rs')
            if r is None:
                continue
            want = {'(Intercept)'} | set(base) | set(extra)
            check(f'{cell} {model}: fixed-effect terms exactly as specified',
                  set(r['cond']) == want,
                  f"extra {set(r['cond']) - want}, missing {want - set(r['cond'])}")
            check(f'{cell} {model}: no interaction terms',
                  not any(':' in t for t in r['cond']))


# ============================================================
# [3] OVERLAP REGRESSION against the frozen ZINB ladder
# ============================================================
def validate_overlap():
    section('3', 'Overlap regression: four fits already exist in the frozen ladder')
    res = load_results()
    with open(GEN + 'count_model_results.json') as fh:
        old = json.load(fh)
    # Both arms estimate by ML (count_models_zinb.R:62 defaults REML = FALSE)
    # on identical rows and formulas, so these should agree to optimizer noise,
    # not merely to a loose tolerance.
    for cell in CELLS:
        for model in MODELS:
            new_key = f'{cell}__{model}__nbinom2__rs'
            old_key = f'{cell}__{model}__nbinom2'
            if old_key not in old:
                skip(f'{new_key} vs frozen ladder',
                     'no counterpart: the ladder fits M2 only under the selected '
                     'family, and nbinom2 is neither cell\'s selected family')
                continue
            a, b = res.get(new_key), old[old_key]
            if a is None:
                check(f'{new_key} present', False, 'missing from results JSON')
                continue
            check(f'{new_key}: n_obs matches ladder', a['n_obs'] == b['n_obs'],
                  f"{a['n_obs']} vs {b['n_obs']}")
            check(f'{new_key}: n_states matches ladder',
                  a['n_states'] == b['n_states'], f"{a['n_states']} vs {b['n_states']}")
            check_close(f'{new_key}: AIC', a['aic'], b['aic'], 1e-6, 'rel')
            check_close(f'{new_key}: sigma2_u0', a['sigma2_u0'], b['sigma2_u0'],
                        1e-4, 'rel')
            check_close(f'{new_key}: dispersion (theta)', a['dispersion'],
                        b['dispersion'], 1e-4, 'rel')
            for term, co in b['cond'].items():
                check_close(f'{new_key}: b[{term}]', a['cond'][term]['b'],
                            co['b'], 1e-4, 'abs')


def main():
    skip_pre = '--skip-precondition' in sys.argv
    validate_precondition(skip_pre)
    validate_panel()
    validate_fits()
    validate_overlap()

    print()
    print("=" * 78)
    print("CHECK BREAKDOWN BY SECTION (do not quote the total on its own: the "
          "per-fit")
    print("sections are record loops, not sets of independent findings)")
    print("=" * 78)
    print(f"{'section':62}{'PASS':>6}{'FAIL':>6}{'SKIP':>6}")
    for sec in SECTIONS:
        print(f"{sec['name'][:62]:62}{sec['pass']:>6}{sec['fail']:>6}{sec['skip']:>6}")
    tp = sum(x['pass'] for x in SECTIONS)
    tf = sum(x['fail'] for x in SECTIONS)
    tsk = sum(x['skip'] for x in SECTIONS)
    print(f"{'TOTAL (see the caveat above)':62}{tp:>6}{tf:>6}{tsk:>6}")
    if tsk:
        print(f"\n{tsk} gated block(s) were SKIPPED -- each is printed above "
              f"with its reason.")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python3 scripts/validate_nb2_stepwise.py --skip-precondition`

Expected: `[1]` passes (the panel already exists), then `[2]` and `[3]` crash with `FileNotFoundError: .../nb2_stepwise_results.json`. That crash *is* the red state — the artifact does not exist yet. Do not add a try/except to soften it; the next step creates the file.

- [ ] **Step 3: Write the R fitting script**

Create `scripts/nb2_stepwise_models.R`.

```r
# NB2-only stepwise count models for the 2011-2021 WPS panel.
# See docs/superpowers/specs/2026-08-28-nb2-stepwise-design.md.
#
# Usage:
#   Rscript scripts/nb2_stepwise_models.R <panel_csv> <out_json>
#
# THIS SCRIPT FITS. It computes no derived statistic -- no Delta sigma^2_u0, no
# ICC, no percentage. Those live in report_nb2_stepwise.py so that exactly one
# script owns each number.
#
# The family is nbinom2 and only nbinom2. That is an EDITORIAL choice, not a
# fit-based one: at the mandated (1 + time | state) tier nbinom2 loses to ZINB
# by ~33 AIC for inspections and to nbinom1 by ~11 for violations. Holding the
# family fixed is what makes the three columns comparable and makes a variance
# reduction sequence meaningful. See spec section 3.

suppressPackageStartupMessages({
  library(glmmTMB)
  library(jsonlite)
})

# Warnings print immediately, with the call that raised them, rather than being
# deferred into an anonymous "There were N warnings" line. Every warning a fit
# raises is captured into that fit's `message` field and muffled, so anything
# reaching stderr is a leak from outside those handlers.
options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: nb2_stepwise_models.R <panel_csv> <out_json>")
panel_csv <- args[1]
out_json <- args[2]

`%||%` <- function(a, b) if (is.null(a)) b else a

CUBIC  <- c("time", "time2", "time3")
M2_ADD <- c("SPEND_APP_z", "SPEND_WORK_z", "lii_2017_z")
M3_ADD <- c(M2_ADD, "h2a_per_farmworker_z", "dol_demand_met_pct_z", "pct_flc_z")
RS_RE  <- "(1 + time | state)"
RI_RE  <- "(1 | state)"

panel <- read.csv(panel_csv, stringsAsFactors = FALSE)

CELLS <- list(
  insp_2021     = list(dv = "inspections", base = CUBIC, outcome = "inspections"),
  viol_cov_2021 = list(dv = "violations",  base = c("log_inspections", CUBIC),
                       outcome = "violations"))

coef_list <- function(mat) {
  if (is.null(mat) || nrow(mat) == 0) return(structure(list(), names = character(0)))
  stats::setNames(lapply(seq_len(nrow(mat)), function(i) {
    list(b = mat[i, 1], se = mat[i, 2], z = mat[i, 3], p = mat[i, 4])
  }), rownames(mat))
}

# fit_nb2: one nbinom2 specification, flattened to a plain list.
#
# `sample_rhs` exists for ONE purpose: the matched Delta baseline in Task 2,
# where the Model-1 formula must be fit on the Model-3 analytic sample. Rows are
# selected by complete.cases over `sample_rhs` (default: `rhs`), so the fitted
# formula and the sample-defining variable set can differ deliberately.
fit_nb2 <- function(d, dv, rhs, re, sample_rhs = NULL) {
  sel <- if (is.null(sample_rhs)) rhs else sample_rhs
  need <- unique(c(dv, sel, "state", "time"))
  need <- need[need %in% names(d)]
  dd <- d[stats::complete.cases(d[, need, drop = FALSE]), , drop = FALSE]
  dd$state <- factor(dd$state)

  form <- stats::as.formula(sprintf("%s ~ %s + %s", dv,
                                    paste(rhs, collapse = " + "), re))

  base <- list(formula = deparse1(form), re_used = re,
               n_obs = nrow(dd), n_states = length(unique(dd$state)),
               obs_zeros = sum(dd[[dv]] == 0, na.rm = TRUE),
               family_name = "nbinom2", sigma2_e = NA_real_)

  warn_msgs <- character(0)
  fit <- tryCatch(
    withCallingHandlers(
      glmmTMB(form, ziformula = ~0, family = nbinom2, data = dd, REML = FALSE),
      warning = function(w) {
        warn_msgs <<- c(warn_msgs, conditionMessage(w))
        invokeRestart("muffleWarning")
      }),
    error = function(e) e)

  if (inherits(fit, "error")) {
    return(c(base, list(converged = FALSE, message = conditionMessage(fit))))
  }

  pd_hess <- isTRUE(fit$sdr$pdHess)
  conv_code <- fit$fit$convergence
  sm <- summary(fit)
  cond <- coef_list(sm$coefficients$cond)
  finite_se <- length(cond) == 0 ||
    all(vapply(cond, function(x) is.finite(x$se), logical(1)))

  vc <- tryCatch(VarCorr(fit)$cond$state, error = function(e) NULL)
  s2_u0 <- if (is.null(vc)) NA_real_ else vc[1, 1]
  s2_u1 <- if (is.null(vc) || nrow(vc) < 2) NA_real_ else vc[2, 2]
  s_u01 <- if (is.null(vc) || nrow(vc) < 2) NA_real_ else vc[1, 2]

  # theta, the NB2 size parameter (Var = mu + mu^2/theta). NOT a residual SD --
  # its square is not a variance component, which is why `sigma2_e` stays NA for
  # every fit in this script.
  disp <- tryCatch(sigma(fit), error = function(e) NA_real_)

  # mu for the ICC's distribution-specific variance, computed here because only
  # the fit object can produce it. re.form = NA gives the POPULATION-level linear
  # predictor (fixed effects only); exponentiating its mean is the mu that
  # Nakagawa's observation-level variance for a log-link count model expects.
  mu_fixed <- tryCatch(
    exp(mean(stats::predict(fit, re.form = NA, type = "link"))),
    error = function(e) NA_real_)

  c(base, list(converged = pd_hess && conv_code == 0 && finite_se,
               message = if (length(warn_msgs)) paste(warn_msgs, collapse = " | ") else "",
               pd_hess = pd_hess, conv_code = conv_code,
               aic = AIC(fit), bic = BIC(fit),
               loglik = as.numeric(logLik(fit)), df = attr(logLik(fit), "df"),
               cond = cond,
               sigma2_u0 = s2_u0, sigma2_u1 = s2_u1, sigma_u01 = s_u01,
               dispersion = disp, mu_fixed = mu_fixed))
}

rhs_for <- function(cl, model) {
  switch(model,
         M1 = cl$base,
         M2 = c(cl$base, M2_ADD),
         M3 = c(cl$base, M3_ADD),
         stop("unknown model: ", model))
}

results <- list()

# ------------------------------------------------------------
# The six substantive fits: (1 + time | state), no fallback tier.
# A fit that fails here is REPORTED as failed. It is not downgraded to
# (1 | state) -- the random slope is the per-state rate of change around the
# WPS revision, and a table whose rows sit at different RE structures is the
# exact defect this arm exists to fix.
# ------------------------------------------------------------
for (cell_name in names(CELLS)) {
  cl <- CELLS[[cell_name]]
  for (model in c("M1", "M2", "M3")) {
    key <- sprintf("%s__%s__nbinom2__rs", cell_name, model)
    cat("fitting", key, "\n")
    res <- fit_nb2(panel, cl$dv, rhs_for(cl, model), RS_RE)
    res$cell <- cell_name; res$model <- model
    res$outcome <- cl$outcome; res$re_tier <- "rs"
    if (!isTRUE(res$converged)) {
      cat(sprintf("WARNING [%s]: did NOT converge at %s; message=%s\n",
                  key, RS_RE, res$message %||% ""))
    }
    results[[key]] <- res
  }
}

write_json(results, out_json, auto_unbox = TRUE, digits = 10, na = "null")
cat("\nWrote", length(results), "entries to", out_json, "\n")
n_bad <- sum(vapply(results, function(r) !isTRUE(r$converged), logical(1)))
if (n_bad > 0) cat("WARNING:", n_bad, "fit(s) did not converge -- see 'converged' flags\n")
```

- [ ] **Step 4: Run the fits**

```bash
Rscript scripts/r_env_check.R
Rscript scripts/nb2_stepwise_models.R \
        data/generated/count_model_panel_2021.csv \
        data/generated/nb2_stepwise_results.json
```

Expected: 6 entries written, no `WARNING: ... did not converge`. A benign TMB ABI-mismatch warning on stderr is expected and is the only thing that should appear there.

**If either M2 fit does not converge, STOP and report it.** Those two fits are the only genuinely unknown ones in this arm (the ladder never fit M2 under `nbinom2` for these cells). Per the spec, a non-converging fit is a finding, not something to work around by relaxing the RE structure.

- [ ] **Step 5: Run the validator and watch it pass**

Run: `python3 scripts/validate_nb2_stepwise.py --skip-precondition`

Expected: sections `[1]`, `[2]`, `[3]` all PASS. `[3]` should show **2 SKIP lines** — the two M2 keys, which have no counterpart in the frozen ladder — and PASS for the four M1/M3 overlaps. Section `[0]` shows 1 SKIP.

Then run the full gate once, including the precondition:

Run: `python3 scripts/validate_nb2_stepwise.py`

Expected: `ALL CHECKS PASSED`, 0 SKIP in section `[0]`. This takes a few minutes — `validate_count_models.py` does live statsmodels refits.

- [ ] **Step 6: Commit**

```bash
git add scripts/nb2_stepwise_models.R scripts/validate_nb2_stepwise.py \
        data/generated/nb2_stepwise_results.json
git commit -m "Fit the six NB2 stepwise models at the random-slope tier

Family fixed at nbinom2 across M1/M2/M3 for both 2021 cells, so the columns
share one model class. The two M2 fits are new -- the frozen ladder only fits
M2 under each cell's selected family, and nbinom2 is neither cell's.

The four overlapping fits reproduce the frozen ladder to optimizer noise,
which is a free regression test on the whole R path.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Random-intercept series and the matched baseline

**Files:**
- Modify: `scripts/nb2_stepwise_models.R` (append a second fitting block after the substantive loop)
- Modify: `scripts/validate_nb2_stepwise.py` (add section `[4]`, register it in `main()`)

**Interfaces:**
- Consumes: `fit_nb2(d, dv, rhs, re, sample_rhs = NULL)` and `rhs_for(cl, model)` from Task 1.
- Produces: 8 more records in `data/generated/nb2_stepwise_results.json` — six `{cell}__{M1,M2,M3}__nbinom2__ri` and two `{cell}__M1matched__nbinom2__ri`. Same field set as Task 1, with `re_tier == "ri"`, `sigma2_u1` and `sigma_u01` null, and on the matched records an extra field `sample_model` (str, `"M3"`) recording which model's sample defined the rows. Task 5 reads `sigma2_u0` off these records and off nothing else.

**Why this series exists:** in the random-slope spec σ²_u0 is the between-state variance *at the 2017 centering year* and trades off against σ²_u1 / σ_u01, so its reduction is not bounded to [0,1] and can go negative — that is what produced the −12.5% artifact in the log-LMM violations table. The published tables' `delta_pct_ri` convention reads the reduction from random-intercept-only refits instead. This arm follows that convention.

- [ ] **Step 1: Write the failing validator section**

Add to `scripts/validate_nb2_stepwise.py`, immediately before `def main()`:

```python
# ============================================================
# [4] RANDOM-INTERCEPT SERIES for the Delta sigma^2_u0 block
# ============================================================
def validate_ri_series():
    section('4', 'Random-intercept-only series and the matched Model-1 baseline')
    res = load_results()
    for cell in CELLS:
        for model in MODELS:
            key = f'{cell}__{model}__nbinom2__ri'
            r = res.get(key)
            if r is None:
                check(f'{key} present', False, 'missing from results JSON')
                continue
            check(f'{key} converged', bool(r['converged']), r.get('message', ''))
            check(f'{key} re_tier == ri', r['re_tier'] == 'ri', f"got {r['re_tier']}")
            check(f'{key} re_used == (1 | state)', r['re_used'] == '(1 | state)',
                  f"got {r['re_used']}")
            check(f'{key} has NO random slope', r['sigma2_u1'] is None,
                  f"got sigma2_u1 = {r['sigma2_u1']!r}")
            check(f'{key} has no intercept-slope covariance',
                  r['sigma_u01'] is None, f"got {r['sigma_u01']!r}")
            check(f'{key} sigma2_u0 finite and > 0',
                  r['sigma2_u0'] is not None and r['sigma2_u0'] > 0)
            # The RI refit must sit on EXACTLY the rows of its random-slope
            # twin. If it did not, the reduction would confound a variance
            # change with a sample change.
            rs = res.get(f'{cell}__{model}__nbinom2__rs', {})
            check(f'{key} shares its random-slope twin\'s sample',
                  r['n_obs'] == rs.get('n_obs') and r['n_states'] == rs.get('n_states'),
                  f"ri {r['n_obs']}/{r['n_states']} vs rs "
                  f"{rs.get('n_obs')}/{rs.get('n_states')}")
            # Same fixed effects as the twin -- only the RE structure differs.
            check(f'{key} fixed effects identical to its random-slope twin',
                  set(r['cond']) == set(rs.get('cond', {})),
                  f"ri {sorted(r['cond'])} vs rs {sorted(rs.get('cond', {}))}")
    # The matched baseline: Model 1's formula on Model 3's rows.
    for cell in CELLS:
        key = f'{cell}__M1matched__nbinom2__ri'
        r = res.get(key)
        if r is None:
            check(f'{key} present', False, 'missing from results JSON')
            continue
        m1 = res.get(f'{cell}__M1__nbinom2__ri', {})
        m3 = res.get(f'{cell}__M3__nbinom2__ri', {})
        check(f'{key} converged', bool(r['converged']), r.get('message', ''))
        check(f'{key} re_tier == ri', r['re_tier'] == 'ri')
        check(f'{key} records the sample it was matched to',
              r.get('sample_model') == 'M3', f"got {r.get('sample_model')!r}")
        check(f'{key} carries Model 1 fixed effects, not Model 3',
              set(r['cond']) == set(m1.get('cond', {})),
              f"got {sorted(r['cond'])}")
        check(f'{key} sits on the Model-3 sample',
              r['n_obs'] == m3.get('n_obs') and r['n_states'] == m3.get('n_states'),
              f"matched {r['n_obs']}/{r['n_states']} vs M3 "
              f"{m3.get('n_obs')}/{m3.get('n_states')}")
        check(f'{key} is a genuinely different sample from the M1 RI fit',
              r['n_obs'] != m1.get('n_obs'),
              'matched baseline equals the unmatched one -- the two Delta '
              'percentages would then be identical and one of them is wrong')
```

Register it in `main()`, after `validate_overlap()`:

```python
    validate_ri_series()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python3 scripts/validate_nb2_stepwise.py --skip-precondition`

Expected: sections `[1]`–`[3]` still PASS; section `[4]` reports 8 FAIL lines of the form `[key] present -- missing from results JSON`.

- [ ] **Step 3: Append the RI fitting block to the R script**

Insert into `scripts/nb2_stepwise_models.R`, **between** the substantive loop and the `write_json(...)` call:

```r
# ------------------------------------------------------------
# Random-intercept-only series, for the Delta sigma^2_u0 block ONLY.
#
# Coefficients are never read off these fits. They exist because in the
# random-slope spec sigma^2_u0 is the between-state variance AT THE 2017
# CENTERING YEAR and trades off against sigma^2_u1 / sigma_u01, so its
# reduction is not bounded to [0, 1] and can go negative -- which is exactly
# what produced the -12.5% artifact in the log-LMM violations table. The
# published tables read Delta from random-intercept-only refits (the
# `delta_pct_ri` convention) and so does this arm.
# ------------------------------------------------------------
for (cell_name in names(CELLS)) {
  cl <- CELLS[[cell_name]]
  for (model in c("M1", "M2", "M3")) {
    key <- sprintf("%s__%s__nbinom2__ri", cell_name, model)
    cat("fitting", key, "(variance-reduction series)\n")
    res <- fit_nb2(panel, cl$dv, rhs_for(cl, model), RI_RE)
    res$cell <- cell_name; res$model <- model
    res$outcome <- cl$outcome; res$re_tier <- "ri"
    if (!isTRUE(res$converged)) {
      cat(sprintf("WARNING [%s]: did NOT converge; message=%s\n", key,
                  res$message %||% ""))
    }
    results[[key]] <- res
  }

  # The matched baseline: Model 1's FORMULA on Model 3's ROWS. M1 keeps all 49
  # states; M2/M3 lose AK/RI/VT, which have no BLS pesticide-applicator series.
  # Without this fit, a Model-1-baseline reduction would silently mix "the
  # covariates explained variance" with "three high-variance states left the
  # sample". Both percentages are reported precisely because neither alone
  # tells the truth.
  key <- sprintf("%s__M1matched__nbinom2__ri", cell_name)
  cat("fitting", key, "(Model-1 baseline on the Model-3 sample)\n")
  res <- fit_nb2(panel, cl$dv, rhs_for(cl, "M1"), RI_RE,
                 sample_rhs = rhs_for(cl, "M3"))
  res$cell <- cell_name; res$model <- "M1matched"
  res$outcome <- cl$outcome; res$re_tier <- "ri"
  res$sample_model <- "M3"
  if (!isTRUE(res$converged)) {
    cat(sprintf("WARNING [%s]: did NOT converge; message=%s\n", key,
                res$message %||% ""))
  }
  results[[key]] <- res
}
```

- [ ] **Step 4: Re-run the fits and the validator**

```bash
Rscript scripts/nb2_stepwise_models.R \
        data/generated/count_model_panel_2021.csv \
        data/generated/nb2_stepwise_results.json
python3 scripts/validate_nb2_stepwise.py --skip-precondition
```

Expected: 14 entries written; sections `[1]`–`[4]` all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/nb2_stepwise_models.R scripts/validate_nb2_stepwise.py \
        data/generated/nb2_stepwise_results.json
git commit -m "Add the random-intercept series and matched Model-1 baseline

Delta sigma^2_u0 is read from (1 | state) refits, not from the random-slope
fits, because under a random slope sigma^2_u0 is the between-state variance at
the 2017 centering year and its reduction is not bounded to [0,1].

The matched baseline refits Model 1 on Model 3's rows, so the reported
reduction can be separated from the loss of AK/RI/VT to the BLS applicator
series.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: COVID robustness under NB2

**Files:**
- Modify: `scripts/nb2_stepwise_models.R` (append a third block)
- Modify: `scripts/validate_nb2_stepwise.py` (add section `[5]`, register in `main()`)

**Interfaces:**
- Consumes: `fit_nb2`, `rhs_for`, `CELLS`, `M3_ADD`, `RS_RE` from Tasks 1–2.
- Produces: 2 records, `{cell}__M3covid__nbinom2__rs`, with `model == "M3covid"` and a `covid` term in `cond`. Task 5 reads these to build the COVID section of the memo.

**Why:** the two existing COVID variants in the frozen ladder are `insp_2021__M3covid__zinb` and `viol_cov_2021__M3covid__nbinom1` — verified, neither is NB2. CLAUDE.md records a load-bearing violations finding (COVID indicator −0.279, p ≈ 0.096, with `time2` moving from p ≈ 2.0e-5 to p ≈ 0.26 once the indicator enters) that is an **NB1** result and does not transfer.

- [ ] **Step 1: Write the failing validator section**

Add to `scripts/validate_nb2_stepwise.py` before `def main()`:

```python
# ============================================================
# [5] COVID ROBUSTNESS under nbinom2
# ============================================================
def validate_covid():
    section('5', 'COVID robustness refit under nbinom2')
    res = load_results()
    import pandas as pd
    p = pd.read_csv(GEN + 'count_model_panel_2021.csv')
    # The indicator must be what it claims to be before any coefficient on it
    # is interpretable.
    covid_years = set(p.loc[p['covid'] == 1, 'year'])
    check('covid indicator marks exactly 2020 and 2021',
          covid_years == {2020, 2021}, f'got {sorted(covid_years)}')
    for cell in CELLS:
        key = f'{cell}__M3covid__nbinom2__rs'
        r = res.get(key)
        if r is None:
            check(f'{key} present', False, 'missing from results JSON')
            continue
        m3 = res.get(f'{cell}__M3__nbinom2__rs', {})
        check(f'{key} converged', bool(r['converged']), r.get('message', ''))
        check(f'{key} re_tier == rs', r['re_tier'] == 'rs')
        check(f'{key} carries a covid term', 'covid' in r['cond'])
        check(f'{key} is M3 + covid and nothing else',
              set(r['cond']) == set(m3.get('cond', {})) | {'covid'},
              f"got {sorted(r['cond'])}")
        check(f'{key} sits on the M3 sample',
              r['n_obs'] == m3.get('n_obs'),
              f"{r['n_obs']} vs M3 {m3.get('n_obs')}")
        check(f'{key} covid SE is finite',
              r['cond']['covid']['se'] == r['cond']['covid']['se']
              and r['cond']['covid']['se'] < float('inf'))
    # The whole point of the refit: it must be a DIFFERENT family from the
    # frozen ladder's COVID variants, or nothing has been re-derived.
    with open(GEN + 'count_model_results.json') as fh:
        old = json.load(fh)
    old_covid = [k for k in old if '__M3covid__' in k]
    check('frozen ladder COVID variants are not nbinom2 (so a refit was needed)',
          all(not k.endswith('__nbinom2') for k in old_covid),
          f'found {old_covid}')
```

Register in `main()` after `validate_ri_series()`:

```python
    validate_covid()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python3 scripts/validate_nb2_stepwise.py --skip-precondition`

Expected: `[5]` reports the covid-indicator check PASS and 2 FAIL lines for the missing keys.

- [ ] **Step 3: Append the COVID block to the R script**

Insert into `scripts/nb2_stepwise_models.R`, before `write_json(...)`:

```r
# ------------------------------------------------------------
# COVID robustness, refit under nbinom2.
#
# The frozen ladder's COVID variants are insp_2021__M3covid__zinb and
# viol_cov_2021__M3covid__nbinom1 -- neither is nbinom2, so neither transfers.
# This matters beyond bookkeeping: the violations sensitivity recorded in
# CLAUDE.md (indicator -0.279, p ~ 0.096, with time2 moving from p ~ 2.0e-5 to
# p ~ 0.26 once the indicator enters) is an NB1 result and must be re-derived
# here. If it moves, the CLAUDE.md note gets corrected; if it holds, that is
# worth stating too.
#
# viol_off_2021 was excluded from the ladder's COVID cells for a recorded
# reason (the offset drops every zero-inspection state-year). That exclusion is
# moot here: this arm does not fit the offset spec at all.
# ------------------------------------------------------------
for (cell_name in names(CELLS)) {
  cl <- CELLS[[cell_name]]
  key <- sprintf("%s__M3covid__nbinom2__rs", cell_name)
  cat("fitting", key, "(COVID robustness)\n")
  res <- fit_nb2(panel, cl$dv, c(rhs_for(cl, "M3"), "covid"), RS_RE)
  res$cell <- cell_name; res$model <- "M3covid"
  res$outcome <- cl$outcome; res$re_tier <- "rs"
  if (!isTRUE(res$converged)) {
    cat(sprintf("WARNING [%s]: did NOT converge; message=%s\n", key,
                res$message %||% ""))
  }
  results[[key]] <- res
}
```

- [ ] **Step 4: Re-run and verify**

```bash
Rscript scripts/nb2_stepwise_models.R \
        data/generated/count_model_panel_2021.csv \
        data/generated/nb2_stepwise_results.json
python3 scripts/validate_nb2_stepwise.py --skip-precondition
```

Expected: 16 entries written; sections `[1]`–`[5]` all PASS.

- [ ] **Step 5: Record what the refit actually says**

Run:

```bash
python3 -c "
import json
d = json.load(open('data/generated/nb2_stepwise_results.json'))
for cell in ('insp_2021','viol_cov_2021'):
    base = d[f'{cell}__M3__nbinom2__rs']['cond']
    cov  = d[f'{cell}__M3covid__nbinom2__rs']['cond']
    print(cell)
    print('  covid    b=%.4f p=%.4g' % (cov['covid']['b'], cov['covid']['p']))
    for t in ('time','time2','time3'):
        print('  %-6s p %.4g -> %.4g' % (t, base[t]['p'], cov[t]['p']))
"
```

This is the number Task 5's memo reports and the number that decides whether the CLAUDE.md note needs correcting. Read it now — do not defer it to the memo.

- [ ] **Step 6: Commit**

```bash
git add scripts/nb2_stepwise_models.R scripts/validate_nb2_stepwise.py \
        data/generated/nb2_stepwise_results.json
git commit -m "Refit the COVID robustness check under nbinom2

The frozen ladder's COVID variants sit under zinb and nbinom1, so the
violations sensitivity currently recorded in CLAUDE.md is an NB1 result and
does not transfer to this arm.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Independent cross-software check

**Files:**
- Create: `scripts/nb2_crosscheck_statsmodels.py`
- Modify: `scripts/validate_nb2_stepwise.py` (add section `[6]`, register in `main()`)

**Interfaces:**
- Consumes: `data/generated/count_model_panel_2021.csv`; `data/generated/nb2_stepwise_results.json` (reads `cond[term]['b']` and `['se']` from the `__M3__nbinom2__rs` records).
- Produces: `data/generated/nb2_stepwise_crosscheck.json` — an object keyed by cell, each value `{"n_obs": int, "n_states": int, "converged": bool, "warnings": [str], "terms": {term: {"b_sm": float, "se_sm": float, "b_tmb": float, "se_tmb": float, "ratio": float, "sign_agrees": bool, "both_nonzero": bool}}}`.

**Why this check and not the old one:** section `[5]` of `validate_count_models.py` cross-checks the zero-inflation component against `ZeroInflatedNegativeBinomialP`. An NB2-only arm has no ZI component, so that check has nothing to test — it is retired here, not re-run, and replaced by the NB2 analogue at the same place in the design.

**What this check honestly is:** `statsmodels` has no multilevel negative binomial, so the comparison uses **state fixed effects (dummies) instead of random effects**. Those are different estimators and the coefficients will *not* match to a tight tolerance. Asserting one would mean tightening a threshold until it passes, which proves nothing. The assertion is therefore **sign agreement among terms both implementations resolve away from zero**, with the ratios reported for the reader to judge.

- [ ] **Step 1: Write the failing validator section**

Add to `scripts/validate_nb2_stepwise.py` before `def main()`:

```python
# ============================================================
# [6] CROSS-SOFTWARE CHECK (statsmodels NB2, state fixed effects)
# ============================================================
def validate_crosscheck():
    section('6', 'Cross-software check: statsmodels NegativeBinomialP(p=2) + state FE')
    try:
        with open(GEN + 'nb2_stepwise_crosscheck.json') as fh:
            cc = json.load(fh)
    except FileNotFoundError:
        check('nb2_stepwise_crosscheck.json exists', False, 'not found')
        return
    for cell in CELLS:
        c = cc.get(cell)
        if c is None:
            check(f'{cell} present in cross-check', False)
            continue
        check(f'{cell}: statsmodels fit converged', bool(c['converged']))
        check(f'{cell}: same analytic sample as the glmmTMB M3 fit',
              c['n_obs'] == load_results()[f'{cell}__M3__nbinom2__rs']['n_obs'],
              f"crosscheck {c['n_obs']}")
        terms = c['terms']
        # Ruling R7. State fixed effects ABSORB any state-invariant regressor,
        # so the check can only cover the time-varying (Level-1) terms. The
        # expected split is re-derived HERE from the panel, independently of
        # what the cross-check script wrote, so the two computations can
        # genuinely disagree.
        import pandas as _pd
        _p = _pd.read_csv(GEN + 'count_model_panel_2021.csv')
        _m3 = [t for t in load_results()[f'{cell}__M3__nbinom2__rs']['cond']
               if t != '(Intercept)']
        _d = _p[[ 'state'] + _m3].dropna()
        _varying = {t for t in _m3
                    if _d.groupby('state')[t].std(ddof=0).max() > 1e-12}
        check(f'{cell}: cross-check covers exactly the time-varying M3 terms',
              set(terms) == _varying, f'got {sorted(terms)}, want {sorted(_varying)}')
        check(f'{cell}: absorbed terms are exactly the state-invariant M3 terms',
              set(c.get('terms_absorbed', [])) == set(_m3) - _varying,
              f"got {sorted(c.get('terms_absorbed', []))}, "
              f"want {sorted(set(_m3) - _varying)}")
        check(f'{cell}: absorbed terms carry a stated reason',
              bool(str(c.get('absorbed_reason', '')).strip()))
        # The assertion. Sign agreement ONLY among terms both implementations
        # resolve away from zero -- a term neither can distinguish from zero has
        # no sign to agree about, and demanding one would be manufacturing a
        # check that passes by luck.
        contested = [t for t, v in terms.items()
                     if v['both_nonzero'] and not v['sign_agrees']]
        check(f'{cell}: all clearly-nonzero terms agree in sign across software',
              not contested, f'disagree: {contested}')
        n_tested = sum(1 for v in terms.values() if v['both_nonzero'])
        check(f'{cell}: at least one term was actually testable',
              n_tested > 0,
              'no term was distinguishable from zero in both fits, so this '
              'section proved nothing')
        print(f"    ({cell}: {n_tested}/{len(terms)} terms distinguishable from "
              f"zero in both fits; ratios reported in the memo, not asserted)")
```

Register in `main()` after `validate_covid()`:

```python
    validate_crosscheck()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python3 scripts/validate_nb2_stepwise.py --skip-precondition`

Expected: `[6]` reports `FAIL nb2_stepwise_crosscheck.json exists -- not found`.

- [ ] **Step 3: Write the cross-check script**

Create `scripts/nb2_crosscheck_statsmodels.py`.

```python
"""
Independent cross-software check for the NB2 stepwise arm.

WHAT THIS IS. statsmodels has no multilevel negative binomial, so this fits the
same Model-3 fixed effects with STATE FIXED EFFECTS (dummies) in place of the
random intercept and slope, using a different implementation in a different
language, and compares the shared coefficients against glmmTMB's.

WHAT THIS IS NOT. It is not a numerical equivalence test. Fixed effects are not
random effects; a random-effects estimator shrinks state deviations toward zero
and a dummy-variable estimator does not, so the coefficients differ by more than
optimizer noise and are expected to. The assertion in validate_nb2_stepwise [6]
is therefore SIGN AGREEMENT among terms both implementations resolve away from
zero. Ratios are reported for the reader; they are not thresholded. Tightening a
tolerance until it passes would prove nothing.

This replaces the ZI cross-check in validate_count_models.py [5], which has no
counterpart here: an NB2-only arm has no zero-inflation component to check.

Spec: docs/superpowers/specs/2026-08-28-nb2-stepwise-design.md section 5.2
Run:  python3 scripts/nb2_crosscheck_statsmodels.py
"""
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import NegativeBinomialP

GEN = '/Users/keshavgoel/Research/data/generated/'

CUBIC = ['time', 'time2', 'time3']
M3_ADD = ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
          'h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']

CELLS = {
    'insp_2021': {'dv': 'inspections', 'base': CUBIC},
    'viol_cov_2021': {'dv': 'violations', 'base': ['log_inspections'] + CUBIC},
}


def fit_one(panel, dv, rhs):
    """NB2 with state dummies. Returns (params, bse, n_obs, n_states, converged,
    warning messages)."""
    need = [dv, 'state'] + rhs
    d = panel[need].dropna().copy()
    X = d[rhs].astype(float)
    dummies = pd.get_dummies(d['state'], prefix='st', drop_first=True).astype(float)
    X = pd.concat([X, dummies], axis=1)
    X.insert(0, 'const', 1.0)
    y = d[dv].astype(float)

    # Warnings are captured and reported, never filtered -- a
    # filterwarnings('ignore') is what hid a non-converged published fit in this
    # project once already.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = NegativeBinomialP(y.values, X.values, p=2)
        fit = model.fit(method='bfgs', maxiter=5000, disp=0)
    msgs = sorted({str(w.message) for w in caught})

    params = pd.Series(fit.params[:len(X.columns)], index=X.columns)
    bse = pd.Series(fit.bse[:len(X.columns)], index=X.columns)
    return params, bse, len(d), d['state'].nunique(), bool(fit.mle_retvals['converged']), msgs


def main():
    panel = pd.read_csv(GEN + 'count_model_panel_2021.csv')
    with open(GEN + 'nb2_stepwise_results.json') as fh:
        res = json.load(fh)

    out = {}
    for cell, cfg in CELLS.items():
        rhs = cfg['base'] + M3_ADD
        params, bse, n_obs, n_states, converged, msgs = fit_one(panel, cfg['dv'], rhs)
        tmb = res[f'{cell}__M3__nbinom2__rs']['cond']

        terms = {}
        for t in rhs:
            b_sm, se_sm = float(params[t]), float(bse[t])
            b_tmb, se_tmb = float(tmb[t]['b']), float(tmb[t]['se'])
            # "Distinguishable from zero" == |b| > 2*SE in BOTH fits. A term
            # neither implementation resolves has no sign to agree about.
            both_nonzero = abs(b_sm) > 2 * se_sm and abs(b_tmb) > 2 * se_tmb
            terms[t] = {
                'b_sm': b_sm, 'se_sm': se_sm,
                'b_tmb': b_tmb, 'se_tmb': se_tmb,
                'ratio': b_sm / b_tmb if b_tmb != 0 else float('nan'),
                'sign_agrees': np.sign(b_sm) == np.sign(b_tmb),
                'both_nonzero': bool(both_nonzero),
            }

        out[cell] = {'n_obs': n_obs, 'n_states': n_states,
                     'converged': converged, 'warnings': msgs, 'terms': terms}

        print(f"\n{cell}  (statsmodels NB2 + {n_states - 1} state dummies, "
              f"N={n_obs}, converged={converged})")
        if msgs:
            print("  warnings raised during the fit (captured, not filtered):")
            for m in msgs:
                print(f"    - {m}")
        print(f"  {'term':24}{'b (statsmodels)':>18}{'b (glmmTMB)':>16}"
              f"{'ratio':>10}{'both != 0':>11}{'sign':>7}")
        for t, v in terms.items():
            print(f"  {t:24}{v['b_sm']:>18.4f}{v['b_tmb']:>16.4f}"
                  f"{v['ratio']:>10.3f}{str(v['both_nonzero']):>11}"
                  f"{('ok' if v['sign_agrees'] else 'DIFFERS'):>7}")

    with open(GEN + 'nb2_stepwise_crosscheck.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f"\nWrote {GEN}nb2_stepwise_crosscheck.json")
    print("\nNOTE: ratios are reported, not thresholded. State fixed effects are "
          "not\nstate random effects -- the estimates differ by more than "
          "optimizer noise by\nconstruction. Only sign agreement among "
          "clearly-nonzero terms is asserted.")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the cross-check, then the validator**

```bash
python3 scripts/nb2_crosscheck_statsmodels.py
python3 scripts/validate_nb2_stepwise.py --skip-precondition
```

Expected: the cross-check prints two tables and writes the JSON; sections `[1]`–`[6]` all PASS.

**If a sign disagreement appears on a clearly-nonzero term, STOP and report it.** That is a substantive discrepancy between two implementations of the same model, not a tolerance to loosen.

- [ ] **Step 5: Commit**

```bash
git add scripts/nb2_crosscheck_statsmodels.py scripts/validate_nb2_stepwise.py \
        data/generated/nb2_stepwise_crosscheck.json
git commit -m "Add an NB2 cross-software check to replace the retired ZI one

An NB2-only arm has no zero-inflation component, so the ladder's ZI
cross-check has nothing to test here. The analogue is statsmodels NB2 with
state fixed effects against glmmTMB's random effects.

Asserts sign agreement among terms both implementations resolve away from
zero, and reports the ratios without thresholding them -- fixed effects are
not random effects and the estimates differ by construction.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Reporting — variance table, coefficient table, memo

**Files:**
- Create: `scripts/report_nb2_stepwise.py`
- Modify: `scripts/validate_nb2_stepwise.py` (add sections `[7]`, `[8]`, `[9]`, register in `main()`)

**Interfaces:**
- Consumes: `data/generated/nb2_stepwise_results.json`, `data/generated/nb2_stepwise_crosscheck.json`.
- Produces: `data/generated/nb2_stepwise_variance.csv` with columns `cell, outcome, model, sigma2_u0_rs, sigma2_u1_rs, sigma_u01_rs, theta, sigma2_u0_ri, icc_ri, mu_fixed, delta_pct_ri, delta_pct_ri_matched, n_obs, n_states`; `data/generated/nb2_stepwise_coefficients.csv` with columns `cell, outcome, model, term, b, se, z, p, irr`; and `docs/nb2_stepwise_models.md`.

**The two derived quantities, defined once, here and nowhere else:**

```
delta_pct_ri          = 100 * (s2_u0_ri[M1]        - s2_u0_ri[model]) / s2_u0_ri[M1]
delta_pct_ri_matched  = 100 * (s2_u0_ri[M1matched] - s2_u0_ri[model]) / s2_u0_ri[M1matched]
icc_ri                = s2_u0_ri / (s2_u0_ri + log(1 + 1/theta + 1/mu_fixed))
```

`icc_ri` is Nakagawa's observation-level (distribution-specific) variance for an NB2 with a log link. It is a **link-scale approximation** and is not comparable to this project's headline "ICC ~77%", which comes from a Gaussian LMM on `log(count + 1)`. `theta` and `mu_fixed` are taken from the same RI record as `s2_u0_ri`, so the ICC is entirely on the RI basis; the `*_rs` columns are the only random-slope quantities in the table and are labelled as such.

For M1 itself, `delta_pct_ri` is 0 by construction (it is its own baseline) and is written as `0.0`, not blank.

- [ ] **Step 1: Write the failing validator sections**

Add to `scripts/validate_nb2_stepwise.py` before `def main()`:

```python
# ============================================================
# [7] VARIANCE ARITHMETIC
# ============================================================
def validate_variance_table():
    section('7', 'Variance components table: arithmetic and basis')
    import math
    import pandas as pd
    try:
        v = pd.read_csv(GEN + 'nb2_stepwise_variance.csv')
    except FileNotFoundError:
        check('nb2_stepwise_variance.csv exists', False, 'not found')
        return
    res = load_results()
    check('one row per cell x model', len(v) == 6, f'got {len(v)}')
    for _, row in v.iterrows():
        cell, model = row['cell'], row['model']
        rs = res[f'{cell}__{model}__nbinom2__rs']
        ri = res[f'{cell}__{model}__nbinom2__ri']
        tag = f'{cell} {model}'
        # The *_rs columns come from the random-slope fits...
        check_close(f'{tag}: sigma2_u0_rs from the rs fit',
                    row['sigma2_u0_rs'], rs['sigma2_u0'], 1e-9, 'rel')
        check_close(f'{tag}: sigma2_u1_rs from the rs fit',
                    row['sigma2_u1_rs'], rs['sigma2_u1'], 1e-9, 'rel')
        # ...and sigma2_u0_ri, theta and the ICC all come from the RI fit, so
        # the ICC and the Delta share one basis.
        check_close(f'{tag}: sigma2_u0_ri from the ri fit',
                    row['sigma2_u0_ri'], ri['sigma2_u0'], 1e-9, 'rel')
        check_close(f'{tag}: theta from the ri fit',
                    row['theta'], ri['dispersion'], 1e-9, 'rel')
        want_icc = ri['sigma2_u0'] / (ri['sigma2_u0'] + math.log(
            1 + 1 / ri['dispersion'] + 1 / ri['mu_fixed']))
        check_close(f'{tag}: icc_ri matches Nakagawa OLV formula',
                    row['icc_ri'], want_icc, 1e-9, 'rel')
        check(f'{tag}: icc_ri in (0, 1)', 0 < row['icc_ri'] < 1,
              f"got {row['icc_ri']}")
        base = res[f'{cell}__M1__nbinom2__ri']['sigma2_u0']
        matched = res[f'{cell}__M1matched__nbinom2__ri']['sigma2_u0']
        check_close(f'{tag}: delta_pct_ri arithmetic',
                    row['delta_pct_ri'],
                    100 * (base - ri['sigma2_u0']) / base, 1e-9, 'abs')
        check_close(f'{tag}: delta_pct_ri_matched arithmetic',
                    row['delta_pct_ri_matched'],
                    100 * (matched - ri['sigma2_u0']) / matched, 1e-9, 'abs')
        check(f'{tag}: n_obs/n_states agree with the rs fit',
              row['n_obs'] == rs['n_obs'] and row['n_states'] == rs['n_states'])
    for cell in CELLS:
        m1 = v[(v['cell'] == cell) & (v['model'] == 'M1')].iloc[0]
        check(f'{cell}: M1 delta_pct_ri == 0 (it is its own baseline)',
              abs(m1['delta_pct_ri']) < 1e-9, f"got {m1['delta_pct_ri']}")
    # No sigma2_e column anywhere -- a count family has no residual variance.
    check('no residual-variance column in the variance table',
          not any(c in v.columns for c in ('sigma2_e', 'sigma_e', 'residual_variance')),
          f'columns: {list(v.columns)}')


# ============================================================
# [8] FROZEN ARTIFACTS
# ============================================================
def validate_frozen():
    section('8', 'Frozen artifacts unchanged')
    frozen = [
        'scripts/count_models_zinb.R',
        'scripts/report_count_models.py',
        'scripts/validate_count_models.py',
        'scripts/build_count_model_panel.py',
        'data/generated/count_model_results.json',
        'data/generated/count_model_panel_2019.csv',
        'data/generated/count_model_panel_2021.csv',
        'docs/count_models_zinb.md',
    ]
    proc = subprocess.run(['git', 'diff', 'HEAD', '--name-only', '--'] + frozen,
                          capture_output=True, text=True, cwd=ROOT)
    changed = [ln for ln in proc.stdout.split('\n') if ln.strip()]
    check('no frozen count-model artifact modified', not changed,
          f'modified: {changed}')
    proc = subprocess.run(['git', 'diff', 'HEAD', '--name-only'],
                          capture_output=True, text=True, cwd=ROOT)
    touched = [ln for ln in proc.stdout.split('\n') if ln.strip()]
    bad = [f for f in touched
           if f.endswith('.docx') or 'paper_table' in f]
    check('no .docx and no paper_table artifact modified', not bad,
          f'modified: {bad}')


# ============================================================
# [9] MEMO
# ============================================================
MEMO_FORBIDDEN_PATTERNS = [
    # NB2 lost to the selected family at every rung. Any of these would read as
    # a fit-based win. Spec section 3.
    'nbinom2 was selected', 'NB2 was selected', 'NB2 is selected',
    'NB2 won', 'NB2 wins', 'nbinom2 wins',
    'best-fitting family', 'the preferred family',
    # A count family has no residual variance. NOTE: this also forbids the memo
    # from DENYING one in those exact words, which is why the memo writes the
    # denial hyphenated ('no residual-variance column'). That is deliberate --
    # do not "fix" the memo by removing the hyphen.
    'residual variance', 'sigma2_e', 'sigma^2_e',
    # The Delta comes from the RI series, never from the random-slope fits.
    'reduction in the random-slope',
]


def validate_memo():
    section('9', 'Memo')
    try:
        with open('/Users/keshavgoel/Research/docs/nb2_stepwise_models.md') as fh:
            memo = fh.read()
    except FileNotFoundError:
        check('docs/nb2_stepwise_models.md exists', False, 'not found')
        return
    check('memo marks itself generated',
          'generated' in memo.lower() and 'never hand-edited' in memo.lower())
    for pat in MEMO_FORBIDDEN_PATTERNS:
        check(f'memo does not contain: {pat!r}', pat.lower() not in memo.lower())
    # The caveats the spec requires by name.
    required = [
        ('AIC penalty is stated', 'editorial'),
        ('link-scale caveat present', 'link scale'),
        ('sample-change caveat present', 'AK, RI and VT'),
        ('ICC formula printed', 'Nakagawa'),
        ('COVID section present', 'COVID'),
        ('cross-check section present', 'state fixed effects'),
    ]
    # Case-SENSITIVE and exact-phrase. A case-folded 'AK' matched 'make',
    # 'take' and 'breaks', so that row passed no matter what the memo said.
    for label, needle in required:
        check(f'memo: {label}', needle in memo, f'{needle!r} not found')
    # The ZI-NB tallies belong to the other memo; restating them here would
    # invite a reader to think this arm re-tested zero-inflation. It did not.
    check('memo does not restate the ZI-NB tallies',
          '17-0' not in memo and '17–0' not in memo and '17 of 17' not in memo)
```

Register all three in `main()` after `validate_crosscheck()`:

```python
    validate_variance_table()
    validate_frozen()
    validate_memo()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python3 scripts/validate_nb2_stepwise.py --skip-precondition`

Expected: `[7]` FAILs on the missing CSV, `[8]` PASSes already (nothing frozen has been touched), `[9]` FAILs on the missing memo.

- [ ] **Step 3: Write the reporter**

Create `scripts/report_nb2_stepwise.py`.

```python
"""
Reads the NB2 stepwise artifacts and produces the tables and the memo.

PURE ARTIFACT READER. It fits nothing. Every number it prints traces to
data/generated/nb2_stepwise_results.json or nb2_stepwise_crosscheck.json.

This is the ONLY script in the arm that computes a derived statistic, so
delta_pct_ri, delta_pct_ri_matched and icc_ri each have exactly one definition:

  delta_pct_ri         = 100 * (s2_u0_ri[M1]        - s2_u0_ri[m]) / s2_u0_ri[M1]
  delta_pct_ri_matched = 100 * (s2_u0_ri[M1matched] - s2_u0_ri[m]) / s2_u0_ri[M1matched]
  icc_ri               = s2 / (s2 + log(1 + 1/theta + 1/mu)),  all from the RI fit

Spec: docs/superpowers/specs/2026-08-28-nb2-stepwise-design.md
Run:  python3 scripts/report_nb2_stepwise.py
"""
import json
import math

import pandas as pd

GEN = '/Users/keshavgoel/Research/data/generated/'
DOCS = '/Users/keshavgoel/Research/docs/'

CELLS = {'insp_2021': 'inspections', 'viol_cov_2021': 'violations'}
MODELS = ('M1', 'M2', 'M3')

TERM_ORDER = ['(Intercept)', 'log_inspections', 'time', 'time2', 'time3',
              'SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
              'h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z',
              'covid']


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''


# The frozen ladder's COVID variants, for the family comparison in the memo.
# They sit under DIFFERENT families -- that is the whole point of re-deriving
# them under NB2 -- so the key carries the family it was fit under.
COVID_LADDER_REF = {
    'insp_2021': ('insp_2021__M3covid__zinb', 'ZINB'),
    'viol_cov_2021': ('viol_cov_2021__M3covid__nbinom1', 'NB1'),
}


def load():
    with open(GEN + 'nb2_stepwise_results.json') as fh:
        res = json.load(fh)
    with open(GEN + 'nb2_stepwise_crosscheck.json') as fh:
        cc = json.load(fh)
    # READ-ONLY. The ladder artifact is frozen; it is opened here only to quote
    # its COVID variants' coefficients beside ours.
    with open(GEN + 'count_model_results.json') as fh:
        ladder = json.load(fh)
    return res, cc, ladder


def icc_nb2(s2_u0, theta, mu):
    """Nakagawa's observation-level variance for an NB2 with a log link. A
    LINK-SCALE approximation -- not comparable to the project's Gaussian-LMM
    ICC on log(count + 1)."""
    return s2_u0 / (s2_u0 + math.log(1 + 1 / theta + 1 / mu))


def variance_table(res):
    rows = []
    for cell, outcome in CELLS.items():
        base = res[f'{cell}__M1__nbinom2__ri']['sigma2_u0']
        matched = res[f'{cell}__M1matched__nbinom2__ri']['sigma2_u0']
        for model in MODELS:
            rs = res[f'{cell}__{model}__nbinom2__rs']
            ri = res[f'{cell}__{model}__nbinom2__ri']
            rows.append({
                'cell': cell, 'outcome': outcome, 'model': model,
                'sigma2_u0_rs': rs['sigma2_u0'],
                'sigma2_u1_rs': rs['sigma2_u1'],
                'sigma_u01_rs': rs['sigma_u01'],
                'theta': ri['dispersion'],
                'sigma2_u0_ri': ri['sigma2_u0'],
                'icc_ri': icc_nb2(ri['sigma2_u0'], ri['dispersion'], ri['mu_fixed']),
                'mu_fixed': ri['mu_fixed'],
                'delta_pct_ri': 100 * (base - ri['sigma2_u0']) / base,
                'delta_pct_ri_matched': 100 * (matched - ri['sigma2_u0']) / matched,
                'n_obs': rs['n_obs'], 'n_states': rs['n_states'],
            })
    return pd.DataFrame(rows)


def coefficient_table(res):
    rows = []
    for cell, outcome in CELLS.items():
        for model in list(MODELS) + ['M3covid']:
            r = res.get(f'{cell}__{model}__nbinom2__rs')
            if r is None:
                continue
            for term, c in r['cond'].items():
                rows.append({'cell': cell, 'outcome': outcome, 'model': model,
                             'term': term, 'b': c['b'], 'se': c['se'],
                             'z': c['z'], 'p': c['p'], 'irr': math.exp(c['b'])})
    df = pd.DataFrame(rows)
    df['_ord'] = df['term'].apply(
        lambda t: TERM_ORDER.index(t) if t in TERM_ORDER else len(TERM_ORDER))
    return df.sort_values(['cell', 'model', '_ord']).drop(columns='_ord')


def md_table(df, cols, fmt):
    head = '| ' + ' | '.join(cols) + ' |'
    rule = '|' + '|'.join(['---'] * len(cols)) + '|'
    body = []
    for _, r in df.iterrows():
        body.append('| ' + ' | '.join(fmt(r, c) for c in cols) + ' |')
    return '\n'.join([head, rule] + body)


def write_memo(res, cc, ladder, vt, ct):
    L = []
    A = L.append
    A('# NB2-Only Stepwise Count Models, 2011-2021')
    A('')
    A('**This file is generated by `scripts/report_nb2_stepwise.py` and is '
      'never hand-edited.** Re-run that script instead. Every number traces to '
      '`data/generated/nb2_stepwise_results.json` or '
      '`nb2_stepwise_crosscheck.json`.')
    A('')
    A('Spec: `docs/superpowers/specs/2026-08-28-nb2-stepwise-design.md`.')
    A('')
    A('## What this is, and what it is not')
    A('')
    A('Six models: inspections M1-M3 and violations M1-M3 on the 2011-2021 WPS '
      'panel, all under `nbinom2`, all at `(1 + time | state)`. Model 1 is '
      'cubic time only; Model 2 adds commodity mix and spending '
      '(`lii_2017_z`, `SPEND_APP_z`, `SPEND_WORK_z`); Model 3 adds the H-2A '
      'block (`h2a_per_farmworker_z`, `dol_demand_met_pct_z`, `pct_flc_z`). '
      'The violations models carry `log_inspections` as a Level-1 covariate, '
      'matching the published Table 3.')
    A('')
    A('**Holding the family at NB2 is an editorial decision, not a fit-based '
      'one.** At this random-effects structure NB2 loses to the family the '
      'six-family ladder selected: by about 33 AIC to ZINB for inspections and '
      'about 11 AIC to NB1 for violations, at both Model 1 and Model 3. What '
      'fixing the family buys is a table whose three columns are the same '
      'model class, so the between-state variance actually forms a reduction '
      'sequence instead of being three different models\' parameters. NB2 is '
      'the least-bad single choice across both outcomes: it beats NB1 for '
      'inspections by about 138 AIC while costing about 11 for violations.')
    A('')
    A('The family-selection evidence, including the zero-inflation comparisons, '
      'lives in `docs/count_models_zinb.md` and is untouched by this arm. '
      'Nothing here re-tests zero-inflation.')
    A('')
    A('## Variance components')
    A('')
    A('**Two bases in one table, labelled per column.** The `*_rs` columns come '
      'from the random-slope fits the coefficients come from. Everything else '
      '-- theta, ICC, and both reductions -- comes from a parallel '
      '`(1 | state)` refit series on the same rows. That split is deliberate: '
      'under a random slope, sigma^2_u0 is the between-state variance *at the '
      '2017 centering year* and trades off against sigma^2_u1, so its '
      'reduction is not bounded to [0, 1] and can go negative. That is not '
      'hypothetical -- it produced a -12.5% figure in the log-LMM violations '
      'table.')
    A('')
    A('All of these are on the log link scale. Their magnitudes are not '
      'comparable to the sigma^2_u0 values in published Tables 2 and 3, though '
      'a percentage reduction is.')
    A('')
    A('There is no residual-variance column. A count family has none; theta is '
      'the NB2 size parameter in `Var = mu + mu^2/theta`, and its square is not '
      'a variance component.')
    A('')
    A('ICC uses Nakagawa\'s observation-level variance for a log-link NB2, '
      '`sigma^2_u0 / (sigma^2_u0 + ln(1 + 1/theta + 1/mu))`, with mu the '
      'exponentiated mean of the fixed-effects linear predictor. It is a '
      'link-scale approximation and is **not** the project\'s headline '
      '"ICC ~77%", which comes from a Gaussian LMM on `log(count + 1)`.')
    A('')
    for cell, outcome in CELLS.items():
        sub = vt[vt['cell'] == cell]
        A(f'### {outcome} ({cell})')
        A('')
        A('| Model | sigma^2_u0 (rs) | sigma^2_u1 (rs) | sigma_u01 (rs) | '
          'theta (ri) | sigma^2_u0 (ri) | ICC (ri) | Delta sigma^2_u0 % | '
          'Delta % matched | N obs | States |')
        A('|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in sub.iterrows():
            A(f"| {r['model']} | {r['sigma2_u0_rs']:.4f} | "
              f"{r['sigma2_u1_rs']:.5f} | {r['sigma_u01_rs']:.5f} | "
              f"{r['theta']:.4f} | {r['sigma2_u0_ri']:.4f} | "
              f"{r['icc_ri']:.3f} | {r['delta_pct_ri']:+.1f} | "
              f"{r['delta_pct_ri_matched']:+.1f} | {int(r['n_obs'])} | "
              f"{int(r['n_states'])} |")
        A('')
    A('**Why two reduction columns.** Model 1 keeps all 49 states; Models 2 and '
      '3 keep 46, because AK, RI and VT have no BLS pesticide-applicator series '
      'and drop by listwise deletion once `SPEND_APP_z` enters. `Delta '
      'sigma^2_u0 %` is measured against a single Model-1 baseline on Model 1\'s '
      'own sample, so every column starts from Model 1. `Delta % matched` refits '
      'the Model-1 baseline on the Model-2/3 sample, so the difference between '
      'the two columns is exactly what those three states contributed. Neither '
      'number alone tells the truth; read both.')
    A('')
    # The direction of the sample effect is NOT the same for both outcomes, so
    # this paragraph is derived per cell rather than asserted. Writing the
    # inspections direction as if it were general would mis-describe the
    # violations column: there the Model-1 basis UNDERstates the covariates.
    A('**And the two columns differ in opposite directions by outcome**, which '
      'is why the generic warning is not written here:')
    A('')
    for cell, outcome in CELLS.items():
        base = res[f'{cell}__M1__nbinom2__ri']['sigma2_u0']
        matched = res[f'{cell}__M1matched__nbinom2__ri']['sigma2_u0']
        m3 = res[f'{cell}__M3__nbinom2__ri']['sigma2_u0']
        d_base = 100 * (base - m3) / base
        d_match = 100 * (matched - m3) / matched
        direction = ('lowers' if matched < base else 'raises')
        reading = ('overstates' if d_base > d_match else 'understates')
        A(f'- **{outcome}:** dropping AK, RI and VT {direction} the Model-1 '
          f'between-state variance ({base:.4f} on 49 states -> {matched:.4f} on '
          f'46), so the Model-1 basis {reading} what the covariates do: '
          f'Model 3 reduces sigma^2_u0 by {d_base:+.1f}% against Model 1 but '
          f'{d_match:+.1f}% against the matched baseline.')
    A('')
    A('The inspections direction reproduces, under a different model class, an '
      'asymmetry this project already documented for the published log-linear '
      'tables: AK/RI/VT carry much of the between-state inspection variance, '
      'and violations are unaffected by their loss.')
    A('')
    A('## Coefficients')
    A('')
    A('`b (SE)` with significance stars, and IRR = exp(b). Stars: '
      '* p<.05, ** p<.01, *** p<.001.')
    A('')
    for cell, outcome in CELLS.items():
        A(f'### {outcome} ({cell})')
        A('')
        sub = ct[(ct['cell'] == cell) & (ct['model'].isin(MODELS))]
        terms = list(dict.fromkeys(sub['term']))
        A('| Term | ' + ' | '.join(f'{m} b (SE)' + ' | ' + f'{m} IRR'
                                   for m in MODELS) + ' |')
        A('|' + '|'.join(['---'] * (1 + 2 * len(MODELS))) + '|')
        for t in terms:
            cells_out = []
            for m in MODELS:
                row = sub[(sub['model'] == m) & (sub['term'] == t)]
                if len(row):
                    r = row.iloc[0]
                    cells_out += [f"{r['b']:.3f} ({r['se']:.3f}){stars(r['p'])}",
                                  f"{r['irr']:.3f}"]
                else:
                    cells_out += ['', '']
            A(f'| {t} | ' + ' | '.join(cells_out) + ' |')
        A('')
    A('## COVID robustness')
    A('')
    A('2020 and 2021 are pandemic years. Each cell\'s Model 3 is refit with a '
      '2020-21 indicator, under NB2 at the same random-effects structure. This '
      'had to be re-derived rather than carried over: the six-family ladder\'s '
      'COVID variants sit under ZINB (inspections) and NB1 (violations).')
    A('')
    A('| Outcome | covid b (SE) | p | IRR | time p, M3 -> M3+covid | '
      'time2 p, M3 -> M3+covid | time3 p, M3 -> M3+covid |')
    A('|---|---|---|---|---|---|---|')
    for cell, outcome in CELLS.items():
        m3 = res[f'{cell}__M3__nbinom2__rs']['cond']
        cv = res[f'{cell}__M3covid__nbinom2__rs']['cond']
        c = cv['covid']
        A(f"| {outcome} | {c['b']:.3f} ({c['se']:.3f}){stars(c['p'])} | "
          f"{c['p']:.3g} | {math.exp(c['b']):.3f} | " +
          ' | '.join(f"{m3[t]['p']:.3g} -> {cv[t]['p']:.3g}"
                     for t in ('time', 'time2', 'time3')) + ' |')
    A('')
    # Ruling R6. The frozen ladder's COVID variants sit under DIFFERENT
    # families (ZINB for inspections, NB1 for violations), and the comparison
    # splits three ways. Reported as three distinct facts, each derived from
    # the two JSONs rather than asserted, because CLAUDE.md currently records
    # the NB1 coefficient as though it were general.
    A('**How this compares to the ladder\'s own COVID variants, which sit '
      'under different families.** Three separate things are true and they '
      'must not be merged:')
    A('')
    for cell, outcome in CELLS.items():
        old_key, old_fam = COVID_LADDER_REF[cell]
        o_cv = ladder[old_key]['cond']
        o_m3 = ladder[old_key.replace('M3covid', 'M3')]['cond']
        n_cv = res[f'{cell}__M3covid__nbinom2__rs']['cond']
        n_m3 = res[f'{cell}__M3__nbinom2__rs']['cond']
        A(f'- **{outcome}, the indicator itself:** {old_fam} gives '
          f'b = {o_cv["covid"]["b"]:+.4f} (p = {o_cv["covid"]["p"]:.3g}); NB2 '
          f'gives b = {n_cv["covid"]["b"]:+.4f} '
          f'(p = {n_cv["covid"]["p"]:.3g}).')
        A(f'- **{outcome}, the time trend under the indicator:** `time2` moves '
          f'{o_m3["time2"]["p"]:.3g} -> {o_cv["time2"]["p"]:.3g} under '
          f'{old_fam}, and {n_m3["time2"]["p"]:.3g} -> '
          f'{n_cv["time2"]["p"]:.3g} under NB2.')
        A(f'- **{outcome}, before COVID enters at all:** the two families '
          f'already disagree on Model 3\'s own time trend -- `time` '
          f'p = {o_m3["time"]["p"]:.3g} under {old_fam} against '
          f'{n_m3["time"]["p"]:.3g} under NB2, and `time2` '
          f'{o_m3["time2"]["p"]:.3g} against {n_m3["time2"]["p"]:.3g}. That '
          f'gap is a property of the family choice, not of the pandemic.')
    A('')
    A('The practical consequence for the violations column: the COVID '
      'coefficient recorded elsewhere in this project is an NB1 estimate and '
      'does not carry over to NB2, while the qualitative caution it supports '
      '-- that `time2` stops being significant once the indicator enters -- '
      'does carry over. Cite the number with its family attached.')
    A('')
    A('## Cross-software check')
    A('')
    A('`statsmodels` has no multilevel negative binomial, so the independent '
      'check fits the same Model-3 fixed effects with **state fixed effects** '
      '(dummies) in place of the random intercept and slope, in a different '
      'language and a different implementation. This is not a numerical '
      'equivalence test and the estimates are not expected to match: a '
      'random-effects estimator shrinks state deviations toward zero and a '
      'dummy-variable estimator does not. What is asserted is sign agreement '
      'among terms both implementations resolve away from zero (|b| > 2 SE in '
      'both). The ratios below are reported for judgement, not thresholded.')
    A('')
    A('**What this check can and cannot cover.** State fixed effects absorb any '
      'regressor that is constant within a state, so the Level-2 covariates are '
      'perfectly collinear with the state dummies and their coefficients are '
      'not identified at all under this estimator -- the design matrix is '
      'rank-deficient by exactly the number of them. They are therefore dropped '
      'from the comparison, by name, below. This is a property of fixed-effects '
      'estimation, not a shortcoming of either implementation, and it means the '
      'cross-check validates the **Level-1 (time-varying) terms only**. The '
      'Level-2 covariates -- the spending, commodity-mix and H-2A-share '
      'variables that Models 2 and 3 are built from -- are NOT independently '
      'confirmed by this check, and no reader should take them as such.')
    A('')
    A('The zero-inflation cross-check in the six-family pipeline has no '
      'counterpart here -- an NB2-only arm has no zero-inflation component.')
    A('')
    for cell, outcome in CELLS.items():
        c = cc[cell]
        A(f'### {outcome} ({cell}) -- N = {c["n_obs"]}, '
          f'{c["n_states"] - 1} state dummies, converged = {c["converged"]}')
        A('')
        absorbed = c.get('terms_absorbed', [])
        if absorbed:
            A(f'Absorbed by the state dummies and excluded from the comparison '
              f'({len(absorbed)}): ' +
              ', '.join(f'`{t}`' for t in absorbed) + '.')
            A('')
        A('| Term | b (statsmodels FE) | b (glmmTMB RE) | ratio | '
          'both distinguishable from 0 | sign |')
        A('|---|---|---|---|---|---|')
        for t, v in c['terms'].items():
            A(f"| {t} | {v['b_sm']:.4f} | {v['b_tmb']:.4f} | "
              f"{v['ratio']:.3f} | {'yes' if v['both_nonzero'] else 'no'} | "
              f"{'agrees' if v['sign_agrees'] else 'DIFFERS'} |")
        A('')
        if c['warnings']:
            A('Warnings raised during the `statsmodels` fit (captured, not '
              'filtered):')
            A('')
            for m in c['warnings']:
                A(f'- {m}')
            A('')
    A('## Caveats')
    A('')
    A('- NB2 is an editorial choice and loses on AIC to the selected family at '
      'every rung. See the first section.')
    A('- Every variance is on the log link scale.')
    A('- Model 1 has 49 states; Models 2 and 3 have 46. Read both reduction '
      'columns.')
    A('- The ICC is a link-scale approximation, not the project\'s Gaussian-LMM '
      'ICC.')
    A('- This arm covers the 2011-2021 WPS-view window only. It does not fit '
      'the 2011-2019 establishments window or the violations-as-offset '
      'specification.')
    A('- No manuscript `.docx` was regenerated.')
    A('')
    with open(DOCS + 'nb2_stepwise_models.md', 'w') as fh:
        fh.write('\n'.join(L) + '\n')


def main():
    res, cc, ladder = load()
    vt = variance_table(res)
    ct = coefficient_table(res)
    vt.to_csv(GEN + 'nb2_stepwise_variance.csv', index=False)
    ct.to_csv(GEN + 'nb2_stepwise_coefficients.csv', index=False)
    print(vt.to_string(index=False))
    print()
    print(ct.to_string(index=False))
    write_memo(res, cc, ladder, vt, ct)
    print(f"\nWrote {GEN}nb2_stepwise_variance.csv, "
          f"{GEN}nb2_stepwise_coefficients.csv, {DOCS}nb2_stepwise_models.md")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the reporter, then the validator**

```bash
python3 scripts/report_nb2_stepwise.py
python3 scripts/validate_nb2_stepwise.py --skip-precondition
```

Expected: sections `[1]`–`[9]` all PASS.

- [ ] **Step 5: Run the full gate including the precondition**

Run: `python3 scripts/validate_nb2_stepwise.py`

Expected: `ALL CHECKS PASSED`, with **0 SKIP in section `[0]`** and exactly 2 SKIP in section `[3]` (the two M2 keys with no counterpart in the frozen ladder). Any other SKIP means a block silently stopped running — investigate before proceeding.

- [ ] **Step 6: Read the memo end to end**

Open `docs/nb2_stepwise_models.md` and read it as Joe would. The validator checks that required phrases are present; it cannot check that the prose is true. Confirm specifically:

- the AIC-penalty paragraph is the first thing after the title, not buried;
- every number in the variance table matches `data/generated/nb2_stepwise_variance.csv`;
- the COVID row for violations either matches CLAUDE.md's recorded NB1 sensitivity or visibly differs from it.

- [ ] **Step 7: Update CLAUDE.md**

Add a section documenting this arm: the run order, the six models, the NB2-as-editorial-choice caveat, the two reduction bases, and the retired-vs-replaced cross-check. If Step 6 showed the violations COVID sensitivity differs under NB2 from the NB1 result CLAUDE.md currently records, correct that note and say NB2 is the source of the new number — do not delete the NB1 finding, which remains true of the NB1 fit.

- [ ] **Step 8: Commit**

```bash
git add scripts/report_nb2_stepwise.py scripts/validate_nb2_stepwise.py \
        data/generated/nb2_stepwise_variance.csv \
        data/generated/nb2_stepwise_coefficients.csv \
        docs/nb2_stepwise_models.md CLAUDE.md
git commit -m "Report the NB2 stepwise tables and memo

Variance components with two clearly separated bases: the random-slope fits
supply the coefficients and sigma^2_u1, the random-intercept series supplies
theta, ICC and both reduction percentages.

Two reduction columns because Model 1 has 49 states and Models 2-3 have 46;
neither number alone separates the covariates from the lost states.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 covariate blocks reused, not redefined | 1 (Global Constraints + `M2_ADD`/`M3_ADD` in the R script) |
| §2.1 two cells, 2021 window only | 1 (`CELLS`), 5 (memo caveat) |
| §2.2 `nbinom2` only | 1 (`fit_nb2` hardcodes `family = nbinom2`, `ziformula = ~0`) |
| §2.3 `(1 + time | state)`, no fallback | 1 (no fallback code path; validator `[2]` asserts `re_tier == 'rs'`) |
| §2.4 Δσ²_u0 from an RI series | 2 (fits), 5 (arithmetic) |
| §2.5 standalone arm, ladder frozen | 5 (validator `[8]`) |
| §3 NB2 override stated, never a fit-based win | 5 (memo first section; validator `[9]` forbidden patterns) |
| §4.1 panel and cells | 1 (validator `[1]`) |
| §4.2 fixed-effect build-up | 1 (validator `[2]` term-set checks) |
| §4.3 ML, four existing fits reproduce | 1 (validator `[3]`) |
| §4.4 variance table, ICC, `sigma2_e` null, two Δ columns | 2, 5 (validator `[7]`) |
| §5.1 Gaussian gate run, not redesigned | 1 (validator `[0]`) |
| §5.2 ZI cross-check retired, NB2 analogue added | 4 |
| §5.3 COVID re-run under NB2 | 3 |
| §6.1–6.7 validation | validator sections `[0]`–`[9]` |
| §7 artifacts, run order, forbidden patterns | 1–5, `[9]` |
| §8 M2 convergence risk surfaced | 1 (Step 4 stop condition) |

No spec requirement is unassigned.

**Placeholder scan:** no `TBD`, no "add appropriate error handling", no "similar to Task N". Every code step carries the actual code.

**Type consistency:** `fit_nb2(d, dv, rhs, re, sample_rhs = NULL)` is defined in Task 1 and called with `sample_rhs` only in Task 2. `rhs_for(cl, model)` is defined in Task 1 and used in Tasks 2–3. The JSON field names in Task 1's Interfaces block (`sigma2_u0`, `sigma2_u1`, `sigma_u01`, `dispersion`, `mu_fixed`, `cond`, `re_tier`, `n_obs`, `n_states`) are the names read in Tasks 2, 3, 4 and 5 and asserted in validator sections `[2]`, `[4]`, `[5]`, `[7]`. The CSV columns listed in Task 5's Interfaces block match `variance_table()`'s dict keys and validator `[7]`'s lookups. `check_close(label, got, want, tol, kind)` has one signature, used identically in `[3]` and `[7]`.

One known asymmetry, deliberate: `sigma2_e` is written by R as `NA_real_` and serialised to JSON `null` by `write_json(..., na = "null")`, so Python reads `None` — which is what validator `[2]` asserts.
