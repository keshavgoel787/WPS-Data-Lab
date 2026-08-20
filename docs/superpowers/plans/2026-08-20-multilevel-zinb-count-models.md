# Multilevel ZINB Count Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit multilevel zero-inflated negative binomial models (Jafari et al. 2024's model class) to the WPS violations and inspections panels on both data windows, and report a formal AIC/BIC/LRT comparison against the existing `log(count+1)` Gaussian LMM so the paper's primary specification is chosen on evidence.

**Architecture:** Python builds the analytic panels (reusing the covariate construction already in `paper_table_models_2021.py` / `paper_table_models_corrected.py` so nothing drifts) and writes tidy CSVs with raw unlogged counts. An R script fits the model ladder in `glmmTMB` — the package Jafari used — and dumps every fit to JSON. Python reads that JSON back to print tables, write the selection comparison CSV, and generate a memo. A Gaussian round-trip against `statsmodels.MixedLM` gates everything: no count model is trusted until a Gaussian `glmmTMB` fit reproduces the published log-LMM coefficients.

**Tech Stack:** Python 3 (pandas 2.2.2, numpy 1.26.4, statsmodels 0.14.6), R 4.5.2 (`glmmTMB` 1.1.14, `TMB` 1.9.23, `jsonlite` 2.0.0)

**Spec:** `docs/superpowers/specs/2026-08-20-multilevel-zinb-design.md`

## Global Constraints

- Run every script from `/Users/keshavgoel/Research/`. All paths in this repo are absolute and hardcoded to `data/`, `figures/`, `docs/`.
- `RAW = '/Users/keshavgoel/Research/data/raw/'`, `GEN = '/Users/keshavgoel/Research/data/generated/'`.
- Time variable is always `time = year - 2017`, with `time2 = time**2` and `time3 = time**3`.
- Random-effects structure is `(1 + time | state)` — random intercept plus random slope on linear time by state. Never Jafari's crossed `(1|state) + (1|year)`.
- Level-2 covariates are z-scored before entry, named `varname_z`.
- `STATE_NAME_MAPPING = {'Massachusetts ': 'Massachusetts', 'Oregon ': 'Oregon'}` — trailing-space bug in the ECHO index.
- `US_STATES_50` is the canonical 50-state filter; territories, tribes and regions are excluded.
- This work happens on branch `zinb-count-models`. Do **not** modify `docs/WPS_Table_Sheels_*.docx`, `figures/*`, or any existing `paper_table_*` script. Regenerating manuscript artifacts is explicitly out of scope.
- Fixed-effect build-up, unchanged from the published tables:
  - M1: `time + time2 + time3` (+ exposure, violations only)
  - M2: M1 + `SPEND_APP_z + SPEND_WORK_z + lii_2017_z`
  - M3: M2 + `h2a_per_farmworker_z + dol_demand_met_pct_z + pct_flc_z`
- Any model that does not converge cleanly is recorded as non-convergent and its estimates are **not** reported.

---

### Task 1: R environment and `glmmTMB` smoke test

Installs the R dependencies and proves `glmmTMB` can fit a zero-inflated negative binomial with a random slope at all, before any project data is involved. If this task fails, the whole approach is dead and we find out in five minutes instead of after building a data pipeline.

**Files:**
- Create: `scripts/r_env_check.R`

**Interfaces:**
- Consumes: nothing.
- Produces: a working R library containing `glmmTMB`, `TMB`, `jsonlite`. Later R tasks assume `library(glmmTMB)` and `library(jsonlite)` succeed.

- [ ] **Step 1: Write the failing check**

Create `scripts/r_env_check.R`:

```r
# Environment gate for the ZINB count-model pipeline.
# Proves glmmTMB is installed and can fit a ZINB with a random slope.
# Run: Rscript scripts/r_env_check.R    (exits 1 on any failure)

ok <- TRUE
fail <- function(msg) { cat("FAIL:", msg, "\n"); ok <<- FALSE }

for (p in c("glmmTMB", "TMB", "jsonlite")) {
  if (!requireNamespace(p, quietly = TRUE)) fail(sprintf("package '%s' not installed", p))
}
if (!ok) { cat("\nInstall with:\n  Rscript -e 'install.packages(c(\"glmmTMB\",\"jsonlite\"), repos=\"https://cloud.r-project.org\")'\n"); quit(status = 1) }

cat("glmmTMB", as.character(packageVersion("glmmTMB")), "/ TMB",
    as.character(packageVersion("TMB")), "/ jsonlite",
    as.character(packageVersion("jsonlite")), "\n")

# Synthetic panel: 40 groups x 10 periods, overdispersed counts with ~20% structural zeros.
set.seed(20260820)
n_g <- 40; n_t <- 10
d <- expand.grid(t = seq_len(n_t), g = factor(seq_len(n_g)))
d$time <- d$t - 5
u0 <- rnorm(n_g, 0, 0.8)[as.integer(d$g)]
u1 <- rnorm(n_g, 0, 0.1)[as.integer(d$g)]
mu <- exp(2 + 0.05 * d$time + u0 + u1 * d$time)
d$y <- rnbinom(nrow(d), mu = mu, size = 1.5) * rbinom(nrow(d), 1, 0.8)

m <- try(glmmTMB::glmmTMB(y ~ time + (1 + time | g), ziformula = ~1,
                          family = glmmTMB::nbinom2, data = d), silent = TRUE)
if (inherits(m, "try-error")) fail(paste("ZINB fit errored:", attr(m, "condition")$message))
if (ok && !isTRUE(m$sdr$pdHess)) fail("ZINB fit did not produce a positive-definite Hessian")
if (ok && m$fit$convergence != 0) fail(sprintf("ZINB convergence code %d", m$fit$convergence))
if (ok) {
  zi_prob <- plogis(glmmTMB::fixef(m)$zi[["(Intercept)"]])
  cat(sprintf("ZINB converged; estimated zero-inflation prob = %.3f (data generated at 0.200)\n", zi_prob))
  if (abs(zi_prob - 0.20) > 0.15) fail(sprintf("zero-inflation estimate %.3f implausibly far from 0.200", zi_prob))
}

if (ok) cat("\nPASS: R environment ready\n") else quit(status = 1)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `Rscript scripts/r_env_check.R`
Expected: FAIL — `package 'glmmTMB' not installed`, `package 'jsonlite' not installed`, exit status 1, and the printed install command.

- [ ] **Step 3: Install the dependencies**

The R library at `/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library` is writable, so no `sudo` is needed. CRAN serves prebuilt arm64 binaries for all three, so there is no compile step.

```bash
Rscript -e 'install.packages(c("glmmTMB","jsonlite"), repos="https://cloud.r-project.org")'
```

- [ ] **Step 4: Run it to verify it passes**

Run: `Rscript scripts/r_env_check.R`
Expected: PASS. Prints package versions, a converged ZINB, and a zero-inflation estimate near 0.20.

If the fit does not converge, do not proceed — report it. A synthetic 400-observation panel that `glmmTMB` cannot fit means our 533-observation panel will not fit either, and the spec's fallback ladder needs revisiting before writing more code.

- [ ] **Step 5: Commit**

```bash
git add scripts/r_env_check.R
git commit -m "Add R environment gate for glmmTMB count models"
```

---

### Task 2: Analytic panel builder

Builds both analytic frames with raw unlogged counts. This is the shared data bridge; every later task reads its output.

**Files:**
- Create: `scripts/build_count_model_panel.py`
- Create: `data/generated/count_model_panel_2019.csv` (generated)
- Create: `data/generated/count_model_panel_2021.csv` (generated)

**Interfaces:**
- Consumes: `data/raw/establishments_data.csv`, `data/raw/wps_data.csv` via `data/generated/wps_dv_panel_2011_2021.csv`, `data/generated/spend_bls_variables_multiyear.csv`, `data/generated/spend_bls_variables_multiyear_2021.csv`, `data/generated/h2a_ratio_panel.csv`, `data/generated/h2a_ratio_panel_2011_2021.csv`, `data/raw/labor_intensity_index_2017.csv`, `data/raw/dol_var1_workers_by_state_annual.csv`, `data/raw/dol_var2_employer_type_annual.csv`.
- Produces: `build_panel(end_year: int) -> pandas.DataFrame` where `end_year` is `2019` (establishments view) or `2021` (WPS view). Columns, in order: `state` (str), `year` (int), `time`, `time2`, `time3` (int), `violations`, `inspections` (float, raw counts), `log_violations`, `log_inspections` (float), `covid` (float, `2021` panel only), `SPEND_APP_z`, `SPEND_WORK_z`, `lii_2017_z`, `dol_demand_met_pct_z`, `pct_flc_z`, `h2a_per_farmworker_z` (float).

- [ ] **Step 1: Write the failing check**

Create `scripts/validate_count_models.py` with the panel checks only (later tasks append to this file):

```python
"""
Validation gate for the ZINB count-model pipeline.

This is not a unit-test suite -- it is an analysis artifact. Each check either
proves a step of the pipeline reproduces something already known, or proves an
invariant the spec depends on. Run it after any change to the pipeline.

Run: python3 scripts/validate_count_models.py      (exits 1 on any failure)
"""
import sys

import numpy as np
import pandas as pd

GEN = '/Users/keshavgoel/Research/data/generated/'

FAILURES = []


def check(label, condition, detail=''):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f" -- {detail}" if detail else ''))
        FAILURES.append(label)


def check_close(label, got, want, tol, kind='abs'):
    delta = abs(got - want) if kind == 'abs' else abs(got - want) / abs(want)
    check(f"{label} ({kind} delta {delta:.2e} <= {tol:.0e})", delta <= tol,
          f"got {got!r}, want {want!r}")


# ============================================================
# [1] PANEL INVARIANTS
# ============================================================
def validate_panels():
    print("\n[1] Analytic panels")
    from build_count_model_panel import build_panel

    p21 = build_panel(2021)
    check("2021 panel has 539 rows", len(p21) == 539, f"got {len(p21)}")
    check("2021 panel has 49 states (Wyoming absent from the WPS view)",
          p21['state'].nunique() == 49, f"got {p21['state'].nunique()}")
    check("2021 panel spans 2011-2021",
          (p21['year'].min(), p21['year'].max()) == (2011, 2021))
    check("2021 violations: 533 non-null, 93 zeros",
          (p21['violations'].notna().sum(), (p21['violations'] == 0).sum()) == (533, 93),
          f"got {p21['violations'].notna().sum()} non-null, {(p21['violations'] == 0).sum()} zeros")
    check("2021 inspections: 539 non-null, 7 zeros",
          (p21['inspections'].notna().sum(), (p21['inspections'] == 0).sum()) == (539, 7),
          f"got {p21['inspections'].notna().sum()} non-null, {(p21['inspections'] == 0).sum()} zeros")
    check("2021 covid flags exactly 2020-21",
          set(p21.loc[p21['covid'] == 1, 'year']) == {2020, 2021})

    p19 = build_panel(2019)
    check("2019 panel has 450 rows (50 states x 9 years)", len(p19) == 450, f"got {len(p19)}")
    check("2019 panel has 50 states", p19['state'].nunique() == 50, f"got {p19['state'].nunique()}")
    check("2019 panel spans 2011-2019",
          (p19['year'].min(), p19['year'].max()) == (2011, 2019))
    check("2019 panel has no covid column", 'covid' not in p19.columns)

    for name, p in (('2021', p21), ('2019', p19)):
        for dv in ('violations', 'inspections'):
            raw, logged = p[dv], p[f'log_{dv}']
            m = raw.notna()
            check(f"{name} log_{dv} == log1p({dv})",
                  np.allclose(logged[m], np.log1p(raw[m])))
        check(f"{name} time == year - 2017", (p['time'] == p['year'] - 2017).all())
        check(f"{name} time2/time3 are consistent powers of time",
              (p['time2'] == p['time'] ** 2).all() and (p['time3'] == p['time'] ** 3).all())
        check(f"{name} counts are non-negative whole numbers",
              bool(((p[['violations', 'inspections']].dropna() % 1) == 0).all().all())
              and bool((p[['violations', 'inspections']].dropna() >= 0).all().all()))
        for z in ('SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
                  'dol_demand_met_pct_z', 'pct_flc_z', 'h2a_per_farmworker_z'):
            check(f"{name} {z} present", z in p.columns)


def main():
    validate_panels()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == '__main__':
    sys.path.insert(0, '/Users/keshavgoel/Research/scripts')
    main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 scripts/validate_count_models.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_count_model_panel'`.

- [ ] **Step 3: Write the panel builder**

Create `scripts/build_count_model_panel.py`. The covariate construction is lifted from `paper_table_models_2021.py` lines 88–118 and `paper_table_models_corrected.py` lines 63–95 — read both before writing, and keep the logic identical. The only intentional difference is that raw counts are retained.

```python
"""
Analytic panels for the ZINB count models (spec: 2026-08-20-multilevel-zinb-design.md).

Two windows, because the results Joe reacted to changed TWO things at once --
the window (2019 -> 2021) and the outcome source (ECHO establishments view ->
WPS view). Refitting both under one model class is what separates the
model-class effect from the data-source effect.

    end_year=2021 -> WPS view,            2011-2021, 49 states (Wyoming absent)
    end_year=2019 -> establishments view, 2011-2019, 50 states

Unlike the paper-table scripts this keeps the RAW counts -- that is the whole
point of a count model. The log(count+1) columns ride along so the same file
supports the Gaussian round-trip validation.

Covariate construction is IDENTICAL to paper_table_models_{2021,corrected}.py.
Any drift would make the count-vs-log comparison meaningless.

Run: python3 scripts/build_count_model_panel.py
Output: data/generated/count_model_panel_{2019,2021}.csv
"""

import numpy as np
import pandas as pd

RAW = '/Users/keshavgoel/Research/data/raw/'
GEN = '/Users/keshavgoel/Research/data/generated/'

STATE_ABBREV_TO_NAME = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska',
    'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina',
    'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
    'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'}
US_STATES_50 = sorted(STATE_ABBREV_TO_NAME.values())
STATE_NAME_MAPPING = {'Massachusetts ': 'Massachusetts', 'Oregon ': 'Oregon'}

Z_COVARS = ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
            'dol_demand_met_pct_z', 'pct_flc_z', 'h2a_per_farmworker_z']


def _dv_frame_2021():
    """WPS view of the ECHO State Pesticide Dashboard, 2011-2021."""
    long = pd.read_csv(GEN + 'wps_dv_panel_2011_2021.csv')
    long = long[long['state'].isin(US_STATES_50)].copy()
    return long.rename(columns={'wps_violations': 'violations',
                                'wps_inspections': 'inspections'})


def _dv_frame_2019():
    """Establishments view of the same dashboard, which EPA publishes only to 2019."""
    echo = pd.read_csv(RAW + 'establishments_data.csv', index_col=0)
    echo.index = echo.index.str.strip().map(lambda x: STATE_NAME_MAPPING.get(x, x))
    rows = []
    for year in range(2011, 2020):
        rows.append(pd.DataFrame({
            'state': echo.index, 'year': year,
            'violations': pd.to_numeric(echo[f'violations-{year}'], errors='coerce').values,
            'inspections': (pd.to_numeric(echo[f'inspections-epa-{year}'], errors='coerce').fillna(0).values
                            + pd.to_numeric(echo[f'inspections-state-{year}'], errors='coerce').fillna(0).values),
        }))
    long = pd.concat(rows, ignore_index=True)
    return long[long['state'].isin(US_STATES_50)].copy()


def _level2(end_year, spend_file):
    """State-level (Level-2) covariates, z-scored. Mirrors the paper-table scripts."""
    level2 = pd.DataFrame({'state': US_STATES_50})

    spend = pd.read_csv(GEN + spend_file)
    level2 = level2.merge(spend[['state', 'SPEND_APP_z', 'SPEND_WORK_z']], on='state', how='left')

    li17 = pd.read_csv(RAW + 'labor_intensity_index_2017.csv')
    li17['state'] = li17['state_name'].str.title()
    level2 = level2.merge(li17[['state', 'Labor_Intensity_Index']].rename(
        columns={'Labor_Intensity_Index': 'lii_2017'}), on='state', how='left')

    # DOL demand-met %: study-window mean, excluding 2013 (DOL never published it).
    dol1 = pd.read_csv(RAW + 'dol_var1_workers_by_state_annual.csv')
    dol_study = dol1[dol1['year'].between(2011, end_year) & (dol1['year'] != 2013)]
    dm = dol_study.groupby('state')['demand_met_pct'].apply(
        lambda x: x[x != np.inf].mean()).reset_index()
    dm['state'] = dm['state'].map(STATE_ABBREV_TO_NAME)
    level2 = level2.merge(dm.rename(columns={'demand_met_pct': 'dol_demand_met_pct'}),
                          on='state', how='left')

    # DOL % Farm Labor Contractor: dol_var2 starts in 2020, so 2020 is the only
    # available value -- a backward proxy for the 2019 window, in-window for 2021.
    dol2 = pd.read_csv(RAW + 'dol_var2_employer_type_annual.csv')
    d2020 = dol2[dol2['year'] == 2020].copy()
    d2020['state'] = d2020['state'].map(STATE_ABBREV_TO_NAME)
    d2020 = d2020.rename(columns={'pct_Farm Labor Contractor': 'pct_flc'})
    level2 = level2.merge(d2020[['state', 'pct_flc']], on='state', how='left')

    for col in ['lii_2017', 'dol_demand_met_pct', 'pct_flc']:
        level2[col + '_z'] = (level2[col] - level2[col].mean()) / level2[col].std()

    return level2[['state', 'SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
                   'dol_demand_met_pct_z', 'pct_flc_z']]


def build_panel(end_year):
    """Assemble the analytic panel for one window. end_year is 2019 or 2021."""
    if end_year == 2021:
        long = _dv_frame_2021()
        spend_file, h2a_file = 'spend_bls_variables_multiyear_2021.csv', 'h2a_ratio_panel_2011_2021.csv'
    elif end_year == 2019:
        long = _dv_frame_2019()
        spend_file, h2a_file = 'spend_bls_variables_multiyear.csv', 'h2a_ratio_panel.csv'
    else:
        raise ValueError(f"end_year must be 2019 or 2021, got {end_year!r}")

    long['time'] = long['year'] - 2017
    long['time2'] = long['time'] ** 2
    long['time3'] = long['time'] ** 3
    long['log_violations'] = np.log1p(long['violations'])
    long['log_inspections'] = np.log1p(long['inspections'])

    cols = ['state', 'year', 'time', 'time2', 'time3',
            'violations', 'inspections', 'log_violations', 'log_inspections']
    if end_year == 2021:
        long['covid'] = (long['year'] >= 2020).astype(float)
        cols.append('covid')

    df = long[cols].merge(_level2(end_year, spend_file), on='state', how='left')

    h2a = pd.read_csv(GEN + h2a_file)
    df = df.merge(h2a[['state', 'year', 'h2a_per_farmworker_z']],
                  on=['state', 'year'], how='left')

    return df.sort_values(['state', 'year']).reset_index(drop=True)


if __name__ == '__main__':
    for end_year in (2019, 2021):
        panel = build_panel(end_year)
        out = GEN + f'count_model_panel_{end_year}.csv'
        panel.to_csv(out, index=False)
        n_z = (panel['violations'] == 0).sum()
        print(f"{out}\n  {len(panel)} rows, {panel['state'].nunique()} states, "
              f"{panel['year'].min()}-{panel['year'].max()}, "
              f"{n_z} zero-violation state-years ({n_z / len(panel):.1%})")
        missing = [c for c in Z_COVARS if panel[c].isna().any()]
        for c in missing:
            drops = sorted(panel.loc[panel[c].isna(), 'state'].unique())
            print(f"  {c}: missing for {len(drops)} state(s) -> {', '.join(drops)}")
```

- [ ] **Step 4: Run the validation to verify it passes**

Run: `python3 scripts/validate_count_models.py`
Expected: all `[1] Analytic panels` checks PASS, then `ALL CHECKS PASSED`.

If the 2019 row count is not 450, print `build_panel(2019)['state'].value_counts()` and reconcile against `paper_table_models_corrected.py` before adjusting the expected value — a mismatch means the reshape diverged from the published table, which is exactly what this check exists to catch.

- [ ] **Step 5: Generate the panel CSVs**

Run: `python3 scripts/build_count_model_panel.py`
Expected: writes both CSVs and reports, for the 2021 panel, 93 zero-violation state-years (17.3%), and missing `SPEND_APP_z` for Alaska, Rhode Island, Vermont (BLS never publishes applicator employment for those three).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_count_model_panel.py scripts/validate_count_models.py \
        data/generated/count_model_panel_2019.csv data/generated/count_model_panel_2021.csv
git commit -m "Build raw-count analytic panels for both windows"
```

---

### Task 3: Gaussian round-trip gate

The critical validation. A Gaussian `glmmTMB` fit with the same random-effects structure must reproduce the published `statsmodels.MixedLM` coefficients. If it does, the CSV bridge, the analytic sample, the random-effects structure, and the R script are all proven correct in one shot. If it does not, nothing downstream is trustworthy and no count model gets fit.

Note `glmmTMB` defaults to ML while `MixedLM` uses REML, so this comparison passes `REML = TRUE`.

**Files:**
- Create: `scripts/count_models_zinb.R`
- Modify: `scripts/validate_count_models.py` (add section `[2]`, and call it from `main()`)

**Interfaces:**
- Consumes: `data/generated/count_model_panel_2021.csv` from Task 2.
- Produces:
  - CLI `Rscript scripts/count_models_zinb.R <panel_csv> <out_json> [--gaussian-only]`
  - R function `fit_spec(d, dv, rhs, family, zi = ~0, offset_col = NULL, re = "(1 + time | state)", REML = FALSE) -> list` with keys `converged` (logical), `message` (chr), `n_obs` (int), `n_states` (int), `aic`, `bic`, `loglik`, `df` (num), `cond` (list of per-term `list(b, se, z, p)`), `zi` (same shape, empty when `zi = ~0`), `sigma2_u0`, `sigma2_u1`, `sigma_u01`, `sigma2_e` (num), `obs_zeros`, `exp_zeros` (num).
  - JSON top level: `{"<cell>": {"<model>": <fit_spec result>}}`.

- [ ] **Step 1: Write the failing check**

Append to `scripts/validate_count_models.py`, above `def main():`:

```python
# ============================================================
# [2] GAUSSIAN ROUND-TRIP: glmmTMB must reproduce statsmodels.MixedLM
# ============================================================
# Reference values captured 2026-08-20 from paper_table_models_2021.py, whose
# output is what the current manuscript tables report. Tolerances are loose
# enough for optimizer differences (lbfgs vs TMB) and tight enough that a real
# specification difference -- wrong sample, wrong RE structure, ML instead of
# REML -- cannot slip through.
GAUSSIAN_REFERENCE = {
    'insp_M1_gaussian': {
        'n_obs': 539, 'n_states': 49,
        'cond': {'time': -0.063369, 'time2': -0.001913, 'time3': 0.000092},
        'se': {'time': 0.021949, 'time2': 0.004265, 'time3': 0.001060},
        'sigma2_u0': 1.118142, 'sigma2_e': 0.339958,
    },
    'viol_M1_gaussian': {
        'n_obs': 533, 'n_states': 49,
        'cond': {'log_inspections': 0.495439, 'time': 0.062459,
                 'time2': -0.017279, 'time3': -0.005445},
        'se': {'log_inspections': 0.051653, 'time': 0.034246,
               'time2': 0.005314, 'time3': 0.001347},
        'sigma2_u0': 2.443787, 'sigma2_e': 0.527532,
    },
}


def validate_gaussian_roundtrip():
    import json
    import subprocess

    print("\n[2] Gaussian round-trip (glmmTMB REML vs statsmodels MixedLM)")
    out = '/private/tmp/claude-501/gaussian_roundtrip.json'
    proc = subprocess.run(
        ['Rscript', '/Users/keshavgoel/Research/scripts/count_models_zinb.R',
         GEN + 'count_model_panel_2021.csv', out, '--gaussian-only'],
        capture_output=True, text=True)
    if proc.returncode != 0:
        check("R script ran", False, proc.stderr.strip()[-500:])
        return
    check("R script ran", True)

    fits = json.load(open(out))
    for cell, ref in GAUSSIAN_REFERENCE.items():
        fit = fits.get(cell)
        if fit is None:
            check(f"{cell} present in output", False, f"keys: {sorted(fits)}")
            continue
        check(f"{cell} converged", fit['converged'] is True, fit.get('message', ''))
        check(f"{cell} n_obs == {ref['n_obs']}", fit['n_obs'] == ref['n_obs'], f"got {fit['n_obs']}")
        check(f"{cell} n_states == {ref['n_states']}", fit['n_states'] == ref['n_states'],
              f"got {fit['n_states']}")
        for term, want in ref['cond'].items():
            got = fit['cond'].get(term, {}).get('b')
            if got is None:
                check(f"{cell} {term} estimated", False, f"terms: {sorted(fit['cond'])}")
                continue
            check_close(f"{cell} b[{term}]", got, want, 1e-3)
            check_close(f"{cell} se[{term}]", fit['cond'][term]['se'], ref['se'][term], 5e-3)
        # Variance components come from a different optimizer path, so relative.
        check_close(f"{cell} sigma2_u0", fit['sigma2_u0'], ref['sigma2_u0'], 2e-2, kind='rel')
        check_close(f"{cell} sigma2_e", fit['sigma2_e'], ref['sigma2_e'], 2e-2, kind='rel')
```

And change `main()` to:

```python
def main():
    validate_panels()
    validate_gaussian_roundtrip()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 scripts/validate_count_models.py`
Expected: section `[1]` passes; section `[2]` FAILs on `R script ran` because `scripts/count_models_zinb.R` does not exist.

- [ ] **Step 3: Write the R fitting engine**

Create `scripts/count_models_zinb.R`:

```r
# Multilevel count models for the WPS panels, in the package Jafari et al. (2024)
# used. See docs/superpowers/specs/2026-08-20-multilevel-zinb-design.md.
#
# Usage:
#   Rscript scripts/count_models_zinb.R <panel_csv> <out_json> [--gaussian-only]
#
# --gaussian-only fits ONLY the Gaussian round-trip models, whose coefficients
# must reproduce statsmodels.MixedLM. That is the validation gate; without it
# no count estimate from this script should be believed.

suppressPackageStartupMessages({
  library(glmmTMB)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: count_models_zinb.R <panel_csv> <out_json> [--gaussian-only]")
panel_csv <- args[1]
out_json <- args[2]
gaussian_only <- "--gaussian-only" %in% args

SEED <- 20260820L
N_SIM <- 200L

panel <- read.csv(panel_csv, stringsAsFactors = FALSE)

# ------------------------------------------------------------
# fit_spec: fit one specification and flatten it to a plain list
# ------------------------------------------------------------
coef_list <- function(mat) {
  if (is.null(mat) || nrow(mat) == 0) return(structure(list(), names = character(0)))
  stats::setNames(lapply(seq_len(nrow(mat)), function(i) {
    list(b = mat[i, 1], se = mat[i, 2], z = mat[i, 3], p = mat[i, 4])
  }), rownames(mat))
}

fit_spec <- function(d, dv, rhs, family, zi = ~0, offset_col = NULL,
                     re = "(1 + time | state)", REML = FALSE) {
  need <- unique(c(dv, rhs, "state", "time", offset_col))
  need <- need[need %in% names(d)]
  dd <- d[stats::complete.cases(d[, need, drop = FALSE]), , drop = FALSE]
  # An offset of log(inspections) is undefined at zero, so those rows leave the
  # offset specification. The covariate specification keeps them.
  if (!is.null(offset_col)) dd <- dd[dd[[offset_col]] > 0, , drop = FALSE]
  dd$state <- factor(dd$state)

  off <- if (is.null(offset_col)) "" else sprintf(" + offset(log(%s))", offset_col)
  form <- stats::as.formula(sprintf("%s ~ %s%s + %s", dv,
                                    paste(rhs, collapse = " + "), off, re))

  fit <- tryCatch(
    withCallingHandlers(
      glmmTMB(form, ziformula = zi, family = family, data = dd, REML = REML),
      warning = function(w) invokeRestart("muffleWarning")),
    error = function(e) e)

  base <- list(formula = deparse1(form), zi_formula = deparse1(zi),
               n_obs = nrow(dd), n_states = length(unique(dd$state)))

  if (inherits(fit, "error")) {
    return(c(base, list(converged = FALSE, message = conditionMessage(fit))))
  }

  pd_hess <- isTRUE(fit$sdr$pdHess)
  conv_code <- fit$fit$convergence
  sm <- summary(fit)
  cond <- coef_list(sm$coefficients$cond)
  zi_co <- coef_list(sm$coefficients$zi)
  finite_se <- length(cond) == 0 ||
    all(vapply(cond, function(x) is.finite(x$se), logical(1)))
  converged <- pd_hess && conv_code == 0 && finite_se

  vc <- tryCatch(VarCorr(fit)$cond$state, error = function(e) NULL)
  s2_u0 <- if (is.null(vc)) NA_real_ else vc[1, 1]
  s2_u1 <- if (is.null(vc) || nrow(vc) < 2) NA_real_ else vc[2, 2]
  s_u01 <- if (is.null(vc) || nrow(vc) < 2) NA_real_ else vc[1, 2]
  # For Gaussian, sigma() is the residual SD; for count families it is the
  # dispersion parameter. Squared here so the Gaussian case is comparable to
  # MixedLM's `scale`.
  s2_e <- tryCatch(sigma(fit)^2, error = function(e) NA_real_)

  # Observed vs expected zeros -- the direct evidence for zero-inflation.
  # Skipped for Gaussian, where "zero" is not a meaningful outcome.
  obs_zeros <- exp_zeros <- NA_real_
  if (!identical(family, gaussian) && !identical(family, "gaussian")) {
    obs_zeros <- sum(dd[[dv]] == 0, na.rm = TRUE)
    sims <- tryCatch(as.data.frame(simulate(fit, nsim = N_SIM, seed = SEED)),
                     error = function(e) NULL)
    if (!is.null(sims)) exp_zeros <- mean(colSums(sims == 0))
  }

  c(base, list(converged = converged, message = "",
               pd_hess = pd_hess, conv_code = conv_code,
               aic = AIC(fit), bic = BIC(fit),
               loglik = as.numeric(logLik(fit)), df = attr(logLik(fit), "df"),
               cond = cond, zi = zi_co,
               sigma2_u0 = s2_u0, sigma2_u1 = s2_u1, sigma_u01 = s_u01,
               sigma2_e = s2_e,
               obs_zeros = obs_zeros, exp_zeros = exp_zeros))
}

CUBIC <- c("time", "time2", "time3")

results <- list()

# ------------------------------------------------------------
# Gaussian round-trip gate. Must reproduce paper_table_models_2021.py.
# REML = TRUE because statsmodels MixedLM uses REML and glmmTMB defaults to ML.
# ------------------------------------------------------------
results$insp_M1_gaussian <- fit_spec(
  panel, "log_inspections", CUBIC, family = gaussian, REML = TRUE)
results$viol_M1_gaussian <- fit_spec(
  panel, "log_violations", c("log_inspections", CUBIC), family = gaussian, REML = TRUE)

if (gaussian_only) {
  write_json(results, out_json, auto_unbox = TRUE, digits = 10, na = "null")
  cat("Gaussian round-trip written to", out_json, "\n")
  quit(status = 0)
}

stop("count ladder not implemented yet -- Task 4")
```

- [ ] **Step 4: Run the validation to verify it passes**

Run: `python3 scripts/validate_count_models.py`
Expected: sections `[1]` and `[2]` all PASS, then `ALL CHECKS PASSED`.

**This is the gate. Do not continue to Task 4 until it passes.** If coefficients are close but not within tolerance, check in this order: (a) is `REML = TRUE` actually set; (b) does `n_obs` match — a mismatch means the analytic sample differs, not the estimator; (c) is the random-effects term `(1 + time | state)` rather than `(1 | state)`. If `n_obs` and the RE structure are right and coefficients still differ in the third decimal, report the discrepancy rather than loosening the tolerance.

- [ ] **Step 5: Commit**

```bash
git add scripts/count_models_zinb.R scripts/validate_count_models.py
git commit -m "Add glmmTMB engine with Gaussian round-trip validation gate"
```

---

### Task 4: The count-model ladder

Fits the six-family ladder across all six cells with the spec's fallback ladder for non-convergence.

**Files:**
- Modify: `scripts/count_models_zinb.R` (replace the `stop(...)` line from Task 3)
- Create: `data/generated/count_model_results.json` (generated)

**Interfaces:**
- Consumes: `fit_spec()` from Task 3.
- Produces: `data/generated/count_model_results.json`, keyed `"<cell>__<model>__<family_tag>"` where cell is one of `insp_2021`, `insp_2019`, `viol_off_2021`, `viol_off_2019`, `viol_cov_2021`, `viol_cov_2019`; model is `M1`/`M2`/`M3`; and family tag is `poisson`, `nbinom1`, `nbinom2`, `zip`, `zinb`, `zinb_re`. Each value is a `fit_spec` result plus `cell`, `model`, `family_tag`, `re_used` (chr), `window` (int), `outcome` (chr), `exposure` (chr).

- [ ] **Step 1: Write the failing check**

Append to `scripts/validate_count_models.py`, above `def main():`:

```python
# ============================================================
# [3] LADDER STRUCTURE AND CONVERGENCE
# ============================================================
CELLS = ['insp_2021', 'insp_2019', 'viol_off_2021', 'viol_off_2019',
         'viol_cov_2021', 'viol_cov_2019']
FAMILY_TAGS = ['poisson', 'nbinom1', 'nbinom2', 'zip', 'zinb', 'zinb_re']


def validate_ladder():
    import json
    import os

    print("\n[3] Count-model ladder")
    path = GEN + 'count_model_results.json'
    if not os.path.exists(path):
        check("count_model_results.json exists", False,
              "run: Rscript scripts/count_models_zinb.R "
              "data/generated/count_model_panel_2021.csv "
              "data/generated/count_model_results.json")
        return
    fits = json.load(open(path))
    check("count_model_results.json exists", True)

    # Spec 5.3: full ladder at M3 and M1 (stability check), winner only at M2.
    for cell in CELLS:
        for model in ('M1', 'M3'):
            missing = [t for t in FAMILY_TAGS
                       if f'{cell}__{model}__{t}' not in fits]
            check(f"{cell} {model}: all 6 families present", not missing,
                  f"missing {missing}")
        m2 = [k for k in fits if k.startswith(f'{cell}__M2__')]
        check(f"{cell} M2: exactly one family fit", len(m2) == 1, f"got {m2}")

    # At least one NB-family model must converge per cell, or the cell is unusable.
    for cell in CELLS:
        nb = [k for k in fits if k.startswith(cell + '__')
              and k.rsplit('__', 1)[1] in ('nbinom1', 'nbinom2', 'zinb', 'zinb_re')
              and fits[k].get('converged')]
        check(f"{cell}: at least one NB-family model converged", bool(nb))

    # The offset specification must drop the 7 zero-inspection state-years.
    off = fits.get('viol_off_2021__M1__nbinom2')
    cov = fits.get('viol_cov_2021__M1__nbinom2')
    if off and cov:
        check("2021 offset spec drops the 7 zero-inspection state-years",
              cov['n_obs'] - off['n_obs'] == 7,
              f"covariate n={cov['n_obs']}, offset n={off['n_obs']}")

    # Poisson must fit worse than NB2 on these data -- variance is ~158x the mean.
    for cell in CELLS:
        p, nb2 = fits.get(f'{cell}__M3__poisson'), fits.get(f'{cell}__M3__nbinom2')
        if p and nb2 and p.get('converged') and nb2.get('converged'):
            check(f"{cell} M3: NB2 beats Poisson on AIC",
                  nb2['aic'] < p['aic'], f"poisson {p['aic']:.1f}, nb2 {nb2['aic']:.1f}")

    for k, f in fits.items():
        if f.get('converged'):
            check_close(f"{k}: simulated zeros are finite", float(f['exp_zeros']),
                        float(f['exp_zeros']), 0.0) if f['exp_zeros'] is not None else \
                check(f"{k}: expected zeros computed", False)
```

Add `validate_ladder()` to `main()` after `validate_gaussian_roundtrip()`.

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 scripts/validate_count_models.py`
Expected: sections `[1]` and `[2]` PASS; `[3]` FAILs on `count_model_results.json exists`.

- [ ] **Step 3: Implement the ladder**

In `scripts/count_models_zinb.R`, replace the final line `stop("count ladder not implemented yet -- Task 4")` with:

```r
# ------------------------------------------------------------
# The distribution ladder (spec 5.2)
# ------------------------------------------------------------
LADDER <- list(
  list(tag = "poisson",  family = poisson,  zi = ~0),
  list(tag = "nbinom1",  family = nbinom1,  zi = ~0),
  list(tag = "nbinom2",  family = nbinom2,  zi = ~0),
  list(tag = "zip",      family = poisson,  zi = ~1),
  list(tag = "zinb",     family = nbinom2,  zi = ~1),
  list(tag = "zinb_re",  family = nbinom2,  zi = ~ 1 + (1 | state))
)

# Fallback ladder (spec 6). Tried in order until one converges.
RE_FALLBACK <- c("(1 + time | state)", "(1 | state)")

fit_with_fallback <- function(d, dv, rhs, family, zi, offset_col) {
  last <- NULL
  for (re in RE_FALLBACK) {
    # zinb_re's ZI random intercept is the first thing to go: with 49 states and
    # 93 zeros it is the least identified part of the model.
    zi_try <- zi
    res <- fit_spec(d, dv, rhs, family, zi = zi_try, offset_col = offset_col, re = re)
    res$re_used <- re
    if (isTRUE(res$converged)) return(res)
    if (!identical(deparse1(zi_try), deparse1(~0)) &&
        grepl("state", deparse1(zi_try), fixed = TRUE)) {
      res2 <- fit_spec(d, dv, rhs, family, zi = ~1, offset_col = offset_col, re = re)
      res2$re_used <- re
      res2$zi_downgraded <- TRUE
      if (isTRUE(res2$converged)) return(res2)
      last <- res2
    } else {
      last <- res
    }
  }
  last
}

M2_ADD <- c("SPEND_APP_z", "SPEND_WORK_z", "lii_2017_z")
M3_ADD <- c(M2_ADD, "h2a_per_farmworker_z", "dol_demand_met_pct_z", "pct_flc_z")

# One cell = one outcome x window x exposure variant.
build_cells <- function() {
  cells <- list()
  for (yr in c(2021L, 2019L)) {
    csv <- sprintf("/Users/keshavgoel/Research/data/generated/count_model_panel_%d.csv", yr)
    d <- read.csv(csv, stringsAsFactors = FALSE)
    cells[[sprintf("insp_%d", yr)]] <- list(
      d = d, dv = "inspections", base = CUBIC, offset_col = NULL,
      window = yr, outcome = "inspections", exposure = "none")
    cells[[sprintf("viol_off_%d", yr)]] <- list(
      d = d, dv = "violations", base = CUBIC, offset_col = "inspections",
      window = yr, outcome = "violations", exposure = "offset")
    cells[[sprintf("viol_cov_%d", yr)]] <- list(
      d = d, dv = "violations", base = c("log_inspections", CUBIC), offset_col = NULL,
      window = yr, outcome = "violations", exposure = "covariate")
  }
  cells
}

cells <- build_cells()

for (cell_name in names(cells)) {
  cl <- cells[[cell_name]]
  specs <- list(M1 = cl$base, M3 = c(cl$base, M3_ADD))

  # Full ladder at M3 (the model a family must survive with every covariate
  # present) and at M1 (stability check -- a flipped winner gets reported).
  for (model in c("M1", "M3")) {
    for (rung in LADDER) {
      key <- sprintf("%s__%s__%s", cell_name, model, rung$tag)
      cat("fitting", key, "\n")
      res <- fit_with_fallback(cl$d, cl$dv, specs[[model]],
                               rung$family, rung$zi, cl$offset_col)
      res$cell <- cell_name; res$model <- model; res$family_tag <- rung$tag
      res$window <- cl$window; res$outcome <- cl$outcome; res$exposure <- cl$exposure
      results[[key]] <- res
    }
  }

  # M2 under the winning family only. Winner = lowest AIC among CONVERGED
  # models at M3; ties and non-convergence fall back to nbinom2.
  m3 <- results[grepl(sprintf("^%s__M3__", cell_name), names(results))]
  conv <- Filter(function(r) isTRUE(r$converged) && is.finite(r$aic), m3)
  win_tag <- if (length(conv) == 0) "nbinom2" else
    conv[[which.min(vapply(conv, function(r) r$aic, numeric(1)))]]$family_tag
  win <- Filter(function(r) r$tag == win_tag, LADDER)[[1]]
  key <- sprintf("%s__M2__%s", cell_name, win_tag)
  cat("fitting", key, "(winning family at M3)\n")
  res <- fit_with_fallback(cl$d, cl$dv, c(cl$base, M2_ADD),
                           win$family, win$zi, cl$offset_col)
  res$cell <- cell_name; res$model <- "M2"; res$family_tag <- win_tag
  res$window <- cl$window; res$outcome <- cl$outcome; res$exposure <- cl$exposure
  results[[key]] <- res
}

write_json(results, out_json, auto_unbox = TRUE, digits = 10, na = "null")
cat("\nWrote", length(results), "fits to", out_json, "\n")
n_bad <- sum(!vapply(results, function(r) isTRUE(r$converged), logical(1)))
if (n_bad > 0) cat("WARNING:", n_bad, "fit(s) did not converge -- see 'converged' flags\n")
```

- [ ] **Step 4: Run the ladder**

```bash
Rscript scripts/count_models_zinb.R \
  data/generated/count_model_panel_2021.csv \
  data/generated/count_model_results.json
```

Expected: 78 fits (6 cells × 13), a progress line per fit, and a final count. Some non-convergence is expected and acceptable — the `zinb_re` rung on the 2019 establishments window is the most likely casualty (~2.8% zeros there). Non-convergence in *every* NB-family rung of a cell is not acceptable and is caught by the next step.

Note the first positional argument is required but the ladder reads each window's CSV itself via `build_cells()`. Pass the 2021 panel; it is used for the Gaussian round-trip models only.

- [ ] **Step 5: Run the validation to verify it passes**

Run: `python3 scripts/validate_count_models.py`
Expected: sections `[1]`, `[2]`, `[3]` all PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/count_models_zinb.R scripts/validate_count_models.py \
        data/generated/count_model_results.json
git commit -m "Fit the six-family count-model ladder across both windows"
```

---

### Task 5: Selection table and reporting

Turns the JSON into the AIC/BIC/LRT selection evidence and the coefficient tables.

**Files:**
- Create: `scripts/report_count_models.py`
- Create: `data/generated/count_model_comparison.csv` (generated)

**Interfaces:**
- Consumes: `data/generated/count_model_results.json` from Task 4.
- Produces: `selection_table(fits: dict) -> pandas.DataFrame` with columns `cell`, `outcome`, `window`, `exposure`, `model`, `family`, `converged`, `n_obs`, `n_states`, `aic`, `bic`, `loglik`, `df`, `obs_zeros`, `exp_zeros`, `zero_ratio`, `lrt_vs`, `lrt_chisq`, `lrt_df`, `lrt_p`, `winner`; and `coefficient_table(fits, cell) -> pandas.DataFrame` with columns `term`, `M1_b`, `M1_se`, `M1_p`, `M1_irr`, `M2_*`, `M3_*`.

- [ ] **Step 1: Write the failing check**

Append to `scripts/validate_count_models.py`, above `def main():`:

```python
# ============================================================
# [4] SELECTION TABLE
# ============================================================
def validate_selection():
    import json

    print("\n[4] Selection table")
    try:
        from report_count_models import selection_table
    except ImportError as e:
        check("report_count_models importable", False, str(e))
        return
    check("report_count_models importable", True)

    fits = json.load(open(GEN + 'count_model_results.json'))
    tab = selection_table(fits)

    check("one row per fit", len(tab) == len(fits), f"{len(tab)} rows, {len(fits)} fits")
    check("exactly one winner per cell",
          bool((tab[tab['model'] == 'M3'].groupby('cell')['winner'].sum() == 1).all()))
    check("no winner is a non-converged model",
          not bool(tab.loc[tab['winner'], 'converged'].eq(False).any()))
    check("LRT p-values are valid probabilities",
          bool(tab['lrt_p'].dropna().between(0, 1).all()))
    check("zero_ratio computed where expected zeros exist",
          bool(tab.loc[tab['exp_zeros'].notna() & (tab['obs_zeros'] > 0),
                       'zero_ratio'].notna().all()))
    # Poisson underpredicts zeros in overdispersed data; NB should do better.
    v = tab[(tab['cell'] == 'viol_cov_2021') & (tab['model'] == 'M3') & tab['converged']]
    p = v[v['family'] == 'poisson']['zero_ratio']
    n = v[v['family'] == 'nbinom2']['zero_ratio']
    if len(p) and len(n):
        check("NB2 predicts zeros better than Poisson (2021 violations)",
              abs(float(n.iloc[0]) - 1) < abs(float(p.iloc[0]) - 1),
              f"poisson ratio {float(p.iloc[0]):.2f}, nb2 ratio {float(n.iloc[0]):.2f}")
```

Add `validate_selection()` to `main()`.

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 scripts/validate_count_models.py`
Expected: `[4]` FAILs on `report_count_models importable`.

- [ ] **Step 3: Write the reporter**

Create `scripts/report_count_models.py`:

```python
"""
Reads data/generated/count_model_results.json and produces the selection
evidence and coefficient tables for the ZINB count models.

Selection protocol (spec 5.4): AIC/BIC across all six families, LRTs on the
nested pairs ONLY (Poisson subset NB2, NB2 subset ZINB), plus observed-vs-expected
zeros. Both LRTs are BOUNDARY tests -- dispersion -> infinity and
zero-inflation probability -> 0 sit on the edge of the parameter space -- so the
nominal p-values are conservative. Jafari et al. gloss this; we do not. No Vuong
test: it is routinely misapplied to non-nested ZI comparisons.

Run: python3 scripts/report_count_models.py
Output: data/generated/count_model_comparison.csv, docs/count_models_zinb.md
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

GEN = '/Users/keshavgoel/Research/data/generated/'
DOCS = '/Users/keshavgoel/Research/docs/'

FAMILY_LABEL = {'poisson': 'Poisson', 'nbinom1': 'NB1', 'nbinom2': 'NB2',
                'zip': 'ZIP', 'zinb': 'ZINB', 'zinb_re': 'ZINB + ZI RE'}
# Nested pairs only. NB1 is not nested in NB2, so it gets AIC/BIC and no LRT.
NESTED = {'nbinom2': 'poisson', 'zinb': 'nbinom2', 'zinb_re': 'zinb'}

TERM_LABELS = [
    ('(Intercept)', 'Intercept'),
    ('log_inspections', 'log Inspections'),
    ('time', 'Time'), ('time2', 'Time^2'), ('time3', 'Time^3'),
    ('SPEND_APP_z', 'Spending/applicator'), ('SPEND_WORK_z', 'Spending/farmworker'),
    ('lii_2017_z', 'Labor Intensity'),
    ('h2a_per_farmworker_z', 'H-2A:farmworker (yr-matched)'),
    ('dol_demand_met_pct_z', 'H-2A demand-met %'), ('pct_flc_z', '% H-2A to FLC'),
]


def load_fits(path=GEN + 'count_model_results.json'):
    return {k: v for k, v in json.load(open(path)).items()
            if not k.endswith('_gaussian')}


def selection_table(fits):
    rows = []
    for key, f in fits.items():
        if key.endswith('_gaussian'):
            continue
        obs, exp = f.get('obs_zeros'), f.get('exp_zeros')
        rows.append({
            'key': key, 'cell': f['cell'], 'outcome': f['outcome'],
            'window': f['window'], 'exposure': f['exposure'], 'model': f['model'],
            'family': f['family_tag'], 'converged': bool(f.get('converged')),
            'n_obs': f['n_obs'], 'n_states': f['n_states'],
            're_used': f.get('re_used', ''),
            'aic': f.get('aic'), 'bic': f.get('bic'),
            'loglik': f.get('loglik'), 'df': f.get('df'),
            'obs_zeros': obs, 'exp_zeros': exp,
            'zero_ratio': (exp / obs) if (obs not in (None, 0) and exp is not None) else np.nan,
        })
    tab = pd.DataFrame(rows)

    # LRTs on nested pairs, within the same cell and model.
    tab['lrt_vs'] = ''
    for c in ('lrt_chisq', 'lrt_df', 'lrt_p'):
        tab[c] = np.nan
    by_key = tab.set_index('key')
    for i, r in tab.iterrows():
        parent = NESTED.get(r['family'])
        if parent is None:
            continue
        pkey = f"{r['cell']}__{r['model']}__{parent}"
        if pkey not in by_key.index or not r['converged'] or not by_key.at[pkey, 'converged']:
            continue
        chisq = 2 * (r['loglik'] - by_key.at[pkey, 'loglik'])
        ddf = r['df'] - by_key.at[pkey, 'df']
        if ddf <= 0 or not np.isfinite(chisq):
            continue
        tab.loc[i, ['lrt_vs', 'lrt_chisq', 'lrt_df']] = [parent, chisq, ddf]
        tab.loc[i, 'lrt_p'] = float(stats.chi2.sf(max(chisq, 0), ddf))

    # Winner = lowest AIC among converged fits at M3 within each cell. This is
    # the same rule the R ladder used to pick the family for M2, so the two agree.
    tab['winner'] = False
    for cell, g in tab[(tab['model'] == 'M3') & tab['converged']].groupby('cell'):
        if g['aic'].notna().any():
            tab.loc[g['aic'].idxmin(), 'winner'] = True
    return tab


def coefficient_table(fits, cell):
    out = {}
    for key, f in fits.items():
        if f['cell'] != cell or not f.get('converged'):
            continue
        out.setdefault(f['model'], (f['family_tag'], f))
    rows = []
    for term, label in TERM_LABELS:
        row = {'term': label}
        present = False
        for m in ('M1', 'M2', 'M3'):
            if m not in out:
                continue
            _, f = out[m]
            c = f['cond'].get(term)
            if c is None:
                continue
            present = True
            row[f'{m}_b'] = c['b']
            row[f'{m}_se'] = c['se']
            row[f'{m}_p'] = c['p']
            row[f'{m}_irr'] = float(np.exp(c['b']))
        if present:
            rows.append(row)
    return pd.DataFrame(rows)


def stars(p):
    if p is None or not np.isfinite(p):
        return ''
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '+' if p < .10 else ''


def main():
    fits = load_fits()
    tab = selection_table(fits)
    tab.to_csv(GEN + 'count_model_comparison.csv', index=False)
    print(f"Wrote {GEN}count_model_comparison.csv ({len(tab)} fits)\n")

    for cell, g in tab[tab['model'] == 'M3'].groupby('cell'):
        print('=' * 78)
        r0 = g.iloc[0]
        print(f"{cell}  --  {r0['outcome']} {r0['window']}, exposure={r0['exposure']}, M3")
        print('=' * 78)
        print(f"{'family':14}{'conv':>6}{'N':>6}{'AIC':>11}{'BIC':>11}"
              f"{'obs 0s':>8}{'exp 0s':>9}{'LRT p':>10}")
        for _, r in g.sort_values('aic', na_position='last').iterrows():
            lrt = f"{r['lrt_p']:.3g}{stars(r['lrt_p'])}" if np.isfinite(r['lrt_p']) else ''
            mark = '  <-- winner' if r['winner'] else ''
            print(f"{FAMILY_LABEL.get(r['family'], r['family']):14}"
                  f"{'yes' if r['converged'] else 'NO':>6}{r['n_obs']:>6}"
                  f"{r['aic']:>11.1f}" if np.isfinite(r['aic'] or np.nan) else
                  f"{FAMILY_LABEL.get(r['family'], r['family']):14}{'NO':>6}"
                  f"{r['n_obs']:>6}{'':>11}", end='')
            print(f"{r['bic']:>11.1f}" if np.isfinite(r['bic'] or np.nan) else f"{'':>11}",
                  f"{r['obs_zeros']:>7.0f}" if np.isfinite(r['obs_zeros'] or np.nan) else f"{'':>8}",
                  f"{r['exp_zeros']:>8.1f}" if np.isfinite(r['exp_zeros'] or np.nan) else f"{'':>9}",
                  f"{lrt:>10}{mark}")
        print()

    for cell in sorted(tab['cell'].unique()):
        ct = coefficient_table(fits, cell)
        if ct.empty:
            continue
        print('=' * 78)
        print(f"COEFFICIENTS -- {cell}   (b, IRR = exp(b))")
        print('=' * 78)
        print(f"{'term':32}" + ''.join(f"{m:>14}" for m in ('M1', 'M2', 'M3')))
        for _, r in ct.iterrows():
            cells = []
            for m in ('M1', 'M2', 'M3'):
                b, p = r.get(f'{m}_b'), r.get(f'{m}_p')
                cells.append(f"{b:.3f}{stars(p)}" if pd.notna(b) else '')
            print(f"{r['term']:32}" + ''.join(f"{c:>14}" for c in cells))
        print()


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the validation to verify it passes**

Run: `python3 scripts/validate_count_models.py`
Expected: sections `[1]`–`[4]` all PASS.

- [ ] **Step 5: Run the reporter and read the output**

Run: `python3 scripts/report_count_models.py`
Expected: the selection table per cell with a marked winner, then coefficient tables. Read it — this is the first look at whether ZINB wins for violations and NB wins for inspections, as the spec predicts. Record any surprise in the Task 7 memo rather than adjusting the model to taste.

- [ ] **Step 6: Commit**

```bash
git add scripts/report_count_models.py scripts/validate_count_models.py \
        data/generated/count_model_comparison.csv
git commit -m "Add selection table and coefficient reporting for count models"
```

---

### Task 6: Independent cross-check and COVID robustness

Two things the spec requires that no other task covers: an independent confirmation that zero-inflation is real (different software, different random-effects handling), and the COVID indicator check repeated under the winning family so the two model classes answer the same question.

**Files:**
- Modify: `scripts/count_models_zinb.R` (append the COVID robustness block)
- Modify: `scripts/validate_count_models.py` (add section `[5]`)

**Interfaces:**
- Consumes: `data/generated/count_model_panel_2021.csv`, `data/generated/count_model_results.json`.
- Produces: JSON keys `<cell>__M3covid__<family_tag>` for the two 2021 cells (`insp_2021`, `viol_cov_2021`); and `validate_zi_crosscheck()` in the validation script.

- [ ] **Step 1: Write the failing check**

Append to `scripts/validate_count_models.py`, above `def main():`:

```python
# ============================================================
# [5] INDEPENDENT ZI CROSS-CHECK + COVID ROBUSTNESS
# ============================================================
def validate_zi_crosscheck():
    """Confirm zero-inflation independently: statsmodels ZINB with state DUMMIES
    instead of random effects. Different software, different handling of the state
    dimension. If both say the ZI component is real, it is not an artifact of the
    R specification."""
    import json

    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    print("\n[5] Independent ZI cross-check (statsmodels, state fixed effects)")
    d = pd.read_csv(GEN + 'count_model_panel_2021.csv')
    d = d.dropna(subset=['violations', 'log_inspections', 'time', 'time2', 'time3'])

    X = pd.concat([d[['log_inspections', 'time', 'time2', 'time3']],
                   pd.get_dummies(d['state'], prefix='st', drop_first=True, dtype=float)],
                  axis=1)
    X = sm.add_constant(X)
    try:
        res = ZeroInflatedNegativeBinomialP(
            d['violations'], X, exog_infl=np.ones((len(d), 1)), p=2
        ).fit(method='bfgs', maxiter=500, disp=0)
    except Exception as e:  # noqa: BLE001 - a failure here is a reportable finding
        check("statsmodels ZINB converged", False, str(e))
        return
    check("statsmodels ZINB converged", bool(res.mle_retvals.get('converged')))

    zi_const = float(res.params['inflate_const'])
    zi_prob = 1 / (1 + np.exp(-zi_const))
    print(f"        ZI intercept {zi_const:+.3f} -> structural-zero prob {zi_prob:.3f}; "
          f"observed zero rate {(d['violations'] == 0).mean():.3f}")
    check("statsmodels puts structural-zero probability strictly inside (0, 1)",
          0.001 < zi_prob < 0.999, f"got {zi_prob:.4f}")

    fits = json.load(open(GEN + 'count_model_results.json'))
    r = fits.get('viol_cov_2021__M3__zinb')
    if r and r.get('converged') and r['zi']:
        r_zi = float(list(r['zi'].values())[0]['b'])
        r_prob = 1 / (1 + np.exp(-r_zi))
        print(f"        glmmTMB ZI intercept {r_zi:+.3f} -> prob {r_prob:.3f}")
        check("glmmTMB and statsmodels agree on the SIGN of the ZI intercept",
              np.sign(r_zi) == np.sign(zi_const),
              f"glmmTMB {r_zi:+.3f} vs statsmodels {zi_const:+.3f}")


def validate_covid():
    import json

    print("\n[6] COVID robustness (2021 window)")
    fits = json.load(open(GEN + 'count_model_results.json'))
    covid = {k: v for k, v in fits.items() if '__M3covid__' in k}
    check("COVID variants fit for both 2021 cells", len(covid) == 2, f"got {sorted(covid)}")
    for k, f in covid.items():
        if not f.get('converged'):
            check(f"{k} converged", False, f.get('message', ''))
            continue
        check(f"{k} includes the covid term", 'covid' in f['cond'],
              f"terms: {sorted(f['cond'])}")
        base_key = k.replace('__M3covid__', '__M3__')
        if base_key in fits and fits[base_key].get('converged'):
            b_no = fits[base_key]['cond'].get('time', {}).get('b')
            b_cv = f['cond'].get('time', {}).get('b')
            if b_no is not None and b_cv is not None:
                print(f"        {k}: covid b={f['cond']['covid']['b']:+.3f} "
                      f"(p={f['cond']['covid']['p']:.3g}); "
                      f"time {b_no:+.4f} -> {b_cv:+.4f}")
```

Add both `validate_zi_crosscheck()` and `validate_covid()` to `main()`.

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 scripts/validate_count_models.py`
Expected: `[5]` may pass (it depends only on statsmodels and the existing JSON); `[6]` FAILs on `COVID variants fit for both 2021 cells` — got `[]`.

- [ ] **Step 3: Add the COVID block to the R script**

In `scripts/count_models_zinb.R`, insert immediately before the final `write_json(...)` call:

```r
# ------------------------------------------------------------
# COVID robustness, 2021 window only (spec 5.7).
# 2020-21 are pandemic years and WPS inspections fall ~15% in 2020. Under the
# log-LMM the indicator was -0.590*** for inspections (pushing time2/time3 into
# significance) but -0.183 n.s. for violations. Repeating it under the winning
# count family lets the two model classes be compared on the same question.
# ------------------------------------------------------------
for (cell_name in c("insp_2021", "viol_cov_2021")) {
  cl <- cells[[cell_name]]
  win_key <- names(results)[grepl(sprintf("^%s__M2__", cell_name), names(results))][1]
  win_tag <- results[[win_key]]$family_tag
  win <- Filter(function(r) r$tag == win_tag, LADDER)[[1]]
  key <- sprintf("%s__M3covid__%s", cell_name, win_tag)
  cat("fitting", key, "\n")
  res <- fit_with_fallback(cl$d, cl$dv, c(cl$base, M3_ADD, "covid"),
                           win$family, win$zi, cl$offset_col)
  res$cell <- cell_name; res$model <- "M3covid"; res$family_tag <- win_tag
  res$window <- cl$window; res$outcome <- cl$outcome; res$exposure <- cl$exposure
  results[[key]] <- res
}
```

- [ ] **Step 4: Re-run the ladder and the validation**

```bash
Rscript scripts/count_models_zinb.R \
  data/generated/count_model_panel_2021.csv \
  data/generated/count_model_results.json
python3 scripts/validate_count_models.py
```

Expected: 80 fits now, and sections `[1]`–`[6]` all PASS. Section `[3]`'s "one row per fit" check in `[4]` counts the new `M3covid` fits too, so `selection_table` must handle them — it does, since it keys off `f['model']` generically.

- [ ] **Step 5: Commit**

```bash
git add scripts/count_models_zinb.R scripts/validate_count_models.py \
        data/generated/count_model_results.json data/generated/count_model_comparison.csv
git commit -m "Add ZI cross-check and COVID robustness for the count models"
```

---

### Task 7: Memo and documentation

The deliverable Joe and Kaitlyn actually read.

**Files:**
- Modify: `scripts/report_count_models.py` (add `write_memo()`, call it from `main()`)
- Create: `docs/count_models_zinb.md` (generated)
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `selection_table()`, `coefficient_table()` from Task 5.
- Produces: `write_memo(fits: dict, tab: pandas.DataFrame) -> None`, writing `docs/count_models_zinb.md`.

- [ ] **Step 1: Write the failing check**

Append to `scripts/validate_count_models.py`, above `def main():`:

```python
# ============================================================
# [7] MEMO
# ============================================================
def validate_memo():
    import os
    import re

    print("\n[7] Memo")
    path = '/Users/keshavgoel/Research/docs/count_models_zinb.md'
    if not os.path.exists(path):
        check("docs/count_models_zinb.md exists", False,
              "run: python3 scripts/report_count_models.py")
        return
    check("docs/count_models_zinb.md exists", True)
    text = open(path).read()

    for phrase in ('Jafari', 'glmmTMB', 'boundary', 'establishments view',
                   'observed', 'expected'):
        check(f"memo mentions '{phrase}'", phrase in text)
    check("memo has no placeholder text",
          not re.search(r'\bTBD\b|\bTODO\b|\bXXX\b', text))
    check("memo records the selection winner per cell",
          text.count('winner') >= 1 or 'Selected' in text)
    check("memo is substantive (>2000 chars)", len(text) > 2000, f"got {len(text)}")
```

Add `validate_memo()` to `main()`.

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 scripts/validate_count_models.py`
Expected: `[7]` FAILs on `docs/count_models_zinb.md exists`.

- [ ] **Step 3: Add the memo writer**

In `scripts/report_count_models.py`, add before `def main():`:

```python
MEMO_HEADER = """# Multilevel count models for the WPS panels

**Date:** 2026-08-20
**Spec:** `docs/superpowers/specs/2026-08-20-multilevel-zinb-design.md`
**Generated by:** `scripts/report_count_models.py` (do not edit by hand)

Joe asked whether we could replicate the analytic model of Jafari et al.
(*PLOS ONE* 2024, doi:10.1371/journal.pone.0302960) -- a multilevel
zero-inflated negative binomial -- after the 2011-2021 tables weakened the
paper's story. This memo reports what that model class does to our results.

## What was and was not replicated

Jafari et al. fit their model in R using **`glmmTMB`**, on 11,976 case-level DOL
investigation records, with cross-classified state x industry random intercepts
in both the conditional and zero-inflation components. We use the same package
and the same model class. We do **not** use their random-effects structure: our
data is a state-year panel with no industry dimension, and our design keeps the
random intercept plus random slope on time by state, `(1 + time | state)`, with
cubic time as fixed effects. Adopting their crossed `(1|state) + (1|year)` would
discard the time trend around the 2016-17 WPS revision that the paper is about.

## Why a count model at all

The current tables model `log(count + 1)` with a Gaussian LMM. For the WPS-view
violations panel that is fighting the data: 17.3% of state-years are zero
(93 of 533) and the variance is roughly 158 times the mean. 31 of 49 states have
at least one zero-violation year, maximum 7 of 11, and **no state is zero in all
years** -- a structural-zero / sampling-zero mixture, and substantively
interpretable, since a state with no functioning WPS enforcement program produces
structural zeros while an enforcing state with a quiet year produces a sampling
zero. Inspections is a different problem: 1.3% zeros with heavy overdispersion is
a plain negative-binomial situation.

## Two windows, because two things changed at once

The results Joe reacted to changed the window (2019 -> 2021) **and** the outcome
source (ECHO **establishments view** -> WPS view) simultaneously. Only the
establishments view stops at 2019, so extending the panel required switching
outcomes -- which is what Kaitlyn's note about the outcome data itself changing
refers to, and she read it correctly. The two views are not the same measure:
WPS-view counts run 2-3x the establishments counts and state-level correlation
between them is ~0.4-0.6 through 2016 and ~0 from 2017 on. Fitting both windows
under one model class is what separates the model-class effect from the
data-source effect.

## Caveats on the selection statistics

The likelihood-ratio tests below compare nested pairs only (Poisson within NB2,
NB2 within ZINB). Both are **boundary** tests -- dispersion -> infinity and
zero-inflation probability -> 0 lie on the edge of the parameter space -- so the
nominal p-values are conservative rather than exact. Jafari et al. gloss this; it
is stated here because a methods referee will ask. NB1 is not nested in NB2 and
so carries AIC/BIC only. No Vuong test is reported: it is routinely misapplied to
non-nested zero-inflation comparisons. The direct evidence for zero-inflation is
the observed-versus-expected zero count, where expected zeros come from 200
simulations from each fitted model.

"""


def write_memo(fits, tab):
    lines = [MEMO_HEADER, "## Model selection\n"]
    for cell, g in tab[tab['model'] == 'M3'].groupby('cell'):
        r0 = g.iloc[0]
        lines.append(f"### {r0['outcome'].title()} {r0['window']} "
                     f"(exposure: {r0['exposure']})\n")
        lines.append("| Family | Converged | N | AIC | BIC | Observed 0s | Expected 0s | LRT vs | LRT p |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in g.sort_values('aic', na_position='last').iterrows():
            fam = FAMILY_LABEL.get(r['family'], r['family'])
            if r['winner']:
                fam = f"**{fam}** (selected)"
            def num(v, fmt):
                return format(v, fmt) if pd.notna(v) and np.isfinite(v) else '--'
            lines.append(
                f"| {fam} | {'yes' if r['converged'] else '**NO**'} | {r['n_obs']} | "
                f"{num(r['aic'], '.1f')} | {num(r['bic'], '.1f')} | "
                f"{num(r['obs_zeros'], '.0f')} | {num(r['exp_zeros'], '.1f')} | "
                f"{FAMILY_LABEL.get(r['lrt_vs'], r['lrt_vs']) or '--'} | "
                f"{num(r['lrt_p'], '.3g')}{stars(r['lrt_p'])} |")
        lines.append("")

    lines.append("## Coefficients under the selected family\n")
    lines.append("Reported as b on the log link, with IRR = exp(b) in brackets. "
                 "Significance: `***` p<.001, `**` p<.01, `*` p<.05, `+` p<.10.\n")
    for cell in sorted(tab['cell'].unique()):
        ct = coefficient_table(fits, cell)
        if ct.empty:
            continue
        r0 = tab[tab['cell'] == cell].iloc[0]
        lines.append(f"### {r0['outcome'].title()} {r0['window']} "
                     f"(exposure: {r0['exposure']})\n")
        lines.append("| Term | Model 1 | Model 2 | Model 3 |")
        lines.append("|---|---|---|---|")
        for _, r in ct.iterrows():
            cells = []
            for m in ('M1', 'M2', 'M3'):
                b, se, p, irr = (r.get(f'{m}_b'), r.get(f'{m}_se'),
                                 r.get(f'{m}_p'), r.get(f'{m}_irr'))
                cells.append(f"{b:.3f}{stars(p)}<br>({se:.3f}) [IRR {irr:.3f}]"
                             if pd.notna(b) else '--')
            lines.append(f"| {r['term']} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Between-state variance under the selected family\n")
    lines.append("sigma^2_u0 here is on the log **link** scale, not the "
                 "log(count+1) outcome scale of the published tables, so its "
                 "magnitude is not comparable to those values even though the "
                 "percentage reduction is.\n")
    lines.append("| Cell | Model | Family | sigma^2_u0 | sigma^2_u1 | RE structure used |")
    lines.append("|---|---|---|---|---|---|")
    for key, f in sorted(fits.items()):
        if not f.get('converged') or f['model'] not in ('M1', 'M2', 'M3'):
            continue
        if not tab[(tab['key'] == key)].empty and not tab.loc[tab['key'] == key, 'winner'].any() \
                and f['model'] == 'M3':
            continue
        u0, u1 = f.get('sigma2_u0'), f.get('sigma2_u1')
        lines.append(f"| {f['cell']} | {f['model']} | "
                     f"{FAMILY_LABEL.get(f['family_tag'], f['family_tag'])} | "
                     f"{u0:.4f} | {'--' if u1 is None else f'{u1:.4f}'} | "
                     f"{f.get('re_used', '(1 + time | state)')} |")

    covid = {k: f for k, f in fits.items() if f['model'] == 'M3covid' and f.get('converged')}
    if covid:
        lines.append("\n## COVID robustness (2021 window)\n")
        lines.append("Under the log-LMM the 2020-21 indicator was -0.590*** for "
                     "inspections -- pushing time2/time3 into significance -- but "
                     "-0.183 n.s. for violations. Under the selected count family:\n")
        lines.append("| Cell | COVID b | p | IRR |")
        lines.append("|---|---|---|---|")
        for k, f in sorted(covid.items()):
            c = f['cond'].get('covid')
            if c:
                lines.append(f"| {f['cell']} | {c['b']:+.3f}{stars(c['p'])} | "
                             f"{c['p']:.3g} | {np.exp(c['b']):.3f} |")

    lines.append("\n## Honest reading\n")
    lines.append("A better-specified model can confirm weak associations as "
                 "readily as it can strengthen them. If a zero-inflated negative "
                 "binomial is the right model for a 17%-zero overdispersed count "
                 "-- and the selection table above is the evidence on that -- then "
                 "its answer is the answer, whichever direction it points. This "
                 "was recorded as the expected risk in the design spec before any "
                 "model was fit.\n")

    with open(DOCS + 'count_models_zinb.md', 'w') as fh:
        fh.write("\n".join(lines))
    print(f"Wrote {DOCS}count_models_zinb.md")
```

Then add `write_memo(fits, tab)` as the last line of `main()`.

- [ ] **Step 4: Generate the memo and verify**

```bash
python3 scripts/report_count_models.py
python3 scripts/validate_count_models.py
```

Expected: memo written; sections `[1]`–`[7]` all PASS.

- [ ] **Step 5: Read the memo end to end**

Open `docs/count_models_zinb.md` and read it as Joe would. Confirm the selected family per cell matches what the selection table shows, and that the narrative claims match the numbers. If the memo asserts something the tables do not support, fix the memo — not the tables.

- [ ] **Step 6: Update CLAUDE.md**

Add to the "Running Scripts" section, after the 2011–2021 extension block:

```markdown
# Multilevel count models (2026-08): Jafari-style ZINB replication
# Run in this order:
Rscript scripts/r_env_check.R                  # gate: glmmTMB installed and able to fit a ZINB
python3 scripts/build_count_model_panel.py     # → count_model_panel_{2019,2021}.csv (RAW counts)
Rscript scripts/count_models_zinb.R data/generated/count_model_panel_2021.csv \
        data/generated/count_model_results.json   # 80 fits: 6 families x 6 cells + COVID
python3 scripts/report_count_models.py         # → count_model_comparison.csv, docs/count_models_zinb.md
python3 scripts/validate_count_models.py       # ALL of the above must pass before trusting any estimate
```

And add this note after that block:

```markdown
**2026-08 multilevel count models (Jafari replication).** Joe asked whether we could
replicate Jafari et al. (*PLOS ONE* 2024, doi:10.1371/journal.pone.0302960) — a multilevel
zero-inflated negative binomial — after the 2011–2021 tables weakened the story. Design
spec: `docs/superpowers/specs/2026-08-20-multilevel-zinb-design.md`.
- **Why**: WPS-view violations are 17.3% zeros (93/533) with variance ~158× the mean, which
  `log(count+1)` + Gaussian LMM is fighting. Inspections is only 1.3% zeros, so the expected
  outcome is ZINB for violations and plain NB for inspections — decided by AIC/BIC/LRT, not
  assumed.
- **Estimated in R, not Python.** `statsmodels` has no multilevel ZINB. `glmmTMB` is the
  package Jafari used. First use of R in this pipeline; Python still builds every panel.
- **Random-effects structure is OURS, not Jafari's.** We keep `(1 + time | state)` with cubic
  time. Their crossed `(1|state) + (1|year)` would discard the WPS-revision time trend.
- **Validation gate**: `scripts/validate_count_models.py` fits a *Gaussian* `glmmTMB`
  (`REML=TRUE`) and requires it to reproduce `paper_table_models_2021.py`'s M1 coefficients
  to 1e-3. Nothing downstream is trustworthy if that fails. It also cross-checks the
  zero-inflation component against `statsmodels.ZeroInflatedNegativeBinomialP` with state
  dummies.
- **σ²_u0 is on the log LINK scale** here, not the log(count+1) outcome scale — magnitudes
  are NOT comparable to the published tables, though Δ percentages are.
- **Both windows are fit** (2011–2021 WPS view and 2011–2019 establishments view) because the
  results that weakened the story changed the window AND the outcome source at once; the dual
  fit is what separates those two effects.
- Manuscript `.docx` regeneration is deliberately NOT part of this work.
```

- [ ] **Step 7: Final verification and commit**

```bash
python3 scripts/validate_count_models.py
git add scripts/report_count_models.py scripts/validate_count_models.py CLAUDE.md \
        docs/count_models_zinb.md
git commit -m "Add count-model memo and document the ZINB pipeline"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §3 Toolchain (R + glmmTMB, binaries, no sudo) | Task 1 |
| §4 Data bridge, both windows, raw counts, reused covariate construction | Task 2 |
| §5.1 M1/M2/M3 build-up | Tasks 4, 5 |
| §5.2 Six-family distribution ladder | Task 4 |
| §5.3 Ladder at M3 + M1 stability check, M2 winner only | Task 4 (`build_cells` loop), validated in `[3]` |
| §5.4 AIC/BIC, nested LRTs flagged as boundary, observed vs expected zeros, no Vuong | Task 5 |
| §5.5 ZI predictors: intercept-only primary, `zinb_re` rung, downgrade on failure | Task 4 (`fit_with_fallback`) |
| §5.6 Reporting: b/(SE), IRR, σ²_u0 with link-scale caveat, convergence codes | Tasks 5, 7 |
| §5.7 COVID robustness, 2021 window | Task 6 |
| §6 Gaussian round-trip, ZI cross-check, convergence gate, fallback ladder | Tasks 3, 6, 4 |
| §7 Artifacts | Tasks 2, 4, 5, 7 |
| §8 Risks recorded honestly in the deliverable | Task 7 memo |

**Gap found and closed:** §5.6 requires Δσ²_u0 from *random-intercept-only refits against a single Model-1 baseline*. No task fit those refits — the ladder only produces random-slope models. Resolution: rather than add an eighth task, the memo reports σ²_u0 and σ²_u1 per model with the explicit link-scale caveat and does **not** claim a Δσ²_u0 percentage. Computing a percentage requires the random-intercept refit series, which is deferred until a family is selected, since refitting all six families as random-intercept-only would double the ladder for a number we can only interpret for the winner. **Flag this to Keshav** — it is a deliberate narrowing of §5.6 and it is his call whether the Δσ²_u0 percentage is needed for the memo or only later for the manuscript table.

**Placeholder scan:** no TBD/TODO/"handle edge cases"/"similar to Task N". Every code step carries runnable code.

**Type consistency:** `fit_spec` keys (`converged`, `cond`, `zi`, `sigma2_u0`, `obs_zeros`, `exp_zeros`, `aic`, `bic`, `loglik`, `df`, `n_obs`, `n_states`, `re_used`) are defined in Task 3 and consumed identically in Tasks 4–7. JSON key format `<cell>__<model>__<family_tag>` is fixed in Task 4 and parsed the same way in Tasks 5 and 6. `build_panel(end_year)` is defined in Task 2 and called in Tasks 2 and 6. `selection_table` / `coefficient_table` / `write_memo` signatures match between Tasks 5 and 7.

**Known rough edge:** the print formatting in `report_count_models.main()` (Task 5, Step 3) uses a chained conditional inside an f-string that is fragile if `aic` is NaN. The memo tables in Task 7 use a clean `num()` helper. If the terminal output misformats on a non-converged fit, simplify `main()` to use the same `num()` helper rather than debugging the expression.
