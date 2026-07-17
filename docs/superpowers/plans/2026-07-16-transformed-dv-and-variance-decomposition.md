# Transformed-DV Violations Models & Variance-Component Decomposition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two exploratory analysis modules — (A) the full violations model suite re-fit under four count transformations, and (B) a full random-effects variance-component decomposition of the paper-table models.

**Architecture:** Two standalone scripts under `scripts/`, each self-contained (own data build, matching the project's no-shared-module convention). Part A loops a registry of model specs over a set of DV transforms and emits a comparison CSV + diagnostics + a markdown write-up. Part B extracts every variance component from the fitted paper models, computes time-varying VPC, and writes a markdown report. Neither touches the manuscript pipeline.

**Tech Stack:** Python 3, pandas, numpy, `statsmodels.MixedLM`, `scipy.stats` (boxcox, shapiro), matplotlib.

## Global Constraints

- Run everything from `/Users/keshavgoel/Research/`; all paths absolute under `data/`, `figures/`, `docs/`.
- `RAW = '/Users/keshavgoel/Research/data/raw/'`, `GEN = '/Users/keshavgoel/Research/data/generated/'`.
- Model estimation always: `MixedLM.from_formula(..., re_formula='~time').fit(method='lbfgs')`, REML.
- `time = year − 2017`; `time2 = time**2`; `time3 = time**3`.
- Level-2 covariates z-scored before entry; spending vars arrive pre-standardized from `data/generated/spend_bls_variables.csv` (do not re-standardize).
- Pseudo-R² / Δσ² baselines re-estimated on each model's own listwise-complete analytic sample (matched N).
- 50-state filter `US_STATES_50`; abbreviation map `STATE_ABBREV_TO_NAME`; `STATE_NAME_MAPPING = {'Massachusetts ': 'Massachusetts', 'Oregon ': 'Oregon'}`.
- Exploratory only: write to terminal, `data/generated/`, `docs/`, `figures/`. Never modify `paper_table_models.py`, `fill_wps_tables_docx.py`, or any `.docx`.
- `spending_bls_models.py` must have been run so `data/generated/spend_bls_variables.csv` exists (it does in this repo).

**Reference constants block** (paste verbatim wherever "the constants block" is called for; copied from `paper_table_models.py` lines 44–77):

```python
US_STATES_50 = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
    'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
    'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
    'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
    'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
    'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
    'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
]
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
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}
STATE_NAME_MAPPING = {'Massachusetts ': 'Massachusetts', 'Oregon ': 'Oregon'}
RAW = '/Users/keshavgoel/Research/data/raw/'
GEN = '/Users/keshavgoel/Research/data/generated/'
```

**Reference data-build block** (paste verbatim wherever "the data-build block" is called for; mirrors `paper_table_models.py` lines 82–153, adds `raw` violations column already present as `violations`):

```python
import pandas as pd, numpy as np

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
long = long[long['state'].isin(US_STATES_50)].copy()
long['time'] = long['year'] - 2017
long['time2'] = long['time'] ** 2
long['time3'] = long['time'] ** 3

# Level-2 covariates (z-scored; identical construction to paper_table_models.py)
level2 = pd.DataFrame({'state': US_STATES_50})
spend = pd.read_csv(GEN + 'spend_bls_variables.csv')
level2 = level2.merge(spend[['state', 'SPEND_APP_z', 'SPEND_WORK_z']], on='state', how='left')
# other SPEND_* z variants for the stepwise/targeted families, if present:
for c in ['SPEND_FLC_z', 'SPEND_OP_z', 'SPEND_AREA_z']:
    if c in spend.columns:
        level2 = level2.merge(spend[['state', c]], on='state', how='left')
li17 = pd.read_csv(RAW + 'labor_intensity_index_2017.csv')
li17['state'] = li17['state_name'].str.title()
level2 = level2.merge(li17[['state', 'Labor_Intensity_Index']].rename(
    columns={'Labor_Intensity_Index': 'lii_2017'}), on='state', how='left')
dol1 = pd.read_csv(RAW + 'dol_var1_workers_by_state_annual.csv')
dol_study = dol1[dol1['year'].between(2011, 2019) & (dol1['year'] != 2013)]
dol_means = dol_study.groupby('state').agg(
    dol_workers_cert=('workers_certified', 'mean'),
    dol_demand_met_pct=('demand_met_pct', lambda x: x[x != np.inf].mean())).reset_index()
dol_means['state'] = dol_means['state'].map(STATE_ABBREV_TO_NAME)
level2 = level2.merge(dol_means, on='state', how='left')
bls = pd.read_csv(RAW + 'bls_oews_panel.csv')
emp = (bls[bls['occ_code'] == '45-2092'].rename(
    columns={'area_title': 'state', 'tot_emp': 'emp_farmworker'})[['state', 'emp_farmworker']])
level2 = level2.merge(emp, on='state', how='left')
level2['h2a_per_farmworker'] = level2['dol_workers_cert'] / level2['emp_farmworker']
dol2 = pd.read_csv(RAW + 'dol_var2_employer_type_annual.csv')
d2020 = dol2[dol2['year'] == 2020].copy()
d2020['state'] = d2020['state'].map(STATE_ABBREV_TO_NAME)
d2020 = d2020.rename(columns={'pct_Farm Labor Contractor': 'pct_flc'})
level2 = level2.merge(d2020[['state', 'pct_flc']], on='state', how='left')
for col in ['lii_2017', 'h2a_per_farmworker', 'dol_demand_met_pct', 'pct_flc']:
    level2[col + '_z'] = (level2[col] - level2[col].mean()) / level2[col].std()

Z_SPEND = ['SPEND_APP_z', 'SPEND_WORK_z']
Z_LABOR = ['lii_2017_z']
Z_H2A = ['h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']
L2_KEEP = [c for c in level2.columns if c.endswith('_z') or c in
           ('lii_2017', 'h2a_per_farmworker', 'dol_demand_met_pct', 'pct_flc')]
df = long.merge(level2[['state'] + L2_KEEP], on='state', how='left')
```

---

## Part A — Transformed-DV Violations Models

### Task A1: Data build + transform functions

**Files:**
- Create: `scripts/transformed_violations_models.py`

**Interfaces:**
- Produces: module-level `df` (long frame with `violations`, `inspections`, `time*`, z-covariates), and `add_transform_columns(df) -> (df, boxcox_lambda)` which adds columns `dv_raw, dv_log, dv_sqrt, dv_anscombe, dv_ft, dv_boxcox` and returns the fitted Box-Cox λ. `TRANSFORMS = {'raw':'dv_raw','log':'dv_log','sqrt':'dv_sqrt','anscombe':'dv_anscombe','boxcox':'dv_boxcox'}` (Freeman–Tukey `dv_ft` is reported as a diagnostic companion, not a primary model DV).

- [ ] **Step 1: Write the script header, constants, and data build**

Create `scripts/transformed_violations_models.py` starting with a docstring, then paste **the constants block** and **the data-build block** from Global Constraints. Add at top:

```python
"""
EXPLORATORY — violations model suite re-fit under count transformations.

Fits the full violations model suite (time-only baseline, Zimmerman stepwise, BLS
spending, targeted FIFRA, labor/DOL stepwise, curated final, paper Table 3 build-up)
under raw / log / sqrt / Anscombe / Box-Cox transforms of the violation count.

AIC/loglik are NOT comparable across DV transforms — transforms are judged on
residual diagnostics (Shapiro-Wilk W, skew, kurtosis, heteroscedasticity) and on
the stability of substantive coefficients. Pseudo-R² (a variance ratio) is reported.

Outputs: data/generated/transform_comparison.csv, docs/transform_exploration.md,
figures/transform_qq_*.png. Manuscript artifacts are untouched.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')
```

- [ ] **Step 2: Add the transform-column function**

```python
def add_transform_columns(df):
    y = df['violations'].astype(float)
    df = df.copy()
    df['dv_raw'] = y
    df['dv_log'] = np.log(y + 1.0)
    df['dv_sqrt'] = np.sqrt(y)
    df['dv_anscombe'] = 2.0 * np.sqrt(y + 3.0 / 8.0)
    df['dv_ft'] = np.sqrt(y) + np.sqrt(y + 1.0)          # Freeman–Tukey companion
    # Box-Cox needs strictly positive input; estimate lambda once on (y+1).
    shifted = (y + 1.0).dropna()
    bc_vals, lam = stats.boxcox(shifted.values)
    df.loc[shifted.index, 'dv_boxcox'] = bc_vals
    return df, float(lam)

TRANSFORMS = {'raw': 'dv_raw', 'log': 'dv_log', 'sqrt': 'dv_sqrt',
              'anscombe': 'dv_anscombe', 'boxcox': 'dv_boxcox'}
```

- [ ] **Step 3: Add sanity-check guardrails at module end (temporary)**

```python
df, BOXCOX_LAMBDA = add_transform_columns(df)
assert df['state'].nunique() == 50, df['state'].nunique()
assert len(df) == 450, len(df)                              # 50 states x 9 years
for col in ['dv_raw', 'dv_log', 'dv_sqrt', 'dv_anscombe', 'dv_boxcox', 'dv_ft']:
    assert col in df.columns, col
assert df['dv_log'].min() >= 0
print(f"OK: df {df.shape}, Box-Cox lambda = {BOXCOX_LAMBDA:.4f}")
```

- [ ] **Step 4: Run and verify the build**

Run: `python3 scripts/transformed_violations_models.py`
Expected: prints `OK: df (450, ...), Box-Cox lambda = <number>` with no assertion error. (N=450 before listwise deletion; covariate models drop to ~39 states later.)

- [ ] **Step 5: Commit**

```bash
git add scripts/transformed_violations_models.py
git commit -m "feat(explore): data build + count transforms for violations DV"
```

---

### Task A2: Model registry, fit/diagnostic driver, comparison CSV

**Files:**
- Modify: `scripts/transformed_violations_models.py`

**Interfaces:**
- Consumes: `df`, `TRANSFORMS`, `BOXCOX_LAMBDA` from Task A1.
- Produces: `MODEL_SPECS` (list of `dict(name, rhs, re_formula='~time')`), `fit_spec(dv_col, rhs, data) -> (res, d)`, `diagnostics(res, d) -> dict`, `pseudo_r2(res, d, dv_col, baseline_rhs) -> float`, and writes `data/generated/transform_comparison.csv`.

- [ ] **Step 1: Replace the temporary guardrail print with the model registry**

Remove the Step-3/Step-4 `print` from Task A1 (keep the two `assert df... ` lines). Add:

```python
# Baseline (cubic time) used for pseudo-R2 of the cubic-time families.
CUBIC_BASE = ['time', 'time2', 'time3']
# Paper Table 3 uses LINEAR time + inspections (matches current manuscript spec).
PAPER_BASE = ['inspections', 'time']

MODEL_SPECS = [
    # name, rhs, baseline_rhs
    ('baseline_cubic', CUBIC_BASE, CUBIC_BASE),
    # Zimmerman one-at-a-time (cubic time + one L2 covariate)
    ('zim_spend_app', CUBIC_BASE + ['SPEND_APP_z'], CUBIC_BASE),
    ('zim_spend_work', CUBIC_BASE + ['SPEND_WORK_z'], CUBIC_BASE),
    ('zim_lii', CUBIC_BASE + ['lii_2017_z'], CUBIC_BASE),
    ('zim_h2a', CUBIC_BASE + ['h2a_per_farmworker_z'], CUBIC_BASE),
    ('zim_demandmet', CUBIC_BASE + ['dol_demand_met_pct_z'], CUBIC_BASE),
    ('zim_flc', CUBIC_BASE + ['pct_flc_z'], CUBIC_BASE),
    # BLS spending combined (the two FIFRA populations)
    ('spend_combined', CUBIC_BASE + ['SPEND_APP_z', 'SPEND_WORK_z'], CUBIC_BASE),
    # Targeted FIFRA (SPEND_AREA_z only if present)
    ('targeted_fifra', CUBIC_BASE + ['SPEND_WORK_z', 'SPEND_APP_z'], CUBIC_BASE),
    # Curated final (cubic time + labor block)
    ('final_labor', CUBIC_BASE + ['lii_2017_z', 'h2a_per_farmworker_z',
                                   'dol_demand_met_pct_z', 'pct_flc_z'], CUBIC_BASE),
    # Paper Table 3 build-up (LINEAR time + inspections)
    ('paper_M1', PAPER_BASE, PAPER_BASE),
    ('paper_M2', PAPER_BASE + ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z'], PAPER_BASE),
    ('paper_M3', PAPER_BASE + ['SPEND_APP_z', 'SPEND_WORK_z', 'lii_2017_z',
                               'h2a_per_farmworker_z', 'dol_demand_met_pct_z',
                               'pct_flc_z', 'pct_flc_z:time'], PAPER_BASE),
]
# Add SPEND_AREA_z to the targeted model only if it exists in the frame.
if 'SPEND_AREA_z' in df.columns:
    MODEL_SPECS = [(('targeted_fifra', CUBIC_BASE + ['SPEND_WORK_z', 'SPEND_APP_z',
                    'SPEND_AREA_z'], CUBIC_BASE) if n == 'targeted_fifra' else (n, r, b))
                   for (n, r, b) in MODEL_SPECS]
```

- [ ] **Step 2: Add fit, pseudo-R², and diagnostics helpers**

```python
def _analytic(dv_col, rhs, data):
    need = {dv_col} | {t for term in rhs for t in term.split(':')}
    need = {c for c in need if c in data.columns}
    d = data.dropna(subset=list(need)).copy()
    d['state'] = pd.Categorical(d['state'])
    return d

def fit_spec(dv_col, rhs, data):
    d = _analytic(dv_col, rhs, data)
    formula = f"{dv_col} ~ " + " + ".join(rhs)
    res = MixedLM.from_formula(formula, data=d, groups=d['state'],
                               re_formula='~time').fit(method='lbfgs')
    return res, d

def pseudo_r2(res, d, dv_col, baseline_rhs):
    base, _ = fit_spec(dv_col, baseline_rhs, d)   # matched-N baseline on this sample
    s_model = float(res.cov_re.iloc[0, 0])
    s_base = float(base.cov_re.iloc[0, 0])
    return (s_base - s_model) / s_base * 100.0

def diagnostics(res, d):
    resid = np.asarray(res.resid)
    fitted = np.asarray(res.fittedvalues)
    # Shapiro on residuals (cap n at 5000; here always small)
    w, p = stats.shapiro(resid)
    skew = float(stats.skew(resid))
    kurt = float(stats.kurtosis(resid))          # excess kurtosis
    # Heteroscedasticity: OLS of resid^2 on fitted -> R^2 * n ~ chi2 (Breusch-Pagan-ish)
    X = np.column_stack([np.ones_like(fitted), fitted])
    beta, *_ = np.linalg.lstsq(X, resid ** 2, rcond=None)
    pred = X @ beta
    ss_tot = np.sum((resid ** 2 - (resid ** 2).mean()) ** 2)
    ss_res = np.sum((resid ** 2 - pred) ** 2)
    bp_r2 = 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return {'shapiro_w': float(w), 'shapiro_p': float(p),
            'resid_skew': skew, 'resid_kurtosis': kurt, 'bp_r2': float(bp_r2)}
```

- [ ] **Step 3: Loop specs × transforms, collect rows, write CSV**

```python
def key_terms(rhs):
    """Non-time fixed effects worth reporting for stability."""
    drop = {'time', 'time2', 'time3', 'inspections'}
    return [t for t in rhs if t not in drop]

records = []
for name, rhs, base_rhs in MODEL_SPECS:
    for tkey, dv_col in TRANSFORMS.items():
        try:
            res, d = fit_spec(dv_col, rhs, df)
        except Exception as e:
            records.append({'model': name, 'transform': tkey, 'error': str(e)})
            continue
        diag = diagnostics(res, d)
        row = {'model': name, 'transform': tkey,
               'n_obs': len(d), 'n_states': d['state'].nunique(),
               'pseudo_r2_pct': pseudo_r2(res, d, dv_col, base_rhs), **diag}
        for t in key_terms(rhs):
            if t in res.fe_params.index:
                row[f'b__{t}'] = float(res.fe_params[t])
                row[f'p__{t}'] = float(res.pvalues[t])
        records.append(row)

out = pd.DataFrame(records)
out.to_csv(GEN + 'transform_comparison.csv', index=False)
print(f"Wrote {GEN}transform_comparison.csv  ({len(out)} rows)")
print(f"Box-Cox lambda = {BOXCOX_LAMBDA:.4f}")

# Headline diagnostics table to terminal
head = out[out['model'].isin(['baseline_cubic', 'final_labor', 'paper_M3'])]
cols = ['model', 'transform', 'shapiro_w', 'shapiro_p', 'resid_skew',
        'resid_kurtosis', 'bp_r2', 'pseudo_r2_pct']
print(head[cols].to_string(index=False))
```

- [ ] **Step 4: Run and verify the CSV**

Run: `python3 scripts/transformed_violations_models.py`
Expected: prints `Wrote .../transform_comparison.csv (N rows)` where N = len(MODEL_SPECS) × 5 (minus any error rows), the Box-Cox λ, and a headline table with a Shapiro W column that is **higher (closer to 1)** and skew **closer to 0** for at least one transform vs `raw`. Confirm the CSV has columns `shapiro_w`, `pseudo_r2_pct`, and `b__SPEND_APP_z`.

Run: `python3 -c "import pandas as pd; d=pd.read_csv('data/generated/transform_comparison.csv'); print(d.shape); print(sorted(d['transform'].unique())); print(d.columns.tolist())"`
Expected: 5 transforms present; no all-NaN key columns for the headline models.

- [ ] **Step 5: Commit**

```bash
git add scripts/transformed_violations_models.py data/generated/transform_comparison.csv
git commit -m "feat(explore): fit violations suite across transforms + diagnostics CSV"
```

---

### Task A3: QQ / residual figures for headline models

**Files:**
- Modify: `scripts/transformed_violations_models.py`

**Interfaces:**
- Consumes: `fit_spec`, `MODEL_SPECS`, `TRANSFORMS`, `df` from Tasks A1–A2.
- Produces: `figures/transform_qq_<model>.png` (2×5 grid: QQ + resid-vs-fitted per transform) for the three headline models.

- [ ] **Step 1: Add the figure routine**

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = '/Users/keshavgoel/Research/figures/'
HEADLINE = {'baseline_cubic', 'final_labor', 'paper_M3'}
spec_by_name = {n: (r, b) for (n, r, b) in MODEL_SPECS}

def make_figs():
    for name in HEADLINE:
        rhs, _ = spec_by_name[name]
        fig, axes = plt.subplots(2, len(TRANSFORMS), figsize=(4 * len(TRANSFORMS), 8))
        for j, (tkey, dv_col) in enumerate(TRANSFORMS.items()):
            res, d = fit_spec(dv_col, rhs, df)
            resid = np.asarray(res.resid)
            stats.probplot(resid, dist='norm', plot=axes[0, j])
            axes[0, j].set_title(f'{tkey}: QQ')
            axes[1, j].scatter(np.asarray(res.fittedvalues), resid, s=8, alpha=.5)
            axes[1, j].axhline(0, color='r', lw=.8)
            axes[1, j].set_title(f'{tkey}: resid vs fitted')
        fig.suptitle(f'Violations model "{name}" — residual diagnostics by transform')
        fig.tight_layout()
        fig.savefig(f'{FIG}transform_qq_{name}.png', dpi=120)
        plt.close(fig)
        print(f"Wrote {FIG}transform_qq_{name}.png")

make_figs()
```

- [ ] **Step 2: Run and verify figures exist**

Run: `python3 scripts/transformed_violations_models.py`
Expected: three `Wrote .../transform_qq_*.png` lines.

Run: `ls -1 figures/transform_qq_*.png | wc -l`
Expected: `3`

- [ ] **Step 3: Commit**

```bash
git add scripts/transformed_violations_models.py figures/transform_qq_*.png
git commit -m "feat(explore): residual QQ / resid-vs-fitted figures per transform"
```

---

### Task A4: Write-up `docs/transform_exploration.md`

**Files:**
- Create: `docs/transform_exploration.md`

**Interfaces:**
- Consumes: `data/generated/transform_comparison.csv`, Box-Cox λ (from terminal), the three figures.

- [ ] **Step 1: Read the produced numbers**

Run: `python3 -c "import pandas as pd; d=pd.read_csv('data/generated/transform_comparison.csv'); print(d[d.model.isin(['baseline_cubic','final_labor','paper_M3'])][['model','transform','shapiro_w','shapiro_p','resid_skew','resid_kurtosis','bp_r2','pseudo_r2_pct']].to_string(index=False))"`
Record the printed values — they populate the tables below.

- [ ] **Step 2: Write the doc**

Create `docs/transform_exploration.md` with these sections, filling the tables from Step 1's output (no placeholders — paste the actual numbers):
- **Purpose & method** — the four transforms, the `+1` shift for log/Box-Cox, the fitted Box-Cox λ, and the AIC-non-comparability caveat.
- **Diagnostics by transform** — a table (rows = transform, cols = Shapiro W, Shapiro p, skew, excess kurtosis, BP-R²) for each headline model, with the raw row first.
- **Coefficient stability** — whether the sign/significance of `SPEND_APP_z`, `SPEND_WORK_z`, `pct_flc_z` (and its ×time) hold across transforms.
- **Recommendation** — which transform best satisfies normality/homoscedasticity while preserving conclusions, and whether it warrants a manuscript robustness appendix.
- **Figures** — reference the three `figures/transform_qq_*.png`.

- [ ] **Step 3: Commit**

```bash
git add docs/transform_exploration.md
git commit -m "docs(explore): transform exploration write-up + recommendation"
```

---

## Part B — Variance-Component Decomposition

### Task B1: Component extraction + VPC + terminal tables + CSV

**Files:**
- Create: `scripts/variance_decomposition.py`

**Interfaces:**
- Produces: `components(res) -> dict(sigma2_u0, sigma2_u1, sigma_u01, corr_u01, sigma2_e, icc)`, `vpc(comp, t) -> float`, and writes `data/generated/variance_components.csv` (one row per DV×model).

- [ ] **Step 1: Header, constants, data build**

Create `scripts/variance_decomposition.py` with a docstring, then paste **the constants block** and **the data-build block**. Docstring:

```python
"""
EXPLORATORY — full variance-component decomposition of the paper-table models.

For violations and inspections M1/M2/M3 (matching paper_table_models.py specs),
extract sigma^2_u0 (between-state intercept), sigma^2_u1 (random slope for time),
sigma_u01 (intercept-slope covariance), sigma^2_e (Level-1 residual); derive ICC
and the time-varying VPC(t). Diagnoses the negative inspections Delta-sigma^2.

Outputs: data/generated/variance_components.csv, docs/variance_decomposition.md,
figures/vpc_violations.png, figures/vpc_inspections.png. No manuscript artifact touched.
"""
import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings('ignore')
```

- [ ] **Step 2: Define the paper-table model specs (mirror `paper_table_models.py`, linear time)**

```python
Z_SPEND = ['SPEND_APP_z', 'SPEND_WORK_z']
Z_LABOR = ['lii_2017_z']
Z_H2A = ['h2a_per_farmworker_z', 'dol_demand_met_pct_z', 'pct_flc_z']

SPECS = {
    'inspections': {
        'M1': ['time'],
        'M2': ['time'] + Z_SPEND + Z_LABOR,
        'M3': ['time'] + Z_SPEND + Z_LABOR + Z_H2A,
    },
    'violations': {
        'M1': ['inspections', 'time'],
        'M2': ['inspections', 'time'] + Z_SPEND + Z_LABOR,
        'M3': ['inspections', 'time'] + Z_SPEND + Z_LABOR + Z_H2A + ['pct_flc_z:time'],
    },
}
```

- [ ] **Step 3: Add fit + component extraction + VPC**

```python
def fit_spec(dv, rhs, data):
    need = {dv} | {t for term in rhs for t in term.split(':')}
    d = data.dropna(subset=[c for c in need if c in data.columns]).copy()
    d['state'] = pd.Categorical(d['state'])
    formula = f"{dv} ~ " + " + ".join(rhs)
    res = MixedLM.from_formula(formula, data=d, groups=d['state'],
                               re_formula='~time').fit(method='lbfgs')
    return res, d

def components(res):
    u0 = float(res.cov_re.iloc[0, 0])
    u1 = float(res.cov_re.iloc[1, 1]) if res.cov_re.shape[0] > 1 else 0.0
    u01 = float(res.cov_re.iloc[0, 1]) if res.cov_re.shape[0] > 1 else 0.0
    e = float(res.scale)
    corr = u01 / np.sqrt(u0 * u1) if u0 > 0 and u1 > 0 else np.nan
    icc = u0 / (u0 + e) if (u0 + e) > 0 else np.nan
    return {'sigma2_u0': u0, 'sigma2_u1': u1, 'sigma_u01': u01,
            'corr_u01': corr, 'sigma2_e': e, 'icc': icc}

def vpc(comp, t):
    between = comp['sigma2_u0'] + 2 * t * comp['sigma_u01'] + t ** 2 * comp['sigma2_u1']
    return between / (between + comp['sigma2_e'])
```

- [ ] **Step 4: Loop DV×model, print tables, write CSV**

```python
records = []
for dv, models in SPECS.items():
    print("\n" + "=" * 70 + f"\n{dv.upper()}\n" + "=" * 70)
    print(f"{'model':6}{'u0':>12}{'u1':>12}{'cov':>12}{'corr':>10}{'resid':>12}{'ICC':>8}")
    for m, rhs in models.items():
        res, d = fit_spec(dv, rhs, df)
        c = components(res)
        print(f"{m:6}{c['sigma2_u0']:>12.2f}{c['sigma2_u1']:>12.3f}"
              f"{c['sigma_u01']:>12.3f}{c['corr_u01']:>10.2f}"
              f"{c['sigma2_e']:>12.2f}{c['icc']:>8.2f}")
        rec = {'dv': dv, 'model': m, 'n_obs': len(d),
               'n_states': d['state'].nunique(), **c}
        for yr in range(2011, 2020):
            rec[f'vpc_{yr}'] = vpc(c, yr - 2017)
        records.append(rec)

vc = pd.DataFrame(records)
vc.to_csv(GEN + 'variance_components.csv', index=False)
print(f"\nWrote {GEN}variance_components.csv ({len(vc)} rows)")
```

- [ ] **Step 5: Run and verify**

Run: `python3 scripts/variance_decomposition.py`
Expected: two component tables (inspections, violations), ICC for violations ≈ 0.5–0.8, and `Wrote .../variance_components.csv (6 rows)`. Confirm the inspections M2/M3 `u0` values reproduce the manuscript σ²_u0 (≈ 812.64 / 821.20) so the extraction is correct.

- [ ] **Step 6: Commit**

```bash
git add scripts/variance_decomposition.py data/generated/variance_components.csv
git commit -m "feat(explore): variance-component extraction + time-varying VPC"
```

---

### Task B2: VPC + component figures

**Files:**
- Modify: `scripts/variance_decomposition.py`

**Interfaces:**
- Consumes: `vc` DataFrame from Task B1.
- Produces: `figures/vpc_<dv>.png` (VPC across 2011–2019, one line per model) and `figures/varcomp_<dv>.png` (stacked component bars per model).

- [ ] **Step 1: Add the figure routine**

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG = '/Users/keshavgoel/Research/figures/'
YEARS = list(range(2011, 2020))

def make_figs(vc):
    for dv in vc['dv'].unique():
        sub = vc[vc['dv'] == dv]
        # VPC curves
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for _, r in sub.iterrows():
            ax.plot(YEARS, [r[f'vpc_{y}'] for y in YEARS], marker='o', label=r['model'])
        ax.set_title(f'{dv}: variance partition coefficient over time')
        ax.set_xlabel('year'); ax.set_ylabel('VPC (between-state share)')
        ax.set_ylim(0, 1); ax.legend()
        fig.tight_layout(); fig.savefig(f'{FIG}vpc_{dv}.png', dpi=120); plt.close(fig)
        print(f"Wrote {FIG}vpc_{dv}.png")
        # Component bars (u0, u1 scaled, residual)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(sub)); w = 0.6
        ax.bar(x, sub['sigma2_u0'], w, label='sigma^2_u0 (intercept)')
        ax.bar(x, sub['sigma2_e'], w, bottom=sub['sigma2_u0'], label='sigma^2_e (residual)')
        ax.set_xticks(x); ax.set_xticklabels(sub['model'])
        ax.set_title(f'{dv}: intercept vs residual variance'); ax.legend()
        fig.tight_layout(); fig.savefig(f'{FIG}varcomp_{dv}.png', dpi=120); plt.close(fig)
        print(f"Wrote {FIG}varcomp_{dv}.png")

make_figs(vc)
```

- [ ] **Step 2: Run and verify**

Run: `python3 scripts/variance_decomposition.py`
Expected: four `Wrote .../*.png` lines.
Run: `ls -1 figures/vpc_*.png figures/varcomp_*.png | wc -l`
Expected: `4`

- [ ] **Step 3: Commit**

```bash
git add scripts/variance_decomposition.py figures/vpc_*.png figures/varcomp_*.png
git commit -m "feat(explore): VPC and variance-component figures"
```

---

### Task B3: Write-up `docs/variance_decomposition.md`

**Files:**
- Create: `docs/variance_decomposition.md`

**Interfaces:**
- Consumes: `data/generated/variance_components.csv`, the four figures.

- [ ] **Step 1: Read the produced numbers**

Run: `python3 -c "import pandas as pd; d=pd.read_csv('data/generated/variance_components.csv'); print(d[['dv','model','sigma2_u0','sigma2_u1','sigma_u01','corr_u01','sigma2_e','icc','vpc_2011','vpc_2017','vpc_2019']].to_string(index=False))"`
Record the values.

- [ ] **Step 2: Write the doc**

Create `docs/variance_decomposition.md` with (fill tables from Step 1, no placeholders):
- **The mixed-model variance structure** — what σ²_u0, σ²_u1, σ_u01, σ²_e each mean here; the VPC(t) formula from the spec.
- **Component tables** — one per DV (rows = M1/M2/M3, cols = the five components + ICC).
- **VPC over time** — read the VPC 2011/2017/2019 columns; note whether the random slope makes the between-state share rise or fall across the window; reference `figures/vpc_*.png`.
- **Why the current pseudo-R² is partial** — it uses only σ²_u0; show where covariate variance actually lands.
- **Negative inspections Δσ² diagnosis** — using the inspections M1→M2→M3 component movement, show which component absorbed the change and explain that σ²_u0 is not bounded to fall when fixed effects are added.

- [ ] **Step 3: Commit**

```bash
git add docs/variance_decomposition.md
git commit -m "docs(explore): variance-component decomposition write-up"
```

---

## Task C: Register both scripts in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add run commands and the analysis-sequence rows**

Add under "Running Scripts" (near the manuscript-tables block):

```bash
# Exploratory (2026-07): transformed-DV robustness + variance decomposition
python3 scripts/transformed_violations_models.py   # violations suite across log/sqrt/Anscombe/Box-Cox
python3 scripts/variance_decomposition.py          # full random-effects component breakdown + VPC
```

Add two rows to the analysis-sequence table:

```markdown
| `transformed_violations_models.py` | Full violations suite re-fit under raw/log/sqrt/Anscombe/Box-Cox; residual diagnostics + pseudo-R² (AIC not cross-comparable). Exploratory. | `data/generated/transform_comparison.csv`, `docs/transform_exploration.md`, `figures/transform_qq_*.png` |
| `variance_decomposition.py` | Full variance-component decomposition (σ²_u0/σ²_u1/σ_u01/σ²_e, ICC, time-varying VPC) of the paper-table models; diagnoses the negative inspections Δσ². Exploratory. | `data/generated/variance_components.csv`, `docs/variance_decomposition.md`, `figures/vpc_*.png` |
```

- [ ] **Step 2: Verify and commit**

Run: `grep -c "transformed_violations_models.py\|variance_decomposition.py" CLAUDE.md`
Expected: `4` (two mentions each).

```bash
git add CLAUDE.md
git commit -m "docs: register exploratory transform + variance-decomposition scripts"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** Part A transforms (A1), full model suite (A2 registry), AIC caveat + diagnostics (A2/A4), figures (A3), write-up (A4). Part B components + VPC (B1), figures (B2), negative-Δσ² diagnosis (B3). Shared conventions in Global Constraints. CLAUDE.md registration (Task C). All spec sections mapped.
- **Placeholder scan:** Doc-writing steps (A4, B3) intentionally defer prose to after results exist, but each specifies exact sections + the exact command that produces the numbers to paste — no vague "add analysis". No TBD/TODO.
- **Type consistency:** `fit_spec` signature differs intentionally between the two scripts (Part A keys on `dv_col`; Part B on `dv` name) — they are separate modules, never cross-imported. `components`/`vpc`/`diagnostics`/`pseudo_r2` names are consistent within each script.
- **Known caveat to watch at execution:** `res.resid`/`res.fittedvalues` in statsmodels MixedLM are marginal (fixed-effects) residuals; that is the intended basis for the normality/heteroscedasticity diagnostics and is stated as such in the write-up.
