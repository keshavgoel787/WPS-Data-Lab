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
    # Unused by this task's checks -- scaffold for later sections (e.g. comparing
    # fitted ZINB parameters/statistics against known tolerances).
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
    check("2019 violations: 394 non-null, 10 zeros",
          (p19['violations'].notna().sum(), (p19['violations'] == 0).sum()) == (394, 10),
          f"got {p19['violations'].notna().sum()} non-null, {(p19['violations'] == 0).sum()} zeros")
    check("2019 inspections: 450 non-null, 15 zeros",
          (p19['inspections'].notna().sum(), (p19['inspections'] == 0).sum()) == (450, 15),
          f"got {p19['inspections'].notna().sum()} non-null, {(p19['inspections'] == 0).sum()} zeros")

    # Asymmetry invariant: the establishments reshape's fillna(0) on the two
    # inspections components (see build_count_model_panel.py comment) manufactures
    # a 0 whenever a state-year is missing a component, instead of propagating
    # NaN. Violations has no such fillna and keeps its NaNs. Net effect: 2019
    # inspections has 0 NaN while 2019 violations has 56 NaN.
    check("2019 inspections has 0 NaN while 2019 violations has 56 NaN (fillna(0) asymmetry)",
          (p19['inspections'].isna().sum(), p19['violations'].isna().sum()) == (0, 56),
          f"got {p19['inspections'].isna().sum()} / {p19['violations'].isna().sum()}")

    # Direct proof the 2019 inspection zeros are (at least partly) manufactured,
    # not measured: every one of the 15 zero-inspection state-years has at least
    # one raw component (inspections-epa-{year} / inspections-state-{year})
    # missing in the source CSV, which fillna(0) silently converted to a 0.
    from build_count_model_panel import RAW, US_STATES_50, STATE_NAME_MAPPING
    echo = pd.read_csv(RAW + 'establishments_data.csv', index_col=0)
    echo.index = echo.index.str.strip().map(lambda x: STATE_NAME_MAPPING.get(x, x))
    echo = echo[echo.index.isin(US_STATES_50)]
    n_zero_with_raw_nan = 0
    for year in range(2011, 2020):
        epa = pd.to_numeric(echo[f'inspections-epa-{year}'], errors='coerce')
        st = pd.to_numeric(echo[f'inspections-state-{year}'], errors='coerce')
        zero_mask = (epa.fillna(0) + st.fillna(0)) == 0
        n_zero_with_raw_nan += int((zero_mask & (epa.isna() | st.isna())).sum())
    check("all 15 2019 zero-inspection state-years have >=1 raw component missing (manufactured, not measured, zeros)",
          n_zero_with_raw_nan == 15, f"got {n_zero_with_raw_nan}")

    # 2021 (WPS view) is read directly from wps_dv_panel_2011_2021.csv with no
    # fillna step in this script -- any NaN there is genuine upstream missingness,
    # not manufactured. It happens to also show inspections fully populated, but
    # for a different reason than 2019: the WPS source simply has no missing
    # inspections cells in this window, not because a fillna(0) forced it.
    check("2021 violations has 6 NaN, inspections has 0 NaN (genuine WPS-source missingness, not fillna-manufactured)",
          (p21['violations'].isna().sum(), p21['inspections'].isna().sum()) == (6, 0),
          f"got violations NaN={p21['violations'].isna().sum()}, inspections NaN={p21['inspections'].isna().sum()}")

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
    # sys.path mutation lives here (not at module scope) because this file is run
    # as a script, not imported; a future import-by-name use would need its own
    # sys.path setup first.
    sys.path.insert(0, '/Users/keshavgoel/Research/scripts')
    main()
