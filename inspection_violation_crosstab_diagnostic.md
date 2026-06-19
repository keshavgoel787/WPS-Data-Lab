# Diagnostic: Inspections × Violations Crosstab by State-Year

**Question:** Do any state-years record violations without an inspection? Under WPS a violation is cited through an inspection, so a `violations > 0` with `inspections == 0` row would signal a data-integrity issue.

**Source:** `establishments-data (2).csv`, 50 U.S. states, 2011–2019.

**DV definitions:** `inspections = inspections-epa + inspections-state` (single-source NaN treated as 0); `violations = violations-YYYY`.

**Total state-year observations:** 450 (50 states × 9 years = 450; `violations` columns exist for all years).


## 1. Crosstab (counts of state-years)

| insp_cat      |   violations=0 |   violations=NaN |   violations>0 |   Total |
|:--------------|---------------:|-----------------:|---------------:|--------:|
| inspections=0 |              1 |                8 |              6 |      15 |
| inspections>0 |              9 |               48 |            378 |     435 |
| Total         |             10 |               56 |            384 |     450 |


## 2. Headline result

⚠️ **6 state-year(s) record `violations > 0` with `inspections == 0`.** These are listed below and warrant inspection of the raw source.


## 3. Per-year summary

|   year |   n_stateyears |   n_viol_pos |   n_insp_zero |   n_flagged |
|-------:|---------------:|-------------:|--------------:|------------:|
|   2011 |             50 |           44 |             3 |           1 |
|   2012 |             50 |           41 |             1 |           0 |
|   2013 |             50 |           36 |             3 |           2 |
|   2014 |             50 |           36 |             2 |           1 |
|   2015 |             50 |           39 |             2 |           1 |
|   2016 |             50 |           47 |             0 |           0 |
|   2017 |             50 |           47 |             1 |           0 |
|   2018 |             50 |           46 |             1 |           0 |
|   2019 |             50 |           48 |             2 |           1 |


- `n_viol_pos` = state-years with at least one violation
- `n_insp_zero` = state-years with zero total inspections
- `n_flagged` = state-years with violations but zero inspections


## 4. Flagged state-years (violations > 0 AND inspections == 0)

| state     |   year |   insp_epa |   insp_state |   inspections |   violations |
|:----------|-------:|-----------:|-------------:|--------------:|-------------:|
| Vermont   |   2011 |        nan |            0 |             0 |            2 |
| Alaska    |   2013 |        nan |            0 |             0 |            1 |
| Utah      |   2013 |        nan |            0 |             0 |            4 |
| Alaska    |   2014 |        nan |            0 |             0 |            1 |
| Montana   |   2015 |        nan |            0 |             0 |            1 |
| Minnesota |   2019 |        nan |            0 |             0 |           15 |


## 5. Note on missing inspection values

State-years where **both** inspection sources are NaN (sum = NaN): 0.

