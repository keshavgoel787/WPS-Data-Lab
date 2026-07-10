/*
Hierarchical Mixed-Effects Model for Predicting Violations
State-Year Level Analysis (2011-2019)

This script:
1. Loads and reshapes EPA ECHO and WPS data
2. Loads, cleans, and inflation-adjusts agricultural spending data
3. Filters to 50 US states only
4. Constructs time variables centered on 2017
5. Fits a hierarchical mixed-effects model with:
   - Fixed effects: time, time^2, time^3, (+ spending in Model 2)
   - Random effects: random intercept, random slope for time by state
*/

clear all
set more off
version 14

display "{hline 70}"
display "HIERARCHICAL MIXED-EFFECTS MODEL FOR VIOLATIONS"
display "{hline 70}"

* ============================================================
* CPI-U LOOKUP TABLE (BLS annual averages, base year 2017)
* ============================================================
* Will be used to compute deflators: spending_2017 = nominal * (245.120 / CPI_year)

tempfile cpi_data
quietly {
    clear
    input int year double cpi_u
    2011 224.939
    2012 229.594
    2013 232.957
    2014 236.736
    2015 237.017
    2016 240.007
    2017 245.120
    2018 251.107
    2019 255.657
    end
    save `cpi_data'
}

* ============================================================
* [1] LOAD VIOLATIONS DATA
* ============================================================
display _newline "[1] Loading violations data..."

* EPA ECHO data: state names as first column (empty header), then violations-YYYY columns
* Stata import delimited converts dashes in column names to underscores
import delimited "/Users/keshavgoel/Research/establishments-data (2).csv", ///
    clear varnames(1) bindquote(strict)

* First column is state names (the pandas index); rename from auto-assigned v1
rename v1 state

display "  EPA ECHO data loaded: " c(N) " rows, " c(k) " columns"

* Also load WPS data (loaded for completeness; not used in model fitting)
preserve
import delimited "/Users/keshavgoel/Research/wps-data (2).csv", ///
    clear varnames(1) bindquote(strict)
display "  WPS data loaded: " c(N) " rows, " c(k) " columns"
restore

* ============================================================
* [2] RESHAPE VIOLATIONS TO LONG FORMAT
* ============================================================
display _newline "[2] Reshaping violations data to long format..."

* Clean up state names (strip leading/trailing whitespace; fix known variants)
replace state = strtrim(state)
replace state = "Massachusetts" if state == "Massachusetts "
replace state = "Oregon"        if state == "Oregon "

* Stata replaces the dash in "violations-YYYY" with underscore on import,
* yielding violations_2011 ... violations_2019. Reshape using that stub.
reshape long violations_, i(state) j(year)
rename violations_ violations

* ============================================================
* [3] FILTER TO 50 US STATES
* ============================================================
display _newline "[3] Filtering to 50 US states only..."

* Keep only rows whose state matches one of the 50 state full names
generate keep50 = 0
foreach s in "Alabama" "Alaska" "Arizona" "Arkansas" "California" ///
    "Colorado" "Connecticut" "Delaware" "Florida" "Georgia" ///
    "Hawaii" "Idaho" "Illinois" "Indiana" "Iowa" ///
    "Kansas" "Kentucky" "Louisiana" "Maine" "Maryland" ///
    "Massachusetts" "Michigan" "Minnesota" "Mississippi" "Missouri" ///
    "Montana" "Nebraska" "Nevada" "New Hampshire" "New Jersey" ///
    "New Mexico" "New York" "North Carolina" "North Dakota" "Ohio" ///
    "Oklahoma" "Oregon" "Pennsylvania" "Rhode Island" "South Carolina" ///
    "South Dakota" "Tennessee" "Texas" "Utah" "Vermont" ///
    "Virginia" "Washington" "West Virginia" "Wisconsin" "Wyoming" {
        replace keep50 = 1 if state == "`s'"
}
keep if keep50 == 1
drop keep50

count
display "  Observations after filtering: " r(N)
quietly levelsof state
display "  Unique states: " r(r)

* ============================================================
* [4] CONSTRUCT TIME VARIABLES (CENTERED ON 2017)
* ============================================================
display _newline "[4] Constructing time variables centered on 2017..."

generate time  = year - 2017
generate time2 = time^2
generate time3 = time^3

display _newline "  Time variable mapping:"
quietly levelsof year, local(years_list)
foreach yr of local years_list {
    local t = `yr' - 2017
    display "    `yr': time = `t', time^2 = " `t'^2 ", time^3 = " `t'^3
}

* ============================================================
* [5] CLEAN VIOLATIONS DATA
* ============================================================
display _newline "[5] Cleaning violations data..."

destring violations, replace force
count if missing(violations)
display "  Missing violation values: " r(N) " (will be dropped)"
drop if missing(violations)

encode state, generate(state_id)

count
quietly levelsof state
display "  Final violations dataset: " c(N) " observations, " r(r) " states"

tempfile df_model
save `df_model'

* ============================================================
* [6] LOAD AND PROCESS SPENDING DATA
* ============================================================
display _newline "[6] Loading and processing spending data..."

import delimited "/Users/keshavgoel/Research/spending_data_master(in) (1).csv", ///
    clear varnames(1) bindquote(strict)

display "  Raw spending data: " c(N) " rows"

* --- 6a. Filter to study years 2011-2019 ---
* Column "Year" may import as "year" (lowercase) depending on CSV header
capture rename Year year_raw
capture rename year year_raw
keep if year_raw >= 2011 & year_raw <= 2019
rename year_raw year

display "  After filtering to 2011-2019: " c(N) " rows"

* --- 6b. Map state abbreviations to full names; keep only 50 states ---
* "State" column may import as "state" or "State"
capture rename State state_abbrev
capture rename state state_abbrev

generate state = ""
replace state = "Alabama"        if state_abbrev == "AL"
replace state = "Alaska"         if state_abbrev == "AK"
replace state = "Arizona"        if state_abbrev == "AZ"
replace state = "Arkansas"       if state_abbrev == "AR"
replace state = "California"     if state_abbrev == "CA"
replace state = "Colorado"       if state_abbrev == "CO"
replace state = "Connecticut"    if state_abbrev == "CT"
replace state = "Delaware"       if state_abbrev == "DE"
replace state = "Florida"        if state_abbrev == "FL"
replace state = "Georgia"        if state_abbrev == "GA"
replace state = "Hawaii"         if state_abbrev == "HI"
replace state = "Idaho"          if state_abbrev == "ID"
replace state = "Illinois"       if state_abbrev == "IL"
replace state = "Indiana"        if state_abbrev == "IN"
replace state = "Iowa"           if state_abbrev == "IA"
replace state = "Kansas"         if state_abbrev == "KS"
replace state = "Kentucky"       if state_abbrev == "KY"
replace state = "Louisiana"      if state_abbrev == "LA"
replace state = "Maine"          if state_abbrev == "ME"
replace state = "Maryland"       if state_abbrev == "MD"
replace state = "Massachusetts"  if state_abbrev == "MA"
replace state = "Michigan"       if state_abbrev == "MI"
replace state = "Minnesota"      if state_abbrev == "MN"
replace state = "Mississippi"    if state_abbrev == "MS"
replace state = "Missouri"       if state_abbrev == "MO"
replace state = "Montana"        if state_abbrev == "MT"
replace state = "Nebraska"       if state_abbrev == "NE"
replace state = "Nevada"         if state_abbrev == "NV"
replace state = "New Hampshire"  if state_abbrev == "NH"
replace state = "New Jersey"     if state_abbrev == "NJ"
replace state = "New Mexico"     if state_abbrev == "NM"
replace state = "New York"       if state_abbrev == "NY"
replace state = "North Carolina" if state_abbrev == "NC"
replace state = "North Dakota"   if state_abbrev == "ND"
replace state = "Ohio"           if state_abbrev == "OH"
replace state = "Oklahoma"       if state_abbrev == "OK"
replace state = "Oregon"         if state_abbrev == "OR"
replace state = "Pennsylvania"   if state_abbrev == "PA"
replace state = "Rhode Island"   if state_abbrev == "RI"
replace state = "South Carolina" if state_abbrev == "SC"
replace state = "South Dakota"   if state_abbrev == "SD"
replace state = "Tennessee"      if state_abbrev == "TN"
replace state = "Texas"          if state_abbrev == "TX"
replace state = "Utah"           if state_abbrev == "UT"
replace state = "Vermont"        if state_abbrev == "VT"
replace state = "Virginia"       if state_abbrev == "VA"
replace state = "Washington"     if state_abbrev == "WA"
replace state = "West Virginia"  if state_abbrev == "WV"
replace state = "Wisconsin"      if state_abbrev == "WI"
replace state = "Wyoming"        if state_abbrev == "WY"

keep if state != ""
display "  After filtering to 50 states: " c(N) " rows"
quietly levelsof state
display "  States with spending data: " r(r)

* --- 6c. Handle negative obligations (deobligations are kept in the sum) ---
* "Total Obligation" imports with spaces removed or underscored
capture rename totalobligation total_obligation
capture rename `"total obligation"' total_obligation
count if total_obligation < 0
display "  Negative obligation rows (deobligations, kept in sum): " r(N)

* --- 6d. Aggregate to state-year level (sum of all grants) ---
collapse (sum) spending_nominal = total_obligation, by(state year)
display "  State-year spending observations: " c(N)

* --- 6e. Inflate to 2017 dollars using CPI-U ---
* spending_2017 = spending_nominal × (245.120 / CPI_year)
merge m:1 year using `cpi_data', keep(match master) nogen

generate cpi_deflator  = 245.120 / cpi_u
generate spending_2017 = spending_nominal * cpi_deflator
generate spending_2017m = spending_2017 / 1000000       // scale to millions

display _newline "  CPI-U deflators applied (base year 2017):"
list year cpi_u cpi_deflator if inrange(year, 2011, 2019), noobs separator(0)

display _newline "  Spending summary (2017 dollars, millions) by year:"
tabstat spending_2017m, by(year) stat(mean sum min max n) nototal

export delimited "/Users/keshavgoel/Research/spending_aggregated.csv", replace
display "  Saved: spending_aggregated.csv"

tempfile spending_agg
save `spending_agg'

* ============================================================
* [7] MERGE VIOLATIONS AND SPENDING
* ============================================================
display _newline "[7] Merging violations and spending data..."

use `df_model', clear

merge m:1 state year using `spending_agg', keepusing(spending_2017 spending_2017m) nogen

count if missing(spending_2017m)
display "  Observations missing spending data: " r(N)

* Drop rows where spending is missing so both models use the same sample
drop if missing(spending_2017m)

* Re-encode state as numeric factor for mixed-effects estimation
drop state_id
encode state, generate(state_id)

count
quietly levelsof state
display "  Final analytic dataset (all variables present): " c(N) ///
    " observations, " r(r) " states"

* ============================================================
* DESCRIPTIVE STATISTICS
* ============================================================
display _newline "{hline 70}"
display "DESCRIPTIVE STATISTICS"
display "{hline 70}"

display _newline "Violations summary:"
summarize violations, detail

display _newline "Spending summary (2017 $, millions):"
summarize spending_2017m, detail

display _newline "Violations by year:"
tabstat violations, by(year) stat(mean sd min max n) nototal

display _newline "Spending by year (2017 $M):"
tabstat spending_2017m, by(year) stat(mean sd min max) nototal

* Save analytic dataset
export delimited "/Users/keshavgoel/Research/model_data_long.csv", replace
display "  Saved: model_data_long.csv"

* ============================================================
* FIT MODELS
* ============================================================
display _newline "{hline 70}"
display "FITTING HIERARCHICAL MIXED-EFFECTS MODELS"
display "{hline 70}"

display _newline "All models use:"
display "  Outcome:        violations"
display "  Grouping:       state (50 levels)"
display "  Random effects: random intercept + random slope for time, by state"
display "  Estimation:     REML"
display ""
display "  Model 1 (baseline):   violations ~ time + time^2 + time^3"
display "  Model 2 (+ spending): violations ~ time + time^2 + time^3 + spending_2017m"

* --- Model 1: Time only (baseline) ---
display _newline "{hline 70}"
display "MODEL 1: Baseline — Time Terms Only"
display "  violations ~ time + time^2 + time^3 + (1 + time | state)"
display "{hline 70}"

mixed violations time time2 time3 || state: time, reml covariance(unstructured)
estimates store model1

* --- Model 2: Add inflation-adjusted spending ---
display _newline "{hline 70}"
display "MODEL 2: With Spending — Time Terms + Inflation-Adjusted Spending"
display "  violations ~ time + time^2 + time^3 + spending_2017m + (1 + time | state)"
display "{hline 70}"

mixed violations time time2 time3 spending_2017m || state: time, reml covariance(unstructured)
estimates store model2

* ============================================================
* DETAILED RESULTS — MODEL 2
* ============================================================
display _newline "{hline 70}"
display "DETAILED RESULTS — MODEL 2 (WITH SPENDING)"
display "{hline 70}"

estimates restore model2

display _newline "--- FIXED EFFECTS ---"
display "(National-average effects, holding all else constant)"
display ""
display "  Intercept:       " %10.4f _b[_cons]
display "    -> Expected violations at time=0 (2017) for a state with 0 spending"
display ""
display "  time:            " %10.4f _b[time]
display ""
display "  time^2:          " %10.4f _b[time2]
display ""
display "  time^3:          " %10.4f _b[time3]
display ""
display "  spending_2017m:  " %10.4f _b[spending_2017m]
display "    -> Each additional $1M in 2017-adjusted agricultural spending"

display _newline "--- RANDOM EFFECTS ---"
display "(Between-state variation)"
estat recovariance

display _newline "--- VARIANCE DECOMPOSITION ---"
estat icc

* ============================================================
* MODEL COMPARISON (Likelihood Ratio Test)
* ============================================================
display _newline "{hline 70}"
display "MODEL COMPARISON"
display "{hline 70}"

lrtest model1 model2

* ============================================================
* STATE-SPECIFIC RANDOM EFFECTS (BLUPs)
* ============================================================
display _newline "{hline 70}"
display "STATE-SPECIFIC RANDOM EFFECTS — MODEL 2 (BLUPs)"
display "{hline 70}"

estimates restore model2

* Predict random effects (BLUPs): intercept deviation and slope deviation
predict re_int re_slope, reffects

* Collapse to one row per state
preserve
collapse (mean) random_intercept = re_int random_slope_time = re_slope, by(state)
gsort -random_intercept

display _newline "  States with HIGHEST baseline violations:"
list state random_intercept random_slope_time in 1/10, noobs separator(0)

display _newline "  States with LOWEST baseline violations:"
list state random_intercept random_slope_time in -10/-1, noobs separator(0)

export delimited "/Users/keshavgoel/Research/state_random_effects.csv", replace
display "  Saved: state_random_effects.csv"
restore

* ============================================================
* PREDICTED NATIONAL TREND (Model 2, at mean spending)
* ============================================================
display _newline "{hline 70}"
display "PREDICTED NATIONAL TREND — MODEL 2"
display "{hline 70}"

estimates restore model2

* Fixed effects
local b_intercept = _b[_cons]
local b_time      = _b[time]
local b_time2     = _b[time2]
local b_time3     = _b[time3]
local b_spend     = _b[spending_2017m]

* Mean spending across the analytic sample
quietly summarize spending_2017m
local mean_spend = r(mean)
display "  Predictions evaluated at mean spending = $" %9.3f `mean_spend' "M (2017 $)"

* Actual mean violations by year
preserve
collapse (mean) actual_mean = violations, by(year)
tempfile actuals
save `actuals'
restore

* Build prediction frame (one row per year 2011-2019)
preserve
clear
input int year
2011
2012
2013
2014
2015
2016
2017
2018
2019
end

generate time  = year - 2017
generate time2 = time^2
generate time3 = time^3
generate predicted_violations = `b_intercept'          ///
    + `b_time'  * time                                  ///
    + `b_time2' * time2                                 ///
    + `b_time3' * time3                                 ///
    + `b_spend' * `mean_spend'

merge 1:1 year using `actuals', nogen

display _newline "  Year | time | Predicted | Actual Mean"
display "  " "{hline 40}"
list year time predicted_violations actual_mean, noobs separator(0)

export delimited "/Users/keshavgoel/Research/predicted_trend.csv", replace
display "  Saved: predicted_trend.csv"
restore

display _newline "{hline 70}"
display "ANALYSIS COMPLETE"
display "{hline 70}"
