/*
Visualizations for Hierarchical Mixed-Effects Model Results
Creates 6 visualizations to illustrate model findings

Requires: model_data_long.csv, state_random_effects.csv, predicted_trend.csv
Run hierarchical_violations_model.do first to generate those files.
*/

clear all
set more off
version 14

* Set a clean graph scheme (closest to seaborn-whitegrid)
set scheme s1color

display "Loading data..."

import delimited "/Users/keshavgoel/Research/model_data_long.csv", clear varnames(1)
tempfile df_model
save `df_model'

import delimited "/Users/keshavgoel/Research/state_random_effects.csv", clear varnames(1)
tempfile re_data
save `re_data'

import delimited "/Users/keshavgoel/Research/predicted_trend.csv", clear varnames(1)
tempfile pred_data
save `pred_data'

* ============================================================
* FIGURE 1: National Trend — Predicted vs Actual
* ============================================================
display "Creating Figure 1: National Trend..."

use `pred_data', clear

twoway ///
    (connected actual_mean year, ///
        mcolor("46 134 171") lcolor("46 134 171") msymbol(circle) ///
        msize(medlarge) lwidth(medthick) lpattern(solid) ///
        legend(label(1 "Actual Mean"))) ///
    (connected predicted_violations year, ///
        mcolor("233 79 55") lcolor("233 79 55") msymbol(square) ///
        msize(medium) lwidth(medthick) lpattern(dash) ///
        legend(label(2 "Model Prediction"))) ///
    (scatteri . 2017, xline(2017, lcolor(gray) lpattern(dot) lwidth(thin)) ///
        legend(label(3 "Reference Year (2017)"))), ///
    xlabel(2011(1)2019, angle(45)) ///
    ylabel(, angle(0)) ///
    xtitle("Year", size(medlarge)) ///
    ytitle("Mean Violations", size(medlarge)) ///
    title("National Trend: EPA Violations (2011-2019)" ///
          "Hierarchical Model Predictions vs Actual", size(medlarge)) ///
    legend(order(1 2) rows(1) position(6)) ///
    xline(2017, lcolor(gray) lpattern(dot))

graph export "/Users/keshavgoel/Research/fig1_national_trend.png", ///
    replace width(1500) height(900)
display "  Saved: fig1_national_trend.png"

* ============================================================
* FIGURE 2: State Random Effects (Caterpillar Plot)
* ============================================================
display "Creating Figure 2: State Random Effects..."

use `re_data', clear

* Create a numeric rank sorted by random_intercept (low to high)
sort random_intercept
generate rank = _n

* Color indicator: 1 = above average (positive), 0 = below average (negative)
generate above_avg = (random_intercept > 0)

twoway ///
    (rbar zero_line random_intercept rank if above_avg == 0, ///
        barwidth(0.6) color("46 134 171%80") horizontal ///
        legend(label(1 "Below Average"))) ///
    (rbar zero_line random_intercept rank if above_avg == 1, ///
        barwidth(0.6) color("233 79 55%80") horizontal ///
        legend(label(2 "Above Average"))), ///
    ytitle("") xtitle("Random Intercept (Deviation from National Average)", size(medlarge)) ///
    title("State-Level Random Effects (BLUPs)" ///
          "Deviation from National Baseline Violations", size(medlarge)) ///
    ylabel(1(1)50, valuelabel angle(0) labsize(vsmall)) ///
    xline(0, lcolor(black) lwidth(thin)) ///
    legend(order(2 1) rows(1) position(6))

* Add state labels to y-axis using value labels
quietly levelsof state, local(state_names)
label define state_lbl 0 " "
local r = 1
foreach s of local state_names {
    label define state_lbl `r' "`s'", add
    local ++r
}
label values rank state_lbl

graph export "/Users/keshavgoel/Research/fig2_state_random_effects.png", ///
    replace width(1200) height(1800)
display "  Saved: fig2_state_random_effects.png"

* ============================================================
* FIGURE 3: State-Level Trajectories (Spaghetti Plot)
* ============================================================
display "Creating Figure 3: State Trajectories..."

use `df_model', clear

* Encode state to numeric for byable looping
encode state, generate(state_id)

* Build the spaghetti: one line per state in gray
* Use twoway with over() emulation via a local macro
quietly levelsof state_id, local(all_states)

* Stack all state lines, then overlay national trend
local state_plots ""
foreach s of local all_states {
    local state_plots `"`state_plots' (line violations year if state_id == `s', lcolor(gray%30) lwidth(vthin))"'
}

* Load predicted trend into a tempfile for overlay
preserve
import delimited "/Users/keshavgoel/Research/predicted_trend.csv", clear varnames(1)
tempfile pred_overlay
save `pred_overlay'
restore

* Merge predictions onto model data for plotting
merge m:1 year using `pred_overlay', nogen keepusing(predicted_violations actual_mean)

* Plot all state lines + national trend overlay
twoway `state_plots' ///
    (line predicted_violations year, sort lcolor("233 79 55") lwidth(vthick) ///
        legend(label(`= wordcount("`all_states'") + 1' "National Trend (Model)"))) ///
    (line actual_mean year, sort lcolor("46 134 171") lwidth(vthick) lpattern(dash) ///
        legend(label(`= wordcount("`all_states'") + 2' "National Mean (Actual)"))), ///
    xlabel(2011(1)2019, angle(45)) ///
    xtitle("Year", size(medlarge)) ytitle("Violations", size(medlarge)) ///
    title("State-Level Violation Trajectories (2011-2019)" ///
          "Individual States (gray) vs National Trend", size(medlarge)) ///
    legend(order(`= wordcount("`all_states'") + 1' `= wordcount("`all_states'") + 2') rows(1) position(6))

graph export "/Users/keshavgoel/Research/fig3_state_trajectories.png", ///
    replace width(1500) height(1000)
display "  Saved: fig3_state_trajectories.png"

* ============================================================
* FIGURE 4: Random Intercept vs Random Slope Scatter
* ============================================================
display "Creating Figure 4: Random Effects Scatter..."

use `re_data', clear

* Check if random slopes are non-trivially non-zero
quietly summarize random_slope_time
local slope_range = r(max) - r(min)

if `slope_range' > 0.001 {
    * Compute 1.5 SD thresholds for labeling extremes
    quietly summarize random_intercept
    local sd_int = r(sd)
    quietly summarize random_slope_time
    local sd_slope = r(sd)

    generate extreme = (abs(random_intercept) > 1.5 * `sd_int') | ///
                       (abs(random_slope_time) > 1.5 * `sd_slope')

    twoway ///
        (scatter random_slope_time random_intercept if extreme == 0, ///
            mcolor("46 134 171%70") msymbol(circle) msize(medlarge)) ///
        (scatter random_slope_time random_intercept if extreme == 1, ///
            mcolor("46 134 171%70") msymbol(circle) msize(medlarge) ///
            mlabel(state) mlabsize(vsmall) mlabposition(1)), ///
        yline(0, lcolor(gray) lpattern(dash) lwidth(thin)) ///
        xline(0, lcolor(gray) lpattern(dash) lwidth(thin)) ///
        xtitle("Random Intercept (Baseline Deviation)", size(medlarge)) ///
        ytitle("Random Slope (Time Trend Deviation)", size(medlarge)) ///
        title("State Random Effects: Baseline vs Time Trend" ///
              "Each point represents a state", size(medlarge)) ///
        legend(off)
}
else {
    * If no meaningful random slopes, show distribution of intercepts
    histogram random_intercept, bin(15) ///
        color("46 134 171%70") fcolor("46 134 171%70") lcolor(white) ///
        xline(0, lcolor("233 79 55") lpattern(dash) lwidth(medthick)) ///
        xtitle("Random Intercept (Baseline Deviation)", size(medlarge)) ///
        ytitle("Frequency", size(medlarge)) ///
        title("Distribution of State Random Intercepts", size(medlarge))
}

graph export "/Users/keshavgoel/Research/fig4_random_effects_scatter.png", ///
    replace width(1200) height(1000)
display "  Saved: fig4_random_effects_scatter.png"

* ============================================================
* FIGURE 5: Violations Heatmap (States x Years)
* ============================================================
display "Creating Figure 5: Violations Heatmap..."

* Stata's heatplot (from SSC: ssc install heatplot) replicates seaborn heatmap
* If heatplot is not installed, an alternative using graph matrix is shown.

use `df_model', clear

* Compute state means for sorting
bysort state: egen state_mean_viol = mean(violations)

* Rank states from highest to lowest average violations
gsort -state_mean_viol state year
egen state_rank = group(state_mean_viol state), label  // unique numeric ID per state

capture which heatplot
if _rc == 0 {
    * heatplot is available (ssc install heatplot)
    heatplot violations state year, ///
        colors(YlOrRd) ///
        xlabel(2011(1)2019, angle(45)) ///
        xtitle("Year") ytitle("State") ///
        title("EPA Violations by State and Year (2011-2019)" ///
              "Heatmap sorted by average violations", size(medlarge))
}
else {
    * Fallback: reshape wide and display as a dotplot sorted by state mean
    display "  Note: install heatplot (ssc install heatplot) for a true heatmap."
    display "  Showing bubble chart approximation instead."

    twoway (scatter state_rank year [w=violations], ///
        msymbol(circle) mcolor("255 120 50%50") ///
        xlabel(2011(1)2019, angle(45)) ///
        xtitle("Year") ytitle("State (ranked by avg violations)") ///
        title("EPA Violations by State and Year (2011-2019)" ///
              "Bubble size = violations; sorted by average", size(medlarge)))
}

graph export "/Users/keshavgoel/Research/fig5_violations_heatmap.png", ///
    replace width(1200) height(2000)
display "  Saved: fig5_violations_heatmap.png"

* ============================================================
* FIGURE 6: ICC Visualization (Variance Decomposition)
* ============================================================
display "Creating Figure 6: Variance Decomposition..."

use `df_model', clear

* Estimate between-state and within-state variance from data
* Between-state variance: variance of state means
* Within-state variance: mean of per-state variances
bysort state: egen state_mean = mean(violations)
quietly {
    preserve
    collapse (mean) state_mean, by(state)
    summarize state_mean
    local between_var = r(Var)
    restore
}
bysort state: egen state_var = sd(violations)
replace state_var = state_var^2
quietly summarize state_var
local within_var = r(mean)

local total_var  = `between_var' + `within_var'
local icc        = `between_var' / `total_var'
local icc_pct    = string(`icc' * 100, "%5.1f")
local wicc_pct   = string((1 - `icc') * 100, "%5.1f")

display "  Between-state variance: " %9.3f `between_var'
display "  Within-state variance:  " %9.3f `within_var'
display "  ICC = " %6.4f `icc'

* Panel A: Pie chart of variance decomposition
graph pie, over(dummy_var) ///  // pie requires categorical; build the data manually
    plabel(_all percent, format(%5.1f)) angle0(90)

* Build a 2-row dataset for pie + bar charts
preserve
clear
input str40 component double variance
"Between-State Variance"    `between_var'
"Within-State (Residual)"   `within_var'
end

* Pie chart
graph pie variance, over(component) ///
    pie(1, color("46 134 171")) pie(2, color("163 206 241")) ///
    plabel(_all percent, format(%5.1f) size(medlarge)) ///
    angle0(90) ///
    title("Variance Decomposition" "(Intraclass Correlation)", size(medlarge)) ///
    legend(rows(2) size(small))

graph save "/tmp/fig6_pie.gph", replace

* Bar chart
graph bar variance, over(component, label(angle(15))) ///
    bar(1, color("46 134 171")) bar(2, color("163 206 241")) ///
    blabel(bar, format(%7.1f) size(small)) ///
    ytitle("Variance", size(medlarge)) ///
    title("Variance Components" "ICC = `icc_pct'%", size(medlarge))

graph save "/tmp/fig6_bar.gph", replace

graph combine "/tmp/fig6_pie.gph" "/tmp/fig6_bar.gph", ///
    cols(2) title("") xsize(12) ysize(5)
restore

graph export "/Users/keshavgoel/Research/fig6_variance_decomposition.png", ///
    replace width(1800) height(750)
display "  Saved: fig6_variance_decomposition.png"

* ============================================================
* BONUS: Combined Summary Figure (4 panels)
* ============================================================
display "Creating Bonus: Combined Summary Figure..."

* --- Panel A: National Trend ---
use `pred_data', clear

twoway ///
    (connected actual_mean year, ///
        mcolor("46 134 171") lcolor("46 134 171") msymbol(circle) ///
        msize(medium) lwidth(medthick) legend(label(1 "Actual Mean"))) ///
    (connected predicted_violations year, ///
        mcolor("233 79 55") lcolor("233 79 55") msymbol(square) ///
        msize(small) lwidth(medthick) lpattern(dash) ///
        legend(label(2 "Model Prediction"))), ///
    xline(2017, lcolor(gray) lpattern(dot)) ///
    xlabel(2011(1)2019, angle(45) labsize(small)) ///
    xtitle("Year", size(small)) ytitle("Mean Violations", size(small)) ///
    title("A. National Trend: Predicted vs Actual", size(small) fweight(bold)) ///
    legend(order(1 2) rows(1) size(vsmall) position(6))

graph save "/tmp/panel_a.gph", replace

* --- Panel B: Top 5 and Bottom 5 States ---
use `re_data', clear
sort random_intercept
generate rank = _n

* Keep only top 5 and bottom 5
generate keep_tb = (rank <= 5) | (rank >= _N - 4)
keep if keep_tb

sort random_intercept
generate rank2 = _n
generate above = (random_intercept > 0)

twoway ///
    (rbar zero_ref random_intercept rank2 if above == 0, ///
        barwidth(0.6) color("46 134 171%80") horizontal) ///
    (rbar zero_ref random_intercept rank2 if above == 1, ///
        barwidth(0.6) color("233 79 55%80") horizontal), ///
    xline(0, lcolor(black) lwidth(thin)) ///
    ylabel(1(1)10, valuelabel angle(0) labsize(vsmall)) ///
    xtitle("Random Intercept", size(small)) ///
    title("B. Top 5 and Bottom 5 States", size(small) fweight(bold)) ///
    legend(off)

graph save "/tmp/panel_b.gph", replace

* --- Panel C: State Trajectories ---
use `df_model', clear
merge m:1 year using `pred_data', nogen keepusing(predicted_violations)
encode state, generate(state_id)
quietly levelsof state_id, local(all_states)

local sp_plots ""
foreach s of local all_states {
    local sp_plots `"`sp_plots' (line violations year if state_id == `s', lcolor(gray%20) lwidth(vthin))"'
}

twoway `sp_plots' ///
    (line predicted_violations year, sort lcolor("233 79 55") lwidth(thick)), ///
    xlabel(2011(1)2019, angle(45) labsize(vsmall)) ///
    xtitle("Year", size(small)) ytitle("Violations", size(small)) ///
    title("C. State Trajectories with National Trend", size(small) fweight(bold)) ///
    legend(off)

graph save "/tmp/panel_c.gph", replace

* --- Panel D: Year-over-Year Changes with Variability ---
use `df_model', clear
collapse (mean) mean_viol = violations (sd) sd_viol = violations, by(year)
generate upper = mean_viol + sd_viol / 2
generate lower = mean_viol - sd_viol / 2

twoway ///
    (rarea upper lower year, fcolor("46 134 171%20") lwidth(none)) ///
    (rcap upper lower year, lcolor("46 134 171") lwidth(thin)) ///
    (connected mean_viol year, ///
        mcolor("46 134 171") lcolor("46 134 171") msymbol(circle) ///
        msize(medium) lwidth(medthick)), ///
    xlabel(2011(1)2019, angle(45) labsize(vsmall)) ///
    xtitle("Year", size(small)) ytitle("Mean Violations", size(small)) ///
    title("D. Violations by Year (with variability)", size(small) fweight(bold)) ///
    legend(off)

graph save "/tmp/panel_d.gph", replace

* Combine all 4 panels
graph combine "/tmp/panel_a.gph" "/tmp/panel_b.gph" ///
              "/tmp/panel_c.gph" "/tmp/panel_d.gph", ///
    cols(2) rows(2) ///
    title("Hierarchical Mixed-Effects Model: EPA Violations Analysis (2011-2019)", ///
          size(medlarge) fweight(bold)) ///
    xsize(14) ysize(12)

graph export "/Users/keshavgoel/Research/fig_summary_combined.png", ///
    replace width(2100) height(1800)
display "  Saved: fig_summary_combined.png"

display _newline "{hline 50}"
display "All visualizations created successfully!"
display "{hline 50}"
display _newline "Files saved:"
display "  1. fig1_national_trend.png"
display "  2. fig2_state_random_effects.png"
display "  3. fig3_state_trajectories.png"
display "  4. fig4_random_effects_scatter.png"
display "  5. fig5_violations_heatmap.png"
display "  6. fig6_variance_decomposition.png"
display "  7. fig_summary_combined.png (bonus: 4-panel summary)"
