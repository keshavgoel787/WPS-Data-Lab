/*
Visualization of Polynomial Time Terms (time, time^2, time^3)
Shows how each term contributes to the overall predicted trend.

Fixed effects from Model 2 are hardcoded (same as Python version):
  Intercept = 16.2075
  beta_time  = 2.1927
  beta_time2 = -0.9532
  beta_time3 = -0.1831
*/

clear all
set more off
version 14

set scheme s1color

* ============================================================
* SETUP: Define fixed effects and compute all components
* ============================================================
local intercept  = 16.2075
local b1         =  2.1927
local b2         = -0.9532
local b3         = -0.1831

* Build a dataset with one row per year 2011-2019
clear
input int year double actual_mean
2011  6.66
2012  8.46
2013  7.00
2014  6.83
2015  5.96
2016 14.19
2017 20.57
2018 16.67
2019 15.54
end

* Centered time variable
generate time  = year - 2017
generate time2 = time^2
generate time3 = time^3

* Individual term contributions
generate comp_intercept  = `intercept'
generate comp_linear     = `b1' * time
generate comp_quadratic  = `b2' * time2
generate comp_cubic      = `b3' * time3

* Cumulative (step-by-step) build-up
generate cum_intercept   = comp_intercept
generate cum_linear      = cum_intercept + comp_linear
generate cum_quadratic   = cum_linear    + comp_quadratic
generate cum_total       = cum_quadratic + comp_cubic     // final prediction

tempfile poly_data
save `poly_data'

* ============================================================
* FIGURE 1: Cumulative Area Chart (Stacked Contribution)
* ============================================================
display "Saving: fig_polynomial_stacked.png"

* Stata area plots stack automatically when using area() overlaps.
* We approximate the stacked fill by plotting successive bands.
twoway ///
    (area comp_intercept year, ///
        fcolor("46 134 171%30") lwidth(none) ///
        legend(label(1 "Intercept = `intercept'"))) ///
    (rarea cum_intercept cum_linear year, ///
        fcolor("233 79 55%30") lwidth(none) ///
        legend(label(2 "+ time (b=`b1')"))) ///
    (rarea cum_linear cum_quadratic year, ///
        fcolor("246 174 45%30") lwidth(none) ///
        legend(label(3 "+ time^2 (b=`b2')"))) ///
    (rarea cum_quadratic cum_total year, ///
        fcolor("134 186 144%30") lwidth(none) ///
        legend(label(4 "+ time^3 (b=`b3')"))) ///
    (connected cum_total year, ///
        lcolor(black) lwidth(vthick) msymbol(circle) mcolor(black) msize(medlarge) ///
        legend(label(5 "Total Prediction"))) ///
    (connected actual_mean year, ///
        lcolor(purple) lwidth(medthick) lpattern(dash) msymbol(square) ///
        mcolor(purple) msize(medlarge) legend(label(6 "Actual Mean"))), ///
    xline(2017, lcolor(gray) lpattern(dot)) ///
    xlabel(2011(1)2019, angle(45)) ///
    xtitle("Year", size(medlarge)) ytitle("Violations", size(medlarge)) ///
    title("How Polynomial Terms Build the Prediction" ///
          "(Cumulative Contribution of Each Term)", size(medlarge)) ///
    legend(order(1 2 3 4 5 6) rows(2) position(6) size(small))

graph export "/Users/keshavgoel/Research/fig_polynomial_stacked.png", ///
    replace width(1800) height(1050)
display "Saved: fig_polynomial_stacked.png"

* ============================================================
* FIGURE 2: Individual Component Contributions (4-Panel)
* ============================================================
display "Saving: fig_polynomial_components.png"

use `poly_data', clear

* Panel A: Linear term
generate zero = 0

twoway ///
    (bar comp_linear year, barwidth(0.6) color("233 79 55%70") lcolor(white)) ///
    (scatter comp_linear year, msymbol(none) ///
        mlabel(comp_linear) mlabformat(%5.1f) mlabsize(vsmall) ///
        mlabvposition(12) mlabcolor(black)), ///
    yline(0, lcolor(black) lwidth(thin)) ///
    xline(2017, lcolor(gray) lpattern(dot)) ///
    xlabel(2011(1)2019, angle(45) labsize(small)) ///
    xtitle("Year", size(small)) ytitle("Contribution to Prediction", size(small)) ///
    title("A. Linear Term: b1 x time" "(b1 = `b1')", size(small) fweight(bold)) ///
    legend(off)
graph save "/tmp/comp_a.gph", replace

* Panel B: Quadratic term
twoway ///
    (bar comp_quadratic year, barwidth(0.6) color("246 174 45%70") lcolor(white)) ///
    (scatter comp_quadratic year, msymbol(none) ///
        mlabel(comp_quadratic) mlabformat(%5.1f) mlabsize(vsmall) ///
        mlabvposition(6) mlabcolor(black)), ///
    yline(0, lcolor(black) lwidth(thin)) ///
    xline(2017, lcolor(gray) lpattern(dot)) ///
    xlabel(2011(1)2019, angle(45) labsize(small)) ///
    xtitle("Year", size(small)) ytitle("Contribution to Prediction", size(small)) ///
    title("B. Quadratic Term: b2 x time^2" "(b2 = `b2')", size(small) fweight(bold)) ///
    legend(off)
graph save "/tmp/comp_b.gph", replace

* Panel C: Cubic term (color by sign)
generate cubic_pos = comp_cubic if comp_cubic >= 0
generate cubic_neg = comp_cubic if comp_cubic < 0

twoway ///
    (bar cubic_pos year, barwidth(0.6) color("134 186 144%70") lcolor(white)) ///
    (bar cubic_neg year, barwidth(0.6) color("233 79 55%70") lcolor(white)) ///
    (scatter comp_cubic year, msymbol(none) ///
        mlabel(comp_cubic) mlabformat(%5.1f) mlabsize(vsmall) ///
        mlabvposition(12) mlabcolor(black)), ///
    yline(0, lcolor(black) lwidth(thin)) ///
    xline(2017, lcolor(gray) lpattern(dot)) ///
    xlabel(2011(1)2019, angle(45) labsize(small)) ///
    xtitle("Year", size(small)) ytitle("Contribution to Prediction", size(small)) ///
    title("C. Cubic Term: b3 x time^3" "(b3 = `b3')", size(small) fweight(bold)) ///
    legend(off)
graph save "/tmp/comp_c.gph", replace

* Panel D: All components side-by-side (grouped bar)
* Reshape to long for grouped bar chart
preserve
keep year comp_linear comp_quadratic comp_cubic cum_total
generate sum_terms = comp_linear + comp_quadratic + comp_cubic

* Use catplot or graph bar over two variables
* Reshape: one row per year-component
rename comp_linear    term1
rename comp_quadratic term2
rename comp_cubic     term3
rename sum_terms      term4

reshape long term, i(year) j(term_id)

label define term_lbl 1 "time (b=`b1')" 2 "time^2 (b=`b2')" ///
                      3 "time^3 (b=`b3')" 4 "Sum (excl. intercept)"
label values term_id term_lbl

graph bar term, over(term_id, label(angle(30) labsize(vsmall))) over(year) ///
    bar(1, color("233 79 55%70")) bar(2, color("246 174 45%70")) ///
    bar(3, color("134 186 144%70")) bar(4, color("46 134 171%70")) ///
    yline(0, lcolor(black) lwidth(thin)) ///
    ytitle("Contribution to Prediction", size(small)) ///
    title("D. All Components Side-by-Side" "(Excluding Intercept)", size(small) fweight(bold)) ///
    legend(order(1 2 3 4) rows(2) size(vsmall) position(6))
restore

graph save "/tmp/comp_d.gph", replace

graph combine "/tmp/comp_a.gph" "/tmp/comp_b.gph" ///
              "/tmp/comp_c.gph" "/tmp/comp_d.gph", ///
    cols(2) rows(2) ///
    title("Decomposition of Polynomial Time Terms", ///
          size(medlarge) fweight(bold)) ///
    xsize(14) ysize(10)

graph export "/Users/keshavgoel/Research/fig_polynomial_components.png", ///
    replace width(2100) height(1500)
display "Saved: fig_polynomial_components.png"

* ============================================================
* FIGURE 3: Smooth Curves Showing Each Term's Shape
* ============================================================
display "Saving: fig_polynomial_curves.png"

* Create a smooth grid of 100 points from time=-6 to time=+2
clear
local n = 100
local tmin = -6
local tmax = 2
set obs `n'
generate time_smooth = `tmin' + (_n - 1) * (`tmax' - `tmin') / (`n' - 1)
generate year_smooth = time_smooth + 2017

generate linear_smooth    = `b1' * time_smooth
generate quadratic_smooth = `b2' * time_smooth^2
generate cubic_smooth     = `b3' * time_smooth^3
generate total_smooth     = `intercept' + linear_smooth + quadratic_smooth + cubic_smooth
generate sum_poly_smooth  = linear_smooth + quadratic_smooth + cubic_smooth

tempfile smooth_data
save `smooth_data'

* Merge discrete data points for scatter overlay
use `poly_data', clear
rename comp_linear    disc_linear
rename comp_quadratic disc_quadratic
rename comp_cubic     disc_cubic
keep year disc_linear disc_quadratic disc_cubic
generate year_smooth = year

merge 1:1 year_smooth using `smooth_data', nogen

twoway ///
    (line linear_smooth    year_smooth, lcolor("233 79 55") lwidth(medthick)) ///
    (line quadratic_smooth year_smooth, lcolor("246 174 45") lwidth(medthick)) ///
    (line cubic_smooth     year_smooth, lcolor("134 186 144") lwidth(medthick)) ///
    (line sum_poly_smooth  year_smooth, lcolor("46 134 171") lwidth(medthick) lpattern(dash)) ///
    (scatter disc_linear    year, mcolor("233 79 55") msize(medlarge) msymbol(circle) mlwidth(medthick)) ///
    (scatter disc_quadratic year, mcolor("246 174 45") msize(medlarge) msymbol(circle) mlwidth(medthick)) ///
    (scatter disc_cubic     year, mcolor("134 186 144") msize(medlarge) msymbol(circle) mlwidth(medthick)), ///
    yline(0, lcolor(black) lwidth(thin) lpattern(solid)) ///
    xline(2017, lcolor(gray) lpattern(dot)) ///
    xlabel(2011(1)2019, angle(45)) ///
    xscale(range(2010.5 2019.5)) ///
    xtitle("Year", size(medlarge)) ///
    ytitle("Contribution to Prediction (excl. intercept)", size(medlarge)) ///
    title("Shape of Each Polynomial Term" ///
          "(How each term changes across years)", size(medlarge)) ///
    legend(order(1 "Linear: `b1' x time" ///
                 2 "Quadratic: `b2' x time^2" ///
                 3 "Cubic: `b3' x time^3" ///
                 4 "Sum of time terms") ///
           rows(2) position(6) size(small))

graph export "/Users/keshavgoel/Research/fig_polynomial_curves.png", ///
    replace width(1800) height(1050)
display "Saved: fig_polynomial_curves.png"

* ============================================================
* FIGURE 4: Building the Prediction Step-by-Step
* ============================================================
display "Saving: fig_polynomial_buildup.png"

use `poly_data', clear

twoway ///
    (connected cum_intercept year, ///
        lcolor("46 134 171") lwidth(medthick) msymbol(circle) mcolor("46 134 171") msize(medlarge) ///
        legend(label(1 "Step 1: Intercept only (`intercept')"))) ///
    (connected cum_linear year, ///
        lcolor("233 79 55") lwidth(medthick) msymbol(square) mcolor("233 79 55") msize(medlarge) ///
        legend(label(2 "Step 2: + Linear term"))) ///
    (connected cum_quadratic year, ///
        lcolor("246 174 45") lwidth(medthick) msymbol(triangle) mcolor("246 174 45") msize(medlarge) ///
        legend(label(3 "Step 3: + Quadratic term"))) ///
    (connected cum_total year, ///
        lcolor("134 186 144") lwidth(vthick) msymbol(diamond) mcolor("134 186 144") msize(vlarge) ///
        legend(label(4 "Step 4: + Cubic term (FINAL)"))) ///
    (connected actual_mean year, ///
        lcolor(purple) lwidth(medthick) lpattern(dash) msymbol(star) mcolor(purple) msize(vlarge) ///
        legend(label(5 "Actual Mean Violations"))), ///
    xline(2017, lcolor(gray) lpattern(dot)) ///
    xlabel(2011(1)2019, angle(45)) ///
    xtitle("Year", size(medlarge)) ytitle("Predicted Violations", size(medlarge)) ///
    title("Building the Prediction: Adding Each Polynomial Term" ///
          "(From Intercept to Full Model)", size(medlarge)) ///
    legend(order(1 2 3 4 5) rows(2) position(6) size(small))

graph export "/Users/keshavgoel/Research/fig_polynomial_buildup.png", ///
    replace width(1800) height(1050)
display "Saved: fig_polynomial_buildup.png"

* ============================================================
* FIGURE 5: Calculation Table (displayed via list + exported)
* ============================================================
display "Saving: fig_polynomial_table.png"

use `poly_data', clear

* Compute derived values
generate b1_x_time  = `b1' * time
generate b2_x_time2 = `b2' * time2
generate b3_x_time3 = `b3' * time3
generate poly_sum   = b1_x_time + b2_x_time2 + b3_x_time3
generate intercept_plus_sum = `intercept' + poly_sum

format b1_x_time b2_x_time2 b3_x_time3 poly_sum intercept_plus_sum actual_mean %8.2f

display _newline "Polynomial Term Calculation Table"
display "(Yellow row = 2017, the reference year where time=0)"
display "{hline 95}"
display %6s "Year" %7s "time" %7s "time^2" %7s "time^3" ///
    %10s "b1*t" %11s "b2*t^2" %11s "b3*t^3" ///
    %10s "Sum" %14s "Int+Sum" %10s "Actual"
display "{hline 95}"
list year time time2 time3 ///
    b1_x_time b2_x_time2 b3_x_time3 ///
    poly_sum intercept_plus_sum actual_mean, ///
    noobs separator(0) clean

* Export the table as a dataset (CSV), since Stata can't render HTML tables as PNG natively
export delimited year time time2 time3 b1_x_time b2_x_time2 b3_x_time3 ///
    poly_sum intercept_plus_sum actual_mean ///
    using "/Users/keshavgoel/Research/polynomial_table.csv", ///
    replace

* Create a simple tabular visualization using a dot chart as proxy
twoway ///
    (scatter actual_mean year,          msymbol(none) mlabel(actual_mean) mlabcolor(purple)        mlabsize(small) mlabvposition(12)) ///
    (scatter intercept_plus_sum year,   msymbol(none) mlabel(intercept_plus_sum) mlabcolor(black)  mlabsize(small) mlabvposition(6)) ///
    (connected intercept_plus_sum year, lcolor(black) lwidth(medthick) msymbol(diamond) mcolor(black) msize(medlarge)) ///
    (connected actual_mean year,        lcolor(purple) lwidth(medthick) lpattern(dash) msymbol(star) mcolor(purple) msize(medlarge)), ///
    xline(2017, lcolor("255 210 0") lwidth(thick)) ///
    xlabel(2011(1)2019, angle(45)) ///
    xtitle("Year", size(medlarge)) ytitle("Violations", size(medlarge)) ///
    title("Polynomial Term Calculation Table" ///
          "(Yellow line = 2017, reference year where time=0)", size(medlarge)) ///
    legend(order(3 "Predicted (Int+Sum)" 4 "Actual Mean") rows(1) position(6) size(small))

graph export "/Users/keshavgoel/Research/fig_polynomial_table.png", ///
    replace width(1800) height(1050)
display "Saved: fig_polynomial_table.png"
display "(Full numeric table also saved: polynomial_table.csv)"

* ============================================================
* Print interpretation summary
* ============================================================
display _newline "{hline 60}"
display "POLYNOMIAL TERM VISUALIZATIONS COMPLETE"
display "{hline 60}"
display _newline "Files created:"
display "  1. fig_polynomial_stacked.png    - Cumulative area chart"
display "  2. fig_polynomial_components.png - 4-panel component breakdown"
display "  3. fig_polynomial_curves.png     - Smooth curves showing term shapes"
display "  4. fig_polynomial_buildup.png    - Step-by-step prediction building"
display "  5. fig_polynomial_table.png      - Calculation chart"
display "  6. polynomial_table.csv          - Full numeric table"

display _newline "{hline 60}"
display "INTERPRETATION SUMMARY"
display "{hline 60}"

display _newline "The polynomial model:"
display "  violations = `intercept' + `b1'*time + `b2'*time^2 + `b3'*time^3"
display ""
display "Each term's role:"
display "{hline 60}"
display ""
display "1. INTERCEPT (`intercept'):"
display "   Baseline prediction at time=0 (year 2017)"
display "   This is the anchor of the prediction"
display ""
display "2. LINEAR TERM (b1 = `b1'):"
display "   Adds `b1' violations for each year after 2017"
display "   Subtracts " %5.2f abs(`b1') " violations for each year before 2017"
display "   Creates a straight upward slope"
display "   At 2011 (time=-6): contributes " %6.2f `b1' * -6
display "   At 2019 (time=+2): contributes " %6.2f `b1' * 2
display ""
display "3. QUADRATIC TERM (b2 = `b2'):"
display "   Negative coefficient -> INVERTED U-SHAPE"
display "   Pulls DOWN predictions at years far from 2017"
display "   At 2011 (time^2=36): contributes " %6.2f `b2' * 36
display "   At 2017 (time^2=0):  contributes 0"
display ""
display "4. CUBIC TERM (b3 = `b3'):"
display "   Negative coefficient -> ASYMMETRY"
display "   Positive contribution before 2017, negative after"
display "   At 2011 (time^3=-216): contributes " %6.2f `b3' * -216
display "   At 2019 (time^3=+8):  contributes " %6.2f `b3' * 8
display ""
display "COMBINED EFFECT:"
display "{hline 60}"
display "The three terms together create a curve that:"
display "  - Starts low in 2011-2014"
display "  - Rises sharply into 2016-2017"
display "  - Peaks around 2017-2018"
display "  - Declines after 2018"
display "  - The decline is steeper than the rise (asymmetry from cubic term)"
