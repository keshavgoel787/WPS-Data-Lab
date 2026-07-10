# Ridge / Spline Regression for the Inspections–Violations Relationship

**Task:** Read up on ridge and spline regression as tools for modeling the conditional relationship between inspections and violations.

## Why this matters for the project

Right now inspections and violations are fit as two separate outcomes (the `inspections_*` scripts mirror the violations scripts). But the two are mechanically linked: a state with more inspections will tend to surface more violations simply because there is more looking. Treating violation counts as a clean measure of non-compliance ignores this detection effect.

The open question is how to model violations *conditional on* inspection effort, so that the spending/covariate effects we report are not just proxies for how hard a state inspects. Two candidate tools:

- **Ridge regression** — when we want to keep a set of correlated predictors (spending denominators, labor covariates, inspection counts) in the model without unstable coefficients.
- **Spline regression** — when the violations-given-inspections curve is nonlinear (e.g. diminishing returns: the first inspections catch most violations, later ones add fewer).

## Ridge regression — what to read for

Ridge is L2-penalized least squares: it shrinks coefficients toward zero to trade a little bias for lower variance, which helps when predictors are collinear.

Read up on:
- The bias–variance tradeoff and why shrinkage helps with multicollinearity (relevant here — our spending variables share a numerator and the labor/DOL covariates are correlated, e.g. `dol_workers_cert` vs Yuri's `h2a_workers` at r≈0.83).
- Choosing the penalty λ by cross-validation.
- The need to standardize predictors first (we already z-score Level-2 covariates, so this fits the existing convention).
- **Mixed-model angle:** the random effects in `MixedLM` are already a form of shrinkage/ridge penalty on the state intercepts and slopes. Worth understanding how an explicit ridge penalty on the fixed effects would interact with that. Look at penalized/regularized mixed models (e.g. `glmmLasso`, `ggmix`, or ridge via the equivalence between random effects and penalized fixed effects).

## Spline regression — what to read for

Splines fit a smooth nonlinear curve by joining piecewise polynomials at knots, giving flexibility without committing to a global polynomial shape.

Read up on:
- Natural cubic splines vs B-splines vs restricted cubic splines, and how knot placement is chosen.
- **Penalized / smoothing splines and GAMs** — penalized splines pick smoothness automatically via a penalty, avoiding manual knot selection. This is probably the most relevant framing.
- How a spline term enters a regression: model violations as a smooth function of inspections, `violations ~ s(inspections) + covariates`, and read the fitted curve for diminishing returns or a saturation point.
- **Connection to the current model:** the cubic time polynomial (`time`, `time2`, `time3`) the project already uses is a fixed-form version of what a spline does more flexibly. A spline in `time` could be an alternative worth noting, but the primary target is a spline in inspections.

## Suggested reading

- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* — Ch. 3.4 (ridge/shrinkage), Ch. 5 (basis expansions and splines). Free PDF online.
- Harrell, *Regression Modeling Strategies* — restricted cubic splines, very applied.
- Wood, *Generalized Additive Models: An Introduction with R* — penalized splines / GAMs, the standard reference. The `mgcv` package docs are a good shorter entry point.
- James et al., *An Introduction to Statistical Learning* — gentler versions of the same ridge and spline chapters.
- For the mixed-model intersection: search "penalized splines as mixed models" (Ruppert, Wand & Carroll, *Semiparametric Regression*) — splines can be written as random effects, which ties directly back to the `MixedLM` setup here.

## Open questions to resolve while reading

- Is the goal (a) to *adjust* violations for inspection effort, or (b) to *describe* the inspections→violations curve? That changes whether inspections is a nuisance covariate or the focus.
- Python tooling: `statsmodels` has GAM/spline support (`statsmodels.gam`); `patsy` provides `bs()`/`cr()` spline bases usable inside the existing `MixedLM.from_formula` calls. Ridge for mixed models is thinner in Python and may push toward R (`mgcv`, `lme4` + penalties).
- This is flagged as pending Stephane's professor's advice on model form, so treat this as background reading rather than a committed modeling decision.
