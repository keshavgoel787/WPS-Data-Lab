# Jafari-aligned crossed random-effects count models.
# See docs/superpowers/specs/2026-09-10-jafari-crossed-re-design.md.
#
# Usage:
#   Rscript scripts/jafari_crossed_models.R <out_json> [--gaussian-only]
#
# There is NO panel argument: build_cells() opens BOTH
# data/generated/count_model_panel_{2019,2021}.csv by absolute path, because
# the dual-window design is not optional -- it is what separates the
# model-class effect from the data-source change (the 2019 establishments view
# and the 2021 WPS view are different measures, not a longer one).
#
# THIS SCRIPT FITS. It computes no derived statistic -- no tally, no
# percentage, no marginal probability. Those live in report_jafari_crossed.py
# so exactly one script owns each number.
#
# --gaussian-only fits ONLY the crossed-RE Gaussian round-trip models, whose
# coefficients and variance components must reproduce statsmodels.MixedLM.
# That is the validation gate for this arm's NEW random-effects structure;
# without it no count estimate here should be believed.

suppressPackageStartupMessages({
  library(glmmTMB)
  library(jsonlite)
})

options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: jafari_crossed_models.R <out_json> [--gaussian-only]")
out_json <- args[1]
gaussian_only <- "--gaussian-only" %in% args

`%||%` <- function(a, b) if (is.null(a)) b else a

SEED  <- 20260910L
N_SIM <- 2000L

CUBIC  <- c("time", "time2", "time3")
M2_ADD <- c("SPEND_APP_z", "SPEND_WORK_z", "lii_2017_z")
M3_ADD <- c(M2_ADD, "h2a_per_farmworker_z", "dol_demand_met_pct_z", "pct_flc_z")

# The ONE random-effects structure in this arm. Jafari fit random intercepts,
# crossed over state and industry; this project has no industry dimension, so
# time is substituted for it (PI direction, 2026-09-10). There is deliberately
# no fallback: the existing ladder needed rs/ri tier machinery only because
# (1 | state) was a convergence fallback from a mandated (1 + time | state).
CROSSED_RE <- "(1 | state) + (1 | year)"

PANEL <- function(yr) sprintf(
  "/Users/keshavgoel/Research/data/generated/count_model_panel_%d.csv", yr)

coef_list <- function(mat) {
  if (is.null(mat) || nrow(mat) == 0) return(structure(list(), names = character(0)))
  stats::setNames(lapply(seq_len(nrow(mat)), function(i) {
    list(b = mat[i, 1], se = mat[i, 2], z = mat[i, 3], p = mat[i, 4])
  }), rownames(mat))
}

fit_spec <- function(d, dv, rhs, family, zi = ~0, re = CROSSED_RE, REML = FALSE) {
  # The analytic sample is defined by complete.cases over the CONDITIONAL
  # predictors only. The spec requires ZI predictors to be a subset of them
  # (section 5.2), so every ZI tier within a (cell, model) shares one sample --
  # which is exactly what makes cross-tier AIC comparison valid (section 5.3).
  need <- unique(c(dv, rhs, "state", "year", "time"))
  need <- need[need %in% names(d)]
  dd <- d[stats::complete.cases(d[, need, drop = FALSE]), , drop = FALSE]
  dd$state <- factor(dd$state)
  dd$year  <- factor(dd$year)

  form <- stats::as.formula(sprintf("%s ~ %s + %s", dv,
                                    paste(rhs, collapse = " + "), re))

  warn_msgs <- character(0)
  fit <- tryCatch(
    withCallingHandlers(
      glmmTMB(form, ziformula = zi, family = family, data = dd, REML = REML),
      warning = function(w) {
        warn_msgs <<- c(warn_msgs, conditionMessage(w))
        invokeRestart("muffleWarning")
      }),
    error = function(e) e)

  base <- list(formula = deparse1(form), zi_formula = deparse1(zi),
               re_used = re, n_obs = nrow(dd),
               n_states = length(unique(dd$state)),
               n_years = length(unique(dd$year)))

  if (inherits(fit, "error")) {
    return(c(base, list(converged = FALSE, message = conditionMessage(fit))))
  }

  pd_hess   <- isTRUE(fit$sdr$pdHess)
  conv_code <- fit$fit$convergence
  sm    <- summary(fit)
  cond  <- coef_list(sm$coefficients$cond)
  zi_co <- coef_list(sm$coefficients$zi)
  finite_se <- length(cond) == 0 ||
    all(vapply(cond, function(x) is.finite(x$se), logical(1)))

  # Crossed REs: TWO separate grouping factors, each with its own variance.
  # There is no covariance between them by construction -- that is what
  # "crossed" means here -- so no sigma_u01 is emitted anywhere in this arm.
  vc_st <- tryCatch(VarCorr(fit)$cond$state, error = function(e) NULL)
  vc_yr <- tryCatch(VarCorr(fit)$cond$year,  error = function(e) NULL)
  s2_state <- if (is.null(vc_st)) NA_real_ else vc_st[1, 1]
  s2_year  <- if (is.null(vc_yr)) NA_real_ else vc_yr[1, 1]

  # The ZI component's own state random-intercept variance. Reading only
  # VarCorr(fit)$cond and never $zi is the defect that made every reported
  # "structural-zero probability" in the previous arm the MEDIAN-STATE value
  # presented as marginal -- off by ~835x in the headline cell.
  vc_zi <- tryCatch(VarCorr(fit)$zi$state, error = function(e) NULL)
  s2_zi <- if (is.null(vc_zi)) NA_real_ else vc_zi[1, 1]

  fam_obj  <- if (is.function(family)) family() else family
  fam_name <- if (is.character(fam_obj)) fam_obj else fam_obj$family
  is_gaussian <- identical(fam_name, "gaussian")

  # sigma() is a residual SD ONLY for Gaussian. For nbinom2 it is theta, for
  # nbinom1 the dispersion multiplier. Exporting its square as `sigma2_e` for a
  # count fit puts a dispersion parameter under a name that reads as a variance
  # component -- a documented defect in the previous arm.
  disp <- tryCatch(sigma(fit), error = function(e) NA_real_)
  s2_e <- if (is_gaussian && is.finite(disp)) disp^2 else NA_real_

  # Year random-intercept BLUPs. These REPLACE the previous arm's COVID
  # indicator (spec 7.3): (1 | year) absorbs the pandemic shock by
  # construction, so a COVID dummy would compete with the term that already
  # contains it and be near-unidentifiable. The 2020/2021 BLUPs show the drop
  # directly.
  year_blups <- tryCatch({
    re_yr <- ranef(fit)$cond$year
    stats::setNames(as.list(as.numeric(re_yr[, 1])), rownames(re_yr))
  }, error = function(e) NULL)

  obs_zeros <- exp_zeros <- exp_zeros_se <- NA_real_
  sim_warn <- character(0)
  if (!is_gaussian) {
    obs_zeros <- sum(dd[[dv]] == 0, na.rm = TRUE)
    sims <- tryCatch(
      withCallingHandlers(
        as.data.frame(simulate(fit, nsim = N_SIM, seed = SEED)),
        warning = function(w) {
          sim_warn <<- c(sim_warn, conditionMessage(w))
          invokeRestart("muffleWarning")
        }),
      error = function(e) { sim_warn <<- c(sim_warn, conditionMessage(e)); NULL })
    if (!is.null(sims)) {
      zc <- colSums(sims == 0)
      exp_zeros <- mean(zc)
      exp_zeros_se <- stats::sd(zc) / sqrt(N_SIM)
    }
  }

  c(base, list(converged = pd_hess && conv_code == 0 && finite_se,
               message = if (length(warn_msgs)) paste(warn_msgs, collapse = " | ") else "",
               pd_hess = pd_hess, conv_code = conv_code,
               aic = AIC(fit), bic = BIC(fit),
               loglik = as.numeric(logLik(fit)), df = attr(logLik(fit), "df"),
               cond = cond, zi = zi_co,
               sigma2_state = s2_state, sigma2_year = s2_year,
               sigma2_zi_state = s2_zi,
               sigma_zi_state = if (is.na(s2_zi)) NA_real_ else sqrt(s2_zi),
               family_name = fam_name, dispersion = disp, sigma2_e = s2_e,
               year_blups = year_blups,
               obs_zeros = obs_zeros, exp_zeros = exp_zeros,
               exp_zeros_se = exp_zeros_se,
               sim_message = if (length(sim_warn))
                 paste(unique(sim_warn), collapse = " | ") else ""))
}

# One cell = one outcome x window. The violations-as-offset variant present in
# the previous arm is deliberately ABSENT: the log(inspections) offset is
# undefined at zero and therefore deletes every zero-inspection state-year --
# in 2011-2021 all 7 rows it drops are also zero-violation rows. An offset
# specification throws away the structural zeros this arm exists to model.
build_cells <- function() {
  list(
    insp_2019 = list(d = read.csv(PANEL(2019), stringsAsFactors = FALSE),
                     dv = "inspections", base = CUBIC,
                     window = 2019L, outcome = "inspections"),
    insp_2021 = list(d = read.csv(PANEL(2021), stringsAsFactors = FALSE),
                     dv = "inspections", base = CUBIC,
                     window = 2021L, outcome = "inspections"),
    viol_2019 = list(d = read.csv(PANEL(2019), stringsAsFactors = FALSE),
                     dv = "violations", base = c("log_inspections", CUBIC),
                     window = 2019L, outcome = "violations"),
    viol_2021 = list(d = read.csv(PANEL(2021), stringsAsFactors = FALSE),
                     dv = "violations", base = c("log_inspections", CUBIC),
                     window = 2021L, outcome = "violations"))
}

rhs_for <- function(cl, model) {
  switch(model,
         M1 = cl$base,
         M2 = c(cl$base, M2_ADD),
         M3 = c(cl$base, M3_ADD),
         stop("unknown model: ", model))
}

cells <- build_cells()
results <- list()

# ------------------------------------------------------------
# Crossed-RE Gaussian round-trip gate. Must reproduce statsmodels.MixedLM
# fitted with vc_formula for the SAME crossed structure. REML = TRUE because
# MixedLM uses REML and glmmTMB defaults to ML.
#
# The previous arm's Gaussian gate validated (1 + time | state). It says
# nothing about whether glmmTMB and statsmodels agree on a CROSSED structure,
# which is the structure every estimate in this arm rests on -- so this gate
# is new, not inherited. M1 and M3 are both fit so the level-2 covariates are
# themselves checked across the R/Python bridge.
# ------------------------------------------------------------
for (cell_name in names(cells)) {
  cl <- cells[[cell_name]]
  gdv <- if (cl$outcome == "inspections") "log_inspections" else "log_violations"
  gbase <- if (cl$outcome == "inspections") CUBIC else c("log_inspections", CUBIC)
  for (gmodel in c("M1", "M3")) {
    grhs <- if (gmodel == "M1") gbase else c(gbase, M3_ADD)
    key <- sprintf("%s__%s__gaussian", cell_name, gmodel)
    cat("fitting", key, "(crossed-RE Gaussian round-trip gate)\n")
    res <- fit_spec(cl$d, gdv, grhs, family = gaussian, REML = TRUE)
    res$cell <- cell_name; res$model <- gmodel; res$family_tag <- "gaussian"
    res$window <- cl$window; res$outcome <- cl$outcome
    res$gaussian_dv <- gdv; res$gaussian_rhs <- grhs
    results[[key]] <- res
  }
}

if (gaussian_only) {
  write_json(results, out_json, auto_unbox = TRUE, digits = 10, na = "null")
  cat("Gaussian round-trip written to", out_json, "\n")
  quit(status = 0)
}

write_json(results, out_json, auto_unbox = TRUE, digits = 10, na = "null")
cat("\nWrote", length(results), "entries to", out_json, "\n")
