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
