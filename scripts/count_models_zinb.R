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

  # Collect (not discard) every warning glmmTMB raises during the fit --
  # muffling them outright is the same idiom as the module-level
  # `warnings.filterwarnings('ignore')` in paper_table_models_2021.py that let
  # a non-converged fit become "ground truth" (see task-3-report.md). They are
  # surfaced in `message` below so Task 4's ZI/NB fits, which are more fragile
  # than this Gaussian gate, cannot silently hide a false-convergence or
  # boundary warning behind a numerically "converged" flag.
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

  # Family-name resolution robust to call style: `family` may arrive as a bare
  # function (gaussian), a called family object (gaussian(), nbinom2(link=
  # "log")), or a string ("gaussian"). identical(family, gaussian) only
  # matches the first of these and silently falls through -- running zero
  # simulation on a continuous DV, or skipping it for a count family called
  # with an explicit link -- for the other two.
  fam_obj <- if (is.function(family)) family() else family
  fam_name <- if (is.character(fam_obj)) fam_obj else fam_obj$family
  is_gaussian <- identical(fam_name, "gaussian")

  # Observed vs expected zeros -- the direct evidence for zero-inflation.
  # Skipped for Gaussian, where "zero" is not a meaningful outcome.
  obs_zeros <- exp_zeros <- NA_real_
  if (!is_gaussian) {
    obs_zeros <- sum(dd[[dv]] == 0, na.rm = TRUE)
    sims <- tryCatch(as.data.frame(simulate(fit, nsim = N_SIM, seed = SEED)),
                     error = function(e) NULL)
    if (!is.null(sims)) exp_zeros <- mean(colSums(sims == 0))
  }

  c(base, list(converged = converged,
               message = if (length(warn_msgs)) paste(warn_msgs, collapse = " | ") else "",
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
