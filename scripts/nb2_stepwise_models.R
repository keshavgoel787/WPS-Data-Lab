# NB2-only stepwise count models for the 2011-2021 WPS panel.
# See docs/superpowers/specs/2026-08-28-nb2-stepwise-design.md.
#
# Usage:
#   Rscript scripts/nb2_stepwise_models.R <panel_csv> <out_json>
#
# THIS SCRIPT FITS. It computes no derived statistic -- no Delta sigma^2_u0, no
# ICC, no percentage. Those live in report_nb2_stepwise.py so that exactly one
# script owns each number.
#
# The family is nbinom2 and only nbinom2. That is an EDITORIAL choice, not a
# fit-based one: at the mandated (1 + time | state) tier nbinom2 loses to ZINB
# by ~33 AIC for inspections and to nbinom1 by ~11 for violations. Holding the
# family fixed is what makes the three columns comparable and makes a variance
# reduction sequence meaningful. See spec section 3.

suppressPackageStartupMessages({
  library(glmmTMB)
  library(jsonlite)
})

# Warnings print immediately, with the call that raised them, rather than being
# deferred into an anonymous "There were N warnings" line. Every warning a fit
# raises is captured into that fit's `message` field and muffled, so anything
# reaching stderr is a leak from outside those handlers.
options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: nb2_stepwise_models.R <panel_csv> <out_json>")
panel_csv <- args[1]
out_json <- args[2]

`%||%` <- function(a, b) if (is.null(a)) b else a

CUBIC  <- c("time", "time2", "time3")
M2_ADD <- c("SPEND_APP_z", "SPEND_WORK_z", "lii_2017_z")
M3_ADD <- c(M2_ADD, "h2a_per_farmworker_z", "dol_demand_met_pct_z", "pct_flc_z")
RS_RE  <- "(1 + time | state)"
RI_RE  <- "(1 | state)"

panel <- read.csv(panel_csv, stringsAsFactors = FALSE)

CELLS <- list(
  insp_2021     = list(dv = "inspections", base = CUBIC, outcome = "inspections"),
  viol_cov_2021 = list(dv = "violations",  base = c("log_inspections", CUBIC),
                       outcome = "violations"))

coef_list <- function(mat) {
  if (is.null(mat) || nrow(mat) == 0) return(structure(list(), names = character(0)))
  stats::setNames(lapply(seq_len(nrow(mat)), function(i) {
    list(b = mat[i, 1], se = mat[i, 2], z = mat[i, 3], p = mat[i, 4])
  }), rownames(mat))
}

# fit_nb2: one nbinom2 specification, flattened to a plain list.
#
# `sample_rhs` exists for ONE purpose: the matched Delta baseline in Task 2,
# where the Model-1 formula must be fit on the Model-3 analytic sample. Rows are
# selected by complete.cases over `sample_rhs` (default: `rhs`), so the fitted
# formula and the sample-defining variable set can differ deliberately.
fit_nb2 <- function(d, dv, rhs, re, sample_rhs = NULL) {
  sel <- if (is.null(sample_rhs)) rhs else sample_rhs
  need <- unique(c(dv, sel, "state", "time"))
  need <- need[need %in% names(d)]
  dd <- d[stats::complete.cases(d[, need, drop = FALSE]), , drop = FALSE]
  dd$state <- factor(dd$state)

  form <- stats::as.formula(sprintf("%s ~ %s + %s", dv,
                                    paste(rhs, collapse = " + "), re))

  base <- list(formula = deparse1(form), re_used = re,
               n_obs = nrow(dd), n_states = length(unique(dd$state)),
               obs_zeros = sum(dd[[dv]] == 0, na.rm = TRUE),
               family_name = "nbinom2", sigma2_e = NA_real_)

  warn_msgs <- character(0)
  fit <- tryCatch(
    withCallingHandlers(
      glmmTMB(form, ziformula = ~0, family = nbinom2, data = dd, REML = FALSE),
      warning = function(w) {
        warn_msgs <<- c(warn_msgs, conditionMessage(w))
        invokeRestart("muffleWarning")
      }),
    error = function(e) e)

  if (inherits(fit, "error")) {
    return(c(base, list(converged = FALSE, message = conditionMessage(fit))))
  }

  pd_hess <- isTRUE(fit$sdr$pdHess)
  conv_code <- fit$fit$convergence
  sm <- summary(fit)
  cond <- coef_list(sm$coefficients$cond)
  finite_se <- length(cond) == 0 ||
    all(vapply(cond, function(x) is.finite(x$se), logical(1)))

  vc <- tryCatch(VarCorr(fit)$cond$state, error = function(e) NULL)
  s2_u0 <- if (is.null(vc)) NA_real_ else vc[1, 1]
  s2_u1 <- if (is.null(vc) || nrow(vc) < 2) NA_real_ else vc[2, 2]
  s_u01 <- if (is.null(vc) || nrow(vc) < 2) NA_real_ else vc[1, 2]

  # theta, the NB2 size parameter (Var = mu + mu^2/theta). NOT a residual SD --
  # its square is not a variance component, which is why `sigma2_e` stays NA for
  # every fit in this script.
  disp <- tryCatch(sigma(fit), error = function(e) NA_real_)

  # mu for the ICC's distribution-specific variance, computed here because only
  # the fit object can produce it. re.form = NA gives the POPULATION-level linear
  # predictor (fixed effects only); exponentiating its mean is the mu that
  # Nakagawa's observation-level variance for a log-link count model expects.
  mu_fixed <- tryCatch(
    exp(mean(stats::predict(fit, re.form = NA, type = "link"))),
    error = function(e) NA_real_)

  c(base, list(converged = pd_hess && conv_code == 0 && finite_se,
               message = if (length(warn_msgs)) paste(warn_msgs, collapse = " | ") else "",
               pd_hess = pd_hess, conv_code = conv_code,
               aic = AIC(fit), bic = BIC(fit),
               loglik = as.numeric(logLik(fit)), df = attr(logLik(fit), "df"),
               cond = cond,
               sigma2_u0 = s2_u0, sigma2_u1 = s2_u1, sigma_u01 = s_u01,
               dispersion = disp, mu_fixed = mu_fixed))
}

rhs_for <- function(cl, model) {
  switch(model,
         M1 = cl$base,
         M2 = c(cl$base, M2_ADD),
         M3 = c(cl$base, M3_ADD),
         stop("unknown model: ", model))
}

results <- list()

# ------------------------------------------------------------
# The six substantive fits: (1 + time | state), no fallback tier.
# A fit that fails here is REPORTED as failed. It is not downgraded to
# (1 | state) -- the random slope is the per-state rate of change around the
# WPS revision, and a table whose rows sit at different RE structures is the
# exact defect this arm exists to fix.
# ------------------------------------------------------------
for (cell_name in names(CELLS)) {
  cl <- CELLS[[cell_name]]
  for (model in c("M1", "M2", "M3")) {
    key <- sprintf("%s__%s__nbinom2__rs", cell_name, model)
    cat("fitting", key, "\n")
    res <- fit_nb2(panel, cl$dv, rhs_for(cl, model), RS_RE)
    res$cell <- cell_name; res$model <- model
    res$outcome <- cl$outcome; res$re_tier <- "rs"
    if (!isTRUE(res$converged)) {
      cat(sprintf("WARNING [%s]: did NOT converge at %s; message=%s\n",
                  key, RS_RE, res$message %||% ""))
    }
    results[[key]] <- res
  }
}

write_json(results, out_json, auto_unbox = TRUE, digits = 10, na = "null")
cat("\nWrote", length(results), "entries to", out_json, "\n")
n_bad <- sum(vapply(results, function(r) !isTRUE(r$converged), logical(1)))
if (n_bad > 0) cat("WARNING:", n_bad, "fit(s) did not converge -- see 'converged' flags\n")
