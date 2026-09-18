# Violations-per-inspection ("ratio") models, crossed random intercepts.
#
# Usage:
#   Rscript scripts/jafari_ratio_models.R <panel_csv> <out_json>
#
# WHY THIS ARM EXISTS. Joe Grzywacz, 2026-09-11: Jafari et al. had only a
# violations COUNT and could not separate "inspected, found clean" from "never
# inspected". A violations-per-inspection outcome is closer to their data, so
# fit the project's stepwise specification on that outcome -- with inspections
# REMOVED as a covariate, because it is now the denominator.
#
# THE DESIGN TENSION THIS SCRIPT EXISTS TO MEASURE. A ratio is not a count, so
# nbinom2/ZINB cannot take it directly. The count-preserving form of "violations
# per inspection" is an OFFSET: violations ~ ... + offset(log(inspections)),
# which models the rate directly on a count likelihood. But log(0) is undefined,
# so the offset DELETES every zero-inspection state-year -- and in this window
# those rows are also the zero-violation rows this modelling tradition exists to
# explain. That is exactly why the crossed arm dropped its offset cell. The
# response here is not to pick a side but to fit both defensible forms and one
# all-rows fallback, and let the PI choose:
#
#   Spec A  nbinom2, DV = violations, offset(log(inspections)), inspections > 0
#   Spec B  Spec A + zero-inflation (mirrored ZI formula, laddered down)
#   Spec C  Gaussian LMM on log((violations + 1) / (inspections + 1)), ALL rows
#
# THIS SCRIPT FITS, and derives only facts that require the raw panel (which
# rows the offset drops; observed zero counts per analytic sample). Every
# percentage, tally and comparison lives in report_jafari_ratio.py so exactly
# one script owns each number.

suppressPackageStartupMessages({
  library(glmmTMB)
  library(jsonlite)
})

# Warnings print immediately with their call rather than being deferred into an
# anonymous "There were N warnings" line. Every warning a fit raises is captured
# into that fit's `message` field; anything reaching stderr is a leak from
# outside those handlers (the benign TMB ABI-mismatch notice is expected).
options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: jafari_ratio_models.R <panel_csv> <out_json>")
panel_csv <- args[1]
out_json  <- args[2]

`%||%` <- function(a, b) if (is.null(a)) b else a

# `l[["missing"]]` is an ERROR in R, not NULL, so `results[[k]] %||% NULL` never
# reaches the %||%. This is the only way this file looks up a possibly-absent
# name.
lget <- function(l, k) if (is.null(k) || !(k %in% names(l))) NULL else l[[k]]

SEED  <- 20260917L
N_SIM <- 2000L

CUBIC  <- c("time", "time2", "time3")
M2_ADD <- c("SPEND_APP_z", "SPEND_WORK_z", "lii_2017_z")
M3_ADD <- c(M2_ADD, "h2a_per_farmworker_z", "dol_demand_met_pct_z", "pct_flc_z")

# The ONE random-effects structure in this arm: crossed random INTERCEPTS,
# state and year. No random slope anywhere -- standing PI directive for the
# Jafari-aligned work (2026-09-10). There is deliberately no fallback tier.
CROSSED_RE <- "(1 | state) + (1 | year)"

# log_inspections is NOT a predictor in any specification here. It is the
# denominator now. Its presence in the panel is inherited from the count-model
# arm and is used only to build Spec C's ratio outcome.
DV_COUNT <- "violations"
DENOM    <- "inspections"
RATIO_DV <- "log_ratio"   # log((violations + 1) / (inspections + 1))

panel <- read.csv(panel_csv, stringsAsFactors = FALSE)
panel[[RATIO_DV]] <- log((panel[[DV_COUNT]] + 1) / (panel[[DENOM]] + 1))

coef_list <- function(mat) {
  if (is.null(mat) || nrow(mat) == 0) return(structure(list(), names = character(0)))
  stats::setNames(lapply(seq_len(nrow(mat)), function(i) {
    list(b = mat[i, 1], se = mat[i, 2], z = mat[i, 3], p = mat[i, 4])
  }), rownames(mat))
}

# analytic_rows: the ONE place rows are selected, so Spec A/B/C and the
# derivation block cannot drift apart.
#
# `sample_rhs` exists for one purpose: the matched baseline, where the Model-1
# FORMULA is fit on the Model-2/3 ROWS. M1 keeps all 49 states; M2/M3 lose
# AK/RI/VT, which have no BLS pesticide-applicator series. Without that fit a
# Model-1-baseline reduction silently mixes "the covariates explained variance"
# with "three states left the sample".
#
# `positive_denom` is what makes the offset legal, and is also the thing this
# arm is measuring the cost of.
analytic_rows <- function(d, dv, rhs, sample_rhs = NULL, positive_denom = FALSE) {
  sel  <- if (is.null(sample_rhs)) rhs else sample_rhs
  need <- unique(c(dv, sel, "state", "year", "time"))
  need <- need[need %in% names(d)]
  dd <- d[stats::complete.cases(d[, need, drop = FALSE]), , drop = FALSE]
  if (positive_denom) dd <- dd[dd[[DENOM]] > 0, , drop = FALSE]
  dd
}

fit_spec <- function(d, dv, rhs, family, zi = ~0, use_offset = FALSE,
                     sample_rhs = NULL, REML = FALSE) {
  dd <- analytic_rows(d, dv, rhs, sample_rhs, positive_denom = use_offset)
  dd$state <- factor(dd$state)
  dd$year  <- factor(dd$year)

  # offset(log(inspections)) -- the true log, NOT the panel's log_inspections
  # column, which is log(x + 1) and would make the rate a rate per (inspections
  # + 1). Legal only because `positive_denom` has already removed the zeros.
  off <- if (use_offset) " + offset(log(inspections))" else ""
  form <- stats::as.formula(sprintf("%s ~ %s%s + %s", dv,
                                    paste(rhs, collapse = " + "), off, CROSSED_RE))

  base <- list(formula = deparse1(form), zi_formula = deparse1(zi),
               re_used = CROSSED_RE, uses_offset = use_offset,
               n_obs = nrow(dd), n_states = length(unique(dd$state)),
               n_years = length(unique(dd$year)),
               obs_zeros = sum(dd[[DV_COUNT]] == 0, na.rm = TRUE),
               sample_rhs = if (is.null(sample_rhs)) rhs else sample_rhs)

  warn_msgs <- character(0)
  fit <- tryCatch(
    withCallingHandlers(
      glmmTMB(form, ziformula = zi, family = family, data = dd, REML = REML),
      warning = function(w) {
        warn_msgs <<- c(warn_msgs, conditionMessage(w))
        invokeRestart("muffleWarning")
      }),
    error = function(e) e)

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

  # Crossed REs: two grouping factors, each with its own variance, and NO
  # covariance between them by construction -- that is what "crossed" means
  # here, so no sigma_u01 is emitted anywhere in this arm.
  vc_st <- tryCatch(VarCorr(fit)$cond$state, error = function(e) NULL)
  vc_yr <- tryCatch(VarCorr(fit)$cond$year,  error = function(e) NULL)
  s2_state <- if (is.null(vc_st)) NA_real_ else vc_st[1, 1]
  s2_year  <- if (is.null(vc_yr)) NA_real_ else vc_yr[1, 1]

  fam_obj  <- if (is.function(family)) family() else family
  fam_name <- if (is.character(fam_obj)) fam_obj else fam_obj$family
  is_gaussian <- identical(fam_name, "gaussian")

  # sigma() is a residual SD ONLY for Gaussian. For nbinom2 it is theta
  # (Var = mu + mu^2/theta). Exporting its square as `sigma2_e` for a count fit
  # would put a dispersion parameter under a name that reads as a variance
  # component -- a documented defect in an earlier arm -- so sigma2_e stays NA
  # for every count fit here.
  disp <- tryCatch(sigma(fit), error = function(e) NA_real_)
  s2_e <- if (is_gaussian && is.finite(disp)) disp^2 else NA_real_

  year_blups <- tryCatch({
    re_yr <- ranef(fit)$cond$year
    stats::setNames(as.list(as.numeric(re_yr[, 1])), rownames(re_yr))
  }, error = function(e) NULL)

  # SAMPLE-AVERAGE structural-zero probability: the mean over the analytic
  # sample of each row's own zero-inflation probability, conditional on the
  # fitted random effects. Never plogis(zi_intercept): once the ZI block carries
  # covariates that evaluates the ZI linear predictor at x = 0, which describes
  # no state in the data.
  zi_prob_mean <- NA_real_
  if (!identical(deparse1(zi), deparse1(~0))) {
    zi_prob_mean <- tryCatch(mean(stats::predict(fit, type = "zprob")),
                             error = function(e) NA_real_)
  }

  exp_zeros <- exp_zeros_se <- NA_real_
  sim_warn <- character(0)
  if (!is_gaussian) {
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
               family_name = fam_name, dispersion = disp, sigma2_e = s2_e,
               year_blups = year_blups, zi_prob_mean = zi_prob_mean,
               exp_zeros = exp_zeros, exp_zeros_se = exp_zeros_se,
               sim_message = if (length(sim_warn))
                 paste(unique(sim_warn), collapse = " | ") else ""))
}

rhs_for <- function(model) {
  switch(model,
         M1 = CUBIC,
         M2 = c(CUBIC, M2_ADD),
         M3 = c(CUBIC, M3_ADD),
         stop("unknown model: ", model))
}

# ------------------------------------------------------------
# The ZI tier ladder (Spec B).
#
# A ZI tier may NEVER introduce a predictor absent from the conditional model.
# That is Jafari's mirroring principle, and -- decisively -- it guarantees every
# tier of a model shares ONE analytic sample, because listwise deletion is
# computed over the conditional predictor set and the ZI set adds nothing to it.
# Attaching spending terms to an M1 ZI block would silently drop AK/RI/VT and
# make the AIC comparison against Spec A invalid.
#
# Within a family this is a FIDELITY rule, not an AIC rule: keep the richest
# mirror that is estimable. AIC is used only for Spec B vs Spec A, where the
# sample and conditional formula are identical and the ZI block is the only
# difference -- which is exactly the "did zero-inflation survive the offset"
# question.
# ------------------------------------------------------------
zi_tiers_for <- function(cond_rhs) {
  raw <- list(
    list(tag = "mirror",     vars = cond_rhs),
    list(tag = "covariates", vars = intersect(M3_ADD, cond_rhs)),
    list(tag = "reduced",    vars = intersect(M2_ADD, cond_rhs)),
    list(tag = "intercept",  vars = character(0)))

  kept <- list(); out <- list()
  for (t in raw) {
    reason <- ""
    if (t$tag != "intercept" && length(t$vars) == 0) {
      reason <- "empty predictor set at this model (no such terms in the conditional part)"
    } else {
      dup <- Find(function(k) identical(sort(k$vars), sort(t$vars)), kept)
      if (!is.null(dup)) reason <- sprintf("duplicate of tier '%s'", dup$tag)
    }
    t$skipped <- nzchar(reason)
    t$skip_reason <- reason
    if (!t$skipped) kept[[length(kept) + 1]] <- t
    out[[length(out) + 1]] <- t
  }
  out
}

zi_formula <- function(vars) {
  stats::as.formula(paste("~", if (length(vars))
    paste(c("1", vars), collapse = " + ") else "1"))
}

# ZI-boundary degeneracy: a fit reporting converged = TRUE whose zero-inflation
# parameter has wandered to the edge of identifiability, or which has bought
# exactly zero likelihood over its non-ZI counterpart. A flag, never a deletion
# -- the record stays in the JSON with its reason.
mark_zi_degenerate <- function(res, counterpart) {
  reasons <- character(0)
  zi_int <- res$zi[["(Intercept)"]]
  if (!is.null(zi_int)) {
    if (is.finite(zi_int$b) && abs(zi_int$b) > 15) reasons <- c(reasons, "|zi_intercept| > 15")
    if (is.finite(zi_int$se) && zi_int$se > 100) reasons <- c(reasons, "se(zi_intercept) > 100")
  }
  applicable <- !is.null(counterpart) && isTRUE(res$converged) &&
    isTRUE(counterpart$converged) && is.finite(res$loglik %||% NA_real_) &&
    is.finite(counterpart$loglik %||% NA_real_)
  if (applicable && abs(res$loglik - counterpart$loglik) < 1e-4) {
    reasons <- c(reasons, "loglik matches non-ZI counterpart within 1e-4")
  }
  res$zi_loglik_criterion_applied <- applicable
  res$zi_degenerate <- length(reasons) > 0
  res$zi_degenerate_reason <- paste(reasons, collapse = "; ")
  res
}

fit_zi_ladder <- function(d, rhs, counterpart, sample_rhs = NULL) {
  attempts <- list(); last <- NULL
  for (t in zi_tiers_for(rhs)) {
    if (t$skipped) {
      attempts[[length(attempts) + 1]] <- list(
        tier = t$tag, skipped = TRUE, skip_reason = t$skip_reason)
      next
    }
    res <- fit_spec(d, DV_COUNT, rhs, nbinom2, zi = zi_formula(t$vars),
                    use_offset = TRUE, sample_rhs = sample_rhs)
    res$zi_tier_reached <- t$tag
    res <- mark_zi_degenerate(res, counterpart)
    attempts[[length(attempts) + 1]] <- list(
      tier = t$tag, skipped = FALSE, skip_reason = "",
      converged = isTRUE(res$converged),
      zi_degenerate = isTRUE(res$zi_degenerate),
      zi_degenerate_reason = res$zi_degenerate_reason,
      aic = res$aic %||% NA_real_, message = res$message %||% "")
    last <- res
    if (isTRUE(res$converged) && !isTRUE(res$zi_degenerate)) {
      res$zi_tier_attempts <- attempts
      return(res)
    }
  }
  # Nothing converged cleanly at any tier. Return the LAST attempt with its
  # flags intact so the failure is reported, never masked by a substitution.
  if (is.null(last)) return(NULL)
  last$zi_tier_attempts <- attempts
  last
}

results <- list()

# ------------------------------------------------------------
# Derivation block: what the offset actually costs, computed from the panel
# rather than asserted. CLAUDE.md records 7 dropped rows (CT-2018, KS-2018,
# MA-2018, OR-2020, OR-2021, UT-2017, WV-2012), all zero-violation. This
# re-derives it; the reporter states whether it held.
# ------------------------------------------------------------
drop_records <- list()
for (model in c("M1", "M2", "M3")) {
  rhs <- rhs_for(model)
  keep_all <- analytic_rows(panel, DV_COUNT, rhs, positive_denom = FALSE)
  keep_pos <- analytic_rows(panel, DV_COUNT, rhs, positive_denom = TRUE)
  dropped  <- keep_all[keep_all[[DENOM]] <= 0, , drop = FALSE]
  drop_records[[model]] <- list(
    model = model,
    n_all_rows = nrow(keep_all),
    n_states_all_rows = length(unique(keep_all$state)),
    n_offset_rows = nrow(keep_pos),
    n_states_offset = length(unique(keep_pos$state)),
    n_dropped = nrow(dropped),
    n_dropped_zero_violation = sum(dropped[[DV_COUNT]] == 0, na.rm = TRUE),
    n_dropped_positive_violation = sum(dropped[[DV_COUNT]] > 0, na.rm = TRUE),
    dropped_rows = lapply(seq_len(nrow(dropped)), function(i) list(
      state = dropped$state[i], year = dropped$year[i],
      violations = dropped[[DV_COUNT]][i], inspections = dropped[[DENOM]][i])),
    zeros_all_rows = sum(keep_all[[DV_COUNT]] == 0, na.rm = TRUE),
    zeros_offset_rows = sum(keep_pos[[DV_COUNT]] == 0, na.rm = TRUE))
}
results[["__offset_derivation"]] <- list(
  panel_csv = panel_csv,
  panel_rows = nrow(panel),
  panel_states = length(unique(panel$state)),
  panel_years = length(unique(panel$year)),
  rows_missing_violations = sum(is.na(panel[[DV_COUNT]])),
  rows_zero_inspections = sum(panel[[DENOM]] == 0, na.rm = TRUE),
  by_model = drop_records)

# ------------------------------------------------------------
# Spec A -- primary, count-preserving. nbinom2 with offset(log(inspections)).
# Spec B -- the same conditional model plus zero-inflation, ZI tiers laddered.
#           Spec B vs Spec A at the same model is the ZI-survival test: same
#           rows, same conditional formula, ZI block the only difference.
# Spec C -- Gaussian on log((v + 1) / (i + 1)), all rows. Keeps the states the
#           offset deletes, at the cost of leaving the count likelihood. Its AIC
#           is on a DIFFERENT likelihood and is never comparable to A or B;
#           REML = FALSE so that at least the A-vs-B-style ladder WITHIN Spec C
#           is on a common (ML) footing.
# ------------------------------------------------------------
SPECS <- list(
  A = list(label = "nbinom2 + offset(log(inspections))", zero_inflated = FALSE),
  B = list(label = "zero-inflated nbinom2 + offset(log(inspections))", zero_inflated = TRUE),
  C = list(label = "Gaussian LMM on log((violations+1)/(inspections+1))", zero_inflated = FALSE))

fit_one <- function(spec, model, sample_rhs = NULL) {
  rhs <- rhs_for(model)
  if (spec == "A") {
    res <- fit_spec(panel, DV_COUNT, rhs, nbinom2, zi = ~0, use_offset = TRUE,
                    sample_rhs = sample_rhs)
    res$zi_tier_reached <- NA_character_; res$zi_tier_attempts <- list()
    res$zi_degenerate <- FALSE; res$zi_degenerate_reason <- ""
    res$zi_loglik_criterion_applied <- FALSE
  } else if (spec == "B") {
    key_a <- if (is.null(sample_rhs)) sprintf("A__%s", model)
             else sprintf("A__%s", "M1matched")
    res <- fit_zi_ladder(panel, rhs, lget(results, key_a), sample_rhs = sample_rhs)
  } else {
    res <- fit_spec(panel, RATIO_DV, rhs, gaussian, zi = ~0, use_offset = FALSE,
                    sample_rhs = sample_rhs)
    res$zi_tier_reached <- NA_character_; res$zi_tier_attempts <- list()
    res$zi_degenerate <- FALSE; res$zi_degenerate_reason <- ""
    res$zi_loglik_criterion_applied <- FALSE
  }
  res$spec <- spec; res$spec_label <- SPECS[[spec]]$label
  res$model <- model
  res$dv <- if (spec == "C") RATIO_DV else DV_COUNT
  res
}

for (spec in names(SPECS)) {
  for (model in c("M1", "M2", "M3")) {
    key <- sprintf("%s__%s", spec, model)
    cat("fitting", key, "--", SPECS[[spec]]$label, "\n")
    res <- fit_one(spec, model)
    if (!isTRUE(res$converged))
      cat(sprintf("  NOT CONVERGED [%s]: %s\n", key, res$message %||% ""))
    results[[key]] <- res
  }

  # The matched baseline: Model 1's FORMULA on Model 2/3's ROWS. Fit for every
  # spec, because delta_pct_matched is reported for every spec.
  key <- sprintf("%s__M1matched", spec)
  cat("fitting", key, "(Model-1 baseline on the Model-2/3 sample)\n")
  res <- fit_one(spec, "M1", sample_rhs = rhs_for("M3"))
  res$model <- "M1matched"; res$sample_model <- "M3"
  if (!isTRUE(res$converged))
    cat(sprintf("  NOT CONVERGED [%s]: %s\n", key, res$message %||% ""))
  results[[key]] <- res
}

write_json(results, out_json, auto_unbox = TRUE, digits = 10, na = "null")
cat("\nWrote", length(results), "entries to", out_json, "\n")
n_bad <- sum(vapply(results, function(r) !is.null(r$converged) && !isTRUE(r$converged),
                    logical(1)))
if (n_bad > 0) cat("WARNING:", n_bad, "fit(s) did not converge -- see 'converged' flags\n")
