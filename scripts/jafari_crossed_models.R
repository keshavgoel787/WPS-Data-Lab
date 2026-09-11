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

# ------------------------------------------------------------
# The ZI tier ladder (spec 5).
#
# A ZI tier may NEVER introduce a predictor absent from the conditional model.
# Two reasons: it preserves Jafari's mirroring principle, and -- decisively --
# it guarantees every tier within a (cell, model) shares ONE analytic sample,
# because listwise deletion is computed over the conditional predictor set and
# the ZI set adds nothing to it. Attaching a `reduced` ZI block to an M1 whose
# conditional part has no covariates would silently drop AK/RI/VT via
# SPEND_APP_z and change N, making the tiers non-comparable.
#
# Consequently: at M1 `covariates` and `reduced` are empty and are SKIPPED; at
# M2 `reduced` duplicates `covariates` and is SKIPPED. Every skip carries a
# reason so a reader never has to infer one from a gap in the output.
# ------------------------------------------------------------
zi_tiers_for <- function(cond_rhs) {
  raw <- list(
    list(tag = "mirror",     vars = cond_rhs),
    list(tag = "covariates", vars = intersect(M3_ADD, cond_rhs)),
    list(tag = "reduced",    vars = intersect(M2_ADD, cond_rhs)),
    list(tag = "intercept",  vars = character(0)))

  kept <- list()
  out <- list()
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

zi_formula <- function(vars, with_re) {
  rhs <- if (length(vars)) paste(c("1", vars), collapse = " + ") else "1"
  if (with_re) rhs <- paste(rhs, "+ (1 | state)")
  stats::as.formula(paste("~", rhs))
}

# ZI-boundary degeneracy (spec 6.2): a fit that reports converged = TRUE but
# whose zero-inflation parameter has wandered to the edge of identifiability.
# A SELECTION-ELIGIBILITY flag, not deletion -- the record stays in the JSON
# with its reason.
#
# The previous arm had to guard the log-likelihood criterion with a
# tier-equality test, because comparing a rs-tier ZI fit against a ri-tier
# non-ZI one is not a degeneracy test at all. That guard is UNNECESSARY here
# (one RE structure), but whether the criterion was APPLICABLE is still
# recorded so the validator can assert it fired wherever a counterpart existed.
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

# Walk the tiers in order, keeping the FIRST that converges cleanly and is not
# ZI-boundary-degenerate. This is a FIDELITY rule, not an AIC rule: within a
# family we want the richest mirror that is estimable (spec 5.3). AIC is used
# only ACROSS families, where it is valid because all tiers of a (cell, model)
# share one sample.
#
# `counterpart` is the same-cell/model plain-family fit used for the
# log-likelihood degeneracy criterion; pass NULL when none exists.
fit_zi_ladder <- function(d, dv, rhs, family, with_re, counterpart = NULL) {
  attempts <- list()
  last <- NULL
  for (t in zi_tiers_for(rhs)) {
    if (t$skipped) {
      attempts[[length(attempts) + 1]] <- list(
        tier = t$tag, skipped = TRUE, skip_reason = t$skip_reason)
      next
    }
    res <- fit_spec(d, dv, rhs, family, zi = zi_formula(t$vars, with_re))
    res$zi_tier_reached <- t$tag
    res <- mark_zi_degenerate(res, counterpart)
    attempts[[length(attempts) + 1]] <- list(
      tier = t$tag, skipped = FALSE, skip_reason = "",
      converged = isTRUE(res$converged),
      zi_degenerate = isTRUE(res$zi_degenerate),
      zi_degenerate_reason = res$zi_degenerate_reason,
      aic = res$aic %||% NA_real_,
      message = res$message %||% "")
    last <- res
    if (isTRUE(res$converged) && !isTRUE(res$zi_degenerate)) {
      res$zi_tier_attempts <- attempts
      return(res)
    }
  }
  # Nothing converged cleanly. Return the LAST attempt as-is, flags intact, so
  # the failure is reported rather than masked by a silent substitution.
  if (is.null(last)) return(NULL)
  last$zi_tier_attempts <- attempts
  last
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

# ------------------------------------------------------------
# The family ladder (spec 4.4). EIGHT families, not the previous arm's six.
#
# Two deliberate changes. (1) A zero-inflated NB1 now exists: the old ladder's
# ZI rungs were all NB2-based, so when plain NB1 won both 2021 violations cells
# it beat a candidate set lacking its own ZI counterpart -- a documented blind
# spot left open only because closing it would have reopened selection.
# Selection is reopened here, so it is closed. (2) ZI random intercepts are on
# `state` only, never `year`: a year-level ZI random intercept would be
# identified off as few as 7 zeros spread across 11 years.
# ------------------------------------------------------------
LADDER <- list(
  list(tag = "poisson",  family = poisson, zi = FALSE, zi_re = FALSE),
  list(tag = "nbinom1",  family = nbinom1, zi = FALSE, zi_re = FALSE),
  list(tag = "nbinom2",  family = nbinom2, zi = FALSE, zi_re = FALSE),
  list(tag = "zip",      family = poisson, zi = TRUE,  zi_re = FALSE),
  list(tag = "zinb1",    family = nbinom1, zi = TRUE,  zi_re = FALSE),
  list(tag = "zinb2",    family = nbinom2, zi = TRUE,  zi_re = FALSE),
  list(tag = "zinb1_re", family = nbinom1, zi = TRUE,  zi_re = TRUE),
  list(tag = "zinb2_re", family = nbinom2, zi = TRUE,  zi_re = TRUE))

FAMILY_TAGS <- vapply(LADDER, `[[`, "", "tag")

# Which plain family each ZI rung is the zero-inflated version OF. Used for the
# log-likelihood degeneracy criterion and for the matched ZI-strength tally.
COUNTERPART <- c(zip = "poisson", zinb1 = "nbinom1", zinb2 = "nbinom2",
                 zinb1_re = "nbinom1", zinb2_re = "nbinom2")

# `x[["missing"]]` is an ERROR in R for both named vectors and lists -- it does
# NOT return NULL -- so `COUNTERPART[[tag]] %||% ""` never reaches the %||%.
# These two guarded accessors are the only way this file looks up an
# optionally-absent name.
counterpart_of <- function(tag) {
  if (!is.null(tag) && tag %in% names(COUNTERPART)) unname(COUNTERPART[[tag]])
  else NA_character_
}
lget <- function(l, k) {
  if (is.null(k) || length(k) != 1 || is.na(k) || !(k %in% names(l))) NULL else l[[k]]
}

# A *_re fit whose ZI random intercept has collapsed becomes
# formula-for-formula identical to its non-RE twin at the same cell, model and
# ZI tier. That is not a distinct competitor -- it is the twin wearing another
# name -- so it is marked and excluded from selection.
mark_collapse <- function(res_re, res_plain) {
  if (!is.null(res_plain) && !is.null(res_re) &&
      identical(res_re$formula, res_plain$formula) &&
      identical(res_re$zi_formula, res_plain$zi_formula)) {
    res_re$collapsed_to <- res_plain$family_tag
  }
  res_re
}

# Lowest AIC among converged, non-degenerate, non-collapsed fits. The
# empty-candidate-set fallback is RETAINED (erroring here would abort the whole
# ladder over one cell) but RECORDED: a defaulted winner is not a selection
# result and must never be reported as one.
pick_winner <- function(fit_list) {
  eligible <- Filter(function(r) isTRUE(r$converged) && is.finite(r$aic %||% NA_real_) &&
                       is.null(r$collapsed_to) && !isTRUE(r$zi_degenerate), fit_list)
  if (length(eligible) == 0) {
    warning("pick_winner: no eligible fit; defaulting to nbinom2",
            call. = FALSE, immediate. = TRUE)
    return(list(tag = "nbinom2", defaulted = TRUE, n_eligible = 0L))
  }
  list(tag = eligible[[which.min(vapply(eligible, function(r) r$aic, numeric(1)))]]$family_tag,
       defaulted = FALSE, n_eligible = length(eligible))
}

eligible_tags <- function(fit_list) {
  ok <- Filter(function(r) isTRUE(r$converged) && is.null(r$collapsed_to) &&
                 !isTRUE(r$zi_degenerate), fit_list)
  sort(vapply(ok, function(r) r$family_tag, character(1)))
}

# ------------------------------------------------------------
# The ladder: 8 families x {M1, M3} x 4 cells, all at (1 | state) + (1 | year).
# Selection races at M3 -- the model a family must survive with every covariate
# present. M1 is refit across the full ladder as a stability check and gets its
# OWN recorded winner, so "the winner changed as covariates entered" is a
# readable fact rather than an inference.
# ------------------------------------------------------------
fit_one <- function(cl, model, rung, plain_fits) {
  rhs <- rhs_for(cl, model)
  cp_tag <- counterpart_of(rung$tag)
  counterpart <- lget(plain_fits, cp_tag)
  if (rung$zi) {
    res <- fit_zi_ladder(cl$d, cl$dv, rhs, rung$family, rung$zi_re, counterpart)
  } else {
    res <- fit_spec(cl$d, cl$dv, rhs, rung$family, zi = ~0)
    res$zi_tier_reached <- NA_character_
    res$zi_tier_attempts <- list()
    res$zi_degenerate <- FALSE
    res$zi_degenerate_reason <- ""
    res$zi_loglik_criterion_applied <- FALSE
  }
  res$cell <- cl$name; res$model <- model; res$family_tag <- rung$tag
  res$window <- cl$window; res$outcome <- cl$outcome
  res$zi_counterpart <- cp_tag
  res$is_winner <- FALSE
  res
}

for (cell_name in names(cells)) {
  cl <- cells[[cell_name]]
  cl$name <- cell_name

  for (model in c("M1", "M3")) {
    plain_fits <- list()
    for (rung in LADDER) {
      key <- sprintf("%s__%s__%s", cell_name, model, rung$tag)
      cat("fitting", key, "\n")
      res <- fit_one(cl, model, rung, plain_fits)

      if (rung$zi_re) {
        twin <- lget(plain_fits, sub("_re$", "", rung$tag))
        res <- mark_collapse(res, twin)
      }
      res$eligible_for_selection <- isTRUE(res$converged) &&
        is.null(res$collapsed_to) && !isTRUE(res$zi_degenerate)

      if (!isTRUE(res$converged)) {
        cat(sprintf("  NOT CONVERGED at %s; message=%s\n",
                    CROSSED_RE, res$message %||% ""))
      }
      results[[key]] <- res
      plain_fits[[rung$tag]] <- res
    }
  }

  m1 <- results[sprintf("%s__M1__%s", cell_name, FAMILY_TAGS)]
  m3 <- results[sprintf("%s__M3__%s", cell_name, FAMILY_TAGS)]
  pw1 <- pick_winner(m1); pw3 <- pick_winner(m3)

  results[[sprintf("%s__M1__%s", cell_name, pw1$tag)]]$is_winner <- TRUE
  results[[sprintf("%s__M3__%s", cell_name, pw3$tag)]]$is_winner <- TRUE

  # M2 under the M3 winner only -- a downstream product of selection, never a
  # competitor. It walks its OWN ZI tier ladder, because the mirror is defined
  # against M2's conditional predictor set and M2's mirror is not M3's.
  win <- Filter(function(r) r$tag == pw3$tag, LADDER)[[1]]
  key <- sprintf("%s__M2__%s", cell_name, pw3$tag)
  cat("fitting", key, "(winning family at M3)\n")
  m2_plain <- list()
  if (win$zi) {
    cp_tag_m2 <- counterpart_of(win$tag)
    cp_rung <- Filter(function(r) r$tag == cp_tag_m2, LADDER)[[1]]
    m2_plain[[cp_tag_m2]] <- fit_one(cl, "M2", cp_rung, list())
  }
  res <- fit_one(cl, "M2", win, m2_plain)
  res$is_winner <- FALSE
  res$eligible_for_selection <- FALSE
  results[[key]] <- res

  results[[sprintf("%s__meta", cell_name)]] <- list(
    cell = cell_name, window = cl$window, outcome = cl$outcome,
    m1_winner = pw1$tag, m3_winner = pw3$tag,
    stable = identical(pw1$tag, pw3$tag),
    winner_defaulted_m1 = isTRUE(pw1$defaulted),
    winner_defaulted_m3 = isTRUE(pw3$defaulted),
    n_eligible_m1 = pw1$n_eligible, n_eligible_m3 = pw3$n_eligible,
    eligible_m1 = eligible_tags(m1), eligible_m3 = eligible_tags(m3),
    m2_family = pw3$tag)
}

write_json(results, out_json, auto_unbox = TRUE, digits = 10, na = "null")
cat("\nWrote", length(results), "entries to", out_json, "\n")
n_bad <- sum(vapply(results, function(r) !is.null(r$converged) && !isTRUE(r$converged),
                    logical(1)))
if (n_bad > 0) cat("WARNING:", n_bad, "fit(s) did not converge -- see 'converged' flags\n")
