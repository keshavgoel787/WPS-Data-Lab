# Jafari-aligned STEPWISE crossed random-effects count models, WPS view only.
#
# Usage:
#   Rscript scripts/jafari_stepwise_models.R <panel_csv> <out_json>
#
# WHY THIS ARM EXISTS (PI direction, meeting of 2026-09-11). The existing
# crossed arm (scripts/jafari_crossed_models.R) enters every covariate at once
# and computes no variance reduction at all. Joe named that as the single thing
# blocking submission: he needs an M1 -> M2 -> M3 build-up showing how much of
# the state-to-state variance the covariates explain. This script produces that
# series. It also answers his second request -- a violations series WITHOUT
# inspections as a covariate, since Jafari had no inspections variable -- by
# fitting `viol_2021_ni` alongside `viol_2021_wi`.
#
# THIS SCRIPT FITS. It computes no derived statistic -- no Delta sigma^2, no
# percentage, no IRR, no odds ratio. Those live in report_jafari_stepwise.py so
# that exactly one script owns each number.
#
# NOTHING EXISTING IS MODIFIED. This file only reads
# data/generated/count_model_panel_2021.csv (frozen) and
# data/generated/jafari_crossed_results.json (read-only, and only to record
# which family the crossed arm selected). It writes one new artifact.
#
# THE CENTRAL DESIGN RULE: the family is HELD FIXED across M1/M2/M3 within a
# cell. If the family were allowed to change between steps, a change in
# sigma^2_state would reflect the family swap and not the covariates, and the
# whole deliverable would be meaningless. See "Family" below.
#
# COMPANION FITS (STEP 3, added 2026-09-17). The two violations cells answer
# Joe's Thing 2 by differing in `log_inspections` -- but they also differ in
# FAMILY, because each raced or inherited its own. So the raw gap between their
# sigma^2_state is an upper bound on what `log_inspections` does, not an
# estimate of it. Step 3 fits each violations cell a SECOND time under the
# OTHER one's family, in the headline `zi_intercept` series only, which brackets
# the answer between two family-matched comparisons. These records are tagged
# `is_companion = TRUE` and carry a `__companion` key suffix; they are NOT the
# reported models and each cell's own selected family is unchanged.

suppressPackageStartupMessages({
  library(glmmTMB)
  library(jsonlite)
})

# Warnings print immediately, with the call that raised them, instead of being
# deferred into an anonymous "There were N warnings" line. Every warning a fit
# raises is captured into that fit's `message` field and muffled, so anything
# that still reaches stderr is a leak from outside those handlers. (The benign
# TMB ABI version-mismatch warning is one such leak and is expected.)
options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: jafari_stepwise_models.R <panel_csv> <out_json>")
panel_csv <- args[1]
out_json  <- args[2]

`%||%` <- function(a, b) if (is.null(a)) b else a

SEED  <- 20260917L
N_SIM <- 2000L

CUBIC  <- c("time", "time2", "time3")
M2_ADD <- c("SPEND_APP_z", "SPEND_WORK_z", "lii_2017_z")
M3_ADD <- c(M2_ADD, "h2a_per_farmworker_z", "dol_demand_met_pct_z", "pct_flc_z")

# The ONE random-effects structure in this arm: crossed random INTERCEPTS, no
# random slope anywhere. That is a standing PI directive for the Jafari-aligned
# work (2026-09-10, restated 2026-09-11) and there is deliberately no fallback
# tier -- a fit that fails here is reported as failed.
CROSSED_RE <- "(1 | state) + (1 | year)"

CROSSED_JSON <- "/Users/keshavgoel/Research/data/generated/jafari_crossed_results.json"

panel <- read.csv(panel_csv, stringsAsFactors = FALSE)

coef_list <- function(mat) {
  if (is.null(mat) || nrow(mat) == 0) return(structure(list(), names = character(0)))
  stats::setNames(lapply(seq_len(nrow(mat)), function(i) {
    list(b = mat[i, 1], se = mat[i, 2], z = mat[i, 3], p = mat[i, 4])
  }), rownames(mat))
}

# fit_spec: one specification, flattened to a plain list.
#
# `sample_rhs` exists for ONE purpose: the MATCHED baseline, where Model 1's
# formula must be fit on Model 2/3's analytic sample. Rows are selected by
# complete.cases over `sample_rhs` (default: `rhs`), so the fitted formula and
# the sample-defining variable set can differ deliberately. Without it, an
# M1 -> M3 variance reduction would silently mix "the covariates explained
# variance" with "three states left the sample".
fit_spec <- function(d, dv, rhs, family, zi = ~0, sample_rhs = NULL,
                     re = CROSSED_RE, REML = FALSE) {
  # The analytic sample is defined by complete.cases over the CONDITIONAL
  # predictors only. ZI predictors are always a subset of them (see
  # zi_tiers_for), so every ZI tier within a (cell, model) shares one sample --
  # which is what makes AIC comparable across tiers.
  sel  <- if (is.null(sample_rhs)) rhs else sample_rhs
  need <- unique(c(dv, sel, "state", "year", "time"))
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
               n_years = length(unique(dd$year)),
               sample_rhs = if (is.null(sample_rhs)) NA_character_
                            else paste(sample_rhs, collapse = " + "),
               obs_zeros = sum(dd[[dv]] == 0, na.rm = TRUE))

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

  # Crossed REs: two separate grouping factors, each with its own variance, and
  # no covariance between them by construction -- so no sigma_u01 anywhere.
  vc_st <- tryCatch(VarCorr(fit)$cond$state, error = function(e) NULL)
  vc_yr <- tryCatch(VarCorr(fit)$cond$year,  error = function(e) NULL)
  s2_state <- if (is.null(vc_st)) NA_real_ else vc_st[1, 1]
  s2_year  <- if (is.null(vc_yr)) NA_real_ else vc_yr[1, 1]

  # The ZI component's OWN state random-intercept variance. Reading only
  # VarCorr(fit)$cond and never $zi is a documented past defect in this project.
  vc_zi <- tryCatch(VarCorr(fit)$zi$state, error = function(e) NULL)
  s2_zi <- if (is.null(vc_zi)) NA_real_ else vc_zi[1, 1]

  fam_obj  <- if (is.function(family)) family() else family
  fam_name <- if (is.character(fam_obj)) fam_obj else fam_obj$family

  # sigma() is a residual SD ONLY for Gaussian. For nbinom2 it is theta, for
  # nbinom1 the dispersion multiplier. Its square is NOT a variance component,
  # which is why nothing here exports a `sigma2_e` for a count fit.
  disp <- tryCatch(sigma(fit), error = function(e) NA_real_)

  year_blups <- tryCatch({
    re_yr <- ranef(fit)$cond$year
    stats::setNames(as.list(as.numeric(re_yr[, 1])), rownames(re_yr))
  }, error = function(e) NULL)

  # SAMPLE-AVERAGE structural-zero probability. NOT plogis(zi_intercept) and
  # NOT the Gauss-Hermite marginal E[plogis(b0 + u)]: both of those evaluate
  # the ZI linear predictor at x = 0, which describes no state in the data once
  # the ZI block carries covariates (log_inspections has mean ~3.5 and is never
  # z-scored). A previous draft of the crossed memo reported 0.4639 for a model
  # that simulates 78 zeros in 501 rows.
  zi_prob_mean <- NA_real_
  if (!identical(deparse1(zi), deparse1(~0))) {
    zi_prob_mean <- tryCatch(mean(stats::predict(fit, type = "zprob")),
                             error = function(e) NA_real_)
  }

  exp_zeros <- exp_zeros_se <- NA_real_
  sim_warn <- character(0)
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

  c(base, list(converged = pd_hess && conv_code == 0 && finite_se,
               message = if (length(warn_msgs)) paste(warn_msgs, collapse = " | ") else "",
               pd_hess = pd_hess, conv_code = conv_code,
               aic = AIC(fit), bic = BIC(fit),
               loglik = as.numeric(logLik(fit)), df = attr(logLik(fit), "df"),
               cond = cond, zi = zi_co,
               sigma2_state = s2_state, sigma2_year = s2_year,
               sigma2_zi_state = s2_zi,
               sigma_zi_state = if (is.na(s2_zi)) NA_real_ else sqrt(s2_zi),
               family_name = fam_name, dispersion = disp,
               year_blups = year_blups, zi_prob_mean = zi_prob_mean,
               exp_zeros = exp_zeros, exp_zeros_se = exp_zeros_se,
               sim_message = if (length(sim_warn))
                 paste(unique(sim_warn), collapse = " | ") else ""))
}

# ------------------------------------------------------------
# The three cells. WPS view (2011-2021) ONLY -- per PI direction 2026-09-11 the
# 2011-2019 establishments window is out of scope for this deliverable.
#
# viol_2021_wi and viol_2021_ni differ in ONE term: log_inspections. That is
# Joe's Thing 2 -- Jafari had no inspections variable, and he wants to see
# whether ours is needed.
# ------------------------------------------------------------
CELLS <- list(
  insp_2021 = list(
    dv = "inspections", base = CUBIC, outcome = "inspections",
    with_inspections = FALSE,
    family_tag = "zinb2",
    family_source = "M3 winner of the existing crossed arm (jafari_crossed_results.json)",
    crossed_key = "insp_2021"),
  viol_2021_wi = list(
    dv = "violations", base = c("log_inspections", CUBIC), outcome = "violations",
    with_inspections = TRUE,
    family_tag = "zinb2_re",
    family_source = "M3 winner of the existing crossed arm (jafari_crossed_results.json)",
    crossed_key = "viol_2021"),
  viol_2021_ni = list(
    dv = "violations", base = CUBIC, outcome = "violations",
    with_inspections = FALSE,
    family_tag = NA_character_,   # selected at M3 below; no precedent exists
    family_source = "selected at M3 in THIS script (no precedent: the crossed arm never fit a violations model without log_inspections)",
    crossed_key = NA_character_))

rhs_for <- function(cl, model) {
  switch(model,
         M1        = cl$base,
         M1matched = cl$base,
         M2        = c(cl$base, M2_ADD),
         M3        = c(cl$base, M3_ADD),
         stop("unknown model: ", model))
}

# The family ladder, identical to the crossed arm's (8 rungs). Used ONLY to
# select viol_2021_ni's family at M3; the other two cells inherit their family
# from the crossed arm so that this series and that one are comparable.
LADDER <- list(
  list(tag = "poisson",  family = poisson, zi = FALSE, zi_re = FALSE),
  list(tag = "nbinom1",  family = nbinom1, zi = FALSE, zi_re = FALSE),
  list(tag = "nbinom2",  family = nbinom2, zi = FALSE, zi_re = FALSE),
  list(tag = "zip",      family = poisson, zi = TRUE,  zi_re = FALSE),
  list(tag = "zinb1",    family = nbinom1, zi = TRUE,  zi_re = FALSE),
  list(tag = "zinb2",    family = nbinom2, zi = TRUE,  zi_re = FALSE),
  list(tag = "zinb1_re", family = nbinom1, zi = TRUE,  zi_re = TRUE),
  list(tag = "zinb2_re", family = nbinom2, zi = TRUE,  zi_re = TRUE))

rung_of <- function(tag) Filter(function(r) identical(r$tag, tag), LADDER)[[1]]

COUNTERPART <- c(zip = "poisson", zinb1 = "nbinom1", zinb2 = "nbinom2",
                 zinb1_re = "nbinom1", zinb2_re = "nbinom2")

# `x[["missing"]]` is an ERROR in R for both named vectors and lists -- it does
# NOT return NULL -- so `COUNTERPART[[tag]] %||% ""` never reaches the %||%.
counterpart_of <- function(tag) {
  if (!is.null(tag) && !is.na(tag) && tag %in% names(COUNTERPART))
    unname(COUNTERPART[[tag]]) else NA_character_
}
lget <- function(l, k) {
  if (is.null(k) || length(k) != 1 || is.na(k) || !(k %in% names(l))) NULL else l[[k]]
}

# ------------------------------------------------------------
# The ZI tier ladder, copied in spirit from the crossed arm so that the
# `mirror` series here reproduces that arm's fits exactly where the two are
# comparable (that reproduction is the correctness anchor for this script).
#
# A ZI tier may NEVER introduce a predictor absent from the conditional model:
# it preserves Jafari's mirroring principle, and it guarantees every tier within
# a (cell, model) shares ONE analytic sample.
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

zi_formula <- function(vars, with_re) {
  rhs <- if (length(vars)) paste(c("1", vars), collapse = " + ") else "1"
  if (with_re) rhs <- paste(rhs, "+ (1 | state)")
  stats::as.formula(paste("~", rhs))
}

# ZI-boundary degeneracy: a fit reporting converged = TRUE whose zero-inflation
# parameter has wandered to the edge of identifiability. A selection-eligibility
# flag, never deletion -- the record stays in the JSON with its reason.
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
# ZI-boundary-degenerate. A FIDELITY rule, not an AIC rule: within a family we
# want the richest estimable mirror. AIC is used only ACROSS families.
fit_zi_ladder <- function(d, dv, rhs, family, with_re, counterpart = NULL,
                          sample_rhs = NULL) {
  attempts <- list(); last <- NULL
  for (t in zi_tiers_for(rhs)) {
    if (t$skipped) {
      attempts[[length(attempts) + 1]] <- list(
        tier = t$tag, skipped = TRUE, skip_reason = t$skip_reason)
      next
    }
    res <- fit_spec(d, dv, rhs, family, zi = zi_formula(t$vars, with_re),
                    sample_rhs = sample_rhs)
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
  if (is.null(last)) return(NULL)
  last$zi_tier_attempts <- attempts
  last
}

mark_collapse <- function(res_re, res_plain) {
  if (!is.null(res_plain) && !is.null(res_re) &&
      identical(res_re$formula, res_plain$formula) &&
      identical(res_re$zi_formula, res_plain$zi_formula)) {
    res_re$collapsed_to <- res_plain$family_tag
  }
  res_re
}

results <- list()

# ------------------------------------------------------------
# STEP 0. Record which family the crossed arm selected, straight out of its
# JSON, so the two cells that inherit a family carry their provenance rather
# than a hardcoded claim about it. Read-only; the crossed artifact is frozen.
# ------------------------------------------------------------
crossed_meta <- list()
if (file.exists(CROSSED_JSON)) {
  cj <- fromJSON(CROSSED_JSON, simplifyVector = FALSE)
  for (cell_name in names(CELLS)) {
    ck <- CELLS[[cell_name]]$crossed_key
    if (is.na(ck)) next
    mk <- sprintf("%s__meta", ck)
    if (!is.null(cj[[mk]])) {
      crossed_meta[[cell_name]] <- list(
        crossed_cell = ck,
        crossed_m1_winner = cj[[mk]]$m1_winner,
        crossed_m3_winner = cj[[mk]]$m3_winner)
    }
    # The anchor values the report checks this arm's `mirror` series against.
    for (m in c("M1", "M2", "M3")) {
      k <- sprintf("%s__%s__%s", ck, m, CELLS[[cell_name]]$family_tag)
      if (!is.null(cj[[k]])) {
        crossed_meta[[cell_name]][[sprintf("anchor_%s", m)]] <- list(
          key = k, sigma2_state = cj[[k]]$sigma2_state, n_obs = cj[[k]]$n_obs,
          aic = cj[[k]]$aic)
      }
    }
  }
}

# ------------------------------------------------------------
# STEP 1. Family selection for viol_2021_ni ONLY.
#
# The other two cells inherit the crossed arm's M3 winner. This cell has no
# precedent -- the crossed arm never fit a violations model without
# log_inspections -- so it races the same eight families at M3, under the same
# rules, and the full AIC table is recorded in the JSON so the choice is
# auditable rather than asserted.
# ------------------------------------------------------------
select_family_at_m3 <- function(cl, cell_name) {
  rhs <- rhs_for(cl, "M3")
  plain_fits <- list(); rows <- list()
  for (rung in LADDER) {
    key <- sprintf("%s__M3__%s__selection", cell_name, rung$tag)
    cat("fitting", key, "(family selection)\n")
    cp_tag <- counterpart_of(rung$tag)
    counterpart <- lget(plain_fits, cp_tag)
    if (rung$zi) {
      res <- fit_zi_ladder(cl$d, cl$dv, rhs, rung$family, rung$zi_re, counterpart)
    } else {
      res <- fit_spec(cl$d, cl$dv, rhs, rung$family, zi = ~0)
      res$zi_tier_reached <- NA_character_; res$zi_tier_attempts <- list()
      res$zi_degenerate <- FALSE; res$zi_degenerate_reason <- ""
      res$zi_loglik_criterion_applied <- FALSE
    }
    res$cell <- cell_name; res$model <- "M3"; res$family_tag <- rung$tag
    res$series <- "selection"; res$outcome <- cl$outcome
    if (rung$zi_re) res <- mark_collapse(res, lget(plain_fits, sub("_re$", "", rung$tag)))
    res$eligible_for_selection <- isTRUE(res$converged) &&
      is.null(res$collapsed_to) && !isTRUE(res$zi_degenerate)
    if (!isTRUE(res$converged)) {
      cat(sprintf("  NOT CONVERGED at %s; message=%s\n", CROSSED_RE, res$message %||% ""))
    }
    results[[key]] <<- res
    plain_fits[[rung$tag]] <- res
    rows[[length(rows) + 1]] <- list(
      family_tag = rung$tag, aic = res$aic %||% NA_real_,
      loglik = res$loglik %||% NA_real_, df = res$df %||% NA_integer_,
      n_obs = res$n_obs, converged = isTRUE(res$converged),
      zi_degenerate = isTRUE(res$zi_degenerate),
      collapsed_to = res$collapsed_to %||% NA_character_,
      zi_tier_reached = res$zi_tier_reached %||% NA_character_,
      eligible = isTRUE(res$eligible_for_selection))
  }
  eligible <- Filter(function(r) isTRUE(r$eligible) && is.finite(r$aic), rows)
  if (length(eligible) == 0) stop("no eligible family for ", cell_name)
  win <- eligible[[which.min(vapply(eligible, function(r) r$aic, numeric(1)))]]
  list(winner = win$family_tag, table = rows, n_eligible = length(eligible))
}

for (cell_name in names(CELLS)) {
  if (!is.na(CELLS[[cell_name]]$family_tag)) next
  cl <- CELLS[[cell_name]]; cl$d <- panel
  sel <- select_family_at_m3(cl, cell_name)
  CELLS[[cell_name]]$family_tag <- sel$winner
  results[[sprintf("%s__selection_meta", cell_name)]] <- list(
    cell = cell_name, selected_family = sel$winner,
    n_eligible = sel$n_eligible, model_selected_at = "M3",
    candidates = sel$table,
    note = paste("Selected by minimum AIC among converged, non-degenerate,",
                 "non-collapsed fits of the same eight-family ladder the",
                 "crossed arm uses, at Model 3. Held FIXED for M1 and M2."))
  cat(sprintf("\n>>> %s: selected family = %s (of %d eligible)\n\n",
              cell_name, sel$winner, sel$n_eligible))
}

# ------------------------------------------------------------
# STEP 2. The stepwise series themselves.
#
# TWO ZI series per cell, because the two answer different questions:
#
#   `mirror`       -- ziformula mirrors the conditional predictors at every
#                     step (the crossed arm's fidelity rule). Consistent with
#                     Jafari, and it reproduces the crossed arm's fits, but the
#                     ZI block GROWS as covariates enter, so a change in
#                     sigma^2_state confounds the conditional covariates with
#                     the ZI block.
#
#   `zi_intercept` -- ziformula = ~1, held CONSTANT across all three steps. The
#                     clean variance-reduction series: the only thing that
#                     changes between M1, M2 and M3 is the conditional
#                     predictor set. This is the one the memo headlines.
#
# A plain (non-zero-inflated) family has no ZI block at all, so the two series
# would be the same fit; in that case one `plain` series is fit and the fact is
# recorded rather than two identical copies being written.
#
# Within each series, M1matched is Model 1's FORMULA on Model 2/3's ROWS. M1
# keeps all 49 states; M2/M3 lose AK/RI/VT, which have no BLS
# pesticide-applicator series (SPEND_APP_z). Without that fit, a Model-1-based
# reduction would silently mix the covariates' effect with a sample change.
# ------------------------------------------------------------
series_for <- function(rung) {
  if (!rung$zi) return(list(list(tag = "plain", mode = "none")))
  list(list(tag = "mirror",       mode = "ladder"),
       list(tag = "zi_intercept", mode = "intercept"))
}

for (cell_name in names(CELLS)) {
  cl <- CELLS[[cell_name]]; cl$d <- panel
  fam_tag <- cl$family_tag
  rung <- rung_of(fam_tag)
  cp_tag <- counterpart_of(fam_tag)

  for (ser in series_for(rung)) {
    for (model in c("M1", "M2", "M3", "M1matched")) {
      rhs <- rhs_for(cl, model)
      samp <- if (model == "M1matched") rhs_for(cl, "M3") else NULL
      key <- sprintf("%s__%s__%s__%s", cell_name, model, fam_tag, ser$tag)
      cat("fitting", key, "\n")

      # The plain-family counterpart at the same (cell, model, sample), used
      # ONLY for the log-likelihood degeneracy criterion. Fitting it here
      # reproduces the crossed arm's tier decisions exactly.
      counterpart <- NULL
      if (rung$zi && !is.na(cp_tag)) {
        cp_rung <- rung_of(cp_tag)
        counterpart <- fit_spec(cl$d, cl$dv, rhs, cp_rung$family, zi = ~0,
                                sample_rhs = samp)
        counterpart$family_tag <- cp_tag
      }

      if (ser$mode == "none") {
        res <- fit_spec(cl$d, cl$dv, rhs, rung$family, zi = ~0, sample_rhs = samp)
        res$zi_tier_reached <- NA_character_; res$zi_tier_attempts <- list()
        res$zi_degenerate <- FALSE; res$zi_degenerate_reason <- ""
        res$zi_loglik_criterion_applied <- FALSE
      } else if (ser$mode == "ladder") {
        res <- fit_zi_ladder(cl$d, cl$dv, rhs, rung$family, rung$zi_re,
                             counterpart, sample_rhs = samp)
      } else {
        res <- fit_spec(cl$d, cl$dv, rhs, rung$family,
                        zi = zi_formula(character(0), rung$zi_re),
                        sample_rhs = samp)
        res$zi_tier_reached <- "intercept"; res$zi_tier_attempts <- list()
        res <- mark_zi_degenerate(res, counterpart)
      }

      res$cell <- cell_name; res$model <- model; res$family_tag <- fam_tag
      res$series <- ser$tag; res$outcome <- cl$outcome
      res$with_inspections <- cl$with_inspections
      res$sample_model <- if (model == "M1matched") "M3" else model
      res$zi_counterpart <- cp_tag
      res$counterpart_aic <- if (is.null(counterpart)) NA_real_
                             else (counterpart$aic %||% NA_real_)
      if (!isTRUE(res$converged)) {
        cat(sprintf("WARNING [%s]: did NOT converge at %s; message=%s\n",
                    key, CROSSED_RE, res$message %||% ""))
      }
      results[[key]] <- res
    }
  }

  results[[sprintf("%s__meta", cell_name)]] <- list(
    cell = cell_name, outcome = cl$outcome, window = 2021L,
    dv = cl$dv, base_terms = cl$base,
    with_inspections = cl$with_inspections,
    family_tag = fam_tag, family_source = cl$family_source,
    family_is_zi = rung$zi, family_has_zi_re = rung$zi_re,
    series = vapply(series_for(rung), function(s) s$tag, character(1)),
    re_structure = CROSSED_RE,
    crossed_arm = crossed_meta[[cell_name]] %||% NULL,
    m2_add = M2_ADD, m3_add = setdiff(M3_ADD, M2_ADD))
}

# ------------------------------------------------------------
# STEP 3. MATCHED-FAMILY COMPANION FITS.
#
# WHY. `viol_2021_wi` and `viol_2021_ni` are meant to differ in exactly one
# thing -- `log_inspections` in the conditional block -- but per-cell family
# selection made them differ in TWO: `wi` carries a state random intercept in
# its zero-inflation block (ZINB2 + ZI-RE, inherited from the crossed arm's M3
# winner) and `ni` does not (plain ZINB2, which won its own race by 14.31 AIC).
# Between-state heterogeneity in the zero process therefore has somewhere to go
# in `wi` and nowhere to go in `ni`, where it must load onto the conditional
# sigma^2_state. Comparing the two as selected confounds the covariate with the
# variance structure.
#
# WHAT. Each violations cell is refit under the OTHER one's family, so the
# comparison can be made with family held constant. BOTH directions are fit,
# never one: each direction forces a cell onto a family that loses for that
# cell, so neither is privileged, and together they BRACKET the answer.
#
# SCOPE. The headline `zi_intercept` series only (M1, M2, M3, M1matched). The
# `mirror` series is explicitly secondary and already carries its own caveat;
# duplicating it here would add fits nobody reads.
#
# STATUS. These are COMPANIONS, not reported models. Each cell's own selected
# family is untouched, every record here is tagged `is_companion = TRUE`, and
# the key carries a `__companion` suffix so nothing downstream can mistake one
# for a selected fit.
#
# The job list is DERIVED, not typed: for every ordered pair of cells sharing a
# DV whose selected families differ, fit the first under the second's family.
# If the two cells ever land on the same family, this loop produces nothing --
# correctly, because then no companion is needed.
# ------------------------------------------------------------
companion_jobs <- list()
for (a in names(CELLS)) for (b in names(CELLS)) {
  if (identical(a, b)) next
  if (!identical(CELLS[[a]]$dv, CELLS[[b]]$dv)) next
  if (identical(CELLS[[a]]$family_tag, CELLS[[b]]$family_tag)) next
  companion_jobs[[length(companion_jobs) + 1]] <- list(
    cell = a, family_tag = CELLS[[b]]$family_tag, matched_to = b,
    own_family_tag = CELLS[[a]]$family_tag)
}

for (job in companion_jobs) {
  cl <- CELLS[[job$cell]]; cl$d <- panel
  rung <- rung_of(job$family_tag)
  cp_tag <- counterpart_of(job$family_tag)
  # The headline series is `zi_intercept`, which only exists for a ZI family.
  # A plain family has no ZI block, so there would be nothing to hold constant.
  if (!rung$zi) stop("companion family ", job$family_tag, " is not ",
                     "zero-inflated; the zi_intercept series is undefined for it")

  for (model in c("M1", "M2", "M3", "M1matched")) {
    rhs  <- rhs_for(cl, model)
    samp <- if (model == "M1matched") rhs_for(cl, "M3") else NULL
    key  <- sprintf("%s__%s__%s__zi_intercept__companion",
                    job$cell, model, job$family_tag)
    cat("fitting", key, "(matched-family companion)\n")

    counterpart <- NULL
    if (!is.na(cp_tag)) {
      cp_rung <- rung_of(cp_tag)
      counterpart <- fit_spec(cl$d, cl$dv, rhs, cp_rung$family, zi = ~0,
                              sample_rhs = samp)
      counterpart$family_tag <- cp_tag
    }

    res <- fit_spec(cl$d, cl$dv, rhs, rung$family,
                    zi = zi_formula(character(0), rung$zi_re),
                    sample_rhs = samp)
    res$zi_tier_reached <- "intercept"; res$zi_tier_attempts <- list()
    res <- mark_zi_degenerate(res, counterpart)

    res$cell <- job$cell; res$model <- model
    res$family_tag <- job$family_tag
    res$series <- "zi_intercept"; res$outcome <- cl$outcome
    res$with_inspections <- cl$with_inspections
    res$sample_model <- if (model == "M1matched") "M3" else model
    res$zi_counterpart <- cp_tag
    res$counterpart_aic <- if (is.null(counterpart)) NA_real_
                           else (counterpart$aic %||% NA_real_)
    # The tags that keep a companion from ever being read as a selected fit.
    res$is_companion <- TRUE
    res$family_matched_to <- job$matched_to
    res$own_family_tag <- job$own_family_tag
    res$companion_of <- sprintf("%s__%s__%s__zi_intercept",
                                job$cell, model, job$own_family_tag)
    if (!isTRUE(res$converged)) {
      cat(sprintf("WARNING [%s]: companion did NOT converge at %s; message=%s\n",
                  key, CROSSED_RE, res$message %||% ""))
    }
    results[[key]] <- res
  }
}

results[["__companion_meta"]] <- list(
  n_jobs = length(companion_jobs),
  series = "zi_intercept",
  models = c("M1", "M2", "M3", "M1matched"),
  jobs = lapply(companion_jobs, function(j) list(
    cell = j$cell, own_family_tag = j$own_family_tag,
    companion_family_tag = j$family_tag, family_matched_to = j$matched_to)),
  note = paste("Matched-family companions for the with/without-inspections",
               "comparison. Each violations cell refit under the OTHER cell's",
               "selected family so that family can be held constant. BOTH",
               "directions are fit; neither is privileged, and together they",
               "bracket what log(inspections) absorbs. NOT reported models:",
               "every reported series still uses its cell's own selected",
               "family, which these fits do not change."))

results[["__run_meta"]] <- list(
  script = "scripts/jafari_stepwise_models.R",
  panel = panel_csv, window = "2011-2021 (WPS view)",
  re_structure = CROSSED_RE,
  n_panel_rows = nrow(panel), n_panel_states = length(unique(panel$state)),
  glmmTMB_version = as.character(utils::packageVersion("glmmTMB")),
  R_version = R.version.string, seed = SEED, n_sim = N_SIM,
  note = paste("Stepwise M1 -> M2 -> M3 variance-reduction series with the",
               "family HELD FIXED within each cell. Derived quantities",
               "(Delta sigma^2, IRR, OR) are computed in",
               "scripts/report_jafari_stepwise.py, never here."))

write_json(results, out_json, auto_unbox = TRUE, digits = 10, na = "null")
cat("\nWrote", length(results), "entries to", out_json, "\n")
n_bad <- sum(vapply(results, function(r) !is.null(r$converged) && !isTRUE(r$converged),
                    logical(1)))
if (n_bad > 0) cat("WARNING:", n_bad, "fit(s) did not converge -- see 'converged' flags\n")
