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
RS_RE <- "(1 + time | state)"
RI_RE <- "(1 | state)"
RE_FALLBACK <- c(RS_RE, RI_RE)

fit_with_fallback <- function(d, dv, rhs, family, zi, offset_col, re_fallback = RE_FALLBACK) {
  last <- NULL
  for (re in re_fallback) {
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

# Ruling R13 (2026-08-20 coordinator review): AIC alone is a weak argument for
# zero-inflation -- the observed-vs-expected zero count is the direct evidence,
# so every fit carries its own discrepancy, visible without re-deriving it later.
add_zero_fit_discrepancy <- function(res) {
  obs <- res$obs_zeros; exp <- res$exp_zeros
  res$zero_fit_discrepancy <- if (is.null(obs) || is.null(exp) || is.na(obs) || is.na(exp))
    NA_real_ else abs(exp - obs) / max(obs, 1)
  res
}

# Ruling R14: wherever zinb_re's ZI random intercept downgrades to a plain ~1
# ZI, it becomes formula-for-formula identical to the "zinb" rung already fit
# in this same (cell, model, tier) -- same rhs, same family, same zi, same re,
# same data. That is not a sixth family, it is zinb wearing zinb_re's name.
# Mark it so winner selection treats it as a duplicate, not a distinct
# competitor (and so no cell can report a winner that is a duplicate rung).
mark_collapse <- function(res_zinb_re, res_zinb) {
  if (!is.null(res_zinb) && isTRUE(res_zinb_re$zi_downgraded) &&
      identical(res_zinb_re$formula, res_zinb$formula) &&
      identical(res_zinb_re$zi_formula, res_zinb$zi_formula) &&
      identical(res_zinb_re$re_used, res_zinb$re_used)) {
    res_zinb_re$collapsed_to <- "zinb"
  }
  res_zinb_re
}

# Ruling R12: lowest-AIC family among a set of candidate fits, excluding any
# collapsed duplicate (R14). Falls back to "nbinom2" on total non-convergence
# or an empty candidate set -- same rule as the original Task 4 tie-break.
pick_winner <- function(fit_list) {
  eligible <- Filter(function(r) isTRUE(r$converged) && is.finite(r$aic) &&
                        is.null(r$collapsed_to), fit_list)
  if (length(eligible) == 0) return(list(tag = "nbinom2"))
  list(tag = eligible[[which.min(vapply(eligible, function(r) r$aic, numeric(1)))]]$family_tag)
}

# Lowest zero-fit-discrepancy family among a set of candidate fits (R13),
# ignoring collapsed duplicates for the same reason as pick_winner.
pick_best_zero_fit <- function(fit_list) {
  eligible <- Filter(function(r) isTRUE(r$converged) && is.finite(r$zero_fit_discrepancy) &&
                        is.null(r$collapsed_to), fit_list)
  if (length(eligible) == 0) return(NA_character_)
  eligible[[which.min(vapply(eligible, function(r) r$zero_fit_discrepancy, numeric(1)))]]$family_tag
}

M2_ADD <- c("SPEND_APP_z", "SPEND_WORK_z", "lii_2017_z")
M3_ADD <- c(M2_ADD, "h2a_per_farmworker_z", "dol_demand_met_pct_z", "pct_flc_z")
FAMILY_TAGS_R <- vapply(LADDER, `[[`, "", "tag")

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

  # ---- Tier 1: full ladder at M3 (the model a family must survive with every
  # covariate present) and M1 (stability check), constraint-compliant RE with
  # the documented convergence fallback (spec 6). ----
  for (model in c("M1", "M3")) {
    zinb_res <- NULL
    for (rung in LADDER) {
      key <- sprintf("%s__%s__%s", cell_name, model, rung$tag)
      cat("fitting", key, "\n")
      res <- fit_with_fallback(cl$d, cl$dv, specs[[model]],
                               rung$family, rung$zi, cl$offset_col)
      res$cell <- cell_name; res$model <- model; res$family_tag <- rung$tag
      res$window <- cl$window; res$outcome <- cl$outcome; res$exposure <- cl$exposure
      res$re_tier <- if (identical(res$re_used, RS_RE)) "rs" else "ri"
      if (rung$tag == "zinb") zinb_res <- res
      if (rung$tag == "zinb_re") res <- mark_collapse(res, zinb_res)
      res <- add_zero_fit_discrepancy(res)
      results[[key]] <- res
    }
  }

  # ---- Ruling R12: does this cell need a second, RE-homogeneous tier? True
  # iff at least one family in the tier-1 M1 or M3 ladder could not reach the
  # constraint-compliant (1 + time | state) structure -- comparing its AIC
  # against families that DID reach it conflates "this family needs a simpler
  # RE structure" with "this family fits the data better." ----
  tier1_keys <- sprintf("%s__%s__%s", cell_name, rep(c("M1", "M3"), each = length(FAMILY_TAGS_R)),
                        rep(FAMILY_TAGS_R, times = 2))
  needs_tier2 <- any(vapply(results[tier1_keys], function(r) r$re_tier == "ri", logical(1)))

  # ---- Tier 2 (only when needed): force ALL SIX families to (1 | state), so
  # this tier is internally apples-to-apples even though tier 1 is not. ----
  if (needs_tier2) {
    for (model in c("M1", "M3")) {
      zinb_res2 <- NULL
      for (rung in LADDER) {
        key <- sprintf("%s__%s__%s__ri2", cell_name, model, rung$tag)
        cat("fitting", key, "(tier-2, forced (1 | state))\n")
        res <- fit_with_fallback(cl$d, cl$dv, specs[[model]],
                                 rung$family, rung$zi, cl$offset_col,
                                 re_fallback = RI_RE)
        res$cell <- cell_name; res$model <- model; res$family_tag <- rung$tag
        res$window <- cl$window; res$outcome <- cl$outcome; res$exposure <- cl$exposure
        res$re_tier <- "ri"
        if (rung$tag == "zinb") zinb_res2 <- res
        if (rung$tag == "zinb_re") res <- mark_collapse(res, zinb_res2)
        res <- add_zero_fit_discrepancy(res)
        results[[key]] <- res
      }
    }
  }

  # ---- Per-cell winners, one per tier -- never collapsed into one headline
  # (R12). M2 is fit under winner_rs: (1 + time | state) is this project's
  # stated RE spec (spec 5.2); (1 | state) is a documented convergence
  # fallback, not an alternative spec to pick a headline model from. ----
  m3_rs <- Filter(function(r) r$re_tier == "rs",
                   results[sprintf("%s__M3__%s", cell_name, FAMILY_TAGS_R)])
  m1_rs <- Filter(function(r) r$re_tier == "rs",
                   results[sprintf("%s__M1__%s", cell_name, FAMILY_TAGS_R)])
  w_rs_m3 <- pick_winner(m3_rs)$tag
  w_rs_m1 <- pick_winner(m1_rs)$tag
  best_zero_rs <- pick_best_zero_fit(m3_rs)

  w_ri_m3 <- NA_character_; w_ri_m1 <- NA_character_; best_zero_ri <- NA_character_
  stable_ri <- NA
  if (needs_tier2) {
    m3_ri <- results[sprintf("%s__M3__%s__ri2", cell_name, FAMILY_TAGS_R)]
    m1_ri <- results[sprintf("%s__M1__%s__ri2", cell_name, FAMILY_TAGS_R)]
    w_ri_m3 <- pick_winner(m3_ri)$tag
    w_ri_m1 <- pick_winner(m1_ri)$tag
    best_zero_ri <- pick_best_zero_fit(m3_ri)
    stable_ri <- identical(w_ri_m1, w_ri_m3)
  }

  # Ruling R13: is the AIC winner ALSO the best zero-fit family? An explicit,
  # visible flag per tier so a later task does not have to re-derive it.
  winner_rs_is_best_zero_fit <- !is.na(best_zero_rs) && identical(best_zero_rs, w_rs_m3)
  winner_ri_is_best_zero_fit <- if (needs_tier2)
    (!is.na(best_zero_ri) && identical(best_zero_ri, w_ri_m3)) else NA

  if (!winner_rs_is_best_zero_fit) {
    cat(sprintf("NOTE [%s]: AIC winner at (1+time|state) is '%s' but best zero-fit family is '%s' -- AIC and zero-count evidence disagree.\n",
                cell_name, w_rs_m3, best_zero_rs))
  }
  if (needs_tier2 && !isTRUE(winner_ri_is_best_zero_fit)) {
    cat(sprintf("NOTE [%s]: AIC winner at (1|state) is '%s' but best zero-fit family is '%s' -- AIC and zero-count evidence disagree.\n",
                cell_name, w_ri_m3, best_zero_ri))
  }

  results[[sprintf("%s__meta", cell_name)]] <- list(
    cell = cell_name, needs_tier2 = needs_tier2,
    m1_winner_rs = w_rs_m1, m3_winner_rs = w_rs_m3, stable_rs = identical(w_rs_m1, w_rs_m3),
    m1_winner_ri = w_ri_m1, m3_winner_ri = w_ri_m3, stable_ri = stable_ri,
    m2_family = w_rs_m3,
    best_zero_fit_rs = best_zero_rs, best_zero_fit_ri = best_zero_ri,
    winner_rs_is_best_zero_fit = winner_rs_is_best_zero_fit,
    winner_ri_is_best_zero_fit = winner_ri_is_best_zero_fit
  )

  # M2 under winner_rs only (spec: M2 fit under the single winning family).
  win <- Filter(function(r) r$tag == w_rs_m3, LADDER)[[1]]
  key <- sprintf("%s__M2__%s", cell_name, w_rs_m3)
  cat("fitting", key, "(winning family at M3, constraint-compliant tier)\n")
  res <- fit_with_fallback(cl$d, cl$dv, c(cl$base, M2_ADD),
                           win$family, win$zi, cl$offset_col)
  res$cell <- cell_name; res$model <- "M2"; res$family_tag <- w_rs_m3
  res$window <- cl$window; res$outcome <- cl$outcome; res$exposure <- cl$exposure
  res$re_tier <- if (identical(res$re_used, RS_RE)) "rs" else "ri"
  res <- add_zero_fit_discrepancy(res)
  results[[key]] <- res
}

write_json(results, out_json, auto_unbox = TRUE, digits = 10, na = "null")
cat("\nWrote", length(results), "entries to", out_json, "\n")
n_bad <- sum(vapply(results, function(r) !is.null(r$converged) && !isTRUE(r$converged), logical(1)))
if (n_bad > 0) cat("WARNING:", n_bad, "fit(s) did not converge -- see 'converged' flags\n")
