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
# Ruling R17 (2026-08-20 coordinator review): raised from 200 to 2000 so the
# Monte Carlo SE of exp_zeros (sd/sqrt(N_SIM)) is tight enough to distinguish a
# genuine zero-fit difference from simulation noise -- at N_SIM=200 the
# reviewer measured +/-1-2 counts of noise between numerically identical fits.
N_SIM <- 2000L

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
                     re = "(1 + time | state)", REML = FALSE,
                     control = glmmTMBControl()) {
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
      glmmTMB(form, ziformula = zi, family = family, data = dd, REML = REML,
              control = control),
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
  # Skipped for Gaussian, where "zero" is not a meaningful outcome. Ruling R17:
  # also computes the Monte Carlo SE of exp_zeros (sd/sqrt(N_SIM)) so a later
  # comparison of two fits' zero counts can be judged against simulation
  # noise instead of as if it were noise-free.
  obs_zeros <- exp_zeros <- exp_zeros_se <- NA_real_
  if (!is_gaussian) {
    obs_zeros <- sum(dd[[dv]] == 0, na.rm = TRUE)
    sims <- tryCatch(as.data.frame(simulate(fit, nsim = N_SIM, seed = SEED)),
                     error = function(e) NULL)
    if (!is.null(sims)) {
      zero_counts <- colSums(sims == 0)
      exp_zeros <- mean(zero_counts)
      exp_zeros_se <- stats::sd(zero_counts) / sqrt(N_SIM)
    }
  }

  c(base, list(converged = converged,
               message = if (length(warn_msgs)) paste(warn_msgs, collapse = " | ") else "",
               pd_hess = pd_hess, conv_code = conv_code,
               aic = AIC(fit), bic = BIC(fit),
               loglik = as.numeric(logLik(fit)), df = attr(logLik(fit), "df"),
               cond = cond, zi = zi_co,
               sigma2_u0 = s2_u0, sigma2_u1 = s2_u1, sigma_u01 = s_u01,
               sigma2_e = s2_e,
               obs_zeros = obs_zeros, exp_zeros = exp_zeros, exp_zeros_se = exp_zeros_se))
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

# Ruling R17 (2026-08-20 coordinator review, Critical): extends R14's collapse
# concept to ZI-BOUNDARY DEGENERACY -- a ZI fit that reports converged=TRUE but
# whose zero-inflation parameter has wandered to the edge of identifiability
# (huge intercept magnitude, huge SE, or a log-likelihood indistinguishable
# from dropping ZI entirely). Unlike R14's collapse (an EXACT formula
# duplicate), this catches a fit that is nominally distinct but statistically
# meaningless as a zero-inflation model -- e.g. viol_cov_2019__M3__zinb, ZI
# intercept -21.43 (SE 2533), loglik identical to nbinom2 to 1.6e-07. This is a
# SELECTION-ELIGIBILITY flag, not deletion: the record stays in the JSON with
# its `zi_degenerate` / `zi_degenerate_reason` fields so a later task can still
# see exactly what glmmTMB reported.
mark_zi_degenerate <- function(res, counterpart) {
  reasons <- character(0)
  zi_int <- res$zi[["(Intercept)"]]
  if (!is.null(zi_int)) {
    if (is.finite(zi_int$b) && abs(zi_int$b) > 15) reasons <- c(reasons, "|zi_intercept| > 15")
    if (is.finite(zi_int$se) && zi_int$se > 100) reasons <- c(reasons, "se(zi_intercept) > 100")
  }
  if (!is.null(counterpart) && isTRUE(res$converged) && isTRUE(counterpart$converged) &&
      is.finite(res$loglik) && is.finite(counterpart$loglik) &&
      abs(res$loglik - counterpart$loglik) < 1e-4) {
    reasons <- c(reasons, "loglik matches non-ZI counterpart within 1e-4")
  }
  res$zi_degenerate <- length(reasons) > 0
  res$zi_degenerate_reason <- paste(reasons, collapse = "; ")
  res
}

# Ruling R12: lowest-AIC family among a set of candidate fits, excluding any
# collapsed duplicate (R14) or ZI-boundary-degenerate fit (R17). Falls back to
# "nbinom2" on total non-convergence or an empty candidate set -- same rule as
# the original Task 4 tie-break.
pick_winner <- function(fit_list) {
  eligible <- Filter(function(r) isTRUE(r$converged) && is.finite(r$aic) &&
                        is.null(r$collapsed_to) && !isTRUE(r$zi_degenerate), fit_list)
  if (length(eligible) == 0) return(list(tag = "nbinom2"))
  list(tag = eligible[[which.min(vapply(eligible, function(r) r$aic, numeric(1)))]]$family_tag)
}

# Lowest zero-fit-discrepancy family among a set of candidate fits (R13),
# ignoring collapsed duplicates and ZI-degenerate fits for the same reason as
# pick_winner.
pick_best_zero_fit <- function(fit_list) {
  eligible <- Filter(function(r) isTRUE(r$converged) && is.finite(r$zero_fit_discrepancy) &&
                        is.null(r$collapsed_to) && !isTRUE(r$zi_degenerate), fit_list)
  if (length(eligible) == 0) return(NA_character_)
  eligible[[which.min(vapply(eligible, function(r) r$zero_fit_discrepancy, numeric(1)))]]$family_tag
}

# Shared default for "no zero-fit comparison was made" (needs_tier2 == FALSE,
# or nothing eligible). Extracted so credible_better_zero_fit()'s internal
# default and the cbz_ri else-branch cannot drift apart -- that exact drift
# (a field present in one hand-built list but not the other) is what produced
# the `{}`-instead-of-`null` bug in the R12/13/14 pass.
empty_zero_fit_result <- function() {
  list(zero_signed = NA_real_, zero_signed_rel = NA_real_,
       zero_discrepancy = NA_real_, zero_discrepancy_se = NA_real_,
       better_tag = NA_character_, better_discrepancy = NA_real_, has_better = FALSE)
}

# Ruling R15 (supersedes R13's flag), refined by Ruling R17 (Critical):
# "is the AIC winner also the best zero-fit family" fired False against
# comparators no one would ever report (e.g. ZIP beating the AIC winner by
# fitting only the zero count, at a cost of +2000 AIC) -- fixed in the R15
# pass by restricting comparators to a CREDIBLE (delta_aic <= 10) set. R17
# found a second defect: the flag also fired on differences SMALLER than the
# Monte Carlo simulation noise in exp_zeros (viol_cov_2019__M3__zinb vs
# nbinom2, 21.495 vs 21.685 expected zeros -- a ~0.4x-SE "difference"), and
# that specific comparator (zinb) turned out to be ZI-boundary-degenerate
# (R17) in the first place. Both are fixed here: the comparator pool excludes
# zi_degenerate fits, and an "improvement" must exceed 2x the combined MC SE
# of the two fits' exp_zeros estimates (converted to the discrepancy scale) to
# count as credible.
credible_better_zero_fit <- function(winner_tag, candidates, delta_aic = 10) {
  winner <- Find(function(r) identical(r$family_tag, winner_tag), candidates)
  out <- empty_zero_fit_result()
  if (is.null(winner) || is.null(winner$obs_zeros) || is.null(winner$exp_zeros) ||
      is.na(winner$obs_zeros) || is.na(winner$exp_zeros)) return(out)
  denom <- max(winner$obs_zeros, 1)
  out$zero_signed <- winner$exp_zeros - winner$obs_zeros
  out$zero_signed_rel <- out$zero_signed / denom
  out$zero_discrepancy <- winner$zero_fit_discrepancy
  win_se <- winner$exp_zeros_se
  out$zero_discrepancy_se <- if (!is.null(win_se) && is.finite(win_se)) win_se / denom else NA_real_
  if (!is.finite(out$zero_discrepancy)) return(out)
  pool <- Filter(function(r) !identical(r$family_tag, winner_tag) &&
                    isTRUE(r$converged) && is.finite(r$aic) && is.finite(r$zero_fit_discrepancy) &&
                    is.null(r$collapsed_to) && !isTRUE(r$zi_degenerate) &&
                    (r$aic - winner$aic) <= delta_aic,
                 candidates)
  if (length(pool) == 0) return(out)
  best <- pool[[which.min(vapply(pool, function(r) r$zero_fit_discrepancy, numeric(1)))]]
  best_se <- best$exp_zeros_se
  best_disc_se <- if (!is.null(best_se) && is.finite(best_se)) best_se / denom else NA_real_
  improvement <- out$zero_discrepancy - best$zero_fit_discrepancy
  combined_se <- if (is.finite(out$zero_discrepancy_se) && is.finite(best_disc_se))
    sqrt(out$zero_discrepancy_se^2 + best_disc_se^2) else NA_real_
  if (is.finite(improvement) && is.finite(combined_se) && improvement > 2 * combined_se) {
    out$better_tag <- best$family_tag
    out$better_discrepancy <- best$zero_fit_discrepancy
    out$has_better <- TRUE
  }
  out
}

# Ruling R16 (refined for R18): tier membership is read from `re_tier`, NEVER
# parsed from the key string -- the "__ri2" suffix exists only to avoid a
# dict-key collision when a family has two genuinely different fits (one per
# tier); it carries no meaning by itself. `resolve_tier_key` is the single
# source of truth for "which key holds family X's fit at tier Y" -- used both
# to READ a fit (`tier_fit`) and to WRITE the `is_winner_*` flag onto the
# correct record (R18), so the two operations cannot disagree about where a
# family's tier-2 fit lives.
resolve_tier_key <- function(cell_name, model, tag, tier) {
  plain_key <- sprintf("%s__%s__%s", cell_name, model, tag)
  plain <- results[[plain_key]]
  if (!is.null(plain) && identical(plain$re_tier, tier)) return(plain_key)
  if (identical(tier, "ri")) {
    ri2_key <- sprintf("%s__%s__%s__ri2", cell_name, model, tag)
    ri2 <- results[[ri2_key]]
    if (!is.null(ri2) && identical(ri2$re_tier, "ri")) return(ri2_key)
  }
  NA_character_
}

tier_fit <- function(cell_name, model, tag, tier) {
  k <- resolve_tier_key(cell_name, model, tag, tier)
  if (is.na(k)) NULL else results[[k]]
}

# Family tags that actually achieved a given tier (converged, not a collapsed
# duplicate, not ZI-boundary-degenerate) -- so the per-tier candidate SET is
# itself visible, not just the winner. This is what makes an M1-vs-M3 "winner
# change" interpretable as either a preference reversal or (as in
# viol_off_2021) a change in which families could even reach that tier.
eligible_tags <- function(fit_list) {
  ok <- Filter(function(r) isTRUE(r$converged) && is.null(r$collapsed_to) &&
                 !isTRUE(r$zi_degenerate), fit_list)
  sort(vapply(ok, function(r) r$family_tag, character(1)))
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
    zinb_res <- NULL; poisson_res <- NULL; nbinom2_res <- NULL
    for (rung in LADDER) {
      key <- sprintf("%s__%s__%s", cell_name, model, rung$tag)
      cat("fitting", key, "\n")
      res <- fit_with_fallback(cl$d, cl$dv, specs[[model]],
                               rung$family, rung$zi, cl$offset_col)
      res$cell <- cell_name; res$model <- model; res$family_tag <- rung$tag
      res$window <- cl$window; res$outcome <- cl$outcome; res$exposure <- cl$exposure
      res$re_tier <- if (identical(res$re_used, RS_RE)) "rs" else "ri"

      # R17: ZI-boundary degeneracy, checked against the non-ZI counterpart at
      # the SAME tier (poisson for zip; nbinom2 for zinb/zinb_re).
      if (rung$tag %in% c("zip", "zinb", "zinb_re")) {
        counterpart <- if (rung$tag == "zip") poisson_res else nbinom2_res
        res$zi_degenerate <- FALSE; res$zi_degenerate_reason <- ""
        res <- mark_zi_degenerate(res, counterpart)
      } else {
        res$zi_degenerate <- FALSE; res$zi_degenerate_reason <- ""
      }
      if (rung$tag == "zinb_re") res <- mark_collapse(res, zinb_res)

      # R18: selection-eligibility fields live ON the record, not just derived
      # in `__meta` -- a consumer that only knows plain keys (e.g. Task 5's
      # selection rule, which does not know the __ri2 convention) can still
      # get the tier-1 winner right by filtering on these fields alone.
      res$is_winner_rs <- FALSE; res$is_winner_ri <- FALSE
      res$eligible_for_selection <- isTRUE(res$converged) && is.null(res$collapsed_to) &&
        !isTRUE(res$zi_degenerate)

      res <- add_zero_fit_discrepancy(res)
      results[[key]] <- res

      if (rung$tag == "poisson") poisson_res <- res
      if (rung$tag == "nbinom2") nbinom2_res <- res
      if (rung$tag == "zinb") zinb_res <- res
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
  # Ruling R16: a family whose tier-1 attempt ALREADY fell back to
  # `(1 | state)` already IS this tier's fit for that family -- refitting it
  # again under a "__ri2" key would store the identical fit twice. Skip those
  # and reuse the tier-1 record (tier membership is read from `re_tier`, so
  # `tier_fit()` finds it either way regardless of which key holds it).
  if (needs_tier2) {
    for (model in c("M1", "M3")) {
      zinb_res2 <- NULL; poisson_res2 <- NULL; nbinom2_res2 <- NULL
      for (rung in LADDER) {
        tier1_rec <- results[[sprintf("%s__%s__%s", cell_name, model, rung$tag)]]
        if (!is.null(tier1_rec) && identical(tier1_rec$re_tier, "ri")) {
          if (rung$tag == "poisson") poisson_res2 <- tier1_rec
          if (rung$tag == "nbinom2") nbinom2_res2 <- tier1_rec
          if (rung$tag == "zinb") zinb_res2 <- tier1_rec
          next
        }
        key <- sprintf("%s__%s__%s__ri2", cell_name, model, rung$tag)
        cat("fitting", key, "(tier-2, forced (1 | state))\n")
        res <- fit_with_fallback(cl$d, cl$dv, specs[[model]],
                                 rung$family, rung$zi, cl$offset_col,
                                 re_fallback = RI_RE)
        res$cell <- cell_name; res$model <- model; res$family_tag <- rung$tag
        res$window <- cl$window; res$outcome <- cl$outcome; res$exposure <- cl$exposure
        res$re_tier <- "ri"

        if (rung$tag %in% c("zip", "zinb", "zinb_re")) {
          counterpart <- if (rung$tag == "zip") poisson_res2 else nbinom2_res2
          res$zi_degenerate <- FALSE; res$zi_degenerate_reason <- ""
          res <- mark_zi_degenerate(res, counterpart)
        } else {
          res$zi_degenerate <- FALSE; res$zi_degenerate_reason <- ""
        }
        if (rung$tag == "zinb_re") res <- mark_collapse(res, zinb_res2)

        res$is_winner_rs <- FALSE; res$is_winner_ri <- FALSE
        res$eligible_for_selection <- isTRUE(res$converged) && is.null(res$collapsed_to) &&
          !isTRUE(res$zi_degenerate)

        res <- add_zero_fit_discrepancy(res)
        results[[key]] <- res

        if (rung$tag == "poisson") poisson_res2 <- res
        if (rung$tag == "nbinom2") nbinom2_res2 <- res
        if (rung$tag == "zinb") zinb_res2 <- res
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

  w_ri_m3 <- NA_character_; w_ri_m1 <- NA_character_
  stable_ri <- NA
  eligible_ri_m1 <- character(0); eligible_ri_m3 <- character(0)
  m3_ri <- list()
  if (needs_tier2) {
    m3_ri <- Filter(Negate(is.null), lapply(FAMILY_TAGS_R, function(t) tier_fit(cell_name, "M3", t, "ri")))
    m1_ri <- Filter(Negate(is.null), lapply(FAMILY_TAGS_R, function(t) tier_fit(cell_name, "M1", t, "ri")))
    w_ri_m3 <- pick_winner(m3_ri)$tag
    w_ri_m1 <- pick_winner(m1_ri)$tag
    stable_ri <- identical(w_ri_m1, w_ri_m3)
    eligible_ri_m1 <- eligible_tags(m1_ri)
    eligible_ri_m3 <- eligible_tags(m3_ri)
  }

  # Ruling R15/R17: report the WINNER's own zero-fit facts unconditionally,
  # and only flag a "better zero-fit exists" competitor if one is (a) within
  # delta_aic=10 of the winner, (b) not ZI-boundary-degenerate, and (c) beats
  # the winner by more than 2x the combined Monte Carlo SE.
  cbz_rs <- credible_better_zero_fit(w_rs_m3, m3_rs)
  cbz_ri <- if (needs_tier2) credible_better_zero_fit(w_ri_m3, m3_ri) else empty_zero_fit_result()

  if (cbz_rs$has_better) {
    cat(sprintf("NOTE [%s]: within 10 AIC of the (1+time|state) winner '%s' (disc %.3f), '%s' fits zeros credibly better (disc %.3f), exceeding 2x MC SE.\n",
                cell_name, w_rs_m3, cbz_rs$zero_discrepancy, cbz_rs$better_tag, cbz_rs$better_discrepancy))
  }
  if (needs_tier2 && cbz_ri$has_better) {
    cat(sprintf("NOTE [%s]: within 10 AIC of the (1|state) winner '%s' (disc %.3f), '%s' fits zeros credibly better (disc %.3f), exceeding 2x MC SE.\n",
                cell_name, w_ri_m3, cbz_ri$zero_discrepancy, cbz_ri$better_tag, cbz_ri$better_discrepancy))
  }

  # Ruling R19 prep: does the rs-tier winner carry a live glmmTMB warning while
  # a clean (no-message) competitor sits within 10 AIC? Reported, not acted on
  # -- the interpretation and any winner change are the PI's call.
  winner_rs_fit <- Find(function(r) identical(r$family_tag, w_rs_m3), m3_rs)
  winner_rs_has_warning <- !is.null(winner_rs_fit) && nzchar(winner_rs_fit$message)
  clean_rs_competitor <- NA_character_
  if (winner_rs_has_warning) {
    clean_pool <- Filter(function(r) !identical(r$family_tag, w_rs_m3) &&
                            isTRUE(r$converged) && !nzchar(r$message) &&
                            is.null(r$collapsed_to) && !isTRUE(r$zi_degenerate) &&
                            (r$aic - winner_rs_fit$aic) <= 10,
                          m3_rs)
    if (length(clean_pool)) {
      clean_rs_competitor <- clean_pool[[which.min(vapply(clean_pool, function(r) r$aic, numeric(1)))]]$family_tag
    }
    cat(sprintf("NOTE [%s]: rs-tier winner '%s' carries a live warning (%s)%s\n",
                cell_name, w_rs_m3, winner_rs_fit$message,
                if (!is.na(clean_rs_competitor))
                  sprintf(" -- clean competitor '%s' sits within 10 AIC.", clean_rs_competitor)
                else " -- no clean competitor within 10 AIC."))
  }

  # I-2: machine-readable flag that this cell's DV zeros are (at least partly)
  # `fillna(0)`-manufactured missingness, not measured non-occurrence -- see
  # the panel-invariant checks in [1]. Verified directly against the M3
  # analytic sample (423 rows): its 4 surviving zero-inspection state-years
  # are Minnesota-2019, Montana-2011, Montana-2015, and Utah-2013, all of
  # which have >=1 raw inspections-epa/inspections-state component missing in
  # the source CSV. Alaska's 8 zero-inspection state-years and Vermont's 3
  # drop from M1 (450 rows) to M3 (423 rows) via BLS-applicator listwise
  # deletion, not because they stopped being zero.
  zeros_partly_manufactured <- identical(cell_name, "insp_2019")
  zeros_partly_manufactured_note <- if (zeros_partly_manufactured)
    paste("4 of 4 surviving M3 zero-inspection rows (Minnesota-2019, Montana-2011,",
          "Montana-2015, Utah-2013) are fillna(0)-manufactured missingness, not",
          "measured non-inspection (see build_count_model_panel.py). Alaska's 8 and",
          "Vermont's 3 zero-inspection rows drop via BLS-applicator listwise",
          "deletion before M3, not because they stopped being zero.") else ""

  results[[sprintf("%s__meta", cell_name)]] <- list(
    cell = cell_name, needs_tier2 = needs_tier2,
    m1_winner_rs = w_rs_m1, m3_winner_rs = w_rs_m3, stable_rs = identical(w_rs_m1, w_rs_m3),
    m1_winner_ri = w_ri_m1, m3_winner_ri = w_ri_m3, stable_ri = stable_ri,
    m2_family = w_rs_m3,
    # Documents the CANDIDATE SET per tier/model (R16 follow-up), so an
    # M1-vs-M3 winner change is interpretable as a preference reversal vs. a
    # change in which families could even reach that tier (see viol_off_2021).
    eligible_rs_m1 = eligible_tags(m1_rs), eligible_rs_m3 = eligible_tags(m3_rs),
    eligible_ri_m1 = eligible_ri_m1, eligible_ri_m3 = eligible_ri_m3,
    # R15/R17: the winner's own zero-fit facts, always reported, plus a
    # credible-set-and-noise-aware "better zero-fit exists" flag/name.
    winner_rs_zero_signed = cbz_rs$zero_signed, winner_rs_zero_signed_rel = cbz_rs$zero_signed_rel,
    winner_rs_zero_discrepancy = cbz_rs$zero_discrepancy,
    winner_rs_zero_discrepancy_se = cbz_rs$zero_discrepancy_se,
    credible_better_zero_fit_rs = cbz_rs$better_tag, has_better_zero_fit_rs = cbz_rs$has_better,
    credible_better_zero_fit_rs_discrepancy = cbz_rs$better_discrepancy,
    winner_ri_zero_signed = cbz_ri$zero_signed, winner_ri_zero_signed_rel = cbz_ri$zero_signed_rel,
    winner_ri_zero_discrepancy = cbz_ri$zero_discrepancy,
    winner_ri_zero_discrepancy_se = cbz_ri$zero_discrepancy_se,
    credible_better_zero_fit_ri = cbz_ri$better_tag, has_better_zero_fit_ri = cbz_ri$has_better,
    credible_better_zero_fit_ri_discrepancy = cbz_ri$better_discrepancy,
    # R19: winner optimizer-warning diagnostic (reported, not acted on).
    winner_rs_has_warning = winner_rs_has_warning,
    clean_rs_competitor_within_10_aic = clean_rs_competitor,
    # I-2: manufactured-zero machine-readable flag.
    zeros_partly_manufactured = zeros_partly_manufactured,
    zeros_partly_manufactured_note = zeros_partly_manufactured_note
  )

  # Ruling R18: mark the winning records directly, via the SAME key-resolution
  # logic used to read them (`resolve_tier_key`) -- so "is this the winner" is
  # answerable from the record alone, without knowing the __ri2 convention.
  for (spec in list(list("M3", w_rs_m3, "rs", "is_winner_rs"),
                     list("M1", w_rs_m1, "rs", "is_winner_rs"))) {
    k <- resolve_tier_key(cell_name, spec[[1]], spec[[2]], spec[[3]])
    if (!is.na(k)) results[[k]][[spec[[4]]]] <- TRUE
  }
  if (needs_tier2) {
    for (spec in list(list("M3", w_ri_m3, "ri", "is_winner_ri"),
                       list("M1", w_ri_m1, "ri", "is_winner_ri"))) {
      k <- resolve_tier_key(cell_name, spec[[1]], spec[[2]], spec[[3]])
      if (!is.na(k)) results[[k]][[spec[[4]]]] <- TRUE
    }
  }

  # M2 under winner_rs only (spec: M2 fit under the single winning family).
  win <- Filter(function(r) r$tag == w_rs_m3, LADDER)[[1]]
  key <- sprintf("%s__M2__%s", cell_name, w_rs_m3)
  cat("fitting", key, "(winning family at M3, constraint-compliant tier)\n")
  res <- fit_with_fallback(cl$d, cl$dv, c(cl$base, M2_ADD),
                           win$family, win$zi, cl$offset_col)
  res$cell <- cell_name; res$model <- "M2"; res$family_tag <- w_rs_m3
  res$window <- cl$window; res$outcome <- cl$outcome; res$exposure <- cl$exposure
  res$re_tier <- if (identical(res$re_used, RS_RE)) "rs" else "ri"
  if (w_rs_m3 %in% c("zip", "zinb", "zinb_re")) {
    # No non-ZI M2 counterpart exists (M2 only fits the winning family), so
    # only the magnitude criteria of R17 apply here.
    res$zi_degenerate <- FALSE; res$zi_degenerate_reason <- ""
    res <- mark_zi_degenerate(res, NULL)
  } else {
    res$zi_degenerate <- FALSE; res$zi_degenerate_reason <- ""
  }
  # M2 is a downstream product of the M3 selection, not itself a ladder
  # competitor -- never a winner, never eligible for a future selection.
  res$is_winner_rs <- FALSE; res$is_winner_ri <- FALSE; res$eligible_for_selection <- FALSE
  res <- add_zero_fit_discrepancy(res)
  results[[key]] <- res
}

# ------------------------------------------------------------
# Ruling R19: optimizer-stability diagnostic. Post-R12 the headline nbinom1
# fits for viol_off_2021/viol_cov_2021 carry a live "NA/NaN function
# evaluation" warning at M1/M2/M3 while the clean nbinom2 runner-up (same df)
# does not, and the two disagree substantially on sigma2_u1 -- a quantity the
# manuscript's variance block reports. Refits the SAME nbinom1 spec under a
# second optimizer (BFGS via optim) and reports whether sigma2_u1 is stable.
# Diagnostic ONLY -- it does not change any winner (that would be tuning to
# taste); the interpretation of any instability is the PI's.
# ------------------------------------------------------------
ALT_OPTIM_CELLS <- c("viol_off_2021", "viol_cov_2021")
ALT_CONTROL <- glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS"))
for (cell_name in ALT_OPTIM_CELLS) {
  cl <- cells[[cell_name]]
  meta_key <- sprintf("%s__meta", cell_name)
  meta <- results[[meta_key]]
  if (is.null(meta) || !identical(meta$m2_family, "nbinom1")) next
  for (model in c("M1", "M2", "M3")) {
    rhs <- if (model == "M1") cl$base else if (model == "M2") c(cl$base, M2_ADD) else c(cl$base, M3_ADD)
    orig <- results[[sprintf("%s__%s__nbinom1", cell_name, model)]]
    if (is.null(orig)) next
    key <- sprintf("%s__%s__nbinom1__altopt", cell_name, model)
    cat("fitting", key, "(R19 optimizer-stability check, BFGS)\n")
    alt <- fit_spec(cl$d, cl$dv, rhs, nbinom1, zi = ~0, offset_col = cl$offset_col,
                    re = "(1 + time | state)", REML = FALSE, control = ALT_CONTROL)
    alt$cell <- cell_name; alt$model <- model; alt$family_tag <- "nbinom1"
    alt$window <- cl$window; alt$outcome <- cl$outcome; alt$exposure <- cl$exposure
    alt$re_used <- "(1 + time | state)"; alt$re_tier <- "rs"
    alt$optimizer <- "BFGS (via optim)"
    alt$zi_degenerate <- FALSE; alt$zi_degenerate_reason <- ""
    alt <- add_zero_fit_discrepancy(alt)
    alt$is_winner_rs <- FALSE; alt$is_winner_ri <- FALSE; alt$eligible_for_selection <- FALSE
    alt$sigma2_u1_default_optimizer <- orig$sigma2_u1
    alt$sigma2_u1_rel_diff_vs_default <- if (isTRUE(alt$converged) && is.finite(alt$sigma2_u1) &&
                                              is.finite(orig$sigma2_u1) && orig$sigma2_u1 != 0)
      abs(alt$sigma2_u1 - orig$sigma2_u1) / abs(orig$sigma2_u1) else NA_real_
    results[[key]] <- alt
  }
  m3_alt <- results[[sprintf("%s__M3__nbinom1__altopt", cell_name)]]
  if (!is.null(m3_alt)) {
    results[[meta_key]]$sigma2_u1_default_optimizer <- results[[sprintf("%s__M3__nbinom1", cell_name)]]$sigma2_u1
    results[[meta_key]]$sigma2_u1_alt_optimizer <- m3_alt$sigma2_u1
    results[[meta_key]]$sigma2_u1_rel_diff <- m3_alt$sigma2_u1_rel_diff_vs_default
    results[[meta_key]]$sigma2_u1_alt_optimizer_converged <- isTRUE(m3_alt$converged)
  }
}

write_json(results, out_json, auto_unbox = TRUE, digits = 10, na = "null")
cat("\nWrote", length(results), "entries to", out_json, "\n")
n_bad <- sum(vapply(results, function(r) !is.null(r$converged) && !isTRUE(r$converged), logical(1)))
if (n_bad > 0) cat("WARNING:", n_bad, "fit(s) did not converge -- see 'converged' flags\n")
