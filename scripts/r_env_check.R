# Environment gate for the ZINB count-model pipeline.
# Proves glmmTMB is installed and can fit a ZINB with a random slope.
# Run: Rscript scripts/r_env_check.R    (exits 1 on any failure)

ok <- TRUE
fail <- function(msg) { cat("FAIL:", msg, "\n"); ok <<- FALSE }

for (p in c("glmmTMB", "TMB", "jsonlite")) {
  if (!requireNamespace(p, quietly = TRUE)) fail(sprintf("package '%s' not installed", p))
}
if (!ok) { cat("\nInstall with:\n  Rscript -e 'install.packages(c(\"glmmTMB\",\"jsonlite\"), repos=\"https://cloud.r-project.org\")'\n"); quit(status = 1) }

cat("glmmTMB", as.character(packageVersion("glmmTMB")), "/ TMB",
    as.character(packageVersion("TMB")), "/ jsonlite",
    as.character(packageVersion("jsonlite")), "\n")

# Synthetic panel: 40 groups x 10 periods, overdispersed counts with ~20% structural zeros.
set.seed(20260820)
n_g <- 40; n_t <- 10
d <- expand.grid(t = seq_len(n_t), g = factor(seq_len(n_g)))
d$time <- d$t - 5
u0 <- rnorm(n_g, 0, 0.8)[as.integer(d$g)]
u1 <- rnorm(n_g, 0, 0.1)[as.integer(d$g)]
mu <- exp(2 + 0.05 * d$time + u0 + u1 * d$time)
d$y <- rnbinom(nrow(d), mu = mu, size = 1.5) * rbinom(nrow(d), 1, 0.8)

m <- try(glmmTMB::glmmTMB(y ~ time + (1 + time | g), ziformula = ~1,
                          family = glmmTMB::nbinom2, data = d), silent = TRUE)
if (inherits(m, "try-error")) fail(paste("ZINB fit errored:", attr(m, "condition")$message))
if (ok && !isTRUE(m$sdr$pdHess)) fail("ZINB fit did not produce a positive-definite Hessian")
if (ok && m$fit$convergence != 0) fail(sprintf("ZINB convergence code %d", m$fit$convergence))
if (ok) {
  zi_prob <- plogis(glmmTMB::fixef(m)$zi[["(Intercept)"]])
  cat(sprintf("ZINB converged; estimated zero-inflation prob = %.3f (data generated at 0.200)\n", zi_prob))
  if (abs(zi_prob - 0.20) > 0.15) fail(sprintf("zero-inflation estimate %.3f implausibly far from 0.200", zi_prob))
}

if (ok) cat("\nPASS: R environment ready\n") else quit(status = 1)
