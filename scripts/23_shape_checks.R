# Shape of the exposure effect, called by scripts/23_shape_checks.py with a settings file.
#
# Per model, on the tables script 19 writes (primary readout):
#   gamm  f1 ~ s(log_homologs) + vienna_f1 + log_length + dataset + family random intercept and slope (mgcv::bam)
#   bins  f1 ~ exposure bin + vienna_f1 + log_length + dataset + (1 | family), reference bin = exactly 1 homolog
#   beta  beta mixed model (glmmTMB) with the main model's terms, F1 squeezed into (0, 1); slope also given as an
#         average marginal effect in F1 per 10x homologs
suppressPackageStartupMessages({library(mgcv); library(lme4); library(lmerTest); library(glmmTMB); library(jsonlite)})

settings <- fromJSON(commandArgs(trailingOnly = TRUE)[1])
grid_counts <- settings$curve_counts

fit_model <- function(key) {
  d <- read.delim(file.path(settings$table_dir, paste0(key, ".tsv")), stringsAsFactors = FALSE)
  d$family <- factor(d$family); d$dataset <- factor(d$dataset)
  d$v <- d$vienna_f1 - mean(d$vienna_f1); d$ll <- log(d$length) - mean(log(d$length))
  d$xc <- d$log_homologs - mean(d$log_homologs)   # centred for the family slope, as in script 19
  out <- list()

  smooth <- bam(as.formula(sprintf("f1_primary ~ s(log_homologs, k = %d) + v + ll + dataset + s(family, bs = 're') + s(family, xc, bs = 're')", settings$gamm_k)),
                data = d, method = "fREML", discrete = TRUE)
  linear <- bam(f1_primary ~ log_homologs + v + ll + dataset + s(family, bs = "re") + s(family, xc, bs = "re"),
                data = d, method = "fREML", discrete = TRUE)
  grid <- data.frame(log_homologs = log1p(grid_counts[log1p(grid_counts) <= max(d$log_homologs)]))
  nd <- cbind(grid, xc = 0, v = 0, ll = 0, dataset = levels(d$dataset)[1], family = levels(d$family)[1])
  # gain = f(x) - f(0), with its own SE: difference rows of the smooth's columns of the linear predictor matrix
  X <- predict(smooth, nd, type = "lpmatrix")
  cols <- grep("^s\\(log_homologs\\)", colnames(X))
  D <- sweep(X[, cols, drop = FALSE], 2, X[1, cols])
  V <- vcov(smooth)[cols, cols]
  gain <- as.numeric(D %*% coef(smooth)[cols]); se <- sqrt(pmax(rowSums((D %*% V) * D), 0))
  st <- summary(smooth)$s.table
  out$gamm <- list(edf = st["s(log_homologs)", "edf"], p_smooth = st["s(log_homologs)", "p-value"],
                   aic_smooth = AIC(smooth), aic_linear = AIC(linear),
                   curve = data.frame(homologs = round(expm1(grid$log_homologs)), gain = gain, se = se))

  n <- round(expm1(d$log_homologs)); edges <- settings$bins
  labels <- c(sapply(seq_along(edges)[-length(edges)], function(i)
    if (edges[i + 1] - edges[i] == 1) as.character(edges[i]) else sprintf("%d-%d", edges[i], edges[i + 1] - 1)),
    sprintf("%d+", edges[length(edges)]))
  d$bin <- relevel(cut(n, c(edges - 0.5, Inf), labels = labels), ref = "1")
  # family random intercept only: bin-specific family slopes don't identify; SEs a little optimistic vs script 19
  b <- lmer(f1_primary ~ bin + v + ll + dataset + (1 | family), data = d, REML = TRUE, control = lmerControl(optimizer = "bobyqa"))
  cf <- summary(b)$coefficients; rows <- grep("^bin", rownames(cf))
  out$bins <- list(n = as.list(table(d$bin)), estimates = data.frame(bin = sub("^bin", "", rownames(cf)[rows]),
                   estimate = cf[rows, "Estimate"], se = cf[rows, "Std. Error"], p = cf[rows, "Pr(>|t|)"]))

  N <- nrow(d); d$y <- if (isTRUE(settings$beta_squeeze)) (d$f1_primary * (N - 1) + 0.5) / N else d$f1_primary
  bt <- tryCatch(glmmTMB(y ~ log_homologs + v + ll + dataset + (1 + log_homologs | family), data = d, family = beta_family()),
                 error = function(e) NULL)
  if (!is.null(bt)) {
    co <- summary(bt)$coefficients$cond["log_homologs", ]; mu <- predict(bt, type = "response")
    out$beta <- list(logit_slope = co[["Estimate"]], se = co[["Std. Error"]], p = co[["Pr(>|z|)"]],
                     f1_per_10x = mean(mu * (1 - mu)) * co[["Estimate"]] * log(10),   # approximate: fixed slope only
                     converged = bt$fit$convergence == 0 && isTRUE(bt$sdr$pdHess))
  } else out$beta <- list(status = "did not fit")
  out
}

res <- list()
for (key in settings$models) { message("shape checks ", key); res[[key]] <- fit_model(key) }
write(toJSON(res, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"), settings$out)
