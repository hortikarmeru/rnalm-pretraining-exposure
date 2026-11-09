# Fit the mixed models for scripts/19_run_analysis.py, which writes the tables and a settings file.
#
# Usage: Rscript scripts/19_fit_models.R <settings.json>
#
# Per model (primary readout):  f1 ~ log_homologs + vienna_f1 + log_length + dataset + (1 + log_homologs | family)
# Joint models (one per model family, sizes stacked):
#   f1 ~ log_homologs * size + vienna_f1 + log_length + dataset + (1 + log_homologs | family) + (1 | seq_id)
# Predictors are centred on the fitted data (log_homologs stays in natural-log units); dataset uses
# sum-to-zero contrasts. REML, bobyqa. Random effects step down full -> uncorrelated -> random
# intercept when a fit warns (lme4's gradient and Hessian checks are on), is singular, or lmerTest
# cannot compute Satterthwaite degrees of freedom; the step reached is reported. Tests: Satterthwaite t,
# two-sided, 95% CI from the same t distribution. The permutation check shuffles each sequence's
# log_homologs within its family, refits at the step the observed fit reached (lme4 only, t = estimate
# / SE, derivative checks off for speed), and compares |t|: p = (k + 1) / (n + 1).
# Placebo and next-release fits replace log_homologs with that corpus's count when its table exists;
# the placebo is only fitted when |r| with log_homologs is below checks.placebo.max_correlation_with_exposure.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(jsonlite)
  library(parallel)
})

settings <- fromJSON(commandArgs(trailingOnly = TRUE)[1])
CTRL <- lmerControl(optimizer = "bobyqa")
CTRL_REFIT <- lmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
STEPS <- c("full", "uncorrelated", "random_intercept")
COVARIATES <- c("log_homologs", "vienna_f1", "log_length")

set_contrasts <- function(d) {
  # sum-to-zero contrasts: with dataset:log_homologs, the log_homologs term is then the average slope
  # over datasets rather than one reference dataset's slope (set again after rbind, which drops them)
  d$dataset <- factor(d$dataset)
  contrasts(d$dataset) <- contr.sum(nlevels(d$dataset))
  d
}

read_table <- function(key) {
  d <- read.delim(file.path(settings$table_dir, paste0(key, ".tsv")), stringsAsFactors = FALSE)
  d$model <- key
  d$log_length <- log(d$length)
  set_contrasts(d)
}

centre <- function(d, cols) {
  for (col in cols) d[[col]] <- d[[col]] - mean(d[[col]])
  d
}

random_part <- function(step, slopes, group = "family") {
  switch(step,
         full = sprintf("(1 + %s | %s)", slopes, group),
         uncorrelated = sprintf("(1 + %s || %s)", slopes, group),
         random_intercept = sprintf("(1 | %s)", group),
         step)   # any other step is given as the random-effects term itself
}

# One fit, with warnings and messages caught instead of printed. A fit that errors is returned with
# model = NULL. refit = TRUE: plain lme4 without derivative checks (permutations need only t).
fit_once <- function(formula, data, refit = FALSE) {
  warnings <- character(0)
  messages <- character(0)
  m <- tryCatch(withCallingHandlers(
    if (refit) lme4::lmer(as.formula(formula), data = data, REML = TRUE, control = CTRL_REFIT)
    else lmerTest::lmer(as.formula(formula), data = data, REML = TRUE, control = CTRL),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    },
    message = function(m) {
      messages <<- c(messages, trimws(conditionMessage(m)))
      invokeRestart("muffleMessage")
    }), error = function(e) {
      warnings <<- c(warnings, paste("error:", conditionMessage(e)))
      NULL
    })
  if (is.null(m)) return(list(model = NULL, problems = warnings, messages = messages, singular = NA))
  problems <- unique(c(warnings, m@optinfo$conv$lme4$messages))
  if (!refit && !inherits(m, "lmerModLmerTest")) problems <- c(problems, "no Satterthwaite df")
  list(model = m, problems = problems, messages = unique(messages), singular = isSingular(m))
}

# Step down the random effects until a fit is clean: no warning, not singular, df available.
fit_with_fallback <- function(fixed, data, steps = STEPS, slopes = "log_homologs", extra = "") {
  for (step in steps) {
    formula <- paste(fixed, "+", random_part(step, slopes), extra)
    f <- fit_once(formula, data)
    f$step <- step
    f$formula <- formula
    if (!is.null(f$model) && length(f$problems) == 0 && !f$singular) return(f)
  }
  f   # last step, its problems reported
}

term <- function(m, name) {
  if (is.null(m)) return(NULL)
  s <- summary(m)$coefficients
  if (!name %in% rownames(s) || !"df" %in% colnames(s)) return(NULL)
  est <- s[name, "Estimate"]; se <- s[name, "Std. Error"]; df <- s[name, "df"]
  half <- qt(0.975, df) * se
  list(estimate = est, se = se, df = df, t = s[name, "t value"], p = s[name, "Pr(>|t|)"],
       ci_low = est - half, ci_high = est + half)
}

refit_t <- function(m, name) {
  if (is.null(m) || !name %in% names(fixef(m))) return(NA_real_)
  unname(fixef(m)[name] / sqrt(diag(vcov(m)))[name])
}

variances <- function(m) {
  if (is.null(m)) return(NULL)
  v <- as.data.frame(VarCorr(m))
  lapply(split(v, seq_len(nrow(v))), function(r)
    list(group = r$grp, var1 = r$var1, var2 = r$var2, variance = r$vcov, sd_or_cor = r$sdcor))
}

summarise_fit <- function(f, data, terms) {
  coefs <- if (is.null(f$model)) character(0) else rownames(summary(f$model)$coefficients)
  wanted <- unique(c(terms, grep(":", coefs, value = TRUE)))
  seqs <- data[!duplicated(data$seq_id), ]
  list(formula = f$formula, step = f$step, problems = f$problems, messages = f$messages,
       singular = f$singular, n = nrow(data), n_sequences = nrow(seqs),
       n_families = length(unique(seqs$family)), singleton_families = sum(table(seqs$family) == 1),
       dataset_levels = levels(data$dataset),
       terms = Filter(Negate(is.null), setNames(lapply(wanted, function(t) term(f$model, t)), wanted)),
       random_effects = unname(variances(f$model)))
}

# Within-family shuffles of each sequence's log_homologs (in the joint models a sequence keeps one
# value across sizes), drawn in order from one seed so the result does not depend on the number of
# cores. Every model's check uses the same seed, so the same sequence of shuffles.
permutations <- function(data, n) {
  set.seed(settings$seed)
  seqs <- data[!duplicated(data$seq_id), c("seq_id", "family", "log_homologs")]
  idx <- split(seq_len(nrow(seqs)), seqs$family)
  lapply(seq_len(n), function(i) {
    perm <- seq_len(nrow(seqs))
    for (g in idx) if (length(g) > 1) perm[g] <- g[sample.int(length(g))]
    setNames(seqs$log_homologs[perm], seqs$seq_id)[data$seq_id]
  })
}

permutation_test <- function(f, data, names = "log_homologs") {
  n <- settings$n_permutations
  t_obs <- sapply(names, function(x) term(f$model, x)$t)
  refits <- mclapply(permutations(data, n), function(x) {
    d <- data
    d$log_homologs <- unname(x)
    r <- fit_once(f$formula, d, refit = TRUE)
    list(t = sapply(names, function(x) refit_t(r$model, x)),
         failed = is.null(r$model), warned = length(r$problems) > 0, singular = isTRUE(r$singular))
  }, mc.cores = max(1, detectCores() - 1))
  refits <- lapply(refits, function(r) if (inherits(r, "try-error")) list(t = setNames(rep(NA, length(names)), names),
                                                                           failed = TRUE, warned = FALSE, singular = FALSE) else r)
  t_perm <- do.call(rbind, lapply(refits, function(r) r$t))
  count <- function(field) sum(vapply(refits, function(r) isTRUE(r[[field]]), logical(1)))
  out <- list(n = n, refits_failed = count("failed"), refits_with_warnings = count("warned"),
              refits_singular = count("singular"))
  for (x in names) {
    ok <- !is.na(t_perm[, x])   # failed refits are left out of k and n
    k <- sum(abs(t_perm[ok, x]) >= abs(t_obs[[x]]))
    out[[x]] <- list(t_observed = t_obs[[x]], k = k, n_used = sum(ok), p = (k + 1) / (sum(ok) + 1))
  }
  out
}

f1_summary <- function(d, readouts) {
  by_set <- function(col) lapply(split(d[[col]], d$dataset), function(v) list(n = length(v), median = median(v), mean = mean(v)))
  c(setNames(lapply(paste0("f1_", readouts), by_set), readouts), list(vienna_f1 = by_set("vienna_f1")))
}

FIXED <- "%s ~ log_homologs + vienna_f1 + log_length%s"

per_model <- function(key) {
  raw <- read_table(key)
  out <- list(f1_by_dataset = f1_summary(raw, settings$readouts))
  fit <- function(name, d, f1 = "f1_primary", dataset = " + dataset", extra_fixed = "") {
    d <- centre(d, COVARIATES)
    f <- fit_with_fallback(paste0(sprintf(FIXED, f1, dataset), extra_fixed), d)
    out[[name]] <<- summarise_fit(f, d, "log_homologs")
    f
  }
  main <- fit("main", raw)
  fit("no_dataset_term", raw, dataset = "")
  fit("dataset_slopes", raw, extra_fixed = " + dataset:log_homologs")
  for (cut in settings$min_length) fit(sprintf("min_length_%d", cut), raw[raw$length >= cut, ])
  for (r in setdiff(settings$readouts, "primary")) fit(paste0("readout_", r), raw, f1 = paste0("f1_", r))
  for (alt in c("next_release", "placebo")) {   # another corpus's count in place of log_homologs
    col <- paste0("log_", alt)
    if (!col %in% names(raw) || all(is.na(raw[[col]]))) {
      out[[alt]] <- list(status = "no exposure table")
      next
    }
    r <- cor(raw[[col]], raw$log_homologs)
    if (alt == "placebo" && abs(r) >= settings$max_placebo_correlation) {
      out[[alt]] <- list(status = "not fitted: correlated with log_homologs", correlation = r)
      next
    }
    d <- raw
    d$log_homologs <- d[[col]]
    fit(alt, d)
    out[[alt]]$correlation_with_log_homologs <- r
  }
  out$main$permutation <- permutation_test(main, centre(raw, COVARIATES))
  out
}

joint <- function(keys) {
  d <- set_contrasts(do.call(rbind, lapply(keys, read_table)))
  d$size <- unlist(settings$log10_parameters[d$model])
  d <- centre(d, c(COVARIATES, "size"))
  fixed <- "f1_primary ~ log_homologs * size + vienna_f1 + log_length + dataset"
  terms <- c("log_homologs", "size", "log_homologs:size")
  out <- list(sizes = keys)
  main <- fit_with_fallback(fixed, d, extra = "+ (1 | seq_id)")
  out$main <- summarise_fit(main, d, terms)
  # by-family slopes for size and the interaction; its last step is the main model's full structure,
  # so the fallback never drops the exposure slope
  slopes <- fit_with_fallback(fixed, d, steps = c("full", "uncorrelated", "(1 + log_homologs | family)"),
                              slopes = "log_homologs * size", extra = "+ (1 | seq_id)")
  out$by_family_size_slopes <- summarise_fit(slopes, d, terms)
  if (length(keys) > 2) {   # size as categories instead of log10 parameters
    d$size_category <- factor(d$model, levels = keys)
    cat_fit <- fit_with_fallback("f1_primary ~ log_homologs * size_category + vienna_f1 + log_length + dataset",
                                 d, extra = "+ (1 | seq_id)")
    coefs <- if (is.null(cat_fit$model)) character(0) else rownames(summary(cat_fit$model)$coefficients)
    out$size_categories <- summarise_fit(cat_fit, d, c("log_homologs", grep("size_category", coefs, value = TRUE)))
  }
  out$main$permutation <- permutation_test(main, d, c("log_homologs", "log_homologs:size"))
  out
}

results <- list(models = list(), joint_models = list())
for (key in settings$models) {
  message("fitting ", key)
  results$models[[key]] <- per_model(key)
}
for (name in names(settings$joint_models)) {
  keys <- settings$joint_models[[name]]
  if (length(keys) < 2) next
  message("fitting joint ", name)
  results$joint_models[[name]] <- joint(keys)
}
write(toJSON(results, auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null", na = "null"), settings$out)
