# Fit the mixed models for scripts/19_run_analysis.py, which writes the tables and a settings file.
#
# Usage: Rscript scripts/19_fit_models.R <settings.json>
#
# Per model (primary readout):  f1 ~ log_homologs + vienna_f1 + log_length + dataset + (1 + log_homologs | family)
# Joint models (one per model family, sizes stacked):
#   f1 ~ log_homologs * size + vienna_f1 + log_length + dataset + (1 + log_homologs | family) + (1 | seq_id)
# Predictors are centred on the fitted data (log_homologs stays in natural-log units). REML, bobyqa.
# Random effects step down full -> uncorrelated -> random intercept when a fit gives a convergence
# warning or is singular; the step reached is reported. Tests: Satterthwaite t (two-sided), 95% CI
# from the same t distribution. The permutation check shuffles log_homologs within family, refits the
# model at the step the observed fit reached, and compares |t| (p = (k + 1) / (n + 1)).

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(jsonlite)
  library(parallel)
})

settings <- fromJSON(commandArgs(trailingOnly = TRUE)[1])
CTRL <- lmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
STEPS <- c("full", "uncorrelated", "random_intercept")

read_table <- function(key) {
  d <- read.delim(file.path(settings$table_dir, paste0(key, ".tsv")), stringsAsFactors = FALSE)
  d$model <- key
  # sum-to-zero contrasts: with dataset:log_homologs, the log_homologs term is then the average slope
  # over datasets rather than one reference dataset's slope
  d$dataset <- factor(d$dataset)
  contrasts(d$dataset) <- contr.sum(nlevels(d$dataset))
  d$log_length <- log(d$length)
  d
}

centre <- function(d, cols) {
  for (col in cols) d[[col]] <- d[[col]] - mean(d[[col]])
  d
}

random_part <- function(step, slopes, group = "family") {
  switch(step,
         full = sprintf("(1 + %s | %s)", slopes, group),
         uncorrelated = sprintf("(1 + %s || %s)", slopes, group),
         random_intercept = sprintf("(1 | %s)", group))
}

# One lmer fit, with its warnings caught instead of printed.
fit_once <- function(formula, data) {
  warnings <- character(0)
  m <- withCallingHandlers(
    lmer(as.formula(formula), data = data, REML = TRUE, control = CTRL),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  problems <- c(warnings, m@optinfo$conv$lme4$messages)
  list(model = m, problems = unique(problems), singular = isSingular(m))
}

# Step down the random effects until a fit converges cleanly and is not singular.
fit_with_fallback <- function(fixed, data, slopes = "log_homologs", extra = "") {
  for (step in STEPS) {
    formula <- paste(fixed, "+", random_part(step, slopes), extra)
    f <- fit_once(formula, data)
    f$step <- step
    f$formula <- formula
    if (length(f$problems) == 0 && !f$singular) return(f)
  }
  f   # last step, problems reported
}

term <- function(m, name) {
  s <- summary(m)$coefficients
  if (!name %in% rownames(s)) return(NULL)
  est <- s[name, "Estimate"]; se <- s[name, "Std. Error"]; df <- s[name, "df"]
  half <- qt(0.975, df) * se
  list(estimate = est, se = se, df = df, t = s[name, "t value"], p = s[name, "Pr(>|t|)"],
       ci_low = est - half, ci_high = est + half)
}

summarise_fit <- function(f, data, terms) {
  coefs <- rownames(summary(f$model)$coefficients)
  wanted <- unique(c(terms, grep(":", coefs, value = TRUE)))
  list(formula = f$formula, step = f$step, problems = f$problems, singular = f$singular,
       n = nrow(data), n_families = length(unique(data$family)),
       singleton_families = sum(table(data$family) == 1),
       terms = Filter(Negate(is.null), setNames(lapply(wanted, function(t) term(f$model, t)), wanted)))
}

# Within-family shuffles of each sequence's log_homologs (in the joint models a sequence keeps one
# value across sizes), drawn in order from one seed so the result does not depend on the number of cores.
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
    r <- fit_once(f$formula, d)
    list(t = sapply(names, function(x) term(r$model, x)$t), problem = length(r$problems) > 0 || r$singular)
  }, mc.cores = max(1, detectCores() - 1))
  t_perm <- do.call(rbind, lapply(refits, function(r) r$t))
  out <- list(n = n, refits_with_problems = sum(vapply(refits, function(r) r$problem, logical(1))))
  for (x in names) {
    k <- sum(abs(t_perm[, x]) >= abs(t_obs[[x]]))
    out[[x]] <- list(t_observed = t_obs[[x]], k = k, p = (k + 1) / (n + 1))
  }
  out
}

FIXED <- "%s ~ log_homologs + vienna_f1 + log_length%s"
COVARIATES <- c("log_homologs", "vienna_f1", "log_length")

per_model <- function(key) {
  raw <- read_table(key)
  out <- list()
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
  d <- centre(raw, COVARIATES)
  out$main$permutation <- permutation_test(main, d)
  out
}

joint <- function(keys) {
  d <- do.call(rbind, lapply(keys, read_table))
  d$size <- unlist(settings$log10_parameters[d$model])
  d <- centre(d, c(COVARIATES, "size"))
  fixed <- "f1_primary ~ log_homologs * size + vienna_f1 + log_length + dataset"
  out <- list(sizes = keys)
  main <- fit_with_fallback(fixed, d, extra = "+ (1 | seq_id)")
  out$main <- summarise_fit(main, d, c("log_homologs", "size", "log_homologs:size"))
  slopes <- fit_with_fallback(fixed, d, slopes = "log_homologs * size", extra = "+ (1 | seq_id)")
  out$by_family_size_slopes <- summarise_fit(slopes, d, c("log_homologs", "size", "log_homologs:size"))
  if (length(keys) > 2) {   # size as categories instead of log10 parameters
    d$size_category <- factor(d$model, levels = keys)
    cat_fit <- fit_with_fallback("f1_primary ~ log_homologs * size_category + vienna_f1 + log_length + dataset",
                                 d, extra = "+ (1 | seq_id)")
    out$size_categories <- summarise_fit(cat_fit, d, "log_homologs")
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
write(toJSON(results, auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null"), settings$out)
