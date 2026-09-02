#!/usr/bin/env Rscript
# R reticulate-backend benchmark for MossyForest.
#
# Times the R package's mf(..., backend="python") path, which calls
# inst/python/mf_python_backend.py through reticulate. Loaded via
# devtools::load_all so the in-tree (updated) backend is used.
#
# Usage:  Rscript bench_r_reticulate.R <sf_dev_path> <num_processors> <seed>
# Prints one parseable line:  RESULT <seconds> <n_stable>

args <- commandArgs(trailingOnly = TRUE)
sf_dev_path    <- if (length(args) >= 1) args[[1]] else "/Users/tzshen/Documents/Github/sf_dev"
num_processors <- if (length(args) >= 2) as.integer(args[[2]]) else 1L
seed           <- if (length(args) >= 3) as.integer(args[[3]]) else 42L

suppressMessages(devtools::load_all(sf_dev_path, quiet = TRUE))

set.seed(0)
n   <- 300
lat <- lapply(1:4, function(i) rnorm(n))
cols <- list(); feats <- c(); labels <- c()
for (m in 1:4) for (j in 1:12) {
  cols[[length(cols) + 1]] <- lat[[m]] + rnorm(n, sd = 0.3)
  feats  <- c(feats,  sprintf("M%d_%d", m - 1, j - 1))
  labels <- c(labels, sprintf("M%d", m - 1))
}
for (j in 1:16) {
  cols[[length(cols) + 1]] <- rnorm(n)
  feats  <- c(feats,  sprintf("ind%d", j - 1))
  labels <- c(labels, "unassigned")
}
X  <- as.data.frame(setNames(cols, feats))
y  <- as.factor(as.integer((lat[[1]] + lat[[2]] - lat[[3]]) > 0))
mm <- setNames(labels, feats)

sc <- screen_control(drop_fraction = 0.3, keep_fraction = 0.5, min_ntree = 150L)
sl <- select_control(n_boots = 30L, min_ntree = 150L, pi_thr = 0.4,
                     shadow_mode = "split", unassigned_modules = "unassigned")

t0  <- Sys.time()
res <- mf(X, y, module_membership = mm, screen_params = sc, select_params = sl,
          final_ntree = 150L, num_processors = num_processors,
          verbose = 0L, seed = seed, backend = "python")
dt  <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

cat(sprintf("RESULT %.3f %d\n", dt, nrow(res$final_SHAP)))
