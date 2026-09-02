#!/usr/bin/env Rscript
# R side of the R<->Python parity harness.
#
# Reads the SAME data CSV the Python side used, runs the R package's
# mf(..., backend="python") through reticulate (pinned to the same Python
# interpreter as the standalone package), and writes the selected features +
# stability frequencies to JSON for comparison.
#
# Usage: Rscript parity_check.R <data_csv> <out_json> <seed>

args     <- commandArgs(trailingOnly = TRUE)
data_csv <- args[[1]]
out_json <- args[[2]]
seed     <- as.integer(args[[3]])

suppressMessages({
  library(reticulate)
  library(jsonlite)
  devtools::load_all("/Users/tzshen/Documents/Github/sf_dev", quiet = TRUE)
})

df   <- read.csv(data_csv, check.names = FALSE)
y    <- as.factor(df[["__y__"]])
X    <- df[, setdiff(colnames(df), "__y__"), drop = FALSE]
mods <- read.csv(sub("\\.csv$", "_mods.csv", data_csv), check.names = FALSE)
mm   <- setNames(as.character(mods$module), as.character(mods$feature))

sc <- screen_control(drop_fraction = 0.3, keep_fraction = 0.5, min_ntree = 100L)
sl <- select_control(n_boots = 30L, min_ntree = 100L, pi_thr = 0.4,
                     shadow_mode = "split", unassigned_modules = "unassigned")

res <- mf(X, y, module_membership = mm, screen_params = sc, select_params = sl,
          final_ntree = 100L, num_processors = 1L, verbose = 0L, seed = seed,
          backend = "python")

out <- list(
  selected = as.character(res$final_SHAP$feature_name),
  freq     = setNames(as.list(as.numeric(res$stability_freq)),
                      names(res$stability_freq))
)
write_json(out, out_json, auto_unbox = TRUE, pretty = TRUE)
cat("R done:", length(out$selected), "selected\n")
