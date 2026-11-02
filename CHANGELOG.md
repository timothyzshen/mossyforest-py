# Changelog

## 0.1.0

First release.

* Module stratified screening, shadow stability selection and a final TreeSHAP
  pass, matching the R package `mossyforest` (bit exact on the same data, seed
  and software stack).
* The default `mtry` rule is the standard one: `sqrt(p)` for classification and
  `p/3` for regression.
* A `UserWarning` is raised when an entry of `unassigned_modules` does not match
  any module in `module_membership`.
* Selection results no longer depend on Python's string hash order, so repeated
  runs give the same stable set.
* `PyWGCNA` is an optional extra (`pip install "mossyforest[wgcna]"`). It is only
  needed for `wmf()` and `detect_modules()`.
* Supports Python 3.9 to 3.13.
