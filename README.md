# mossyforest

<!-- badges: start -->
[![tests](https://github.com/timothyzshen/mossyforest-py/actions/workflows/tests.yaml/badge.svg)](https://github.com/timothyzshen/mossyforest-py/actions/workflows/tests.yaml)
[![docs](https://github.com/timothyzshen/mossyforest-py/actions/workflows/docs.yaml/badge.svg)](https://timothyzshen.github.io/mossyforest-py/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
<!-- badges: end -->

Network aware stable feature selection via Shapley values and random forests.

`mossyforest` is the Python implementation of the
[<img src="https://www.r-project.org/logo/Rlogo.png" alt="R" height="16"> mossyforest](https://timothyzshen.github.io/mossyforest/) R package. It is a
bit-exact replica of the R algorithm (verified — see
[Parity](#parity-with-the-r-package)). It is recommended to use this package for large
high dimensional data sets.

It selects a small, stable set of features from high-dimensional, correlated data
by combining:

1. **Module stratified screening** — recursive feature elimination (RFE) within
   each correlated module (or WGCNA-derived module).
2. **Shadow stability selection** — Boruta style subsample selection where each
   feature must beat its own permuted "shadow" across many resamples.
3. **A single TreeSHAP pass** on the stable set for importance, confidence
   intervals, and a pairwise interaction matrix.

---

## Installation

```bash
pip install mossyforest
```

Optional extras:

```bash
# PyWGCNA, only needed for wmf() and detect_modules()
pip install "mossyforest[wgcna]"

# numba and fasttreeshap (fasttreeshap needs Python 3.11 or lower)
pip install "mossyforest[fast]"
```

From a clone, for development:

```bash
pip install -e .
pip install -e ".[wgcna]"   # PyWGCNA
pip install -e ".[fast]"    # numba and fasttreeshap
pip install -e ".[dev]"     # test and lint tooling
```

Requires Python ≥3.9, numpy, pandas, scikit-learn, shap. Module detection
(`wmf`) additionally needs [PyWGCNA](https://github.com/mortazavilab/PyWGCNA),
which is an optional extra: `pip install "mossyforest[wgcna]"`.

---

## Quick start

```python
import numpy as np, pandas as pd
from mossyforest import mf, screen_control, select_control

# X: samples x features (DataFrame keeps feature names); y: int/bool = classification
res = mf(
    X, y,
    module_membership = {"gene1": "M1", "gene2": "M1", "gene3": "M2", ...},
    screen_params = screen_control(),      # Phase 1 defaults
    select_params = select_control(),      # Phase 2 defaults
    num_processors = 4,                    # parallel replicates/modules
    seed = 42,
)

print(res)                                 # stable features + SHAP + stability
res.final_SHAP                             # DataFrame: importance, 95% CIs, freq
res.predict(X_new)                         # predict from the final RF
res.detect_interaction(thresh=0.01)        # pairwise TreeSHAP interactions
```

### With automatic WGCNA module detection

```python
from mossyforest import wmf, WGCNA_control
res = wmf(X, y, WGCNA_params=WGCNA_control(power=6, min_features=20))
res.WGCNA_object                           # raw module labels
```

### sklearn-compatible estimator

```python
from mossyforest import MossyForest
model = MossyForest(n_jobs=4, random_state=42)
model.fit(X, y, module_membership=mm)
model.transform(X)          # X reduced to stable features
model.predict(X_new)
```

---

## Parameters

### `screen_control` — Phase 1 (screening)
| param | default | meaning |
|---|---|---|
| `drop_fraction` | 0.25 | fraction of features dropped per RFE step |
| `keep_fraction` | 0.50 | fraction of each module retained after screening |
| `mtry_factor` | 1.0 | `mtry` factor (√p classification, p/3 regression) |
| `min_ntree` | 500 | minimum trees per forest |
| `ntree_factor` | 1 | trees = `max(p·factor, min_ntree)` |

### `select_control` — Phase 2 (shadow stability)
| param | default | meaning |
|---|---|---|
| `shadow_mode` | `"split"` | `"split"` (corr + unassigned pools) or `"within_module"` |
| `n_reps` | 50 | subsample replicates (half the rows, drawn without replacement) |
| `pi_thr` | 0.60 | stability threshold for the correlated pool |
| `pi_thr_unassigned` | `None` | unassigned-pool threshold (`None` = elbow detection) |
| `shadow_percentile` | 95 | percentile of the shadow null used as the cutoff |
| `unassigned_modules` | `None` | module names treated as the unassigned pool |
| `use_cfres` | `False` | cross-fitted cfRes residualization of `y` |
| `cfres_scope` | `"unassigned"` | `"unassigned"` or `"all_modules"` |
| `early_stop` | `False` | stop replicates early once frequencies converge |

### `mf()` execution / performance
| param | default | meaning |
|---|---|---|
| `num_processors` | 1 | parallel workers for replicates + modules |
| `blas_threads` | `None` | inner BLAS/OpenMP cap (auto: 1 when parallel) |
| `shap_backend` | `"shap"` | or `"fasttreeshap"` (Python ≤3.11 only) |
| `use_numba` | `False` | JIT the mean\|SHAP\| reduction |
| `screen_cache` | `None` | `True` (memory) / path (disk) — cache Phase 1 for sweeps |
| `compute_interactions` | `True` | retain RF so interactions compute lazily on access |

---

## Result object (`MossyForestResult`)

| attribute / method | description |
|---|---|
| `final_SHAP` | stable features: importance, mean/abs SHAP, 95% CIs, stability freq |
| `shap_obj["shapley_values"]` | signed n×p SHAP matrix |
| `interaction_matrix` | p×p mean \|TreeSHAP interaction\| (computed lazily) |
| `stability_freq` | per-feature shadow-subsample selection frequency |
| `stability_data` | per-pool stability curves (for `plot_stability_elbow`) |
| `survivor_list` | Phase-1 survivors per module |
| `runtimes` | seconds for Screen / Selection / Final_RF |
| `predict()` / `predict_proba()` | predict from the final RF |
| `detect_interaction()` | ranked pairwise interaction table |

---

## Plots

```python
from mossyforest import (
    plot_importance, plot_waterfall, plot_force, plot_stability,
    plot_stability_elbow, plot_potential_interactions, plot_decisions,
    plot_modules, plot_elbow, plot_soft_threshold,
)
plot_importance(res, kind="beeswarm")     # or "bar" / "both"
plot_stability_elbow(res)                 # per-pool selection curves
plot_potential_interactions(res)          # TreeSHAP interaction heatmap
```

Benchmarks and a parity harness live in `benchmarks/`:

```bash
python benchmarks/bench_common.py            # config sweep + R comparison + cache
python benchmarks/bench_common.py --out results.csv   # persist results
python benchmarks/parity_in_process.py       # bit-exact parity vs the R backend
```

---

## Parity with the R package

`mossyforest` (Python) is a bit-exact replica of the R package. Verified in
one process on identical data/seed/stack, the two agree exactly:

- screening survivors identical, selected set identical
- stability frequencies match to 0.0000
- signed SHAP matrix bit-identical (max \|Δ\| = 0.000)

Two things were required for exactness: matching R's **survivor-column ordering**
(original-X order), and a shared **per-replicate RNG scheme**. float32 (Python)
vs float64 (R) is *not* observable — sklearn trees are float32 internally either
way. Re-verify anytime with `python benchmarks/parity_in_process.py`.

---

## License

GPL-3.0-or-later.
