"""
Phase 1: Per-module Screening RFE.

Runs rfe_loop on each module independently on the FULL dataset (no bootstrap).
Retains keep_fraction of each module's features as candidates for Phase 2.

Matches run_screen_rfe() from sf_python_backend.py v18.
"""

from __future__ import annotations

import math
import numpy as np
from joblib import Parallel, delayed

from .rfe import rfe_loop, blockwise_reduce, _map_labels
from .threads import thread_limit_ctx, resolve_blas_threads
from .parameters import screen_control


def run_screen_rfe(
    X_mat: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    module_assignments: list,
    module_list: list,
    screen_params: screen_control,
    nodesize: int,
    classification: bool,
    seed: int,
    n_jobs: int,
    max_modsize: int = 5000,
    backend: str = "shap",
    use_numba: bool = False,
    blas_threads=None,
    verbose: bool = False,
) -> dict:
    """
    Phase 1: per-module screening RFE on the full dataset.

    Parameters
    ----------
    X_mat              : (n, p) full feature matrix
    y                  : (n,) outcome
    feature_names      : list of p column names
    module_assignments : list of p module labels (aligned with feature_names)
    module_list        : ordered list of unique module names to process
    screen_params      : screen_control instance
    nodesize           : min terminal node size
    classification     : bool
    seed               : RNG seed
    n_jobs             : parallel jobs for RF
    max_modsize        : blockwise reduction trigger (default 5000)
    verbose            : print progress

    Returns
    -------
    dict with keys:
        survivor_features  : list of lists — per-module surviving feature names
        survivor_vims      : list of lists — per-module VIM scores
        initial_feat       : list — feature names from first RFE step
        initial_vim        : list — VIMs from first RFE step
        initial_survivor   : list of bool — survivor flag at first step
        initial_module     : list — module label per row of initial_* lists
        module_list        : echoed module_list
    """
    X_mat         = np.ascontiguousarray(X_mat, dtype=np.float32)
    y             = np.asarray(y).ravel()
    feature_names = [str(f) for f in feature_names]
    module_assignments = [str(m) for m in module_assignments]
    module_list   = [str(m) for m in module_list]

    sc = screen_params

    if classification:
        y = _map_labels(y)
    else:
        y = y.astype(np.float64)

    feat_to_col = {f: i for i, f in enumerate(feature_names)}
    mod_to_cols: dict[str, list] = {}
    for feat, mod in zip(feature_names, module_assignments):
        mod_to_cols.setdefault(mod, []).append(feat_to_col[feat])

    # inner-thread cap when modules run in parallel
    inner_limit = resolve_blas_threads(blas_threads, n_jobs)

    def _screen_one_module(mod):
        """Screen a single module. Deterministic given `seed` → order-independent."""
        cols = mod_to_cols.get(mod, [])
        if not cols:
            return mod, [], [], None

        X_mod    = X_mat[:, cols]
        feat_mod = [feature_names[c] for c in cols]
        n_mod    = len(cols)
        target   = max(1, math.ceil(n_mod * sc.keep_fraction))

        if n_mod > max_modsize:
            if verbose:
                print(f"  Module {mod}: blockwise pre-reduction "
                      f"({n_mod} > {max_modsize} features)")
            X_mod, feat_mod = blockwise_reduce(
                X_mod, feat_mod, y, max_modsize, 0.9,
                sc.drop_fraction, sc.mtry_factor, sc.ntree_factor,
                sc.min_ntree, nodesize, classification, seed, 1,
                use_shap=True, backend=backend, use_numba=use_numba,
            )
            n_mod  = X_mod.shape[1]
            target = max(1, math.ceil(n_mod * sc.keep_fraction))

        if verbose:
            print(f"  Module {mod}: {n_mod} features → target {target}")

        survivors, vims, init_info, _ = rfe_loop(
            X_mod, feat_mod, y, target,
            sc.drop_fraction, sc.mtry_factor, sc.ntree_factor,
            sc.min_ntree, nodesize, classification, seed, 1,
            use_shap=True, backend=backend, use_numba=use_numba,
        )
        return mod, survivors, vims, init_info

    # modules are independent and deterministic → parallelize the outer loop
    if n_jobs == 1:
        results = [_screen_one_module(mod) for mod in module_list]
    else:
        with thread_limit_ctx(inner_limit):
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_screen_one_module)(mod) for mod in module_list
            )

    # reassemble in module_list order (results already ordered by Parallel)
    all_survivor_features: list = []
    all_survivor_vims: list     = []
    init_feats: list    = []
    init_vims: list     = []
    init_survivors: list = []
    init_modules: list  = []

    for mod, survivors, vims, init_info in results:
        all_survivor_features.append(survivors)
        all_survivor_vims.append(vims)
        if init_info is not None:
            n_rows = len(init_info["feature"])
            init_feats.extend(init_info["feature"])
            init_vims.extend(init_info["vim"])
            init_survivors.extend(init_info["survivor"])
            init_modules.extend([mod] * n_rows)

    return {
        "survivor_features": all_survivor_features,
        "survivor_vims":     all_survivor_vims,
        "initial_feat":      init_feats,
        "initial_vim":       init_vims,
        "initial_survivor":  init_survivors,
        "initial_module":    init_modules,
        "module_list":       module_list,
    }
