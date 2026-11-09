"""
Phase 1: Per module Screening RFE.

Runs rfe_loop on each module independently on the FULL dataset (no resampling).
Retains keep_fraction of each module's features as candidates for Phase 2.

Matches run_screen_rfe() from mf_python_backend.py.

This module also holds the optional Phase 1 result cache (see the section below).
"""

from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from .rfe import rfe_loop, blockwise_reduce, _map_labels
from .utils import thread_limit_ctx, resolve_blas_threads
from .parameters import screen_control


# Phase 1 (screening) result cache
#
# Screening depends only on X, y, module membership, screen_params, nodesize,
# classification, seed, max_modsize, and the SHAP backend, NOT on select_params.
# The common tuning / cross validation workflow reruns mf() many times varying
# only select_params (pi_thr, shadow_percentile, n_reps, ...), so caching Phase 1
# lets those runs skip screening entirely.
#
# Two backends, chosen via the `screen_cache` argument to mf():
#   None / False  means no caching (default; current behavior)
#   True          means in process memory cache (persists for the Python session)
#   str / Path    means on disk cache directory (persists across sessions)
#
# The cache is keyed on a hash of everything that affects screening output, so a
# hit returns byte identical survivors and a change to any relevant input misses.
# `use_numba` and `num_processors` are deliberately excluded from the key: they do
# not change screening results (only speed / determinism preserving), so a cached
# result is valid regardless of their values.

# in process cache: {key_hex: screen_result_dict}
_MEM_CACHE: dict[str, dict] = {}

# counters for observability / tests
STATS = {"hits": 0, "misses": 0}


def compute_screen_key(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    feature_names: list,
    module_assign: list,
    module_list: list,
    screen_params,
    nodesize: int,
    classification: bool,
    seed: int,
    max_modsize: int,
    backend: str,
) -> str:
    """Deterministic hash of all inputs that affect screening output."""
    h = hashlib.blake2b(digest_size=32)

    # data: exact bytes + shape + dtype
    Xc = np.ascontiguousarray(X_arr)
    h.update(str(Xc.dtype).encode()); h.update(str(Xc.shape).encode())
    h.update(Xc.tobytes())

    yc = np.ascontiguousarray(np.asarray(y_arr))
    h.update(str(yc.dtype).encode()); h.update(yc.tobytes())

    # feature names + module structure (order matters)
    h.update("\x00".join(map(str, feature_names)).encode())
    h.update("\x00".join(map(str, module_assign)).encode())
    h.update("\x00".join(map(str, module_list)).encode())

    # screen params (all fields that steer the RFE)
    sp = screen_params
    for v in (sp.drop_fraction, sp.keep_fraction, sp.mtry_factor,
              sp.min_ntree, sp.ntree_factor):
        h.update(repr(v).encode())

    # scalar execution inputs that affect results
    for v in (nodesize, bool(classification), int(seed), int(max_modsize), str(backend)):
        h.update(repr(v).encode())

    return h.hexdigest()


def load_cached(key: str, cache) -> dict | None:
    """Return a cached screen result for `key`, or None on miss."""
    if cache is True:
        return _MEM_CACHE.get(key)
    # disk cache: cache is a directory path
    fp = Path(cache) / f"screen_{key}.pkl"
    if fp.exists():
        try:
            with open(fp, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None
    return None


def store_cached(key: str, cache, result: dict) -> None:
    """Persist a screen result under `key`."""
    if cache is True:
        _MEM_CACHE[key] = result
        return
    d = Path(cache)
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"screen_{key}.pkl"
    tmp = fp.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(fp)   # atomic


def clear_memory_cache() -> None:
    """Drop the in process screening cache (does not touch on disk caches)."""
    _MEM_CACHE.clear()
    STATS["hits"] = 0
    STATS["misses"] = 0




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
    Phase 1: per module screening RFE on the full dataset.

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
        survivor_features  : list of lists: per module surviving feature names
        survivor_vims      : list of lists: per module VIM scores
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

    # inner thread cap when modules run in parallel
    inner_limit = resolve_blas_threads(blas_threads, n_jobs)

    def _screen_one_module(mod):
        """Screen a single module. Deterministic given `seed`, so order independent."""
        cols = mod_to_cols.get(mod, [])
        if not cols:
            return mod, [], []

        X_mod    = X_mat[:, cols]
        feat_mod = [feature_names[c] for c in cols]
        n_mod    = len(cols)
        target   = max(1, math.ceil(n_mod * sc.keep_fraction))

        if n_mod > max_modsize:
            if verbose:
                print(f"  Module {mod}: blockwise pre reduction "
                      f"({n_mod} > {max_modsize} features)")
            X_mod, feat_mod = blockwise_reduce(
                X_mod, feat_mod, y, max_modsize, 0.9,
                sc.drop_fraction, sc.mtry_factor, sc.ntree_factor,
                sc.min_ntree, nodesize, classification, seed, 1,
                use_shap=True, backend=backend, use_numba=use_numba,
                mtry_rule=sc.mtry_rule,
            )
            n_mod  = X_mod.shape[1]
            target = max(1, math.ceil(n_mod * sc.keep_fraction))

        if verbose:
            print(f"  Module {mod}: {n_mod} features, target {target}")

        survivors, vims, _ = rfe_loop(
            X_mod, feat_mod, y, target,
            sc.drop_fraction, sc.mtry_factor, sc.ntree_factor,
            sc.min_ntree, nodesize, classification, seed, 1,
            use_shap=True, backend=backend, use_numba=use_numba,
            mtry_rule=sc.mtry_rule,
        )
        return mod, survivors, vims

    # modules are independent and deterministic, so parallelize the outer loop
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

    for mod, survivors, vims in results:
        all_survivor_features.append(survivors)
        all_survivor_vims.append(vims)

    return {
        "survivor_features": all_survivor_features,
        "survivor_vims":     all_survivor_vims,
        "module_list":       module_list,
    }
