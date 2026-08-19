"""
Phase 2: Shadow-stability selection + single-shot RF/SHAP.

Implements shadow_mode = "split" and "within_module", DML y-residualization,
and the v18 single-shot final RF + TreeSHAP + interaction matrix.

Matches run_select_rfe() from sf_python_backend.py v18.
"""

from __future__ import annotations

import warnings
import numpy as np

from .rfe import (
    _build_rf,
    _compute_mtry,
    _compute_ntree,
    _map_labels,
    shap_full_signed,
)
from .dml import cross_fit_residuals
from .shadow import boruta_shadow_stability, apply_stable_threshold
from .parameters import select_control


def run_select_rfe(
    X_surv_mat: np.ndarray,
    y: np.ndarray,
    feature_names_surv: list,
    module_assignments_surv: list | None,
    select_params: select_control,
    nodesize: int,
    classification: bool,
    seed: int,
    n_jobs: int,
    max_modsize: int = 5000,
    compute_interactions: bool = True,
    backend: str = "shap",
    use_numba: bool = False,
    blas_threads=None,
    verbose: bool = False,
) -> dict:
    """
    Phase 2: shadow-stability selection on the survivor pool.

    shadow_mode = "split"
        Corr pool → shadow stability → stable_corr  (pi_thr threshold)
        Ind  pool → shadow stability → stable_ind   (elbow or pi_thr_indep)
        stable_union = stable_corr | stable_ind
        → single RF on X[stable_union] + TreeSHAP → return all features

    shadow_mode = "within_module"
        Per-module shadow stability → per-module stable sets
        stable_union = union of all module stable sets
        → single RF + TreeSHAP → return all features

    DML (use_dml_residual=True):
        dml_scope="indep": y_resid = y - RF_cv(X_corr); ind pool uses y_resid.
        dml_scope="all_modules": For each module m, y_resid_m = y - RF_cv(X_{-m}).

    Returns
    -------
    dict with keys:
        final_features     : list[str]         sorted by mean|SHAP| desc
        final_vims         : list[float]        mean|SHAP| per feature
        stability_freq     : list[float]        shadow-boot frequency
        shap_matrix        : ndarray (n, p)     signed SHAP values
        interaction_matrix : ndarray (p, p) or None
        pool_stability     : list[dict]         per-pool selection data
        selection_list_feats : list of lists    compat with R structure
        selection_list_vims  : list of lists
    """
    sl = select_params

    # ── coerce types ──────────────────────────────────────────────────────────
    X_surv_mat        = np.ascontiguousarray(X_surv_mat, dtype=np.float32)
    y                 = np.asarray(y).ravel()
    feature_names_surv = [str(f) for f in feature_names_surv]
    shadow_mode       = str(sl.shadow_mode).lower()
    use_shap          = True

    if classification:
        y = _map_labels(y)
    else:
        y = y.astype(np.float64)

    indep_set = set(str(m) for m in sl.indep_modules) if sl.indep_modules else set()
    if module_assignments_surv is not None:
        raw_mods    = [str(m) for m in module_assignments_surv]
        name_to_mod = dict(zip(feature_names_surv, raw_mods))
    else:
        name_to_mod = {}

    if verbose:
        print(f"[Phase 2] shadow_mode={shadow_mode}  "
              f"shadow_percentile={sl.shadow_percentile}  "
              f"pi_thr={sl.pi_thr}  n_boots={sl.n_boots}")

    # (Phase-2 blockwise pre-reduction removed by design — the full survivor
    #  pool goes straight into shadow stability.)

    # ── DML: compute y_resid for ind pool (scope="indep") ────────────────────
    # Computed once here regardless of shadow_mode branch.
    if sl.use_dml_residual and indep_set and name_to_mod:
        all_corr_idx = [i for i, f in enumerate(feature_names_surv)
                        if name_to_mod.get(f, "") not in indep_set]
        if len(all_corr_idx) == 0:
            if verbose:
                print("  DML: no corr survivors — using original y for ind pool")
            y_for_ind = y
        else:
            X_corr_surv = X_surv_mat[:, all_corr_idx]
            if verbose:
                print(f"  DML: cross-fitted residualization "
                      f"({sl.dml_n_folds} folds, ntree={sl.dml_ntree}, "
                      f"p_corr={len(all_corr_idx)})")
            y_for_ind = cross_fit_residuals(
                X_corr   = X_corr_surv,
                y        = y,
                k_folds  = sl.dml_n_folds,
                ntree    = sl.dml_ntree,
                mtry_factor = float(sl.mtry_factor),
                nodesize    = int(nodesize),
                seed        = int(seed) + 88317,
                n_jobs      = int(n_jobs),
            )
    else:
        y_for_ind = y

    # ── helper: run shadow stability + threshold on a column subset ───────────
    def _shadow_on_cols(col_idx, label, seed_offset=0, is_indep=False, y_override=None):
        if len(col_idx) == 0:
            return [], {}, [], 0.0, [], None

        X_sub   = X_surv_mat[:, col_idx]
        names_s = [feature_names_surv[i] for i in col_idx]

        if y_override is not None:
            y_arr = y_override
        elif is_indep:
            y_arr = y_for_ind
        else:
            y_arr = y

        # A DML-residualized target is CONTINUOUS (y - RF_cv(...)), even when the
        # original task is classification. So this pool's shadow-stability RF must
        # be fit as a REGRESSOR — otherwise the continuous residual is cast to int
        # (mostly zeros) and the selection is degenerate.
        using_resid = (y_override is not None) or (is_indep and sl.use_dml_residual)
        pool_classification = classification and not using_resid

        if verbose:
            print(f"  Shadow stability on {label}: {len(col_idx)} features"
                  + (" [y=y_resid → regression]" if using_resid else ""))

        names_sort, freqs_sort = boruta_shadow_stability(
            X_pool           = X_sub,
            feat_names       = names_s,
            y                = y_arr,
            n_boots          = sl.n_boots,
            mtry_factor      = sl.mtry_factor,
            ntree_factor     = sl.ntree_factor,
            min_ntree        = sl.min_ntree,
            nodesize         = nodesize,
            classification   = pool_classification,
            seed             = seed + seed_offset,
            n_jobs           = n_jobs,
            use_shap         = use_shap,
            shadow_percentile= sl.shadow_percentile,
            backend          = backend,
            use_numba        = use_numba,
            blas_threads     = blas_threads,
            early_stop_boots = sl.early_stop_boots,
            early_stop_tol   = sl.early_stop_tol,
            early_stop_check_every = sl.early_stop_check_every,
            mtry_rule        = sl.mtry_rule,
            mtry_on_real     = sl.mtry_on_real,
        )
        freq_map = dict(zip(names_sort, freqs_sort))

        if sl.threshold_mode == "elbow":
            # Alternative A: unified adaptive elbow rule for BOTH pools.
            stable, threshold, elbow_rank = apply_stable_threshold(
                names_sort, freqs_sort, sl.pi_thr, use_elbow=True, label=label,
            )
        elif is_indep:
            use_elbow = (sl.pi_thr_indep is None)
            thr       = sl.pi_thr_indep if not use_elbow else sl.pi_thr
            stable, threshold, elbow_rank = apply_stable_threshold(
                names_sort, freqs_sort, thr, use_elbow=use_elbow, label=label,
            )
        else:
            stable, threshold, elbow_rank = apply_stable_threshold(
                names_sort, freqs_sort, sl.pi_thr, use_elbow=False, label=label,
            )

        return names_sort, freq_map, stable, threshold, freqs_sort, elbow_rank

    # ─────────────────────────────────────────────────────────────────────────
    # MODE A: "split" — corr pool vs ind pool
    # ─────────────────────────────────────────────────────────────────────────
    if shadow_mode == "split":
        if indep_set and name_to_mod:
            corr_idx  = [i for i, f in enumerate(feature_names_surv)
                         if name_to_mod.get(f, "") not in indep_set]
            indep_idx = [i for i, f in enumerate(feature_names_surv)
                         if name_to_mod.get(f, "") in indep_set]
        else:
            corr_idx  = list(range(len(feature_names_surv)))
            indep_idx = []

        # optional: residualize y for corr pool too (scope="all_modules")
        y_for_corr = y
        if sl.use_dml_residual and sl.dml_scope == "all_modules" and indep_idx:
            X_ind_surv = X_surv_mat[:, indep_idx]
            if verbose:
                print(f"  split+mDML: residualizing y for corr pool using "
                      f"{len(indep_idx)} ind features")
            y_for_corr = cross_fit_residuals(
                X_corr      = X_ind_surv,
                y           = y,
                k_folds     = sl.dml_n_folds,
                ntree       = sl.dml_ntree,
                mtry_factor = float(sl.mtry_factor),
                nodesize    = int(nodesize),
                seed        = int(seed) + 88317 + 2,
                n_jobs      = int(n_jobs),
            )

        corr_override = (y_for_corr
                         if sl.use_dml_residual and sl.dml_scope == "all_modules"
                         else None)

        names_corr,  freq_corr,  stable_corr,  thr_corr,  freqs_corr,  elbow_corr  = \
            _shadow_on_cols(corr_idx,  "corr",  seed_offset=0, is_indep=False,
                            y_override=corr_override)
        names_indep, freq_indep, stable_indep, thr_indep, freqs_indep, elbow_indep = \
            _shadow_on_cols(indep_idx, "indep", seed_offset=1, is_indep=True)

        freq_map_all = {**freq_corr, **freq_indep}
        stable_union = list(set(stable_corr) | set(stable_indep))

        pool_stability = []
        if names_corr:
            stable_set_c = set(stable_corr)
            pool_stability.append({
                "pool":         "corr",
                "feature_names": names_corr,
                "freqs":        freqs_corr,
                "threshold":    float(thr_corr),
                "selected":     [f in stable_set_c for f in names_corr],
                "elbow_rank":   elbow_corr,
            })
        if names_indep:
            stable_set_i = set(stable_indep)
            pool_stability.append({
                "pool":         "indep",
                "feature_names": names_indep,
                "freqs":        freqs_indep,
                "threshold":    float(thr_indep),
                "selected":     [f in stable_set_i for f in names_indep],
                "elbow_rank":   elbow_indep,
            })

    # ─────────────────────────────────────────────────────────────────────────
    # MODE B: "within_module" — one shadow run per module
    # ─────────────────────────────────────────────────────────────────────────
    elif shadow_mode == "within_module":
        pool_stability = []

        if not name_to_mod:
            if verbose:
                print("  within_module: no module assignments — running as single pool")
            names_all, freq_map_all, stable_union, thr_all, freqs_all, elbow_all = \
                _shadow_on_cols(list(range(len(feature_names_surv))),
                                "all", seed_offset=0, is_indep=False)
            if names_all:
                stable_set_all = set(stable_union)
                pool_stability.append({
                    "pool":         "all",
                    "feature_names": names_all,
                    "freqs":        freqs_all,
                    "threshold":    float(thr_all),
                    "selected":     [f in stable_set_all for f in names_all],
                    "elbow_rank":   elbow_all,
                })
        else:
            mod_to_idx: dict = {}
            for i, f in enumerate(feature_names_surv):
                mod = name_to_mod.get(f, "__unknown__")
                mod_to_idx.setdefault(mod, []).append(i)

            freq_map_all: dict = {}
            stable_union: list = []

            for m_offset, (mod_label, col_idx) in enumerate(mod_to_idx.items()):
                is_indep_mod = (mod_label in indep_set)

                # per-module DML: residualize y by all OTHER modules
                y_mod = None
                if sl.use_dml_residual and sl.dml_scope == "all_modules":
                    other_idx = [i for i in range(len(feature_names_surv))
                                 if i not in col_idx]
                    if not other_idx:
                        if verbose:
                            print(f"  mDML: no other-module features for "
                                  f"module:{mod_label} — using original y")
                        y_mod = y
                    else:
                        X_other = X_surv_mat[:, other_idx]
                        if verbose:
                            print(f"  mDML: residualizing y for module:{mod_label} "
                                  f"using {len(other_idx)} other-module features")
                        y_mod = cross_fit_residuals(
                            X_corr      = X_other,
                            y           = y,
                            k_folds     = sl.dml_n_folds,
                            ntree       = sl.dml_ntree,
                            mtry_factor = float(sl.mtry_factor),
                            nodesize    = int(nodesize),
                            seed        = int(seed) + 88317 + m_offset * 997,
                            n_jobs      = int(n_jobs),
                        )

                names_mod, freq_mod, stable_mod, thr_mod, freqs_mod, elbow_mod = \
                    _shadow_on_cols(col_idx, f"module:{mod_label}",
                                    seed_offset=m_offset, is_indep=is_indep_mod,
                                    y_override=y_mod)
                freq_map_all.update(freq_mod)
                stable_union.extend(stable_mod)

                if names_mod:
                    stable_set_mod = set(stable_mod)
                    pool_stability.append({
                        "pool":         mod_label,
                        "feature_names": names_mod,
                        "freqs":        freqs_mod,
                        "threshold":    float(thr_mod),
                        "selected":     [f in stable_set_mod for f in names_mod],
                        "elbow_rank":   elbow_mod,
                    })

            stable_union = list(set(stable_union))

    else:
        raise ValueError(
            f"shadow_mode must be 'split' or 'within_module', got '{shadow_mode}'"
        )

    n_stable = len(stable_union)
    if verbose:
        print(f"  stable_union: {n_stable} features"
              + (" (empty)" if n_stable == 0 else " → single-shot RF + TreeSHAP"))

    # ── sort all features by freq for selection_list compat ───────────────────
    all_sorted = sorted(freq_map_all, key=lambda f: freq_map_all[f], reverse=True)
    all_freqs  = [freq_map_all[f] for f in all_sorted]

    if n_stable == 0:
        return {
            "final_features":       [],
            "final_vims":           [],
            "stability_freq":       [],
            "shap_matrix":          np.zeros((X_surv_mat.shape[0], 0)),
            "interaction_matrix":   None,
            "pool_stability":       pool_stability,
            "selection_list_feats": [all_sorted, []],
            "selection_list_vims":  [all_freqs,  []],
        }

    # ── v18: single-shot RF on all stable features → TreeSHAP ────────────────
    stable_sorted = sorted(stable_union,
                           key=lambda f: freq_map_all.get(f, 0.0), reverse=True)

    feat_idx_map = {f: i for i, f in enumerate(feature_names_surv)}
    stable_cols  = [feat_idx_map[f] for f in stable_sorted]
    X_stable     = X_surv_mat[:, stable_cols]

    ntree_s = _compute_ntree(n_stable, sl.ntree_factor, sl.min_ntree)
    mtry_s  = _compute_mtry(n_stable, sl.mtry_factor, classification, sl.mtry_rule)

    if verbose:
        print(f"  Fitting single-shot RF: n_stable={n_stable}  "
              f"ntree={ntree_s}  mtry={mtry_s}")

    rf_stable = _build_rf(ntree_s, mtry_s, nodesize,
                          (seed + 8888) % (2**31 - 1), n_jobs, classification)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_stable.fit(X_stable, y)

    # signed SHAP matrix (n × n_stable)
    sv_signed     = shap_full_signed(rf_stable, X_stable, classification, backend=backend)
    shap_mean_vec = np.mean(np.abs(sv_signed), axis=0)

    # sort by mean|SHAP| descending
    order          = np.argsort(-shap_mean_vec)
    final_features = [stable_sorted[i] for i in order]
    final_vims     = [float(shap_mean_vec[i]) for i in order]
    stab_freqs     = [float(freq_map_all.get(f, 0.0)) for f in final_features]

    shap_matrix_ordered = sv_signed[:, order]   # (n, n_stable)

    # ── TreeSHAP interaction matrix is computed LAZILY ────────────────────────
    # The interaction call is ~p× the cost of regular SHAP, so we defer it: the
    # ingredients (fitted RF, stable submatrix, final ordering) are returned and
    # ShapleyForestResult.interaction_matrix computes + caches on first access.
    # `compute_interactions=False` disables it entirely (and drops the retained RF
    # to save memory).
    return {
        "final_features":       final_features,
        "final_vims":           final_vims,
        "stability_freq":       stab_freqs,
        "shap_matrix":          shap_matrix_ordered,          # ndarray (n, p)
        "interaction_matrix":   None,                         # filled lazily
        "pool_stability":       pool_stability,
        "selection_list_feats": [all_sorted, list(stable_sorted)],
        "selection_list_vims":  [all_freqs,
                                  [freq_map_all.get(f, 0.0) for f in stable_sorted]],
        # ── lazy-interaction ingredients (None when compute_interactions=False) ──
        "_rf_stable":           rf_stable if compute_interactions else None,
        "_X_stable":            X_stable if compute_interactions else None,
        "_shap_order":          order if compute_interactions else None,
        "_shap_backend":        backend,
        "_classification":      classification,
    }
