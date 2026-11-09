"""
Phase 2: Shadow stability selection + single shot RF/SHAP.

Implements shadow_mode = "split" and "within_module", cfRes y residualization
(the cross_fit_residuals helper below), and the single shot final RF + TreeSHAP
+ interaction matrix.

Matches run_select_rfe() from mf_python_backend.py.
"""

from __future__ import annotations

import math
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .rfe import (
    _build_rf,
    _compute_mtry,
    _compute_ntree,
    _map_labels,
    shap_full_signed,
)
from .shadow import boruta_shadow_stability, apply_stable_threshold
from .parameters import select_control


# cross fit residualization (cfRes)
#
# Mirrors _cross_fit_residuals() in inst/python/mf_python_backend.py and R's cfRes:
#   y_resid = y - RF_cv(X_corr)
# After residualization, y_resid ~ theta * X_unassigned + eps, boosting the SNR
# for unassigned pool shadow stability.

def cross_fit_residuals(
    X_corr: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
    ntree: int = 300,
    mtry_factor: float = 1.0,
    nodesize: int = 5,
    seed: int = 42,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Robinson / cfRes cross fitted residualization.

    Each observation's predicted value comes from a model trained on the
    other K-1 folds (no obs influences its own residual).

    Parameters
    ----------
    X_corr : (n, p_corr)  correlated pool feature matrix
    y      : (n,)          original continuous or mapped integer outcome
    k_folds : int           number of cross fitting folds (default 5)
    ntree   : int           trees per fold RF (default 300)
    mtry_factor : float     mtry multiplier (default 1.0)
    nodesize : int          min samples per leaf (default 5)
    seed    : int           RNG seed
    n_jobs  : int           joblib n_jobs for RF tree building

    Returns
    -------
    y_resid : (n,) float64   y - RF_hat(X_corr), cross fitted
    """
    n      = len(y)
    p_corr = X_corr.shape[1]
    rng    = np.random.default_rng(int(seed))

    # balanced folds by cyclic permutation
    perm     = rng.permutation(n)
    fold_ids = np.empty(n, dtype=int)
    for k in range(k_folds):
        fold_ids[perm[k::k_folds]] = k

    y_hat = np.zeros(n, dtype=np.float64)
    mtry  = max(1, min(p_corr, int(math.ceil(mtry_factor * math.sqrt(p_corr)))))

    for k in range(k_folds):
        tr = np.where(fold_ids != k)[0]
        te = np.where(fold_ids == k)[0]

        rf = RandomForestRegressor(
            n_estimators     = int(ntree),
            max_features     = mtry,
            min_samples_leaf = int(nodesize),
            random_state     = (int(seed) + k * 1999) % (2**31 - 1),
            n_jobs           = int(n_jobs),
        )
        rf.fit(X_corr[tr], y[tr].astype(np.float64))
        y_hat[te] = rf.predict(X_corr[te])

    return y.astype(np.float64) - y_hat


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
    compute_interactions: bool = True,
    backend: str = "shap",
    use_numba: bool = False,
    blas_threads=None,
    verbose: bool = False,
) -> dict:
    """
    Phase 2: shadow stability selection on the survivor pool.

    shadow_mode = "split"
        Corr pool: shadow stability gives stable_corr  (pi_thr threshold)
        Unassigned pool: shadow stability gives stable_unassigned   (elbow or pi_thr_unassigned)
        stable_union = stable_corr | stable_unassigned
        then a single RF on X[stable_union] + TreeSHAP, and return all features

    shadow_mode = "within_module"
        Per module shadow stability gives per module stable sets
        stable_union = union of all module stable sets
        then a single RF + TreeSHAP, and return all features

    cfRes (use_cfres=True):
        cfres_scope="unassigned": y_resid = y - RF_cv(X_corr); unassigned pool uses y_resid.
        cfres_scope="all_modules": For each module m, y_resid_m = y - RF_cv(X_{-m}).

    Returns
    -------
    dict with keys:
        final_features     : list[str]         sorted by mean|SHAP| desc
        final_vims         : list[float]        mean|SHAP| per feature
        stability_freq     : list[float]        shadow subsample frequency
        shap_matrix        : ndarray (n, p)     signed SHAP values
        interaction_matrix : ndarray (p, p) or None
        pool_stability     : list[dict]         per pool selection data
        selection_list_feats : list of lists    compat with R structure
        selection_list_vims  : list of lists
    """
    sl = select_params

    # coerce types
    X_surv_mat        = np.ascontiguousarray(X_surv_mat, dtype=np.float32)
    y                 = np.asarray(y).ravel()
    feature_names_surv = [str(f) for f in feature_names_surv]
    shadow_mode       = str(sl.shadow_mode).lower()
    use_shap          = True

    if classification:
        y = _map_labels(y)
    else:
        y = y.astype(np.float64)

    unassigned_set = set(str(m) for m in sl.unassigned_modules) if sl.unassigned_modules else set()
    # Deterministic feature order: set iteration order changes between Python
    # processes (string hashing), which would change the final forest's columns.
    _feat_order = {str(f): i for i, f in enumerate(feature_names_surv)}
    _by_order = lambda names: sorted(set(names), key=lambda f: _feat_order.get(f, len(_feat_order)))
    if module_assignments_surv is not None:
        raw_mods    = [str(m) for m in module_assignments_surv]
        name_to_mod = dict(zip(feature_names_surv, raw_mods))
    else:
        name_to_mod = {}

    if verbose:
        print(f"[Phase 2] shadow_mode={shadow_mode}  "
              f"shadow_percentile={sl.shadow_percentile}  "
              f"pi_thr={sl.pi_thr}  n_reps={sl.n_reps}")

    # The full survivor pool goes straight into shadow stability
    # (no blockwise pre reduction in Phase 2).

    # cfRes: compute y_resid for unassigned pool (scope="unassigned")
    # Computed once here regardless of shadow_mode branch.
    if sl.use_cfres and unassigned_set and name_to_mod:
        all_corr_idx = [i for i, f in enumerate(feature_names_surv)
                        if name_to_mod.get(f, "") not in unassigned_set]
        if len(all_corr_idx) == 0:
            if verbose:
                print("  cfRes: no corr survivors, using original y for unassigned pool")
            y_for_unassigned = y
        else:
            X_corr_surv = X_surv_mat[:, all_corr_idx]
            if verbose:
                print(f"  cfRes: cross fitted residualization "
                      f"({sl.cfres_n_folds} folds, ntree={sl.cfres_ntree}, "
                      f"p_corr={len(all_corr_idx)})")
            y_for_unassigned = cross_fit_residuals(
                X_corr   = X_corr_surv,
                y        = y,
                k_folds  = sl.cfres_n_folds,
                ntree    = sl.cfres_ntree,
                mtry_factor = float(sl.mtry_factor),
                nodesize    = int(nodesize),
                seed        = int(seed) + 88317,
                n_jobs      = int(n_jobs),
            )
    else:
        y_for_unassigned = y

    # helper: run shadow stability + threshold on a column subset
    def _shadow_on_cols(col_idx, label, seed_offset=0, is_unassigned=False, y_override=None):
        if len(col_idx) == 0:
            return [], {}, [], 0.0, [], None

        X_sub   = X_surv_mat[:, col_idx]
        names_s = [feature_names_surv[i] for i in col_idx]

        if y_override is not None:
            y_arr = y_override
        elif is_unassigned:
            y_arr = y_for_unassigned
        else:
            y_arr = y

        # A cfRes residualized target is CONTINUOUS (y - RF_cv(...)), even when the
        # original task is classification. So this pool's shadow stability RF must
        # be fit as a REGRESSOR; otherwise the continuous residual is cast to int
        # (mostly zeros) and the selection is degenerate.
        using_resid = (y_override is not None) or (is_unassigned and sl.use_cfres)
        pool_classification = classification and not using_resid

        if verbose:
            print(f"  Shadow stability on {label}: {len(col_idx)} features"
                  + (" [y=y_resid, regression]" if using_resid else ""))

        names_sort, freqs_sort = boruta_shadow_stability(
            X_pool           = X_sub,
            feat_names       = names_s,
            y                = y_arr,
            n_reps          = sl.n_reps,
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
            early_stop = sl.early_stop,
            early_stop_tol   = sl.early_stop_tol,
            early_stop_check_every = sl.early_stop_check_every,
            mtry_rule        = sl.mtry_rule,
            mtry_on_real     = sl.mtry_on_real,
        )
        freq_map = dict(zip(names_sort, freqs_sort))

        if sl.threshold_mode == "elbow":
            # Alternative A: unified adaptive elbow rule for BOTH pools.
            stable, threshold, elbow_rank = apply_stable_threshold(
                names_sort, freqs_sort, sl.pi_thr, use_elbow=True,
            )
        elif is_unassigned:
            use_elbow = (sl.pi_thr_unassigned is None)
            thr       = sl.pi_thr_unassigned if not use_elbow else sl.pi_thr
            stable, threshold, elbow_rank = apply_stable_threshold(
                names_sort, freqs_sort, thr, use_elbow=use_elbow,
            )
        else:
            stable, threshold, elbow_rank = apply_stable_threshold(
                names_sort, freqs_sort, sl.pi_thr, use_elbow=False,
            )

        return names_sort, freq_map, stable, threshold, freqs_sort, elbow_rank

    # MODE A: "split": corr pool vs unassigned pool
    if shadow_mode == "split":
        if unassigned_set and name_to_mod:
            corr_idx  = [i for i, f in enumerate(feature_names_surv)
                         if name_to_mod.get(f, "") not in unassigned_set]
            unassigned_idx = [i for i, f in enumerate(feature_names_surv)
                         if name_to_mod.get(f, "") in unassigned_set]
        else:
            corr_idx  = list(range(len(feature_names_surv)))
            unassigned_idx = []

        # optional: residualize y for corr pool too (scope="all_modules")
        y_for_corr = y
        if sl.use_cfres and sl.cfres_scope == "all_modules" and unassigned_idx:
            X_ind_surv = X_surv_mat[:, unassigned_idx]
            if verbose:
                print(f"  split+mcfRes: residualizing y for corr pool using "
                      f"{len(unassigned_idx)} unassigned features")
            y_for_corr = cross_fit_residuals(
                X_corr      = X_ind_surv,
                y           = y,
                k_folds     = sl.cfres_n_folds,
                ntree       = sl.cfres_ntree,
                mtry_factor = float(sl.mtry_factor),
                nodesize    = int(nodesize),
                seed        = int(seed) + 88317 + 2,
                n_jobs      = int(n_jobs),
            )

        corr_override = (y_for_corr
                         if sl.use_cfres and sl.cfres_scope == "all_modules"
                         else None)

        names_corr,  freq_corr,  stable_corr,  thr_corr,  freqs_corr,  elbow_corr  = \
            _shadow_on_cols(corr_idx,  "corr",  seed_offset=0, is_unassigned=False,
                            y_override=corr_override)
        names_unassigned, freq_unassigned, stable_unassigned, thr_unassigned, freqs_unassigned, elbow_unassigned = \
            _shadow_on_cols(unassigned_idx, "unassigned", seed_offset=1, is_unassigned=True)

        freq_map_all = {**freq_corr, **freq_unassigned}
        stable_union = _by_order(list(stable_corr) + list(stable_unassigned))

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
        if names_unassigned:
            stable_set_i = set(stable_unassigned)
            pool_stability.append({
                "pool":         "unassigned",
                "feature_names": names_unassigned,
                "freqs":        freqs_unassigned,
                "threshold":    float(thr_unassigned),
                "selected":     [f in stable_set_i for f in names_unassigned],
                "elbow_rank":   elbow_unassigned,
            })

    # MODE B: "within_module": one shadow run per module
    elif shadow_mode == "within_module":
        pool_stability = []

        if not name_to_mod:
            if verbose:
                print("  within_module: no module assignments, running as single pool")
            names_all, freq_map_all, stable_union, thr_all, freqs_all, elbow_all = \
                _shadow_on_cols(list(range(len(feature_names_surv))),
                                "all", seed_offset=0, is_unassigned=False)
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
                is_unassigned_mod = (mod_label in unassigned_set)

                # per module cfRes: residualize y by all OTHER modules
                y_mod = None
                if sl.use_cfres and sl.cfres_scope == "all_modules":
                    other_idx = [i for i in range(len(feature_names_surv))
                                 if i not in col_idx]
                    if not other_idx:
                        if verbose:
                            print(f"  mcfRes: no other module features for "
                                  f"module:{mod_label}, using original y")
                        y_mod = y
                    else:
                        X_other = X_surv_mat[:, other_idx]
                        if verbose:
                            print(f"  mcfRes: residualizing y for module:{mod_label} "
                                  f"using {len(other_idx)} other module features")
                        y_mod = cross_fit_residuals(
                            X_corr      = X_other,
                            y           = y,
                            k_folds     = sl.cfres_n_folds,
                            ntree       = sl.cfres_ntree,
                            mtry_factor = float(sl.mtry_factor),
                            nodesize    = int(nodesize),
                            seed        = int(seed) + 88317 + m_offset * 997,
                            n_jobs      = int(n_jobs),
                        )

                names_mod, freq_mod, stable_mod, thr_mod, freqs_mod, elbow_mod = \
                    _shadow_on_cols(col_idx, f"module:{mod_label}",
                                    seed_offset=m_offset, is_unassigned=is_unassigned_mod,
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

            stable_union = _by_order(stable_union)

    else:
        raise ValueError(
            f"shadow_mode must be 'split' or 'within_module', got '{shadow_mode}'"
        )

    n_stable = len(stable_union)
    if verbose:
        print(f"  stable_union: {n_stable} features"
              + (" (empty)" if n_stable == 0 else ", single shot RF + TreeSHAP"))

    # sort all features by freq for selection_list compat
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

    # single shot RF on all stable features, then TreeSHAP
    stable_sorted = sorted(stable_union,
                           key=lambda f: freq_map_all.get(f, 0.0), reverse=True)

    feat_idx_map = {f: i for i, f in enumerate(feature_names_surv)}
    stable_cols  = [feat_idx_map[f] for f in stable_sorted]
    X_stable     = X_surv_mat[:, stable_cols]

    ntree_s = _compute_ntree(n_stable, sl.ntree_factor, sl.min_ntree)
    mtry_s  = _compute_mtry(n_stable, sl.mtry_factor, classification, sl.mtry_rule)

    if verbose:
        print(f"  Fitting single shot RF: n_stable={n_stable}  "
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

    # TreeSHAP interaction matrix is computed LAZILY
    # The interaction call is ~p× the cost of regular SHAP, so we defer it: the
    # ingredients (fitted RF, stable submatrix, final ordering) are returned and
    # MossyForestResult.interaction_matrix computes + caches on first access.
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
        # lazy interaction ingredients (None when compute_interactions=False)
        "_rf_stable":           rf_stable if compute_interactions else None,
        "_X_stable":            X_stable if compute_interactions else None,
        "_shap_order":          order if compute_interactions else None,
        "_shap_backend":        backend,
        "_classification":      classification,
    }
