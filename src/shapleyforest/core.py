"""
ShapleyForest — main user-facing API.

Primary functions (matching R's API):
  sf(X, y, module_membership, ...)    — two-phase feature selection
  wsf(X, y, WGCNA_params, ...)        — same but runs WGCNA first

sklearn-compatible estimator:
  ShapleyForest(...)                  — wraps sf() for pipeline use

Algorithm (matching R's sf() / sf_python_backend.py v18):
  Phase 1 (Screen):    per-module RFE on the full dataset
  Phase 2 (Select):    shadow-stability selection on survivors
  Phase 3 (Final RF):  RF on the stable set for prediction
"""

from __future__ import annotations

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .parameters import screen_control, select_control, WGCNA_control
from .rfe import _build_rf, _compute_mtry, _compute_ntree, _map_labels, shap_full_signed
from .screen import run_screen_rfe
from .select import run_select_rfe
from .result import ShapleyForestResult
from .modules import detect_modules
from . import cache
from .cache import compute_screen_key, load_cached, store_cached


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_module_membership(module_membership, feature_names: list) -> dict:
    """
    Normalize module_membership to {feature_name: module_label}.

    Accepts:
      - dict  {feature: module}
      - list/ndarray  (aligned with feature_names)
      - pd.Series     (index = feature names)
    """
    if isinstance(module_membership, dict):
        return {str(k): str(v) for k, v in module_membership.items()}
    if isinstance(module_membership, pd.Series):
        return {str(k): str(v) for k, v in module_membership.items()}
    # list / ndarray — align with feature_names
    mm = list(module_membership)
    if len(mm) != len(feature_names):
        raise ValueError(
            f"module_membership length ({len(mm)}) does not match "
            f"number of features ({len(feature_names)})."
        )
    return {str(f): str(m) for f, m in zip(feature_names, mm)}


def _build_stability_data(pool_stability: list) -> pd.DataFrame | None:
    """Convert pool_stability dicts to a tidy DataFrame for plot_stability_elbow."""
    if not pool_stability:
        return None
    rows = []
    for pool_dict in pool_stability:
        pool    = pool_dict["pool"]
        names   = pool_dict["feature_names"]
        freqs   = pool_dict["freqs"]
        thr     = pool_dict["threshold"]
        sels    = pool_dict["selected"]
        elbow_r = pool_dict.get("elbow_rank")

        for rank, (f, fr, s) in enumerate(zip(names, freqs, sels), start=1):
            rows.append({
                "pool":       pool,
                "rank":       rank,
                "feature":    f,
                "freq":       fr,
                "selected":   s,
                "threshold":  thr,
                "elbow_rank": elbow_r,
            })
    if not rows:
        return None
    return pd.DataFrame(rows)


def _shap_direction(feature_values: np.ndarray, shap_matrix: np.ndarray,
                    ambiguity_thresh: float = 0.1) -> dict:
    """
    Global effect direction per feature from the feature-value ↔ SHAP relationship.

    Because mean signed SHAP ≈ 0 (contributions cancel), direction lives in HOW
    a feature's SHAP tracks its value. We use the Spearman correlation between
    x_j and phi_j (rank-based → captures monotone direction without assuming
    linearity):

        dir_corr_j = spearman(x_j, phi_j)
        direction_j = sign(dir_corr_j)          (+1 promotes, -1 suppresses)
        ambiguous_j = |dir_corr_j| < ambiguity_thresh   (non-monotone / unclear;
                       a single direction is misleading — use a dependence plot)

    Returns arrays: dir_corr, direction, ambiguous.
    """
    from scipy.stats import spearmanr
    p = shap_matrix.shape[1]
    dir_corr = np.zeros(p, dtype=np.float64)
    for j in range(p):
        xj, sj = feature_values[:, j], shap_matrix[:, j]
        if np.ptp(xj) == 0 or np.ptp(sj) == 0:   # constant → undefined
            dir_corr[j] = 0.0
            continue
        rho, _ = spearmanr(xj, sj)
        dir_corr[j] = 0.0 if np.isnan(rho) else rho
    direction = np.sign(dir_corr).astype(int)
    ambiguous = np.abs(dir_corr) < ambiguity_thresh
    return {"dir_corr": dir_corr, "direction": direction, "ambiguous": ambiguous}


def _build_final_shap_df(
    stable_features: list,
    shap_matrix: np.ndarray,
    feature_values: np.ndarray,
    final_vims: list,
    stab_freq: list,
    module_membership_map: dict,
) -> pd.DataFrame:
    """Build the final_SHAP DataFrame matching R's structure (with CIs + direction)."""
    n_obs = shap_matrix.shape[0]
    feat_ord = stable_features

    df = pd.DataFrame({
        "feature_name":        feat_ord,
        "variable_importance": [round(v, 4) for v in final_vims],
        "module_membership":   [module_membership_map.get(f, ".") for f in feat_ord],
    })

    if n_obs > 0 and shap_matrix.shape[1] > 0:
        abs_mat     = np.abs(shap_matrix)
        mean_abs    = np.mean(abs_mat, axis=0)
        se_abs      = np.std(abs_mat, axis=0, ddof=1) / np.sqrt(n_obs)
        mean_signed = np.mean(shap_matrix, axis=0)
        se_signed   = np.std(shap_matrix, axis=0, ddof=1) / np.sqrt(n_obs)

        df["mean_abs_shap"]    = np.round(mean_abs, 4)
        df["se_abs_shap"]      = np.round(se_abs, 4)
        df["ci_lower_abs"]     = np.round(np.maximum(0, mean_abs - 1.96 * se_abs), 4)
        df["ci_upper_abs"]     = np.round(mean_abs + 1.96 * se_abs, 4)
        df["mean_signed_shap"] = np.round(mean_signed, 4)
        df["se_signed_shap"]   = np.round(se_signed, 4)
        df["ci_lower_signed"]  = np.round(mean_signed - 1.96 * se_signed, 4)
        df["ci_upper_signed"]  = np.round(mean_signed + 1.96 * se_signed, 4)
        df["stability_freq"]   = np.round(stab_freq, 4)

        # ── global effect direction (feature-value ↔ SHAP correlation) ────────
        d = _shap_direction(feature_values, shap_matrix)
        df["direction"]         = d["direction"]                    # +1 / -1 / 0
        df["dir_corr"]          = np.round(d["dir_corr"], 4)        # Spearman ρ
        # signed importance = direction × magnitude (for a diverging-bar report)
        df["signed_importance"] = np.round(d["direction"] * mean_abs, 4)
        df["direction_ambiguous"] = d["ambiguous"]                 # non-monotone flag

    return df.reset_index(drop=True)


# ── sf() — main two-phase function ───────────────────────────────────────────

def sf(
    X,
    y,
    module_membership,
    screen_params: screen_control | None    = None,
    select_params: select_control | None    = None,
    final_ntree: int                        = 500,
    nodesize: int | None                    = None,
    num_processors: int                     = 1,
    max_modsize: int                        = 5000,
    auto_initial: str                       = "4",
    verbose: int                            = 1,
    seed: int | None                        = None,
    compute_interactions: bool              = True,
    shap_backend: str                       = "shap",
    use_numba: bool                         = False,
    blas_threads: int | None                = None,
    screen_cache=None,
) -> ShapleyForestResult:
    """
    Shapley Forest Feature Selection.

    Matches R's sf().

    Parameters
    ----------
    X : pd.DataFrame or array-like (n, p)
        Feature matrix. Column names are used when X is a DataFrame.
    y : array-like (n,)
        Outcome. Integer/bool → classification; float → regression.
    module_membership : dict, list, ndarray, or pd.Series
        Module label per feature. dict: {feature: label};
        list/array: aligned with X columns; Series: index = feature names.
    screen_params : screen_control
        Phase 1 parameters. Default: screen_control().
    select_params : select_control
        Phase 2 parameters. Default: select_control().
    final_ntree : int
        Trees in the final prediction RF. Default 500.
    nodesize : int or None
        Min terminal node size. None → 1 for classification, 5 for regression.
    num_processors : int
        Parallel workers for RF fitting. Default 1.
    max_modsize : int
        Module size above which blockwise pre-reduction is applied. Default 5000.
    auto_initial : str
        Initial-screen checkpoint.
        "1" — save CSV and stop.
        "2" — save CSV and continue.
        "3" — stop without saving.
        "4" — continue without saving (default).
    verbose : int
        0 = silent; 1 = progress messages; 2 = detailed. Default 1.
    seed : int or None
        RNG seed. None = random.
    compute_interactions : bool
        Retain the single-shot RF so the TreeSHAP interaction matrix can be
        computed lazily on first access (via result.interaction_matrix /
        detect_interaction / plot_potential_interactions). False disables
        interactions entirely and frees the retained RF. Default True.
    shap_backend : str
        "shap" (default) or "fasttreeshap". The latter returns identical
        values (exact mode) and is faster, but requires Python <=3.11 with
        numpy<2 / shap<0.46 (fasttreeshap's dependency pins have no Python
        3.12+ wheels). On a modern stack keep "shap".
    use_numba : bool
        Use a Numba-JIT mean|SHAP| reduction when Numba is installed.
        Default False.
    blas_threads : int or None
        Inner BLAS/OpenMP thread cap during parallel sections. None (default)
        = auto: capped to 1 when num_processors > 1, unrestricted otherwise.
    screen_cache : None, bool, or str/Path
        Cache Phase-1 (screening) results, which depend only on X, y, module
        membership, screen_params, nodesize, seed, max_modsize, and shap_backend
        — NOT on select_params. Lets tuning / CV runs that vary only
        select_params skip screening.
        None/False (default) = no caching;
        True = in-process memory cache (this Python session);
        str/Path = on-disk cache directory (persists across sessions).
        A cache hit returns byte-identical survivors.

    Returns
    -------
    ShapleyForestResult
    """
    if screen_params is None:
        screen_params = screen_control()
    if select_params is None:
        select_params = select_control()
    if shap_backend not in ("shap", "fasttreeshap"):
        raise ValueError("shap_backend must be 'shap' or 'fasttreeshap'")
    if seed is None:
        seed = int(time.time()) % (2**31 - 1)

    # ── validate inputs ───────────────────────────────────────────────────────
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = np.ascontiguousarray(X.values, dtype=np.float32)
    else:
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        feature_names = [f"X{i}" for i in range(X_arr.shape[1])]

    y_arr = np.asarray(y).ravel()
    CLASSIFICATION = (
        np.issubdtype(y_arr.dtype, np.integer) or y_arr.dtype == bool
        or (y_arr.dtype.kind in ("i", "u"))
        or (len(np.unique(y_arr)) <= 10 and y_arr.dtype.kind != "f")
    )

    if nodesize is None:
        nodesize = 1 if CLASSIFICATION else 5

    mm_map = _parse_module_membership(module_membership, feature_names)
    module_assign = [mm_map.get(f, "independent") for f in feature_names]
    module_list   = list(dict.fromkeys(module_assign))   # preserves order, deduplicates

    runtimes = {"Screen": np.nan, "Selection": np.nan, "Final_RF": np.nan}

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: Screening RFE  (optionally cached — independent of select_params)
    # ══════════════════════════════════════════════════════════════════════════
    if verbose >= 1:
        print("Screening ...")

    t0 = time.time()

    screen_result = None
    cache_key = None
    if screen_cache:
        cache_key = compute_screen_key(
            X_arr, y_arr, feature_names, module_assign, module_list,
            screen_params, nodesize, CLASSIFICATION, seed, max_modsize, shap_backend,
        )
        screen_result = load_cached(cache_key, screen_cache)
        if screen_result is not None:
            cache.STATS["hits"] += 1
            if verbose >= 1:
                print("  (screening cache hit — skipping Phase 1)")

    if screen_result is None:
        if screen_cache:
            cache.STATS["misses"] += 1
        screen_result = run_screen_rfe(
            X_mat              = X_arr,
            y                  = y_arr,
            feature_names      = feature_names,
            module_assignments = module_assign,
            module_list        = module_list,
            screen_params      = screen_params,
            nodesize           = nodesize,
            classification     = CLASSIFICATION,
            seed               = seed,
            n_jobs             = num_processors,
            max_modsize        = max_modsize,
            backend            = shap_backend,
            use_numba          = use_numba,
            blas_threads       = blas_threads,
            verbose            = (verbose >= 2),
        )
        if screen_cache:
            store_cached(cache_key, screen_cache, screen_result)

    runtimes["Screen"] = time.time() - t0

    # ── build initial_screen DataFrame ────────────────────────────────────────
    initial_screen = None
    if screen_result["initial_feat"]:
        initial_screen = pd.DataFrame({
            "Module":   screen_result["initial_module"],
            "Feature":  screen_result["initial_feat"],
            "VIM":      screen_result["initial_vim"],
            "Survivor": screen_result["initial_survivor"],
        })

    # ── auto_initial checkpoint ───────────────────────────────────────────────
    if auto_initial in ("1", "2") and initial_screen is not None:
        initial_screen.to_csv("initial_screen.csv", index=False)
        if verbose >= 1:
            print("Initial screen saved to 'initial_screen.csv'.")
    if auto_initial == "1":
        raise RuntimeError("Stopped after initial screen (auto_initial='1').")
    if auto_initial == "3":
        raise RuntimeError("Stopped after initial screen (auto_initial='3').")

    # ── assemble survivor pool ────────────────────────────────────────────────
    survivor_list: dict = {}
    survivors_flat: list = []
    surv_to_mod: dict   = {}
    for mod, surv_feats, surv_vims in zip(
        module_list,
        screen_result["survivor_features"],
        screen_result["survivor_vims"],
    ):
        df_surv = pd.DataFrame({
            "features":    surv_feats,
            "mod_varlist": surv_vims,
        })
        survivor_list[mod] = df_surv
        for f in surv_feats:
            survivors_flat.append(f)
            surv_to_mod[f] = mod

    # Order survivors in the ORIGINAL X column order to match R's sf(), which
    # builds X_surv via `X[, names(X) %in% survivors]` (original order). Column
    # order changes sklearn RF's mtry feature subsampling (it samples by index),
    # so matching the order is required for exact parity with the R backend.
    surv_set = set(survivors_flat)
    survivors_flat = [f for f in feature_names if f in surv_set]
    X_surv = X_arr[:, [feature_names.index(f) for f in survivors_flat]]

    if verbose >= 1:
        print(f"Selecting stable features ... ({len(survivors_flat)} candidates)")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: Shadow-stability selection
    # ══════════════════════════════════════════════════════════════════════════
    t0 = time.time()
    select_result = run_select_rfe(
        X_surv_mat           = X_surv,
        y                    = y_arr,
        feature_names_surv   = survivors_flat,
        module_assignments_surv = [surv_to_mod.get(f, "independent")
                                    for f in survivors_flat],
        select_params        = select_params,
        nodesize             = nodesize,
        classification       = CLASSIFICATION,
        seed                 = seed,
        n_jobs               = num_processors,
        max_modsize          = max_modsize,
        compute_interactions = compute_interactions,
        backend              = shap_backend,
        use_numba            = use_numba,
        blas_threads         = blas_threads,
        verbose              = (verbose >= 2),
    )
    runtimes["Selection"] = time.time() - t0

    stable_features = select_result["final_features"]
    final_vims      = select_result["final_vims"]
    stab_freq       = select_result["stability_freq"]
    shap_matrix     = np.asarray(select_result["shap_matrix"])
    pool_stability  = select_result.get("pool_stability", [])

    # ── lazy-interaction ingredients (interaction matrix computed on demand) ──
    interaction_ingredients = {
        "rf_stable":      select_result.get("_rf_stable"),
        "X_stable":       select_result.get("_X_stable"),
        "order":          select_result.get("_shap_order"),
        "classification": select_result.get("_classification", CLASSIFICATION),
        "backend":        select_result.get("_shap_backend", shap_backend),
    }

    n_stable = len(stable_features)
    if verbose >= 1:
        print(f"Stable features: {n_stable}")

    # ── stability_freq dict (only stable features) ────────────────────────────
    stability_freq_dict = dict(zip(stable_features, stab_freq))

    # ── stability_data DataFrame ──────────────────────────────────────────────
    stability_data = _build_stability_data(pool_stability)

    # ── selection_list ────────────────────────────────────────────────────────
    selection_list = [
        pd.DataFrame({
            "feature_name":        select_result["selection_list_feats"][i],
            "variable_importance": np.round(select_result["selection_list_vims"][i], 4),
        })
        for i in range(len(select_result["selection_list_feats"]))
    ]

    # ── final_SHAP DataFrame (with CIs + direction) ───────────────────────────
    # feature values aligned with the SHAP matrix columns (for the direction test)
    stable_value_mat = (
        X_arr[:, [feature_names.index(f) for f in stable_features]]
        if len(stable_features) > 0 else np.zeros((X_arr.shape[0], 0))
    )
    final_shap_df = _build_final_shap_df(
        stable_features  = stable_features,
        shap_matrix      = shap_matrix,
        feature_values   = stable_value_mat,
        final_vims       = final_vims,
        stab_freq        = stab_freq,
        module_membership_map = mm_map,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3: Final prediction RF (with final_ntree)
    # ══════════════════════════════════════════════════════════════════════════
    if verbose >= 1:
        print("Fitting final model ...")

    t0 = time.time()

    if n_stable == 0:
        final_rf = None
        final_X_df = pd.DataFrame()
        feature_list_df = pd.DataFrame(columns=["feature_name", "variable_importance",
                                                  "module_membership"])
    else:
        final_X_arr = X_arr[:, [feature_names.index(f) for f in stable_features]]
        final_X_df  = pd.DataFrame(final_X_arr, columns=stable_features)

        p_final   = n_stable
        mtry_fin  = _compute_mtry(p_final, select_params.mtry_factor, CLASSIFICATION,
                                  select_params.mtry_rule)
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if CLASSIFICATION else RandomForestRegressor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_rf = cls(
                n_estimators     = int(final_ntree),
                max_features     = mtry_fin,
                min_samples_leaf = int(nodesize),
                random_state     = seed,
                n_jobs           = num_processors,
            )
            y_fit = _map_labels(y_arr) if CLASSIFICATION else y_arr.astype(np.float64)
            final_rf.fit(final_X_arr, y_fit)

        feature_list_df = pd.DataFrame({
            "feature_name":        stable_features,
            "variable_importance": np.round(final_vims, 4),
            "module_membership":   [mm_map.get(f, ".") for f in stable_features],
        })

    runtimes["Final_RF"] = time.time() - t0

    # ── shap_obj (signed matrix; interaction matrix filled lazily) ────────────
    shap_obj = {
        "shapley_values":    shap_matrix,
        "interaction_matrix": None,   # populated on first access via result property
    }

    # ── module_membership DataFrame ───────────────────────────────────────────
    module_membership_df = pd.DataFrame({
        "feature_name": feature_names,
        "module":       module_assign,
    })

    if verbose >= 1:
        print("Done.")

    return ShapleyForestResult(
        final_rf          = final_rf,
        final_X           = final_X_df,
        module_membership = module_membership_df,
        survivor_list     = survivor_list,
        selection_list    = selection_list,
        final_SHAP        = final_shap_df,
        shap_obj          = shap_obj,
        runtimes          = runtimes,
        feature_list      = feature_list_df,
        stability_freq    = stability_freq_dict,
        stability_data    = stability_data,
        WGCNA_object      = None,
        _interaction_ingredients = interaction_ingredients,
    )


# ── wsf() — sf() with integrated WGCNA module detection ──────────────────────

def wsf(
    X,
    y,
    WGCNA_params: WGCNA_control | None  = None,
    screen_params: screen_control | None = None,
    select_params: select_control | None = None,
    final_ntree: int                     = 500,
    nodesize: int | None                 = None,
    num_processors: int                  = 1,
    max_modsize: int                     = 5000,
    auto_initial: str                    = "4",
    verbose: int                         = 1,
    seed: int | None                     = None,
    compute_interactions: bool           = True,
    shap_backend: str                    = "shap",
    use_numba: bool                      = False,
    blas_threads: int | None             = None,
    screen_cache=None,
) -> ShapleyForestResult:
    """
    Shapley Forest with WGCNA Module Detection.

    Runs PyWGCNA to derive module_membership, then calls sf().
    Matches R's wsf().

    Parameters
    ----------
    WGCNA_params : WGCNA_control
        WGCNA parameters. Default: WGCNA_control(power=6).
    All other parameters as in sf().

    Returns
    -------
    ShapleyForestResult with WGCNA_object populated.
    """
    if WGCNA_params is None:
        WGCNA_params = WGCNA_control()

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = X.values
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        feature_names = [f"X{i}" for i in range(X_arr.shape[1])]

    if verbose >= 1:
        print("Detecting modules via WGCNA ...")

    module_labels = detect_modules(
        X_arr,
        feature_names   = feature_names,
        r2_threshold    = 0.85,
        min_module_size = WGCNA_params.min_features,
        verbose         = (verbose >= 2),
        **WGCNA_params.extra_args,
    )

    module_membership = dict(zip(feature_names, module_labels))

    if verbose >= 1:
        unique_mods = len(set(module_labels))
        print(f"WGCNA complete: {unique_mods} modules detected.")

    result = sf(
        X                    = X,
        y                    = y,
        module_membership    = module_membership,
        screen_params        = screen_params,
        select_params        = select_params,
        final_ntree          = final_ntree,
        nodesize             = nodesize,
        num_processors       = num_processors,
        max_modsize          = max_modsize,
        auto_initial         = auto_initial,
        verbose              = verbose,
        seed                 = seed,
        compute_interactions = compute_interactions,
        shap_backend         = shap_backend,
        use_numba            = use_numba,
        blas_threads         = blas_threads,
        screen_cache         = screen_cache,
    )

    result.WGCNA_object = module_labels
    return result


# ── ShapleyForest — sklearn-compatible estimator ──────────────────────────────

class ShapleyForest(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper around sf().

    Suitable for use in sklearn pipelines. Module membership must be supplied
    at fit() time because it is data-dependent (not a hyperparameter).

    Parameters
    ----------
    screen_params : screen_control
        Phase 1 parameters. Default: screen_control().
    select_params : select_control
        Phase 2 parameters. Default: select_control().
    final_ntree : int
        Trees in the prediction RF. Default 500.
    nodesize : int or None
        Min terminal node size. None → 1 (classification) / 5 (regression).
    n_jobs : int
        Parallel workers. Default 1.
    max_modsize : int
        Blockwise pre-reduction trigger. Default 5000.
    auto_initial : str
        Checkpoint control ("4" = continue silently). Default "4".
    verbose : int
        Verbosity level 0/1/2. Default 1.
    random_state : int or None
        RNG seed. Default None (random).
    compute_interactions : bool
        Retain RF for lazy TreeSHAP interaction matrix. Default True.
    shap_backend : str
        "shap" (default) or "fasttreeshap".
    use_numba : bool
        Use Numba-JIT mean|SHAP| reduction when available. Default False.
    blas_threads : int or None
        Inner thread cap during parallel sections. None = auto. Default None.
    """

    def __init__(
        self,
        screen_params: screen_control | None = None,
        select_params: select_control | None = None,
        final_ntree: int                     = 500,
        nodesize: int | None                 = None,
        n_jobs: int                          = 1,
        max_modsize: int                     = 5000,
        auto_initial: str                    = "4",
        verbose: int                         = 1,
        random_state: int | None             = None,
        compute_interactions: bool           = True,
        shap_backend: str                    = "shap",
        use_numba: bool                      = False,
        blas_threads: int | None             = None,
        screen_cache=None,
    ) -> None:
        self.screen_params        = screen_params
        self.select_params        = select_params
        self.final_ntree          = final_ntree
        self.nodesize             = nodesize
        self.n_jobs               = n_jobs
        self.max_modsize          = max_modsize
        self.auto_initial         = auto_initial
        self.verbose              = verbose
        self.random_state         = random_state
        self.compute_interactions = compute_interactions
        self.shap_backend         = shap_backend
        self.use_numba            = use_numba
        self.blas_threads         = blas_threads
        self.screen_cache         = screen_cache

    def fit(self, X, y, module_membership=None) -> "ShapleyForest":
        """
        Run the two-phase algorithm.

        Parameters
        ----------
        X : array-like or pd.DataFrame
        y : array-like
        module_membership : dict, list, ndarray, or pd.Series
            Module assignment per feature. Required.
        """
        if module_membership is None:
            raise ValueError(
                "module_membership is required. Pass a dict {feature: module}, "
                "a list/array aligned with X columns, or a pd.Series."
            )

        self.result_ = sf(
            X                    = X,
            y                    = y,
            module_membership    = module_membership,
            screen_params        = self.screen_params,
            select_params        = self.select_params,
            final_ntree          = self.final_ntree,
            nodesize             = self.nodesize,
            num_processors       = self.n_jobs,
            max_modsize          = self.max_modsize,
            auto_initial         = self.auto_initial,
            verbose              = self.verbose,
            seed                 = self.random_state,
            compute_interactions = self.compute_interactions,
            shap_backend         = self.shap_backend,
            use_numba            = self.use_numba,
            blas_threads         = self.blas_threads,
            screen_cache         = self.screen_cache,
        )

        # sklearn-standard attributes
        self.selected_features_  = list(self.result_.final_X.columns)
        self.selected_indices_   = self._get_selected_indices(X)
        self.selection_freq_     = self.result_.stability_freq
        self.module_labels_      = self.result_.module_membership
        self.shap_values_        = self.result_.shap_obj["shapley_values"]
        self.final_rf_           = self.result_.final_rf

        return self

    def _get_selected_indices(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
        else:
            cols = [f"X{i}" for i in range(np.asarray(X).shape[1])]
        return np.array([cols.index(f) for f in self.selected_features_
                         if f in cols], dtype=int)

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "result_")
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_].values
        return np.asarray(X)[:, self.selected_indices_]

    def fit_transform(self, X, y, module_membership=None) -> np.ndarray:
        return self.fit(X, y, module_membership).transform(X)

    def predict(self, X) -> np.ndarray:
        """Predict using the final RF on the stable feature subset."""
        check_is_fitted(self, "result_")
        X_stable = self.transform(X)
        return self.result_.final_rf.predict(X_stable)

    def predict_proba(self, X) -> np.ndarray:
        """predict_proba using the final RF on the stable feature subset."""
        check_is_fitted(self, "result_")
        X_stable = self.transform(X)
        return self.result_.final_rf.predict_proba(X_stable)

    def __repr__(self) -> str:
        sc = self.screen_params or screen_control()
        sl = self.select_params or select_control()
        return (
            f"ShapleyForest("
            f"shadow_mode='{sl.shadow_mode}', "
            f"n_boots={sl.n_boots}, "
            f"pi_thr={sl.pi_thr}, "
            f"final_ntree={self.final_ntree}, "
            f"n_jobs={self.n_jobs})"
        )
