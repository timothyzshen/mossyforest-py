"""
RFE loop, blockwise reduction, and shared RF/SHAP helpers.

All helpers here match the Python backend (sf_python_backend.py v18) used
by the R package.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ── Optional Numba acceleration ───────────────────────────────────────────────
try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _mean_abs_axis0_numba(arr):
        """Mean absolute value along axis 0 — parallel over columns."""
        n, p = arr.shape
        out = np.empty(p, dtype=np.float64)
        for j in prange(p):
            s = 0.0
            for i in range(n):
                v = arr[i, j]
                s += v if v >= 0.0 else -v
            out[j] = s / n
        return out

    _HAS_NUMBA = True
except Exception:   # pragma: no cover - numba optional
    _HAS_NUMBA = False

    def _mean_abs_axis0_numba(arr):   # type: ignore[misc]
        return np.mean(np.abs(arr), axis=0)


def _mean_abs_axis0(arr: np.ndarray, use_numba: bool = False) -> np.ndarray:
    """Mean |value| along axis 0. NumPy by default; Numba when requested+available."""
    if use_numba and _HAS_NUMBA:
        return _mean_abs_axis0_numba(np.ascontiguousarray(arr, dtype=np.float64))
    return np.mean(np.abs(arr), axis=0)


# ── SHAP backend factory ──────────────────────────────────────────────────────

def _tree_explainer(rf, backend: str = "shap"):
    """
    Return a TreeSHAP explainer from the requested backend.

    backend="shap"          → shap.TreeExplainer (default).
    backend="fasttreeshap"  → fasttreeshap.TreeExplainer in exact mode
                              (algorithm="v2"), which returns identical values
                              to shap but is faster. Requires fasttreeshap.
    """
    if backend == "fasttreeshap":
        try:
            import fasttreeshap
        except ImportError as e:   # pragma: no cover - optional dep
            raise ImportError(
                "shap_backend='fasttreeshap' requires the fasttreeshap package.\n"
                "Install with: pip install fasttreeshap   (or pip install shapleyforest[fast])"
            ) from e
        return fasttreeshap.TreeExplainer(rf, algorithm="v2", n_jobs=1)
    import shap as _shap
    return _shap.TreeExplainer(rf)


# ── RF / mtry / ntree helpers ─────────────────────────────────────────────────

def _compute_mtry(n_features: int, mtry_factor: float, classification: bool) -> int:
    if classification:
        return min(max(1, math.ceil(mtry_factor * n_features / 3)), n_features)
    else:
        return min(max(1, math.ceil(mtry_factor * math.sqrt(n_features))), n_features)


def _compute_ntree(n_features: int, ntree_factor: float, min_ntree: int) -> int:
    return max(int(n_features * ntree_factor), int(min_ntree))


def _build_rf(
    n_estimators: int,
    max_features: int,
    min_samples_leaf: int,
    random_state: int,
    n_jobs: int,
    classification: bool,
):
    max_features = max(1, int(max_features))
    kw = dict(
        n_estimators=int(n_estimators),
        max_features=max_features,
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )
    return RandomForestClassifier(**kw) if classification else RandomForestRegressor(**kw)


def _map_labels(y: np.ndarray) -> np.ndarray:
    """Contiguous integer label encoding (0, 1, ...)."""
    uniq = sorted(np.unique(y))
    lmap = {v: i for i, v in enumerate(uniq)}
    return np.array([lmap[v] for v in y], dtype=int)


# ── SHAP helpers ──────────────────────────────────────────────────────────────

def shap_mean_abs(
    rf, X: np.ndarray, classification: bool,
    backend: str = "shap", use_numba: bool = False,
) -> np.ndarray:
    """
    Mean absolute SHAP values — shape (n_features,).

    Handles both the legacy list format and modern ndarray format from shap.
    For binary classification: class-1 SHAP. For multiclass: mean |SHAP| across classes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = _tree_explainer(rf, backend).shap_values(X)

    if isinstance(sv, list):
        if classification and len(rf.classes_) == 2:
            return _mean_abs_axis0(np.asarray(sv[1]), use_numba)
        elif classification:
            return np.mean([_mean_abs_axis0(np.asarray(s), use_numba) for s in sv], axis=0)
        else:
            return _mean_abs_axis0(np.asarray(sv[0]), use_numba)
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            if classification and arr.shape[2] == 2:
                return _mean_abs_axis0(arr[:, :, 1], use_numba)
            elif classification:
                return np.mean(np.abs(arr), axis=(0, 2))
            else:
                return _mean_abs_axis0(arr[:, :, 0], use_numba)
        else:
            return _mean_abs_axis0(arr, use_numba)


def shap_full_signed(
    rf, X: np.ndarray, classification: bool, backend: str = "shap",
) -> np.ndarray:
    """
    Signed SHAP matrix — shape (n_samples, n_features).

    For binary classification: class-1 SHAP values.
    For multiclass: mean across classes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = _tree_explainer(rf, backend).shap_values(X)

    if isinstance(sv, list):
        if classification and len(rf.classes_) == 2:
            return np.asarray(sv[1])
        elif classification:
            return np.mean([np.asarray(s) for s in sv], axis=0)
        else:
            return np.asarray(sv[0])
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            if classification and arr.shape[2] == 2:
                return arr[:, :, 1]
            elif classification:
                return np.mean(arr, axis=2)
            else:
                return arr[:, :, 0]
        else:
            return arr


def shap_interaction_matrix(
    rf, X: np.ndarray, classification: bool, backend: str = "shap",
) -> np.ndarray:
    """
    p×p mean absolute TreeSHAP interaction matrix.

    Diagonal = mean |main effect|. Off-diagonal = mean |pairwise interaction|.
    Matrix is symmetric and non-negative.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iv_raw = _tree_explainer(rf, backend).shap_interaction_values(X)

    if isinstance(iv_raw, list):
        if classification and len(rf.classes_) == 2:
            iv = np.asarray(iv_raw[1])
        elif classification:
            iv = np.mean([np.abs(np.asarray(s)) for s in iv_raw], axis=0)
        else:
            iv = np.asarray(iv_raw[0])
    else:
        arr = np.asarray(iv_raw)
        if arr.ndim == 4:
            if classification and arr.shape[3] == 2:
                iv = arr[:, :, :, 1]
            elif classification:
                iv = np.mean(np.abs(arr), axis=3)
            else:
                iv = arr[:, :, :, 0]
        else:
            iv = arr

    return np.mean(np.abs(iv), axis=0)


# ── RFE loop ──────────────────────────────────────────────────────────────────

def rfe_loop(
    X_mod: np.ndarray,
    feat_names: list,
    y: np.ndarray,
    target: int,
    drop_fraction: float,
    mtry_factor: float,
    ntree_factor: float,
    min_ntree: int,
    nodesize: int,
    classification: bool,
    seed: int,
    n_jobs: int,
    use_shap: bool = True,
    backend: str = "shap",
    use_numba: bool = False,
) -> tuple:
    """
    Recursive feature elimination until `target` features remain.

    At each step: fit RF → compute mean|SHAP| → drop top drop_fraction
    weakest features. Repeats until exactly `target` features remain.

    Returns
    -------
    survivors : list of str
    survivor_vims : list of float
    initial_info : dict   (feature, vim, survivor at first step)
    all_steps : list of (names_sorted, vims_sorted) per step
    """
    feat_names = list(feat_names)
    current_X = np.ascontiguousarray(X_mod, dtype=np.float32)
    current_names = list(feat_names)
    is_initial = True
    initial_info = None
    survivors = None
    survivor_vims = None
    all_steps = []

    while len(current_names) >= target:
        n_cur = len(current_names)
        mtry  = _compute_mtry(n_cur, mtry_factor, classification)
        ntree = _compute_ntree(n_cur, ntree_factor, min_ntree)

        rf = _build_rf(ntree, mtry, nodesize, seed, n_jobs, classification)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf.fit(
                current_X,
                y.astype(int) if classification else y.astype(np.float64),
            )

        if use_shap:
            try:
                vim = shap_mean_abs(rf, current_X, classification,
                                    backend=backend, use_numba=use_numba)
            except Exception:
                vim = rf.feature_importances_
        else:
            vim = rf.feature_importances_

        order        = np.argsort(vim)[::-1]
        vim_sorted   = vim[order]
        names_sorted = [current_names[i] for i in order]
        reduction    = math.ceil(n_cur * drop_fraction)

        all_steps.append((list(names_sorted), vim_sorted.tolist()))

        if is_initial:
            n_keep_init   = max(n_cur - reduction, target)
            survived_init = [True] * n_keep_init + [False] * (n_cur - n_keep_init)
            initial_info  = {
                "feature":  names_sorted,
                "vim":      vim_sorted.tolist(),
                "survivor": survived_init,
            }
            is_initial = False

        if n_cur - reduction > target:
            keep_n        = n_cur - reduction
            keep_idx      = list(order[:keep_n])
            current_X     = current_X[:, keep_idx]
            current_names = names_sorted[:keep_n]
        else:
            survivors     = names_sorted[:target]
            survivor_vims = vim_sorted[:target].tolist()
            break

    if survivors is None:
        survivors     = current_names[:target]
        survivor_vims = [0.0] * len(survivors)

    return survivors, survivor_vims, initial_info, all_steps


# ── Blockwise pre-reduction ───────────────────────────────────────────────────

def blockwise_reduce(
    X_mod: np.ndarray,
    feat_names: list,
    y: np.ndarray,
    max_block_size: int,
    keep_fraction: float,
    drop_fraction: float,
    mtry_factor: float,
    ntree_factor: float,
    min_ntree: int,
    nodesize: int,
    classification: bool,
    seed: int,
    n_jobs: int,
    use_shap: bool = True,
    backend: str = "shap",
    use_numba: bool = False,
) -> tuple:
    """
    Iteratively partition a large module into random blocks and run
    rfe_loop on each block to reduce to keep_fraction features, until
    the total module size <= max_block_size.
    """
    feat_names = list(feat_names)
    rng = np.random.default_rng(seed)

    while X_mod.shape[1] > max_block_size:
        n_feat   = X_mod.shape[1]
        perm     = rng.permutation(n_feat)
        n_blocks = math.ceil(n_feat / max_block_size)
        blocks   = np.array_split(perm, n_blocks)

        kept_cols = []
        for block in blocks:
            X_blk      = X_mod[:, block]
            names_blk  = [feat_names[i] for i in block]
            n_blk      = len(block)
            target_blk = max(1, math.ceil(n_blk * keep_fraction))
            surv_b, _, _, _ = rfe_loop(
                X_blk, names_blk, y, target_blk,
                drop_fraction, mtry_factor, ntree_factor, min_ntree,
                nodesize, classification, seed, n_jobs, use_shap,
                backend=backend, use_numba=use_numba,
            )
            name_to_orig = {feat_names[i]: i for i in block}
            kept_cols.extend([name_to_orig[f] for f in surv_b])

        kept_cols  = sorted(set(kept_cols))
        X_mod      = X_mod[:, kept_cols]
        feat_names = [feat_names[i] for i in kept_cols]

    return X_mod, feat_names
