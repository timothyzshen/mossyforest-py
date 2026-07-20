"""
Shadow (Boruta-style) stability selection.

Per-bootstrap algorithm:
  1. Subsample n//2 rows.
  2. Permute each real column independently → X_shadow.
  3. X_aug = [X_real | X_shadow]  (n//2 × 2p).
  4. Fit RF; compute mean|SHAP| on X_aug.
  5. shadow_threshold = percentile(mean|SHAP|[p:], shadow_percentile)
     (only the p shadow columns contribute to the null).
  6. selected_b = { j : mean|SHAP|[j] > shadow_threshold }

stability_freq(j) = |selected_b| / n_valid_boots

Thresholding:
  corr pool: stable = { j : freq(j) >= pi_thr }
  ind  pool: elbow detection (or pi_thr_indep if given)

Matches _boruta_shadow_stability() from sf_python_backend.py v18.
"""

from __future__ import annotations

import warnings
import numpy as np
from joblib import Parallel, delayed

from .rfe import (
    _build_rf,
    _compute_mtry,
    _compute_ntree,
    shap_mean_abs,
)
from .threads import thread_limit_ctx


# ── Shadow matrix ─────────────────────────────────────────────────────────────

def make_shadow_matrix(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Return X_aug = [X | X_shadow] where each shadow column is an independent
    within-column permutation of the corresponding column of X.

    Uses rng.permuted(axis=0) — a single C-level call that shuffles each
    column independently (replaces the per-column Python loop).
    """
    X_shadow = rng.permuted(X, axis=0)
    return np.concatenate([X, X_shadow], axis=1)


# ── Elbow detection ───────────────────────────────────────────────────────────

def find_elbow_idx(freqs_sorted: list | np.ndarray) -> int:
    """
    Perpendicular-distance elbow on a descending frequency array.

    Normalises both axes to [0, 1], draws the chord from the first to the
    last point, and returns the 0-based index of the point furthest from
    that chord (the elbow).

    Matches _find_elbow_idx() in sf_python_backend.py v18.
    """
    freqs = np.asarray(freqs_sorted, dtype=np.float64)
    n     = len(freqs)
    if n <= 1:
        return 0
    rng_val = freqs[0] - freqs[-1]
    if rng_val < 1e-10:
        return 0
    x  = np.linspace(0.0, 1.0, n)
    yr = (freqs - freqs[-1]) / rng_val
    dx, dy = x[-1] - x[0], yr[-1] - yr[0]   # = 1 and -1
    dists  = np.abs(dy * (x - x[0]) - dx * (yr - yr[0])) / np.sqrt(dx**2 + dy**2)
    return int(np.argmax(dists))


def apply_stable_threshold(
    names_sorted: list,
    freqs_sorted: list,
    pi_thr: float,
    use_elbow: bool = False,
    label: str = "pool",
) -> tuple:
    """
    Apply stable-set threshold to shadow-stability outputs.

    use_elbow=False  (corr pool):
        Standard threshold: stable = { j : freq(j) >= pi_thr }.
    use_elbow=True   (ind pool):
        Elbow detection: stable = { j : freq(j) >= freqs[elbow_idx] }.

    Returns
    -------
    stable      : list[str]   selected feature names
    threshold   : float       frequency cutoff applied
    elbow_rank  : int or None 1-based elbow rank (use_elbow only)
    """
    if not names_sorted:
        return [], 0.0, None

    freqs = np.asarray(freqs_sorted, dtype=np.float64)

    if use_elbow:
        elbow_idx  = find_elbow_idx(freqs_sorted)
        elbow_freq = float(freqs[elbow_idx])
        stable     = [f for f, fr in zip(names_sorted, freqs_sorted)
                      if fr >= elbow_freq and fr > 0.0]
        return stable, elbow_freq, int(elbow_idx) + 1   # 1-based rank
    else:
        thr    = float(pi_thr)
        stable = [f for f, fr in zip(names_sorted, freqs_sorted) if fr >= thr]
        return stable, thr, None


def bh_fdr_select(pvals, q: float, method: str = "bh") -> np.ndarray:
    """
    FDR selection at level q.

    method="bh" — Benjamini–Hochberg: reject up to the last p_(i) <= q·i/m.
                  Valid under independence / PRDS (positive dependence).
    method="by" — Benjamini–Yekutieli: same, with q scaled by 1/Σ(1/i). Valid
                  under ARBITRARY dependence (more conservative). Use when the
                  correlated modules make positive dependence unsafe to assume.

    Returns a boolean mask aligned with `pvals`.
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    qq = q
    if method == "by":
        qq = q / float(np.sum(1.0 / np.arange(1, m + 1)))
    order  = np.argsort(p)
    ranked = p[order]
    line   = qq * (np.arange(1, m + 1) / m)
    below  = ranked <= line
    if not below.any():
        return np.zeros(m, dtype=bool)
    cutoff = ranked[np.max(np.where(below)[0])]
    return p <= cutoff


# ── One bootstrap round ───────────────────────────────────────────────────────

def _one_boot(
    boot_seed: int,
    X_pool: np.ndarray,
    y: np.ndarray,
    p: int,
    n: int,
    mtry_factor: float,
    ntree_factor: float,
    min_ntree: int,
    nodesize: int,
    classification: bool,
    use_shap: bool,
    shadow_percentile: float,
    backend: str,
    use_numba: bool,
):
    """
    Run one shadow bootstrap. Returns (mask, vim_real, vim_shadow) where
    mask is the boolean (p,) selection (real importance > shadow percentile)
    and vim_real / vim_shadow are the length-p continuous importances — or
    None if the round was skipped (degenerate label subsample / fit failure).

    Shadow column k is a permutation of real column k, so vim_shadow[k] is the
    matched noise-null for feature k (used for the FDR margin test downstream).

    Fully determined by `boot_seed` so the result is independent of execution
    order (and therefore of the number of parallel workers).
    """
    rng = np.random.default_rng(int(boot_seed))

    idx        = rng.choice(n, size=n // 2, replace=False)
    X_real_sub = X_pool[idx]
    y_sub      = y[idx]

    if classification and len(np.unique(y_sub)) < 2:
        return None

    # shadow: each column independently permuted (C-level), preallocated buffer
    n_sub = X_real_sub.shape[0]
    X_aug = np.empty((n_sub, 2 * p), dtype=np.float32)
    X_aug[:, :p] = X_real_sub
    X_aug[:, p:] = rng.permuted(X_real_sub, axis=0)

    n_aug    = 2 * p
    mtry_aug = _compute_mtry(n_aug, mtry_factor, classification)
    ntree    = _compute_ntree(n_aug, ntree_factor, min_ntree)

    rf = _build_rf(ntree, mtry_aug, nodesize, int(boot_seed), 1, classification)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf.fit(
                X_aug,
                y_sub.astype(int) if classification else y_sub.astype(np.float64),
            )
    except Exception:
        return None

    if use_shap:
        try:
            vim_aug = shap_mean_abs(rf, X_aug, classification,
                                    backend=backend, use_numba=use_numba)
        except Exception:
            vim_aug = rf.feature_importances_
    else:
        vim_aug = rf.feature_importances_

    vim_real   = np.asarray(vim_aug[:p], dtype=np.float64)
    vim_shadow = np.asarray(vim_aug[p:], dtype=np.float64)

    threshold = np.percentile(vim_shadow, float(shadow_percentile))
    mask = vim_real > threshold
    return mask, vim_real, vim_shadow


# ── Boruta shadow stability ───────────────────────────────────────────────────

def boruta_shadow_stability(
    X_pool: np.ndarray,
    feat_names: list,
    y: np.ndarray,
    n_boots: int,
    mtry_factor: float,
    ntree_factor: float,
    min_ntree: int,
    nodesize: int,
    classification: bool,
    seed: int,
    n_jobs: int,
    use_shap: bool = True,
    shadow_percentile: float = 95.0,
    backend: str = "shap",
    use_numba: bool = False,
    blas_threads=None,
    early_stop_boots: bool = False,
    early_stop_tol: float = 0.01,
    early_stop_check_every: int = 10,
) -> tuple:
    """
    Shadow-feature bootstrap stability selection on a single feature pool.

    Each bootstrap gets a deterministic seed derived from `seed` via
    SeedSequence, so results are independent of the number of parallel workers
    (n_jobs=1 and n_jobs=8 produce identical output).

    When early_stop_boots is True, bootstraps run in batches of
    early_stop_check_every and the loop stops once the maximum change in any
    feature's frequency between batches falls below early_stop_tol.

    Returns
    -------
    names_sorted : list[str]   all pool features sorted by freq descending
    freqs_sorted : list[float] parallel selection frequencies in [0, 1]
    pvals_sorted : list[float] importance p-value per feature (threshold_mode
                   "fdr"): mean real importance vs the shadow-mean null, Gaussian
                   tail (Altmann et al. 2010).
    signtest_pvals_sorted : list[float] own-shadow sign-test p-value per feature
                   (threshold_mode "signtest"): win-rate vs Binomial(n, 0.5).
                   Same null (0.5) for every feature → common scale across pools.
    """
    feat_names = list(feat_names)
    p = X_pool.shape[1]
    n = X_pool.shape[0]

    if p == 0:
        return [], [], [], []

    X_pool = np.ascontiguousarray(X_pool, dtype=np.float32)

    # deterministic per-bootstrap seeds (order-independent)
    boot_seeds = np.random.SeedSequence(int(seed)).generate_state(int(n_boots))

    counts     = np.zeros(p, dtype=np.float64)   # selection counts → frequency
    sum_real   = np.zeros(p, dtype=np.float64)   # Σ real importance   → mean
    sum_shadow = np.zeros(p, dtype=np.float64)   # Σ shadow importance → null mean
    win_counts = np.zeros(p, dtype=np.float64)   # # bootstraps feature beat own shadow
    n_valid    = 0
    limit      = blas_threads

    def _run_batch(seeds_batch):
        if n_jobs == 1:
            with thread_limit_ctx(limit):
                return [
                    _one_boot(s, X_pool, y, p, n, mtry_factor, ntree_factor,
                              min_ntree, nodesize, classification, use_shap,
                              shadow_percentile, backend, use_numba)
                    for s in seeds_batch
                ]
        with thread_limit_ctx(limit):
            return Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_one_boot)(
                    s, X_pool, y, p, n, mtry_factor, ntree_factor,
                    min_ntree, nodesize, classification, use_shap,
                    shadow_percentile, backend, use_numba)
                for s in seeds_batch
            )

    def _accumulate(res):
        nonlocal n_valid
        if res is None:
            return
        mask, vr, vs = res
        counts[mask]   += 1.0
        sum_real[:]     += vr
        sum_shadow[:]   += vs
        win_counts[:]   += (vr > vs)   # own-shadow win (sign test)
        n_valid          += 1

    if not early_stop_boots:
        for res in _run_batch(boot_seeds):
            _accumulate(res)
    else:
        prev_freqs = None
        step = max(1, int(early_stop_check_every))
        for start in range(0, len(boot_seeds), step):
            batch = boot_seeds[start:start + step]
            for res in _run_batch(batch):
                _accumulate(res)
            if n_valid == 0:
                continue
            cur_freqs = counts / n_valid
            if prev_freqs is not None:
                if np.max(np.abs(cur_freqs - prev_freqs)) < float(early_stop_tol):
                    break
            prev_freqs = cur_freqs

    if n_valid == 0:
        return feat_names, [0.0] * p, [1.0] * p, [1.0] * p

    freqs = counts / n_valid

    # ── sign-test p-value: own-shadow win-rate vs Binomial(n_valid, 0.5) ───────
    # Each bootstrap, feature j races its OWN matched shadow (a permutation of
    # itself) — a coin flip under the null, for EVERY feature regardless of pool.
    # So win_counts ~ Binomial(n_valid, 0.5) under H0, and the one-sided p-value
    # p_j = P(X >= win_counts[j]) puts correlated and independent features on one
    # common scale. This is the "level playing field" statistic.
    from scipy.stats import binom
    signtest_pvals = binom.sf(win_counts - 1, int(n_valid), 0.5)

    # ── p-value: mean real importance vs the matched shadow-null distribution ──
    # Compare each feature's mean importance S_j = sum_real[j]/n_valid to the
    # distribution of shadow mean importances {N_k = sum_shadow[k]/n_valid}
    # (same aggregation → like-for-like). Altmann et al. (2010): use a PARAMETRIC
    # (Gaussian) tail so the p-value is not floored at 1/(p+1) — critical for
    # small pools (e.g. the independent pool), where an empirical p-value can
    # never clear the BH line. Falls back to the empirical p-value if the null
    # has no spread.
    mean_real = sum_real / n_valid
    ms        = sum_shadow / n_valid
    null_mu   = float(ms.mean())
    null_sd   = float(ms.std(ddof=1)) if p > 1 else 0.0
    if null_sd > 1e-12:
        from scipy.stats import norm
        pvals = norm.sf((mean_real - null_mu) / null_sd)   # P(null >= observed)
    else:
        ms_sorted = np.sort(ms)
        n_ge  = p - np.searchsorted(ms_sorted, mean_real, side="left")
        pvals = (n_ge + 1.0) / (p + 1.0)
    pvals = np.clip(pvals, 1e-12, 1.0)

    order = np.argsort(freqs)[::-1]
    return (
        [feat_names[i] for i in order],
        freqs[order].tolist(),
        pvals[order].tolist(),
        signtest_pvals[order].tolist(),
    )
