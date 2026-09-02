"""
cross-fit residualization (cfRes) — cross-fitted y residualization.

Matches the _cross_fit_residuals() function from mf_python_backend.py v18.

Unlike the old Python code (which residualized X), this matches R's cfRes:
  y_resid = y - RF_cv(X_corr)

After residualization, y_resid ≈ theta * X_ind + eps, boosting the SNR
for unassigned-pool shadow stability.
"""

from __future__ import annotations

import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor


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
    Robinson / cfRes cross-fitted residualization.

    Each observation's predicted value comes from a model trained on the
    other K-1 folds (no obs influences its own residual).

    Parameters
    ----------
    X_corr : (n, p_corr)  correlated-pool feature matrix
    y      : (n,)          original continuous or mapped-integer outcome
    k_folds : int           number of cross-fitting folds (default 5)
    ntree   : int           trees per fold-RF (default 300)
    mtry_factor : float     mtry multiplier (default 1.0)
    nodesize : int          min samples per leaf (default 5)
    seed    : int           RNG seed
    n_jobs  : int           joblib n_jobs for RF tree building

    Returns
    -------
    y_resid : (n,) float64   y - RF_hat(X_corr), cross-fitted
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
