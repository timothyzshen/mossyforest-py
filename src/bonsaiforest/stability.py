"""
StabilityTracker — backward-compatible helper.

The shadow-stability counting now lives in shadow.py (boruta_shadow_stability).
This class is retained for backward compatibility with existing code that
imported it directly.
"""

from __future__ import annotations

import numpy as np


class StabilityTracker:
    """
    Accumulate per-feature selection counts across iterations.

    Parameters
    ----------
    n_features : int
        Number of candidate features being tracked.
    """

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.counts     = np.zeros(n_features, dtype=int)
        self.n_iter     = 0

    def update(self, selected: np.ndarray) -> None:
        """
        Parameters
        ----------
        selected : boolean array of shape (n_features,) or integer index array
        """
        self.counts[selected] += 1
        self.n_iter += 1

    @property
    def frequencies(self) -> np.ndarray:
        """Selection frequency in [0, 1] for each feature."""
        if self.n_iter == 0:
            return np.zeros(self.n_features)
        return self.counts / self.n_iter

    def stable_set(self, pi_thr: float = 0.6) -> np.ndarray:
        """Integer indices of features selected in >= pi_thr fraction of iters."""
        return np.where(self.frequencies >= pi_thr)[0]

    def reset(self) -> None:
        self.counts[:] = 0
        self.n_iter    = 0
