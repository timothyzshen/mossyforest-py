"""
Parameter control objects for ShapleyForest.
Exact replicas of R's screen_control(), select_control(), and WGCNA_control().
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class screen_control:
    """
    Parameters for Phase 1 (per-module screening RFE).

    Matches R's screen_control().

    Parameters
    ----------
    drop_fraction : float
        Fraction of features eliminated per RFE step. Default 0.25.
    keep_fraction : float
        Fraction of features retained per module after screening. Default 0.50.
    mtry_factor : float
        Multiplier for mtry: ceil(factor * sqrt(p)) for regression,
        ceil(factor * p/3) for classification. Default 1.0.
    min_ntree : int
        Minimum number of trees per screening forest. Default 500.
    ntree_factor : int
        Trees per forest = max(p * ntree_factor, min_ntree). Default 1.
    """
    drop_fraction: float = 0.25
    keep_fraction: float = 0.50
    mtry_factor: float = 1.0
    min_ntree: int = 500
    ntree_factor: int = 1

    def __post_init__(self):
        if not 0 < self.drop_fraction < 1:
            raise ValueError("drop_fraction must be in (0, 1)")
        if not 0 < self.keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1]")
        if self.min_ntree < 1:
            raise ValueError("min_ntree must be >= 1")
        if self.ntree_factor < 1:
            raise ValueError("ntree_factor must be >= 1")


@dataclass
class select_control:
    """
    Parameters for Phase 2 (shadow-stability selection).

    Matches R's select_control().

    Parameters
    ----------
    drop_fraction : float
        Fraction of features dropped per RFE step during survivor-pool
        pre-reduction. Default 0.10.
    mtry_factor : float
        mtry multiplier for shadow-stability RFs. Default 1.0.
    min_ntree : int
        Minimum trees per forest. Default 500.
    ntree_factor : int
        Trees per forest scaling factor. Default 1.
    shadow_mode : str
        "split"  — one corr pool + one ind pool evaluated separately.
        "within_module" — one shadow run per module. Default "split".
    n_boots : int
        Bootstrap replicates for shadow stability. Default 50.
    pi_thr : float
        Stability frequency threshold for the corr pool. Default 0.60.
    pi_thr_indep : float or None
        Threshold for the ind pool. None (default) = automatic elbow detection.
    shadow_percentile : float
        Percentile of the shadow-feature distribution used as threshold
        within each bootstrap. Default 95.
    indep_modules : list of str or None
        Module names treated as the independent pool. None = no split-pool.
    use_dml_residual : bool
        If True, cross-fitted RF residualizes y before the ind-pool (or
        per-module) stability step. Default False.
    dml_n_folds : int
        Cross-fitting folds for DML residualization. Default 5.
    dml_ntree : int
        Trees per fold-RF in DML residualization. Default 300.
    dml_scope : str
        "indep" — DML applied only to the independent pool (default).
        "all_modules" — per-module DML applied to every module.
    early_stop_boots : bool
        If True, stop the bootstrap loop early once selection frequencies
        converge (changes results by design — fewer effective boots).
        Default False.
    early_stop_tol : float
        Convergence tolerance: stop when the max change in any feature's
        frequency between consecutive batches is below this. Default 0.01.
    early_stop_check_every : int
        Bootstraps per batch; convergence is checked between batches.
        Also the parallel granularity when early stopping is on. Default 10.
    """
    drop_fraction: float = 0.10
    mtry_factor: float = 1.0
    min_ntree: int = 500
    ntree_factor: int = 1
    shadow_mode: str = "split"
    n_boots: int = 50
    pi_thr: float = 0.60
    pi_thr_indep: Optional[float] = None
    shadow_percentile: float = 95.0
    indep_modules: Optional[List[str]] = None
    use_dml_residual: bool = False
    dml_n_folds: int = 5
    dml_ntree: int = 300
    dml_scope: str = "indep"
    early_stop_boots: bool = False
    early_stop_tol: float = 0.01
    early_stop_check_every: int = 10

    def __post_init__(self):
        if self.shadow_mode not in ("split", "within_module"):
            raise ValueError("shadow_mode must be 'split' or 'within_module'")
        if self.dml_scope not in ("indep", "all_modules"):
            raise ValueError("dml_scope must be 'indep' or 'all_modules'")
        if not 0 <= self.pi_thr <= 1:
            raise ValueError("pi_thr must be in [0, 1]")
        if self.pi_thr_indep is not None and not 0 <= self.pi_thr_indep <= 1:
            raise ValueError("pi_thr_indep must be in [0, 1]")
        if not 0 < self.shadow_percentile <= 100:
            raise ValueError("shadow_percentile must be in (0, 100]")
        if self.n_boots < 1:
            raise ValueError("n_boots must be >= 1")
        if self.dml_n_folds < 2:
            raise ValueError("dml_n_folds must be >= 2")
        if self.early_stop_tol <= 0:
            raise ValueError("early_stop_tol must be > 0")
        if self.early_stop_check_every < 1:
            raise ValueError("early_stop_check_every must be >= 1")


@dataclass
class WGCNA_control:
    """
    Parameters for WGCNA module detection.

    Matches R's WGCNA_control().

    Parameters
    ----------
    power : int
        Soft-thresholding power for blockwiseModules. Default 6.
    min_features : int
        Minimum number of features per module. Default 20.
    extra_args : dict
        Additional keyword arguments forwarded to PyWGCNA.
    """
    power: int = 6
    min_features: int = 20
    extra_args: dict = field(default_factory=dict)
