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
        Multiplier applied to the mtry rule. Default 1.0.
    mtry_rule : str
        How mtry is derived from the feature count p.
        "auto"     (default) — p/3 for classification, sqrt(p) for regression
                   (the package's historical rule; preserves R parity).
        "sqrt"     — sqrt(p) for both. Much cheaper for classification, where
                   "auto" evaluates p/3 features per split.
        "p_over_3" — p/3 for both.
        NOTE: the standard RF convention is the reverse of "auto"
        (classification sqrt(p), regression p/3); "auto" is kept as the default
        so results and R parity are unchanged unless you opt in.
    min_ntree : int
        Minimum number of trees per screening forest. Default 500.
    ntree_factor : float
        Trees per forest = max(p * ntree_factor, min_ntree). Default 1.0.
        May be < 1 to reduce the tree count (e.g. 0.5 halves the p-scaled term).
    """
    drop_fraction: float = 0.25
    keep_fraction: float = 0.50
    mtry_factor: float = 1.0
    mtry_rule: str = "auto"
    min_ntree: int = 500
    ntree_factor: float = 1.0

    def __post_init__(self):
        if not 0 < self.drop_fraction < 1:
            raise ValueError("drop_fraction must be in (0, 1)")
        if not 0 < self.keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1]")
        if self.min_ntree < 1:
            raise ValueError("min_ntree must be >= 1")
        if self.mtry_rule not in ("auto", "sqrt", "p_over_3"):
            raise ValueError("mtry_rule must be 'auto', 'sqrt', or 'p_over_3'")


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
    mtry_rule : str
        "auto" (default) / "sqrt" / "p_over_3" — see screen_control.mtry_rule.
    mtry_on_real : bool
        How mtry is sized on the AUGMENTED [real | shadow] matrix (2p columns).
        False (default) — mtry = rule(2p), the historical behaviour.
        True — mtry = 2 * rule(p_real), so the expected number of REAL features
        offered per split matches an unaugmented forest (the shadows otherwise
        dilute real-feature sampling by ~half).
        NOTE: this is a STATISTICAL correction, not a speed optimisation — for
        the "sqrt" rule it is ~1.4x SLOWER (2*sqrt(p) > sqrt(2p)); for "p_over_3"
        it is identical (2*(p/3) == 2p/3).
    min_ntree : int
        Minimum trees per forest. Default 500.
    ntree_factor : float
        Trees per forest scaling factor. Default 1.0. May be < 1.
    shadow_mode : str
        "split"  — one corr pool + one ind pool evaluated separately.
        "within_module" — one shadow run per module. Default "split".
    n_boots : int
        Bootstrap replicates for shadow stability. Default 50.
    pi_thr : float
        Stability frequency threshold for the corr pool (used when
        threshold_mode="pi_thr"). Default 0.60.
    pi_thr_indep : float or None
        Threshold for the ind pool. None (default) = automatic elbow detection.
    threshold_mode : str
        How the per-pool selection cutoff is chosen.
        "pi_thr" (default) — correlated pool uses the fixed pi_thr; the
            independent pool uses elbow detection (or pi_thr_indep). This
            matches the R package.
        "elbow" — BOTH pools use the same adaptive elbow rule (unified
            selection philosophy; a Python-only extension not in the R
            package, so it breaks bit-exact R parity when enabled).
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
    mtry_rule: str = "auto"
    mtry_on_real: bool = False
    min_ntree: int = 500
    ntree_factor: float = 1.0
    shadow_mode: str = "split"
    n_boots: int = 50
    pi_thr: float = 0.60
    pi_thr_indep: Optional[float] = None
    threshold_mode: str = "pi_thr"
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
        if self.mtry_rule not in ("auto", "sqrt", "p_over_3"):
            raise ValueError("mtry_rule must be 'auto', 'sqrt', or 'p_over_3'")
        if self.ntree_factor <= 0:
            raise ValueError("ntree_factor must be > 0")
        if self.threshold_mode not in ("pi_thr", "elbow"):
            raise ValueError("threshold_mode must be 'pi_thr' or 'elbow'")
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
