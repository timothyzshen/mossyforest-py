"""
MossyForest

Primary API (matching R):
    mf():             two phase feature selection
    wmf():            mf() with integrated WGCNA module detection
    MossyForest:    sklearn compatible estimator wrapper

Parameter objects:
    screen_control:   Phase 1 (screening RFE) parameters
    select_control:   Phase 2 (shadow stability) parameters
    WGCNA_control:    WGCNA module detection parameters

Module detection:
    detect_modules:       run PyWGCNA on a feature matrix
    pick_soft_threshold:  WGCNA soft power selection

Plot functions:
    plot_importance, plot_waterfall, plot_force,
    plot_stability, plot_stability_elbow,
    plot_potential_interactions, plot_decisions,
    plot_modules, plot_elbow, plot_soft_threshold

Utilities:
    check_blas, check_numba
"""

from .core import mf, wmf, MossyForest
from .parameters import screen_control, select_control, WGCNA_control
from .result import MossyForestResult
from .modules import detect_modules, pick_soft_threshold
from .plots import (
    plot_importance,
    plot_signed_importance,
    plot_waterfall,
    plot_force,
    plot_stability,
    plot_stability_elbow,
    plot_potential_interactions,
    plot_decisions,
    plot_modules,
    plot_elbow,
    plot_soft_threshold,
)
from .utils import check_blas, check_numba
from .screen import clear_memory_cache

__all__ = [
    # main API
    "mf",
    "wmf",
    "MossyForest",
    # screening cache
    "clear_memory_cache",
    # parameter objects
    "screen_control",
    "select_control",
    "WGCNA_control",
    # result object
    "MossyForestResult",
    # module detection
    "detect_modules",
    "pick_soft_threshold",
    # plots
    "plot_importance",
    "plot_signed_importance",
    "plot_waterfall",
    "plot_force",
    "plot_stability",
    "plot_stability_elbow",
    "plot_potential_interactions",
    "plot_decisions",
    "plot_modules",
    "plot_elbow",
    "plot_soft_threshold",
    # utilities
    "check_blas",
    "check_numba",
]

__version__ = "0.1.0"
