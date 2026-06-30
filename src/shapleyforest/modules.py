"""
Module detection via PyWGCNA.

Wraps the real PyWGCNA API to return a simple array of module labels
aligned with the columns of X.  The "grey" module (unassigned genes)
is relabelled "independent" to match the R package convention.

PyWGCNA API reference:
    https://github.com/mortazavilab/PyWGCNA
    Constructor: WGCNA(name, geneExp, minModuleSize, MEDissThres,
                       RsquaredCut, MeanCut, networkType, TOMType,
                       TPMcutoff, save, ...)
    Static:      WGCNA.pickSoftThreshold(data, powerVector, RsquaredCut, ...)
                 returns (powerEstimate, datout_DataFrame)
    Run:         wgcna.runWGCNA()
    Labels:      wgcna.datExpr.var['moduleColors']  (after runWGCNA)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_modules(
    X: np.ndarray,
    feature_names: list[str] | None = None,
    r2_threshold: float = 0.85,
    mean_cut: float = 100.0,
    min_module_size: int = 30,
    merge_cut_height: float = 0.25,
    network_type: str = "signed hybrid",
    tom_type: str = "signed",
    verbose: bool = False,
) -> np.ndarray:
    """
    Run PyWGCNA module detection on feature matrix X.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Expression / feature matrix — rows are samples, columns are features.
    feature_names : list of str, optional
        Column names for X. Defaults to ["X0", "X1", ...].
    r2_threshold : float
        Minimum scale-free topology R² for automatic soft-power selection.
        PyWGCNA picks the lowest power whose R² >= r2_threshold.
    mean_cut : float
        Maximum mean connectivity allowed during soft-power selection.
    min_module_size : int
        Minimum number of features per module.
    merge_cut_height : float
        Dissimilarity threshold for merging similar modules (MEDissThres).
    network_type : str
        One of "signed hybrid", "signed", "unsigned".
    tom_type : str
        One of "signed", "unsigned".
    verbose : bool
        Print PyWGCNA progress messages.

    Returns
    -------
    labels : np.ndarray of shape (n_features,), dtype str
        Module name per feature. Unassigned features → "independent".
    """
    try:
        import PyWGCNA
    except ImportError as e:
        raise ImportError(
            "PyWGCNA is required for module detection.\n"
            "Install with: pip install PyWGCNA"
        ) from e

    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]

    # PyWGCNA expects genes × samples (rows = genes/features, cols = samples)
    sample_names = [f"S{i}" for i in range(n_samples)]
    gene_expr_df = pd.DataFrame(
        X.T,
        index=feature_names,
        columns=sample_names,
    )

    wgcna = PyWGCNA.WGCNA(
        name="shapleyforest",
        geneExp=gene_expr_df,
        TPMcutoff=0,            # data already preprocessed — no TPM filter
        RsquaredCut=r2_threshold,
        MeanCut=mean_cut,
        minModuleSize=min_module_size,
        MEDissThres=merge_cut_height,
        networkType=network_type,
        TOMType=tom_type,
        naColor="grey",
        save=False,
    )

    if not verbose:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wgcna.runWGCNA()
    else:
        wgcna.runWGCNA()

    # Module labels live in datExpr.var after runWGCNA()
    raw_labels = wgcna.datExpr.var["moduleColors"].values.astype(str)

    # Recode grey (unassigned) → "independent"
    labels = np.where(raw_labels == "grey", "independent", raw_labels)
    return labels


def pick_soft_threshold(
    X: np.ndarray,
    feature_names: list[str] | None = None,
    powers: list[int] | None = None,
    r2_threshold: float = 0.85,
    network_type: str = "signed hybrid",
) -> tuple[int, pd.DataFrame]:
    """
    Run PyWGCNA's soft-threshold selection and return the chosen power
    plus the full fit table (for plotting).

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    powers : list of candidate powers; defaults to 1–20
    r2_threshold : float
        R² target; passed to PyWGCNA as RsquaredCut.
    network_type : str

    Returns
    -------
    power_estimate : int
        Chosen soft-threshold power.
    datout : pd.DataFrame
        Full fit table — columns include "Power", "SFT.R.sq", "mean.k."
    """
    try:
        import PyWGCNA
    except ImportError as e:
        raise ImportError("Install PyWGCNA with: pip install PyWGCNA") from e

    if powers is None:
        powers = list(range(1, 21))

    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]

    gene_expr_df = pd.DataFrame(
        X.T,
        index=feature_names,
        columns=[f"S{i}" for i in range(n_samples)],
    )

    power_estimate, datout = PyWGCNA.WGCNA.pickSoftThreshold(
        data=gene_expr_df,
        dataIsExpr=True,
        powerVector=powers,
        RsquaredCut=r2_threshold,
        networkType=network_type,
    )

    return int(power_estimate), datout
