"""
MossyForestResult: the return object from mf() / wmf().

Matches the R mossy_forest S3 class in structure and methods:
  - print()          maps to __repr__ / __str__
  - predict()        maps to predict(new_X)
  - detect_interaction()
  - All plot methods (see plots.py)

Attributes mirror R's mossy_forest list slots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MossyForestResult:
    """
    Result object returned by mf() and wmf().

    Attributes
    ----------
    final_rf : sklearn RF
        Final RF fitted on the stable feature set (for prediction).
    final_X : pd.DataFrame
        Feature matrix restricted to stable features.
    module_membership : pd.DataFrame
        Two column DataFrame: feature_name, module.
    WGCNA_object : object or None
        Raw PyWGCNA output (wmf only), else None.
    survivor_list : dict
        Per module screening survivors {module: DataFrame(features, vims)}.
    selection_list : list of pd.DataFrame
        Per pool feature/frequency tables.
    final_SHAP : pd.DataFrame
        Stable features with mean|SHAP|, 95% CIs, stability frequencies.
        Columns: feature_name, variable_importance, module_membership,
                 mean_abs_shap, se_abs_shap, ci_lower_abs, ci_upper_abs,
                 mean_signed_shap, se_signed_shap, ci_lower_signed,
                 ci_upper_signed, stability_freq.
    shap_obj : dict
        {"shapley_values": ndarray (n, p), "interaction_matrix": ndarray (p, p) or None}
    runtimes : dict
        {"Screen": float, "Selection": float, "Final_RF": float}  (seconds)
    feature_list : pd.DataFrame
        Stable features with feature_name, variable_importance, module_membership.
    stability_freq : dict
        {feature_name: selection_frequency}
    stability_data : pd.DataFrame or None
        Per pool stability curves for plot_stability_elbow().
        Columns: pool, rank, feature, freq, selected, threshold, elbow_rank.
    """

    def __init__(
        self,
        final_rf,
        final_X: pd.DataFrame,
        module_membership: pd.DataFrame,
        survivor_list: dict,
        selection_list: list,
        final_SHAP: pd.DataFrame,
        shap_obj: dict,
        runtimes: dict,
        feature_list: pd.DataFrame | None = None,
        stability_freq: dict | None = None,
        stability_data: pd.DataFrame | None = None,
        WGCNA_object=None,
        _interaction_ingredients: dict | None = None,
    ):
        self.final_rf         = final_rf
        self.final_X          = final_X
        self.module_membership = module_membership
        self.WGCNA_object     = WGCNA_object
        self.survivor_list    = survivor_list
        self.selection_list   = selection_list
        self.final_SHAP       = final_SHAP
        self.shap_obj         = shap_obj
        self.runtimes         = runtimes
        self.feature_list     = feature_list
        self.stability_freq   = stability_freq or {}
        self.stability_data   = stability_data

        # ingredients to compute the interaction matrix lazily on first access
        self._interaction_ingredients = _interaction_ingredients
        self._interaction_cache = None
        self._interaction_computed = False

    # lazy interaction matrix

    @property
    def interaction_matrix(self):
        """
        p×p mean absolute TreeSHAP interaction matrix, computed on first access
        and cached. Returns None if interactions were disabled
        (compute_interactions=False) or no stable features were found.

        The interaction call is ~p× the cost of regular SHAP, so deferring it
        until something actually needs it keeps fit() fast.
        """
        if self._interaction_computed:
            return self._interaction_cache

        self._interaction_computed = True
        ing = self._interaction_ingredients
        if ing is None or ing.get("rf_stable") is None:
            self._interaction_cache = None
            return None

        from .rfe import shap_interaction_matrix
        try:
            int_mat_raw = shap_interaction_matrix(
                ing["rf_stable"], ing["X_stable"], ing["classification"],
                backend=ing.get("backend", "shap"),
            )
            order = ing["order"]
            self._interaction_cache = int_mat_raw[np.ix_(order, order)]
        except Exception:
            self._interaction_cache = None

        # keep shap_obj consistent for callers that read it directly
        self.shap_obj["interaction_matrix"] = self._interaction_cache
        return self._interaction_cache

    # print / repr

    def __repr__(self) -> str:
        n_stable = len(self.final_SHAP) if self.final_SHAP is not None else 0
        lines = [
            "--- Mossy Forest ---",
            f"Stable features selected: {n_stable}",
            "",
        ]
        if self.final_SHAP is not None and n_stable > 0:
            lines.append(self.final_SHAP.to_string(index=False))
        oob = getattr(getattr(self.final_rf, "oob_score_", None), "__float__", None)
        if oob is not None:
            lines.append(f"\nOOB score: {self.final_rf.oob_score_:.4f}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    # predict

    def predict(self, new_X) -> np.ndarray:
        """
        Predict from the final RF on stable features.

        Parameters
        ----------
        new_X : pd.DataFrame or array like
            New observations.
            - pd.DataFrame: columns are matched by name to the stable set.
            - array like with the same number of columns as the stable set:
              used directly.
            - array like with more columns than the stable set: raises ValueError;
              use a DataFrame or pass MossyForest.predict() which subsets for you.

        Returns
        -------
        ndarray of predictions.
        """
        stable_feats = list(self.final_X.columns)
        if isinstance(new_X, pd.DataFrame):
            new_X = new_X[stable_feats].values
        else:
            new_X = np.asarray(new_X)
            if new_X.shape[1] != len(stable_feats):
                raise ValueError(
                    f"new_X has {new_X.shape[1]} columns but the stable set has "
                    f"{len(stable_feats)} features. Pass a DataFrame with named "
                    "columns, or use MossyForest.predict() which subsets automatically."
                )
        return self.final_rf.predict(new_X)

    def predict_proba(self, new_X) -> np.ndarray:
        """predict_proba for classifiers."""
        import pandas as pd
        stable_feats = list(self.final_X.columns)
        if isinstance(new_X, pd.DataFrame):
            new_X = new_X[stable_feats].values
        else:
            new_X = np.asarray(new_X)
        return self.final_rf.predict_proba(new_X)

    # detect_interaction

    def detect_interaction(
        self,
        thresh: float = 0.0,
        all: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Return exact TreeSHAP pairwise interaction strengths.

        Matches R's detect_interaction().

        Parameters
        ----------
        thresh  : float
            Only pairs above this mean |SHAP interaction| are returned.
        all     : bool
            If True, return all off diagonal pairs regardless of thresh.
        verbose : bool
            Print the significant pair table.

        Returns
        -------
        pd.DataFrame  columns: FeatureA, FeatureB, Interaction_Strength
        """
        int_mat = self.interaction_matrix   # lazy: computes + caches on first access
        if int_mat is None:
            raise ValueError(
                "No interaction matrix in this result. "
                "Refit with compute_interactions=True (the default)."
            )

        feats = list(self.final_X.columns)
        p     = len(feats)
        if p < 2:
            raise ValueError(
                "At least two stable features are required to report interactions."
            )

        rows = []
        for j in range(p - 1):
            for l in range(j + 1, p):
                rows.append({
                    "FeatureA":            feats[j],
                    "FeatureB":            feats[l],
                    "Interaction_Strength": float(int_mat[j, l]),
                })

        all_pairs = (
            pd.DataFrame(rows)
            .sort_values("Interaction_Strength", ascending=False)
            .reset_index(drop=True)
        )

        if all:
            return all_pairs

        sig = all_pairs[all_pairs["Interaction_Strength"] > thresh].reset_index(drop=True)

        if verbose:
            if len(sig) > 0:
                print(sig.to_string(index=False))
            else:
                print(
                    f"No interactions found above threshold ({thresh:.4g}). "
                    "Use thresh=0 or all=True to see all pairs."
                )
        return sig
