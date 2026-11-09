"""
End-to-end tests for MossyForest two phase algorithm.

Uses a synthetic block-structured dataset that mimics correlated gene modules.
Tests cover all shadow_mode / cfRes combinations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mossyforest import mf, screen_control, select_control


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def block_data():
    """
    Synthetic dataset with a known block correlation structure.

    Module A : 10 features driven by latent z1
    Module B : 10 features driven by latent z2
    Module C : 10 features driven by latent z3
    independent: 5 pure-noise features

    Binary outcome: y = 1 iff z1 + z2 - z3 > 0.
    True signals are the first feature of each module (cols 0, 10, 20).
    """
    rng = np.random.default_rng(42)
    n, noise = 200, 0.3

    z1, z2, z3 = (rng.standard_normal(n) for _ in range(3))

    mod_A = np.column_stack([z1 + rng.standard_normal(n) * noise for _ in range(10)])
    mod_B = np.column_stack([z2 + rng.standard_normal(n) * noise for _ in range(10)])
    mod_C = np.column_stack([z3 + rng.standard_normal(n) * noise for _ in range(10)])
    ind   = rng.standard_normal((n, 5))

    X = np.hstack([mod_A, mod_B, mod_C, ind])
    y = (z1 + z2 - z3 > 0).astype(int)

    labels = (["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["unassigned"] * 5)

    feat_names = (
        [f"A{i}" for i in range(10)]
        + [f"B{i}" for i in range(10)]
        + [f"C{i}" for i in range(10)]
        + [f"ind{i}" for i in range(5)]
    )
    return np.c_[X], y, labels, feat_names


def _fast_params():
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=30)
    sl = select_control(
        n_reps=8, min_ntree=30, pi_thr=0.4,
        shadow_mode="split",
        unassigned_modules=["unassigned"],
        shadow_percentile=95,
    )
    return sc, sl


# ── Split shadow mode ─────────────────────────────────────────────────────────

def test_split_mode_runs(block_data):
    X, y, labels, feat_names = block_data
    sc, sl = _fast_params()
    result = mf(
        pd.DataFrame(X, columns=feat_names), y,
        module_membership=dict(zip(feat_names, labels)),
        screen_params=sc, select_params=sl,
        final_ntree=30, verbose=0, seed=42,
    )
    assert result is not None
    assert hasattr(result, "final_SHAP")


def test_split_mode_result_structure(block_data):
    X, y, labels, feat_names = block_data
    sc, sl = _fast_params()
    result = mf(
        X, y,
        module_membership=labels,
        screen_params=sc, select_params=sl,
        final_ntree=30, verbose=0, seed=42,
    )
    # shap_matrix shape
    n_stable = len(result.stability_freq)
    assert result.shap_obj["shapley_values"].shape == (X.shape[0], n_stable)
    # runtimes populated
    assert all(k in result.runtimes for k in ("Screen", "Selection", "Final_RF"))


def test_split_mode_stability_data(block_data):
    X, y, labels, feat_names = block_data
    sc, sl = _fast_params()
    result = mf(
        X, y, module_membership=labels,
        screen_params=sc, select_params=sl,
        final_ntree=30, verbose=0, seed=0,
    )
    if result.stability_data is not None:
        assert "pool" in result.stability_data.columns
        assert "freq" in result.stability_data.columns
        assert "selected" in result.stability_data.columns


def test_split_mode_shap_ci_columns(block_data):
    X, y, labels, _ = block_data
    sc, sl = _fast_params()
    result = mf(
        X, y, module_membership=labels,
        screen_params=sc, select_params=sl,
        final_ntree=30, verbose=0, seed=1,
    )
    if len(result.final_SHAP) > 0:
        for col in ("mean_abs_shap", "ci_lower_abs", "ci_upper_abs",
                    "mean_signed_shap", "stability_freq"):
            assert col in result.final_SHAP.columns


# ── Within-module shadow mode ─────────────────────────────────────────────────

def test_threshold_mode_elbow_runs(block_data):
    """threshold_mode='elbow' (Alternative A) applies the elbow rule to BOTH pools."""
    X, y, labels, _ = block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=30)
    sl = select_control(n_reps=8, min_ntree=30, pi_thr=0.4,
                        shadow_mode="split", unassigned_modules=["unassigned"],
                        threshold_mode="elbow")
    result = mf(X, y, module_membership=labels,
                screen_params=sc, select_params=sl,
                final_ntree=30, verbose=0, seed=5)
    assert result is not None
    # under elbow mode the CORRELATED pool must carry an elbow_rank (it's None
    # in the default pi_thr mode), proving the unified rule is applied to it.
    sd = result.stability_data
    if sd is not None and (sd["pool"] == "corr").any():
        corr_elbow = sd.loc[sd["pool"] == "corr", "elbow_rank"].dropna()
        assert len(corr_elbow) > 0, "corr pool should have an elbow_rank under elbow mode"



def test_shap_direction_recovers_sign(block_data):
    """
    Direction should recover the sign of a feature's effect: features positively
    associated with the outcome get direction +1, negatively-associated get -1.
    """
    import numpy as np
    rng = np.random.default_rng(1)
    n = 250
    a = rng.standard_normal(n)   # positive effect
    b = rng.standard_normal(n)   # negative effect
    Xd = pd.DataFrame({
        **{f"pos{i}": a + rng.standard_normal(n) * 0.3 for i in range(6)},
        **{f"neg{i}": b + rng.standard_normal(n) * 0.3 for i in range(6)},
        **{f"noise{i}": rng.standard_normal(n) for i in range(6)},
    })
    y = ((2 * a - 2 * b) > 0).astype(int)
    mm = {c: ("A" if c.startswith("pos") else "B" if c.startswith("neg")
              else "unassigned") for c in Xd.columns}
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.6, min_ntree=60)
    sl = select_control(n_reps=15, min_ntree=60, pi_thr=0.3,
                        shadow_mode="split", unassigned_modules=["unassigned"])
    res = mf(Xd, y, module_membership=mm, screen_params=sc, select_params=sl,
             final_ntree=60, verbose=0, seed=3)
    df = res.final_SHAP
    assert "direction" in df.columns
    assert "signed_importance" in df.columns
    assert "dir_corr" in df.columns
    # among selected features, pos* should trend +1 and neg* should trend -1
    pos = df[df["feature_name"].str.startswith("pos")]
    neg = df[df["feature_name"].str.startswith("neg")]
    if len(pos) > 0:
        assert (pos["direction"] >= 0).mean() > 0.5
    if len(neg) > 0:
        assert (neg["direction"] <= 0).mean() > 0.5
    # signed_importance sign matches direction, magnitude matches mean_abs
    np.testing.assert_allclose(
        df["signed_importance"].abs().values, df["mean_abs_shap"].values, atol=1e-4)




def test_within_module_mode_runs(block_data):
    X, y, labels, _ = block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=30)
    sl = select_control(n_reps=8, min_ntree=30, pi_thr=0.4,
                        shadow_mode="within_module", shadow_percentile=95)
    result = mf(
        X, y, module_membership=labels,
        screen_params=sc, select_params=sl,
        final_ntree=30, verbose=0, seed=5,
    )
    assert result is not None


def test_within_module_stability_data_pools(block_data):
    """within_module mode should produce one pool per module in stability_data."""
    X, y, labels, _ = block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=30)
    sl = select_control(n_reps=6, min_ntree=30, pi_thr=0.4,
                        shadow_mode="within_module", shadow_percentile=95)
    result = mf(
        X, y, module_membership=labels,
        screen_params=sc, select_params=sl,
        final_ntree=30, verbose=0, seed=7,
    )
    if result.stability_data is not None:
        unique_pools = set(result.stability_data["pool"])
        # should contain at least one of the module labels
        assert len(unique_pools) >= 1


# ── cfRes y-residualization ─────────────────────────────────────────────────────

def test_dml_unassigned_scope_runs(block_data):
    """use_cfres=True with cfres_scope='unassigned' must complete."""
    X, y, labels, _ = block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=30)
    sl = select_control(
        n_reps=6, min_ntree=30, pi_thr=0.4,
        shadow_mode="split", unassigned_modules=["unassigned"],
        use_cfres=True, cfres_scope="unassigned",
        cfres_n_folds=2, cfres_ntree=20,
    )
    result = mf(
        X, y, module_membership=labels,
        screen_params=sc, select_params=sl,
        final_ntree=30, verbose=0, seed=99,
    )
    assert result is not None


def test_dml_all_modules_scope_within(block_data):
    """use_cfres=True with cfres_scope='all_modules' within_module."""
    X, y, labels, _ = block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=30)
    sl = select_control(
        n_reps=5, min_ntree=30, pi_thr=0.4,
        shadow_mode="within_module",
        use_cfres=True, cfres_scope="all_modules",
        cfres_n_folds=2, cfres_ntree=20,
    )
    result = mf(
        X, y, module_membership=labels,
        screen_params=sc, select_params=sl,
        final_ntree=30, verbose=0, seed=11,
    )
    assert result is not None


# ── Interaction matrix ────────────────────────────────────────────────────────

def test_interaction_matrix_shape(block_data):
    X, y, labels, _ = block_data
    sc, sl = _fast_params()
    result = mf(
        X, y, module_membership=labels,
        screen_params=sc, select_params=sl,
        final_ntree=30, verbose=0, seed=3,
        compute_interactions=True,
    )
    int_mat = result.shap_obj["interaction_matrix"]
    n_stable = len(result.stability_freq)
    if int_mat is not None and n_stable > 0:
        assert int_mat.shape == (n_stable, n_stable)
        # symmetric
        np.testing.assert_allclose(int_mat, int_mat.T, atol=1e-10)
        # non-negative
        assert (int_mat >= 0).all()


def test_detect_interaction(block_data):
    X, y, labels, _ = block_data
    sc, sl = _fast_params()
    result = mf(
        X, y, module_membership=labels,
        screen_params=sc, select_params=sl,
        final_ntree=30, verbose=0, seed=3,
        compute_interactions=True,
    )
    if (result.shap_obj["interaction_matrix"] is not None
            and len(result.stability_freq) >= 2):
        df = result.detect_interaction(thresh=0, verbose=False)
        assert "FeatureA" in df.columns
        assert "FeatureB" in df.columns
        assert "Interaction_Strength" in df.columns
        # sorted descending
        assert (df["Interaction_Strength"].diff().dropna() <= 0).all()


# ── predict() ────────────────────────────────────────────────────────────────

def test_predict_returns_array(block_data):
    X, y, labels, feat_names = block_data
    sc, sl = _fast_params()
    import pandas as pd
    result = mf(
        pd.DataFrame(X, columns=feat_names), y,
        module_membership=dict(zip(feat_names, labels)),
        screen_params=sc, select_params=sl,
        final_ntree=30, verbose=0, seed=4,
    )
    if result.final_rf is not None:
        # Pass a DataFrame so result.predict can subset to stable features by name
        preds = result.predict(pd.DataFrame(X, columns=feat_names))
        assert preds.shape == (X.shape[0],)


# ── DataFrame input ───────────────────────────────────────────────────────────

def test_dataframe_input(block_data):
    X, y, labels, feat_names = block_data
    X_df = pd.DataFrame(X, columns=feat_names)
    sc, sl = _fast_params()
    result = mf(
        X_df, y,
        module_membership=dict(zip(feat_names, labels)),
        screen_params=sc, select_params=sl,
        final_ntree=30, verbose=0, seed=0,
    )
    if len(result.stability_freq) > 0:
        assert all(f in feat_names for f in result.stability_freq)


# ── Signal vs noise (statistical sanity) ─────────────────────────────────────

def test_corr_module_survivors_beat_noise_in_frequency(block_data):
    """
    Features that survive Phase 1 from signal modules (A, B, C) should have
    higher mean shadow stability frequency than survivors from the unassigned pool.

    Within each module, all features are equally driven by the same latent variable,
    so screening keeps whichever half have the highest SHAP, not necessarily A0/B0/C0.
    We compare ANY corr-module survivor (pool="corr") against noise survivors
    (pool="unassigned") using the stability_data per-pool frequencies.
    """
    X, y, labels, feat_names = block_data
    X_df = pd.DataFrame(X, columns=feat_names)

    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=50)
    sl = select_control(
        n_reps=20, min_ntree=50, pi_thr=0.01,
        shadow_mode="split",
        unassigned_modules=["unassigned"],
        shadow_percentile=95,
    )
    result = mf(
        X_df, y,
        module_membership=dict(zip(feat_names, labels)),
        screen_params=sc, select_params=sl,
        final_ntree=30, verbose=0, seed=42,
    )

    sd = result.stability_data
    if sd is None or len(sd) == 0:
        pytest.skip("No stability data available.")

    corr_freqs  = sd[sd["pool"] == "corr"]["freq"].values
    unassigned_freqs = sd[sd["pool"] == "unassigned"]["freq"].values

    if len(corr_freqs) == 0 or len(unassigned_freqs) == 0:
        pytest.skip("Missing pool data.")

    assert corr_freqs.mean() > unassigned_freqs.mean(), (
        f"Corr pool mean freq {corr_freqs.mean():.3f} not > "
        f"unassigned pool mean freq {unassigned_freqs.mean():.3f}"
    )


def test_dml_classification_uses_regression_on_residual():
    """
    Classification + use_cfres: the residualized (continuous) target must
    be fit as REGRESSION in stability selection, not truncated to int.
    Regression finds a weak independent signal that the buggy int-cast misses.
    """
    import numpy as np
    from mossyforest.shadow import boruta_shadow_stability
    from mossyforest.select import cross_fit_residuals
    rng = np.random.default_rng(0); n = 300
    z, ind = rng.standard_normal(n), rng.standard_normal(n)
    Xcorr = np.column_stack([z + rng.standard_normal(n) * 0.3 for _ in range(6)])
    Xind  = np.column_stack([ind] + [rng.standard_normal(n) for _ in range(5)]).astype(np.float32)
    y_int = ((z + 0.7 * ind) > 0).astype(int)
    y_res = cross_fit_residuals(Xcorr, y_int, k_folds=5, ntree=200, seed=0)
    feats = [f"ind{i}" for i in range(6)]

    def freq(classif):
        names, freqs = boruta_shadow_stability(
            X_pool=Xind, feat_names=feats, y=y_res, n_reps=20, mtry_factor=1.0,
            ntree_factor=1, min_ntree=100, nodesize=5, classification=classif,
            seed=1, n_jobs=1, shadow_percentile=95)
        return dict(zip(names, freqs))["ind0"]

    # regression (the fix) recovers the signal; classification (bug) truncates it away
    assert freq(False) > freq(True)
    assert freq(False) >= 0.5


# ── unassigned_modules validation ─────────────────────────────────────────────

def test_unassigned_modules_name_not_in_membership_warns(block_data):
    X, y, labels, feat_names = block_data
    sc, _ = _fast_params()
    sl = select_control(n_reps=4, min_ntree=30, pi_thr=0.4,
                        unassigned_modules=["no_such_module"])
    with pytest.warns(UserWarning, match="not found in `module_membership`.*no_such_module"):
        mf(pd.DataFrame(X, columns=feat_names), y,
           module_membership=dict(zip(feat_names, labels)),
           screen_params=sc, select_params=sl,
           final_ntree=30, verbose=0, seed=42)


def test_unassigned_modules_valid_name_is_silent(block_data):
    import warnings
    X, y, labels, feat_names = block_data
    sc, sl = _fast_params()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mf(pd.DataFrame(X, columns=feat_names), y,
           module_membership=dict(zip(feat_names, labels)),
           screen_params=sc, select_params=sl,
           final_ntree=30, verbose=0, seed=42)
