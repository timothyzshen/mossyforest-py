"""
Unit tests for BonsaiForest core components.

Tests are written against the new two-phase API that exactly mirrors
the R package's bf() / bf_python_backend.py v18.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

from bonsaiforest import (
    BonsaiForest,
    screen_control,
    select_control,
    WGCNA_control,
)
from bonsaiforest.shadow import (
    make_shadow_matrix,
    boruta_shadow_stability,
    apply_stable_threshold,
    find_elbow_idx,
)
from bonsaiforest.stability import StabilityTracker
from bonsaiforest.cfres import cross_fit_residuals
from bonsaiforest.rfe import rfe_loop, _compute_mtry, _compute_ntree


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def classification_data():
    rng = np.random.default_rng(42)
    X   = rng.standard_normal((120, 20))
    y   = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X   = rng.standard_normal((120, 20))
    y   = X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(120) * 0.1
    return X, y


# ── Parameter objects ─────────────────────────────────────────────────────────

def test_screen_control_defaults():
    sc = screen_control()
    assert sc.drop_fraction == 0.25
    assert sc.keep_fraction == 0.50
    assert sc.mtry_factor   == 1.0
    assert sc.min_ntree     == 500
    assert sc.ntree_factor  == 1


def test_select_control_defaults():
    sl = select_control()
    assert sl.drop_fraction     == 0.10
    assert sl.shadow_mode       == "split"
    assert sl.n_boots           == 50
    assert sl.pi_thr            == 0.60
    assert sl.pi_thr_unassigned      is None
    assert sl.threshold_mode    == "pi_thr"
    assert sl.shadow_percentile == 95.0
    assert sl.use_cfres  is False
    assert sl.cfres_n_folds       == 5
    assert sl.cfres_ntree         == 300
    assert sl.cfres_scope         == "unassigned"


def test_screen_control_bad_drop():
    with pytest.raises(ValueError, match="drop_fraction"):
        screen_control(drop_fraction=1.5)


def test_select_control_bad_shadow_mode():
    with pytest.raises(ValueError, match="shadow_mode"):
        select_control(shadow_mode="bad")


def test_select_control_bad_cfres_scope():
    with pytest.raises(ValueError, match="cfres_scope"):
        select_control(cfres_scope="bad")


def test_select_control_bad_threshold_mode():
    with pytest.raises(ValueError, match="threshold_mode"):
        select_control(threshold_mode="bad")




# ── BonsaiForest instantiation ───────────────────────────────────────────────

def test_instantiation():
    model = BonsaiForest()
    assert model.final_ntree == 500
    assert model.n_jobs      == 1


def test_instantiation_custom_params():
    sc = screen_control(drop_fraction=0.30, keep_fraction=0.40)
    sl = select_control(n_boots=20, pi_thr=0.70, shadow_mode="within_module")
    model = BonsaiForest(screen_params=sc, select_params=sl, final_ntree=200)
    assert model.screen_params.drop_fraction == 0.30
    assert model.select_params.n_boots       == 20
    assert model.final_ntree                 == 200


def test_fit_requires_module_membership(classification_data):
    X, y = classification_data
    model = BonsaiForest()
    with pytest.raises(ValueError, match="module_membership"):
        model.fit(X, y)


# ── mtry / ntree helpers ──────────────────────────────────────────────────────

def test_compute_mtry_regression():
    assert _compute_mtry(100, 1.0, classification=False) == 10

def test_compute_mtry_classification():
    assert _compute_mtry(30, 1.0, classification=True)   == 10

def test_compute_ntree():
    assert _compute_ntree(100, 1, 500) == 500
    assert _compute_ntree(600, 1, 500) == 600


# ── Shadow matrix ─────────────────────────────────────────────────────────────

def test_shadow_matrix_shape():
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((50, 10))
    aug = make_shadow_matrix(X, rng)
    assert aug.shape == (50, 20)


def test_shadow_matrix_real_unchanged():
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((50, 10))
    aug = make_shadow_matrix(X, rng)
    np.testing.assert_array_equal(aug[:, :10], X)


def test_shadow_cols_are_permutations():
    """Shadow columns contain the same values as real columns, just permuted."""
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((50, 5))
    aug = make_shadow_matrix(X, rng)
    for j in range(5):
        assert set(aug[:, j]) == set(aug[:, 5 + j])


# ── Elbow detection ───────────────────────────────────────────────────────────

def test_find_elbow_idx_flat():
    assert find_elbow_idx([0.5, 0.5, 0.5, 0.5]) == 0


def test_find_elbow_idx_clear_elbow():
    freqs = [1.0, 0.9, 0.8, 0.2, 0.1, 0.05]
    idx   = find_elbow_idx(freqs)
    # elbow should be around index 3 (the big drop)
    assert idx in (2, 3)


def test_apply_stable_threshold_fixed():
    names = ["a", "b", "c", "d"]
    freqs = [0.9, 0.7, 0.5, 0.3]
    stable, thr, elbow = apply_stable_threshold(names, freqs, pi_thr=0.6)
    assert stable == ["a", "b"]
    assert thr    == 0.6
    assert elbow  is None


def test_apply_stable_threshold_elbow():
    names = ["a", "b", "c", "d", "e"]
    freqs = [0.95, 0.90, 0.85, 0.20, 0.10]
    stable, thr, elbow = apply_stable_threshold(names, freqs, pi_thr=0.6, use_elbow=True)
    assert len(stable) >= 1
    assert elbow is not None and elbow >= 1


# ── RFE loop ──────────────────────────────────────────────────────────────────

def test_rfe_loop_returns_target(classification_data):
    X, y = classification_data
    feat_names = [f"X{i}" for i in range(X.shape[1])]
    survivors, vims, init_info, steps = rfe_loop(
        X, feat_names, (y > 0).astype(int), target=5,
        drop_fraction=0.25, mtry_factor=1.0, ntree_factor=1, min_ntree=50,
        nodesize=1, classification=True, seed=42, n_jobs=1,
    )
    assert len(survivors) == 5
    assert len(vims)       == 5
    assert init_info is not None


def test_rfe_loop_survivor_names_are_subset(classification_data):
    X, y = classification_data
    feat_names = [f"X{i}" for i in range(X.shape[1])]
    survivors, _, _, _ = rfe_loop(
        X, feat_names, (y > 0).astype(int), target=3,
        drop_fraction=0.25, mtry_factor=1.0, ntree_factor=1, min_ntree=50,
        nodesize=1, classification=True, seed=0, n_jobs=1,
    )
    assert all(f in feat_names for f in survivors)


# ── cfRes cross-fitted residualization ─────────────────────────────────────────

def test_cross_fit_residuals_shape(regression_data):
    X, y = regression_data
    y_resid = cross_fit_residuals(X[:, :5], y, k_folds=2, ntree=20, seed=0)
    assert y_resid.shape == (len(y),)
    assert not np.allclose(y_resid, y)   # residuals differ from original y


def test_cross_fit_residuals_reduces_variance(regression_data):
    """y_resid should have lower variance than y when X is predictive."""
    rng = np.random.default_rng(7)
    n   = 200
    X   = rng.standard_normal((n, 5))
    y   = X[:, 0] * 3 + X[:, 1] * 2 + rng.standard_normal(n) * 0.2
    y_resid = cross_fit_residuals(X, y, k_folds=5, ntree=100, seed=0)
    assert np.var(y_resid) < np.var(y)


# ── StabilityTracker ──────────────────────────────────────────────────────────

def test_stability_tracker():
    tracker = StabilityTracker(n_features=10)
    tracker.update(np.array([0, 1, 2]))
    tracker.update(np.array([1, 2, 3]))
    freqs = tracker.frequencies
    assert freqs[1] == 1.0
    assert freqs[2] == 1.0
    assert freqs[0] == 0.5
    assert freqs[9] == 0.0


def test_stability_tracker_stable_set():
    tracker = StabilityTracker(n_features=5)
    tracker.update(np.array([0, 1]))
    tracker.update(np.array([1, 2]))
    stable = tracker.stable_set(pi_thr=0.8)
    assert set(stable) == {1}


def test_stability_tracker_reset():
    tracker = StabilityTracker(n_features=4)
    tracker.update(np.array([0, 1]))
    tracker.reset()
    assert tracker.n_iter == 0
    assert tracker.counts.sum() == 0


# ── Boruta shadow stability (smoke test) ──────────────────────────────────────

def test_boruta_shadow_stability_returns_all_features(classification_data):
    X, y = classification_data
    feat_names = [f"X{i}" for i in range(X.shape[1])]
    names_sort, freqs_sort = boruta_shadow_stability(
        X_pool=X, feat_names=feat_names, y=y.astype(int),
        n_boots=5, mtry_factor=1.0, ntree_factor=1, min_ntree=30,
        nodesize=1, classification=True, seed=0, n_jobs=1,
        shadow_percentile=95,
    )
    assert len(names_sort) == X.shape[1]
    assert len(freqs_sort) == X.shape[1]
    assert all(0.0 <= f <= 1.0 for f in freqs_sort)
    # frequencies should be in descending order
    assert all(freqs_sort[i] >= freqs_sort[i+1] for i in range(len(freqs_sort)-1))


# ── BonsaiForest end-to-end (mocked Phase 1) ────────────────────────────────

def _make_labels(n_feat, mods):
    """Create module_membership for n_feat features in equal chunks."""
    per = n_feat // len(mods)
    labels = []
    for m in mods:
        labels.extend([m] * per)
    labels.extend([mods[-1]] * (n_feat - len(labels)))
    return labels


@pytest.fixture
def small_block_data():
    rng  = np.random.default_rng(0)
    n, p = 100, 20
    z    = rng.standard_normal(n)
    X    = np.column_stack([z + rng.standard_normal(n) * 0.3 for _ in range(10)]
                           + [rng.standard_normal(n) for _ in range(10)])
    y    = (z > 0).astype(int)
    mods = ["A"] * 10 + ["unassigned"] * 10
    return X, y, mods


def test_bonsaiforest_fit_runs(small_block_data):
    """BonsaiForest.fit() must complete without errors."""
    X, y, mods = small_block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=20)
    sl = select_control(n_boots=5, min_ntree=20, shadow_percentile=95,
                        pi_thr=0.4, shadow_mode="split",
                        unassigned_modules=["unassigned"])
    model = BonsaiForest(
        screen_params=sc, select_params=sl,
        final_ntree=20, random_state=42, verbose=0,
    )
    model.fit(X, y, module_membership=mods)
    assert hasattr(model, "result_")
    assert hasattr(model, "selected_features_")


def test_bonsaiforest_transform(small_block_data):
    X, y, mods = small_block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=20)
    sl = select_control(n_boots=5, min_ntree=20, pi_thr=0.4,
                        shadow_mode="split", unassigned_modules=["unassigned"])
    model = BonsaiForest(screen_params=sc, select_params=sl,
                          final_ntree=20, random_state=0, verbose=0)
    model.fit(X, y, module_membership=mods)
    X_t = model.transform(X)
    assert X_t.shape == (X.shape[0], len(model.selected_features_))


def test_bonsaiforest_within_module(small_block_data):
    """within_module shadow_mode must run without error."""
    X, y, mods = small_block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=20)
    sl = select_control(n_boots=5, min_ntree=20, pi_thr=0.4,
                        shadow_mode="within_module")
    model = BonsaiForest(screen_params=sc, select_params=sl,
                          final_ntree=20, random_state=1, verbose=0)
    model.fit(X, y, module_membership=mods)
    assert hasattr(model, "result_")


def test_bonsaiforest_predict(small_block_data):
    X, y, mods = small_block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=20)
    sl = select_control(n_boots=5, min_ntree=20, pi_thr=0.4,
                        shadow_mode="split", unassigned_modules=["unassigned"])
    model = BonsaiForest(screen_params=sc, select_params=sl,
                          final_ntree=20, random_state=0, verbose=0)
    model.fit(X, y, module_membership=mods)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_result_attributes(small_block_data):
    """BonsaiForestResult must have all expected attributes."""
    X, y, mods = small_block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=20)
    sl = select_control(n_boots=5, min_ntree=20, pi_thr=0.4,
                        shadow_mode="split", unassigned_modules=["unassigned"])
    model = BonsaiForest(screen_params=sc, select_params=sl,
                          final_ntree=20, random_state=0, verbose=0)
    model.fit(X, y, module_membership=mods)
    res = model.result_
    assert hasattr(res, "final_rf")
    assert hasattr(res, "final_X")
    assert hasattr(res, "module_membership")
    assert hasattr(res, "survivor_list")
    assert hasattr(res, "selection_list")
    assert hasattr(res, "final_SHAP")
    assert hasattr(res, "shap_obj")
    assert hasattr(res, "runtimes")
    assert hasattr(res, "stability_freq")
    # shap_obj structure
    assert "shapley_values" in res.shap_obj
    assert "interaction_matrix" in res.shap_obj
    # runtimes
    for key in ("Screen", "Selection", "Final_RF"):
        assert key in res.runtimes


def test_final_shap_columns(small_block_data):
    """final_SHAP DataFrame must have the expected CI columns."""
    X, y, mods = small_block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=20)
    sl = select_control(n_boots=5, min_ntree=20, pi_thr=0.4,
                        shadow_mode="split", unassigned_modules=["unassigned"])
    model = BonsaiForest(screen_params=sc, select_params=sl,
                          final_ntree=20, random_state=0, verbose=0)
    model.fit(X, y, module_membership=mods)
    df = model.result_.final_SHAP
    expected_cols = [
        "feature_name", "variable_importance", "module_membership",
        "mean_abs_shap", "se_abs_shap", "ci_lower_abs", "ci_upper_abs",
        "mean_signed_shap", "se_signed_shap",
        "ci_lower_signed", "ci_upper_signed", "stability_freq",
    ]
    if len(df) > 0:
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
