"""
Tests for the performance-optimization layer:
  - parallelization determinism (n_jobs-independent results)
  - lazy interaction matrix
  - shap_backend / use_numba value-equivalence
  - early-stop bootstraps
  - float32 path
  - blas_threads / thread helpers
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from mossyforest import mf, screen_control, select_control
from mossyforest.threads import resolve_blas_threads
from mossyforest.shadow import make_shadow_matrix


# ── Shared data ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def block_data():
    rng = np.random.default_rng(0)
    n, noise = 160, 0.3
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    mod_A = np.column_stack([z1 + rng.standard_normal(n) * noise for _ in range(8)])
    mod_B = np.column_stack([z2 + rng.standard_normal(n) * noise for _ in range(8)])
    ind   = rng.standard_normal((n, 6))
    X = np.hstack([mod_A, mod_B, ind])
    y = (z1 - z2 > 0).astype(int)
    labels = ["A"] * 8 + ["B"] * 8 + ["unassigned"] * 6
    feats  = ([f"A{i}" for i in range(8)] + [f"B{i}" for i in range(8)]
              + [f"ind{i}" for i in range(6)])
    return pd.DataFrame(X, columns=feats), y, dict(zip(feats, labels))


def _params():
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=30)
    sl = select_control(n_boots=12, min_ntree=30, pi_thr=0.4,
                        shadow_mode="split", unassigned_modules=["unassigned"])
    return sc, sl


def _run(block_data, n_jobs, **kw):
    X, y, mm = block_data
    sc, sl = _params()
    return mf(X, y, module_membership=mm, screen_params=sc, select_params=sl,
              final_ntree=30, num_processors=n_jobs, verbose=0, seed=123, **kw)


# ── #1 Parallelization determinism (THE key guarantee) ────────────────────────

def test_parallel_matches_sequential(block_data):
    """n_jobs=1 and n_jobs=4 must produce identical results."""
    r1 = _run(block_data, n_jobs=1)
    r4 = _run(block_data, n_jobs=4)

    assert r1.final_SHAP["feature_name"].tolist() == r4.final_SHAP["feature_name"].tolist()
    assert r1.stability_freq == r4.stability_freq
    np.testing.assert_allclose(
        r1.shap_obj["shapley_values"], r4.shap_obj["shapley_values"], rtol=0, atol=0
    )


def test_parallel_stability_data_identical(block_data):
    r1 = _run(block_data, n_jobs=1)
    r2 = _run(block_data, n_jobs=2)
    if r1.stability_data is not None and r2.stability_data is not None:
        pd.testing.assert_frame_equal(
            r1.stability_data.reset_index(drop=True),
            r2.stability_data.reset_index(drop=True),
        )


# ── #2 Lazy interaction matrix ────────────────────────────────────────────────

def test_interaction_lazy_then_cached(block_data):
    r = _run(block_data, n_jobs=1)
    # not computed at fit time
    assert r._interaction_computed is False
    assert r.shap_obj["interaction_matrix"] is None

    n_stable = len(r.stability_freq)
    if n_stable >= 1:
        mat1 = r.interaction_matrix          # triggers compute
        assert r._interaction_computed is True
        mat2 = r.interaction_matrix          # cached
        if mat1 is not None:
            assert mat1 is mat2              # same object → cached
            assert mat1.shape == (n_stable, n_stable)
            np.testing.assert_allclose(mat1, mat1.T, atol=1e-10)
            # shap_obj now reflects the computed matrix
            assert r.shap_obj["interaction_matrix"] is mat1


def test_interaction_disabled(block_data):
    X, y, mm = block_data
    sc, sl = _params()
    r = mf(X, y, module_membership=mm, screen_params=sc, select_params=sl,
           final_ntree=30, verbose=0, seed=123, compute_interactions=False)
    assert r.interaction_matrix is None
    with pytest.raises(ValueError, match="No interaction matrix"):
        r.detect_interaction(verbose=False)


# ── #14 Early-stop bootstraps ─────────────────────────────────────────────────

def test_early_stop_runs(block_data):
    X, y, mm = block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=30)
    sl = select_control(n_boots=40, min_ntree=30, pi_thr=0.4,
                        shadow_mode="split", unassigned_modules=["unassigned"],
                        early_stop_boots=True, early_stop_tol=0.05,
                        early_stop_check_every=5)
    r = mf(X, y, module_membership=mm, screen_params=sc, select_params=sl,
           final_ntree=30, verbose=0, seed=7)
    # frequencies still valid in [0, 1]
    for f in r.stability_freq.values():
        assert 0.0 <= f <= 1.0


def test_early_stop_deterministic_across_workers(block_data):
    X, y, mm = block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=30)
    sl = select_control(n_boots=40, min_ntree=30, pi_thr=0.4,
                        shadow_mode="split", unassigned_modules=["unassigned"],
                        early_stop_boots=True, early_stop_tol=0.05,
                        early_stop_check_every=5)
    r1 = mf(X, y, module_membership=mm, screen_params=sc, select_params=sl,
            final_ntree=30, verbose=0, seed=7, num_processors=1)
    r2 = mf(X, y, module_membership=mm, screen_params=sc, select_params=sl,
            final_ntree=30, verbose=0, seed=7, num_processors=3)
    assert r1.stability_freq == r2.stability_freq


# ── #3 float32 path ───────────────────────────────────────────────────────────

def test_shap_matrix_finite(block_data):
    r = _run(block_data, n_jobs=1)
    assert np.isfinite(r.shap_obj["shapley_values"]).all()


def test_shadow_matrix_permuted():
    """make_shadow_matrix: real half unchanged; shadow half is a column permutation."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 5)).astype(np.float32)
    aug = make_shadow_matrix(X, rng)
    assert aug.shape == (40, 10)
    np.testing.assert_array_equal(aug[:, :5], X)
    for j in range(5):
        assert set(aug[:, j]) == set(aug[:, 5 + j])


# ── #13 Numba backend value-equivalence (skip if numba absent) ────────────────

@pytest.mark.skipif(importlib.util.find_spec("numba") is None,
                    reason="numba not installed")
def test_numba_matches_numpy(block_data):
    r_np = _run(block_data, n_jobs=1, use_numba=False)
    r_nb = _run(block_data, n_jobs=1, use_numba=True)
    assert (r_np.final_SHAP["feature_name"].tolist()
            == r_nb.final_SHAP["feature_name"].tolist())
    np.testing.assert_allclose(
        r_np.shap_obj["shapley_values"], r_nb.shap_obj["shapley_values"],
        rtol=1e-5, atol=1e-6,
    )


# ── #12 fasttreeshap backend (skip if absent) ─────────────────────────────────

@pytest.mark.skipif(importlib.util.find_spec("fasttreeshap") is None,
                    reason="fasttreeshap not installed")
def test_fasttreeshap_matches_shap(block_data):
    r_shap = _run(block_data, n_jobs=1, shap_backend="shap")
    r_fts  = _run(block_data, n_jobs=1, shap_backend="fasttreeshap")
    assert (r_shap.final_SHAP["feature_name"].tolist()
            == r_fts.final_SHAP["feature_name"].tolist())


def test_fasttreeshap_missing_raises_clearly(block_data):
    if importlib.util.find_spec("fasttreeshap") is not None:
        pytest.skip("fasttreeshap is installed")
    with pytest.raises(ImportError, match="fasttreeshap"):
        _run(block_data, n_jobs=1, shap_backend="fasttreeshap")


# ── #10 thread helper ─────────────────────────────────────────────────────────

def test_resolve_blas_threads_auto():
    assert resolve_blas_threads(None, 1) is None     # serial → unrestricted
    assert resolve_blas_threads(None, 4) == 1        # parallel → cap inner to 1
    assert resolve_blas_threads(2, 4) == 2           # explicit override wins
    assert resolve_blas_threads(8, 1) == 8


def test_bad_shap_backend(block_data):
    X, y, mm = block_data
    sc, sl = _params()
    with pytest.raises(ValueError, match="shap_backend"):
        mf(X, y, module_membership=mm, screen_params=sc, select_params=sl,
           verbose=0, seed=1, shap_backend="bogus")
