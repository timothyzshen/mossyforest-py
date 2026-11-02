"""
End-to-end smoke test for wmf() — the WGCNA + MossyForest path.

Runs real PyWGCNA module detection (not mocked), then the full two-phase
selection. Marked 'wgcna' so it can be deselected with `-m "not wgcna"` if
PyWGCNA is slow/unavailable in a given environment.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyWGCNA")

from mossyforest import wmf, WGCNA_control, screen_control, select_control
from mossyforest.modules import detect_modules


@pytest.fixture(scope="module")
def block_data():
    """3 correlated blocks (latent-driven) + noise, sized so WGCNA finds modules."""
    rng = np.random.default_rng(0)
    n = 80
    lat = [rng.standard_normal(n) for _ in range(3)]
    cols, feats = [], []
    for b, z in enumerate(lat):
        for j in range(15):
            cols.append(z + rng.standard_normal(n) * 0.2)
            feats.append(f"b{b}_{j}")
    for j in range(10):
        cols.append(rng.standard_normal(n))
        feats.append(f"noise{j}")
    X = pd.DataFrame(np.column_stack(cols), columns=feats)
    y = ((lat[0] + lat[1] - lat[2]) > 0).astype(int)
    return X, y


@pytest.mark.wgcna
def test_detect_modules_runs(block_data):
    X, _ = block_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = detect_modules(X.values, feature_names=list(X.columns),
                                min_module_size=5, verbose=False)
    assert labels.shape == (X.shape[1],)
    assert len(set(labels)) >= 2   # more than one module found


@pytest.mark.wgcna
def test_wsf_end_to_end(block_data):
    X, y = block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=50)
    sl = select_control(n_reps=12, min_ntree=50, pi_thr=0.4,
                        shadow_mode="split")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wmf(X, y,
                  WGCNA_params=WGCNA_control(power=6, min_features=5),
                  screen_params=sc, select_params=sl,
                  final_ntree=50, verbose=0, seed=42)

    # WGCNA output attached
    assert res.WGCNA_object is not None
    assert len(res.WGCNA_object) == X.shape[1]

    # standard result structure intact
    assert res.module_membership.shape[0] == X.shape[1]
    assert hasattr(res, "final_SHAP")
    assert "shapley_values" in res.shap_obj
    for k in ("Screen", "Selection", "Final_RF"):
        assert k in res.runtimes

    # predictions work on the fitted model
    if res.final_rf is not None:
        preds = res.predict(X)
        assert preds.shape == (X.shape[0],)


@pytest.mark.wgcna
def test_wsf_within_module_mode(block_data):
    """wmf with within_module shadow mode should also complete."""
    X, y = block_data
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=50)
    sl = select_control(n_reps=10, min_ntree=50, pi_thr=0.4,
                        shadow_mode="within_module")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wmf(X, y,
                  WGCNA_params=WGCNA_control(power=6, min_features=5),
                  screen_params=sc, select_params=sl,
                  final_ntree=50, verbose=0, seed=1)
    assert res.WGCNA_object is not None
