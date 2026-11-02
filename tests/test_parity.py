"""
Bit-exact parity between the standalone package and the R package's Python
backend (mf_python_backend.py), run in one process on identical data/seed/stack.

Marked 'parity'. Auto-skips if the R backend isn't found on disk, so the main
suite still runs anywhere. Searches the canonical package first, then sf_dev.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mossyforest import mf, screen_control, select_control

# candidate locations for the R package's Python backend (canonical first)
_BACKEND_DIRS = [
    "/Users/tzshen/Documents/Github/mossyforest/inst/python",
    "/Users/tzshen/Documents/Github/sf_dev/inst/python",
]


def _load_backend():
    for d in _BACKEND_DIRS:
        fp = Path(d) / "mf_python_backend.py"
        if fp.exists():
            spec = importlib.util.spec_from_file_location("sf_r_backend", fp)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["sf_r_backend"] = mod
            spec.loader.exec_module(mod)
            return mod, d
    return None, None


_BK, _BK_DIR = _load_backend()
pytestmark = [
    pytest.mark.parity,
    pytest.mark.skipif(_BK is None, reason="R package Python backend not found on disk"),
]


def _make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    lat = [rng.standard_normal(n) for _ in range(3)]
    cols, feats, labels = [], [], []
    for m, z in enumerate(lat):
        for j in range(10):
            cols.append(z + rng.standard_normal(n) * 0.3)
            feats.append(f"M{m}_{j}"); labels.append(f"M{m}")
    for j in range(10):
        cols.append(rng.standard_normal(n))
        feats.append(f"ind{j}"); labels.append("unassigned")
    X = np.column_stack(cols)
    y = (lat[0] + lat[1] - lat[2] > 0).astype(int)
    return X, y, feats, labels


def _run_backend(X, y, feats, labels, seed):
    module_list = list(dict.fromkeys(labels))
    scr = _BK.run_screen_rfe(
        X_mat=X.astype(np.float64), y=y, feature_names=feats,
        module_assignments=labels, module_list=module_list,
        drop_fraction=0.3, keep_fraction=0.5, mtry_factor=1.0, ntree_factor=1,
        min_ntree=100, nodesize=1, classification=True, seed=seed, n_jobs=1,
        max_modsize=5000)
    surv_mod = {}
    for m, sfeats in zip(module_list, scr["survivor_features"]):
        for f in sfeats:
            surv_mod[f] = m
    survivors = {m: sorted(sfeats) for m, sfeats in zip(module_list, scr["survivor_features"])}
    order = [f for f in feats if f in surv_mod]                 # R's original-X order
    Xs = X[:, [feats.index(f) for f in order]].astype(np.float64)
    sel = _BK.run_select_rfe(
        X_surv_mat=Xs, y=y, feature_names_surv=order,
        mtry_factor=1.0, ntree_factor=1, min_ntree=100,
        nodesize=1, classification=True, seed=seed, n_jobs=1,
        n_reps=30, pi_thr=0.4, shadow_mode="split", shadow_percentile=95,
        module_assignments_surv=[surv_mod[f] for f in order],
        unassigned_modules=["unassigned"], verbose=False)
    return survivors, sel


@pytest.fixture(scope="module")
def both():
    warnings.simplefilter("ignore")
    seed = 42
    X, y, feats, labels = _make_data()
    Xdf = pd.DataFrame(X, columns=feats)
    mm = dict(zip(feats, labels))

    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=100)
    sl = select_control(n_reps=30, min_ntree=100, pi_thr=0.4,
                        shadow_mode="split", unassigned_modules=["unassigned"])
    res = mf(Xdf, y, module_membership=mm, screen_params=sc, select_params=sl,
             final_ntree=100, verbose=0, seed=seed, num_processors=1)

    bk_surv, bk_out = _run_backend(X, y, feats, labels, seed)
    return res, bk_surv, bk_out


def test_screening_survivors_identical(both):
    res, bk_surv, _ = both
    pkg = {m: sorted(res.survivor_list[m]["features"]) for m in res.survivor_list}
    for m in bk_surv:
        assert pkg.get(m) == bk_surv[m], f"module {m} survivors differ"


def test_selected_set_identical(both):
    res, _, bk_out = both
    assert set(res.final_SHAP["feature_name"]) == set(bk_out["final_features"])


def test_stability_frequencies_identical(both):
    res, _, bk_out = both
    pkg = dict(res.stability_freq)
    bk = dict(zip(bk_out["final_features"], bk_out["stability_freq"]))
    for f in set(pkg) & set(bk):
        assert abs(pkg[f] - bk[f]) == 0.0, f"freq differs for {f}"


def test_shap_matrix_identical(both):
    res, _, bk_out = both
    pkg_ord = list(res.final_SHAP["feature_name"])
    bk_ord = list(bk_out["final_features"])
    if set(pkg_ord) != set(bk_ord):
        pytest.skip("selected sets differ (covered by another test)")
    idx = [bk_ord.index(f) for f in pkg_ord]
    bk_sv = np.asarray(bk_out["shap_matrix"])[:, idx]
    assert np.abs(res.shap_obj["shapley_values"] - bk_sv).max() == 0.0
