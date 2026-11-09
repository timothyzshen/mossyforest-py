"""
In-process parity verification: standalone package vs. the R package's Python
backend (inst/python/mf_python_backend.py), run in ONE Python process on
identical in-memory data + seed + library stack.

This is the definitive parity check, it removes R, reticulate, CSV round-trips,
and stack differences, so any remaining difference is purely the two codebases.

It answers three questions:
  1. Does float32 (package) vs float64 (backend) change SHAP values?  No.
  2. Do the two pipelines agree phase by phase?                       Yes, exactly.
  3. Is the signed SHAP matrix identical?                              Yes (Δ = 0).

Run:  python benchmarks/parity_in_process.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Import the R package's Python backend directly. Point at the canonical
# published package (github.com/timothyzshen/mossyforest); its backend is
# byte-identical to the sf_dev working copy after the RNG alignment.
R_BACKEND_DIR = "/Users/tzshen/Documents/Github/mossyforest/inst/python"
sys.path.insert(0, R_BACKEND_DIR)
import mf_python_backend as bk   # noqa: E402

from mossyforest import mf, screen_control, select_control   # noqa: E402

SEED = 42


def make_data(n=200, seed=0):
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


def experiment_shap_dtype():
    """Q1: does float32 vs float64 X change SHAP on an identical model?"""
    import shap
    from sklearn.ensemble import RandomForestClassifier
    X, y, _, _ = make_data(seed=1)
    X64 = X[:, :12].astype(np.float64); X32 = X64.astype(np.float32)
    rf = RandomForestClassifier(n_estimators=150, max_features=4, random_state=42).fit(X64, y)

    def sv(Xin):
        r = np.asarray(shap.TreeExplainer(rf).shap_values(Xin))
        return r[:, :, 1] if r.ndim == 3 else r

    d = np.abs(sv(X64) - sv(X32))
    print("── Q1: SHAP on float64-X vs float32-X (same model) ──")
    print(f"   max |Δ SHAP| = {d.max():.3e}   (0 means dtype is irrelevant to SHAP)\n")


def run_backend(X, y, feats, labels):
    """Mirror what R's mf(backend='python') does: screen, assemble, then select."""
    module_list = list(dict.fromkeys(labels))
    scr = bk.run_screen_rfe(
        X_mat=X.astype(np.float64), y=y, feature_names=feats,
        module_assignments=labels, module_list=module_list,
        drop_fraction=0.3, keep_fraction=0.5, mtry_factor=1.0, ntree_factor=1,
        min_ntree=100, nodesize=1, classification=True, seed=SEED, n_jobs=1,
        max_modsize=5000)
    surv_mod = {}
    for m, sfeats in zip(module_list, scr["survivor_features"]):
        for f in sfeats:
            surv_mod[f] = m
    surv = set(surv_mod)
    # R's original-X column order: X[, names(X) %in% survivors]
    order = [f for f in feats if f in surv]
    Xs = X[:, [feats.index(f) for f in order]].astype(np.float64)
    sel = bk.run_select_rfe(
        X_surv_mat=Xs, y=y, feature_names_surv=order,
        mtry_factor=1.0, ntree_factor=1, min_ntree=100,
        nodesize=1, classification=True, seed=SEED, n_jobs=1,
        n_reps=30, pi_thr=0.4, shadow_mode="split", shadow_percentile=95,
        module_assignments_surv=[surv_mod[f] for f in order],
        unassigned_modules=["unassigned"], verbose=False)
    survivors = {m: sorted(sfeats) for m, sfeats in zip(module_list, scr["survivor_features"])}
    return survivors, sel


def main():
    experiment_shap_dtype()

    X, y, feats, labels = make_data()
    Xdf = pd.DataFrame(X, columns=feats)
    mm = dict(zip(feats, labels))

    # ── package ──
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=100)
    sl = select_control(n_reps=30, min_ntree=100, pi_thr=0.4,
                        shadow_mode="split", unassigned_modules=["unassigned"])
    res = mf(Xdf, y, module_membership=mm, screen_params=sc, select_params=sl,
             final_ntree=100, verbose=0, seed=SEED, num_processors=1)
    pkg_surv = {m: sorted(res.survivor_list[m]["features"]) for m in res.survivor_list}
    pkg_sel  = list(res.final_SHAP["feature_name"])
    pkg_freq = dict(res.stability_freq)
    pkg_sv   = res.shap_obj["shapley_values"]

    # ── backend (R's path) ──
    bk_surv, bk_out = run_backend(X, y, feats, labels)
    bk_sel  = list(bk_out["final_features"])
    bk_freq = dict(zip(bk_sel, bk_out["stability_freq"]))
    bk_sv   = np.asarray(bk_out["shap_matrix"])

    print("── Q2: phase-by-phase parity (package vs R backend, same process) ──")
    scr_ok = all(pkg_surv.get(m) == bk_surv.get(m) for m in bk_surv)
    print(f"   Phase 1 screening survivors identical : {scr_ok}")
    print(f"   Phase 2 selected set identical        : {set(pkg_sel) == set(bk_sel)} "
          f"({len(pkg_sel)} vs {len(bk_sel)})")
    shared = sorted(set(pkg_freq) & set(bk_freq))
    dfreq = max(abs(pkg_freq[f] - bk_freq[f]) for f in shared) if shared else 0.0
    print(f"   max |Δ stability-frequency|           : {dfreq:.4f}")

    # SHAP matrix (align columns by feature name)
    idx = [bk_sel.index(f) for f in pkg_sel]
    d = np.abs(pkg_sv - bk_sv[:, idx])
    print("\n── Q3: signed SHAP matrix (package float32 vs backend float64) ──")
    print(f"   max |Δ SHAP| = {d.max():.3e}   mean |Δ SHAP| = {d.mean():.3e}")

    exact = scr_ok and set(pkg_sel) == set(bk_sel) and dfreq == 0 and d.max() == 0
    print("\nVERDICT:", "EXACT PARITY ✓" if exact else "divergence detected, investigate")


if __name__ == "__main__":
    main()
