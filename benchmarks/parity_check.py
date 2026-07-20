"""
R <-> Python end-to-end parity harness.

Runs the standalone Python package and the R package's sf(backend="python")
on IDENTICAL data + seed, with R's reticulate pinned to the same Python
interpreter as the package (so any difference is algorithmic, not a
library-version artifact). Reports set overlap and per-feature frequency
agreement.

Usage:  python parity_check.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shapleyforest import sf, screen_control, select_control

SEED   = 42
SF_DEV = "/Users/tzshen/Documents/Github/sf_dev"
SCRATCH = Path(os.environ.get("SF_SCRATCH", "/tmp"))


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    n = 200
    lat = [rng.standard_normal(n) for _ in range(3)]
    cols, feats, labels = [], [], []
    for m, z in enumerate(lat):
        for j in range(10):
            cols.append(z + rng.standard_normal(n) * 0.3)
            feats.append(f"M{m}_{j}"); labels.append(f"M{m}")
    for j in range(10):
        cols.append(rng.standard_normal(n))
        feats.append(f"ind{j}"); labels.append("independent")
    X = pd.DataFrame(np.column_stack(cols), columns=feats)
    y = ((lat[0] + lat[1] - lat[2]) > 0).astype(int)
    return X, y, dict(zip(feats, labels))


def main():
    X, y, mm = make_data()

    # ── Python package ────────────────────────────────────────────────────────
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=100)
    sl = select_control(n_boots=30, min_ntree=100, pi_thr=0.4,
                        shadow_mode="split", indep_modules=["independent"])
    res = sf(X, y, module_membership=mm, screen_params=sc, select_params=sl,
             final_ntree=100, num_processors=1, verbose=0, seed=SEED)
    py_sel  = list(res.final_SHAP["feature_name"])
    py_freq = dict(res.stability_freq)
    print(f"Python package: {len(py_sel)} selected")

    # ── write shared data for R ───────────────────────────────────────────────
    data_csv = SCRATCH / "parity_data.csv"
    df = X.copy(); df["__y__"] = y
    df.to_csv(data_csv, index=False)
    pd.DataFrame({"feature": list(mm), "module": list(mm.values())}).to_csv(
        str(data_csv).replace(".csv", "_mods.csv"), index=False)

    # ── R package (reticulate pinned to THIS python) ─────────────────────────
    out_json = SCRATCH / "parity_R.json"
    env = dict(os.environ, RETICULATE_PYTHON=sys.executable)
    r = subprocess.run(
        ["Rscript", str(Path(__file__).parent / "parity_check.R"),
         str(data_csv), str(out_json), str(SEED)],
        capture_output=True, text=True, env=env, timeout=900,
    )
    if not out_json.exists():
        print("R run FAILED:\n", r.stderr[-1500:]); return

    R = json.loads(out_json.read_text())
    r_sel  = R["selected"]
    r_freq = {k: float(v) for k, v in R["freq"].items()}
    print(f"R reticulate:   {len(r_sel)} selected")

    # ── compare ───────────────────────────────────────────────────────────────
    a, b = set(py_sel), set(r_sel)
    inter, union = a & b, a | b
    jaccard = len(inter) / len(union) if union else 1.0

    print("\n── selected-set agreement ──")
    print(f"  Python-only: {sorted(a - b)}")
    print(f"  R-only:      {sorted(b - a)}")
    print(f"  shared:      {len(inter)}/{len(union)}   Jaccard = {jaccard:.3f}")

    common = sorted(inter)
    if common:
        diffs = [abs(py_freq.get(f, 0) - r_freq.get(f, 0)) for f in common]
        print("\n── stability-frequency agreement (shared features) ──")
        print(f"  max |Δfreq| = {max(diffs):.4f}   mean |Δfreq| = {np.mean(diffs):.4f}")

    print("\nVERDICT:",
          "STRONG parity" if jaccard >= 0.8 else
          "PARTIAL parity" if jaccard >= 0.5 else "WEAK parity")


if __name__ == "__main__":
    main()
