"""
Shared benchmark helpers for MossyForest speed comparisons.

Used by both benchmark.qmd and a plain `python bench_common.py` run.
Keeps the synthetic dataset identical to bench_r_reticulate.R so the
standalone-Python and R-reticulate timings are directly comparable.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mossyforest import mf, screen_control, select_control


def make_data(n: int = 300, seed: int = 0):
    """4 correlated modules (12 feats each) + 16 independent noise features."""
    rng = np.random.default_rng(seed)
    lat = [rng.standard_normal(n) for _ in range(4)]
    cols, feats, labels = [], [], []
    for m, z in enumerate(lat):
        for j in range(12):
            cols.append(z + rng.standard_normal(n) * 0.3)
            feats.append(f"M{m}_{j}")
            labels.append(f"M{m}")
    for j in range(16):
        cols.append(rng.standard_normal(n))
        feats.append(f"ind{j}")
        labels.append("unassigned")
    X = pd.DataFrame(np.column_stack(cols), columns=feats)
    y = ((lat[0] + lat[1] - lat[2]) > 0).astype(int)
    return X, y, dict(zip(feats, labels))


def _params():
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=150)
    sl = select_control(n_boots=30, min_ntree=150, pi_thr=0.4,
                        shadow_mode="split", unassigned_modules=["unassigned"])
    return sc, sl


def time_config(X, y, mm, label, access_interaction=False, seed=42, **kw):
    """Time one mf() configuration. Returns (label, seconds, n_stable)."""
    sc, sl = _params()
    t = time.time()
    r = mf(X, y, module_membership=mm, screen_params=sc, select_params=sl,
           final_ntree=150, verbose=0, seed=seed, **kw)
    if access_interaction and r.interaction_matrix is not None:
        pass   # force the lazy interaction matrix to materialize
    dt = time.time() - t
    return label, dt, len(r.stability_freq)


def run_python_suite(X, y, mm):
    """Benchmark the standalone-Python package across configurations."""
    # warmup: trigger imports + any JIT compile so timings are steady-state
    sc, sl = _params()
    mf(X, y, module_membership=mm, screen_params=sc, select_params=sl,
       final_ntree=50, verbose=0, seed=1, num_processors=1)

    rows = [
        time_config(X, y, mm, "py: baseline (n_jobs=1)", num_processors=1),
        time_config(X, y, mm, "py: parallel (n_jobs=2)", num_processors=2),
        time_config(X, y, mm, "py: parallel (n_jobs=4)", num_processors=4),
        time_config(X, y, mm, "py: n_jobs=1 + numba", num_processors=1, use_numba=True),
        time_config(X, y, mm, "py: parallel + numba (n_jobs=4)", num_processors=4, use_numba=True),
        time_config(X, y, mm, "py: parallel(4) + interaction matrix",
                    num_processors=4, access_interaction=True),
    ]
    return rows


def run_r_reticulate(sf_dev_path: str, num_processors_list=(1, 4), seed=42):
    """Benchmark the R mf(..., backend='python') path via subprocess Rscript."""
    here   = Path(__file__).resolve().parent
    r_file = here / "bench_r_reticulate.R"
    rows = []
    for npx in num_processors_list:
        try:
            out = subprocess.run(
                ["Rscript", str(r_file), sf_dev_path, str(npx), str(seed)],
                capture_output=True, text=True, timeout=900,
            )
            line = next((ln for ln in out.stdout.splitlines()
                         if ln.startswith("RESULT")), None)
            if line is None:
                rows.append((f"R reticulate (num_processors={npx})", float("nan"),
                             f"FAILED: {out.stderr.strip()[-200:]}"))
            else:
                _, secs, nstab = line.split()
                rows.append((f"R reticulate (num_processors={npx})",
                             float(secs), int(nstab)))
        except Exception as e:   # noqa: BLE001
            rows.append((f"R reticulate (num_processors={npx})", float("nan"), f"ERROR: {e}"))
    return rows


def run_cache_sweep(X, y, mm, pi_values=(0.3, 0.4, 0.5, 0.6)):
    """
    Time a tuning sweep over pi_thr with and without screen_cache. Screening is
    identical across the sweep, so the cache lets every run after the first skip
    Phase 1. Returns (no_cache_seconds, cached_seconds, speedup).
    """
    from mossyforest import clear_memory_cache

    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=150)

    def one(cache, pi):
        sl = select_control(n_boots=30, min_ntree=150, pi_thr=pi,
                            shadow_mode="split", unassigned_modules=["unassigned"])
        mf(X, y, module_membership=mm, screen_params=sc, select_params=sl,
           final_ntree=150, verbose=0, seed=42, screen_cache=cache)

    # warmup
    one(None, 0.4)

    clear_memory_cache()
    t = time.time()
    for p in pi_values:
        one(None, p)
    no_cache = time.time() - t

    clear_memory_cache()
    t = time.time()
    for p in pi_values:
        one(True, p)
    cached = time.time() - t

    return no_cache, cached, (no_cache / cached if cached else float("nan"))


def print_table(rows, baseline_label="py: baseline (n_jobs=1)"):
    base = next((s for lbl, s, _ in rows if lbl == baseline_label and s == s), None)
    print(f"\n{'configuration':42s} {'time(s)':>9s} {'speedup':>8s} {'stable':>7s}")
    print("-" * 70)
    for lbl, secs, nstab in rows:
        sp = f"{base/secs:.2f}x" if (base and secs == secs and secs > 0) else "—"
        secs_s = f"{secs:.2f}" if secs == secs else "n/a"
        print(f"{lbl:42s} {secs_s:>9s} {sp:>8s} {str(nstab):>7s}")


def append_results(path, rows, n, p):
    """
    Append this run's rows to a CSV history so performance can be tracked over
    time. One row per configuration, tagged with an ISO timestamp, git commit,
    and dataset shape. Creates the file (with header) if it doesn't exist.
    """
    import csv
    import datetime
    import subprocess

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"

    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["timestamp", "commit", "n_samples", "n_features",
                        "configuration", "seconds", "n_stable"])
        for lbl, secs, nstab in rows:
            w.writerow([ts, commit, n, p, lbl,
                        f"{secs:.3f}" if secs == secs else "nan", nstab])
    print(f"\nAppended {len(rows)} rows to {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MossyForest speed benchmark")
    ap.add_argument("--out", metavar="CSV",
                    help="append results to this CSV history file")
    ap.add_argument("--mf-dev", default="/Users/tzshen/Documents/Github/sf_dev",
                    help="path to the R package source (for the reticulate comparison)")
    ap.add_argument("--no-r", action="store_true",
                    help="skip the R reticulate comparison")
    args = ap.parse_args()

    X, y, mm = make_data()
    print(f"dataset: {X.shape[0]} samples x {X.shape[1]} features "
          f"(4 corr modules + 16 independent)")
    rows = run_python_suite(X, y, mm)
    if not args.no_r:
        rows += run_r_reticulate(args.sf_dev)
    print_table(rows)
    if args.out:
        append_results(args.out, rows, X.shape[0], X.shape[1])
