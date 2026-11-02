"""
Smoke tests for every plotting function.

Renders each plot on a fitted MossyForestResult using the non-interactive
Agg backend and asserts it runs without raising. plt.show is stubbed so nothing
blocks, and figures are closed after each test to bound memory.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")   # headless — must precede pyplot import
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from mossyforest import (
    mf, screen_control, select_control,
    plot_importance, plot_signed_importance, plot_waterfall, plot_force,
    plot_stability, plot_stability_elbow,
    plot_potential_interactions, plot_decisions,
    plot_modules, plot_elbow, plot_soft_threshold,
)


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    """Stub plt.show and always close figures so nothing blocks or leaks."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def fitted():
    """A fitted result with several stable features (needs >=2 for interactions)."""
    rng = np.random.default_rng(0)
    n = 200
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    X = np.hstack([
        np.column_stack([z1 + rng.standard_normal(n) * 0.3 for _ in range(10)]),
        np.column_stack([z2 + rng.standard_normal(n) * 0.3 for _ in range(10)]),
        rng.standard_normal((n, 8)),
    ])
    y = (z1 - z2 > 0).astype(int)
    feats = ([f"A{i}" for i in range(10)] + [f"B{i}" for i in range(10)]
             + [f"ind{i}" for i in range(8)])
    mm = dict(zip(feats, ["A"] * 10 + ["B"] * 10 + ["unassigned"] * 8))
    sc = screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=50)
    sl = select_control(n_reps=15, min_ntree=50, pi_thr=0.35,
                        shadow_mode="split", unassigned_modules=["unassigned"])
    res = mf(pd.DataFrame(X, columns=feats), y, module_membership=mm,
             screen_params=sc, select_params=sl, final_ntree=50,
             verbose=0, seed=42)
    assert len(res.stability_freq) >= 2, "fixture needs >=2 stable features"
    return res


# ── SHAP-summary based plots ──────────────────────────────────────────────────

def test_plot_importance_bar(fitted):
    ax = plot_importance(fitted, kind="bar")
    assert ax is not None


def test_plot_importance_beeswarm(fitted):
    plot_importance(fitted, kind="beeswarm")   # calls shap.summary_plot + show


def test_plot_importance_both(fitted):
    plot_importance(fitted, kind="both")


def test_plot_importance_bad_kind(fitted):
    with pytest.raises(ValueError, match="kind"):
        plot_importance(fitted, kind="nope")


def test_plot_signed_importance(fitted):
    ax = plot_signed_importance(fitted)
    assert ax is not None


# ── per-observation plots ─────────────────────────────────────────────────────

def test_plot_waterfall_single(fitted):
    plot_waterfall(fitted, row_id=0)


def test_plot_waterfall_multi(fitted):
    plot_waterfall(fitted, row_id=[0, 1, 2])   # averaged


def test_plot_force_single(fitted):
    plot_force(fitted, row_id=0)


# ── stability plots ───────────────────────────────────────────────────────────

def test_plot_stability(fitted):
    ax = plot_stability(fitted)
    assert ax is not None


def test_plot_stability_elbow(fitted):
    plot_stability_elbow(fitted)


def test_plot_stability_elbow_labels(fitted):
    plot_stability_elbow(fitted, show_labels=True)


# ── interaction / decision / module / elbow ───────────────────────────────────

def test_plot_potential_interactions(fitted):
    plot_potential_interactions(fitted)


def test_plot_decisions(fitted):
    plot_decisions(fitted)


def test_plot_decisions_highlight(fitted):
    plot_decisions(fitted, highlight=[0, 5, 10], geom_point=True)


def test_plot_modules(fitted):
    plot_modules(fitted)


def test_plot_elbow(fitted):
    plot_elbow(fitted)


def test_plot_elbow_no_color(fitted):
    plot_elbow(fitted, color=False)


# ── soft-threshold (takes a WGCNA fit table, not a result) ────────────────────

def test_plot_soft_threshold():
    datout = pd.DataFrame({
        "Power":    list(range(1, 11)),
        "SFT.R.sq": [0.1, 0.3, 0.5, 0.7, 0.82, 0.88, 0.9, 0.91, 0.92, 0.93],
        "mean.k.":  [50, 40, 32, 25, 20, 16, 13, 11, 9, 8],
    })
    fig = plot_soft_threshold(datout, chosen_power=6, r2_threshold=0.85)
    assert fig is not None
