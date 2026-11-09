"""
Tests for the Phase-1 (screening) cache.

Verifies:
  - a second run differing only in select_params hits the cache
  - cached survivors are byte-identical (Phase 1 not re-run)
  - changing screen_params / seed / X / y misses the cache
  - both memory (True) and on-disk (path) backends work
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import mossyforest.core as core
from mossyforest import mf, screen_control, select_control, clear_memory_cache
from mossyforest import screen as cache_mod


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 150
    z = rng.standard_normal(n)
    X = np.column_stack(
        [z + rng.standard_normal(n) * 0.3 for _ in range(8)]
        + [rng.standard_normal(n) for _ in range(8)]
    )
    y = (z > 0).astype(int)
    feats = [f"A{i}" for i in range(8)] + [f"ind{i}" for i in range(8)]
    mm = dict(zip(feats, ["A"] * 8 + ["unassigned"] * 8))
    return pd.DataFrame(X, columns=feats), y, mm


def _sf(data, cache, n_reps, seed=42, sp=None):
    X, y, mm = data
    sc = sp or screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=30)
    sl = select_control(n_reps=n_reps, min_ntree=30, pi_thr=0.4,
                        shadow_mode="split", unassigned_modules=["unassigned"])
    return mf(X, y, module_membership=mm, screen_params=sc, select_params=sl,
              final_ntree=30, verbose=0, seed=seed, screen_cache=cache)


def setup_function():
    clear_memory_cache()


# ── memory cache: hit on differing select_params ──────────────────────────────

def test_memory_cache_hit_skips_screening(data):
    # first run: miss, so screening runs once
    _sf(data, cache=True, n_reps=8)
    assert cache_mod.STATS["misses"] == 1
    assert cache_mod.STATS["hits"] == 0

    # second run differs ONLY in select_params (n_reps), so should hit
    with patch("mossyforest.core.run_screen_rfe",
               side_effect=AssertionError("screening should not re-run on a cache hit")):
        _sf(data, cache=True, n_reps=20)
    assert cache_mod.STATS["hits"] == 1


def test_cached_survivors_identical(data):
    r1 = _sf(data, cache=True, n_reps=8)
    r2 = _sf(data, cache=True, n_reps=8)   # identical inputs, so cache hit
    s1 = {m: list(df["features"]) for m, df in r1.survivor_list.items()}
    s2 = {m: list(df["features"]) for m, df in r2.survivor_list.items()}
    assert s1 == s2


# ── cache misses on relevant changes ──────────────────────────────────────────

def test_changing_seed_misses(data):
    _sf(data, cache=True, n_reps=8, seed=1)
    _sf(data, cache=True, n_reps=8, seed=2)
    assert cache_mod.STATS["misses"] == 2
    assert cache_mod.STATS["hits"] == 0


def test_changing_screen_params_misses(data):
    _sf(data, cache=True, n_reps=8,
        sp=screen_control(drop_fraction=0.3, keep_fraction=0.5, min_ntree=30))
    _sf(data, cache=True, n_reps=8,
        sp=screen_control(drop_fraction=0.3, keep_fraction=0.6, min_ntree=30))
    assert cache_mod.STATS["misses"] == 2


def test_changing_X_misses(data):
    X, y, mm = data
    _sf(data, cache=True, n_reps=8)
    X2 = X.copy(); X2.iloc[0, 0] += 1.0
    _sf((X2, y, mm), cache=True, n_reps=8)
    assert cache_mod.STATS["misses"] == 2


# ── on-disk cache ─────────────────────────────────────────────────────────────

def test_disk_cache(tmp_path, data):
    d = str(tmp_path / "sfcache")
    _sf(data, cache=d, n_reps=8)
    files = list((tmp_path / "sfcache").glob("screen_*.pkl"))
    assert len(files) == 1

    # second run (new process would reload from disk); here just confirm hit
    with patch("mossyforest.core.run_screen_rfe",
               side_effect=AssertionError("should load from disk, not re-screen")):
        _sf(data, cache=d, n_reps=25)


# ── no caching by default ─────────────────────────────────────────────────────

def test_no_cache_by_default(data):
    # screen_cache=None, so screening always runs; STATS untouched
    before = dict(cache_mod.STATS)
    _sf(data, cache=None, n_reps=8)
    assert cache_mod.STATS == before
