"""
Phase-1 (screening) result cache.

Screening depends only on X, y, module membership, screen_params, nodesize,
classification, seed, max_modsize, and the SHAP backend — NOT on select_params.
The common tuning / cross-validation workflow re-runs mf() many times varying
only select_params (pi_thr, shadow_percentile, n_reps, ...), so caching Phase 1
lets those runs skip screening entirely.

Two backends, chosen via the `screen_cache` argument to mf():
  None / False  → no caching (default; current behaviour)
  True          → in-process memory cache (persists for the Python session)
  str / Path    → on-disk cache directory (persists across sessions)

The cache is keyed on a hash of everything that affects screening output, so a
hit returns byte-identical survivors and a change to any relevant input misses.
`use_numba` and `num_processors` are deliberately excluded from the key: they do
not change screening results (only speed / determinism-preserving), so a cached
result is valid regardless of their values.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

# in-process cache: {key_hex: screen_result_dict}
_MEM_CACHE: dict[str, dict] = {}

# counters for observability / tests
STATS = {"hits": 0, "misses": 0}


def compute_screen_key(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    feature_names: list,
    module_assign: list,
    module_list: list,
    screen_params,
    nodesize: int,
    classification: bool,
    seed: int,
    max_modsize: int,
    backend: str,
) -> str:
    """Deterministic hash of all inputs that affect screening output."""
    h = hashlib.blake2b(digest_size=32)

    # data: exact bytes + shape + dtype
    Xc = np.ascontiguousarray(X_arr)
    h.update(str(Xc.dtype).encode()); h.update(str(Xc.shape).encode())
    h.update(Xc.tobytes())

    yc = np.ascontiguousarray(np.asarray(y_arr))
    h.update(str(yc.dtype).encode()); h.update(yc.tobytes())

    # feature names + module structure (order matters)
    h.update("\x00".join(map(str, feature_names)).encode())
    h.update("\x00".join(map(str, module_assign)).encode())
    h.update("\x00".join(map(str, module_list)).encode())

    # screen params (all fields that steer the RFE)
    sp = screen_params
    for v in (sp.drop_fraction, sp.keep_fraction, sp.mtry_factor,
              sp.min_ntree, sp.ntree_factor):
        h.update(repr(v).encode())

    # scalar execution inputs that affect results
    for v in (nodesize, bool(classification), int(seed), int(max_modsize), str(backend)):
        h.update(repr(v).encode())

    return h.hexdigest()


def load_cached(key: str, cache) -> dict | None:
    """Return a cached screen result for `key`, or None on miss."""
    if cache is True:
        return _MEM_CACHE.get(key)
    # disk cache: cache is a directory path
    fp = Path(cache) / f"screen_{key}.pkl"
    if fp.exists():
        try:
            with open(fp, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None
    return None


def store_cached(key: str, cache, result: dict) -> None:
    """Persist a screen result under `key`."""
    if cache is True:
        _MEM_CACHE[key] = result
        return
    d = Path(cache)
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"screen_{key}.pkl"
    tmp = fp.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(fp)   # atomic


def clear_memory_cache() -> None:
    """Drop the in-process screening cache (does not touch on-disk caches)."""
    _MEM_CACHE.clear()
    STATS["hits"] = 0
    STATS["misses"] = 0
