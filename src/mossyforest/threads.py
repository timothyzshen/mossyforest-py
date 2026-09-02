"""
Thread-pool management to prevent nested-parallelism oversubscription.

When the bootstrap / module loops are parallelized across `num_processors`
workers, each inner fit (RF tree building, BLAS calls in cfRes/PCA) would also
spawn its own threads — N_workers × M_threads on N cores → cache thrash and
context-switch overhead that can erase the parallel gain.

`thread_limit_ctx` caps the inner thread pools (BLAS + OpenMP) during parallel
sections. Uses threadpoolctl, which ships as a scikit-learn dependency, so no
new requirement is introduced.
"""

from __future__ import annotations

import contextlib


def resolve_blas_threads(blas_threads, num_processors: int):
    """
    Decide the inner thread cap.

    blas_threads=None (auto):
        - num_processors > 1 (outer parallel active) → cap inner pools to 1
        - num_processors == 1 → None (leave inner threads unrestricted so a
          single large fit can use all cores)
    blas_threads=int:
        explicit cap regardless of num_processors.
    """
    if blas_threads is not None:
        return int(blas_threads)
    return 1 if num_processors and num_processors > 1 else None


@contextlib.contextmanager
def thread_limit_ctx(limit):
    """
    Limit BLAS/OpenMP thread pools within the context.

    limit=None → no-op (do not touch thread pools).
    limit=int  → cap all native thread pools to that many threads.
    """
    if limit is None:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:   # pragma: no cover - threadpoolctl ships with sklearn
        yield
        return
    with threadpool_limits(limits=int(limit)):
        yield
