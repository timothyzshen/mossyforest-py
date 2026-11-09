"""
Utility functions for MossyForest: environment checks (BLAS, Numba) and
inner thread pool control.

Thread pool management to prevent nested parallelism oversubscription.

When the replicate / module loops are parallelized across `num_processors`
workers, each inner fit (RF tree building, BLAS calls in cfRes/PCA) would also
spawn its own threads: N_workers x M_threads on N cores causes cache thrash and
context switch overhead that can erase the parallel gain.

`thread_limit_ctx` caps the inner thread pools (BLAS + OpenMP) during parallel
sections. It uses threadpoolctl, which ships as a scikit-learn dependency, so no
new requirement is introduced.
"""

from __future__ import annotations

import contextlib


def check_blas() -> dict:
    """
    Report the BLAS/LAPACK backend linked to NumPy.

    For best performance, NumPy should be backed by MKL or OpenBLAS,
    not the reference (slow) BLAS.  Install via:
        conda install numpy "blas=*=mkl"     # Intel MKL
        conda install numpy "blas=*=openblas"  # OpenBLAS

    Returns
    -------
    info : dict with keys "blas_opt_info", "blas_mkl_info", "openblas_info"
    """
    import numpy as np

    config = np.__config__
    info = {
        "numpy_version": np.__version__,
        "blas_opt":    getattr(config, "blas_opt_info",    {}) or {},
        "blas_mkl":    getattr(config, "blas_mkl_info",    {}) or {},
        "openblas":    getattr(config, "openblas_info",     {}) or {},
    }

    if info["blas_mkl"]:
        backend = "MKL (optimal)"
    elif info["openblas"]:
        backend = "OpenBLAS (good)"
    elif info["blas_opt"]:
        backend = "Other BLAS"
    else:
        backend = "Reference BLAS (slow, consider installing MKL or OpenBLAS)"

    info["backend"] = backend
    print(f"NumPy {info['numpy_version']}, BLAS backend: {backend}")
    return info


def check_numba() -> bool:
    """Return True if Numba is installed and functional."""
    try:
        import numba  # noqa: F401
        print(f"Numba {numba.__version__} available, shadow permutation will be JIT compiled.")
        return True
    except ImportError:
        print("Numba not installed, using NumPy fallback (pip install numba for speedup).")
        return False


def resolve_blas_threads(blas_threads, num_processors: int):
    """
    Decide the inner thread cap.

    blas_threads=None (auto):
        - num_processors > 1 (outer parallel active), so cap inner pools to 1
        - num_processors == 1, so None (leave inner threads unrestricted so a
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

    limit=None is a no op (do not touch thread pools).
    limit=int  caps all native thread pools to that many threads.
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
