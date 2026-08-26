"""
Utility functions for BonsaiForest.
"""

from __future__ import annotations


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
        backend = "Reference BLAS (slow — consider installing MKL or OpenBLAS)"

    info["backend"] = backend
    print(f"NumPy {info['numpy_version']} — BLAS backend: {backend}")
    return info


def check_numba() -> bool:
    """Return True if Numba is installed and functional."""
    try:
        import numba  # noqa: F401
        print(f"Numba {numba.__version__} available — shadow permutation will be JIT-compiled.")
        return True
    except ImportError:
        print("Numba not installed — using NumPy fallback (pip install numba for speedup).")
        return False
