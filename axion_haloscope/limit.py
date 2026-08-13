# axion_haloscope/limit.py
"""
Limit
=====

Computes coupling limits and related functions
"""
from __future__ import annotations
import numpy as np

def compute_local_snr_template(
    sigma_r: np.ndarray,
    lineshape_template: np.ndarray
) -> np.ndarray:
    """
    local_snr(center) = sqrt( sum_i lineshape_template[i]^2 / sigma_r[i]^2 ) at each valid center
    (NaN at edges).

    Parametes
    =========
    sigma_r: 1D array
        Sigmas of rebinned data
    lineshape_template: 1D array
        Axion lineshape

    Returns
    =======
    local_snr: 1D array
        Defined above
    """
    n, k = len(sigma_r), len(lineshape_template)
    local_snr = np.full(n, np.nan)
    for r in range(n - k + 1):
        segs = sigma_r[r:r+k]
        denom = (lineshape_template*lineshape_template / (segs**2 + 1e-18)).sum()
        if denom > 0:
            local_snr[r + k//2] = np.sqrt(denom)
    return local_snr

def coupling_limit(
    local_snr: np.ndarray,
    target_snr: float = 5.0,
    g0: float = 1e-13,
    snr_efficiency: float = 0.90
) -> np.ndarray:
    """
    g_min(f) = g0 * sqrt( target_snr / (snr_efficiency * local_snr) ).

    Parameters
    ==========
    local_snr: 1D array
        See `compute_local_snr_template`
    target_snr: float
        The ideal snr
    g0: float
        Relative Coupling constant, used as a reference in exclusion plot
    snr_efficiency: float
        Assumed snr efficiency

    Returns
    =======
    out: 1D array
        Coupling constants relative to g0
    """
    out = np.full_like(local_snr, np.nan, float)
    good = np.isfinite(local_snr) & (local_snr > 0)
    out[good] = g0 * np.sqrt(target_snr / (snr_efficiency * local_snr[good]))
    return out
