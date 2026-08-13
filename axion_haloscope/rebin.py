# axion_haloscope/rebin.py
"""
Rebin
=====
"""

from __future__ import annotations
from typing import Tuple, List
import numpy as np

def rebin_ml(
    combined: np.ndarray,
    sigma_c: np.ndarray,
    rebin_width: int = 10
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    ML rebin by factor rebin_width (non-overlapping) for noise stationarity and speed.
    Returns data_r (rebinned data), sigma_r (rebinned sigma), and spans.

    Parameters
    ==========
    combined: 1D array
        Array of averaged spectra
    sigma_c: 1D array
        sigmas of `combined`
    rebin_width: int
        Number of old bins about to be rebinned into 1 bin

    Returns
    =======
    data_r: 1D array
        Rebinned data, where each bin is the weighted average of `rebin_width` bins
    sigma_r: 1D array
        Sigmas of `data_r`
    spans: List
        List of which old bins have been converted into which new bins (uses indexes)
    """
    n   = len(combined)
    n_r = n // rebin_width
    data_r  = np.zeros(n_r, float)
    sigma_r  = np.zeros(n_r, float)
    spans = []
    for r in range(n_r):
        a, b = r*rebin_width, r*rebin_width + rebin_width
        spans.append((a, b))
        seg, sseg = combined[a:b], sigma_c[a:b]
        w = np.where(np.isfinite(sseg), 1.0/(sseg**2 + 1e-18), 0.0)
        w_sum = w.sum()
        data_r[r] = (w*seg).sum() / (w_sum + 1e-18) if w_sum>0 else 0.0
        sigma_r[r] = np.sqrt(1.0 / (w_sum + 1e-18)) if w_sum>0 else np.inf
    return data_r, sigma_r, spans

def grand_spectrum_ml(
    data_r: np.ndarray,
    sigma_r: np.ndarray,
    lineshape_template: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlapping matched-filter: for each center, ML-combine k=len(lineshape_template) rebinned bins.
    Returns data_g (grand), sigma_g (its sigma). Undefined (edges) are left at 0/inf.

    Parameters
    ==========
    data_r: 1D array
        Rebinned data
    sigma_r: 1D array
        Rebinned data sigmas
    lineshape_template: 1D array
        Axion lineshape

    Returns
    =======
    data_g: 1D array
        Grand Spectrum, after rebinned data has been fited to the lineshape template
    sigma_g: 1D array
        Sigmas on data_g
    """
    k = len(lineshape_template)
    n = len(data_r)
    data_g = np.zeros(n, float)
    sigma_g = np.full(n, np.inf, float)
    for r in range(n - k + 1):
        seg, sseg = data_r[r:r+k], sigma_r[r:r+k]
        denom = (lineshape_template*lineshape_template / (sseg**2 + 1e-18)).sum()
        if denom > 0:
            num = (lineshape_template * seg / (sseg**2 + 1e-18)).sum()
            c = r + k//2
            data_g[c] = num / (denom + 1e-18)
            sigma_g[c] = np.sqrt(1.0 / (denom + 1e-18))
    return data_g, sigma_g
