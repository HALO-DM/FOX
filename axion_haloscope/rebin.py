# axion_haloscope/rebin.py
from __future__ import annotations
import numpy as np

def rebin_ml(combined: np.ndarray, sigma_c: np.ndarray, C: int = 10):
    """
    ML rebin by factor C (non-overlapping) for noise stationarity and speed.
    Returns Dr (rebinned data), sr (rebinned sigma), and spans.
    """
    n   = len(combined)
    n_r = n // C
    Dr  = np.zeros(n_r, float)
    sr  = np.zeros(n_r, float)
    spans = []
    for r in range(n_r):
        a, b = r*C, r*C + C
        spans.append((a, b))
        seg, sseg = combined[a:b], sigma_c[a:b]
        w = np.where(np.isfinite(sseg), 1.0/(sseg**2 + 1e-18), 0.0)
        W = w.sum()
        Dr[r] = (w*seg).sum() / (W + 1e-18) if W>0 else 0.0
        sr[r] = np.sqrt(1.0 / (W + 1e-18)) if W>0 else np.inf
    return Dr, sr, spans

def axion_template_gaussian(K: int, bin_width_hz: float, sigma_hz: float) -> np.ndarray:
    """
    K-bin Gaussian template (unit sum) with physical width sigma_hz and bin spacing bin_width_hz.
    """
    centers = np.arange(K) - (K - 1)/2.0
    hz = centers * bin_width_hz
    L = np.exp(-0.5 * (hz/sigma_hz)**2)
    L /= L.sum() if L.sum() > 0 else 1.0
    return L

def grand_spectrum_ml(Dr: np.ndarray, sr: np.ndarray, Lq: np.ndarray):
    """
    Overlapping matched-filter: for each center, ML-combine K=len(Lq) rebinned bins.
    Returns Dg (grand), sg (its sigma). Undefined (edges) are left at 0/inf.
    """
    K = len(Lq); n = len(Dr)
    Dg = np.zeros(n, float)
    sg = np.full(n, np.inf, float)
    for r in range(n - K + 1):
        seg, sseg = Dr[r:r+K], sr[r:r+K]
        denom = (Lq*Lq / (sseg**2 + 1e-18)).sum()
        if denom > 0:
            num = (Lq * seg / (sseg**2 + 1e-18)).sum()
            c = r + K//2
            Dg[c] = num / (denom + 1e-18)
            sg[c] = np.sqrt(1.0 / (denom + 1e-18))
    return Dg, sg
