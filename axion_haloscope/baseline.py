# axion_haloscope/baseline.py
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter

def remove_baseline(
    raw_spectrum: np.ndarray,
    window_length: int = 501,
    polyorder: int = 4,
    subtract_one: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SG-fit baseline, then return processed spectrum:
      processed = (raw / baseline) - 1    (if subtract_one=True)
    Also returns the estimated baseline.
    """
    n = len(raw_spectrum)
    wl = window_length | 1                      # force odd
    wl = min(wl, n - ((n + 1) & 1))            # <= n and odd
    base = savgol_filter(raw_spectrum, wl, polyorder, mode="mirror")
    base = np.where(base <= 0, np.median(base[base > 0]), base)
    spec = raw_spectrum / base
    if subtract_one:
        spec = spec - 1.0
    return spec.astype(float), base.astype(float)

def mask_bins(spec: np.ndarray, bad_indices: np.ndarray | list[int]) -> np.ndarray:
    out = spec.copy()
    out[np.asarray(bad_indices, int)] = 0.0
    return out
