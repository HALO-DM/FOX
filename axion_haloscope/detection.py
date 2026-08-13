# axion_haloscope/detection.py
"""
Detection
=========

Calculating the threshold for detecting candiates, and then finding the candidates using a peak
finding algorithm
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm


def threshold_for_detection(
    target_snr: float,
    confidence: float,
    n_trials: int = 1
) -> float:
    """
    Return the z-threshold (σ units) so that the global false-alarm probability
    across n_trials is 1 - confidence.

    Parameters
    ----------
    target_snr : float
        (Kept for API compatibility; not used in the tail calculation.)
    confidence : float
        Desired global confidence (e.g. 0.95, 0.99).
    n_trials : int
        Number of independent trials (look-elsewhere factor).

    Returns
    -------
    theta : float
        Right-tail z cut. Monotone in 'confidence' and 'n_trials'.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")

    # Global → per-trial false-alarm rate
    alpha_global = 1.0 - confidence
    n_trials = max(1, int(n_trials))
    alpha_per = 1.0 - (1.0 - alpha_global)**(1.0 / n_trials)

    # Right-tail threshold: P(Z > theta) = alpha_per
    theta = norm.isf(alpha_per)
    return float(theta)


def find_candidates(
    grand_spectrum: np.ndarray,
    sigma_gs: np.ndarray,
    thresh: float,
    min_separation: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find candidates based on the spectra its errors that are above a given threshold.
    If there are multiple points in a row above threshold, makes sure that only the peak is 
    selected (from min_separation).

    Parameters
    ==========
    grand_spectrum: 1D array
        The grand spectrum, calculated from combined spectra and rebinned before fitted to a 
        specific lineshape
    sigma_gs: 1D array
        Weighted uncertainty on `grand_spectrum`
    thresh: float
        Threshold for candidate deteection - any points above threshold are potential candidates
    min_separation: int
        Minimum seperation between points above the threshold to register as a peak

    Returns
    =======
    merged: 1D array
        Array of candidates, where muliple peaks have been merged together if within
        `min_seperation`
    z: 1D array
        Array of grand_spectrum/sigma_gs >= thresh

    See Also
    ========
    axion_haloscope.lineshape
    axion_haloscope.combined
    axion_haloscope.rebin
        see how grand spectra is calculated

    """
    z = np.zeros_like(grand_spectrum)
    m = np.isfinite(sigma_gs) & (sigma_gs > 0)
    z[m] = grand_spectrum[m] / sigma_gs[m]
    raw = np.where(z >= thresh)[0].tolist()
    if not raw:
        return [], z
    merged = []
    i = 0
    while i < len(raw):
        j = i
        group = [raw[i]]
        while j + 1 < len(raw) and (raw[j+1] - raw[j]) <= min_separation:
            j += 1
            group.append(raw[j])
        merged.append(max(group, key=lambda k: z[k]))
        i = j + 1
    return merged, z
