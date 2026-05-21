# axion_haloscope/combine.py
from __future__ import annotations
from typing import Optional
import numpy as np

def lorentzian_height(f0: float, f: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the height of a Lorentzian profile at frequency f, given a center frequency f0 and width sigma.
    """
    return 1.0 / (1.0 + ((f - f0) / sigma)**2)

def combine_ml(
    processed_spectra: list[np.ndarray],
    rf_index_map: list[np.ndarray],
    total_rf_bins: int,
    per_spec_sigma: list[float] | None = None,
    lorentzian_weight: Optional[bool] = False,
    lorentz_params: list[tuple[float, float]] | None = None,  # (f0_hz, bw_hz) per spectrum
    spec_freqs: list[np.ndarray] | None = None,               # actual RF freqs per spectrum
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Put all spectra on the common RF grid and ML-average overlaps.
    Returns combined, sigma (1/sqrt(sum w)), and counts per RF bin.
    """
    combined = np.zeros(total_rf_bins, float) # Initialise arrays to hold the combined spectrum
    wsum     = np.zeros(total_rf_bins, float) # the sum of weights
    counts   = np.zeros(total_rf_bins, int) # and the count of contributions

    if per_spec_sigma is None: # If no sigma provided, use std
        per_spec_sigma = [max(np.std(s), 1e-6) for s in processed_spectra]

    for s, idx, sig, freqs, (f0, bw) in zip(
        processed_spectra, rf_index_map, per_spec_sigma, spec_freqs, lorentz_params
    ):
        if lorentzian_weight:
            h = lorentzian_height(f0=f0, f=freqs, sigma=bw)  # shape matches s
            w = h / (sig * sig)
        else:
            w = 1.0 / (sig * sig) # uniform weight
        np.add.at(combined, idx, w * s)
        np.add.at(wsum,     idx, w)
        np.add.at(counts,   idx, 1)

        nz = wsum > 0
        out = np.zeros_like(combined) 
        out[nz] = combined[nz] / wsum[nz] # output weighted average

        sigma = np.full_like(combined, np.inf)
        sigma[nz] = np.sqrt(1.0 / wsum[nz]) # uncertainty of the weighted average
    return out, sigma, counts
