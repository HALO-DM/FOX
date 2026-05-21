# axion_haloscope/combine.py
from __future__ import annotations
import numpy as np

def combine_ml(
    processed_spectra: list[np.ndarray],
    rf_index_map: list[np.ndarray],
    total_rf_bins: int,
    per_spec_sigma: list[float] | None = None,
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

    for s, idx, sig in zip(processed_spectra, rf_index_map, per_spec_sigma):
        w = 1.0 / (sig*sig) # Weight is inverse variance
        combined[idx] += w * s # Add the weighted spectrum to the combined array
        wsum[idx]     += w 
        counts[idx]   += 1 

    nz = wsum > 0
    out = np.zeros_like(combined) 
    out[nz] = combined[nz] / wsum[nz] # output weighted average

    sigma = np.full_like(combined, np.inf)
    sigma[nz] = np.sqrt(1.0 / wsum[nz]) # uncertainty of the weighted average
    return out, sigma, counts
