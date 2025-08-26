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
    combined = np.zeros(total_rf_bins, float)
    wsum     = np.zeros(total_rf_bins, float)
    counts   = np.zeros(total_rf_bins, int)

    if per_spec_sigma is None:
        per_spec_sigma = [max(np.std(s), 1e-6) for s in processed_spectra]

    for s, idx, sig in zip(processed_spectra, rf_index_map, per_spec_sigma):
        w = 1.0 / (sig*sig)
        combined[idx] += w * s
        wsum[idx]     += w
        counts[idx]   += 1

    nz = wsum > 0
    out = np.zeros_like(combined)
    out[nz] = combined[nz] / wsum[nz]

    sigma = np.full_like(combined, np.inf)
    sigma[nz] = np.sqrt(1.0 / wsum[nz])
    return out, sigma, counts
