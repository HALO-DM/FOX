# axion_haloscope/combine.py
"""
Combine
=======

"""
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

    Parameters
    ----------
    processed_spectra: list of 1D arrays
        spectra that has been cleaned and normalised 
    rf_index_map: list of 1D arrays
        slices gives the integer indices into `rf_grid` corresponding to spectrum 
        `i`'s bins.
    total_rf_bins: int
        global number of unique bins across all spectra
    per_spec_sigma: list of floats, optional
        sigma of a given spectra

    Returns
    ------
    out: 1D array
        y_axis averaged values
    sigma: 1D array
        (1/sqrt(sum w))
    counts: 1D array
        counts per rf bin

    See also
    --------

    axion.haloscope.simulation.make_frequency_axis: Source of `rf_index_map` generation
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
