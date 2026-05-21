# axion_haloscope/combine.py
from __future__ import annotations
from typing import Optional
import numpy as np


def lorentzian_height(f0: float, f: np.ndarray, bw: float) -> np.ndarray:
    """
    Compute the Lorentzian coupling factor at frequencies f, given cavity
    resonance frequency f0 and bandwidth (FWHM) bw.

    Parameters
    ----------
    f0 : float
        Cavity resonance frequency [Hz].
    f : np.ndarray
        Frequencies at which to evaluate the profile [Hz].
    bw : float
        Cavity bandwidth (FWHM) [Hz].

    Returns
    -------
    np.ndarray
        Lorentzian height in [0, 1], same shape as f.
    """
    return 1.0 / (1.0 + ((f - f0) / (0.5 * bw)) ** 2)


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

    Parameters
    ----------
    processed_spectra : list of np.ndarray
        Baseline-removed power spectra, one per tuning step.
    rf_index_map : list of np.ndarray
        For each spectrum, the indices into the common RF grid that its
        samples map to.  Same length as the corresponding spectrum.
    total_rf_bins : int
        Total number of bins in the common RF grid.
    per_spec_sigma : list of float or None
        Noise estimate (std) for each spectrum.  If None, estimated from
        the IQR of each spectrum.
    lorentzian_weight : bool
        If True, apply per-bin Lorentzian cavity coupling weights
        w_ij = h(f_ij) / sigma_i^2.  Requires lorentz_params and
        spec_freqs to be provided.
    lorentz_params : list of (f0_hz, bw_hz) or None
        Per-spectrum cavity resonance frequency and bandwidth [Hz].
        Required when lorentzian_weight=True.
    spec_freqs : list of np.ndarray or None
        Actual RF frequencies [Hz] for each spectrum's samples.
        Required when lorentzian_weight=True.

    Returns
    -------
    out : np.ndarray
        Weighted-average combined spectrum over the RF grid.
    sigma : np.ndarray
        Per-bin uncertainty: 1 / sqrt(sum of weights).
        Bins with no coverage are set to np.inf.
    counts : np.ndarray
        Number of spectra contributing to each RF bin.
    """

    if lorentzian_weight: # Validation of required parameters for Lorentzian weighting
        if lorentz_params is None or spec_freqs is None:
            raise ValueError(
                "lorentz_params and spec_freqs must both be provided "
                "when lorentzian_weight=True."
            )
        if len(lorentz_params) != len(processed_spectra):
            raise ValueError(
                f"lorentz_params length ({len(lorentz_params)}) must match "
                f"processed_spectra length ({len(processed_spectra)})."
            )
        if len(spec_freqs) != len(processed_spectra):
            raise ValueError(
                f"spec_freqs length ({len(spec_freqs)}) must match "
                f"processed_spectra length ({len(processed_spectra)})."
            )

    combined = np.zeros(total_rf_bins, float)   # Initialise arrays to hold the combined spectrum
    wsum     = np.zeros(total_rf_bins, float)   # the sum of weights
    counts   = np.zeros(total_rf_bins, int)     # and the count of contributions


    if lorentzian_weight:
        for s, idx, sig, freqs, (f0, bw) in zip(
            processed_spectra, rf_index_map, per_spec_sigma, spec_freqs, lorentz_params
        ):
            h = lorentzian_height(f0=f0, f=freqs, bw=bw)   # per-sample coupling
            w = h / (sig * sig)  # per sample weight
            np.add.at(combined, idx, w * s)
            np.add.at(wsum,     idx, w)
            np.add.at(counts,   idx, 1)
    else:
        for s, idx, sig in zip(processed_spectra, rf_index_map, per_spec_sigma):
            w = 1.0 / (sig * sig)   # uniform weight
            np.add.at(combined, idx, w * s)
            np.add.at(wsum,     idx, w)
            np.add.at(counts,   idx, 1)

    nz = wsum > 0
    out = np.zeros_like(combined)
    out[nz] = combined[nz] / wsum[nz] # Output weighted average

    sigma_out = np.full_like(combined, np.inf)
    sigma_out[nz] = np.sqrt(1.0 / wsum[nz])    # uncertainty on the weighted mean

    return out, sigma_out, counts
