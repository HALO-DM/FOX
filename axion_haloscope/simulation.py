# axion_haloscope/simulation.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import datetime
from collections import defaultdict

from axion_haloscope.noise import external_noise
from axion_haloscope.io_working import SpectrumMetadata
from axion_haloscope.width_fq   import width_from_fq

@dataclass
class AxionParams:
    """Parameters for an injected axion-like signal."""
    f_axion_hz : float     # central frequency [Hz]
    sigma_hz   : float     # spectral width (1-sigma) [Hz]
    total_power: float     # integrated power in spectrum units (arb.)

def make_frequency_axes(
    n_spectra: int,
    n_bins: int,
    bin_width_hz: float,
    f_start_hz: float,
    tune_step_bins: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Build per-spectrum RF axes on a common global RF grid.

    Returns
    -------
    freqs_per_spec : (n_spectra, n_bins) float array
        RF frequency of each bin for each spectrum.
    rf_grid : (N_total,) float array
        Global RF axis covering the whole scan.
    rf_index_map : list of length n_spectra
        rf_index_map[i] are integer indices into rf_grid for spectrum i.
    """
    total_bins = n_bins + (n_spectra - 1) * tune_step_bins
    rf_grid = f_start_hz + np.arange(total_bins, dtype=float) * bin_width_hz

    freqs_per_spec = np.zeros((n_spectra, n_bins), dtype=float)
    rf_index_map: List[np.ndarray] = []
    for i in range(n_spectra):
        off = i * tune_step_bins
        idx = np.arange(off, off + n_bins, dtype=int)
        rf_index_map.append(idx)
        freqs_per_spec[i] = rf_grid[idx]
    return freqs_per_spec, rf_grid, rf_index_map

def simulate_baseline(
    n_bins: int,
    rng: np.random.Generator,
    amplitude: float = 0.05,
    corr_bins: int = 400,
) -> np.ndarray:
    """
    Slow multiplicative baseline ~ 1 + smooth noise.

    Parameters
    ----------
    amplitude : float
        RMS amplitude of baseline undulations (fractional).
    corr_bins : int
        Correlation length in bins (larger -> smoother).
    """
    # White noise -> smooth via Hann kernel
    pad = 8 * corr_bins
    x = rng.normal(0.0, 1.0, size=n_bins + pad)
    kernel = np.hanning(2 * corr_bins + 1)
    kernel /= kernel.sum()
    y = np.convolve(x, kernel, mode="same")
    # Center-crop to exactly n_bins
    start = (len(y) - n_bins) // 2
    y = y[start : start + n_bins]
    y = y / (np.std(y) + 1e-12) * amplitude
    baseline = 1.0 + y
    return np.maximum(1e-6, baseline)

def axion_lineshape_gaussian(
    freqs_hz: np.ndarray, f_axion_hz: float, sigma_hz: float
) -> np.ndarray:
    """
    Gaussian lineshape (unit *area* w.r.t. discrete bins).
    Suitable as a simple SHM proxy; replace later with a Maxwellian if desired.
    """
    x = (freqs_hz - f_axion_hz) / (sigma_hz + 1e-30)
    L = np.exp(-0.5 * x * x)
    # Normalize to unit sum over bins
    Lsum = L.sum()
    if Lsum <= 0:
        return np.zeros_like(L)
    return L / Lsum

def injected_axion_power(
    rf_grid_hz: np.ndarray, f_axion_hz: float, sigma_hz: float, total_power: float
) -> np.ndarray:
    """
    Distribute 'total_power' across rf_grid_hz with a Gaussian lineshape.
    Returns a per-bin power array aligned with rf_grid_hz.
    """
    Lb = axion_lineshape_gaussian(rf_grid_hz, f_axion_hz, sigma_hz)
    return total_power * Lb

def _simulate_one_spectrum(
    i: int,
    freqs: np.ndarray,
    rng: np.random.Generator,
    n_bins: int,
    f_start_hz: float,
    f_range: float,
    noise_sigma: float,
    baseline_amp: float,
    baseline_corr_bins: int,
    baseline_key: Optional[np.ndarray],
    axion: AxionParams | None,
    axion_power_global: np.ndarray,
    rf_index_map: List[np.ndarray],
) -> Tuple[np.ndarray, dict]:
    """
    Simulate one tuned spectrum plus its metadata.

    Returns
    """
    baseline = simulate_baseline(n_bins, rng, amplitude=baseline_amp, corr_bins=baseline_corr_bins)
    white_noise = rng.normal(0.0, noise_sigma, size=n_bins)
    external = external_noise(freqs, f_start_hz, f_range, baseline_key)
    raw = external + white_noise

    if axion is not None:
        raw = raw + axion_power_global[rf_index_map[i]]

    metadata = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": None,
        "invalid_files": None,
        "b_vals": None,
        "q_factor": None,
        "temps": None,
        "tuning_angle": None,
        "volume": None,
        "res_freq": axion.f_axion_hz if axion is not None else None,
        "cw_freq": None,
        "bandwidth": axion.sigma_hz if axion is not None else None,
    }
    return raw.astype(np.float64), metadata

def simulate_spectra(
    n_spectra: int = 60,
    n_bins: int = 6000,
    bin_width_hz: float = 100.0,
    f_start_hz: float = 5.70e9,
    tune_step_bins: int = 60,
    noise_sigma: float = 1.0,
    rng_seed: int | None = 1234,
    injected_axion: AxionParams | None = None,
    baseline_amp: float = 0.05,
    baseline_corr_bins: int = 400,
    baseline_key: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Simulate multiple tuned spectra: slow baseline × (1 + Gaussian noise),
    placed on a shared RF grid, with an optional injected axion-like line.

    Returns
    -------
    spectra : list of (n_bins,) float arrays
        Raw spectra (pre-baseline-removal), one per tuning step.
    freqs_per_spec, rf_grid, rf_index_map
        Frequency bookkeeping from make_frequency_axes().
    """
    rng = np.random.default_rng(rng_seed)
    freqs_per_spec, rf_grid, rf_index_map = make_frequency_axes(
        n_spectra=n_spectra,
        n_bins=n_bins,
        bin_width_hz=bin_width_hz,
        f_start_hz=f_start_hz,
        tune_step_bins=tune_step_bins,
    )

    f_range = np.max(freqs_per_spec) - np.min(freqs_per_spec)

    axion = None
    if injected_axion["enabled"]:
        total_bins = n_bins + n_spectra - 1 * tune_step_bins
        f_ax = injected_axion["f_axion_hz"]
        if f_ax is None:
            f_ax = f_start_hz + 0.5 * total_bins * bin_width_hz
        s_ax = width_from_fq(f_ax)
        axion = AxionParams(f_axion_hz=float(f_ax), sigma_hz=s_ax, total_power=injected_axion["total_power"])

    # Optional axion power on the global RF grid
    axion_power_global = (
        injected_axion_power(rf_grid, axion.f_axion_hz, axion.sigma_hz, axion.total_power)
        if axion is not None
        else np.zeros_like(rf_grid)
    )

    spectra: List[np.ndarray] = []
    meta_lists: dict[str, list] = defaultdict(list)

    for i in range(n_spectra):
        # Simulate a singluar spectra and its metadata
        raw, record = _simulate_one_spectrum(
            i, freqs_per_spec[i], rng, n_bins, f_start_hz, f_range,
            noise_sigma, baseline_amp, baseline_corr_bins, baseline_key,
            axion, axion_power_global, rf_index_map,
        )
        spectra.append(raw)
        for k, v in record.items():
            meta_lists[k].append(v)

    metadata = SpectrumMetadata(**{k: np.array(v) for k, v in meta_lists.items()})
    return spectra, freqs_per_spec, rf_grid, rf_index_map, metadata

# --- Minimal demo (optional) ---
if __name__ == "__main__":
    ax = AxionParams(f_axion_hz=5.705e9, sigma_hz=2500.0, total_power=20.0)
    specs, f_per, rf, idx_map, ax_power_dist = simulate_spectra(
        n_spectra=10, n_bins=4000, bin_width_hz=100.0,
        f_start_hz=5.70e9, tune_step_bins=80,
        noise_sigma=1.0, rng_seed=1, axion=ax
    )
    print(f"{len(specs)} spectra; RF span = {rf[0]/1e9:.6f}–{rf[-1]/1e9:.6f} GHz")