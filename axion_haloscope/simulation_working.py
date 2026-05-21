# axion_haloscope/simulation.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

from axion_haloscope.filter import pass_filter
from axion_haloscope.wavepacket import wavepacket_generation
from axion_haloscope.noise import simulate_baseline
from axion_haloscope.downmixing import downmix_signal
from axion_haloscope.graphs import aliasing, simulation_stages


@dataclass
class AxionParams:
    """Parameters for an injected axion-like signal."""
    f_axion_hz : float     # central frequency [Hz]
    sigma_hz   : float     # spectral width (1-sigma) [Hz]
    total_power: float     # integrated power in spectrum units (arb.)

def make_frequency_axes(
    n_spectra: int,
    freqs: np.ndarray,
    mask_show: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Build per-spectrum RF axes on a common global RF grid.

    Returns
    -------
    freqs_per_spec : (n_spectra, n_masked) float array
        RF frequency of each kept bin for each spectrum.
    rf_grid : (n_bins,) float array
        Global RF axis (unmasked).
    rf_index_map : list of length n_spectra
        rf_index_map[i] are integer indices into rf_grid for spectrum i.
    """
    rf_grid = freqs
    idx = np.where(mask_show)[0]          # compute once; same for every spectrum
    n_masked = idx.size

    freqs_per_spec = np.empty((n_spectra, n_masked), dtype=np.float64)
    rf_index_map: List[np.ndarray] = []

    for i in range(n_spectra):
        rf_index_map.append(idx)
        freqs_per_spec[i] = rf_grid[idx]

    return freqs_per_spec, rf_grid, rf_index_map

def simulate_spectra(
    n_spectra: int = 1,
    n_bins: int = 1000000,
    freq_axion: float = 30e9,
    freq_downmixed: float = 6e6,
    samples_per_cycle: float = 125/12,
    amplitude: float = 1,
    run_dir: str = "",
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[np.ndarray]]:
 
    spectra: List[np.ndarray] = []

    freq_local_oscillator = freq_axion - freq_downmixed
    combined_freq = freq_axion + freq_local_oscillator
    fs = freq_axion * samples_per_cycle

    bandwidth = 1e-6 * freq_axion
    dt = 1.0 / fs
    t = np.arange(n_bins) * dt

    freqs = np.fft.rfftfreq(n_bins, d=dt)
    mask_show = (freqs >= 0.2e6) & (freqs <= 200e6)
    #mask_show = freqs >= 0
    freqs_per_spec, rf_grid, rf_index_map = make_frequency_axes(
        n_spectra=n_spectra,
        freqs=freqs,
        mask_show=mask_show,
    )
    
    H_linear, freqs = pass_filter(n_bins, dt, run_dir=run_dir, name="low_pass_filter")

    L_linear, _ = pass_filter(n_bins, dt, x = np.array([
    0.20, 0.21, 0.23, 0.26, 0.29, 0.31, 0.34, 0.38, 0.39,
    0.43, 0.44, 0.46, 0.47, 0.53, 0.57, 0.59, 0.60,
    0.61, 0.64, 0.65, 0.73, 0.78, 0.82, 0.90, 0.99, 1.02,
    1.14, 1.41, 2.04, 2.72, 3.63, 5.86, 6.44
    ]), y = np.array([
    57.17, 57.48, 58.77, 61.74, 80.53, 65.50, 57.71, 67.91, 72.50,
    41.42, 39.58, 32.16, 30.34, 15.79, 6.33, 3.30, 2.52,
    1.91, 1.01, 0.93, 0.50, 0.24, 0.09, 0.00, 0.05, 0.05,
    0.18, 0.33, 0.28, 0.17, 0.08, 0.01, 0.01
    ]), run_dir=run_dir,name="high_pass_filter")


    for i in range(n_spectra):
        x_raw_signal = wavepacket_generation(freq_axion, bandwidth, amplitude, n=n_bins, samples_per_cycle=samples_per_cycle)
        baseline = simulate_baseline(x_raw_signal[:, 0])

        x_signal = x_raw_signal[:, 1] + baseline

        x_filtered, x_mixed = downmix_signal(x_signal, t, freq_local_oscillator, H_linear, L_linear)

        X_filt  = np.fft.rfft(x_filtered, n=n_bins)
        psd_filt = (np.abs(X_filt)**2) / (n_bins * fs)      

        spectra.append(psd_filt[mask_show].astype(np.float64))

        if i == 0:
            # Graphs
            tag, psd_mixed = simulation_stages(freq_axion, freq_local_oscillator,fs, freq_downmixed, n_bins, x_signal, 
                                    x_mixed, x_filtered, freqs,psd_filt, mask_show, 
                                    H_linear, L_linear, run_dir, t)
            aliasing(freqs, psd_mixed, freq_downmixed, fs, combined_freq, tag, run_dir)

    return spectra, freqs_per_spec, rf_grid, rf_index_map

# --- Minimal demo (optional) ---
if __name__ == "__main__":
    ax = AxionParams(f_axion_hz=5.705e9, sigma_hz=2500.0, total_power=20.0)
    specs, f_per, rf, idx_map, ax_power_dist = simulate_spectra(
        n_spectra=10, n_bins=4000, bin_width_hz=100.0,
        f_start_hz=5.70e9, tune_step_bins=80,
        noise_sigma=1.0, rng_seed=1, axion=ax
    )
    print(f"{len(specs)} spectra; RF span = {rf[0]/1e9:.6f}–{rf[-1]/1e9:.6f} GHz")