# axion_haloscope/simulation_working.py
"""
Simulation Working
==================
New simulation designed by Blue Carn during MPhys project. Basic wavepackets are superimposed
with noise in he time domain, before being amplified, downmixed and filtered to produce a 
more physical simulation.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

from axion_haloscope.io_working import SpectrumSet, SpectrumMetadata
from axion_haloscope.filter import pass_filter
from axion_haloscope.wavepacket import wavepacket_generation
from axion_haloscope.noise import simulate_baseline
from axion_haloscope.downmixing import downmix_signal
from axion_haloscope.graphs import aliasing, simulation_stages
from axion_haloscope.width_fq import width_from_fq


@dataclass
class AxionParams:
    """Parameters for an injected axion-like signal. Currently unused in this simulation, update
    to contain simualted axion information.

    Used to configure a synthetic narrow-band signal added to simulated spectra, approximating
    the expected lineshape of an axion-photon conversion signal (e.g. under the Standard Halo
    Model) as a Gaussian centered at a given frequency.

    Attributes
    ==========
    f_axion_hz : float
        Central frequency of the injected signal, in Hz.
    sigma_hz : float
        Spectral width (1-sigma) of the injected signal, in Hz.
    total_power : float
        Total integrated power of the injected signal, in the same (arbitrary) power units as
        the simulated spectra.
    """
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

    Parameters
    ==========
    n_spectra : int
        Number of spectra (tuning steps) in the scan.
    freqs : 1D array
        Array of frequency bins
    mask_show : 1D array
        Array of which freqs should be shown. Currently hardcoded to be between 0.2MHz and 200MHz

    Returns
    =======
    freqs_per_spec : 1D array of shape (n_spectra, n_bins)
        RF frequency of each bin, for each spectrum.
    rf_grid : 1D array of shape (N_total,)
        Global RF axis covering the whole scan
    rf_index_map : list of 1D array
        `rf_index_map[i]` gives the integer indices into `rf_grid` corresponding to spectrum 
        `i`'s bins.

    Notes
    =====
    If `tune_step_bins >= n_bins`, consecutive spectra don't overlap at all on `rf_grid`.
    Note for non-tunable cavities, `tune_step_bins` should be set to 0
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
    run_dir: Path = "",
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Simulate multiple physical spectra, mimicking the current pathfinder status. Uses physical
    power scalings and realistic noise floors given hardcoded parameters. Does not simualte a 
    tunable cavity, only 1 resonant frequency 

    Parameters
    ==========
    n_spectra: int
        Number of spectra to be generated
    n_bins: int
        Number of bins per spectra
    freq_axion: float
        Resonant Frequency of cavity/Injected Axion frequency (as doesn't simulate tunable cavity)
    freq_downmixed: float
        Target frequency to downmix to
    samples_per_cycle: float
        How many bins per wavelength
    run_dir: Path
        Save directory for all graphs/data

    Returns
    =======
    spectra: list of (n_bins,) float arrays
        Raw spectra (pre-baseline-removal)
    freqs_per_spec: rf_grid, rf_index_map
        Frequency bookkeeping from make_frequency_axes().
    metadata: SpectrumMetadata
        metadata extracted from simualtion

    Notes
    =====
    For all simuations, bandwidth is not calculated and therefore not passed. If QC check for min
    bandwidth is left on, all simualtions will be discarded. Turn off this seting in the config
    file.

    See Also
    ========
    axion_haloscope.io.SpectrumSet: see the data structure being exported
    axion_haloscope.io.SpectrumMetadata: see the metadata structure being exported
    axion_haloscope.wavepacket: see the initial signal generation and current hardcoded values
    """

    spectra: List[np.ndarray] = []
    dates = []
    file_names = []
    invalid_files = []
    b_vals = []
    temps = []
    q_factors = []
    res_freqs = []
    cw_freqs = []
    bandwidths = []

    freq_local_oscillator = freq_axion - freq_downmixed
    combined_freq = freq_axion + freq_local_oscillator
    fs = freq_axion * samples_per_cycle

    axion_bandwidth = width_from_fq(freq_axion)
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

    h_linear, freqs = pass_filter(n_bins, dt, run_dir=run_dir, file_name="low_pass_filter")

    l_linear, _ = pass_filter(n_bins, dt, x = np.array([
    0.20, 0.21, 0.23, 0.26, 0.29, 0.31, 0.34, 0.38, 0.39,
    0.43, 0.44, 0.46, 0.47, 0.53, 0.57, 0.59, 0.60,
    0.61, 0.64, 0.65, 0.73, 0.78, 0.82, 0.90, 0.99, 1.02,
    1.14, 1.41, 2.04, 2.72, 3.63, 5.86, 6.44
    ]), y = np.array([
    57.17, 57.48, 58.77, 61.74, 80.53, 65.50, 57.71, 67.91, 72.50,
    41.42, 39.58, 32.16, 30.34, 15.79, 6.33, 3.30, 2.52,
    1.91, 1.01, 0.93, 0.50, 0.24, 0.09, 0.00, 0.05, 0.05,
    0.18, 0.33, 0.28, 0.17, 0.08, 0.01, 0.01
    ]), run_dir=run_dir,file_name="high_pass_filter")


    for i in range(n_spectra):
        x_raw_signal = wavepacket_generation(freq_axion, axion_bandwidth, n=n_bins,
                                             samples_per_cycle=samples_per_cycle)
        baseline = simulate_baseline(x_raw_signal[:, 0])

        x_signal = x_raw_signal[:, 1] + baseline

        x_filtered, x_mixed = downmix_signal(x_signal, t, freq_local_oscillator, h_linear,
                                             l_linear)

        x_filt  = np.fft.rfft(x_filtered, n=n_bins)
        psd_filt = (np.abs(x_filt)**2) / (n_bins * fs)

        spectra.append(psd_filt[mask_show].astype(np.float64))
        dates.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        file_names.append(f"FOX_simulation_{datetime.now().strftime("%Y-%m-%d")}_{i:05d}")
        invalid_files.append(None)
        b_vals.append(None)
        temps.append(None)
        q_factors.append(None)
        res_freqs.append(freq_axion)
        cw_freqs.append(freq_axion)
        bandwidths.append(None) # Readout Chain Bandwidth not axion bandwidth

        # Please note, with bandwidth at None, if QC check for min bandwidth is on, all spectra
        # will be discarded.

        if i == 0:
            # Graphs
            tag, psd_mixed = simulation_stages(freq_axion, freq_local_oscillator,fs,
                                               freq_downmixed, n_bins, x_signal,
                                    x_mixed, x_filtered, freqs,psd_filt, mask_show,
                                    h_linear, l_linear, run_dir, t)
            aliasing(freqs, psd_mixed, freq_downmixed, fs, combined_freq, tag, run_dir)

    metadata = SpectrumMetadata(
        dates = dates,
        file_names = file_names,
        invalid_files = invalid_files,
        b_vals = b_vals,
        temps = temps,
        q_factors = q_factors,
        res_freqs = res_freqs,
        cw_freqs = cw_freqs,
        bandwidths = bandwidths,)
    return SpectrumSet(spectra, freqs_per_spec, rf_grid, rf_index_map, metadata)

# --- Minimal demo (optional) ---
if __name__ == "__main__":
    sset = simulate_spectra(
        n_spectra=1, n_bins = 1000000, freq_axion = 30e9,
        freq_downmixed = 6e6, samples_per_cycle = 125/12, run_dir = "demo",
    )
    specs, f_per, rf, idx_map, ax_power_dist = (sset.spectra, sset.freqs_per_spec, sset.rf_grid,
                                                sset.rf_index_map, sset.metadata)
    print(f"{len(specs)} spectra; RF span = {rf[0]/1e9:.6f}–{rf[-1]/1e9:.6f} GHz")
