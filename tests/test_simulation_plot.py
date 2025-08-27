# tests/test_simulation_plot.py
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt

from axion_haloscope.simulation import simulate_spectra, AxionParams


def test_simulated_spectrum_plot(tmp_path):
    # Small, fast config for a unit test
    n_spectra = 3
    n_bins = 4000
    bin_width = 100.0  # Hz
    f_start = 5.70e9
    tune_step_bins = 100

    ax = AxionParams(
        f_axion_hz = f_start + 0.45*(n_bins + (n_spectra-1)*tune_step_bins)*bin_width,
        sigma_hz   = 2500.0,
        total_power= 15.0,
    )

    spectra, freqs_per_spec, rf_grid, rf_index_map = simulate_spectra(
        n_spectra=n_spectra,
        n_bins=n_bins,
        bin_width_hz=bin_width,
        f_start_hz=f_start,
        tune_step_bins=tune_step_bins,
        noise_sigma=1.0,
        rng_seed=42,
        axion=ax,
    )

    # Basic sanity checks
    assert len(spectra) == n_spectra
    assert spectra[0].shape == (n_bins,)
    assert rf_grid.ndim == 1 and rf_grid.size == n_bins + (n_spectra-1)*tune_step_bins

    # Plot the first simulated spectrum on its RF axis
    fig, axp = plt.subplots(figsize=(8, 3))
    idx0 = rf_index_map[0]
    axp.plot(freqs_per_spec[0]/1e9, spectra[0], lw=0.7)
    axp.set_xlabel("Frequency [GHz]")
    axp.set_ylabel("Power [arb.]")
    axp.set_title("Simulated haloscope spectrum (toy)")

    out = tmp_path / "sim_spectrum.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)

    # File should exist and be non-empty
    assert out.exists()
    assert out.stat().st_size > 0
    print ("Saved to:", out)




def test_plot_first_spectrum(tmp_path):
    specs, fper, rf, rf_map = simulate_spectra(
        n_spectra=3, n_bins=3000, bin_width_hz=100.0,
        f_start_hz=5.70e9, tune_step_bins=80, rng_seed=42,
        axion=AxionParams(5.705e9, 2500.0, 20.0)
    )
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(fper[0]/1e9, specs[0], lw=0.7)
    ax.set(xlabel="Frequency [GHz]", ylabel="Power [arb.]",
           title="Simulated spectrum (toy)")
    out = tmp_path / "sim_spectrum.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    assert out.exists() and out.stat().st_size > 0
    print ("Saved to:", out)
    
