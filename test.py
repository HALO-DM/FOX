from axion_haloscope.simulation import simulate_spectra, AxionParams

ax = AxionParams(f_axion_hz=5.705e9, sigma_hz=2500.0, total_power=20.0)
spectra, freqs_per_spec, rf_grid, rf_index_map = simulate_spectra(
    n_spectra=60, n_bins=6000, bin_width_hz=100.0,
    f_start_hz=5.70e9, tune_step_bins=60,
    noise_sigma=1.0, rng_seed=1234, axion=ax
)
