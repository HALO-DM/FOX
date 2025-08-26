# examples/run_pipeline.py
import numpy as np
from axion_haloscope.simulation import simulate_spectra, AxionParams
from axion_haloscope.baseline   import remove_baseline
from axion_haloscope.combine    import combine_ml
from axion_haloscope.rebin      import rebin_ml, axion_template_gaussian, grand_spectrum_ml
from axion_haloscope.detection  import threshold_for_detection, find_candidates
from axion_haloscope.limit      import compute_local_snr_template, coupling_limit, plot_exclusion

def run_demo():
    # ----- step 1: simulate -----
    ax = AxionParams(f_axion_hz=5.705e9, sigma_hz=2500.0, total_power=20.0)
    spectra, f_per, rf_grid, rf_map = simulate_spectra(
        n_spectra=60, n_bins=6000, bin_width_hz=100.0,
        f_start_hz=5.70e9, tune_step_bins=60, rng_seed=1234, axion=ax
    )

    # ----- step 2: baseline removal -----
    proc = [remove_baseline(s, window_length=401, polyorder=4)[0] for s in spectra]

    # ----- step 3: vertical ML combine -----
    combined, sigma_c, counts = combine_ml(proc, rf_map, total_rf_bins=len(rf_grid))

    # ----- step 4: rebin + matched-filter grand spectrum -----
    C = 10                                   # rebin factor
    Dr, sr, _ = rebin_ml(combined, sigma_c, C=C)
    freqs_r = rf_grid[:len(Dr)*C:C] + (C//2)*100.0
    K = 7                                    # template length (rebinned bins)
    Lq = axion_template_gaussian(K, bin_width_hz=C*100.0, sigma_hz=2500.0)
    Dg, sg = grand_spectrum_ml(Dr, sr, Lq)

    # ----- step 5: threshold & candidates -----
    theta = threshold_for_detection(target_snr=5.0, confidence=0.95)
    cands, z = find_candidates(Dg, sg, theta, min_separation=K-1)

    # ----- step 6: exclusion curve -----
    Rloc = compute_local_snr_template(sr, Lq)
    gmin = coupling_limit(Rloc, target_snr=5.0, g0=1.0, snr_efficiency=0.90)
    plot_exclusion(freqs_r, gmin, outfile=None)

    return dict(freqs_r=freqs_r, gmin=gmin, candidates=cands, z=z)
