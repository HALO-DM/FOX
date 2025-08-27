#!/usr/bin/env python
"""
Run a full simulated axion haloscope scan and save results in ./output/run_<timestamp>/
"""
import argparse, datetime, pathlib
import matplotlib.pyplot as plt
from axion_haloscope.simulation import simulate_spectra, AxionParams
from axion_haloscope.baseline import remove_baseline
from axion_haloscope.combine import combine_ml
from axion_haloscope.rebin import rebin_ml, grand_spectrum_ml
from axion_haloscope.lineshape import shm_maxwell_template
from axion_haloscope.detection import threshold_for_detection, find_candidates
from axion_haloscope.limit import compute_local_snr_template, coupling_limit, plot_exclusion

def main():
    p = argparse.ArgumentParser(description="Simulate a haloscope scan (toy HAYSTAC-like)")
    p.add_argument("--n-spectra", type=int, default=80)
    p.add_argument("--n-bins", type=int, default=8000)
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--f-start", type=float, default=5.70e9)
    p.add_argument("--tune-step-bins", type=int, default=100)
    p.add_argument("--axion", action="store_true", help="Inject an axion signal")
    args = p.parse_args()

    # timestamped output folder
    outdir = pathlib.Path("output") / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    outdir.mkdir(parents=True, exist_ok=True)

    ax = None
    if args.axion:
        f_ax = args.f_start + (args.n_bins + (args.n_spectra-1)*args.tune_step_bins)*args.bin_width*0.5
        ax = AxionParams(f_axion_hz=f_ax, sigma_hz=2500.0, total_power=20.0)

    # 1) simulate
    specs, fper, rf, rf_map = simulate_spectra(
        n_spectra=args.n_spectra, n_bins=args.n_bins,
        bin_width_hz=args.bin_width, f_start_hz=args.f_start,
        tune_step_bins=args.tune_step_bins, rng_seed=1234, axion=ax
    )

    # save one raw spectrum
    plt.figure(figsize=(9,3))
    plt.plot(fper[0]/1e9, specs[0], lw=0.6)
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Raw Power [arb]")
    plt.title("Example raw spectrum")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(outdir/"raw_spectrum.png", dpi=150); plt.close()

    # 2) baseline removal
    proc = [remove_baseline(s, window_length=401, polyorder=4)[0] for s in specs]

    # 3) combine
    combined, sigma_c, counts = combine_ml(proc, rf_map, total_rf_bins=len(rf))

    # 4) rebin + grand spectrum (SHM template)
    C, K = 10, 9
    Dr, sr, _ = rebin_ml(combined, sigma_c, C=C)
    freqs_r = rf[:len(Dr)*C:C] + (C//2)*args.bin_width
    Lq = shm_maxwell_template(K, bin_width_hz=C*args.bin_width, f0_hz=freqs_r[len(freqs_r)//2])
    Dg, sg = grand_spectrum_ml(Dr, sr, Lq)

    # 5) candidates
    theta = threshold_for_detection(target_snr=5.0, confidence=0.95)
    cands, z = find_candidates(Dg, sg, theta, min_separation=K-1)

    # 6) exclusion curve
    Rloc = compute_local_snr_template(sr, Lq)
    gmin = coupling_limit(Rloc, target_snr=5.0, g0=1.0, snr_efficiency=0.9)
    plot_exclusion(freqs_r, gmin, outfile=outdir/"exclusion.png")

    print(f"Simulation finished, results in {outdir}")
    print(f"Candidates flagged: {len(cands)}")

if __name__ == "__main__":
    main()
