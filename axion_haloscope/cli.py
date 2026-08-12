# axion_haloscope/cli.py
"""
cli
===
"""
from __future__ import annotations
import argparse
import csv
import json
import numpy as np
from .simulation import simulate_spectra
from .baseline   import remove_baseline
from .combine    import combine_ml
from .rebin      import rebin_ml
from .lineshape  import shm_maxwell_template
from .rebin      import grand_spectrum_ml
from .detection  import threshold_for_detection, find_candidates
from .limit      import compute_local_snr_template, coupling_limit
from .graphs     import plot_exclusion

def main():
    """Stripped Back Simulation and data analysis"""
    p = argparse.ArgumentParser(description="Axion haloscope toy pipeline (HAYSTAC-like)")
    p.add_argument("--n-spectra", type=int, default=60)
    p.add_argument("--n-bins", type=int, default=6000)
    p.add_argument("--bin-width-hz", type=float, default=100.0)
    p.add_argument("--f-start-hz", type=float, default=5.70e9)
    p.add_argument("--tune-step-bins", type=int, default=60)
    p.add_argument("--sg-window", type=int, default=401)
    p.add_argument("--sg-poly", type=int, default=4)
    p.add_argument("--rebin-C", type=int, default=10)
    p.add_argument("--k", type=int, default=9, help="template length (rebinned bins)")
    p.add_argument("--target-snr", type=float, default=5.0)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--snr-eff", type=float, default=0.90)
    p.add_argument("--g0", type=float, default=1.0)
    p.add_argument("--rng-seed", type=int, default=1234)
    # SHM parameters
    p.add_argument("--v0-kms", type=float, default=220.0, help="local circular speed (km/s)")
    p.add_argument("--vesc-kms", type=float, default=544.0, help="escape speed (km/s)")
    p.add_argument("--ve-kms", type=float, default=232.0,
                   help="Earth speed (km/s); set 0 to ignore boost")
    # Axion injection (optional)
    p.add_argument("--inject", action="store_true")
    p.add_argument("--axion-f-hz", type=float, default=None)
    p.add_argument("--axion-total", type=float, default=20.0)
    p.add_argument("--axion-width-hz", type=float, default=None,
                   help="if set, overrides SHM width w/ Gaussian")
    p.add_argument("--out-prefix", type=str, default="/mnt/data/axion_out")
    p.add_argument("--dry-run", action="store_true",
                   help="Just print what would be done; skip simulation and file writes")
    args = p.parse_args()

    # If dry-run, just show configuration and exit
    if args.dry_run:
        total_bins = args.n_bins + (args.n_spectra-1)*args.tune_step_bins
        rf_lo = args.f_start_hz
        rf_hi = args.f_start_hz + total_bins*args.bin_width_hz
        print("[DRY RUN] Pipeline configuration:")
        print(f"  spectra: {args.n_spectra} × {args.n_bins} bins,"
              " bin width = {args.bin_width_hz} Hz")
        print(f"  RF span: {rf_lo/1e9:.6f}–{rf_hi/1e9:.6f} GHz")
        if args.inject:
            f_ax = args.axion_f_hz if args.axion_f_hz else (args.f_start_hz
                                                            + total_bins*args.bin_width_hz*0.6)
            print(f"  Would inject axion at {f_ax/1e9:.6f} GHz, total_power={args.axion_total}")
        print(f"  Outputs would be written to {args.out_prefix}_exclusion.csv/.png")
        return  # Exit cleanly without running analysis

    # 1) simulate
    ax = {
    "enabled": True,
    "f_axion_hz": args.axion_f_hz,  # can be None; simulate_spectra fills default
    "total_power": args.axion_total,
    }
    sset = simulate_spectra(
        n_spectra=args.n_spectra, n_bins=args.n_bins, bin_width_hz=args.bin_width_hz,
        f_start_hz=args.f_start_hz, tune_step_bins=args.tune_step_bins,
        rng_seed=args.rng_seed, injected_axion=ax
    )
    spectra, _, rf_grid, rf_map, _ = (sset.spectra, sset.freqs_per_spec, sset.rf_grid,
                                                sset.rf_index_map, sset.metadata)

    # 2) baseline removal
    proc = [remove_baseline(s, window_length=args.sg_window,
                            polyorder=args.sg_poly)[0] for s in spectra]

    # 3) vertical ML combine
    combined, sigma_c, _ = combine_ml(proc, rf_map, total_rf_bins=len(rf_grid))

    # 4) rebin & SHM matched filter
    dr, sr, _ = rebin_ml(combined, sigma_c, C=args.rebin_C)
    freqs_r = rf_grid[:len(dr)*args.rebin_C:args.rebin_C] + (args.rebin_C//2)*args.bin_width_hz

    # Build SHM template on the rebinned grid (k bins)
    k = args.k
    lq = shm_maxwell_template(
        K=k, bin_width_hz=args.rebin_C*args.bin_width_hz, f0_hz=freqs_r[len(freqs_r)//2],
        v0=args.v0_kms*1e3, v_esc=args.vesc_kms*1e3, v_earth=args.ve_kms*1e3
    )
    dg, sg = grand_spectrum_ml(dr, sr, lq)

    # 5) threshold & candidates
    theta = threshold_for_detection(args.target_snr, args.confidence)
    cands, _ = find_candidates(dg, sg, theta, min_separation=k-1)

    # 6) exclusion
    rloc = compute_local_snr_template(sr, lq)
    gmin = coupling_limit(rloc, target_snr=args.target_snr, g0=args.g0, snr_efficiency=args.snr_eff)

    # save simple CSV + plot

    csv_path = args.out_prefix + "_exclusion.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["freq_Hz","g_min_rel_to_g0"])
        for f,g in zip(freqs_r, gmin):
            if np.isfinite(g):
                w.writerow([f, g])
    png_path = args.out_prefix + "_exclusion.png"
    plot_exclusion(freqs_r, gmin, outfile=png_path, title="95% CL Exclusion (toy, SHM)")
    # meta
    meta = dict(
        theta=theta, n_candidates=len(cands),
        candidates=[int(c) for c in cands]
    )
    with open(args.out_prefix + "_meta.json","w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Wrote {csv_path}, {png_path} and meta JSON; candidates: {len(cands)}")

if __name__ == "__main__":
    main()
