#!/usr/bin/env python
"""
Load spectra.npz and run the full HAYSTAC-like pipeline:
- baseline removal
- vertical ML combine
- rebin
- SHM matched filter (grand spectrum)
- candidate search
- 95% CL exclusion curve
"""
import argparse, json, pathlib, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from axion_haloscope.io        import read_npz
from axion_haloscope.baseline  import remove_baseline
from axion_haloscope.combine   import combine_ml
from axion_haloscope.rebin     import rebin_ml, grand_spectrum_ml
from axion_haloscope.lineshape import shm_maxwell_template
from axion_haloscope.detection import threshold_for_detection, find_candidates
from axion_haloscope.limit     import compute_local_snr_template, coupling_limit, plot_exclusion


def main():
    p = argparse.ArgumentParser(description="Run pipeline on spectra.npz")
    p.add_argument("npz_file", type=str, help="Path to spectra.npz")
    p.add_argument("--outdir", type=str, default="output/", help="Output dir")
    # Baseline
    p.add_argument("--sg-window", type=int, default=401)
    p.add_argument("--sg-poly", type=int, default=4)
    # Rebin + template
    p.add_argument("--rebin-C", type=int, default=10, dest="C")
    p.add_argument("--K", type=int, default=9, help="template length (rebinned bins)")
    # Detection & limit
    p.add_argument("--target-snr", type=float, default=5.0)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--snr-eff", type=float, default=0.90)
    p.add_argument("--g0", type=float, default=1.0)
    args = p.parse_args()


    npz_path = pathlib.Path(args.npz_file).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    
    # --- bookkeeping: enforce run_YYYYmmdd_HHMMSS convention ---
    run_dir = npz_path.parent
    m = re.match(r"^run_\d{8}_\d{6}$", run_dir.name)
    if not m:
        raise SystemExit(
            f"ERROR: input file must be inside a folder named run_YYYYmmdd_HHMMSS, got '{run_dir.name}'"
        )

    outdir = run_dir.parent / f"{run_dir.name}_pipeline"
    outdir.mkdir(parents=True, exist_ok=True)
    

    # --- Load bundle ---
    sset = read_npz(args.npz_file)
    n_spec = sset.n_spectra()
    print(f"Loaded {n_spec} spectra; RF span {sset.rf_grid[0]/1e9:.6f}–{sset.rf_grid[-1]/1e9:.6f} GHz")

    # --- Baseline removal ---
    proc = []
    for i, s in enumerate(sset.spectra):
        p_s, _ = remove_baseline(s, window_length=args.sg_window, polyorder=args.sg_poly, subtract_one=True)
        proc.append(p_s)

    # --- Vertical ML combine ---
    combined, sigma_c, counts = combine_ml(proc, sset.rf_index_map, total_rf_bins=len(sset.rf_grid))

    # Quick plot: combined spectrum
    plt.figure(figsize=(10,3))
    plt.plot(sset.rf_grid/1e9, combined, lw=0.8)
    plt.title("Combined spectrum (baseline-removed)")
    plt.xlabel("Frequency [GHz]"); plt.ylabel("Excess power [arb]"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(outdir/"combined.png", dpi=150); plt.close()

    # --- Rebin ---
    C = args.C
    Dr, sr, _ = rebin_ml(combined, sigma_c, C=C)
    # rebinned frequency centers
    freqs_r = sset.rf_grid[:len(Dr)*C:C] + (C//2) * (sset.rf_grid[1]-sset.rf_grid[0])

    # --- SHM template & grand spectrum ---
    K = args.K
    # Center SHM template at mid-band (center choice cancels in matched filter normalization)
    f0 = freqs_r[len(freqs_r)//2]
    Lq = shm_maxwell_template(K=K, bin_width_hz=C*(sset.rf_grid[1]-sset.rf_grid[0]), f0_hz=f0)
    Dg, sg = grand_spectrum_ml(Dr, sr, Lq)

    # Grand z-plot
    z = np.zeros_like(Dg)
    m = np.isfinite(sg) & (sg>0)
    z[m] = Dg[m] / sg[m]
    plt.figure(figsize=(10,3))
    plt.plot(freqs_r/1e9, z, lw=0.8)
    plt.title("Grand spectrum z-score (SHM matched filter)")
    plt.xlabel("Frequency [GHz]"); plt.ylabel("z"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(outdir/"grand_z.png", dpi=150); plt.close()

    # --- Candidates ---
    theta = threshold_for_detection(args.target_snr, args.confidence)
    cands, z = find_candidates(Dg, sg, theta, min_separation=K-1)
    with open(outdir/"candidates.json","w") as fh:
        json.dump({"threshold": float(theta),
                   "indices": [int(i) for i in cands],
                   "freqs_Hz": [float(freqs_r[i]) for i in cands]}, fh, indent=2)
    print(f"Candidates: {len(cands)} (threshold={theta:.2f}σ)")

    # --- Exclusion curve ---
    Rloc = compute_local_snr_template(sr, Lq)
    gmin = coupling_limit(Rloc, target_snr=args.target_snr, g0=args.g0, snr_efficiency=args.snr_eff)
    plot_exclusion(freqs_r, gmin, outfile=outdir/"exclusion.png", title="95% CL Exclusion (SHM)")
    # CSV
    with open(outdir/"exclusion.csv","w") as fh:
        fh.write("freq_Hz,g_min_rel_to_g0\n")
        for f,g in zip(freqs_r, gmin):
            if np.isfinite(g):
                fh.write(f"{f},{g}\n")

    print(f"Saved: {outdir/'combined.png'}, {outdir/'grand_z.png'}, {outdir/'exclusion.png'}, {outdir/'exclusion.csv'}")


if __name__ == "__main__":
    main()
