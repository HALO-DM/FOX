#!/usr/bin/env pythonA
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import datetime


from axion_haloscope.io import read_qshs_hdf5_dir
from axion_haloscope.data_quality import filter_spectrum_set, too_noisy, restrict_frequency_range
from axion_haloscope.baseline import remove_baseline
from axion_haloscope.combine import combine_ml
from axion_haloscope.rebin import rebin_ml, grand_spectrum_ml
from axion_haloscope.lineshape import shm_maxwell_template
from axion_haloscope.detection import threshold_for_detection, find_candidates
from axion_haloscope.limit import compute_local_snr_template, coupling_limit, plot_exclusion




def main():
    p = argparse.ArgumentParser(description="Run FOX analysis chain on QSHS HDF5 files.")

    p.add_argument("--input-dir", default="input/Jan_QSHS")
    p.add_argument("--pattern", default="*.hdf5")
    p.add_argument("--outdir", default="output/qshs_full_chain")

    # quality control
    p.add_argument("--qc", action="store_true")
    p.add_argument("--qc-rms-max", type=float, default=3.0)
    p.add_argument("--fmin-mhz", type=float, default=None)
    p.add_argument("--fmax-mhz", type=float, default=None)

    # baseline
    p.add_argument("--sg-window", type=int, default=401)
    p.add_argument("--sg-poly", type=int, default=4)
    p.add_argument("--baseline-diagnostic", action="store_true")

    # rebin / grand spectrum
    p.add_argument("--rebin-C", type=int, default=2)
    p.add_argument("--K", type=int, default=2)

    # detection / limits
    p.add_argument("--target-snr", type=float, default=5.0)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--snr-eff", type=float, default=0.90)
    p.add_argument("--g0", type=float, default=1.0)

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)


    timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    run_dir = outdir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)


    # ------------------------------------------------------------------
    # 1. Read QSHS HDF5 directory
    # ------------------------------------------------------------------
    sset = read_qshs_hdf5_dir(
        args.input_dir,
        pattern=args.pattern,
        use_shifted_frequency=True,
        sort_frequency=True,

        run_dir=run_dir,

    )

    print(f"[I/O] Loaded {sset.n_spectra()} QSHS spectra")
    print(
        f"[I/O] Frequency-offset span: "
        f"{sset.rf_grid[0]/1e6:.6f} to {sset.rf_grid[-1]/1e6:.6f} MHz"
    )

    # ------------------------------------------------------------------
    # 2. Optional data quality cut
    # ------------------------------------------------------------------
    if args.qc:
        sset, kept, bad = filter_spectrum_set(
            sset,
            predicate=lambda s, f, i: too_noisy(
                s,
                f,
                i,
                rms_max=args.qc_rms_max,
            ),
        )
        print(f"[QC] kept {len(kept)} spectra; dropped {len(bad)}: {bad}")



    # ------------------------------------------------------------------
    # 3. Baseline removal
    # ------------------------------------------------------------------
    proc = []
    raw_all = []
    baseline_all = []
    freqs_all = []

    for i, (s, f) in enumerate(zip(sset.spectra, sset.freqs_per_spec)):
        processed, baseline = remove_baseline(
            s,
            window_length=args.sg_window,
            polyorder=args.sg_poly,
            subtract_one=True,
            diagnostic=None,   # turn off per-spectrum file output
            freqs_hz=f,
        )

        proc.append(processed)
        raw_all.append(np.asarray(s))
        baseline_all.append(np.asarray(baseline))
        freqs_all.append(np.asarray(f))
        
    # ------------------------------------------------------------------
    # Optional frequency-range cut
    # ------------------------------------------------------------------
    #if args.fmin_mhz is not None or args.fmax_mhz is not None:
    #    fmin_hz = None if args.fmin_mhz is None else args.fmin_mhz * 1e6
    #    fmax_hz = None if args.fmax_mhz is None else args.fmax_mhz * 1e6
    #    
    #    sset = restrict_frequency_range(
    #        sset,
    #        fmin_hz=fmin_hz,
    #        fmax_hz=fmax_hz,
    #    )
        
    #    print(
    #        f"[Freq cut] kept range: "
    #        f"{sset.rf_grid[0]/1e6:.6f} to {sset.rf_grid[-1]/1e6:.6f} MHz"
    #    )
        
        
        
    plot_all_baseline_diagnostics(
        freqs_all,
        raw_all,
        baseline_all,
        proc,
        outfile=outdir,
    )

    '''
    # ------------------------------------------------------------------
    # 3. Baseline removal
    # ------------------------------------------------------------------
    proc = []

    for i, (s, f) in enumerate(zip(sset.spectra, sset.freqs_per_spec)):
        diagnostic = None
        if args.baseline_diagnostic and i == 0:
            diagnostic = {
                "outfile": outdir / "baseline_s000_before_after.png",
                "title": "QSHS baseline removal diagnostic: spectrum 0",
            }

        processed, baseline = remove_baseline(
            s,
            window_length=args.sg_window,
            polyorder=args.sg_poly,
            subtract_one=True,
            diagnostic=diagnostic,
            freqs_hz=f,
        )

        proc.append(processed)
    '''
    # ------------------------------------------------------------------
    # 4. Vertical combination
    # ------------------------------------------------------------------
    combined, sigma_c, counts = combine_ml(
        proc,
        sset.rf_index_map,
        total_rf_bins=len(sset.rf_grid),
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(sset.rf_grid / 1e6, combined, lw=0.8)
    ax.set(
        xlabel="Frequency offset [MHz]",
        ylabel="Excess power [arb.]",
        title="QSHS combined spectrum",
    )
    ax.grid(alpha=0.3)
    ax.set_xlim([0,2.5])
    ax.set_ylim([-0.1,0.1])
    fig.tight_layout()
    fig.savefig(outdir / "combined.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Rebin + grand spectrum
    # ------------------------------------------------------------------
    C = args.rebin_C
    K = args.K

    Dr, sr, spans = rebin_ml(combined, sigma_c, C=C)

    df = float(np.median(np.diff(sset.rf_grid)))
    freqs_r = sset.rf_grid[:len(Dr) * C:C] + (C // 2) * df



    template_f0_hz = 4.45e9  # temporary, approximate RF scale for shifted-frequency analysis

    Lq = shm_maxwell_template(
        K=K,
        bin_width_hz=C * df,
        f0_hz=template_f0_hz,
    )

    '''
    Lq = shm_maxwell_template(
        K=K,
        bin_width_hz=C * df,
        f0_hz=freqs_r[len(freqs_r) // 2],
    )
    '''
    Dg, sg = grand_spectrum_ml(Dr, sr, Lq)

    z = np.zeros_like(Dg)
    good = np.isfinite(sg) & (sg > 0)
    z[good] = Dg[good] / sg[good]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(freqs_r / 1e6, z, lw=0.8)
    ax.set(
        xlabel="Frequency offset [MHz]",
        ylabel="z",
        title="QSHS grand spectrum z-score",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "grand_z.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. Candidate finding
    # ------------------------------------------------------------------
    theta = threshold_for_detection(
        target_snr=args.target_snr,
        confidence=args.confidence,
    )

    cands, z = find_candidates(
        Dg,
        sg,
        theta,
        min_separation=K - 1,
    )

    with (outdir / "candidates.json").open("w") as fh:
        json.dump(
            {
                "threshold_sigma": float(theta),
                "indices": [int(i) for i in cands],
                "freq_offset_Hz": [float(freqs_r[i]) for i in cands],
                "z": [float(z[i]) for i in cands],
            },
            fh,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(freqs_r / 1e6, z, lw=0.7, color="black", label="grand z-score")
    ax.axhline(theta, color="tab:red", ls="--", label=f"threshold = {theta:.2f}σ")

    if len(cands) > 0:
        ax.scatter(
            freqs_r[cands] / 1e6,
            z[cands],
            s=35,
            color="tab:orange",
            zorder=5,
            label="candidates",
        )

    ax.set(
        xlabel="Frequency offset [MHz]",
        ylabel="z",
        title="QSHS candidate visualization",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlim([0,2.5])
    ax.set_ylim([-1.0,3.])
    fig.tight_layout()
    fig.savefig(outdir / "candidates.png", dpi=150)
    plt.close(fig)

    print(f"[Detection] Candidates flagged: {len(cands)}")

    # ------------------------------------------------------------------
    # 7. Exclusion curve, relative units
    # ------------------------------------------------------------------
    Rloc = compute_local_snr_template(sr, Lq)

    gmin = coupling_limit(
        Rloc,
        target_snr=args.target_snr,
        g0=args.g0,
        snr_efficiency=args.snr_eff,
    )

    plot_exclusion(
        freqs_r,
        gmin,
        outfile=outdir / "exclusion.png",
        title="QSHS relative exclusion, shifted frequency",
    )

    with (outdir / "exclusion.csv").open("w") as fh:
        fh.write("freq_offset_Hz,g_min_rel_to_g0\n")
        for f, g in zip(freqs_r, gmin):
            if np.isfinite(g):
                fh.write(f"{f},{g}\n")

    print(f"[OK] Results saved in {outdir}")



def plot_all_baseline_diagnostics(freqs_list, raw_list, baseline_list, proc_list, outfile):
    """
    Save two overlay diagnostic plots:
      1) all raw spectra + all SG baselines
      2) all processed spectra
    """

    # ------------------------------------------------------------
    # Plot 1: raw + baseline overlays
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))

    for i, (f, raw, base) in enumerate(zip(freqs_list, raw_list, baseline_list)):
        f_mhz = np.asarray(f) / 1e6
        ax.plot(f_mhz, raw, lw=0.6, alpha=0.6, label="raw" if i == 0 else None)
        ax.plot(f_mhz, base, lw=1.0, alpha=0.8, label="baseline (SG)" if i == 0 else None)

    ax.set_xlabel("Frequency offset [MHz]")
    ax.set_ylabel("Power")
    ax.set_title("Baseline diagnostic: all raw spectra + SG baselines")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlim([0,2.5])
    fig.tight_layout()
    fig.savefig(outfile / "baseline_overlay_raw_and_baseline.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------
    # Plot 2: processed overlays
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))

    for f, proc in zip(freqs_list, proc_list):
        f_mhz = np.asarray(f) / 1e6
        ax.plot(f_mhz, np.asarray(proc), lw=0.7, alpha=0.7)

    ax.set_xlabel("Frequency offset [MHz]")
    ax.set_ylabel("Processed power")
    ax.set_title("Baseline diagnostic: all processed spectra")
    ax.grid(alpha=0.3)
    ax.set_xlim([0,2.5])
    ax.set_ylim([-0.1,0.1])
    fig.tight_layout()
    fig.savefig(outfile / "baseline_overlay_processed.png", dpi=150)
    plt.close(fig)
    


if __name__ == "__main__":
    main()
