#!/usr/bin/env python
"""
Run a full simulated axion haloscope scan and save results in ./output/run_<timestamp>/
"""
import argparse, datetime, pathlib
import matplotlib.pyplot as plt
import numpy as np
from axion_haloscope.simulation import simulate_spectra, AxionParams

def main():
    p = argparse.ArgumentParser(description="Simulate a haloscope scan (toy HAYSTAC-like)")
    p.add_argument("--n-spectra"     , type=int  , default=    10) # number of acquired spectra
    p.add_argument("--n-bins"        , type=int  , default=  8000) # number of bins in the acquired spectra
    p.add_argument("--bin-width"     , type=float, default= 100.0) # bin width of the power spectra
    p.add_argument("--f-start"       , type=float, default=5.70e9) # Starting frequency
    p.add_argument("--tune-step-bins", type=int  , default=     0) # 100 Hz is probably more suitable when we want to actually do a scan
    p.add_argument("--save-n-spectra", type=int  , default=     2) # number of saved plots
    p.add_argument('--save-data', action=argparse.BooleanOptionalAction)
    p.add_argument("--axion", action="store_true", help="Inject an axion signal")
    args = p.parse_args()
    
    
    # timestamped output folder
    outdir = pathlib.Path("output/sim_spectra/") / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    outdir.mkdir(parents=True, exist_ok=True)

    ax = None
    if args.axion:
        f_ax = args.f_start + (args.n_bins + (args.n_spectra-1)*args.tune_step_bins)*args.bin_width*0.5
        ax = AxionParams(f_axion_hz=f_ax, sigma_hz=2500.0, total_power=200.0)

    # 1) simulate
    specs, fper, rf, rf_map = simulate_spectra(
        n_spectra=args.n_spectra, n_bins=args.n_bins,
        bin_width_hz=args.bin_width, f_start_hz=args.f_start,
        tune_step_bins=args.tune_step_bins, rng_seed=1234, axion=ax
    )

    

    if args.save_n_spectra: 
        # choose how many to plot (1 by default). You can throttle with step or max_plots.
        step = 1        # e.g., plot every 5th: step = 5
        max_plots = min(args.save_n_spectra, args.n_spectra)   # e.g., cap at 100: max_plots = 100
        count = 0
        for i, (freqs, spec) in enumerate(zip(fper, specs)):
            if i % step != 0:
                continue
            if max_plots is not None and count >= max_plots:
                break
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(freqs/1e9, spec, lw=0.6)
            ax.set_xlabel("Frequency [GHz]")
            ax.set_ylabel("Raw Power [arb]")
            ax.set_title(f"Spectrum {i:03d}")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / f"spectrum_{i:03d}.png", dpi=150)
            plt.close(fig)
            count += 1

    if args.save_data:
        # compact binary for analysis
        np.savez(outdir/"spectra.npz",
                 spectra=np.array(specs),   # (n_spectra, n_bins)
                 freqs=fper,                # (n_spectra, n_bins)
                 rf_grid=rf)                # (total_rf_bins,)


    print(f"Raw spectra simulation finished, results in {outdir}")


if __name__ == "__main__":
    main()
