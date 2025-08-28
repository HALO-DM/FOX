#!/usr/bin/env python
"""
Read a spectra.npz bundle using the axion_haloscope.io module and make quick plots.
"""

import argparse
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from axion_haloscope.io import read_npz


def main():
    p = argparse.ArgumentParser(description="Read spectra.npz and plot a few spectra.")
    p.add_argument("npz_file"   , type=str, help="Path to spectra.npz")
    p.add_argument("--max-plots", type=int, default=5, help="How many spectra to plot")
    p.add_argument("--outdir"   , type=str, default="output/read_npz_demo", help="Where to save plots")
    args = p.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load spectra ---
    sset = read_npz(args.npz_file)
    print(f"Loaded {sset.n_spectra()} spectra")
    print(f"Global RF range: {sset.rf_grid[0]/1e9:.6f}–{sset.rf_grid[-1]/1e9:.6f} GHz")
    print(f"Average bins per spectrum: {np.mean([len(s) for s in sset.spectra]):.0f}")

    # --- Plot first few spectra ---
    for i in range(min(args.max_plots, sset.n_spectra())):
        freqs = sset.freqs_per_spec[i]
        spec = sset.spectra[i]
        plt.figure(figsize=(9, 3))
        plt.plot(freqs/1e9, spec, lw=0.6)
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Power [arb.]")
        plt.title(f"Spectrum {i:03d}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"spectrum_{i:03d}.png", dpi=120)
        plt.close()

    print(f"Saved plots in {outdir}")


if __name__ == "__main__":
    main()
