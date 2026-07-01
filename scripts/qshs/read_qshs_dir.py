#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from axion_haloscope.io import read_qshs_hdf5_dir, write_hdf5


def main():
    p = argparse.ArgumentParser(
        description="Read a directory of QSHS HDF5 files into FOX SpectrumSet format."
    )
    p.add_argument(
        "--input-dir",
        default="input/Jan_QSHS",
        help="Directory containing QSHS .hdf5 files.",
    )
    p.add_argument(
        "--pattern",
        default="*.hdf5",
        help="Glob pattern for QSHS files.",
    )
    p.add_argument(
        "--outdir",
        default="output/qshs_import",
        help="Output directory for diagnostics and converted file.",
    )
    p.add_argument(
        "--save-fox-h5",
        action="store_false",
        help="Save merged SpectrumSet as FOX-native spectra.h5.",
    )
    p.add_argument(
        "--max-plot",
        type=int,
        default=100,
        help="Number of spectra to plot for diagnostics.",
    )
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sset = read_qshs_hdf5_dir(
        input_dir,
        pattern=args.pattern,
        use_shifted_frequency=True,
        sort_frequency=True,
    )

    print(f"[QSHS] Loaded {sset.n_spectra()} spectra")
    print(
        f"[QSHS] Shifted frequency span: "
        f"{sset.rf_grid[0]/1e6:.6f} to {sset.rf_grid[-1]/1e6:.6f} MHz"
    )

    # Plot a few spectra to verify import.
    nplot = min(args.max_plot, sset.n_spectra())
    for i in range(nplot):
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(sset.freqs_per_spec[i] / 1e6, sset.spectra[i], lw=0.7)
        ax.set(
            xlabel="Frequency offset [MHz]",
            ylabel="Power",
            title=f"QSHS imported spectrum {i}",
        )
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"qshs_spectrum_{i:03d}.png", dpi=150)
        plt.close(fig)

    if args.save_fox_h5:
        out_h5 = outdir / "spectra.h5"
        write_hdf5(sset, out_h5)
        print(f"[QSHS] Saved FOX-native HDF5: {out_h5}")

    print(f"[QSHS] Diagnostics saved in {outdir}")


if __name__ == "__main__":
    main()
