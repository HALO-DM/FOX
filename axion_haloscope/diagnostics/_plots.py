"""
Diagnostic Plots
"""

__all__ = ['vary_set_size_plots']

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from axion_haloscope.baseline import remove_baseline

def vary_set_size_plots(var_results, spacings_config, sets_by_spacing, var_dir, base):
    spacings_plot = [r["spacing_minutes"] for r in var_results]
    resid_av      = [r["average_residual_average"] for r in var_results]
    resid_std     = [r["average_residual_std"] for r in var_results]
    total_masked = [r["total_masked"] for r in var_results]


    # Plot absolute residual average vs set size
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(spacings_plot, resid_av, marker="o", alpha=0.7)
    ax.set_xlabel("Set size [minutes]")
    ax.set_ylabel("Average residuals  [V²/Hz]")
    ax.set_title(f"SG fit residual mean vs. set size ({base['clipping_mode']} clipping mode)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{var_dir}/residual_avg_vs_set_size.png", dpi=150, bbox_inches='tight')
    plt.close()


    # Plot residual std vs set size
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(spacings_plot, resid_std, marker="o")
    ax.set_xlabel("Set size threshold  [minutes]")
    ax.set_ylabel("Average residual std  [V²/Hz]")
    ax.set_title(f"Average residual std vs. set size ({base['clipping_mode']} clipping mode)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{var_dir}/residual_std_vs_set_size.png", dpi=150, bbox_inches='tight')
    plt.close()


    # Plot SG fit for first set for each set size
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = cm.viridis(np.linspace(0, 1, len(spacings_config)))

    for c_idx, test_spacing in enumerate(spacings_config):
        var_sets = sets_by_spacing[test_spacing]
        rep_set = next((s for s in var_sets if len(s) > 0), None)
        if rep_set is None:
            continue

        freqs_avg = np.mean([x[1] for x in rep_set], axis=0)
        spec_avg = np.mean([x[0] for x in rep_set], axis=0)
        try:
            _, baseline = remove_baseline(
                spectrum=spec_avg,
                window_length=base["sg_window_warm"],
                polyorder=base["sg_poly_warm"],
            )
        except Exception as e:
            print(f"[SG overlay] spacing={test_spacing}minutes: SG fit failed ({e}), skipping")
            continue

        ax.plot(freqs_avg / 1e6, baseline, color=colors[c_idx],
                label=f"{test_spacing} min  (n={len(rep_set)})")

    norm_spacing = mcolors.Normalize(vmin=min(spacings_config), vmax=max(spacings_config))
    sm_res = ScalarMappable(cmap=cm.viridis, norm=norm_spacing)
    sm_res.set_array([])
    fig.colorbar(sm_res, ax=ax, label="Set size [minutes]")
    ax.set_xlabel("IF frequency  [MHz]")
    ax.set_ylabel("PSD  [V²/Hz]")
    ax.set_title("Set averaged spectra for set size variation (1st set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{var_dir}/sg_fit_set_size_var.png", dpi=150, bbox_inches='tight')
    plt.close()


    # Plot per-set size diagnostic plots: set + average, set + average with errors, zoomed
    for test_spacing in tqdm(spacings_config, desc="Set size variation diagnostic plots"):

        var_sets = sets_by_spacing[test_spacing]
        rep_set = next((s for s in var_sets if len(s) > 0), None)

        if rep_set is None:
            continue

        freqs_avg = np.mean([x[1] for x in rep_set], axis=0) / 1e6
        spec_avg = np.mean([x[0] for x in rep_set], axis=0)
        spec_std = np.std([x[0] for x in rep_set], axis=0)


        # Plot set + average spectra 
        fig, ax = plt.subplots(figsize=(13, 7))
        greys = cm.Greys(np.linspace(0.3, 0.9, len(rep_set)))
        for i_spec, x in enumerate(rep_set):
            ax.plot(x[1] / 1e6, x[0], color=greys[i_spec])
        ax.plot(freqs_avg, spec_avg, alpha=0.8, color="red", label="set averaged")
        norm = mcolors.Normalize(vmin=0, vmax=len(rep_set))
        sm = ScalarMappable(cmap=cm.Greys, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Spectrum index in set")
        ax.set_xlabel("IF frequency  [MHz]")
        ax.set_ylabel("PSD  [V²/Hz]")
        ax.set_title(f"Set-averaged spectra and individual spectra — spacing {test_spacing} minutes (n={len(rep_set)})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{var_dir}/set_and_average_spectra_spacing_{test_spacing}.png", dpi=150, bbox_inches='tight')
        plt.close()


        # Plot average spectra with errors
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.errorbar(freqs_avg, spec_avg, spec_std, alpha=0.10, ecolor="blue", color="white", label="std on average")
        ax.plot(freqs_avg, spec_avg, alpha=1, color="red", label="set averaged")
        ax.set_xlabel("IF frequency  [MHz]")
        ax.set_ylabel("PSD  [V²/Hz]")
        ax.set_title(f"Set-averaged spectra with errors — spacing {test_spacing} minutes (n={len(rep_set)})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{var_dir}/average_spectra_errors_spacing_{test_spacing}.png", dpi=150, bbox_inches='tight')
        plt.close()


        # Plot average spectra with errors, zoomed
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.errorbar(freqs_avg, spec_avg, spec_std, alpha=0.5, ecolor="blue", color="white", label="std on average")
        ax.plot(freqs_avg, spec_avg, alpha=1, color="red", label="set averaged")
        ax.set_xlabel("IF frequency  [MHz]")
        ax.set_ylabel("PSD  [V²/Hz]")
        ax.set_title(f"Set-averaged spectra with errors zoomed - spacing {test_spacing} min (n={len(rep_set)})")

        x_min, x_max = 1.5, 1.75
        ax.set_xlim(x_min, x_max)
        in_range = (freqs_avg >= x_min) & (freqs_avg <= x_max)
        if in_range.any():
            y_lower = np.min(spec_avg[in_range] - spec_std[in_range])
            y_upper = np.max(spec_avg[in_range] + spec_std[in_range])
            y_pad = 0.05 * (y_upper - y_lower)
            ax.set_ylim(y_lower - y_pad, y_upper + y_pad)

        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{var_dir}/average_spectra_errors_zoom_spacing_{test_spacing}.png", dpi=150, bbox_inches='tight')
        plt.close()


    # Plot avg-of-avg std vs set size
    spacing_avg_of_avg_std = []
    for spacing in spacings_config:
        var_sets = sets_by_spacing[spacing]
        set_avg_stds = [np.mean(np.std([x[0] for x in set], axis=0)) for set in var_sets if len(set) > 0]
        spacing_avg_of_avg_std.append(np.mean(set_avg_stds) if set_avg_stds else np.nan)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(spacings_config, spacing_avg_of_avg_std, marker="o")
    ax.set_xlabel("Set spacing (minutes)")
    ax.set_ylabel("Average standard deviation  [V²/Hz]")
    ax.set_title("Average standard deviation vs set size")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{var_dir}/avg_avg_std_vs_set_size.png", dpi=150, bbox_inches='tight')
    plt.close()


    # Plot average cavity resonance drift within a set againist set size
    spacing_avg_res_spread = []
    for spacing in spacings_config:
        var_sets = sets_by_spacing[spacing]
        res_spreads_this = []

        for set in var_sets:
            if len(set) < 2:
                continue 
            res_freqs_in_set = [item[2] for item in set]
            res_freqs_in_set = np.asarray(res_freqs_in_set, dtype=float)
            finite_vals = res_freqs_in_set[np.isfinite(res_freqs_in_set)]
            if len(finite_vals) < 2:
                continue
            spread = np.max(finite_vals) - np.min(finite_vals)
            res_spreads_this.append(spread)

        if len(res_spreads_this) == 0:
            spacing_avg_res_spread.append(np.nan)
            continue

        spacing_avg_res_spread.append(np.mean(res_spreads_this))

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(spacings_config, spacing_avg_res_spread, marker="o")
    ax.set_xlabel("Set size  [minutes]")
    ax.set_ylabel("Average resonance frequency spread  [GHz]")
    ax.set_title("Cavity resonance drift within a set vs. set size")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{var_dir}/resonance_drift_vs_set_size.png", dpi=150, bbox_inches='tight')
    plt.close()


    # Plot total bins masked againist set size
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(spacings_config, total_masked, marker="o")
    ax.set_xlabel("Set size [minutes]")
    ax.set_ylabel("Total masked bins ")
    ax.set_title(f"Total masked bins vs. set size ({base['clipping_mode']} clipping mode)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{var_dir}/total_masked_vs_set_size.png", dpi=150, bbox_inches='tight')
    plt.close()